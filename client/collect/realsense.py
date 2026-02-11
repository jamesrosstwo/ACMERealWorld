"""Intel RealSense multi-camera capture and bag file playback.

Provides :class:`RealSenseInterface` for live multi-camera RGB-D capture with
per-camera threading and automatic bag file recording, and :class:`RSBagProcessor`
for offline playback of recorded ``.bag`` files during postprocessing. Cameras
are identified by serial number throughout.
"""
import threading
import time
import traceback
from typing import Callable
import pyrealsense2 as rs
import numpy as np
from pathlib import Path
from typing import List

from client.utils import enumerate_devices


class RSBagProcessor:
    def __init__(self, bag_paths: List[Path], n_frames: int, width: int, height: int, fps: int):
        rs.log_to_console(min_severity=rs.log_severity.warn)
        self.bag_paths = bag_paths
        self.width = width
        self.height = height
        self.fps = fps

    def process_all_frames(self):
        for bag_path in self.bag_paths:
            serial = bag_path.stem
            yield from self.process_frames_for_bag(serial, bag_path)

    def process_frames_for_bag(self, serial: str, bag_path: Path):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(str(bag_path), repeat_playback=False)
        pipeline.start(config)
        align = rs.align(rs.stream.color)

        try:
            while True:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)

                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                color = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())

                col_tmstmp = self.get_tmstmp(color_frame)
                dep_tmstmp = self.get_tmstmp(depth_frame)
                yield color, col_tmstmp, depth, dep_tmstmp, serial

        except RuntimeError as e:
            print(f"Error while processing bag {bag_path}: {e}")
        finally:
            pipeline.stop()

    def get_tmstmp(self, frame):
        return frame.get_timestamp()


class RealSenseInterface:
    class _FrameGrabberThread(threading.Thread):
        def __init__(self, serial, pipe, callback, stop_event):
            super().__init__()
            self.serial = serial
            self.pipe = pipe
            self.callback = callback
            self.stop_event = stop_event

        def run(self):
            while not self.stop_event.is_set():
                try:
                    fs = self.pipe.wait_for_frames(timeout_ms=5000)
                    self.callback(self.serial, fs.get_timestamp())
                except Exception as e:
                    print(f"Camera {self.serial} failed to grab frame: {e}")
                    traceback.print_exc()
            print(f"Stopping capture pipeline {self.serial}")
            self.pipe.stop()

    @property
    def n_cameras(self):
        return len(self._pipelines)

    @property
    def serials(self) -> List[str]:
        return list(self._serials)

    def __init__(self, path: Path, *, n_frames: int, width: int, height: int, fps: int):
        rs.log_to_console(min_severity=rs.log_severity.warn)
        self._path = path
        self._n_frames = n_frames
        self._width = width
        self._height = height
        self._fps = fps
        self._all_intr = []
        self._counts_lock = threading.Lock()
        self._serials: List[str] = []
        self._serial_to_idx = {}
        self._pipelines = self._initialize_cameras()
        self._stop_events = []
        self._threads = []
        self.frame_counts = {s: 0 for s in self._serials}

    def _initialize_cameras(self):
        cameras = enumerate_devices()
        if not cameras:
            print("No RealSense devices detected – exiting.")
            return []

        print(f"Found {len(cameras)} camera(s):")
        for idx, (serial, product) in enumerate(cameras):
            print(f"   Camera {idx}: {serial}  ({product})")

        pipelines = []
        for idx, (serial, _) in enumerate(cameras):
            self._serials.append(serial)
            self._serial_to_idx[serial] = idx

            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.depth, self._width, self._height, rs.format.z16, self._fps)
            cfg.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)
            bag_path = str(self._path / f"{serial}.bag")
            cfg.enable_record_to_file(bag_path)

            pipe = rs.pipeline()
            pipelines.append((pipe, cfg))

        print("Waiting for cameras to start..")
        time.sleep(8.0)
        return pipelines

    def start_capture(self, on_receive_frame: Callable = None, on_warmup: Callable = None):
        def _callback_wrapper(serial, timestamp):
            if on_receive_frame is not None:
                on_receive_frame(serial, timestamp)
            with self._counts_lock:
                self.frame_counts[serial] += 1
                if self.frame_counts[serial] >= self._n_frames:
                    print("stopping capture", serial)
                    self._stop_capture_by_serial(serial)

        # Phase 1: Start all pipelines, pause recording, and configure exposure.
        started = []
        recorders = []
        for idx, (pipe, cfg) in enumerate(self._pipelines):
            serial = self._serials[idx]
            profile = pipe.start(cfg)

            # Pause bag recording immediately so warmup frames are not saved.
            recorder = profile.get_device().as_recorder()
            recorder.pause()
            recorders.append(recorder)

            dep = profile.get_stream(rs.stream.depth)
            intr = dep.as_video_stream_profile().get_intrinsics()
            intr_vals = intr.fx, intr.fy, intr.ppx, intr.ppy
            print(f"Camera {serial} intrinsics: {intr_vals})")
            self._all_intr.append(np.asarray(intr_vals))

            col_sensor = profile.get_device().query_sensors()[1]
            col_sensor.set_option(rs.option.enable_auto_exposure, 0)
            col_sensor.set_option(rs.option.exposure, 250)
            col_sensor.set_option(rs.option.gain, 128)

            started.append(pipe)

        if self._all_intr:
            np.save(self._path / "intrinsics.npy", np.stack(self._all_intr))

        # Phase 2: Run on_warmup callback (e.g. gripper homing) concurrently
        # with the warmup drain.
        if on_warmup is not None:
            on_warmup()

        # Phase 3: Drain warmup frames so manual exposure takes effect.
        for pipe in started:
            for _ in range(30):
                pipe.wait_for_frames(timeout_ms=5000)

        # Phase 4: Resume recording and start capture threads.
        for recorder in recorders:
            recorder.resume()

        for idx, pipe in enumerate(started):
            serial = self._serials[idx]
            stop_event = threading.Event()
            t = self._FrameGrabberThread(serial, pipe, _callback_wrapper, stop_event)
            t.start()
            self._threads.append(t)
            self._stop_events.append(stop_event)

    def get_frame_counts(self):
        with self._counts_lock:
            return dict(self.frame_counts)

    def reset_frame_counts(self):
        with self._counts_lock:
            for k in self.frame_counts:
                self.frame_counts[k] = 0

    def _stop_capture_by_serial(self, serial: str):
        idx = self._serial_to_idx[serial]
        self._stop_events[idx].set()

    def stop_all_captures(self):
        for event in self._stop_events:
            event.set()
        for t in self._threads:
            t.join()
        print("Capture stopped.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_all_captures()
