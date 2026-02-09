import threading
import time
import traceback
from typing import Callable
import pyrealsense2 as rs
import numpy as np
from pathlib import Path
from typing import List


def enumerate_devices():
    ctx = rs.context()
    devs = []
    for d in ctx.query_devices():
        serial = d.get_info(rs.camera_info.serial_number)
        product = d.get_info(rs.camera_info.product_line)
        devs.append((serial, product))
    return devs

def get_tmstmp(frame):
    return frame.get_frame_metadata(rs.frame_metadata_value.backend_timestamp)


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
                    self.callback(self.serial)
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

    def __init__(self, path: Path = None, *, n_frames: int, width: int, height: int, fps: int, init=True):
        rs.log_to_console(min_severity=rs.log_severity.warn)
        self._path = path
        self._n_frames = n_frames
        self._width = width
        self._height = height
        self._fps = fps
        self._all_intr = []
        self._recording_bagpaths = []
        self._counts_lock = threading.Lock()
        self._serials: List[str] = []
        self._serial_to_idx = {}
        if init:
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
            pipe, cfg = self.create_pipeline(serial, self._width, self._height, self._fps)
            pipelines.append((pipe, cfg))

        print("Waiting for cameras to start..")
        time.sleep(8.0)
        return pipelines

    def create_pipeline(self, serial: str, w: int, h: int, fps: int):
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)

        if self._path is not None:
            tmp_path = str(self._path / f"{serial}.bag")
            cfg.enable_record_to_file(tmp_path)
            self._recording_bagpaths.append(tmp_path)

        pipe = rs.pipeline()
        return pipe, cfg

    def _on_pipeline_started(self, serial, profile):
        dep = profile.get_stream(rs.stream.depth)
        i = dep.as_video_stream_profile().get_intrinsics()
        intr_vals = i.fx, i.fy, i.ppx, i.ppy
        print(f"Camera {serial} intrinsics: {intr_vals})")
        self._all_intr.append(np.asarray(intr_vals))

    def _on_all_pipelines_started(self):
        if self._path is not None and self._all_intr:
            intrinsics_path = self._path / "intrinsics.npy"
            np.save(intrinsics_path, np.stack(self._all_intr))

    def _create_frame_thread(self, serial, pipe, callback, stop_event):
        return self._FrameGrabberThread(serial, pipe, callback, stop_event)

    def start_capture(self, on_receive_frame: Callable = None):
        def _callback_wrapper(serial):
            if on_receive_frame is not None:
                on_receive_frame(serial)
            with self._counts_lock:
                self.frame_counts[serial] += 1
                if self.frame_counts[serial] >= self._n_frames:
                    print("stopping capture", serial)
                    self._stop_capture_by_serial(serial)

        for idx, (pipe, cfg) in enumerate(self._pipelines):
            serial = self._serials[idx]
            stop_event = threading.Event()
            profile = pipe.start(cfg)

            self._on_pipeline_started(serial, profile)

            col_sensor = profile.get_device().query_sensors()[1]
            col_sensor.set_option(rs.option.exposure, 250)
            col_sensor.set_option(rs.option.gain, 128)
            t = self._create_frame_thread(serial, pipe, _callback_wrapper, stop_event)
            t.start()
            self._threads.append(t)
            self._stop_events.append(stop_event)

        self._on_all_pipelines_started()

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
