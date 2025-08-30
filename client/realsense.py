import shutil
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, List

import numpy as np
import pyrealsense2 as rs
import multiprocessing as mp

from tqdm import tqdm


def enumerate_devices():
    ctx = rs.context()
    devs = []
    for d in ctx.query_devices():
        serial = d.get_info(rs.camera_info.serial_number)
        product = d.get_info(rs.camera_info.product_line)
        # sensors = d.query_sensors()
        # for sensor in sensors:
        #     for profile in sensor.get_stream_profiles():
        #         try:
        #             video_profile = profile.as_video_stream_profile()
        #             intrinsics = video_profile.get_intrinsics()
        #             print(f"Serial: {serial}, Stream: {video_profile.stream_type()}, Index: {video_profile.stream_index()}, Format: {video_profile.format()}, Resolution: {intrinsics.width}x{intrinsics.height}, Intrinsics: {intrinsics}")
        #         except:
        #             pass
        devs.append((serial, product))
    return devs

def get_tmstmp(frame):
    return frame.get_frame_metadata(rs.frame_metadata_value.backend_timestamp)



import pyrealsense2 as rs
import numpy as np
from pathlib import Path
from typing import List

class RSBagProcessor:
    def __init__(self, bag_paths: List[Path], n_frames: int, width: int, height: int, fps: int):
        rs.log_to_console(min_severity=rs.log_severity.warn)
        self.bag_paths = bag_paths
        self.width = width
        self.height = height
        self.fps = fps

    def process_all_frames(self):
        # Iterate over each bag file and process frames sequentially
        for cam_idx, bag_path in enumerate(self.bag_paths):
            yield from self.process_frames_for_bag(cam_idx, bag_path)

    def process_frames_for_bag(self, cam_idx: int, bag_path: Path):
        # Initialize the pipeline and start streaming
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(str(bag_path), repeat_playback=False)
        pipeline.start(config)
        align = rs.align(rs.stream.color)
        frame_idx = 0

        try:
            while True:
                # Wait for the next frames
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)

                # Get color and depth frames
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                # Convert frames to numpy arrays
                color = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())

                # Extract timestamps
                col_tmstmp = self.get_tmstmp(color_frame)
                dep_tmstmp = self.get_tmstmp(depth_frame)
                # Yield the frame data
                if frame_idx > 15:
                    yield color, col_tmstmp, depth, dep_tmstmp, cam_idx
                frame_idx += 1


        except RuntimeError as e:
            print(f"Error while processing bag {bag_path}: {e}")
        finally:
            pipeline.stop()

    def get_tmstmp(self, frame):
        return frame.get_timestamp()


class RealSenseInterface:
    class _FrameGrabberThread(threading.Thread):
        def __init__(self, idx, pipe, callback, stop_event):
            super().__init__()
            self.idx = idx
            self.pipe = pipe
            self.callback = callback
            self.stop_event = stop_event

        def run(self):
            while not self.stop_event.is_set():
                try:
                    # config results in these being written to bagfile in /dev/shm
                    fs = self.pipe.wait_for_frames(timeout_ms=5000)
                    self.callback(self.idx)
                except Exception as e:
                    print(f"Camera {self.idx} failed to grab frame: {e}")
                    traceback.print_exc()
            print(f"Stopping capture pipeline {self.idx}")
            self.pipe.stop()

    @property
    def n_cameras(self):
        return len(self._pipelines)

    def __init__(self, path: Path, n_frames: int, width: int, height: int, fps: int, init=True):
        rs.log_to_console(min_severity=rs.log_severity.warn)
        self._path = path
        self._n_frames = n_frames
        self._width = width
        self._height = height
        self._fps = fps
        if init:
            self._pipelines, self._recording_bagpaths = self._initialize_cameras()
            self._stop_events = []
            self._threads = []
            self._start_indices = []
            self.frame_counts = {i: 0 for i in range(len(self._pipelines))}

    def start_capture(self, on_receive_frame: Callable = None):
        def _callback_wrapper(cap_idx):
            if on_receive_frame is not None:
                on_receive_frame(cap_idx)
            self.frame_counts[cap_idx] += 1
            if self.frame_counts[cap_idx] >= self._n_frames:
                self.stop_capture(cap_idx)

        for idx, (pipe, cfg) in enumerate(self._pipelines):
            stop_event = threading.Event()
            pipe.start(cfg)
            t = self._FrameGrabberThread(idx, pipe, _callback_wrapper, stop_event)
            t.start()
            self._threads.append(t)
            self._stop_events.append(stop_event)

    def create_pipeline(self, serial: str, w: int, h: int, fps: int):
        tmp_path = str(self._path / f"{serial}.bag")

        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)

        cfg.enable_record_to_file(tmp_path)

        pipe = rs.pipeline()
        return pipe, cfg, tmp_path

    def _initialize_cameras(self):
        cameras = enumerate_devices()
        if not cameras:
            print("No RealSense devices detected – exiting.")
            return []

        print(f"Found {len(cameras)} camera(s):")
        for idx, (serial, product) in enumerate(cameras):
            print(f"   Camera {idx}: {serial}  ({product})")

        pipelines = []
        bagpaths = []
        for idx, (serial, _) in enumerate(cameras):
            pipe, cfg, bagpath = self.create_pipeline(serial, self._width, self._height, self._fps)
            pipelines.append((pipe, cfg))
            bagpaths.append(bagpath)

        print("Waiting for cameras to start..")
        time.sleep(5.0)
        return pipelines, bagpaths

    def stop_capture(self, capture_idx: int):
        self._stop_events[capture_idx].set()

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
        # for shm_path in tqdm(self._recording_bagpaths, "Saving recordings"):
        #     shutil.move(str(shm_path), str(self._path / Path(shm_path).name))
