import threading
import traceback
from collections import defaultdict
from typing import Callable

import numpy as np
from client.record import start_pipeline, enumerate_devices, get_tmstmp
import pyrealsense2 as rs


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
                    fs = self.pipe.wait_for_frames(timeout_ms=5000)
                    self.callback(
                        self.idx,
                        fs
                    )
                except Exception as e:
                    print(f"Camera {self.idx} failed to grab frame: {e}")
                    traceback.print_exc()
            print(f"Stopping capture pipeline {self.idx}")
            self.pipe.stop()

    @property
    def n_cameras(self):
        return len(self._pipelines)

    def __init__(self, n_frames: int, width: int, height: int, fps: int):
        rs.log_to_console(min_severity=rs.log_severity.warn)
        self._n_frames = n_frames
        self._width = width
        self._height = height
        self._fps = fps
        self._aligner = rs.align(rs.stream.color)
        self._pipelines = self._initialize_cameras()
        self._stop_events = []
        self._threads = []
        self.frame_counts = defaultdict(int)
        self._frame_streams = defaultdict(list)

    def start_capture(self, on_receive_frame: Callable):
        def _callback_wrapper(cap_idx, fs):
            on_receive_frame(cap_idx)
            self.frame_counts[cap_idx] += 1
            self._frame_streams[cap_idx].append(fs)
            if self.frame_counts[cap_idx] >= self._n_frames:
                self.stop_capture(cap_idx)

        for idx, pipe in enumerate(zip(self._pipelines)):
            stop_event = threading.Event()
            t = self._FrameGrabberThread(idx, pipe, _callback_wrapper, stop_event)
            t.start()
            self._threads.append(t)
            self._stop_events.append(stop_event)

    def _initialize_cameras(self):
        cameras = enumerate_devices()
        if not cameras:
            print("No RealSense devices detected – exiting.")
            return []

        print(f"Found {len(cameras)} camera(s):")
        for idx, (serial, product) in enumerate(cameras):
            print(f"   Camera {idx}: {serial}  ({product})")

        pipelines = []
        for serial, _ in cameras:
            pipe = start_pipeline(serial, self._width, self._height, self._fps)
            pipelines.append(pipe)
        return pipelines

    def stop_capture(self, capture_idx: int):
        self._stop_events[capture_idx].set()

    def stop_all_captures(self):
        for event in self._stop_events:
            event.set()
        for t in self._threads:
            t.join()
        print("Capture stopped.")


    def process_frames(self, capture_idx: int):
        for fs in self._frame_streams[capture_idx]:
            aligned_fs = self._aligner.process(fs)
            col_frame = aligned_fs.get_color_frame()
            dep_frame = aligned_fs.get_depth_frame()
            color = np.asanyarray(col_frame.get_data())
            depth = np.asanyarray(dep_frame.get_data())
            col_tmstmp = get_tmstmp(color)
            dep_tmstmp = get_tmstmp(depth)
            yield color, col_tmstmp, depth, dep_tmstmp





    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_all_captures()