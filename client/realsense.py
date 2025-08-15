import threading
from collections import defaultdict
from typing import Callable

import numpy as np
from client.record import start_pipeline, enumerate_devices, get_tmstmp
import pyrealsense2 as rs


class RealSenseInterface:
    class _FrameGrabberThread(threading.Thread):
        def __init__(self, idx, pipe, align, callback, stop_event):
            super().__init__()
            self.idx = idx
            self.pipe = pipe
            self.align = align
            self.callback = callback
            self.stop_event = stop_event

        def run(self):
            while not self.stop_event.is_set():
                try:
                    fs = self.pipe.wait_for_frames(timeout_ms=5000)
                    fs = self.align.process(fs) if self.align else fs
                    depth_frame = fs.get_depth_frame()
                    color_frame = fs.get_color_frame()

                    depth = np.asanyarray(depth_frame.get_data())
                    color = np.asanyarray(color_frame.get_data())

                    depth_ts = get_tmstmp(depth_frame)
                    color_ts = get_tmstmp(color_frame)

                    self.callback(self.idx, color, color_ts, depth, depth_ts)
                except Exception as e:
                    print(f"Camera {self.idx} failed to grab frame: {e}")

    def __init__(self, n_frames: int, width: int , height: int, fps: int, frame_callback: Callable):
        rs.log_to_console(min_severity=rs.log_severity.warn)

        self._n_frames = n_frames
        self._width = width
        self._height = height
        self._fps = fps
        self._pipelines, self._aligners = self._initialize_cameras()
        self._stop_events = []
        self._threads = []
        self._frame_callback = frame_callback
        self._frame_counts = defaultdict(int)

        def _callback_wrapper(cap_idx, **data):
            self._frame_callback(cap_idx, **data)
            self._frame_counts[cap_idx] += 1
            if self._frame_counts[cap_idx] > self._n_frames:
                self.stop_capture(cap_idx)

        for idx, (pipe, align) in enumerate(zip(self._pipelines, self._aligners)):
            stop_event = threading.Event()
            t = self._FrameGrabberThread(idx, pipe, align, _callback_wrapper, stop_event)
            t.start()
            self._threads.append(t)
            self._stop_events.append(stop_event)

    def _initialize_cameras(self):
        cameras = enumerate_devices()
        if not cameras:
            print("No RealSense devices detected – exiting.")
            return [], []

        print(f"Found {len(cameras)} camera(s):")
        for idx, (serial, product) in enumerate(cameras):
            print(f"   Camera {idx}: {serial}  ({product})")

        pipelines = []
        aligners = []
        for serial, _ in cameras:
            pipe, align, _ = start_pipeline(serial, self._width, self._height, self._fps)
            pipelines.append(pipe)
            aligners.append(align)
        return pipelines, aligners


    def stop_capture(self, capture_idx: int):
        pass

    def shutdown(self):
        self._stop_event.set()
        for t in self._threads:
            t.join()
        print("All camera threads stopped.")
