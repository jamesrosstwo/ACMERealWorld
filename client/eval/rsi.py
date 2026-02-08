import threading
import traceback
from collections import deque
from typing import Callable, List

import numpy as np
import torch

from client.realsense import RealSenseInterface, enumerate_devices
import pyrealsense2 as rs



class EvalRSI(RealSenseInterface):
    class _FrameGrabberThread(threading.Thread):
        def __init__(self, idx, pipe, callback, stop_event, cache_size=2):
            super().__init__()
            self.idx = idx
            self.pipe = pipe
            self.callback = callback
            self.stop_event = stop_event
            self._cache = deque(maxlen=cache_size)

        def run(self):
            while not self.stop_event.is_set():
                try:
                    fs = self.pipe.wait_for_frames(timeout_ms=5000)
                    color_frame = fs.get_color_frame()
                    # Convert frames to numpy arrays
                    color = torch.tensor(np.asanyarray(color_frame.get_data()))
                    self._cache.append(color)
                    self.callback(self.idx)
                except Exception as e:
                    print(f"Camera {self.idx} failed to grab frame: {e}")
                    traceback.print_exc()
            print(f"Stopping capture pipeline {self.idx}")
            self.pipe.stop()

        def get_obs(self):
            return torch.stack(list(self._cache))

    def _initialize_cameras(self):
        cameras = enumerate_devices()
        if not cameras:
            print("No RealSense devices detected – exiting.")
            return []

        print(f"Found {len(cameras)} camera(s):")
        for idx, (serial, product) in enumerate(cameras):
            print(f"   Camera {idx}: {serial}  ({product})")

        pipelines = []

        for serial in self._obs_cam_serial:
            pipe, cfg = self.create_pipeline(serial, self._width, self._height, self._fps)
            pipelines.append((pipe, cfg))
        return pipelines

    def __init__(self, n_frames: int, width: int, height: int, fps: int, obs_cams: List[str], init=True):
        rs.log_to_console(min_severity=rs.log_severity.warn)
        self._n_frames = n_frames
        self._width = width
        self._height = height
        self._fps = fps
        self._obs_cam_serial: List[str] = obs_cams
        if init:
            self._pipelines = self._initialize_cameras()
            self._stop_events = []
            self._threads = []
            self._start_indices = []
            self.frame_counts = {i: 0 for i in range(len(self._pipelines))}

    def get_rgb_obs(self) -> List[torch.Tensor]:
        return [t.get_obs() for t in self._threads]

    def start_capture(self, on_receive_frame: Callable = None):
        def _callback_wrapper(cap_idx):
            if on_receive_frame is not None:
                on_receive_frame(cap_idx)
            self.frame_counts[cap_idx] += 1
            if self.frame_counts[cap_idx] >= self._n_frames:
                self.stop_capture(cap_idx)

        for idx, (pipe, cfg) in enumerate(self._pipelines):
            stop_event = threading.Event()
            profile = pipe.start(cfg)

            col_sensor = profile.get_device().query_sensors()[1]
            col_sensor.set_option(rs.option.exposure, 250)
            col_sensor.set_option(rs.option.gain, 128)
            t = self._FrameGrabberThread(idx, pipe, _callback_wrapper, stop_event)
            t.start()
            self._threads.append(t)
            self._stop_events.append(stop_event)

    def create_pipeline(self, serial: str, w: int, h: int, fps: int):
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        pipe = rs.pipeline()
        return pipe, cfg

