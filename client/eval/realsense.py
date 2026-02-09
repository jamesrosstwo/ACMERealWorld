"""Evaluation-mode RealSense interface.

:class:`EvalRealsense` is a specialized subclass of
:class:`~client.realsense.RealSenseInterface` that captures only RGB frames
(no depth or bag recording) and caches recent observations in a fixed-size
deque for low-latency policy inference.
"""
import threading
import traceback
from collections import deque
from typing import List

import numpy as np
import torch
import pyrealsense2 as rs

from client.realsense import RealSenseInterface, enumerate_devices


class EvalRealsense(RealSenseInterface):
    class _FrameGrabberThread(threading.Thread):
        def __init__(self, serial, pipe, callback, stop_event, cache_size=2):
            super().__init__()
            self.serial = serial
            self.pipe = pipe
            self.callback = callback
            self.stop_event = stop_event
            self._cache = deque(maxlen=cache_size)

        def run(self):
            while not self.stop_event.is_set():
                try:
                    fs = self.pipe.wait_for_frames(timeout_ms=5000)
                    color_frame = fs.get_color_frame()
                    color = torch.tensor(np.asanyarray(color_frame.get_data()))
                    self._cache.append(color)
                    self.callback(self.serial)
                except Exception as e:
                    print(f"Camera {self.serial} failed to grab frame: {e}")
                    traceback.print_exc()
            print(f"Stopping capture pipeline {self.serial}")
            self.pipe.stop()

        def get_obs(self):
            return torch.stack(list(self._cache))

    def __init__(self, n_frames: int, width: int, height: int, fps: int, obs_cams: List[str], init=True):
        self._obs_cam_serial: List[str] = obs_cams
        super().__init__(n_frames=n_frames, width=width, height=height, fps=fps, init=init)

    def _initialize_cameras(self):
        cameras = enumerate_devices()
        if not cameras:
            print("No RealSense devices detected – exiting.")
            return []

        print(f"Found {len(cameras)} camera(s):")
        for idx, (serial, product) in enumerate(cameras):
            print(f"   Camera {idx}: {serial}  ({product})")

        pipelines = []
        for idx, serial in enumerate(self._obs_cam_serial):
            self._serials.append(serial)
            self._serial_to_idx[serial] = idx
            pipe, cfg = self.create_pipeline(serial, self._width, self._height, self._fps)
            pipelines.append((pipe, cfg))
        return pipelines

    def create_pipeline(self, serial: str, w: int, h: int, fps: int):
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        pipe = rs.pipeline()
        return pipe, cfg

    def _on_pipeline_started(self, serial, profile):
        pass  # No depth intrinsics in eval mode

    def _on_all_pipelines_started(self):
        pass  # No intrinsics file to save

    def _create_frame_thread(self, serial, pipe, callback, stop_event):
        return self._FrameGrabberThread(serial, pipe, callback, stop_event)

    def get_rgb_obs(self) -> List[torch.Tensor]:
        return [t.get_obs() for t in self._threads]
