"""Evaluation-mode RealSense interface.

:class:`EvalRealsense` captures RGB-only frames from a configured set of
RealSense cameras and caches recent observations in a fixed-size deque
for low-latency policy inference. Unlike the collection-mode
:class:`~client.collect.realsense.RealSenseInterface`, it does not record
bag files or capture depth.
"""
import threading
import traceback
from collections import deque
from typing import Callable, List

import numpy as np
import torch
import pyrealsense2 as rs

from client.utils import enumerate_devices


class EvalRealsense:
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

    def __init__(self, n_frames: int, width: int, height: int, fps: int, obs_cams: List[str]):
        rs.log_to_console(min_severity=rs.log_severity.warn)
        self._n_frames = n_frames
        self._width = width
        self._height = height
        self._fps = fps
        self._obs_cam_serials = obs_cams
        self._counts_lock = threading.Lock()
        self._serials: List[str] = []
        self._serial_to_idx = {}
        self._pipelines = self._initialize_cameras()
        self._stop_events = []
        self._threads = []
        self.frame_counts = {s: 0 for s in self._serials}

    @property
    def serials(self) -> List[str]:
        return list(self._serials)

    def _initialize_cameras(self):
        cameras = enumerate_devices()
        if not cameras:
            print("No RealSense devices detected – exiting.")
            return []

        print(f"Found {len(cameras)} camera(s):")
        for idx, (serial, product) in enumerate(cameras):
            print(f"   Camera {idx}: {serial}  ({product})")

        pipelines = []
        for idx, serial in enumerate(self._obs_cam_serials):
            self._serials.append(serial)
            self._serial_to_idx[serial] = idx

            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)
            pipe = rs.pipeline()
            pipelines.append((pipe, cfg))

        return pipelines

    def start_capture(self, on_receive_frame: Callable = None, on_warmup: Callable = None):
        def _callback_wrapper(serial):
            if on_receive_frame is not None:
                on_receive_frame(serial)
            with self._counts_lock:
                self.frame_counts[serial] += 1
                if self.frame_counts[serial] >= self._n_frames:
                    print("stopping capture", serial)
                    self._stop_capture_by_serial(serial)

        # Phase 1: Start all pipelines and configure exposure.
        started = []
        for idx, (pipe, cfg) in enumerate(self._pipelines):
            profile = pipe.start(cfg)
            col_sensor = profile.get_device().query_sensors()[1]
            col_sensor.set_option(rs.option.enable_auto_exposure, 0)
            col_sensor.set_option(rs.option.exposure, 250)
            col_sensor.set_option(rs.option.gain, 128)
            started.append(pipe)

        # Phase 2: Run on_warmup callback (e.g. gripper homing) concurrently
        # with the warmup drain.
        if on_warmup is not None:
            on_warmup()

        # Phase 3: Drain warmup frames so manual exposure takes effect.
        for pipe in started:
            for _ in range(20):
                pipe.wait_for_frames(timeout_ms=5000)

        # Phase 4: Start capture threads (no frames counted until now).
        for idx, pipe in enumerate(started):
            serial = self._serials[idx]
            stop_event = threading.Event()
            t = self._FrameGrabberThread(serial, pipe, _callback_wrapper, stop_event)
            t.start()
            self._threads.append(t)
            self._stop_events.append(stop_event)

    def get_rgb_obs(self) -> List[torch.Tensor]:
        return [t.get_obs() for t in self._threads]

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
