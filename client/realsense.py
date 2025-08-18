import os
import tempfile
import threading
import traceback
from collections import defaultdict
from typing import Callable, List

import numpy as np
import pyrealsense2 as rs

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

    def __init__(self, n_frames: int, width: int, height: int, fps: int):
        rs.log_to_console(min_severity=rs.log_severity.warn)
        self._n_frames = n_frames
        self._width = width
        self._height = height
        self._fps = fps
        self._pipelines, self._recording_bagpaths = self._initialize_cameras()
        self._stop_events = []
        self._threads = []
        self.frame_counts = defaultdict(int)

    def start_capture(self, on_receive_frame: Callable = None):
        def _callback_wrapper(cap_idx):
            if on_receive_frame is not None:
                on_receive_frame(cap_idx)
            self.frame_counts[cap_idx] += 1
            if self.frame_counts[cap_idx] >= self._n_frames:
                self.stop_capture(cap_idx)

        for idx, pipe in enumerate(self._pipelines):
            stop_event = threading.Event()
            t = self._FrameGrabberThread(idx, pipe, _callback_wrapper, stop_event)
            t.start()
            self._threads.append(t)
            self._stop_events.append(stop_event)
            self._savers.append(rs.save_single_frameset())

    def start_pipeline(self, serial: str, w: int, h: int, fps: int):
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".bag", dir="/dev/shm")
        os.close(tmp_fd)

        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)

        cfg.enable_record_to_file(tmp_path)

        pipe = rs.pipeline()
        pipe.start(cfg)

        return pipe, tmp_path

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
        for serial, _ in cameras:
            pipe, bagpath = self.start_pipeline(serial, self._width, self._height, self._fps)
            pipelines.append(pipe)
            bagpaths.append(bagpath)
        return pipelines, bagpaths

    def stop_capture(self, capture_idx: int):
        self._stop_events[capture_idx].set()

    def stop_all_captures(self):
        for event in self._stop_events:
            event.set()
        for t in self._threads:
            t.join()
        print("Capture stopped.")


    def process_frames(self, capture_idx: int):
        pipeline = rs.pipeline()
        config = rs.config()
        bag_path = self._recording_bagpaths[capture_idx]
        config.enable_device_from_file(bag_path, repeat_playback=False)
        pipeline.start(config)
        align = rs.align(rs.stream.color)

        try:
            while True:
                fs = pipeline.wait_for_frames()
                aligned_fs = align.process(fs)
                col_frame = aligned_fs.get_color_frame()
                dep_frame = aligned_fs.get_depth_frame()
                color = np.asanyarray(col_frame.get_data())
                depth = np.asanyarray(dep_frame.get_data())
                col_tmstmp = get_tmstmp(color)
                dep_tmstmp = get_tmstmp(depth)
                yield color, col_tmstmp, depth, dep_tmstmp
        except:
            pass
        finally:
            pipeline.stop()


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_all_captures()
        for path in self._recording_bagpaths:
            os.remove(path)
