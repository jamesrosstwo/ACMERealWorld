import threading
import time
import traceback
from typing import Callable

import numpy as np
import pyrealsense2 as rs
import multiprocessing as mp

def enumerate_devices():
    ctx = rs.context()
    devs = []
    for d in ctx.query_devices():
        serial = d.get_info(rs.camera_info.serial_number)
        product = d.get_info(rs.camera_info.product_line)
        sensors = d.query_sensors()
        for sensor in sensors:
            for profile in sensor.get_stream_profiles():
                try:
                    video_profile = profile.as_video_stream_profile()
                    intrinsics = video_profile.get_intrinsics()
                    print(f"Serial: {serial}, Stream: {video_profile.stream_type()}, Index: {video_profile.stream_index()}, Format: {video_profile.format()}, Resolution: {intrinsics.width}x{intrinsics.height}, Intrinsics: {intrinsics}")
                except:
                    pass
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

    def __init__(self, n_frames: int, width: int, height: int, fps: int, init=True):
        rs.log_to_console(min_severity=rs.log_severity.warn)
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
        import uuid
        tmp_path = f"/dev/shm/{uuid.uuid4().hex}.bag"

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
        for serial, _ in cameras:
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
                col_tmstmp = get_tmstmp(col_frame)
                dep_tmstmp = get_tmstmp(dep_frame)
                yield color, col_tmstmp, depth, dep_tmstmp
        except RuntimeError:
            pass
        finally:
            pipeline.stop()

    def process_frames_for_bag(self, cam_idx, bag_path, output_queue):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(str(bag_path), repeat_playback=False)
        align = rs.align(rs.stream.color)

        pipeline.start(config)

        try:
            while True:
                fs = pipeline.wait_for_frames()
                aligned_fs = align.process(fs)

                col_frame = aligned_fs.get_color_frame()
                dep_frame = aligned_fs.get_depth_frame()

                if not col_frame or not dep_frame:
                    continue

                color = np.asanyarray(col_frame.get_data())
                depth = np.asanyarray(dep_frame.get_data())

                col_tmstmp = get_tmstmp(col_frame)
                dep_tmstmp = get_tmstmp(dep_frame)

                # Send data back to main process
                output_queue.put((color, col_tmstmp, depth, dep_tmstmp, cam_idx))
        except RuntimeError:
            # Playback finished
            pass
        finally:
            pipeline.stop()
            output_queue.put(("EOF", bag_path))

    def process_all_frames_parallel(self):
        output_queue = mp.Queue()
        processes = []

        for cam_idx, bag_path in enumerate(self._recording_bagpaths):
            p = mp.Process(target=self.process_frames_for_bag, args=(cam_idx, bag_path, output_queue))
            p.start()
            processes.append(p)

        active_bags = set(self._recording_bagpaths)
        while active_bags:
            result = output_queue.get()
            if result[0] == "EOF":
                bag_path = result[1]
                active_bags.remove(bag_path)
            else:
                color, col_tmstmp, depth, dep_tmstmp, cam_idx = result
                yield color, col_tmstmp, depth, dep_tmstmp, cam_idx

        for p in processes:
            p.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_all_captures()
        # for path in self._recording_bagpaths:
        #     os.remove(path)
