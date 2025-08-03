#!/usr/bin/env python3
"""
multi_realsense_sync.py
Grab synchronized frames from *all* attached RealSense cameras and save them with wall-clock timestamps.
"""


"""
FOLDER=$(pwd)
xhost +local:root
docker run -it -e "DISPLAY" -e "QT_X11_NO_MITSHM=1" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "$FOLDER:/data" kalibr


source devel/setup.bash

rosrun kalibr kalibr_bagcreater --folder /data --output-bag calib_0.bag

rosrun kalibr kalibr_calibrate_cameras \
    --bag /catkin_ws/calib_0.bag --target /data/target.yaml \
    --models pinhole-radtan pinhole-radtan pinhole-radtan pinhole-radtan\
    --topics /cam0/image_raw /cam1/image_raw /cam2/image_raw /cam3/image_raw --dont-show-report

"""
import time
from datetime import datetime
from pathlib import Path

import pyrealsense2 as rs
import numpy as np
import cv2
from tqdm import tqdm

WIDTH, HEIGHT, FPS = 1920, 1080, 30
ENABLE_DEPTH = False
ENABLE_COLOR = True

base_out_dir = Path("outputs").resolve().absolute()
out_dir = base_out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir.mkdir(parents=True)

def enumerate_devices():
    ctx = rs.context()
    devs = []
    for d in ctx.query_devices():
        serial = d.get_info(rs.camera_info.serial_number)
        product = d.get_info(rs.camera_info.product_line)
        devs.append((serial, product))
    return devs

def start_pipeline(serial: str):
    cfg = rs.config()
    cfg.enable_device(serial)
    if ENABLE_DEPTH:
        cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    if ENABLE_COLOR:
        cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    pipe = rs.pipeline()
    align = rs.align(rs.stream.color) if ENABLE_DEPTH else None
    start_time = time.time()  # Record global time at pipeline start
    pipe.start(cfg)
    return pipe, align, start_time

import threading

def _get_tmstmp(frame):
    return frame.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)

def get_synchronized_frames(pipelines, aligners, system_start_times, n=1000):
    for _ in tqdm(range(n)):
        frame_data = [None] * len(pipelines)

        def grab(idx):
            pipe, align = pipelines[idx], aligners[idx]
            try:
                fs = pipe.wait_for_frames(timeout_ms=5000)
                fs = align.process(fs) if align else fs
                depth_frame = fs.get_depth_frame() if ENABLE_DEPTH else None
                color_frame = fs.get_color_frame()

                depth = np.asanyarray(depth_frame.get_data()) if ENABLE_DEPTH else None
                color = np.asanyarray(color_frame.get_data()) if color_frame else None

                depth_ts = _get_tmstmp(depth_frame) if ENABLE_DEPTH else None
                color_ts = _get_tmstmp(color_frame)

                frame_data[idx] = ((color, color_ts), (depth, depth_ts))
            except Exception as e:
                print(f"Camera {idx} failed to grab frame: {e}")
                frame_data[idx] = ((None, None), (None, None))

        threads = [threading.Thread(target=grab, args=(i,)) for i in range(len(pipelines))]
        [t.start() for t in threads]
        [t.join() for t in threads]

        colors = [fd[0] for fd in frame_data]
        depths = [fd[1] for fd in frame_data]
        yield colors, depths


def main():
    cameras = enumerate_devices()
    if not cameras:
        print("No RealSense devices detected – exiting.")
        return

    print(f"Found {len(cameras)} camera(s):")
    for idx, (serial, product) in enumerate(cameras):
        print(f"   Camera {idx}: {serial}  ({product})")

    pipelines = []
    aligners = []
    system_start_times = []
    for serial, _ in cameras:
        pipe, align, sys_time = start_pipeline(serial)
        pipelines.append(pipe)
        aligners.append(align)
        system_start_times.append(sys_time)

    dirs = [out_dir / f"cam_{i}" for i in range(len(pipelines))]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    time.sleep(4.0)

    try:
        for idx, (colors, depths) in enumerate(get_synchronized_frames(pipelines, aligners, system_start_times)):
            for cam_dir, (color, color_time), (depth, depth_time) in zip(dirs, colors, depths):
                if color is not None and color_time is not None:
                    color_path = cam_dir / f"{color_time}.png"
                    cv2.imwrite(str(color_path), color)

                if depth is not None and depth_time is not None:
                    depth_path = cam_dir / f"{color_time}.png"
                    cv2.imwrite(str(depth_path), depth)
        print("Frames saved successfully.")

    finally:
        for p in pipelines:
            p.stop()

if __name__ == "__main__":
    main()
