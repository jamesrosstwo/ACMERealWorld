#!/usr/bin/env python3
"""
multi_realsense_sync.py
Grab synchronized frames from *all* attached RealSense cameras.
"""

import time
from datetime import datetime
from pathlib import Path

import pyrealsense2 as rs
import numpy as np

WIDTH, HEIGHT, FPS = 1280, 720, 15
ENABLE_DEPTH = True
ENABLE_COLOR = True

base_out_dir = Path("outputs").resolve().absolute()
out_dir = base_out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir.mkdir(parents=True)

def enumerate_devices():
    """Return list of (serial, product_line) for every connected camera."""
    ctx = rs.context()
    devs = []
    for d in ctx.query_devices():
        serial = d.get_info(rs.camera_info.serial_number)
        product = d.get_info(rs.camera_info.product_line)
        devs.append((serial, product))
    return devs

def start_pipeline(serial: str):
    """Create and start a pipeline for one camera with given serial."""
    cfg = rs.config()
    cfg.enable_device(serial)

    if ENABLE_DEPTH:
        cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    if ENABLE_COLOR:
        cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    pipe = rs.pipeline()
    align = rs.align(rs.stream.color)  # align depth to color
    profile = pipe.start(cfg)
    return pipe, align


def get_single_frames(pipelines, aligners):
    framesets = []
    for pipe, align in zip(pipelines, aligners):
        fs = pipe.wait_for_frames(timeout_ms=5000)
        fs = align.process(fs)
        framesets.append(fs)


    depths = []
    colors = []
    for cam_idx, fs in enumerate(framesets):
        depth = np.asanyarray(fs.get_depth_frame().get_data())
        color = np.asanyarray(fs.get_color_frame().get_data())
        depths.append(depth)
        colors.append(color)

    return colors, depths

def main():
    cameras = enumerate_devices()
    if not cameras:
        print("No RealSense devices detected – exiting.")
        return

    print(f"Found {len(cameras)} camera(s):")
    for serial, product in cameras:
        print(f"   {serial}  ({product})")

    # Create a pipeline + align object for every camera
    pipelines   = []
    aligners    = []
    for serial, _ in cameras:
        pipe, align = start_pipeline(serial)
        pipelines.append(pipe)
        aligners.append(align)

    # Warm-up auto-exposure etc.
    time.sleep(1.0)

    try:

        dirs = [out_dir / f"cam_{i}" for i in range(len(pipelines))]
        for dir, color, depth in zip(dirs, get_single_frames(pipelines, aligners)):


    finally:
        for p in pipelines:
            p.stop()

if __name__ == "__main__":
    main()