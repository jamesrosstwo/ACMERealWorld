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
import pyrealsense2 as rs


def enumerate_devices():
    ctx = rs.context()
    devs = []
    for d in ctx.query_devices():
        serial = d.get_info(rs.camera_info.serial_number)
        product = d.get_info(rs.camera_info.product_line)
        devs.append((serial, product))
    return devs

def start_pipeline(serial: str, w: int, h: int, fps: int):
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    pipe = rs.pipeline()
    pipe.start(cfg)
    return pipe


def get_tmstmp(frame):
    return frame.get_frame_metadata(rs.frame_metadata_value.backend_timestamp)
