"""Shared utility functions for episode management and device enumeration."""
from pathlib import Path
from typing import List, Tuple

import pyrealsense2 as rs


def enumerate_devices() -> List[Tuple[str, str]]:
    """Return a list of ``(serial, product_line)`` for all connected RealSense devices."""
    ctx = rs.context()
    devs = []
    for d in ctx.query_devices():
        serial = d.get_info(rs.camera_info.serial_number)
        product = d.get_info(rs.camera_info.product_line)
        devs.append((serial, product))
    return devs


def get_latest_ep_path(base_episodes_path: Path, prefix: str) -> Path:
    ep_idxs = [int(x.stem.split("_")[-1]) for x in base_episodes_path.iterdir()]
    ep_idx = 0
    if len(ep_idxs) > 0:
        ep_idx = max(ep_idxs) + 1
    current_episode_name = f"{prefix}_{ep_idx}"
    ep_path = base_episodes_path / current_episode_name
    ep_path.mkdir(exist_ok=False)
    return ep_path
