"""Shared utility functions for episode management and device enumeration."""
from pathlib import Path
from typing import List, Tuple

import pyrealsense2 as rs

EXPECTED_N_BAGS = 6


def enumerate_devices() -> List[Tuple[str, str]]:
    """Return a list of ``(serial, product_line)`` for all connected RealSense devices."""
    ctx = rs.context()
    devs = []
    for d in ctx.query_devices():
        serial = d.get_info(rs.camera_info.serial_number)
        product = d.get_info(rs.camera_info.product_line)
        devs.append((serial, product))
    return devs


def validate_episode(ep_path: Path, expected_n_bags: int = EXPECTED_N_BAGS) -> Tuple[bool, List[str]]:
    """Verify an episode directory contains the expected raw collection artifacts.

    Returns ``(ok, errors)`` where ``errors`` is a list of human-readable
    descriptions of every problem found. An episode is valid when it has
    exactly ``expected_n_bags`` ``*.bag`` files (excluding ``*.orig.bag``)
    and a ``raw_episode.zarr`` directory.
    """
    errors: List[str] = []
    bags = [p for p in ep_path.glob("*.bag") if not p.stem.endswith(".orig")]
    if len(bags) != expected_n_bags:
        errors.append(f"expected {expected_n_bags} bags, found {len(bags)}")
    if not (ep_path / "raw_episode.zarr").is_dir():
        errors.append("missing raw_episode.zarr")
    return (len(errors) == 0, errors)


def get_latest_ep_path(base_episodes_path: Path, prefix: str) -> Path:
    ep_idxs = [int(x.stem.split("_")[-1]) for x in base_episodes_path.iterdir()]
    ep_idx = 0
    if len(ep_idxs) > 0:
        ep_idx = max(ep_idxs) + 1
    current_episode_name = f"{prefix}_{ep_idx:03d}"
    ep_path = base_episodes_path / current_episode_name
    ep_path.mkdir(exist_ok=False)
    return ep_path
