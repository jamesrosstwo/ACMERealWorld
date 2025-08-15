from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import yaml
import zarr
from omegaconf import DictConfig, OmegaConf


class KalibrWriter:
    class _CaptureWriter:
        def __init__(self, path: Path, max_episode_len: int, frame_width: int, frame_height: int):
            self._path = path
            self._path.mkdir()
            self._max_episode_len = max_episode_len
            self._frame_width = frame_width
            self._frame_height = frame_height
            self._highest_seen_index = 0
            self._grayscale_cache = np.zeros((max_episode_len, self._frame_height, self._frame_width), dtype=np.uint8)
            self._timestamps = np.zeros((max_episode_len,), dtype=np.uint64)

        def write_frame(self, timestamp, color):
            self._grayscale_cache[self._highest_seen_index] = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            self._timestamps[self._highest_seen_index] = timestamp
            self._highest_seen_index = self._highest_seen_index + 1

        def flush(self):
            for index in range(self._highest_seen_index):
                rgb_frame: np.ndarray = self._grayscale_cache[index]
                timestamp_us = int(self._timestamps[index]) * 1000000
                out_path = self._path / f"{timestamp_us}.png"
                cv2.imwrite(str(out_path), rgb_frame)

    def __init__(self, episodes_path: str, target: DictConfig, max_episode_len: int, n_cameras: int, captures: DictConfig):
        self._base_episodes_path = Path(episodes_path)
        self._base_episodes_path.mkdir(exist_ok=True, parents=True)

        ep_idxs = [int(x.stem.split("_")[-1]) for x in self._base_episodes_path.iterdir()]
        ep_idx = 0
        if len(ep_idxs) > 0:
            ep_idx = max(ep_idxs) + 1
        current_episode_name = f"calibration_{ep_idx}"
        self.path = self._base_episodes_path / current_episode_name
        self.path.mkdir(exist_ok=False)

        with open(self.path / "target.yaml", "w") as f:
            target_dict = OmegaConf.to_container(target, resolve=True)
            yaml.dump(target_dict, f, default_flow_style=False)

        self._n_cameras = n_cameras
        self._max_episode_len = max_episode_len
        self._captures: List = self._init_captures(captures)

    def _init_captures(self, captures: DictConfig):
        cap_writers = []
        for i in range(self._n_cameras):
            c_path = self.path / f"cam_{i}"
            cw = self._CaptureWriter(c_path, **captures)
            cap_writers.append(cw)
        return cap_writers

    def write_captures_frame(self, timestamps, rgbs):
        for cap, timestamp, rgb in zip(self._captures, timestamps, rgbs):
            cap.write_frame(timestamp, rgb)

    def flush(self):
        # Only captures require flushing at the end of the collection
        for cap in self._captures:
            cap.flush()


class ACMEWriter:
    class _CaptureWriter:
        def __init__(self, path: Path, max_episode_len: int, frame_width: int, frame_height: int, fps: int):
            self._path = path
            self._max_episode_len = max_episode_len
            self._frame_width = frame_width
            self._frame_height = frame_height
            self._highest_seen_timestep = -1
            self._path.mkdir()
            self._rgb_path = self._path / "rgb.mp4"
            self._fps = fps
            self._depth_store = zarr.DirectoryStore(str(self._path / "depth.zarr"))
            self._depth_cache = np.zeros((max_episode_len, self._frame_height, self._frame_width), dtype=np.uint16)
            self._rgb_cache = np.zeros((max_episode_len, self._frame_height, self._frame_width, 3), dtype=np.uint8)

        def write_frame(self, timestep, color, depth):
            if color is not None:
                self._rgb_cache[timestep] = color
            if depth is not None:
                self._depth_cache[timestep] = depth
            self._highest_seen_timestep = max(timestep, self._highest_seen_timestep)

        def flush(self):
            depth_arr = zarr.array(self._depth_cache[:self._highest_seen_timestep],
                                   chunks=(16, None, None),
                                   dtype=np.int16,
                                   store=self._depth_store)
            rgb_data = self._rgb_cache[:self._highest_seen_timestep]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            rgb_out = cv2.VideoWriter(str(self._rgb_path), fourcc, self._fps,
                                      (int(self._frame_width), int(self._frame_height)), True)
            for rgb_frame in rgb_data:
                rgb_out.write(rgb_frame)
            rgb_out.release()

    def __init__(self, episodes_path: str, max_episode_len: int, n_cameras: int, captures: DictConfig):
        self._base_episodes_path = Path(episodes_path)
        self._base_episodes_path.mkdir(exist_ok=True, parents=True)
        ep_idxs = [int(x.stem.split("_")[-1]) for x in self._base_episodes_path.iterdir()]
        ep_idx = 0
        if len(ep_idxs) > 0:
            ep_idx = max(ep_idxs) + 1
        current_episode_name = f"episode_{ep_idx}"
        self._path = self._base_episodes_path / current_episode_name
        self._path.mkdir(exist_ok=False)
        self._store = zarr.DirectoryStore(str(self._path / "episode.zarr"))
        self._root = zarr.group(store=self._store)
        self._n_cameras = n_cameras
        self._max_episode_len = max_episode_len
        self._captures: List = self._init_captures(captures)
        self._highest_seen_timestep = 0
        self._written_timesteps = np.full((self._max_episode_len,), False)

    def _init_captures(self, captures: DictConfig):
        capture_path_base = self._path / "captures"
        capture_path_base.mkdir()

        cap_writers = []
        for i in range(self._n_cameras):
            c_path = capture_path_base / f"capture_{i}"
            cw = self._CaptureWriter(c_path, **captures)
            cap_writers.append(cw)
        return cap_writers

    # Write a single frame for all cameras
    def write_frame(self, timestep, colors, depths):
        for cap, col, dep in zip(self._captures, colors, depths):
            cap.write_frame(timestep, col, dep)
        self._highest_seen_timestep = max(self._highest_seen_timestep, timestep)

    def write_state(self, timestep, **state):
        for k, v in state.items():
            if k not in self._root:
                self._root.create_dataset(
                    name=k,
                    shape=(self._max_episode_len, *v.shape),
                    chunks=(self._max_episode_len, *v.shape),
                    dtype=v.dtype,
                )
            self._root[k][timestep] = v
        self._highest_seen_timestep = max(self._highest_seen_timestep, timestep)

    def flush(self):
        # Only captures require flushing at the end of the collection
        for cap in self._captures:
            cap.flush()
