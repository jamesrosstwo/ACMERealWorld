from pathlib import Path
from typing import List

import numpy as np
import zarr
from omegaconf import DictConfig

class DataWriter:
    class _CaptureWriter:
        def __init__(self, path: Path, frame_width: int, frame_height: int, block_size: int):
            self._path = path
            self._path.mkdir()
            self._block_size = block_size
            self._rgb_path = self._path / "rgb.mp4"
            depth_store = zarr.DirectoryStore(str(self._path / "depth.zarr"))
            self._depth = zarr.zeros(shape=(0, frame_height, frame_width),
                                     chunks=(self._block_size, None, None),
                                     dtype=np.int16,
                                     store=depth_store)
            self._rgb_cache = np.zeros((self._block_size, frame_width, frame_height, 3), dtype=np.uint8)
            self._rgb_cache_mask = np.full((self._block_size), fill_value=False)

        def _dump_rgb_cache(self):
            # TODO: dump block to disk with decord
            self._rgb_cache_mask = np.full((self._block_size), fill_value=False)

        def write_frame(self, timestep, color, depth):
            cache_idx = timestep % self._block_size
            assert not self._rgb_cache_mask[cache_idx]
            self._rgb_cache[cache_idx] = color
            self._rgb_cache_mask[cache_idx] = True
            if cache_idx == self._block_size - 1:
                self._dump_rgb_cache()
            self._depth[timestep] = depth

    def __init__(self, path: Path, max_episode_len: int, n_cameras: int, captures: DictConfig):
        self._path = path
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
