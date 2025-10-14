import shutil
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import yaml
import zarr
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


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
            if self._highest_seen_index >= self._max_episode_len:
                raise IndexError
            self._grayscale_cache[self._highest_seen_index] = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            self._timestamps[self._highest_seen_index] = timestamp
            self._highest_seen_index = self._highest_seen_index + 1

        def flush(self):
            for index in range(self._highest_seen_index):
                rgb_frame: np.ndarray = self._grayscale_cache[index]
                timestamp_us = int(self._timestamps[index]) * 1000000
                out_path = self._path / f"{timestamp_us}.png"
                cv2.imwrite(str(out_path), rgb_frame)

    def __init__(self, episodes_path: str, target: DictConfig, max_episode_len: int, n_cameras: int,
                 captures: DictConfig):
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

    def write_capture_frame(self, cap_idx, timestamp, rgb):
        self._captures[cap_idx].write_frame(timestamp, rgb)

    def flush(self):
        # Only captures require flushing at the end of the collection
        print("FLUSHING CAPTURES")
        for cap in self._captures:
            cap.flush()


class ACMEWriter:
    class _CaptureWriter:
        def __init__(self, path: Path, max_episode_len: int, frame_width: int, frame_height: int, fps: int,
                     save_interval: int = 1):
            self._path = path
            self._max_episode_len = max_episode_len
            self._frame_width = frame_width
            self._frame_height = frame_height
            self.highest_written_index = 0
            self._path.mkdir()
            self._rgb_path = self._path / "rgb.mp4"
            self._fps = fps
            self._depth_store = zarr.DirectoryStore(str(self._path / "depth.zarr"))
            self._depth_cache = np.zeros((max_episode_len, self._frame_height, self._frame_width), dtype=np.uint16)
            self.col_tmstmps = np.zeros((max_episode_len,))
            self.depth_tmstmps = np.zeros((max_episode_len,))
            self._rgb_cache = np.zeros((max_episode_len, self._frame_height, self._frame_width, 3), dtype=np.uint8)
            self._save_interval = 1

        def write_frame(self, color, col_tmstmp, depth, depth_tmstmp):
            if self.highest_written_index >= self._max_episode_len:
                raise IndexError
            self._rgb_cache[self.highest_written_index] = color
            self._depth_cache[self.highest_written_index] = depth
            self.col_tmstmps[self.highest_written_index] = col_tmstmp
            self.depth_tmstmps[self.highest_written_index] = depth_tmstmp
            self.highest_written_index += 1

    def __init__(self, path: Path, max_episode_len: int, n_cameras: int, instruction: str, calibration_path: str,
                 captures: DictConfig):
        self.path = path
        assert self.path.exists()
        self.instruction = instruction
        self.calibration_path = calibration_path
        self._store = zarr.DirectoryStore(str(self.path / "episode.zarr"))
        self._root = zarr.group(store=self._store)
        self._n_cameras = n_cameras
        self._max_episode_len = max_episode_len
        self._captures: List = self._init_captures(captures)
        self._state_write_counter = 0

    @property
    def episode_path(self):
        return self.path

    def _init_captures(self, captures: DictConfig):
        capture_path_base = self.path / "captures"
        capture_path_base.mkdir()

        cap_writers = []
        for i in range(self._n_cameras):
            c_path = capture_path_base / f"capture_{i}"
            cw = self._CaptureWriter(c_path, **captures)
            cap_writers.append(cw)
        return cap_writers

    def write_frame(self, colors, depths):
        for cap, col, dep in zip(self._captures, colors, depths):
            cap.write_frame(col, dep)

    def write_capture_frame(self, capture_index, col_tmstmp, depth_tmstmp, color, depth):
        self._captures[capture_index].write_frame(color, col_tmstmp, depth, depth_tmstmp)

    def write_state(self, **state):
        for k, v in state.items():
            if k not in self._root:
                if isinstance(v, (float, int)):
                    self._root.create_dataset(
                        name=k,
                        shape=(self._max_episode_len,),
                        chunks=(self._max_episode_len,),
                    )
                else:
                    self._root.create_dataset(
                        name=k,
                        shape=(self._max_episode_len, *v.shape),
                        chunks=(self._max_episode_len, *v.shape),
                        dtype=v.dtype,
                    )
            self._root[k][self._state_write_counter] = v
        self._state_write_counter += 1

    def flush(self):
        try:
            # We write state on the fly, only captures require flushing at the end of the collection
            all_rgb_ts = [c.col_tmstmps[:c.highest_written_index] for c in self._captures]
            all_depth_ts = [c.depth_tmstmps[:c.highest_written_index] for c in self._captures]

            t0 = max(
                max(ts[0] for ts in all_rgb_ts),
                max(ts[0] for ts in all_depth_ts)
            )
            t1 = min(
                min(ts[-1] for ts in all_rgb_ts),
                min(ts[-1] for ts in all_depth_ts)
            )

            ref_ts = all_rgb_ts[0]
            ref_ts = ref_ts[(ref_ts >= t0) & (ref_ts <= t1)]
            sync_len = len(ref_ts)

            global_synced = []
            for cap in self._captures:
                depth_frames = cap._depth_cache[:cap.highest_written_index]
                rgb_frames = cap._rgb_cache[:cap.highest_written_index]
                depth_ts = cap.depth_tmstmps[:cap.highest_written_index]
                rgb_ts = cap.col_tmstmps[:cap.highest_written_index]

                synced_rgb_frames = []
                synced_depth_frames = []
                synced_timestamps = []

                d_idx, r_idx = 0, 0
                for t in ref_ts:
                    # nearest RGB
                    while (r_idx + 1 < len(rgb_ts) and
                           abs(rgb_ts[r_idx + 1] - t) < abs(rgb_ts[r_idx] - t)):
                        r_idx += 1
                    # nearest depth
                    while (d_idx + 1 < len(depth_ts) and
                           abs(depth_ts[d_idx + 1] - t) < abs(depth_ts[d_idx] - t)):
                        d_idx += 1

                    synced_rgb_frames.append(rgb_frames[r_idx])
                    synced_depth_frames.append(depth_frames[d_idx])
                    synced_timestamps.append((rgb_ts[r_idx], depth_ts[d_idx]))

                global_synced.append((cap, synced_rgb_frames, synced_depth_frames, synced_timestamps))

            for cap, rgb_frames, depth_frames, synced_timestamps in global_synced:
                # depth
                zarr.array(np.array(depth_frames, dtype=np.int16),
                           chunks=(16, None, None),
                           dtype=np.int16,
                           store=cap._depth_store)

                # rgb
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                rgb_out = cv2.VideoWriter(str(cap._rgb_path), fourcc, cap._fps,
                                          (int(cap._frame_width), int(cap._frame_height)), True)
                for frame in rgb_frames:
                    rgb_out.write(frame)
                rgb_out.release()

                # timestamps
                with open(str(cap._path / "timestamps.npz"), "wb") as f:
                    np.savez_compressed(f,
                                        color=[c for c, _ in synced_timestamps],
                                        depth=[d for _, d in synced_timestamps])

            state = zarr.open_group(str(self.path / "episode.zarr"), mode="r+")
            orig_len = len(next(iter(state.values())))
            if orig_len != sync_len:
                orig_idx = np.linspace(0, 1, orig_len)
                new_idx = np.linspace(0, 1, sync_len)
                for key, arr in list(state.items()):
                    data = np.array(arr)

                    state.create_dataset(f"{key}_original", data=data)
                    del state[key]

                    if data.ndim == 1:
                        interp = np.interp(new_idx, orig_idx, data)
                    else:
                        interp = np.empty((sync_len,) + data.shape[1:], dtype=data.dtype)
                        for j in range(data.shape[1]):
                            interp[:, j] = np.interp(new_idx, orig_idx, data[:, j])

                    state.create_dataset(f"{key}", data=interp)

            metadata = dict(
                n_timesteps=sync_len,
                instruction=self.instruction
            )
            with open(self.episode_path / "metadata.yaml", "w") as f:
                yaml.dump(metadata, f)

            shutil.copy(str(self.calibration_path), self.episode_path / "params.yaml")
        except IndexError:
            pass
