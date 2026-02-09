"""Episode data writer.

:class:`ACMEWriter` writes synchronized multi-camera RGB frames (MP4),
depth maps (zarr), and robot state (zarr) for a single episode. During
collection frames are buffered in memory; :meth:`ACMEWriter.flush` performs
cross-camera temporal synchronization, encodes video, and writes metadata.
"""
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import yaml
import zarr
from omegaconf import DictConfig


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

    def __init__(self, path: Path, max_episode_len: int, serials: List[str], instruction: str,
                 captures: DictConfig):
        self.path = path
        assert self.path.exists()
        self.instruction = instruction
        self._store = zarr.DirectoryStore(str(self.path / "episode.zarr"))
        self._root = zarr.group(store=self._store)
        self._max_episode_len = max_episode_len
        self._captures: Dict[str, ACMEWriter._CaptureWriter] = self._init_captures(serials, captures)
        self._state_write_counter = 0

    @property
    def episode_path(self):
        return self.path

    def _init_captures(self, serials: List[str], captures: DictConfig):
        capture_path_base = self.path / "captures"
        capture_path_base.mkdir()

        cap_writers = {}
        for serial in serials:
            c_path = capture_path_base / f"capture_{serial}"
            cw = self._CaptureWriter(c_path, **captures)
            cap_writers[serial] = cw
        return cap_writers

    def write_capture_frame(self, serial: str, col_tmstmp, depth_tmstmp, color, depth):
        self._captures[serial].write_frame(color, col_tmstmp, depth, depth_tmstmp)

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
        # We write state on the fly, only captures require flushing at the end of the collection
        captures = list(self._captures.values())
        empty_captures = [s for s, c in self._captures.items() if c.highest_written_index == 0]
        if empty_captures:
            print(f"Warning: captures {empty_captures} have no frames, skipping flush")
            return

        all_rgb_ts = [c.col_tmstmps[:c.highest_written_index] for c in captures]
        all_depth_ts = [c.depth_tmstmps[:c.highest_written_index] for c in captures]

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
        for cap in captures:
            depth_frames = cap._depth_cache[:cap.highest_written_index]
            rgb_frames = cap._rgb_cache[:cap.highest_written_index]
            depth_ts = cap.depth_tmstmps[:cap.highest_written_index]
            rgb_ts = cap.col_tmstmps[:cap.highest_written_index]

            synced_rgb_frames = []
            synced_depth_frames = []
            synced_timestamps = []

            t_idx = 0
            for t in ref_ts:
                while (t_idx + 1 < len(rgb_ts) and
                       abs(rgb_ts[t_idx + 1] - t) < abs(rgb_ts[t_idx] - t)):
                    t_idx += 1

                synced_rgb_frames.append(rgb_frames[t_idx])
                # Assume depth and rgb are captured simultaneously (usually <1ms off in practice)
                synced_depth_frames.append(depth_frames[t_idx])
                synced_timestamps.append(rgb_ts[t_idx])

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
            with open(str(cap._path / "timestamps.npy"), "wb") as f:
                np.savez(f, np.asarray(synced_timestamps))

        state = zarr.open_group(str(self.path / "episode.zarr"), mode="r+")
        orig_len = len(next(iter(state.values())))
        # Aligns timesteps when some cameras are slightly slower
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
            instruction=self.instruction,
            dynamic_captures=[5]
        )
        with open(self.episode_path / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f)
