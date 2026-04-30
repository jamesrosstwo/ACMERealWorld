"""Episode data writer.

:class:`ACMEWriter` writes synchronized multi-camera RGB frames (MP4),
IR stereo pairs (zarr), and robot state (zarr) for a single episode. Frames
and state are buffered in memory during collection; :meth:`ACMEWriter.flush`
performs cross-camera temporal synchronization, aligns state to the synced
reference timestamps, encodes video, and writes metadata.
"""
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import yaml
import zarr
from omegaconf import DictConfig


def compute_sync_window(all_rgb_ts):
    """Compute the common temporal window across all cameras.

    Returns (t0, t1) where t0 is the latest start and t1 is the earliest end
    across all RGB timestamp arrays.
    """
    t0 = max(ts[0] for ts in all_rgb_ts)
    t1 = min(ts[-1] for ts in all_rgb_ts)
    return t0, t1


def align_frames_to_reference(ref_ts, cam_rgb_ts, cam_rgb_frames,
                              cam_ir_left_frames=None, cam_ir_right_frames=None):
    """Align a camera's frames to reference timestamps using nearest-neighbor matching.

    Uses a greedy forward search: for each reference timestamp, finds the closest
    camera frame timestamp (only advancing forward through the camera timestamps).

    Returns (synced_rgb_frames, synced_ir_left_frames, synced_ir_right_frames,
    synced_timestamps). IR lists are None if not provided.
    """
    synced_rgb_frames = []
    synced_ir_left_frames = [] if cam_ir_left_frames is not None else None
    synced_ir_right_frames = [] if cam_ir_right_frames is not None else None
    synced_timestamps = []

    t_idx = 0
    for t in ref_ts:
        while (t_idx + 1 < len(cam_rgb_ts) and
               abs(cam_rgb_ts[t_idx + 1] - t) < abs(cam_rgb_ts[t_idx] - t)):
            t_idx += 1

        synced_rgb_frames.append(cam_rgb_frames[t_idx])
        if synced_ir_left_frames is not None:
            synced_ir_left_frames.append(cam_ir_left_frames[t_idx])
        if synced_ir_right_frames is not None:
            synced_ir_right_frames.append(cam_ir_right_frames[t_idx])
        synced_timestamps.append(cam_rgb_ts[t_idx])

    return synced_rgb_frames, synced_ir_left_frames, synced_ir_right_frames, synced_timestamps


def nearest_neighbor_indices(ref_ts, source_ts):
    """For each reference timestamp, find the index of the nearest source timestamp.

    Uses the same greedy forward search as align_frames_to_reference: the source
    index only advances forward, so the mapping is monotonic.

    Returns a list of integer indices into source_ts (one per ref_ts entry).
    """
    indices = []
    s_idx = 0
    for t in ref_ts:
        while (s_idx + 1 < len(source_ts) and
               abs(source_ts[s_idx + 1] - t) < abs(source_ts[s_idx] - t)):
            s_idx += 1
        indices.append(s_idx)
    return indices


def resample_timeseries(data, target_len):
    """Resample a 1D or 2D time series to target_len using linear interpolation.

    For 2D data, each column is interpolated independently.
    """
    orig_len = len(data)
    if orig_len == target_len:
        return data
    orig_idx = np.linspace(0, 1, orig_len)
    new_idx = np.linspace(0, 1, target_len)
    if data.ndim == 1:
        return np.interp(new_idx, orig_idx, data)
    else:
        interp = np.empty((target_len,) + data.shape[1:], dtype=data.dtype)
        for j in range(data.shape[1]):
            interp[:, j] = np.interp(new_idx, orig_idx, data[:, j])
        return interp


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
            self._ir_left_store = zarr.DirectoryStore(str(self._path / "ir_left.zarr"))
            self._ir_right_store = zarr.DirectoryStore(str(self._path / "ir_right.zarr"))
            self._ir_left_cache = np.zeros((max_episode_len, self._frame_height, self._frame_width), dtype=np.uint8)
            self._ir_right_cache = np.zeros((max_episode_len, self._frame_height, self._frame_width), dtype=np.uint8)
            self.col_tmstmps = np.zeros((max_episode_len,))
            self._rgb_cache = np.zeros((max_episode_len, self._frame_height, self._frame_width, 3), dtype=np.uint8)
            self._save_interval = 1

        def write_frame(self, color, col_tmstmp, ir_left, ir_right):
            if self.highest_written_index >= self._max_episode_len:
                raise IndexError
            self._rgb_cache[self.highest_written_index] = color
            self._ir_left_cache[self.highest_written_index] = ir_left
            self._ir_right_cache[self.highest_written_index] = ir_right
            self.col_tmstmps[self.highest_written_index] = col_tmstmp
            self.highest_written_index += 1

    def __init__(self, path: Path, max_episode_len: int, serials: List[str], instruction: str,
                 captures: DictConfig):
        self.path = path
        assert self.path.exists()
        self.instruction = instruction
        self._max_episode_len = max_episode_len
        self._captures: Dict[str, ACMEWriter._CaptureWriter] = self._init_captures(serials, captures)
        # State is buffered in memory and written to Zarr in flush(). The previous
        # per-step DirectoryStore writes used a single chunk per array, so each
        # call rewrote the whole chunk file from the logging hot path.
        self._state_buffer: Dict[str, list] = {}
        self._state_timestamps: list = []

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

    def write_capture_frame(self, serial: str, col_tmstmp, color, ir_left, ir_right):
        self._captures[serial].write_frame(color, col_tmstmp, ir_left, ir_right)

    def write_state(self, timestamp=None, **state):
        if timestamp is not None:
            self._state_timestamps.append(timestamp)
        for k, v in state.items():
            if k not in self._state_buffer:
                self._state_buffer[k] = []
            if isinstance(v, (int, float)):
                self._state_buffer[k].append(v)
            elif isinstance(v, np.ndarray):
                # Copy to avoid aliasing libfranka / panda-py internal buffers
                # that may be shared across calls.
                self._state_buffer[k].append(v.copy())
            else:
                try:
                    self._state_buffer[k].append(np.asarray(v))
                except ValueError as e:
                    print(f"write_state: cannot convert key={k!r} type={type(v).__name__} repr={v!r}: {e}")
                    raise

    def _persist_raw_state(self):
        """Write the in-memory state buffer to ``raw_episode.zarr`` in one batch.

        Done at flush time (not per call) so the previous single-chunk
        rewrite-on-every-write hot-path cost is avoided.
        """
        if not self._state_buffer and not self._state_timestamps:
            return
        raw_store = zarr.DirectoryStore(str(self.path / "raw_episode.zarr"))
        raw_root = zarr.group(store=raw_store, overwrite=True)
        if self._state_timestamps:
            raw_root.create_dataset(
                "_state_timestamps",
                data=np.asarray(self._state_timestamps),
            )
        for k, values in self._state_buffer.items():
            data = np.stack([np.asarray(v) for v in values])
            raw_root.create_dataset(k, data=data)

    def _load_raw_state(self):
        """Return ``(state_ts, state_data)`` from ``raw_episode.zarr``, or ``(None, None)``."""
        raw_path = self.path / "raw_episode.zarr"
        if not raw_path.is_dir():
            return None, None
        raw_root = zarr.open_group(str(raw_path), mode="r")
        if "_state_timestamps" not in raw_root:
            return None, None
        state_ts = np.asarray(raw_root["_state_timestamps"])
        state_data = {k: np.asarray(raw_root[k]) for k in raw_root if k != "_state_timestamps"}
        return state_ts, state_data

    def flush(self):
        # Persist raw state to disk before any early returns so live collection
        # (where captures are filled by the bag pipeline, not this writer) still
        # leaves recoverable state on disk for postprocessing.
        self._persist_raw_state()

        captures = list(self._captures.values())
        empty_captures = [s for s, c in self._captures.items() if c.highest_written_index == 0]
        if empty_captures:
            print(f"Note: captures {empty_captures} have no in-memory frames, "
                  f"skipping capture sync (raw_episode.zarr written: "
                  f"{(self.path / 'raw_episode.zarr').is_dir()})")
            return

        all_rgb_ts = [c.col_tmstmps[:c.highest_written_index] for c in captures]

        t0, t1 = compute_sync_window(all_rgb_ts)

        ref_ts = all_rgb_ts[0]
        ref_mask = (ref_ts >= t0) & (ref_ts <= t1)
        ref_ts = ref_ts[ref_mask]
        sync_len = len(ref_ts)

        global_synced = []
        for cap in captures:
            rgb_frames = cap._rgb_cache[:cap.highest_written_index]
            ir_left_frames = cap._ir_left_cache[:cap.highest_written_index]
            ir_right_frames = cap._ir_right_cache[:cap.highest_written_index]
            rgb_ts = cap.col_tmstmps[:cap.highest_written_index]

            synced_rgb, synced_ir_left, synced_ir_right, synced_ts = align_frames_to_reference(
                ref_ts, rgb_ts, rgb_frames, ir_left_frames, ir_right_frames
            )
            global_synced.append((cap, synced_rgb, synced_ir_left, synced_ir_right, synced_ts))

        for cap, rgb_frames, ir_left_frames, ir_right_frames, synced_timestamps in global_synced:
            # ir left
            zarr.array(np.array(ir_left_frames, dtype=np.uint8),
                       chunks=(16, None, None),
                       dtype=np.uint8,
                       store=cap._ir_left_store)

            # ir right
            zarr.array(np.array(ir_right_frames, dtype=np.uint8),
                       chunks=(16, None, None),
                       dtype=np.uint8,
                       store=cap._ir_right_store)

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

        synced_store = zarr.DirectoryStore(str(self.path / "episode.zarr"))
        synced_root = zarr.group(store=synced_store)
        # Prefer the in-memory buffer (live-collection path); fall back to
        # raw_episode.zarr on disk (postprocess path, where this writer is
        # freshly constructed and the buffer is empty).
        if self._state_timestamps:
            state_ts = np.asarray(self._state_timestamps)
            state_data = {k: np.stack([np.asarray(v) for v in vs])
                          for k, vs in self._state_buffer.items()}
        else:
            state_ts, state_data = self._load_raw_state()

        if state_ts is not None and len(state_ts) > 0:
            sync_indices = nearest_neighbor_indices(ref_ts, state_ts)
            for key, data in state_data.items():
                synced_root.create_dataset(key, data=data[sync_indices])

        metadata = dict(
            n_timesteps=sync_len,
            instruction=self.instruction,
            dynamic_captures=[217222061106]
        )
        with open(self.episode_path / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f)
