"""Mixed-backend eval camera manager.

Presents the :class:`~client.eval.realsense.EvalRealsense` eval interface over a
heterogeneous set of cameras (RealSense + ZED), preserving the configured
``obs_cams`` order so policy inputs (``rgb_0``, ``rgb_1``, ...) and writer
outputs (``output_video_<idx>``) line up regardless of which backend a given
camera uses.

This is what lets a droid-style ZED Mini wrist camera coexist with RealSense
cameras elsewhere: all RealSense cameras are grouped into a single
:class:`EvalRealsense` (its behavior is unchanged), each ZED gets its own
:class:`~client.eval.zed.EvalZed`, and this manager re-interleaves their
observations back into the global ``obs_cams`` order.

``obs_cams`` is a mapping of *server observation key* -> camera spec, e.g.
``{"observation/wrist_image_left": {serial: wrist_zed, backend: zed, ...}}``.
The key is the rgb key sent to the policy server (and what ``rgb_keys`` is
derived from); the spec selects the physical camera. Each spec may be either:

* a bare serial string (treated as RealSense, backwards compatible); or
* a mapping ``{serial: <str>, backend: realsense|zed, zed_serial: <int?>}``.

A plain list of specs is still accepted for backwards compatibility, in which
case the keys default to ``rgb_0``, ``rgb_1``, ...
"""
from typing import Callable, Dict, List, Optional, Tuple

import torch

from client.eval.realsense import EvalRealsense
from client.eval.zed import EvalZed


def _parse_cam(cam) -> Tuple[str, str, Optional[int]]:
    """Return ``(backend, serial, device_serial)`` for one ``obs_cams`` entry."""
    if isinstance(cam, str):
        return "realsense", cam, None
    serial = str(cam["serial"])
    backend = str(cam.get("backend", "realsense")).lower()
    device_serial = cam.get("zed_serial", None)
    return backend, serial, device_serial


class EvalCameras:
    def __init__(self, n_frames: int, width: int, height: int, fps: int,
                 obs_history: int, obs_cams, laser_power: int = 0):
        self._fps = fps
        self._backends: List = []
        self._serials: List[str] = []

        # obs_cams maps a server rgb key -> camera spec. A plain list is still
        # accepted (keys default to rgb_0, rgb_1, ...). Order is preserved: it
        # drives get_rgb_obs() ordering and the rgb_keys sent to the server.
        if hasattr(obs_cams, "keys"):
            entries = [(str(k), v) for k, v in obs_cams.items()]
        else:
            entries = [(f"rgb_{i}", v) for i, v in enumerate(obs_cams)]
        self._rgb_keys: List[str] = [k for k, _ in entries]

        # global obs index -> (backend_position, local_index_within_backend)
        self._routing: List[Optional[Tuple[int, int]]] = [None] * len(entries)

        rs_serials: List[str] = []
        rs_global_indices: List[int] = []
        zed_entries: List[Tuple[int, EvalZed]] = []

        for global_idx, (_key, cam) in enumerate(entries):
            backend, serial, device_serial = _parse_cam(cam)
            self._serials.append(serial)
            if backend == "realsense":
                rs_global_indices.append(global_idx)
                rs_serials.append(serial)
            elif backend == "zed":
                zed_entries.append((global_idx, EvalZed(
                    serial=serial, n_frames=n_frames, width=width, height=height,
                    fps=fps, obs_history=obs_history, device_serial=device_serial,
                )))
            else:
                raise ValueError(
                    f"Unknown camera backend {backend!r} for serial {serial!r} "
                    f"(expected 'realsense' or 'zed')"
                )

        # All RealSense cameras share one EvalRealsense so the existing
        # multi-cam pipeline/exposure setup is reused verbatim.
        if rs_serials:
            rs_pos = len(self._backends)
            self._backends.append(EvalRealsense(
                n_frames=n_frames, width=width, height=height, fps=fps,
                obs_cams=rs_serials, obs_history=obs_history, laser_power=laser_power,
            ))
            for local_idx, global_idx in enumerate(rs_global_indices):
                self._routing[global_idx] = (rs_pos, local_idx)

        for global_idx, zed in zed_entries:
            pos = len(self._backends)
            self._backends.append(zed)
            self._routing[global_idx] = (pos, 0)

    @property
    def serials(self) -> List[str]:
        return list(self._serials)

    @property
    def rgb_keys(self) -> List[str]:
        """Server observation keys, in obs order (aligns with get_rgb_obs())."""
        return list(self._rgb_keys)

    def start_capture(self, on_receive_frame: Callable = None, on_warmup: Callable = None):
        # Only the first backend runs the warmup hook (e.g. gripper homing) so
        # a side-effecting callback isn't invoked once per backend.
        for i, backend in enumerate(self._backends):
            backend.start_capture(
                on_receive_frame=on_receive_frame,
                on_warmup=on_warmup if i == 0 else None,
            )

    def get_rgb_obs(self) -> List[torch.Tensor]:
        per_backend = [b.get_rgb_obs() for b in self._backends]
        return [per_backend[pos][local] for (pos, local) in self._routing]

    def get_frame_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for backend in self._backends:
            counts.update(backend.get_frame_counts())
        return counts

    def reset_frame_counts(self):
        for backend in self._backends:
            backend.reset_frame_counts()

    def stop_all_captures(self):
        for backend in self._backends:
            backend.stop_all_captures()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_all_captures()
