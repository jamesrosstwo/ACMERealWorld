"""Mixed-backend collection camera manager.

The collection counterpart of :class:`~client.eval.cameras.EvalCameras`, but for
the *recording* interfaces: it composes one
:class:`~client.collect.realsense.RealSenseInterface` (all RealSense cameras,
auto-enumerated and recorded to ``.bag``) with one
:class:`~client.collect.zed.CollectZed` per declared ZED (recorded to ``.svo2``),
and presents the single surface :mod:`client.collect.__main__` already relies on:
``serials``, ``start_capture``, ``get_frame_counts``, ``reset_frame_counts``,
``stop_all_captures`` and the context-manager protocol.

RealSense cameras are listed first, so ``serials[0]`` (the primary serial that
drives state-logging cadence in ``__main__``) stays a RealSense.
"""
from pathlib import Path
from typing import Callable, Dict, List, Optional

from client.collect.realsense import RealSenseInterface
from client.collect.zed import CollectZed


class CollectCameras:
    def __init__(self, path: Path, *, realsense: dict, zed_cameras: Optional[List] = None):
        zed_cameras = list(zed_cameras or [])
        self._backends: List = []
        self._serials: List[str] = []

        # RealSense backend (unchanged: auto-enumerates every connected device).
        self._rs = RealSenseInterface(path, **realsense)
        self._backends.append(self._rs)
        self._serials.extend(self._rs.serials)
        self._n_realsense = len(self._rs.serials)

        # One ZED recorder per declared wrist camera, sharing the RealSense
        # capture dimensions so every capture lands at the same resolution/fps.
        for cam in zed_cameras:
            serial = str(cam["serial"])
            device_serial = cam.get("zed_serial", None)
            self._backends.append(CollectZed(
                path, serial=serial,
                n_frames=realsense["n_frames"], width=realsense["width"],
                height=realsense["height"], fps=realsense["fps"],
                device_serial=device_serial,
            ))
            self._serials.append(serial)
        self._n_zed = len(zed_cameras)

    @property
    def serials(self) -> List[str]:
        return list(self._serials)

    @property
    def n_realsense(self) -> int:
        return self._n_realsense

    @property
    def n_zed(self) -> int:
        return self._n_zed

    def start_capture(self, on_receive_frame: Callable = None, on_warmup: Callable = None):
        # Only the first backend (RealSense) runs the warmup hook so a
        # side-effecting callback isn't invoked once per backend.
        for i, backend in enumerate(self._backends):
            backend.start_capture(
                on_receive_frame=on_receive_frame,
                on_warmup=on_warmup if i == 0 else None,
            )

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
