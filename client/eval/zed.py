"""Evaluation-mode ZED (Stereolabs) camera interface.

:class:`EvalZed` is the droid-style wrist-camera counterpart to
:class:`~client.eval.realsense.EvalRealsense`: it captures RGB-only frames from
a single ZED (e.g. a ZED Mini) and caches recent observations in a fixed-size
deque for low-latency policy inference. The ZED ``LEFT`` view is retrieved as
BGRA and sliced to BGR so the frame format matches RealSense's ``bgr8`` exactly
-- the policy preprocessing (BGR->RGB) is then identical regardless of which
camera produced the frame. Depth is disabled; eval only needs RGB.

This implements the subset of the :class:`EvalRealsense` interface used by the
eval loop and :class:`~client.eval.cameras.EvalCameras` manager: ``serials``,
``start_capture``, ``get_rgb_obs``, ``get_frame_counts``,
``reset_frame_counts``, ``stop_all_captures`` and the context-manager protocol.
"""
import os
import threading
import traceback
from collections import deque
from typing import Callable, List, Optional

import numpy as np
import torch

# Default user-writable dir for ZED factory calibration (SN<serial>.conf). The
# SDK's default (/usr/local/zed/settings) is root:zed and usually not writable,
# so it can't auto-download calibration there; pointing the SDK here via
# InitParameters.optional_settings_path lets open() find/fetch calibration
# without sudo. Pre-seed it with: curl -sSL -o <dir>/SN<serial>.conf \
#   "https://calib.stereolabs.com/?SN=<serial>"
DEFAULT_ZED_SETTINGS_PATH = os.path.expanduser("~/.config/stereolabs/zed")

try:
    import pyzed.sl as sl
except ImportError as e:  # pragma: no cover - depends on the ZED SDK install
    sl = None
    _ZED_IMPORT_ERROR: Optional[ImportError] = e
else:
    _ZED_IMPORT_ERROR = None


def _resolution_for(width: int, height: int):
    """Map a (width, height) to the matching fixed ZED resolution mode.

    ZED resolutions are discrete stereo modes; the per-eye ``LEFT`` image we
    retrieve has exactly these dimensions. Falls back to HD720 (the ZED Mini's
    native 1280x720) with a warning if no exact match exists.
    """
    table = {
        (2208, 1242): sl.RESOLUTION.HD2K,
        (1920, 1080): sl.RESOLUTION.HD1080,
        (1280, 720): sl.RESOLUTION.HD720,
        (672, 376): sl.RESOLUTION.VGA,
    }
    res = table.get((width, height))
    if res is None:
        print(f"[zed] No exact ZED resolution for {width}x{height}; "
              f"falling back to HD720 (1280x720).")
        res = sl.RESOLUTION.HD720
    return res


class EvalZed:
    def __init__(self, serial: str, n_frames: int, width: int, height: int,
                 fps: int, obs_history: int, device_serial: Optional[int] = None,
                 settings_path: Optional[str] = None):
        """
        Args:
            serial: Stable label keying this camera in writer output
                (``output_video_<idx>``) and frame counts. Need not be the ZED
                hardware serial -- it is positional within ``obs_cams``.
            device_serial: Optional ZED hardware serial to select a specific
                camera. If ``None``, the first available ZED is opened (the
                common single-wrist-camera case).
            settings_path: Directory holding ZED factory calibration
                (``SN<serial>.conf``). Defaults to a user-writable location so
                open() can find/fetch calibration without write access to the
                root-owned ``/usr/local/zed/settings``.
        """
        if sl is None:
            raise ImportError(
                "pyzed is required for ZED cameras but could not be imported. "
                "Install the ZED SDK and its Python API (pyzed). "
                f"Original error: {_ZED_IMPORT_ERROR}"
            )
        self._serial = str(serial)
        self._n_frames = n_frames
        self._width = width
        self._height = height
        self._fps = fps
        self._obs_history = obs_history
        self._device_serial = device_serial
        self._settings_path = settings_path or DEFAULT_ZED_SETTINGS_PATH
        self._counts_lock = threading.Lock()
        self.frame_counts = {self._serial: 0}
        self._cam = sl.Camera()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cache_lock = threading.Lock()
        self._cache = deque(maxlen=obs_history)

    @property
    def serials(self) -> List[str]:
        return [self._serial]

    def _open(self):
        init = sl.InitParameters()
        init.camera_resolution = _resolution_for(self._width, self._height)
        init.camera_fps = self._fps
        # Eval needs RGB only; disabling depth saves GPU/latency on the wrist.
        init.depth_mode = sl.DEPTH_MODE.NONE
        init.coordinate_units = sl.UNIT.METER
        # Read/auto-download factory calibration from a user-writable dir so
        # open() doesn't fail with "CALIBRATION FILE NOT AVAILABLE" when the
        # default root-owned settings dir isn't writable.
        os.makedirs(self._settings_path, exist_ok=True)
        init.optional_settings_path = self._settings_path
        if self._device_serial is not None:
            init.set_from_serial_number(int(self._device_serial))
        err = self._cam.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera '{self._serial}': {err}")
        hw_serial = self._cam.get_camera_information().serial_number
        print(f"[zed] Opened ZED hw_serial={hw_serial} as '{self._serial}' "
              f"({self._width}x{self._height}@{self._fps}, LEFT view, depth off)")

    def start_capture(self, on_receive_frame: Callable = None, on_warmup: Callable = None):
        self._open()
        runtime = sl.RuntimeParameters()
        image = sl.Mat()

        # Run the warmup hook (e.g. gripper homing) concurrently with the
        # warmup drain, then drain frames so auto-exposure settles before any
        # frame is counted -- mirrors the RealSense warmup phase.
        if on_warmup is not None:
            on_warmup()
        for _ in range(20):
            self._cam.grab(runtime)

        def _run():
            while not self._stop_event.is_set():
                try:
                    if self._cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                        continue
                    self._cam.retrieve_image(image, sl.VIEW.LEFT)
                    # ZED returns BGRA; drop alpha to match RealSense bgr8 so
                    # downstream preprocessing (BGR->RGB) is backend-agnostic.
                    bgra = image.get_data()
                    bgr = np.ascontiguousarray(bgra[:, :, :3])
                    with self._cache_lock:
                        self._cache.append(torch.tensor(bgr))
                    if on_receive_frame is not None:
                        on_receive_frame(self._serial, bgr)
                    with self._counts_lock:
                        self.frame_counts[self._serial] += 1
                        if self.frame_counts[self._serial] >= self._n_frames:
                            print("stopping capture", self._serial)
                            self._stop_event.set()
                except Exception as e:
                    print(f"ZED camera {self._serial} failed to grab frame: {e}")
                    traceback.print_exc()
            print(f"Stopping ZED capture pipeline {self._serial}")
            self._cam.close()

        self._thread = threading.Thread(target=_run)
        self._thread.start()

    def get_rgb_obs(self) -> List[torch.Tensor]:
        with self._cache_lock:
            return [torch.stack(list(self._cache))]

    def get_frame_counts(self):
        with self._counts_lock:
            return dict(self.frame_counts)

    def reset_frame_counts(self):
        with self._counts_lock:
            self.frame_counts[self._serial] = 0

    def stop_all_captures(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        print("ZED capture stopped.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_all_captures()
