"""ZED (Stereolabs) capture and SVO playback for data collection.

The collection counterpart of :mod:`client.collect.realsense`. Where RealSense
cameras record to ``.bag`` and are decoded offline by :class:`RSBagProcessor`,
the wrist ZED records to ``.svo2`` and is decoded offline by
:class:`ZEDSvoProcessor` -- both feed the same ``(color, ts, ir_left, ir_right,
serial)`` tuple into :class:`~client.collect.write.ACMEWriter`, so the dataset
format is identical regardless of which backend produced a capture.

Unlike :class:`~client.eval.zed.EvalZed` (which only caches the most recent
frames for low-latency inference), :class:`CollectZed` persists *every* frame to
the SVO so the full episode is recoverable in postprocessing.

Timestamps use ``TIME_REFERENCE.IMAGE`` in milliseconds since the Unix epoch --
the same host-referenced wall clock RealSense reports under global-time mode --
so the cross-camera sync in :class:`~client.collect.write.ACMEWriter` lines the
wrist frames up with the arm cameras and robot state.
"""
import os
import threading
import traceback
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import yaml

# Reuse the eval ZED helpers (resolution table + user-writable calibration dir)
# so the open path behaves identically across collect and eval.
from client.eval.zed import _resolution_for, DEFAULT_ZED_SETTINGS_PATH

try:
    import pyzed.sl as sl
except ImportError as e:  # pragma: no cover - depends on the ZED SDK install
    sl = None
    _ZED_IMPORT_ERROR: Optional[ImportError] = e
else:
    _ZED_IMPORT_ERROR = None


def _require_sl():
    if sl is None:
        raise ImportError(
            "pyzed is required for ZED cameras but could not be imported. "
            "Install the ZED SDK and its Python API (pyzed). "
            f"Original error: {_ZED_IMPORT_ERROR}"
        )


def _bgr_from_view(image: "sl.Mat") -> np.ndarray:
    """ZED retrieve_image LEFT is BGRA; drop alpha to match RealSense bgr8."""
    bgra = image.get_data()
    return np.ascontiguousarray(bgra[:, :, :3])


def _gray_from_view(image: "sl.Mat") -> np.ndarray:
    """A ``*_GRAY`` view is single-channel; return a contiguous (H, W) uint8
    array matching the RealSense IR frame shape."""
    data = image.get_data()
    return np.ascontiguousarray(np.squeeze(data)).astype(np.uint8)


class CollectZed:
    """Record a single ZED to an ``.svo2`` for the duration of an episode.

    Implements the subset of the :class:`~client.collect.realsense.RealSenseInterface`
    surface that :mod:`client.collect.__main__` (via
    :class:`~client.collect.cameras.CollectCameras`) relies on: ``serials``,
    ``start_capture``, ``get_frame_counts``, ``reset_frame_counts``,
    ``stop_all_captures`` and the context-manager protocol.
    """

    class _GrabThread(threading.Thread):
        def __init__(self, parent: "CollectZed"):
            super().__init__()
            self._parent = parent

        def run(self):
            p = self._parent
            runtime = sl.RuntimeParameters()
            while not p._stop_event.is_set():
                try:
                    if p._cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                        continue
                    # grab() writes the frame to the SVO automatically while
                    # recording is enabled; we only need a timestamp + count.
                    ts_ms = p._cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
                    if p._on_receive_frame is not None:
                        p._on_receive_frame(p._serial, ts_ms)
                    with p._counts_lock:
                        p.frame_counts[p._serial] += 1
                        if p.frame_counts[p._serial] >= p._n_frames:
                            print("stopping capture", p._serial)
                            p._stop_event.set()
                except Exception as e:
                    print(f"ZED camera {p._serial} failed to grab frame: {e}")
                    traceback.print_exc()
            print(f"Stopping ZED capture pipeline {p._serial}")
            p._cam.disable_recording()
            p._cam.close()

    def __init__(self, path: Path, *, serial: str, n_frames: int, width: int,
                 height: int, fps: int, device_serial: Optional[int] = None,
                 settings_path: Optional[str] = None):
        _require_sl()
        self._path = Path(path)
        self._serial = str(serial)
        self._n_frames = n_frames
        self._width = width
        self._height = height
        self._fps = fps
        self._device_serial = device_serial
        self._settings_path = settings_path or DEFAULT_ZED_SETTINGS_PATH
        self._svo_path = str(self._path / f"{self._serial}.svo2")
        self._counts_lock = threading.Lock()
        self.frame_counts = {self._serial: 0}
        self._cam = sl.Camera()
        self._stop_event = threading.Event()
        self._thread: Optional[CollectZed._GrabThread] = None
        self._on_receive_frame: Optional[Callable] = None

    @property
    def serials(self) -> List[str]:
        return [self._serial]

    def _open(self):
        init = sl.InitParameters()
        init.camera_resolution = _resolution_for(self._width, self._height)
        init.camera_fps = self._fps
        # Recording the rectified LEFT/RIGHT pair does not require depth; keep it
        # off to save GPU/latency. Rectification is independent of depth mode.
        init.depth_mode = sl.DEPTH_MODE.NONE
        init.coordinate_units = sl.UNIT.METER
        os.makedirs(self._settings_path, exist_ok=True)
        init.optional_settings_path = self._settings_path
        if self._device_serial is not None:
            init.set_from_serial_number(int(self._device_serial))
        err = self._cam.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera '{self._serial}': {err}")
        hw_serial = self._cam.get_camera_information().serial_number
        print(f"[zed] Opened ZED hw_serial={hw_serial} as '{self._serial}' "
              f"({self._width}x{self._height}@{self._fps}, recording stereo)")

    def start_capture(self, on_receive_frame: Callable = None, on_warmup: Callable = None):
        self._on_receive_frame = on_receive_frame
        self._open()

        # Run the warmup hook (e.g. gripper homing) concurrently with the warmup
        # drain, then drain frames so auto-exposure settles -- mirrors the
        # RealSense warmup phase. Recording is enabled *after* the drain (as
        # RealSense pauses its recorder during warmup) so warmup frames aren't
        # written to the SVO.
        runtime = sl.RuntimeParameters()
        if on_warmup is not None:
            on_warmup()
        for _ in range(20):
            self._cam.grab(runtime)

        rec_params = sl.RecordingParameters(self._svo_path, sl.SVO_COMPRESSION_MODE.H264)
        err = self._cam.enable_recording(rec_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to enable ZED recording to {self._svo_path}: {err}")

        self._thread = CollectZed._GrabThread(self)
        self._thread.start()

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


def _zed_calibration(cam: "sl.Camera") -> dict:
    """Build a calibration dict in the same schema RealSense captures use
    (consumed by scripts/foundation_stereo.py and scripts/diagnose_alignment.py).

    rgb.mp4 is the rectified LEFT view, so the "color" and "ir" cameras are the
    same physical sensor -> color==ir1, T_color_to_ir1 is identity, and depth is
    computed in the rgb frame directly. The stereo baseline goes into
    T_ir1_to_ir2 (foundation_stereo reads ``abs(T_ir1_to_ir2[0][3])``).
    """
    calib = cam.get_camera_information().camera_configuration.calibration_parameters
    left = calib.left_cam
    intr = [float(left.fx), float(left.fy), float(left.cx), float(left.cy)]

    # Baseline magnitude from the stereo transform's x-translation. ZED reports
    # calibration translation in millimetres; convert to metres. (Values > ~1
    # are unambiguously mm for any real baseline.)
    tx = float(calib.stereo_transform.get_translation().get()[0])
    baseline_m = abs(tx)
    if baseline_m > 1.0:
        baseline_m /= 1000.0

    T_ir1_to_ir2 = np.eye(4)
    T_ir1_to_ir2[0, 3] = -baseline_m
    identity = np.eye(4)

    return {
        "intrinsics": {
            "depth": intr,
            "color": intr,
            "ir": intr,
        },
        "extrinsics": {
            "T_ir1_to_ir2": T_ir1_to_ir2.tolist(),
            "T_color_to_ir1": identity.tolist(),
            "T_depth_to_color": identity.tolist(),
        },
    }


class ZEDSvoProcessor:
    """Decode recorded ``.svo2`` files into the same per-frame tuple
    :class:`~client.collect.realsense.RSBagProcessor` yields, so
    :mod:`client.collect.postprocess` can treat both backends uniformly.

    Accepts the same ``(n_frames, width, height, fps)`` keyword set as
    ``RSBagProcessor`` (``n_frames``/``width``/``height``/``fps`` are accepted
    for signature parity; the SVO carries its own dimensions and frame count).
    """

    def __init__(self, svo_paths: List[Path], n_frames: int = 0, width: int = 0,
                 height: int = 0, fps: int = 0, settings_path: Optional[str] = None):
        _require_sl()
        self.svo_paths = svo_paths
        # Factory-calibration dir (SN<serial>.conf). Without it open() tries to
        # download the calibration and fails offline -- same fix EvalZed/CollectZed
        # apply to live capture.
        self._settings_path = settings_path or DEFAULT_ZED_SETTINGS_PATH

    def process_all_frames(self):
        for svo_path in self.svo_paths:
            serial = svo_path.stem
            try:
                yield from self.process_frames_for_svo(serial, svo_path)
            except RuntimeError as e:
                print(f"Skipping svo {svo_path.name}: {e}")
                continue

    def extract_calibration(self, cam: "sl.Camera", serial: str, svo_path: Path):
        cap_dir = svo_path.parent / "captures" / f"capture_{serial}"
        cap_dir.mkdir(parents=True, exist_ok=True)
        with open(cap_dir / "calibration.yaml", "w") as f:
            yaml.dump(_zed_calibration(cam), f, default_flow_style=None, sort_keys=False)

    def process_frames_for_svo(self, serial: str, svo_path: Path):
        cam = sl.Camera()
        init = sl.InitParameters()
        init.set_from_svo_file(str(svo_path))
        init.svo_real_time_mode = False
        init.depth_mode = sl.DEPTH_MODE.NONE
        init.coordinate_units = sl.UNIT.METER
        os.makedirs(self._settings_path, exist_ok=True)
        init.optional_settings_path = self._settings_path
        err = cam.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open SVO {svo_path}: {err}")

        # calibration.yaml is written here (not by the live recorder) because the
        # SVO embeds the rectified calibration and postprocess regenerates the
        # captures/ tree from scratch.
        self.extract_calibration(cam, serial, svo_path)

        runtime = sl.RuntimeParameters()
        left, left_gray, right_gray = sl.Mat(), sl.Mat(), sl.Mat()
        try:
            while True:
                err = cam.grab(runtime)
                if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    break
                if err != sl.ERROR_CODE.SUCCESS:
                    continue
                cam.retrieve_image(left, sl.VIEW.LEFT)
                cam.retrieve_image(left_gray, sl.VIEW.LEFT_GRAY)
                cam.retrieve_image(right_gray, sl.VIEW.RIGHT_GRAY)
                color = _bgr_from_view(left)
                ir_left = _gray_from_view(left_gray)
                ir_right = _gray_from_view(right_gray)
                ts_ms = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
                yield color, ts_ms, ir_left, ir_right, serial
        finally:
            cam.close()
