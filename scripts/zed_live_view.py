"""Live preview of the ZED wrist camera's LEFT view (eval-equivalent frame).

Opens the ZED with the *same* InitParameters as the eval pipeline
(:class:`~client.eval.zed.EvalZed`): the configured fixed resolution, depth
disabled, and ``optional_settings_path`` pointed at a user-writable dir so
factory calibration can auto-download (avoids "CALIBRATION FILE NOT AVAILABLE"
against the root-owned ``/usr/local/zed/settings``). It retrieves ``VIEW.LEFT``,
slices BGRA->BGR exactly as eval does, and shows it live -- so this is a faithful
preview of the frame the policy receives, useful for aiming/focusing the wrist
camera before a run.

The ZED is exclusive-access: eval/collect must not be holding it while this runs.

Usage::

    python -m scripts.zed_live_view                      # first available ZED
    python -m scripts.zed_live_view --zed-serial 17785901
    python -m scripts.zed_live_view --width 1280 --height 720 --fps 15
    python -m scripts.zed_live_view --scale 0.5           # half-size window
    python -m scripts.zed_live_view --save preview.jpg   # headless: no window

With a display, press ``q`` or ``Esc`` to quit. Without one (e.g. over plain
SSH), pass ``--save`` to write the latest frame to a file on an interval
instead of opening a window.
"""
import argparse
import os
import time

import numpy as np

import pyzed.sl as sl

from client.eval.zed import DEFAULT_ZED_SETTINGS_PATH, _resolution_for


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zed-serial", type=int, default=None,
                    help="ZED hardware serial to pin a specific camera "
                         "(default: first available)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="scale factor for the displayed window only "
                         "(e.g. 0.5 = half size); does not affect capture")
    ap.add_argument("--settings-path", default=DEFAULT_ZED_SETTINGS_PATH,
                    help="dir holding/fetching SN<serial>.conf calibration")
    ap.add_argument("--save", default=None,
                    help="headless mode: write latest frame to this path on an "
                         "interval instead of opening a window")
    ap.add_argument("--save-interval", type=float, default=0.5,
                    help="seconds between writes in --save mode")
    args = ap.parse_args()

    init = sl.InitParameters()
    init.camera_resolution = _resolution_for(args.width, args.height)
    init.camera_fps = args.fps
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.coordinate_units = sl.UNIT.METER
    os.makedirs(args.settings_path, exist_ok=True)
    init.optional_settings_path = args.settings_path
    if args.zed_serial is not None:
        init.set_from_serial_number(int(args.zed_serial))

    cam = sl.Camera()
    err = cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED camera: {err}")
    hw_serial = cam.get_camera_information().serial_number
    print(f"[zed] Opened hw_serial={hw_serial} "
          f"({args.width}x{args.height}@{args.fps}, LEFT view, depth off)")

    runtime = sl.RuntimeParameters()
    image = sl.Mat()

    # cv2 is only needed to display/encode; import lazily so the SDK errors
    # above surface first and a missing cv2 doesn't mask a camera problem.
    import cv2

    win = f"ZED LEFT (sn={hw_serial})"
    if args.save is None:
        # WINDOW_NORMAL lets you also drag-resize the window freely at runtime,
        # on top of the fixed --scale downsize applied to each frame.
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        if args.scale != 1.0:
            cv2.resizeWindow(win, int(args.width * args.scale),
                             int(args.height * args.scale))

    last_save = 0.0
    try:
        while True:
            if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue
            cam.retrieve_image(image, sl.VIEW.LEFT)
            bgr = np.ascontiguousarray(image.get_data()[:, :, :3])

            if args.save is not None:
                now = time.monotonic()
                if now - last_save >= args.save_interval:
                    cv2.imwrite(args.save, bgr)
                    last_save = now
                continue

            disp = bgr
            if args.scale != 1.0:
                disp = cv2.resize(bgr, None, fx=args.scale, fy=args.scale,
                                  interpolation=cv2.INTER_AREA)
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or Esc
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        if args.save is None:
            cv2.destroyAllWindows()
        print("[zed] closed.")


if __name__ == "__main__":
    main()
