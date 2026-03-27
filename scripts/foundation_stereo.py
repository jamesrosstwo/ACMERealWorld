"""Compute depth from IR stereo pairs using FoundationStereo.

Reads ``ir_left.zarr`` and ``ir_right.zarr`` from each capture directory,
runs FoundationStereo inference, and writes ``depth.zarr`` (uint16, millimetres).

Can be used standalone::

    python scripts/foundation_stereo.py --episodes_path /path/to/episodes \
        --ckpt_dir /path/to/pretrained_models/23-51-11/model_best_bp2.pth

Or called programmatically from the postprocessing pipeline via
:func:`process_episode`.
"""
import argparse
import logging
import queue
import sys
import os
import threading
from pathlib import Path

import yaml

import numpy as np
import torch
import zarr
from tqdm import tqdm

# FoundationStereo is installed as a separate repo; add it to sys.path.
_fs_dir = str(Path(__file__).resolve().parent.parent / "submodules" / "FoundationStereo")
if _fs_dir not in sys.path:
    sys.path.insert(0, _fs_dir)
from omegaconf import OmegaConf
from core.foundation_stereo import FoundationStereo
from core.utils.utils import InputPadder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_model(ckpt_dir: str, scale: float = 1.0, valid_iters: int = 32):
    """Load a FoundationStereo model from a checkpoint directory.

    Returns (model, args) where *args* carries inference hyper-parameters.
    """
    cfg = OmegaConf.load(f"{os.path.dirname(ckpt_dir)}/cfg.yaml")
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg["scale"] = scale
    cfg["valid_iters"] = valid_iters
    cfg["hiera"] = 0
    args = OmegaConf.create(cfg)

    model = FoundationStereo(args)
    ckpt = torch.load(ckpt_dir, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.cuda()
    model.eval()
    return model, args


def stereo_to_depth(model, args, ir_left: np.ndarray, ir_right: np.ndarray,
                    focal_length: float, baseline: float,
                    depth_min_mm: int = 1, depth_max_mm: int = 3000) -> np.ndarray:
    """Run FoundationStereo on a single stereo pair and return a depth map.

    Parameters
    ----------
    model : FoundationStereo
        Loaded model (on GPU, eval mode).
    args : OmegaConf
        Model configuration carrying ``scale``, ``valid_iters``, ``hiera``.
    ir_left, ir_right : np.ndarray
        Grayscale uint8 images, shape ``(H, W)``.
    focal_length : float
        Focal length in pixels (from IR intrinsics).
    baseline : float
        Stereo baseline in metres (magnitude of the translation between IR
        emitters, typically ~0.05 m for D400-series RealSense cameras).

    Returns
    -------
    depth : np.ndarray
        Uint16 depth map in millimetres, shape ``(H, W)``.  Pixels with
        invalid disparity or depth > 65535 mm are set to 0.
    """
    # FoundationStereo expects 3-channel float tensors (N, 3, H, W).
    img0 = np.stack([ir_left] * 3, axis=-1).astype(np.float32)
    img1 = np.stack([ir_right] * 3, axis=-1).astype(np.float32)

    scale = args.scale
    if scale < 1.0:
        import cv2
        img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
        img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)

    H, W = img0.shape[:2]

    img0_t = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    img1_t = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0_t.shape, divis_by=32, force_square=False)
    img0_t, img1_t = padder.pad(img0_t, img1_t)

    with torch.cuda.amp.autocast(True):
        if not args.hiera:
            disp = model.forward(img0_t, img1_t, iters=args.valid_iters, test_mode=True)
        else:
            disp = model.run_hierachical(img0_t, img1_t, iters=args.valid_iters,
                                         test_mode=True, small_ratio=0.5)

    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W)

    # depth = f * b / disparity  (disparity in pixels at the *scaled* resolution)
    effective_focal = focal_length * scale
    valid = disp > 0
    depth_m = np.zeros_like(disp, dtype=np.float32)
    depth_m[valid] = effective_focal * baseline / disp[valid]

    # Convert to uint16 millimetres (0 stays 0 for invalid pixels).
    depth_mm = depth_m * 1000.0
    depth_mm[(depth_mm < depth_min_mm) | (depth_mm > depth_max_mm)] = 0
    depth_mm[depth_mm > 65535] = 0
    return depth_mm.astype(np.uint16)


def reproject_depth_ir_to_color(
    depth_ir: np.ndarray,
    ir_intrinsics: list,
    color_intrinsics: list,
    T_color_to_ir1: list,
) -> np.ndarray:
    """Reproject a depth map from the IR camera frame to the color camera frame.

    Back-projects each valid depth pixel to 3D using IR intrinsics, transforms
    the point cloud into the color camera coordinate frame via the extrinsic
    calibration, then projects onto the color image plane.  A simple z-buffer
    resolves occlusions when multiple IR pixels map to the same color pixel.

    Parameters
    ----------
    depth_ir : np.ndarray
        Depth map in the IR frame, uint16 millimetres, shape ``(H, W)``.
    ir_intrinsics : list
        ``[fx, fy, cx, cy]`` for the IR camera.
    color_intrinsics : list
        ``[fx, fy, cx, cy]`` for the color camera.
    T_color_to_ir1 : list
        4x4 extrinsic matrix (nested lists) mapping **color → IR1**.

    Returns
    -------
    depth_color : np.ndarray
        Depth map aligned with the color camera, uint16 millimetres,
        same shape as *depth_ir*.
    """
    H, W = depth_ir.shape
    fx_ir, fy_ir, cx_ir, cy_ir = ir_intrinsics
    fx_c, fy_c, cx_c, cy_c = color_intrinsics

    # We have color→IR1; invert to get IR1→color.
    T_ir1_to_color = np.linalg.inv(np.asarray(T_color_to_ir1, dtype=np.float64))

    # Pixel coordinate grid
    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32))

    valid = depth_ir > 0
    Z = depth_ir.astype(np.float32) / 1000.0  # mm → m

    # Back-project valid pixels to 3D in the IR frame
    X_ir = (u - cx_ir) * Z / fx_ir
    Y_ir = (v - cy_ir) * Z / fy_ir

    mask = valid.ravel()
    pts = np.stack([X_ir.ravel()[mask],
                    Y_ir.ravel()[mask],
                    Z.ravel()[mask],
                    np.ones(mask.sum(), dtype=np.float32)], axis=0)  # (4, N)

    # Transform to color camera frame
    pts_c = T_ir1_to_color @ pts  # (4, N)
    X_c, Y_c, Z_c = pts_c[0], pts_c[1], pts_c[2]

    # Discard points behind the color camera
    in_front = Z_c > 0
    X_c, Y_c, Z_c = X_c[in_front], Y_c[in_front], Z_c[in_front]

    # Project onto the color image plane
    u_c = np.round(fx_c * X_c / Z_c + cx_c).astype(np.int32)
    v_c = np.round(fy_c * Y_c / Z_c + cy_c).astype(np.int32)

    # Keep only pixels that land inside the image
    in_bounds = (u_c >= 0) & (u_c < W) & (v_c >= 0) & (v_c < H)
    u_c, v_c, Z_c = u_c[in_bounds], v_c[in_bounds], Z_c[in_bounds]

    # Z-buffer: sort far-to-near so closer depths overwrite farther ones
    order = np.argsort(-Z_c)
    u_c, v_c, Z_c = u_c[order], v_c[order], Z_c[order]

    depth_color = np.zeros((H, W), dtype=np.float32)
    depth_color[v_c, u_c] = Z_c

    depth_mm = depth_color * 1000.0
    depth_mm[depth_mm > 65535] = 0
    return depth_mm.astype(np.uint16)


def postprocess_depth(
    depth_mm: np.ndarray,
    spatial_kernel: int = 5,
    spatial_iters: int = 2,
    hole_fill_mode: str = "farthest",
    hole_fill_max_radius: int = 2,
) -> np.ndarray:
    """Apply RealSense-style post-processing to fill reprojection grid artifacts.

    Follows the Intel RealSense recommended pipeline (spatial filter then hole
    filling filter), adapted to preserve the larger occlusion gaps expected
    from the IR-to-color extrinsic shift.

    Stage 1 — **Spatial filter**: an edge-preserving median filter applied only
    to invalid (zero) pixels.  Fills isolated grid-artifact holes while leaving
    valid depth untouched and preserving depth edges.

    Stage 2 — **Hole filling filter**: fills any remaining small gaps (up to
    *hole_fill_max_radius* pixels wide) by propagating from valid neighbours.
    Three modes match the RealSense ``hole_filling_filter``:

    * ``"left"``     – fill from the left neighbour (scan-line order)
    * ``"farthest"`` – fill with the farthest (deepest) of the 4-connected
      neighbours, avoiding foreground bleeding onto background
    * ``"nearest"``  – fill with the nearest (shallowest) of the 4-connected
      neighbours

    Parameters
    ----------
    depth_mm : np.ndarray
        Depth map (uint16, millimetres).
    spatial_kernel : int
        Median filter kernel size (must be odd, 3 or 5 recommended).
    spatial_iters : int
        Number of spatial-filter passes.
    hole_fill_mode : str
        One of ``"left"``, ``"farthest"``, ``"nearest"``.
    hole_fill_max_radius : int
        Maximum distance (in pixels) to search for a valid neighbour in the
        hole-filling stage.  Keeps large occlusion gaps intact.  Set to 0 to
        disable hole filling entirely.
    """
    import cv2

    depth = depth_mm.copy()

    # --- Stage 1: Spatial filter (median, applied only to holes) -----------
    for _ in range(spatial_iters):
        filtered = cv2.medianBlur(depth, spatial_kernel)
        holes = depth == 0
        depth[holes] = filtered[holes]

    # --- Stage 2: Hole filling filter (limited radius) ---------------------
    if hole_fill_max_radius > 0:
        for _ in range(hole_fill_max_radius):
            holes = depth == 0
            if not holes.any():
                break

            if hole_fill_mode == "left":
                # Scan-line fill: propagate from the left neighbour
                filled = depth.copy()
                filled[:, 1:] = np.where(
                    (filled[:, 1:] == 0) & (depth[:, :-1] > 0),
                    depth[:, :-1],
                    filled[:, 1:],
                )
                depth = filled
            else:
                # 4-connected: gather up/down/left/right neighbours
                pad = np.pad(depth, 1, mode='constant', constant_values=0)
                up    = pad[:-2, 1:-1]
                down  = pad[2:,  1:-1]
                left  = pad[1:-1, :-2]
                right = pad[1:-1, 2:]
                neighbors = np.stack([up, down, left, right], axis=-1)  # (H,W,4)

                # Mask out invalid neighbours (zero)
                valid_nb = neighbors > 0
                any_valid = valid_nb.any(axis=-1) & holes

                if not any_valid.any():
                    break

                if hole_fill_mode == "farthest":
                    # Use the deepest valid neighbour
                    candidates = np.where(valid_nb, neighbors, 0)
                    fill_vals = candidates.max(axis=-1)
                elif hole_fill_mode == "nearest":
                    # Use the shallowest valid neighbour
                    candidates = np.where(valid_nb, neighbors, np.uint16(65535))
                    fill_vals = candidates.min(axis=-1)
                else:
                    raise ValueError(f"Unknown hole_fill_mode: {hole_fill_mode!r}")

                depth[any_valid] = fill_vals[any_valid]

    return depth


def process_episode(ep_path: Path, ckpt_dir: str, model=None, args=None,
                    scale: float = 1.0, valid_iters: int = 32,
                    depth_min_mm: int = 1, depth_max_mm: int = 3000,
                    spatial_kernel: int = 5, spatial_iters: int = 2,
                    hole_fill_mode: str = "farthest", hole_fill_max_radius: int = 2):
    """Run FoundationStereo on every capture in an episode directory.

    Reads ``ir_left.zarr`` / ``ir_right.zarr`` and the per-camera intrinsics
    file, then writes ``depth.zarr`` (uint16, millimetres) alongside the IR stores.

    Parameters
    ----------
    ep_path : Path
        Episode directory (e.g. ``episodes/episode_0``).
    ckpt_dir : str
        Path to the FoundationStereo checkpoint (``.pth``).
    model, args : optional
        Pre-loaded model and config.  If *None* they are loaded from
        *ckpt_dir* (useful when processing a single episode in isolation).
    scale : float
        Image downscale factor passed to FoundationStereo.
    valid_iters : int
        Number of GRU refinement iterations.
    """
    if model is None:
        model, args = load_model(ckpt_dir, scale=scale, valid_iters=valid_iters)

    depth_marker = ep_path / "DEPTH_COMPLETE"
    if depth_marker.exists():
        log.info("DEPTH_COMPLETE marker exists for %s, skipping", ep_path.name)
        return

    captures_dir = ep_path / "captures"
    if not captures_dir.exists():
        log.warning("No captures/ directory in %s, skipping", ep_path)
        return

    log.info("Starting FoundationStereo depth estimation for %s", ep_path.name)

    for cap_dir in sorted(captures_dir.iterdir()):
        if not cap_dir.is_dir() or not cap_dir.name.startswith("capture_"):
            continue

        serial = cap_dir.name.replace("capture_", "")
        ir_left_path = cap_dir / "ir_left.zarr"
        ir_right_path = cap_dir / "ir_right.zarr"
        depth_path = cap_dir / "depth.zarr"

        if not ir_left_path.exists() or not ir_right_path.exists():
            log.warning("Missing IR zarr for %s, skipping", cap_dir.name)
            continue

        # Load calibration
        calib_file = cap_dir / "calibration.yaml"
        if not calib_file.exists():
            log.warning("Missing calibration file %s, skipping", calib_file)
            continue
        with open(calib_file) as f:
            calib = yaml.safe_load(f)
        ir_intrinsics = calib["intrinsics"]["ir"]        # [fx, fy, cx, cy]
        color_intrinsics = calib["intrinsics"]["color"]  # [fx, fy, cx, cy]
        focal_length = ir_intrinsics[0]  # fx
        T_ir = calib["extrinsics"]["T_ir1_to_ir2"]
        baseline = abs(T_ir[0][3])  # x-component of translation
        T_color_to_ir1 = calib["extrinsics"]["T_color_to_ir1"]

        ir_left_arr = zarr.open(str(ir_left_path), mode="r")
        ir_right_arr = zarr.open(str(ir_right_path), mode="r")
        n_frames = ir_left_arr.shape[0]
        H, W = ir_left_arr.shape[1], ir_left_arr.shape[2]

        log.info("Processing %s: %d frames (%dx%d), f=%.1f baseline=%.4f",
                 cap_dir.name, n_frames, W, H, focal_length, baseline)

        depth_store = zarr.DirectoryStore(str(depth_path))
        depth_out = zarr.open_array(depth_store, mode="w",
                                    shape=(n_frames, H, W),
                                    chunks=(16, H, W),
                                    dtype=np.uint16)

        for i in tqdm(range(n_frames), desc=cap_dir.name):
            left = np.array(ir_left_arr[i])
            right = np.array(ir_right_arr[i])
            depth_ir = stereo_to_depth(model, args, left, right, focal_length, baseline,
                                       depth_min_mm=depth_min_mm, depth_max_mm=depth_max_mm)
            depth = reproject_depth_ir_to_color(
                depth_ir, ir_intrinsics, color_intrinsics, T_color_to_ir1,
            )
            depth = postprocess_depth(
                depth,
                spatial_kernel=spatial_kernel,
                spatial_iters=spatial_iters,
                hole_fill_mode=hole_fill_mode,
                hole_fill_max_radius=hole_fill_max_radius,
            )
            depth_out[i] = depth

    depth_marker.touch()
    log.info("Done with episode %s — DEPTH_COMPLETE", ep_path.name)


def _gpu_worker(gpu_id: int, work_queue: queue.Queue, ckpt_dir: str,
                scale: float, valid_iters: int,
                depth_min_mm: int = 1, depth_max_mm: int = 3000,
                spatial_kernel: int = 5, spatial_iters: int = 2,
                hole_fill_mode: str = "farthest", hole_fill_max_radius: int = 2):
    """Worker thread: loads a model on one GPU and processes episodes from the queue."""
    torch.cuda.set_device(gpu_id)
    torch.autograd.set_grad_enabled(False)
    model, model_args = load_model(ckpt_dir, scale=scale, valid_iters=valid_iters)
    log.info("GPU %d worker ready", gpu_id)

    while True:
        ep_path = work_queue.get()
        if ep_path is None:
            break
        try:
            process_episode(ep_path, ckpt_dir, model=model, args=model_args,
                           depth_min_mm=depth_min_mm, depth_max_mm=depth_max_mm,
                           spatial_kernel=spatial_kernel, spatial_iters=spatial_iters,
                           hole_fill_mode=hole_fill_mode,
                           hole_fill_max_radius=hole_fill_max_radius)
        except Exception:
            log.exception("Depth failed for %s on GPU %d", ep_path, gpu_id)
        finally:
            work_queue.task_done()


def main():
    parser = argparse.ArgumentParser(description="Run FoundationStereo depth estimation on postprocessed episodes.")
    parser.add_argument("--episodes_path", type=str, required=True,
                        help="Root directory containing episode_* folders")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Path to FoundationStereo checkpoint (.pth)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Downscale factor for inference (<=1.0)")
    parser.add_argument("--valid_iters", type=int, default=32,
                        help="Number of GRU refinement iterations")
    parser.add_argument("--depth_min_mm", type=int, default=1,
                        help="Minimum valid depth in millimetres (default: 1)")
    parser.add_argument("--depth_max_mm", type=int, default=3000,
                        help="Maximum valid depth in millimetres (default: 3000)")
    parser.add_argument("--spatial_kernel", type=int, default=5,
                        help="Median filter kernel size for spatial post-processing (default: 5)")
    parser.add_argument("--spatial_iters", type=int, default=2,
                        help="Number of spatial filter passes (default: 2)")
    parser.add_argument("--hole_fill_mode", type=str, default="farthest",
                        choices=["left", "farthest", "nearest"],
                        help="Hole filling strategy (default: farthest)")
    parser.add_argument("--hole_fill_max_radius", type=int, default=2,
                        help="Max pixel radius for hole filling; 0 to disable (default: 2)")
    args = parser.parse_args()

    base = Path(args.episodes_path)
    ep_paths = [p for p in sorted(base.iterdir(), key=lambda p: int(p.stem.split("_")[-1]))
                if (p / "COMPLETE").exists() and not (p / "DEPTH_COMPLETE").exists()]

    if not ep_paths:
        log.info("No episodes to process")
        return

    num_gpus = torch.cuda.device_count()
    log.info("Found %d GPU(s), %d episode(s) to process", num_gpus, len(ep_paths))

    work_queue: queue.Queue = queue.Queue()
    workers = []
    for gpu_id in range(num_gpus):
        t = threading.Thread(target=_gpu_worker, daemon=True,
                             args=(gpu_id, work_queue, args.ckpt_dir,
                                   args.scale, args.valid_iters,
                                   args.depth_min_mm, args.depth_max_mm,
                                   args.spatial_kernel, args.spatial_iters,
                                   args.hole_fill_mode, args.hole_fill_max_radius))
        t.start()
        workers.append(t)

    for ep_path in ep_paths:
        work_queue.put(ep_path)

    work_queue.join()
    for _ in workers:
        work_queue.put(None)
    for t in workers:
        t.join()
    log.info("All episodes done")


if __name__ == "__main__":
    main()
