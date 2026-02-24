"""Compute depth from IR stereo pairs using FoundationStereo.

Reads ``ir_left.zarr`` and ``ir_right.zarr`` from each capture directory,
runs FoundationStereo inference, and writes ``depth.zarr`` (float32, metres).

Can be used standalone::

    python scripts/foundation_stereo.py --episodes_path /path/to/episodes \
        --ckpt_dir /path/to/pretrained_models/23-51-11/model_best_bp2.pth

Or called programmatically from the postprocessing pipeline via
:func:`process_episode`.
"""
import argparse
import logging
import sys
import os
from pathlib import Path

import yaml

import numpy as np
import torch
import zarr
from tqdm import tqdm

# FoundationStereo is installed as a separate repo; its location must be on
# sys.path (see docs/foundation_stereo.md for setup instructions).
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
                    focal_length: float, baseline: float) -> np.ndarray:
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
        Float32 depth map in metres, shape ``(H, W)``.  Pixels with invalid
        disparity are set to 0.
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
    depth = np.zeros_like(disp, dtype=np.float32)
    depth[valid] = effective_focal * baseline / disp[valid]
    return depth


def process_episode(ep_path: Path, ckpt_dir: str, model=None, args=None,
                    scale: float = 1.0, valid_iters: int = 32):
    """Run FoundationStereo on every capture in an episode directory.

    Reads ``ir_left.zarr`` / ``ir_right.zarr`` and the per-camera intrinsics
    file, then writes ``depth.zarr`` (float32, metres) alongside the IR stores.

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

    captures_dir = ep_path / "captures"
    if not captures_dir.exists():
        log.warning("No captures/ directory in %s, skipping", ep_path)
        return

    for cap_dir in sorted(captures_dir.iterdir()):
        if not cap_dir.is_dir() or not cap_dir.name.startswith("capture_"):
            continue

        serial = cap_dir.name.replace("capture_", "")
        ir_left_path = cap_dir / "ir_left.zarr"
        ir_right_path = cap_dir / "ir_right.zarr"
        depth_path = cap_dir / "depth.zarr"

        if depth_path.exists():
            log.info("depth.zarr already exists for %s, skipping", cap_dir.name)
            continue

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
        focal_length = calib["intrinsics"]["ir"][0]  # fx
        T_ir = calib["extrinsics"]["T_ir1_to_ir2"]
        baseline = abs(T_ir[0][3])  # x-component of translation

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
                                    dtype=np.float32)

        for i in tqdm(range(n_frames), desc=cap_dir.name):
            left = np.array(ir_left_arr[i])
            right = np.array(ir_right_arr[i])
            depth = stereo_to_depth(model, args, left, right, focal_length, baseline)
            depth_out[i] = depth

    log.info("Done with episode %s", ep_path.name)


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
    args = parser.parse_args()

    torch.autograd.set_grad_enabled(False)

    model, model_args = load_model(args.ckpt_dir, scale=args.scale, valid_iters=args.valid_iters)

    base = Path(args.episodes_path)
    ep_paths = sorted(base.iterdir(), key=lambda p: int(p.stem.split("_")[-1]))

    for ep_path in tqdm(ep_paths, desc="Episodes"):
        if not (ep_path / "COMPLETE").exists():
            log.info("Episode %s not postprocessed yet, skipping", ep_path.name)
            continue
        process_episode(ep_path, args.ckpt_dir, model=model, args=model_args)


if __name__ == "__main__":
    main()
