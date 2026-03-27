"""Quick test: run FoundationStereo + reprojection + postprocessing on a single
frame and save before/after depth PNGs for visual comparison."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'submodules', 'FoundationStereo'))

import numpy as np
import yaml
import zarr
import cv2
import torch

from scripts.foundation_stereo import (
    load_model, stereo_to_depth, reproject_depth_ir_to_color, postprocess_depth,
)

CAP_DIR = "/home/james/ACME/data/mug_stereo_6cam_clamped/episode_0/captures/capture_848312071702"
CKPT = "./submodules/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth"
OUT_DIR = "/home/james/ACMERealWorld/depth_samples/postproc_test"
FRAME_IDX = 100  # pick a mid-episode frame

os.makedirs(OUT_DIR, exist_ok=True)

# --- Load data ---------------------------------------------------------------
with open(f"{CAP_DIR}/calibration.yaml") as f:
    calib = yaml.safe_load(f)

ir_intrinsics = calib["intrinsics"]["ir"]
color_intrinsics = calib["intrinsics"]["color"]
focal_length = ir_intrinsics[0]
T_ir = calib["extrinsics"]["T_ir1_to_ir2"]
baseline = abs(T_ir[0][3])
T_color_to_ir1 = calib["extrinsics"]["T_color_to_ir1"]

ir_left = np.array(zarr.open(f"{CAP_DIR}/ir_left.zarr", "r")[FRAME_IDX])
ir_right = np.array(zarr.open(f"{CAP_DIR}/ir_right.zarr", "r")[FRAME_IDX])
print(f"Loaded frame {FRAME_IDX}: {ir_left.shape}")

# --- FoundationStereo depth --------------------------------------------------
torch.cuda.set_device(0)
torch.autograd.set_grad_enabled(False)
model, args = load_model(CKPT, scale=1.0, valid_iters=32)

depth_ir = stereo_to_depth(model, args, ir_left, ir_right, focal_length, baseline)
print(f"Depth IR: valid={np.count_nonzero(depth_ir)}/{depth_ir.size}, "
      f"range=[{depth_ir[depth_ir>0].min()}, {depth_ir[depth_ir>0].max()}] mm")

# --- Reproject to color frame ------------------------------------------------
depth_reprojected = reproject_depth_ir_to_color(
    depth_ir, ir_intrinsics, color_intrinsics, T_color_to_ir1,
)
n_holes_before = np.count_nonzero(depth_reprojected == 0)
print(f"After reprojection: {n_holes_before} zero pixels ({100*n_holes_before/depth_reprojected.size:.1f}%)")

# --- Postprocess (RealSense-style) -------------------------------------------
depth_postprocessed = postprocess_depth(
    depth_reprojected,
    spatial_kernel=5,
    spatial_iters=2,
    hole_fill_mode="farthest",
    hole_fill_max_radius=2,
)
n_holes_after = np.count_nonzero(depth_postprocessed == 0)
print(f"After postprocessing: {n_holes_after} zero pixels ({100*n_holes_after/depth_postprocessed.size:.1f}%)")
print(f"Filled {n_holes_before - n_holes_after} holes")

# --- Render depth as colormapped PNG -----------------------------------------
def depth_to_color_png(depth_mm, path, vmin=200, vmax=2000):
    """Normalise uint16 depth to 0-255 and apply a colourmap."""
    d = depth_mm.astype(np.float32)
    d[d == 0] = np.nan
    d = np.clip((d - vmin) / (vmax - vmin), 0, 1)
    d_u8 = (d * 255).astype(np.uint8)
    # Set invalid pixels to 0 before colourmap
    d_u8[np.isnan(d)] = 0
    colored = cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)
    # Make invalid pixels black
    colored[np.isnan(depth_mm.astype(np.float32).copy())] = 0
    mask = depth_mm == 0
    colored[mask] = 0
    cv2.imwrite(path, colored)
    print(f"Saved {path}")

depth_to_color_png(depth_reprojected, f"{OUT_DIR}/depth_before_postproc.png")
depth_to_color_png(depth_postprocessed, f"{OUT_DIR}/depth_after_postproc.png")

# Also save the IR left for reference alignment
ir_bgr = cv2.cvtColor(np.stack([ir_left]*3, axis=-1), cv2.COLOR_RGB2BGR)
cv2.imwrite(f"{OUT_DIR}/ir_left_reference.png", ir_left)
print(f"Saved {OUT_DIR}/ir_left_reference.png")
print("Done!")
