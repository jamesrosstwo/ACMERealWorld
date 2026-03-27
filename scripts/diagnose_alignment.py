"""Diagnose depth-to-color alignment by overlaying depth edges on color.

Generates separate overlays for:
  1. No correction (raw IR depth pixels on color)
  2. Intrinsic-only correction (remap for different f/cx/cy, no extrinsic)
  3. Full correction (intrinsic + extrinsic, our reproject_depth_ir_to_color)
  4. Full correction using T_depth_to_color directly (no inversion)

This isolates whether the overcorrection comes from the intrinsic remap,
the extrinsic transform, or an inversion bug.
"""
import sys
import numpy as np
import yaml
import zarr
import cv2
from pathlib import Path


def load_frame(cap_dir: Path, frame_idx: int):
    """Load color, reprojected depth for one frame."""
    # Color is stored as rgb.mp4
    mp4_path = cap_dir / "rgb.mp4"
    cap = cv2.VideoCapture(str(mp4_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, color = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {mp4_path}")
    depth_reproj = np.array(zarr.open(str(cap_dir / "depth.zarr"), mode="r")[frame_idx])
    return color, depth_reproj


def load_raw_depth(raw_cap_dir: Path, frame_idx: int):
    """Load non-reprojected depth from the raw backup."""
    return np.array(zarr.open(str(raw_cap_dir / "depth.zarr"), mode="r")[frame_idx])


def depth_edges(depth: np.ndarray, threshold: int = 50) -> np.ndarray:
    """Compute binary edge mask from a uint16 depth map."""
    valid = (depth > 0).astype(np.uint8)
    d_f = depth.astype(np.float32)
    sx = cv2.Sobel(d_f, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(d_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    edges = (mag > threshold).astype(np.uint8)
    # Also mark boundaries of valid depth
    boundary = cv2.Canny(valid * 255, 50, 150)
    return np.clip(edges + (boundary > 0).astype(np.uint8), 0, 1).astype(np.uint8)


def color_edges(color_bgr: np.ndarray) -> np.ndarray:
    """Compute binary edge mask from a color image."""
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return (edges > 0).astype(np.uint8)


def remap_intrinsic_only(depth_ir: np.ndarray, ir_intr, color_intr) -> np.ndarray:
    """Remap depth from IR pixel grid to color pixel grid using ONLY intrinsics.

    No extrinsic (3D translation/rotation) applied — just accounts for
    different focal lengths and principal points.
    """
    H, W = depth_ir.shape
    fx_ir, fy_ir, cx_ir, cy_ir = ir_intr
    fx_c, fy_c, cx_c, cy_c = color_intr

    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32))

    valid = depth_ir > 0
    Z = depth_ir.astype(np.float32) / 1000.0

    # Back-project using IR intrinsics to get ray direction
    X = (u - cx_ir) * Z / fx_ir
    Y = (v - cy_ir) * Z / fy_ir

    # Project using color intrinsics (same 3D point, no extrinsic shift)
    mask = valid.ravel()
    X_flat, Y_flat, Z_flat = X.ravel()[mask], Y.ravel()[mask], Z.ravel()[mask]

    u_c = np.round(fx_c * X_flat / Z_flat + cx_c).astype(np.int32)
    v_c = np.round(fy_c * Y_flat / Z_flat + cy_c).astype(np.int32)

    in_bounds = (u_c >= 0) & (u_c < W) & (v_c >= 0) & (v_c < H)
    u_c, v_c, Z_flat = u_c[in_bounds], v_c[in_bounds], Z_flat[in_bounds]

    order = np.argsort(-Z_flat)
    u_c, v_c, Z_flat = u_c[order], v_c[order], Z_flat[order]

    out = np.zeros((H, W), dtype=np.float32)
    out[v_c, u_c] = Z_flat
    depth_mm = out * 1000.0
    depth_mm[depth_mm > 65535] = 0
    return depth_mm.astype(np.uint16)


def reproject_full(depth_ir: np.ndarray, ir_intr, color_intr, T_4x4) -> np.ndarray:
    """Full reprojection: intrinsics + extrinsic transform.

    T_4x4 should be the IR1-to-color transform (already in the right direction).
    """
    H, W = depth_ir.shape
    fx_ir, fy_ir, cx_ir, cy_ir = ir_intr
    fx_c, fy_c, cx_c, cy_c = color_intr

    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32))

    valid = depth_ir > 0
    Z = depth_ir.astype(np.float32) / 1000.0

    X_ir = (u - cx_ir) * Z / fx_ir
    Y_ir = (v - cy_ir) * Z / fy_ir

    mask = valid.ravel()
    pts = np.stack([X_ir.ravel()[mask], Y_ir.ravel()[mask],
                    Z.ravel()[mask], np.ones(mask.sum(), dtype=np.float32)], axis=0)

    pts_c = T_4x4 @ pts
    X_c, Y_c, Z_c = pts_c[0], pts_c[1], pts_c[2]

    in_front = Z_c > 0
    X_c, Y_c, Z_c = X_c[in_front], Y_c[in_front], Z_c[in_front]

    u_c = np.round(fx_c * X_c / Z_c + cx_c).astype(np.int32)
    v_c = np.round(fy_c * Y_c / Z_c + cy_c).astype(np.int32)

    in_bounds = (u_c >= 0) & (u_c < W) & (v_c >= 0) & (v_c < H)
    u_c, v_c, Z_c = u_c[in_bounds], v_c[in_bounds], Z_c[in_bounds]

    order = np.argsort(-Z_c)
    u_c, v_c, Z_c = u_c[order], v_c[order], Z_c[order]

    out = np.zeros((H, W), dtype=np.float32)
    out[v_c, u_c] = Z_c
    depth_mm = out * 1000.0
    depth_mm[depth_mm > 65535] = 0
    return depth_mm.astype(np.uint16)


def overlay(color_bgr: np.ndarray, c_edges: np.ndarray, d_edges: np.ndarray,
            crop=None) -> np.ndarray:
    """Overlay color edges (green) and depth edges (red) on the color image."""
    vis = color_bgr.copy()
    # Dim the background slightly
    vis = (vis * 0.6).astype(np.uint8)
    vis[c_edges > 0] = [0, 255, 0]   # color edges = green
    vis[d_edges > 0] = [0, 0, 255]   # depth edges = red
    # Where both overlap = yellow
    both = (c_edges > 0) & (d_edges > 0)
    vis[both] = [0, 255, 255]
    if crop is not None:
        y1, y2, x1, x2 = crop
        vis = vis[y1:y2, x1:x2]
    return vis


def compute_edge_offset(c_edges: np.ndarray, d_edges: np.ndarray,
                        search_range: int = 30) -> dict:
    """Estimate the median horizontal pixel offset between depth and color edges.

    For each row, cross-correlate the edge signals to find the best shift.
    """
    H, W = c_edges.shape
    shifts = []
    for row in range(0, H, 4):  # sample every 4th row
        c_row = c_edges[row].astype(np.float32)
        d_row = d_edges[row].astype(np.float32)
        if c_row.sum() < 5 or d_row.sum() < 5:
            continue
        best_shift, best_corr = 0, -1
        for dx in range(-search_range, search_range + 1):
            if dx >= 0:
                corr = np.sum(c_row[dx:] * d_row[:W - dx])
            else:
                corr = np.sum(c_row[:W + dx] * d_row[-dx:])
            if corr > best_corr:
                best_corr = corr
                best_shift = dx
        if best_corr > 0:
            shifts.append(best_shift)
    if not shifts:
        return {"median_shift_px": 0, "n_rows": 0}
    return {
        "median_shift_px": float(np.median(shifts)),
        "mean_shift_px": float(np.mean(shifts)),
        "std_shift_px": float(np.std(shifts)),
        "n_rows": len(shifts),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="/home/james/ACME/data/mug_stereo_6cam")
    parser.add_argument("--raw_dataset", type=str,
                        default="/home/james/ACME/data/mug_stereo_6cam_raw")
    parser.add_argument("--serial", type=str, default="148122060186")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=100)
    parser.add_argument("--outdir", type=str, default="depth_samples/alignment_diag")
    args = parser.parse_args()

    cap_dir = Path(args.dataset) / f"episode_{args.episode}" / "captures" / f"capture_{args.serial}"
    raw_cap_dir = Path(args.raw_dataset) / f"episode_{args.episode}" / "captures" / f"capture_{args.serial}"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load calibration
    with open(cap_dir / "calibration.yaml") as f:
        calib = yaml.safe_load(f)
    ir_intr = calib["intrinsics"]["ir"]
    color_intr = calib["intrinsics"]["color"]
    T_color_to_ir1 = np.array(calib["extrinsics"]["T_color_to_ir1"], dtype=np.float64)
    T_depth_to_color = np.array(calib["extrinsics"]["T_depth_to_color"], dtype=np.float64)

    print("=== Calibration ===")
    print(f"IR intrinsics:    fx={ir_intr[0]:.2f} fy={ir_intr[1]:.2f} cx={ir_intr[2]:.2f} cy={ir_intr[3]:.2f}")
    print(f"Color intrinsics: fx={color_intr[0]:.2f} fy={color_intr[1]:.2f} cx={color_intr[2]:.2f} cy={color_intr[3]:.2f}")
    print(f"Δfx={color_intr[0]-ir_intr[0]:.2f}  Δcx={color_intr[2]-ir_intr[2]:.2f}  Δcy={color_intr[3]-ir_intr[3]:.2f}")
    print(f"T_color_to_ir1 translation: {T_color_to_ir1[:3, 3]}")
    print(f"T_depth_to_color translation: {T_depth_to_color[:3, 3]}")

    T_ir1_to_color_inv = np.linalg.inv(T_color_to_ir1)
    print(f"inv(T_color_to_ir1) translation: {T_ir1_to_color_inv[:3, 3]}")
    print(f"Difference inv vs T_d2c: {T_ir1_to_color_inv[:3, 3] - T_depth_to_color[:3, 3]}")

    # Load frames
    print(f"\n=== Loading frame {args.frame} from camera {args.serial} ===")
    color_bgr, depth_reproj_stored = load_frame(cap_dir, args.frame)
    depth_raw = load_raw_depth(raw_cap_dir, args.frame)

    print(f"Color shape: {color_bgr.shape}")
    print(f"Depth reproj (stored): valid pixels = {(depth_reproj_stored > 0).sum()}")
    print(f"Depth raw (no reproj): valid pixels = {(depth_raw > 0).sum()}")

    # Compute reprojections on the fly from the raw depth
    print("\n=== Computing reprojections ===")
    depth_intrinsic_only = remap_intrinsic_only(depth_raw, ir_intr, color_intr)
    depth_full_via_inv = reproject_full(depth_raw, ir_intr, color_intr, T_ir1_to_color_inv)
    depth_full_via_d2c = reproject_full(depth_raw, ir_intr, color_intr, T_depth_to_color)

    print(f"Intrinsic-only: valid = {(depth_intrinsic_only > 0).sum()}")
    print(f"Full (inv T_c2i): valid = {(depth_full_via_inv > 0).sum()}")
    print(f"Full (T_d2c direct): valid = {(depth_full_via_d2c > 0).sum()}")

    # Compute edges
    c_edges = color_edges(color_bgr)

    cases = {
        "1_no_correction": depth_raw,
        "2_intrinsic_only": depth_intrinsic_only,
        "3_full_inv_Tc2i": depth_full_via_inv,
        "4_full_direct_Td2c": depth_full_via_d2c,
        "5_stored_reproj": depth_reproj_stored,
    }

    # Find a good crop region around objects (center of image)
    H, W = color_bgr.shape[:2]
    crop = (H // 4, 3 * H // 4, W // 4, 3 * W // 4)

    print("\n=== Edge offset analysis (depth edges vs color edges) ===")
    print(f"{'Case':<25s} {'Median shift (px)':>18s} {'Mean shift':>12s} {'Std':>8s}")
    print("-" * 70)

    for name, depth in cases.items():
        d_edges = depth_edges(depth)
        vis = overlay(color_bgr, c_edges, d_edges, crop=crop)
        cv2.imwrite(str(outdir / f"{name}.png"), vis)

        # Also save full-resolution
        vis_full = overlay(color_bgr, c_edges, d_edges)
        cv2.imwrite(str(outdir / f"{name}_full.png"), vis_full)

        stats = compute_edge_offset(c_edges, d_edges)
        print(f"{name:<25s} {stats['median_shift_px']:>18.1f} {stats.get('mean_shift_px', 0):>12.1f} {stats.get('std_shift_px', 0):>8.1f}")

    # Pixel-level comparison between stored reproj and our fresh reproj
    diff = depth_reproj_stored.astype(np.int32) - depth_full_via_inv.astype(np.int32)
    both_valid = (depth_reproj_stored > 0) & (depth_full_via_inv > 0)
    if both_valid.sum() > 0:
        print(f"\n=== Stored vs fresh reproj (inv T_c2i) ===")
        print(f"Both valid pixels: {both_valid.sum()}")
        print(f"Mean diff (mm): {diff[both_valid].mean():.2f}")
        print(f"Max abs diff (mm): {np.abs(diff[both_valid]).max()}")
        print(f"Pixels with >1mm diff: {(np.abs(diff[both_valid]) > 1).sum()}")

    diff2 = depth_full_via_inv.astype(np.int32) - depth_full_via_d2c.astype(np.int32)
    both2 = (depth_full_via_inv > 0) & (depth_full_via_d2c > 0)
    if both2.sum() > 0:
        print(f"\n=== inv(T_c2i) vs direct T_d2c ===")
        print(f"Both valid pixels: {both2.sum()}")
        print(f"Mean diff (mm): {diff2[both2].mean():.4f}")
        print(f"Max abs diff (mm): {np.abs(diff2[both2]).max()}")

    print(f"\nImages saved to {outdir}/")
    print("Green = color edges, Red = depth edges, Yellow = overlap (good alignment)")


if __name__ == "__main__":
    main()
