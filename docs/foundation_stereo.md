# FoundationStereo Setup

FoundationStereo computes dense depth maps from the IR stereo pairs captured
by the RealSense cameras.  It runs as an optional step after the main bag-file
postprocessing.

## Requirements

- **Python >= 3.10** — DINOv2 (used internally by FoundationStereo) uses
  `float | None` union syntax which requires Python 3.10+.
- **PyTorch with CUDA** — must support your GPU architecture. For Blackwell
  GPUs (sm_120), use PyTorch nightly with cu128:
  ```bash
  pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
  ```
  This also supports Ada Lovelace (sm_90) and older architectures.

## Installation

FoundationStereo has its own heavy dependencies (DINOv2, DepthAnything, etc.)
so it is installed as a separate repo alongside this project rather than
merged into `environment.yaml`.

```bash
# 1. FoundationStereo is a git submodule — ensure it's checked out:
git submodule update --init submodules/FoundationStereo

# 2. Install pip dependencies into the ACMERealWorld environment.
conda activate ACMERealWorld
pip install timm einops imageio open3d scikit-image tensorboard trimesh

# 3. (Optional) flash-attn — FoundationStereo does NOT require flash-attn.
#    DINOv2 will use it if available, but falls back to standard attention
#    (with xFormers warnings) if not. If you do want it:
#      pip install flash-attn --no-build-isolation
#    Note: flash-attn must be compiled from source and requires CUDA toolkit
#    headers matching your torch CUDA version. This can be tricky with
#    conda environments — skip it unless you need the speedup.

# 4. Download a pretrained checkpoint.  The best-performing model uses a
#    ViT-Large backbone (23-51-11).  A smaller/faster ViT-Small variant
#    (11-33-40) is also available.
#    Download from the HuggingFace link in the FoundationStereo README and
#    place under:
#      submodules/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth
#    The cfg.yaml that ships in the same directory is required.

# 5. Make the FoundationStereo package importable.  Add it to PYTHONPATH
#    before running postprocessing:
export PYTHONPATH="${PYTHONPATH}:$(pwd)/submodules/FoundationStereo"
```

## Usage

### Integrated with postprocessing

Enable stereo depth in `config/postprocess.yaml`:

```yaml
stereo:
  enabled: true
  ckpt_dir: "./submodules/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth"
  scale: 1.0         # downscale factor (1.0 = full res, 0.5 = half)
  valid_iters: 32     # GRU refinement iterations (fewer = faster)
```

Then run the normal postprocessing pipeline — depth estimation happens
automatically after bag extraction:

```bash
PYTHONPATH=".:./submodules/FoundationStereo" python client/postprocess.py
```

### Standalone

If episodes are already postprocessed (have `COMPLETE` marker and IR zarrs),
you can run depth estimation by itself:

```bash
PYTHONPATH=".:./submodules/FoundationStereo" \
  python scripts/foundation_stereo.py \
    --episodes_path /path/to/episodes \
    --ckpt_dir ./submodules/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth \
    --scale 1.0 \
    --valid_iters 32
```

## Output

For each capture directory the script writes:

```
capture_<serial>/
  depth.zarr/     # (n_frames, H, W) uint16 — depth in millimetres
```

Depth is computed as `focal_length * baseline / disparity` using the IR
intrinsics and stereo baseline saved by the bag extractor in
`<serial>_calibration.yaml`.

## Performance tips

- Use `scale: 0.5` for ~4x faster inference at the cost of spatial resolution.
- Reduce `valid_iters` to 16 for faster (but slightly less accurate) results.
- For images >1000 px, hierarchical mode can be enabled by setting `hiera: 1`
  in the config (not yet exposed in the YAML; pass manually if needed).

## Troubleshooting

### `CUDA error: no kernel image is available for execution on the device`
Your PyTorch build doesn't support your GPU architecture. Check supported
architectures with:
```python
import torch; print(torch.cuda.get_arch_list())
```
For Blackwell (sm_120), install PyTorch nightly with cu128 (see Requirements).

### `float | None` syntax error from DINOv2
DINOv2 (loaded from `torch.hub`) uses Python 3.10+ union type syntax.
Upgrade to Python >= 3.10.

### `_pickle.UnpicklingError: Weights only load failed`
PyTorch >= 2.6 defaults `torch.load` to `weights_only=True`. The
FoundationStereo checkpoint contains numpy scalars that aren't in the
safe-load allowlist. The loading code already passes `weights_only=False`.

### `xFormers is not available` warnings
These are harmless. DINOv2 falls back to standard attention when xFormers /
flash-attn are not installed. No impact on accuracy, minor impact on speed.
