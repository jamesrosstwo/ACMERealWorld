# FoundationStereo Setup

FoundationStereo computes dense depth maps from the IR stereo pairs captured
by the RealSense cameras.  It runs as an optional step after the main bag-file
postprocessing.

## Installation

FoundationStereo has its own heavy dependencies (DINOv2, DepthAnything,
flash-attn) so it is installed as a separate repo alongside this project
rather than merged into `environment.yaml`.

```bash
# 1. Clone into third_party/
mkdir -p third_party
git clone https://github.com/NVlabs/FoundationStereo.git third_party/FoundationStereo

# 2. Install its conda deps into the ACMEReal environment.
#    (The repo ships environment.yml — cherry-pick what's needed on top of
#    ACMEReal rather than creating a second env.)
conda activate ACMEReal
pip install timm einops imageio open3d scikit-image tensorboard

# 3. flash-attn must be compiled separately (needs CUDA toolkit headers):
pip install flash-attn --no-build-isolation

# 4. Download a pretrained checkpoint.  The best-performing model uses a
#    ViT-Large backbone (23-51-11).  A smaller/faster ViT-Small variant
#    (11-33-40) is also available.
#    Download from the HuggingFace link in the FoundationStereo README and
#    place under:
#      third_party/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth
#    The cfg.yaml that ships in the same directory is required.

# 5. Make the FoundationStereo package importable.  Add it to PYTHONPATH
#    before running postprocessing:
export PYTHONPATH="${PYTHONPATH}:$(pwd)/third_party/FoundationStereo"
```

## Tested GPUs

NVIDIA 3090, 4090, A100, V100, and Jetson Orin.

## Usage

### Integrated with postprocessing

Enable stereo depth in `config/postprocess.yaml`:

```yaml
stereo:
  enabled: true
  ckpt_dir: "./third_party/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth"
  scale: 1.0         # downscale factor (1.0 = full res, 0.5 = half)
  valid_iters: 32     # GRU refinement iterations (fewer = faster)
```

Then run the normal postprocessing pipeline — depth estimation happens
automatically after bag extraction:

```bash
PYTHONPATH="${PYTHONPATH}:./third_party/FoundationStereo" python client/postprocess.py
```

### Standalone

If episodes are already postprocessed (have `COMPLETE` marker and IR zarrs),
you can run depth estimation by itself:

```bash
PYTHONPATH="${PYTHONPATH}:./third_party/FoundationStereo" \
  python scripts/foundation_stereo.py \
    --episodes_path /path/to/episodes \
    --ckpt_dir ./third_party/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth \
    --scale 1.0 \
    --valid_iters 32
```

## Output

For each capture directory the script writes:

```
capture_<serial>/
  depth.zarr/     # (n_frames, H, W) float32 — depth in metres
```

Depth is computed as `focal_length * baseline / disparity` using the IR
intrinsics and stereo baseline saved by the bag extractor in
`<serial>_intrinsics.npz`.

## Performance tips

- Use `scale: 0.5` for ~4x faster inference at the cost of spatial resolution.
- Reduce `valid_iters` to 16 for faster (but slightly less accurate) results.
- For images >1000 px, hierarchical mode can be enabled by setting `hiera: 1`
  in the config (not yet exposed in the YAML; pass manually if needed).
