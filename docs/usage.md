# Usage

## Collecting demonstrations

Record teleoperated demonstrations using the GELLO controller:

```bash
conda activate ACMEReal
python -m client.collect
```

By default this uses the `pusht` task. Switch tasks with:

```bash
python -m client.collect task=pickandplace_mug
```

Episodes are saved to the path specified by `episodes_path` in the config (default: `/mnt/ssd/james/episodes`).

## Postprocessing

After collection, decode the RealSense `.bag` files into synchronized video, state, and IR data:

```bash
python client/postprocess.py
```

With FoundationStereo enabled, this also computes depth from the IR stereo pairs:

```bash
PYTHONPATH="${PYTHONPATH}:./submodules/FoundationStereo" python client/postprocess.py
```

## Running FoundationStereo standalone

If episodes are already postprocessed (have a `COMPLETE` marker and IR zarrs), run depth estimation separately:

```bash
PYTHONPATH="${PYTHONPATH}:./submodules/FoundationStereo" \
  python scripts/foundation_stereo.py \
    --episodes_path /path/to/episodes \
    --ckpt_dir ./submodules/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth \
    --scale 1.0 \
    --valid_iters 32
```

See [foundation_stereo.md](foundation_stereo.md) for full details and performance tips.

## Evaluating a policy

Run a learned policy on the robot. The policy must be served over HTTP (default port 52968):

```bash
conda activate ACMEReal
python -m client.eval
```

## Utility scripts

| Script | Purpose |
|--------|---------|
| `scripts/foundation_stereo.py` | Standalone depth estimation from IR stereo pairs |
| `scripts/reindex_bags.py` | Re-index corrupted RealSense `.bag` files |
| `scripts/rewrite_bags.py` | Rewrite RealSense `.bag` files |

## Running tests

```bash
conda activate ACMEReal
pytest tests/
```
