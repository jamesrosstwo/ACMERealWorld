# ACMERealWorld

Client-side interfaces for the ACME robot learning data collection and evaluation system. This project provides synchronized multi-camera RGB-D recording with teleoperated demonstrations on a Franka Panda robot arm, along with policy evaluation for learned behaviors.

## Documentation

| Document | Description |
|----------|-------------|
| [Installation](docs/installation.md) | Hardware requirements, environment setup, submodules, network config |
| [Usage](docs/usage.md) | Collecting demonstrations, postprocessing, evaluation, utility scripts |
| [Configuration](docs/configuration.md) | Hydra config files and override syntax |
| [Dataset Format](docs/dataset_format.md) | On-disk episode structure, zarr schemas, synchronization details |
| [FoundationStereo](docs/foundation_stereo.md) | Stereo depth estimation setup, usage, and troubleshooting |

## Quick start

```bash
git clone --recurse-submodules git@github.com:jamesrosstwo/ACMERealWorld.git
cd ACMERealWorld
conda env create -f environment.yaml
conda activate ACMEReal
python -m client.collect
```

## Project structure

```
ACMERealWorld/
  client/                   # Main Python package
    collect/                # Teleoperated data collection
    eval/                   # Policy evaluation on the robot
    nuc.py                  # NUC / Franka Panda interface
    utils.py                # RealSense device enumeration, path helpers
    postprocess.py          # Postprocessing entry point
  config/                   # Hydra configuration files
  docs/                     # Documentation
  scripts/                  # Standalone utility scripts
  tests/                    # Unit tests
  submodules/               # Git submodules (GELLO, FoundationStereo)
  environment.yaml          # Conda environment specification
```
