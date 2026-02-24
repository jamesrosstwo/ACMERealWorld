# Installation

## Hardware Requirements

- **Franka Emika Panda** robot arm (7-DOF) with parallel-jaw gripper
- **Intel RealSense D400-series cameras** (2+) — streams RGB, depth, and IR stereo at 1280x720 @ 15 fps
- **GELLO teleoperation controller** — 7 Dynamixel servos + gripper, connected via USB serial
- **Intel NUC** (or equivalent) running the Franka control server at `192.168.1.106`
- **Franka robot controller** reachable at `192.168.1.107`

## 1. Clone the repository

```bash
git clone --recurse-submodules git@github.com:jamesrosstwo/ACMERealWorld.git
cd ACMERealWorld
```

If you already cloned without `--recurse-submodules`, initialize the submodules separately:

```bash
git submodule update --init --recursive
```

## 2. Create the Conda environment

```bash
conda env create -f environment.yaml
conda activate ACMEReal
```

This installs Python 3.10 and all core dependencies:

| Package | Purpose |
|---------|---------|
| `pyrealsense2` / `librealsense` | Intel RealSense camera SDK and Python bindings |
| `panda-python` | Franka Panda robot control (panda_py / libfranka) |
| `torch` | PyTorch (used by FoundationStereo and policy inference) |
| `hydra-core` | Configuration management |
| `zarr` / `blosc` | Chunked array storage for episode data |
| `opencv-contrib-python` | Image and video processing |
| `scipy` | Scientific computing |
| `h5py` | HDF5 file support |
| `tqdm` | Progress bars |
| `Pillow` | Image I/O |

## 3. Install FoundationStereo (optional)

FoundationStereo computes dense depth maps from IR stereo pairs captured by the RealSense cameras. Skip this section if you do not need stereo depth estimation.

FoundationStereo is included as a git submodule at `submodules/FoundationStereo/`. If you cloned with `--recurse-submodules` it's already checked out.

```bash
# Install additional Python dependencies
conda activate ACMEReal
pip install timm einops imageio open3d scikit-image tensorboard

# Compile flash-attn (requires CUDA toolkit headers)
pip install flash-attn --no-build-isolation
```

**Download a pretrained checkpoint** from the link in the [FoundationStereo README](https://github.com/NVlabs/FoundationStereo) and place it at:

```
submodules/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth
```

The `cfg.yaml` that ships in the same HuggingFace directory is also required — place it alongside the checkpoint.

A smaller/faster ViT-Small variant (`11-33-40`) is also available if GPU memory is limited.

**Make FoundationStereo importable** by adding it to `PYTHONPATH` before running postprocessing:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/submodules/FoundationStereo"
```

See [foundation_stereo.md](foundation_stereo.md) for full details and performance tips.

## 4. GELLO controller setup

The GELLO submodule is cloned at `submodules/gello_software/`. Follow its own setup instructions for Dynamixel SDK installation and servo calibration.

Ensure the GELLO device is accessible at its serial port. The default configuration expects:

```
/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA2U2CD-if00-port0
```

You may need to add your user to the `dialout` group for serial port access:

```bash
sudo usermod -aG dialout $USER
# Log out and back in for the group change to take effect
```

## 5. Network configuration

The system communicates with the Franka Panda through a NUC over a local network. Ensure connectivity between:

| Device | Default IP |
|--------|-----------|
| NUC (Franka control server) | `192.168.1.106` |
| Franka robot controller | `192.168.1.107` |

These addresses are configurable in `config/collect.yaml` and `config/eval.yaml` under `nuc.ip` and `nuc.franka_ip`.
