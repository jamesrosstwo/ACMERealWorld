# Dataset Format

This document describes the on-disk format produced by the ACME data collection and postprocessing pipeline. An episode goes through two stages:

1. **Collection** — raw RealSense `.bag` files and incrementally-written robot state.
2. **Postprocessing** — synchronized multi-camera RGB video, IR stereo pairs, aligned robot state, and optionally FoundationStereo depth maps.

All paths below are relative to an individual episode directory (e.g. `episodes/episode_0/`).

---

## Raw episode (after collection, before postprocessing)

```
episode_N/
  <serial_1>.bag
  <serial_2>.bag
  intrinsics.npz
  raw_episode.zarr/
    qpos
    ee_pos
    ee_rot
    gripper_force
    action
    _state_timestamps
```

### `.bag` files

One RealSense `.bag` file per connected camera, named by the camera's serial number (e.g. `217222061106.bag`). Each bag contains the raw streams recorded by `pyrealsense2`:

| Stream | Format | Resolution | Frame rate |
|--------|--------|------------|------------|
| Color (RGB) | BGR8 | 1280 x 720 | 15 fps |
| Depth | Z16 | 1280 x 720 | 15 fps |
| Infrared left (IR1) | Y8 (grayscale) | 1280 x 720 | 15 fps |
| Infrared right (IR2) | Y8 (grayscale) | 1280 x 720 | 15 fps |

Warmup frames (used to let auto-exposure settle) are excluded — recording is paused during the warmup drain and resumed before capture threads start.

### `intrinsics.npz`

A combined NumPy `.npz` archive written at the start of capture, containing intrinsics for **all** cameras in a single file. Keys are prefixed with the camera serial number:

| Key pattern | Shape | Dtype | Description |
|-------------|-------|-------|-------------|
| `<serial>_depth` | `(4,)` | float64 | `[fx, fy, ppx, ppy]` for the depth stream |
| `<serial>_color` | `(4,)` | float64 | `[fx, fy, ppx, ppy]` for the color stream |
| `<serial>_ir` | `(4,)` | float64 | `[fx, fy, ppx, ppy]` for the infrared stream (left) |
| `<serial>_ir_baseline` | `(3,)` | float64 | Translation vector `[tx, ty, tz]` from IR1 to IR2 |

`fx`, `fy` are the focal lengths in pixels. `ppx`, `ppy` are the principal point (optical centre) coordinates. The IR baseline is the physical distance between the left and right infrared emitters (typically ~0.05 m in the x-component for D400-series cameras).

### `raw_episode.zarr/`

A zarr directory store containing robot state written incrementally during collection. Each dataset is pre-allocated to `max_episode_len` rows and filled as frames arrive.

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `qpos` | `(N, 7)` | float64 | Franka Panda joint positions (7-DOF) |
| `ee_pos` | `(N, 3)` | float64 | End-effector Cartesian position in world frame (metres) |
| `ee_rot` | `(N, 4)` | float64 | End-effector orientation as quaternion (`xyzw` convention) |
| `gripper_force` | `(N, 1)` | float64 | Gripper force reading (currently zero-filled) |
| `action` | `(N, 8)` | float64 | Operator action: `[ee_pos (3), ee_rot (4), gripper_force (1)]` |
| `_state_timestamps` | `(N,)` | float64 | RealSense SDK timestamp (milliseconds) at which each state was recorded |

`N` is the pre-allocated `max_episode_len`; only the first `n_written` entries contain valid data (the remainder are zeros). State is written once per primary-camera frame callback.

---

## Postprocessed episode

Postprocessing (`client/postprocess.py`) decodes the `.bag` files, synchronizes frames across cameras, encodes video, and writes aligned state. The `COMPLETE` marker file is created only after all steps succeed.

```
episode_N/
  <serial_1>.bag                         # Original raw recording (retained)
  <serial_2>.bag
  intrinsics.npz                         # Combined intrinsics from collection
  <serial_1>_intrinsics.npz             # Per-camera intrinsics (from bag playback)
  <serial_2>_intrinsics.npz
  raw_episode.zarr/                      # Raw state from collection (retained)
  episode.zarr/                          # Synchronized state (created by postprocessing)
    qpos
    ee_pos
    ee_rot
    gripper_force
    action
  captures/
    capture_<serial_1>/
      rgb.mp4
      ir_left.zarr/
      ir_right.zarr/
      timestamps.npy
      depth.zarr/                        # Only if FoundationStereo is enabled
    capture_<serial_2>/
      ...
  metadata.yaml
  COMPLETE
```

### Per-camera intrinsics (`<serial>_intrinsics.npz`)

During postprocessing, intrinsics are re-extracted from each `.bag` file and saved as individual per-camera `.npz` files. These use simpler, unprefixed keys:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `depth` | `(4,)` | float64 | `[fx, fy, ppx, ppy]` for the depth stream |
| `color` | `(4,)` | float64 | `[fx, fy, ppx, ppy]` for the color stream |
| `ir` | `(4,)` | float64 | `[fx, fy, ppx, ppy]` for the infrared stream (left) |
| `ir_baseline` | `(3,)` | float64 | Translation vector `[tx, ty, tz]` from IR1 to IR2 |

These per-camera files are the ones consumed by downstream code (e.g. FoundationStereo depth estimation).

### `captures/capture_<serial>/`

One subdirectory per camera, containing the synchronized and encoded capture data.

#### `rgb.mp4`

Encoded RGB video.

| Property | Value |
|----------|-------|
| Codec | mp4v (MPEG-4 Part 2) |
| Colour space | BGR (OpenCV default) |
| Resolution | 1280 x 720 |
| Frame rate | 15 fps |
| Frame count | `T` (synchronized length) |

#### `ir_left.zarr/` and `ir_right.zarr/`

Zarr directory stores containing the left and right infrared frames.

| Property | Value |
|----------|-------|
| Shape | `(T, 720, 1280)` |
| Dtype | `uint8` |
| Chunks | `(16, 720, 1280)` |
| Description | Grayscale IR frames from the stereo emitter pair |

#### `timestamps.npy`

Synchronized per-frame timestamps saved with `np.savez`. Despite the `.npy` extension the file is actually in `.npz` format. Load with:

```python
data = np.load("timestamps.npy")
timestamps = data["arr_0"]  # float64, shape (T,), milliseconds
```

Timestamps originate from the RealSense SDK (`frame.get_timestamp()`) and are in milliseconds.

#### `depth.zarr/` (optional)

Present only when FoundationStereo depth estimation is enabled (`stereo.enabled: true` in config, or run separately via `scripts/foundation_stereo.py`).

| Property | Value |
|----------|-------|
| Shape | `(T, 720, 1280)` |
| Dtype | `uint16` |
| Chunks | `(16, 720, 1280)` |
| Units | Millimetres |
| Invalid pixels | `0` (where stereo disparity was non-positive or depth > 65535 mm) |

Depth is computed from the IR stereo pair using the formula:

```
depth = (focal_length * scale * baseline) / disparity
```

Where `focal_length` is `ir[0]` (fx) from the intrinsics, `baseline` is `|ir_baseline[0]|` (the x-component of the stereo translation, ~0.05 m), and `scale` is the optional downscale factor (default 1.0).

### `episode.zarr/` (synchronized)

Created during postprocessing from `raw_episode.zarr`. Contains temporally-aligned state arrays with exactly `T` entries (matching the synchronized frame count). The alignment uses nearest-neighbor index mapping from the reference camera timestamps to the recorded state timestamps in `raw_episode.zarr`.

| Dataset | Shape | Description |
|---------|-------|-------------|
| `qpos` | `(T, 7)` | Synchronized joint positions |
| `ee_pos` | `(T, 3)` | Synchronized end-effector position |
| `ee_rot` | `(T, 4)` | Synchronized end-effector quaternion (xyzw) |
| `gripper_force` | `(T, 1)` | Synchronized gripper force |
| `action` | `(T, 8)` | Synchronized operator action |

The original unsynchronized state is preserved in `raw_episode.zarr/`.

### `metadata.yaml`

```yaml
n_timesteps: 237          # Synchronized frame count (T)
instruction: "Push the green T shape into the black outline of the T shape on the table"
dynamic_captures:
  - 217222061106           # Serial number of the primary (reference) camera
```

### `COMPLETE`

An empty marker file created by `touch` after postprocessing succeeds. Its presence tells the pipeline to skip this episode on subsequent runs.

---

## Temporal synchronization

Postprocessing aligns all cameras and robot state to a common timeline:

1. **Common window** — For each camera, the first and last frame timestamps define its recording interval. The sync window is `[t0, t1]` where `t0 = max(start times)` and `t1 = min(end times)` across all cameras. Frames outside this window are discarded.

2. **Reference camera** — The first camera (by enumeration order) serves as the temporal reference. Its timestamps within the sync window define the `T` output frames.

3. **Frame alignment** — Each non-reference camera's frames are aligned to the reference timestamps using a greedy forward nearest-neighbour search. The search index only advances forward, ensuring monotonic mapping.

4. **State alignment** — Robot state entries from `raw_episode.zarr` are mapped to the reference timestamps using the same nearest-neighbour approach on `_state_timestamps`. The synchronized `T`-length arrays are written to a new `episode.zarr`.

---

## Coordinate conventions

| Quantity | Convention | Units |
|----------|-----------|-------|
| End-effector position (`ee_pos`) | World frame (Franka base frame) | Metres |
| End-effector orientation (`ee_rot`) | Quaternion, `xyzw` order (scipy convention) | Unitless |
| Joint positions (`qpos`) | Franka Panda joint angles | Radians |
| Depth values | Distance from camera | Millimetres (uint16) |
| Timestamps | RealSense SDK clock | Milliseconds |
| IR baseline | Translation from IR1 to IR2 | Metres |
| Intrinsics (`fx`, `fy`, `ppx`, `ppy`) | Pinhole camera model | Pixels |
