# Hardware Setup

This guide covers the physical setup and wiring for the ACME data collection system: networking between the Franka controller and workstation, USB connectivity for cameras and peripherals, and the GELLO teleoperation arm.

---

## Network setup

The Franka Panda is controlled through an Intel NUC that runs a real-time control server (`panda_py` / libfranka). The NUC, Franka controller, and your workstation must all be on the same local network.

### Topology

```
Workstation ──── Ethernet switch ──── NUC ──── Franka controller
 192.168.1.x         (GbE)       192.168.1.106    192.168.1.107
```

The NUC connects to the Franka controller over a dedicated Ethernet link. Your workstation connects to the NUC through the same subnet.

### Addressing

| Device | Default IP | Notes |
|--------|-----------|-------|
| Intel NUC (control server) | `192.168.1.106` | Runs `panda_py` with real-time kernel |
| Franka controller | `192.168.1.107` | Franka Desk web interface also available here |
| Workstation | `192.168.1.*` | Any free address on the same subnet |

These addresses are configurable in `config/collect.yaml` and `config/eval.yaml` under `nuc.ip` and `nuc.franka_ip`.

### Workstation network configuration

Assign a static IP on the `192.168.1.0/24` subnet to the Ethernet interface connected to the switch:

```bash
# Identify your Ethernet interface
ip link show

# Example: assign a static IP using nmcli
sudo nmcli con mod "Wired connection 1" \
  ipv4.addresses 192.168.1.100/24 \
  ipv4.method manual

sudo nmcli con up "Wired connection 1"
```

Verify connectivity:

```bash
ping 192.168.1.106   # NUC
ping 192.168.1.107   # Franka controller
```

You should also be able to access the Franka Desk web interface at `https://192.168.1.107` in a browser to unlock the robot joints and check system status.

### Tips

- Use a **gigabit Ethernet switch** — the NUC control loop is latency-sensitive.
- If your workstation also needs internet access, use a second network interface (e.g. Wi-Fi or a second Ethernet port) for the WAN connection. Do not route the `192.168.1.0/24` subnet through a gateway.
- The NUC should run a real-time kernel (`PREEMPT_RT`) for reliable 1 kHz Franka control. See the [Franka documentation](https://frankaemika.github.io/docs/) for kernel setup.
- This project uses a [custom fork of panda-py](https://github.com/jamesrosstwo/panda-py/tree/polymetis-impedance) that adds a `PolymetisImpedance` controller with per-axis Cartesian damping and per-joint nullspace gains (matching [Polymetis/fairo](https://github.com/facebookresearch/fairo) defaults). It is installed automatically via `environment.yaml`.

---

## USB cameras and PCIe expansion

### RealSense camera connectivity

Each Intel RealSense D400-series camera streams RGB, depth, and stereo IR at 1280x720 @ 15 fps. This requires sustained USB 3.x bandwidth (~1.5 Gbps per camera with all streams active).

Most workstations have 1–2 USB 3.x host controllers on the motherboard. When running **two or more cameras simultaneously**, you will likely saturate a single controller's bandwidth, causing dropped frames or failed pipeline starts.

### Adding a USB PCIe card

A dedicated USB 3.x PCIe card adds an independent host controller with its own bandwidth pool. This is the recommended approach when using more than one RealSense camera.

**What to look for:**

- A PCIe x1 (or larger) USB 3.1+ card with a dedicated controller chip (e.g. Renesas uPD720202, ASMedia ASM3242). Avoid cards that are just internal hubs behind a single controller.
- At least as many ports as you have cameras. Spread cameras across different controllers — ideally one camera per controller.
- External power input (Molex/SATA) is a plus; some cards need it to supply enough current for multiple cameras.

**After installation:**

1. Verify the new controller is detected:
   ```bash
   lspci | grep -i usb
   ```
   You should see an additional USB host controller entry.

2. Check that cameras enumerate on the new controller:
   ```bash
   # List RealSense devices
   rs-enumerate-devices | grep "Serial Number"

   # See which USB bus each camera is on
   lsusb -t
   ```

3. Distribute cameras so that each USB host controller serves at most one camera. If you have two cameras and two controllers (onboard + PCIe card), plug one camera into each.

### Troubleshooting USB bandwidth

If you see errors like `failed to start pipeline` or `frame didn't arrive within 5000ms`:

- Check `dmesg` for USB errors: `dmesg | grep -i usb`
- Reduce the number of active streams or resolution in `config/collect.yaml`
- Ensure cameras are on **USB 3.x ports** (not USB 2.0) — the RealSense viewer (`realsense-viewer`) shows the connection type
- Avoid USB hubs; connect cameras directly to host controller ports

---

## GELLO teleoperation arm

[GELLO](https://github.com/wuphilipp/gello_software) is a low-cost teleoperation controller built from 7 Dynamixel servos that mirrors the Franka Panda's kinematics. The operator moves the GELLO arm and the robot follows.

### Hardware

The GELLO arm hardware is described in the [GELLO project page](https://wuphilipp.github.io/gello/) and the [gello_software repository](https://github.com/wuphilipp/gello_software). Refer to these resources for the bill of materials, 3D-printed parts, and assembly instructions.

### Connection

GELLO connects to the workstation via a USB-to-serial adapter (FTDI). The default serial port expected by the configuration is:

```
/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA2U2CD-if00-port0
```

This path is stable across reboots (unlike `/dev/ttyUSB*` which can change). Update the path in `config/collect.yaml` under `gello.port` if your adapter has a different ID.

### Software setup

The GELLO driver is included as a git submodule at `submodules/gello_software/`. Follow its README for Dynamixel SDK installation and servo calibration.

Ensure your user has serial port access:

```bash
sudo usermod -aG dialout $USER
# Log out and back in for the group change to take effect
```

---

## Robotiq 2F-85 gripper

The Robotiq 2F-85 replaces the Franka Panda Hand. Unlike the Panda Hand (which is
driven through the Franka control box over libfranka), the Robotiq is a third-party
gripper that talks **Modbus RTU over RS-485** directly to the control PC. `panda_py`
keeps the arm; the [`RobotiqGripper`](../client/nuc.py) backend owns the gripper on the
same PC — **no NUC is involved in gripper control**. Select it with
`gripper.type: robotiq` under `nuc.server` in `config/eval.yaml` / `config/collect.yaml`.

### Wiring

```
2F-85 ── coupling ── 24 V supply
                  └── RS-485 pigtail ── ACC-ADT-USB-RS485 ── PC USB
```

- Power the gripper coupling from a **24 V** supply (the Robotiq does **not** draw power
  from the Franka).
- Connect the RS-485 pigtail to the `ACC-ADT-USB-RS485` converter that ships with the
  gripper, then to a PC USB port. Confirm it enumerates: `ls -l /dev/ttyUSB*`.

### Serial permissions and a stable port

The converter is an FTDI device, like GELLO — so give your user serial access
(`sudo usermod -aG dialout $USER`, then re-login) and prefer a stable
`/dev/serial/by-id/...` symlink over `/dev/ttyUSB*` so it can't get swapped with the
GELLO FTDI:

```bash
ls -l /dev/serial/by-id/   # find the Robotiq FTDI symlink
```

Put that path in `gripper.port` (instead of `"auto"`) in the eval/collect configs.

### Franka Desk end-effector configuration (important)

With the Panda Hand removed, configure the new end-effector in Franka Desk, otherwise the
impedance controller's gravity compensation will be wrong (the arm sags/drifts):

1. Remove the "Franka Hand" end-effector.
2. Set the **end-effector mass / centre-of-mass / inertia** for the 2F-85 + coupling
   (~0.9 kg).
3. Set the **TCP transform** (`NE_T_EE` / `F_T_EE`) to the new fingertip.

Because `panda_py.get_position()` returns the TCP-inclusive EE pose, the new TCP shifts
the reported pose — re-check `task.home_pos` / `home_rot` after setting the Desk TCP
(cartesian mode is TCP-frame; qpos mode FKs in flange frame).

### Driver and quick test

The driver is [`pyRobotiqGripper`](https://github.com/castetsb/pyRobotiqGripper), installed
via `environment.yaml`. Bench-test the gripper on its own (arm idle) with:

```bash
python -m scripts.test_robotiq            # auto-detect port
python -m scripts.test_robotiq --port /dev/serial/by-id/usb-FTDI...-if00-port0
```

It activates/calibrates the gripper, opens and closes it, and prints the `gripper_force`
observation (≈0.0 open, ≈1.0 closed) so you can confirm serial, activation, and the obs
mapping before running the full collect/eval loops.

### Caveats

- **Obs distribution shift**: a policy trained on Panda-Hand `gripper_force` will see the
  Robotiq's value. The convention (0=open, 1=closed) matches and the hysteresis dispatch
  makes the *command* binary, but the continuous obs trace won't be identical — expect to
  recollect data / re-tune for best results. `task.zero_gripper_obs: true` is an escape
  hatch.
- The default Robotiq Modbus slave id is `9` (`gripper.device_id`); change it only if a
  non-default was set on the gripper.

---

## Wrist ZED camera (data collection)

The wrist camera is a ZED Mini. Eval already streams it ([client/eval/zed.py](../client/eval/zed.py));
collection records it too via [client/collect/zed.py](../client/collect/zed.py).

### How it fits the pipeline

Collection is two-phase, and the ZED mirrors the RealSense flow:

| Phase | RealSense | ZED |
|-------|-----------|-----|
| Live (`client.collect`) | records `<serial>.bag` | records `<serial>.svo2` |
| Offline (`client.collect.postprocess`) | `RSBagProcessor` decodes the bag | `ZEDSvoProcessor` decodes the svo2 |

Both decoders emit the same `(color, ts, ir_left, ir_right, serial)` tuple, so
`ACMEWriter` and the dataset format are unchanged. The ZED is stored as **full stereo**:
`rgb.mp4` = rectified LEFT, `ir_left/ir_right.zarr` = the rectified LEFT/RIGHT grayscale pair,
and a `calibration.yaml` (ZED intrinsics + baseline) so `scripts/foundation_stereo.py` can
produce wrist depth later.

Declare it in [config/collect.yaml](../config/collect.yaml) under `zed_cameras`, keeping the
`serial` label and ordering aligned with `eval.yaml`'s `obs_cams` so the `rgb_<i>` wrist slot
matches between collection and eval.

### Timestamp synchronization (Tier-1)

All streams are put on one host-referenced Unix-epoch clock (milliseconds):

- RealSense: collection **enables and asserts** the global-time domain
  ([client/collect/realsense.py](../client/collect/realsense.py)); if a camera isn't on the
  global-time domain, collection aborts rather than silently misalign.
- ZED: the SVO's `TIME_REFERENCE.IMAGE` timestamps are already epoch ms.
- Postprocess asserts the per-camera sync residual median is under half a frame and records it
  in `metadata.yaml` (`camera_sync_ms`) — a loud failure beats silent misalignment.

### Requirements (collection PC **and** postprocess box)

- ZED SDK + matching `pyzed` (see [environment.yaml](../environment.yaml); not a clean PyPI
  package — install via the SDK or the repo's `pyzed-4.2-...whl`).
- An **NVIDIA GPU + compatible driver** — the ZED SDK initializes CUDA even for SVO playback,
  so a CPU-only postprocess box won't work.
- The `.svo2` files must sit alongside the `.bag` files in each episode dir (relevant if
  postprocess runs on a different machine than collection).