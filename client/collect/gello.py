"""GELLO teleoperation interface.

Reads joint angles from a GELLO controller (Dynamixel servo-based input device)
and maps them to Franka Panda joint space for teleoperated demonstration
collection. The Dynamixel reader runs in a dedicated subprocess so it has its
own GIL — main-process Python load (cameras, teleop loop, state worker)
cannot starve the read thread, which previously caused tens-to-hundreds of
ms stalls and stale-then-jump on the cached joint values.
"""
import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from omegaconf import DictConfig


# Shared-buffer layout (float64 entries):
#   [0..n-1]  joint angles in radians (n = num_joints)
#   [n]       wall-clock timestamp of the most recent successful publish
#   [n+1]     alive flag (0.0 until the first successful publish, then 1.0)
_TS_OFFSET = 0
_ALIVE_OFFSET = 1


def _reader_proc(port, ids, baudrate, shared_buf, stop_flag, n):
    """Subprocess entry: owns the FTDI port, polls servos, writes radians to
    shared_buf. Imports are local so the child only pulls in what it needs.
    """
    from gello.dynamixel.driver import (
        DynamixelDriver, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION,
    )
    from dynamixel_sdk.robotis_def import COMM_SUCCESS

    shared = np.frombuffer(shared_buf, dtype=np.float64, count=n + 2)

    driver = DynamixelDriver(ids, port=port, baudrate=baudrate)
    driver._portHandler.ser.timeout = 0.001
    # Stop the SDK's internal reader thread; we drive reads from this loop.
    driver._stop_thread.set()
    driver._reading_thread.join()

    gsr = driver._groupSyncRead
    ph = gsr.ph
    port_h = gsr.port
    data_len = gsr.data_length
    raw = np.zeros(n, dtype=int)

    try:
        while not stop_flag.value:
            time.sleep(0.001)
            rc = gsr.txPacket()
            if rc != COMM_SUCCESS:
                gsr.last_result = False
                continue
            complete = True
            for i, dxl_id in enumerate(ids):
                data, result, _ = ph.readRx(port_h, dxl_id, data_len)
                if result != COMM_SUCCESS:
                    complete = False
                    break
                gsr.data_dict[dxl_id] = data
            if not complete:
                gsr.last_result = False
                continue
            gsr.last_result = True
            for i, dxl_id in enumerate(ids):
                if gsr.isAvailable(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                    angle = gsr.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                    raw[i] = np.int32(np.uint32(angle))
                else:
                    complete = False
                    break
            if complete:
                shared[:n] = raw.astype(np.float64) / 2048.0 * np.pi
                shared[n + _TS_OFFSET] = time.time()
                shared[n + _ALIVE_OFFSET] = 1.0
    finally:
        try:
            driver.close()
        except Exception:
            pass


@dataclass
class _GelloArgs:
    port: str
    """The port that GELLO is connected to."""

    start_joints: Tuple[float, ...]
    """The joint angles that the GELLO is placed in at (in radians)."""

    joint_signs: Tuple[float, ...]
    """The joint angles that the GELLO is placed in at (in radians)."""

    base_index: int = 1
    baud_rate: int = 57600

    gripper: bool = True
    """Whether or not the gripper is attached."""

    def __post_init__(self):
        assert len(self.joint_signs) == len(self.start_joints)
        for idx, j in enumerate(self.joint_signs):
            assert (
                j == -1 or j == 1
            ), f"Joint idx: {idx} should be -1 or 1, but got {j}."

    @property
    def num_robot_joints(self) -> int:
        return len(self.start_joints)

    @property
    def num_joints(self) -> int:
        extra_joints = 1 if self.gripper else 0
        return self.num_robot_joints + extra_joints

    @property
    def joint_ids(self) -> List[int]:
        return list(range(self.base_index, self.num_joints + self.base_index))


class GELLOInterface:
    def __init__(self, joint_signs: List[int], gripper: DictConfig, *args, **kwargs):
        self._args = _GelloArgs(joint_signs=tuple(joint_signs), *args, **kwargs)
        n = self._args.num_joints
        self._n = n

        # Lock-free shared memory inherited via fork. Tearing across the n
        # joint-angle entries is harmless: even a half-old, half-new sample
        # is sub-ms worth of angular change.
        self._shared_buf = mp.RawArray("d", n + 2)
        self._shared = np.frombuffer(self._shared_buf, dtype=np.float64, count=n + 2)
        self._shared[:] = 0.0
        self._stop_flag = mp.Value("i", 0, lock=False)

        self._process = mp.Process(
            target=_reader_proc,
            args=(self._args.port, self._args.joint_ids, self._args.baud_rate,
                  self._shared_buf, self._stop_flag, n),
            daemon=True,
        )
        self._process.start()

        # Block until the subprocess publishes its first complete reading.
        deadline = time.time() + 30.0
        while self._shared[n + _ALIVE_OFFSET] == 0.0:
            if not self._process.is_alive():
                raise RuntimeError(
                    f"GELLO subprocess exited before producing a reading "
                    f"(exitcode={self._process.exitcode})"
                )
            if time.time() > deadline:
                self.close()
                raise RuntimeError("GELLO subprocess did not produce a reading within 30s")
            time.sleep(0.05)

        self._joint_signs = np.array(joint_signs)
        self._joint_adjustment = np.zeros(7)
        self._eef_pos_adjustment = np.zeros(3)
        self._gripper = gripper

    def close(self):
        try:
            self._stop_flag.value = 1
        except Exception:
            pass
        if hasattr(self, "_process"):
            try:
                self._process.join(timeout=2.0)
                if self._process.is_alive():
                    self._process.terminate()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _read_angles(self):
        return self._shared[: self._n].copy()

    def get_gripper(self):
        jointval = self._read_angles()[-1]
        # Map from dynamixel range to [0=open, 1=closed].
        # sign determines direction: +1 means open < closed, -1 means open > closed.
        if self._gripper.sign > 0:
            gripper_range = [self._gripper.open, self._gripper.closed]
        else:
            gripper_range = [self._gripper.closed, self._gripper.open]
        return float(np.clip(np.interp(jointval, gripper_range, [1., 0.]), 0., 1.))

    def _get_joints(self):
        return self._joint_signs * self._read_angles()[:7]

    def get_joint_angles(self) -> np.ndarray:
        return self._get_joints() + self._joint_adjustment

    def zero_controls(self, qpos):
        self._joint_adjustment = qpos - self._get_joints()
        print(f"[gello] zero_controls: joint_adjustment={np.array2string(self._joint_adjustment, precision=4)}")
