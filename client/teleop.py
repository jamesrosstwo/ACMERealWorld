from dataclasses import dataclass
from typing import Tuple, List

import numpy as np

from gello.dynamixel.driver import DynamixelDriver
from polymetis import FrankaArm


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
    def __init__(self, joint_signs: List[int], *args, **kwargs):
        self._args = _GelloArgs(joint_signs=joint_signs, *args, **kwargs)
        self._driver = DynamixelDriver(self._args.joint_ids, port=self._args.port, baudrate=self._args.baud_rate)
        self._joint_signs = np.array(joint_signs)
        self._joint_adjustment = np.zeros(7)
        self._eef_pos_adjustment = np.zeros(3)
        self._reference_robot = FrankaArm() # Assuming this GELLO is modeled after a franka.

    def _get_joints(self):
        return (self._joint_signs * self._driver.get_joints()[:7])

    def get_joint_angles(self):
        return self._get_joints() + self._joint_adjustment

    def get_eef_pose(self):
        return self._reference_robot.get_ee_pose(self.get_joint_angles())

    def zero_controls(self, qpos):
        self._joint_adjustment = qpos - self._get_joints()
