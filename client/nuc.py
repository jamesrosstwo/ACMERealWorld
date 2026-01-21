import os
import threading
import time

import numpy as np
import polymetis_pb2
import torch
from omegaconf import DictConfig
from polymetis import RobotInterface
import paramiko


def stream_output(stream, prefix=''):
    for line in iter(stream.readline, ""):
        print(f"{prefix}{line.strip()}")


from scipy.spatial.transform import Rotation
import panda_py

class GripperInterface:
    def __init__(self, panda_robot):
        # reuse the existing connection
        self._panda = panda_robot
    
    def get_state(self):
        state = self._panda.get_state()
        return GripperState(
            width=state.gripper_width,
            max_width=state.gripper_max_width,
            is_grasped=state.gripper_width < 0.08
        )
    
    def goto(self, width: float, speed: float = 0.1, force: float = 10.0, blocking: bool = True):
        self._panda.get_gripper().move(width=width, speed=speed)
    
    def grasp(self, grasp_width: float = 0.0, speed: float = 0.1, force: float = 10.0, blocking: bool = True):
        try:
            return self._panda.get_gripper().grasp(width=grasp_width, speed=speed, force=force, epsilon=0.005)
        except RuntimeError:
            return False
    
    def stop(self):
        self._panda.get_gripper().stop()
    
    def close(self):
        pass


class GripperState:
    def __init__(self, width: float, max_width: float, is_grasped: bool):
        self.width = width
        self.max_width = max_width
        self.is_grasped = is_grasped


class NUCInterface:
    @property
    def pusht_home(self):
        return np.array([0.425, -0.375, 0.38]), np.array([0.942, 0.336, 0, 0])

    def __init__(self, ip: str, server: DictConfig, franka_ip: str):
        self._franka_ip = franka_ip
        self._nuc_ip = ip
        self._server_cfg = server

        print(f"Connecting to Panda at {self._franka_ip}")
        self._panda = panda_py.Panda(self._franka_ip)
        self._controller = None

        self._gripper = GripperInterface(self._panda)

        self._desired_eef_pos, self._desired_eef_rot = self.pusht_home

    def get_desired_ee_pose(self):
        return np.concatenate([self._desired_eef_pos, self._desired_eef_rot]).copy()

    def get_robot_state(self):
        state = self._panda.get_state()
        qpos = np.array(state.q)
        qvel = np.array(state.dq)
        
        # O_T_EE is a list/array of 16 floats (column-major usually in libfranka, panda-py might wrap it)
        # panda-py usually returns numpy array 4x4
        O_T_EE = np.array(state.O_T_EE).reshape(4, 4).T # Transpose because libfranka is col-major
        
        ee_pos = O_T_EE[:3, 3]
        ee_rot = Rotation.from_matrix(O_T_EE[:3, :3]).as_quat() # xyzw
        
        # Match expected dict keys
        st = dict(qpos=qpos, qvel=qvel, ee_pos=ee_pos, ee_rot=ee_rot, gripper_force=np.zeros(1))
        return st

    def forward_kinematics(self, joint_positions: torch.Tensor):
        # panda-py usually exposes a helper for FK
        # If not available directly on instance, we might need simple kinematics.
        # Assuming panda-py has `fk(q)` based on common bindings.
        # If not, this method will fail and user will report.
        # Since input is tensor, convert to list/numpy
        q = joint_positions.cpu().numpy().tolist()
        try:
            pose = panda_py.fk(q) # Potential API guess, or self._panda.get_model().pose(frames.EE, q)
            # If panda_py returns 4x4 matrix
            mat = np.array(pose).reshape(4, 4).T
            pos = mat[:3, 3]
            rot = Rotation.from_matrix(mat[:3, :3]).as_quat()
            return pos, rot
        except AttributeError:
            print("WARNING: forward_kinematics not implemented in panda-py NUCInterface")
            return np.zeros(3), np.array([0, 0, 0, 1])

    def send_control(self, eef_pos: np.ndarray, eef_rot: np.ndarray, gripper: np.ndarray):
        self._desired_eef_pos = eef_pos.copy()
        self._desired_eef_rot = eef_rot.copy()
        if self._controller:
             # Convert xyzw to wxyz if needed? Scipy is xyzw. Libfranka usually xyzw.
             # panda_py set_attractor expected format?
             self._controller.set_attractor(eef_pos, eef_rot)

    def send_control_tensor(self, eef_pos: torch.Tensor, eef_rot: torch.Tensor, gripper: torch.Tensor):
        self.send_control(eef_pos.cpu().numpy(), eef_rot.cpu().numpy(), None)

    def reset(self):
        # Move to home
        pos, rot = self.pusht_home
        self._panda.move_to_pose(pos, rot)
        
        self._gripper.grasp(speed=0.01, force=1, blocking=True)
        print("Grasping")

    def start(self):
        # Start impedance controller
        # We need to install/import controllers if they are separate
        # Assuming panda_py.controllers.CartesianImpedance exists
        try:
            from panda_py import controllers
            self._controller = controllers.CartesianImpedance()
            self._panda.start_controller(self._controller)
        except ImportError:
            print("ERROR: panda_py.controllers not found. Control will not work.")

    def close(self):
        if self._controller:
            self._panda.stop_controller()
