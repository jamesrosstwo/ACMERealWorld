"""NUC robot interface.

Communicates with a Franka Panda robot arm via panda-py (libfranka).
Provides end-effector Cartesian impedance control, forward kinematics, gripper
commands, and robot state queries (joint positions, EE pose).
"""
import threading
import time

import numpy as np
import torch
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation
import panda_py


class GripperInterface:
    def __init__(self, ip_address: str, hysteresis: float = 0.1):
        print(f"Initializing GripperInterface via panda-py at {ip_address}")
        self._gripper = panda_py.libfranka.Gripper(ip_address)

        # Async state
        self._thread = None
        self._target_grasp_state = False  # False=Open, True=Grasped
        self._hysteresis = hysteresis

    def get_state(self):
        state = self._gripper.read_once()
        return GripperState(
            width=state.width,
            max_width=state.max_width,
            is_grasped=state.is_grasped
        )

    def goto(self, width: float, speed: float = 0.1, force: float = 10.0, blocking: bool = True):
        self._gripper.move(width=width, speed=speed)

    def grasp(self, grasp_width: float = 0.0, speed: float = 0.1, force: float = 10.0, blocking: bool = True):
        try:
            return self._gripper.grasp(width=grasp_width, speed=speed, force=force, epsilon_outer=0.1)
        except RuntimeError:
            return False

    def stop(self):
        self._gripper.stop()

    def close(self):
        pass

    def _do_grasp(self):
        self.grasp(speed=0.1, force=10.0, blocking=True)

    def _do_open(self):
        mx = self.get_state().max_width
        self.goto(width=mx, speed=0.1, blocking=True)

    def act_async(self, gripper_val: float):
        val = float(gripper_val)

        grasp_threshold = 0.5 + self._hysteresis
        open_threshold = 0.5 - self._hysteresis

        if self._thread and self._thread.is_alive():
            return

        if val > grasp_threshold and not self._target_grasp_state:
            self._target_grasp_state = True
            self._thread = threading.Thread(target=self._do_grasp)
            self._thread.start()
        elif val < open_threshold and self._target_grasp_state:
            self._target_grasp_state = False
            self._thread = threading.Thread(target=self._do_open)
            self._thread.start()


class GripperState:
    def __init__(self, width: float, max_width: float, is_grasped: bool):
        self.width = width
        self.max_width = max_width
        self.is_grasped = is_grasped


class NUCInterface:
    @property
    def home(self):
        return self._home_pos.copy(), self._home_rot.copy()

    def __init__(self, ip: str, server: DictConfig, franka_ip: str,
                 home_pos=(0.30582778, 0., 0.48467681), home_rot=(1., 0., 0., 0.)):
        self._franka_ip = franka_ip
        self._nuc_ip = ip
        self._server_cfg = server
        self._home_pos = np.array(home_pos)
        self._home_rot = np.array(home_rot)

        print(f"Connecting to Panda at {self._franka_ip}")
        self._panda = panda_py.Panda(self._franka_ip)
        self._controller = None

        hysteresis = getattr(server, 'gripper_hysteresis', 0.1)
        self._gripper = GripperInterface(self._franka_ip, hysteresis=hysteresis)

        self._desired_eef_pos = self._panda.get_position()
        self._desired_eef_rot = self._panda.get_orientation(scalar_first=False)

    def get_desired_ee_pose(self):
        return np.concatenate([self._desired_eef_pos, self._desired_eef_rot]).copy()

    def get_robot_state(self):
        R = self._panda.get_pose()[:3, :3]
        t = self._panda.get_position()
        ee_rot = Rotation.from_matrix(R).as_quat()  # xyzw
        return dict(qpos=self._panda.q, ee_pos=t, ee_rot=ee_rot, gripper_force=np.zeros(1))

    def forward_kinematics(self, joint_positions: torch.Tensor):
        q = joint_positions.cpu().numpy().reshape(7, 1)
        try:
            pose = panda_py.fk(q)
            mat = np.array(pose).reshape(4, 4)
            pos = mat[:3, 3]
            rot = Rotation.from_matrix(mat[:3, :3]).as_quat()
            return pos, rot
        except AttributeError:
            print("WARNING: forward_kinematics (panda_py.fk) not found/working")
            return np.zeros(3), np.array([1, 0, 0, 0])

    def send_control(self, eef_pos: np.ndarray, eef_rot: np.ndarray, gripper: np.ndarray):
        self._desired_eef_pos = eef_pos.copy()
        self._desired_eef_rot = eef_rot.copy()
        if self._controller:
            self._controller.set_control(eef_pos, eef_rot)

        if gripper is not None:
            g_val = gripper.item() if hasattr(gripper, 'item') else float(gripper)
            self._gripper.act_async(g_val)

    def send_control_tensor(self, eef_pos: torch.Tensor, eef_rot: torch.Tensor, gripper: torch.Tensor):
        g = gripper.cpu().numpy() if gripper is not None else None
        self.send_control(eef_pos.cpu().numpy(), eef_rot.cpu().numpy(), g)

    def home_gripper(self):
        """Calibrate the gripper via homing in a background thread."""
        threading.Thread(target=self._gripper._gripper.homing, daemon=True).start()

    def reset(self):
        self._panda.move_to_start()
        home_pos, home_rot = self.home
        self._panda.move_to_pose([home_pos], [home_rot])

    def start(self):
        try:
            from panda_py import controllers
            impedance = np.eye(6)
            impedance[:3, :3] *= 400   # translational stiffness (N/m)
            impedance[3:, 3:] *= 60    # rotational stiffness (Nm/rad)
            self._controller = controllers.CartesianImpedance(
                impedance=impedance, nullspace_stiffness=15.0
            )
            self._panda.start_controller(self._controller)
        except ImportError:
            print("ERROR: panda_py.controllers not found. Control will not work.")

    def close(self):
        if self._controller:
            self._panda.stop_controller()
