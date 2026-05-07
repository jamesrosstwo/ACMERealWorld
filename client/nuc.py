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
    def __init__(self, ip_address: str, hysteresis: float, logging: bool = True):
        self._logging = logging
        self._log(f"Initializing GripperInterface via panda-py at {ip_address}")
        self._gripper = panda_py.libfranka.Gripper(ip_address)

        # Async state
        self._thread = None
        self._target_grasp_state = False  # False=Open, True=Grasped
        self._hysteresis = hysteresis

    def _log(self, msg: str):
        if self._logging:
            print(f"\t[gripper] {msg}")

    def get_state(self):
        state = self._gripper.read_once()
        return GripperState(
            width=state.width,
            max_width=state.max_width,
            is_grasped=state.is_grasped
        )

    def goto(self, width: float, speed: float = 0.1, force: float = 10.0, blocking: bool = False):
        self._gripper.move(width=width, speed=speed)

    def grasp(self, grasp_width: float = 0.0, speed: float = 0.1, force: float = 10.0, blocking: bool = False):
        try:
            return self._gripper.grasp(width=grasp_width, speed=speed, force=force, epsilon_outer=0.04)
        except RuntimeError:
            return False

    def stop(self):
        self._gripper.stop()

    def close(self):
        pass

    def _do_grasp(self):
        result = self.grasp(speed=0.1, force=10.0, blocking=False)
        self._log(f"grasp result: {result}")

    def _do_open(self):
        mx = self.get_state().max_width
        self._log(f"opening to max_width={mx}")
        self.goto(width=mx, speed=0.1, blocking=False)

    def act_async(self, gripper_val: float):
        val = float(gripper_val)
        self._log(f"val={val:.3f}")

        grasp_threshold = 0.5 + self._hysteresis
        open_threshold = 0.5 - self._hysteresis

        if self._thread and self._thread.is_alive():
            return

        if val > grasp_threshold and not self._target_grasp_state:
            self._log(f"val={val:.3f} > {grasp_threshold} — GRASPING")
            self._target_grasp_state = True
            self._thread = threading.Thread(target=self._do_grasp)
            self._thread.start()
        elif val < open_threshold and self._target_grasp_state:
            self._log(f"val={val:.3f} < {open_threshold} — OPENING")
            self._target_grasp_state = False
            self._thread = threading.Thread(target=self._do_open)
            self._thread.start()
        else:
            pass


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
                 home_pos=None, home_rot=None):
        self._franka_ip = franka_ip
        self._nuc_ip = ip
        self._server_cfg = server
        self._home_pos = np.array(home_pos)
        self._home_rot = np.array(home_rot)

        print(f"Connecting to Panda at {self._franka_ip}")
        self._panda = panda_py.Panda(self._franka_ip)
        self._controller = None
        self._is_joint_space = False
        self._prev_q = None
        self._prev_t = None

        hysteresis = server.gripper_hysteresis
        gripper_logging = server.get("gripper_logging", True)
        self._gripper = GripperInterface(self._franka_ip, hysteresis=hysteresis, logging=gripper_logging)

        self._desired_eef_pos = self._panda.get_position()
        self._desired_eef_rot = self._panda.get_orientation(scalar_first=False)

    def get_desired_ee_pose(self):
        return np.concatenate([self._desired_eef_pos, self._desired_eef_rot]).copy()

    def get_robot_state(self):
        R = self._panda.get_pose()[:3, :3]
        t = self._panda.get_position()
        ee_rot = Rotation.from_matrix(R).as_quat()  # xyzw
        gripper_state = self._gripper.get_state()
        gripper_force = np.array([1.0 - gripper_state.width / gripper_state.max_width])
        return dict(qpos=self._panda.q, ee_pos=t, ee_rot=ee_rot, gripper_force=gripper_force)

    def get_controller_diagnostics(self):
        """Compute low-level controller diagnostics.

        Returns a dict with:
          - cart_pos_error (6,): Cartesian pose error (3 translation + 3 rotation)
          - cart_vel_error (6,): Cartesian velocity (finite-difference estimate)
          - tau_stiffness (7,): Joint torques from Cartesian stiffness (J^T K x_err)
          - tau_damping (7,): Joint torques from Cartesian damping (-J^T D dx)
          - tau_nullspace (7,): Joint torques from nullspace stiffness + damping
        """
        impedance, damping, ns_stiffness, ns_damping = self._parse_impedance()

        q = self._panda.q.copy()
        pose = self._panda.get_pose()  # 4x4 homogeneous
        actual_pos = pose[:3, 3]
        actual_rot = Rotation.from_matrix(pose[:3, :3])

        desired_pos = self._desired_eef_pos
        desired_rot = Rotation.from_quat(self._desired_eef_rot)

        # Cartesian position error
        pos_err = desired_pos - actual_pos
        # Orientation error as rotation vector (angle-axis)
        rot_err = (desired_rot * actual_rot.inv()).as_rotvec()
        cart_pos_error = np.concatenate([pos_err, rot_err])

        # Estimate Cartesian velocity via finite-difference on joint positions
        now = time.time()
        if self._prev_q is not None and self._prev_t is not None:
            dt = now - self._prev_t
            if dt > 0:
                dq = (q - self._prev_q) / dt
            else:
                dq = np.zeros(7)
        else:
            dq = np.zeros(7)
        self._prev_q = q.copy()
        self._prev_t = now

        # Get Jacobian at current configuration
        try:
            J = np.array(self._panda.get_jacobian()).reshape(6, 7)
        except (AttributeError, RuntimeError):
            # Fallback: zero Jacobian means we can't decompose torques
            J = np.zeros((6, 7))

        cart_vel = J @ dq  # 6D Cartesian velocity
        cart_vel_error = cart_vel  # velocity error (desired vel is zero for impedance)

        # Torque contributions
        # Stiffness: J^T K x_err
        tau_stiffness = J.T @ (impedance @ cart_pos_error)
        # Damping: -J^T D dx  (opposes motion)
        tau_damping = -J.T @ (damping @ cart_vel)
        # Nullspace (approximate): project into nullspace of J
        # For logging, just show raw joint-space stiffness/damping terms
        q_home = panda_py.ik(self._home_pos, self._home_rot, q_init=q)
        tau_nullspace = ns_stiffness * (q_home - q) - ns_damping * dq

        return dict(
            cart_pos_error=cart_pos_error.astype(np.float32),
            cart_vel_error=cart_vel_error.astype(np.float32),
            tau_stiffness=tau_stiffness.astype(np.float32),
            tau_damping=tau_damping.astype(np.float32),
            tau_nullspace=tau_nullspace.astype(np.float32),
        )

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
            if self._is_joint_space:
                q_desired = panda_py.ik(eef_pos, eef_rot, q_init=self._panda.q)
                self._controller.set_control(q_desired)
            else:
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

    def _parse_impedance(self):
        imp_cfg = self._server_cfg.impedance
        trans = list(imp_cfg.translational_stiffness)
        rot = list(imp_cfg.rotational_stiffness)
        trans_d = list(imp_cfg.translational_damping)
        rot_d = list(imp_cfg.rotational_damping)
        impedance = np.diag(trans + rot).astype(np.float64)
        damping = np.diag(trans_d + rot_d).astype(np.float64)
        ns_stiffness = np.array(imp_cfg.nullspace_stiffness, dtype=np.float64)
        ns_damping = np.array(imp_cfg.nullspace_damping, dtype=np.float64)
        return impedance, damping, ns_stiffness, ns_damping

    def _make_controller(self):
        from panda_py import controllers
        ctrl_type = self._server_cfg.get("controller", "cartesian")
        impedance, damping, ns_stiffness, ns_damping = self._parse_impedance()
        if ctrl_type == "hybrid_joint":
            self._is_joint_space = True
            return controllers.HybridJointImpedance(
                Kx=impedance,
                Kxd=damping,
                Kq=ns_stiffness,
                Kqd=ns_damping,
            )
        else:
            self._is_joint_space = False
            return controllers.PolymetisImpedance(
                impedance=impedance,
                damping=damping,
                nullspace_stiffness=ns_stiffness,
                nullspace_damping=ns_damping,
            )

    def reset(self, open_gripper: bool = True):
        home_pos, home_rot = self.home
        reset_pos = home_pos + np.array([0.0, 0.0, 0.04])
        # Drive to home with libfranka's motion generator before handing off to the impedance controller.
        if self._controller:
            self._panda.stop_controller()
            self._controller = None
        self._panda.move_to_pose([reset_pos], [home_rot])
        self.start()
        self.send_control(home_pos, home_rot, gripper=None)

        if open_gripper:
            self._gripper._do_open()
            self._gripper._target_grasp_state = False

    def start(self):
        if self._controller:
            self._panda.stop_controller()
        self._controller = self._make_controller()
        self._panda.start_controller(self._controller)

    def close(self):
        if self._controller:
            self._panda.stop_controller()
