"""NUC robot interface.

Communicates with a Franka Panda robot arm via panda-py (libfranka).
Provides end-effector Cartesian impedance control, forward kinematics, gripper
commands, and robot state queries (joint positions, EE pose).

The gripper is pluggable (see :func:`make_gripper`): the Franka Panda Hand
(:class:`PandaGripper`, through the control box) or a Robotiq 2F-85
(:class:`RobotiqGripper`, over USB->RS485 Modbus RTU on the control PC,
independent of the Franka). Selected via ``server.gripper.type`` in config.
"""
import threading
import time

import numpy as np
import torch
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation
import panda_py
import panda_py.constants


class GripperState:
    def __init__(self, width: float, max_width: float, is_grasped: bool):
        self.width = width
        self.max_width = max_width
        self.is_grasped = is_grasped


class BaseGripper:
    """Backend-agnostic gripper interface.

    The threshold + hysteresis dispatch in :meth:`act_async` and the async
    bookkeeping are identical across grippers and live here. A backend talks to
    its specific hardware by implementing :meth:`get_state`, :meth:`_do_grasp`,
    :meth:`_do_open`, and (optionally) :meth:`home` / :meth:`stop`.
    """

    def __init__(self, hysteresis: float, logging: bool = True):
        self._logging = logging
        self._hysteresis = hysteresis
        # Async state
        self._thread = None
        self._target_grasp_state = False  # False=Open, True=Grasped

    def _log(self, msg: str):
        if self._logging:
            print(f"\t[gripper] {msg}")

    # --- backend contract ------------------------------------------------
    def get_state(self) -> "GripperState":
        raise NotImplementedError

    def _do_grasp(self):
        raise NotImplementedError

    def _do_open(self):
        raise NotImplementedError

    def home(self):
        """Calibrate / home the gripper (backend specific; no-op by default)."""
        pass

    def stop(self):
        pass

    def close(self):
        pass

    # --- shared async dispatch -------------------------------------------
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


class PandaGripper(BaseGripper):
    """Franka Panda Hand via panda-py (libfranka), routed through the Franka
    control box at ``ip_address``. Requires the Panda Hand to be physically
    attached and configured as the end-effector in Franka Desk.
    """

    def __init__(self, ip_address: str, hysteresis: float, logging: bool = True):
        super().__init__(hysteresis=hysteresis, logging=logging)
        self._log(f"Initializing PandaGripper via panda-py at {ip_address}")
        self._gripper = panda_py.libfranka.Gripper(ip_address)

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

    def home(self):
        self._gripper.homing()

    def _do_grasp(self):
        result = self.grasp(speed=0.1, force=10.0, blocking=False)
        self._log(f"grasp result: {result}")

    def _do_open(self):
        mx = self.get_state().max_width
        self._log(f"opening to max_width={mx}")
        self.goto(width=mx, speed=0.1, blocking=False)


class RobotiqGripper(BaseGripper):
    """Robotiq 2F-85 over Modbus RTU (USB->RS485) via the pyRobotiqGripper
    driver. Runs on the control PC alongside panda-py and is fully independent
    of the Franka — no Panda Hand / control-box gripper server is involved, and
    no NUC is required.

    Position is in bits, ``0 = open .. 255 = closed``. :meth:`get_state` reports
    a *synthetic* width so the policy observation computed in
    ``NUCInterface.get_robot_state`` (``1 - width/max_width``) comes out as
    ``pos/255`` — 0 when open, 1 when closed — matching the Panda Hand
    convention the policy was trained against.
    """

    OPEN_BIT = 0
    CLOSED_BIT = 255

    def __init__(self, port: str = "auto", device_id: int = 9, speed: int = 255,
                 force: int = 100, max_width_m: float = 0.085,
                 hysteresis: float = 0.1, logging: bool = True):
        super().__init__(hysteresis=hysteresis, logging=logging)
        # Imported lazily so the panda-only path never needs pyRobotiqGripper.
        from pyrobotiqgripper import RobotiqGripper as _Robotiq
        self._log(f"Initializing RobotiqGripper via pyRobotiqGripper on port={port}")
        self._speed = int(speed)
        self._force = int(force)
        self._max_width = float(max_width_m)
        # pymodbus serial access is not concurrency-safe; the state read (camera
        # / state thread) and the move dispatch (act_async thread) both touch the
        # bus, so every transaction is serialized behind this lock.
        self._lock = threading.Lock()
        self._gripper = _Robotiq(com_port=port, device_id=int(device_id))
        with self._lock:
            self._gripper.connect()
            self._gripper.activate()  # physically homes the gripper on first activation
            self._gripper.start()
            # Set the open/close bit references without a slow re-probe — the
            # 2F-85 spans the full 0..255 stroke after activation.
            self._gripper.calibrate_bit(openbit=self.OPEN_BIT, closebit=self.CLOSED_BIT)

    def get_state(self):
        with self._lock:
            pos = self._gripper.position(refreshStatus=True)
        if pos is None:
            pos = self.OPEN_BIT
        frac_closed = pos / 255.0
        # Synthetic width: get_robot_state computes 1 - width/max_width, which
        # must equal frac_closed -> width = max_width * (1 - frac_closed).
        width = self._max_width * (1.0 - frac_closed)
        return GripperState(width=width, max_width=self._max_width,
                            is_grasped=self._target_grasp_state)

    def _move(self, position: int):
        # wait=False so the call returns as soon as the request is written and
        # the lock is released — holding it across the ~1s travel would stall
        # the obs read. readStatus=False keeps the locked section minimal.
        with self._lock:
            self._gripper.move(position, speed=self._speed, force=self._force,
                               wait=False, readStatus=False)

    def _do_grasp(self):
        self._log(f"GRASP -> bit {self.CLOSED_BIT} (speed={self._speed}, force={self._force})")
        self._move(self.CLOSED_BIT)

    def _do_open(self):
        self._log(f"OPEN -> bit {self.OPEN_BIT}")
        self._move(self.OPEN_BIT)

    def home(self):
        with self._lock:
            self._gripper.activate()
            self._gripper.calibrate_bit(openbit=self.OPEN_BIT, closebit=self.CLOSED_BIT)

    def stop(self):
        with self._lock:
            self._gripper.stop()


def make_gripper(server_cfg: DictConfig, franka_ip: str, hysteresis: float,
                 logging: bool) -> BaseGripper:
    """Construct the gripper backend selected by ``server.gripper.type``.

    Defaults to the Panda Hand when no ``gripper`` block is present so existing
    configs keep working unchanged; set ``type: robotiq`` to drive a Robotiq
    2F-85 over USB/RS485 instead.
    """
    gcfg = server_cfg.get("gripper", {}) or {}
    gtype = str(gcfg.get("type", "panda")).lower()
    if gtype == "panda":
        return PandaGripper(franka_ip, hysteresis=hysteresis, logging=logging)
    if gtype == "robotiq":
        return RobotiqGripper(
            port=gcfg.get("port", "auto"),
            device_id=int(gcfg.get("device_id", 9)),
            speed=int(gcfg.get("speed", 255)),
            force=int(gcfg.get("force", 100)),
            max_width_m=float(gcfg.get("max_width_m", 0.085)),
            hysteresis=hysteresis,
            logging=logging,
        )
    raise ValueError(f"Unknown gripper.type {gtype!r}; expected 'panda' or 'robotiq'.")


class NUCInterface:
    @property
    def home(self):
        return self._home_pos.copy(), self._home_rot.copy()

    def __init__(self, ip: str, server: DictConfig, franka_ip: str,
                 home_pos=None, home_rot=None, home_q=None,
                 home_q_noise_enabled=False, home_q_noise_std=0.0):
        self._franka_ip = franka_ip
        self._nuc_ip = ip
        self._server_cfg = server
        # Optional joint-space home. When provided it is authoritative: the arm
        # resets to this exact joint configuration and the Cartesian home below
        # is overridden by its FK (resolved after connecting, see _home_q).
        self._home_q_cfg = None if home_q is None else np.asarray(home_q, dtype=np.float64).reshape(7)
        self._uses_joint_home = self._home_q_cfg is not None
        # Optional domain randomization: add fresh per-joint Gaussian noise to
        # the joint home on every reset (see _sample_initial_q). Only meaningful
        # with a joint-space home. The std is a scalar (broadcast to all 7
        # joints) or a length-7 list of per-joint stddevs, in radians.
        self._home_q_noise_enabled = bool(home_q_noise_enabled)
        self._home_q_noise_std = np.broadcast_to(
            np.asarray(home_q_noise_std, dtype=np.float64), (7,)).copy()
        if self._home_q_noise_enabled and not self._uses_joint_home:
            print("WARNING: home_q_noise_enabled is set but no home_q is "
                  "configured; initial-pose noise will not be applied.")
        self._home_pos = None if home_pos is None else np.array(home_pos)
        self._home_rot = None if home_rot is None else np.array(home_rot)

        print(f"Connecting to Panda at {self._franka_ip}")
        self._panda = panda_py.Panda(self._franka_ip)
        self._controller = None
        self._is_joint_space = False
        self._prev_q = None
        self._prev_t = None

        hysteresis = server.gripper_hysteresis
        gripper_logging = server.get("gripper_logging", True)
        self._gripper = make_gripper(server, self._franka_ip,
                                     hysteresis=hysteresis, logging=gripper_logging)

        self._desired_eef_pos = self._panda.get_position()
        self._desired_eef_rot = self._panda.get_orientation(scalar_first=False)
        # Resolve self._home_q, the joint configuration at home. It is the fixed
        # IK seed for joint-space mode (so every send_control IK call lives in
        # the same branch neighbourhood, eliminating shoulder/elbow jiggle from
        # IK re-solving against a moving seed) and, when a joint home is
        # configured, the reset target itself.
        if self._uses_joint_home:
            # Joint home is authoritative: derive the Cartesian home from FK so
            # every consumer of self.home (freeze masking, the Cartesian
            # impedance setpoint) matches the posture we actually drive to.
            self._home_q = self._home_q_cfg
            mat = np.array(panda_py.fk(self._home_q.reshape(7, 1))).reshape(4, 4)
            self._home_pos = mat[:3, 3]
            self._home_rot = Rotation.from_matrix(mat[:3, :3]).as_quat()  # xyzw
            print(f"Joint-space home: q={self._home_q.tolist()} -> "
                  f"pos={self._home_pos.tolist()} rot(xyzw)={self._home_rot.tolist()}")
        else:
            # Seed IK with panda_py.constants.JOINT_POSITION_START (the canonical
            # high-manipulability home) and back out the joint config at the
            # configured Cartesian home.
            jps = np.asarray(panda_py.constants.JOINT_POSITION_START, dtype=np.float64)
            self._home_q = np.asarray(
                panda_py.ik(self._home_pos, self._home_rot, q_init=jps, q_7=float(jps[6])),
                dtype=np.float64,
            ).reshape(7)

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
                # Seed IK from a fixed home-pose qpos. Analytical IK is
                # deterministic — same target + same seed -> same q_desired —
                # so all calls land in the same branch neighbourhood and
                # nearby cartesian targets map to nearby q_desired.
                q_desired = panda_py.ik(
                    eef_pos, eef_rot,
                    q_init=self._home_q,
                    q_7=float(self._home_q[6]),
                )
                self._controller.set_control(q_desired)
            else:
                self._controller.set_control(eef_pos, eef_rot)

        if gripper is not None:
            g_val = gripper.item() if hasattr(gripper, 'item') else float(gripper)
            self._gripper.act_async(g_val)

    def send_control_tensor(self, eef_pos: torch.Tensor, eef_rot: torch.Tensor, gripper: torch.Tensor):
        g = gripper.cpu().numpy() if gripper is not None else None
        self.send_control(eef_pos.cpu().numpy(), eef_rot.cpu().numpy(), g)

    def get_fk_ee_pos(self) -> np.ndarray:
        """FK-frame EE position of the current joint config — i.e. the same
        frame ``_desired_eef_pos`` lives in when commands come from FK of a
        commanded qpos. Use this (not ``get_position``) when comparing against
        a target derived from FK of commanded qpos: ``get_position``'s baked-in
        TCP offset would otherwise introduce a constant bias that masks the
        true joint tracking error.
        """
        q = self._panda.q.copy()
        mat = np.array(panda_py.fk(q.reshape(7, 1))).reshape(4, 4)
        return mat[:3, 3]

    def send_qpos_control(self, qpos: np.ndarray, gripper):
        """Command a joint configuration. Works for both controllers:

        - joint-space (HybridJointImpedance / JointPosition): qpos is sent
          straight to the controller; no IK involved.
        - Cartesian (PolymetisImpedance): FK once to derive the EE pose and
          dispatch ``set_control(pos, quat)`` — qpos itself is the natural
          nullspace target but the panda_py 0.7.x PolymetisImpedance binding
          does not expose ``q_nullspace`` as a kwarg, so the controller's own
          configured nullspace stiffness handles redundancy.
        """
        q = np.asarray(qpos, dtype=np.float64).reshape(7)
        try:
            mat = np.array(panda_py.fk(q.reshape(7, 1))).reshape(4, 4)
            new_pos = mat[:3, 3]
            new_rot = Rotation.from_matrix(mat[:3, :3]).as_quat()
            # Quaternion double-cover: q and -q represent the same rotation,
            # but scipy's canonical choice can flip between consecutive FK
            # calls on similar joint configs. The Cartesian impedance
            # controller treats a sign flip as a ~360° rotation request and
            # spikes the torque. Anchor to the previous setpoint's hemisphere.
            if np.dot(new_rot, self._desired_eef_rot) < 0:
                new_rot = -new_rot
            self._desired_eef_pos = new_pos
            self._desired_eef_rot = new_rot
        except (AttributeError, RuntimeError):
            new_pos = None

        if self._controller:
            if self._is_joint_space:
                self._controller.set_control(q)
            else:
                if new_pos is None:
                    raise RuntimeError(
                        "send_qpos_control: FK failed; cannot dispatch to "
                        "Cartesian controller without a pose."
                    )
                self._controller.set_control(self._desired_eef_pos, self._desired_eef_rot)

        if gripper is not None:
            g_val = gripper.item() if hasattr(gripper, 'item') else float(gripper)
            self._gripper.act_async(g_val)

    def send_qpos_control_tensor(self, qpos: torch.Tensor, gripper):
        g = gripper.cpu().numpy() if gripper is not None else None
        self.send_qpos_control(qpos.cpu().numpy(), g)

    def home_gripper(self):
        """Calibrate/home the gripper in a background thread (backend specific)."""
        threading.Thread(target=self._gripper.home, daemon=True).start()

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
        elif ctrl_type == "joint_impedance":
            # libfranka JointPosition. The per-joint gains live under the task's
            # impedance config (joint_stiffness / joint_damping); these are
            # distinct from the Cartesian translational/rotational gains, which
            # don't translate to pure joint-space impedance. Fall back to
            # panda_py's stock gains when the task doesn't override them.
            self._is_joint_space = True
            imp_cfg = self._server_cfg.impedance
            kq_list = imp_cfg.get("joint_stiffness", [600., 600., 600., 600., 250., 150., 50.])
            kqd_list = imp_cfg.get("joint_damping", [50., 50., 50., 20., 20., 20., 10.])
            kq = np.array([list(kq_list)], dtype=np.float64).T
            kqd = np.array([list(kqd_list)], dtype=np.float64).T
            return controllers.JointPosition(
                stiffness=kq,
                damping=kqd,
                filter_coeff=0.008,
            )
        else:
            # Cartesian impedance using the per-task impedance gains.
            self._is_joint_space = False
            return controllers.PolymetisImpedance(
                impedance=impedance,
                damping=damping,
                nullspace_stiffness=ns_stiffness,
                nullspace_damping=ns_damping,
            )

    def _sample_initial_q(self):
        """Joint configuration to reset to, optionally perturbed by noise.

        When home-q noise is enabled, draw fresh per-joint Gaussian noise and
        add it to the nominal joint home so each reset starts from a slightly
        different posture (domain randomization). self._home_q is left untouched
        — it stays the canonical IK seed and impedance reference.
        """
        if not self._home_q_noise_enabled:
            return self._home_q.copy()
        noise = np.random.normal(0.0, self._home_q_noise_std)
        noisy_q = self._home_q + noise
        print(f"Home-q noise (std={self._home_q_noise_std.tolist()}): "
              f"q={noisy_q.tolist()}")
        return noisy_q

    def reset(self, open_gripper: bool = True):
        home_pos, home_rot = self.home
        # Drive to home with libfranka's motion generator before handing off to the impedance controller.
        if self._controller:
            self._panda.stop_controller()
            self._controller = None
        if self._uses_joint_home:
            # Joint-space home: drive straight to the configured joint config,
            # then command it (send_qpos_control FKs to the matching pose for a
            # Cartesian controller, or passes q through for a joint-space one).
            initial_q = self._sample_initial_q()
            self._panda.move_to_joint_position(initial_q)
            self.start()
            self.send_qpos_control(initial_q, gripper=None)
        else:
            reset_pos = home_pos + np.array([0.0, 0.0, 0.04])
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
