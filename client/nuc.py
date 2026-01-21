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


import panda_py

class GripperInterface:
    def __init__(self, ip_address: str, port: int = None):
        print(f"DEBUG: Initializing GripperInterface via panda-py at {ip_address}")
        # Initialize Panda interface, just for gripper control if possible
        self._panda = panda_py.Panda(ip_address)
    
    def get_state(self):
        # panda-py state is usually a dict or object
        state = self._panda.get_state()
        return GripperState(
            width=state.gripper_width,
            max_width=state.gripper_max_width, # Assuming attribute names, need to verify or be defensive
            is_grasped=state.gripper_width < 0.08 # simplified check
            # Real panda-py might not expose 'is_grasped' directly in state, often inferred from width and last command success
        )
    
    def goto(self, width: float, speed: float = 0.1, force: float = 10.0, blocking: bool = True):
        self._panda.get_gripper().move(width=width, speed=speed)
    
    def grasp(self, grasp_width: float = 0.0, speed: float = 0.1, force: float = 10.0, blocking: bool = True):
        # returns boolean success
        try:
            return self._panda.get_gripper().grasp(width=grasp_width, speed=speed, force=force, epsilon=0.005)
        except RuntimeError:
            return False
    
    def stop(self):
        self._panda.get_gripper().stop()
    
    def close(self):
        pass # panda-py handles cleanup likely


class GripperState:
    def __init__(self, width: float, max_width: float, is_grasped: bool):
        self.width = width
        self.max_width = max_width
        self.is_grasped = is_grasped


class NUCInterface:
    def _launch_server(self, conda, ip, port, user, pwd):
        c = f"{conda}/bin/conda"
        launch = f"{conda}/envs/nuc_polymetis_server/bin/launch_robot.py"
        launch_dir = os.path.dirname(launch)

        cmd = (
            f"cd {launch_dir} && "
            f"sudo -S {c} run -n nuc_polymetis_server python launch_robot.py "
            f"robot_client=franka_hardware robot_client.executable_cfg.robot_ip=\"{self._franka_ip}\""
        )

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=ip,
            port=port,
            username=user,
            password=pwd,
        )

        stdin, stdout, stderr = client.exec_command(cmd)
        stdin.write(pwd + "\n")

        threading.Thread(target=stream_output, args=(stdout,), daemon=True).start()
        threading.Thread(target=stream_output, args=(stderr, '[ERR] '), daemon=True).start()

        return client

    @property
    def pusht_home(self):
        return np.array([0.425, -0.375, 0.38]), np.array([0.942, 0.336, 0, 0])

    def __init__(self, ip: str, server: DictConfig, franka_ip: str):
        self._franka_ip = franka_ip
        self._nuc_ip = ip
        self._server_cfg = server
        self._last_gripper_state = False
        # self._nuc_client = self._launch_server(**self._server_cfg)
        # print("Waiting for ser ver to start")
        # time.sleep(20.0)

        self._robot = RobotInterface(
            ip_address=self._nuc_ip,
        )

        self._gripper = GripperInterface(
            ip_address=self._franka_ip, # Use direct robot IP, not NUC IP
            port=None
        )
        # NUCInterface alive check simplified or removed as panda-py connection throws on init if failed

        self._desired_eef_pos, self._desired_eef_rot = self.pusht_home

    def get_desired_ee_pose(self):
        return np.concatenate([self._desired_eef_pos, self._desired_eef_rot]).copy()

    def get_robot_state(self):
        # gripper_state = self._gripper.get_state()
        qpos = self._robot.get_joint_positions()
        qvel = self._robot.get_joint_velocities()
        ee_pos, ee_rot = self._robot.get_ee_pose()
        # gripper_force = gripper_state.force  # idk
        state = dict(qpos=qpos, qvel=qvel, ee_pos=ee_pos, ee_rot=ee_rot, gripper_force=torch.zeros(1))
        return {k: v.detach().cpu().numpy() for k, v in state.items()}

    def forward_kinematics(self, joint_positions: torch.Tensor):
        return tuple([x.cpu().numpy() for x in self._robot.robot_model.forward_kinematics(joint_positions)])

    def send_control(self, eef_pos: np.ndarray, eef_rot: np.ndarray, gripper: np.ndarray):
        self._desired_eef_pos = eef_pos.copy()
        self._desired_eef_rot = eef_rot.copy()
        self.send_control_tensor(torch.tensor(eef_pos), torch.tensor(eef_rot), None)

    def send_control_tensor(self, eef_pos: torch.Tensor, eef_rot: torch.Tensor, gripper: torch.Tensor):
        self._robot.update_desired_ee_pose(eef_pos, eef_rot)
        # if gripper > 0:
        #     if not self._last_gripper_state:
        #         self._gripper.grasp(speed=0.1, force=1., blocking=False)
        #         self._last_gripper_state = True
        # else:
        #     if self._last_gripper_state:
        #         self._gripper.grasp(grasp_width=0.25, speed=0.1, force=0.5, blocking=False)
        #         self._last_gripper_state = False

    def reset(self):
        current_ee_pos, current_ee_rot = self._robot.get_ee_pose()
        self._robot.move_to_ee_pose(current_ee_pos + torch.tensor([0, 0, 0.15]))
        # PUSH T FREEZES
        self._robot.move_to_ee_pose(*self.pusht_home)
        # self._gripper.goto(width=0.25, speed=0.1, force=0.5, blocking=True)
        # time.sleep(6)

        self._gripper.grasp(speed=0.01, force=1, blocking=True)
        print("Grasping")

    def start(self):
        self._robot.start_cartesian_impedance()

    def close(self):
        # self._nuc_client.close()
        pass
