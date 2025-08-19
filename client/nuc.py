import os
import threading
import time

import torch
from omegaconf import DictConfig
from polymetis import RobotInterface, GripperInterface
import paramiko


def stream_output(stream, prefix=''):
    for line in iter(stream.readline, ""):
        print(f"{prefix}{line.strip()}")


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

    def __init__(self, ip: str, server: DictConfig, franka_ip: str):
        self._franka_ip = franka_ip
        self._nuc_ip = ip
        self._server_cfg = server
        # self._nuc_client = self._launch_server(**self._server_cfg)
        # print("Waiting for server to start")
        # time.sleep(20.0)

        self._robot = RobotInterface(
            ip_address=self._nuc_ip,
        )
        self._gripper = GripperInterface(
            ip_address=self._nuc_ip,
            port=self._server_cfg.gripper_port
        )

    def get_robot_state(self):
        # gripper_state = self._gripper.get_state()
        qpos = self._robot.get_joint_positions()
        qvel = self._robot.get_joint_velocities()
        ee_pos, ee_rot = self._robot.get_ee_pose()
        # gripper_force = gripper_state.force  # idk
        state = dict(qpos=qpos, qvel=qvel, ee_pos=ee_pos, ee_rot=ee_rot, gripper_force=torch.zeros(1))
        return {k: v.detach().cpu().numpy() for k, v in state.items()}

    def forward_kinematics(self, joint_positions):
        return self._robot.robot_model.forward_kinematics(joint_positions)

    def send_control(self, eef_pos, eef_rot, gripper_pos):
        # self._robot.update_desired_joint_positions(torch.tensor(joint_angles))
        self._robot.update_desired_ee_pose(eef_pos, eef_rot)

    def reset(self):
        self._robot.go_home()
        # self._gripper.goto(width=0, speed=0.05, force=0.1)

    def start(self):
        self._robot.start_joint_impedance(self._robot.Kq_default / 10, self._robot.Kqd_default / 10)

    def close(self):
        # self._nuc_client.close()
        pass