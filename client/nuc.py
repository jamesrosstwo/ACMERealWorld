import os
import threading
import time

import numpy as np
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

    def forward_kinematics(self, joint_positions: torch.Tensor):
        return tuple([x.cpu().numpy() for x in self._robot.robot_model.forward_kinematics(joint_positions)])

    def send_control(self, eef_pos, eef_rot, gripper_pos):
        # self._robot.update_desired_joint_positions(torch.tensor(joint_angles))
        self._robot.update_desired_ee_pose(torch.tensor(eef_pos), torch.tensor(eef_rot))

    def reset(self):
        self._robot.go_home()

        # PUSH T FREEZES
        self._robot.move_to_ee_pose(torch.tensor([0.5, 0, 0.3]))
        # self._gripper.goto(width=0, speed=0.05, force=0.2)

    def start(self):
        self._robot.start_cartesian_impedance()

    def close(self):
        # self._nuc_client.close()
        pass



from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

_REALSENSE_INTRINSICS = np.asarray([
    [900, 0, 350],
    [0, 900, 350],
    [0, 0, 1]
])


def effective_reprojection_error(
        keypoint_position,  # World position of point when captured by cam 0
        keypoint_velocity,  # Velocity of keypoint
        delta_time,  # Time offset in ms between cam 0 capture and cam 1 capture
        T_cam1_cam0,
):
    pos_world_t0 = keypoint_position
    pos_world_t1 = keypoint_position + keypoint_velocity * delta_time

    pos_world_t0_homogenous = np.vstack([pos_world_t0[None].T, [[1]]])
    pos_world_t1_homogenous = np.vstack([pos_world_t1[None].T, [[1]]])

    base_extr = np.eye(4)  # we only really care about the relative offset between cam0 and 1

    P_cam0 = _REALSENSE_INTRINSICS @ base_extr[:3, :]
    P_cam1 = _REALSENSE_INTRINSICS @ (T_cam1_cam0 @ base_extr)[:3, :]

    pos_cam0_t0 = P_cam0 @ pos_world_t0_homogenous
    pos_cam0_t1 = P_cam0 @ pos_world_t1_homogenous

    pos_cam1_t0 = P_cam1 @ pos_world_t0_homogenous
    pos_cam1_t1 = P_cam1 @ pos_world_t1_homogenous

    uv_cam1_t0 = pos_cam1_t0[:2] / pos_cam1_t0[2]
    uv_cam1_t1 = pos_cam1_t1[:2] / pos_cam1_t1[2]

    return np.abs(uv_cam1_t0 - uv_cam1_t1).squeeze(1)


def generate_reprojection_figure(
        keypoint_speed: float,
        time_offset: float,
        n_samples: int,
        min_keypoint_depth: float,
        max_keypoint_depth: float
):
    reproj_errors = []
    for i in range(100000):
        T_cam = np.eye(4)
        point_position = np.random.uniform(low=0.5, high=1.5, size=(3,))
        # point_position = np.zeros(3)
        point_position[:2] = 0
        keypoint_vel = np.random.uniform(low=-1, high=1, size=(3,))  # m/s
        keypoint_vel = keypoint_vel / np.linalg.norm(keypoint_vel) * keypoint_speed

        time_offset = 0.007  # 7ms
        reproj_error = effective_reprojection_error(
            keypoint_position=point_position,
            keypoint_velocity=keypoint_vel,
            delta_time=time_offset,
            T_cam1_cam0=T_cam  # For the time being, imagine the reproj is for two cameras in the same position.
        )
        reproj_errors.append(reproj_error)

    errors = np.asarray(reproj_errors)
    df = pd.DataFrame(errors, columns=["u_error", "v_error"])
    f_error_hist = px.histogram(df.melt(), x="value", color="variable", barmode="overlay")
    return errors, f_error_hist


if __name__ == "__main__":
    base_dir = Path("synchronization_study")
    f_error_hist.write_html("reprojection_errors.html")

    print(f"Reprojection errors mean:{errors.mean():.4f}, var: {errors.var():.4f}")