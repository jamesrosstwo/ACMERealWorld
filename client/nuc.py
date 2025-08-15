import torch
from polymetis import RobotInterface, GripperInterface
import paramiko


class NUCInterface:
    def _launch_server(self):
        cmd = f"conda run -n nuc_polymetis_server python $CONDA_PREFIX/bin/launch_robot.py robot_client=franka_hardware robot_client.executable.robot_ip=\"{self._franka_ip}\""

    def __init__(self, ip: str, franka_ip: str):
        self._franka_ip = franka_ip
        self._nuc_ip = ip

        self._robot = RobotInterface(
            ip_address=self._nuc_ip,
        )
        self._gripper = GripperInterface(
            ip_address="localhost",
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

    def send_control(self, joint_angles, gripper_pos):
        self._robot.update_desired_joint_positions(torch.tensor(joint_angles))

    def reset(self):
        self._robot.go_home()
        self._gripper.goto(width=0, speed=0.05, force=0.1)

    def start(self):
        self._robot.start_joint_impedance(self._robot.Kq_default / 10, self._robot.Kqd_default / 10)
