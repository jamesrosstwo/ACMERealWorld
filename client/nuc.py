from polymetis import RobotInterface, GripperInterface
from polymetis_pb2 import GripperState


class NUCInterface:
    def __init__(self, ip: str):
        self._nuc_ip = ip
        self._robot = RobotInterface(
            ip_address=self._nuc_ip,
        )
        self._gripper = GripperInterface(
            ip_address=self._nuc_ip,
        )

    def get_robot_state(self):
        gripper_state: GripperState = self._gripper.get_state()
        qpos = self._robot.get_joint_positions()
        qvel = self._robot.get_joint_velocities()
        ee_pos, ee_rot = self._robot.get_ee_pose()
        gripper_force = gripper_state.force  # idk
        return dict(qpos=qpos, qvel=qvel, ee_pos=ee_pos, ee_rot=ee_rot, gripper_force=gripper_force)

    def send_control(self, desired_eef_pos, desired_gripper_force):
        self._robot.move_to_ee_pose(desired_eef_pos)
        self._gripper.grasp(speed=0.05, force=desired_gripper_force)