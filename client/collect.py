from polymetis import RobotInterface
from polymetis import GripperInterface

robot = RobotInterface(
    ip_address="192.168.1.106",
)


gripper = GripperInterface(
    ip_address="localhost",
)

gripper_state = gripper.get_state()

print(gripper_state)