import threading
import time

import numpy as np
import torch
from omegaconf import DictConfig
import zmq


class RobotInterface:
    """Client interface to the Panda server, mimics polymetis RobotInterface."""
    
    def __init__(self, ip_address: str, port: int = 5556):
        self._context = zmq.Context()
        
        self._state_socket = self._context.socket(zmq.SUB)
        self._state_socket.connect(f"tcp://{ip_address}:{port}")
        self._state_socket.setsockopt_string(zmq.SUBSCRIBE, '')
        
        self._cmd_socket = self._context.socket(zmq.REQ)
        self._cmd_socket.connect(f"tcp://{ip_address}:{port + 1}")
        
        self._latest_state = None
        self._state_lock = threading.Lock()
        self._running = True
        
        self._state_thread = threading.Thread(target=self._state_receiver_loop, daemon=True)
        self._state_thread.start()
        
        time.sleep(0.1)
    
    def _state_receiver_loop(self):
        while self._running:
            try:
                if self._state_socket.poll(timeout=10):
                    state = self._state_socket.recv_json()
                    with self._state_lock:
                        self._latest_state = state
            except Exception:
                pass
    
    def _send_command(self, cmd: dict) -> dict:
        self._cmd_socket.send_json(cmd)
        return self._cmd_socket.recv_json()
    
    def get_joint_positions(self) -> torch.Tensor:
        with self._state_lock:
            if self._latest_state is not None:
                return torch.tensor(self._latest_state['q'])
        response = self._send_command({'cmd': 'get_state'})
        return torch.tensor(response['q'])
    
    def get_joint_velocities(self) -> torch.Tensor:
        with self._state_lock:
            if self._latest_state is not None:
                return torch.tensor(self._latest_state['dq'])
        response = self._send_command({'cmd': 'get_state'})
        return torch.tensor(response['dq'])
    
    def get_ee_pose(self):
        response = self._send_command({'cmd': 'get_ee_pose'})
        return torch.tensor(response['ee_pos']), torch.tensor(response['ee_rot'])
    
    def move_to_ee_pose(self, pos, quat=None):
        if quat is None:
            quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        pos_np = pos.cpu().numpy() if isinstance(pos, torch.Tensor) else np.array(pos)
        quat_np = quat.cpu().numpy() if isinstance(quat, torch.Tensor) else np.array(quat)
        self._send_command({
            'cmd': 'move_to_ee_pose',
            'pos': pos_np.tolist(),
            'quat': quat_np.tolist()
        })
    
    def update_desired_ee_pose(self, pos: torch.Tensor, quat: torch.Tensor):
        pos_np = pos.cpu().numpy() if isinstance(pos, torch.Tensor) else np.array(pos)
        quat_np = quat.cpu().numpy() if isinstance(quat, torch.Tensor) else np.array(quat)
        self._send_command({
            'cmd': 'update_desired_ee_pose',
            'pos': pos_np.tolist(),
            'quat': quat_np.tolist()
        })
    
    def start_cartesian_impedance(self):
        self._send_command({'cmd': 'start_cartesian_impedance'})
    
    def stop_controller(self):
        self._send_command({'cmd': 'stop_controller'})
    
    @property
    def robot_model(self):
        return RobotModel(self)
    
    def close(self):
        self._running = False
        self._state_thread.join(timeout=1.0)
        self._context.term()


class RobotModel:
    """Provides forward kinematics, mimics polymetis robot_model."""
    
    def __init__(self, robot: RobotInterface):
        self._robot = robot
    
    def forward_kinematics(self, joint_positions: torch.Tensor):
        q = joint_positions.cpu().numpy() if isinstance(joint_positions, torch.Tensor) else np.array(joint_positions)
        response = self._robot._send_command({
            'cmd': 'forward_kinematics',
            'q': q.tolist()
        })
        pose = np.array(response['pose']).reshape(4, 4)
        ee_pos = torch.tensor(pose[:3, 3])
        ee_rot = torch.tensor(pose[:3, :3].flatten())
        return ee_pos, ee_rot


class GripperInterface:
    """Client interface to the Panda gripper, mimics polymetis GripperInterface."""
    
    def __init__(self, ip_address: str, port: int = 5558):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{ip_address}:{port}")
    
    def _send_command(self, cmd: dict) -> dict:
        self._socket.send_json(cmd)
        return self._socket.recv_json()
    
    def get_state(self):
        response = self._send_command({'cmd': 'get_state'})
        return GripperState(
            width=response['width'],
            max_width=response['max_width'],
            is_grasped=response['is_grasped']
        )
    
    def goto(self, width: float, speed: float = 0.1, force: float = 10.0, blocking: bool = True):
        self._send_command({
            'cmd': 'move',
            'width': width,
            'speed': speed
        })
    
    def grasp(self, grasp_width: float = 0.0, speed: float = 0.1, force: float = 10.0, blocking: bool = True):
        response = self._send_command({
            'cmd': 'grasp',
            'width': grasp_width,
            'speed': speed,
            'force': force
        })
        return response.get('grasped', False)
    
    def stop(self):
        self._send_command({'cmd': 'stop'})
    
    def close(self):
        self._context.term()


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
        self._last_gripper_state = False

        self._robot = RobotInterface(
            ip_address=self._nuc_ip,
        )

        self._gripper = GripperInterface(
            ip_address=self._nuc_ip,
            port=self._server_cfg.gripper_port
        )

        self._desired_eef_pos, self._desired_eef_rot = self.pusht_home

    def get_desired_ee_pose(self):
        return np.concatenate([self._desired_eef_pos, self._desired_eef_rot]).copy()

    def get_robot_state(self):
        qpos = self._robot.get_joint_positions()
        qvel = self._robot.get_joint_velocities()
        ee_pos, ee_rot = self._robot.get_ee_pose()
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

    def reset(self):
        current_ee_pos, current_ee_rot = self._robot.get_ee_pose()
        self._robot.move_to_ee_pose(current_ee_pos + torch.tensor([0, 0, 0.15]))
        self._robot.move_to_ee_pose(*self.pusht_home)

        self._gripper.grasp(speed=0.01, force=1, blocking=True)
        print("Grasping")

    def start(self):
        self._robot.start_cartesian_impedance()

    def close(self):
        self._robot.close()
        self._gripper.close()
