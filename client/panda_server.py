"""
Panda Server - Runs on the NUC connected to the Franka robot.
Provides a network interface for remote control, similar to polymetis.

Usage:
    python panda_server.py --robot-ip 192.168.1.107 --port 5556
"""

import argparse
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import panda_py
from panda_py import libfranka
from scipy.spatial.transform import Rotation
import zmq


@dataclass
class RobotState:
    q: np.ndarray
    dq: np.ndarray
    O_T_EE: np.ndarray
    gripper_width: float


class PandaServer:
    def __init__(self, robot_ip: str, port: int = 5556, gripper_port: int = 5557):
        self._robot = panda_py.Panda(robot_ip)
        self._gripper = libfranka.Gripper(robot_ip)
        
        self._context = zmq.Context()
        
        self._state_socket = self._context.socket(zmq.PUB)
        self._state_socket.bind(f"tcp://*:{port}")
        
        self._cmd_socket = self._context.socket(zmq.REP)
        self._cmd_socket.bind(f"tcp://*:{port + 1}")
        
        self._gripper_socket = self._context.socket(zmq.REP)
        self._gripper_socket.bind(f"tcp://*:{gripper_port}")
        
        self._desired_ee_pos: Optional[np.ndarray] = None
        self._desired_ee_rot: Optional[np.ndarray] = None
        self._control_active = False
        self._running = True
        
        self._state_lock = threading.Lock()
        self._cmd_lock = threading.Lock()
        
        print(f"Panda server started on ports {port}, {port + 1}, {gripper_port}")
    
    def get_state(self) -> RobotState:
        robot_state = self._robot.get_state()
        gripper_state = self._gripper.read_once()
        O_T_EE = np.array(robot_state.O_T_EE).reshape(4, 4, order='F')
        return RobotState(
            q=np.array(robot_state.q),
            dq=np.array(robot_state.dq),
            O_T_EE=O_T_EE,
            gripper_width=gripper_state.width
        )
    
    def _state_publisher_loop(self):
        """Publishes robot state at high frequency."""
        while self._running:
            try:
                state = self.get_state()
                state_dict = {
                    'q': state.q.tolist(),
                    'dq': state.dq.tolist(),
                    'O_T_EE': state.O_T_EE.flatten().tolist(),
                    'gripper_width': state.gripper_width
                }
                self._state_socket.send_json(state_dict, zmq.NOBLOCK)
            except zmq.ZMQError:
                pass
            time.sleep(0.001)  # 1kHz state publishing
    
    def _command_handler_loop(self):
        """Handles incoming commands."""
        while self._running:
            try:
                if self._cmd_socket.poll(timeout=10):
                    msg = self._cmd_socket.recv_json()
                    response = self._handle_command(msg)
                    self._cmd_socket.send_json(response)
            except Exception as e:
                print(f"Command handler error: {e}")
    
    def _gripper_handler_loop(self):
        """Handles gripper commands."""
        while self._running:
            try:
                if self._gripper_socket.poll(timeout=10):
                    msg = self._gripper_socket.recv_json()
                    response = self._handle_gripper_command(msg)
                    self._gripper_socket.send_json(response)
            except Exception as e:
                print(f"Gripper handler error: {e}")
    
    def _handle_command(self, msg: dict) -> dict:
        cmd = msg.get('cmd')
        
        if cmd == 'get_state':
            state = self.get_state()
            return {
                'success': True,
                'q': state.q.tolist(),
                'dq': state.dq.tolist(),
                'O_T_EE': state.O_T_EE.flatten().tolist(),
                'gripper_width': state.gripper_width
            }
        
        elif cmd == 'get_ee_pose':
            state = self.get_state()
            ee_pos = state.O_T_EE[:3, 3]
            ee_rot_matrix = state.O_T_EE[:3, :3]
            rot = Rotation.from_matrix(ee_rot_matrix)
            ee_quat = rot.as_quat()  # scipy returns (x, y, z, w) - use directly
            return {
                'success': True,
                'ee_pos': ee_pos.tolist(),
                'ee_rot': ee_quat.tolist()
            }
        
        elif cmd == 'move_to_ee_pose':
            pos = np.array(msg['pos'])
            quat = np.array(msg['quat'])  # assume (x, y, z, w) format like scipy
            rot = Rotation.from_quat(quat)
            rot_matrix = rot.as_matrix()
            transform = np.eye(4)
            transform[:3, :3] = rot_matrix
            transform[:3, 3] = pos
            self._robot.move_to_pose(transform)
            return {'success': True}
        
        elif cmd == 'update_desired_ee_pose':
            pos = np.array(msg['pos'])
            quat = np.array(msg['quat'])
            with self._cmd_lock:
                self._desired_ee_pos = pos
                self._desired_ee_rot = quat
            return {'success': True}
        
        elif cmd == 'start_cartesian_impedance':
            self._robot.start_controller(panda_py.controllers.CartesianImpedance())
            self._control_active = True
            return {'success': True}
        
        elif cmd == 'stop_controller':
            self._control_active = False
            return {'success': True}
        
        elif cmd == 'forward_kinematics':
            q = np.array(msg['q'])
            # Use panda model for FK
            model = self._robot.get_model()
            pose_flat = model.pose(panda_py.constants.Frame.kEndEffector, q)
            pose = np.array(pose_flat).reshape(4, 4, order='F')
            return {
                'success': True,
                'pose': pose.flatten().tolist()
            }
        
        else:
            return {'success': False, 'error': f'Unknown command: {cmd}'}
    
    def _handle_gripper_command(self, msg: dict) -> dict:
        cmd = msg.get('cmd')
        
        if cmd == 'get_state':
            state = self._gripper.read_once()
            return {
                'success': True,
                'width': state.width,
                'max_width': state.max_width,
                'is_grasped': state.is_grasped
            }
        
        elif cmd == 'move':
            width = msg['width']
            speed = msg.get('speed', 0.1)
            # Run move in background thread since it may be blocking
            def do_move():
                try:
                    self._gripper.move(width, speed)
                except Exception as e:
                    print(f"Gripper move error: {e}")
            threading.Thread(target=do_move, daemon=True).start()
            return {'success': True}
        
        elif cmd == 'grasp':
            width = msg.get('width', 0.0)
            speed = msg.get('speed', 0.1)
            force = msg.get('force', 10.0)
            # Run grasp in background thread since it's blocking
            def do_grasp():
                try:
                    self._gripper.grasp(width, speed, force)
                except Exception as e:
                    print(f"Grasp error: {e}")
            threading.Thread(target=do_grasp, daemon=True).start()
            return {'success': True, 'grasped': True}
        
        elif cmd == 'stop':
            self._gripper.stop()
            return {'success': True}
        
        else:
            return {'success': False, 'error': f'Unknown gripper command: {cmd}'}
    
    def _control_loop(self):
        """High-frequency control loop for Cartesian impedance control."""
        while self._running:
            if self._control_active:
                with self._cmd_lock:
                    if self._desired_ee_pos is not None and self._desired_ee_rot is not None:
                        pos = self._desired_ee_pos.copy()
                        quat = self._desired_ee_rot.copy()
                
                if pos is not None:
                    rot = Rotation.from_quat(quat)  # assume (x, y, z, w) format
                    rot_matrix = rot.as_matrix()
                    transform = np.eye(4)
                    transform[:3, :3] = rot_matrix
                    transform[:3, 3] = pos
                    try:
                        self._robot.set_cartesian_impedance_target(transform)
                    except Exception as e:
                        print(f"Control error: {e}")
            
            time.sleep(0.001)  # 1kHz control loop
    
    def run(self):
        """Start all server threads."""
        threads = [
            threading.Thread(target=self._state_publisher_loop, daemon=True),
            threading.Thread(target=self._command_handler_loop, daemon=True),
            threading.Thread(target=self._gripper_handler_loop, daemon=True),
            threading.Thread(target=self._control_loop, daemon=True),
        ]
        
        for t in threads:
            t.start()
        
        print("Server running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            self._running = False
            for t in threads:
                t.join(timeout=1.0)
            self._context.term()


def main():
    parser = argparse.ArgumentParser(description='Panda robot server')
    parser.add_argument('--robot-ip', type=str, default='192.168.1.107',
                        help='IP address of the Franka robot')
    parser.add_argument('--port', type=int, default=5556,
                        help='Base port for ZMQ sockets')
    parser.add_argument('--gripper-port', type=int, default=5558,
                        help='Port for gripper commands')
    args = parser.parse_args()
    
    server = PandaServer(args.robot_ip, args.port, args.gripper_port)
    server.run()


if __name__ == '__main__':
    main()

