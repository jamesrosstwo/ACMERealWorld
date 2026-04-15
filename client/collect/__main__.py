"""Episode data collection entry point.

Orchestrates teleoperated demonstration recording by coordinating the GELLO
controller, RealSense cameras, and Franka Panda robot. Each episode captures
synchronized multi-camera RGB-D frames alongside robot state and operator actions,
written to disk via :class:`~client.collect.write.ACMEWriter`.

Usage::

    python -m client.collect
"""
import shutil
import traceback
import time
import threading
import queue
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation

from client.nuc import NUCInterface
from client.collect.realsense import RealSenseInterface
from client.collect.gello import GELLOInterface
from client.utils import get_latest_ep_path
from client.collect.write import ACMEWriter



def action_step(gello: GELLOInterface, nuc: NUCInterface, task_cfg: DictConfig):
    joint_angles = gello.get_joint_angles()
    eef_pos, eef_rot = nuc.forward_kinematics(torch.tensor(joint_angles))

    home_pos, home_rot = nuc.home
    pos_mask = np.array(task_cfg.pos_mask)
    eef_pos = np.where(pos_mask, eef_pos, home_pos)
    if task_cfg.freeze_rotation:
        eef_rot = home_rot

    gripper_force = np.array([gello.get_gripper()])
    gripper_cmd = None if task_cfg.freeze_gripper else gripper_force

    nuc.send_control(eef_pos, eef_rot, gripper_cmd)
    action = np.concatenate([eef_pos, eef_rot, gripper_force])
    return action


@hydra.main(config_path="../../config", config_name="collect")
def main(cfg: DictConfig):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    nuc = NUCInterface(**cfg.nuc)
    gello = GELLOInterface(**cfg.gello)
    base_ep_path = Path(cfg.episodes_path)
    base_ep_path.mkdir(exist_ok=True, parents=True)
    while True:
        try:
            ep_path = get_latest_ep_path(base_ep_path, prefix="episode")
            print(f"Recording to {ep_path}")
            with RealSenseInterface(ep_path, **cfg.realsense) as rs_interface:
                episode_writer = ACMEWriter(ep_path, serials=rs_interface.serials, **cfg.writer)

                nuc.reset(open_gripper=cfg.task.open_gripper_on_reset)
                state = nuc.get_robot_state()
                gello.zero_controls(state["qpos"])
                
                primary_serial = rs_interface.serials[0]

                # Queue decouples frame grabbing from heavy robot state / control
                # work so the primary camera thread never blocks on network or I/O.
                state_queue = queue.Queue()
                state_worker_stop = threading.Event()

                def state_worker():
                    steps = 0
                    while not state_worker_stop.is_set():
                        try:
                            timestamp = state_queue.get(timeout=1.0)
                        except queue.Empty:
                            continue
                        if steps >= cfg.max_episode_timesteps:
                            continue
                        try:
                            state = nuc.get_robot_state()

                            # Log tracking error (actual vs previously commanded pose)
                            desired_pose = nuc.get_desired_ee_pose()
                            desired_pos, desired_rot = desired_pose[:3], desired_pose[3:]
                            pos_err = np.linalg.norm(state["ee_pos"] - desired_pos)
                            rot_err = (Rotation.from_quat(state["ee_rot"])
                                       * Rotation.from_quat(desired_rot).inv()).magnitude()
                            # print(f"\t[tracking] pos_err={pos_err*1000:.1f}mm  rot_err={np.degrees(rot_err):.1f}deg")
                            # print(f"\t[ee_pos] x={state['ee_pos'][0]:.4f}  y={state['ee_pos'][1]:.4f}  z={state['ee_pos'][2]:.4f}")

                            current_action = action_step(gello, nuc, cfg.task)
                            episode_writer.write_state(timestamp=timestamp, action=current_action, **state)
                            steps += 1
                        except Exception as e:
                            print(f"State worker error: {e}")
                            traceback.print_exc()

                state_thread = threading.Thread(target=state_worker, daemon=True)
                state_thread.start()

                def on_receive_frame(serial, timestamp):
                    if serial == primary_serial:
                        state_queue.put(timestamp)

                rs_interface.reset_frame_counts()
                rs_interface.start_capture(on_receive_frame)#, on_warmup=nuc.home_gripper)
                while any([c < cfg.max_episode_timesteps for c in rs_interface.get_frame_counts().values()]):
                    time.sleep(2.0)
                    qsize = state_queue.qsize()
                    print("Episode progress:", np.array(list(rs_interface.get_frame_counts().values())) / cfg.max_episode_timesteps,
                          f"  state_queue: {qsize}")

                # Wait for state worker to drain remaining items
                state_worker_stop.set()
                state_thread.join()
            episode_writer.flush()
        except Exception as e:
            print(e)
            traceback.print_exc()
        finally:
            ep_control_msg = "1: Continue and start the next recording\n0: to delete this recording.\nx: Exit"
            ep_control_cmd = str(input(ep_control_msg))
            while ep_control_cmd not in ["1", "x"]:
                if ep_control_cmd == "0":
                    cmd_confirm = input(f"Input 0 again to confirm deletion of {episode_writer.episode_path}")
                    if cmd_confirm == "0":
                        shutil.rmtree(episode_writer.episode_path)
                ep_control_cmd = str(input(ep_control_msg))
            if ep_control_cmd == "x":
                break



if __name__ == "__main__":
    main()
