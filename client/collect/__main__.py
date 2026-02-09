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
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from client.nuc import NUCInterface
from client.collect.realsense import RealSenseInterface
from client.collect.gello import GELLOInterface
from client.utils import get_latest_ep_path
from client.collect.write import ACMEWriter


def start_control(gello: GELLOInterface, nuc: NUCInterface):
    state = nuc.get_robot_state()
    gello.zero_controls(state["qpos"])
    nuc.start()



def action_step(gello: GELLOInterface, nuc: NUCInterface):
    joint_angles = gello.get_joint_angles()
    eef_pos, eef_rot = nuc.forward_kinematics(torch.tensor(joint_angles))
    gripper_force = np.zeros(1)
    nuc.send_control(eef_pos, eef_rot, None)
    action = np.concatenate([eef_pos, eef_rot, gripper_force])
    return action


@hydra.main(config_path="../../config", config_name="collect")
def main(cfg: DictConfig):
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
                nuc.reset()
                start_control(gello, nuc)
                primary_serial = rs_interface.serials[0]
                def on_receive_frame(serial):
                    if serial == primary_serial:
                        state = nuc.get_robot_state()
                        # for episode collection we just assume all cameras are synchronized to the
                        # primary camera, and that this synchronous operation of getting robot state
                        # from the NUC takes no time.
                        current_action = action_step(gello, nuc)
                        episode_writer.write_state(action=current_action, **state)

                rs_interface.start_capture(on_receive_frame)
                rs_interface.reset_frame_counts()
                while any([c < cfg.max_episode_timesteps for c in rs_interface.get_frame_counts().values()]):
                    time.sleep(2.0)
                    print("Episode progress:", np.array(list(rs_interface.get_frame_counts().values())) / cfg.max_episode_timesteps)
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
