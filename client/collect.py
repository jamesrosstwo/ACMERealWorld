import time

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from client.nuc import NUCInterface
from client.realsense import RealSenseInterface
from client.teleop import GELLOInterface
from client.write import DataWriter


@hydra.main(config_path="config", config_name="collect")
def main(cfg: DictConfig):
    nuc = NUCInterface(**cfg.nuc)
    realsense = RealSenseInterface(**cfg.realsense)
    gello = GELLOInterface(**cfg.gello)
    writer = DataWriter(**cfg.writer)

    nuc.reset()

    for i in tqdm(range(cfg.max_episode_timesteps)):
        colors, color_timesteps, depths, depth_timesteps = realsense.get_synchronized_frame()
        writer.write_frame(i, colors, depths)
        state = nuc.get_robot_state()
        if i == 25:
            nuc.start()
            gello.zero_controls(state["qpos"])
            print("Control active")

        joint_angles = gello.get_joint_angles()
        writer.write_state(timestep=i, action=joint_angles, **state)

        if i >= 25:
            pass
            nuc.send_control(joint_angles[:7], joint_angles[7:])
    writer.flush()
    exit(0)

if __name__ == "__main__":
    main()