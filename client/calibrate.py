import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from client.realsense import RealSenseInterface
from client.write import DataWriter

@hydra.main(config_path="calibrate", config_name="collect")
def main(cfg: DictConfig):
    realsense = RealSenseInterface(**cfg.realsense)
    writer = DataWriter(**cfg.writer)

    for i in tqdm(range(cfg.max_episode_timesteps)):
        colors, color_timesteps, depths, depth_timesteps = realsense.get_synchronized_frame()
        writer.write_frame(i, colors, depths)
    realsense.shutdown()
    writer.flush()
    exit(0)

if __name__ == "__main__":
    main()