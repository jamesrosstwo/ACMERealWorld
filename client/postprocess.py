import shutil
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from client.realsense import RSBagProcessor
from client.write import ACMEWriter


def gather_data(episodes_path: str, n_frames: int, writer_cfg: DictConfig, realsense: DictConfig) -> Path:
    base_episodes_path = Path(episodes_path)
    base_episodes_path.mkdir(exist_ok=True, parents=True)

    ep_paths = sorted(base_episodes_path.iterdir(), key=lambda p: int(p.stem.split("_")[-1]))
    for ep_path in tqdm(ep_paths, "Postprocessing episodes"):
        try:
            completion_marker = ep_path / "COMPLETE"
            if completion_marker.exists():
                print(f"Skipping {ep_path}: already postprocessed")
                continue
            print(f"Processing episode {ep_path}")
            try:
                shutil.rmtree(Path(ep_path / "captures"))
            except FileNotFoundError:
                pass
            bagpaths = sorted(Path(ep_path).glob("*.bag"))
            serials = [p.stem for p in bagpaths]
            writer = ACMEWriter(ep_path, serials=serials, **writer_cfg)
            rs_interface = RSBagProcessor(bagpaths, **realsense)
            print(f"found {len(bagpaths)} bags: {serials}")
            for color, color_tmstmp, depth, depth_tmstmp, serial in tqdm(rs_interface.process_all_frames()):
                try:
                    writer.write_capture_frame(serial, color_tmstmp, depth_tmstmp, color, depth)
                except IndexError:
                    continue
            writer.flush()
            completion_marker.touch()
        except RuntimeError as e:
            print(f"bags unindexed for episode {ep_path}")


@hydra.main(config_path="../config", config_name="collect")
def main(cfg: DictConfig):
    n_frames = cfg.max_episode_timesteps
    gather_data(cfg.episodes_path, n_frames, cfg.writer, cfg.realsense)


if __name__ == "__main__":
    main()
