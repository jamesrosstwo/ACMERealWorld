import shutil
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from client.realsense import RealSenseInterface
from client.write import KalibrWriter


def gather_data(n_frames: int, writer: DictConfig, realsense: DictConfig) -> Path:
    writer = KalibrWriter(**writer)
    with RealSenseInterface(**realsense, init=False) as rs_interface:
        bagpaths = list(Path("/home/rvl_root/calib_3k").glob("*.bag"))
        rs_interface._recording_bagpaths = bagpaths
        print(f"found {len(bagpaths)} bags")
        for color, color_tmstmp, depth, depth_tmstmp, cap_idx in tqdm(rs_interface.process_all_frames_parallel()):
            try:
                writer.write_capture_frame(cap_idx, color_tmstmp, color)
            except IndexError:
                break
        writer.flush()
    return writer.path


@hydra.main(config_path="config", config_name="calibrate")
def main(cfg: DictConfig):
    calibration_episode_path = gather_data(cfg.n_frames, cfg.writer, cfg.realsense)
    # KalibrInterface().run_calibration(calibration_episode_path)

if __name__ == "__main__":
    main()
