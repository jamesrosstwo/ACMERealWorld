import os
import time
from collections import defaultdict
from pathlib import Path

import docker
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from client.realsense import RealSenseInterface
from client.write import KalibrWriter


class KalibrInterface:
    def __init__(self):
        self._client = docker.from_env()

    def run_calibration(self, folder: Path):
        os.system("xhost +local:docker")

        folder_path = str(folder.resolve().absolute())

        cmd = (
            "source /catkin_ws/devel/setup.bash && "
            "rosrun kalibr kalibr_bagcreater --folder /data --output-bag /catkin_ws/calib_0.bag && "
            "rosrun kalibr kalibr_calibrate_cameras --bag /catkin_ws/calib_0.bag --target /data/target.yaml --models pinhole-radtan pinhole-radtan pinhole-radtan pinhole-radtan --topics /cam_0/image_raw /cam_1/image_raw /cam_2/image_raw --dont-show-report"
        )

        container = self._client.containers.run(
            "kalibr",
            command=f"bash -c '{cmd}'",
            environment={
                "DISPLAY": os.environ.get("DISPLAY", ":0"),
                "QT_X11_NO_MITSHM": "1"
            },
            volumes={
                "/tmp/.X11-unix": {"bind": "/tmp/.X11-unix", "mode": "rw"},
                folder_path: {"bind": "/data", "mode": "rw"}
            },
            stdin_open=True,
            tty=True,
            remove=True,
            detach=False  # Run in foreground, output streams to console
        )


def gather_data(n_frames: int, writer: DictConfig, realsense: DictConfig) -> Path:
    n_frames: int = n_frames
    writer = KalibrWriter(**writer)

    counts = defaultdict(int)

    def on_receive_frame(cap_idx, color, color_tmstmp, depth, depth_tmstmp):
        writer.write_capture_frame(cap_idx, color_tmstmp, color)
        counts[cap_idx] += 1


    rs_interface = RealSenseInterface(**realsense, frame_callback=on_receive_frame)
    for i in range(rs_interface.n_cameras):
        counts[i] = 0
    while any([c < n_frames for c in counts.values()]):
        time.sleep(5.0)
        print("Progress:", np.array(list(counts.values())) / n_frames)
    writer.flush()
    return writer.path


@hydra.main(config_path="config", config_name="calibrate")
def main(cfg: DictConfig):
    calibration_episode_path = gather_data(cfg.n_frames, cfg.writer, cfg.realsense)
    KalibrInterface().run_calibration(calibration_episode_path)

if __name__ == "__main__":
    main()
