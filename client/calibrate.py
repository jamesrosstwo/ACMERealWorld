from pathlib import Path
from typing import Dict

import docker
import hydra
import yaml
from omegaconf import DictConfig
from tqdm import tqdm

from client.realsense import RealSenseInterface
from client.write import ACMEWriter, KalibrWriter


class KalibrInterface:
    def __init__(self):
        self._client = docker.from_env()

    def run_calibration(self, folder: Path):
        container = self._client.containers.run(
            "kalibr",
            command="/bin/bash",
            environment={
                "DISPLAY": "",
                "QT_X11_NO_MITSHM": "1"
            },
            volumes={
                "/tmp/.X11-unix": {"bind": "/tmp/.X11-unix", "mode": "rw"},
                str(folder.resolve().absolute()): {"bind": "/data", "mode": "rw"}
            },
            stdin_open=True,
            tty=True,
            detach=True,
            remove=True
        )

        commands = [
            "source devel/setup.bash",
            "rosrun kalibr kalibr_bagcreater --folder /data --output-bag calib_0.bag",
            "rosrun kalibr kalibr_calibrate_cameras "
            "--bag /catkin_ws/calib_0.bag --target /data/target.yaml "
            "--models pinhole-radtan pinhole-radtan pinhole-radtan pinhole-radtan "
            "--topics /cam0/image_raw /cam1/image_raw /cam2/image_raw /cam3/image_raw --dont-show-report"
        ]

        for cmd in commands:
            exit_code, output = container.exec_run(f"bash -c '{cmd}'", tty=True)
            print(f"Ran: {cmd}")
            print(output.decode())


@hydra.main(config_path="calibrate", config_name="collect")
def main(cfg: DictConfig):
    realsense = RealSenseInterface(**cfg.realsense)
    writer = KalibrWriter(**cfg.writer)

    for i in tqdm(range(cfg.max_episode_timesteps)):
        colors, color_timesteps, depths, depth_timestamps = realsense.get_synchronized_frame()
        writer.write_frame(i, colors)
    realsense.shutdown()
    writer.flush()


    kalibr = KalibrInterface(writer.path)
    kalibr.run_calibration(cfg.folder)

if __name__ == "__main__":
    main()
