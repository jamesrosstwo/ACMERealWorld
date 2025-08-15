import os
from pathlib import Path

import docker
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from client.realsense import RealSenseInterface
from client.write import ACMEWriter, KalibrWriter


class KalibrInterface:
    def __init__(self):
        self._client = docker.from_env()

    def run_calibration(self, folder: Path):
        os.system("xhost +local:docker")

        folder_path = str(folder.resolve().absolute())

        cmd = (
            "source /catkin_ws/devel/setup.bash && "
            "rosrun kalibr kalibr_bagcreater --folder /data --output-bag /catkin_ws/calib_0.bag && "
            "rosrun kalibr kalibr_calibrate_cameras "
            "--bag /catkin_ws/calib_0.bag --target /data/target.yaml "
            "--models pinhole-radtan pinhole-radtan pinhole-radtan "
            "--topics /cam0/image_raw /cam1/image_raw /cam2/image_raw --dont-show-report"
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


@hydra.main(config_path="config", config_name="calibrate")
def main(cfg: DictConfig):
    realsense = RealSenseInterface(**cfg.realsense)
    writer = KalibrWriter(**cfg.writer)

    for i in tqdm(range(cfg.max_episode_timesteps)):
        colors, color_timesteps, depths, depth_timestamps = realsense.get_synchronized_frame()
        writer.write_captures_frame(color_timesteps, colors)
    realsense.shutdown()
    writer.flush()

    kalibr = KalibrInterface()
    kalibr.run_calibration(writer.path)


if __name__ == "__main__":
    main()
