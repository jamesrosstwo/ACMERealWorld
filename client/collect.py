from collections import defaultdict
import time

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from client.nuc import NUCInterface
from client.realsense import RealSenseInterface
from client.teleop import GELLOInterface
from client.write import ACMEWriter
import threading


def start_control_loop(gello: GELLOInterface, nuc: NUCInterface):
    state = nuc.get_robot_state()
    gello.zero_controls(state["qpos"])
    nuc.start()

    stop_event = threading.Event()
    joint_lock = threading.Lock()
    latest_joint_angles = None

    def _loop_iter():
        nonlocal latest_joint_angles
        joint_angles = gello.get_joint_angles()
        with joint_lock:
            latest_joint_angles = joint_angles.copy()
        nuc.send_control(joint_angles[:7], joint_angles[7:])

    def _loop_runner():
        while not stop_event.is_set():
            _loop_iter()

    loop_thread = threading.Thread(target=_loop_runner, daemon=True)
    loop_thread.start()

    def stop_loop():
        stop_event.set()
        loop_thread.join()

    def get_latest_joint_angles():
        with joint_lock:
            return None if latest_joint_angles is None else latest_joint_angles.copy()

    return stop_loop, get_latest_joint_angles


@hydra.main(config_path="config", config_name="collect")
def main(cfg: DictConfig):
    nuc = NUCInterface(**cfg.nuc)
    gello = GELLOInterface(**cfg.gello)

    rs_interface = RealSenseInterface(**cfg.realsense)

    while True:
        episode_writer = ACMEWriter(**cfg.writer)
        nuc.reset()
        stop_control, get_action = start_control_loop(gello, nuc)

        def on_receive_frame(capture_idx):
            if capture_idx == 0:
                state = nuc.get_robot_state()
                # for episode collection we just assume all cameras are synchronized to cam0,
                # and that this synchronous operation of getting robot state from the NUC takes
                # no time.
                episode_writer.write_state(timestamp=color_tmstmp, action=get_action(), **state)

        rs_interface.start_capture(on_receive_frame)
        for i in range(rs_interface.n_cameras):
            rs_interface.frame_counts[i] = 0
        while any([c < cfg.max_episode_timesteps for c in rs_interface.frame_counts.values()]):
            time.sleep(2.0)
            print("Episode progress:", np.array(list(rs_interface.frame_counts.values())) / cfg.max_episode_timesteps)

        stop_control()
        for cap_idx in range(rs_interface.n_cameras):
            m = f"processing capture {cap_idx}"
            for color, color_tmstmp, depth, depth_tmstmp in tqdm(rs_interface.process_frames(cap_idx), m):
                episode_writer.write_capture_frame(cap_idx, color_tmstmp, color, depth)
        episode_writer.flush()


if __name__ == "__main__":
    main()
