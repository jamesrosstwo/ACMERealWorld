import threading
import time

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from client.nuc import NUCInterface
from client.realsense import RealSenseInterface
from client.teleop import GELLOInterface
from client.write import DataWriter


import threading

import threading
import time

def start_control_loop(gello: GELLOInterface, nuc: NUCInterface):
    nuc.reset()
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
    realsense = RealSenseInterface(**cfg.realsense)
    gello = GELLOInterface(**cfg.gello)
    writer = DataWriter(**cfg.writer)
    stop_control, get_action = start_control_loop(gello, nuc)

    for i in tqdm(range(cfg.max_episode_timesteps)):
        colors, color_timesteps, depths, depth_timesteps = realsense.get_synchronized_frame()
        state = nuc.get_robot_state()
        writer.write_frame(i, colors, depths)
        writer.write_state(timestep=i, action=get_action(), **state)
    stop_control()
    writer.flush()
    exit(0)

if __name__ == "__main__":
    main()