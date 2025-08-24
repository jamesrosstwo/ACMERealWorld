import shutil
import traceback
from collections import defaultdict
import time

import hydra
import numpy as np
import torch
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
    pose_lock = threading.Lock()
    latest_eef_pos, latest_eef_rot = None, None

    def _loop_iter():
        nonlocal latest_eef_pos, latest_eef_rot, pose_lock
        joint_angles = gello.get_joint_angles()
        eef_pos, eef_rot = nuc.forward_kinematics(torch.tensor(joint_angles))

        with pose_lock:
            latest_eef_pos = eef_pos
            latest_eef_rot = eef_rot
        # PUSH T FREEZES
        eef_pos[-1] = 0.3
        eef_rot = np.array([0.94, 0.335, 0, -0.03])
        nuc.send_control(eef_pos, eef_rot, None)

    def _loop_runner():
        while not stop_event.is_set():
            _loop_iter()

    loop_thread = threading.Thread(target=_loop_runner, daemon=True)
    loop_thread.start()

    def stop_loop():
        stop_event.set()
        loop_thread.join()

    def get_latest_eef_pos():
        nonlocal latest_eef_pos, latest_eef_rot, pose_lock
        with pose_lock:
            assert latest_eef_rot is not None and latest_eef_pos is not None
            return latest_eef_pos.copy(), latest_eef_rot.copy()

    return stop_loop, get_latest_eef_pos



@hydra.main(config_path="config", config_name="collect")
def main(cfg: DictConfig):
    nuc = NUCInterface(**cfg.nuc)
    gello = GELLOInterface(**cfg.gello)
    while True:
        with RealSenseInterface(**cfg.realsense) as rs_interface:
            try:
                episode_writer = ACMEWriter(**cfg.writer)
                nuc.reset()
                stop_control, get_desired_eef_pose = start_control_loop(gello, nuc)

                def get_action():
                    return np.concatenate(get_desired_eef_pose())

                def on_receive_frame(capture_idx):
                    if capture_idx == 0:
                        state = nuc.get_robot_state()
                        # for episode collection we just assume all cameras are synchronized to cam0,
                        # and that this synchronous operation of getting robot state from the NUC takes
                        # no time.
                        episode_writer.write_state(action=get_action(), **state)

                rs_interface.start_capture(on_receive_frame)
                for i in range(rs_interface.n_cameras):
                    rs_interface.frame_counts[i] = 0
                while any([c < cfg.max_episode_timesteps for c in rs_interface.frame_counts.values()]):
                    time.sleep(2.0)
                    print("Episode progress:", np.array(list(rs_interface.frame_counts.values())) / cfg.max_episode_timesteps)
                stop_control()
                time.sleep(5.0)
                m = f"processing capture"
                for color, color_tmstmp,depth, depth_tmstmp, cap_idx in tqdm(rs_interface.process_all_frames_parallel(), m):
                    try:
                        episode_writer.write_capture_frame(cap_idx, color_tmstmp, color, depth)
                    except IndexError:
                        break
            except Exception as e:
                print(e)
                traceback.print_exc()
            finally:
                episode_writer.flush()
                ep_control_msg = "Input 1 to continue to start the next recording, or 0 to delete this recording."
                ep_control_cmd = str(input(ep_control_msg))
                while ep_control_cmd != "1":
                    if ep_control_cmd == "0":
                        cmd_confirm = input(f"Input 0 again to confirm deletion of {episode_writer.episode_path}")
                        if cmd_confirm == "0":
                            shutil.rmtree(episode_writer.episode_path)
                    ep_control_cmd = str(input(ep_control_msg))



if __name__ == "__main__":
    main()
