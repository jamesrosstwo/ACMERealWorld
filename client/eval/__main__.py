import select
import shutil
import sys
import traceback
import time
import threading
from pathlib import Path
from typing import List

import hydra
import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from client.eval.rsi import EvalRSI
from client.eval.writer import EvalWriter
from client.eval.policy import EvalPolicyInterface
from client.nuc import NUCInterface


def listen_for_keypress(cancel_event):
    print("Press 'c' to cancel the episode.")
    while not cancel_event.is_set():
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key.lower() == 'c':
                cancel_event.set()
                print("\nEpisode cancelled by user (keypress 'c').")


def start_control_loop(
        policy: EvalPolicyInterface,
        realsense: EvalRSI,
        writer: EvalWriter,
        nuc: NUCInterface,
):
    stop_event = threading.Event()

    def _loop_iter():
        frames: List[torch.Tensor] = realsense.get_rgb_obs()
        resized_frames = [policy.preprocess_frame(f) for f in frames]
        # TODO:  A little weird this goes through the writer, but whatever
        all_states = list(writer.states)
        eef_pos = np.stack([s["ee_pos"] for s in all_states[-policy.obs_history_size:]])
        eef_rot = np.stack([s["ee_rot"] for s in all_states[-policy.obs_history_size:]])

        cur_eef_pos = nuc.get_robot_state()["ee_pos"]
        print(f"\tInput positional error: {(cur_eef_pos - eef_pos[-1]) * 100}cm")
        gripper_force = np.zeros((policy.obs_history_size, 1))
        eef_pos[:, -1] = 0

        desired_eef_pos, desired_eef_quat, desired_gripper_force = policy(
            rgb_0=resized_frames[0].unsqueeze(0),
            rgb_1=resized_frames[1].unsqueeze(0),
            eef_pos=np.expand_dims(eef_pos, 0),
            eef_quat=np.expand_dims(eef_rot, 0),
            gripper_force=np.expand_dims(gripper_force, 0)
        )


        horizon_len = desired_eef_pos.shape[0]
        home_eef_pos, home_eef_rot = nuc.pusht_home
        desired_eef_pos[:, -1] = home_eef_pos[-1]
        desired_eef_pos = desired_eef_pos.to(torch.float64)
        desired_eef_quat = torch.zeros((horizon_len, 4))
        desired_eef_quat[:] = torch.from_numpy(home_eef_rot)
        desired_eef_quat = desired_eef_quat.to(torch.float64)
        writer.on_inference(
            ee_pos=eef_pos,
            ee_quat=eef_rot,
            gripper_force=gripper_force,
            desired_ee_pos=desired_eef_pos.numpy(),
            desired_ee_quat=desired_eef_quat.numpy(),
            desired_gripper_force=np.zeros((desired_eef_pos.shape[0], 1))
        )
        per_step_sleep = 1.0 / (policy.control_frequency * horizon_len)
        for i in range(horizon_len):
            nuc.send_control_tensor(desired_eef_pos[i], desired_eef_quat[i], None)
            time.sleep(per_step_sleep)
        time.sleep(2 * per_step_sleep)
        eef_pos = nuc.get_robot_state()["ee_pos"]

    def _loop_runner():
        while not stop_event.is_set():
            _loop_iter()

    loop_thread = threading.Thread(target=_loop_runner, daemon=True)
    loop_thread.start()

    def stop_loop():
        stop_event.set()
        loop_thread.join()

    return stop_loop


def record_episode(cfg, ep_path, nuc, policy):
    with EvalRSI(**cfg.realsense) as rsi:
        writer = EvalWriter(path=ep_path, **cfg.writer)
        nuc.reset()

        def on_receive_frame(cap_idx):
            if cap_idx == 0:
                c_state = nuc.get_robot_state()
                c_state.update(dict(
                    action=nuc.get_desired_ee_pose()
                ))
                writer.on_state_update(c_state)

        rsi.start_capture(on_receive_frame)
        print("Waiting for realsense caches to fill")
        time.sleep(5.0)

        nuc.start()

        stop_control = start_control_loop(policy, rsi, writer, nuc)

        cancel_event = threading.Event()
        keypress_thread = threading.Thread(target=listen_for_keypress, args=(cancel_event,), daemon=True)
        keypress_thread.start()

        for i in range(rsi.n_cameras):
            rsi.frame_counts[i] = 0
        while any([c < cfg.max_episode_timesteps for c in rsi.frame_counts.values()]):
            time.sleep(2.0)
            print("Episode progress:",
                  np.array(list(rsi.frame_counts.values())) / cfg.max_episode_timesteps)
            # Check for the cancel event (keypress 'c' to cancel)
            if cancel_event.is_set():
                print("Episode stopped by user.")
                break
        stop_control()
    writer.flush()


@hydra.main(config_path="../../config", config_name="eval")
def main(cfg: DictConfig):
    nuc = NUCInterface(**cfg.nuc)
    policy = EvalPolicyInterface(**cfg.policy)

    ep_idx = cfg.start_index

    timestamp = int(time.time())
    out_path = Path(f"../outputs/evaluation/{timestamp}")
    out_path.mkdir(parents=True, exist_ok=True)

    # Save Hydra config to the output directory
    config_out_path = out_path / "config.yaml"
    OmegaConf.save(config=cfg, f=config_out_path)
    ep_path = None
    successes = []

    should_exit = False

    while not should_exit:
        try:
            ep_path = out_path / f"episode_{ep_idx}"
            ep_path.mkdir()
            record_episode(cfg, ep_path, nuc, policy)
            ep_idx += 1
        except Exception as e:
            print(e)
            traceback.print_exc()
        finally:
            ep_control_msg = "1: Record Success\n2: Record Failure\n0: Delete this recording.\nx: Exit\nz: Next episode\n"
            while True:
                ep_control_cmd = str(input(ep_control_msg)).strip()
                if ep_control_cmd == "1":
                    successes.append(True)
                    break
                elif ep_control_cmd == "2":
                    successes.append(False)
                    break
                elif ep_control_cmd == "0":
                    if ep_path and ep_path.exists():
                        shutil.rmtree(ep_path)
                elif ep_control_cmd == "x":
                    should_exit = True
                    break
                elif ep_control_cmd == "z":
                    break

        if len(successes) > 0:
            stats_path = out_path / "stats.yaml"
            stats_path.unlink(missing_ok=True)
            stats = dict(
                n_successes=sum(successes),
                success_rate=sum(successes) / len(successes),
                successes=successes,
            )

            with open(stats_path, 'w') as f:
                yaml.dump(stats, f)


if __name__ == "__main__":
    main()
