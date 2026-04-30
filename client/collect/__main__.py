"""Episode data collection entry point.

Orchestrates teleoperated demonstration recording by coordinating the GELLO
controller, RealSense cameras, and Franka Panda robot. Each episode captures
synchronized multi-camera RGB-D frames alongside robot state and operator actions,
written to disk via :class:`~client.collect.write.ACMEWriter`.

Usage::

    python -m client.collect
"""
import shutil
import traceback
import time
import threading
import queue
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from client.nuc import NUCInterface
from client.collect.realsense import RealSenseInterface
from client.collect.gello import GELLOInterface
from client.utils import get_latest_ep_path, validate_episode
from client.collect.write import ACMEWriter



def action_step(gello: GELLOInterface, nuc: NUCInterface, task_cfg: DictConfig):
    joint_angles = gello.get_joint_angles()
    eef_pos, eef_rot = nuc.forward_kinematics(torch.tensor(joint_angles))

    home_pos, home_rot = nuc.home
    pos_mask = np.array(task_cfg.pos_mask)
    eef_pos = np.where(pos_mask, eef_pos, home_pos)
    if task_cfg.freeze_rotation:
        eef_rot = home_rot

    gripper_force = np.array([gello.get_gripper()])
    gripper_cmd = None if task_cfg.freeze_gripper else gripper_force

    nuc.send_control(eef_pos, eef_rot, gripper_cmd)
    action = np.concatenate([eef_pos, eef_rot, gripper_force])
    return action


@hydra.main(config_path="../../config", config_name="collect")
def main(cfg: DictConfig):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    nuc = NUCInterface(**cfg.nuc)
    gello = GELLOInterface(**cfg.gello)
    base_ep_path = Path(cfg.episodes_path)
    base_ep_path.mkdir(exist_ok=True, parents=True)
    while True:
        try:
            ep_path = get_latest_ep_path(base_ep_path, prefix="episode")
            print(f"Recording to {ep_path}")
            with RealSenseInterface(ep_path, **cfg.realsense) as rs_interface:
                episode_writer = ACMEWriter(ep_path, serials=rs_interface.serials, **cfg.writer)

                nuc.reset(open_gripper=cfg.task.open_gripper_on_reset)

                primary_serial = rs_interface.serials[0]

                ctrl_logging = cfg.task.get("controller_logging", False)

                # Fast teleop thread: drives GELLO -> Franka commands at cfg.teleop_rate,
                # independent of camera FPS. The most recently issued action is published
                # via latest_action[0] for the logging thread to sample. Thread is
                # constructed here but only started after start_capture so the operator
                # gets control at the same moment recording begins.
                home_pos, home_rot = nuc.home
                latest_action = [np.concatenate([home_pos, home_rot, np.zeros(1)])]
                teleop_stop = threading.Event()
                teleop_period = 1.0 / float(cfg.teleop_rate)

                def teleop_loop():
                    next_t = time.time()
                    while not teleop_stop.is_set():
                        try:
                            latest_action[0] = action_step(gello, nuc, cfg.task)
                        except Exception as e:
                            print(f"Teleop error: {e}")
                            traceback.print_exc()
                        next_t += teleop_period
                        sleep_for = next_t - time.time()
                        if sleep_for > 0:
                            time.sleep(sleep_for)
                        else:
                            next_t = time.time()

                teleop_thread = threading.Thread(target=teleop_loop, daemon=True)

                # Logging-only worker: one entry per primary RGB frame, samples robot
                # state and the most recent commanded action. Bounded queue with
                # drop-old policy so logging backlog can never lag the camera.
                state_queue = queue.Queue(maxsize=1)
                state_worker_stop = threading.Event()

                def state_worker():
                    steps = 0
                    while not state_worker_stop.is_set():
                        try:
                            timestamp = state_queue.get(timeout=1.0)
                        except queue.Empty:
                            continue
                        if steps >= cfg.max_episode_timesteps:
                            continue
                        try:
                            state = nuc.get_robot_state()

                            if ctrl_logging:
                                diag = nuc.get_controller_diagnostics()
                                state.update(diag)
                                print(f"\t[ctrl] pos_err={np.linalg.norm(diag['cart_pos_error'][:3])*1000:.1f}mm "
                                      f"rot_err={np.linalg.norm(diag['cart_pos_error'][3:])*180/np.pi:.1f}deg | "
                                      f"tau: stiff={np.linalg.norm(diag['tau_stiffness']):.2f} "
                                      f"damp={np.linalg.norm(diag['tau_damping']):.2f} "
                                      f"null={np.linalg.norm(diag['tau_nullspace']):.2f}")

                            episode_writer.write_state(timestamp=timestamp, action=latest_action[0], **state)
                            steps += 1
                        except Exception as e:
                            print(f"State worker error: {e}")
                            traceback.print_exc()

                state_thread = threading.Thread(target=state_worker, daemon=True)
                state_thread.start()

                def on_receive_frame(serial, timestamp):
                    if serial != primary_serial:
                        return
                    try:
                        state_queue.put_nowait(timestamp)
                    except queue.Full:
                        # Logging is behind the camera — drop the older pending
                        # timestamp in favor of the fresher one.
                        try:
                            state_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            state_queue.put_nowait(timestamp)
                        except queue.Full:
                            pass

                rs_interface.reset_frame_counts()
                rs_interface.start_capture(on_receive_frame)#, on_warmup=nuc.home_gripper)
                # Recording is now live (phase 4 of start_capture). Zero the
                # GELLO offset against the robot's current pose and engage
                # teleop so the operator gets control at the same moment the
                # first frame is recorded.
                gello.zero_controls(nuc.get_robot_state()["qpos"])
                teleop_thread.start()
                while any([c < cfg.max_episode_timesteps for c in rs_interface.get_frame_counts().values()]):
                    time.sleep(2.0)
                    qsize = state_queue.qsize()
                    backlog_warn = "  WARN: logging backlog" if qsize > 0 else ""
                    print("Episode progress:", np.array(list(rs_interface.get_frame_counts().values())) / cfg.max_episode_timesteps,
                          f"  state_queue: {qsize}{backlog_warn}")

                teleop_stop.set()
                teleop_thread.join()
                state_worker_stop.set()
                state_thread.join()
            episode_writer.flush()
            ok, errors = validate_episode(episode_writer.episode_path)
            if not ok:
                print(f"WARNING: episode {episode_writer.episode_path} failed validation:")
                for err in errors:
                    print(f"  - {err}")
        except Exception as e:
            print(e)
            traceback.print_exc()
        finally:
            ep_control_msg = "1: Continue and start the next recording\n0: to delete this recording.\nx: Exit"
            ep_control_cmd = str(input(ep_control_msg))
            while ep_control_cmd not in ["1", "x"]:
                if ep_control_cmd == "0":
                    cmd_confirm = input(f"Input 0 again to confirm deletion of {episode_writer.episode_path}")
                    if cmd_confirm == "0":
                        shutil.rmtree(episode_writer.episode_path)
                ep_control_cmd = str(input(ep_control_msg))
            if ep_control_cmd == "x":
                break



if __name__ == "__main__":
    main()
