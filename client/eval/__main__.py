"""Policy evaluation entry point.

Runs a learned policy on the Franka Panda robot using RealSense camera
observations. A threaded control loop queries the policy server for actions,
sends them to the robot, and records episode trajectories with success/failure
labels for computing evaluation metrics.

Usage::

    python -m client.eval
"""
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

from client.eval.realsense import EvalRealsense
from client.eval.writer import EvalWriter
from client.eval.policy import EvalPolicyInterface
from client.eval.live_plotter import LiveControlErrorPlotter
from client.nuc import NUCInterface


class AggressiveMotionError(RuntimeError):
    """Raised when a policy-issued horizon would exceed configured velocity limits."""


def _check_safety(pos_seq: np.ndarray, quat_seq: np.ndarray, dt: float,
                  max_lin_vel: float, max_ang_vel: float):
    """Validate that the implied per-step velocities stay under the configured limits.

    `pos_seq` is (N+1, 3) and `quat_seq` is (M+1, 4); each consecutive pair is
    treated as `dt` apart. Rotation is measured via quaternion dot product so the
    result is invariant to xyzw vs wxyz convention as long as the sequence is
    self-consistent.
    """
    lin_vel = np.linalg.norm(np.diff(pos_seq, axis=0), axis=1) / dt
    if lin_vel.size and lin_vel.max() > max_lin_vel:
        i = int(np.argmax(lin_vel))
        return False, (
            f"linear velocity {lin_vel[i]:.3f} m/s exceeds limit {max_lin_vel:.3f} m/s "
            f"at horizon step {i}"
        )

    if quat_seq.shape[0] >= 2:
        norms = np.linalg.norm(quat_seq, axis=1, keepdims=True)
        # Avoid div-by-zero on degenerate quats; treat as identity.
        norms = np.where(norms > 0, norms, 1.0)
        unit = quat_seq / norms
        dots = np.abs(np.sum(unit[1:] * unit[:-1], axis=1))
        angles = 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))
        ang_vel = angles / dt
        if ang_vel.max() > max_ang_vel:
            i = int(np.argmax(ang_vel))
            return False, (
                f"angular velocity {ang_vel[i]:.3f} rad/s exceeds limit "
                f"{max_ang_vel:.3f} rad/s at horizon step {i}"
            )
    return True, ""


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
        realsense: EvalRealsense,
        writer: EvalWriter,
        nuc: NUCInterface,
        task_cfg: DictConfig,
        safety_cfg: DictConfig,
):
    stop_event = threading.Event()
    safety_state = {"violation": None}

    pos_mask = np.array(list(task_cfg.pos_mask))
    safety_enabled = bool(safety_cfg.get("enabled", True))
    max_lin_vel = float(safety_cfg.max_linear_velocity)
    max_ang_vel = float(safety_cfg.max_angular_velocity)
    prev_cmd = {"pos": None, "quat": None}
    if not safety_enabled:
        print("[SAFETY] Custom motion safety check disabled; relying on robot-side limits.")

    def _loop_iter():
        frames: List[torch.Tensor] = realsense.get_rgb_obs()
        resized_frames = [policy.preprocess_frame(f) for f in frames]
        # TODO:  A little weird this goes through the writer, but whatever
        all_states = writer.get_states_snapshot()
        eef_pos = np.stack([s["ee_pos"] for s in all_states[-policy.obs_history_size:]])
        eef_rot = np.stack([s["ee_rot"] for s in all_states[-policy.obs_history_size:]])

        if task_cfg.zero_gripper_obs:
            gripper_force = np.zeros((policy.obs_history_size, 1))
        else:
            gripper_force = np.stack([s["gripper_force"] for s in all_states[-policy.obs_history_size:]]).reshape(-1, 1)


        # Slice observation to active position dims (frozen dims excluded)
        eef_pos = eef_pos[:, pos_mask]

        desired_eef_pos, desired_eef_quat, desired_gripper_force = policy(
            rgb_0=resized_frames[0].unsqueeze(0),
            rgb_1=resized_frames[1].unsqueeze(0),
            eef_pos=np.expand_dims(eef_pos, 0),
            eef_quat=np.expand_dims(eef_rot, 0),
            gripper_force=np.expand_dims(gripper_force, 0)
        )




        horizon_len = desired_eef_pos.shape[0]
        home_eef_pos, home_eef_rot = nuc.home
        desired_eef_pos = desired_eef_pos.to(torch.float64)
        desired_eef_quat = desired_eef_quat.to(torch.float64)

        # Replace frozen position dims with home values
        frozen = torch.from_numpy(~pos_mask)
        desired_eef_pos[:, frozen] = torch.from_numpy(home_eef_pos)[frozen]
        if task_cfg.freeze_rotation:
            desired_eef_quat = torch.zeros((horizon_len, 4), dtype=torch.float64)
            desired_eef_quat[:] = torch.from_numpy(home_eef_rot)

        writer.on_inference(
            ee_pos=eef_pos,
            ee_quat=eef_rot,
            gripper_force=gripper_force,
            desired_ee_pos=desired_eef_pos.numpy(),
            desired_ee_quat=desired_eef_quat.numpy(),
            desired_gripper_force=desired_gripper_force.numpy().reshape(-1, 1)
        )
        per_step_sleep = 1.0 / (policy.control_frequency * horizon_len)

        # Build the sequences used for the safety check. Position is convention-free,
        # so we anchor it to the previous command (or current robot pose on the very
        # first iter). For rotation we anchor only when we have a previous policy
        # command to compare against — otherwise the quaternion convention may not
        # match (e.g. home_rot vs policy output) and produce a spurious violation.
        desired_pos_np = desired_eef_pos.numpy()
        desired_quat_np = desired_eef_quat.numpy()
        if safety_enabled:
            if prev_cmd["pos"] is None:
                anchor_pos = nuc.get_robot_state()["ee_pos"]
                pos_seq = np.concatenate([anchor_pos[None, :], desired_pos_np], axis=0)
                quat_seq = desired_quat_np
            else:
                pos_seq = np.concatenate([prev_cmd["pos"][None, :], desired_pos_np], axis=0)
                quat_seq = np.concatenate([prev_cmd["quat"][None, :], desired_quat_np], axis=0)

            ok, msg = _check_safety(pos_seq, quat_seq, per_step_sleep, max_lin_vel, max_ang_vel)
            if not ok:
                safety_state["violation"] = msg
                stop_event.set()
                print(f"\n[SAFETY] Aggressive motion command rejected: {msg}")
                return

            prev_cmd["pos"] = desired_pos_np[-1]
            prev_cmd["quat"] = desired_quat_np[-1]

        for i in range(horizon_len):
            gripper_cmd = None if task_cfg.freeze_gripper else desired_gripper_force[i]
            nuc.send_control_tensor(desired_eef_pos[i], desired_eef_quat[i], gripper_cmd)
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

    return stop_loop, stop_event, safety_state


def record_episode(cfg, ep_path, nuc, policy):
    writer = None
    safety_state = {"violation": None}
    try:
        with EvalRealsense(**cfg.realsense) as rsi:
            writer = EvalWriter(path=ep_path, **cfg.writer)
            try:
                plotter = LiveControlErrorPlotter()
            except Exception as e:
                print(f"[viz] Failed to start live control-error plot ({e}); continuing without it.")
                plotter = None
            nuc.reset(open_gripper=cfg.task.open_gripper_on_reset)

            primary_serial = rsi.serials[0]
            def on_receive_frame(serial):
                if serial == primary_serial:
                    c_state = nuc.get_robot_state()
                    c_state.update(dict(
                        action=nuc.get_desired_ee_pose()
                    ))
                    writer.on_state_update(c_state)

            rsi.start_capture(on_receive_frame)#, on_warmup=nuc.home_gripper)
            print("Waiting for realsense caches to fill")
            time.sleep(5.0)

            nuc.start()

            stop_control, control_stop_event, safety_state = start_control_loop(
                policy, rsi, writer, nuc, cfg.task, cfg.safety
            )

            cancel_event = threading.Event()
            keypress_thread = threading.Thread(target=listen_for_keypress, args=(cancel_event,), daemon=True)
            keypress_thread.start()

            rsi.reset_frame_counts()
            last_print = time.time()
            try:
                while any([c < cfg.max_episode_timesteps for c in rsi.get_frame_counts().values()]):
                    if plotter is not None:
                        states = writer.get_states_snapshot()
                        if states:
                            ee_hist = np.stack([s["ee_pos"] for s in states])
                            desired_hist = np.stack([s["action"][:3] for s in states])
                        else:
                            ee_hist = np.empty((0, 3))
                            desired_hist = np.empty((0, 3))
                        try:
                            plotter.update(ee_hist, desired_hist)
                        except Exception as e:
                            print(f"[viz] Live plot update failed ({e}); disabling.")
                            plotter.close()
                            plotter = None
                    time.sleep(0.1)
                    if time.time() - last_print >= 2.0:
                        print("Episode progress:",
                              np.array(list(rsi.get_frame_counts().values())) / cfg.max_episode_timesteps)
                        last_print = time.time()
                    # Check for the cancel event (keypress 'c' to cancel)
                    if cancel_event.is_set():
                        print("Episode stopped by user.")
                        break
                    if control_stop_event.is_set():
                        # Control loop self-terminated (e.g. safety violation).
                        break
            finally:
                stop_control()
                if plotter is not None:
                    plotter.close()
    finally:
        if writer is not None:
            writer.flush()
    if safety_state["violation"] is not None:
        raise AggressiveMotionError(safety_state["violation"])


@hydra.main(config_path="../../config", config_name="eval")
def main(cfg: DictConfig):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
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
    safety_abort = False

    while not should_exit:
        try:
            ep_path = out_path / f"episode_{ep_idx:03d}"
            ep_path.mkdir()
            record_episode(cfg, ep_path, nuc, policy)
            ep_idx += 1
        except AggressiveMotionError as e:
            print(f"\nSAFETY ABORT: {e}")
            print("Exiting eval to protect the robot.")
            safety_abort = True
            should_exit = True
        except Exception as e:
            print(e)
            traceback.print_exc()
        finally:
            if not safety_abort:
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
