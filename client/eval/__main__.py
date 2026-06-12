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

from client.eval.cameras import EvalCameras
from client.eval.writer import EvalWriter
from client.eval.policy import EvalPolicyInterface
from client.eval.live_plotter import LiveControlErrorPlotter
from client.nuc import NUCInterface

# OopsieData failure-mode logging. Imports are deferred so the module can be
# used without oopsie-tools installed when cfg.oopsie.enabled is false.
try:
    from oopsie_tools.annotation_tool.episode_recorder import EpisodeRecorder
    from oopsie_tools.utils.robot_profile import load_robot_profile
    _OOPSIE_AVAILABLE = True
except ImportError:
    _OOPSIE_AVAILABLE = False


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
        realsense: EvalCameras,
        writer: EvalWriter,
        nuc: NUCInterface,
        task_cfg: DictConfig,
        safety_cfg: DictConfig,
        settle_cfg: DictConfig,
        print_action: bool = False,
):
    stop_event = threading.Event()
    safety_state = {"violation": None}

    pos_mask = np.array(list(task_cfg.pos_mask))
    safety_enabled = bool(safety_cfg.get("enabled", True))
    max_lin_vel = float(safety_cfg.max_linear_velocity)
    max_ang_vel = float(safety_cfg.max_angular_velocity)
    settle_threshold_m = float(settle_cfg.threshold_mm) / 1000.0
    settle_timeout_s = float(settle_cfg.timeout_s)
    settle_poll_s = float(settle_cfg.poll_period_s)
    action_type = str(task_cfg.get("action_type", "cartesian"))
    if action_type not in ("qpos", "cartesian"):
        raise ValueError(f"task.action_type must be 'qpos' or 'cartesian', got {action_type!r}")
    prev_cmd = {"pos": None, "quat": None}
    if not safety_enabled:
        print("[SAFETY] Custom motion safety check disabled; relying on robot-side limits.")

    def _loop_iter():
        frames: List[torch.Tensor] = realsense.get_rgb_obs()
        resized_frames = [policy.preprocess_frame(f) for f in frames]
        # TODO:  A little weird this goes through the writer, but whatever
        all_states = writer.get_states_snapshot()
        recent = all_states[-policy.obs_history_size:]
        eef_pos = np.stack([s["ee_pos"] for s in recent])
        eef_rot = np.stack([s["ee_rot"] for s in recent])
        qpos = np.stack([s["qpos"] for s in recent])

        if task_cfg.zero_gripper_obs:
            gripper_force = np.zeros((policy.obs_history_size, 1))
        else:
            gripper_force = np.stack([s["gripper_force"] for s in recent]).reshape(-1, 1)


        # Slice observation to active position dims (frozen dims excluded)
        eef_pos = eef_pos[:, pos_mask]

        desired_eef_pos, desired_eef_quat, desired_gripper_force, desired_qpos = policy(
            rgb_0=resized_frames[0].unsqueeze(0),
            rgb_1=resized_frames[1].unsqueeze(0),
            eef_pos=np.expand_dims(eef_pos, 0),
            eef_quat=np.expand_dims(eef_rot, 0),
            gripper_force=np.expand_dims(gripper_force, 0),
            qpos=np.expand_dims(qpos, 0),
        )

        horizon_len = desired_eef_pos.shape[0]
        home_eef_pos, home_eef_rot = nuc.home
        desired_eef_pos = desired_eef_pos.to(torch.float64)
        desired_eef_quat = desired_eef_quat.to(torch.float64)
        desired_qpos = desired_qpos.to(torch.float64)

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
            desired_gripper_force=desired_gripper_force.numpy().reshape(-1, 1),
            desired_qpos=desired_qpos.numpy(),
        )
        per_step_sleep = 1.0 / (policy.control_frequency * horizon_len)

        # Build the sequences used for the safety check. Position is convention-free,
        # so we anchor it to the previous command (or current robot pose on the very
        # first iter). For rotation we anchor only when we have a previous policy
        # command to compare against — otherwise the quaternion convention may not
        # match (e.g. home_rot vs policy output) and produce a spurious violation.
        desired_pos_np = desired_eef_pos.numpy()
        desired_quat_np = desired_eef_quat.numpy()

        if print_action:
            if action_type == "qpos":
                print(f"[action] qpos horizon={horizon_len} qpos={desired_qpos.numpy().tolist()} gripper={desired_gripper_force.numpy().tolist()}")
            else:
                print(f"[action] eef horizon={horizon_len} pos={desired_pos_np.tolist()} quat={desired_quat_np.tolist()} gripper={desired_gripper_force.numpy().tolist()}")

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
            if action_type == "qpos":
                nuc.send_qpos_control_tensor(desired_qpos[i], gripper_cmd)
            else:
                nuc.send_control_tensor(desired_eef_pos[i], desired_eef_quat[i], gripper_cmd)
            time.sleep(per_step_sleep)

        # Block until the EE is within settle_threshold_m of the commanded
        # final pose (or we time out). Replaces a fixed post-horizon sleep
        # so the policy never sees an observation while the robot is still
        # catching up to the previous horizon's target. Measured-side frame
        # must match the target frame: qpos mode commands go through FK so
        # target is in flange frame (no TCP) — measure via get_fk_ee_pos();
        # cartesian mode targets are the controller setpoint (O_T_EE / TCP
        # frame) — measure via get_robot_state()["ee_pos"]. Mixing them
        # leaks F_T_EE as a constant bias.
        target_pos = desired_pos_np[-1]
        target_quat = desired_quat_np[-1]
        measure_pos = (
            nuc.get_fk_ee_pos
            if action_type == "qpos"
            else (lambda: nuc.get_robot_state()["ee_pos"])
        )
        settle_start = time.time()
        settled = False
        err_m = float("inf")
        last_measured = measure_pos()
        while time.time() - settle_start < settle_timeout_s:
            last_measured = measure_pos()
            err_m = float(np.linalg.norm(last_measured - target_pos))
            if err_m <= settle_threshold_m:
                settled = True
                break
            time.sleep(settle_poll_s)
        if not settled:
            settle_elapsed_ms = (time.time() - settle_start) * 1000
            per_axis_err_mm = (last_measured - target_pos) * 1000.0
            per_axis_str = np.array2string(per_axis_err_mm, precision=2, suppress_small=True)
            measured_quat = nuc.get_robot_state()["ee_rot"]
            q_t = target_quat / max(float(np.linalg.norm(target_quat)), 1e-12)
            q_m = measured_quat / max(float(np.linalg.norm(measured_quat)), 1e-12)
            rot_err_deg = float(np.degrees(
                2.0 * np.arccos(np.clip(abs(float(np.dot(q_t, q_m))), 0.0, 1.0))
            ))
            print(
                f"[settle] timeout after {settle_elapsed_ms:.0f}ms; "
                f"pos err {err_m*1000:.2f}mm > {settle_threshold_m*1000:.2f}mm; "
                f"per-axis (mm) {per_axis_str}; "
                f"rot err {rot_err_deg:.2f}deg"
            )

    def _loop_runner():
        while not stop_event.is_set():
            _loop_iter()

    loop_thread = threading.Thread(target=_loop_runner, daemon=True)
    loop_thread.start()

    def stop_loop():
        stop_event.set()
        loop_thread.join()

    return stop_loop, stop_event, safety_state


def record_episode(cfg, ep_path, nuc, policy, oopsie_recorder=None):
    writer = None
    safety_state = {"violation": None}
    try:
        with EvalCameras(**cfg.cameras) as rsi:
            writer = EvalWriter(path=ep_path, **cfg.writer)
            try:
                plotter = LiveControlErrorPlotter()
            except Exception as e:
                print(f"[viz] Failed to start live control-error plot ({e}); continuing without it.")
                plotter = None
            nuc.reset(open_gripper=cfg.task.open_gripper_on_reset)

            if oopsie_recorder is not None:
                oopsie_recorder.reset_episode_recorder()

            if bool(cfg.get("render_eval", False)):
                writer.register_cameras(rsi.serials, fps=cfg.cameras.fps)

            # The oopsie EpisodeRecorder profile (acme_franka.yaml) names cameras
            # "external" and "wrist", positionally matching cfg.cameras.obs_cams
            # (rsi.serials[0] -> external, rsi.serials[1] -> wrist).
            oopsie_serial_to_cam = dict(zip(rsi.serials, ["external", "wrist"]))
            # Latest single frame per serial. on_receive_frame delivers one current
            # BGR frame (H,W,3) per camera; we stash the newest from each so the
            # primary tick can assemble a synchronized multi-cam observation. This
            # deliberately avoids rsi.get_rgb_obs(), whose per-backend obs-history
            # caches are empty during warmup (ZED stack would raise) and return a
            # whole history stack, not a single frame.
            oopsie_latest_frames = {}

            primary_serial = rsi.serials[0]
            def on_receive_frame(serial, frame):
                writer.on_frame(serial, frame)
                if oopsie_recorder is not None:
                    oopsie_latest_frames[serial] = frame
                if serial == primary_serial:
                    c_state = nuc.get_robot_state()
                    desired_pose = nuc.get_desired_ee_pose()
                    c_state.update(dict(action=desired_pose))
                    writer.on_state_update(c_state)
                    if oopsie_recorder is not None and all(
                        s in oopsie_latest_frames for s in oopsie_serial_to_cam
                    ):
                        image_obs = {}
                        for s, cam_name in oopsie_serial_to_cam.items():
                            raw = oopsie_latest_frames[s]
                            arr = raw.numpy() if hasattr(raw, "numpy") else np.asarray(raw)
                            # Cameras emit BGR (rs.format.bgr8 / ZED BGRA->BGR); the
                            # recorder writes RGB mp4s, so flip channels to RGB.
                            image_obs[cam_name] = np.ascontiguousarray(arr[..., ::-1]).astype(np.uint8)
                        cartesian_position = np.concatenate(
                            [c_state["ee_pos"], c_state["ee_rot"]]
                        )
                        oopsie_recorder.record_step(
                            observation={
                                "image_observation": image_obs,
                                "robot_state": {
                                    "joint_position": np.asarray(c_state["qpos"]),
                                    "cartesian_position": cartesian_position,
                                    "gripper_position": np.asarray(c_state["gripper_force"]),
                                },
                            },
                            action={
                                "cartesian_position": desired_pose,
                                "gripper_position": np.asarray(c_state["gripper_force"]),
                            },
                        )

            rsi.start_capture(on_receive_frame)#, on_warmup=nuc.home_gripper)
            print("Waiting for realsense caches to fill")
            time.sleep(5.0)

            nuc.start()

            stop_control, control_stop_event, safety_state = start_control_loop(
                policy, rsi, writer, nuc, cfg.task, cfg.safety, cfg.settle,
                print_action=bool(cfg.get("print_action", False)),
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

    oopsie_recorder = None
    oopsie_cfg = cfg.get("oopsie", None)
    if oopsie_cfg is not None and oopsie_cfg.get("enabled", False):
        if not _OOPSIE_AVAILABLE:
            raise ImportError(
                "cfg.oopsie.enabled=true but oopsie_tools is not importable. "
                "Install via `pip install -e oopsie-tools` in the active env."
            )
        profile = load_robot_profile(oopsie_cfg.robot_profile)
        oopsie_recorder = EpisodeRecorder(
            robot_profile=profile,
            data_root_dir=oopsie_cfg.data_root_dir,
            operator_name=oopsie_cfg.operator_name or None,
        )

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
            record_episode(cfg, ep_path, nuc, policy, oopsie_recorder=oopsie_recorder)
            if oopsie_recorder is not None:
                oopsie_recorder.finish_rollout(instruction=cfg.task.instruction)
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
