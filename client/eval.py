import io
import logging
import select
import shutil
import sys
import traceback
import time
from collections import deque
from pathlib import Path
from typing import Callable, List, Dict

import torch.nn.functional as F

import hydra
import numpy as np
import requests
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
import plotly.graph_objects as go

from client.nuc import NUCInterface
from client.realsense import RealSenseInterface, enumerate_devices
import pyrealsense2 as rs
import threading




class EvalWriter:
    def __init__(self, path: Path, max_episode_len: int):
        self.path = path
        assert self.path.exists()
        self._max_episode_len = max_episode_len
        self.states = []

    @property
    def episode_path(self):
        return self.path

    import plotly.graph_objects as go
    import numpy as np

    def write_trajectory_plot(self, stacked_states: Dict[str, np.ndarray]):
        plot_path = self.path / "trajectory.html"

        # Extract positions from the stacked states
        eef_pos = stacked_states["ee_pos"]  # shape: (N, 3)
        action_pos = stacked_states["action"][:, :3]  # Take only first 3 components, shape: (N, 3)

        # Get bounds for the cube (using the min and max positions of eef_pos and action_pos)
        min_pos = np.min(np.vstack([eef_pos, action_pos]), axis=0)
        max_pos = np.max(np.vstack([eef_pos, action_pos]), axis=0)

        # Calculate the range for each axis
        range_x = max_pos[0] - min_pos[0]
        range_y = max_pos[1] - min_pos[1]
        range_z = max_pos[2] - min_pos[2]

        # Ensure that all axes have the same range (so the plot is cubic)
        max_range = max(range_x, range_y, range_z)

        # Extend the boundaries by 10% in both directions for some padding
        padding = max_range * 0.1
        center_pos = (min_pos + max_pos) / 2  # Get the center of the data

        # Coerce the bounds to be a cube centered around the data
        min_pos = center_pos - max_range / 2 - padding
        max_pos = center_pos + max_range / 2 + padding

        # Create a 3D plot
        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=eef_pos[:, 0],
            y=eef_pos[:, 1],
            z=eef_pos[:, 2],
            mode='lines+markers',  # 'lines' for the trajectory, 'markers' for the points
            name='End-Effector Trajectory',
            line=dict(color='blue', width=4),
            marker=dict(size=2, color='blue')
        ))

        # Add trace for action positions (action_pos)
        fig.add_trace(go.Scatter3d(
            x=action_pos[:, 0],
            y=action_pos[:, 1],
            z=action_pos[:, 2],
            mode='lines+markers',
            name='Action Trajectory',
            line=dict(color='red', width=4),
            marker=dict(size=2, color='red')
        ))

        # Update layout for better visualization
        fig.update_layout(
            title="End-Effector and Action Trajectories",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                xaxis=dict(range=[min_pos[0], max_pos[0]]),
                yaxis=dict(range=[min_pos[1], max_pos[1]]),
                zaxis=dict(range=[min_pos[2], max_pos[2]]),
            ),
            margin=dict(l=0, r=0, b=0, t=40),  # Adjust margins for better layout
        )

        # Save the plot to an HTML file
        fig.write_html(plot_path)

    def flush(self):
        state_hist_path = self.path / f"state_action.npz"
        all_keys = self.states[0].keys()
        stacked_states = {k: np.stack([s[k] for s in self.states]) for k in all_keys}
        np.savez(state_hist_path, **stacked_states)
        self.write_trajectory_plot(stacked_states)

class PolicyInterface:
    """
    Interface for an ACMEPolicy running on some local port
    """

    def __init__(self, control_frequency: float, obs_history: int, rgb_keys: List[str], lowdim_keys: List[str],
                 frame_shape: List[int],
                 port: int):
        self._control_frequency = control_frequency
        self._obs_history = obs_history
        self._rgb_keys = rgb_keys
        self._lowdim_keys = lowdim_keys
        self._frame_shape = frame_shape
        self._port = port
        self._server_url = f'http://localhost:{self._port}'

    @property
    def obs_history_size(self):
        return self._obs_history

    @property
    def control_frequency(self):
        return self._control_frequency

    def coerce_frame_shape(self, frame: torch.Tensor):
        """
        Resize a batch of image tensors to match self._frame_shape.

        Input shape:  (N, H, W, C), dtype: uint8 or float
        Output shape: (N, C, H, W), dtype: uint8
        """
        target_shape = self._frame_shape  # (N, H, W, C)

        assert frame.ndim == 4, f"Expected 4D input (N, H, W, C), got {frame.shape}"
        assert frame.shape[-1] == 3, "Expected 3 channels (RGB)"

        n, h, w, c = frame.shape
        tc, th, tw = target_shape

        # Convert to float and normalize if needed
        if frame.dtype == torch.uint8:
            frame = frame.float() / 255.0
        else:
            frame = frame.float()

        # NHWC → NCHW
        frame = frame.permute(0, 3, 1, 2)

        # Resize if necessary
        if (h, w) != (th, tw):
            frame = F.interpolate(frame, size=(th, tw), mode='bilinear', align_corners=False)

        # Convert back to uint8
        frame = (frame * 255.0).clamp(0, 255).to(torch.uint8)

        return frame

    def __call__(self,
                 rgb_0: torch.Tensor,
                 rgb_1: torch.Tensor,
                 eef_pos: np.ndarray,
                 eef_quat: np.ndarray,
                 gripper_force: np.ndarray) -> torch.Tensor:
        """
        Send binary data using multipart/form-data for efficient transfer.
        """
        files = {}
        data = {}

        # Add RGB frames as binary data
        rgb_data = {"rgb_0": rgb_0, "rgb_1": rgb_1}

        for rgb_key, frames in rgb_data.items():
            buffer = io.BytesIO()
            torch.save(frames, buffer)
            buffer.seek(0)
            files[f"{rgb_key}"] = (
                f"{rgb_key}.pt",
                buffer,
                "application/octet-stream"
            )

        lowdim_data = {
            "eef_pos": eef_pos,
            "eef_quat": eef_quat,
            "gripper_force": gripper_force
        }

        lowdim_buffer = io.BytesIO()
        np.savez(lowdim_buffer, **lowdim_data)
        lowdim_buffer.seek(0)
        files["lowdim_data"] = (
            "lowdim_data.npz",
            lowdim_buffer,
            "application/octet-stream"
        )

        obs_steps = rgb_0.shape[1]
        data.update({
            "rgb_keys": ",".join(self._rgb_keys),
            "lowdim_keys": ",".join(self._lowdim_keys),
            "rgb_shape": ",".join(map(str, self._frame_shape)),
            "obs_steps": str(obs_steps)
        })

        try:
            resp = requests.post(
                f"{self._server_url}/predict",
                files=files,
                data=data,
                timeout=30
            )
            resp.raise_for_status()
            result = resp.json()
            return torch.tensor(result["action"])
        except Exception as err:
            logging.info(f"Error communicating with the server: {err}")
            raise err

class EvalRSI(RealSenseInterface):
    class _FrameGrabberThread(threading.Thread):
        def __init__(self, idx, pipe, callback, stop_event, cache_size=2):
            super().__init__()
            self.idx = idx
            self.pipe = pipe
            self.callback = callback
            self.stop_event = stop_event
            self._cache = deque(maxlen=cache_size)

        def run(self):
            while not self.stop_event.is_set():
                try:
                    fs = self.pipe.wait_for_frames(timeout_ms=5000)
                    color_frame = fs.get_color_frame()
                    # Convert frames to numpy arrays
                    color = torch.tensor(np.asanyarray(color_frame.get_data()))
                    self._cache.append(color)
                    self.callback(self.idx)
                except Exception as e:
                    print(f"Camera {self.idx} failed to grab frame: {e}")
                    traceback.print_exc()
            print(f"Stopping capture pipeline {self.idx}")
            self.pipe.stop()

        def get_obs(self):
            return torch.stack(list(self._cache))

    def _initialize_cameras(self):
        cameras = enumerate_devices()
        if not cameras:
            print("No RealSense devices detected – exiting.")
            return []

        print(f"Found {len(cameras)} camera(s):")
        for idx, (serial, product) in enumerate(cameras):
            print(f"   Camera {idx}: {serial}  ({product})")

        pipelines = []
        for idx, (serial, _) in enumerate(cameras):
            if idx not in self._obs_cam_idx:
                continue
            pipe, cfg = self.create_pipeline(serial, self._width, self._height, self._fps)
            pipelines.append((pipe, cfg))
        return pipelines

    def __init__(self, n_frames: int, width: int, height: int, fps: int, obs_cams: List[int], init=True):
        rs.log_to_console(min_severity=rs.log_severity.warn)
        self._n_frames = n_frames
        self._width = width
        self._height = height
        self._fps = fps
        self._obs_cam_idx = obs_cams
        if init:
            self._pipelines = self._initialize_cameras()
            self._stop_events = []
            self._threads = []
            self._start_indices = []
            self.frame_counts = {i: 0 for i in range(len(self._pipelines))}

    def get_rgb_obs(self) -> List[torch.Tensor]:
        return [t.get_obs() for t in self._threads]

    def start_capture(self, on_receive_frame: Callable = None):
        def _callback_wrapper(cap_idx):
            if on_receive_frame is not None:
                on_receive_frame(cap_idx)
            self._latest_rgb_frames = None
            self.frame_counts[cap_idx] += 1
            if self.frame_counts[cap_idx] >= self._n_frames:
                self.stop_capture(cap_idx)

        for idx, (pipe, cfg) in enumerate(self._pipelines):
            stop_event = threading.Event()
            pipe.start(cfg)
            t = self._FrameGrabberThread(idx, pipe, _callback_wrapper, stop_event)
            t.start()
            self._threads.append(t)
            self._stop_events.append(stop_event)

    def create_pipeline(self, serial: str, w: int, h: int, fps: int):
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        pipe = rs.pipeline()
        return pipe, cfg

def get_latest_ep_path(base_episodes_path: Path, prefix: str):
    ep_idxs = [int(x.stem.split("_")[-1]) for x in base_episodes_path.iterdir()]
    ep_idx = 0
    if len(ep_idxs) > 0:
        ep_idx = max(ep_idxs) + 1
    current_episode_name = f"{prefix}_{ep_idx}"
    ep_path = base_episodes_path / current_episode_name
    ep_path.mkdir(exist_ok=False)
    return ep_path

def listen_for_keypress(cancel_event):
    print("Press 'c' to cancel the episode.")
    while not cancel_event.is_set():
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key.lower() == 'c':
                cancel_event.set()
                print("\nEpisode cancelled by user (keypress 'x').")

def start_control_loop(policy: PolicyInterface, realsense: EvalRSI, nuc: NUCInterface, states):
    nuc.start()

    stop_event = threading.Event()
    pose_lock = threading.Lock()
    desired_eef_pos, desired_eef_rot = None, None

    def _loop_iter():
        nonlocal desired_eef_pos, desired_eef_rot, pose_lock
        frames: List[torch.Tensor] = realsense.get_rgb_obs()
        resized_frames = [policy.coerce_frame_shape(f) for f in frames]
        current_states = list(states)
        eef_pos = np.stack([s["ee_pos"] for s in current_states[-policy.obs_history_size:]])
        eef_rot = np.stack([s["ee_rot"] for s in current_states[-policy.obs_history_size:]])
        action = policy(
            rgb_0=resized_frames[0].unsqueeze(0),
            rgb_1=resized_frames[1].unsqueeze(0),
            eef_pos=np.expand_dims(eef_pos, 0),
            eef_quat=np.expand_dims(eef_rot, 0),
            gripper_force=np.zeros((1, policy.obs_history_size, 1))
        )[0]  # un-batch

        horizon_len = action.shape[0]

        home_eef_pos, home_eef_rot = nuc.pusht_home

        for i in range(horizon_len):
            desired_eef_pos = action[i, :3]
            desired_eef_rot = action[i, 3:7]
            # PUSH T FREEZES
            desired_eef_pos[-1] = home_eef_pos[-1]
            desired_eef_pos = desired_eef_pos.to(torch.float64)
            desired_eef_rot = torch.from_numpy(home_eef_rot)
            print(desired_eef_pos, desired_eef_rot)
            nuc.send_control_tensor(desired_eef_pos, desired_eef_rot, None)
            time.sleep(1.0 / (policy.control_frequency * horizon_len))

    def _loop_runner():
        while not stop_event.is_set():
            _loop_iter()

    loop_thread = threading.Thread(target=_loop_runner, daemon=True)
    loop_thread.start()

    def stop_loop():
        stop_event.set()
        loop_thread.join()

    def get_action():
        nonlocal desired_eef_pos, desired_eef_rot, pose_lock
        with pose_lock:
            assert desired_eef_rot is not None and desired_eef_pos is not None
            return desired_eef_pos.copy(), desired_eef_rot.copy()

    return stop_loop, get_action

@hydra.main(config_path="config", config_name="eval")
def main(cfg: DictConfig):
    nuc = NUCInterface(**cfg.nuc)
    policy = PolicyInterface(**cfg.policy)

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
            with EvalRSI(**cfg.realsense) as rsi:
                writer = EvalWriter(path=ep_path, **cfg.writer)
                nuc.reset()

                def on_receive_frame(cap_idx):
                    if cap_idx == 0:
                        c_state = nuc.get_robot_state()
                        c_state.update(dict(
                            action=nuc.get_desired_ee_pose()
                        ))
                        writer.states.append(c_state)

                rsi.start_capture(on_receive_frame)
                print("Waiting for realsense caches to fill")
                time.sleep(2.0)

                stop_control, get_action = start_control_loop(policy, rsi, nuc, writer.states)

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
            ep_idx += 1
        except Exception as e:
            print(e)
            traceback.print_exc()
        finally:
            ep_control_msg = "1: Record Success \n2: Record Failure\n0: Delete this recording.\nx: Exit\nz: Next episode"
            ep_control_cmd = str(input(ep_control_msg))
            while ep_control_cmd not in ["x", "z"]:
                ep_control_cmd = str(input(ep_control_msg))
                if ep_control_cmd == "1":
                    successes.append(True)
                elif ep_control_cmd == "2":
                    successes.append(False)
                elif ep_control_cmd == "0":
                    if ep_path and ep_path.exists():
                        shutil.rmtree(ep_path)
                elif ep_control_cmd == "x":
                    should_exit = True
                elif ep_control_cmd == "z":
                    break

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
