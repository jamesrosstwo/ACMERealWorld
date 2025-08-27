import io
import logging
import shutil
import traceback
import time
from pathlib import Path
from typing import Callable, List, Tuple

import hydra
import numpy as np
import requests
import torch
from omegaconf import DictConfig

from client.nuc import NUCInterface
from client.realsense import RealSenseInterface
from client.teleop import GELLOInterface
from client.write import ACMEWriter
import pyrealsense2 as rs
import threading


class PolicyInterface:
    """
    Interface for an ACMEPolicy running on some local port
    """

    def __init__(self, rgb_keys: List[str], lowdim_keys: List[str], frame_size: List[int], port: int):
        self._rgb_keys = rgb_keys
        self._lowdim_keys = lowdim_keys
        self._frame_size = frame_size
        self._port = port
        self._server_url = f'http://localhost:{self._port}'
        self._torch_device = torch.device('cuda')

    def __call__(self,
                 rgb_0: torch.Tensor,
                 rgb_1: torch.Tensor,
                 eef_pos: torch.Tensor,
                 eef_quat: torch.Tensor,
                 gripper_force: torch.Tensor) -> torch.Tensor:
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
            "eef_pos": eef_pos.cpu().numpy(),
            "eef_quat": eef_quat.cpu().numpy(),
            "gripper_force": gripper_force.cpu().numpy()
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
            return torch.tensor(result["action"], device=self._torch_device)
        except Exception as err:
            logging.info(f"Error communicating with the server: {err}")
            raise err


class EvalRSI(RealSenseInterface):
    class _FrameGrabberThread(threading.Thread):
        def __init__(self, idx, pipe, callback, stop_event):
            super().__init__()
            self.idx = idx
            self.pipe = pipe
            self.callback = callback
            self.stop_event = stop_event

        def run(self):
            while not self.stop_event.is_set():
                try:
                    fs = self.pipe.wait_for_frames(timeout_ms=5000)
                    color_frame = fs.get_color_frame()
                    # Convert frames to numpy arrays
                    color = np.asanyarray(color_frame.get_data())
                    self.callback(self.idx, color)
                except Exception as e:
                    print(f"Camera {self.idx} failed to grab frame: {e}")
                    traceback.print_exc()
            print(f"Stopping capture pipeline {self.idx}")
            self.pipe.stop()

    def __init__(self, path: Path, n_frames: int, width: int, height: int, fps: int, init=True):
        rs.log_to_console(min_severity=rs.log_severity.warn)
        self._path = path
        self._n_frames = n_frames
        self._width = width
        self._height = height
        self._fps = fps
        self._latest_rgb_frames: Tuple[torch.Tensor, torch.Tensor] = None
        if init:
            self._pipelines, self._recording_bagpaths = self._initialize_cameras()
            self._stop_events = []
            self._threads = []
            self._start_indices = []
            self.frame_counts = {i: 0 for i in range(len(self._pipelines))}

    def start_capture(self):
        def _callback_wrapper(cap_idx, fs):
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
        tmp_path = str(self._path / f"{serial}.bag")
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        pipe = rs.pipeline()
        return pipe, cfg, tmp_path


def get_latest_ep_path(base_episodes_path: Path, prefix: str):
    ep_idxs = [int(x.stem.split("_")[-1]) for x in base_episodes_path.iterdir()]
    ep_idx = 0
    if len(ep_idxs) > 0:
        ep_idx = max(ep_idxs) + 1
    current_episode_name = f"{prefix}_{ep_idx}"
    ep_path = base_episodes_path / current_episode_name
    ep_path.mkdir(exist_ok=False)
    return ep_path


def start_control_loop(policy: PolicyInterface, realsense: EvalRSI, nuc: NUCInterface):
    state = nuc.get_robot_state()
    nuc.start()

    stop_event = threading.Event()
    pose_lock = threading.Lock()
    latest_eef_pos, latest_eef_rot = None, None

    def _loop_iter():
        nonlocal latest_eef_pos, latest_eef_rot, pose_lock

        frames: Tuple[torch.Tensor] = realsense.get()
        eef_pos, eef_rot = policy(
            rgb_0=frames[0],
            rgb_1=frames[1],
            eef_pos=state["eef_pos"],
            eef_quat=state["eef_rot"],
            gripper_force=torch.zeros((1,))
        )

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
    base_ep_path = Path(cfg.episodes_path)
    base_ep_path.mkdir(exist_ok=True, parents=True)
    policy = PolicyInterface(**cfg.policy)
    while True:
        try:
            with EvalRSI(**cfg.realsense) as rsi:
                nuc.reset()
                stop_control, get_desired_eef_pose = start_control_loop(policy, rsi, nuc)

                def get_action():
                    return np.concatenate(get_desired_eef_pose())

                rsi.start_capture()
                for i in range(rsi.n_cameras):
                    rsi.frame_counts[i] = 0
                while any([c < cfg.max_episode_timesteps for c in rsi.frame_counts.values()]):
                    time.sleep(2.0)
                    print("Episode progress:",
                          np.array(list(rsi.frame_counts.values())) / cfg.max_episode_timesteps)
            stop_control()
        except Exception as e:
            print(e)
            traceback.print_exc()
        finally:
            ep_control_msg = "1: Continue and start the next recording\n0: to delete this recording.\nx: Exit"
            ep_control_cmd = str(input(ep_control_msg))
            while ep_control_cmd not in ["1", "x"]:
                ep_control_cmd = str(input(ep_control_msg))
            if ep_control_cmd == "x":
                break


if __name__ == "__main__":
    main()
