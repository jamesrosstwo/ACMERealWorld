import io
import logging

import requests
import torch
import torch.nn.functional as F
from typing import List, Tuple

import numpy as np


class EvalPolicyInterface:
    """
    Interface for an ACMEPolicy running on some local port
    """

    def __init__(self, control_frequency: float,
                 obs_history: int,
                 action_horizon: int,
                 action_start_offset: int,
                 rgb_keys: List[str],
                 lowdim_keys: List[str],
                 frame_shape: List[int],
                 port: int,
                 delta_actions: bool = False):
        self._control_frequency = control_frequency
        self._obs_history = obs_history
        self._action_horizon = action_horizon
        self._offset = action_start_offset
        self._rgb_keys = rgb_keys
        self._lowdim_keys = lowdim_keys
        self._frame_shape = frame_shape
        self._port = port
        self._server_url = f'http://localhost:{self._port}'
        self._delta_actions = delta_actions

    @property
    def obs_history_size(self):
        return self._obs_history

    @property
    def control_frequency(self):
        return self._control_frequency

    def preprocess_frame(self, frame: torch.Tensor):
        """
        Resize a batch of image tensors to match self._frame_shape.

        Input shape:  (N, H, W, C), dtype: uint8 or float
        Output shape: (N, C, H, W), dtype: uint8
        """
        target_shape = self._frame_shape  # Expected (C, H, W)

        assert frame.ndim == 4, f"Expected 4D input (N, H, W, C), got {frame.shape}"
        assert frame.shape[-1] == 3, "Expected 3 channels (BGR or RGB)"

        n, h, w, c = frame.shape
        tc, th, tw = target_shape

        # Convert to float and normalize if needed
        if frame.dtype == torch.uint8:
            frame = frame.float() / 255.0
        else:
            frame = frame.float()

        # BGR → RGB (flip channel 0 and 2)
        frame = frame[..., [2, 1, 0]]

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
                 gripper_force: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Send binary data using multipart/form-data for efficient transfer.
        """
        data = {}

        files = dict()
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
            action = torch.tensor(result["action"])
            if self._delta_actions:
                cumulative = torch.cumsum(action, dim=1)
                cumulative[:, :, :3] += eef_pos[:, -1]
                cumulative[:, :, 3:7] += eef_quat[:, -1]
                raise ValueError()
                return cumulative

            # un-batch
            o = self._offset
            desired_eef_pos = action[0, o:self._action_horizon + o, :3]
            desired_eef_rot = action[0, o:self._action_horizon + o, 3:7]
            desired_gripper_force = torch.zeros((desired_eef_pos.shape[0], 1))
            return desired_eef_pos, desired_eef_rot, desired_gripper_force
        except Exception as err:
            logging.info(f"Error communicating with the server: {err}")
            raise err