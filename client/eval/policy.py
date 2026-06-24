"""Policy inference server client.

:class:`EvalPolicyInterface` communicates with an HTTP policy server to obtain
robot actions from camera observations and low-dimensional state. Handles frame
preprocessing (BGR to RGB, aspect-preserving resize-with-pad, channel reordering)
and serialization of observation tensors for network transfer.
"""
import io
import logging
from pathlib import Path

import cv2
import requests
import torch
import torch.nn.functional as F
from typing import List, Tuple

import numpy as np
import panda_py
from scipy.spatial.transform import Rotation


class EvalPolicyInterface:
    """
    Interface for an ACMEPolicy running on a remote or local HTTP server
    """

    def __init__(self, control_frequency: float,
                 obs_history: int,
                 action_horizon: int,
                 action_start_offset: int,
                 rgb_keys: List[str],
                 lowdim_keys: List[str],
                 frame_shape: List[int],
                 port: int,
                 delta_actions: bool = False,
                 action_type: str = "cartesian",
                 host: str = "localhost",
                 prompt: str = "",
                 antialias: bool = True,
                 resize_with_pad: bool = True,
                 dump_frames_dir: str = ""):
        if action_type not in ("qpos", "cartesian"):
            raise ValueError(f"action_type must be 'qpos' or 'cartesian', got {action_type!r}")
        self._control_frequency = control_frequency
        self._obs_history = obs_history
        self._action_horizon = action_horizon
        self._offset = action_start_offset
        self._rgb_keys = rgb_keys
        self._lowdim_keys = lowdim_keys
        self._frame_shape = frame_shape
        self._port = port
        self._host = host
        self._server_url = f'http://{self._host}:{self._port}'
        self._delta_actions = delta_actions
        self._action_type = action_type
        self._prompt = prompt
        self._antialias = antialias
        self._resize_with_pad = resize_with_pad
        # Frame-tracking diagnostics. When dump_frames_dir is set, every frame
        # actually sent to the server is written to disk as a PNG. The geometry
        # (padding / letterboxing) is logged once on the first preprocessed
        # frame regardless, so each run records what it sent.
        self._dump_frames_dir = Path(dump_frames_dir) if dump_frames_dir else None
        if self._dump_frames_dir is not None:
            self._dump_frames_dir.mkdir(parents=True, exist_ok=True)
        self._dump_idx = 0
        self._logged_geom = False

    @property
    def obs_history_size(self):
        return self._obs_history

    @property
    def control_frequency(self):
        return self._control_frequency

    def preprocess_frame(self, frame: torch.Tensor):
        """
        Resize a batch of image tensors to match self._frame_shape.

        When ``resize_with_pad`` is true, aspect ratio is preserved by
        zero-padding (matches openpi's ``resize_with_pad``) — required for
        openpi/π0-style VLAs. When false, the frame is stretched directly to
        the target shape; this is what older diffusion policies were trained
        on. ``antialias`` toggles the PIL-style prefilter inside the bilinear
        downsample — turn it off to reproduce pre-VLA preprocessing exactly.

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

        if (h, w) != (th, tw):
            if self._resize_with_pad:
                # Scale down so the longer side fits the target, preserving aspect ratio.
                ratio = max(w / tw, h / th)
                rh = int(h / ratio)
                rw = int(w / ratio)
                resized = F.interpolate(frame, size=(rh, rw), mode='bilinear',
                                        align_corners=False, antialias=self._antialias)
                padded = torch.zeros((n, tc, th, tw), dtype=resized.dtype, device=resized.device)
                pad_top = max(0, (th - rh) // 2)
                pad_left = max(0, (tw - rw) // 2)
                padded[:, :, pad_top:pad_top + rh, pad_left:pad_left + rw] = resized
                frame = padded
                if not self._logged_geom:
                    bars = "top/bottom" if pad_top > 0 else ("left/right" if pad_left > 0 else "none")
                    print(f"[frame-geom] resize_with_pad=True src={h}x{w} -> "
                          f"resized={rh}x{rw} -> canvas={th}x{tw}; "
                          f"pad_top={pad_top}px pad_left={pad_left}px "
                          f"({'LETTERBOXED ' + bars if (pad_top or pad_left) else 'no padding'})")
                    self._logged_geom = True
            else:
                frame = F.interpolate(frame, size=(th, tw), mode='bilinear',
                                      align_corners=False, antialias=self._antialias)
                if not self._logged_geom:
                    print(f"[frame-geom] resize_with_pad=False src={h}x{w} -> "
                          f"stretched to {th}x{tw} (aspect not preserved, no padding)")
                    self._logged_geom = True
        elif not self._logged_geom:
            print(f"[frame-geom] src already {th}x{tw}; no resize, no padding")
            self._logged_geom = True

        # Convert back to uint8
        frame = (frame * 255.0).clamp(0, 255).to(torch.uint8)

        return frame

    def _dump_sent_frames(self, rgb_data):
        """Write the exact frames being sent to the server to disk as PNGs.

        Each value is the post-preprocess tensor (uint8, RGB) of shape
        (1, obs_steps, C, H, W) — i.e. literally what gets serialized into the
        request. Files are named ``<call>_<rgb_key>_t<step>.png`` so you can
        diff successive inference calls and confirm padding/letterboxing
        visually (black bars survive the round-trip to disk).
        """
        for rgb_key, frames in rgb_data.items():
            arr = frames.detach().cpu()
            if arr.dtype != torch.uint8:
                arr = arr.clamp(0, 255).to(torch.uint8)
            arr = arr.numpy()
            # (1, T, C, H, W) -> iterate over T
            batch = arr[0]
            for t in range(batch.shape[0]):
                rgb = batch[t].transpose(1, 2, 0)  # CHW -> HWC, RGB
                bgr = rgb[..., ::-1]               # RGB -> BGR for cv2
                safe_key = rgb_key.replace("/", "_")
                fname = self._dump_frames_dir / f"{self._dump_idx:05d}_{safe_key}_t{t}.png"
                cv2.imwrite(str(fname), bgr)
        self._dump_idx += 1

    @staticmethod
    def _fk_horizon(qpos_horizon: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """FK a (T, 7) joint trajectory into (T, 3) EEF position and (T, 4) xyzw quaternion."""
        q_np = qpos_horizon.detach().cpu().numpy().astype(np.float64)
        T = q_np.shape[0]
        eef_pos = np.zeros((T, 3), dtype=np.float64)
        eef_rot = np.zeros((T, 4), dtype=np.float64)
        for i in range(T):
            mat = np.array(panda_py.fk(q_np[i].reshape(7, 1))).reshape(4, 4)
            eef_pos[i] = mat[:3, 3]
            eef_rot[i] = Rotation.from_matrix(mat[:3, :3]).as_quat()
        return torch.from_numpy(eef_pos), torch.from_numpy(eef_rot)

    def __call__(self,
                 rgbs: List[torch.Tensor],
                 eef_pos: np.ndarray,
                 eef_quat: np.ndarray,
                 gripper_force: np.ndarray,
                 qpos: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Send binary data using multipart/form-data for efficient transfer.

        ``rgbs`` is one frame tensor per camera, positionally matching
        ``self._rgb_keys`` (entry 0 -> rgb_0, entry 1 -> rgb_1, ...). The count
        must equal ``len(self._rgb_keys)`` so we never silently drop a camera
        the server expects.

        Returns (desired_eef_pos, desired_eef_rot, desired_gripper_force, desired_qpos)
        regardless of action_type — the qpos vs cartesian wire format only
        determines how the 8-dim action returned by the server is parsed; both
        downstream consumers (writer, settle, dispatch) get the same shape.
        """
        if len(rgbs) != len(self._rgb_keys):
            raise ValueError(
                f"got {len(rgbs)} camera frames but rgb_keys has "
                f"{len(self._rgb_keys)} entries ({self._rgb_keys}); "
                "obs_cams and rgb_keys must line up positionally"
            )
        data = {}

        files = dict()
        # Add RGB frames as binary data, keyed positionally by rgb_keys.
        rgb_data = dict(zip(self._rgb_keys, rgbs))

        if self._dump_frames_dir is not None:
            self._dump_sent_frames(rgb_data)

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
            "gripper_force": gripper_force,
            "qpos": qpos,
        }

        lowdim_buffer = io.BytesIO()
        np.savez(lowdim_buffer, **lowdim_data)
        lowdim_buffer.seek(0)
        files["lowdim_data"] = (
            "lowdim_data.npz",
            lowdim_buffer,
            "application/octet-stream"
        )

        obs_steps = rgbs[0].shape[1]
        data.update({
            "rgb_keys": ",".join(self._rgb_keys),
            "lowdim_keys": ",".join(self._lowdim_keys),
            "rgb_shape": ",".join(map(str, self._frame_shape)),
            "obs_steps": str(obs_steps),
            "prompt": self._prompt,
        })

        try:
            resp = requests.post(
                f"{self._server_url}/predict",
                files=files,
                data=data,
                timeout=30
            )
            if not resp.ok:
                # Surface what we sent + the server's own explanation. Without
                # this, raise_for_status() only reports the status code and hides
                # the response body where the server says why it rejected us.
                raise RuntimeError(
                    f"policy server returned {resp.status_code} from /predict.\n"
                    f"  sent file fields: {list(files.keys())}\n"
                    f"  sent rgb_keys:    {data['rgb_keys']!r}\n"
                    f"  sent lowdim_keys: {data['lowdim_keys']!r}\n"
                    f"  server response:  {resp.text[:2000]}"
                )
            result = resp.json()
            action = torch.tensor(result["action"])
            if self._delta_actions:
                cumulative = torch.cumsum(action, dim=1)
                cumulative[:, :, :3] += eef_pos[:, -1]
                cumulative[:, :, 3:7] += eef_quat[:, -1]
                return cumulative

            # un-batch. Server always returns 8 dims per step; the meaning
            # depends on action_type. In both modes the return tuple is the
            # same shape so downstream code (writer / settle / dispatch)
            # doesn't branch.
            o = self._offset
            window = action[0, o:self._action_horizon + o]
            if self._action_type == "cartesian":
                # Layout: [px, py, pz, qx, qy, qz, qw, gripper] per step.
                desired_eef_pos = window[:, :3]
                desired_eef_rot = window[:, 3:7]
                desired_gripper_force = window[:, 7]
                # Placeholder: broadcast latest observed qpos across horizon.
                # Used only by writer / diagnostics; cartesian dispatch never
                # consults desired_qpos for control.
                q_obs = torch.as_tensor(qpos[0, -1], dtype=torch.float64).reshape(1, 7)
                desired_qpos = q_obs.expand(desired_eef_pos.shape[0], 7).clone()
            else:
                # Layout: [qpos(7), gripper(1)] per step.
                desired_qpos = window[:, :7]
                desired_gripper_force = window[:, 7]
                # Derive EEF trajectory via FK so safety / writer / settle
                # still have a Cartesian view of the commanded motion.
                desired_eef_pos, desired_eef_rot = self._fk_horizon(desired_qpos)
            return desired_eef_pos, desired_eef_rot, desired_gripper_force, desired_qpos
        except Exception as err:
            logging.info(f"Error communicating with the server: {err}")
            raise err