import time
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from client.eval.policy import EvalPolicyInterface
import zarr

from client.eval.writer import EvalWriter

DO_CHANNELS_FIRST=True
DO_BGR = False
RAND_RGB = False
RAND_EEF_POS = False

def get_obs(root, idx: int, obs_hist: int):
    rgb_0 = torch.from_numpy(root.data.rgb_0[idx - obs_hist: idx])
    rgb_1 = torch.from_numpy(root.data.rgb_1[idx - obs_hist: idx])

    if DO_BGR:
        rgb_0 = rgb_0[..., [2, 1, 0]]
        rgb_1 = rgb_1[..., [2, 1, 0]]
    if DO_CHANNELS_FIRST:
        rgb_0 = rgb_0.permute(0, 3, 1, 2)
        rgb_1 = rgb_1.permute(0, 3, 1, 2)

    if RAND_RGB:
        rgb_0 = (torch.rand_like(rgb_0.to(torch.float32)) * 255).to(torch.uint8)
        rgb_1 = (torch.rand_like(rgb_1.to(torch.float32)) * 255).to(torch.uint8)

    eef_pos = root.data.eef_pos[idx - obs_hist: idx][None]

    if RAND_EEF_POS:
        eef_pos = np.random.uniform(low=-1, high=1, size=eef_pos.shape)
        eef_pos[..., -1] = 0


    obs = dict(
        eef_pos=eef_pos,
        eef_quat=root.data.eef_quat[idx - obs_hist: idx][None],
        gripper_force=root.data.gripper_force[idx - obs_hist: idx][None],
        rgb_0=rgb_0.unsqueeze(0),
        rgb_1=rgb_1.unsqueeze(0),
    )
    return obs


def error(desired_eef_pos, desired_eef_rot, desired_gripper_pos, gt_action) -> Tuple[float, float, float]:
    pos_mse = np.mean((gt_action[:, :2] - desired_eef_pos[:, :2].numpy()) ** 2)
    return np.sqrt(pos_mse), 0, 0


def plot_inferences(tracking):
    import plotly.graph_objects as go
    fig = go.Figure()

    obs_col = 1
    action_col = 0 # on colorscale
    def _add_trace(obs_pos, action_pos, trace_idx, suffix: str):
        obs_colors = np.full(obs_pos.shape[0], obs_col)
        action_colors = np.full(action_pos.shape[0], action_col)

        points = np.vstack([obs_pos, action_pos])
        points[:, 2] = 0
        colors = np.concatenate([obs_colors, action_colors])

        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='lines+markers',
            name=f'trace_{trace_idx}_{suffix}',
            line=dict(
                width=3,
                color="yellow" if suffix == "gt" else "purple",
            ),
            marker=dict(
                size=2,
                opacity=0.5,
                color="black",
            )
        ))
    for idx, trace in enumerate(tracking):
        obs_trace = trace["eef_pos"][0]
        gt_action_trace = trace["gt_action"][:, :3]
        action_trace = trace["desired_eef_pos"]

        _add_trace(obs_trace, gt_action_trace, idx, "gt")
        _add_trace(obs_trace, action_trace, idx, "inferred")



    # Update layout
    fig.update_layout(
        title="Inference plots",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


@hydra.main(config_path="../../config", config_name="offline_eval")
def main(cfg: DictConfig) -> None:
    histsize = cfg.policy.obs_history
    policy = EvalPolicyInterface(**cfg.policy)
    root = zarr.open_group(cfg.dtst_path, "r")
    data_len = root.data.eef_pos.shape[0]
    sample_horizon = cfg.sample_horizon

    timestamp = int(time.time())
    out_path = Path(f"/home/rvl_root/Desktop/ACMERealWorld/outputs/offline_eval/{timestamp}")
    out_path.mkdir(parents=True, exist_ok=True)
    print(out_path.resolve())



    sum_error = 0
    n_iters = 0
    tracking = []

    for obs_idx in range(histsize, data_len, 4)[:100]:
        obs = get_obs(root, obs_idx, histsize)
        desired_eef_pos, desired_eef_rot, desired_gripper_pos = policy(**obs)
        gt_action = root.data.action[obs_idx:obs_idx + sample_horizon]
        pos_error, rot_error, gripper_error = error(desired_eef_pos, desired_eef_rot, desired_gripper_pos,
                                                    gt_action)


        obs.update(dict(gt_action=gt_action, desired_eef_pos=desired_eef_pos))
        tracking.append(obs)

        sum_error += pos_error
        n_iters += 1
        print("Mean error:", sum_error / n_iters * 100, "cm")
    fig = plot_inferences(tracking)
    fig.write_html(out_path / "inference.html")


if __name__ == "__main__":
    main()
