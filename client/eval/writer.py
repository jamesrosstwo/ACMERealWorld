from pathlib import Path
from typing import Dict, List
import numpy as np
import plotly.graph_objects as go
import torch


def states_fig(stacked_states) -> go.Figure:
    # Extract positions from the stacked states
    eef_pos = stacked_states["ee_pos"]  # shape: (N, 3)
    action_pos = stacked_states["action"][:, :3]  # shape: (N, 3)

    # Get bounds for the cube
    min_pos = np.min(np.vstack([eef_pos, action_pos]), axis=0)
    max_pos = np.max(np.vstack([eef_pos, action_pos]), axis=0)

    # Calculate the range for each axis
    range_x = max_pos[0] - min_pos[0]
    range_y = max_pos[1] - min_pos[1]
    range_z = max_pos[2] - min_pos[2]
    max_range = max(range_x, range_y, range_z)

    # Add padding
    padding = max_range * 0.1
    center_pos = (min_pos + max_pos) / 2
    min_pos = center_pos - max_range / 2 - padding
    max_pos = center_pos + max_range / 2 + padding

    # Create a 3D plot
    fig = go.Figure()

    # End-effector trajectory (solid blue)
    fig.add_trace(go.Scatter3d(
        x=eef_pos[:, 0],
        y=eef_pos[:, 1],
        z=eef_pos[:, 2],
        mode='lines+markers',
        name='End-Effector Trajectory',
        line=dict(color='blue', width=4),
        marker=dict(size=2, color='blue')
    ))

    # Action trajectory with gradient from black to red
    num_points = action_pos.shape[0]
    gradient_values = np.linspace(0, 1, num_points)  # Used for color scaling

    fig.add_trace(go.Scatter3d(
        x=action_pos[:, 0],
        y=action_pos[:, 1],
        z=action_pos[:, 2],
        mode='lines+markers',
        name='Action Trajectory',
        line=dict(
            color='red',
            width=4
        ),
        marker=dict(
            size=3,
            color=gradient_values,  # Gradient value for each point
            colorscale=[[0, 'black'], [1, 'red']],
            cmin=0,
            cmax=1,
            showscale=False  # Hide the color bar
        )
    ))

    # Update layout
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
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig

def inference_fig(all_inferences: Dict[str, np.ndarray]) -> go.Figure:

    # Extract positions from the stacked states
    eef_pos = all_inferences["ee_pos"] # N, obs_hist_size, 3
    desired_pos = all_inferences["desired_ee_pos"] # N, action_horizon_size, 3

    # Get bounds for the cube
    min_pos = np.min(np.vstack([eef_pos.reshape(-1, 3), desired_pos.reshape(-1, 3)]), axis=0)
    max_pos = np.max(np.vstack([eef_pos.reshape(-1, 3), desired_pos.reshape(-1, 3)]), axis=0)

    # Calculate the range for each axis
    range_x = max_pos[0] - min_pos[0]
    range_y = max_pos[1] - min_pos[1]
    range_z = max_pos[2] - min_pos[2]
    max_range = max(range_x, range_y, range_z)

    # Add padding
    padding = max_range * 0.1
    center_pos = (min_pos + max_pos) / 2
    min_pos = center_pos - max_range / 2 - padding
    max_pos = center_pos + max_range / 2 + padding

    # Create a 3D plot
    fig = go.Figure()

    obs_col = 0
    action_col = 1 # on colorscale
    for trace_idx, (obs_trace, action_trace) in enumerate(zip(eef_pos, desired_pos)):
        obs_colors = np.full(obs_trace.shape[0], obs_col)
        action_colors = np.full(action_trace.shape[0], action_col)

        points = np.vstack([obs_trace, action_trace])
        points[:, 2] = 0
        colors = np.concatenate([obs_colors, action_colors])

        # End-effector trajectory (solid blue)
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='lines+markers',
            name=f'trace_{trace_idx}',
            line=dict(color='black', width=3),
            marker=dict(
                size=4,
                color=colors,
                colorscale="Viridis"
            )
        ))

    # Update layout
    fig.update_layout(
        title="Inference plots",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(range=[min_pos[0], max_pos[0]]),
            yaxis=dict(range=[min_pos[1], max_pos[1]]),
            zaxis=dict(range=[min_pos[2], max_pos[2]]),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


class EvalWriter:
    def __init__(self, path: Path, max_episode_len: int):
        self.path = path
        assert self.path.exists()
        self._max_episode_len = max_episode_len
        self.states = []
        self.inferences = []
        self._state_keys = ["qpos", "qvel", "ee_pos", "ee_rot", "gripper_force", "action"]
        self._inference_keys = ["ee_pos", "ee_quat", "gripper_force", "desired_ee_pos", "desired_ee_quat", "desired_gripper_force"]


    def on_state_update(self, new_state):
        assert all([k in new_state for k in self._state_keys])
        self.states.append(new_state)

    def on_inference(self, **inference: np.ndarray):
        assert all([k in inference for k in self._inference_keys])
        self.inferences.append(inference)

    @property
    def episode_path(self):
        return self.path

    def write_trajectory_plot(self, plot_path: Path):
        state_hist_path = self.path / f"state_action.npz"
        all_keys = self.states[0].keys()
        stacked_states = {k: np.stack([s[k] for s in self.states]) for k in all_keys}
        np.savez(state_hist_path, **stacked_states)
        fig = states_fig(stacked_states)
        fig.write_html(plot_path)


    def write_inference_plot(self, plot_path: Path):
        stacked_infs = {k: np.stack([s[k] for s in self.inferences]) for k in self._inference_keys}
        fig = inference_fig(stacked_infs)
        fig.write_html(plot_path)

    def flush(self):
        self.write_trajectory_plot(self.path / "states.html")
        self.write_inference_plot(self.path / "inference.html")


