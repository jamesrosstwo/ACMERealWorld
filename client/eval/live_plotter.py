"""Live 3D matplotlib visualization of the policy action trace during eval.

A non-blocking matplotlib window that plots the actual end-effector path
alongside the most recent policy-issued action horizon. Intended to be
constructed and ticked from the main thread.
"""
from __future__ import annotations

import numpy as np


class LiveActionPlotter:
    def __init__(self, title: str = "EEF trajectory (live)"):
        import matplotlib.pyplot as plt  # imported lazily so headless eval still works
        self._plt = plt
        plt.ion()
        self.fig = plt.figure(figsize=(7, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title(title)
        (self._eef_line,) = self.ax.plot([], [], [], color="tab:blue", linewidth=2, label="EEF")
        (self._action_line,) = self.ax.plot(
            [], [], [], color="tab:red", linewidth=2, marker="o", markersize=3, label="Action trace"
        )
        self.ax.legend(loc="upper left")
        self.fig.canvas.draw()
        plt.show(block=False)

    def update(self, eef_pos: np.ndarray, action_pos: np.ndarray) -> None:
        eef_pos = np.asarray(eef_pos).reshape(-1, 3) if eef_pos is not None else np.empty((0, 3))
        action_pos = np.asarray(action_pos).reshape(-1, 3) if action_pos is not None else np.empty((0, 3))

        if eef_pos.shape[0]:
            self._eef_line.set_data(eef_pos[:, 0], eef_pos[:, 1])
            self._eef_line.set_3d_properties(eef_pos[:, 2])
        if action_pos.shape[0]:
            self._action_line.set_data(action_pos[:, 0], action_pos[:, 1])
            self._action_line.set_3d_properties(action_pos[:, 2])

        pts = [p for p in (eef_pos, action_pos) if p.shape[0]]
        if pts:
            stacked = np.vstack(pts)
            mn, mx = stacked.min(axis=0), stacked.max(axis=0)
            ctr = (mn + mx) / 2
            half = max((mx - mn).max() / 2, 0.05) * 1.2
            self.ax.set_xlim(ctr[0] - half, ctr[0] + half)
            self.ax.set_ylim(ctr[1] - half, ctr[1] + half)
            self.ax.set_zlim(ctr[2] - half, ctr[2] + half)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        self._plt.close(self.fig)
