"""Live matplotlib visualization of end-effector tracking error during eval.

A non-blocking matplotlib window that plots the per-axis position error
(commanded EE pose minus measured EE pose) over time. Used to spot impedance
saturation, lag, or controller instability while a policy is running.
Constructed and ticked from the main thread.
"""
from __future__ import annotations

import numpy as np


class LiveControlErrorPlotter:
    def __init__(self, title: str = "EEF control error (live)", window: int = 600):
        import matplotlib.pyplot as plt  # imported lazily so headless eval still works
        self._plt = plt
        self._window = window
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.set_xlabel("sample")
        self.ax.set_ylabel("commanded − measured (m)")
        self.ax.set_title(title)
        self.ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
        (self._x_line,) = self.ax.plot([], [], color="tab:red", linewidth=1.2, label="x")
        (self._y_line,) = self.ax.plot([], [], color="tab:green", linewidth=1.2, label="y")
        (self._z_line,) = self.ax.plot([], [], color="tab:blue", linewidth=1.2, label="z")
        (self._norm_line,) = self.ax.plot(
            [], [], color="black", linewidth=1.8, alpha=0.7, label="‖err‖"
        )
        self.ax.legend(loc="upper right", ncols=4, fontsize=8)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        plt.show(block=False)

    def update(self, ee_pos: np.ndarray, desired_ee_pos: np.ndarray) -> None:
        """Plot the trailing ``window`` samples of (desired − measured) per axis.

        Both inputs are expected to be (N, 3) and aligned sample-for-sample.
        Mismatched lengths are truncated to the shorter tail.
        """
        ee = np.asarray(ee_pos).reshape(-1, 3) if ee_pos is not None else np.empty((0, 3))
        des = np.asarray(desired_ee_pos).reshape(-1, 3) if desired_ee_pos is not None else np.empty((0, 3))
        n = min(ee.shape[0], des.shape[0])
        if n == 0:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            return

        err = des[-n:] - ee[-n:]
        if n > self._window:
            err = err[-self._window:]
            n = self._window
        norm = np.linalg.norm(err, axis=1)

        # x-axis is the absolute sample index of the trailing window, so the
        # view scrolls with the episode rather than rebasing to 0 each tick.
        end = ee.shape[0]
        idx = np.arange(end - n, end)

        self._x_line.set_data(idx, err[:, 0])
        self._y_line.set_data(idx, err[:, 1])
        self._z_line.set_data(idx, err[:, 2])
        self._norm_line.set_data(idx, norm)

        self.ax.set_xlim(idx[0], max(idx[-1], idx[0] + 1))
        ymax = max(float(np.abs(err).max()), float(norm.max()), 1e-3) * 1.15
        self.ax.set_ylim(-ymax, ymax)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        self._plt.close(self.fig)
