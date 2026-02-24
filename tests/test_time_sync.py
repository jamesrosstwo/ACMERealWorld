"""Unit tests for the time-synchronization functions in client.collect.write."""
import numpy as np
import pytest

from client.collect.write import (
    align_frames_to_reference,
    compute_sync_window,
    nearest_neighbor_indices,
    resample_timeseries,
)


# ---------------------------------------------------------------------------
# nearest_neighbor_indices
# ---------------------------------------------------------------------------

class TestNearestNeighborIndices:
    def test_exact_match(self):
        ref = np.array([1.0, 2.0, 3.0])
        src = np.array([1.0, 2.0, 3.0])
        assert nearest_neighbor_indices(ref, src) == [0, 1, 2]

    def test_source_faster(self):
        """Source has more samples; extras are skipped."""
        ref = np.array([0.0, 1.0, 2.0])
        src = np.array([0.0, 0.3, 0.7, 1.0, 1.4, 1.8, 2.0])
        assert nearest_neighbor_indices(ref, src) == [0, 3, 6]

    def test_source_slower(self):
        """Source has fewer samples; same index is reused."""
        ref = np.array([0.0, 0.4, 1.0, 1.6, 2.0])
        src = np.array([0.0, 1.0, 2.0])
        # ref 1.6: |2.0-1.6|=0.4 < |1.0-1.6|=0.6 → advances to idx 2
        assert nearest_neighbor_indices(ref, src) == [0, 0, 1, 2, 2]

    def test_single_source(self):
        ref = np.array([1.0, 2.0, 3.0])
        src = np.array([1.5])
        assert nearest_neighbor_indices(ref, src) == [0, 0, 0]

    def test_greedy_forward_only(self):
        """Index only advances forward, never goes back."""
        ref = np.array([1.0, 5.0, 5.5])
        src = np.array([1.0, 4.9, 5.1, 5.4])
        assert nearest_neighbor_indices(ref, src) == [0, 1, 3]


# ---------------------------------------------------------------------------
# State-to-frame alignment via timestamps (regression for interpolation bug)
# ---------------------------------------------------------------------------

class TestStateSyncWithTimestamps:
    """State entries have their own timestamps (same clock domain as cameras).
    Nearest-neighbor matching on timestamps properly aligns state to video.
    """

    def test_state_starts_late(self):
        """State recording began 2 frames after the camera started."""
        # Camera ref timestamps after sync window
        ref_ts = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        # State was written starting at t=4 (e.g. NUC was slow to start)
        state_ts = np.array([4.0, 5.0, 6.0, 7.0])
        state_vals = np.array([40.0, 50.0, 60.0, 70.0])

        idx = nearest_neighbor_indices(ref_ts, state_ts)
        synced = state_vals[idx]
        # ref 2.0 -> nearest state 4.0 (idx 0), ref 3.0 -> 4.0, ref 4.0 -> 4.0,
        # ref 5.0 -> 5.0 (idx 1), ref 6.0 -> 6.0 (idx 2)
        np.testing.assert_array_equal(synced, [40.0, 40.0, 40.0, 50.0, 60.0])

    def test_state_same_rate_offset(self):
        """State timestamps offset by a small constant from camera."""
        ref_ts = np.array([100.0, 200.0, 300.0, 400.0])
        # State written ~5ms after each frame trigger
        state_ts = np.array([105.0, 205.0, 305.0, 405.0])
        state_vals = np.arange(4, dtype=np.float64)

        idx = nearest_neighbor_indices(ref_ts, state_ts)
        # Small constant offset — each ref maps to the corresponding state
        np.testing.assert_array_equal(state_vals[idx], [0, 1, 2, 3])

    def test_dropped_state_entry(self):
        """A state callback was missed (e.g. NUC timeout)."""
        ref_ts = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        # State entry at t=2 was lost
        state_ts = np.array([0.0, 1.0, 3.0, 4.0])
        state_vals = np.array([10.0, 20.0, 30.0, 40.0])

        idx = nearest_neighbor_indices(ref_ts, state_ts)
        synced = state_vals[idx]
        # ref 2.0 -> nearest is state t=1.0 or t=3.0, both dist=1.0, greedy stays at idx 1
        np.testing.assert_array_equal(synced, [10.0, 20.0, 20.0, 30.0, 40.0])

    def test_2d_state(self):
        """Multi-dimensional state (joint positions) aligned by timestamp."""
        ref_ts = np.array([1.0, 2.0, 3.0])
        state_ts = np.array([0.5, 1.5, 2.5, 3.5])
        state_vals = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.float64)

        idx = nearest_neighbor_indices(ref_ts, state_ts)
        synced = state_vals[idx]
        # ref 1.0 -> state 0.5 (idx 0) vs 1.5 (idx 1): 0.5 vs 0.5, stays at 0
        # ref 2.0 -> from idx 0, advance to 1.5 (idx 1), check 2.5 (idx 2): 0.5 vs 0.5, stays at 1
        # ref 3.0 -> from idx 1, advance to 2.5 (idx 2): 0.5 vs 0.5, stays. Check 3.5: 0.5 vs 0.5, stays at 2
        np.testing.assert_array_equal(synced, [[1, 10], [2, 20], [3, 30]])


# ---------------------------------------------------------------------------
# compute_sync_window
# ---------------------------------------------------------------------------

class TestComputeSyncWindow:
    def test_two_cameras_overlapping(self):
        rgb = [np.array([1.0, 2.0, 3.0, 4.0]),
               np.array([2.0, 3.0, 4.0, 5.0])]
        t0, t1 = compute_sync_window(rgb)
        assert t0 == 2.0
        assert t1 == 4.0

    def test_three_cameras_one_starts_late_one_ends_early(self):
        rgb = [np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
               np.array([1.5, 2.5, 3.5, 4.5]),
               np.array([0.5, 1.5, 2.5, 3.0])]
        t0, t1 = compute_sync_window(rgb)
        assert t0 == 1.5
        assert t1 == 3.0

    def test_identical_timestamp_ranges(self):
        ts = np.array([10.0, 20.0, 30.0])
        rgb = [ts.copy(), ts.copy()]
        t0, t1 = compute_sync_window(rgb)
        assert t0 == 10.0
        assert t1 == 30.0

    def test_single_camera(self):
        rgb = [np.array([5.0, 10.0, 15.0])]
        t0, t1 = compute_sync_window(rgb)
        assert t0 == 5.0
        assert t1 == 15.0


# ---------------------------------------------------------------------------
# align_frames_to_reference
# ---------------------------------------------------------------------------

class TestAlignFramesToReference:
    def _make_frames(self, n, val=None):
        """Create n dummy 2x2 RGB frames."""
        rgb = np.arange(n).reshape(n, 1, 1, 1) * np.ones((1, 2, 2, 3), dtype=np.uint8) if val is None \
            else np.full((n, 2, 2, 3), val, dtype=np.uint8)
        return rgb

    def test_exact_match(self):
        """When camera timestamps exactly match reference, output is identity."""
        ref_ts = np.array([1.0, 2.0, 3.0])
        cam_ts = np.array([1.0, 2.0, 3.0])
        rgb = self._make_frames(3)
        s_rgb, s_ir_left, s_ir_right, s_ts = align_frames_to_reference(ref_ts, cam_ts, rgb)
        assert len(s_rgb) == 3
        np.testing.assert_array_equal(s_ts, [1.0, 2.0, 3.0])
        # frame ordering preserved
        for i in range(3):
            np.testing.assert_array_equal(s_rgb[i], rgb[i])

    def test_camera_faster_than_reference(self):
        """Camera has more frames than reference; extras are skipped."""
        ref_ts = np.array([0.0, 1.0, 2.0])
        cam_ts = np.array([0.0, 0.3, 0.7, 1.0, 1.4, 1.8, 2.0, 2.3])
        rgb = self._make_frames(8)
        s_rgb, s_ir_left, s_ir_right, s_ts = align_frames_to_reference(ref_ts, cam_ts, rgb)
        assert len(s_rgb) == 3
        # ref 0.0 -> cam 0.0 (idx 0), ref 1.0 -> cam 1.0 (idx 3), ref 2.0 -> cam 2.0 (idx 6)
        np.testing.assert_array_equal(s_ts, [0.0, 1.0, 2.0])

    def test_camera_slower_than_reference(self):
        """Camera has fewer frames than reference; frames are reused."""
        ref_ts = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        cam_ts = np.array([0.0, 1.0, 2.0])
        rgb = self._make_frames(3)
        s_rgb, s_ir_left, s_ir_right, s_ts = align_frames_to_reference(ref_ts, cam_ts, rgb)
        assert len(s_rgb) == 5
        # ref 0.0 -> cam 0.0, ref 0.5 -> cam 0.0 or 1.0 (equidistant, stays at 0),
        # ref 1.0 -> cam 1.0, ref 1.5 -> cam 1.0 or 2.0, ref 2.0 -> cam 2.0
        # The greedy forward search: at ref 0.5, idx=0, check idx=1 (1.0): |1.0-0.5|=0.5 vs |0.0-0.5|=0.5, not strictly <, stays at 0
        expected_ts = [0.0, 0.0, 1.0, 1.0, 2.0]
        np.testing.assert_array_equal(s_ts, expected_ts)

    def test_dropped_frames_gap(self):
        """Large gap in camera timestamps; nearest frame is still selected."""
        ref_ts = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        # Camera drops frames around t=2-3
        cam_ts = np.array([0.0, 1.0, 3.8, 4.0])
        rgb = self._make_frames(4)
        s_rgb, s_ir_left, s_ir_right, s_ts = align_frames_to_reference(ref_ts, cam_ts, rgb)
        assert len(s_rgb) == 5
        # ref 0 -> cam 0, ref 1 -> cam 1, ref 2 -> cam 1 (closer than 3.8),
        # ref 3 -> cam 3.8 (idx 2, advance from 1), ref 4 -> cam 4.0 (idx 3)
        np.testing.assert_array_equal(s_ts, [0.0, 1.0, 1.0, 3.8, 4.0])

    def test_single_frame(self):
        """Single camera frame matched to all reference timestamps."""
        ref_ts = np.array([1.0, 2.0, 3.0])
        cam_ts = np.array([1.5])
        rgb = self._make_frames(1)
        s_rgb, s_ir_left, s_ir_right, s_ts = align_frames_to_reference(ref_ts, cam_ts, rgb)
        assert len(s_rgb) == 3
        # All map to the only available frame
        np.testing.assert_array_equal(s_ts, [1.5, 1.5, 1.5])

    def test_greedy_forward_only(self):
        """The search only advances t_idx forward, never backwards."""
        ref_ts = np.array([1.0, 5.0, 5.5])
        cam_ts = np.array([1.0, 4.9, 5.1, 5.4])
        rgb = self._make_frames(4)
        s_rgb, s_ir_left, s_ir_right, s_ts = align_frames_to_reference(ref_ts, cam_ts, rgb)
        # ref 1.0 -> cam 1.0 (idx 0)
        # ref 5.0 -> advance to idx 1 (4.9), check idx 2 (5.1): |5.1-5.0|=0.1 < |4.9-5.0|=0.1? No (not <). Stay at 1 -> ts 4.9
        # ref 5.5 -> from idx 1, check idx 2 (5.1): |5.1-5.5|=0.4 < |4.9-5.5|=0.6? Yes, advance.
        #            check idx 3 (5.4): |5.4-5.5|=0.1 < |5.1-5.5|=0.4? Yes, advance. idx=3 -> ts 5.4
        np.testing.assert_array_equal(s_ts, [1.0, 4.9, 5.4])


# ---------------------------------------------------------------------------
# resample_timeseries
# ---------------------------------------------------------------------------

class TestResampleTimeseries:
    def test_noop_same_length(self):
        data = np.array([1.0, 2.0, 3.0, 4.0])
        result = resample_timeseries(data, 4)
        np.testing.assert_array_equal(result, data)

    def test_upsample_1d(self):
        data = np.array([0.0, 10.0])
        result = resample_timeseries(data, 5)
        assert len(result) == 5
        np.testing.assert_allclose(result, [0.0, 2.5, 5.0, 7.5, 10.0])

    def test_downsample_1d(self):
        data = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
        result = resample_timeseries(data, 3)
        assert len(result) == 3
        np.testing.assert_allclose(result, [0.0, 5.0, 10.0])

    def test_2d_multi_joint(self):
        """2D array: each column interpolated independently."""
        data = np.array([[0.0, 100.0],
                         [10.0, 200.0]])
        result = resample_timeseries(data, 3)
        assert result.shape == (3, 2)
        np.testing.assert_allclose(result[:, 0], [0.0, 5.0, 10.0])
        np.testing.assert_allclose(result[:, 1], [100.0, 150.0, 200.0])

    def test_single_element(self):
        data = np.array([42.0])
        result = resample_timeseries(data, 3)
        assert len(result) == 3
        # interp from a single point: all values should be 42.0
        np.testing.assert_allclose(result, [42.0, 42.0, 42.0])

    def test_preserves_endpoints(self):
        data = np.array([1.0, 3.0, 7.0, 15.0])
        result = resample_timeseries(data, 6)
        assert result[0] == pytest.approx(1.0)
        assert result[-1] == pytest.approx(15.0)

    def test_2d_preserves_dtype(self):
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = resample_timeseries(data, 4)
        assert result.dtype == np.float32
        assert result.shape == (4, 3)
