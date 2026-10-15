from __future__ import annotations

import math
import unittest

import numpy as np

from audit_r1_1 import (
    constant_velocity_trace,
    observed_geometry_metrics,
    route_geometry_metrics,
    wrap_angle_array,
)


class TestR11Audit(unittest.TestCase):
    def test_constant_velocity_starts_at_observed_state(self) -> None:
        observed = np.array(
            [[3.0, -2.0, 4.0, math.pi / 3.0], [0.0, 0.0, 0.0, 0.0]],
            dtype=float,
        )
        trace = constant_velocity_trace(observed, 0.1)
        np.testing.assert_allclose(trace[0], observed[0], rtol=0.0, atol=0.0)

    def test_constant_velocity_is_straight_and_uniform(self) -> None:
        observed = np.array(
            [[1.0, 2.0, 5.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            dtype=float,
        )
        trace = constant_velocity_trace(observed, 0.2)
        np.testing.assert_allclose(trace[:, 0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(trace[:, 1], [2.0, 2.0, 2.0])
        np.testing.assert_allclose(trace[:, 2], 5.0)
        np.testing.assert_allclose(trace[:, 3], 0.0)

    def test_straight_synthetic_trajectory_has_zero_curvature(self) -> None:
        observed = np.zeros((4, 4), dtype=float)
        observed[:, 0] = [0.0, 1.0, 2.0, 3.0]
        metrics = observed_geometry_metrics(observed, 0.1)
        self.assertEqual(metrics["path_length_m"], 3.0)
        self.assertEqual(metrics["net_heading_change_deg"], 0.0)
        self.assertEqual(metrics["cumulative_abs_heading_change_deg"], 0.0)
        self.assertEqual(metrics["mean_abs_curvature_rad_per_m"], 0.0)

    def test_straight_route_has_zero_curvature(self) -> None:
        metrics = route_geometry_metrics(
            np.array([0.5, 1.0, 1.5]),
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 0.0, 0.0]),
        )
        self.assertEqual(metrics["route_net_heading_change_deg"], 0.0)
        self.assertEqual(metrics["route_cumulative_abs_heading_change_deg"], 0.0)
        self.assertEqual(metrics["route_mean_abs_curvature_rad_per_m"], 0.0)

    def test_angle_wrapping_at_branch_cut(self) -> None:
        values = wrap_angle_array(np.array([0.0, math.pi, -math.pi, 3.0 * math.pi]))
        np.testing.assert_allclose(values, [0.0, -math.pi, -math.pi, -math.pi])


if __name__ == "__main__":
    unittest.main(verbosity=2)
