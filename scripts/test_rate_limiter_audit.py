from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("run_rate_limiter_audit.py")
SPEC = importlib.util.spec_from_file_location("run_rate_limiter_audit", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load {SCRIPT}")
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


class TargetHeadingLimiterTests(unittest.TestCase):
    def setUp(self) -> None:
        audit.configure_table1(0.1)

    def test_first_value_is_not_artificially_ramped_from_zero(self) -> None:
        raw = 1.7
        self.assertAlmostEqual(audit.limited_target(None, raw, 0.5), raw)

    def test_wrapped_step_is_rate_bounded(self) -> None:
        previous = math.radians(179.0)
        raw = math.radians(-120.0)
        limited = audit.limited_target(previous, raw, 0.5)
        change = float(audit.core.wrap_angle(limited - previous))
        self.assertLessEqual(abs(change), 0.05 + 1e-12)

    def test_unlimited_target_equals_wrapped_raw_target(self) -> None:
        values = np.linspace(-4.0 * math.pi, 4.0 * math.pi, 101)
        for value in values:
            self.assertAlmostEqual(
                audit.limited_target(0.4, float(value), None),
                float(audit.core.wrap_angle(value)),
                places=12,
            )

    def test_metric_indices_are_self_consistent(self) -> None:
        error = np.array([1.0, -1.0, 2.0])
        observed = np.array([2.0, 2.0, 4.0])
        result = audit.metric_indices(error, observed, 0.1)
        self.assertAlmostEqual(result["MSE"], 2.0)
        self.assertAlmostEqual(result["RMSE"], math.sqrt(2.0))
        self.assertAlmostEqual(result["MAE"], 4.0 / 3.0)
        self.assertAlmostEqual(result["MAPE"], 0.5)


if __name__ == "__main__":
    unittest.main()
