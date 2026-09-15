"""Seed-42 expanded-bound sensitivity audit for AV-IDM.

This is supplementary only.  The primary comparison retains the parameter
ranges published by Sharath and Velaga (2020).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_m11_calibration as calibration


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "reproduced_outputs" / "M11" / "expanded_bound_audit"

EXPANDED_BOUNDS = np.array(
    [
        [0.1, 5.0],
        [0.1, 5.0],
        [0.1, 5.0],
        [0.1, 5.0],
        [0.1, 1.5],
        [0.5, 6.0],
        [0.1, 6.0],
        [0.1, 3.0],
        [0.1, 6.0],
        [3.0, 30.0],
        [0.5, 6.0],
        [0.1, 8.0],
        [0.1, 3.0],
    ],
    dtype=float,
)


def main() -> None:
    protocol = calibration.load_protocol(calibration.DEFAULT_PROTOCOL)
    settings = protocol["optimizer"].copy()
    data = calibration.load_cache(calibration.DEFAULT_CACHE)
    original_bounds = calibration.MODEL_BOUNDS["AV-IDM"].copy()
    calibration.MODEL_BOUNDS["AV-IDM"] = EXPANDED_BOUNDS
    parameters, calibration_objective, holdout_objective, convergence, elapsed = (
        calibration.run_ga("AV-IDM", 42, data, settings)
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for name, estimate, original, expanded in zip(
        calibration.MODEL_PARAMETERS["AV-IDM"],
        parameters,
        original_bounds,
        EXPANDED_BOUNDS,
    ):
        records.append(
            {
                "parameter": name,
                "estimate": float(estimate),
                "original_lower_bound": float(original[0]),
                "original_upper_bound": float(original[1]),
                "expanded_lower_bound": float(expanded[0]),
                "expanded_upper_bound": float(expanded[1]),
                "expanded_relative_bound_position": float(
                    (estimate - expanded[0]) / (expanded[1] - expanded[0])
                ),
            }
        )
    pd.DataFrame(records).to_csv(
        OUTPUT / "expanded_bound_parameter_estimates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(convergence).to_csv(
        OUTPUT / "expanded_bound_convergence.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "model": "AV-IDM",
        "seed": 42,
        "status": "supplementary expanded-bound sensitivity; not the primary comparison",
        "calibration_objective_m": calibration_objective,
        "holdout_objective_m": holdout_objective,
        "elapsed_seconds": elapsed,
        "evaluations": int(settings["population_size"]) * (int(settings["generations"]) + 1),
    }
    with (OUTPUT / "expanded_bound_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
