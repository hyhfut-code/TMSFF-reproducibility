"""Reviewer 1, Comment 1: temporal-block difficulty and constant-velocity audit.

This audit is deliberately post hoc and parameter-free.  It does not refit
TMSFF or AV-IDM and it does not change the frozen 70/30 temporal split.

The default paths follow the public reproducibility-package layout.  The
CitySim-derived cache is intentionally not distributed; create it first with
``python scripts/m11_data.py`` after placing the authorized source files in
``external_data/`` as documented in the README.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


EXPECTED = {
    "trajectory_count": 38,
    "total_frames": 9009,
    "calibration_frames": 6289,
    "holdout_frames": 2720,
    "dt": 1.0 / 30.0,
}

METRICS = (
    "frames",
    "duration_s",
    "path_length_m",
    "net_heading_change_deg",
    "cumulative_abs_heading_change_deg",
    "mean_abs_curvature_rad_per_m",
    "route_net_heading_change_deg",
    "route_cumulative_abs_heading_change_deg",
    "route_mean_abs_curvature_rad_per_m",
    "cv_raw_rmse_m",
    "cv_penalized_rmse_m",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wrap_angle_array(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap radians to [-pi, pi), suitable for differences."""
    return (np.asarray(angle) + math.pi) % (2.0 * math.pi) - math.pi


def constant_velocity_trace(observed_block: np.ndarray, dt: float) -> np.ndarray:
    """Propagate from the observed split state with fixed speed and heading."""
    if len(observed_block) < 1:
        raise ValueError("A block must contain at least one frame")
    x0, y0 = observed_block[0, :2]
    speed0 = max(float(observed_block[0, 2]), 0.0)
    heading0 = float(observed_block[0, 3])
    elapsed = np.arange(len(observed_block), dtype=float) * float(dt)
    trace = np.empty((len(observed_block), 4), dtype=float)
    trace[:, 0] = x0 + speed0 * math.cos(heading0) * elapsed
    trace[:, 1] = y0 + speed0 * math.sin(heading0) * elapsed
    trace[:, 2] = speed0
    trace[:, 3] = heading0
    return trace


def observed_geometry_metrics(observed_block: np.ndarray, dt: float) -> dict[str, float]:
    """Describe horizon length and turning demand from the observed trajectory."""
    if len(observed_block) < 1:
        raise ValueError("A block must contain at least one frame")
    if len(observed_block) == 1:
        path_length = 0.0
        net_change = 0.0
        cumulative_change = 0.0
    else:
        displacement = np.diff(observed_block[:, :2], axis=0)
        path_length = float(np.hypot(displacement[:, 0], displacement[:, 1]).sum())
        heading = np.unwrap(observed_block[:, 3])
        net_change = abs(float(heading[-1] - heading[0]))
        cumulative_change = float(np.abs(np.diff(heading)).sum())
    return {
        "frames": int(len(observed_block)),
        "duration_s": float(max(len(observed_block) - 1, 0) * dt),
        "path_length_m": path_length,
        "net_heading_change_deg": math.degrees(net_change),
        "cumulative_abs_heading_change_deg": math.degrees(cumulative_change),
        "mean_abs_curvature_rad_per_m": (
            cumulative_change / path_length if path_length > 1e-12 else 0.0
        ),
    }


def route_geometry_metrics(
    observed_station: np.ndarray,
    route_station: np.ndarray,
    route_heading: np.ndarray,
) -> dict[str, float]:
    """Curvature sensitivity check using the reconstructed route reference."""
    station_min = float(np.min(observed_station))
    station_max = float(np.max(observed_station))
    route_heading_unwrapped = np.unwrap(route_heading)
    interior = route_station[(route_station > station_min) & (route_station < station_max)]
    sample_station = np.unique(
        np.concatenate(([station_min], np.asarray(interior, dtype=float), [station_max]))
    )
    sample_heading = np.interp(sample_station, route_station, route_heading_unwrapped)
    if len(sample_heading) < 2:
        net_change = 0.0
        cumulative_change = 0.0
    else:
        net_change = abs(float(sample_heading[-1] - sample_heading[0]))
        cumulative_change = float(np.abs(np.diff(sample_heading)).sum())
    route_distance = max(station_max - station_min, 0.0)
    return {
        "route_net_heading_change_deg": math.degrees(net_change),
        "route_cumulative_abs_heading_change_deg": math.degrees(cumulative_change),
        "route_mean_abs_curvature_rad_per_m": (
            cumulative_change / route_distance if route_distance > 1e-12 else 0.0
        ),
    }


def route_penalized_mse(
    trace_xy: np.ndarray,
    observed_block: np.ndarray,
    observed_station: np.ndarray,
    route_arrays: tuple[np.ndarray, ...],
    project_route,
) -> tuple[float, float, float]:
    """Return raw MSE, Eq. (35)-convention penalized MSE, and penalty MSE."""
    if len(trace_xy) != len(observed_block):
        raise ValueError("Trace and observation lengths differ")
    squared_error = np.sum((trace_xy - observed_block[:, :2]) ** 2, axis=1)
    station_min = float(np.min(observed_station))
    station_max = float(np.max(observed_station))
    penalty = np.zeros(len(trace_xy), dtype=float)
    previous = -1
    for index, (x, y) in enumerate(trace_xy):
        previous, _, _, _, _, station = project_route(
            float(x), float(y), *route_arrays, previous
        )
        if station < station_min:
            penalty[index] = (station_min - station) ** 2
        elif station > station_max:
            penalty[index] = (station - station_max) ** 2
    raw_mse = float(np.mean(squared_error))
    penalty_mse = float(np.mean(penalty))
    return raw_mse, raw_mse + penalty_mse, penalty_mse


def iqr(values: np.ndarray) -> float:
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def safe_wilcoxon(first: np.ndarray, second: np.ndarray) -> float:
    difference = np.asarray(second, dtype=float) - np.asarray(first, dtype=float)
    if np.allclose(difference, 0.0):
        return 1.0
    return float(stats.wilcoxon(difference, alternative="two-sided").pvalue)


def load_modules(project_root: Path):
    scripts = project_root / "scripts"
    sys.path.insert(0, str(scripts))
    from m11_models import (  # type: ignore
        block_objective_avidm,
        block_objective_tmsff,
        project_route,
    )
    from run_m11_calibration import MODEL_PARAMETERS, load_cache  # type: ignore

    return (
        block_objective_tmsff,
        block_objective_avidm,
        project_route,
        MODEL_PARAMETERS,
        load_cache,
    )


def preflight(m11_results: Path, data: dict) -> dict[str, object]:
    lengths = np.asarray(data["lengths"], dtype=int)
    splits = np.asarray(data["split_indices"], dtype=int)
    checks = {
        "trajectory_count": int(len(lengths)),
        "total_frames": int(lengths.sum()),
        "calibration_frames": int(splits.sum()),
        "holdout_frames": int((lengths - splits).sum()),
        "dt": float(data["dt"]),
    }
    for key, expected in EXPECTED.items():
        actual = checks[key]
        if not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"Preflight failed for {key}: {actual} != {expected}")

    runs_path = m11_results / "common_protocol" / "ga_runs.csv"
    table_path = m11_results / "analysis" / "Table_M11_common_protocol_comparison.csv"
    runs = pd.read_csv(runs_path)
    table = pd.read_csv(table_path).set_index("model")
    for model in ("TMSFF", "AV-IDM"):
        group = runs[runs["model"] == model]
        for source, target in (
            ("calibration_objective_m", "calibration_objective_mean_m"),
            ("holdout_objective_m", "holdout_objective_mean_m"),
        ):
            actual = float(group[source].mean())
            expected = float(table.loc[model, target])
            if not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
                raise RuntimeError(
                    f"Table 4 preflight failed for {model} {source}: {actual} != {expected}"
                )
    return checks


def seed_parameters(
    m11_results: Path, model: str, seed: int, model_parameters: dict
) -> np.ndarray:
    estimates = pd.read_csv(
        m11_results / "common_protocol" / "ga_parameter_estimates.csv"
    )
    subset = estimates[
        (estimates["model"] == model) & (estimates["seed"] == seed)
    ].set_index("parameter")
    return subset.loc[list(model_parameters[model]), "estimate"].to_numpy(float)


def block_metrics(
    data: dict,
    pair_index: int,
    start: int,
    stop: int,
    block_name: str,
    project_route,
) -> dict[str, object]:
    observed = data["observed"][pair_index, start:stop]
    trace = constant_velocity_trace(observed, float(data["dt"]))
    geometry = observed_geometry_metrics(observed, float(data["dt"]))
    route_geometry = route_geometry_metrics(
        observed[:, 5],
        data["route_station"][pair_index],
        data["route_heading"][pair_index],
    )
    route_arrays = (
        data["route_x"][pair_index],
        data["route_y"][pair_index],
        data["route_heading"][pair_index],
        data["route_left_width"][pair_index],
        data["route_right_width"][pair_index],
        data["route_station"][pair_index],
    )
    raw_mse, penalized_mse, penalty_mse = route_penalized_mse(
        trace[:, :2], observed, observed[:, 5], route_arrays, project_route
    )
    metadata = data["pair_metadata"][pair_index]
    return {
        "pair_index": pair_index,
        "trajectory_no": int(metadata[0]),
        "leader_id": int(metadata[1]),
        "follower_id": int(metadata[2]),
        "entry_lane": int(metadata[3]),
        "exit_lane": int(metadata[4]),
        "block": block_name,
        "start_index": int(start),
        "stop_index_exclusive": int(stop),
        **geometry,
        **route_geometry,
        "cv_raw_mse_m2": raw_mse,
        "cv_raw_rmse_m": math.sqrt(raw_mse),
        "cv_range_penalty_mse_m2": penalty_mse,
        "cv_penalized_mse_m2": penalized_mse,
        "cv_penalized_rmse_m": math.sqrt(penalized_mse),
    }


def summarize_blocks(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    for block, group in records.groupby("block", sort=False):
        for metric in METRICS:
            values = group[metric].to_numpy(float)
            summaries.append(
                {
                    "block": block,
                    "metric": metric,
                    "n_trajectories": len(values),
                    "mean": float(values.mean()),
                    "sample_sd": float(values.std(ddof=1)),
                    "median": float(np.median(values)),
                    "q1": float(np.percentile(values, 25)),
                    "q3": float(np.percentile(values, 75)),
                    "iqr": iqr(values),
                }
            )

    wide = records.pivot(index="pair_index", columns="block", values=list(METRICS))
    paired: list[dict[str, object]] = []
    for metric in METRICS:
        calibration = wide[(metric, "calibration")].to_numpy(float)
        holdout = wide[(metric, "holdout")].to_numpy(float)
        calibration_median = float(np.median(calibration))
        holdout_median = float(np.median(holdout))
        paired.append(
            {
                "metric": metric,
                "calibration_median": calibration_median,
                "calibration_iqr": iqr(calibration),
                "holdout_median": holdout_median,
                "holdout_iqr": iqr(holdout),
                "median_paired_difference_holdout_minus_calibration": float(
                    np.median(holdout - calibration)
                ),
                "holdout_vs_calibration_percent_from_medians": (
                    100.0 * (holdout_median / calibration_median - 1.0)
                    if abs(calibration_median) > 1e-12
                    else np.nan
                ),
                "paired_wilcoxon_two_sided_p": safe_wilcoxon(calibration, holdout),
            }
        )

    global_cv: list[dict[str, object]] = []
    for block, group in records.groupby("block", sort=False):
        global_cv.append(
            {
                "block": block,
                "equal_trajectory_raw_objective_m": math.sqrt(
                    float(group["cv_raw_mse_m2"].mean())
                ),
                "equal_trajectory_penalized_objective_m": math.sqrt(
                    float(group["cv_penalized_mse_m2"].mean())
                ),
                "mean_range_penalty_mse_m2": float(
                    group["cv_range_penalty_mse_m2"].mean()
                ),
            }
        )
    return pd.DataFrame(summaries), pd.DataFrame(paired), pd.DataFrame(global_cv)


def equal_horizon_diagnostic(
    m11_results: Path,
    data: dict,
    project_route,
    block_objective_tmsff,
    block_objective_avidm,
    model_parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parameters = {
        "TMSFF": seed_parameters(m11_results, "TMSFF", 42, model_parameters),
        "AV-IDM": seed_parameters(m11_results, "AV-IDM", 42, model_parameters),
    }
    rows: list[dict[str, object]] = []
    for pair_index, (length, split) in enumerate(
        zip(data["lengths"], data["split_indices"])
    ):
        length = int(length)
        split = int(split)
        horizon = length - split
        windows = {
            "late_calibration_equal_horizon": (split - horizon, split),
            "holdout": (split, length),
        }
        for block_name, (start, stop) in windows.items():
            row = block_metrics(
                data, pair_index, start, stop, block_name, project_route
            )
            row["tmsff_penalized_mse_m2"] = float(
                block_objective_tmsff(
                    parameters["TMSFF"],
                    pair_index,
                    start,
                    stop,
                    data["observed"],
                    data["leader"],
                    data["route_x"],
                    data["route_y"],
                    data["route_heading"],
                    data["route_left_width"],
                    data["route_right_width"],
                    data["route_station"],
                    float(data["dt"]),
                )
            )
            row["avidm_penalized_mse_m2"] = float(
                block_objective_avidm(
                    parameters["AV-IDM"],
                    pair_index,
                    start,
                    stop,
                    data["observed"],
                    data["environment"],
                    data["route_x"],
                    data["route_y"],
                    data["route_heading"],
                    data["route_left_width"],
                    data["route_right_width"],
                    data["route_station"],
                    float(data["dt"]),
                )
            )
            row["tmsff_penalized_rmse_m"] = math.sqrt(
                row["tmsff_penalized_mse_m2"]
            )
            row["avidm_penalized_rmse_m"] = math.sqrt(
                row["avidm_penalized_mse_m2"]
            )
            rows.append(row)
    per_trajectory = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    objective_columns = {
        "constant_velocity_raw": "cv_raw_mse_m2",
        "constant_velocity_penalized": "cv_penalized_mse_m2",
        "TMSFF_seed42_penalized": "tmsff_penalized_mse_m2",
        "AV-IDM_seed42_penalized": "avidm_penalized_mse_m2",
    }
    for model, column in objective_columns.items():
        values = {
            block: math.sqrt(float(group[column].mean()))
            for block, group in per_trajectory.groupby("block", sort=False)
        }
        summary_rows.append(
            {
                "model_or_reference": model,
                "late_calibration_equal_horizon_objective_m": values[
                    "late_calibration_equal_horizon"
                ],
                "holdout_objective_m": values["holdout"],
                "holdout_minus_late_calibration_m": values["holdout"]
                - values["late_calibration_equal_horizon"],
            }
        )
    comparison_ready = per_trajectory.copy()
    comparison_ready["block"] = comparison_ready["block"].replace(
        {"late_calibration_equal_horizon": "calibration"}
    )
    _, paired, _ = summarize_blocks(comparison_ready)
    wide = comparison_ready.pivot(
        index="pair_index", columns="block", values=[
            "tmsff_penalized_rmse_m",
            "avidm_penalized_rmse_m",
        ]
    )
    extra_rows: list[dict[str, object]] = []
    for metric in ("tmsff_penalized_rmse_m", "avidm_penalized_rmse_m"):
        calibration = wide[(metric, "calibration")].to_numpy(float)
        holdout = wide[(metric, "holdout")].to_numpy(float)
        calibration_median = float(np.median(calibration))
        holdout_median = float(np.median(holdout))
        extra_rows.append(
            {
                "metric": metric,
                "calibration_median": calibration_median,
                "calibration_iqr": iqr(calibration),
                "holdout_median": holdout_median,
                "holdout_iqr": iqr(holdout),
                "median_paired_difference_holdout_minus_calibration": float(
                    np.median(holdout - calibration)
                ),
                "holdout_vs_calibration_percent_from_medians": (
                    100.0 * (holdout_median / calibration_median - 1.0)
                    if abs(calibration_median) > 1e-12
                    else np.nan
                ),
                "paired_wilcoxon_two_sided_p": safe_wilcoxon(calibration, holdout),
            }
        )
    paired = pd.concat([paired, pd.DataFrame(extra_rows)], ignore_index=True)
    return per_trajectory, pd.DataFrame(summary_rows), paired


def main() -> None:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help=(
            "CitySim cache created by scripts/m11_data.py (default: "
            "<project-root>/reproduced_outputs/cache/m11_citysim_cache_k16.npz)"
        ),
    )
    parser.add_argument(
        "--m11-results",
        type=Path,
        default=None,
        help="Archived M11 result root (default: <project-root>/outputs/M11)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Audit output directory (default: <project-root>/reproduced_outputs/R1_1)",
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    cache_path = (
        args.cache.resolve()
        if args.cache is not None
        else project_root
        / "reproduced_outputs"
        / "cache"
        / "m11_citysim_cache_k16.npz"
    )
    m11_results = (
        args.m11_results.resolve()
        if args.m11_results is not None
        else project_root / "outputs" / "M11"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else project_root / "reproduced_outputs" / "R1_1"
    )
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"CitySim cache not found: {cache_path}. Run scripts/m11_data.py first."
        )
    if not m11_results.is_dir():
        raise FileNotFoundError(f"M11 result directory not found: {m11_results}")
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        block_objective_tmsff,
        block_objective_avidm,
        project_route,
        model_parameters,
        load_cache,
    ) = load_modules(project_root)
    data = load_cache(cache_path)
    checks = preflight(m11_results, data)

    records: list[dict[str, object]] = []
    for pair_index, (length, split) in enumerate(
        zip(data["lengths"], data["split_indices"])
    ):
        records.append(
            block_metrics(data, pair_index, 0, int(split), "calibration", project_route)
        )
        records.append(
            block_metrics(
                data, pair_index, int(split), int(length), "holdout", project_route
            )
        )
    per_trajectory = pd.DataFrame(records)
    block_summary, paired, global_cv = summarize_blocks(per_trajectory)

    paired_by_metric = paired.set_index("metric")
    cv_by_block = global_cv.set_index("block")
    duration_lower = (
        paired_by_metric.loc[
            "duration_s", "holdout_vs_calibration_percent_from_medians"
        ]
        < 0.0
    )
    turning_lower = (
        paired_by_metric.loc[
            "cumulative_abs_heading_change_deg",
            "holdout_vs_calibration_percent_from_medians",
        ]
        < 0.0
        and paired_by_metric.loc[
            "net_heading_change_deg", "holdout_vs_calibration_percent_from_medians"
        ]
        < 0.0
    )
    cv_lower = (
        cv_by_block.loc["holdout", "equal_trajectory_penalized_objective_m"]
        < cv_by_block.loc[
            "calibration", "equal_trajectory_penalized_objective_m"
        ]
    )
    easier_block_supported = bool(duration_lower and turning_lower and cv_lower)
    equal_horizon_needed = not easier_block_supported

    per_trajectory.to_csv(
        output_dir / "r1_1_per_trajectory_block_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    block_summary.to_csv(
        output_dir / "r1_1_block_descriptive_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    paired.to_csv(
        output_dir / "r1_1_paired_block_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )
    global_cv.to_csv(
        output_dir / "r1_1_constant_velocity_global_objectives.csv",
        index=False,
        encoding="utf-8-sig",
    )

    equal_horizon_files: list[str] = []
    if equal_horizon_needed:
        equal_per_trajectory, equal_summary, equal_paired = equal_horizon_diagnostic(
            m11_results,
            data,
            project_route,
            block_objective_tmsff,
            block_objective_avidm,
            model_parameters,
        )
        per_path = output_dir / "r1_1_equal_horizon_per_trajectory.csv"
        summary_path = output_dir / "r1_1_equal_horizon_summary.csv"
        paired_path = output_dir / "r1_1_equal_horizon_paired_comparisons.csv"
        equal_per_trajectory.to_csv(per_path, index=False, encoding="utf-8-sig")
        equal_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        equal_paired.to_csv(paired_path, index=False, encoding="utf-8-sig")
        equal_horizon_files = [per_path.name, summary_path.name, paired_path.name]

    def portable_path_label(path: Path) -> str:
        try:
            return path.relative_to(project_root).as_posix()
        except ValueError:
            return path.name

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Reviewer 1 Comment 1 temporal-block difficulty audit",
        "preflight": checks,
        "cache_path": portable_path_label(cache_path),
        "cache_sha256": sha256(cache_path),
        "m11_results_path": portable_path_label(m11_results),
        "constant_velocity_definition": (
            "Observed block-start x, y, speed, and heading; fixed speed and fixed "
            "heading thereafter; no leader, route, future observation, acceleration, "
            "steering, or fitted parameter."
        ),
        "duration_definition": "(frames - 1) / 30 seconds",
        "curvature_definition": (
            "sum(abs(diff(unwrap(observed heading)))) / observed path length"
        ),
        "objective_aggregation": (
            "square root of the mean per-trajectory MSE, with equal trajectory weight"
        ),
        "easier_block_supported": easier_block_supported,
        "decision_components": {
            "holdout_shorter": bool(duration_lower),
            "holdout_less_total_turning": bool(turning_lower),
            "holdout_lower_constant_velocity_penalized_objective": bool(cv_lower),
        },
        "equal_horizon_diagnostic_triggered": equal_horizon_needed,
        "equal_horizon_output_files": equal_horizon_files,
        "model_refitting_performed": False,
        "frozen_split_changed": False,
    }
    with (output_dir / "r1_1_audit_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("\nPaired block comparisons:\n")
    print(paired.to_string(index=False))
    print("\nConstant-velocity global objectives:\n")
    print(global_cv.to_string(index=False))


if __name__ == "__main__":
    main()
