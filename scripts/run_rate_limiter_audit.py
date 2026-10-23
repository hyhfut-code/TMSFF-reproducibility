"""Closed-loop target-heading slew-rate audit for TMSFF.

The hard branch in Eqs. (17)-(19) selects either the leader heading or the
line-of-sight heading.  This script inserts a symmetric slew-rate limiter on
that selected target heading before the existing cubic path is constructed.
All longitudinal equations, fixed Table 1 parameters, path construction, and
trajectory-level metric definitions remain unchanged.

The unlimited condition calls the archived baseline simulator directly.  It
therefore serves as a numerical identity check against Tables 2 and 3.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
CORE_PATH = SCRIPT_DIR / "citysim_core.py"
DEFAULT_CITYSIM_INPUT = REPOSITORY_ROOT / "external_data" / "IntersectionA_cleaned.csv"
DEFAULT_CITYSIM_MANIFEST = SCRIPT_DIR / "citysim_selected_following_pairs.csv"
DEFAULT_HEFEI_INPUT = (
    REPOSITORY_ROOT
    / "external_data"
    / "private_normalized_observed_trajectories.csv"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "reproduced_outputs" / "rate_limiter"


def load_core():
    spec = importlib.util.spec_from_file_location("rate_limiter_citysim_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load simulation core: {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


core = load_core()

WHEELBASE_M = 2.5
STEERING_RATIO = 16.0
BASELINE_SC_M = 15.8
RATE_LIMITS_RAD_S = (0.5, 1.0, 2.0, 4.0, 8.0)
VARIABLES = (
    ("Heading angle", "rad", "heading"),
    ("Arc spacing", "m", "arc_spacing"),
    ("Velocity", "m/s", "velocity"),
    ("Acceleration", "m/s^2", "acceleration"),
)
INDICES = ("MSE", "MAE", "RMSE", "MAPE")
MAPE_FLOORS = {
    "heading": 0.01,
    "arc_spacing": 0.10,
    "velocity": 0.10,
    "acceleration": 0.10,
}
TABLE1_TIDM_PARAMETERS = {
    "a_max_mps2": 2.81,
    "b_comfort_mps2": 2.96,
    "time_gap_s": 1.28,
    "desired_speed_mps": 18.06,
    "minimum_safe_arc_spacing_m": 4.81,
}
TABLE1_TFVD_PARAMETERS = {
    "kappa": 0.28,
    "lambda_speed": 0.30,
    "maximum_speed_mps": 10.90,
    "safety_arc_spacing_m": 9.11,
}
EXPECTED_BASELINE_RMSE = {
    ("Hefei Table 2", "T-IDM", "Heading angle"): 0.09289168916020436,
    ("Hefei Table 2", "T-IDM", "Arc spacing"): 3.8893817229726406,
    ("Hefei Table 2", "T-IDM", "Velocity"): 0.7473799944672581,
    ("Hefei Table 2", "T-IDM", "Acceleration"): 0.7222404447724353,
    ("Hefei Table 2", "T-FVD", "Heading angle"): 0.15521448948339853,
    ("Hefei Table 2", "T-FVD", "Arc spacing"): 7.238591786879526,
    ("Hefei Table 2", "T-FVD", "Velocity"): 1.343788668865656,
    ("Hefei Table 2", "T-FVD", "Acceleration"): 0.9338221034182903,
    ("CitySim Table 3", "T-IDM", "Heading angle"): 0.12010617063530922,
    ("CitySim Table 3", "T-IDM", "Arc spacing"): 4.657420973166916,
    ("CitySim Table 3", "T-IDM", "Velocity"): 1.1713938133466106,
    ("CitySim Table 3", "T-IDM", "Acceleration"): 1.186364947158445,
    ("CitySim Table 3", "T-FVD", "Heading angle"): 0.07697180742103645,
    ("CitySim Table 3", "T-FVD", "Arc spacing"): 2.354052436720675,
    ("CitySim Table 3", "T-FVD", "Velocity"): 0.7358316597660841,
    ("CitySim Table 3", "T-FVD", "Acceleration"): 0.95482192449368,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_reference(path: Path) -> str:
    """Return metadata paths that do not disclose an author's local directory."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.name


def configure_table1(dt_s: float) -> None:
    core.HEFEI_TIDM_PARAMETERS.update(TABLE1_TIDM_PARAMETERS)
    core.HEFEI_TFVD_PARAMETERS.update(TABLE1_TFVD_PARAMETERS)
    core.SHORT_RANGE_THRESHOLD_M = BASELINE_SC_M
    core.DT = float(dt_s)
    core.FPS = 1.0 / float(dt_s)


def metric_indices(error: np.ndarray, observed: np.ndarray, floor: float) -> dict[str, float]:
    error = np.asarray(error, dtype=float)
    observed = np.asarray(observed, dtype=float)
    denominator = np.maximum(np.abs(observed), floor)
    return {
        "MSE": float(np.mean(np.square(error))),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(np.square(error)))),
        "MAPE": float(np.mean(np.abs(error) / denominator)),
    }


def evaluate_all_indices(
    pair: pd.DataFrame,
    simulation: pd.DataFrame,
    observed_arc_spacing: np.ndarray,
) -> dict[tuple[str, str], float]:
    dynamic = slice(1, len(pair))
    acceleration = slice(0, len(pair) - 1)
    errors_and_observations = {
        "heading": (
            core.wrap_angle(
                simulation["heading_sim"].to_numpy()[dynamic]
                - pair["heading_follower"].to_numpy()[dynamic]
            ),
            core.wrap_angle(pair["heading_follower"].to_numpy()[dynamic]),
        ),
        "arc_spacing": (
            simulation["arc_spacing_sim"].to_numpy()[dynamic]
            - observed_arc_spacing[dynamic],
            observed_arc_spacing[dynamic],
        ),
        "velocity": (
            simulation["velocity_sim"].to_numpy()[dynamic]
            - pair["velocity_follower"].to_numpy()[dynamic],
            pair["velocity_follower"].to_numpy()[dynamic],
        ),
        "acceleration": (
            simulation["acceleration_sim"].to_numpy()[acceleration]
            - pair["acceleration_follower"].to_numpy()[acceleration],
            pair["acceleration_follower"].to_numpy()[acceleration],
        ),
    }
    result: dict[tuple[str, str], float] = {}
    for _, _, key in VARIABLES:
        error, observed = errors_and_observations[key]
        for index, value in metric_indices(error, observed, MAPE_FLOORS[key]).items():
            result[(key, index)] = value
    return result


def raw_target_heading(
    follower_x: float,
    follower_y: float,
    leader_x: float,
    leader_y: float,
    leader_heading: float,
) -> tuple[float, str, float]:
    euclidean = float(np.hypot(leader_x - follower_x, leader_y - follower_y))
    if euclidean <= core.SHORT_RANGE_THRESHOLD_M:
        target = math.atan2(leader_y - follower_y, leader_x - follower_x)
        branch = "line-of-sight"
    else:
        target = leader_heading
        branch = "leader-heading"
    return float(core.wrap_angle(target)), branch, euclidean


def build_path_with_target(
    follower_x: float,
    follower_y: float,
    follower_heading: float,
    leader_x: float,
    leader_y: float,
    leader_heading: float,
    target_heading_original: float,
):
    follower, leader, turning_angle = core.coordinate_system_transformation(
        follower_x,
        follower_y,
        follower_heading,
        leader_x,
        leader_y,
        leader_heading,
    )
    target_heading_rotated = float(core.wrap_angle(target_heading_original - turning_angle))
    euclidean = float(np.hypot(leader[0] - follower[0], leader[1] - follower[1]))
    delta_x = float(leader[0] - follower[0])
    delta_y = float(leader[1] - follower[1])
    if delta_x <= 1e-6:
        raise ValueError(f"Non-positive longitudinal separation after rotation: {delta_x}")

    matrix = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [delta_x**3, delta_x**2, delta_x, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [3.0 * delta_x**2, 2.0 * delta_x, 1.0, 0.0],
        ]
    )
    rhs = np.array(
        [0.0, delta_y, math.tan(follower[2]), math.tan(target_heading_rotated)]
    )
    try:
        coefficients = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(matrix, rhs, rcond=None)[0]

    path = core.PlannedPath(
        x_origin=follower[0],
        y_origin=follower[1],
        x_leader_local=delta_x,
        coefficients=coefficients,
        follower_heading_rotated=follower[2],
        turning_angle=turning_angle,
        arc_spacing_m=0.0,
        euclidean_spacing_m=euclidean,
    )
    path.arc_spacing_m = path.arc_length(delta_x)
    return path


def limited_target(previous: float | None, raw: float, rate_rad_s: float | None) -> float:
    if previous is None or rate_rad_s is None:
        return float(core.wrap_angle(raw))
    maximum_step = float(rate_rad_s) * core.DT
    change = float(core.wrap_angle(raw - previous))
    return float(core.wrap_angle(previous + np.clip(change, -maximum_step, maximum_step)))


def acceleration_for_path(model_name: str, speed: float, leader_speed: float, arc_spacing: float) -> float:
    if model_name == "T-IDM":
        value = core.t_idm_acceleration(speed, leader_speed, arc_spacing)
    elif model_name == "T-FVD":
        value = core.t_fvd_acceleration(speed, leader_speed, arc_spacing)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    value = float(np.clip(value, *core.ACCELERATION_LIMITS_MPS2))
    return max(value, -speed / core.DT)


def steering_from_heading_step(current: float, following: float, speed: float) -> float:
    heading_rate = float(core.wrap_angle(following - current)) / core.DT
    reference_speed = max(float(speed), 0.5)
    front_wheel = math.atan2(WHEELBASE_M * heading_rate, reference_speed)
    return float(STEERING_RATIO * front_wheel)


def steering_history_from_simulation(simulation: pd.DataFrame) -> np.ndarray:
    headings = simulation["heading_sim"].to_numpy(dtype=float)
    speeds = simulation["velocity_sim"].to_numpy(dtype=float)
    values = np.full(len(simulation), np.nan)
    for index in range(len(simulation) - 1):
        values[index] = steering_from_heading_step(
            headings[index], headings[index + 1], speeds[index]
        )
    return values


def branch_history(pair: pd.DataFrame, simulation: pd.DataFrame) -> np.ndarray:
    dx = pair["carCenterXm_leader"].to_numpy(float) - simulation["x_sim"].to_numpy(float)
    dy = pair["carCenterYm_leader"].to_numpy(float) - simulation["y_sim"].to_numpy(float)
    return np.where(np.hypot(dx, dy) <= BASELINE_SC_M, "line-of-sight", "leader-heading")


def simulate_pair_limited(
    pair: pd.DataFrame,
    model_name: str,
    rate_rad_s: float | None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    if rate_rad_s is None:
        simulation, observed_arc_spacing, _ = core.simulate_pair(pair, model_name)
        commands = steering_history_from_simulation(simulation)
        branches = branch_history(pair, simulation)
        audit = pd.DataFrame(
            {
                "raw_target_heading_rad": np.nan,
                "limited_target_heading_rad": np.nan,
                "branch": branches,
                "steering_wheel_command_rad": commands,
            }
        )
        return simulation, observed_arc_spacing, audit

    n = len(pair)
    simulation = pd.DataFrame(
        {
            "frameNum": pair["frameNum"].to_numpy(),
            "x_sim": np.nan,
            "y_sim": np.nan,
            "heading_sim": np.nan,
            "velocity_sim": np.nan,
            "acceleration_sim": np.nan,
            "arc_spacing_sim": np.nan,
        }
    )
    simulation.loc[0, ["x_sim", "y_sim", "heading_sim", "velocity_sim"]] = [
        pair.loc[0, "carCenterXm_follower"],
        pair.loc[0, "carCenterYm_follower"],
        pair.loc[0, "heading_follower"],
        pair.loc[0, "velocity_follower"],
    ]
    observed_arc_spacing = np.full(n, np.nan)
    raw_targets = np.full(n, np.nan)
    filtered_targets = np.full(n, np.nan)
    branches = np.full(n, "", dtype=object)
    commands = np.full(n, np.nan)
    previous_target: float | None = None

    for index in range(n):
        observed_path = core.build_planned_path(
            pair.loc[index, "carCenterXm_follower"],
            pair.loc[index, "carCenterYm_follower"],
            pair.loc[index, "heading_follower"],
            pair.loc[index, "carCenterXm_leader"],
            pair.loc[index, "carCenterYm_leader"],
            pair.loc[index, "heading_leader"],
        )
        observed_arc_spacing[index] = observed_path.arc_spacing_m

        follower_x = float(simulation.loc[index, "x_sim"])
        follower_y = float(simulation.loc[index, "y_sim"])
        follower_heading = float(simulation.loc[index, "heading_sim"])
        follower_speed = float(simulation.loc[index, "velocity_sim"])
        leader_x = float(pair.loc[index, "carCenterXm_leader"])
        leader_y = float(pair.loc[index, "carCenterYm_leader"])
        leader_heading = float(pair.loc[index, "heading_leader"])
        leader_speed = float(pair.loc[index, "velocity_leader"])

        raw_target, branch, _ = raw_target_heading(
            follower_x, follower_y, leader_x, leader_y, leader_heading
        )
        target = limited_target(previous_target, raw_target, rate_rad_s)
        raw_targets[index] = raw_target
        filtered_targets[index] = target
        branches[index] = branch
        previous_target = target

        path = build_path_with_target(
            follower_x,
            follower_y,
            follower_heading,
            leader_x,
            leader_y,
            leader_heading,
            target,
        )
        simulation.loc[index, "arc_spacing_sim"] = path.arc_spacing_m
        if index == n - 1:
            break

        acceleration = acceleration_for_path(
            model_name, follower_speed, leader_speed, path.arc_spacing_m
        )
        next_speed = max(0.0, follower_speed + acceleration * core.DT)
        travel_distance = max(
            follower_speed * core.DT + 0.5 * acceleration * core.DT**2, 0.0
        )
        x_next, y_next, heading_change = core.advance_on_path(path, travel_distance)
        heading_next = float(core.wrap_angle(follower_heading + heading_change))
        commands[index] = steering_from_heading_step(
            follower_heading, heading_next, follower_speed
        )
        simulation.loc[index, "acceleration_sim"] = acceleration
        simulation.loc[index + 1, ["x_sim", "y_sim", "heading_sim", "velocity_sim"]] = [
            x_next,
            y_next,
            heading_next,
            next_speed,
        ]

    audit = pd.DataFrame(
        {
            "raw_target_heading_rad": raw_targets,
            "limited_target_heading_rad": filtered_targets,
            "branch": branches,
            "steering_wheel_command_rad": commands,
        }
    )
    return simulation, observed_arc_spacing, audit


def load_hefei_trajectories(path: Path) -> list[tuple[str, pd.DataFrame]]:
    data = pd.read_csv(path)
    required = {
        "trajectory_id",
        "frame",
        "x1_m",
        "y1_m",
        "x2_m",
        "y2_m",
        "v1_mps",
        "v2_mps",
        "a1_mps2",
        "a2_mps2",
        "c1_rad",
        "c2_rad",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Hefei input is missing columns: {sorted(missing)}")
    trajectories: list[tuple[str, pd.DataFrame]] = []
    for trajectory_id, group in data.groupby("trajectory_id", sort=True):
        group = group.sort_values("frame").reset_index(drop=True)
        pair = pd.DataFrame(
            {
                "frameNum": group["frame"].astype(int),
                "carCenterXm_leader": group["x1_m"].astype(float),
                "carCenterYm_leader": group["y1_m"].astype(float),
                "carCenterXm_follower": group["x2_m"].astype(float),
                "carCenterYm_follower": group["y2_m"].astype(float),
                "velocity_leader": group["v1_mps"].astype(float),
                "velocity_follower": group["v2_mps"].astype(float),
                "acceleration_leader": group["a1_mps2"].astype(float),
                "acceleration_follower": group["a2_mps2"].astype(float),
                "heading_leader": group["c1_rad"].astype(float),
                "heading_follower": group["c2_rad"].astype(float),
            }
        )
        trajectories.append((str(trajectory_id), pair))
    if len(trajectories) != 13:
        raise RuntimeError(f"Expected 13 Hefei trajectories, found {len(trajectories)}")
    return trajectories


def load_citysim_trajectories(
    tracks_path: Path, manifest_path: Path
) -> list[tuple[str, pd.DataFrame]]:
    _, tracks = core.load_tracks(tracks_path)
    manifest = pd.read_csv(manifest_path)
    trajectories = []
    for _, selection in manifest.iterrows():
        pair = core.prepare_pair_data(selection, tracks)
        trajectories.append((f"CitySim_{int(selection['trajectory_no']):02d}", pair))
    if len(trajectories) != 38:
        raise RuntimeError(f"Expected 38 CitySim trajectories, found {len(trajectories)}")
    return trajectories


def summarize_command_history(audit: pd.DataFrame) -> dict[str, float | int]:
    omega = audit["steering_wheel_command_rad"].to_numpy(float)
    valid = np.isfinite(omega)
    omega = omega[valid]
    if len(omega) < 2:
        return {
            "steering_step_count": 0,
            "steering_step_mean_rad": math.nan,
            "steering_step_p95_rad": math.nan,
            "steering_step_max_rad": math.nan,
        }
    steps = np.abs(np.diff(omega))
    return {
        "steering_step_count": int(len(steps)),
        "steering_step_mean_rad": float(np.mean(steps)),
        "steering_step_p95_rad": float(np.percentile(steps, 95)),
        "steering_step_max_rad": float(np.max(steps)),
    }


def branch_crossing_steps(audit: pd.DataFrame) -> np.ndarray:
    omega = audit["steering_wheel_command_rad"].to_numpy(float)
    branches = audit["branch"].astype(str).to_numpy()
    values = []
    for index in range(1, len(audit) - 1):
        if branches[index] != branches[index - 1] and np.isfinite(omega[index]) and np.isfinite(omega[index - 1]):
            values.append(abs(omega[index] - omega[index - 1]))
    return np.asarray(values, dtype=float)


def run_dataset(
    dataset: str,
    trajectories: list[tuple[str, pd.DataFrame]],
    dt_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    configure_table1(dt_s)
    metric_rows: list[dict[str, Any]] = []
    smoothness_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    configurations: tuple[float | None, ...] = (None, *RATE_LIMITS_RAD_S)

    for rate in configurations:
        label = "hard switch" if rate is None else f"{rate:g} rad/s"
        for model_name in core.MODEL_NAMES:
            crossing_values: list[float] = []
            for trajectory_id, pair in trajectories:
                simulation, observed_arc, audit = simulate_pair_limited(
                    pair, model_name, rate
                )
                metrics = evaluate_all_indices(pair, simulation, observed_arc)
                for variable, unit, key in VARIABLES:
                    for index in INDICES:
                        metric_rows.append(
                            {
                                "dataset": dataset,
                                "trajectory_id": trajectory_id,
                                "model": model_name,
                                "limiter": label,
                                "target_heading_rate_limit_rad_s": rate,
                                "variable": variable,
                                "unit": unit,
                                "evaluation_index": index,
                                "value": metrics[(key, index)],
                            }
                        )
                command_stats = summarize_command_history(audit)
                crossings = branch_crossing_steps(audit)
                crossing_values.extend(crossings.tolist())
                smoothness_rows.append(
                    {
                        "dataset": dataset,
                        "trajectory_id": trajectory_id,
                        "model": model_name,
                        "limiter": label,
                        "target_heading_rate_limit_rad_s": rate,
                        **command_stats,
                        "branch_crossing_count": int(len(crossings)),
                        "branch_crossing_step_mean_rad": (
                            float(np.mean(crossings)) if len(crossings) else math.nan
                        ),
                        "branch_crossing_step_max_rad": (
                            float(np.max(crossings)) if len(crossings) else math.nan
                        ),
                    }
                )
                if len(crossings) and len(example_rows) < 10000:
                    local = pd.DataFrame(
                        {
                            "dataset": dataset,
                            "trajectory_id": trajectory_id,
                            "model": model_name,
                            "limiter": label,
                            "target_heading_rate_limit_rad_s": rate,
                            "frame_index": np.arange(len(audit)),
                            "time_s": np.arange(len(audit)) * dt_s,
                            "branch": audit["branch"],
                            "raw_target_heading_rad": audit["raw_target_heading_rad"],
                            "limited_target_heading_rad": audit["limited_target_heading_rad"],
                            "steering_wheel_command_rad": audit["steering_wheel_command_rad"],
                        }
                    )
                    example_rows.extend(local.to_dict("records"))
            print(
                f"{dataset}: {model_name}, {label}, trajectories={len(trajectories)}, "
                f"crossings={len(crossing_values)}",
                flush=True,
            )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(smoothness_rows),
        pd.DataFrame(example_rows),
    )


def aggregate_metrics(per_trajectory: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset",
        "model",
        "limiter",
        "target_heading_rate_limit_rad_s",
        "variable",
        "unit",
        "evaluation_index",
    ]
    summary = (
        per_trajectory.groupby(keys, dropna=False, sort=False)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "equal_trajectory_mean",
                "std": "between_trajectory_sd",
                "count": "number_of_trajectories",
            }
        )
    )
    baseline = summary[summary["limiter"].eq("hard switch")][
        ["dataset", "model", "variable", "evaluation_index", "equal_trajectory_mean"]
    ].rename(columns={"equal_trajectory_mean": "hard_switch_mean"})
    summary = summary.merge(
        baseline,
        on=["dataset", "model", "variable", "evaluation_index"],
        how="left",
        validate="many_to_one",
    )
    summary["relative_change_percent"] = 100.0 * (
        summary["equal_trajectory_mean"] / summary["hard_switch_mean"] - 1.0
    )
    return summary


def aggregate_smoothness(per_trajectory: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "model", "limiter", "target_heading_rate_limit_rad_s"]
    rows = []
    for group_key, group in per_trajectory.groupby(keys, dropna=False, sort=False):
        dataset, model, limiter, rate = group_key
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "limiter": limiter,
                "target_heading_rate_limit_rad_s": rate,
                "equal_trajectory_mean_steering_step_rad": float(
                    group["steering_step_mean_rad"].mean()
                ),
                "equal_trajectory_mean_p95_steering_step_rad": float(
                    group["steering_step_p95_rad"].mean()
                ),
                "maximum_steering_step_rad": float(group["steering_step_max_rad"].max()),
                "total_branch_crossings": int(group["branch_crossing_count"].sum()),
                "mean_crossing_step_rad": float(
                    np.average(
                        group["branch_crossing_step_mean_rad"].fillna(0.0),
                        weights=group["branch_crossing_count"],
                    )
                )
                if int(group["branch_crossing_count"].sum())
                else math.nan,
                "maximum_crossing_step_rad": float(
                    group["branch_crossing_step_max_rad"].max()
                ),
            }
        )
    result = pd.DataFrame(rows)
    baseline = result[result["limiter"].eq("hard switch")][
        [
            "dataset",
            "model",
            "equal_trajectory_mean_p95_steering_step_rad",
            "maximum_steering_step_rad",
            "mean_crossing_step_rad",
            "maximum_crossing_step_rad",
        ]
    ].rename(
        columns={
            "equal_trajectory_mean_p95_steering_step_rad": "hard_switch_mean_p95_step_rad",
            "maximum_steering_step_rad": "hard_switch_max_step_rad",
            "mean_crossing_step_rad": "hard_switch_mean_crossing_step_rad",
            "maximum_crossing_step_rad": "hard_switch_max_crossing_step_rad",
        }
    )
    result = result.merge(baseline, on=["dataset", "model"], how="left")
    result["p95_step_reduction_percent"] = 100.0 * (
        1.0
        - result["equal_trajectory_mean_p95_steering_step_rad"]
        / result["hard_switch_mean_p95_step_rad"]
    )
    result["max_step_reduction_percent"] = 100.0 * (
        1.0 - result["maximum_steering_step_rad"] / result["hard_switch_max_step_rad"]
    )
    result["mean_crossing_step_reduction_percent"] = 100.0 * (
        1.0
        - result["mean_crossing_step_rad"]
        / result["hard_switch_mean_crossing_step_rad"]
    )
    result["max_crossing_step_reduction_percent"] = 100.0 * (
        1.0
        - result["maximum_crossing_step_rad"]
        / result["hard_switch_max_crossing_step_rad"]
    )
    return result


def make_compact_table(metric_summary: pd.DataFrame, smoothness: pd.DataFrame) -> pd.DataFrame:
    rmse = metric_summary[metric_summary["evaluation_index"].eq("RMSE")]
    identity = ["dataset", "model", "limiter"]
    rates = rmse[identity + ["target_heading_rate_limit_rad_s"]].drop_duplicates(identity)
    means = rmse.pivot(
        index=identity,
        columns="variable",
        values="equal_trajectory_mean",
    ).rename(columns=lambda value: f"equal_trajectory_mean_{value.lower().replace(' ', '_')}")
    changes = rmse.pivot(
        index=identity,
        columns="variable",
        values="relative_change_percent",
    ).rename(columns=lambda value: f"relative_change_percent_{value.lower().replace(' ', '_')}")
    wide = means.join(changes).reset_index().merge(
        rates, on=identity, how="left", validate="one_to_one"
    )
    columns = [
        "dataset",
        "model",
        "limiter",
        "target_heading_rate_limit_rad_s",
        "equal_trajectory_mean_heading_angle",
        "relative_change_percent_heading_angle",
        "equal_trajectory_mean_arc_spacing",
        "relative_change_percent_arc_spacing",
        "equal_trajectory_mean_velocity",
        "relative_change_percent_velocity",
        "equal_trajectory_mean_acceleration",
        "relative_change_percent_acceleration",
    ]
    return wide[columns].merge(
        smoothness,
        on=["dataset", "model", "limiter"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_smoothness"),
    )


def choose_rate(compact: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    candidates = compact[compact["limiter"].ne("hard switch")].copy()
    metric_change_columns = [
        "relative_change_percent_heading_angle",
        "relative_change_percent_arc_spacing",
        "relative_change_percent_velocity",
        "relative_change_percent_acceleration",
    ]
    grouped = candidates.groupby("target_heading_rate_limit_rad_s", sort=True)
    rows = []
    for rate, group in grouped:
        rows.append(
            {
                "target_heading_rate_limit_rad_s": float(rate),
                "worst_absolute_rmse_change_percent": float(
                    group[metric_change_columns].abs().to_numpy().max()
                ),
                "median_absolute_rmse_change_percent": float(
                    np.median(group[metric_change_columns].abs().to_numpy())
                ),
                "median_p95_step_reduction_percent": float(
                    group["p95_step_reduction_percent"].median()
                ),
                "minimum_p95_step_reduction_percent": float(
                    group["p95_step_reduction_percent"].min()
                ),
                "median_max_step_reduction_percent": float(
                    group["max_step_reduction_percent"].median()
                ),
                "minimum_max_step_reduction_percent": float(
                    group["max_step_reduction_percent"].min()
                ),
                "median_mean_crossing_step_reduction_percent": float(
                    group["mean_crossing_step_reduction_percent"].median()
                ),
                "minimum_mean_crossing_step_reduction_percent": float(
                    group["mean_crossing_step_reduction_percent"].min()
                ),
            }
        )
    decision = pd.DataFrame(rows)
    feasible = decision[
        (decision["worst_absolute_rmse_change_percent"] <= 2.0)
        & (decision["minimum_max_step_reduction_percent"] >= 40.0)
    ]
    if len(feasible):
        selected = feasible.sort_values(
            ["median_max_step_reduction_percent", "worst_absolute_rmse_change_percent"],
            ascending=[False, True],
        ).iloc[0]
    else:
        selected = decision.sort_values(
            ["worst_absolute_rmse_change_percent", "median_max_step_reduction_percent"],
            ascending=[True, False],
        ).iloc[0]
    return float(selected["target_heading_rate_limit_rad_s"]), decision


def validate_baseline(metric_summary: pd.DataFrame) -> dict[str, Any]:
    baseline = metric_summary[
        metric_summary["limiter"].eq("hard switch")
        & metric_summary["evaluation_index"].eq("RMSE")
    ]
    rows = []
    for key, expected in EXPECTED_BASELINE_RMSE.items():
        dataset, model, variable = key
        match = baseline[
            baseline["dataset"].eq(dataset)
            & baseline["model"].eq(model)
            & baseline["variable"].eq(variable)
        ]
        if len(match) != 1:
            raise RuntimeError(f"Baseline lookup failed for {key}: found {len(match)} rows")
        actual = float(match.iloc[0]["equal_trajectory_mean"])
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "variable": variable,
                "expected": expected,
                "actual": actual,
                "absolute_difference": abs(actual - expected),
            }
        )
    maximum_difference = max(row["absolute_difference"] for row in rows)
    if maximum_difference > 1e-10:
        raise RuntimeError(
            f"Hard-switch baseline does not reproduce the archived tables: {maximum_difference}"
        )
    return {
        "status": "pass",
        "tolerance": 1e-10,
        "maximum_absolute_difference": maximum_difference,
        "comparisons": rows,
    }


def selected_effect_tables(
    metric_summary: pd.DataFrame,
    smoothness: pd.DataFrame,
    selected_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hard = metric_summary[metric_summary["limiter"].eq("hard switch")][
        [
            "dataset",
            "model",
            "variable",
            "unit",
            "evaluation_index",
            "equal_trajectory_mean",
        ]
    ].rename(columns={"equal_trajectory_mean": "hard_switch_value"})
    limited = metric_summary[
        np.isclose(metric_summary["target_heading_rate_limit_rad_s"], selected_rate)
    ][
        [
            "dataset",
            "model",
            "variable",
            "unit",
            "evaluation_index",
            "equal_trajectory_mean",
            "relative_change_percent",
        ]
    ].rename(columns={"equal_trajectory_mean": "rate_limited_value"})
    all_indices = hard.merge(
        limited,
        on=["dataset", "model", "variable", "unit", "evaluation_index"],
        how="inner",
        validate="one_to_one",
    )
    rmse = all_indices[all_indices["evaluation_index"].eq("RMSE")].reset_index(drop=True)

    hard_smooth = smoothness[smoothness["limiter"].eq("hard switch")][
        [
            "dataset",
            "model",
            "maximum_steering_step_rad",
            "mean_crossing_step_rad",
            "maximum_crossing_step_rad",
        ]
    ].rename(
        columns={
            "maximum_steering_step_rad": "hard_switch_maximum_step_rad",
            "mean_crossing_step_rad": "hard_switch_mean_crossing_step_rad",
            "maximum_crossing_step_rad": "hard_switch_max_crossing_step_rad",
        }
    )
    limited_smooth = smoothness[
        np.isclose(smoothness["target_heading_rate_limit_rad_s"], selected_rate)
    ][
        [
            "dataset",
            "model",
            "maximum_steering_step_rad",
            "mean_crossing_step_rad",
            "maximum_crossing_step_rad",
            "max_step_reduction_percent",
            "mean_crossing_step_reduction_percent",
            "max_crossing_step_reduction_percent",
        ]
    ].rename(
        columns={
            "maximum_steering_step_rad": "rate_limited_maximum_step_rad",
            "mean_crossing_step_rad": "rate_limited_mean_crossing_step_rad",
            "maximum_crossing_step_rad": "rate_limited_max_crossing_step_rad",
        }
    )
    smooth_table = hard_smooth.merge(
        limited_smooth,
        on=["dataset", "model"],
        how="inner",
        validate="one_to_one",
    )
    return all_indices, rmse, smooth_table


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": 9,
        }
    )


def plot_tradeoff(decision: pd.DataFrame, selected_rate: float, output: Path) -> None:
    apply_plot_style()
    fig, axis = plt.subplots(figsize=(5.6, 3.5))
    axis.plot(
        decision["worst_absolute_rmse_change_percent"],
        decision["median_max_step_reduction_percent"],
        color="#2F5597",
        marker="o",
        linewidth=1.3,
    )
    for _, row in decision.iterrows():
        rate = float(row["target_heading_rate_limit_rad_s"])
        if np.isclose(rate, selected_rate):
            offset = (-68, -16)
        else:
            offset = (4, 4)
        axis.annotate(
            f"{rate:g} rad/s",
            (
                row["worst_absolute_rmse_change_percent"],
                row["median_max_step_reduction_percent"],
            ),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            fontname="Times New Roman",
        )
    selected = decision[np.isclose(decision["target_heading_rate_limit_rad_s"], selected_rate)].iloc[0]
    axis.scatter(
        selected["worst_absolute_rmse_change_percent"],
        selected["median_max_step_reduction_percent"],
        s=70,
        facecolors="none",
        edgecolors="#C00000",
        linewidths=1.4,
        label="Selected diagnostic setting",
        zorder=4,
    )
    axis.set_xlabel("Worst absolute change in Tables 2-3 RMSE metrics (%)")
    axis.set_ylabel("Median reduction in maximum steering-command step (%)")
    axis.grid(True, color="#D9D9D9", linewidth=0.6)
    axis.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_transition_example(examples: pd.DataFrame, output: Path) -> None:
    query = examples[
        examples["dataset"].eq("Hefei Table 2")
        & examples["model"].eq("T-FVD")
        & examples["trajectory_id"].eq("Hefei_13")
        & examples["limiter"].isin(["hard switch", "0.5 rad/s"])
    ].copy()
    if set(query["limiter"]) != {"hard switch", "0.5 rad/s"}:
        return
    query = query[query["time_s"].between(4.6, 5.7)]
    apply_plot_style()
    fig, axis = plt.subplots(figsize=(5.7, 3.2))
    styles = {
        "hard switch": ("#C00000", "--", "Hard switch"),
        "0.5 rad/s": ("#2F5597", "-", "0.5-rad/s target-heading limiter"),
    }
    for limiter, (color, linestyle, label) in styles.items():
        part = query[query["limiter"].eq(limiter)].sort_values("time_s")
        axis.plot(
            part["time_s"],
            part["steering_wheel_command_rad"],
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            marker="o",
            markersize=2.8,
            label=label,
        )
    axis.axvline(5.1, color="#666666", linewidth=0.8, linestyle=":")
    axis.annotate(
        "Branch crossing",
        xy=(5.1, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(4, -5),
        textcoords="offset points",
        rotation=90,
        va="top",
        ha="left",
        fontsize=8,
    )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel(r"Steering command $\omega(t)$ (rad)")
    axis.grid(True, color="#D9D9D9", linewidth=0.6)
    axis.legend(frameon=False, loc="best", fontsize=8.5)
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.19, top=0.98)
    fig.savefig(output, dpi=600)
    plt.close(fig)


def format_relative_change_percent(value: float) -> str:
    """Keep small nonzero changes distinct from exact zero in display output."""
    value = float(value)
    if value == 0.0:
        return "0.0000%"
    if abs(value) < 0.0001:
        return "negligible"
    return f"{value:+.4f}%"


def write_report(
    selected_rate: float,
    decision: pd.DataFrame,
    compact: pd.DataFrame,
    output: Path,
) -> None:
    selected = compact[np.isclose(compact["target_heading_rate_limit_rad_s"], selected_rate)]
    decision_row = decision[
        np.isclose(decision["target_heading_rate_limit_rad_s"], selected_rate)
    ].iloc[0]
    lines = [
        "# Rate-limiter audit summary",
        "",
        f"Selected diagnostic target-heading slew-rate limit: {selected_rate:g} rad/s.",
        f"Worst absolute change across the 16 dataset/model/RMSE comparisons: {decision_row['worst_absolute_rmse_change_percent']:.4f}%.",
        f"Median reduction in the maximum steering-command step: {decision_row['median_max_step_reduction_percent']:.2f}%.",
        "",
        "Relative changes use unrounded values and are displayed to four decimal places. "
        "Absolute relative changes below 0.0001% are labeled negligible; this does not imply exact equality.",
        "",
        "## Selected-setting results",
        "",
    ]
    for _, row in selected.iterrows():
        lines.append(
            f"- {row['dataset']} / {row['model']}: heading {row['equal_trajectory_mean_heading_angle']:.6f} rad "
            f"({format_relative_change_percent(row['relative_change_percent_heading_angle'])}), spacing {row['equal_trajectory_mean_arc_spacing']:.6f} m "
            f"({format_relative_change_percent(row['relative_change_percent_arc_spacing'])}), velocity {row['equal_trajectory_mean_velocity']:.6f} m/s "
            f"({format_relative_change_percent(row['relative_change_percent_velocity'])}), acceleration {row['equal_trajectory_mean_acceleration']:.6f} m/s^2 "
            f"({format_relative_change_percent(row['relative_change_percent_acceleration'])}), maximum steering-step reduction "
            f"{row['max_step_reduction_percent']:.2f}%, mean crossing-step reduction "
            f"{row['mean_crossing_step_reduction_percent']:.2f}%."
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--citysim",
        type=Path,
        default=DEFAULT_CITYSIM_INPUT,
        help="CitySim IntersectionA_cleaned.csv (default: external_data/...).",
    )
    parser.add_argument(
        "--citysim-manifest",
        type=Path,
        default=DEFAULT_CITYSIM_MANIFEST,
        help="Frozen 38-pair manifest distributed with the repository.",
    )
    parser.add_argument(
        "--hefei-private",
        type=Path,
        default=DEFAULT_HEFEI_INPUT,
        help="Authorized normalized 13-trajectory Hefei input; not redistributed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory (default: reproduced_outputs/rate_limiter).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    missing = [
        path
        for path in (args.citysim, args.citysim_manifest, args.hefei_private)
        if not path.is_file()
    ]
    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            "Required input file(s) not found:\n"
            f"{rendered}\n"
            "See README.md, section 'Table C14 and Fig. C9 rate-limiter audit'."
        )
    args.output.mkdir(parents=True, exist_ok=True)

    hefei = load_hefei_trajectories(args.hefei_private)
    configure_table1(1.0 / 30.0)
    citysim = load_citysim_trajectories(args.citysim, args.citysim_manifest)

    h_metrics, h_smooth, h_examples = run_dataset("Hefei Table 2", hefei, 0.1)
    c_metrics, c_smooth, c_examples = run_dataset(
        "CitySim Table 3", citysim, 1.0 / 30.0
    )
    per_metric = pd.concat([h_metrics, c_metrics], ignore_index=True)
    per_smooth = pd.concat([h_smooth, c_smooth], ignore_index=True)
    examples = pd.concat([h_examples, c_examples], ignore_index=True)
    summary = aggregate_metrics(per_metric)
    smoothness = aggregate_smoothness(per_smooth)
    compact = make_compact_table(summary, smoothness)
    selected_rate, decision = choose_rate(compact)
    baseline_validation = validate_baseline(summary)
    selected_all, selected_rmse, selected_smoothness = selected_effect_tables(
        summary, smoothness, selected_rate
    )

    per_metric.to_csv(
        args.output / "rate_limiter_per_trajectory_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    per_smooth.to_csv(
        args.output / "rate_limiter_per_trajectory_smoothness.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        args.output / "rate_limiter_all_table2_table3_indices.csv",
        index=False,
        encoding="utf-8-sig",
    )
    smoothness.to_csv(
        args.output / "rate_limiter_smoothness_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    compact.to_csv(
        args.output / "rate_limiter_compact_rmse_table.csv",
        index=False,
        encoding="utf-8-sig",
    )
    decision.to_csv(
        args.output / "rate_limiter_setting_decision.csv",
        index=False,
        encoding="utf-8-sig",
    )
    examples.to_csv(
        args.output / "rate_limiter_crossing_timeseries.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected_all.to_csv(
        args.output / "Table_rate_limiter_all_indices_selected.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected_rmse.to_csv(
        args.output / "Table_rate_limiter_RMSE_effect.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected_smoothness.to_csv(
        args.output / "Table_rate_limiter_smoothness_effect.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output / "baseline_reproduction_validation.json").write_text(
        json.dumps(baseline_validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_tradeoff(decision, selected_rate, args.output / "Fig_rate_limiter_tradeoff.png")
    plot_transition_example(examples, args.output / "Fig_rate_limiter_transition.png")
    write_report(selected_rate, decision, compact, args.output / "rate_limiter_summary.md")

    metadata = {
        "experiment": "Closed-loop target-heading slew-rate audit",
        "limiter_location": "selected target heading before cubic-path construction",
        "initialization": "first target heading initialized to the raw selected target",
        "rates_rad_s": list(RATE_LIMITS_RAD_S),
        "selected_rate_rad_s": selected_rate,
        "baseline": "archived hard-switch simulator called directly",
        "fixed_parameter_source": "Table 1",
        "hefei_input": {
            "path": portable_reference(args.hefei_private),
            "sha256": file_sha256(args.hefei_private),
            "trajectory_count": len(hefei),
            "dt_s": 0.1,
        },
        "citysim_input": {
            "path": portable_reference(args.citysim),
            "sha256": file_sha256(args.citysim),
            "manifest": portable_reference(args.citysim_manifest),
            "manifest_sha256": file_sha256(args.citysim_manifest),
            "trajectory_count": len(citysim),
            "dt_s": 1.0 / 30.0,
        },
        "metric_aggregation": "equal arithmetic mean of trajectory-level indices",
        "mape_floors": MAPE_FLOORS,
        "figure": "Times New Roman, 600 dpi PNG",
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output / "rate_limiter_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Selected rate: {selected_rate:g} rad/s", flush=True)
    print(decision.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
