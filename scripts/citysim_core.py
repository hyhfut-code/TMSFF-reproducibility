"""Multi-trajectory external validation of TMSFF on CitySim Intersection A.

This script broadens the original single-trajectory CitySim check without
recalibrating on CitySim.  The T-IDM and T-FVD parameters are fixed to their
respective seed-42 values obtained under the same Hefei calibration protocol
and reported in Appendix Table C2 of the revised manuscript.

The workflow is intentionally auditable:
1. identify complete turning trajectories from the cleaned CitySim file;
2. identify the immediate leader for each follower using geometry only;
3. retain every qualified immediate-leader/follower pair;
4. simulate TMSFF with both models under their fixed Hefei parameters;
5. report trajectory-level RMSE and the across-trajectory mean and sample SD.

A one-pair-per-movement route flag is also retained for a route-balanced
sensitivity check, but the primary result uses all qualified pairs.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FPS = 30.0
DT = 1.0 / FPS
VEHICLE_LENGTH_M = 5.0
SHORT_RANGE_THRESHOLD_M = 15.8
GAUSS_NODES, GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(24)

# Seed-42 T-IDM calibration under the common Hefei protocol; no CitySim
# recalibration.  The full-precision values are reported in Appendix Table C2.
HEFEI_TIDM_PARAMETERS = {'a_max_mps2': 2.81, 'b_comfort_mps2': 2.96, 'time_gap_s': 1.28, 'desired_speed_mps': 18.06, 'minimum_safe_arc_spacing_m': 4.81}

# Seed-42 T-FVD calibration obtained with the same Hefei data split, objective,
# optimizer, population size, generation count, and parameter bounds as T-IDM.
HEFEI_TFVD_PARAMETERS = {'kappa': 0.28, 'lambda_speed': 0.3, 'maximum_speed_mps': 10.9, 'safety_arc_spacing_m': 9.11}

PARAMETER_SOURCE = (
    "Respective seed-42 Hefei calibrations obtained under the common protocol "
    "and reported in Appendix Table C2"
)

MODEL_NAMES = ("T-IDM", "T-FVD")
ACCELERATION_LIMITS_MPS2 = (-5.0, 3.0)
SAVGOL_WINDOW_FRAMES = 31
SAVGOL_POLYNOMIAL_ORDER = 3


@dataclass(frozen=True)
class ScreeningProtocol:
    minimum_track_frames: int = 150
    maximum_track_frames: int = 1500
    minimum_moving_frames: int = 80
    minimum_displacement_m: float = 25.0
    minimum_full_turn_deg: float = 20.0
    maximum_full_turn_deg: float = 150.0
    minimum_overlap_frames: int = 150
    minimum_overlap_turn_deg: float = 15.0
    minimum_forward_share: float = 0.95
    minimum_heading_agreement_share: float = 0.95
    heading_agreement_limit_deg: float = 60.0
    minimum_median_spacing_m: float = 3.0
    maximum_median_spacing_m: float = 60.0
    maximum_spacing_p95_m: float = 100.0
    maximum_median_lateral_offset_m: float = 6.0


PROTOCOL = ScreeningProtocol()


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def circular_mean(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.angle(np.mean(np.exp(1j * array))))


def circular_rmse(simulated: np.ndarray, observed: np.ndarray) -> float:
    error = wrap_angle(np.asarray(simulated) - np.asarray(observed))
    return float(np.sqrt(np.mean(np.square(error))))


def rmse(simulated: np.ndarray, observed: np.ndarray) -> float:
    error = np.asarray(simulated, dtype=float) - np.asarray(observed, dtype=float)
    return float(np.sqrt(np.mean(np.square(error))))


def coordinate_system_transformation(
    x_follower: float,
    y_follower: float,
    angle_follower: float,
    x_leader: float,
    y_leader: float,
    angle_leader: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Appendix-B rotation used by the original TMSFF implementation."""
    angle_follower = float(wrap_angle(angle_follower))
    angle_leader = float(wrap_angle(angle_leader))
    angle_gap = abs(angle_follower - angle_leader)
    tiny_angle = 0.005

    if abs(angle_gap / 2.0 - np.pi / 2.0) <= tiny_angle:
        turning_angle = (angle_follower + angle_leader) / 2.0 - np.pi
        if angle_follower >= angle_leader:
            new_angle_follower = angle_gap / 2.0 + tiny_angle - np.pi
            new_angle_leader = np.pi - angle_gap / 2.0 - tiny_angle
        else:
            new_angle_follower = np.pi - angle_gap / 2.0 - tiny_angle
            new_angle_leader = angle_gap / 2.0 + tiny_angle - np.pi
    elif angle_gap >= np.pi:
        turning_angle = (angle_follower + angle_leader) / 2.0 - np.pi
        if angle_follower >= angle_leader:
            new_angle_follower = angle_gap / 2.0 - np.pi
            new_angle_leader = np.pi - angle_gap / 2.0
        else:
            new_angle_follower = np.pi - angle_gap / 2.0
            new_angle_leader = angle_gap / 2.0 - np.pi
    else:
        turning_angle = (angle_follower + angle_leader) / 2.0
        if angle_follower >= angle_leader:
            new_angle_follower = angle_gap / 2.0
            new_angle_leader = -angle_gap / 2.0
        else:
            new_angle_follower = -angle_gap / 2.0
            new_angle_leader = angle_gap / 2.0

    rotation = np.array(
        [
            [np.cos(turning_angle), np.sin(turning_angle)],
            [-np.sin(turning_angle), np.cos(turning_angle)],
        ]
    )
    new_x_follower, new_y_follower = rotation @ np.array([x_follower, y_follower])
    new_x_leader, new_y_leader = rotation @ np.array([x_leader, y_leader])
    return (
        (float(new_x_follower), float(new_y_follower), float(new_angle_follower)),
        (float(new_x_leader), float(new_y_leader), float(new_angle_leader)),
        float(turning_angle),
    )


def coordinate_system_inversion(
    x_rotated: float,
    y_rotated: float,
    original_heading: float,
    heading_change: float,
    turning_angle: float,
) -> tuple[float, float, float]:
    rotation = np.array(
        [
            [np.cos(turning_angle), np.sin(turning_angle)],
            [-np.sin(turning_angle), np.cos(turning_angle)],
        ]
    )
    x_original, y_original = np.linalg.solve(rotation, np.array([x_rotated, y_rotated]))
    return (
        float(x_original),
        float(y_original),
        float(wrap_angle(original_heading + heading_change)),
    )


@dataclass
class PlannedPath:
    x_origin: float
    y_origin: float
    x_leader_local: float
    coefficients: np.ndarray
    follower_heading_rotated: float
    turning_angle: float
    arc_spacing_m: float
    euclidean_spacing_m: float

    def polynomial(self, x_local: float | np.ndarray) -> float | np.ndarray:
        return np.polyval(self.coefficients, x_local)

    def derivative(self, x_local: float | np.ndarray) -> float | np.ndarray:
        return np.polyval(np.polyder(self.coefficients), x_local)

    def arc_length(self, x_local: float) -> float:
        # Fixed-order Gauss-Legendre quadrature is accurate for the smooth
        # cubic path and keeps the validation script independent of SciPy.
        x_local = float(x_local)
        sample_x = 0.5 * x_local * (GAUSS_NODES + 1.0)
        integrand = np.sqrt(1.0 + np.asarray(self.derivative(sample_x)) ** 2)
        return float(0.5 * x_local * np.dot(GAUSS_WEIGHTS, integrand))


def build_planned_path(
    follower_x: float,
    follower_y: float,
    follower_heading: float,
    leader_x: float,
    leader_y: float,
    leader_heading: float,
) -> PlannedPath:
    follower, leader, turning_angle = coordinate_system_transformation(
        follower_x,
        follower_y,
        follower_heading,
        leader_x,
        leader_y,
        leader_heading,
    )
    euclidean = float(np.hypot(leader[0] - follower[0], leader[1] - follower[1]))
    target_heading = (
        math.atan2(leader[1] - follower[1], leader[0] - follower[0])
        if euclidean <= SHORT_RANGE_THRESHOLD_M
        else leader[2]
    )

    # Translation to a follower-centred local frame improves numerical stability
    # without changing the cubic boundary conditions used in the manuscript.
    delta_x = float(leader[0] - follower[0])
    delta_y = float(leader[1] - follower[1])
    if delta_x <= 1e-6:
        raise ValueError(f"Non-positive longitudinal separation after rotation: {delta_x}")

    slope_follower = math.tan(follower[2])
    slope_leader = math.tan(target_heading)
    matrix = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [delta_x**3, delta_x**2, delta_x, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [3.0 * delta_x**2, 2.0 * delta_x, 1.0, 0.0],
        ]
    )
    rhs = np.array([0.0, delta_y, slope_follower, slope_leader])
    try:
        coefficients = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(matrix, rhs, rcond=None)[0]

    path = PlannedPath(
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


def t_idm_acceleration(speed: float, leader_speed: float, arc_spacing: float) -> float:
    p = HEFEI_TIDM_PARAMETERS
    net_arc_gap = max(arc_spacing - VEHICLE_LENGTH_M, 0.5)
    closing_speed = speed - leader_speed
    desired_spacing = p["minimum_safe_arc_spacing_m"] + max(
        0.0,
        speed * p["time_gap_s"]
        + speed
        * closing_speed
        / (2.0 * math.sqrt(p["a_max_mps2"] * p["b_comfort_mps2"])),
    )
    return float(
        p["a_max_mps2"]
        * (
            1.0
            - (speed / p["desired_speed_mps"]) ** 4
            - (desired_spacing / net_arc_gap) ** 2
        )
    )


def t_fvd_acceleration(speed: float, leader_speed: float, arc_spacing: float) -> float:
    """T-FVD Eqs. (17)-(18) using arc spacing and the standard relative speed."""
    p = HEFEI_TFVD_PARAMETERS
    optimal_speed = 0.5 * p["maximum_speed_mps"] * (
        np.tanh(arc_spacing - p["safety_arc_spacing_m"])
        + np.tanh(p["safety_arc_spacing_m"])
    )
    return float(
        p["kappa"] * (optimal_speed - speed)
        + p["lambda_speed"] * (leader_speed - speed)
    )


def advance_on_path(path: PlannedPath, travel_distance: float) -> tuple[float, float, float]:
    travel_distance = max(float(travel_distance), 0.0)
    if travel_distance == 0.0:
        x_local = 0.0
    else:
        upper = min(max(path.x_leader_local, 1.0), 5.0)
        while path.arc_length(upper) < travel_distance:
            upper *= 1.5
            if upper > max(20.0, 2.0 * path.x_leader_local):
                raise ValueError("Unable to bracket the next TMSFF position")
        # Monotone bisection on arc length; 32 iterations are well beyond the
        # positional precision of the 30-Hz trajectory observations.
        lower = 0.0
        for _ in range(32):
            midpoint = 0.5 * (lower + upper)
            if path.arc_length(midpoint) < travel_distance:
                lower = midpoint
            else:
                upper = midpoint
        x_local = 0.5 * (lower + upper)

    y_local = float(path.polynomial(x_local))
    heading_rotated = math.atan(float(path.derivative(x_local)))
    heading_change = heading_rotated - path.follower_heading_rotated
    return coordinate_system_inversion(
        path.x_origin + x_local,
        path.y_origin + y_local,
        0.0,
        heading_rotated,
        path.turning_angle,
    )[:2] + (heading_change,)


def next_follower_state(
    model_name: str,
    follower_x: float,
    follower_y: float,
    follower_heading: float,
    follower_speed: float,
    leader_x: float,
    leader_y: float,
    leader_heading: float,
    leader_speed: float,
) -> tuple[float, float, float, float, float, float]:
    path = build_planned_path(
        follower_x,
        follower_y,
        follower_heading,
        leader_x,
        leader_y,
        leader_heading,
    )
    if model_name == "T-IDM":
        acceleration = t_idm_acceleration(follower_speed, leader_speed, path.arc_spacing_m)
    elif model_name == "T-FVD":
        acceleration = t_fvd_acceleration(follower_speed, leader_speed, path.arc_spacing_m)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Apply the same physical acceleration limits used during the common Hefei
    # calibration/holdout comparison, then retain the non-reversing constraint
    # for very low speeds.
    acceleration = float(np.clip(acceleration, *ACCELERATION_LIMITS_MPS2))
    acceleration = max(acceleration, -follower_speed / DT)
    next_speed = max(0.0, follower_speed + acceleration * DT)
    travel_distance = follower_speed * DT + 0.5 * acceleration * DT**2

    x_next_rotated, y_next_rotated, heading_change = advance_on_path(path, travel_distance)
    # advance_on_path returns the inverted position but the local heading change;
    # apply that change to the actual follower heading.
    next_heading = float(wrap_angle(follower_heading + heading_change))
    return (
        x_next_rotated,
        y_next_rotated,
        next_heading,
        next_speed,
        acceleration,
        path.arc_spacing_m,
    )


def load_tracks(input_csv: Path) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    data = pd.read_csv(input_csv)
    required = {
        "frameNum",
        "carId",
        "carCenterXm",
        "carCenterYm",
        "velocity",
        "acceleration",
        "heading",
        "laneId",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = data.sort_values(["carId", "frameNum"]).reset_index(drop=True)
    tracks = {int(vehicle_id): group.reset_index(drop=True) for vehicle_id, group in data.groupby("carId")}
    return data, tracks


def describe_tracks(tracks: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for vehicle_id, group in tracks.items():
        if not PROTOCOL.minimum_track_frames <= len(group) <= PROTOCOL.maximum_track_frames:
            continue
        moving = group[group["velocity"] > 1.0]
        if len(moving) < PROTOCOL.minimum_moving_frames:
            continue
        window = max(10, min(40, len(moving) // 5))
        heading_start = circular_mean(moving["heading"].iloc[:window])
        heading_end = circular_mean(moving["heading"].iloc[-window:])
        turn_deg = float(np.degrees(wrap_angle(heading_end - heading_start)))
        displacement = float(
            np.hypot(
                group["carCenterXm"].iloc[-1] - group["carCenterXm"].iloc[0],
                group["carCenterYm"].iloc[-1] - group["carCenterYm"].iloc[0],
            )
        )
        if displacement < PROTOCOL.minimum_displacement_m:
            continue
        if not PROTOCOL.minimum_full_turn_deg <= abs(turn_deg) <= PROTOCOL.maximum_full_turn_deg:
            continue
        rows.append(
            {
                "vehicle_id": vehicle_id,
                "frames": len(group),
                "start_frame": int(group["frameNum"].iloc[0]),
                "end_frame": int(group["frameNum"].iloc[-1]),
                "entry_lane": int(group["laneId"].iloc[0]),
                "exit_lane": int(group["laneId"].iloc[-1]),
                "full_turn_deg": turn_deg,
                "displacement_m": displacement,
            }
        )
    return pd.DataFrame(rows)


def assess_ordered_pair(
    leader_id: int,
    follower_id: int,
    tracks: dict[int, pd.DataFrame],
    entry_lane: int,
    exit_lane: int,
) -> dict[str, float | int] | None:
    leader = tracks[leader_id]
    follower = tracks[follower_id]
    start_frame = int(max(leader["frameNum"].iloc[0], follower["frameNum"].iloc[0]))
    end_frame = int(min(leader["frameNum"].iloc[-1], follower["frameNum"].iloc[-1]))
    if end_frame - start_frame + 1 < PROTOCOL.minimum_overlap_frames:
        return None

    overlap = follower[
        follower["frameNum"].between(start_frame, end_frame)
    ].merge(
        leader[leader["frameNum"].between(start_frame, end_frame)],
        on="frameNum",
        suffixes=("_follower", "_leader"),
    )
    if len(overlap) < PROTOCOL.minimum_overlap_frames:
        return None

    relative_x = overlap["carCenterXm_leader"] - overlap["carCenterXm_follower"]
    relative_y = overlap["carCenterYm_leader"] - overlap["carCenterYm_follower"]
    direction_x = np.cos(overlap["heading_follower"])
    direction_y = np.sin(overlap["heading_follower"])
    forward = relative_x * direction_x + relative_y * direction_y
    lateral = np.abs(-relative_x * direction_y + relative_y * direction_x)
    spacing = np.hypot(relative_x, relative_y)
    heading_difference = np.abs(
        np.degrees(wrap_angle(overlap["heading_leader"] - overlap["heading_follower"]))
    )
    moving_heading = overlap.loc[overlap["velocity_follower"] > 1.0, "heading_follower"]
    if len(moving_heading) < 40:
        return None
    overlap_turn = float(
        np.degrees(
            wrap_angle(
                circular_mean(moving_heading.tail(20)) - circular_mean(moving_heading.head(20))
            )
        )
    )

    row = {
        "leader_id": leader_id,
        "follower_id": follower_id,
        "entry_lane": entry_lane,
        "exit_lane": exit_lane,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "overlap_frames": len(overlap),
        "duration_s": len(overlap) / FPS,
        "overlap_turn_deg": overlap_turn,
        "leader_ahead_share": float(np.mean(forward > 0.0)),
        "heading_agreement_share": float(
            np.mean(heading_difference < PROTOCOL.heading_agreement_limit_deg)
        ),
        "median_spacing_m": float(np.median(spacing)),
        "spacing_p95_m": float(np.quantile(spacing, 0.95)),
        "median_lateral_offset_m": float(np.median(lateral)),
    }
    if abs(overlap_turn) < PROTOCOL.minimum_overlap_turn_deg:
        return None
    if row["leader_ahead_share"] < PROTOCOL.minimum_forward_share:
        return None
    if row["heading_agreement_share"] < PROTOCOL.minimum_heading_agreement_share:
        return None
    if not (
        PROTOCOL.minimum_median_spacing_m
        <= row["median_spacing_m"]
        <= PROTOCOL.maximum_median_spacing_m
    ):
        return None
    if row["spacing_p95_m"] > PROTOCOL.maximum_spacing_p95_m:
        return None
    if row["median_lateral_offset_m"] > PROTOCOL.maximum_median_lateral_offset_m:
        return None
    return row


def select_pairs(tracks: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = describe_tracks(tracks)
    eligible: list[dict[str, float | int]] = []

    for (entry_lane, exit_lane), route_group in metadata.groupby(["entry_lane", "exit_lane"]):
        vehicle_ids = route_group["vehicle_id"].astype(int).tolist()
        for follower_id in vehicle_ids:
            follower_candidates: list[dict[str, float | int]] = []
            for leader_id in vehicle_ids:
                if leader_id == follower_id:
                    continue
                candidate = assess_ordered_pair(
                    leader_id,
                    follower_id,
                    tracks,
                    int(entry_lane),
                    int(exit_lane),
                )
                if candidate is not None:
                    follower_candidates.append(candidate)
            if follower_candidates:
                # The closest eligible vehicle is the immediate leader.
                eligible.append(min(follower_candidates, key=lambda row: row["median_spacing_m"]))

    eligible_df = (
        pd.DataFrame(eligible)
        .drop_duplicates(["leader_id", "follower_id"])
        .sort_values(["entry_lane", "exit_lane", "start_frame", "follower_id"])
        .reset_index(drop=True)
    )
    if eligible_df.empty:
        raise RuntimeError("No CitySim pairs satisfy the pre-specified screening protocol")

    # Mark a route-balanced subset for sensitivity analysis.  The primary
    # validation below retains all eligible immediate-leader/follower pairs.
    route_balanced_indices: list[int] = []
    for _, group in eligible_df.groupby(["entry_lane", "exit_lane"], sort=True):
        best = group.sort_values(
            ["median_lateral_offset_m", "spacing_p95_m", "duration_s"],
            ascending=[True, True, False],
        ).iloc[0]
        route_balanced_indices.append(int(best.name))

    selected_df = eligible_df.copy()
    selected_df.insert(0, "trajectory_no", np.arange(1, len(selected_df) + 1))
    selected_df["route_balanced_subset"] = selected_df.index.isin(route_balanced_indices)
    selected_df["representative"] = (
        (selected_df["leader_id"] == 6) & (selected_df["follower_id"] == 59)
    )
    return eligible_df, selected_df


def prepare_pair_data(
    selection: pd.Series, tracks: dict[int, pd.DataFrame]
) -> pd.DataFrame:
    leader = tracks[int(selection["leader_id"])]
    follower = tracks[int(selection["follower_id"])]
    start_frame = int(selection["start_frame"])
    end_frame = int(selection["end_frame"])
    pair = follower[follower["frameNum"].between(start_frame, end_frame)].merge(
        leader[leader["frameNum"].between(start_frame, end_frame)],
        on="frameNum",
        suffixes=("_follower", "_leader"),
    )
    return pair.sort_values("frameNum").reset_index(drop=True)


def simulate_pair(
    pair: pd.DataFrame,
    model_name: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
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

    for index in range(n):
        observed_path = build_planned_path(
            pair.loc[index, "carCenterXm_follower"],
            pair.loc[index, "carCenterYm_follower"],
            pair.loc[index, "heading_follower"],
            pair.loc[index, "carCenterXm_leader"],
            pair.loc[index, "carCenterYm_leader"],
            pair.loc[index, "heading_leader"],
        )
        observed_arc_spacing[index] = observed_path.arc_spacing_m

        if index == n - 1:
            simulated_path = build_planned_path(
                simulation.loc[index, "x_sim"],
                simulation.loc[index, "y_sim"],
                simulation.loc[index, "heading_sim"],
                pair.loc[index, "carCenterXm_leader"],
                pair.loc[index, "carCenterYm_leader"],
                pair.loc[index, "heading_leader"],
            )
            simulation.loc[index, "arc_spacing_sim"] = simulated_path.arc_spacing_m
            break

        state = next_follower_state(
            model_name,
            simulation.loc[index, "x_sim"],
            simulation.loc[index, "y_sim"],
            simulation.loc[index, "heading_sim"],
            simulation.loc[index, "velocity_sim"],
            pair.loc[index, "carCenterXm_leader"],
            pair.loc[index, "carCenterYm_leader"],
            pair.loc[index, "heading_leader"],
            pair.loc[index, "velocity_leader"],
        )
        (
            x_next,
            y_next,
            heading_next,
            speed_next,
            acceleration,
            arc_spacing,
        ) = state
        simulation.loc[index, "acceleration_sim"] = acceleration
        simulation.loc[index, "arc_spacing_sim"] = arc_spacing
        simulation.loc[index + 1, ["x_sim", "y_sim", "heading_sim", "velocity_sim"]] = [
            x_next,
            y_next,
            heading_next,
            speed_next,
        ]

    return simulation, observed_arc_spacing, pair["acceleration_follower"].to_numpy()


def evaluate_pair(
    model_name: str,
    trajectory_no: int,
    selection: pd.Series,
    pair: pd.DataFrame,
    simulation: pd.DataFrame,
    observed_arc_spacing: np.ndarray,
) -> dict[str, float | int | bool]:
    dynamic_slice = slice(1, len(pair))
    acceleration_slice = slice(0, len(pair) - 1)
    return {
        "model": model_name,
        "trajectory_no": trajectory_no,
        "leader_id": int(selection["leader_id"]),
        "follower_id": int(selection["follower_id"]),
        "entry_lane": int(selection["entry_lane"]),
        "exit_lane": int(selection["exit_lane"]),
        "evaluation_frames": len(pair) - 1,
        "duration_s": float((len(pair) - 1) / FPS),
        "heading_rmse_rad": circular_rmse(
            simulation["heading_sim"].to_numpy()[dynamic_slice],
            pair["heading_follower"].to_numpy()[dynamic_slice],
        ),
        "arc_spacing_rmse_m": rmse(
            simulation["arc_spacing_sim"].to_numpy()[dynamic_slice],
            observed_arc_spacing[dynamic_slice],
        ),
        "velocity_rmse_mps": rmse(
            simulation["velocity_sim"].to_numpy()[dynamic_slice],
            pair["velocity_follower"].to_numpy()[dynamic_slice],
        ),
        "acceleration_rmse_mps2": rmse(
            simulation["acceleration_sim"].to_numpy()[acceleration_slice],
            pair["acceleration_follower"].to_numpy()[acceleration_slice],
        ),
        "route_balanced_subset": bool(selection["route_balanced_subset"]),
        "representative": bool(selection["representative"]),
    }


def make_summary(results: pd.DataFrame) -> pd.DataFrame:
    metric_rows = [
        ("Heading angle", "rad", "heading_rmse_rad"),
        ("Arc spacing", "m", "arc_spacing_rmse_m"),
        ("Velocity", "m/s", "velocity_rmse_mps"),
        ("Acceleration", "m/s²", "acceleration_rmse_mps2"),
    ]
    rows = []
    for model_name in MODEL_NAMES:
        model_results = results.loc[results["model"] == model_name]
        for metric, unit, column in metric_rows:
            values = model_results[column].astype(float)
            rows.append(
                {
                    "Model": model_name,
                    "Metric": metric,
                    "Unit": unit,
                    "Mean RMSE": float(values.mean()),
                    "Standard deviation": float(values.std(ddof=1)),
                    "Number of trajectories": int(values.count()),
                }
            )
    return pd.DataFrame(rows)


def make_comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ("Heading angle", "Arc spacing", "Velocity", "Acceleration"):
        metric_rows = summary.loc[summary["Metric"] == metric].set_index("Model")
        rows.append(
            {
                "Metric": metric,
                "Unit": metric_rows.loc["T-IDM", "Unit"],
                "T-IDM Mean RMSE": metric_rows.loc["T-IDM", "Mean RMSE"],
                "T-IDM Standard deviation": metric_rows.loc["T-IDM", "Standard deviation"],
                "T-FVD Mean RMSE": metric_rows.loc["T-FVD", "Mean RMSE"],
                "T-FVD Standard deviation": metric_rows.loc["T-FVD", "Standard deviation"],
                "Number of trajectories": int(metric_rows.loc["T-IDM", "Number of trajectories"]),
            }
        )
    return pd.DataFrame(rows)


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.linewidth": 1.1,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
        }
    )


def plot_representative_single_model_legacy(
    pair: pd.DataFrame,
    simulation: pd.DataFrame,
    observed_arc_spacing: np.ndarray,
    metrics: pd.Series,
    output_path: Path,
) -> None:
    apply_plot_style()
    colors = {
        "leader": "#8CC5BE",
        "observed": "#074166",
        "simulated": "#CC011F",
    }
    time_s = pair["frameNum"].to_numpy() / FPS
    fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.6), dpi=600)
    leader_style = dict(color=colors["leader"], linestyle="-.", linewidth=2.0)
    observed_style = dict(color=colors["observed"], linestyle="-.", linewidth=2.0)
    simulated_style = dict(color=colors["simulated"], linestyle="--", linewidth=2.3)

    axes[0].plot(
        pair["carCenterXm_leader"],
        pair["carCenterYm_leader"],
        label="Preceding vehicle",
        **leader_style,
    )
    axes[0].plot(
        pair["carCenterXm_follower"],
        pair["carCenterYm_follower"],
        label="Real follower",
        **observed_style,
    )
    axes[0].plot(simulation["x_sim"], simulation["y_sim"], label="Simulated follower (T-IDM)", **simulated_style)
    marker_indices = np.linspace(0, len(pair) - 1, 8).astype(int)
    for x_column, y_column, color in [
        ("carCenterXm_leader", "carCenterYm_leader", colors["leader"]),
        ("carCenterXm_follower", "carCenterYm_follower", colors["observed"]),
    ]:
        axes[0].scatter(
            pair.loc[marker_indices, x_column],
            pair.loc[marker_indices, y_column],
            s=24,
            facecolors="none",
            edgecolors=color,
            linewidths=0.9,
            zorder=4,
        )
    axes[0].set_xlabel("x-position (m)")
    axes[0].set_ylabel("y-position (m)")
    axes[0].set_title(f"Heading-angle RMSE: {metrics['heading_rmse_rad']:.3f} rad", fontsize=10.5, fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8, loc="best")

    axes[1].plot(time_s, observed_arc_spacing, label="Real follower", **observed_style)
    axes[1].plot(time_s, simulation["arc_spacing_sim"], label="Simulated follower (T-IDM)", **simulated_style)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Arc spacing (m)")
    axes[1].set_title(f"Arc-spacing RMSE: {metrics['arc_spacing_rmse_m']:.3f} m", fontsize=10.5, fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8, loc="best")

    axes[2].plot(time_s, pair["velocity_leader"], label="Preceding vehicle", **leader_style)
    axes[2].plot(time_s, pair["velocity_follower"], label="Real follower", **observed_style)
    axes[2].plot(time_s, simulation["velocity_sim"], label="Simulated follower (T-IDM)", **simulated_style)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Velocity (m/s)")
    axes[2].set_title(f"Velocity RMSE: {metrics['velocity_rmse_mps']:.3f} m/s", fontsize=10.5, fontweight="bold")
    axes[2].legend(frameon=False, fontsize=8, loc="best")

    axes[3].plot(time_s, pair["acceleration_leader"], label="Preceding vehicle", **leader_style)
    axes[3].plot(time_s, pair["acceleration_follower"], label="Real follower", **observed_style)
    axes[3].plot(time_s, simulation["acceleration_sim"], label="Simulated follower (T-IDM)", **simulated_style)
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Acceleration (m/s²)")
    axes[3].set_title(
        f"Acceleration RMSE: {metrics['acceleration_rmse_mps2']:.3f} m/s²",
        fontsize=10.5,
        fontweight="bold",
    )
    axes[3].legend(frameon=False, fontsize=8, loc="best")

    panel_labels = ["(a) Position", "(b) Arc spacing-Time", "(c) Velocity-Time", "(d) Acceleration-Time"]
    for axis, panel_label in zip(axes, panel_labels):
        axis.tick_params(labelsize=9)
        axis.text(0.5, -0.24, panel_label, transform=axis.transAxes, ha="center", va="top", fontsize=10.5, fontweight="bold")
    fig.subplots_adjust(left=0.055, right=0.995, top=0.86, bottom=0.25, wspace=0.28)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def savitzky_golay_for_display(
    values: np.ndarray | pd.Series,
    window_length: int = SAVGOL_WINDOW_FRAMES,
    polynomial_order: int = SAVGOL_POLYNOMIAL_ORDER,
) -> np.ndarray:
    """Central Savitzky-Golay smoothing with reflected edges for display only."""
    array = np.asarray(values, dtype=float)
    if window_length % 2 != 1 or window_length <= polynomial_order:
        raise ValueError("Savitzky-Golay window must be odd and exceed the polynomial order")
    if len(array) < window_length:
        return array.copy()
    half_window = window_length // 2
    local_x = np.arange(-half_window, half_window + 1, dtype=float)
    design = np.vander(local_x, polynomial_order + 1, increasing=True)
    coefficients = np.linalg.pinv(design)[0]
    padded = np.pad(array, (half_window, half_window), mode="reflect")
    return np.convolve(padded, coefficients[::-1], mode="valid")


def plot_representative_dual_model(
    pair: pd.DataFrame,
    simulations: dict[str, pd.DataFrame],
    observed_arc_spacing: np.ndarray,
    metrics: dict[str, pd.Series],
    output_path: Path,
) -> None:
    apply_plot_style()
    colors = {
        "leader": "#8CC5BE",
        "observed": "#074166",
        "T-IDM": "#CC011F",
        "T-FVD": "#7A5195",
    }
    time_s = (pair["frameNum"].to_numpy() - pair["frameNum"].iloc[0]) / FPS
    fig, axes = plt.subplots(1, 4, figsize=(18.2, 4.8), dpi=600)
    leader_style = dict(color=colors["leader"], linestyle="-.", linewidth=2.0)
    observed_style = dict(color=colors["observed"], linestyle="-", linewidth=2.0)
    tidm_style = dict(color=colors["T-IDM"], linestyle="--", linewidth=2.25)
    tfvd_style = dict(color=colors["T-FVD"], linestyle=":", linewidth=2.45)

    axes[0].plot(
        pair["carCenterXm_leader"],
        pair["carCenterYm_leader"],
        label="Preceding vehicle",
        **leader_style,
    )
    axes[0].plot(
        pair["carCenterXm_follower"],
        pair["carCenterYm_follower"],
        label="Real follower",
        **observed_style,
    )
    axes[0].plot(
        simulations["T-IDM"]["x_sim"],
        simulations["T-IDM"]["y_sim"],
        label="Simulated follower (T-IDM)",
        **tidm_style,
    )
    axes[0].plot(
        simulations["T-FVD"]["x_sim"],
        simulations["T-FVD"]["y_sim"],
        label="Simulated follower (T-FVD)",
        **tfvd_style,
    )
    marker_indices = np.linspace(0, len(pair) - 1, 8).astype(int)
    for x_column, y_column, color in [
        ("carCenterXm_leader", "carCenterYm_leader", colors["leader"]),
        ("carCenterXm_follower", "carCenterYm_follower", colors["observed"]),
    ]:
        axes[0].scatter(
            pair.loc[marker_indices, x_column],
            pair.loc[marker_indices, y_column],
            s=24,
            facecolors="none",
            edgecolors=color,
            linewidths=0.9,
            zorder=4,
        )
    axes[0].set_xlabel("x-position (m)")
    axes[0].set_ylabel("y-position (m)")
    axes[0].set_title(
        "Heading-angle RMSE\n"
        f"T-IDM: {metrics['T-IDM']['heading_rmse_rad']:.3f}; "
        f"T-FVD: {metrics['T-FVD']['heading_rmse_rad']:.3f} rad",
        fontsize=9.5,
        fontweight="bold",
    )

    axes[1].plot(time_s, observed_arc_spacing, label="Real follower", **observed_style)
    axes[1].plot(
        time_s,
        simulations["T-IDM"]["arc_spacing_sim"],
        label="Simulated follower (T-IDM)",
        **tidm_style,
    )
    axes[1].plot(
        time_s,
        simulations["T-FVD"]["arc_spacing_sim"],
        label="Simulated follower (T-FVD)",
        **tfvd_style,
    )
    axes[1].set_xlabel("Elapsed time (s)")
    axes[1].set_ylabel("Arc spacing (m)")
    axes[1].set_title(
        "Arc-spacing RMSE\n"
        f"T-IDM: {metrics['T-IDM']['arc_spacing_rmse_m']:.3f}; "
        f"T-FVD: {metrics['T-FVD']['arc_spacing_rmse_m']:.3f} m",
        fontsize=9.5,
        fontweight="bold",
    )

    axes[2].plot(time_s, pair["velocity_leader"], label="Preceding vehicle", **leader_style)
    axes[2].plot(time_s, pair["velocity_follower"], label="Real follower", **observed_style)
    axes[2].plot(
        time_s,
        simulations["T-IDM"]["velocity_sim"],
        label="Simulated follower (T-IDM)",
        **tidm_style,
    )
    axes[2].plot(
        time_s,
        simulations["T-FVD"]["velocity_sim"],
        label="Simulated follower (T-FVD)",
        **tfvd_style,
    )
    axes[2].set_xlabel("Elapsed time (s)")
    axes[2].set_ylabel("Velocity (m/s)")
    axes[2].set_title(
        "Velocity RMSE\n"
        f"T-IDM: {metrics['T-IDM']['velocity_rmse_mps']:.3f}; "
        f"T-FVD: {metrics['T-FVD']['velocity_rmse_mps']:.3f} m/s",
        fontsize=9.5,
        fontweight="bold",
    )

    acceleration_time = time_s[:-1]
    axes[3].plot(
        acceleration_time,
        savitzky_golay_for_display(pair["acceleration_leader"].to_numpy()[:-1]),
        label="Preceding vehicle",
        **leader_style,
    )
    axes[3].plot(
        acceleration_time,
        savitzky_golay_for_display(pair["acceleration_follower"].to_numpy()[:-1]),
        label="Real follower",
        **observed_style,
    )
    axes[3].plot(
        acceleration_time,
        savitzky_golay_for_display(
            simulations["T-IDM"]["acceleration_sim"].to_numpy()[:-1]
        ),
        label="Simulated follower (T-IDM)",
        **tidm_style,
    )
    axes[3].plot(
        acceleration_time,
        savitzky_golay_for_display(
            simulations["T-FVD"]["acceleration_sim"].to_numpy()[:-1]
        ),
        label="Simulated follower (T-FVD)",
        **tfvd_style,
    )
    axes[3].set_xlabel("Elapsed time (s)")
    axes[3].set_ylabel(r"Acceleration (m/s$^2$)")
    axes[3].set_title(
        "Acceleration RMSE\n"
        f"T-IDM: {metrics['T-IDM']['acceleration_rmse_mps2']:.3f}; "
        f"T-FVD: {metrics['T-FVD']['acceleration_rmse_mps2']:.3f} m/s$^2$",
        fontsize=9.5,
        fontweight="bold",
    )
    axes[3].text(
        0.98,
        0.03,
        "Smoothed for display only",
        transform=axes[3].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.4,
        color="#555555",
        style="italic",
    )

    panel_labels = [
        "(a) Position",
        "(b) Arc spacing-Time",
        "(c) Velocity-Time",
        "(d) Acceleration-Time",
    ]
    for axis, panel_label in zip(axes, panel_labels):
        axis.tick_params(labelsize=9)
        axis.text(
            0.5,
            -0.24,
            panel_label,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=10.5,
            fontweight="bold",
        )

    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=4,
        fontsize=9,
    )
    fig.subplots_adjust(left=0.055, right=0.995, top=0.79, bottom=0.25, wspace=0.28)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_selected_pairs(
    selected: pd.DataFrame,
    tracks: dict[int, pd.DataFrame],
    output_path: Path,
) -> None:
    apply_plot_style()
    columns = 4
    rows = math.ceil(len(selected) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(12.0, 2.75 * rows), dpi=220)
    axes_array = np.atleast_1d(axes).ravel()
    for axis, (_, selection) in zip(axes_array, selected.iterrows()):
        pair = prepare_pair_data(selection, tracks)
        axis.plot(pair["carCenterXm_leader"], pair["carCenterYm_leader"], color="#8CC5BE", linestyle="-.", linewidth=1.7, label="Leader")
        axis.plot(pair["carCenterXm_follower"], pair["carCenterYm_follower"], color="#074166", linestyle="-.", linewidth=1.7, label="Follower")
        axis.set_title(
            f"No. {int(selection['trajectory_no'])}: {int(selection['leader_id'])}→{int(selection['follower_id'])} "
            f"(lanes {int(selection['entry_lane'])}→{int(selection['exit_lane'])})",
            fontsize=8.5,
        )
        axis.set_aspect("equal", adjustable="datalim")
        axis.tick_params(labelsize=7)
    for axis in axes_array[len(selected) :]:
        axis.axis("off")
    handles, labels = axes_array[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, fontsize=9)
    fig.supxlabel("x-position (m)", fontsize=10)
    fig.supylabel("y-position (m)", fontsize=10)
    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.07, right=0.99, hspace=0.40, wspace=0.30)
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def save_protocol(
    output_dir: Path,
    input_csv: Path,
    selected_count: int,
    eligible_count: int,
    route_balanced_count: int,
) -> None:
    payload = {
        "source_file": str(input_csv),
        "source_data_role": "CitySim Intersection A cleaned kinematic trajectories",
        "sampling_frequency_hz": FPS,
        "screening_protocol": asdict(PROTOCOL),
        "selection_rule": "All geometry-qualified immediate-leader/follower pairs are retained; no model-error criterion is used. A one-pair-per-entry-to-exit-route subset is flagged only for sensitivity analysis.",
        "eligible_pair_count": eligible_count,
        "primary_validation_pair_count": selected_count,
        "route_balanced_sensitivity_pair_count": route_balanced_count,
        "models": ["TMSFF with T-IDM", "TMSFF with T-FVD"],
        "parameter_source": PARAMETER_SOURCE,
        "parameters_fixed_without_citysim_recalibration": {
            "T-IDM": HEFEI_TIDM_PARAMETERS,
            "T-FVD": HEFEI_TFVD_PARAMETERS,
        },
        "average_vehicle_length_m": VEHICLE_LENGTH_M,
        "common_acceleration_limits_mps2": list(ACCELERATION_LIMITS_MPS2),
        "metric_standard_deviation": "sample SD across trajectory-level RMSE values (ddof=1)",
        "acceleration_plot_smoothing": {
            "method": "Savitzky-Golay",
            "window_frames": SAVGOL_WINDOW_FRAMES,
            "polynomial_order": SAVGOL_POLYNOMIAL_ORDER,
            "scope": "Fig. 10(d) display only; all RMSE values use unsmoothed data",
        },
    }
    (output_dir / "citysim_external_validation_protocol.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="IntersectionA_cleaned.csv")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--tidm-parameters-json",
        type=Path,
        help="Optional JSON containing a 'parameters' mapping used to override T-IDM values",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.tidm_parameters_json is not None:
        payload = json.loads(args.tidm_parameters_json.read_text(encoding="utf-8"))
        parameters = payload.get("parameters", payload)
        required = set(HEFEI_TIDM_PARAMETERS)
        missing = required.difference(parameters)
        if missing:
            raise ValueError(f"T-IDM parameter override is missing: {sorted(missing)}")
        HEFEI_TIDM_PARAMETERS.update(
            {name: float(parameters[name]) for name in HEFEI_TIDM_PARAMETERS}
        )
        global PARAMETER_SOURCE
        PARAMETER_SOURCE = (
            "T-IDM target-calibrated on the 38 CitySim pairs using the supplied "
            "parameter JSON; T-FVD fixed at its seed-42 common-protocol Hefei values"
        )

    _, tracks = load_tracks(args.input)
    eligible, selected = select_pairs(tracks)
    eligible.to_csv(args.output / "citysim_eligible_following_pairs.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(args.output / "citysim_selected_following_pairs.csv", index=False, encoding="utf-8-sig")

    result_rows = []
    representative_pair = None
    representative_observed_arc_spacing = None
    representative_simulations: dict[str, pd.DataFrame] = {}
    representative_metrics: dict[str, dict[str, float | int | bool | str]] = {}
    for _, selection in selected.iterrows():
        pair = prepare_pair_data(selection, tracks)
        pair_observed_arc_spacing = None
        for model_name in MODEL_NAMES:
            simulation, observed_arc_spacing, _ = simulate_pair(pair, model_name)
            if pair_observed_arc_spacing is None:
                pair_observed_arc_spacing = observed_arc_spacing
            elif not np.allclose(
                pair_observed_arc_spacing,
                observed_arc_spacing,
                rtol=0.0,
                atol=1e-10,
            ):
                raise RuntimeError("Observed arc spacing changed between model runs")
            metrics = evaluate_pair(
                model_name,
                int(selection["trajectory_no"]),
                selection,
                pair,
                simulation,
                observed_arc_spacing,
            )
            result_rows.append(metrics)
            if bool(selection["representative"]):
                representative_pair = pair
                representative_observed_arc_spacing = observed_arc_spacing
                representative_simulations[model_name] = simulation
                representative_metrics[model_name] = metrics

    results = (
        pd.DataFrame(result_rows)
        .sort_values(["model", "trajectory_no"])
        .reset_index(drop=True)
    )
    summary = make_summary(results)
    comparison_summary = make_comparison_table(summary)
    route_balanced_summary = make_summary(results.loc[results["route_balanced_subset"]])
    route_balanced_comparison = make_comparison_table(route_balanced_summary)
    results.to_csv(args.output / "citysim_external_validation_per_trajectory.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output / "citysim_external_validation_summary.csv", index=False, encoding="utf-8-sig")
    comparison_summary.to_csv(
        args.output / "citysim_external_validation_summary_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    route_balanced_summary.to_csv(
        args.output / "citysim_route_balanced_sensitivity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    route_balanced_comparison.to_csv(
        args.output / "citysim_route_balanced_sensitivity_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if (
        representative_pair is None
        or representative_observed_arc_spacing is None
        or set(representative_simulations) != set(MODEL_NAMES)
    ):
        raise RuntimeError("The original representative pair (leader 6, follower 59) was not selected")
    pair = representative_pair
    observed_arc_spacing = representative_observed_arc_spacing
    elapsed_time_s = (pair["frameNum"] - pair["frameNum"].iloc[0]) / FPS

    def smoothed_simulation_acceleration(model_name: str) -> np.ndarray:
        output = np.full(len(pair), np.nan)
        output[:-1] = savitzky_golay_for_display(
            representative_simulations[model_name]["acceleration_sim"].to_numpy()[:-1]
        )
        return output

    observed_acceleration_smoothed = savitzky_golay_for_display(
        pair["acceleration_follower"].to_numpy()
    )
    leader_acceleration_smoothed = savitzky_golay_for_display(
        pair["acceleration_leader"].to_numpy()
    )
    representative_timeseries = pd.DataFrame(
        {
            "frameNum": pair["frameNum"],
            "elapsed_time_s": elapsed_time_s,
            "leader_x_m": pair["carCenterXm_leader"],
            "leader_y_m": pair["carCenterYm_leader"],
            "observed_follower_x_m": pair["carCenterXm_follower"],
            "observed_follower_y_m": pair["carCenterYm_follower"],
            "tidm_follower_x_m": representative_simulations["T-IDM"]["x_sim"],
            "tidm_follower_y_m": representative_simulations["T-IDM"]["y_sim"],
            "tfvd_follower_x_m": representative_simulations["T-FVD"]["x_sim"],
            "tfvd_follower_y_m": representative_simulations["T-FVD"]["y_sim"],
            "observed_heading_rad": pair["heading_follower"],
            "tidm_heading_rad": representative_simulations["T-IDM"]["heading_sim"],
            "tfvd_heading_rad": representative_simulations["T-FVD"]["heading_sim"],
            "observed_arc_spacing_m": observed_arc_spacing,
            "tidm_arc_spacing_m": representative_simulations["T-IDM"]["arc_spacing_sim"],
            "tfvd_arc_spacing_m": representative_simulations["T-FVD"]["arc_spacing_sim"],
            "observed_velocity_mps": pair["velocity_follower"],
            "tidm_velocity_mps": representative_simulations["T-IDM"]["velocity_sim"],
            "tfvd_velocity_mps": representative_simulations["T-FVD"]["velocity_sim"],
            "leader_acceleration_raw_mps2": pair["acceleration_leader"],
            "observed_acceleration_mps2": pair["acceleration_follower"],
            "tidm_acceleration_raw_mps2": representative_simulations["T-IDM"]["acceleration_sim"],
            "tfvd_acceleration_raw_mps2": representative_simulations["T-FVD"]["acceleration_sim"],
            "leader_acceleration_display_smoothed_mps2": leader_acceleration_smoothed,
            "observed_acceleration_display_smoothed_mps2": observed_acceleration_smoothed,
            "tidm_acceleration_display_smoothed_mps2": smoothed_simulation_acceleration("T-IDM"),
            "tfvd_acceleration_display_smoothed_mps2": smoothed_simulation_acceleration("T-FVD"),
        }
    )
    representative_timeseries.to_csv(
        args.output / "citysim_representative_no59_timeseries.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_representative_dual_model(
        pair,
        representative_simulations,
        observed_arc_spacing,
        {
            model_name: pd.Series(representative_metrics[model_name])
            for model_name in MODEL_NAMES
        },
        args.output / "Fig_10_CitySim_multitrajectory_protocol.png",
    )
    route_balanced = selected.loc[selected["route_balanced_subset"]].copy()
    plot_selected_pairs(
        route_balanced,
        tracks,
        args.output / "CitySim_route_balanced_subset_overview.png",
    )
    save_protocol(
        args.output,
        args.input,
        len(selected),
        len(eligible),
        len(route_balanced),
    )

    print("Qualified trajectory pairs:", len(selected))
    print("Route-balanced sensitivity pairs:", len(route_balanced))
    print("Validation outputs written to:", args.output)


if __name__ == "__main__":
    main()
