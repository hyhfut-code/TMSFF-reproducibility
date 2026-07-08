# -*- coding: utf-8 -*-
"""Run the curved-road baseline comparison simulation.

The script uses a processed leader profile recovered from
``outputs/curved_cf_timeseries.csv`` when available. This avoids releasing the
restricted raw trajectory Excel file. If the processed output is absent, a
synthetic leader profile is generated for demonstration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DT = 0.1
T_END = 19.9
TIME = np.arange(0.0, T_END + DT, DT)
VEHICLE_LENGTH = 5.0
TARGET_GAP = 22.0
MODELS = ["TMSFF", "C-FVD", "GACF"]


def centerline_y(s):
    return 3.2 * np.sin(2.0 * np.pi * s / 260.0) + 1.7 * np.sin(2.0 * np.pi * s / 135.0 + 0.8)


def centerline_dy(s):
    return (
        3.2 * (2.0 * np.pi / 260.0) * np.cos(2.0 * np.pi * s / 260.0)
        + 1.7 * (2.0 * np.pi / 135.0) * np.cos(2.0 * np.pi * s / 135.0 + 0.8)
    )


def centerline_ddy(s):
    return (
        -3.2 * (2.0 * np.pi / 260.0) ** 2 * np.sin(2.0 * np.pi * s / 260.0)
        - 1.7 * (2.0 * np.pi / 135.0) ** 2 * np.sin(2.0 * np.pi * s / 135.0 + 0.8)
    )


def curvature(s):
    dy = centerline_dy(s)
    ddy = centerline_ddy(s)
    return np.abs(ddy) / np.power(1.0 + dy * dy, 1.5)


def unit_normal(s):
    dy = centerline_dy(s)
    scale = np.sqrt(1.0 + dy * dy)
    return -dy / scale, 1.0 / scale


def leader_lateral_offset(s):
    return 0.95 * np.exp(-((s - 330.0) / 58.0) ** 2) - 0.52 * np.exp(-((s - 515.0) / 46.0) ** 2)


def pose_xy(s, lateral_offset):
    nx, ny = unit_normal(s)
    return s + nx * lateral_offset, centerline_y(s) + ny * lateral_offset


def load_or_create_leader() -> pd.DataFrame:
    """Load processed leader states from existing outputs, or create an example profile."""
    ts_file = OUT / "curved_cf_timeseries.csv"
    if ts_file.exists():
        ts = pd.read_csv(ts_file)
        leader = (
            ts[["time_s", "leader_s_m", "leader_v_mps", "leader_a_mps2"]]
            .drop_duplicates("time_s")
            .sort_values("time_s")
            .reset_index(drop=True)
        )
        return leader

    time = TIME
    speed = 5.0 + 2.25 * (1.0 - np.exp(-time / 4.5)) + 0.35 * np.sin(0.75 * time)
    accel = np.gradient(speed, DT)
    station = 10.0 + np.cumsum(speed) * DT
    return pd.DataFrame(
        {"time_s": time, "leader_s_m": station, "leader_v_mps": speed, "leader_a_mps2": accel}
    )


def curve_speed_limit(station: float, base_speed: float, penalty: float) -> float:
    return max(5.8, base_speed - penalty * float(curvature(station)))


def compute_acceleration(model, station, speed, prev_accel, leader_speed, leader_accel, gap):
    effective_gap = max(gap - VEHICLE_LENGTH, 1.5)
    rel_speed = leader_speed - speed

    if model == "TMSFF":
        desired_speed = curve_speed_limit(station, 12.7, 190.0)
        max_accel = 1.55
        comfortable_decel = 2.25
        min_gap = 4.2
        time_gap = 1.06
        dynamic_gap = min_gap + time_gap * speed + max(
            0.0, speed * (speed - leader_speed) / (2.0 * np.sqrt(max_accel * comfortable_decel))
        )
        raw = max_accel * (1.0 - (speed / max(desired_speed, 0.1)) ** 4 - (dynamic_gap / effective_gap) ** 2)
        raw += 0.18 * leader_accel + 0.038 * (effective_gap - (TARGET_GAP - VEHICLE_LENGTH))
        return 0.58 * prev_accel + 0.42 * float(np.clip(raw, -2.20, 1.45))

    if model == "C-FVD":
        desired_speed = curve_speed_limit(station, 12.8, 130.0)
        optimal_velocity = desired_speed * (
            np.tanh((effective_gap - 18.5) / 6.2) + np.tanh(18.5 / 6.2)
        ) / (1.0 + np.tanh(18.5 / 6.2))
        return float(np.clip(0.46 * (optimal_velocity - speed) + 0.58 * rel_speed, -3.10, 2.05))

    if model == "GACF":
        desired_speed = curve_speed_limit(station, 12.1, 340.0)
        local_curvature = float(curvature(station))
        desired_gap = 6.5 + 1.34 * speed + 72.0 * local_curvature * speed * speed
        raw = 1.05 * (1.0 - (speed / max(desired_speed, 0.1)) ** 4 - (desired_gap / effective_gap) ** 2)
        raw += 0.31 * rel_speed - 42.0 * local_curvature
        return 0.35 * prev_accel + 0.65 * float(np.clip(raw, -2.75, 1.65))

    raise ValueError(f"Unknown model: {model}")


def lateral_target(model, station):
    if model == "TMSFF":
        return float(leader_lateral_offset(station)), 1.85
    if model == "C-FVD":
        return 0.0, 0.34
    if model == "GACF":
        return 0.28 * float(leader_lateral_offset(station)), 0.64
    raise ValueError(f"Unknown model: {model}")


def simulate_follower(model: str, leader: pd.DataFrame) -> pd.DataFrame:
    n = len(leader)
    time = leader["time_s"].to_numpy()
    station = np.zeros(n)
    speed = np.zeros(n)
    accel = np.zeros(n)
    offset = np.zeros(n)
    station[0] = leader.loc[0, "leader_s_m"] - TARGET_GAP - 1.0
    speed[0] = 8.7
    offset[0] = -0.25

    for i in range(n - 1):
        dt = float(time[i + 1] - time[i])
        gap = leader.loc[i, "leader_s_m"] - station[i]
        accel[i + 1] = compute_acceleration(
            model, station[i], speed[i], accel[i], leader.loc[i, "leader_v_mps"], leader.loc[i, "leader_a_mps2"], gap
        )
        speed[i + 1] = np.clip(speed[i] + accel[i + 1] * dt, 0.1, 15.0)
        station[i + 1] = station[i] + speed[i] * dt + 0.5 * accel[i + 1] * dt * dt
        target_offset, gain = lateral_target(model, station[i])
        offset[i + 1] = offset[i] + dt * gain * (target_offset - offset[i])

    x, y = pose_xy(station, offset)
    return pd.DataFrame(
        {
            "time_s": time,
            "model": model,
            "s_m": station,
            "v_mps": speed,
            "a_mps2": accel,
            "offset_m": offset,
            "x_m": x,
            "y_m": y,
            "gap_m": leader["leader_s_m"].to_numpy() - station,
        }
    )


def compute_metrics(leader: pd.DataFrame, follower: pd.DataFrame):
    mask = follower["time_s"].to_numpy() >= 3.0
    speed = follower["v_mps"].to_numpy()
    accel = follower["a_mps2"].to_numpy()
    jerk = np.gradient(accel, DT)
    gap = follower["gap_m"].to_numpy()
    ref_s = leader["leader_s_m"].to_numpy() - TARGET_GAP
    ref_x, ref_y = pose_xy(ref_s, leader_lateral_offset(ref_s))
    traj_err = np.hypot(follower["x_m"].to_numpy() - ref_x, follower["y_m"].to_numpy() - ref_y)
    distance_m = max(follower["s_m"].iloc[-1] - follower["s_m"].iloc[0], 1e-3)
    fuel_rate = (
        1.95
        + 0.105 * speed
        + 0.0115 * speed * speed
        + 0.58 * np.maximum(accel, 0.0) * speed
        + 0.095 * np.abs(accel) * speed
    )
    fuel_l_per_100km = float(np.sum(fuel_rate[mask]) * DT / 1000.0 / (distance_m / 1000.0) * 100.0)
    return {
        "mean_abs_jerk_mps3": float(np.mean(np.abs(jerk[mask]))),
        "speed_std_mps": float(np.std(speed[mask])),
        "spacing_rmse_m": float(np.sqrt(np.mean((gap[mask] - TARGET_GAP) ** 2))),
        "trajectory_rmse_m": float(np.sqrt(np.mean(traj_err[mask] ** 2))),
        "fuel_L_per_100km": fuel_l_per_100km,
        "mean_gap_m": float(np.mean(gap[mask])),
        "mean_speed_mps": float(np.mean(speed[mask])),
        "max_abs_accel_mps2": float(np.max(np.abs(accel[mask]))),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    leader = load_or_create_leader()
    followers = {model: simulate_follower(model, leader) for model in MODELS}
    rows = []
    series = []
    for model, follower in followers.items():
        row = {"model": model}
        row.update(compute_metrics(leader, follower))
        rows.append(row)
        merged = follower.copy()
        merged["leader_s_m"] = leader["leader_s_m"].to_numpy()
        merged["leader_v_mps"] = leader["leader_v_mps"].to_numpy()
        merged["leader_a_mps2"] = leader["leader_a_mps2"].to_numpy()
        merged["spacing_error_m"] = merged["gap_m"] - TARGET_GAP
        merged["jerk_mps3"] = np.gradient(merged["a_mps2"].to_numpy(), DT)
        series.append(merged)

    pd.DataFrame(rows).to_csv(OUT / "curved_cf_metrics.csv", index=False)
    pd.concat(series, ignore_index=True).to_csv(OUT / "curved_cf_timeseries.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"Outputs written to: {OUT}")


if __name__ == "__main__":
    main()
