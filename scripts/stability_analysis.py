# -*- coding: utf-8 -*-
"""Run traffic-flow stability simulations and generate stability figures.

The equilibrium speed is fixed to the value derived from the measured
curved-road leader trajectory used in the manuscript. The raw trajectory data
are not included because of data-use restrictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DT = 0.1
T_END = 80.0
VEHICLE_LENGTH = 5.0
N_FOLLOWERS = 8
EQUILIBRIUM_SPEED = 7.081830396538565
REPRESENTATIVE_STATION = 80.0
MODELS = ["TMSFF", "C-FVD", "GACF"]
COLORS = {"TMSFF": "#0072B2", "C-FVD": "#D55E00", "GACF": "#7F7F7F"}


@dataclass(frozen=True)
class Scenario:
    name: str
    accel_pulse: float
    decel_pulse: float


SCENARIOS = [
    Scenario("mild", 0.75, 0.50),
    Scenario("moderate", 1.20, 0.80),
    Scenario("severe", 1.65, 1.10),
]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 8.8,
            "axes.labelsize": 9.2,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
        }
    )


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
        raw += 0.18 * leader_accel + 0.038 * (effective_gap - (22.0 - VEHICLE_LENGTH))
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


def equilibrium_gap(model: str, speed: float) -> float:
    candidate_gaps = np.linspace(6.0, 90.0, 6000)
    accel_values = np.array(
        [
            compute_acceleration(model, REPRESENTATIVE_STATION, speed, 0.0, speed, 0.0, gap)
            for gap in candidate_gaps
        ]
    )
    return float(candidate_gaps[int(np.argmin(np.abs(accel_values)))])


def local_linear_stability(model: str, speed: float) -> dict:
    gap = equilibrium_gap(model, speed)
    h = 1.0e-3
    station = REPRESENTATIVE_STATION
    f_gap = (
        compute_acceleration(model, station, speed, 0.0, speed, 0.0, gap + h)
        - compute_acceleration(model, station, speed, 0.0, speed, 0.0, gap - h)
    ) / (2.0 * h)
    f_speed = (
        compute_acceleration(model, station, speed + h, 0.0, speed, 0.0, gap)
        - compute_acceleration(model, station, speed - h, 0.0, speed, 0.0, gap)
    ) / (2.0 * h)
    f_rel_speed = (
        compute_acceleration(model, station, speed, 0.0, speed + h, 0.0, gap)
        - compute_acceleration(model, station, speed, 0.0, speed - h, 0.0, gap)
    ) / (2.0 * h)
    matrix = np.array([[0.0, -1.0], [f_gap, f_speed - f_rel_speed]])
    eig = np.linalg.eigvals(matrix)
    max_real = float(np.max(np.real(eig)))
    return {
        "model": model,
        "equilibrium_speed_mps": speed,
        "equilibrium_gap_m": gap,
        "f_gap": float(f_gap),
        "f_speed": float(f_speed),
        "f_rel_speed": float(f_rel_speed),
        "eig_real_1": float(np.real(eig[0])),
        "eig_real_2": float(np.real(eig[1])),
        "max_eig_real": max_real,
        "locally_stable": bool(max_real < 0.0),
    }


def leader_perturbation(time: np.ndarray, speed: float, scenario: Scenario):
    leader_speed = (
        speed
        + scenario.accel_pulse * np.exp(-((time - 18.0) / 3.5) ** 2)
        - scenario.decel_pulse * np.exp(-((time - 38.0) / 4.5) ** 2)
    )
    leader_speed = np.clip(leader_speed, 0.5, 15.0)
    leader_accel = np.gradient(leader_speed, DT)
    return leader_speed, leader_accel


def simulate_platoon(model: str, speed0: float, scenario: Scenario) -> pd.DataFrame:
    time = np.arange(0.0, T_END + DT, DT)
    n_steps = len(time)
    n_vehicles = N_FOLLOWERS + 1
    gap0 = equilibrium_gap(model, speed0)
    station = np.zeros((n_steps, n_vehicles))
    speed = np.zeros((n_steps, n_vehicles))
    accel = np.zeros((n_steps, n_vehicles))
    speed[:, 0], accel[:, 0] = leader_perturbation(time, speed0, scenario)
    station[0, 0] = 100.0
    speed[0, 1:] = speed0
    for veh in range(1, n_vehicles):
        station[0, veh] = station[0, veh - 1] - gap0

    for step in range(n_steps - 1):
        station[step + 1, 0] = station[step, 0] + speed[step, 0] * DT + 0.5 * accel[step, 0] * DT * DT
        for veh in range(1, n_vehicles):
            gap = station[step, veh - 1] - station[step, veh]
            accel[step + 1, veh] = compute_acceleration(
                model,
                station[step, veh],
                speed[step, veh],
                accel[step, veh],
                speed[step, veh - 1],
                accel[step, veh - 1],
                gap,
            )
            speed[step + 1, veh] = np.clip(speed[step, veh] + accel[step + 1, veh] * DT, 0.1, 15.0)
            station[step + 1, veh] = station[step, veh] + speed[step, veh] * DT + 0.5 * accel[step + 1, veh] * DT * DT

    records = []
    for veh in range(n_vehicles):
        if veh == 0:
            gap_series = np.full(n_steps, np.nan)
            spacing_error = np.full(n_steps, np.nan)
        else:
            gap_series = station[:, veh - 1] - station[:, veh]
            spacing_error = gap_series - gap0
        records.append(
            pd.DataFrame(
                {
                    "scenario": scenario.name,
                    "model": model,
                    "vehicle_index": veh,
                    "time_s": time,
                    "station_m": station[:, veh],
                    "speed_mps": speed[:, veh],
                    "accel_mps2": accel[:, veh],
                    "gap_m": gap_series,
                    "spacing_error_m": spacing_error,
                    "speed_deviation_mps": speed[:, veh] - speed0,
                    "equilibrium_speed_mps": speed0,
                    "equilibrium_gap_m": gap0,
                }
            )
        )
    return pd.concat(records, ignore_index=True)


def propagation_metrics(timeseries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, model), group in timeseries.groupby(["scenario", "model"]):
        leader = group[(group["vehicle_index"] == 0) & (group["time_s"] >= 5.0)]
        last = group[(group["vehicle_index"] == N_FOLLOWERS) & (group["time_s"] >= 5.0)]
        follower = group[(group["vehicle_index"] > 0) & (group["time_s"] >= 5.0)]
        leader_peak = float(np.max(np.abs(leader["speed_deviation_mps"])))
        last_peak = float(np.max(np.abs(last["speed_deviation_mps"])))
        leader_rms = float(np.sqrt(np.mean(leader["speed_deviation_mps"] ** 2)))
        last_rms = float(np.sqrt(np.mean(last["speed_deviation_mps"] ** 2)))
        rows.append(
            {
                "scenario": scenario,
                "model": model,
                "leader_peak_speed_dev_mps": leader_peak,
                "last_peak_speed_dev_mps": last_peak,
                "peak_speed_amplification_ratio": last_peak / leader_peak,
                "leader_rms_speed_dev_mps": leader_rms,
                "last_rms_speed_dev_mps": last_rms,
                "rms_speed_amplification_ratio": last_rms / leader_rms,
                "max_abs_spacing_error_m": float(np.max(np.abs(follower["spacing_error_m"]))),
            }
        )
    return pd.DataFrame(rows)


def plot_heatmap(timeseries: pd.DataFrame) -> None:
    set_style()
    data = timeseries[timeseries["scenario"] == "moderate"].copy()
    vmax = max(float(np.nanpercentile(np.abs(data["speed_deviation_mps"]), 99.0)), 1.2)
    time = np.sort(data["time_s"].unique())
    vehicles = sorted(data["vehicle_index"].unique())
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.15), dpi=600, sharex=True, gridspec_kw={"hspace": 0.12})
    image = None
    for ax, model, label in zip(axes, MODELS, ["(a)", "(b)", "(c)"]):
        sub = data[data["model"] == model]
        matrix = (
            sub.pivot_table(index="vehicle_index", columns="time_s", values="speed_deviation_mps", aggfunc="mean")
            .reindex(index=vehicles, columns=time)
            .to_numpy()
        )
        image = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
            extent=[time.min(), time.max(), min(vehicles) - 0.5, max(vehicles) + 0.5],
        )
        ax.set_ylabel("Vehicle index")
        ax.text(
            0.012,
            0.88,
            f"{label} {model}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            bbox={"boxstyle": "square,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
    axes[-1].set_xlabel("Time (s)")
    cbar = fig.colorbar(image, ax=axes, location="right", fraction=0.034, pad=0.018)
    cbar.set_label("Speed deviation (m/s)")
    fig.savefig(OUT / "figure_stability_heatmap.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_amplification(timeseries: pd.DataFrame) -> None:
    set_style()
    data = timeseries[timeseries["scenario"] == "moderate"].copy()
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=600)
    line_styles = {"TMSFF": "-", "C-FVD": "--", "GACF": "-."}
    for model in MODELS:
        sub = data[data["model"] == model]
        leader_peak = float(
            np.max(np.abs(sub[(sub["vehicle_index"] == 0) & (sub["time_s"] >= 5.0)]["speed_deviation_mps"]))
        )
        ratios = []
        vehicle_indices = list(range(N_FOLLOWERS + 1))
        for veh in vehicle_indices:
            veh_data = sub[(sub["vehicle_index"] == veh) & (sub["time_s"] >= 5.0)]
            ratios.append(float(np.max(np.abs(veh_data["speed_deviation_mps"]))) / leader_peak)
        ax.plot(vehicle_indices, ratios, marker="o", lw=1.35, color=COLORS[model], ls=line_styles[model], label=model)
    ax.axhline(1.0, color="#404040", lw=0.7, ls=":", label="Neutral amplification")
    ax.set_xlabel("Vehicle index")
    ax.set_ylabel("Peak speed disturbance ratio")
    ax.set_xticks(range(N_FOLLOWERS + 1))
    ax.set_ylim(0.0, 2.25)
    ax.grid(True, color="#C8C8C8", linewidth=0.42, alpha=0.55)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "figure_stability_amplification.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    linear = pd.DataFrame([local_linear_stability(model, EQUILIBRIUM_SPEED) for model in MODELS])
    all_series = [
        simulate_platoon(model, EQUILIBRIUM_SPEED, scenario)
        for scenario in SCENARIOS
        for model in MODELS
    ]
    timeseries = pd.concat(all_series, ignore_index=True)
    metrics = propagation_metrics(timeseries)
    linear.to_csv(OUT / "linear_stability_results.csv", index=False)
    timeseries.to_csv(OUT / "platoon_stability_timeseries.csv", index=False)
    metrics.to_csv(OUT / "stability_metrics.csv", index=False)
    plot_heatmap(timeseries)
    plot_amplification(timeseries)
    print(linear.round(4).to_string(index=False))
    print(metrics.round(4).to_string(index=False))
    print(f"Outputs written to: {OUT}")


if __name__ == "__main__":
    main()
