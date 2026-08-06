from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from second_review_experiments import (
    ACCELERATION_LIMITS,
    DT,
    MODEL_SPECS,
    VEHICLE_LENGTH,
)


T_END = 72.0
TIME = np.arange(0.0, T_END + DT, DT)
TARGET_GAP = 22.0
RANDOM_SEED = 20260630
MODELS = ("T-IDM", "C-FVD", "GAPB")
COLORS = {"T-IDM": "#0072B2", "C-FVD": "#D55E00", "GAPB": "#7F7F7F"}
LINESTYLES = {"T-IDM": "-", "C-FVD": "--", "GAPB": "-."}


def centerline_y(station: np.ndarray | float) -> np.ndarray | float:
    return 3.2 * np.sin(2.0 * np.pi * station / 260.0) + 1.7 * np.sin(
        2.0 * np.pi * station / 135.0 + 0.8
    )


def centerline_dy(station: np.ndarray | float) -> np.ndarray | float:
    return (
        3.2 * (2.0 * np.pi / 260.0) * np.cos(2.0 * np.pi * station / 260.0)
        + 1.7
        * (2.0 * np.pi / 135.0)
        * np.cos(2.0 * np.pi * station / 135.0 + 0.8)
    )


def centerline_ddy(station: np.ndarray | float) -> np.ndarray | float:
    return (
        -3.2
        * (2.0 * np.pi / 260.0) ** 2
        * np.sin(2.0 * np.pi * station / 260.0)
        - 1.7
        * (2.0 * np.pi / 135.0) ** 2
        * np.sin(2.0 * np.pi * station / 135.0 + 0.8)
    )


def curvature(station: np.ndarray | float) -> np.ndarray | float:
    derivative = centerline_dy(station)
    second_derivative = centerline_ddy(station)
    return np.abs(second_derivative) / np.power(1.0 + derivative * derivative, 1.5)


def generate_leader() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    ripple = 0.04 * rng.standard_normal(len(TIME))
    speed = (
        9.6
        + 0.82 * np.sin(2.0 * np.pi * TIME / 24.0)
        + 0.36 * np.sin(2.0 * np.pi * TIME / 8.5 + 0.7)
        - 1.20 * np.exp(-((TIME - 31.0) / 5.6) ** 2)
        + 0.72 * np.exp(-((TIME - 51.0) / 6.8) ** 2)
        + ripple
    )
    speed = np.clip(speed, 6.6, 11.5)
    station = 35.0 + np.cumsum(speed) * DT
    acceleration = np.gradient(speed, DT)
    return pd.DataFrame(
        {
            "time_s": TIME,
            "leader_s_m": station,
            "leader_v_mps": speed,
            "leader_a_mps2": acceleration,
        }
    )


def load_seed_parameters(runs_path: Path, seed: int) -> dict[str, np.ndarray]:
    runs = pd.read_csv(runs_path)
    specs = {spec.name: spec for spec in MODEL_SPECS}
    result: dict[str, np.ndarray] = {}
    for model in MODELS:
        row = runs[(runs["model"] == model) & (runs["seed"] == seed)]
        if len(row) != 1:
            raise ValueError(f"Expected one row for {model}, seed {seed}")
        spec = specs[model]
        result[model] = row.loc[:, list(spec.parameter_names)].iloc[0].to_numpy(dtype=float)
    return result


def compute_model_acceleration(
    model: str,
    params: np.ndarray,
    station: float,
    speed: float,
    leader_speed: float,
    center_gap: float,
) -> float:
    specs = {spec.name: spec for spec in MODEL_SPECS}
    raw = specs[model].acceleration(
        params,
        speed,
        leader_speed,
        center_gap,
        float(curvature(station)),
    )
    return float(np.clip(raw, ACCELERATION_LIMITS[0], ACCELERATION_LIMITS[1]))


def simulate_follower(
    model: str,
    params: np.ndarray,
    leader: pd.DataFrame,
) -> pd.DataFrame:
    count = len(leader)
    station = np.zeros(count)
    speed = np.zeros(count)
    acceleration = np.zeros(count)
    station[0] = leader.loc[0, "leader_s_m"] - TARGET_GAP - 1.0
    speed[0] = 8.7
    for index in range(count - 1):
        center_gap = leader.loc[index, "leader_s_m"] - station[index]
        acceleration[index] = compute_model_acceleration(
            model,
            params,
            station[index],
            speed[index],
            leader.loc[index, "leader_v_mps"],
            center_gap,
        )
        speed[index + 1] = max(0.0, speed[index] + acceleration[index] * DT)
        station[index + 1] = (
            station[index]
            + speed[index] * DT
            + 0.5 * acceleration[index] * DT * DT
        )
    acceleration[-1] = acceleration[-2]
    return pd.DataFrame(
        {
            "time_s": TIME,
            "model": model,
            "s_m": station,
            "v_mps": speed,
            "a_mps2": acceleration,
            "gap_m": leader["leader_s_m"].to_numpy() - station,
        }
    )


def compute_fuel_l_per_100km(speed: np.ndarray, acceleration: np.ndarray, distance_m: float) -> float:
    positive_acceleration = np.maximum(acceleration, 0.0)
    fuel_rate_ml_s = (
        1.95
        + 0.105 * speed
        + 0.0115 * speed * speed
        + 0.58 * positive_acceleration * speed
        + 0.095 * np.abs(acceleration) * speed
    )
    fuel_l = float(np.sum(fuel_rate_ml_s) * DT / 1000.0)
    distance_km = max(distance_m / 1000.0, 0.001)
    return fuel_l / distance_km * 100.0


def compute_metrics(follower: pd.DataFrame) -> dict[str, float]:
    mask = follower["time_s"].to_numpy() >= 3.0
    speed = follower["v_mps"].to_numpy()
    acceleration = follower["a_mps2"].to_numpy()
    jerk = np.gradient(acceleration, DT)
    gap = follower["gap_m"].to_numpy()
    distance_m = follower["s_m"].iloc[-1] - follower["s_m"].iloc[0]
    return {
        "mean_abs_jerk_mps3": float(np.mean(np.abs(jerk[mask]))),
        "speed_std_mps": float(np.std(speed[mask])),
        "spacing_rmse_m": float(np.sqrt(np.mean((gap[mask] - TARGET_GAP) ** 2))),
        "fuel_L_per_100km": compute_fuel_l_per_100km(
            speed[mask],
            acceleration[mask],
            distance_m,
        ),
        "mean_gap_m": float(np.mean(gap[mask])),
        "mean_speed_mps": float(np.mean(speed[mask])),
        "max_abs_accel_mps2": float(np.max(np.abs(acceleration[mask]))),
    }


def plot_dynamic_response(
    leader: pd.DataFrame,
    followers: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.05), dpi=300, sharex=True)
    axes[0, 0].plot(
        leader["time_s"],
        leader["leader_v_mps"],
        color="#222222",
        linewidth=1.2,
        label="Leader",
    )
    for model in MODELS:
        follower = followers[model]
        mask = follower["time_s"] >= 3.0
        time = follower.loc[mask, "time_s"]
        axes[0, 0].plot(
            time,
            follower.loc[mask, "v_mps"],
            color=COLORS[model],
            linestyle=LINESTYLES[model],
            linewidth=1.2,
            label=model,
        )
        axes[0, 1].plot(
            time,
            follower.loc[mask, "gap_m"] - TARGET_GAP,
            color=COLORS[model],
            linestyle=LINESTYLES[model],
            linewidth=1.2,
        )
        axes[1, 0].plot(
            time,
            follower.loc[mask, "a_mps2"],
            color=COLORS[model],
            linestyle=LINESTYLES[model],
            linewidth=1.2,
        )
        jerk = np.gradient(follower["a_mps2"].to_numpy(), DT)
        axes[1, 1].plot(
            time,
            jerk[mask.to_numpy()],
            color=COLORS[model],
            linestyle=LINESTYLES[model],
            linewidth=1.2,
        )
    axes[0, 0].set_ylabel("Speed (m/s)")
    axes[0, 1].set_ylabel("Spacing error (m)")
    axes[1, 0].set_ylabel("Acceleration (m/s$^2$)")
    axes[1, 1].set_ylabel("Jerk (m/s$^3$)")
    for axis in axes.ravel():
        axis.grid(True, color="#D8D8D8", linewidth=0.5)
    for axis in axes[1]:
        axis.set_xlabel("Time (s)")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=600)
    plt.close(figure)


def plot_normalized_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    metric_columns = [
        "mean_abs_jerk_mps3",
        "speed_std_mps",
        "spacing_rmse_m",
        "fuel_L_per_100km",
    ]
    labels = ["Mean abs. jerk", "Speed std.", "Spacing RMSE", "Fuel/100 km"]
    normalized = metrics.set_index("model")[metric_columns].copy()
    for column in metric_columns:
        normalized[column] /= normalized[column].max()
    x = np.arange(len(labels))
    width = 0.24
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
        }
    )
    figure, axis = plt.subplots(figsize=(6.03, 3.2), dpi=300)
    for index, model in enumerate(MODELS):
        axis.bar(
            x + (index - 1) * width,
            normalized.loc[model].to_numpy(),
            width=width,
            color=COLORS[model],
            label=model,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=12, ha="right")
    axis.set_ylabel("Normalized value (lower is better)")
    axis.set_ylim(0.0, 1.08)
    axis.grid(True, axis="y", color="#D8D8D8", linewidth=0.5)
    axis.legend(frameon=False, ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=600)
    plt.close(figure)


def equilibrium_gap(
    model: str,
    params: np.ndarray,
    equilibrium_speed: float,
    station: float,
) -> float:
    gaps = np.linspace(6.0, 90.0, 8401)
    accelerations = np.array(
        [
            compute_model_acceleration(
                model,
                params,
                station,
                equilibrium_speed,
                equilibrium_speed,
                gap,
            )
            for gap in gaps
        ]
    )
    return float(gaps[int(np.argmin(np.abs(accelerations)))])


def leader_perturbation(time: np.ndarray, equilibrium_speed: float) -> np.ndarray:
    return (
        equilibrium_speed
        + 0.80 * np.exp(-((time - 22.0) / 4.0) ** 2)
        - 0.70 * np.exp(-((time - 39.0) / 4.8) ** 2)
    )


def simulate_platoon(
    model: str,
    params: np.ndarray,
    equilibrium_speed: float = 9.5,
    followers: int = 8,
    duration: float = 65.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    time = np.arange(0.0, duration + DT, DT)
    count = followers + 1
    speed = np.full((len(time), count), equilibrium_speed, dtype=float)
    station = np.zeros((len(time), count), dtype=float)
    acceleration = np.zeros((len(time), count), dtype=float)
    speed[:, 0] = leader_perturbation(time, equilibrium_speed)
    station[0, 0] = 250.0
    gap = equilibrium_gap(model, params, equilibrium_speed, station[0, 0])
    for vehicle in range(1, count):
        station[0, vehicle] = station[0, vehicle - 1] - gap
    for index in range(len(time) - 1):
        station[index + 1, 0] = station[index, 0] + speed[index, 0] * DT
        for vehicle in range(1, count):
            current_gap = station[index, vehicle - 1] - station[index, vehicle]
            acceleration[index, vehicle] = compute_model_acceleration(
                model,
                params,
                station[index, vehicle],
                speed[index, vehicle],
                speed[index, vehicle - 1],
                current_gap,
            )
            speed[index + 1, vehicle] = max(
                0.0,
                speed[index, vehicle] + acceleration[index, vehicle] * DT,
            )
            station[index + 1, vehicle] = (
                station[index, vehicle]
                + speed[index, vehicle] * DT
                + 0.5 * acceleration[index, vehicle] * DT * DT
            )
    deviation = speed - equilibrium_speed
    leader_peak = float(np.max(np.abs(deviation[:, 0])))
    ratios = np.max(np.abs(deviation), axis=0) / max(leader_peak, 1e-9)
    time_rows = []
    for vehicle in range(count):
        for index, current_time in enumerate(time):
            time_rows.append(
                {
                    "model": model,
                    "time_s": current_time,
                    "vehicle_index": vehicle,
                    "speed_deviation_mps": deviation[index, vehicle],
                }
            )
    ratio_rows = [
        {
            "model": model,
            "vehicle_index": vehicle,
            "peak_amplification_ratio": float(ratios[vehicle]),
            "equilibrium_gap_m": gap,
        }
        for vehicle in range(count)
    ]
    return pd.DataFrame(time_rows), pd.DataFrame(ratio_rows)


def plot_disturbance_heatmaps(timeseries: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
        }
    )
    figure, axes = plt.subplots(3, 1, figsize=(5.82, 4.0), dpi=300, sharex=True)
    maximum = float(np.max(np.abs(timeseries["speed_deviation_mps"])))
    image = None
    for panel_index, (axis, model) in enumerate(zip(axes, MODELS)):
        subset = timeseries[timeseries["model"] == model]
        pivot = subset.pivot(
            index="vehicle_index",
            columns="time_s",
            values="speed_deviation_mps",
        )
        image = axis.imshow(
            pivot.to_numpy(),
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            vmin=-maximum,
            vmax=maximum,
            extent=[
                float(pivot.columns.min()),
                float(pivot.columns.max()),
                float(pivot.index.min()),
                float(pivot.index.max()),
            ],
        )
        axis.text(
            0.012,
            0.86,
            f"({chr(97 + panel_index)}) {model}",
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
        )
        axis.set_ylabel("Vehicle index")
    axes[-1].set_xlabel("Time (s)")
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes, fraction=0.035, pad=0.025)
        colorbar.set_label("Speed deviation (m/s)")
    figure.subplots_adjust(left=0.12, right=0.86, bottom=0.12, top=0.98, hspace=0.10)
    figure.savefig(output_path, dpi=600)
    plt.close(figure)


def plot_amplification_ratios(ratios: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
        }
    )
    figure, axis = plt.subplots(figsize=(4.755, 3.0), dpi=300)
    for model in MODELS:
        subset = ratios[ratios["model"] == model]
        axis.plot(
            subset["vehicle_index"],
            subset["peak_amplification_ratio"],
            color=COLORS[model],
            linestyle=LINESTYLES[model],
            marker="o",
            markersize=3.0,
            linewidth=1.4,
            label=model,
        )
    axis.axhline(1.0, color="#333333", linewidth=0.8, linestyle=":")
    axis.set_xlabel("Vehicle index")
    axis.set_ylabel("Peak amplification ratio")
    axis.grid(True, color="#D8D8D8", linewidth=0.5)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=600)
    plt.close(figure)


def find_default_runs_csv(script_dir: Path) -> Path:
    candidates = (
        script_dir / "fair_calibration_runs.csv",
        script_dir.parent / "outputs" / "fair_calibration_runs.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def default_output_dir(script_dir: Path) -> Path:
    if (script_dir.parent / "outputs").is_dir():
        return script_dir.parent / "reproduced_outputs"
    return script_dir / "reproduced_outputs"


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Reproduce Figs. 15-16 and the underlying baseline outputs."
    )
    parser.add_argument(
        "fair_calibration_runs_csv",
        type=Path,
        nargs="?",
        default=None,
        help="Defaults to fair_calibration_runs.csv next to the script or in ../outputs.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Defaults to a reproduced_outputs directory without overwriting published files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    runs_csv = args.fair_calibration_runs_csv or find_default_runs_csv(script_dir)
    output_dir = args.output_dir or default_output_dir(script_dir)
    if not runs_csv.is_file():
        parser.error(
            "Cannot find fair_calibration_runs.csv. Place it next to this script "
            "or in the repository outputs directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    parameters = load_seed_parameters(runs_csv, args.seed)
    leader = generate_leader()
    followers = {
        model: simulate_follower(model, parameters[model], leader)
        for model in MODELS
    }
    metrics = pd.DataFrame(
        [
            {"model": model, **compute_metrics(followers[model])}
            for model in MODELS
        ]
    )
    timeseries = pd.concat(list(followers.values()), ignore_index=True)
    metrics.to_csv(
        output_dir / "recalibrated_baseline_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    timeseries.to_csv(
        output_dir / "recalibrated_baseline_timeseries.csv",
        index=False,
        encoding="utf-8-sig",
    )
    leader.to_csv(
        output_dir / "recalibrated_baseline_leader.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_dynamic_response(
        leader,
        followers,
        output_dir / "Fig_21_recalibrated.png",
    )
    plot_normalized_metrics(
        metrics,
        output_dir / "Fig_22_recalibrated.png",
    )

    platoon_series = []
    platoon_ratios = []
    for model in MODELS:
        series, ratios = simulate_platoon(model, parameters[model])
        platoon_series.append(series)
        platoon_ratios.append(ratios)
    disturbance_series = pd.concat(platoon_series, ignore_index=True)
    amplification_ratios = pd.concat(platoon_ratios, ignore_index=True)
    disturbance_series.to_csv(
        output_dir / "recalibrated_disturbance_timeseries.csv",
        index=False,
        encoding="utf-8-sig",
    )
    amplification_ratios.to_csv(
        output_dir / "recalibrated_amplification_ratios.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_disturbance_heatmaps(
        disturbance_series,
        output_dir / "Fig_15_recalibrated.png",
    )
    plot_amplification_ratios(
        amplification_ratios,
        output_dir / "Fig_16_recalibrated.png",
    )

    print(metrics.round(4).to_string(index=False))
    print(
        amplification_ratios[
            amplification_ratios["vehicle_index"] == amplification_ratios["vehicle_index"].max()
        ].round(4).to_string(index=False)
    )
    print(f"Inputs read from {runs_csv}")
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
