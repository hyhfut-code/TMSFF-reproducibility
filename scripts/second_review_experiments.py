from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DT = 0.1
VEHICLE_LENGTH = 5.0
POPULATION_SIZE = 100
GENERATIONS = 80
TOURNAMENT_SIZE = 3
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.2
GENE_MUTATION_PROBABILITY = 0.2
POLYNOMIAL_ETA = 20.0
ELITE_COUNT = 2
SEEDS = (42, 43, 44, 45, 46)
TRAIN_FRACTION = 0.7
ACCELERATION_LIMITS = (-5.0, 3.0)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    parameter_names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]
    acceleration: Callable[[np.ndarray, float, float, float, float], float]


@dataclass(frozen=True)
class Trajectory:
    leader_s: np.ndarray
    leader_v: np.ndarray
    follower_s: np.ndarray
    follower_v: np.ndarray
    follower_x: np.ndarray
    follower_y: np.ndarray
    curvature: np.ndarray

    def slice(self, start: int, stop: int) -> "Trajectory":
        return Trajectory(
            leader_s=self.leader_s[start:stop].copy(),
            leader_v=self.leader_v[start:stop].copy(),
            follower_s=self.follower_s[start:stop].copy(),
            follower_v=self.follower_v[start:stop].copy(),
            follower_x=self.follower_x[start:stop].copy(),
            follower_y=self.follower_y[start:stop].copy(),
            curvature=self.curvature[start:stop].copy(),
        )


def moving_average(values: np.ndarray, width: int = 5) -> np.ndarray:
    if len(values) < width:
        return values.copy()
    pad = width // 2
    padded = np.pad(values, pad, mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def path_curvature(station: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    station = np.asarray(station, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    station_safe = station.copy()
    for index in range(1, len(station_safe)):
        if station_safe[index] <= station_safe[index - 1]:
            station_safe[index] = station_safe[index - 1] + 1e-6
    dx = np.gradient(x, station_safe)
    dy = np.gradient(y, station_safe)
    ddx = np.gradient(dx, station_safe)
    ddy = np.gradient(dy, station_safe)
    denominator = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)
    return moving_average(np.abs(dx * ddy - dy * ddx) / denominator, width=5)


def load_trajectory(path: Path) -> Trajectory:
    data = pd.read_csv(path)
    required = {"x1", "y1", "x2", "y2", "v1", "v2", "s1", "s2"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="raise")
    curvature = path_curvature(
        data["s2"].to_numpy(),
        data["x2"].to_numpy(),
        data["y2"].to_numpy(),
    )
    return Trajectory(
        leader_s=data["s1"].to_numpy(dtype=float),
        leader_v=data["v1"].to_numpy(dtype=float),
        follower_s=data["s2"].to_numpy(dtype=float),
        follower_v=data["v2"].to_numpy(dtype=float),
        follower_x=data["x2"].to_numpy(dtype=float),
        follower_y=data["y2"].to_numpy(dtype=float),
        curvature=curvature,
    )


def acceleration_tidm(
    params: np.ndarray,
    speed: float,
    leader_speed: float,
    center_gap: float,
    curvature: float,
) -> float:
    del curvature
    a_max, b_comfort, time_gap, desired_speed, minimum_gap = params
    net_gap = max(center_gap - VEHICLE_LENGTH, 0.5)
    closing_speed = speed - leader_speed
    desired_gap = minimum_gap + max(
        0.0,
        speed * time_gap
        + speed * closing_speed / (2.0 * np.sqrt(max(a_max * b_comfort, 1e-9))),
    )
    return float(
        a_max
        * (
            1.0
            - (speed / max(desired_speed, 0.1)) ** 4
            - (desired_gap / net_gap) ** 2
        )
    )


def acceleration_tfvd(
    params: np.ndarray,
    speed: float,
    leader_speed: float,
    center_gap: float,
    curvature: float,
) -> float:
    del curvature
    kappa, lambda_speed, maximum_speed, critical_gap = params
    optimal_speed = 0.5 * maximum_speed * (
        np.tanh(center_gap - critical_gap) + np.tanh(critical_gap)
    )
    return float(
        kappa * (optimal_speed - speed)
        + lambda_speed * (leader_speed - speed)
    )


def acceleration_cfvd(
    params: np.ndarray,
    speed: float,
    leader_speed: float,
    center_gap: float,
    curvature: float,
) -> float:
    kappa, lambda_speed, maximum_speed, critical_gap, transition_width, curvature_penalty = params
    curve_limited_speed = max(3.0, maximum_speed - curvature_penalty * curvature)
    denominator = 1.0 + np.tanh(critical_gap / max(transition_width, 0.1))
    optimal_speed = curve_limited_speed * (
        np.tanh((center_gap - critical_gap) / max(transition_width, 0.1))
        + np.tanh(critical_gap / max(transition_width, 0.1))
    ) / max(denominator, 1e-6)
    return float(
        kappa * (optimal_speed - speed)
        + lambda_speed * (leader_speed - speed)
    )


def acceleration_gapb(
    params: np.ndarray,
    speed: float,
    leader_speed: float,
    center_gap: float,
    curvature: float,
) -> float:
    (
        a_max,
        b_comfort,
        desired_speed,
        time_gap,
        minimum_gap,
        curvature_gap_gain,
        curvature_speed_penalty,
        lambda_speed,
    ) = params
    net_gap = max(center_gap - VEHICLE_LENGTH, 0.5)
    curve_limited_speed = max(3.0, desired_speed - curvature_speed_penalty * curvature)
    closing_speed = speed - leader_speed
    desired_gap = (
        minimum_gap
        + time_gap * speed
        + curvature_gap_gain * curvature * speed * speed
        + max(
            0.0,
            speed * closing_speed / (2.0 * np.sqrt(max(a_max * b_comfort, 1e-9))),
        )
    )
    return float(
        a_max
        * (
            1.0
            - (speed / max(curve_limited_speed, 0.1)) ** 4
            - (desired_gap / net_gap) ** 2
        )
        + lambda_speed * (leader_speed - speed)
    )


MODEL_SPECS = (
    ModelSpec(
        name="T-IDM",
        parameter_names=("a_max", "b_comfort", "time_gap", "desired_speed", "minimum_gap"),
        bounds=((0.1, 3.0), (0.1, 3.0), (0.1, 2.0), (3.0, 30.0), (0.1, 5.0)),
        acceleration=acceleration_tidm,
    ),
    ModelSpec(
        name="T-FVD",
        parameter_names=("kappa", "lambda_speed", "maximum_speed", "critical_gap"),
        bounds=((0.1, 5.0), (0.1, 1.0), (3.0, 30.0), (0.1, 20.0)),
        acceleration=acceleration_tfvd,
    ),
    ModelSpec(
        name="C-FVD",
        parameter_names=(
            "kappa",
            "lambda_speed",
            "maximum_speed",
            "critical_gap",
            "transition_width",
            "curvature_penalty",
        ),
        bounds=((0.1, 5.0), (0.0, 2.0), (3.0, 30.0), (2.0, 30.0), (0.5, 15.0), (0.0, 500.0)),
        acceleration=acceleration_cfvd,
    ),
    ModelSpec(
        name="GAPB",
        parameter_names=(
            "a_max",
            "b_comfort",
            "desired_speed",
            "time_gap",
            "minimum_gap",
            "curvature_gap_gain",
            "curvature_speed_penalty",
            "lambda_speed",
        ),
        bounds=(
            (0.1, 3.0),
            (0.1, 3.0),
            (3.0, 30.0),
            (0.1, 3.0),
            (0.1, 10.0),
            (0.0, 200.0),
            (0.0, 500.0),
            (0.0, 1.0),
        ),
        acceleration=acceleration_gapb,
    ),
)


def simulate(spec: ModelSpec, params: np.ndarray, trajectory: Trajectory) -> dict[str, np.ndarray]:
    count = len(trajectory.follower_s)
    station = np.zeros(count, dtype=float)
    speed = np.zeros(count, dtype=float)
    acceleration = np.zeros(count, dtype=float)
    station[0] = trajectory.follower_s[0]
    speed[0] = trajectory.follower_v[0]

    for index in range(count - 1):
        center_gap = trajectory.leader_s[index] - station[index]
        if not np.isfinite(center_gap) or center_gap <= VEHICLE_LENGTH + 0.25:
            acceleration[index] = ACCELERATION_LIMITS[0]
        else:
            acceleration[index] = spec.acceleration(
                params,
                speed[index],
                trajectory.leader_v[index],
                center_gap,
                trajectory.curvature[index],
            )
        acceleration[index] = float(
            np.clip(
                acceleration[index],
                ACCELERATION_LIMITS[0],
                ACCELERATION_LIMITS[1],
            )
        )
        next_speed = max(0.0, speed[index] + acceleration[index] * DT)
        station[index + 1] = (
            station[index]
            + speed[index] * DT
            + 0.5 * acceleration[index] * DT * DT
        )
        speed[index + 1] = next_speed
    acceleration[-1] = acceleration[-2] if count > 1 else 0.0

    x_sim = np.interp(
        station,
        trajectory.follower_s,
        trajectory.follower_x,
    )
    y_sim = np.interp(
        station,
        trajectory.follower_s,
        trajectory.follower_y,
    )
    return {
        "station": station,
        "speed": speed,
        "acceleration": acceleration,
        "x": x_sim,
        "y": y_sim,
    }


def trajectory_rmse(spec: ModelSpec, params: np.ndarray, trajectory: Trajectory) -> float:
    simulation = simulate(spec, params, trajectory)
    squared_error = (
        (simulation["x"] - trajectory.follower_x) ** 2
        + (simulation["y"] - trajectory.follower_y) ** 2
    )
    lower_excess = np.maximum(trajectory.follower_s.min() - simulation["station"], 0.0)
    upper_excess = np.maximum(simulation["station"] - trajectory.follower_s.max(), 0.0)
    range_penalty = np.mean(lower_excess * lower_excess + upper_excess * upper_excess)
    return float(np.sqrt(np.mean(squared_error) + range_penalty))


def polynomial_mutation(
    individual: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    mutated = individual.copy()
    for index in range(len(mutated)):
        if rng.random() >= GENE_MUTATION_PROBABILITY:
            continue
        x = mutated[index]
        xl = low[index]
        xu = high[index]
        delta_1 = (x - xl) / (xu - xl)
        delta_2 = (xu - x) / (xu - xl)
        mutation_power = 1.0 / (POLYNOMIAL_ETA + 1.0)
        random_value = rng.random()
        if random_value < 0.5:
            xy = 1.0 - delta_1
            value = (
                2.0 * random_value
                + (1.0 - 2.0 * random_value) * xy ** (POLYNOMIAL_ETA + 1.0)
            )
            delta_q = value**mutation_power - 1.0
        else:
            xy = 1.0 - delta_2
            value = (
                2.0 * (1.0 - random_value)
                + 2.0 * (random_value - 0.5) * xy ** (POLYNOMIAL_ETA + 1.0)
            )
            delta_q = 1.0 - value**mutation_power
        mutated[index] = np.clip(x + delta_q * (xu - xl), xl, xu)
    return mutated


def evaluate_population(
    spec: ModelSpec,
    population: np.ndarray,
    trajectory: Trajectory,
) -> np.ndarray:
    return np.array(
        [trajectory_rmse(spec, individual, trajectory) for individual in population],
        dtype=float,
    )


def tournament_pick(
    fitness: np.ndarray,
    rng: np.random.Generator,
) -> int:
    candidates = rng.integers(0, len(fitness), size=TOURNAMENT_SIZE)
    return int(candidates[np.argmin(fitness[candidates])])


def run_ga(
    spec: ModelSpec,
    trajectory: Trajectory,
    seed: int,
) -> tuple[np.ndarray, float, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    bounds = np.asarray(spec.bounds, dtype=float)
    low = bounds[:, 0]
    high = bounds[:, 1]
    population = rng.uniform(low, high, size=(POPULATION_SIZE, len(bounds)))
    fitness = evaluate_population(spec, population, trajectory)
    log_rows: list[dict[str, float | int | str]] = []

    for generation in range(GENERATIONS + 1):
        log_rows.append(
            {
                "model": spec.name,
                "seed": seed,
                "generation": generation,
                "best_rmse_m": float(np.min(fitness)),
                "median_rmse_m": float(np.median(fitness)),
                "mean_rmse_m": float(np.mean(fitness)),
            }
        )
        if generation == GENERATIONS:
            break

        elite_indices = np.argsort(fitness)[:ELITE_COUNT]
        next_population = [population[index].copy() for index in elite_indices]
        while len(next_population) < POPULATION_SIZE:
            first = population[tournament_pick(fitness, rng)].copy()
            second = population[tournament_pick(fitness, rng)].copy()
            if rng.random() < CROSSOVER_PROBABILITY:
                exchange_mask = rng.random(len(bounds)) < 0.5
                first_values = first.copy()
                first[exchange_mask] = second[exchange_mask]
                second[exchange_mask] = first_values[exchange_mask]
            if rng.random() < MUTATION_PROBABILITY:
                first = polynomial_mutation(first, low, high, rng)
            if rng.random() < MUTATION_PROBABILITY:
                second = polynomial_mutation(second, low, high, rng)
            next_population.append(first)
            if len(next_population) < POPULATION_SIZE:
                next_population.append(second)
        population = np.asarray(next_population, dtype=float)
        fitness = evaluate_population(spec, population, trajectory)

    best_index = int(np.argmin(fitness))
    return population[best_index].copy(), float(fitness[best_index]), pd.DataFrame(log_rows)


def plot_convergence(logs: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
        }
    )
    colors = {
        "T-IDM": "#0072B2",
        "T-FVD": "#D55E00",
        "C-FVD": "#009E73",
        "GAPB": "#7F7F7F",
    }
    figure, axis = plt.subplots(figsize=(5.2, 3.2), dpi=300)
    for model_name, group in logs.groupby("model"):
        pivot = group.pivot(
            index="generation",
            columns="seed",
            values="best_rmse_m",
        )
        mean = pivot.mean(axis=1)
        std = pivot.std(axis=1)
        axis.plot(
            mean.index,
            mean,
            label=model_name,
            color=colors[model_name],
            linewidth=1.6,
        )
        axis.fill_between(
            mean.index,
            np.maximum(mean - std, 0.0),
            mean + std,
            color=colors[model_name],
            alpha=0.12,
            linewidth=0,
        )
    axis.set_xlabel("Generation")
    axis.set_ylabel("Best trajectory RMSE (m)")
    axis.grid(True, color="#D8D8D8", linewidth=0.5)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=600)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trajectory = load_trajectory(args.data_csv)
    split_index = max(10, min(len(trajectory.follower_s) - 10, int(len(trajectory.follower_s) * TRAIN_FRACTION)))
    calibration = trajectory.slice(0, split_index)
    validation = trajectory.slice(split_index, len(trajectory.follower_s))

    run_rows: list[dict[str, object]] = []
    all_logs: list[pd.DataFrame] = []
    best_by_model: dict[str, dict[str, object]] = {}

    for spec in MODEL_SPECS:
        model_runs: list[dict[str, object]] = []
        for seed in SEEDS:
            params, calibration_rmse, log = run_ga(spec, calibration, seed)
            validation_rmse = trajectory_rmse(spec, params, validation)
            row: dict[str, object] = {
                "model": spec.name,
                "seed": seed,
                "calibration_rmse_m": calibration_rmse,
                "validation_rmse_m": validation_rmse,
            }
            for name, value in zip(spec.parameter_names, params):
                row[name] = float(value)
            run_rows.append(row)
            model_runs.append(row)
            all_logs.append(log)
        selected = min(model_runs, key=lambda item: float(item["calibration_rmse_m"]))
        best_by_model[spec.name] = selected

    runs = pd.DataFrame(run_rows)
    logs = pd.concat(all_logs, ignore_index=True)
    summary = (
        runs.groupby("model", sort=False)
        .agg(
            calibration_rmse_mean_m=("calibration_rmse_m", "mean"),
            calibration_rmse_sd_m=("calibration_rmse_m", "std"),
            validation_rmse_mean_m=("validation_rmse_m", "mean"),
            validation_rmse_sd_m=("validation_rmse_m", "std"),
        )
        .reset_index()
    )

    runs.to_csv(args.output_dir / "fair_calibration_runs.csv", index=False, encoding="utf-8-sig")
    logs.to_csv(args.output_dir / "fair_calibration_convergence.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "fair_calibration_summary.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "selected_calibrated_parameters.json").write_text(
        json.dumps(best_by_model, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_convergence(logs, args.output_dir / "fair_calibration_convergence.png")

    settings = {
        "data_file": str(args.data_csv),
        "records": len(trajectory.follower_s),
        "calibration_records": len(calibration.follower_s),
        "validation_records": len(validation.follower_s),
        "split_rule": f"first {TRAIN_FRACTION:.0%} calibration; remaining temporal block validation",
        "objective": "two-dimensional follower-trajectory RMSE on the common empirical path",
        "population_size": POPULATION_SIZE,
        "generations": GENERATIONS,
        "fitness_evaluations_per_run": POPULATION_SIZE * (GENERATIONS + 1),
        "seeds": list(SEEDS),
        "selection": f"tournament size {TOURNAMENT_SIZE}",
        "crossover": f"uniform, probability {CROSSOVER_PROBABILITY}",
        "mutation": (
            f"bounded polynomial, individual probability {MUTATION_PROBABILITY}, "
            f"gene probability {GENE_MUTATION_PROBABILITY}, eta {POLYNOMIAL_ETA}"
        ),
        "elitism": ELITE_COUNT,
        "acceleration_limits_mps2": list(ACCELERATION_LIMITS),
    }
    (args.output_dir / "fair_calibration_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
