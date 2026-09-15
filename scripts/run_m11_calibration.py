"""Run the frozen same-objective, equal-budget M11 calibration protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from m11_models import (
    evaluate_population_avidm,
    evaluate_population_tmsff,
    individual_objective_avidm,
    individual_objective_tmsff,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = PROJECT_ROOT / "reproduced_outputs" / "cache" / "m11_citysim_cache_k16.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "reproduced_outputs" / "M11" / "common_protocol"
DEFAULT_PROTOCOL = PROJECT_ROOT / "scripts" / "m11_protocol.json"

MODEL_PARAMETERS = {
    "TMSFF": (
        "a_max",
        "b_comfort",
        "T",
        "v_desired",
        "s0",
    ),
    "AV-IDM": (
        "C_front",
        "C_rear",
        "C_left",
        "C_right",
        "a_lat",
        "b_lat",
        "s0_lat",
        "T_lat",
        "a_long",
        "v_desired_long",
        "b_long",
        "s0_long",
        "T_long",
    ),
}

MODEL_BOUNDS = {
    "TMSFF": np.array(
        [[0.1, 3.0], [0.1, 3.0], [0.1, 2.0], [3.0, 30.0], [0.1, 5.0]],
        dtype=float,
    ),
    "AV-IDM": np.array(
        [
            [0.5, 3.0],
            [0.5, 3.0],
            [0.5, 3.0],
            [0.5, 3.0],
            [0.25, 1.0],
            [1.0, 4.0],
            [0.2, 3.0],
            [0.5, 2.0],
            [1.0, 4.0],
            [5.0, 20.0],
            [1.0, 4.0],
            [1.0, 4.0],
            [0.5, 2.0],
        ],
        dtype=float,
    ),
}


def load_protocol(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_cache(path: Path) -> dict[str, np.ndarray | float]:
    source = np.load(path)
    data: dict[str, np.ndarray | float] = {key: source[key] for key in source.files}
    data["dt"] = float(source["dt"][0])
    return data


def polynomial_mutation(
    individual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    gene_probability: float,
    eta: float,
) -> np.ndarray:
    mutated = individual.copy()
    for index in range(len(mutated)):
        if rng.random() >= gene_probability:
            continue
        x = mutated[index]
        xl = lower[index]
        xu = upper[index]
        delta_1 = (x - xl) / (xu - xl)
        delta_2 = (xu - x) / (xu - xl)
        mutation_power = 1.0 / (eta + 1.0)
        random_value = rng.random()
        if random_value < 0.5:
            xy = 1.0 - delta_1
            value = (
                2.0 * random_value
                + (1.0 - 2.0 * random_value) * xy ** (eta + 1.0)
            )
            delta_q = value**mutation_power - 1.0
        else:
            xy = 1.0 - delta_2
            value = (
                2.0 * (1.0 - random_value)
                + 2.0 * (random_value - 0.5) * xy ** (eta + 1.0)
            )
            delta_q = 1.0 - value**mutation_power
        mutated[index] = np.clip(x + delta_q * (xu - xl), xl, xu)
    return mutated


def tournament_pick(
    fitness: np.ndarray,
    rng: np.random.Generator,
    size: int,
) -> int:
    candidates = rng.integers(0, len(fitness), size=size)
    return int(candidates[np.argmin(fitness[candidates])])


def evaluate_population(model: str, population: np.ndarray, data: dict) -> np.ndarray:
    common = (
        data["observed"],
        data["lengths"],
        data["split_indices"],
        data["route_x"],
        data["route_y"],
        data["route_heading"],
        data["route_left_width"],
        data["route_right_width"],
        data["route_station"],
        data["dt"],
    )
    if model == "TMSFF":
        return evaluate_population_tmsff(population, common[0], data["leader"], *common[1:])
    return evaluate_population_avidm(population, common[0], data["environment"], *common[1:])


def block_objective(model: str, parameters: np.ndarray, calibration: bool, data: dict) -> float:
    common = (
        data["observed"],
        data["lengths"],
        data["split_indices"],
        data["route_x"],
        data["route_y"],
        data["route_heading"],
        data["route_left_width"],
        data["route_right_width"],
        data["route_station"],
        data["dt"],
    )
    if model == "TMSFF":
        return float(
            individual_objective_tmsff(
                parameters, calibration, common[0], data["leader"], *common[1:]
            )
        )
    return float(
        individual_objective_avidm(
            parameters, calibration, common[0], data["environment"], *common[1:]
        )
    )


def run_ga(
    model: str,
    seed: int,
    data: dict,
    settings: dict,
) -> tuple[np.ndarray, float, float, list[dict], float]:
    rng = np.random.default_rng(seed)
    bounds = MODEL_BOUNDS[model]
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    population_size = int(settings["population_size"])
    generations = int(settings["generations"])
    population = rng.uniform(lower, upper, size=(population_size, len(bounds)))
    started = time.perf_counter()
    fitness = evaluate_population(model, population, data)
    log: list[dict] = []

    def append_log(generation: int) -> None:
        best_index = int(np.argmin(fitness))
        log.append(
            {
                "model": model,
                "seed": seed,
                "generation": generation,
                "evaluations": population_size * (generation + 1),
                "best_objective_m": float(fitness[best_index]),
                "median_objective_m": float(np.median(fitness)),
                "mean_objective_m": float(np.mean(fitness)),
            }
        )

    append_log(0)
    for generation in range(1, generations + 1):
        elite_indices = np.argsort(fitness)[: int(settings["elitism"])]
        next_population = [population[index].copy() for index in elite_indices]
        while len(next_population) < population_size:
            first = population[
                tournament_pick(fitness, rng, int(settings["tournament_size"]))
            ].copy()
            second = population[
                tournament_pick(fitness, rng, int(settings["tournament_size"]))
            ].copy()
            if rng.random() < float(settings["crossover_probability"]):
                exchange = rng.random(len(bounds)) < float(
                    settings["uniform_gene_exchange_probability"]
                )
                first_original = first.copy()
                first[exchange] = second[exchange]
                second[exchange] = first_original[exchange]
            if rng.random() < float(settings["mutation_individual_probability"]):
                first = polynomial_mutation(
                    first,
                    lower,
                    upper,
                    rng,
                    float(settings["mutation_gene_probability"]),
                    float(settings["polynomial_mutation_index"]),
                )
            if rng.random() < float(settings["mutation_individual_probability"]):
                second = polynomial_mutation(
                    second,
                    lower,
                    upper,
                    rng,
                    float(settings["mutation_gene_probability"]),
                    float(settings["polynomial_mutation_index"]),
                )
            next_population.append(first)
            if len(next_population) < population_size:
                next_population.append(second)
        population = np.asarray(next_population, dtype=float)
        fitness = evaluate_population(model, population, data)
        append_log(generation)
        if generation == 1 or generation % 10 == 0 or generation == generations:
            print(
                f"{model} seed {seed}: generation {generation}/{generations}, "
                f"best={fitness.min():.6f} m",
                flush=True,
            )

    best_index = int(np.argmin(fitness))
    parameters = population[best_index].copy()
    calibration_objective = float(fitness[best_index])
    holdout_objective = block_objective(model, parameters, False, data)
    elapsed = time.perf_counter() - started
    return parameters, calibration_objective, holdout_objective, log, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_PARAMETERS), default=list(MODEL_PARAMETERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    optimizer = protocol["optimizer"].copy()
    if args.population_size is not None:
        optimizer["population_size"] = args.population_size
    if args.generations is not None:
        optimizer["generations"] = args.generations
    seeds = args.seeds if args.seeds is not None else list(optimizer["seeds"])
    data = load_cache(args.cache)
    args.output.mkdir(parents=True, exist_ok=True)
    run_records: list[dict] = []
    parameter_records: list[dict] = []
    convergence_records: list[dict] = []

    for model in args.models:
        for seed in seeds:
            print(f"Starting {model}, seed {seed}", flush=True)
            parameters, calibration, holdout, log, elapsed = run_ga(
                model, seed, data, optimizer
            )
            run_records.append(
                {
                    "model": model,
                    "seed": seed,
                    "calibration_objective_m": calibration,
                    "holdout_objective_m": holdout,
                    "elapsed_seconds": elapsed,
                    "population_size": optimizer["population_size"],
                    "generations": optimizer["generations"],
                    "evaluations": int(optimizer["population_size"])
                    * (int(optimizer["generations"]) + 1),
                }
            )
            for name, value, bound in zip(
                MODEL_PARAMETERS[model], parameters, MODEL_BOUNDS[model]
            ):
                parameter_records.append(
                    {
                        "model": model,
                        "seed": seed,
                        "parameter": name,
                        "estimate": float(value),
                        "lower_bound": float(bound[0]),
                        "upper_bound": float(bound[1]),
                        "relative_bound_position": float((value - bound[0]) / (bound[1] - bound[0])),
                    }
                )
            convergence_records.extend(log)
            pd.DataFrame(run_records).to_csv(
                args.output / "ga_runs.csv", index=False, encoding="utf-8-sig"
            )
            pd.DataFrame(parameter_records).to_csv(
                args.output / "ga_parameter_estimates.csv", index=False, encoding="utf-8-sig"
            )
            pd.DataFrame(convergence_records).to_csv(
                args.output / "ga_convergence.csv", index=False, encoding="utf-8-sig"
            )

    metadata = {
        "cache": str(args.cache.resolve()),
        "models": args.models,
        "seeds": seeds,
        "optimizer": optimizer,
        "actual_evaluations_per_run": int(optimizer["population_size"])
        * (int(optimizer["generations"]) + 1),
        "trajectory_count": int(len(data["lengths"])),
        "calibration_frames": int(np.asarray(data["split_indices"]).sum()),
        "holdout_frames": int(
            (np.asarray(data["lengths"]) - np.asarray(data["split_indices"])).sum()
        ),
    }
    with (args.output / "calibration_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(pd.DataFrame(run_records).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
