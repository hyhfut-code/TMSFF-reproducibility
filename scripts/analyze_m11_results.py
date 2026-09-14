"""Independent evaluation, paired statistics, tables and journal figures for M11."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

from m11_models import trace_avidm, trace_tmsff, wrap_angle
from run_m11_calibration import (
    MODEL_PARAMETERS,
    block_objective,
    load_cache,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_K16 = PROJECT_ROOT / "reproduced_outputs" / "cache" / "m11_citysim_cache_k16.npz"
CACHE_K80 = PROJECT_ROOT / "reproduced_outputs" / "cache" / "m11_citysim_cache_k80.npz"
PAIRS_PATH = PROJECT_ROOT / "scripts" / "citysim_selected_following_pairs.csv"
CALIBRATION_OUTPUT = PROJECT_ROOT / "outputs" / "M11" / "common_protocol"
ANALYSIS_OUTPUT = PROJECT_ROOT / "reproduced_outputs" / "M11" / "analysis"
FIGURE_OUTPUT = PROJECT_ROOT / "reproduced_outputs"
EXPANDED_OUTPUT = PROJECT_ROOT / "outputs" / "M11" / "expanded_bound_audit"
FIGURE_DPI = 600

COLORS = {
    "TMSFF": "#0077BB",
    "AV-IDM": "#EE7733",
    "observed": "#222222",
    "leader": "#009988",
    "neutral": "#BBBBBB",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "mathtext.sf": "Times New Roman",
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.5,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
        }
    )


def seed_parameters(model: str, seed: int) -> np.ndarray:
    estimates = pd.read_csv(CALIBRATION_OUTPUT / "ga_parameter_estimates.csv")
    subset = estimates[(estimates["model"] == model) & (estimates["seed"] == seed)].set_index(
        "parameter"
    )
    return subset.loc[list(MODEL_PARAMETERS[model]), "estimate"].to_numpy(float)


def projected_lateral(
    x: np.ndarray,
    y: np.ndarray,
    route_x: np.ndarray,
    route_y: np.ndarray,
    route_heading: np.ndarray,
) -> np.ndarray:
    output = np.zeros(len(x), dtype=float)
    previous = 0
    for time_index, (px, py) in enumerate(zip(x, y)):
        start = max(0, previous - 14)
        stop = min(len(route_x), previous + 25)
        local = (route_x[start:stop] - px) ** 2 + (route_y[start:stop] - py) ** 2
        previous = start + int(np.argmin(local))
        beta = route_heading[previous]
        output[time_index] = (
            -(px - route_x[previous]) * math.sin(beta)
            + (py - route_y[previous]) * math.cos(beta)
        )
    return output


def trace_metrics(
    model: str,
    trace: np.ndarray,
    observed: np.ndarray,
    route_x: np.ndarray,
    route_y: np.ndarray,
    route_heading: np.ndarray,
    left_width: np.ndarray,
    right_width: np.ndarray,
) -> dict[str, float]:
    position_error = np.hypot(trace[:, 0] - observed[:, 0], trace[:, 1] - observed[:, 1])
    speed_error = trace[:, 2] - observed[:, 2]
    heading_error = np.array(
        [wrap_angle(float(a - b)) for a, b in zip(trace[:, 3], observed[:, 3])]
    )
    observed_lateral = projected_lateral(
        observed[:, 0], observed[:, 1], route_x, route_y, route_heading
    )
    simulated_lateral = projected_lateral(
        trace[:, 0], trace[:, 1], route_x, route_y, route_heading
    )
    lateral_error = simulated_lateral - observed_lateral
    outside = np.zeros(len(trace), dtype=bool)
    previous = 0
    for index, (x, y) in enumerate(zip(trace[:, 0], trace[:, 1])):
        start = max(0, previous - 14)
        stop = min(len(route_x), previous + 25)
        local = (route_x[start:stop] - x) ** 2 + (route_y[start:stop] - y) ** 2
        previous = start + int(np.argmin(local))
        outside[index] = (
            simulated_lateral[index] > left_width[previous]
            or simulated_lateral[index] < -right_width[previous]
        )
    acceleration_error = trace[:, 4] - observed[:, 4]
    return {
        "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "position_mae_m": float(np.mean(position_error)),
        "speed_rmse_mps": float(np.sqrt(np.mean(speed_error**2))),
        "heading_mae_deg": float(np.degrees(np.mean(np.abs(heading_error)))),
        "lateral_rmse_m": float(np.sqrt(np.mean(lateral_error**2))),
        "acceleration_rmse_mps2": float(np.sqrt(np.mean(acceleration_error**2))),
        "boundary_exceedance_share": float(np.mean(outside)),
        "minimum_front_gap_m": float(np.nanmin(trace[:-1, 6])) if len(trace) > 1 else np.nan,
    }


def evaluate_seed42(data: dict, pairs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    dt = float(data["dt"])
    parameters = {
        "TMSFF": seed_parameters("TMSFF", 42),
        "AV-IDM": seed_parameters("AV-IDM", 42),
    }
    records: list[dict] = []
    representative: dict = {}
    for pair_index in range(len(data["lengths"])):
        start = int(data["split_indices"][pair_index])
        stop = int(data["lengths"][pair_index])
        observed = data["observed"][pair_index, start:stop]
        metadata = data["pair_metadata"][pair_index]
        selection = pairs.iloc[pair_index]
        traces = {
            "TMSFF": trace_tmsff(
                parameters["TMSFF"],
                pair_index,
                start,
                stop,
                data["observed"],
                data["leader"],
                dt,
            ),
            "AV-IDM": trace_avidm(
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
                dt,
            ),
        }
        for model, trace in traces.items():
            metrics = trace_metrics(
                model,
                trace,
                observed,
                data["route_x"][pair_index],
                data["route_y"][pair_index],
                data["route_heading"][pair_index],
                data["route_left_width"][pair_index],
                data["route_right_width"][pair_index],
            )
            records.append(
                {
                    "model": model,
                    "seed": 42,
                    "trajectory_no": int(metadata[0]),
                    "leader_id": int(metadata[1]),
                    "follower_id": int(metadata[2]),
                    "entry_lane": int(metadata[3]),
                    "exit_lane": int(metadata[4]),
                    "holdout_frames": stop - start,
                    "route_balanced_subset": bool(selection["route_balanced_subset"]),
                    **metrics,
                }
            )
        if int(metadata[1]) == 6 and int(metadata[2]) == 59:
            representative = {
                "pair_index": pair_index,
                "start": start,
                "stop": stop,
                "observed": observed,
                "leader": data["leader"][pair_index, start:stop],
                "TMSFF": traces["TMSFF"],
                "AV-IDM": traces["AV-IDM"],
                "dt": dt,
                "metadata": metadata,
            }
    if not representative:
        raise RuntimeError("Pre-specified representative pair 6->59 was not found")
    return pd.DataFrame(records), representative


def bootstrap_mean_difference(
    first: np.ndarray,
    second: np.ndarray,
    samples: int = 10000,
    seed: int = 20260829,
) -> tuple[float, float, float]:
    difference = first - second
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(samples, len(difference)))
    means = difference[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(difference.mean()), float(lower), float(upper)


def paired_statistics(per_trajectory: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "position_rmse_m",
        "speed_rmse_mps",
        "heading_mae_deg",
        "lateral_rmse_m",
        "acceleration_rmse_mps2",
    ]
    output = []
    for metric in metrics:
        pivot = per_trajectory.pivot(index="trajectory_no", columns="model", values=metric)
        tmsff = pivot["TMSFF"].to_numpy(float)
        avidm = pivot["AV-IDM"].to_numpy(float)
        mean_difference, ci_lower, ci_upper = bootstrap_mean_difference(tmsff, avidm)
        test = stats.wilcoxon(tmsff, avidm, alternative="two-sided", zero_method="wilcox")
        output.append(
            {
                "metric": metric,
                "trajectories": len(pivot),
                "TMSFF_mean": float(np.mean(tmsff)),
                "TMSFF_sd": float(np.std(tmsff, ddof=1)),
                "AV_IDM_mean": float(np.mean(avidm)),
                "AV_IDM_sd": float(np.std(avidm, ddof=1)),
                "mean_paired_difference_TMSFF_minus_AVIDM": mean_difference,
                "bootstrap_95ci_lower": ci_lower,
                "bootstrap_95ci_upper": ci_upper,
                "TMSFF_lower_error_count": int(np.sum(tmsff < avidm)),
                "tie_count": int(np.sum(np.isclose(tmsff, avidm))),
                "wilcoxon_statistic": float(test.statistic),
                "wilcoxon_two_sided_p": float(test.pvalue),
            }
        )
    return pd.DataFrame(output)


def aggregate_per_trajectory(per_trajectory: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "position_rmse_m",
        "speed_rmse_mps",
        "heading_mae_deg",
        "lateral_rmse_m",
        "acceleration_rmse_mps2",
        "boundary_exceedance_share",
    ]
    records = []
    for subset_name, subset in (
        ("all_38", per_trajectory),
        ("route_balanced_13", per_trajectory[per_trajectory["route_balanced_subset"]]),
    ):
        for model, group in subset.groupby("model"):
            for metric in metrics:
                values = group[metric].to_numpy(float)
                records.append(
                    {
                        "subset": subset_name,
                        "model": model,
                        "metric": metric,
                        "trajectory_count": len(values),
                        "mean": float(np.mean(values)),
                        "sample_sd": float(np.std(values, ddof=1)),
                        "median": float(np.median(values)),
                        "q25": float(np.quantile(values, 0.25)),
                        "q75": float(np.quantile(values, 0.75)),
                    }
                )
    return pd.DataFrame(records)


def global_seed_summary() -> pd.DataFrame:
    runs = pd.read_csv(CALIBRATION_OUTPUT / "ga_runs.csv")
    records = []
    for model, group in runs.groupby("model"):
        records.append(
            {
                "model": model,
                "seeds": len(group),
                "calibration_objective_mean_m": group["calibration_objective_m"].mean(),
                "calibration_objective_sd_m": group["calibration_objective_m"].std(ddof=1),
                "holdout_objective_mean_m": group["holdout_objective_m"].mean(),
                "holdout_objective_sd_m": group["holdout_objective_m"].std(ddof=1),
                "mean_elapsed_seconds": group["elapsed_seconds"].mean(),
            }
        )
    return pd.DataFrame(records)


def parameter_summary() -> pd.DataFrame:
    estimates = pd.read_csv(CALIBRATION_OUTPUT / "ga_parameter_estimates.csv")
    records = []
    for (model, parameter), group in estimates.groupby(["model", "parameter"], sort=False):
        seed42 = group.loc[group["seed"] == 42].iloc[0]
        records.append(
            {
                "model": model,
                "parameter": parameter,
                "seed42_estimate": seed42["estimate"],
                "five_seed_mean": group["estimate"].mean(),
                "five_seed_sd": group["estimate"].std(ddof=1),
                "lower_bound": seed42["lower_bound"],
                "upper_bound": seed42["upper_bound"],
                "seed42_relative_bound_position": seed42["relative_bound_position"],
                "near_bound_within_2pct": bool(
                    seed42["relative_bound_position"] < 0.02
                    or seed42["relative_bound_position"] > 0.98
                ),
            }
        )
    return pd.DataFrame(records)


def candidate_and_bound_audits(data_k16: dict) -> pd.DataFrame:
    avidm = seed_parameters("AV-IDM", 42)
    data_k80 = load_cache(CACHE_K80)
    records = []
    for label, data in (("K16 calibration implementation", data_k16), ("all visible vehicles", data_k80)):
        records.append(
            {
                "audit": "dynamic-candidate sensitivity",
                "case": label,
                "calibration_objective_m": block_objective("AV-IDM", avidm, True, data),
                "holdout_objective_m": block_objective("AV-IDM", avidm, False, data),
            }
        )
    expanded_path = EXPANDED_OUTPUT / "expanded_bound_summary.json"
    if expanded_path.exists():
        with expanded_path.open("r", encoding="utf-8") as handle:
            expanded = json.load(handle)
        records.append(
            {
                "audit": "parameter-bound sensitivity",
                "case": "published original bounds, seed 42",
                "calibration_objective_m": block_objective("AV-IDM", avidm, True, data_k16),
                "holdout_objective_m": block_objective("AV-IDM", avidm, False, data_k16),
            }
        )
        records.append(
            {
                "audit": "parameter-bound sensitivity",
                "case": "expanded bounds, independent seed-42 GA",
                "calibration_objective_m": expanded["calibration_objective_m"],
                "holdout_objective_m": expanded["holdout_objective_m"],
            }
        )
    return pd.DataFrame(records)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png",):
        fig.savefig(
            FIGURE_OUTPUT / f"{stem}.{extension}",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_protocol_figure() -> None:
    convergence = pd.read_csv(CALIBRATION_OUTPUT / "ga_convergence.csv")
    runs = pd.read_csv(CALIBRATION_OUTPUT / "ga_runs.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), constrained_layout=True)
    for model in ("TMSFF", "AV-IDM"):
        summary = (
            convergence[convergence["model"] == model]
            .groupby("generation")["best_objective_m"]
            .agg(["mean", "std"])
            .reset_index()
        )
        x = summary["generation"].to_numpy(float)
        mean = summary["mean"].to_numpy(float)
        sd = summary["std"].fillna(0.0).to_numpy(float)
        axes[0].plot(x, mean, color=COLORS[model], lw=1.7, label=model)
        axes[0].fill_between(x, mean - sd, mean + sd, color=COLORS[model], alpha=0.16, linewidth=0)
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Best calibration objective (m)")
    axes[0].set_xlim(0, 80)
    axes[0].grid(axis="y", color="#DDDDDD", lw=0.6)
    axes[0].legend(frameon=False)
    axes[0].text(-0.12, 1.04, "(a)", transform=axes[0].transAxes, fontweight="bold")

    positions = {("TMSFF", "calibration"): 0, ("TMSFF", "holdout"): 1, ("AV-IDM", "calibration"): 2.5, ("AV-IDM", "holdout"): 3.5}
    for model in ("TMSFF", "AV-IDM"):
        group = runs[runs["model"] == model].sort_values("seed")
        x1 = positions[(model, "calibration")]
        x2 = positions[(model, "holdout")]
        for row in group.itertuples(index=False):
            axes[1].plot(
                [x1, x2],
                [row.calibration_objective_m, row.holdout_objective_m],
                color=COLORS[model],
                alpha=0.42,
                lw=0.8,
            )
            axes[1].scatter(
                [x1, x2],
                [row.calibration_objective_m, row.holdout_objective_m],
                color=COLORS[model],
                s=17,
                zorder=3,
            )
    axes[1].set_xticks([0, 1, 2.5, 3.5], ["Calibration", "Holdout", "Calibration", "Holdout"], rotation=18)
    axes[1].text(0.5, -0.27, "TMSFF", ha="center", transform=axes[1].get_xaxis_transform())
    axes[1].text(3.0, -0.27, "AV-IDM", ha="center", transform=axes[1].get_xaxis_transform())
    axes[1].set_ylabel("Two-dimensional objective (m)")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(axis="y", color="#DDDDDD", lw=0.6)
    axes[1].legend(
        handles=[
            Line2D([0], [0], color=COLORS["TMSFF"], marker="o", lw=1.2, markersize=4, label="TMSFF"),
            Line2D([0], [0], color=COLORS["AV-IDM"], marker="o", lw=1.2, markersize=4, label="AV-IDM"),
        ],
        frameon=False,
        loc="lower center",
        ncol=2,
    )
    axes[1].text(-0.12, 1.04, "(b)", transform=axes[1].transAxes, fontweight="bold")
    save_figure(fig, "Fig_M11_1_common_protocol_and_generalization")


def plot_holdout_figure(
    per_trajectory: pd.DataFrame,
    representative: dict,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    axis = axes[0, 0]
    axis.plot(
        representative["leader"][:, 0],
        representative["leader"][:, 1],
        color=COLORS["leader"],
        ls="--",
        lw=1.3,
        label="Leader",
    )
    axis.plot(
        representative["observed"][:, 0],
        representative["observed"][:, 1],
        color=COLORS["observed"],
        lw=1.8,
        label="Observed follower",
    )
    for model in ("TMSFF", "AV-IDM"):
        axis.plot(
            representative[model][:, 0],
            representative[model][:, 1],
            color=COLORS[model],
            lw=1.4,
            label=model,
        )
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.legend(frameon=False, loc="best")
    axis.text(-0.12, 1.04, "(a)", transform=axis.transAxes, fontweight="bold")

    axis = axes[0, 1]
    time_s = np.arange(len(representative["observed"])) * representative["dt"]
    for model in ("TMSFF", "AV-IDM"):
        error = np.hypot(
            representative[model][:, 0] - representative["observed"][:, 0],
            representative[model][:, 1] - representative["observed"][:, 1],
        )
        axis.plot(time_s, error, color=COLORS[model], lw=1.5, label=model)
    axis.set_xlabel("Holdout time (s)")
    axis.set_ylabel("Position error (m)")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#DDDDDD", lw=0.6)
    axis.legend(frameon=False)
    axis.text(-0.12, 1.04, "(b)", transform=axis.transAxes, fontweight="bold")

    axis = axes[1, 0]
    pivot = per_trajectory.pivot(index="trajectory_no", columns="model", values="position_rmse_m")
    for _, row in pivot.iterrows():
        axis.plot([0, 1], [row["TMSFF"], row["AV-IDM"]], color="#CCCCCC", lw=0.65, zorder=1)
        axis.scatter(0, row["TMSFF"], color=COLORS["TMSFF"], s=13, zorder=2)
        axis.scatter(1, row["AV-IDM"], color=COLORS["AV-IDM"], s=13, zorder=2)
    axis.set_xticks([0, 1], ["TMSFF", "AV-IDM"])
    axis.set_ylabel("Per-trajectory holdout RMSE (m)")
    axis.set_xlim(-0.35, 1.35)
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#DDDDDD", lw=0.6)
    axis.legend(
        handles=[
            Line2D([0], [0], color=COLORS["TMSFF"], marker="o", lw=0, markersize=5, label="TMSFF"),
            Line2D([0], [0], color=COLORS["AV-IDM"], marker="o", lw=0, markersize=5, label="AV-IDM"),
        ],
        frameon=False,
        loc="upper left",
    )
    axis.text(-0.12, 1.04, "(c)", transform=axis.transAxes, fontweight="bold")

    axis = axes[1, 1]
    route = (
        per_trajectory.groupby(["entry_lane", "exit_lane", "model"])["position_rmse_m"]
        .mean()
        .unstack("model")
        .sort_values("TMSFF")
    )
    y = np.arange(len(route))
    labels = [f"{int(a)}→{int(b)}" for a, b in route.index]
    for index in range(len(route)):
        axis.plot(
            [route.iloc[index]["TMSFF"], route.iloc[index]["AV-IDM"]],
            [y[index], y[index]],
            color="#CCCCCC",
            lw=0.8,
        )
    axis.scatter(route["TMSFF"], y, color=COLORS["TMSFF"], s=20, label="TMSFF", zorder=3)
    axis.scatter(route["AV-IDM"], y, color=COLORS["AV-IDM"], s=20, label="AV-IDM", zorder=3)
    axis.set_yticks(y, labels)
    axis.set_xlabel("Route-mean holdout RMSE (m)")
    axis.set_ylabel("Entry→exit route")
    axis.set_xlim(left=0)
    axis.grid(axis="x", color="#DDDDDD", lw=0.6)
    axis.legend(frameon=False, loc="lower right")
    axis.text(-0.12, 1.04, "(d)", transform=axis.transAxes, fontweight="bold")
    save_figure(fig, "Fig_M11_2_CitySim_holdout_comparison")


def make_main_table(
    global_summary: pd.DataFrame,
    aggregate: pd.DataFrame,
    statistics: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    position_stats = statistics.set_index("metric").loc["position_rmse_m"]
    for model in ("TMSFF", "AV-IDM"):
        global_row = global_summary.set_index("model").loc[model]
        trajectory_row = aggregate[
            (aggregate["subset"] == "all_38")
            & (aggregate["model"] == model)
            & (aggregate["metric"] == "position_rmse_m")
        ].iloc[0]
        records.append(
            {
                "model": model,
                "free_parameters": len(MODEL_PARAMETERS[model]),
                "calibration_objective_mean_m": global_row["calibration_objective_mean_m"],
                "calibration_objective_sd_m": global_row["calibration_objective_sd_m"],
                "holdout_objective_mean_m": global_row["holdout_objective_mean_m"],
                "holdout_objective_sd_m": global_row["holdout_objective_sd_m"],
                "seed42_per_trajectory_position_RMSE_mean_m": trajectory_row["mean"],
                "seed42_per_trajectory_position_RMSE_sd_m": trajectory_row["sample_sd"],
                "seed42_per_trajectory_position_RMSE_median_m": trajectory_row["median"],
                "TMSFF_lower_error_trajectories_out_of_38": int(position_stats["TMSFF_lower_error_count"]),
                "paired_wilcoxon_two_sided_p": position_stats["wilcoxon_two_sided_p"],
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    configure_matplotlib()
    ANALYSIS_OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_cache(CACHE_K16)
    pairs = pd.read_csv(PAIRS_PATH)
    per_trajectory, representative = evaluate_seed42(data, pairs)
    statistics_table = paired_statistics(per_trajectory)
    aggregate = aggregate_per_trajectory(per_trajectory)
    global_summary = global_seed_summary()
    parameters = parameter_summary()
    audits = candidate_and_bound_audits(data)
    main_table = make_main_table(global_summary, aggregate, statistics_table)

    per_trajectory.to_csv(
        ANALYSIS_OUTPUT / "seed42_holdout_per_trajectory.csv", index=False, encoding="utf-8-sig"
    )
    statistics_table.to_csv(
        ANALYSIS_OUTPUT / "paired_statistics.csv", index=False, encoding="utf-8-sig"
    )
    aggregate.to_csv(
        ANALYSIS_OUTPUT / "holdout_aggregate_summary.csv", index=False, encoding="utf-8-sig"
    )
    global_summary.to_csv(
        ANALYSIS_OUTPUT / "five_seed_global_objective_summary.csv", index=False, encoding="utf-8-sig"
    )
    parameters.to_csv(
        ANALYSIS_OUTPUT / "parameter_summary.csv", index=False, encoding="utf-8-sig"
    )
    audits.to_csv(
        ANALYSIS_OUTPUT / "robustness_audits.csv", index=False, encoding="utf-8-sig"
    )
    main_table.to_csv(
        ANALYSIS_OUTPUT / "Table_M11_common_protocol_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_protocol_figure()
    plot_holdout_figure(per_trajectory, representative)

    representative_rows = []
    for index in range(len(representative["observed"])):
        representative_rows.append(
            {
                "time_s": index * representative["dt"],
                "leader_x_m": representative["leader"][index, 0],
                "leader_y_m": representative["leader"][index, 1],
                "observed_x_m": representative["observed"][index, 0],
                "observed_y_m": representative["observed"][index, 1],
                "TMSFF_x_m": representative["TMSFF"][index, 0],
                "TMSFF_y_m": representative["TMSFF"][index, 1],
                "AV_IDM_x_m": representative["AV-IDM"][index, 0],
                "AV_IDM_y_m": representative["AV-IDM"][index, 1],
            }
        )
    pd.DataFrame(representative_rows).to_csv(
        ANALYSIS_OUTPUT / "representative_pair_6_59_holdout_timeseries.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\nFive-seed global summary")
    print(global_summary.to_string(index=False))
    print("\nPaired statistics")
    print(statistics_table.to_string(index=False))
    print("\nRobustness audits")
    print(audits.to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-output", type=Path, default=PROJECT_ROOT / "outputs" / "M11")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "reproduced_outputs" / "cache")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reproduced_outputs" / "M11" / "analysis")
    args = parser.parse_args()
    CALIBRATION_OUTPUT = args.source_output / "common_protocol"
    EXPANDED_OUTPUT = args.source_output / "expanded_bound_audit"
    CACHE_K16 = args.cache_dir / "m11_citysim_cache_k16.npz"
    CACHE_K80 = args.cache_dir / "m11_citysim_cache_k80.npz"
    ANALYSIS_OUTPUT = args.output
    main()
