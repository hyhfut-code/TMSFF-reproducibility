from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = ("T-IDM", "C-FVD", "GAPB")
COLORS = {
    "T-IDM": "#0072B2",
    "C-FVD": "#D55E00",
    "GAPB": "#7F7F7F",
}
# Match the palette used by the earlier Fig. 22: deep blue, muted gold,
# and medium gray. Fig. 21 retains its established line palette.
FIG22_COLORS = {
    "T-IDM": "#0057B8",
    "C-FVD": "#D1AC2F",
    "GAPB": "#8F8F8F",
}
LINESTYLES = {"T-IDM": "-", "C-FVD": "--", "GAPB": "-."}
MARKERS = {"T-IDM": "o", "C-FVD": "D", "GAPB": "^"}
DT = 0.1
INITIALIZATION_TIME_S = 3.0
TARGET_CENTER_GAP_M = 22.0
SAVGOL_WINDOW_SAMPLES = 41
SAVGOL_POLYNOMIAL_ORDER = 3


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10.5,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 10.5,
            "axes.linewidth": 1.0,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "butt",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def savitzky_golay_for_display(
    values: np.ndarray | pd.Series,
    window_length: int = SAVGOL_WINDOW_SAMPLES,
    polynomial_order: int = SAVGOL_POLYNOMIAL_ORDER,
) -> np.ndarray:
    """Central Savitzky-Golay smoothing with reflected edges for display only."""
    array = np.asarray(values, dtype=float)
    if window_length % 2 != 1 or window_length <= polynomial_order:
        raise ValueError("The smoothing window must be odd and exceed the polynomial order")
    if len(array) < window_length:
        return array.copy()
    half_window = window_length // 2
    local_x = np.arange(-half_window, half_window + 1, dtype=float)
    design = np.vander(local_x, polynomial_order + 1, increasing=True)
    coefficients = np.linalg.pinv(design)[0]
    padded = np.pad(array, (half_window, half_window), mode="reflect")
    return np.convolve(padded, coefficients[::-1], mode="valid")


def load_inputs(
    timeseries_path: Path, leader_path: Path, metrics_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timeseries = pd.read_csv(timeseries_path)
    leader = pd.read_csv(leader_path)
    metrics = pd.read_csv(metrics_path)

    required_timeseries = {"time_s", "model", "v_mps", "a_mps2", "gap_m"}
    required_leader = {"time_s", "leader_v_mps"}
    required_metrics = {
        "model",
        "mean_abs_jerk_mps3",
        "speed_std_mps",
        "spacing_rmse_m",
        "fuel_L_per_100km",
    }
    if not required_timeseries.issubset(timeseries.columns):
        raise ValueError(f"Missing time-series columns: {required_timeseries - set(timeseries.columns)}")
    if not required_leader.issubset(leader.columns):
        raise ValueError(f"Missing leader columns: {required_leader - set(leader.columns)}")
    if not required_metrics.issubset(metrics.columns):
        raise ValueError(f"Missing metrics columns: {required_metrics - set(metrics.columns)}")

    if set(timeseries["model"].unique()) != set(MODELS):
        raise ValueError("The time-series file does not contain exactly T-IDM, C-FVD, and GAPB")
    if set(metrics["model"].unique()) != set(MODELS):
        raise ValueError("The metrics file does not contain exactly T-IDM, C-FVD, and GAPB")
    return timeseries, leader, metrics


def finish_axis(axis: plt.Axes, zero_line: bool = False) -> None:
    axis.grid(True, color="#D8D8D8", linewidth=0.65, alpha=0.72)
    axis.set_axisbelow(True)
    axis.spines["top"].set_color("#333333")
    axis.spines["right"].set_color("#333333")
    axis.spines["bottom"].set_color("#333333")
    axis.spines["left"].set_color("#333333")
    if zero_line:
        axis.axhline(0.0, color="#555555", linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.025,
        0.94,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        color="#111111",
    )


def save_figure(figure: plt.Figure, output_base: Path) -> None:
    figure.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(
        output_base.with_suffix(".tif"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def plot_fig21(
    timeseries: pd.DataFrame, leader: pd.DataFrame, output_dir: Path
) -> None:
    apply_style()
    figure, axes = plt.subplots(2, 2, figsize=(7.25, 5.55), sharex=True)
    mask_leader = leader["time_s"] >= INITIALIZATION_TIME_S
    leader_eval = leader.loc[mask_leader].copy()
    leader_time = leader_eval["time_s"].to_numpy() - INITIALIZATION_TIME_S
    leader_speed = savitzky_golay_for_display(leader_eval["leader_v_mps"])
    axes[0, 0].plot(
        leader_time,
        leader_speed,
        color="#222222",
        linewidth=1.65,
        label="Leader",
        zorder=5,
    )

    for model in MODELS:
        follower = timeseries[timeseries["model"] == model].sort_values("time_s")
        acceleration_all = follower["a_mps2"].to_numpy(dtype=float)
        jerk_all = np.gradient(acceleration_all, DT)
        mask = follower["time_s"].to_numpy() >= INITIALIZATION_TIME_S
        time = follower.loc[mask, "time_s"].to_numpy() - INITIALIZATION_TIME_S

        speed = savitzky_golay_for_display(follower.loc[mask, "v_mps"])
        spacing_error = savitzky_golay_for_display(
            follower.loc[mask, "gap_m"].to_numpy() - TARGET_CENTER_GAP_M
        )
        acceleration = savitzky_golay_for_display(follower.loc[mask, "a_mps2"])
        # Jerk is derivative-based and therefore contains substantially more
        # high-frequency numerical noise. A second zero-phase display pass
        # removes the remaining spikes without changing any reported metric.
        jerk = savitzky_golay_for_display(
            savitzky_golay_for_display(jerk_all[mask])
        )

        plot_kwargs = {
            "color": COLORS[model],
            "linestyle": LINESTYLES[model],
            "linewidth": 1.9,
            "label": model,
            "zorder": 4,
        }
        axes[0, 0].plot(time, speed, **plot_kwargs)
        axes[0, 1].plot(time, spacing_error, **plot_kwargs)
        axes[1, 0].plot(time, acceleration, **plot_kwargs)
        axes[1, 1].plot(time, jerk, **plot_kwargs)

    axes[0, 0].set_ylabel("Speed (m/s)")
    axes[0, 1].set_ylabel("Spacing error (m)")
    axes[1, 0].set_ylabel(r"Acceleration (m/s$^2$)")
    axes[1, 1].set_ylabel(r"Jerk (m/s$^3$)")
    axes[1, 0].set_xlabel("Time after initialization (s)")
    axes[1, 1].set_xlabel("Time after initialization (s)")

    panel_labels = ("(a) Speed", "(b) Spacing error", "(c) Acceleration", "(d) Jerk")
    for axis, label in zip(axes.ravel(), panel_labels):
        finish_axis(axis, zero_line=axis in (axes[0, 1], axes[1, 0], axes[1, 1]))
        add_panel_label(axis, label)
        axis.set_xlim(0.0, float(timeseries["time_s"].max() - INITIALIZATION_TIME_S))

    spacing_lines = [
        line.get_ydata() for line in axes[0, 1].lines if len(line.get_ydata()) > 2
    ]
    spacing_max = max(float(np.max(values)) for values in spacing_lines)
    axes[0, 1].set_ylim(0.0, spacing_max * 1.09)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        handlelength=3.0,
        columnspacing=1.7,
    )
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.105, top=0.90, wspace=0.24, hspace=0.14)
    save_figure(figure, output_dir / "Fig21_dynamic_response_smoothed_display")
    plt.close(figure)


def plot_fig22(metrics: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    apply_style()
    metric_columns = (
        "mean_abs_jerk_mps3",
        "speed_std_mps",
        "spacing_rmse_m",
        "fuel_L_per_100km",
    )
    labels = (r"Mean $|$jerk$|$", "Speed std.", "Spacing RMSE", "Fuel / 100 km")
    normalized = metrics.set_index("model").loc[list(MODELS), list(metric_columns)].copy()
    normalized = normalized / normalized.max(axis=0)

    x = np.arange(len(labels), dtype=float)
    width = 0.19
    offsets = {"T-IDM": -width, "C-FVD": 0.0, "GAPB": width}
    figure, axis = plt.subplots(figsize=(7.25, 4.15))

    for model in MODELS:
        values = normalized.loc[model].to_numpy(dtype=float)
        positions = x + offsets[model]
        axis.bar(
            positions,
            values,
            width=width,
            color=FIG22_COLORS[model],
            edgecolor=FIG22_COLORS[model],
            linewidth=1.05,
            alpha=0.34,
            zorder=2,
        )
        axis.plot(
            positions,
            values,
            color=FIG22_COLORS[model],
            linewidth=1.55,
            marker=MARKERS[model],
            markersize=6.8,
            markerfacecolor="white",
            markeredgecolor="#111111",
            markeredgewidth=0.9,
            label=model,
            zorder=4,
        )
        for xpos, value in zip(positions, values):
            axis.text(
                xpos,
                value + 0.035,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                color=FIG22_COLORS[model],
                fontsize=9.4,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.6},
                zorder=5,
            )

    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=17, ha="right")
    axis.set_ylabel("Normalized value (lower is better)")
    axis.set_ylim(0.0, 1.16)
    axis.set_xlim(-0.55, len(labels) - 0.45)
    axis.set_yticks(np.arange(0.0, 1.01, 0.2))
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.65, alpha=0.78)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(1.0)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=3,
        frameon=False,
        handlelength=2.8,
        columnspacing=2.1,
    )
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.22, top=0.83)
    save_figure(figure, output_dir / "Fig22_normalized_performance_bar_line")
    plt.close(figure)

    normalized.index.name = "model"
    normalized.to_csv(output_dir / "Fig22_normalized_values.csv", float_format="%.9f")
    return normalized


def write_notes(output_dir: Path, normalized: pd.DataFrame) -> None:
    metadata = {
        "figure_21": {
            "display_smoothing": "Savitzky-Golay",
            "polynomial_order": SAVGOL_POLYNOMIAL_ORDER,
            "window_samples": SAVGOL_WINDOW_SAMPLES,
            "time_step_s": DT,
            "physical_window_s": SAVGOL_WINDOW_SAMPLES * DT,
            "jerk_display": "The same zero-phase filter is applied twice to the derivative-based jerk trace.",
            "metric_calculation": "All reported metrics remain based on the original unsmoothed signals.",
        },
        "figure_22": {
            "normalization": "Each metric is divided by the maximum value among the three models.",
            "metrics": list(normalized.columns),
            "models": list(normalized.index),
        },
    }
    (output_dir / "figure_redraw_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    captions = (
        "Fig. 21. Dynamic responses of the same-data-calibrated T-IDM, C-FVD, and GAPB "
        "models under the common controlled curved-road scenario: (a) speed, (b) center-to-center "
        "spacing error, (c) acceleration, and (d) jerk. The curves were smoothed using a "
        "zero-phase, third-order Savitzky-Golay filter with a 41-sample (4.1 s) window for visualization "
        "only; the filter was applied twice to the derivative-based jerk traces to suppress differentiation "
        "noise. All reported performance indicators were calculated from the original unsmoothed signals.\n\n"
        "Fig. 22. Normalized longitudinal performance comparison among the same-data-calibrated "
        "T-IDM, C-FVD, and GAPB models. Each indicator was divided by the maximum value among the "
        "three models, and lower values indicate better performance.\n"
    )
    (output_dir / "recommended_captions.txt").write_text(captions, encoding="utf-8")


def find_default_data_dir(script_dir: Path) -> Path:
    required_names = (
        "recalibrated_baseline_timeseries.csv",
        "recalibrated_baseline_leader.csv",
        "recalibrated_baseline_metrics.csv",
    )
    candidates = (
        script_dir / "reproduced_outputs",
        script_dir,
        script_dir.parent / "reproduced_outputs",
        script_dir.parent / "outputs",
    )
    for candidate in candidates:
        if all((candidate / name).is_file() for name in required_names):
            return candidate
    return candidates[0]


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_data_dir = find_default_data_dir(script_dir)
    parser = argparse.ArgumentParser(
        description="Create the publication versions of Figs. 21 and 22."
    )
    parser.add_argument("--timeseries", type=Path, default=None)
    parser.add_argument("--leader", type=Path, default=None)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    timeseries_path = args.timeseries or default_data_dir / "recalibrated_baseline_timeseries.csv"
    leader_path = args.leader or default_data_dir / "recalibrated_baseline_leader.csv"
    metrics_path = args.metrics or default_data_dir / "recalibrated_baseline_metrics.csv"
    output_dir = args.output_dir or default_data_dir
    missing = [
        path
        for path in (timeseries_path, leader_path, metrics_path)
        if not path.is_file()
    ]
    if missing:
        parser.error(
            "Cannot find the required input CSV files: "
            + ", ".join(str(path) for path in missing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    timeseries, leader, metrics = load_inputs(timeseries_path, leader_path, metrics_path)
    plot_fig21(timeseries, leader, output_dir)
    normalized = plot_fig22(metrics, output_dir)
    write_notes(output_dir, normalized)
    print(f"Inputs read from {default_data_dir}")
    print(f"Publication figures written to {output_dir}")


if __name__ == "__main__":
    main()
