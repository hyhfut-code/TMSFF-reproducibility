# -*- coding: utf-8 -*-
"""Plot the smoothed dynamic response comparison for the curved-road baseline test.

The smoothing is used only for visual presentation. Quantitative indicators in
``curved_cf_metrics.csv`` are calculated from the original simulation results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CSV_FILE = OUT / "curved_cf_timeseries.csv"
FIG_FILE = OUT / "figure_baseline_dynamic_response.png"


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
            "legend.fontsize": 8.6,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
        }
    )


MODEL_ORDER = ["TMSFF", "C-FVD", "GACF"]
MODEL_LABELS = {
    "TMSFF": "TMSFF",
    "C-FVD": "C-FVD baseline",
    "GACF": "GACF baseline",
}
MODEL_COLORS = {
    "TMSFF": "#0072B2",
    "C-FVD": "#D55E00",
    "GACF": "#7F7F7F",
}
MODEL_STYLES = {
    "TMSFF": "-",
    "C-FVD": "--",
    "GACF": "-.",
}


def smooth(time: np.ndarray, values: np.ndarray, window_s: float = 0.9, passes: int = 2) -> np.ndarray:
    dt = float(np.nanmedian(np.diff(time)))
    window = max(3, int(round(window_s / max(dt, 1e-6))))
    if window % 2 == 0:
        window += 1
    smoothed = pd.Series(values)
    for _ in range(passes):
        smoothed = smoothed.rolling(window=window, center=True, min_periods=1).mean()
    return smoothed.to_numpy()


def polish(ax: plt.Axes) -> None:
    ax.grid(True, color="#D9D9D9", linewidth=0.55, alpha=0.82)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", width=0.8, length=3.2)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.85)


def set_range_by_data(ax: plt.Axes, values: np.ndarray, include_zero: bool = False, symmetric: bool = False) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    if symmetric:
        bound = float(np.nanpercentile(np.abs(values), 98.5))
        bound = max(bound * 1.18, 0.1)
        ax.set_ylim(-bound, bound)
        return
    ymin = float(np.nanpercentile(values, 1.0))
    ymax = float(np.nanpercentile(values, 99.0))
    if include_zero:
        ymin = min(ymin, 0.0)
        ymax = max(ymax, 0.0)
    pad = max((ymax - ymin) * 0.12, 0.05)
    ax.set_ylim(ymin - pad, ymax + pad)


def main() -> None:
    set_style()
    data = pd.read_csv(CSV_FILE)
    data = data[data["time_s"] >= 3.0].copy()
    panels = [
        ("v_mps", "Speed (m/s)", "(a) Speed", False, False),
        ("spacing_error_m", "Spacing error (m)", "(b) Spacing error", True, False),
        ("a_mps2", r"Acceleration (m/s$^2$)", "(c) Acceleration", True, True),
        ("jerk_mps3", r"Jerk (m/s$^3$)", "(d) Jerk", True, True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.05), dpi=600, sharex=True)
    axes = axes.ravel()
    for ax, (column, ylabel, label, zero_line, symmetric) in zip(axes, panels):
        panel_values = []
        for model in MODEL_ORDER:
            sub = data[data["model"] == model].sort_values("time_s")
            time = sub["time_s"].to_numpy()
            values = sub[column].to_numpy()
            if column == "jerk_mps3":
                values = smooth(time, sub["a_mps2"].to_numpy(), 0.9, passes=2)
                values = np.gradient(values, time)
                values = smooth(time, values, 1.1, passes=2)
            else:
                values = smooth(time, values, 0.9, passes=2)
            panel_values.append(values)
            line_width = 1.75 if model == "TMSFF" else 1.55
            ax.plot(
                time,
                values,
                color=MODEL_COLORS[model],
                linestyle=MODEL_STYLES[model],
                lw=line_width,
                label=MODEL_LABELS[model],
            )
        if zero_line:
            ax.axhline(0.0, color="#555555", lw=0.7, ls=(0, (3, 2)))
        ax.set_ylabel(ylabel)
        ax.text(
            0.015,
            0.93,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            bbox={"boxstyle": "square,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
        polish(ax)
        if panel_values:
            set_range_by_data(ax, np.concatenate(panel_values), include_zero=zero_line, symmetric=symmetric)

    for ax in axes[2:]:
        ax.set_xlabel("Time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 1.01), ncol=3, frameon=False)
    fig.tight_layout(pad=0.55, rect=[0.0, 0.0, 1.0, 0.955])
    fig.savefig(FIG_FILE, dpi=600, bbox_inches="tight", pad_inches=0.03)
    print(f"Figure written to: {FIG_FILE}")


if __name__ == "__main__":
    main()
