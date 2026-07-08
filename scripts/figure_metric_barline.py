# -*- coding: utf-8 -*-
"""Plot normalized baseline-comparison metrics as bars plus profile lines."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CSV_FILE = OUT / "curved_cf_metrics.csv"
FIG_FILE = OUT / "figure_metric_barline.png"


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
            "axes.labelsize": 9.4,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
        }
    )


def polish(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.82)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.85)


def add_label(ax: plt.Axes, x: float, y: float, color: str) -> None:
    ypos = y - 0.045 if y >= 0.94 else y + 0.035
    va = "top" if y >= 0.94 else "bottom"
    ax.text(
        x,
        ypos,
        f"{y:.2f}",
        ha="center",
        va=va,
        fontsize=6.8,
        color=color,
        bbox={"boxstyle": "square,pad=0.06", "facecolor": "white", "edgecolor": "none", "alpha": 0.70},
        zorder=6,
    )


def main() -> None:
    set_style()
    metrics = pd.read_csv(CSV_FILE)
    metric_columns = [
        "mean_abs_jerk_mps3",
        "speed_std_mps",
        "spacing_rmse_m",
        "trajectory_rmse_m",
        "fuel_L_per_100km",
    ]
    metric_labels = ["Mean |jerk|", "Speed std.", "Spacing RMSE", "Trajectory RMSE", "Fuel / 100 km"]
    normalized = metrics[["model"] + metric_columns].copy()
    for column in metric_columns:
        normalized[column] = normalized[column] / normalized[column].max()

    models = ["TMSFF", "C-FVD", "GACF"]
    fill = {"TMSFF": "#5B9BD5", "C-FVD": "#F1E4B7", "GACF": "#E6E6E6"}
    line = {"TMSFF": "#0057B8", "C-FVD": "#D1AC2F", "GACF": "#8F8F8F"}
    markers = {"TMSFF": "o", "C-FVD": "D", "GACF": "^"}
    x = np.arange(len(metric_columns))
    bar_width = 0.19
    offsets = {"TMSFF": -bar_width, "C-FVD": 0.0, "GACF": bar_width}

    fig, ax = plt.subplots(figsize=(7.2, 3.85), dpi=600)
    for model in models:
        row = normalized[normalized["model"] == model]
        values = row[metric_columns].iloc[0].to_numpy(dtype=float)
        xpos = x + offsets[model]
        ax.bar(xpos, values, width=bar_width, color=fill[model], edgecolor=line[model], linewidth=0.85, alpha=0.74)
        ax.plot(
            xpos,
            values,
            color=line[model],
            marker=markers[model],
            ms=4.6,
            lw=1.55 if model == "TMSFF" else 1.25,
            markerfacecolor=fill[model],
            markeredgecolor="black",
            markeredgewidth=0.55,
            label=model if model == "TMSFF" else f"{model} baseline",
            zorder=4,
        )
        for xi, yi in zip(xpos, values):
            add_label(ax, float(xi), float(yi), line[model])

    ax.set_ylabel("Normalized value (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=18, ha="right", rotation_mode="anchor")
    ax.set_ylim(0.0, 1.12)
    ax.set_xlim(-0.55, len(metric_columns) - 0.45)
    polish(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=3, frameon=False)
    fig.tight_layout(pad=0.35)
    fig.savefig(FIG_FILE, dpi=600, bbox_inches="tight", pad_inches=0.03)
    print(f"Figure written to: {FIG_FILE}")


if __name__ == "__main__":
    main()
