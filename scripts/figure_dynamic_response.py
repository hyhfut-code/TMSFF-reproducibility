# -*- coding: utf-8 -*-
"""Plot the dynamic response comparison for the curved-road baseline test."""

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


def smooth(time: np.ndarray, values: np.ndarray, window_s: float = 0.9) -> np.ndarray:
    dt = float(np.nanmedian(np.diff(time)))
    window = max(3, int(round(window_s / max(dt, 1e-6))))
    if window % 2 == 0:
        window += 1
    return pd.Series(values).rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def polish(ax: plt.Axes) -> None:
    ax.grid(True, color="#D9D9D9", linewidth=0.55, alpha=0.82)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.85)


def main() -> None:
    set_style()
    data = pd.read_csv(CSV_FILE)
    data = data[data["time_s"] >= 3.0].copy()
    colors = {"TMSFF": "#0072B2", "C-FVD": "#D55E00", "GACF": "#7F7F7F"}
    styles = {"TMSFF": "-", "C-FVD": "--", "GACF": "-."}
    panels = [
        ("v_mps", "Speed (m/s)", "(a) Speed", False),
        ("spacing_error_m", "Spacing error (m)", "(b) Spacing error", True),
        ("a_mps2", r"Acceleration (m/s$^2$)", "(c) Acceleration", True),
        ("jerk_mps3", r"Jerk (m/s$^3$)", "(d) Jerk", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.05), dpi=600, sharex=True)
    axes = axes.ravel()
    for ax, (column, ylabel, label, zero_line) in zip(axes, panels):
        for model in ["TMSFF", "C-FVD", "GACF"]:
            sub = data[data["model"] == model].sort_values("time_s")
            time = sub["time_s"].to_numpy()
            values = sub[column].to_numpy()
            if column == "jerk_mps3":
                values = smooth(time, smooth(time, sub["a_mps2"].to_numpy()), 1.1)
                values = np.gradient(values, time)
                values = smooth(time, values, 1.1)
            else:
                values = smooth(time, values, 0.9)
            ax.plot(time, values, color=colors[model], linestyle=styles[model], lw=1.7, label=model)
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

    for ax in axes[2:]:
        ax.set_xlabel("Time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 1.01), ncol=3, frameon=False)
    fig.tight_layout(pad=0.55, rect=[0.0, 0.0, 1.0, 0.955])
    fig.savefig(FIG_FILE, dpi=600, bbox_inches="tight", pad_inches=0.03)
    print(f"Figure written to: {FIG_FILE}")


if __name__ == "__main__":
    main()
