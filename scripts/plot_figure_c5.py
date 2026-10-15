"""Regenerate manuscript Fig. C5 from the frozen CitySim and 13-trajectory Hefei summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODULE_ROOT
OUTPUTS = MODULE_ROOT / "outputs"
FIGURES = REPOSITORY_ROOT / "reproduced_outputs"

CITYSIM_FILE = OUTPUTS / "Table_M4_CitySim_full_sensitivity.csv"
HEFEI_FILE = OUTPUTS / "Table_C4a_Hefei_13trajectory_Sc_sensitivity.csv"

SC_BASELINE_M = 15.8
MODELS = ("T-IDM", "T-FVD")
COLORS = {"T-IDM": "#0072B2", "T-FVD": "#D55E00"}
MARKERS = {"T-IDM": "o", "T-FVD": "s"}
METRICS = (
    ("Heading angle RMSE", "Heading angle_RMSE", "Heading_RMSE_rad", "RMSE (rad)"),
    ("Arc spacing RMSE", "Arc spacing_RMSE", "Spacing_RMSE_m", "RMSE (m)"),
    ("Velocity RMSE", "Velocity_RMSE", "Velocity_RMSE_mps", "RMSE (m/s)"),
    ("Acceleration RMSE", "Acceleration_RMSE", "Acceleration_RMSE_mps2", "RMSE (m/s$^2$)"),
)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=OUTPUTS)
    parser.add_argument("--figures", type=Path, default=FIGURES)
    args = parser.parse_args()
    apply_style()
    citysim = pd.read_csv(args.outputs / CITYSIM_FILE.name)
    hefei = pd.read_csv(args.outputs / HEFEI_FILE.name)
    args.figures.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 4, figsize=(15.5, 7.2), constrained_layout=False)
    for row_index, (frame, model_column, row_label) in enumerate(
        (
            (citysim, "model", "CitySim: 38 observed pairs"),
            (hefei, "Model", "Hefei: 13 observed trajectories"),
        )
    ):
        for column_index, (title, citysim_metric, hefei_metric, ylabel) in enumerate(METRICS):
            axis = axes[row_index, column_index]
            metric = citysim_metric if row_index == 0 else hefei_metric
            for model in MODELS:
                subset = frame[frame[model_column] == model].sort_values("S_c_m")
                axis.plot(
                    subset["S_c_m"],
                    subset[metric],
                    color=COLORS[model],
                    marker=MARKERS[model],
                    markersize=4.5,
                    linewidth=1.7,
                    label=model,
                )
            axis.axvline(SC_BASELINE_M, color="#666666", linestyle="--", linewidth=1.0)
            axis.set_title(title, fontweight="bold")
            axis.set_xlabel(r"$S_c$ (m)")
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
            if column_index == 0:
                axis.text(
                    -0.28,
                    0.5,
                    row_label,
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontweight="bold",
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    figure.text(0.985, 0.98, r"Dashed line: manuscript $S_c=15.8$ m", ha="right", va="top", color="#555555")
    figure.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.09, hspace=0.42, wspace=0.35)

    base = args.figures / "Fig_C5"
    figure.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
