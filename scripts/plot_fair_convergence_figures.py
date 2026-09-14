from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODULE_ROOT = Path(__file__).resolve().parents[1]
INPUT = MODULE_ROOT / "outputs" / "fair_calibration_convergence.csv"
OUTPUT_DIR = MODULE_ROOT / "reproduced_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"T-IDM": "#0072B2", "T-FVD": "#D55E00"}


def plot_model(logs: pd.DataFrame, model: str, output_name: str) -> None:
    subset = logs.loc[logs["model"] == model].copy()
    pivot = subset.pivot(
        index="generation",
        columns="seed",
        values="best_rmse_m",
    ).sort_index()
    generation = pivot.index.to_numpy(dtype=float)
    mean = pivot.mean(axis=1).to_numpy(dtype=float)
    standard_deviation = pivot.std(axis=1, ddof=1).to_numpy(dtype=float)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
        }
    )
    figure, axis = plt.subplots(figsize=(5.35, 3.2), dpi=600)
    for seed in pivot.columns:
        axis.plot(
            generation,
            pivot[seed].to_numpy(dtype=float),
            color=COLORS[model],
            alpha=0.18,
            linewidth=0.75,
        )
    axis.fill_between(
        generation,
        np.maximum(0.0, mean - standard_deviation),
        mean + standard_deviation,
        color=COLORS[model],
        alpha=0.18,
        linewidth=0,
        label="Mean ± SD (five seeds)",
    )
    axis.plot(
        generation,
        mean,
        color=COLORS[model],
        linewidth=1.65,
        label="Mean best objective",
    )
    axis.axvline(
        10,
        color="#333333",
        linewidth=0.9,
        linestyle=":",
        label="Legacy 10-generation limit",
    )
    axis.set_xlabel("Generation")
    axis.set_ylabel("Best 2D trajectory RMSE (m)")
    axis.set_xlim(0, 80)
    axis.set_ylim(bottom=0.0)
    axis.grid(True, color="#D8D8D8", linewidth=0.5)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / output_name, dpi=600)
    plt.close(figure)


def main() -> None:
    logs = pd.read_csv(INPUT)
    plot_model(logs, "T-IDM", "Fig_C1_fair_TIDM.png")
    plot_model(logs, "T-FVD", "Fig_C2_fair_TFVD.png")


if __name__ == "__main__":
    main()
