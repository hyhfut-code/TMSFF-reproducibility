"""Reproduce Fig. E1 from the archived Reviewer 1 Comment 1 audit tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
BLUE = "#1F77B4"
ORANGE = "#E67E22"
DARK_GRAY = "#595959"
PAIR_GRAY = "#B8B8B8"
GRID_GRAY = "#D9D9D9"


def configure_matplotlib() -> None:
    """Configure the 600-dpi Times New Roman manuscript style."""
    for font_path in (
        Path(r"C:\Windows\Fonts\times.ttf"),
        Path(r"C:\Windows\Fonts\timesbd.ttf"),
        Path(r"C:\Windows\Fonts\timesi.ttf"),
        Path(r"C:\Windows\Fonts\timesbi.ttf"),
    ):
        if font_path.is_file():
            font_manager.fontManager.addfont(font_path)
    try:
        font_manager.findfont("Times New Roman", fallback_to_default=False)
    except ValueError as exc:
        raise RuntimeError(
            "Times New Roman is required to reproduce the manuscript figure."
        ) from exc

    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def panel_style(ax: plt.Axes, letter: str) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID_GRAY, linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.75)
    ax.tick_params(direction="out", colors="black")
    ax.text(
        -0.17,
        1.065,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        clip_on=False,
    )


def paired_panel(
    ax: plt.Axes,
    left: np.ndarray,
    right: np.ndarray,
    labels: tuple[str, str],
    ylabel: str,
    p_text: str,
    letter: str,
    ylim: tuple[float, float] | None = None,
    log_y: bool = False,
) -> None:
    x = np.array([0.0, 1.0])
    for left_value, right_value in zip(left, right, strict=True):
        ax.plot(
            x,
            [left_value, right_value],
            color=PAIR_GRAY,
            linewidth=0.62,
            alpha=0.72,
            zorder=1,
        )
    ax.scatter(
        np.zeros_like(left), left, s=13, color=BLUE, edgecolor="white",
        linewidth=0.28, alpha=0.84, zorder=2,
    )
    ax.scatter(
        np.ones_like(right), right, s=13, color=ORANGE, edgecolor="white",
        linewidth=0.28, alpha=0.84, zorder=2,
    )
    medians = np.array([np.median(left), np.median(right)])
    ax.plot(x, medians, color="black", linewidth=1.45, zorder=3)
    ax.scatter(
        x, medians, marker="D", s=39, c=[BLUE, ORANGE], edgecolor="black",
        linewidth=0.7, zorder=4,
    )
    ax.set_xlim(-0.34, 1.34)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
        ticks = [0.005, 0.01, 0.02, 0.05, 0.1]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
        ax.grid(axis="y", which="minor", visible=False)
    elif ylim is not None:
        ax.set_ylim(*ylim)
    ax.text(
        0.5, 0.975, p_text, transform=ax.transAxes, ha="center", va="top",
        fontsize=8.2,
    )
    panel_style(ax, letter)


def plot_figure(data_dir: Path) -> plt.Figure:
    block_df = pd.read_csv(data_dir / "r1_1_per_trajectory_block_metrics.csv")
    equal_df = pd.read_csv(data_dir / "r1_1_equal_horizon_per_trajectory.csv")
    summary_df = pd.read_csv(data_dir / "r1_1_equal_horizon_summary.csv")
    duration = block_df.pivot(
        index="trajectory_no", columns="block", values="duration_s"
    )
    curvature = block_df.pivot(
        index="trajectory_no", columns="block",
        values="mean_abs_curvature_rad_per_m",
    )
    equal_cv = equal_df.pivot(
        index="trajectory_no", columns="block", values="cv_raw_rmse_m"
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.55), facecolor="white")
    paired_panel(
        axes[0, 0], duration["calibration"].to_numpy(),
        duration["holdout"].to_numpy(), ("Calibration", "Holdout"),
        "Block duration (s)", r"$p < 0.001$", "a", ylim=(0, 14.2),
    )
    axes[0, 0].text(
        0.035, 0.965, r"$n = 38$", transform=axes[0, 0].transAxes,
        ha="left", va="top", fontsize=8.0,
    )
    paired_panel(
        axes[0, 1], curvature["calibration"].to_numpy(),
        curvature["holdout"].to_numpy(), ("Calibration", "Holdout"),
        r"Mean absolute curvature (rad m$^{-1}$)", r"$p = 0.002$", "b",
        log_y=True,
    )
    axes[0, 1].text(
        0.035, 0.06, "log scale", transform=axes[0, 1].transAxes,
        ha="left", va="bottom", fontsize=7.2, fontstyle="italic",
        color=DARK_GRAY,
    )
    paired_panel(
        axes[1, 0], equal_cv["late_calibration_equal_horizon"].to_numpy(),
        equal_cv["holdout"].to_numpy(), ("Late calibration", "Holdout"),
        "Constant-velocity RMSE (m)", r"$p < 0.001$", "c",
        ylim=(0, 10.2),
    )

    ax = axes[1, 1]
    summary_lookup = summary_df.set_index("model_or_reference")
    series = [
        ("Constant velocity", "constant_velocity_penalized", DARK_GRAY, "s", "--"),
        ("TMSFF", "TMSFF_seed42_penalized", BLUE, "o", "-"),
        ("AV-IDM", "AV-IDM_seed42_penalized", ORANGE, "^", "-"),
    ]
    for label, key, color, marker, linestyle in series:
        values = summary_lookup.loc[
            key,
            ["late_calibration_equal_horizon_objective_m", "holdout_objective_m"],
        ].astype(float).to_numpy()
        ax.plot(
            [0, 1], values, color=color, linewidth=1.55, linestyle=linestyle,
            marker=marker, markersize=5.2, markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=0.45, label=label, zorder=3,
        )
        for x_value, y_value in zip([0, 1], values, strict=True):
            offset = 5 if label != "TMSFF" else -12
            ax.annotate(
                f"{y_value:.2f}", (x_value, y_value), xytext=(0, offset),
                textcoords="offset points", ha="center",
                va="bottom" if offset > 0 else "top", color=color, fontsize=7.6,
            )
    ax.set_xlim(-0.28, 1.28)
    ax.set_ylim(0, 3.75)
    ax.set_xticks([0, 1], ["Late calibration", "Holdout"])
    ax.set_ylabel("Aggregate objective (m)")
    ax.legend(loc="upper left", frameon=False, handlelength=2.3, borderaxespad=0.35)
    panel_style(ax, "d")
    fig.subplots_adjust(
        left=0.105, right=0.982, bottom=0.105, top=0.968,
        wspace=0.31, hspace=0.34,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "outputs" / "R1_1" / "analysis"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "reproduced_outputs"
    )
    parser.add_argument(
        "--formats", nargs="+", choices=("png", "tif", "pdf"),
        default=("png",),
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    figure = plot_figure(data_dir)
    for file_format in dict.fromkeys(args.formats):
        output = output_dir / f"Fig_E1.{file_format}"
        save_options: dict[str, object] = {
            "bbox_inches": "tight",
            "pad_inches": 0.035,
        }
        if file_format in {"png", "tif"}:
            save_options["dpi"] = 600
        if file_format == "tif":
            save_options["pil_kwargs"] = {"compression": "tiff_lzw"}
        figure.savefig(output, **save_options)
        print(output)
    plt.close(figure)


if __name__ == "__main__":
    main()
