"""One-click reproduction of Figs. 15, 16, 21, and 22.

Run this file directly in an IDE or with ``python run_all_figures.py``.
No command-line arguments are required.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def find_fair_calibration_csv(script_dir: Path) -> Path:
    candidates = (
        script_dir / "fair_calibration_runs.csv",
        script_dir.parent / "outputs" / "fair_calibration_runs.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Cannot find fair_calibration_runs.csv. Searched:\n" + searched
    )


def default_output_dir(script_dir: Path) -> Path:
    if (script_dir.parent / "outputs").is_dir():
        return script_dir.parent / "reproduced_outputs"
    return script_dir / "reproduced_outputs"


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def run_checked(command: list[str]) -> None:
    print("\nRunning:", " ".join(f'"{item}"' for item in command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    runs_csv = find_fair_calibration_csv(script_dir)
    output_dir = default_output_dir(script_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_script = require_file(script_dir / "recalibrated_baseline_simulation.py")
    redraw_script = require_file(script_dir / "redraw_fig21_fig22_publication.py")

    print("One-click reproduction started.")
    print(f"Input calibration table: {runs_csv}")
    print(f"Output directory: {output_dir}")

    run_checked(
        [
            sys.executable,
            str(baseline_script),
            str(runs_csv),
            str(output_dir),
            "--seed",
            "42",
        ]
    )

    timeseries_csv = require_file(output_dir / "recalibrated_baseline_timeseries.csv")
    leader_csv = require_file(output_dir / "recalibrated_baseline_leader.csv")
    metrics_csv = require_file(output_dir / "recalibrated_baseline_metrics.csv")

    run_checked(
        [
            sys.executable,
            str(redraw_script),
            "--timeseries",
            str(timeseries_csv),
            "--leader",
            str(leader_csv),
            "--metrics",
            str(metrics_csv),
            "--output-dir",
            str(output_dir),
        ]
    )

    # Use the publication redraws as the canonical Fig. 21 and Fig. 22 outputs.
    publication_outputs = {
        "Fig21_dynamic_response_smoothed_display.png": "Fig_21_recalibrated.png",
        "Fig22_normalized_performance_bar_line.png": "Fig_22_recalibrated.png",
    }
    for source_name, target_name in publication_outputs.items():
        source = require_file(output_dir / source_name)
        shutil.copy2(source, output_dir / target_name)

    final_figures = (
        output_dir / "Fig_15_recalibrated.png",
        output_dir / "Fig_16_recalibrated.png",
        output_dir / "Fig_21_recalibrated.png",
        output_dir / "Fig_22_recalibrated.png",
    )
    for figure in final_figures:
        require_file(figure)

    print("\nCompleted. Final figures:")
    for figure in final_figures:
        print(f"  {figure}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print("Please keep this file, the two plotting scripts, and the CSV data together.", file=sys.stderr)
        raise
