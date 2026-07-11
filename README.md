# TMSFF Reproducibility Materials

This repository provides the implementation materials for the revised manuscript:

**TMSFF: A Framework for Multi-State Car-Following Behavior Modelling in Planar Turning**

## Contents

- `scripts/baseline_comparison_simulation.py`  
  Controlled curved-road baseline comparison among TMSFF, C-FVD, and GACF.

- `scripts/figure_dynamic_response.py`  
  Plotting script for the dynamic response comparison.

- `scripts/figure_metric_barline.py`  
  Plotting script for the normalized performance comparison.

- `scripts/stability_analysis.py`  
  Simulation-based platoon perturbation analysis for stability-related performance.

- `outputs/`  
  Processed simulation outputs, performance metrics, and generated figures.

- `data_sample/README_data.md`  
  Description of the processed data and raw-data restrictions.

## Data Availability

The CitySim dataset used for external validation is publicly available from its official repository.

The Hefei UAV trajectory data collected in this study are not publicly available due to institutional data management restrictions and privacy considerations.

To support reproducibility, this repository provides scripts, parameter settings, derived simulation outputs, and plotting files used for the additional curved-road baseline comparison and simulation-based stability-related analysis. Raw UAV trajectory data are not included in this repository.

## Requirements

The scripts were tested with Python 3.10.

Required Python packages are listed in `requirements.txt`:

```bash
numpy
pandas
matplotlib
scipy
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

## How to Run

Run the scripts from the repository root directory.

```bash
python scripts/baseline_comparison_simulation.py
python scripts/figure_dynamic_response.py
python scripts/figure_metric_barline.py
python scripts/stability_analysis.py
```

The generated figures and CSV files will be saved in the `outputs/` directory.

## Notes

The released scripts are intended to reproduce the additional analyses added during manuscript revision, including the curved-road baseline comparison and the simulation-based stability-related analysis.

The stability-related analysis includes local linear stability checking around the equilibrium car-following state and controlled platoon perturbation simulations under prescribed leader-speed disturbances. The platoon perturbation results should be interpreted as simulation-based disturbance-damping evidence rather than direct measured traffic-flow observations.

The raw trajectory data are not redistributed because of data-use restrictions. Additional processed data or reproduction materials can be made available from the corresponding author upon reasonable request.

## Code Availability Statement for Manuscript

The source code, parameter settings, processed simulation outputs, and plotting scripts used for the additional baseline comparison and simulation-based stability-related analysis are available at:

```text
https://github.com/hyhfut-code/TMSFF-reproducibility
```

The raw Hefei UAV trajectory data are not publicly available due to institutional data management restrictions and privacy considerations.
