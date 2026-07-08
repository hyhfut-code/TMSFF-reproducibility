# Data Description

The original trajectory data are not included because of data-use restrictions.

This repository provides processed simulation outputs in `outputs/`, including:

- `curved_cf_timeseries.csv`
- `curved_cf_metrics.csv`
- `stability_metrics.csv`
- `linear_stability_results.csv`
- `platoon_stability_timeseries.csv`

These files are sufficient to reproduce the figures included in the repository.

If users need to rerun the baseline comparison with their own trajectory data,
they should provide a processed leader trajectory with the following fields:

```text
time_s
leader_s_m
leader_v_mps
leader_a_mps2
```

The raw dataset can be requested from the corresponding author only when the
original data-use restrictions permit redistribution.
