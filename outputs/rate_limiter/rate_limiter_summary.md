# Rate-limiter audit summary

Selected diagnostic target-heading slew-rate limit: 0.5 rad/s.
Worst absolute change across the 16 dataset/model/RMSE comparisons: 0.7148%.
Median reduction in the maximum steering-command step: 76.38%.

Relative changes use unrounded values and are displayed to four decimal places. Absolute relative changes below 0.0001% are labeled negligible; this does not imply exact equality.

## Selected-setting results

- Hefei Table 2 / T-IDM: heading 0.092916 rad (+0.0263%), spacing 3.889473 m (+0.0024%), velocity 0.747319 m/s (-0.0082%), acceleration 0.722218 m/s^2 (-0.0031%), maximum steering-step reduction 58.36%, mean crossing-step reduction 56.92%.
- Hefei Table 2 / T-FVD: heading 0.155218 rad (+0.0019%), spacing 7.238309 m (-0.0039%), velocity 1.343789 m/s (negligible), acceleration 0.933822 m/s^2 (negligible), maximum steering-step reduction 63.66%, mean crossing-step reduction 57.80%.
- CitySim Table 3 / T-IDM: heading 0.119471 rad (-0.5290%), spacing 4.624131 m (-0.7148%), velocity 1.176508 m/s (+0.4366%), acceleration 1.186268 m/s^2 (-0.0082%), maximum steering-step reduction 95.85%, mean crossing-step reduction 93.80%.
- CitySim Table 3 / T-FVD: heading 0.077036 rad (+0.0836%), spacing 2.349752 m (-0.1827%), velocity 0.735977 m/s (+0.0198%), acceleration 0.954819 m/s^2 (-0.0003%), maximum steering-step reduction 89.09%, mean crossing-step reduction 85.58%.
