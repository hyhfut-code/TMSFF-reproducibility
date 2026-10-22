# Release changes

## R4 complete model 2026-10-22

Baseline public commit: bf0ddc57c5191a76feda57a6064cf7018671355e.

- Replaced the M11 simplified follower-heading-frame geometry with the
  manuscript's complete thresholded TMSFF formulation, including the
  spacing-dependent target-heading rule and Appendix B coordinate rotation.
- Recalibrated TMSFF under the same data, split, bounds, five-seed GA protocol
  and evaluation budget. Verified existing AV-IDM estimates are retained.
  The TMSFF five-seed calibration/holdout objectives change from
  1.814076/0.872732 m to 1.907173/1.024059 m. Corresponding parameter,
  per-trajectory, paired-statistical and matched-window outputs are updated.
- Updated files supporting Table 4, Table E1, Table E4, Figs. 23–24 and E1.
  The manuscript Eq. (35a) describes the loss aggregation already used in the
  comparison implementation; this written-formula correction is distinct
  from the TMSFF simulation-kernel correction.
- Report pure-position paired tests separately from penalized trajectory
  statistics. Added forward-domain records and fixed-parameter directed-heading
  checks. No empirical validation is claimed for untriggered Case 1 states.
- Aligned the archived validation helper's Case 1 slope regularization with
  the common wrapping/clipping rule. The reported fixed-parameter validation
  updates do not activate Case 1; their numerical results are retained.
- Retained Table 1, Table C2, Table C14, Fig. C9 and other unrelated parameter
  contexts and results. Provider data and private local caches are excluded.

The dated release bundles the numerical correction and additional domain
audit performed on 20–21 October 2026. It does not represent another
parameter recalibration on 22 October.
