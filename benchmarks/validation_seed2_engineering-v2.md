# Doppler geolocation benchmark (re-scored: engineering-v2)

200 seeded Monte Carlo runs (seed 2), grid 15 over 8.0 km, precision target 1000 m, grid strategy `full`. Truth from the independent forward model; frequency noise 0.2–2.0 Hz + 3% outliers; unknown wind 0–10 m/s; ADS-B altitude bias; trusted clock in 50% of runs (error σ 0.3 s).

| subset | runs | supported | false supports | false-support rate | median err (m) | p95 err (m) | supported median / p95 / max err (m) | radius coverage |
|---|---|---|---|---|---|---|---|---|
| all | 200 | 51 (26 %) | 0 | 0.0 % | 172 | 5536 | 71 / 429 / 756 | 88 % |
| 1 flyover(s) | 66 | 0 (0 %) | 0 | 0.0 % | 3331 | 7318 | – / – / – | – |
| 2 flyover(s) | 66 | 28 (42 %) | 0 | 0.0 % | 144 | 1253 | 93 / 506 / 756 | 82 % |
| 3 flyover(s) | 68 | 23 (34 %) | 0 | 0.0 % | 79 | 583 | 40 / 146 / 241 | 96 % |
| clock free | 103 | 23 (22 %) | 0 | 0.0 % | 238 | 6940 | 73 / 454 / 522 | 91 % |
| clock trusted | 97 | 28 (29 %) | 0 | 0.0 % | 139 | 4908 | 55 / 364 / 756 | 86 % |
| geometric | 124 | 31 (25 %) | 0 | 0.0 % | 178 | 5905 | 40 / 247 / 522 | 90 % |
| barometric | 76 | 20 (26 %) | 0 | 0.0 % | 169 | 4980 | 102 / 491 / 756 | 85 % |
| no outliers | 97 | 25 (26 %) | 0 | 0.0 % | 179 | 5742 | 80 / 371 / 756 | 84 % |
| with outliers | 103 | 26 (25 %) | 0 | 0.0 % | 161 | 5429 | 47 / 400 / 522 | 92 % |

Outcome reasons (all runs): `inconsistent_event_match` 74, `supported_by_evidence_checks` 51, `ambiguous_locations` 39, `source_drift_unresolved` 18, `insufficient_independent_events` 12, `refinement_boundary_limited` 6

## Decision-policy sweep (same runs, re-scored)

| variant | supported | false supports | false-support rate |
|---|---|---|---|
| baseline (engineering-v1) | 17 (8 %) | 0 | 0.0 % |
| max_rmse_hz=2.0 | 10 (5 %) | 0 | 0.0 % |
| max_rmse_hz=4.0 | 30 (15 %) | 0 | 0.0 % |
| max_rmse_hz=6.0 | 56 (28 %) | 0 | 0.0 % |
| near_tie_hz=0.25 | 17 (8 %) | 0 | 0.0 % |
| near_tie_hz=1.0 | 17 (8 %) | 0 | 0.0 % |
| near_tie_hz=2.0 | 17 (8 %) | 0 | 0.0 % |
| max_outlier_fraction=0.15 | 17 (8 %) | 0 | 0.0 % |
| max_outlier_fraction=0.4 | 17 (8 %) | 0 | 0.0 % |
| residual_metric=robust, max_rmse_hz=2.0 | 19 (10 %) | 0 | 0.0 % |
| residual_metric=robust, max_rmse_hz=3.0 | 51 (26 %) | 0 | 0.0 % |
| min_independent_flyovers=1 | 24 (12 %) | 1 | 4.2 % |
| require_sensitivity=True | 17 (8 %) | 0 | 0.0 % |

*Radius coverage*: share of supported runs whose true position lies within the reported sensitivity radius (an engineering envelope, not a calibrated confidence interval).
