# Doppler geolocation benchmark

200 seeded Monte Carlo runs (seed 2), grid 15 over 8.0 km, precision target 1000 m, grid strategy `full`. Truth from the independent forward model; frequency noise 0.2–2.0 Hz + 3% outliers; unknown wind 0–10 m/s; ADS-B altitude bias; trusted clock in 50% of runs (error σ 0.3 s).

| subset | runs | supported | false supports | false-support rate | median err (m) | p95 err (m) | supported median / p95 / max err (m) | radius coverage |
|---|---|---|---|---|---|---|---|---|
| all | 200 | 17 (8 %) | 0 | 0.0 % | 172 | 5536 | 74 / 343 / 380 | 88 % |
| 1 flyover(s) | 66 | 0 (0 %) | 0 | 0.0 % | 3331 | 7318 | – / – / – | – |
| 2 flyover(s) | 66 | 9 (14 %) | 0 | 0.0 % | 144 | 1253 | 107 / 362 / 380 | 78 % |
| 3 flyover(s) | 68 | 8 (12 %) | 0 | 0.0 % | 79 | 583 | 35 / 207 / 241 | 100 % |
| clock free | 103 | 6 (6 %) | 0 | 0.0 % | 238 | 6940 | 91 / 217 / 241 | 100 % |
| clock trusted | 97 | 11 (11 %) | 0 | 0.0 % | 139 | 4908 | 56 / 357 / 380 | 82 % |
| geometric | 124 | 11 (9 %) | 0 | 0.0 % | 178 | 5905 | 36 / 242 / 334 | 91 % |
| barometric | 76 | 6 (8 %) | 0 | 0.0 % | 169 | 4980 | 137 / 345 / 380 | 83 % |
| no outliers | 97 | 17 (18 %) | 0 | 0.0 % | 179 | 5742 | 74 / 343 / 380 | 88 % |
| with outliers | 103 | 0 (0 %) | 0 | 0.0 % | 161 | 5429 | – / – / – | – |

Outcome reasons (all runs): `inconsistent_event_match` 144, `ambiguous_locations` 23, `supported_by_evidence_checks` 17, `insufficient_independent_events` 7, `source_drift_unresolved` 6, `refinement_boundary_limited` 3

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
