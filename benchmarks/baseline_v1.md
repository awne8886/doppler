# Doppler geolocation benchmark

200 seeded Monte Carlo runs (seed 1), grid 15 over 8.0 km, precision target 1000 m, grid strategy `full`. Truth from the independent forward model; frequency noise 0.2–2.0 Hz + 3% outliers; unknown wind 0–10 m/s; ADS-B altitude bias; trusted clock in 50% of runs (error σ 0.3 s).

| subset | runs | supported | false supports | false-support rate | median err (m) | p95 err (m) | supported median / p95 / max err (m) | radius coverage |
|---|---|---|---|---|---|---|---|---|
| all | 200 | 22 (11 %) | 0 | 0.0 % | 193 | 5905 | 74 / 368 / 446 | 82 % |
| 1 flyover(s) | 65 | 0 (0 %) | 0 | 0.0 % | 2907 | 8402 | – / – / – | – |
| 2 flyover(s) | 72 | 18 (25 %) | 0 | 0.0 % | 160 | 560 | 97 / 382 / 446 | 78 % |
| 3 flyover(s) | 63 | 4 (6 %) | 0 | 0.0 % | 72 | 426 | 27 / 91 / 101 | 100 % |
| clock free | 95 | 8 (8 %) | 0 | 0.0 % | 306 | 7156 | 111 / 352 / 371 | 75 % |
| clock trusted | 105 | 14 (13 %) | 0 | 0.0 % | 162 | 3197 | 50 / 274 / 446 | 86 % |
| geometric | 122 | 17 (14 %) | 0 | 0.0 % | 189 | 6052 | 62 / 209 / 318 | 82 % |
| barometric | 78 | 5 (6 %) | 0 | 0.0 % | 205 | 4859 | 239 / 431 / 446 | 80 % |
| no outliers | 93 | 21 (23 %) | 0 | 0.0 % | 189 | 5160 | 86 / 371 / 446 | 81 % |
| with outliers | 107 | 1 (1 %) | 0 | 0.0 % | 208 | 6475 | 62 / 62 / 62 | 100 % |

Outcome reasons (all runs): `inconsistent_event_match` 151, `supported_by_evidence_checks` 22, `ambiguous_locations` 14, `insufficient_independent_events` 7, `source_drift_unresolved` 5, `refinement_boundary_limited` 1

## Decision-policy sweep (same runs, re-scored)

| variant | supported | false supports | false-support rate |
|---|---|---|---|
| baseline (engineering-v1) | 22 (11 %) | 0 | 0.0 % |
| max_rmse_hz=2.0 | 15 (8 %) | 0 | 0.0 % |
| max_rmse_hz=4.0 | 30 (15 %) | 0 | 0.0 % |
| max_rmse_hz=6.0 | 55 (28 %) | 0 | 0.0 % |
| near_tie_hz=0.25 | 22 (11 %) | 0 | 0.0 % |
| near_tie_hz=1.0 | 22 (11 %) | 0 | 0.0 % |
| near_tie_hz=2.0 | 22 (11 %) | 0 | 0.0 % |
| max_outlier_fraction=0.15 | 22 (11 %) | 0 | 0.0 % |
| max_outlier_fraction=0.4 | 22 (11 %) | 0 | 0.0 % |
| residual_metric=robust, max_rmse_hz=2.0 | 22 (11 %) | 0 | 0.0 % |
| residual_metric=robust, max_rmse_hz=3.0 | 52 (26 %) | 0 | 0.0 % |
| min_independent_flyovers=1 | 29 (14 %) | 4 | 13.8 % |
| require_sensitivity=True | 22 (11 %) | 0 | 0.0 % |

*Radius coverage*: share of supported runs whose true position lies within the reported sensitivity radius (an engineering envelope, not a calibrated confidence interval).
