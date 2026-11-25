# Doppler geolocation benchmark (re-scored: engineering-v2)

200 seeded Monte Carlo runs (seed 1), grid 15 over 8.0 km, precision target 1000 m, grid strategy `full`. Truth from the independent forward model; frequency noise 0.2–2.0 Hz + 3% outliers; unknown wind 0–10 m/s; ADS-B altitude bias; trusted clock in 50% of runs (error σ 0.3 s).

| subset | runs | supported | false supports | false-support rate | median err (m) | p95 err (m) | supported median / p95 / max err (m) | radius coverage |
|---|---|---|---|---|---|---|---|---|
| all | 200 | 52 (26 %) | 0 | 0.0 % | 193 | 5905 | 59 / 342 / 755 | 90 % |
| 1 flyover(s) | 65 | 0 (0 %) | 0 | 0.0 % | 2907 | 8402 | – / – / – | – |
| 2 flyover(s) | 72 | 31 (43 %) | 0 | 0.0 % | 160 | 560 | 100 / 344 / 446 | 87 % |
| 3 flyover(s) | 63 | 21 (33 %) | 0 | 0.0 % | 72 | 426 | 38 / 182 / 755 | 95 % |
| clock free | 95 | 18 (19 %) | 0 | 0.0 % | 306 | 7156 | 100 / 428 / 755 | 83 % |
| clock trusted | 105 | 34 (32 %) | 0 | 0.0 % | 162 | 3197 | 49 / 222 / 446 | 94 % |
| geometric | 122 | 36 (30 %) | 0 | 0.0 % | 189 | 6052 | 52 / 210 / 318 | 92 % |
| barometric | 78 | 16 (21 %) | 0 | 0.0 % | 205 | 4859 | 105 / 523 / 755 | 88 % |
| no outliers | 93 | 26 (28 %) | 0 | 0.0 % | 189 | 5160 | 50 / 358 / 446 | 85 % |
| with outliers | 107 | 26 (24 %) | 0 | 0.0 % | 208 | 6475 | 60 / 267 / 755 | 96 % |

Outcome reasons (all runs): `inconsistent_event_match` 82, `supported_by_evidence_checks` 52, `ambiguous_locations` 38, `source_drift_unresolved` 15, `insufficient_independent_events` 11, `refinement_boundary_limited` 2

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
