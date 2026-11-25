"""
Evidence checks and inconclusive outcomes.

A ranked best grid cell is a *hypothesis*, not an accepted microphone
position.  ``decide_location`` applies a small set of predeclared engineering
checks to the matcher output and returns either ``supported`` or
``inconclusive`` together with a machine-readable reason code.

The thresholds are deliberately simple and conservative.  They are
regression-tested engineering choices, **not** empirically calibrated
false-localisation rates, and a ``supported`` outcome carries no probability.

Physical motivation for the main gates:

* A single straight flyover is mirror-symmetric about the vertical plane
  containing the flight track: an observer on either side hears the same
  curve.  Unless a second, differently oriented flyover (or other evidence)
  breaks the symmetry, the location is ambiguous by construction.
* Every measured sample must be explained by real trajectory data, without
  extrapolation.
* Residuals, outliers and drift must be consistent with the declared loss
  scale; drift pinned at its bound means the source tone was not stable.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class DecisionPolicy:
    name: str = "engineering-v1"
    max_rmse_hz: float = 3.0
    max_outlier_fraction: float = 0.25
    min_support_fraction: float = 0.95
    min_independent_flyovers: int = 2
    near_tie_hz: float = 0.5
    precision_target_m: float = 1000.0
    require_sensitivity: bool = False


REASONS = {
    "no_doppler_events": "No usable Doppler event was supplied.",
    "no_adsb_data": "No real ADS-B trajectories were available for the search area and time window (demo data is opt-in).",
    "no_matching_flights": "No candidate flight explained every event with full trajectory support.",
    "insufficient_track_support": "Some measured samples fall outside the matched trajectory's arrival-time support.",
    "inconsistent_event_match": "Event residuals or outlier fraction exceed the declared thresholds.",
    "source_drift_unresolved": "Fitted source-frequency drift reached its bound; the tone was not stable enough.",
    "refinement_boundary_limited": "The best position (or clock) sits on the search boundary; enlarge the search.",
    "insufficient_independent_events": "Fewer independent flyovers than required (and no trusted recording clock).",
    "ambiguous_locations": "A materially different location fits almost equally well.",
    "uncertainty_not_established": "Sensitivity analysis was not available or did not complete.",
    "location_uncertainty_too_large": "The sensitivity radius exceeds the precision target.",
    "supported_by_evidence_checks": "Passed all declared evidence checks (uncalibrated; not a probability).",
}


def decide_location(result: Dict[str, Any], policy: Optional[DecisionPolicy] = None,
                    clock_known: bool = False, n_events: Optional[int] = None) -> Dict[str, Any]:
    p = policy or DecisionPolicy()
    checks: List[Dict[str, Any]] = []

    def add(code: str, passed: bool, **detail):
        checks.append({"check": code, "passed": bool(passed), **detail})

    def outcome(code: str) -> Dict[str, Any]:
        status = "supported" if code == "supported_by_evidence_checks" else "inconclusive"
        return {
            "status": status, "reason_code": code, "message": REASONS[code],
            "policy": asdict(p), "checks": checks, "calibrated": False,
        }

    if n_events is not None and n_events == 0:
        return outcome("no_doppler_events")
    if result.get("status") == "no_adsb_data":
        return outcome("no_adsb_data")
    if result.get("status") != "ok":
        return outcome("no_matching_flights")

    diag = result.get("diagnostics", [])
    matched = [d for d in diag if d.get("matched")]
    if len(matched) < len(diag) or not matched:
        add("all_events_matched", False, unmatched=[d["event_id"] for d in diag if not d.get("matched")])
        return outcome("no_matching_flights")
    add("all_events_matched", True)

    first_failure = None

    min_support = min(d["support_fraction"] for d in matched)
    add("track_support", min_support >= p.min_support_fraction, min_support_fraction=min_support)
    if min_support < p.min_support_fraction:
        first_failure = first_failure or "insufficient_track_support"

    worst_rmse = max(d["rmse_hz"] for d in matched)
    worst_out = max(d["outlier_fraction"] for d in matched)
    resid_ok = worst_rmse <= p.max_rmse_hz and worst_out <= p.max_outlier_fraction
    add("residuals", resid_ok, worst_rmse_hz=worst_rmse, worst_outlier_fraction=worst_out)
    if not resid_ok:
        first_failure = first_failure or "inconsistent_event_match"

    drift_bound = [d["event_id"] for d in matched if d.get("drift_at_bound")]
    add("source_drift", not drift_bound, events_at_bound=drift_bound)
    if drift_bound:
        first_failure = first_failure or "source_drift_unresolved"

    boundary = bool(result.get("boundary_limited"))
    add("boundary", not boundary)
    if boundary:
        first_failure = first_failure or "refinement_boundary_limited"

    flights = {d["flight_id"] for d in matched}
    groups = {d["group_id"] for d in matched}
    independent = min(len(flights), len(groups))
    add("independent_flyovers", independent >= p.min_independent_flyovers,
        independent_flyovers=independent, required=p.min_independent_flyovers, clock_known=clock_known)

    best_rr = result.get("robust_rmse_hz", 0.0)
    ties = []
    for alt in result.get("alternatives", []):
        gap = alt["robust_rmse_hz"] - best_rr
        if gap <= p.near_tie_hz and alt["distance_from_best_m"] > p.precision_target_m:
            ties.append({"lat": round(alt["lat"], 6), "lon": round(alt["lon"], 6),
                         "distance_m": round(alt["distance_from_best_m"], 1), "rmse_gap_hz": round(gap, 3)})
    add("ambiguity", not ties, near_tied_alternatives=ties)

    sens = result.get("sensitivity")
    sens_ok = True
    if sens is None or not sens.get("complete"):
        add("sensitivity", not p.require_sensitivity, available=sens is not None,
            complete=bool(sens and sens.get("complete")))
        if p.require_sensitivity:
            sens_ok = False
            first_failure = first_failure or "uncertainty_not_established"
    else:
        rad = sens.get("radius_m") or 0.0
        sens_ok = rad <= p.precision_target_m
        add("sensitivity", sens_ok, radius_m=rad, precision_target_m=p.precision_target_m)

    if first_failure:
        return outcome(first_failure)
    if independent < p.min_independent_flyovers:
        # A single flyover is mirror-ambiguous; report that specific cause when visible.
        return outcome("ambiguous_locations" if ties else "insufficient_independent_events")
    if ties:
        return outcome("ambiguous_locations")
    if not sens_ok:
        return outcome("location_uncertainty_too_large")
    return outcome("supported_by_evidence_checks")
