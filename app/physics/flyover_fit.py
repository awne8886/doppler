"""
Constant-velocity flyover model and fitter.

A straight, constant-speed pass of an aircraft past a stationary microphone
produces a characteristic Doppler "S-curve" whose shape is fully described by
four parameters:

    f0     rest (source) frequency of the tracked tone            [Hz]
    v      aircraft speed                                         [m/s]
    d_cpa  slant distance at the closest point of approach (CPA)  [m]
    tau0   emission time of the CPA (recording clock)             [s]

The model is solved exactly in the *emission* domain.  For a receive time t the
emission time tau satisfies the retarded-time equation

    c (t - tau) = sqrt(d_cpa^2 + v^2 (tau - tau0)^2)

which is a quadratic in s = tau - tau0.  The receding radial velocity at
emission is u_r = v^2 s / r and the received frequency is

    f(t) = f0 * c / (c + u_r)

Fitting this model to a traced or automatically detected ridge gives a
flight-data-independent estimate of the aircraft's speed and closest slant
range, plus a principled estimate of f0 (instead of the heuristic
"steepest-slope" inflection estimate).  These estimates are generous screening
bounds, not exact values: real aircraft turn, accelerate and change engine RPM.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

DEFAULT_SPEED_OF_SOUND = 343.2


def emission_offset(t: np.ndarray, v: float, d_cpa: float, tau0: float, c: float) -> np.ndarray:
    """Solve the retarded-time equation; return s = tau - tau0 for each receive time t."""
    T = np.asarray(t, dtype=np.float64) - tau0
    a = c * c - v * v
    disc = (c * c) * (v * v) * T * T + a * d_cpa * d_cpa
    disc = np.maximum(disc, 0.0)
    # Smaller root keeps the emission strictly before reception (t - tau > 0).
    return (c * c * T - np.sqrt(disc)) / a


def flyover_frequency(
    t: np.ndarray,
    f0: float,
    v: float,
    d_cpa: float,
    tau0: float,
    c: float = DEFAULT_SPEED_OF_SOUND,
) -> np.ndarray:
    """Exact received frequency for a straight constant-velocity pass."""
    s = emission_offset(t, v, d_cpa, tau0, c)
    r = np.sqrt(d_cpa * d_cpa + (v * s) ** 2)
    u_r = (v * v) * s / np.maximum(r, 1e-9)  # positive when receding
    return f0 * c / (c + u_r)


def cpa_arrival_time(d_cpa: float, tau0: float, c: float = DEFAULT_SPEED_OF_SOUND) -> float:
    """Receive time at which the CPA emission is heard."""
    return tau0 + d_cpa / c


def fit_flyover(
    times: Sequence[float],
    freqs: Sequence[float],
    speed_of_sound: float = DEFAULT_SPEED_OF_SOUND,
    loss_scale_hz: float = 2.0,
    speed_bounds: tuple = (15.0, 330.0),
    distance_bounds: tuple = (30.0, 40000.0),
) -> Dict[str, Any]:
    """
    Robustly fit the constant-velocity flyover model to a measured ridge.

    Uses a soft-L1 (pseudo-Huber-like) loss so isolated tracing errors have
    limited influence, and several starting slant distances to avoid local
    minima.  Returns the parameters, residual statistics, approximate 1-sigma
    parameter uncertainties from the Jacobian, and a dense model curve.
    """
    t = np.asarray(times, dtype=np.float64)
    f = np.asarray(freqs, dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    if t.size < 5:
        raise ValueError("At least 5 ridge samples are required for a flyover fit.")

    order = np.argsort(t)
    t, f = t[order], f[order]
    c = float(speed_of_sound)

    f_hi = float(np.percentile(f, 95))
    f_lo = float(np.percentile(f, 5))
    f_mid = 0.5 * (f_hi + f_lo)
    span = float(t[-1] - t[0])

    # Initial speed from asymptotic ratio: f_hi/f_lo ~ (c+v)/(c-v)
    ratio = max(1.0001, f_hi / max(f_lo, 1e-6))
    v_init = float(np.clip(c * (ratio - 1.0) / (ratio + 1.0), speed_bounds[0] + 1, speed_bounds[1] - 1))

    # Initial CPA arrival: time where the trace crosses its mid frequency.
    idx_mid = int(np.argmin(np.abs(f - f_mid)))
    t_mid = float(t[idx_mid])

    lo = [f_lo * 0.5, speed_bounds[0], distance_bounds[0], t[0] - max(30.0, span)]
    hi = [f_hi * 1.5, speed_bounds[1], distance_bounds[1], t[-1] + max(30.0, span)]

    def residuals(p):
        return flyover_frequency(t, p[0], p[1], p[2], p[3], c) - f

    best = None
    for d_init in (300.0, 800.0, 1500.0, 3000.0, 6000.0, 12000.0):
        tau_init = t_mid - d_init / c
        p0 = np.array([f_mid, v_init, d_init, tau_init])
        p0 = np.clip(p0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
        try:
            sol = least_squares(
                residuals, p0, bounds=(lo, hi), loss="soft_l1", f_scale=loss_scale_hz,
                x_scale=[max(f_mid, 1.0) * 0.01, 10.0, 200.0, 1.0], max_nfev=400,
            )
        except Exception:  # pragma: no cover - defensive
            continue
        if best is None or sol.cost < best.cost:
            best = sol

    if best is None:
        raise ValueError("Flyover fit failed to converge.")

    f0, v, d_cpa, tau0 = (float(x) for x in best.x)
    res = residuals(best.x)
    rmse = float(np.sqrt(np.mean(res ** 2)))
    outlier_fraction = float(np.mean(np.abs(res) > 3.0 * loss_scale_hz))

    # Approximate parameter uncertainty from the Gauss-Newton covariance.
    sigmas = [float("nan")] * 4
    try:
        J = best.jac
        dof = max(1, t.size - 4)
        s2 = float(np.sum(res ** 2) / dof)
        cov = np.linalg.pinv(J.T @ J) * s2
        sigmas = [float(math.sqrt(max(0.0, cov[i, i]))) for i in range(4)]
    except Exception:  # pragma: no cover - defensive
        pass

    at_bound = {
        "speed": bool(abs(v - speed_bounds[0]) < 1e-3 or abs(v - speed_bounds[1]) < 1e-3),
        "distance": bool(abs(d_cpa - distance_bounds[0]) < 1e-2 or abs(d_cpa - distance_bounds[1]) < 1e-2),
    }

    t_dense = np.linspace(t[0], t[-1], 120)
    f_dense = flyover_frequency(t_dense, f0, v, d_cpa, tau0, c)

    return {
        "f0_hz": round(f0, 3),
        "speed_mps": round(v, 2),
        "cpa_slant_m": round(d_cpa, 1),
        "cpa_emission_time_s": round(tau0, 3),
        "cpa_arrival_time_s": round(cpa_arrival_time(d_cpa, tau0, c), 3),
        "uncertainty": {
            "f0_hz": round(sigmas[0], 3),
            "speed_mps": round(sigmas[1], 2),
            "cpa_slant_m": round(sigmas[2], 1),
            "cpa_emission_time_s": round(sigmas[3], 3),
        },
        "rmse_hz": round(rmse, 3),
        "outlier_fraction": round(outlier_fraction, 4),
        "n_samples": int(t.size),
        "parameter_at_bound": at_bound,
        "speed_of_sound_mps": c,
        "model_curve": [
            {"time": round(float(a), 3), "freq": round(float(b), 2)} for a, b in zip(t_dense, f_dense)
        ],
        "note": "Constant-velocity screening estimate; not exact for turning or accelerating aircraft.",
    }


def kinematic_consistency(
    fit: Dict[str, Any],
    trajectory: Dict[str, Any],
    observer_alt_m: float = 0.0,
    speed_tolerance_mps: Optional[float] = None,
    slant_tolerance_m: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compare a flyover fit with a candidate ADS-B trajectory.

    The check is deliberately generous and used for diagnostics/ranking only:
    a flight is *compatible* when its median speed is within tolerance of the
    fitted speed and its minimum height above the observer does not exceed the
    fitted CPA slant range (an aircraft cannot be closer than its height).
    """
    speeds = np.asarray(trajectory.get("speeds") or [], dtype=np.float64)
    alts = np.asarray(trajectory.get("altitudes") or [], dtype=np.float64)
    v_fit = float(fit["speed_mps"])
    d_fit = float(fit["cpa_slant_m"])
    unc = fit.get("uncertainty", {})
    v_tol = speed_tolerance_mps if speed_tolerance_mps is not None else max(25.0, 3.0 * (unc.get("speed_mps") or 0.0), 0.2 * v_fit)
    d_tol = slant_tolerance_m if slant_tolerance_m is not None else max(500.0, 3.0 * (unc.get("cpa_slant_m") or 0.0), 0.5 * d_fit)

    flight_speed = float(np.median(speeds)) if speeds.size else float("nan")
    min_height = float(np.min(alts) - observer_alt_m) if alts.size else float("nan")

    speed_ok = bool(np.isfinite(flight_speed) and abs(flight_speed - v_fit) <= v_tol)
    height_ok = bool(np.isfinite(min_height) and min_height <= d_fit + d_tol)
    return {
        "flight_speed_mps": round(flight_speed, 1) if np.isfinite(flight_speed) else None,
        "fitted_speed_mps": round(v_fit, 1),
        "speed_tolerance_mps": round(v_tol, 1),
        "min_height_above_observer_m": round(min_height, 1) if np.isfinite(min_height) else None,
        "fitted_cpa_slant_m": round(d_fit, 1),
        "slant_tolerance_m": round(d_tol, 1),
        "speed_compatible": speed_ok,
        "height_compatible": height_ok,
        "compatible": bool(speed_ok and height_ok),
    }
