"""
Joint multi-flyover Doppler matching.

Compared to the original single-curve least-squares grid search, this module

* fits each ridge's rest frequency (and an optional, bounded, linear source
  drift evaluated at *emission* time) in closed form instead of trusting a
  hand-estimated f0;
* scores residuals with a pseudo-Huber robust loss so isolated tracing errors
  do not dominate;
* requires every measured sample to fall inside the trajectory's arrival-time
  support (no extrapolation);
* locks one shared recording clock for all events: each event keeps its
  actual flight-derived timing instead of getting its own time shift;
* treats ridges sharing a ``group_id`` (harmonics) as one flyover that must
  map to one flight;
* refines the best grid cells continuously, keeps competing local optima as
  alternatives, and re-runs the refinement under declared perturbations
  (sound speed, aircraft altitude, frequency offset/tilt) to produce an
  engineering sensitivity radius.

Scores are ranking statistics, not calibrated probabilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from app.physics.doppler_engine import DopplerEngine
from app.physics.flyover_fit import fit_flyover
from app.physics.parallel import map_matcher


@dataclass
class MatchOptions:
    speed_of_sound: float = 343.2
    fit_f0: bool = True
    drift_fraction: float = 0.02
    loss_scale_hz: float = 2.0
    min_support: float = 0.95
    irls_iterations: int = 2
    coarse_halfwidth_s: float = 6.0
    coarse_step_s: float = 0.25
    fine_halfwidth_s: float = 0.4
    fine_step_s: float = 0.02
    refine_clock_halfwidth_s: float = 1.0
    refine_clock_step_s: float = 0.08  # coarse clock step inside refinement; the fine pass then resolves fine_step_s
    max_gap_s: float = 20.0
    observer_alt_m: float = 0.0
    altitude_bias_m: float = 0.0
    wind_e: float = 0.0  # uniform horizontal wind toward east [m/s]
    wind_n: float = 0.0  # uniform horizontal wind toward north [m/s]
    wind_uncertainty_mps: float = 5.0  # +/- perturbation per axis in the sensitivity analysis
    # Optional ground elevation (callable (x, y) -> metres in the trajectory altitude datum).
    # The observer height is observer_alt_m + elevation(x, y); None means flat ground at observer_alt_m.
    elevation: Optional[Any] = None
    workers: int = 1  # >1: grid chunks and refinements run in a process pool (identical results)
    grid_strategy: str = "full"  # "full" or "coarse_to_fine" (see JointMatcher.grid_search)
    coarse_margin_hz: float = 1.5  # coarse_to_fine: refine around coarse nodes within this robust RMSE of the best
    max_alternatives: int = 6
    grid_max_points: int = 40  # events are decimated to this many samples for the coarse grid


@dataclass
class EventData:
    id: str
    group_id: str
    t: np.ndarray  # receive times relative to the recording start [s]
    f: np.ndarray  # measured frequency [Hz]
    f0_fixed: Optional[float] = None
    flight_ids: Optional[List[str]] = None  # restrict association
    cpa_time_s: float = 0.0  # approximate CPA arrival time in recording time


def build_events(raw_events: Sequence[Dict[str, Any]], opts: MatchOptions) -> List[EventData]:
    """Normalise API event dicts into EventData, estimating each event's CPA arrival."""
    events = []
    for i, ev in enumerate(raw_events):
        pts = ev.get("traced_curve") or ev.get("points") or []
        t = np.array([p["time"] for p in pts], dtype=np.float64)
        f = np.array([p["freq"] for p in pts], dtype=np.float64)
        order = np.argsort(t)
        t, f = t[order], f[order]
        if t.size < 3:
            raise ValueError(f"Event {i + 1} needs at least 3 points.")
        eid = str(ev.get("id") or f"E{i + 1}")
        gid = str(ev.get("group_id") or eid)
        events.append(EventData(
            id=eid, group_id=gid, t=t, f=f,
            f0_fixed=None if opts.fit_f0 else ev.get("f0"),
            flight_ids=[ev["flight_id"]] if ev.get("flight_id") else None,
            cpa_time_s=_estimate_cpa_time(t, f, opts.speed_of_sound),
        ))
    return events


def _estimate_cpa_time(t: np.ndarray, f: np.ndarray, c: float) -> float:
    if t.size >= 5:
        try:
            return float(fit_flyover(t, f, speed_of_sound=c)["cpa_arrival_time_s"])
        except Exception:
            pass
    slope = np.gradient(f, t) if t.size > 2 else np.array([0.0])
    return float(t[int(np.argmin(slope))])


class TrajectoryModel:
    """ADS-B trajectory in local Cartesian metres relative to the search centre."""

    def __init__(self, traj: Dict[str, Any], center_lat: float, center_lon: float,
                 engine: DopplerEngine, altitude_bias_m: float = 0.0, max_gap_s: float = 20.0):
        self.raw = traj
        self.flight_id = str(traj.get("flight_id", "UNKNOWN"))
        self.altitude_basis = str(traj.get("altitude_basis") or "unknown")
        self.altitude_corrected = bool(traj.get("altitude_correction"))
        t = np.asarray(traj["timestamps"], dtype=np.float64)
        order = np.argsort(t)
        self.t_emit = t[order]
        lat = np.asarray(traj["latitudes"], dtype=np.float64)[order]
        lon = np.asarray(traj["longitudes"], dtype=np.float64)[order]
        alt = np.asarray(traj["altitudes"], dtype=np.float64)[order] + altitude_bias_m
        x, y, z = engine.geodetic_to_cartesian(lat, lon, alt, center_lat, center_lon, 0.0)
        self.pos = np.stack([x, y, z], axis=1)
        explicit = [traj.get(k) for k in ("velocity_east", "velocity_north", "velocity_up")]
        if all(v is not None and len(v) == t.size for v in explicit):
            # Conditioned tracks carry fused (reported + smoothed) velocities; use them as-is.
            self.vel = np.stack([np.asarray(v, dtype=np.float64)[order] for v in explicit], axis=1)
        else:
            self.vel = np.stack([np.gradient(x, self.t_emit), np.gradient(y, self.t_emit),
                                 np.gradient(z, self.t_emit)], axis=1)
        gaps = np.diff(self.t_emit)
        self.gap_after = np.where(gaps > max_gap_s)[0]

    def arrival(self, obs: np.ndarray, c: float, wind: Optional[Sequence[float]] = None) -> Dict[str, np.ndarray]:
        """
        Arrival times and Doppler factors at an observer, optionally in a
        uniform horizontal wind ``wind = (east, north)`` [m/s, direction the air
        moves toward].  See :func:`propagation` for the derivation.
        """
        r = self.pos - obs
        d = np.maximum(np.linalg.norm(r, axis=1), 1.0)
        T, g = propagation(-r, self.vel, c, wind)
        t_obs = self.t_emit + T
        order = np.argsort(t_obs)
        gaps = [(t_obs[i], t_obs[i + 1]) for i in self.gap_after]
        return {
            "t": t_obs[order], "g": g[order], "tau": self.t_emit[order], "d": d[order],
            "t_cpa": float(t_obs[int(np.argmin(d))]), "gaps": gaps,
        }


def propagation(R: np.ndarray, vel: np.ndarray, c: float, wind: Optional[Sequence[float]] = None):
    """
    Travel time and Doppler factor from aircraft to observer.

    ``R`` is the vector from the aircraft (at emission) to the observer and
    ``vel`` the aircraft ground velocity.  In a uniform wind W (ground frame,
    horizontal) sound moves at c relative to the air, so the travel time T
    solves ``c T = |R - W T|``::

        T = (-R.W + sqrt((R.W)^2 + (c^2 - |W|^2) |R|^2)) / (c^2 - |W|^2)
          ~ |R| / (c + W.n),  n = R/|R|  (the usual effective sound speed)

    Differentiating with respect to emission time gives the received
    frequency factor::

        f/f0 = (c + W.k) / (c + W.k - v.k),   k = (R - W T) / (c T)

    where k is the *air-frame* wave normal, not the ground ray n (k ~ n -
    W_perp / c).  Using n instead drops a first-order term v.W_perp / c,
    comparable to W.n for aircraft speeds, so the exact form is used.  With
    W = 0 this reduces to T = |R|/c and f/f0 = c / (c + u_r), u_r = -v.n the
    receding radial speed.  The receding speed along k is clipped to 0.95 of
    the local phase speed for numerical safety.
    """
    d = np.maximum(np.linalg.norm(R, axis=1), 1.0)
    if wind is None or (abs(float(wind[0])) < 1e-12 and abs(float(wind[1])) < 1e-12):
        u = -np.sum(R * vel, axis=1) / d  # receding speed along the ray
        u = np.clip(u, -0.95 * c, 0.95 * c)
        return d / c, c / (c + u)
    W = np.array([float(wind[0]), float(wind[1]), 0.0])
    w2 = float(W @ W)
    rw = R @ W
    a = c * c - w2
    T = (-rw + np.sqrt(rw * rw + a * d * d)) / a
    k = (R - np.outer(T, W)) / (c * T)[:, None]
    ck = c + k @ W
    u = np.clip(-np.sum(vel * k, axis=1), -0.95 * ck, 0.95 * ck)
    return T, ck / (ck + u)


def wind_vector(speed_mps: Optional[float], from_deg: Optional[float]) -> Tuple[float, float]:
    """Meteorological wind (speed, direction it blows FROM, deg clockwise from north) -> (east, north) toward."""
    if not speed_mps:
        return (0.0, 0.0)
    th = math.radians(float(from_deg or 0.0))
    return (-float(speed_mps) * math.sin(th), -float(speed_mps) * math.cos(th))


def fit_event(curve: Dict[str, Any], ev: EventData, starts: np.ndarray, opts: MatchOptions) -> Dict[str, np.ndarray]:
    """Vectorised fit of one event against one arrival curve for many recording-start hypotheses."""
    t_obs = curve["t"]
    # Cheap support bound (ignores gaps): skip hypotheses that cannot reach min_support.
    lo = np.searchsorted(ev.t, t_obs[0] - starts, side="left")
    hi = np.searchsorted(ev.t, t_obs[-1] - starts, side="right")
    feasible = (hi - lo) / ev.t.size >= opts.min_support
    K = starts.size
    if not np.any(feasible):
        inf = np.full(K, np.inf)
        return {"loss": inf, "rmse": inf.copy(), "outlier": np.ones(K), "support": (hi - lo) / ev.t.size,
                "f0": np.zeros(K), "drift": np.zeros(K), "drift_at_bound": np.zeros(K, dtype=bool),
                "model": np.full((K, ev.t.size), np.nan)}
    if not np.all(feasible):
        sub = _fit_event_rows(curve, ev, starts[feasible], opts)
        out = {}
        for key, val in sub.items():
            if val.ndim == 2:
                full = np.full((K, val.shape[1]), np.nan)
            elif val.dtype == bool:
                full = np.zeros(K, dtype=bool)
            else:
                full = np.full(K, np.inf if key in ("loss", "rmse") else 0.0)
            full[feasible] = val
            out[key] = full
        out["support"][~feasible] = ((hi - lo) / ev.t.size)[~feasible]
        return out
    return _fit_event_rows(curve, ev, starts, opts)


def _fit_event_rows(curve: Dict[str, Any], ev: EventData, starts: np.ndarray, opts: MatchOptions) -> Dict[str, np.ndarray]:
    t_obs, g, tau = curve["t"], curve["g"], curve["tau"]
    Q = ev.t[None, :] + starts[:, None]  # (K, M) absolute receive times
    inside = (Q >= t_obs[0]) & (Q <= t_obs[-1])
    for a, b in curve["gaps"]:
        inside &= ~((Q > a) & (Q < b))
    support = inside.mean(axis=1)
    G = np.interp(Q.ravel(), t_obs, g).reshape(Q.shape)
    TAU = np.interp(Q.ravel(), t_obs, tau).reshape(Q.shape)
    w_in = inside.astype(np.float64)
    n_in = np.maximum(w_in.sum(axis=1), 1.0)
    tau_c = (TAU * w_in).sum(axis=1) / n_in
    tau_span = np.maximum(np.where(inside, TAU, -np.inf).max(axis=1) - np.where(inside, TAU, np.inf).min(axis=1), 1e-3)
    tau_span = np.where(np.isfinite(tau_span), tau_span, 1.0)
    U = (TAU - tau_c[:, None]) / tau_span[:, None]  # normalised emission time, ~[-0.5, 0.5]
    F = np.broadcast_to(ev.f, Q.shape)
    delta = opts.loss_scale_hz
    df = opts.drift_fraction

    w = w_in.copy()
    b0 = np.zeros(Q.shape[0])
    b1 = np.zeros(Q.shape[0])
    at_bound = np.zeros(Q.shape[0], dtype=bool)
    iters = 1 + max(0, opts.irls_iterations)
    for _ in range(iters):
        if ev.f0_fixed:
            b0[:] = float(ev.f0_fixed)
            b1[:] = 0.0
        else:
            H = G * U
            a11 = (w * G * G).sum(1); a12 = (w * G * H).sum(1); a22 = (w * H * H).sum(1)
            y1 = (w * F * G).sum(1); y2 = (w * F * H).sum(1)
            det = a11 * a22 - a12 * a12
            safe = np.abs(det) > 1e-12 * np.maximum(a11 * a22, 1e-30)
            b0 = np.where(safe, (y1 * a22 - y2 * a12) / np.where(safe, det, 1.0), y1 / np.maximum(a11, 1e-12))
            b1 = np.where(safe, (a11 * y2 - a12 * y1) / np.where(safe, det, 1.0), 0.0)
            ratio = b1 / np.maximum(np.abs(b0), 1e-9)
            at_bound = np.abs(ratio) > df
            if np.any(at_bound):
                sgn = np.sign(ratio)
                Hc = G * (1.0 + (sgn * df)[:, None] * U)
                b0c = (w * F * Hc).sum(1) / np.maximum((w * Hc * Hc).sum(1), 1e-12)
                b0 = np.where(at_bound, b0c, b0)
                b1 = np.where(at_bound, sgn * df * b0c, b1)
        model = G * (b0[:, None] + b1[:, None] * U)
        resid = F - model
        w = w_in / np.sqrt(1.0 + (resid / delta) ** 2)

    r_in = np.where(inside, resid, 0.0)
    loss = (delta * delta * (np.sqrt(1.0 + (r_in / delta) ** 2) - 1.0) * w_in).sum(1) / n_in
    rmse = np.sqrt((r_in ** 2).sum(1) / n_in)
    outliers = ((np.abs(r_in) > 3.0 * delta) & inside).sum(1) / n_in
    bad = (support < opts.min_support) | (b0 <= 0)
    loss = np.where(bad, np.inf, loss)
    return {
        "loss": loss, "rmse": rmse, "outlier": outliers, "support": support,
        "f0": b0, "drift": b1 / np.maximum(np.abs(b0), 1e-9), "drift_at_bound": at_bound,
        "model": model,
    }


class _CurvePack:
    """
    Several arrival curves flattened for batched interpolation: every curve's
    times are offset by ``curve_id * big`` so one ``searchsorted`` serves
    queries against many curves at once (equivalent to ``np.interp`` per curve).
    """

    def __init__(self, curves: Sequence[Dict[str, Any]]):
        lens = np.array([c["t"].size for c in curves])
        self.start = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
        self.last = self.start + lens - 1
        self.ref = float(curves[0]["t"][0])
        t = np.concatenate([c["t"] for c in curves]) - self.ref
        self.big = 2.0 * (float(np.max(np.abs(t))) + 1000.0)
        cid = np.repeat(np.arange(len(curves)), lens)
        self.keys = t + cid * self.big
        self.g = np.concatenate([c["g"] for c in curves])
        self.tau = np.concatenate([c["tau"] for c in curves])
        self.t_first = np.array([c["t"][0] for c in curves])
        self.t_last = np.array([c["t"][-1] for c in curves])
        self.gaps = [(ci, a, b) for ci, c in enumerate(curves) for a, b in c["gaps"]]

    def interp(self, rc: np.ndarray, Q: np.ndarray):
        qk = (Q - self.ref) + (rc * self.big)[:, None]
        j = np.searchsorted(self.keys, qk.ravel(), side="right").reshape(Q.shape) - 1
        j = np.clip(j, self.start[rc][:, None], (self.last[rc] - 1)[:, None])
        x0, x1 = self.keys[j], self.keys[j + 1]
        frac = np.clip((qk - x0) / np.where(x1 > x0, x1 - x0, 1.0), 0.0, 1.0)
        G = self.g[j] + frac * (self.g[j + 1] - self.g[j])
        TAU = self.tau[j] + frac * (self.tau[j + 1] - self.tau[j])
        return G, TAU


class _EventPack:
    """Events padded to a common length for batched fitting."""

    def __init__(self, events: Sequence[EventData]):
        m = max(e.t.size for e in events)
        E = len(events)
        self.t = np.zeros((E, m))
        self.f = np.zeros((E, m))
        self.valid = np.zeros((E, m), dtype=bool)
        self.n = np.array([e.t.size for e in events], dtype=np.float64)
        self.f0_fixed = np.array([float(e.f0_fixed) if e.f0_fixed else np.nan for e in events])
        for i, e in enumerate(events):
            self.t[i, :e.t.size] = e.t
            self.f[i, :e.t.size] = e.f
            self.valid[i, :e.t.size] = True


def fit_rows(cp: _CurvePack, ep: _EventPack, rc: np.ndarray, re: np.ndarray, rs: np.ndarray,
             opts: MatchOptions) -> np.ndarray:
    """
    Robust loss for many (curve, event, recording-start) rows in one pass.

    Same model and arithmetic as :func:`_fit_event_rows` (closed-form f0 and
    bounded drift, pseudo-Huber IRLS, support check); rows failing the support
    check are skipped before the fit and get an infinite loss.
    """
    loss = np.full(rc.size, np.inf)
    if rc.size == 0:
        return loss
    Q = ep.t[re] + rs[:, None]
    inside = ep.valid[re] & (Q >= cp.t_first[rc][:, None]) & (Q <= cp.t_last[rc][:, None])
    for ci, a, b in cp.gaps:
        rows = np.nonzero(rc == ci)[0]
        if rows.size:
            inside[rows] &= ~((Q[rows] > a) & (Q[rows] < b))
    n_ev = ep.n[re]
    support = inside.sum(1) / n_ev
    ok = np.nonzero(support >= opts.min_support)[0]
    if ok.size == 0:
        return loss
    rc, re, Q, inside = rc[ok], re[ok], Q[ok], inside[ok]
    G, TAU = cp.interp(rc, Q)
    w_in = inside.astype(np.float64)
    n_in = np.maximum(w_in.sum(axis=1), 1.0)
    tau_c = (TAU * w_in).sum(axis=1) / n_in
    tau_span = np.maximum(np.where(inside, TAU, -np.inf).max(axis=1) - np.where(inside, TAU, np.inf).min(axis=1), 1e-3)
    tau_span = np.where(np.isfinite(tau_span), tau_span, 1.0)
    U = (TAU - tau_c[:, None]) / tau_span[:, None]
    F = ep.f[re]
    fixed = ep.f0_fixed[re]
    has_fixed = np.isfinite(fixed)
    any_fixed = bool(np.any(has_fixed))
    delta = opts.loss_scale_hz
    df = opts.drift_fraction
    H = G * U
    # Weighted normal-equation terms [G.G, G.H, H.H, F.G, F.H]; only the weights change per IRLS pass.
    P = np.stack([G * G, G * H, H * H, F * G, F * H], axis=2)
    w = w_in
    for _ in range(1 + max(0, opts.irls_iterations)):
        a11, a12, a22, y1, y2 = np.einsum("rm,rmk->kr", w, P)
        det = a11 * a22 - a12 * a12
        safe = np.abs(det) > 1e-12 * np.maximum(a11 * a22, 1e-30)
        b0 = np.where(safe, (y1 * a22 - y2 * a12) / np.where(safe, det, 1.0), y1 / np.maximum(a11, 1e-12))
        b1 = np.where(safe, (a11 * y2 - a12 * y1) / np.where(safe, det, 1.0), 0.0)
        ratio = b1 / np.maximum(np.abs(b0), 1e-9)
        at_bound = np.abs(ratio) > df
        if np.any(at_bound):
            # Drift pinned at +/-df: one-parameter fit with Hc = G (1 + s df U), from the same sums.
            sgn = np.sign(ratio)
            b0c = (y1 + sgn * df * y2) / np.maximum(a11 + 2.0 * sgn * df * a12 + df * df * a22, 1e-12)
            b0 = np.where(at_bound, b0c, b0)
            b1 = np.where(at_bound, sgn * df * b0c, b1)
        if any_fixed:
            b0 = np.where(has_fixed, fixed, b0)
            b1 = np.where(has_fixed, 0.0, b1)
        resid = F - b0[:, None] * G - b1[:, None] * H
        q = np.sqrt(1.0 + (resid / delta) ** 2)
        w = w_in / q
    out = (delta * delta * (q - 1.0) * w_in).sum(1) / n_in
    loss[ok] = np.where(b0 <= 0, np.inf, out)
    return loss


def robust_rmse(loss: float) -> float:
    """Equivalent RMSE of a pseudo-Huber loss (exact for small residuals)."""
    return math.sqrt(max(0.0, 2.0 * loss)) if np.isfinite(loss) else float("inf")


class JointMatcher:
    def __init__(self, trajectories: Sequence[Dict[str, Any]], raw_events: Sequence[Dict[str, Any]],
                 center_lat: float, center_lon: float, opts: Optional[MatchOptions] = None,
                 clock_window: Optional[Tuple[float, float]] = None):
        self.opts = opts or MatchOptions()
        self.center = (center_lat, center_lon)
        self.raw_trajectories = list(trajectories)
        self.engine = DopplerEngine(speed_of_sound=self.opts.speed_of_sound)
        self.models = [TrajectoryModel(t, center_lat, center_lon, self.engine, self.opts.altitude_bias_m, self.opts.max_gap_s)
                       for t in trajectories]
        self.events = build_events(raw_events, self.opts)
        self.events_coarse = [_decimate(e, self.opts.grid_max_points) for e in self.events]
        self.groups: Dict[str, List[int]] = {}
        for i, ev in enumerate(self.events):
            self.groups.setdefault(ev.group_id, []).append(i)
        self.group_ids = list(self.groups.keys())
        self.clock_window = clock_window
        if not self.models:
            raise ValueError("No flight trajectories supplied.")

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_pair_cache", None)
        state.pop("_ep_cache", None)
        state.pop("progress", None)
        return state

    # ------------------------------------------------------------------ core
    @staticmethod
    def observer(x: float, y: float, opts: MatchOptions) -> np.ndarray:
        z = opts.observer_alt_m + (float(opts.elevation(x, y)) if opts.elevation is not None else 0.0)
        return np.array([x, y, z])

    @staticmethod
    def wind(opts: MatchOptions) -> Tuple[float, float]:
        return (opts.wind_e, opts.wind_n)

    def _allowed(self, gid: str) -> List[int]:
        restrict = None
        for ei in self.groups[gid]:
            if self.events[ei].flight_ids:
                restrict = set(self.events[ei].flight_ids) if restrict is None else restrict & set(self.events[ei].flight_ids)
        idx = [k for k, m in enumerate(self.models) if restrict is None or m.flight_id in restrict]
        return idx or list(range(len(self.models)))

    def _candidate_starts(self, curves: List[Dict[str, Any]]) -> np.ndarray:
        o = self.opts
        if self.clock_window is not None:
            lo, hi = self.clock_window
            if hi - lo < 1e-9:
                return np.array([lo])
            n = int(min(4000, max(2, math.ceil((hi - lo) / o.coarse_step_s) + 1)))
            return np.linspace(lo, hi, n)
        offs = np.arange(-o.coarse_halfwidth_s, o.coarse_halfwidth_s + 1e-9, o.coarse_step_s)
        noms = []
        for gid in self.group_ids:
            ev = self.events[self.groups[gid][0]]
            for k in self._allowed(gid):
                noms.append(curves[k]["t_cpa"] - ev.cpa_time_s)
        noms = np.unique(np.round(np.array(noms) / o.coarse_step_s) * o.coarse_step_s)
        return np.unique((noms[:, None] + offs[None, :]).ravel())

    def _clip_clock(self, starts: np.ndarray) -> np.ndarray:
        if self.clock_window is None:
            return starts
        lo, hi = self.clock_window
        s = starts[(starts >= lo - 1e-9) & (starts <= hi + 1e-9)]
        return s if s.size else np.array([min(max(starts.mean(), lo), hi)])

    def _joint(self, curves: List[Dict[str, Any]], starts: np.ndarray, opts: Optional[MatchOptions] = None,
               events: Optional[List[EventData]] = None):
        opts = opts or self.opts
        events = events or self.events
        total = np.zeros(starts.size)
        choice = {}
        for gid in self.group_ids:
            members = self.groups[gid]
            best = np.full(starts.size, np.inf)
            arg = np.full(starts.size, -1)
            for k in self._allowed(gid):
                acc = np.zeros(starts.size)
                for ei in members:
                    acc += fit_event(curves[k], events[ei], starts, opts)["loss"]
                    if not np.any(np.isfinite(acc)):
                        break
                acc /= len(members)
                better = acc < best
                best[better] = acc[better]
                arg[better] = k
            total += best
            choice[gid] = arg
        return total, choice

    def evaluate_point(self, obs: np.ndarray, starts: Optional[np.ndarray] = None,
                       opts: Optional[MatchOptions] = None, models: Optional[List[TrajectoryModel]] = None,
                       events: Optional[List[EventData]] = None) -> Tuple[float, float, Dict[str, int]]:
        """Best joint loss at one observer position, optimising the shared clock."""
        return self.evaluate_cells([obs], starts, opts, models, events)[0]

    def _pairs(self, models: List[TrajectoryModel]):
        """Per group: (model index, event index) pairs the group can be explained by."""
        if not hasattr(self, "_pair_cache") or self._pair_cache[0] is not models:
            pairs = {}
            for gid in self.group_ids:
                ks, es = [], []
                for k in self._allowed(gid):
                    for ei in self.groups[gid]:
                        ks.append(k)
                        es.append(ei)
                pairs[gid] = (np.array(ks, dtype=np.int64), np.array(es, dtype=np.int64))
            self._pair_cache = (models, pairs)
        return self._pair_cache[1]

    def _joint_cells(self, cp: _CurvePack, ep: _EventPack, starts_list: List[np.ndarray], nm: int,
                     opts: MatchOptions, models: List[TrajectoryModel]):
        """
        Joint loss over the shared clock for many cells at once (batched :meth:`_joint`).
        Groups are evaluated one after another and only at clock starts that are
        still finite for every previous group (an infinite group loss makes the
        joint total infinite, so this pruning is exact).
        """
        pairs = self._pairs(models)
        totals = [np.zeros(st.size) for st in starts_list]
        choices: List[Dict[str, np.ndarray]] = [{} for _ in starts_list]
        alive = [np.arange(st.size) for st in starts_list]
        for gid in self.group_ids:
            pk, pe = pairs[gid]
            n_mem = len(self.groups[gid])
            rc, re, rs, layout = [], [], [], []
            n = 0
            for b, starts in enumerate(starts_list):
                choices[b][gid] = np.full(starts.size, -1, dtype=np.int64)
                idx = alive[b]
                K = idx.size
                if K == 0:
                    continue
                rc.append(np.repeat(b * nm + pk, K))
                re.append(np.repeat(pe, K))
                rs.append(np.tile(starts[idx], pk.size))
                layout.append((b, n, K))
                n += pk.size * K
            if not layout:
                break
            loss = fit_rows(cp, ep, np.concatenate(rc), np.concatenate(re), np.concatenate(rs), opts)
            for b, off, K in layout:
                L = loss[off:off + pk.size * K].reshape(pk.size // n_mem, n_mem, K)
                acc = L.sum(axis=1) / n_mem  # mean over harmonic members, per flight
                arg = np.argmin(acc, axis=0)
                best = acc[arg, np.arange(K)]
                idx = alive[b]
                totals[b][idx] += best
                choices[b][gid][idx] = np.where(np.isfinite(best), pk[::n_mem][arg], -1)
                dead = np.setdiff1d(np.arange(starts_list[b].size), idx[np.isfinite(best)], assume_unique=True)
                totals[b][dead] = np.inf
                alive[b] = idx[np.isfinite(best)]
        return totals, choices

    def evaluate_cells(self, observers: Sequence[np.ndarray], starts: Optional[np.ndarray] = None,
                       opts: Optional[MatchOptions] = None, models: Optional[List[TrajectoryModel]] = None,
                       events: Optional[List[EventData]] = None) -> List[Tuple[float, float, Dict[str, int]]]:
        """:meth:`evaluate_point` for many observers in one batched pass (coarse clock search, then fine)."""
        opts = opts or self.opts
        models = models or self.models
        events = events or self.events
        nm = len(models)
        curves_all = [[m.arrival(obs, opts.speed_of_sound, self.wind(opts)) for m in models] for obs in observers]
        cp = _CurvePack([c for cs in curves_all for c in cs])
        ep = self._event_pack(events)
        starts_list = [starts if starts is not None else self._candidate_starts(cs) for cs in curves_all]
        totals, choices = self._joint_cells(cp, ep, starts_list, nm, opts, models)
        out: List[Any] = [None] * len(observers)
        fine_idx, fine_starts, coarse_k = [], [], []
        for b, total in enumerate(totals):
            if not np.any(np.isfinite(total)):
                out[b] = (float("inf"), float(starts_list[b][0]), dict.fromkeys(self.group_ids, -1))
                continue
            k = int(np.argmin(total))
            coarse_k.append(k)
            fine_idx.append(b)
            fine_starts.append(self._clip_clock(
                starts_list[b][k] + np.arange(-opts.fine_halfwidth_s, opts.fine_halfwidth_s + 1e-9, opts.fine_step_s)))
        if fine_idx:
            sub = _CurvePack([c for b in fine_idx for c in curves_all[b]])
            totals_f, choices_f = self._joint_cells(sub, ep, fine_starts, nm, opts, models)
            for j, b in enumerate(fine_idx):
                k, tf = coarse_k[j], totals_f[j]
                if np.any(np.isfinite(tf)) and np.min(tf) <= totals[b][k]:
                    kf = int(np.argmin(tf))
                    out[b] = (float(tf[kf]), float(fine_starts[j][kf]), {g: int(choices_f[j][g][kf]) for g in self.group_ids})
                else:
                    out[b] = (float(totals[b][k]), float(starts_list[b][k]), {g: int(choices[b][g][k]) for g in self.group_ids})
        return out

    def _event_pack(self, events: List[EventData]) -> _EventPack:
        cache = self.__dict__.setdefault("_ep_cache", {})
        key = id(events)
        if key not in cache or cache[key][0] is not events:
            cache[key] = (events, _EventPack(events))
        return cache[key][1]

    # ------------------------------------------------------------------ grid
    def eval_grid_batches(self, batches: Sequence[Sequence[Tuple[float, float]]]) -> List[List[Tuple[float, float, Dict[str, int]]]]:
        """Coarse-event evaluation of grid nodes, batch by batch (picklable worker entry point)."""
        return [self.evaluate_cells([self.observer(x, y, self.opts) for x, y in xy], events=self.events_coarse)
                for xy in batches]

    def _eval_cells_into(self, cells, X, Y, loss, start, progress=None, frac0=0.0, frac1=1.0):
        # Batch composition is independent of the worker count, so serial and
        # parallel runs perform bit-identical computations.
        pairs = self._pairs(self.models)
        rows_per_cell = max(1, sum(p[0].size for p in pairs.values())) * 300
        batch = max(1, 24000 // rows_per_cell)
        batches = [cells[i:i + batch] for i in range(0, len(cells), batch)]
        workers = max(1, int(self.opts.workers))
        per_task = max(1, -(-len(batches) // (3 * workers))) if workers > 1 else 1
        tasks = [batches[i:i + per_task] for i in range(0, len(batches), per_task)]
        calls = [(([[(float(X[c]), float(Y[c])) for c in b] for b in task],), {}) for task in tasks]

        def prog(f):
            if progress:
                progress("grid", frac0 + (frac1 - frac0) * f)
        results = map_matcher(self, "eval_grid_batches", calls, workers, prog)
        for task, res in zip(tasks, results):
            for b, rb in zip(task, res):
                for c, (loss_c, s, _) in zip(b, rb):
                    loss[c] = loss_c
                    start[c] = s

    def grid_search(self, radius_km: float, grid_resolution: int = 25, progress: Optional[Any] = None) -> Dict[str, Any]:
        """
        Evaluate the joint loss on a square grid of candidate observer positions.

        ``grid_strategy="full"`` evaluates every node.  ``"coarse_to_fine"``
        (odd resolutions >= 21) first evaluates every other node, then every
        full-resolution node within two nodes of a coarse local minimum or of
        a coarse node whose robust RMSE is within ``coarse_margin_hz`` of the
        best; remaining nodes are bilinearly interpolated from the coarse grid
        (``grid["evaluated"]`` is False for them) and are never reported as
        local minima.
        """
        R = radius_km * 1000.0
        lin = np.linspace(-R, R, grid_resolution)
        step = float(lin[1] - lin[0])
        X, Y = np.meshgrid(lin, lin)
        loss = np.full(X.shape, np.inf)
        start = np.zeros(X.shape)
        inside = np.hypot(X, Y) <= R * 1.05
        evaluated = np.zeros(X.shape, dtype=bool)
        n = grid_resolution
        c2f = self.opts.grid_strategy == "coarse_to_fine" and n >= 21 and n % 2 == 1
        if not c2f:
            cells = [(iy, ix) for iy in range(n) for ix in range(n) if inside[iy, ix]]
            self._eval_cells_into(cells, X, Y, loss, start, progress)
            evaluated[inside] = True
        else:
            coarse = [(iy, ix) for iy in range(0, n, 2) for ix in range(0, n, 2) if inside[iy, ix]]
            self._eval_cells_into(coarse, X, Y, loss, start, progress, 0.0, 0.5)
            for c in coarse:
                evaluated[c] = True
            Lc = loss[::2, ::2]
            n_groups = max(1, len(self.group_ids))
            rr = np.sqrt(np.maximum(0.0, 2.0 * Lc / n_groups))
            best_rr = np.min(rr[np.isfinite(rr)]) if np.any(np.isfinite(rr)) else np.inf
            seeds = set()
            ny, nx = Lc.shape
            for jy in range(ny):
                for jx in range(nx):
                    v = Lc[jy, jx]
                    if not np.isfinite(v):
                        continue
                    nb = Lc[max(0, jy - 1):jy + 2, max(0, jx - 1):jx + 2]
                    if v <= np.min(nb[np.isfinite(nb)]) or rr[jy, jx] <= best_rr + self.opts.coarse_margin_hz:
                        seeds.add((2 * jy, 2 * jx))
            fine = sorted({(iy, ix) for sy, sx in seeds for iy in range(max(0, sy - 2), min(n, sy + 3))
                           for ix in range(max(0, sx - 2), min(n, sx + 3)) if inside[iy, ix] and not evaluated[iy, ix]})
            self._eval_cells_into(fine, X, Y, loss, start, progress, 0.5, 1.0)
            for c in fine:
                evaluated[c] = True
            # Fill the rest from the coarse grid (bilinear in loss, nearest coarse clock).
            for iy in range(n):
                for ix in range(n):
                    if evaluated[iy, ix] or not inside[iy, ix]:
                        continue
                    y0, x0 = (iy // 2) * 2, (ix // 2) * 2
                    y1, x1 = min(y0 + 2, n - 1), min(x0 + 2, n - 1)
                    ty, tx = (iy - y0) / 2.0, (ix - x0) / 2.0
                    corners = [loss[y0, x0], loss[y0, x1], loss[y1, x0], loss[y1, x1]]
                    if all(np.isfinite(corners)):
                        loss[iy, ix] = ((1 - ty) * ((1 - tx) * corners[0] + tx * corners[1])
                                        + ty * ((1 - tx) * corners[2] + tx * corners[3]))
                    start[iy, ix] = start[min(n - 1, int(round(iy / 2.0)) * 2), min(n - 1, int(round(ix / 2.0)) * 2)]
        self.grid = {"X": X, "Y": Y, "loss": loss, "start": start, "inside": inside, "step": step, "R": R,
                     "evaluated": evaluated, "strategy": "coarse_to_fine" if c2f else "full"}
        return self.grid

    def local_minima(self) -> List[Tuple[int, int]]:
        L = self.grid["loss"]
        ny, nx = L.shape
        mins = []
        evaluated = self.grid.get("evaluated")
        for iy in range(ny):
            for ix in range(nx):
                v = L[iy, ix]
                if not np.isfinite(v) or (evaluated is not None and not evaluated[iy, ix]):
                    continue
                nb = L[max(0, iy - 1):iy + 2, max(0, ix - 1):ix + 2]
                if v <= np.min(nb[np.isfinite(nb)]):
                    mins.append((iy, ix))
        mins.sort(key=lambda p: L[p])
        return mins

    # ------------------------------------------------------------ refinement
    def refine(self, x0: float, y0: float, s0: float, opts: Optional[MatchOptions] = None,
               models: Optional[List[TrajectoryModel]] = None, events: Optional[List[EventData]] = None) -> Dict[str, Any]:
        opts = opts or self.opts
        hw = opts.refine_clock_halfwidth_s
        starts = self._clip_clock(np.arange(s0 - hw, s0 + hw + 1e-9, opts.refine_clock_step_s))
        step = self.grid["step"] if hasattr(self, "grid") else 400.0
        cache = {}

        def obj(p):
            key = (round(p[0], 2), round(p[1], 2))
            if key not in cache:
                cache[key] = self.evaluate_point(self.observer(p[0], p[1], opts), starts, opts, models, events)
            val = cache[key][0]
            return val if np.isfinite(val) else 1e12

        init = np.array([[x0, y0], [x0 + 0.5 * step, y0], [x0, y0 + 0.5 * step]])
        res = minimize(obj, np.array([x0, y0]), method="Nelder-Mead",
                       options={"initial_simplex": init, "xatol": 2.0, "fatol": 1e-4, "maxiter": 160})
        x, y = float(res.x[0]), float(res.x[1])
        loss, s, ch = self.evaluate_point(self.observer(x, y, opts), starts, opts, models, events)
        clock_edge = bool(self.clock_window is None and (abs(s - starts[0]) < 1e-6 or abs(s - starts[-1]) < 1e-6))
        return {"x": x, "y": y, "loss": loss, "start": s, "choice": ch, "clock_at_refine_edge": clock_edge,
                "converged": bool(res.success)}

    def to_geo(self, x: float, y: float) -> Tuple[float, float]:
        lat, lon, _ = self.engine.cartesian_to_geodetic(x, y, 0.0, self.center[0], self.center[1], 0.0)
        return float(lat), float(lon)

    def diagnostics(self, x: float, y: float, start: float, choice: Dict[str, int]) -> List[Dict[str, Any]]:
        obs = self.observer(x, y, self.opts)
        curves = [m.arrival(obs, self.opts.speed_of_sound, self.wind(self.opts)) for m in self.models]
        out = []
        for gid in self.group_ids:
            k = choice.get(gid, -1)
            for ei in self.groups[gid]:
                ev = self.events[ei]
                if k < 0:
                    out.append({"event_id": ev.id, "group_id": gid, "flight_id": None, "matched": False})
                    continue
                r = fit_event(curves[k], ev, np.array([start]), self.opts)
                m = self.models[k]
                out.append({
                    "event_id": ev.id, "group_id": gid, "matched": bool(np.isfinite(r["loss"][0])),
                    "flight_id": m.flight_id, "callsign": m.raw.get("callsign"),
                    "aircraft_type": m.raw.get("aircraft_type"),
                    "f0_hz": round(float(r["f0"][0]), 2),
                    "drift_fraction": round(float(r["drift"][0]), 5),
                    "drift_at_bound": bool(r["drift_at_bound"][0]),
                    "rmse_hz": round(float(r["rmse"][0]), 3),
                    "robust_rmse_hz": round(robust_rmse(float(r["loss"][0])), 3),
                    "outlier_fraction": round(float(r["outlier"][0]), 4),
                    "support_fraction": round(float(r["support"][0]), 4),
                    "n_samples": int(ev.t.size),
                    "times": ev.t.tolist(), "traced": ev.f.tolist(),
                    "modeled": r["model"][0].tolist(),
                })
        return out

    # ----------------------------------------------------------- full search
    def run(self, radius_km: float, grid_resolution: int = 25, sensitivity: bool = True,
            precision_target_m: float = 1000.0, progress: Optional[Any] = None) -> Dict[str, Any]:
        """
        Grid search, refinement of the best local optima, diagnostics and
        sensitivity.  ``progress(stage, fraction)`` is called with stages
        ``grid``, ``refine`` and ``sensitivity``.
        """
        self._run_token = getattr(self, "_run_token", 0) + 1
        self.grid_search(radius_km, grid_resolution, progress=progress)
        mins = self.local_minima()
        if not mins:
            return {"status": "no_match", "message": "No candidate position explained all events with full trajectory support."}

        G = self.grid
        refined = []
        calls = [((float(G["X"][c]), float(G["Y"][c]), float(G["start"][c])), {}) for c in mins[: self.opts.max_alternatives]]
        results = map_matcher(self, "refine", calls, max(1, int(self.opts.workers)),
                              (lambda f: progress("refine", f)) if progress else None)
        for r in results:
            if not np.isfinite(r["loss"]):
                continue
            # merge duplicates converging to the same optimum
            if any(math.hypot(r["x"] - q["x"], r["y"] - q["y"]) < 0.5 * G["step"] for q in refined):
                continue
            refined.append(r)
        if not refined:
            return {"status": "no_match", "message": "Refinement failed for all candidate optima."}
        refined.sort(key=lambda r: r["loss"])
        best = refined[0]

        n_groups = max(1, len(self.group_ids))
        best_rr = robust_rmse(best["loss"] / n_groups)
        for r in refined:
            r["lat"], r["lon"] = self.to_geo(r["x"], r["y"])
            r["robust_rmse_hz"] = robust_rmse(r["loss"] / n_groups)
            r["distance_from_best_m"] = math.hypot(r["x"] - best["x"], r["y"] - best["y"])

        boundary = math.hypot(best["x"], best["y"]) > G["R"] - 0.5 * G["step"]
        diag = self.diagnostics(best["x"], best["y"], best["start"], best["choice"])
        sens = self.sensitivity(best, progress=progress) if sensitivity else None
        return {
            "status": "ok", "best": best, "alternatives": refined[1:], "diagnostics": diag,
            "boundary_limited": bool(boundary or best["clock_at_refine_edge"]),
            "sensitivity": sens, "robust_rmse_hz": best_rr, "n_groups": n_groups,
        }

    def altitude_perturbation_m(self, choice: Optional[Dict[str, int]] = None) -> float:
        """
        Altitude perturbation for the sensitivity analysis, widened when the
        matched trajectories' altitudes are barometric: geometric 150 m,
        barometric corrected with QNH 200 m, barometric uncorrected or unknown
        datum 300 m (+/-30 hPa of unknown QNH is ~250 m).
        """
        idx = [k for k in (choice or {}).values() if k is not None and k >= 0] or range(len(self.models))
        delta = 150.0
        for k in idx:
            m = self.models[k]
            if m.altitude_basis == "geometric":
                continue
            delta = max(delta, 200.0 if (m.altitude_basis == "barometric" and m.altitude_corrected) else 300.0)
        return delta

    def sensitivity(self, best: Dict[str, Any], c_delta: float = 5.0, alt_delta: Optional[float] = None,
                    progress: Optional[Any] = None) -> Dict[str, Any]:
        """Re-run refinement under declared perturbations; report the max displacement."""
        base = self.opts
        if alt_delta is None:
            alt_delta = self.altitude_perturbation_m(best.get("choice"))
        jobs = []  # (name, opts, models, events)
        for dc in (-c_delta, c_delta):
            jobs.append((f"sound_speed_{dc:+.0f}mps", replace(base, speed_of_sound=base.speed_of_sound + dc), None, None))
        for da in (-alt_delta, alt_delta):
            models = [TrajectoryModel(t, self.center[0], self.center[1], self.engine, base.altitude_bias_m + da, base.max_gap_s)
                      for t in self.raw_trajectories]
            jobs.append((f"altitude_{da:+.0f}m", None, models, None))
        dw = base.wind_uncertainty_mps
        if dw > 0:
            for name, de, dn in (("wind_east", dw, 0.0), ("wind_west", -dw, 0.0),
                                 ("wind_north", 0.0, dw), ("wind_south", 0.0, -dw)):
                jobs.append((f"{name}_{dw:.0f}mps", replace(base, wind_e=base.wind_e + de, wind_n=base.wind_n + dn), None, None))
        d = base.loss_scale_hz
        for name, fn in (("freq_offset_+", lambda e: e.f + d), ("freq_offset_-", lambda e: e.f - d),
                         ("freq_tilt_+", lambda e: e.f + d * _tilt(e.t)), ("freq_tilt_-", lambda e: e.f - d * _tilt(e.t))):
            jobs.append((name, None, None, [replace(e, f=fn(e)) for e in self.events]))

        calls = [((best["x"], best["y"], best["start"], o or base, models, events), {}) for _, o, models, events in jobs]
        try:
            results = map_matcher(self, "refine", calls, max(1, int(base.workers)),
                                  (lambda f: progress("sensitivity", f)) if progress else None)
        except Exception as e:  # pragma: no cover - defensive
            results = [e] * len(jobs)
        scenarios = []
        for (name, _, _, _), r in zip(jobs, results):
            if isinstance(r, Exception):  # pragma: no cover - defensive
                scenarios.append({"scenario": name, "ok": False, "error": str(r)})
                continue
            ok = np.isfinite(r["loss"])
            disp = math.hypot(r["x"] - best["x"], r["y"] - best["y"]) if ok else None
            scenarios.append({"scenario": name, "ok": bool(ok), "displacement_m": round(disp, 1) if ok else None,
                              "clock_shift_s": round(r["start"] - best["start"], 3) if ok else None})

        oks = [s for s in scenarios if s.get("ok")]
        radius = max([s["displacement_m"] for s in oks], default=None)
        return {
            "complete": len(oks) == len(scenarios),
            "radius_m": radius,
            "scenarios": scenarios,
            "sources": ["sound_speed", "altitude", "wind", "frequency"] if dw > 0 else ["sound_speed", "altitude", "frequency"],
            "altitude_perturbation_m": alt_delta,
            "wind_perturbation_mps": dw,
            "note": "Declared engineering perturbations; not a calibrated confidence interval.",
        }


class ElevationGrid:
    """
    Ground elevation on a regular local grid (metres, trajectory altitude datum),
    bilinearly interpolated and clamped at the edges.  ``values[iy][ix]`` is the
    elevation at ``(x0 + ix*step, y0 + iy*step)`` in the matcher's local frame.
    """

    def __init__(self, x0: float, y0: float, step: float, values: Sequence[Sequence[float]]):
        self.x0, self.y0, self.step = float(x0), float(y0), float(step)
        self.values = np.asarray(values, dtype=np.float64)
        if self.values.ndim != 2 or min(self.values.shape) < 2 or not np.all(np.isfinite(self.values)):
            raise ValueError("Elevation grid must be a finite 2-D array of at least 2x2 values.")

    def __call__(self, x: float, y: float) -> float:
        ny, nx = self.values.shape
        fx = min(max((x - self.x0) / self.step, 0.0), nx - 1.0)
        fy = min(max((y - self.y0) / self.step, 0.0), ny - 1.0)
        ix, iy = min(int(fx), nx - 2), min(int(fy), ny - 2)
        tx, ty = fx - ix, fy - iy
        v = self.values
        return float((1 - ty) * ((1 - tx) * v[iy, ix] + tx * v[iy, ix + 1]) + ty * ((1 - tx) * v[iy + 1, ix] + tx * v[iy + 1, ix + 1]))


def _decimate(ev: EventData, n: int) -> EventData:
    if ev.t.size <= n:
        return ev
    idx = np.unique(np.round(np.linspace(0, ev.t.size - 1, n)).astype(int))
    return replace(ev, t=ev.t[idx], f=ev.f[idx])


def _tilt(t: np.ndarray) -> np.ndarray:
    span = max(float(t[-1] - t[0]), 1e-6)
    return (t - 0.5 * (t[0] + t[-1])) / span * 2.0
