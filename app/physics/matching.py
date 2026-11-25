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
    max_gap_s: float = 20.0
    observer_alt_m: float = 0.0
    altitude_bias_m: float = 0.0
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

    def arrival(self, obs: np.ndarray, c: float) -> Dict[str, np.ndarray]:
        r = self.pos - obs
        d = np.maximum(np.linalg.norm(r, axis=1), 1.0)
        u_r = np.sum(r * self.vel, axis=1) / d  # receding radial speed at emission
        u_r = np.clip(u_r, -0.95 * c, 0.95 * c)
        t_obs = self.t_emit + d / c
        g = c / (c + u_r)
        order = np.argsort(t_obs)
        gaps = [(t_obs[i], t_obs[i + 1]) for i in self.gap_after]
        return {
            "t": t_obs[order], "g": g[order], "tau": self.t_emit[order], "d": d[order],
            "t_cpa": float(t_obs[int(np.argmin(d))]), "gaps": gaps,
        }


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

    # ------------------------------------------------------------------ core
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
        opts = opts or self.opts
        models = models or self.models
        curves = [m.arrival(obs, opts.speed_of_sound) for m in models]
        if starts is None:
            starts = self._candidate_starts(curves)
        total, choice = self._joint(curves, starts, opts, events)
        if not np.any(np.isfinite(total)):
            return float("inf"), float(starts[0]), {g: -1 for g in self.group_ids}
        k = int(np.argmin(total))
        fine = self._clip_clock(starts[k] + np.arange(-opts.fine_halfwidth_s, opts.fine_halfwidth_s + 1e-9, opts.fine_step_s))
        total_f, choice_f = self._joint(curves, fine, opts, events)
        if np.any(np.isfinite(total_f)) and np.min(total_f) <= total[k]:
            kf = int(np.argmin(total_f))
            return float(total_f[kf]), float(fine[kf]), {g: int(choice_f[g][kf]) for g in self.group_ids}
        return float(total[k]), float(starts[k]), {g: int(choice[g][k]) for g in self.group_ids}

    # ------------------------------------------------------------------ grid
    def grid_search(self, radius_km: float, grid_resolution: int = 25) -> Dict[str, Any]:
        R = radius_km * 1000.0
        lin = np.linspace(-R, R, grid_resolution)
        step = float(lin[1] - lin[0])
        X, Y = np.meshgrid(lin, lin)
        loss = np.full(X.shape, np.inf)
        start = np.zeros(X.shape)
        inside = np.hypot(X, Y) <= R * 1.05
        for iy in range(grid_resolution):
            for ix in range(grid_resolution):
                if not inside[iy, ix]:
                    continue
                obs = np.array([X[iy, ix], Y[iy, ix], self.opts.observer_alt_m])
                l, s, _ = self.evaluate_point(obs, events=self.events_coarse)
                loss[iy, ix] = l
                start[iy, ix] = s
        self.grid = {"X": X, "Y": Y, "loss": loss, "start": start, "inside": inside, "step": step, "R": R}
        return self.grid

    def local_minima(self) -> List[Tuple[int, int]]:
        L = self.grid["loss"]
        ny, nx = L.shape
        mins = []
        for iy in range(ny):
            for ix in range(nx):
                v = L[iy, ix]
                if not np.isfinite(v):
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
        starts = self._clip_clock(np.arange(s0 - hw, s0 + hw + 1e-9, opts.fine_step_s * 2))
        step = self.grid["step"] if hasattr(self, "grid") else 400.0
        cache = {}

        def obj(p):
            key = (round(p[0], 2), round(p[1], 2))
            if key not in cache:
                l, s, ch = self.evaluate_point(np.array([p[0], p[1], opts.observer_alt_m]), starts, opts, models, events)
                cache[key] = (l, s, ch)
            l = cache[key][0]
            return l if np.isfinite(l) else 1e12

        init = np.array([[x0, y0], [x0 + 0.5 * step, y0], [x0, y0 + 0.5 * step]])
        res = minimize(obj, np.array([x0, y0]), method="Nelder-Mead",
                       options={"initial_simplex": init, "xatol": 2.0, "fatol": 1e-4, "maxiter": 160})
        x, y = float(res.x[0]), float(res.x[1])
        l, s, ch = self.evaluate_point(np.array([x, y, opts.observer_alt_m]), starts, opts, models, events)
        clock_edge = bool(self.clock_window is None and (abs(s - starts[0]) < 1e-6 or abs(s - starts[-1]) < 1e-6))
        return {"x": x, "y": y, "loss": l, "start": s, "choice": ch, "clock_at_refine_edge": clock_edge,
                "converged": bool(res.success)}

    def to_geo(self, x: float, y: float) -> Tuple[float, float]:
        lat, lon, _ = self.engine.cartesian_to_geodetic(x, y, 0.0, self.center[0], self.center[1], 0.0)
        return float(lat), float(lon)

    def diagnostics(self, x: float, y: float, start: float, choice: Dict[str, int]) -> List[Dict[str, Any]]:
        obs = np.array([x, y, self.opts.observer_alt_m])
        curves = [m.arrival(obs, self.opts.speed_of_sound) for m in self.models]
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
            precision_target_m: float = 1000.0) -> Dict[str, Any]:
        self.grid_search(radius_km, grid_resolution)
        mins = self.local_minima()
        if not mins:
            return {"status": "no_match", "message": "No candidate position explained all events with full trajectory support."}

        G = self.grid
        refined = []
        for (iy, ix) in mins[: self.opts.max_alternatives]:
            r = self.refine(float(G["X"][iy, ix]), float(G["Y"][iy, ix]), float(G["start"][iy, ix]))
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
        sens = self.sensitivity(best) if sensitivity else None
        return {
            "status": "ok", "best": best, "alternatives": refined[1:], "diagnostics": diag,
            "boundary_limited": bool(boundary or best["clock_at_refine_edge"]),
            "sensitivity": sens, "robust_rmse_hz": best_rr, "n_groups": n_groups,
        }

    def sensitivity(self, best: Dict[str, Any], c_delta: float = 5.0, alt_delta: float = 150.0) -> Dict[str, Any]:
        """Re-run refinement under declared perturbations; report the max displacement."""
        scenarios = []
        base = self.opts

        def run(name, opts=None, models=None, events=None):
            try:
                r = self.refine(best["x"], best["y"], best["start"], opts or base, models, events)
                ok = np.isfinite(r["loss"])
                disp = math.hypot(r["x"] - best["x"], r["y"] - best["y"]) if ok else None
                scenarios.append({"scenario": name, "ok": bool(ok), "displacement_m": round(disp, 1) if ok else None,
                                  "clock_shift_s": round(r["start"] - best["start"], 3) if ok else None})
            except Exception as e:  # pragma: no cover - defensive
                scenarios.append({"scenario": name, "ok": False, "error": str(e)})

        for dc in (-c_delta, c_delta):
            o = replace(base, speed_of_sound=base.speed_of_sound + dc)
            run(f"sound_speed_{dc:+.0f}mps", opts=o)
        for da in (-alt_delta, alt_delta):
            models = [TrajectoryModel(t, self.center[0], self.center[1], self.engine, base.altitude_bias_m + da, base.max_gap_s)
                      for t in self.raw_trajectories]
            run(f"altitude_{da:+.0f}m", models=models)
        d = base.loss_scale_hz
        for name, fn in (("freq_offset_+", lambda e: e.f + d), ("freq_offset_-", lambda e: e.f - d),
                         ("freq_tilt_+", lambda e: e.f + d * _tilt(e.t)), ("freq_tilt_-", lambda e: e.f - d * _tilt(e.t))):
            evs = [replace(e, f=fn(e)) for e in self.events]
            run(name, events=evs)

        oks = [s for s in scenarios if s.get("ok")]
        radius = max([s["displacement_m"] for s in oks], default=None)
        return {
            "complete": len(oks) == len(scenarios),
            "radius_m": radius,
            "scenarios": scenarios,
            "sources": ["sound_speed", "altitude", "frequency"],
            "note": "Declared engineering perturbations; not a calibrated confidence interval.",
        }


def _decimate(ev: EventData, n: int) -> EventData:
    if ev.t.size <= n:
        return ev
    idx = np.unique(np.round(np.linspace(0, ev.t.size - 1, n)).astype(int))
    return replace(ev, t=ev.t[idx], f=ev.f[idx])


def _tilt(t: np.ndarray) -> np.ndarray:
    span = max(float(t[-1] - t[0]), 1e-6)
    return (t - 0.5 * (t[0] + t[-1])) / span * 2.0
