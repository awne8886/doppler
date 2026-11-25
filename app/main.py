"""
Doppler Acoustic Geolocator - FastAPI Application.
Integrates automatic Doppler ridge discovery, ADS-B trajectory matching with a
shared recording clock, and evidence checks that distinguish a supported
location from an inconclusive ranked hypothesis.
"""

import math
import os
import shutil
import tempfile
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.adsb.flight_manager import FlightManager
from app.adsb.opensky_client import OpenSkyClient
from app.audio.discovery import DiscoverySettings, discover_file
from app.physics.confidence import DecisionPolicy, decide_location
from app.physics.doppler_engine import DopplerEngine
from app.physics.flyover_fit import fit_flyover, kinematic_consistency
from app.physics.grid_scorer import GridScorer
from app.physics.matching import JointMatcher, MatchOptions, robust_rmse

APP_VERSION = "1.1.0"

app = FastAPI(
    title="Doppler - Acoustic ADS-B Geolocator",
    description="Geolocates audio recordings by matching acoustic Doppler curves of passing airplanes with ADS-B flight tracking data.",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

doppler_engine = DopplerEngine()
grid_scorer = GridScorer(doppler_engine)
opensky_client = OpenSkyClient()
flight_manager = FlightManager(opensky_client)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
SAMPLES_DIR = os.path.join(STATIC_DIR, "samples")
os.makedirs(UPLOAD_DIR, exist_ok=True)

AUDIO_EXTS = [".wav", ".mp3", ".ogg", ".flac", ".m4a"]


class CurvePoint(BaseModel):
    time: float
    freq: float


class EventIn(BaseModel):
    id: Optional[str] = None
    group_id: Optional[str] = Field(None, description="Ridges sharing a group (harmonics) must map to one flight")
    traced_curve: List[CurvePoint]
    f0: Optional[float] = Field(None, description="Fixed rest frequency (used only when fit_f0 is false)")
    flight_id: Optional[str] = Field(None, description="Restrict this event to one flight")


class GeolocateRequest(BaseModel):
    center_lat: float = Field(..., description="Search area center latitude")
    center_lon: float = Field(..., description="Search area center longitude")
    radius_km: float = Field(10.0, description="Search radius in kilometers", ge=1.0, le=100.0)
    f0: Optional[float] = Field(None, description="Estimated rest frequency in Hz (legacy single-event mode)", ge=20.0, le=10000.0)
    traced_curve: Optional[List[CurvePoint]] = Field(None, description="Legacy single traced Doppler curve (audio time, s)")
    events: Optional[List[EventIn]] = Field(None, description="One or more Doppler events on the same recording clock")
    scenario_id: Optional[str] = None
    flight_id: Optional[str] = None
    time_window_start: Optional[float] = None
    time_window_end: Optional[float] = None
    grid_resolution: Optional[int] = Field(25, ge=11, le=41)
    fit_f0: bool = Field(True, description="Fit each event's rest frequency instead of trusting f0")
    drift_fraction: float = Field(0.02, ge=0.0, le=0.1, description="Max linear source-frequency drift over an event")
    loss_scale_hz: float = Field(2.0, ge=0.25, le=50.0, description="Pseudo-Huber loss scale")
    recording_start_epoch: Optional[float] = Field(None, description="Trusted UTC epoch of the first audio sample")
    clock_tolerance_s: float = Field(2.0, ge=0.0, le=3600.0, description="Half-width of the trusted clock window")
    temperature_c: Optional[float] = Field(None, ge=-60.0, le=60.0, description="Air temperature for the speed of sound")
    observer_alt_m: float = Field(0.0, description="Microphone height in the trajectory altitude datum")
    sensitivity: bool = Field(True, description="Run perturbation sensitivity analysis")
    max_rmse_hz: float = Field(3.0, gt=0.0, description="Decision threshold for per-event RMSE")
    min_independent_flyovers: int = Field(2, ge=1, le=10)
    precision_target_m: float = Field(1000.0, gt=0.0)


class FlightSearchRequest(BaseModel):
    center_lat: float
    center_lon: float
    radius_km: float = 10.0
    scenario_id: Optional[str] = None
    time_window_start: Optional[float] = None
    time_window_end: Optional[float] = None


class FlyoverFitRequest(BaseModel):
    traced_curve: List[CurvePoint]
    temperature_c: Optional[float] = None
    loss_scale_hz: float = 2.0


@app.get("/api/health")
def health_check():
    return {"status": "ok", "app": "doppler-geolocator", "version": APP_VERSION}


@app.get("/api/scenarios")
def list_scenarios():
    """Return available curated flight scenarios for quick loading."""
    scenarios = flight_manager.get_curated_scenario_list()
    for sc in scenarios:
        sc["sample_audio_url"] = f"/static/samples/{sc['id']}.wav"
    return scenarios


@app.post("/api/flights/search")
def search_flights(req: FlightSearchRequest):
    """Search or generate ADS-B flight trajectories in the requested spatial and temporal window."""
    try:
        flights = flight_manager.fetch_or_generate_flights(
            center_lat=req.center_lat,
            center_lon=req.center_lon,
            radius_km=req.radius_km,
            time_window_start=req.time_window_start,
            time_window_end=req.time_window_end,
            scenario_id=req.scenario_id,
        )
        return {"count": len(flights), "flights": flights}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/flyover/fit")
def flyover_fit(req: FlyoverFitRequest):
    """Fit a constant-velocity flyover model: rest frequency, speed, CPA slant range and CPA time."""
    c = DopplerEngine.calculate_speed_of_sound(req.temperature_c) if req.temperature_c is not None else 343.2
    try:
        return fit_flyover([p.time for p in req.traced_curve], [p.freq for p in req.traced_curve],
                           speed_of_sound=c, loss_scale_hz=req.loss_scale_hz)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/audio/discover")
async def discover_audio(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    saved_as: Optional[str] = Form(None),
    fmin_hz: Optional[float] = Form(None),
    fmax_hz: Optional[float] = Form(None),
):
    """
    Automatically detect moving Doppler ridges in a recording and group harmonics.
    Provide an uploaded file, a curated ``sample_id``, or a previously uploaded ``saved_as`` name.
    """
    settings = DiscoverySettings()
    if fmin_hz:
        settings.fmin_hz = float(fmin_hz)
    if fmax_hz:
        settings.fmax_hz = float(fmax_hz)

    tmp_path = None
    try:
        if file is not None:
            ext = os.path.splitext(file.filename or "")[1].lower()
            if ext not in AUDIO_EXTS:
                raise HTTPException(status_code=400, detail="Unsupported audio format.")
            fd, tmp_path = tempfile.mkstemp(suffix=ext)
            with os.fdopen(fd, "wb") as out:
                shutil.copyfileobj(file.file, out)
            path = tmp_path
        elif sample_id:
            path = _safe_child(SAMPLES_DIR, f"{sample_id}.wav")
        elif saved_as:
            path = _safe_child(UPLOAD_DIR, saved_as)
        else:
            raise HTTPException(status_code=400, detail="Provide file, sample_id or saved_as.")
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="Audio not found.")
        try:
            return discover_file(path, settings)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _safe_child(base: str, name: str) -> str:
    path = os.path.realpath(os.path.join(base, os.path.basename(name)))
    if not path.startswith(os.path.realpath(base) + os.sep):
        raise HTTPException(status_code=400, detail="Invalid file name.")
    return path


@app.post("/api/geolocate")
def geolocate(req: GeolocateRequest):
    """
    Jointly match one or more Doppler events against candidate ADS-B trajectories
    over a spatial grid with a shared recording clock, refine the best optima,
    run sensitivity checks and return a supported / inconclusive decision.
    """
    raw_events = _collect_events(req)
    flights = flight_manager.fetch_or_generate_flights(
        center_lat=req.center_lat,
        center_lon=req.center_lon,
        radius_km=req.radius_km,
        time_window_start=req.time_window_start,
        time_window_end=req.time_window_end,
        scenario_id=req.scenario_id,
    )
    if not flights:
        raise HTTPException(status_code=404, detail="No flight trajectories found in the specified spatial/temporal window.")
    if req.flight_id:
        target = [f for f in flights if f.get("flight_id") == req.flight_id]
        flights = target or flights

    c = DopplerEngine.calculate_speed_of_sound(req.temperature_c) if req.temperature_c is not None else 343.2
    opts = MatchOptions(
        speed_of_sound=c, fit_f0=req.fit_f0, drift_fraction=req.drift_fraction,
        loss_scale_hz=req.loss_scale_hz, observer_alt_m=req.observer_alt_m,
    )
    clock_window = None
    if req.recording_start_epoch is not None:
        clock_window = (req.recording_start_epoch - req.clock_tolerance_s, req.recording_start_epoch + req.clock_tolerance_s)

    try:
        matcher = JointMatcher(flights, raw_events, req.center_lat, req.center_lon, opts, clock_window)
        res = matcher.run(req.radius_km, req.grid_resolution or 25, sensitivity=req.sensitivity,
                          precision_target_m=req.precision_target_m)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    policy = DecisionPolicy(max_rmse_hz=req.max_rmse_hz, min_independent_flyovers=req.min_independent_flyovers,
                            precision_target_m=req.precision_target_m)
    decision = decide_location(res, policy, clock_known=clock_window is not None, n_events=len(raw_events))

    flyover_fits = _flyover_fits(raw_events, c)
    if res.get("status") != "ok":
        return {"status": "no_match", "message": res.get("message"), "decision": decision,
                "flyover_fits": flyover_fits, "evaluated_flights": _flight_summaries(flights, [], flyover_fits)}

    return _format_result(matcher, res, decision, flights, flyover_fits, opts)


def _collect_events(req: GeolocateRequest) -> List[Dict[str, Any]]:
    events = []
    if req.events:
        for i, ev in enumerate(req.events):
            events.append({
                "id": ev.id or f"E{i + 1}", "group_id": ev.group_id,
                "traced_curve": [{"time": p.time, "freq": p.freq} for p in ev.traced_curve],
                "f0": ev.f0 if ev.f0 is not None else req.f0, "flight_id": ev.flight_id,
            })
    elif req.traced_curve:
        events.append({"id": "E1", "traced_curve": [{"time": p.time, "freq": p.freq} for p in req.traced_curve], "f0": req.f0})
    if not events:
        raise HTTPException(status_code=400, detail="Provide events or traced_curve.")
    for ev in events:
        if len(ev["traced_curve"]) < 3:
            raise HTTPException(status_code=400, detail="Traced curve must contain at least 3 points.")
        if not req.fit_f0 and not ev.get("f0"):
            raise HTTPException(status_code=400, detail="f0 is required when fit_f0 is false.")
    return events


def _flyover_fits(raw_events: List[Dict[str, Any]], c: float) -> Dict[str, Any]:
    fits = {}
    for ev in raw_events:
        pts = ev["traced_curve"]
        if len(pts) < 5:
            continue
        try:
            fit = fit_flyover([p["time"] for p in pts], [p["freq"] for p in pts], speed_of_sound=c)
            fit.pop("model_curve", None)
            fits[ev["id"]] = fit
        except ValueError:
            pass
    return fits


def _flight_summaries(flights, diagnostics, flyover_fits) -> List[Dict[str, Any]]:
    out = []
    for f in flights:
        fid = f.get("flight_id")
        matched = [d["event_id"] for d in diagnostics if d.get("flight_id") == fid and d.get("matched")]
        kin = {eid: kinematic_consistency(fit, f) for eid, fit in flyover_fits.items()}
        out.append({
            "flight_id": fid, "callsign": f.get("callsign"), "aircraft_type": f.get("aircraft_type"),
            "matched_events": matched, "kinematic_consistency": kin,
        })
    return out


def _format_result(matcher: JointMatcher, res, decision, flights, flyover_fits, opts: MatchOptions) -> Dict[str, Any]:
    G = matcher.grid
    n_groups = res["n_groups"]
    sigma = 3.0 * opts.loss_scale_hz

    def score_of(rr: float) -> float:
        return 100.0 * math.exp(-0.5 * (rr / sigma) ** 2) if math.isfinite(rr) else 0.0

    heatmap = []
    for iy in range(G["X"].shape[0]):
        for ix in range(G["X"].shape[1]):
            if not G["inside"][iy, ix]:
                continue
            lat, lon = matcher.to_geo(float(G["X"][iy, ix]), float(G["Y"][iy, ix]))
            rr = robust_rmse(G["loss"][iy, ix] / n_groups)
            heatmap.append({
                "lat": round(lat, 6), "lon": round(lon, 6),
                "score": round(score_of(rr), 2),
                "rmse": round(rr, 2) if math.isfinite(rr) else None,
            })

    best = res["best"]
    lat, lon = matcher.to_geo(best["x"], best["y"])
    diags = res["diagnostics"]
    primary = next((d for d in diags if d.get("matched")), diags[0])
    pk = next((m for m in matcher.models if m.flight_id == primary.get("flight_id")), matcher.models[0])
    traj0 = float(pk.t_emit[0])

    comparison = []
    events_out = []
    for d in diags:
        times, traced, modeled = d.pop("times", []), d.pop("traced", []), d.pop("modeled", [])
        events_out.append(d)
        for t, fr, md in zip(times, traced, modeled):
            comparison.append({
                "event_id": d["event_id"], "time": round(float(t), 3), "traced_freq": round(float(fr), 2),
                "modeled_freq": round(float(md), 2) if np.isfinite(md) else None,
            })
    comparison.sort(key=lambda p: (p["event_id"], p["time"]))

    alternatives = [{
        "lat": round(a["lat"], 6), "lon": round(a["lon"], 6),
        "robust_rmse_hz": round(a["robust_rmse_hz"], 3),
        "distance_from_best_m": round(a["distance_from_best_m"], 1),
        "recording_start_epoch": round(a["start"], 3),
    } for a in res["alternatives"]]

    matched_flight_ids = {d["flight_id"] for d in diags if d.get("matched")}
    flight_traj = {
        "flight_id": pk.flight_id, "callsign": pk.raw.get("callsign"), "aircraft_type": pk.raw.get("aircraft_type"),
        "latitudes": pk.raw.get("latitudes"), "longitudes": pk.raw.get("longitudes"),
        "altitudes": pk.raw.get("altitudes"), "speeds": pk.raw.get("speeds"),
    }
    return {
        "status": "success",
        "best_flight": {"flight_id": pk.flight_id, "callsign": pk.raw.get("callsign"), "aircraft_type": pk.raw.get("aircraft_type")},
        "best_location": {
            "lat": round(lat, 6), "lon": round(lon, 6),
            "score": round(score_of(res["robust_rmse_hz"]), 2),
            "rmse_hz": round(max(d["rmse_hz"] for d in diags if d.get("matched")), 3),
            "robust_rmse_hz": round(res["robust_rmse_hz"], 3),
            "time_offset_sec": round(best["start"] - traj0, 3),
            "recording_start_epoch": round(best["start"], 3),
            "uncertainty_radius_m": (res["sensitivity"] or {}).get("radius_m"),
        },
        "decision": decision,
        "alternatives": alternatives,
        "sensitivity": res["sensitivity"],
        "events": events_out,
        "flyover_fits": flyover_fits,
        "heatmap": heatmap,
        "curve_comparison": comparison,
        "flight_trajectory": flight_traj,
        "matched_flights": [
            {"flight_id": m.flight_id, "callsign": m.raw.get("callsign"), "aircraft_type": m.raw.get("aircraft_type"),
             "latitudes": m.raw.get("latitudes"), "longitudes": m.raw.get("longitudes")}
            for m in matcher.models if m.flight_id in matched_flight_ids
        ],
        "f0_hz": primary.get("f0_hz"),
        "evaluated_flights": _flight_summaries(flights, diags, flyover_fits),
        "grid_resolution": int(G["X"].shape[0]),
        "speed_of_sound_mps": opts.speed_of_sound,
        "clock_mode": "trusted_window" if matcher.clock_window else "free",
    }


@app.post("/api/audio/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Upload custom MP3/WAV audio file for analysis."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in AUDIO_EXTS:
        raise HTTPException(status_code=400, detail="Only audio files (WAV, MP3, OGG, FLAC, M4A) are supported.")

    unique_filename = f"{uuid.uuid4().hex}{ext}"
    dest_path = os.path.join(UPLOAD_DIR, unique_filename)
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "filename": file.filename,
        "saved_as": unique_filename,
        "url": f"/static/uploads/{unique_filename}",
        "size_bytes": os.path.getsize(dest_path),
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# index.html uses relative asset paths (so the same file works on GitHub Pages).
for _sub in ("css", "js", "samples"):
    app.mount(f"/{_sub}", StaticFiles(directory=os.path.join(STATIC_DIR, _sub)), name=f"static_{_sub}")


@app.get("/")
@app.head("/")
def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
