"""
Doppler Acoustic Geolocator - FastAPI Application.
Integrates automatic Doppler ridge discovery, ADS-B trajectory matching with a
shared recording clock, and evidence checks that distinguish a supported
location from an inconclusive ranked hypothesis.
"""

import logging
import math
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.adsb.conditioning import condition_trajectory
from app.adsb.flight_manager import FlightManager
from app.adsb.opensky_client import OpenSkyClient
from app.adsb.providers import build_providers
from app.adsb.track_import import ImportOptions, TrackImportError, parse_tracks, validate_uploaded_trajectory
from app.audio.discovery import DiscoverySettings, discover_file
from app.audio.metadata import read_recording_start
from app.config import Settings
from app.physics.confidence import DecisionPolicy, decide_location
from app.physics.doppler_engine import DopplerEngine
from app.physics.flyover_fit import fit_flyover, kinematic_consistency
from app.physics.matching import JointMatcher, MatchOptions, robust_rmse
from app.physics.prefilter import prefilter_flights

APP_VERSION = "1.2.0"
DEFAULT_SPEED_OF_SOUND = 343.2

logger = logging.getLogger("doppler.api")
settings = Settings.from_env()

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_DIR, "static")
SAMPLES_DIR = os.path.join(STATIC_DIR, "samples")
AUDIO_EXTS = [".wav", ".mp3", ".ogg", ".flac", ".m4a"]
UPLOAD_CHUNK = 1 << 20


def upload_dir() -> str:
    """Uploads live under the data dir, never under the publicly served static tree."""
    os.makedirs(settings.upload_dir, exist_ok=True)
    return settings.upload_dir


def cleanup_uploads(now: Optional[float] = None) -> int:
    """Delete uploaded files older than ``settings.upload_ttl_s``. Returns the number removed."""
    d = settings.upload_dir
    if not os.path.isdir(d):
        return 0
    now = time.time() if now is None else now
    removed = 0
    for name in os.listdir(d):
        path = os.path.join(d, name)
        try:
            if os.path.isfile(path) and now - os.path.getmtime(path) > settings.upload_ttl_s:
                os.remove(path)
                removed += 1
        except OSError:  # pragma: no cover - concurrent removal
            pass
    return removed


@asynccontextmanager
async def lifespan(_app: FastAPI):
    cleanup_uploads()
    yield


app = FastAPI(
    title="Doppler - Acoustic ADS-B Geolocator",
    description="Geolocates audio recordings by matching acoustic Doppler curves of passing airplanes with ADS-B flight tracking data.",
    version=APP_VERSION,
    lifespan=lifespan,
)

if settings.cors_origins:
    # No cookies or auth headers are used, so credentials stay off (and "*" stays valid).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "HEAD"],
        allow_headers=["*"],
    )

opensky_client = OpenSkyClient()
flight_manager = FlightManager(opensky_client, historical_providers=build_providers(settings.cache_dir),
                               max_live_aircraft=settings.max_live_aircraft)
CONDITIONED_SOURCES = ("historical", "uploaded")


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
    scenario_id: Optional[str] = Field(None, description="Explicitly use a curated demo scenario's trajectories")
    demo: bool = Field(False, description="Allow demo data (nearby curated scenario or synthetic flights) when no real ADS-B data exists")
    flight_id: Optional[str] = None
    trajectories: Optional[List[Dict[str, Any]]] = Field(
        None, description="Uploaded tracks (e.g. from /api/flights/import); used instead of any ADS-B source")
    icao24_hints: Optional[List[str]] = Field(None, description="Candidate aircraft for historical providers")
    condition_tracks: bool = Field(True, description="Resample/smooth historical and uploaded tracks before matching")
    prefilter: bool = Field(True, description="Discard kinematically / temporally impossible flights before the search")
    time_window_start: Optional[float] = None
    time_window_end: Optional[float] = None
    grid_resolution: Optional[int] = Field(25, ge=11, le=41)
    fit_f0: bool = Field(True, description="Fit each event's rest frequency instead of trusting f0")
    drift_fraction: float = Field(0.02, ge=0.0, le=0.1, description="Max linear source-frequency drift over an event")
    loss_scale_hz: float = Field(2.0, ge=0.25, le=50.0, description="Pseudo-Huber loss scale")
    recording_start_epoch: Optional[float] = Field(None, description="Trusted UTC epoch of the first audio sample")
    clock_tolerance_s: float = Field(2.0, ge=0.0, le=3600.0, description="Half-width of the trusted clock window")
    temperature_c: Optional[float] = Field(None, ge=-60.0, le=60.0, description="Air temperature for the speed of sound")
    relative_humidity_pct: Optional[float] = Field(None, ge=0.0, le=100.0, description="Relative humidity for the speed of sound (default 50 % when temperature is given)")
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
    demo: bool = False
    icao24_hints: Optional[List[str]] = None
    time_window_start: Optional[float] = None
    time_window_end: Optional[float] = None


class FlyoverFitRequest(BaseModel):
    traced_curve: List[CurvePoint]
    temperature_c: Optional[float] = None
    relative_humidity_pct: Optional[float] = Field(None, ge=0.0, le=100.0)
    loss_scale_hz: float = 2.0


def speed_of_sound(temperature_c: Optional[float], relative_humidity_pct: Optional[float]) -> float:
    """343.2 m/s (20 degC dry) unless weather is given; then Cramer (1993) with RH defaulting to 50 %."""
    if temperature_c is None and relative_humidity_pct is None:
        return DEFAULT_SPEED_OF_SOUND
    t = 20.0 if temperature_c is None else temperature_c
    rh = 50.0 if relative_humidity_pct is None else relative_humidity_pct
    return DopplerEngine.calculate_speed_of_sound(t, rh / 100.0)


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


def _find_flights(center_lat, center_lon, radius_km, t0, t1, scenario_id, demo, icao24_hints=None) -> Dict[str, Any]:
    try:
        return flight_manager.find_flights(center_lat=center_lat, center_lon=center_lon, radius_km=radius_km,
                                           time_window_start=t0, time_window_end=t1,
                                           scenario_id=scenario_id or None, demo=demo, icao24_hints=icao24_hints)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Flight search failed")
        raise HTTPException(status_code=500, detail="Internal error while retrieving flight data.")


@app.post("/api/flights/search")
def search_flights(req: FlightSearchRequest):
    """
    Retrieve candidate ADS-B trajectories.  Demo data (curated or synthetic) is
    returned only with an explicit ``scenario_id`` or ``demo=true``; otherwise
    ``status`` is ``no_adsb_data`` when no real data exists.
    """
    found = _find_flights(req.center_lat, req.center_lon, req.radius_km, req.time_window_start,
                          req.time_window_end, req.scenario_id, req.demo, req.icao24_hints)
    return {"count": len(found["flights"]), "flights": found["flights"], "status": found["status"],
            "data_source": found["data_source"], "notes": found["notes"], "limitations": found["limitations"]}


@app.post("/api/flyover/fit")
def flyover_fit(req: FlyoverFitRequest):
    """Fit a constant-velocity flyover model: rest frequency, speed, CPA slant range and CPA time."""
    c = speed_of_sound(req.temperature_c, req.relative_humidity_pct)
    try:
        return fit_flyover([p.time for p in req.traced_curve], [p.freq for p in req.traced_curve],
                           speed_of_sound=c, loss_scale_hz=req.loss_scale_hz)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/audio/discover")
def discover_audio(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    saved_as: Optional[str] = Form(None),
    fmin_hz: Optional[float] = Form(None),
    fmax_hz: Optional[float] = Form(None),
):
    """
    Automatically detect moving Doppler ridges in a recording and group harmonics.
    Provide an uploaded file, a curated ``sample_id``, or a previously uploaded ``saved_as`` name.

    Declared as a sync endpoint so FastAPI runs the CPU-heavy analysis in its
    threadpool instead of blocking the event loop.
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
            os.close(fd)
            _save_upload(file, tmp_path)
            path = tmp_path
        elif sample_id:
            path = _safe_child(SAMPLES_DIR, f"{sample_id}.wav")
        elif saved_as:
            path = _safe_child(upload_dir(), saved_as)
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


def _save_upload(file: UploadFile, dest_path: str) -> int:
    """Stream an upload to disk, enforcing ``settings.max_upload_bytes`` (HTTP 413 when exceeded)."""
    total = 0
    try:
        with open(dest_path, "wb") as out:
            while True:
                chunk = file.file.read(UPLOAD_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > settings.max_upload_bytes:
                    raise HTTPException(status_code=413, detail=(
                        f"Upload exceeds the {settings.max_upload_bytes // (1024 * 1024)} MiB limit."))
                out.write(chunk)
    except BaseException:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise
    return total


def _safe_child(base: str, name: str) -> str:
    path = os.path.realpath(os.path.join(base, os.path.basename(name)))
    if not path.startswith(os.path.realpath(base) + os.sep):
        raise HTTPException(status_code=400, detail="Invalid file name.")
    return path


@app.post("/api/flights/import")
def import_flights(
    file: UploadFile = File(...),
    format: Optional[str] = Form(None, description="csv, fr24_csv, kml, geojson or readsb (auto-detected if omitted)"),
    altitude_unit: str = Form("auto"),
    speed_unit: str = Form("auto"),
    vertical_rate_unit: str = Form("auto"),
    timezone: Optional[str] = Form(None, description="Zone for ISO times without an offset, e.g. UTC or +02:00"),
    altitude_basis: Optional[str] = Form(None),
):
    """
    Parse and validate uploaded aircraft tracks.  The returned trajectories
    (``data_source="uploaded"``) can be passed back as ``trajectories`` in
    ``/api/geolocate``; nothing is stored server-side.
    """
    fd, tmp = tempfile.mkstemp(suffix=".tracks")
    os.close(fd)
    try:
        _save_upload(file, tmp)
        with open(tmp, "rb") as fh:
            data = fh.read()
    finally:
        os.remove(tmp)
    try:
        res = parse_tracks(data, file.filename or "", format, ImportOptions(
            altitude_unit=altitude_unit, speed_unit=speed_unit, vertical_rate_unit=vertical_rate_unit,
            timezone=timezone, altitude_basis=altitude_basis or None))
    except TrackImportError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="The file could not be parsed as a track file.")
    return {"count": len(res["flights"]), "format": res["format"], "flights": res["flights"],
            "warnings": res["warnings"], "data_source": "uploaded"}


@app.post("/api/audio/metadata")
def audio_metadata(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    saved_as: Optional[str] = Form(None),
):
    """Suggest the recording start time from file metadata (BWF bext / iXML / ffprobe). Always user-confirmable."""
    tmp_path = None
    try:
        if file is not None:
            ext = os.path.splitext(file.filename or "")[1].lower()
            if ext not in AUDIO_EXTS:
                raise HTTPException(status_code=400, detail="Unsupported audio format.")
            fd, tmp_path = tempfile.mkstemp(suffix=ext)
            os.close(fd)
            _save_upload(file, tmp_path)
            path = tmp_path
        elif sample_id:
            path = _safe_child(SAMPLES_DIR, f"{sample_id}.wav")
        elif saved_as:
            path = _safe_child(upload_dir(), saved_as)
        else:
            raise HTTPException(status_code=400, detail="Provide file, sample_id or saved_as.")
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="Audio not found.")
        return read_recording_start(path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/api/geolocate")
def geolocate(req: GeolocateRequest):
    """
    Jointly match one or more Doppler events against candidate ADS-B trajectories
    over a spatial grid with a shared recording clock, refine the best optima,
    run sensitivity checks and return a supported / inconclusive decision.
    """
    return run_geolocation(req)


def run_geolocation(req: GeolocateRequest, progress=None) -> Dict[str, Any]:
    """Full geolocation pipeline shared by the synchronous endpoint and async jobs."""
    raw_events = _collect_events(req)
    c = speed_of_sound(req.temperature_c, req.relative_humidity_pct)
    clock_window = None
    if req.recording_start_epoch is not None:
        clock_window = (req.recording_start_epoch - req.clock_tolerance_s, req.recording_start_epoch + req.clock_tolerance_s)

    if req.trajectories:
        try:
            flights = [validate_uploaded_trajectory(t, i) for i, t in enumerate(req.trajectories)]
        except TrackImportError as e:
            raise HTTPException(status_code=400, detail=str(e))
        found = {"data_source": "uploaded", "notes": [f"{len(flights)} uploaded track(s) supplied with the request."],
                 "limitations": []}
    else:
        t0, t1 = req.time_window_start, req.time_window_end
        if t0 is None and t1 is None and req.recording_start_epoch is not None:
            # A trusted recording clock defines the ADS-B window to query.
            t0, t1 = _recording_window(req, raw_events)
        found = _find_flights(req.center_lat, req.center_lon, req.radius_km, t0, t1, req.scenario_id, req.demo,
                              req.icao24_hints)
        flights = found["flights"]
    provenance = {"data_source": found["data_source"], "data_notes": found["notes"],
                  "data_limitations": found["limitations"]}
    if not flights:
        decision = decide_location({"status": "no_adsb_data"}, n_events=len(raw_events))
        return {"status": "no_adsb_data", "message": found.get("message"), "decision": decision, **provenance}
    if req.flight_id:
        target = [f for f in flights if f.get("flight_id") == req.flight_id]
        flights = target or flights
    if req.condition_tracks:
        flights = [condition_trajectory(f) if f.get("data_source") in CONDITIONED_SOURCES else f for f in flights]

    flyover_fits = _flyover_fits(raw_events, c)
    discarded: List[Dict[str, Any]] = []
    candidates = flights
    if req.prefilter:
        candidates, discarded = prefilter_flights(flights, raw_events, flyover_fits, clock_window, req.observer_alt_m)
    provenance["discarded_flights"] = discarded
    if not candidates:
        decision = decide_location({"status": "no_match"}, n_events=len(raw_events))
        return {"status": "no_match", "message": "Every candidate flight was excluded by the kinematic / time pre-filter.",
                "decision": decision, "flyover_fits": flyover_fits,
                "evaluated_flights": _flight_summaries(flights, [], flyover_fits, discarded), **provenance}

    opts = MatchOptions(
        speed_of_sound=c, fit_f0=req.fit_f0, drift_fraction=req.drift_fraction,
        loss_scale_hz=req.loss_scale_hz, observer_alt_m=req.observer_alt_m,
    )
    try:
        matcher = JointMatcher(candidates, raw_events, req.center_lat, req.center_lon, opts, clock_window)
        res = matcher.run(req.radius_km, req.grid_resolution or 25, sensitivity=req.sensitivity,
                          precision_target_m=req.precision_target_m)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    policy = DecisionPolicy(max_rmse_hz=req.max_rmse_hz, min_independent_flyovers=req.min_independent_flyovers,
                            precision_target_m=req.precision_target_m)
    decision = decide_location(res, policy, clock_known=clock_window is not None, n_events=len(raw_events))

    if res.get("status") != "ok":
        return {"status": "no_match", "message": res.get("message"), "decision": decision,
                "flyover_fits": flyover_fits, "evaluated_flights": _flight_summaries(flights, [], flyover_fits, discarded),
                **provenance}

    out = _format_result(matcher, res, decision, flights, flyover_fits, opts, discarded)
    out.update(provenance)
    return out


def _recording_window(req: GeolocateRequest, raw_events: List[Dict[str, Any]]):
    t_end = max(p["time"] for ev in raw_events for p in ev["traced_curve"])
    return (req.recording_start_epoch - req.clock_tolerance_s - 60.0,
            req.recording_start_epoch + req.clock_tolerance_s + t_end + 60.0)


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


def _flight_summaries(flights, diagnostics, flyover_fits, discarded=()) -> List[Dict[str, Any]]:
    out = []
    dropped = {d["flight_id"]: d for d in discarded}
    for f in flights:
        fid = f.get("flight_id")
        matched = [d["event_id"] for d in diagnostics if d.get("flight_id") == fid and d.get("matched")]
        kin = {eid: kinematic_consistency(fit, f) for eid, fit in flyover_fits.items()}
        out.append({
            "flight_id": fid, "callsign": f.get("callsign"), "aircraft_type": f.get("aircraft_type"),
            "data_source": f.get("data_source"), "altitude_basis": f.get("altitude_basis"),
            "matched_events": matched, "kinematic_consistency": kin,
            "discarded": fid in dropped, "discard_reasons": dropped.get(fid, {}).get("groups", []),
            "conditioning": f.get("conditioning"),
        })
    return out


def _format_result(matcher: JointMatcher, res, decision, flights, flyover_fits, opts: MatchOptions,
                   discarded=()) -> Dict[str, Any]:
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
        "data_source": pk.raw.get("data_source"),
        "latitudes": pk.raw.get("latitudes"), "longitudes": pk.raw.get("longitudes"),
        "altitudes": pk.raw.get("altitudes"), "speeds": pk.raw.get("speeds"),
    }
    return {
        "status": "success",
        "best_flight": {"flight_id": pk.flight_id, "callsign": pk.raw.get("callsign"), "aircraft_type": pk.raw.get("aircraft_type"),
                        "data_source": pk.raw.get("data_source")},
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
             "data_source": m.raw.get("data_source"),
             "latitudes": m.raw.get("latitudes"), "longitudes": m.raw.get("longitudes")}
            for m in matcher.models if m.flight_id in matched_flight_ids
        ],
        "f0_hz": primary.get("f0_hz"),
        "evaluated_flights": _flight_summaries(flights, diags, flyover_fits, discarded),
        "grid_resolution": int(G["X"].shape[0]),
        "speed_of_sound_mps": opts.speed_of_sound,
        "clock_mode": "trusted_window" if matcher.clock_window else "free",
    }


@app.post("/api/audio/upload")
def upload_audio(file: UploadFile = File(...)):
    """
    Upload a recording for server-side analysis.  Files are stored outside the
    static tree, size-capped, and deleted after ``DOPPLER_UPLOAD_TTL_S``.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in AUDIO_EXTS:
        raise HTTPException(status_code=400, detail="Only audio files (WAV, MP3, OGG, FLAC, M4A) are supported.")
    cleanup_uploads()
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    dest_path = os.path.join(upload_dir(), unique_filename)
    size = _save_upload(file, dest_path)
    return {
        "recording_metadata": read_recording_start(dest_path),
        "filename": file.filename,
        "saved_as": unique_filename,
        "url": f"/api/audio/uploads/{unique_filename}",
        "size_bytes": size,
        "expires_in_s": settings.upload_ttl_s,
    }


@app.get("/api/audio/uploads/{name}")
def get_uploaded_audio(name: str):
    """Serve a previously uploaded file by its unguessable ``saved_as`` name (until it expires)."""
    path = _safe_child(upload_dir(), name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Upload not found or expired.")
    return FileResponse(path)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# index.html uses relative asset paths (so the same file works on GitHub Pages).
for _sub in ("css", "js", "samples"):
    app.mount(f"/{_sub}", StaticFiles(directory=os.path.join(STATIC_DIR, _sub)), name=f"static_{_sub}")


@app.get("/")
@app.head("/")
def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
