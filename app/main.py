"""
Doppler Acoustic Geolocator - FastAPI Application.
Integrates audio Doppler curve extraction, ADS-B trajectory matching,
and 3D acoustic physics engine for microphone geolocation.
"""

import os
import shutil
import uuid
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.physics.doppler_engine import DopplerEngine
from app.physics.grid_scorer import GridScorer
from app.adsb.flight_manager import FlightManager
from app.adsb.opensky_client import OpenSkyClient

app = FastAPI(
    title="Doppler - Acoustic ADS-B Geolocator",
    description="Full-stack web application that geolocates audio recordings by matching acoustic Doppler curves of passing airplanes with ADS-B flight tracking data.",
    version="1.0.0",
)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Physics & ADS-B instances
doppler_engine = DopplerEngine()
grid_scorer = GridScorer(doppler_engine)
opensky_client = OpenSkyClient()
flight_manager = FlightManager(opensky_client)

UPLOAD_DIR = "/workspace/gentle-curie/app/static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class CurvePoint(BaseModel):
    time: float
    freq: float


class GeolocateRequest(BaseModel):
    center_lat: float = Field(..., description="Search area center latitude")
    center_lon: float = Field(..., description="Search area center longitude")
    radius_km: float = Field(10.0, description="Search radius in kilometers", ge=1.0, le=100.0)
    f0: float = Field(..., description="Estimated fundamental frequency in Hz", ge=20.0, le=10000.0)
    traced_curve: List[CurvePoint] = Field(..., description="User traced Doppler curve points")
    scenario_id: Optional[str] = None
    flight_id: Optional[str] = None
    time_window_start: Optional[float] = None
    time_window_end: Optional[float] = None
    grid_resolution: Optional[int] = Field(25, ge=11, le=41)


class FlightSearchRequest(BaseModel):
    center_lat: float
    center_lon: float
    radius_km: float = 10.0
    scenario_id: Optional[str] = None
    time_window_start: Optional[float] = None
    time_window_end: Optional[float] = None


@app.get("/api/health")
def health_check():
    return {"status": "ok", "app": "doppler-geolocator", "version": "1.0.0"}


@app.get("/api/scenarios")
def list_scenarios():
    """Return available curated flight scenarios for quick loading."""
    scenarios = flight_manager.get_curated_scenario_list()
    # Add audio sample links
    for sc in scenarios:
        sc["sample_audio_url"] = f"/static/samples/{sc['id']}.wav"
    return scenarios


@app.post("/api/flights/search")
def search_flights(req: FlightSearchRequest):
    """
    Search or generate ADS-B flight trajectories in the requested spatial and temporal window.
    """
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


@app.post("/api/geolocate")
def geolocate(req: GeolocateRequest):
    """
    Evaluate candidate microphone positions across a spatial grid
    using Least Squares comparison between user's traced Doppler curve
    and 3D acoustic theoretical Doppler curves of passing aircraft.
    """
    if len(req.traced_curve) < 3:
        raise HTTPException(status_code=400, detail="Traced curve must contain at least 3 points.")

    # 1. Obtain candidate flight trajectories in search window
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

    # If specific flight_id requested, filter to that flight
    if req.flight_id:
        target_flights = [f for f in flights if f.get("flight_id") == req.flight_id]
        if target_flights:
            flights = target_flights

    curve_dicts = [{"time": p.time, "freq": p.freq} for p in req.traced_curve]
    res_grid = req.grid_resolution or 25

    flight_evaluations = []
    best_eval = None
    lowest_rmse = float("inf")

    for flight in flights:
        try:
            eval_result = grid_scorer.evaluate_grid(
                center_lat=req.center_lat,
                center_lon=req.center_lon,
                radius_km=req.radius_km,
                flight_trajectory=flight,
                traced_curve=curve_dicts,
                f0=req.f0,
                grid_resolution=res_grid,
            )
            eval_result["flight_trajectory"] = {
                "flight_id": flight.get("flight_id"),
                "callsign": flight.get("callsign"),
                "aircraft_type": flight.get("aircraft_type"),
                "latitudes": flight.get("latitudes"),
                "longitudes": flight.get("longitudes"),
                "altitudes": flight.get("altitudes"),
                "speeds": flight.get("speeds"),
            }
            flight_evaluations.append({
                "flight_id": eval_result["flight_id"],
                "callsign": eval_result["callsign"],
                "aircraft_type": eval_result["aircraft_type"],
                "best_location": eval_result["best_location"],
            })

            rmse = eval_result["best_location"]["rmse_hz"]
            if rmse < lowest_rmse:
                lowest_rmse = rmse
                best_eval = eval_result
        except Exception as e:
            continue

    if not best_eval:
        raise HTTPException(status_code=500, detail="Failed to evaluate acoustic Doppler model against flights.")

    return {
        "status": "success",
        "best_flight": {
            "flight_id": best_eval["flight_id"],
            "callsign": best_eval["callsign"],
            "aircraft_type": best_eval["aircraft_type"],
        },
        "best_location": best_eval["best_location"],
        "heatmap": best_eval["heatmap"],
        "curve_comparison": best_eval["curve_comparison"],
        "flight_trajectory": best_eval["flight_trajectory"],
        "f0_hz": best_eval["f0_hz"],
        "evaluated_flights": flight_evaluations,
    }


@app.post("/api/audio/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Upload custom MP3/WAV audio file for analysis."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".wav", ".mp3", ".ogg", ".flac", ".m4a"]:
        raise HTTPException(status_code=400, detail="Only audio files (WAV, MP3, OGG, FLAC) are supported.")

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


# Mount static assets
app.mount("/static", StaticFiles(directory="/workspace/gentle-curie/app/static"), name="static")


@app.get("/")
def serve_index():
    return FileResponse("/workspace/gentle-curie/app/static/index.html")
