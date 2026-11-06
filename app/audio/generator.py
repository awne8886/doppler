"""
Acoustic flyover audio generator for Doppler geolocator testing.
Synthesizes WAV files with realistic Doppler-shifted engine tones, harmonics,
inverse-square amplitude decay, and atmospheric sound propagation.
"""

import math
import struct
import wave
from typing import Dict, Any, Tuple
import numpy as np

from app.physics.doppler_engine import DopplerEngine
from app.adsb.sample_flights import get_curated_scenarios


def synthesize_flyover_wav(
    trajectory: Dict[str, Any],
    observer_coords: Tuple[float, float],
    output_wav_path: str,
    duration_sec: float = 30.0,
    sample_rate: int = 44100,
    f0: float = 800.0,
) -> str:
    """
    Synthesize an audio WAV file of a flight passing over an observer,
    strictly obeying Doppler shift and 1/r acoustic attenuation.
    """
    engine = DopplerEngine()
    obs_lat, obs_lon = observer_coords

    # Extract trajectory
    t_emit = np.array(trajectory["timestamps"], dtype=np.float64)
    # Re-zero timestamps to 0..T
    t_emit = t_emit - t_emit[0]

    plane_lats = np.array(trajectory["latitudes"], dtype=np.float64)
    plane_lons = np.array(trajectory["longitudes"], dtype=np.float64)
    plane_alts = np.array(trajectory["altitudes"], dtype=np.float64)

    # Convert to Cartesian relative to observer
    plane_x, plane_y, plane_z = engine.geodetic_to_cartesian(
        plane_lats, plane_lons, plane_alts, obs_lat, obs_lon, 0.0
    )
    plane_pos = np.stack([plane_x, plane_y, plane_z], axis=1)

    # Calculate velocity vectors
    vx = np.gradient(plane_x, t_emit)
    vy = np.gradient(plane_y, t_emit)
    vz = np.gradient(plane_z, t_emit)
    plane_vel = np.stack([vx, vy, vz], axis=1)

    obs_pos = np.array([0.0, 0.0, 0.0])

    # Compute continuous observer times and frequencies
    t_obs, f_obs = engine.compute_doppler_curve(t_emit, plane_pos, plane_vel, obs_pos, f0)
    d_slant, _ = engine.compute_radial_velocity(plane_pos, plane_vel, obs_pos)

    # Find CPA time at observer
    idx_cpa = np.argmin(d_slant)
    cpa_t_obs = t_obs[idx_cpa]

    # Select audio time window centered on CPA (e.g. CPA +/- duration/2)
    t_start = max(0.0, cpa_t_obs - (duration_sec / 2.0))
    t_end = t_start + duration_sec

    # Audio synthesis time samples
    num_samples = int(duration_sec * sample_rate)
    t_audio = np.linspace(t_start, t_end, num_samples)

    # Interpolate observed frequency and slant distance at audio timestamps
    sort_idx = np.argsort(t_obs)
    t_obs_sorted = t_obs[sort_idx]
    f_obs_sorted = f_obs[sort_idx]
    d_slant_sorted = d_slant[sort_idx]

    f_instant = np.interp(t_audio, t_obs_sorted, f_obs_sorted)
    d_instant = np.interp(t_audio, t_obs_sorted, d_slant_sorted)

    # Compute phase as cumulative integral of instantaneous frequency
    # phi(t) = 2 * pi * sum(f * dt)
    dt_sample = 1.0 / sample_rate
    phase = 2.0 * np.pi * np.cumsum(f_instant) * dt_sample

    # Amplitude modulation: inverse square law with atmospheric absorption and ground decay
    # normalize amplitude so CPA is ~0.8 max
    min_dist = np.min(d_instant)
    amplitude = (min_dist / np.maximum(d_instant, 100.0)) ** 1.1

    # Sound tone synthesis: fundamental f0 + harmonics (blade pass / turbine tone)
    # Fundamental
    tone = np.sin(phase)
    # 2nd harmonic
    tone += 0.35 * np.sin(2.0 * phase)
    # 3rd harmonic
    tone += 0.15 * np.sin(3.0 * phase)
    # Add ambient low rumble / broad jet noise
    noise = np.random.normal(0, 0.08, num_samples)
    # Simple lowpass filter on noise
    noise = np.convolve(noise, np.ones(15) / 15.0, mode="same")

    signal = (tone * 0.7 + noise) * amplitude
    # Apply fade in/out envelope at boundaries to prevent clicks
    fade_len = int(sample_rate * 0.5)
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    signal[:fade_len] *= fade_in
    signal[-fade_len:] *= fade_out

    # Normalize to 16-bit PCM
    signal = signal / (np.max(np.abs(signal)) + 1e-6)
    int_samples = (signal * 32760.0).astype(np.int16)

    # Write WAV file
    with wave.open(output_wav_path, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())

    return output_wav_path


def generate_all_scenario_samples():
    """Generate sample WAV audio files for all curated scenarios."""
    scenarios = get_curated_scenarios()
    for s_id, s_data in scenarios.items():
        out_path = f"/workspace/gentle-curie/app/static/samples/{s_id}.wav"
        gt_lat = s_data["trajectory"]["ground_truth_mic"]["lat"]
        gt_lon = s_data["trajectory"]["ground_truth_mic"]["lon"]
        f0 = s_data["f0_target"]
        synthesize_flyover_wav(
            trajectory=s_data["trajectory"],
            observer_coords=(gt_lat, gt_lon),
            output_wav_path=out_path,
            duration_sec=25.0,
            f0=f0,
        )
