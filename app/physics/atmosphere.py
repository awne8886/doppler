"""
Altitude-datum helpers.

ADS-B barometric altitude is *pressure altitude*: the ISA height of the
measured static pressure relative to 1013.25 hPa.  True height above mean sea
level differs by roughly 8.4 m per hPa of (QNH - 1013.25) near the ground,
i.e. a few hundred metres on a deep low or high - enough to bias Doppler
geometry.  Geometric (GNSS) altitude is height above the WGS-84 ellipsoid and
needs no pressure correction (it differs from MSL by the local geoid
undulation, typically tens of metres).

The correction here uses the ISA troposphere and ignores temperature
deviations, which add a further error of a few percent of the height above
the station.  The sensitivity analysis widens its altitude perturbation for
barometric tracks to cover what remains.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import numpy as np

T0 = 288.15        # ISA sea-level temperature [K]
L = 0.0065         # ISA lapse rate [K/m]
P0 = 1013.25       # ISA sea-level pressure [hPa]
G = 9.80665
R_AIR = 287.05287
EXP = G / (R_AIR * L)  # ~5.25588


def pressure_at_pressure_altitude(pa_m):
    return P0 * (1.0 - L * np.asarray(pa_m, dtype=np.float64) / T0) ** EXP


def qnh_altitude(pressure_altitude_m, qnh_hpa: float):
    """Height above MSL implied by a pressure altitude under the given QNH (ISA temperature profile)."""
    p = pressure_at_pressure_altitude(pressure_altitude_m)
    return (T0 / L) * (1.0 - (p / float(qnh_hpa)) ** (1.0 / EXP))


def apply_qnh(traj: Dict[str, Any], qnh_hpa: float) -> Dict[str, Any]:
    """Copy of a barometric trajectory with QNH-corrected altitudes; other bases are returned unchanged."""
    if traj.get("altitude_basis") != "barometric" or traj.get("altitude_correction"):
        return traj
    out = copy.deepcopy(traj)
    out["altitudes"] = [round(float(a), 2) for a in qnh_altitude(traj["altitudes"], qnh_hpa)]
    out["altitude_correction"] = {"type": "qnh", "qnh_hpa": float(qnh_hpa)}
    return out
