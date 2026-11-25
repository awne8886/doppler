#!/usr/bin/env python3
"""
Regenerate the small track-import fixtures in tests/fixtures/tracks/ from the
curated ``sfo_dual`` trajectories, written in every supported import format.

    python scripts/make_track_fixtures.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.adsb.sample_flights import get_curated_scenarios  # noqa: E402

OUT = os.path.join(ROOT, "tests", "fixtures", "tracks")
FT, KT, FPM = 0.3048, 0.514444, 0.00508


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    os.makedirs(OUT, exist_ok=True)
    flights = get_curated_scenarios()["sfo_dual"]["flights"]

    # Generic CSV (ISO times with Z, feet, knots, fpm), all three flights.
    lines = ["icao24,callsign,time_utc,lat,lon,alt_ft,speed_kt,track_deg,vertical_rate_fpm"]
    for f in flights:
        for i, t in enumerate(f["timestamps"]):
            lines.append(f"{f['icao24']},{f['callsign']},{iso(t)},{f['latitudes'][i]:.6f},{f['longitudes'][i]:.6f},"
                         f"{f['altitudes'][i] / FT:.1f},{f['speeds'][i] / KT:.1f},{f['headings'][i]:.1f},"
                         f"{f['vertical_rates'][i] / FPM:.0f}")
    _write("generic.csv", "\n".join(lines) + "\n")

    # FlightRadar24 CSV export: one flight, 5 s cadence, integer feet / knots.
    f = flights[0]
    rows = ["Timestamp,UTC,Callsign,Position,Altitude,Speed,Direction"]
    for i in range(0, len(f["timestamps"]), 5):
        t = f["timestamps"][i]
        rows.append(f'{int(t)},{iso(t)},{f["callsign"]},"{f["latitudes"][i]:.5f},{f["longitudes"][i]:.5f}",'
                    f'{round(f["altitudes"][i] / FT)},{round(f["speeds"][i] / KT)},{round(f["headings"][i])}')
    _write("fr24.csv", "\n".join(rows) + "\n")

    # KML gx:Track: second flight, metres.
    f = flights[1]
    whens = "\n".join(f"      <when>{iso(t)}</when>" for t in f["timestamps"])
    coords = "\n".join(f"      <gx:coord>{lo:.6f} {la:.6f} {a:.1f}</gx:coord>"
                       for la, lo, a in zip(f["latitudes"], f["longitudes"], f["altitudes"]))
    _write("track.kml", f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">
<Document>
  <name>{f["callsign"]}</name>
  <Placemark>
    <gx:Track>
      <altitudeMode>absolute</altitudeMode>
{whens}
{coords}
    </gx:Track>
  </Placemark>
</Document>
</kml>
""")

    # GeoJSON LineStrings with times (epoch ms), all three flights, metres.
    feats = [{
        "type": "Feature",
        "properties": {"flight_id": g["flight_id"], "callsign": g["callsign"], "icao24": g["icao24"],
                       "times": [int(t * 1000) for t in g["timestamps"]], "altitude_basis": "geometric"},
        "geometry": {"type": "LineString",
                     "coordinates": [[lo, la, a] for la, lo, a in zip(g["latitudes"], g["longitudes"], g["altitudes"])]},
    } for g in flights]
    _write("tracks.geojson", json.dumps({"type": "FeatureCollection", "features": feats}, indent=1))

    # readsb / tar1090 trace_full JSON: first flight, with geometric altitude (index 10).
    f = flights[0]
    t0 = f["timestamps"][0]
    trace = []
    for i, t in enumerate(f["timestamps"]):
        ac = {"flight": f["callsign"].ljust(8), "type": "adsb_icao"} if i == 0 else None
        trace.append([round(t - t0, 2), f["latitudes"][i], f["longitudes"][i], round(f["altitudes"][i] / FT + 75),
                      round(f["speeds"][i] / KT, 1), f["headings"][i], 0, round(f["vertical_rates"][i] / FPM),
                      ac, "adsb_icao", round(f["altitudes"][i] / FT), round(f["vertical_rates"][i] / FPM), None, None])
    _write("trace_full_a1b2c3.json", json.dumps({"icao": "a1b2c3", "r": "N12345", "t": "B739", "timestamp": t0,
                                                "trace": trace}))


def _write(name: str, text: str):
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write(text)
    print("wrote", os.path.relpath(os.path.join(OUT, name), ROOT), len(text), "bytes")


if __name__ == "__main__":
    main()
