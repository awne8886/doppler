/**
 * Result helpers without DOM dependencies (unit-tested in tests/js/):
 *  - data-source badges
 *  - single-flyover mirror ambiguity (mirror point + ambiguity ridge)
 *  - JSON / GeoJSON export of a geolocation result
 *  - trace-session (events + settings) serialisation
 */
(function (root) {
    const R = 6371000.0;
    const RAD = Math.PI / 180;

    const SOURCE_LABELS = {
        curated_demo: { label: 'DEMO', kind: 'demo', title: 'Curated demo scenario (simulated trajectories)' },
        synthetic: { label: 'SYNTHETIC', kind: 'demo', title: 'Synthetic demo flights; not real aircraft' },
        opensky_live: { label: 'LIVE', kind: 'live', title: 'OpenSky live snapshot (projected +/-30 s)' },
        adsblol_live: { label: 'LIVE', kind: 'live', title: 'ADSB.lol live snapshot (projected +/-30 s)' },
        historical: { label: 'HISTORICAL', kind: 'historical', title: 'Historical ADS-B provider' },
        uploaded: { label: 'UPLOADED', kind: 'uploaded', title: 'Tracks uploaded by the user' },
    };

    function sourceInfo(dataSource) {
        return SOURCE_LABELS[dataSource] || { label: 'UNKNOWN', kind: 'unknown', title: 'Unknown data source' };
    }

    // ---- local tangent plane around a reference point (equirectangular, as in the engine)
    function toLocal(lat, lon, ref) {
        const mid = 0.5 * (lat + ref.lat) * RAD;
        return { x: R * (lon - ref.lon) * RAD * Math.cos(mid), y: R * (lat - ref.lat) * RAD };
    }
    function toGeo(x, y, ref) {
        const lat = ref.lat + (y / R) / RAD;
        const mid = 0.5 * (lat + ref.lat) * RAD;
        return { lat, lon: ref.lon + (x / (R * Math.cos(mid))) / RAD };
    }

    /** Least-squares straight line through a track (local frame): point + unit direction. */
    function trackLine(track, ref) {
        const pts = track.latitudes.map((la, i) => toLocal(la, track.longitudes[i], ref));
        const n = pts.length;
        const mx = pts.reduce((a, p) => a + p.x, 0) / n, my = pts.reduce((a, p) => a + p.y, 0) / n;
        let sxx = 0, sxy = 0, syy = 0;
        for (const p of pts) { sxx += (p.x - mx) ** 2; sxy += (p.x - mx) * (p.y - my); syy += (p.y - my) ** 2; }
        const ang = 0.5 * Math.atan2(2 * sxy, sxx - syy);
        return { px: mx, py: my, ux: Math.cos(ang), uy: Math.sin(ang), pts };
    }

    /**
     * Mirror image of a location about a straight flight track. A single
     * constant-velocity flyover sounds the same on both sides of the vertical
     * plane through its track, so the mirror is an equally good fit. With a
     * free clock the observer can also slide along the track: the returned
     * ridges are lines parallel to the track through the best point and its
     * mirror, spanning the track's extent.
     */
    function mirrorAmbiguity(best, track) {
        const ref = { lat: best.lat, lon: best.lon };
        const L = trackLine(track, ref);
        const t0 = (0 - L.px) * L.ux + (0 - L.py) * L.uy;
        const fx = L.px + t0 * L.ux, fy = L.py + t0 * L.uy; // foot of the perpendicular from best
        const mirror = toGeo(2 * fx, 2 * fy, ref);
        const proj = L.pts.map(p => (p.x - L.px) * L.ux + (p.y - L.py) * L.uy);
        const a = Math.min(...proj) - t0, b = Math.max(...proj) - t0;
        const ridge = (ox, oy) => [toGeo(ox + a * L.ux, oy + a * L.uy, ref), toGeo(ox + b * L.ux, oy + b * L.uy, ref)];
        return {
            mirror, distance_m: 2 * Math.hypot(fx, fy),
            ridges: [ridge(0, 0), ridge(2 * fx, 2 * fy)],
        };
    }

    function isSingleFlyover(result) {
        const chk = ((result.decision && result.decision.checks) || []).find(c => c.check === 'independent_flyovers');
        return !!(chk && chk.independent_flyovers < 2);
    }

    function circlePolygon(lat, lon, radiusM, n = 64) {
        const ref = { lat, lon };
        const ring = [];
        for (let i = 0; i <= n; i++) {
            const a = (2 * Math.PI * i) / n;
            const g = toGeo(radiusM * Math.sin(a), radiusM * Math.cos(a), ref);
            ring.push([+g.lon.toFixed(7), +g.lat.toFixed(7)]);
        }
        return ring;
    }

    /** GeoJSON FeatureCollection: best point, alternatives, sensitivity circle, matched tracks, mirror. */
    function toGeoJSON(result) {
        const features = [];
        const d = result.decision || {};
        const best = result.best_location;
        if (best) {
            features.push({
                type: 'Feature', geometry: { type: 'Point', coordinates: [best.lon, best.lat] },
                properties: {
                    role: 'best_location', decision: d.status || null, reason_code: d.reason_code || null,
                    calibrated: false, robust_rmse_hz: best.robust_rmse_hz, rmse_hz: best.rmse_hz,
                    recording_start_epoch: best.recording_start_epoch, sensitivity_radius_m: best.uncertainty_radius_m,
                    data_source: result.data_source || null, clock_mode: result.clock_mode || null,
                },
            });
            if (best.uncertainty_radius_m) {
                features.push({
                    type: 'Feature', geometry: { type: 'Polygon', coordinates: [circlePolygon(best.lat, best.lon, best.uncertainty_radius_m)] },
                    properties: { role: 'sensitivity_circle', radius_m: best.uncertainty_radius_m,
                        note: 'Engineering sensitivity envelope; not a calibrated confidence region.' },
                });
            }
        }
        (result.alternatives || []).forEach((a, i) => features.push({
            type: 'Feature', geometry: { type: 'Point', coordinates: [a.lon, a.lat] },
            properties: { role: 'alternative', rank: i + 1, robust_rmse_hz: a.robust_rmse_hz, distance_from_best_m: a.distance_from_best_m },
        }));
        for (const f of result.matched_flights || []) {
            if (!f.latitudes || !f.latitudes.length) continue;
            features.push({
                type: 'Feature', geometry: { type: 'LineString', coordinates: f.latitudes.map((la, i) => [f.longitudes[i], la]) },
                properties: { role: 'matched_track', flight_id: f.flight_id, callsign: f.callsign, data_source: f.data_source || null },
            });
        }
        if (best && isSingleFlyover(result) && (result.matched_flights || []).length) {
            const amb = mirrorAmbiguity(best, result.matched_flights[0]);
            features.push({
                type: 'Feature', geometry: { type: 'Point', coordinates: [amb.mirror.lon, amb.mirror.lat] },
                properties: { role: 'mirror_ambiguity', distance_m: amb.distance_m,
                    note: 'A single straight flyover cannot distinguish the two sides of its track.' },
            });
        }
        return { type: 'FeatureCollection', features };
    }

    /** JSON export: the full result plus provenance of the export itself. */
    function toExportJSON(result, extra = {}) {
        return Object.assign({ format: 'doppler-result', version: 1, exported_at: new Date().toISOString() }, extra, { result });
    }

    const SESSION_FORMAT = 'doppler-trace-session';

    function sessionToJSON(events, settings, extra = {}) {
        return Object.assign({
            format: SESSION_FORMAT, version: 1, saved_at: new Date().toISOString(),
            settings,
            events: events.map(e => ({
                id: String(e.id), group_id: e.group_id || null, include: e.include !== false,
                points: e.points.map(p => ({ time: +(+p.time).toFixed(4), freq: +(+p.freq).toFixed(3) })),
                meta: e.meta ? { doppler_like: !!e.meta.doppler_like, harmonic_ratio: e.meta.harmonic_ratio || null,
                    flyover_fit: e.meta.flyover_fit || null } : null,
            })),
        }, extra);
    }

    /** Validate a parsed session object; throws on anything malformed. */
    function parseSession(obj) {
        if (!obj || obj.format !== SESSION_FORMAT) throw new Error('Not a Doppler trace-session file.');
        if (!Array.isArray(obj.events)) throw new Error('Session has no events array.');
        const events = obj.events.map((e, i) => {
            if (!Array.isArray(e.points)) throw new Error(`Event ${i + 1} has no points.`);
            const points = e.points.map(p => ({ time: Number(p.time), freq: Number(p.freq) }))
                .filter(p => Number.isFinite(p.time) && Number.isFinite(p.freq) && p.time >= 0 && p.freq > 0);
            return { id: String(e.id || `S${i + 1}`).slice(0, 32), group_id: e.group_id ? String(e.group_id).slice(0, 32) : null,
                include: e.include !== false, points, meta: e.meta || null };
        });
        return { events, settings: obj.settings && typeof obj.settings === 'object' ? obj.settings : {},
            audio: obj.audio || null, tracks: Array.isArray(obj.tracks) ? obj.tracks : [] };
    }

    root.ResultTools = { sourceInfo, mirrorAmbiguity, isSingleFlyover, toGeoJSON, toExportJSON, circlePolygon,
        sessionToJSON, parseSession, SOURCE_LABELS };
})(typeof window !== 'undefined' ? window : globalThis);
