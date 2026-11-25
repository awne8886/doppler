// Unit tests for the DOM-free frontend helpers (app/static/js/results.js, util.js).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const ctx = vm.createContext({ console, Date, TextDecoder });
ctx.window = ctx;
for (const f of ['util.js', 'results.js']) vm.runInContext(readFileSync(join(ROOT, 'app/static/js', f), 'utf8'), ctx, { filename: f });
const RT = ctx.ResultTools;
const U = ctx.DopplerUtil;
const plain = (v) => JSON.parse(JSON.stringify(v));

// A straight north-south track 1 km east of the best point (local frame around 37.5, -122.3).
const best = { lat: 37.5, lon: -122.3, robust_rmse_hz: 0.4, rmse_hz: 0.5, uncertainty_radius_m: 250, recording_start_epoch: 1.7e9 };
const dLon = 1000 / (6371000 * Math.cos(37.5 * Math.PI / 180)) * 180 / Math.PI;
const track = { flight_id: 'T1', callsign: 'T1', latitudes: [37.45, 37.48, 37.5, 37.52, 37.55], longitudes: Array(5).fill(-122.3 + dLon) };

test('escapeHtml neutralises markup', () => {
    assert.equal(U.escapeHtml('<img src=x onerror="a()">&\''), '&lt;img src=x onerror=&quot;a()&quot;&gt;&amp;&#39;');
    assert.equal(U.escapeHtml(null), '');
    assert.equal(U.fmtNum('abc', 1), '--');
});

test('data source badges cover every source', () => {
    for (const s of ['curated_demo', 'synthetic', 'opensky_live', 'adsblol_live', 'historical', 'uploaded']) {
        assert.notEqual(RT.sourceInfo(s).kind, 'unknown', s);
    }
    assert.equal(RT.sourceInfo('curated_demo').kind, 'demo');
    assert.equal(RT.sourceInfo(undefined).kind, 'unknown');
});

test('mirror ambiguity reflects the location across the track', () => {
    const amb = RT.mirrorAmbiguity(best, track);
    assert.ok(Math.abs(amb.distance_m - 2000) < 1, `distance ${amb.distance_m}`);
    // Within ~1 m (the equirectangular local frame bends slightly with latitude).
    assert.ok(Math.abs(amb.mirror.lat - 37.5) < 1e-5);
    assert.ok(Math.abs(amb.mirror.lon - (-122.3 + 2 * dLon)) < 1e-5);
    assert.equal(amb.ridges.length, 2);
    // Ridges run parallel to the (north-south) track through best and mirror.
    assert.ok(Math.abs(amb.ridges[0][0].lon - best.lon) < 1e-5 && Math.abs(amb.ridges[0][1].lon - best.lon) < 1e-5);
});

test('GeoJSON export includes best, circle, alternatives, tracks and mirror for single flyovers', () => {
    const result = {
        best_location: best, data_source: 'uploaded', clock_mode: 'free',
        decision: { status: 'inconclusive', reason_code: 'insufficient_independent_events',
            checks: [{ check: 'independent_flyovers', passed: false, independent_flyovers: 1 }] },
        alternatives: [{ lat: 37.51, lon: -122.29, robust_rmse_hz: 0.6, distance_from_best_m: 1500 }],
        matched_flights: [track],
    };
    const gj = plain(RT.toGeoJSON(result));
    assert.equal(gj.type, 'FeatureCollection');
    const roles = gj.features.map(f => f.properties.role);
    assert.deepEqual(roles, ['best_location', 'sensitivity_circle', 'alternative', 'matched_track', 'mirror_ambiguity']);
    assert.deepEqual(gj.features[0].geometry.coordinates, [best.lon, best.lat]);
    assert.equal(gj.features[0].properties.calibrated, false);
    const ring = gj.features[1].geometry.coordinates[0];
    assert.deepEqual(ring[0], ring[ring.length - 1]);
    // Two supported flyovers: no mirror feature.
    const two = Object.assign({}, result, { decision: { status: 'supported', checks: [{ check: 'independent_flyovers', independent_flyovers: 2 }] } });
    assert.ok(!plain(RT.toGeoJSON(two)).features.some(f => f.properties.role === 'mirror_ambiguity'));
});

test('trace sessions round-trip and reject malformed input', () => {
    const events = [{ id: 'R1', group_id: 'G1', include: true, points: [{ time: 1.23456, freq: 800.1234, x: 1, y: 2 }],
        meta: { doppler_like: true, harmonic_ratio: '1', flyover_fit: { f0_hz: 800 } } }];
    const saved = plain(RT.sessionToJSON(events, { grid: 25 }, { audio: { filename: 'a.wav', duration_s: 30 } }));
    const parsed = plain(RT.parseSession(saved));
    assert.equal(parsed.events[0].id, 'R1');
    assert.deepEqual(parsed.events[0].points, [{ time: 1.2346, freq: 800.123 }]);
    assert.equal(parsed.settings.grid, 25);
    assert.equal(parsed.audio.filename, 'a.wav');
    assert.throws(() => RT.parseSession({ format: 'other' }));
    assert.throws(() => RT.parseSession({ format: 'doppler-trace-session', events: [{ points: 'x' }] }));
    const dirty = RT.parseSession({ format: 'doppler-trace-session', events: [{ id: 'X', points: [{ time: 'a', freq: 1 }, { time: 1, freq: -5 }, { time: 2, freq: 700 }] }] });
    assert.equal(dirty.events[0].points.length, 1);
});
