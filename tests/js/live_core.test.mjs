// Tests for live listening (app/static/js/live_core.js): resampling, ring buffer, sample clock,
// flyover commit/de-duplication over a sliding window, live ADS-B track accumulation, search
// refinement, and an end-to-end run (synthetic microphone audio of two flyovers -> ridge discovery
// -> committed flyovers -> joint match against polled ADS-B tracks in the browser engine).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
// Loaded into this test process's own global scope (node --test isolates files): global lookups inside a
// vm context are slow enough to make repeated ridge discovery over minutes of audio take far too long.
const ctx = globalThis;
ctx.window = ctx;
for (const f of ['util.js', 'doppler_analysis.js', 'live_core.js']) {
    (0, eval)(`${readFileSync(join(ROOT, 'app/static/js', f), 'utf8')}\n//# sourceURL=${f}`);
}
const LC = ctx.LiveCore;
const DA = ctx.DopplerAnalysis;

// ------------------------------------------------------------------ synthetic world
const RATE = 8000;
const C = 343.2;
const MIC = { lat: 37.5585, lon: -122.2711 };
const R_E = 6371000;
const toLatLon = (e, n) => ({
    lat: MIC.lat + (n / R_E) * 180 / Math.PI,
    lon: MIC.lon + (e / (R_E * Math.cos(MIC.lat * Math.PI / 180))) * 180 / Math.PI,
});
const SESSION_EPOCH = 1700000000.37;

// Straight, constant-velocity flights (local ENU metres around the microphone).
function flight(id, headingDeg, speed, alt, cpaOffsetE, cpaOffsetN, cpaEpoch, f0) {
    const h = headingDeg * Math.PI / 180;
    const v = { e: speed * Math.sin(h), n: speed * Math.cos(h) };
    return {
        id, f0, speed, headingDeg, alt,
        pos(epoch) { const dt = epoch - cpaEpoch; return { e: cpaOffsetE + v.e * dt, n: cpaOffsetN + v.n * dt, u: alt }; },
    };
}
const FLIGHTS = [
    flight('AAA111', 300, 160, 1200, 700, 400, SESSION_EPOCH + 70, 820),
    flight('BBB222', 30, 140, 900, -600, 500, SESSION_EPOCH + 175, 610),
];

/** Received signal: the phase at the microphone equals the phase emitted at the emission time. */
function synthAudio(durationS, seed = 7) {
    let s = seed;
    const rnd = () => { s = (s * 1103515245 + 12345) % 2147483648; return s / 2147483648; };
    const gauss = () => Math.sqrt(-2 * Math.log(Math.max(rnd(), 1e-12))) * Math.cos(2 * Math.PI * rnd());
    const n = Math.floor(durationS * RATE);
    const x = new Float32Array(n);
    for (let i = 0; i < n; i++) x[i] = 0.05 * gauss() + 0.03 * Math.sin(2 * Math.PI * 120 * i / RATE); // noise + mains-like hum
    for (const f of FLIGHTS) {
        // Emission time for each arrival time by fixed-point iteration (from the previous solution).
        let tau = SESSION_EPOCH - 60;
        for (let i = 0; i < n; i++) {
            const t = SESSION_EPOCH + i / RATE;
            let r = 0;
            for (let it = 0; it < 4; it++) {
                const p = f.pos(tau);
                r = Math.hypot(p.e, p.n, p.u);
                tau = t - r / C;
            }
            if (r > 6000) continue;
            x[i] += 0.25 * (1000 / r) * Math.sin(2 * Math.PI * f.f0 * (tau - SESSION_EPOCH));
        }
    }
    return x;
}

/** /api/flights/search response for a live snapshot at `epoch` (FlightManager._project_state layout). */
function liveResponse(epoch) {
    const flights = FLIGHTS.map(f => {
        const ts = [], la = [], lo = [], al = [], sp = [], hd = [], vr = [];
        for (let i = 0; i <= 60; i++) {
            const t = epoch - 30 + i;
            const p = f.pos(t), ll = toLatLon(p.e, p.n);
            ts.push(t); la.push(ll.lat); lo.push(ll.lon); al.push(p.u); sp.push(f.speed); hd.push(f.headingDeg); vr.push(0);
        }
        return { flight_id: f.id, icao24: f.id.toLowerCase(), callsign: f.id, timestamps: ts, latitudes: la, longitudes: lo,
            altitudes: al, speeds: sp, headings: hd, vertical_rates: vr, data_source: 'opensky_live',
            altitude_basis: 'geometric', projection: 'linear_from_single_state', state_time: epoch };
    });
    return { status: 'ok', data_source: 'opensky_live', flights };
}

// -------------------------------------------------------------------------- units
test('resampler: chunked output equals one-shot output and keeps the frequency', () => {
    const inRate = 44100, n = 44100;
    const x = new Float32Array(n);
    for (let i = 0; i < n; i++) x[i] = Math.sin(2 * Math.PI * 440 * i / inRate);
    const one = new LC.Resampler(inRate, RATE).process(x);
    const rs = new LC.Resampler(inRate, RATE);
    const parts = [];
    for (let i = 0; i < n; i += 997) parts.push(...rs.process(x.subarray(i, Math.min(n, i + 997))));
    assert.ok(Math.abs(one.length - n * RATE / inRate) <= 1, `length ${one.length}`);
    assert.equal(parts.length, one.length);
    for (let i = 0; i < one.length; i++) assert.ok(Math.abs(parts[i] - one[i]) < 1e-6, `sample ${i}`);
    for (let k = 0; k < one.length; k += 401) {
        assert.ok(Math.abs(one[k] - Math.sin(2 * Math.PI * 440 * k / RATE)) < 0.01, `value at ${k}`);
    }
});

test('ring buffer keeps the newest samples by absolute index', () => {
    const rb = new LC.RingBuffer(10);
    rb.push(Float32Array.from([0, 1, 2, 3, 4, 5, 6]));
    rb.push(Float32Array.from([7, 8, 9, 10, 11, 12]));
    assert.equal(rb.total, 13);
    assert.equal(rb.oldest, 3);
    const r = rb.read(0, 13);
    assert.equal(r.start, 3);
    assert.deepEqual([...r.samples], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    assert.deepEqual([...rb.read(5, 8).samples], [5, 6, 7]);
});

test('sample clock follows jitter, re-anchors on dropped audio and tracks drift', () => {
    const clock = new LC.SampleClock(RATE);
    const chunk = 1024;
    let k = 0;
    // 60 s with +-5 ms timestamp jitter, then 3 s of audio lost, then an audio clock running 100 ppm fast.
    for (; k < 60 * RATE; k += chunk) clock.addChunk(k, 1000 + k / RATE + 0.005 * Math.sin(k));
    assert.ok(Math.abs(clock.toEpoch(30 * RATE) - (1000 + 30)) < 0.01);
    const kGap = k;
    for (; k < kGap + 600 * RATE; k += chunk) clock.addChunk(k, 1003 + (k / RATE) * (1 - 1e-4));
    assert.equal(clock.discontinuities, 1);
    assert.ok(Math.abs(clock.toEpoch(30 * RATE) - 1030) < 0.01, 'before the gap');
    const kq = kGap + 500 * RATE;
    const truth = 1003 + (kq / RATE) * (1 - 1e-4);
    assert.ok(Math.abs(clock.toEpoch(kq) - truth) < 0.02, `after gap with drift: ${clock.toEpoch(kq)} vs ${truth}`);
});

test('ADS-B store keeps only the reported state of each live snapshot and de-duplicates it', () => {
    const store = new LC.AdsbTrackStore();
    for (let t = SESSION_EPOCH; t < SESSION_EPOCH + 50; t += 10) {
        store.ingest(liveResponse(t));
        store.ingest(liveResponse(t)); // the same state reported twice
    }
    store.ingest({ status: 'ok', flights: [Object.assign(liveResponse(SESSION_EPOCH).flights[0], { data_source: 'synthetic', icao24: 'zzz' })] });
    assert.equal(store.size, 2);
    const trajs = store.trajectories(SESSION_EPOCH, SESSION_EPOCH + 40);
    assert.equal(trajs.length, 2);
    const a = trajs.find(t => t.icao24 === 'aaa111');
    assert.equal(a.timestamps.length, 5);
    const p = FLIGHTS[0].pos(SESSION_EPOCH + 20), ll = toLatLon(p.e, p.n);
    assert.ok(Math.abs(a.latitudes[2] - ll.lat) < 1e-9 && Math.abs(a.longitudes[2] - ll.lon) < 1e-9);
    assert.equal(a.source_format, 'live_poll');
    store.prune(SESSION_EPOCH + 3600 + 25);
    assert.equal(store.trajectories(SESSION_EPOCH, SESSION_EPOCH + 40).length, 0, 'fewer than 3 points left');
});

test('search refinement narrows after a supported result and widens back otherwise', () => {
    const base = { lat: 37.5, lon: -122.3, radius_km: 10, refined: false };
    const ok = { status: 'success', decision: { status: 'supported' }, best_location: { lat: 37.51, lon: -122.31, uncertainty_radius_m: 300 } };
    const a = LC.refineSearchArea(ok, base, base);
    assert.deepEqual(JSON.parse(JSON.stringify(a)), { lat: 37.51, lon: -122.31, radius_km: 2, refined: true });
    assert.ok(LC.areaChanged(a, base));
    const bad = { status: 'success', decision: { status: 'inconclusive' }, best_location: { lat: 37.6, lon: -122.2 } };
    const b = LC.refineSearchArea(bad, a, base);
    assert.equal(b.radius_km, 4);
    assert.equal(b.lat, 37.51);
    const c = LC.refineSearchArea(bad, LC.refineSearchArea(bad, b, base), base);
    assert.deepEqual(JSON.parse(JSON.stringify(c)), JSON.parse(JSON.stringify(base)));
    assert.ok(!LC.areaChanged(a, Object.assign({}, a)));
});

// ---------------------------------------------------------------------- end to end
test('live pipeline: sliding-window discovery commits each flyover once and the joint match finds the microphone', async () => {
    const DUR = 290;
    const audio = synthAudio(DUR);
    // Feed the audio as the browser would, in 2048-sample chunks with jittery time stamps.
    const ring = new LC.RingBuffer(300 * RATE);
    const clock = new LC.SampleClock(RATE);
    const tracker = new LC.FlyoverTracker();
    const adsb = new LC.AdsbTrackStore();
    let nextPoll = SESSION_EPOCH, nextAnalysis = SESSION_EPOCH + 30;
    const commits = [];
    for (let k = 0; k < audio.length; k += 2048) {
        const chunk = audio.subarray(k, Math.min(audio.length, k + 2048));
        const epoch = SESSION_EPOCH + k / RATE + 0.004 * Math.cos(k / 7);
        clock.addChunk(ring.total, epoch);
        ring.push(chunk);
        const now = SESSION_EPOCH + ring.total / RATE;
        if (now >= nextPoll) { adsb.ingest(liveResponse(Math.round(now))); nextPoll += 10; }
        if (now >= nextAnalysis) {
            nextAnalysis += 30;
            const k1 = ring.total, k0 = Math.max(ring.oldest, k1 - 240 * RATE);
            const { start, samples } = ring.read(k0, k1);
            const res = DA.RidgeDiscovery.discoverSignal(samples, RATE, { fmax_hz: 2000 });
            const ch = tracker.ingest(res, start, RATE, (kk) => clock.toEpoch(kk), samples.length / RATE);
            commits.push({ at: now - SESSION_EPOCH, added: ch.added.map(f => f.id), updated: ch.updated.map(f => f.id) });
        }
    }
    assert.equal(tracker.flyovers.length, 2, `flyovers: ${JSON.stringify(commits)}`);
    // Each flyover was added exactly once even though later windows saw it again.
    assert.equal(commits.flatMap(c => c.added).length, 2);
    const [f1, f2] = tracker.flyovers;
    const cpa1 = SESSION_EPOCH + 70 + Math.hypot(700, 400, 1200) / C;
    const cpa2 = SESSION_EPOCH + 175 + Math.hypot(600, 500, 900) / C;
    assert.ok(f1.epoch_start < cpa1 && f1.epoch_end > cpa1, 'flyover 1 spans its CPA arrival');
    assert.ok(f2.epoch_start < cpa2 && f2.epoch_end > cpa2, 'flyover 2 spans its CPA arrival');

    const { origin, events } = tracker.events(8);
    assert.ok(events.length >= 2);
    assert.ok(events.every(e => e.traced_curve[0].time >= 0));
    const trajectories = adsb.trajectories(origin, f2.epoch_end);
    assert.equal(trajectories.length, 2);

    // Search centre 3 km off the truth, as a user would give "roughly here".
    const centre = toLatLon(2200, -2000);
    const payload = {
        center_lat: centre.lat, center_lon: centre.lon, radius_km: 8, events, trajectories,
        recording_start_epoch: origin, clock_tolerance_s: 2, fit_f0: true, grid_resolution: 21, sensitivity: true,
    };
    const found = { status: 'ok', data_source: 'uploaded', notes: [], limitations: [],
        flights: trajectories.map(t => Object.assign({}, t, { data_source: 'uploaded' })) };
    const result = await DA.runPipeline(payload, found);
    assert.equal(result.status, 'success', result.message);
    const best = result.best_location;
    const errM = Math.hypot((best.lat - MIC.lat) * 111000, (best.lon - MIC.lon) * 111000 * Math.cos(MIC.lat * Math.PI / 180));
    assert.ok(errM < 300, `location error ${errM.toFixed(0)} m`);
    assert.equal(result.decision.status, 'supported', JSON.stringify(result.decision));
    const matched = new Set((result.events || []).filter(e => e.matched).map(e => e.flight_id));
    assert.ok(matched.has('AAA111') && matched.has('BBB222'), `matched ${[...matched]}`);

    // Refinement: the next search is centred on the estimate and smaller.
    const next = LC.refineSearchArea(result, { lat: centre.lat, lon: centre.lon, radius_km: 8 }, { lat: centre.lat, lon: centre.lon, radius_km: 8 });
    assert.ok(next.refined && next.radius_km < 8);
});
