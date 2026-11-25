// Python/JS parity: the browser engine (app/static/js) must reproduce the
// Python engine's golden outputs (tests/fixtures/parity/golden.json, written
// by scripts/make_parity_fixtures.py). No dependencies: node:test + vm.
//
//   node --test tests/js/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const golden = JSON.parse(readFileSync(join(ROOT, 'tests/fixtures/parity/golden.json'), 'utf8'));

function loadBrowserScripts(...files) {
    const ctx = vm.createContext({ console, setTimeout, clearTimeout, TextDecoder });
    ctx.window = ctx; // window shim: the scripts attach their APIs to window
    for (const f of files) vm.runInContext(readFileSync(join(ROOT, 'app/static/js', f), 'utf8'), ctx, { filename: f });
    return ctx;
}
const ctx = loadBrowserScripts('util.js', 'doppler_analysis.js', 'track_import.js');
const DA = ctx.DopplerAnalysis;
const TI = ctx.TrackImport;

// Values created inside the vm context have that context's Array/Object prototypes;
// normalise them before strict structural comparison.
const plain = (v) => JSON.parse(JSON.stringify(v));
const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b} (tol ${tol})`);
const relClose = (a, b, rel, msg) => close(a, b, rel * Math.max(1, Math.abs(b)), msg);

test('flyover fit agrees within fit tolerance', () => {
    const g = golden.flyover_fit;
    const fit = DA.FlyoverFit.fit(g.times, g.freqs);
    close(fit.f0_hz, g.expected.f0_hz, 0.5, 'f0');
    close(fit.speed_mps, g.expected.speed_mps, 3.0, 'speed');
    close(fit.cpa_slant_m, g.expected.cpa_slant_m, 60.0, 'slant');
    close(fit.cpa_arrival_time_s, g.expected.cpa_arrival_time_s, 0.1, 'cpa arrival');
    close(fit.rmse_hz, g.expected.rmse_hz, 0.05, 'rmse');
});

test('arrival curves and fit_event match on fixed inputs', () => {
    for (const cs of golden.fit_event) {
        const o = Object.assign({}, DA.DEFAULT_MATCH, cs.opts);
        const engine = new DA.GeoEngine(o.speed_of_sound);
        const center = golden.end_to_end.center;
        const model = new DA.TrajectoryModel(cs.flight, center.lat, center.lon, engine, 0, o.max_gap_s);
        const [x, y, z] = cs.observer;
        const curve = model.arrival({ x, y, z }, o.speed_of_sound, [o.wind_e, o.wind_n]);
        assert.equal(curve.t.length, cs.curve_t.length);
        curve.t.forEach((t, i) => close(t, cs.curve_t[i], 1e-7, `curve t[${i}]`));
        curve.g.forEach((v, i) => close(v, cs.curve_g[i], 1e-10, `curve g[${i}]`));
        const pts = cs.event.traced_curve.slice().sort((a, b) => a.time - b.time);
        const ev = { t: pts.map(p => p.time), f: pts.map(p => p.freq), f0_fixed: o.fit_f0 ? null : cs.event.f0 };
        cs.starts.forEach((s, i) => {
            const r = DA.fitEventAt(curve, ev, s, o);
            if (cs.loss[i] === null) { assert.ok(!Number.isFinite(r.loss), `start ${i} should be infeasible`); return; }
            relClose(r.loss, cs.loss[i], 1e-6, `loss[${i}]`);
            relClose(r.f0, cs.f0[i], 1e-8, `f0[${i}]`);
            close(r.drift, cs.drift[i], 1e-8, `drift[${i}]`);
            relClose(r.rmse, cs.rmse[i], 1e-6, `rmse[${i}]`);
        });
    }
});

test('wind propagation and speed of sound are identical', () => {
    const p = golden.propagation;
    p.R.forEach((r, i) => {
        const v = p.V[i];
        const [T, g] = DA.propagation(r[0], r[1], r[2], v[0], v[1], v[2], p.c, p.wind[0], p.wind[1]);
        relClose(T, p.T[i], 1e-12, `T[${i}]`);
        relClose(g, p.g[i], 1e-12, `g[${i}]`);
    });
    for (const s of golden.speed_of_sound) relClose(DA.speedOfSound(s.t, s.rh), s.c, 1e-12, `c(${s.t}, ${s.rh})`);
});

test('trajectory conditioning matches', () => {
    const c = golden.conditioning;
    const out = DA.conditionTrajectory(c.input);
    for (const k of Object.keys(c.expected)) {
        assert.equal(out[k].length, c.expected[k].length, `${k} length`);
        const tol = k.startsWith('velocity') ? 2e-3 : (k === 'latitudes' || k === 'longitudes' ? 2e-7 : 2e-2);
        out[k].forEach((v, i) => close(v, c.expected[k][i], tol, `${k}[${i}]`));
    }
});

test('pre-filter makes the same decisions', () => {
    const p = golden.prefilter;
    const { kept, discarded } = DA.prefilterFlights(p.flights, p.events, p.fits, p.clock, 0);
    assert.deepEqual(plain(kept.map(f => f.flight_id)), p.kept);
    const summary = (list) => list.map(d => [d.flight_id, d.groups.map(g => [g.group_id, g.reasons.map(r => r.reason)])]);
    assert.deepEqual(plain(summary(discarded)), summary(p.discarded));
});

test('decision layer is identical', () => {
    for (const d of golden.decide) {
        const out = DA.decideLocation(d.result, d.policy, 2);
        assert.equal(out.status, d.status, `${d.name} ${JSON.stringify(d.policy)} status`);
        assert.equal(out.reason_code, d.reason_code, `${d.name} ${JSON.stringify(d.policy)} reason`);
        assert.deepEqual(plain(out.checks.map(c => [c.check, c.passed])), d.checks, `${d.name} checks`);
    }
});

test('track import matches (CSV, FR24, GeoJSON, readsb)', () => {
    for (const t of golden.track_import) {
        const text = readFileSync(join(ROOT, 'tests/fixtures/tracks', t.file), 'utf8');
        const res = TI.parseTracksText(text, t.file);
        assert.equal(res.format, t.format);
        assert.equal(res.flights.length, t.flights.length);
        res.flights.forEach((f, i) => {
            const e = t.flights[i];
            assert.equal(f.flight_id, e.flight_id);
            assert.equal(f.callsign, e.callsign);
            assert.equal(f.altitude_basis, e.altitude_basis);
            assert.deepEqual(plain(f.timestamps), e.timestamps);
            for (const k of ['latitudes', 'longitudes', 'altitudes', 'speeds']) {
                assert.equal(f[k].length, e[k].length, `${t.file} ${k}`);
                f[k].forEach((v, j) => close(v, e[k][j], 1e-9, `${t.file} ${k}[${j}]`));
            }
        });
    }
});

test('end-to-end match: same grid, location and decision', async () => {
    const e = golden.end_to_end;
    const m = new DA.JointMatcher(e.flights, e.events, e.center.lat, e.center.lon, {}, null);
    const res = await m.run(e.radius_km, e.grid, { sensitivity: false });
    assert.equal(res.status, 'ok');
    // Grid losses use identical rows (numpy-faithful clock grids and decimation).
    for (const cell of m.grid.cells) {
        const ref = e.grid_loss[cell.iy][cell.ix];
        if (ref === null) assert.ok(!Number.isFinite(cell.loss), `cell ${cell.iy},${cell.ix} should be infinite`);
        else relClose(cell.loss, ref, 1e-6, `grid loss ${cell.iy},${cell.ix}`);
    }
    const g = m.toGeo(res.best.x, res.best.y);
    const dy = (g.lat - e.best.lat) * 111195, dx = (g.lon - e.best.lon) * 111195 * Math.cos(e.best.lat * Math.PI / 180);
    assert.ok(Math.hypot(dx, dy) < 25, `best location differs by ${Math.hypot(dx, dy).toFixed(1)} m`);
    close(DA.robustRmse(res.best.loss / res.n_groups), e.best.robust_rmse_hz, 0.05, 'robust rmse');
    const matched = [...new Set(res.diagnostics.filter(d => d.matched).map(d => d.flight_id))].sort();
    assert.deepEqual(plain(matched), e.matched);
    const dec = DA.decideLocation(res, {}, e.events.length);
    assert.equal(dec.status, e.status);
    assert.equal(dec.reason_code, e.reason_code);
});
