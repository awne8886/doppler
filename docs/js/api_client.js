/**
 * API Client for Doppler Acoustic Aircraft Geolocator.
 * Automatically detects whether a FastAPI backend is reachable.
 * If running on GitHub Pages (static), it seamlessly executes the client-side
 * physics engine and ADS-B scenario manager with zero server dependencies!
 */
class ApiClient {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
        this.backendAvailable = null; // null = untried, true, false
    }

    async checkBackend() {
        if (this.backendAvailable !== null) return this.backendAvailable;
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 1200);
            const resp = await fetch(`${this.baseUrl}/api/health`, { signal: controller.signal });
            clearTimeout(timeoutId);
            this.backendAvailable = resp.ok;
        } catch (e) {
            this.backendAvailable = false;
        }
        return this.backendAvailable;
    }

    async getScenarios() {
        const hasBackend = await this.checkBackend();
        if (hasBackend) {
            try {
                const resp = await fetch(`${this.baseUrl}/api/scenarios`);
                if (resp.ok) return await resp.json();
            } catch (e) {
                console.warn('Backend scenarios failed, using client data:', e);
            }
        }
        // Fallback to client-side flight manager
        return window.clientFlights ? window.clientFlights.getScenarios() : [];
    }

    async searchFlights(params) {
        const hasBackend = await this.checkBackend();
        if (hasBackend) {
            try {
                const resp = await fetch(`${this.baseUrl}/api/flights/search`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(params),
                });
                if (resp.ok) return await resp.json();
            } catch (e) {
                console.warn('Backend flight search failed, falling back to client:', e);
            }
        }

        // Client-side flight retrieval (demo data only when explicitly requested)
        const found = window.clientFlights.findFlights(
            params.center_lat, params.center_lon, params.radius_km, params.scenario_id,
            { demo: !!params.demo, timeWindowStart: params.time_window_start }
        );
        return { count: found.flights.length, flights: found.flights, status: found.status,
            data_source: found.data_source, notes: found.notes, limitations: found.limitations };
    }

    async geolocateObserver(payload, onProgress = null) {
        const hasBackend = await this.checkBackend();
        if (hasBackend) {
            try {
                const resp = await fetch(`${this.baseUrl}/api/geolocate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                if (resp.ok) return await resp.json();
                const err = await resp.json().catch(() => ({}));
                if (resp.status >= 400 && resp.status < 500) throw new Error(err.detail || `HTTP ${resp.status}`);
            } catch (e) {
                if (e.message && !/fetch|network/i.test(e.message)) throw e;
                console.warn('Backend geolocation failed, falling back to client physics:', e);
            }
        }
        return this.clientGeolocate(payload, onProgress);
    }

    /** Full joint-matching pipeline in the browser (GitHub Pages mode). */
    async clientGeolocate(payload, onProgress) {
        const DA = window.DopplerAnalysis;
        const events = (payload.events && payload.events.length)
            ? payload.events
            : [{ id: 'E1', traced_curve: payload.traced_curve, f0: payload.f0 }];
        const found = window.clientFlights.findFlights(
            payload.center_lat, payload.center_lon, payload.radius_km, payload.scenario_id,
            { demo: !!payload.demo, timeWindowStart: payload.time_window_start }
        );
        let flights = found.flights;
        const provenance = { data_source: found.data_source, data_notes: found.notes, data_limitations: found.limitations };
        if (!flights || flights.length === 0) {
            return { status: 'no_adsb_data', message: found.message, mode: 'client_physics',
                decision: DA.decideLocation({ status: 'no_adsb_data' }, {}, events.length), ...provenance };
        }
        if (payload.flight_id) {
            const target = flights.filter(f => f.flight_id === payload.flight_id);
            if (target.length) flights = target;
        }
        const c = DA.speedOfSoundFor(payload.temperature_c, payload.relative_humidity_pct);
        const opts = {
            speed_of_sound: c,
            fit_f0: payload.fit_f0 !== false,
            loss_scale_hz: payload.loss_scale_hz || 2.0,
            drift_fraction: payload.drift_fraction != null ? payload.drift_fraction : 0.02,
            observer_alt_m: payload.observer_alt_m || 0,
        };
        if (!opts.fit_f0) events.forEach(e => { e.f0 = e.f0 || payload.f0; });
        const clock = payload.recording_start_epoch != null
            ? [payload.recording_start_epoch - (payload.clock_tolerance_s ?? 2), payload.recording_start_epoch + (payload.clock_tolerance_s ?? 2)]
            : null;

        const matcher = new DA.JointMatcher(flights, events, payload.center_lat, payload.center_lon, opts, clock);
        const res = await matcher.run(payload.radius_km, payload.grid_resolution || 21, {
            sensitivity: payload.sensitivity !== false, onProgress,
        });
        const decision = DA.decideLocation(res, {
            max_rmse_hz: payload.max_rmse_hz || 3.0,
            min_independent_flyovers: payload.min_independent_flyovers || 2,
            precision_target_m: payload.precision_target_m || 1000,
        }, events.length);

        const flyoverFits = {};
        for (const ev of events) {
            const pts = ev.traced_curve || [];
            if (pts.length < 5) continue;
            try { flyoverFits[ev.id] = DA.FlyoverFit.fit(pts.map(p => p.time), pts.map(p => p.freq), c); } catch (e) { /* skip */ }
        }
        if (res.status !== 'ok') {
            return { status: 'no_match', message: res.message, decision, flyover_fits: flyoverFits, mode: 'client_physics', ...provenance };
        }

        const sigma = 3 * opts.loss_scale_hz;
        const scoreOf = (rr) => (isFinite(rr) ? 100 * Math.exp(-0.5 * (rr / sigma) ** 2) : 0);
        const nG = res.n_groups;
        const heatmap = matcher.grid.cells.filter(cell => cell.inside).map(cell => {
            const g = matcher.toGeo(cell.x, cell.y);
            const rr = DA.robustRmse(cell.loss / nG);
            return { lat: +g.lat.toFixed(6), lon: +g.lon.toFixed(6), score: +scoreOf(rr).toFixed(2), rmse: isFinite(rr) ? +rr.toFixed(2) : null };
        });
        const best = res.best;
        const bg = matcher.toGeo(best.x, best.y);
        const diags = res.diagnostics;
        const primary = diags.find(d => d.matched) || diags[0];
        const pk = matcher.models.find(m => m.flight_id === primary.flight_id) || matcher.models[0];
        const comparison = [];
        const eventsOut = diags.map(d => {
            const { times = [], traced = [], modeled = [], ...rest } = d;
            times.forEach((t, i) => comparison.push({
                event_id: d.event_id, time: +t.toFixed(3), traced_freq: +traced[i].toFixed(2),
                modeled_freq: modeled[i] != null ? +modeled[i].toFixed(2) : null,
            }));
            return rest;
        });
        const matchedIds = new Set(diags.filter(d => d.matched).map(d => d.flight_id));
        return {
            status: 'success',
            mode: 'client_physics',
            ...provenance,
            best_flight: { flight_id: pk.flight_id, callsign: pk.raw.callsign, aircraft_type: pk.raw.aircraft_type, data_source: pk.raw.data_source },
            best_location: {
                lat: +bg.lat.toFixed(6), lon: +bg.lon.toFixed(6),
                score: +scoreOf(res.robust_rmse_hz).toFixed(2),
                rmse_hz: +Math.max(...diags.filter(d => d.matched).map(d => d.rmse_hz)).toFixed(3),
                robust_rmse_hz: +res.robust_rmse_hz.toFixed(3),
                time_offset_sec: +(best.start - pk.t[0]).toFixed(3),
                recording_start_epoch: +best.start.toFixed(3),
                uncertainty_radius_m: res.sensitivity ? res.sensitivity.radius_m : null,
            },
            decision,
            alternatives: res.alternatives.map(a => ({
                lat: +a.lat.toFixed(6), lon: +a.lon.toFixed(6), robust_rmse_hz: +a.robust_rmse_hz.toFixed(3),
                distance_from_best_m: +a.distance_from_best_m.toFixed(1), recording_start_epoch: +a.start.toFixed(3),
            })),
            sensitivity: res.sensitivity,
            events: eventsOut,
            flyover_fits: flyoverFits,
            heatmap,
            curve_comparison: comparison,
            flight_trajectory: {
                flight_id: pk.flight_id, callsign: pk.raw.callsign, aircraft_type: pk.raw.aircraft_type,
                data_source: pk.raw.data_source, latitudes: pk.raw.latitudes, longitudes: pk.raw.longitudes, altitudes: pk.raw.altitudes, speeds: pk.raw.speeds,
            },
            matched_flights: matcher.models.filter(m => matchedIds.has(m.flight_id)).map(m => ({
                flight_id: m.flight_id, callsign: m.raw.callsign, aircraft_type: m.raw.aircraft_type,
                data_source: m.raw.data_source, latitudes: m.raw.latitudes, longitudes: m.raw.longitudes,
            })),
            f0_hz: primary.f0_hz,
            evaluated_flights: flights.map(f => ({
                flight_id: f.flight_id, callsign: f.callsign, aircraft_type: f.aircraft_type,
                data_source: f.data_source, altitude_basis: f.altitude_basis,
                matched_events: diags.filter(d => d.flight_id === f.flight_id && d.matched).map(d => d.event_id),
            })),
            grid_resolution: payload.grid_resolution || 21,
            speed_of_sound_mps: c,
            clock_mode: clock ? 'trusted_window' : 'free',
        };
    }

    /**
     * Automatic Doppler ridge discovery. Uses the backend when available
     * (curated sample id or uploaded file), otherwise runs in the browser.
     */
    async discoverRidges({ sampleId = null, savedAs = null, audioBuffer = null, settings = {} } = {}) {
        const hasBackend = await this.checkBackend();
        if (hasBackend && (sampleId || savedAs)) {
            try {
                const form = new FormData();
                if (sampleId) form.append('sample_id', sampleId);
                if (savedAs) form.append('saved_as', savedAs);
                if (settings.fmax_hz) form.append('fmax_hz', settings.fmax_hz);
                const resp = await fetch(`${this.baseUrl}/api/audio/discover`, { method: 'POST', body: form });
                if (resp.ok) return Object.assign(await resp.json(), { mode: 'server' });
            } catch (e) {
                console.warn('Backend discovery failed, using browser discovery:', e);
            }
        }
        if (!audioBuffer) throw new Error('No decoded audio available for discovery.');
        return window.DopplerAnalysis.RidgeDiscovery.discover(audioBuffer, settings);
    }

    async uploadAudio(file) {
        const hasBackend = await this.checkBackend();
        if (hasBackend) {
            const formData = new FormData();
            formData.append('file', file);
            try {
                const resp = await fetch(`${this.baseUrl}/api/audio/upload`, {
                    method: 'POST',
                    body: formData,
                });
                if (resp.ok) return await resp.json();
            } catch (e) {
                // Client-side fallback handled by FileReader
            }
        }
        return { filename: file.name, url: URL.createObjectURL(file) };
    }
}

window.apiClient = new ApiClient();
