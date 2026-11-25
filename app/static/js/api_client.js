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
                return await this.serverGeolocate(payload, onProgress);
            } catch (e) {
                if (e && e.clientError) throw new Error(e.message);
                console.warn('Backend geolocation failed, falling back to client physics:', e);
            }
        }
        return this.clientGeolocate(payload, onProgress);
    }

    /**
     * Server geolocation via the async jobs API (progress polling), falling
     * back to the synchronous endpoint on servers without /api/jobs.
     */
    async serverGeolocate(payload, onProgress) {
        const post = (url) => fetch(`${this.baseUrl}${url}`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
        });
        const fail = async (resp) => {
            const err = await resp.json().catch(() => ({}));
            const detail = typeof err.detail === 'string' ? err.detail : `HTTP ${resp.status}`;
            if (resp.status >= 400 && resp.status < 500) throw { clientError: true, message: detail };
            throw new Error(detail);
        };
        const sub = await post('/api/jobs/geolocate');
        if (sub.status === 404 || sub.status === 405) {
            const resp = await post('/api/geolocate');
            if (!resp.ok) await fail(resp);
            return resp.json();
        }
        if (!sub.ok) await fail(sub);
        const { job_id: jobId } = await sub.json();
        for (;;) {
            await new Promise(r => setTimeout(r, 400));
            const resp = await fetch(`${this.baseUrl}/api/jobs/${encodeURIComponent(jobId)}`);
            if (!resp.ok) await fail(resp);
            const job = await resp.json();
            if (onProgress && job.status === 'running') onProgress(`server ${job.stage}`, job.progress);
            if (job.status === 'done') return job.result;
            if (job.status === 'error') {
                const e = job.error || {};
                if (e.status_code >= 400 && e.status_code < 500) throw { clientError: true, message: e.detail };
                throw new Error(e.detail || 'Server geolocation failed.');
            }
            if (job.status === 'cancelled') throw { clientError: true, message: 'Geolocation job was cancelled.' };
        }
    }

    /** Full joint-matching pipeline in the browser (GitHub Pages mode), in a Web Worker when possible. */
    async clientGeolocate(payload, onProgress) {
        const DA = window.DopplerAnalysis;
        const events = (payload.events && payload.events.length)
            ? payload.events
            : [{ id: 'E1', traced_curve: payload.traced_curve, f0: payload.f0 }];
        let found;
        if (payload.trajectories && payload.trajectories.length) {
            found = { status: 'ok', data_source: 'uploaded', notes: [`${payload.trajectories.length} uploaded track(s) supplied with the request.`], limitations: [],
                flights: payload.trajectories.map(t => Object.assign({}, t, { data_source: 'uploaded' })) };
        } else {
            found = window.clientFlights.findFlights(
                payload.center_lat, payload.center_lon, payload.radius_km, payload.scenario_id,
                { demo: !!payload.demo, timeWindowStart: payload.time_window_start }
            );
        }
        const args = { payload: Object.assign({}, payload, { events }), found };
        try {
            return await this.runInWorker('geolocate', args, onProgress);
        } catch (e) {
            if (!e || !e.workerUnavailable) throw e;
            return DA.runPipeline(args.payload, found, onProgress);
        }
    }

    /**
     * Run analysis in a dedicated Web Worker so the UI stays responsive.
     * Rejects with {workerUnavailable: true} when workers cannot be used
     * (e.g. file:// pages), so callers can fall back to the main thread.
     */
    runInWorker(type, args, onProgress = null, transfer = []) {
        if (this.workerBroken || typeof Worker === 'undefined') return Promise.reject({ workerUnavailable: true });
        if (!this.worker) {
            try {
                this.worker = new Worker(new URL('js/analysis_worker.js', document.baseURI));
                this.workerSeq = 0;
                this.workerCalls = new Map();
                this.worker.onmessage = (e) => {
                    const { id, kind } = e.data;
                    const call = this.workerCalls.get(id);
                    if (!call) return;
                    if (kind === 'progress') { if (call.onProgress) call.onProgress(e.data.stage, e.data.frac); return; }
                    this.workerCalls.delete(id);
                    if (kind === 'result') call.resolve(e.data.result);
                    else call.reject(new Error(e.data.message || 'Worker error'));
                };
                this.worker.onerror = (err) => {
                    console.warn('Analysis worker failed; using the main thread.', err.message || err);
                    this.workerBroken = true;
                    for (const call of this.workerCalls.values()) call.reject({ workerUnavailable: true });
                    this.workerCalls.clear();
                    try { this.worker.terminate(); } catch (e2) { /* ignore */ }
                    this.worker = null;
                };
            } catch (e) {
                this.workerBroken = true;
                return Promise.reject({ workerUnavailable: true });
            }
        }
        const id = ++this.workerSeq;
        return new Promise((resolve, reject) => {
            this.workerCalls.set(id, { resolve, reject, onProgress });
            this.worker.postMessage({ id, type, args }, transfer);
        });
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
        return window.DopplerAnalysis.RidgeDiscovery.discover(audioBuffer, settings,
            (type, args, onProgress, transfer) => this.runInWorker(type, args, onProgress, transfer));
    }

    /**
     * Import aircraft tracks (CSV / FR24 CSV / KML / GeoJSON / readsb trace).
     * Uses the server when available, otherwise parses in the browser.
     */
    async importTracks(file, options = {}) {
        const hasBackend = await this.checkBackend();
        if (hasBackend) {
            const form = new FormData();
            form.append('file', file);
            for (const [k, v] of Object.entries(options)) if (v != null && v !== '') form.append(k, v);
            const resp = await fetch(`${this.baseUrl}/api/flights/import`, { method: 'POST', body: form });
            const body = await resp.json().catch(() => ({}));
            if (!resp.ok) throw new Error(body.detail || `HTTP ${resp.status}`);
            return Object.assign(body, { mode: 'server' });
        }
        const res = await window.TrackImport.parseTrackFile(file, options.format || null, options);
        return Object.assign(res, { mode: 'client' });
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
        let meta = null;
        if (/\.wav$/i.test(file.name || '')) {
            try { meta = window.DopplerUtil.readWavStart(await file.arrayBuffer()); } catch (e) { meta = null; }
        }
        return { filename: file.name, url: URL.createObjectURL(file),
            recording_metadata: meta || { start_epoch: null, source: null, timezone_known: false, requires_confirmation: true,
                notes: ['No recording start time found in the file metadata (the browser reads Broadcast-WAV bext/iXML only).'] } };
    }
}

window.apiClient = new ApiClient();
