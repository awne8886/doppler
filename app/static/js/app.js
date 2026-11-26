/**
 * Main Application Controller for Doppler Acoustic Aircraft Geolocator
 */
document.addEventListener('DOMContentLoaded', async () => {
    const { escapeHtml, fmtNum, toast, downloadFile } = window.DopplerUtil;
    const RT = window.ResultTools;
    let lastResult = null;
    const badge = (ds) => {
        const info = RT.sourceInfo(ds);
        return `<span class="src-badge ${info.kind}" title="${escapeHtml(info.title)}">${escapeHtml(info.label)}</span>`;
    };
    const playBtn = document.getElementById('playBtn');
    const playIcon = document.getElementById('playIcon');
    const stopBtn = document.getElementById('stopBtn');
    const dropZone = document.getElementById('dropZone');
    const audioFileInput = document.getElementById('audioFileInput');
    const scenarioPreset = document.getElementById('scenarioPreset');
    const loadPresetBtn = document.getElementById('loadPresetBtn');
    const searchRadius = document.getElementById('searchRadius');
    const radiusDisplay = document.getElementById('radiusDisplay');
    const searchFlightsBtn = document.getElementById('searchFlightsBtn');
    const geolocateBtn = document.getElementById('geolocateBtn');
    const copyCoordsBtn = document.getElementById('copyCoordsBtn');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const loadingMessage = document.getElementById('loadingMessage');

    let currentScenarioId = 'sfo_dual';
    let currentSavedAs = null;
    let availableScenarios = [];
    const autoRidgeBtn = document.getElementById('autoRidgeBtn');
    const discoveryStatus = document.getElementById('discoveryStatus');
    const clockInput = document.getElementById('clockInput');
    const clockNote = document.getElementById('clockNote');
    const trackStatus = document.getElementById('trackStatus');
    let importedTracks = [];

    // Helper to toggle spinner
    const showLoading = (msg = 'Processing acoustic physics...') => {
        loadingMessage.textContent = msg;
        loadingOverlay.classList.remove('hidden');
    };
    const hideLoading = () => {
        loadingOverlay.classList.add('hidden');
    };

    // 1. Load Scenarios from backend
    try {
        availableScenarios = await window.apiClient.getScenarios();
        if (availableScenarios && availableScenarios.length > 0) {
            scenarioPreset.innerHTML = '';
            for (const sc of availableScenarios) {
                const opt = document.createElement('option');
                opt.value = sc.id;
                opt.textContent = sc.n_flights > 1 ? sc.name : `${sc.name} (${sc.aircraft_type})`;
                if (sc.id === 'sfo_dual') opt.selected = true;
                scenarioPreset.appendChild(opt);
            }
        }
    } catch (e) {
        console.warn('Using default preset scenarios:', e);
    }

    // 2. Scenario loader function
    async function loadScenario(scenarioId) {
        if (window.liveListener && window.liveListener.active) window.liveListener.stop('Live listening stopped: a demo scenario was loaded.');
        currentScenarioId = scenarioId;
        const scenario = availableScenarios.find(s => s.id === scenarioId);

        let centerLat = 37.5585;
        let centerLon = -122.2711;
        let radius = 10;
        let f0 = 840.0;
        let audioUrl = `./samples/${scenarioId}.wav`;

        if (scenario) {
            centerLat = scenario.search_center.lat;
            centerLon = scenario.search_center.lon;
            radius = scenario.recommended_radius_km || 10;
            f0 = scenario.f0_target || 800.0;
            audioUrl = scenario.sample_audio_url || audioUrl;
        } else if (scenarioId === 'lhr_baw282') {
            centerLat = 51.4620; centerLon = -0.3080; f0 = 680.0; radius = 12;
        } else if (scenarioId === 'jfk_dal410') {
            centerLat = 40.6900; centerLon = -73.7800; f0 = 910.0; radius = 10;
        }

        document.getElementById('centerLat').value = centerLat.toFixed(5);
        document.getElementById('centerLon').value = centerLon.toFixed(5);
        searchRadius.value = radius;
        radiusDisplay.textContent = `${radius} km`;
        document.getElementById('f0Input').value = f0.toFixed(1);

        window.mapView.setCenter(centerLat, centerLon, 11);
        window.mapView.updateSearchRadiusCircle(centerLat, centerLon, radius);

        showLoading(`Loading audio & ADS-B flight data for ${scenario ? scenario.name : scenarioId}...`);
        try {
            await window.audioAnalyzer.loadAudioFromUrl(audioUrl, `${scenarioId}.wav`);

            // Also query and render the flight path
            const flightResp = await window.apiClient.searchFlights({
                center_lat: centerLat,
                center_lon: centerLon,
                radius_km: radius,
                scenario_id: scenarioId,
            });

            if (flightResp && flightResp.flights && flightResp.flights.length > 0) {
                window.mapView.displayFlightTrajectories(flightResp.flights);
            }
            clockInput.value = '';
            clockNote.textContent = '';
            clockInput.placeholder = scenario && scenario.recording_start_epoch
                ? `optional, e.g. ${new Date(scenario.recording_start_epoch * 1000).toISOString()}`
                : 'ISO 8601 or Unix epoch';
            resetResults();
            // Detect real Doppler ridges in the recording. If none are found the user
            // traces manually; no synthetic trace is ever substituted.
            await runAutoDetect(true);
        } catch (err) {
            toast(`Error loading scenario: ${err.message}`, 'error');
        } finally {
            hideLoading();
        }
    }

    async function runAutoDetect(quiet = false) {
        if (!window.audioAnalyzer.audioBuffer) {
            if (!quiet) toast('Load an audio recording first.', 'warn');
            return false;
        }
        const wasHidden = loadingOverlay.classList.contains('hidden');
        if (wasHidden) showLoading('Scanning the full recording for moving Doppler ridges...');
        else loadingMessage.textContent = 'Scanning the full recording for moving Doppler ridges...';
        try {
            const res = await window.apiClient.discoverRidges({
                sampleId: currentScenarioId,
                savedAs: currentScenarioId ? null : currentSavedAs,
                audioBuffer: window.audioAnalyzer.audioBuffer,
                settings: { fmax_hz: Math.max(2000, window.audioAnalyzer.maxFreq) },
            });
            const dop = (res.ridges || []).filter(r => r.doppler_like);
            const groups = new Set(dop.map(r => r.harmonic_group_id));
            if (!res.ridges || res.ridges.length === 0) {
                discoveryStatus.classList.add('warn');
                discoveryStatus.innerHTML = `No moving tonal ridge cleared the detection threshold. Trace manually instead.`;
                return false;
            }
            window.tracer.loadDiscoveredRidges(res);
            discoveryStatus.classList.toggle('warn', dop.length === 0);
            discoveryStatus.innerHTML = `Detected <strong>${res.ridges.length}</strong> ridge(s), <strong>${dop.length}</strong> Doppler-like in <strong>${groups.size}</strong> harmonic group(s) `
                + `(${res.mode === 'server' ? 'server' : 'browser'} analysis). Non-Doppler ridges are excluded by default; tick to include. `
                + `<em>Ridges are acoustic proposals, not verified aircraft.</em>`;
            return dop.length > 0;
        } catch (err) {
            discoveryStatus.classList.add('warn');
            discoveryStatus.textContent = `Ridge discovery failed: ${err.message}`;
            return false;
        } finally {
            if (wasHidden) hideLoading();
        }
    }

    autoRidgeBtn.addEventListener('click', () => runAutoDetect(false));

    function parseClock(text) {
        const v = (text || '').trim();
        if (!v) return null;
        if (/^-?\d+(\.\d+)?$/.test(v)) {
            const n = parseFloat(v);
            return n > 1e12 ? n / 1000 : n;
        }
        const iso = /[zZ]|[+-]\d\d:?\d\d$/.test(v) ? v : `${v}Z`;
        const ms = Date.parse(iso);
        if (isNaN(ms)) throw new Error(`Could not parse recording start "${v}". Use ISO 8601 UTC or a Unix epoch.`);
        return ms / 1000;
    }

    function resetResults() {
        const banner = document.getElementById('decisionBanner');
        banner.className = 'decision-banner pending';
        document.getElementById('decisionStatus').textContent = 'AWAITING ANALYSIS';
        document.getElementById('decisionMsg').textContent = 'A ranked best cell is a hypothesis; evidence checks decide whether it is a supported location.';
        document.getElementById('eventResults').innerHTML = '';
    }

    // 3. Audio Controls
    playBtn.addEventListener('click', () => {
        window.audioAnalyzer.initContext();
        const audio = window.audioAnalyzer.audioElement;
        if (audio.paused) {
            audio.play();
            playIcon.innerHTML = '&#10074;&#10074;';
        } else {
            audio.pause();
            playIcon.innerHTML = '&#9658;';
        }
    });

    stopBtn.addEventListener('click', () => {
        const audio = window.audioAnalyzer.audioElement;
        audio.pause();
        audio.currentTime = 0;
        playIcon.innerHTML = '&#9658;';
        window.audioAnalyzer.playheadLine.style.display = 'none';
    });

    // 4. File Drop / Upload
    dropZone.addEventListener('click', () => audioFileInput.click());
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', async (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });
    audioFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    async function handleFileUpload(file) {
        if (window.liveListener && window.liveListener.active) window.liveListener.stop('Live listening stopped: an audio file was loaded.');
        showLoading(`Decoding audio file ${file.name}...`);
        try {
            await window.audioAnalyzer.loadAudioFromFile(file);
            currentScenarioId = null; // Custom uploaded audio
            currentSavedAs = null;
            resetResults();
            // Upload to server cache (enables server-side discovery when a backend is running)
            try {
                const up = await window.apiClient.uploadAudio(file);
                currentSavedAs = up && up.saved_as ? up.saved_as : null;
                applyRecordingMetadata(up && up.recording_metadata);
            } catch (e) {
                console.warn('Server audio cache notice:', e);
            }
            discoveryStatus.classList.remove('warn');
            discoveryStatus.innerHTML = 'Audio loaded. Click <strong>Auto-Detect Ridges</strong> or trace manually.';
        } catch (err) {
            toast(`Failed to load audio: ${err.message}`, 'error');
        } finally {
            hideLoading();
        }
    }

    /** Prefill the clock from file metadata. It stays a suggestion the user must confirm. */
    function applyRecordingMetadata(meta) {
        clockNote.textContent = '';
        if (!meta || meta.start_epoch == null) {
            clockInput.value = '';
            if (meta && meta.notes && meta.notes.length) clockNote.textContent = meta.notes[0];
            return;
        }
        clockInput.value = new Date(meta.start_epoch * 1000).toISOString();
        const tz = meta.timezone_known ? 'UTC per file metadata' : 'time zone unknown, read as UTC';
        clockNote.textContent = `Prefilled from ${meta.source} (${tz}). Confirm before relying on it. ${(meta.notes || [])[0] || ''}`;
    }

    // 4a. Settings + trace sessions
    const SETTING_IDS = ['centerLat', 'centerLon', 'searchRadius', 'gridResInput', 'clockInput', 'clockTolInput', 'tempInput',
        'humidityInput', 'windSpeedInput', 'windFromInput', 'qnhInput', 'icaoHintsInput', 'f0Input'];
    const SETTING_CHECKS = ['fitF0Check', 'sensitivityCheck', 'demoCheck'];
    function collectSettings() {
        const out = {};
        for (const id of SETTING_IDS) out[id] = document.getElementById(id).value;
        for (const id of SETTING_CHECKS) out[id] = document.getElementById(id).checked;
        out.scenario_id = currentScenarioId;
        return out;
    }
    function applySettings(st) {
        for (const id of SETTING_IDS) if (st[id] != null) document.getElementById(id).value = String(st[id]);
        for (const id of SETTING_CHECKS) if (st[id] != null) document.getElementById(id).checked = !!st[id];
        radiusDisplay.textContent = `${searchRadius.value} km`;
        const lat = parseFloat(document.getElementById('centerLat').value), lon = parseFloat(document.getElementById('centerLon').value);
        if (Number.isFinite(lat) && Number.isFinite(lon)) window.mapView.updateSearchRadiusCircle(lat, lon, parseFloat(searchRadius.value));
    }
    document.getElementById('saveSessionBtn').addEventListener('click', () => {
        const session = RT.sessionToJSON(window.tracer.events.filter(e => e.points.length), collectSettings(), {
            audio: { filename: document.getElementById('audioFileName').textContent, duration_s: window.audioAnalyzer.duration || null },
            tracks: importedTracks,
        });
        downloadFile(`doppler-session-${new Date().toISOString().replace(/[:.]/g, '-')}.json`, session);
    });
    const sessionInput = document.getElementById('sessionFileInput');
    document.getElementById('loadSessionBtn').addEventListener('click', () => sessionInput.click());
    sessionInput.addEventListener('change', async () => {
        const file = sessionInput.files && sessionInput.files[0];
        sessionInput.value = '';
        if (!file) return;
        try {
            const session = RT.parseSession(JSON.parse(await file.text()));
            applySettings(session.settings);
            if (session.settings.scenario_id !== undefined) currentScenarioId = session.settings.scenario_id;
            window.tracer.loadSessionEvents(session.events);
            importedTracks = session.tracks || [];
            trackStatus.textContent = importedTracks.length ? `${importedTracks.length} uploaded track(s) restored from the session.` : 'No tracks imported.';
            if (importedTracks.length) window.mapView.displayFlightTrajectories(importedTracks);
            const audioNote = session.audio && session.audio.filename ? ` Recorded against "${session.audio.filename}" - load the same audio to see it on the spectrogram.` : '';
            toast(`Session loaded: ${session.events.length} event(s).${audioNote}`, 'ok', 9000);
        } catch (err) {
            toast(`Could not load session: ${err.message}`, 'error');
        }
    });

    // 4b. Real-track import
    document.getElementById('importTracksBtn').addEventListener('click', async () => {
        const files = [...document.getElementById('trackFileInput').files];
        if (!files.length) { toast('Choose one or more track files first.', 'warn'); return; }
        const options = {
            altitude_unit: document.getElementById('trackAltUnit').value,
            timezone: document.getElementById('trackTz').value.trim() || null,
        };
        showLoading('Importing and validating aircraft tracks...');
        const warnings = [];
        try {
            for (const f of files) {
                const res = await window.apiClient.importTracks(f, options);
                importedTracks = importedTracks.concat(res.flights || []);
                warnings.push(...(res.warnings || []));
            }
            window.mapView.displayFlightTrajectories(importedTracks);
            trackStatus.textContent = `${importedTracks.length} uploaded track(s) will be used instead of live/demo ADS-B.`
                + (warnings.length ? ` Warnings: ${warnings.join(' ')}` : '');
            toast(`Imported ${importedTracks.length} track(s).`, warnings.length ? 'warn' : 'ok');
        } catch (err) {
            trackStatus.textContent = `Import failed: ${err.message}`;
        } finally {
            hideLoading();
        }
    });
    document.getElementById('clearTracksBtn').addEventListener('click', () => {
        importedTracks = [];
        trackStatus.textContent = 'No tracks imported.';
    });

    // 5. Preset selection
    loadPresetBtn.addEventListener('click', () => {
        const val = scenarioPreset.value;
        if (val) loadScenario(val);
    });

    // 6. Search Radius Slider
    searchRadius.addEventListener('input', (e) => {
        const r = parseFloat(e.target.value);
        radiusDisplay.textContent = `${r} km`;
        const lat = parseFloat(document.getElementById('centerLat').value);
        const lon = parseFloat(document.getElementById('centerLon').value);
        window.mapView.updateSearchRadiusCircle(lat, lon, r);
    });

    // 7. Query Flights Button
    searchFlightsBtn.addEventListener('click', async () => {
        const lat = parseFloat(document.getElementById('centerLat').value);
        const lon = parseFloat(document.getElementById('centerLon').value);
        const r = parseFloat(searchRadius.value);

        showLoading('Querying ADS-B flights in geographic search sector...');
        try {
            const resp = await window.apiClient.searchFlights({
                center_lat: lat,
                center_lon: lon,
                radius_km: r,
                scenario_id: currentScenarioId,
                demo: document.getElementById('demoCheck').checked,
            });

            if (resp && resp.flights && resp.flights.length > 0) {
                window.mapView.displayFlightTrajectories(resp.flights);
                document.getElementById('mapOverlayBadge').innerHTML = `
                    <span class="status-dot"></span> Found ${fmtNum(resp.count)} flight(s) in sector ${badge(resp.data_source)}
                `;
                if (resp.limitations && resp.limitations.length) toast(resp.limitations.join(' '), 'info', 10000);
            } else {
                toast((resp && resp.notes && resp.notes.length) ? resp.notes[resp.notes.length - 1] : 'No flight trajectories found in this sector.', 'warn', 12000);
            }
        } catch (e) {
            toast(`Flight query error: ${e.message}`, 'error');
        } finally {
            hideLoading();
        }
    });

    /** Matching options and atmosphere inputs shared by manual and live geolocation. */
    function basePayload() {
        const payload = {
            fit_f0: document.getElementById('fitF0Check').checked,
            grid_resolution: parseInt(document.getElementById('gridResInput').value, 10),
            sensitivity: document.getElementById('sensitivityCheck').checked,
        };
        const tempVal = document.getElementById('tempInput').value;
        const humidityVal = document.getElementById('humidityInput').value;
        if (tempVal !== '') payload.temperature_c = parseFloat(tempVal);
        if (humidityVal !== '') payload.relative_humidity_pct = parseFloat(humidityVal);
        const windSpeed = document.getElementById('windSpeedInput').value;
        const windFrom = document.getElementById('windFromInput').value;
        const qnh = document.getElementById('qnhInput').value;
        if (windSpeed !== '' && parseFloat(windSpeed) > 0) {
            payload.wind_speed_mps = parseFloat(windSpeed);
            payload.wind_from_deg = windFrom !== '' ? parseFloat(windFrom) : 0;
        }
        if (qnh !== '') payload.qnh_hpa = parseFloat(qnh);
        return payload;
    }

    // 8. Main Geolocate Button
    geolocateBtn.addEventListener('click', async () => {
        const events = window.tracer.getEvents();
        if (events.length === 0) {
            toast('Trace or auto-detect at least one Doppler event (3+ points) and make sure it is ticked.', 'warn');
            return;
        }
        const fitF0 = document.getElementById('fitF0Check').checked;
        const f0 = parseFloat(document.getElementById('f0Input').value);
        if (!fitF0 && (!f0 || f0 <= 0)) {
            toast('Please specify a valid fundamental frequency f0, or enable "Fit f0 per event".', 'warn');
            return;
        }
        let clock = null;
        try { clock = parseClock(clockInput.value); } catch (e) { toast(e.message, 'warn'); return; }
        const lat = parseFloat(document.getElementById('centerLat').value);
        const lon = parseFloat(document.getElementById('centerLon').value);
        const radius = parseFloat(searchRadius.value);
        const gridRes = parseInt(document.getElementById('gridResInput').value, 10);

        showLoading(`Jointly matching ${events.length} event(s) across ${gridRes * gridRes} candidate positions (shared clock)...`);
        try {
            const payload = Object.assign(basePayload(), {
                center_lat: lat,
                center_lon: lon,
                radius_km: radius,
                events: events.map(e => fitF0 ? e : Object.assign({}, e, { f0 })),
                f0: fitF0 ? undefined : f0,
                scenario_id: currentScenarioId,
                demo: document.getElementById('demoCheck').checked,
            });
            if (importedTracks.length) {
                payload.trajectories = importedTracks;
                payload.scenario_id = null;
            }
            const hints = document.getElementById('icaoHintsInput').value.split(/[\s,;]+/).filter(Boolean);
            if (hints.length) payload.icao24_hints = hints;
            if (clock !== null) {
                payload.recording_start_epoch = clock;
                payload.clock_tolerance_s = parseFloat(document.getElementById('clockTolInput').value) || 0;
            }

            const result = await window.apiClient.geolocateObserver(payload, (stage, frac) => {
                loadingMessage.textContent = `Matching (${stage}): ${Math.round(frac * 100)}%`;
            });
            displayGeolocationResults(result);
        } catch (err) {
            toast(`Geolocation error: ${err.message}`, 'error', 12000);
        } finally {
            hideLoading();
        }
    });

    function displayDecision(decision) {
        const banner = document.getElementById('decisionBanner');
        const status = decision ? decision.status : 'inconclusive';
        banner.className = `decision-banner ${status}`;
        document.getElementById('decisionStatus').textContent = decision
            ? `${status.toUpperCase()} \u2022 ${decision.reason_code}` : 'INCONCLUSIVE';
        document.getElementById('decisionMsg').textContent = decision
            ? `${decision.message}${status === 'supported' ? '' : ' The best-ranked cell is shown as a hypothesis only.'}`
            : '';
    }

    function displayEventTable(result) {
        const host = document.getElementById('eventResults');
        const rows = (result.events || []).map(e => {
            const fit = (result.flyover_fits || {})[e.event_id];
            const bad = (v, lim) => (v > lim ? ' class="bad"' : '');
            const pct = (v, d) => (v != null ? `${fmtNum(v * 100, d)}%` : '--');
            return `<tr>
                <td>${escapeHtml(e.event_id)}</td><td>${escapeHtml(e.group_id || '')}</td>
                <td>${e.matched ? escapeHtml(e.callsign || e.flight_id) : '<span class="bad">unmatched</span>'}</td>
                <td>${fmtNum(e.f0_hz, 1)}</td>
                <td${bad(e.rmse_hz, 3)}>${fmtNum(e.rmse_hz, 2)}</td>
                <td${bad(e.outlier_fraction, 0.25)}>${pct(e.outlier_fraction, 0)}</td>
                <td${e.support_fraction < 0.95 ? ' class="bad"' : ''}>${pct(e.support_fraction, 0)}</td>
                <td${e.drift_at_bound ? ' class="bad"' : ''}>${pct(e.drift_fraction, 2)}</td>
                <td>${fit ? `${fmtNum(fit.speed_mps)} m/s, ${fmtNum(fit.cpa_slant_m / 1000, 2)} km` : '--'}</td>
            </tr>`;
        }).join('');
        const checks = ((result.decision && result.decision.checks) || []).map(c =>
            `<span class="check-pill ${c.passed ? 'pass' : 'fail'}">${c.passed ? '\u2713' : '\u2717'} ${escapeHtml(String(c.check).replace(/_/g, ' '))}</span>`).join('');
        const flights = (result.evaluated_flights || []).map(f => {
            const kin = Object.entries(f.kinematic_consistency || {}).map(([eid, k]) =>
                `<span class="${k.compatible ? 'kin-ok' : 'kin-bad'}" title="fitted ${fmtNum(k.fitted_speed_mps)} m/s vs flight ${fmtNum(k.flight_speed_mps)} m/s; CPA slant ${fmtNum(k.fitted_cpa_slant_m)} m vs min height ${fmtNum(k.min_height_above_observer_m)} m">${escapeHtml(eid)} ${k.compatible ? '\u2713' : '\u2717'}</span>`).join(' ');
            const reasons = (f.discard_reasons || []).map(g => `${escapeHtml(g.group_id)}: ${g.reasons.map(r => escapeHtml(String(r.reason).replace(/_/g, ' '))).join(', ')}`).join('; ');
            return `<tr>
                <td>${escapeHtml(f.callsign || f.flight_id)} ${badge(f.data_source)}</td>
                <td>${escapeHtml(f.altitude_basis || 'unknown')}</td>
                <td>${(f.matched_events || []).map(escapeHtml).join(', ') || '--'}</td>
                <td>${kin || '--'}</td>
                <td class="${f.discarded ? 'reasons' : ''}">${f.discarded ? `discarded (${reasons})` : 'kept'}</td>
            </tr>`;
        }).join('');
        const notes = [...(result.data_limitations || []), ...((result.data_notes || []).slice(-2))]
            .map(n => `<div>${escapeHtml(n)}</div>`).join('');
        host.innerHTML = `
            <table>
                <thead><tr><th>Event</th><th>Group</th><th>Flight</th><th>f&#8320; Hz</th><th>RMSE Hz</th><th>Outliers</th><th>Support</th><th>Drift</th><th>Flyover fit</th></tr></thead>
                <tbody>${rows}</tbody>
            </table>
            <div class="checks">${checks}</div>
            <table class="flight-table">
                <thead><tr><th>Candidate flight</th><th>Altitude</th><th>Matched</th><th>Kinematics (fit vs ADS-B)</th><th>Pre-filter</th></tr></thead>
                <tbody>${flights}</tbody>
            </table>
            <div class="data-notes">${notes}</div>
            <div class="result-actions">
                <button id="exportJsonBtn" class="btn btn-sm">Export JSON</button>
                <button id="exportGeojsonBtn" class="btn btn-sm">Export GeoJSON</button>
            </div>`;
        document.getElementById('exportJsonBtn').addEventListener('click', () => exportResult('json'));
        document.getElementById('exportGeojsonBtn').addEventListener('click', () => exportResult('geojson'));
    }

    function exportResult(kind) {
        if (!lastResult) { toast('Run a geolocation first.', 'warn'); return; }
        const stamp = new Date().toISOString().replace(/[:.]/g, '-');
        if (kind === 'geojson') downloadFile(`doppler-result-${stamp}.geojson`, RT.toGeoJSON(lastResult), 'application/geo+json');
        else downloadFile(`doppler-result-${stamp}.json`, RT.toExportJSON(lastResult, { settings: collectSettings() }));
    }

    function displayGeolocationResults(result, { quiet = false } = {}) {
        lastResult = result;
        if (result.status !== 'success') {
            displayDecision(result.decision);
            const discarded = (result.discarded_flights || []).map(d => `${escapeHtml(d.callsign || d.flight_id)} (${d.groups.map(g => g.reasons.map(r => escapeHtml(r.reason)).join(', ')).join('; ')})`).join(', ');
            document.getElementById('eventResults').innerHTML = `<p class="small-text">${escapeHtml(result.message || 'No match.')}</p>`
                + (discarded ? `<p class="small-text">Discarded by the pre-filter: ${discarded}</p>` : '');
            window.mapView.clearAmbiguity();
            if (result.status === 'no_adsb_data' && !quiet) toast('No real ADS-B data for this area/time. Import tracks for the recording, or tick "Allow demo / synthetic flights".', 'warn', 12000);
            return;
        }
        const best = result.best_location;
        const flight = result.best_flight;
        const decision = result.decision;
        displayDecision(decision);
        displayEventTable(result);

        const matchedNames = (result.matched_flights || []).map(f => f.callsign || f.flight_id);
        document.getElementById('matchScoreNum').textContent = best.score.toFixed(1);
        document.getElementById('resLatLon').textContent = `${best.lat.toFixed(5)}, ${best.lon.toFixed(5)}`;
        document.getElementById('resFlightId').textContent = matchedNames.length ? matchedNames.join(' + ') : `${flight.callsign || flight.flight_id}`;
        document.getElementById('resAircraftType').textContent = [...new Set((result.events || []).filter(e => e.matched).map(e => e.aircraft_type))].join(', ') || flight.aircraft_type || '--';
        document.getElementById('resRmse').textContent = `${best.rmse_hz.toFixed(2)} Hz RMSE`;
        document.getElementById('resTimeShift').textContent = best.recording_start_epoch
            ? `${new Date(best.recording_start_epoch * 1000).toISOString().replace('.000', '')} (${result.clock_mode === 'trusted_window' ? 'trusted window' : 'fitted'})`
            : `${best.time_offset_sec.toFixed(2)} s`;
        document.getElementById('resUncertainty').textContent = best.uncertainty_radius_m != null
            ? `${Math.round(best.uncertainty_radius_m)} m (uncalibrated)` : 'not assessed';
        const ties = ((decision && decision.checks) || []).find(c => c.check === 'ambiguity');
        const nTies = ties && ties.near_tied_alternatives ? ties.near_tied_alternatives.length : 0;
        document.getElementById('resAlternatives').textContent = `${(result.alternatives || []).length} (${nTies} near-tied)`;
        document.getElementById('resTotalTested').textContent = `${result.heatmap.length} points`;

        window.mapView.displayHeatmap(result.heatmap, best, flight, { alternatives: result.alternatives, decision });
        if (result.matched_flights && result.matched_flights.length) {
            window.mapView.displayFlightTrajectories(result.matched_flights);
        } else if (result.flight_trajectory) {
            window.mapView.displayFlightTrajectory(result.flight_trajectory);
        }
        window.mapView.renderComparisonChart(result.curve_comparison, result.f0_hz);
        if (RT.isSingleFlyover(result) && (result.matched_flights || []).length) {
            window.mapView.displayAmbiguity(best, RT.mirrorAmbiguity(best, result.matched_flights[0]), result.clock_mode !== 'trusted_window');
        } else {
            window.mapView.clearAmbiguity();
        }

        const supported = decision && decision.status === 'supported';
        const dotColor = supported ? '#10b981' : '#f59e0b';
        document.getElementById('mapOverlayBadge').innerHTML = `
            <span class="status-dot" style="background:${dotColor}; box-shadow:0 0 8px ${dotColor};"></span>
            ${supported ? 'SUPPORTED' : 'INCONCLUSIVE'}: ${fmtNum(best.lat, 4)}, ${fmtNum(best.lon, 4)} ${badge(result.data_source)}
        `;

        if (!quiet && window.innerWidth < 1200) {
            document.getElementById('resultsCard').scrollIntoView({ behavior: 'smooth' });
        }
    }

    // 9. Copy Coordinates Button
    copyCoordsBtn.addEventListener('click', () => {
        const txt = document.getElementById('resLatLon').textContent;
        if (txt && txt !== '--') {
            navigator.clipboard.writeText(txt).then(() => {
                const orig = copyCoordsBtn.innerHTML;
                copyCoordsBtn.innerHTML = '&#10003; Copied!';
                setTimeout(() => { copyCoordsBtn.innerHTML = orig; }, 1800);
            });
        }
    });

    // 10. Hooks for live microphone listening (live_listener.js)
    window.DopplerApp = {
        basePayload,
        displayGeolocationResults,
        resetResults,
        getImportedTracks: () => importedTracks,
        getSearchArea: () => ({
            lat: parseFloat(document.getElementById('centerLat').value),
            lon: parseFloat(document.getElementById('centerLon').value),
            radius_km: parseFloat(searchRadius.value),
        }),
        setSearchCenter(lat, lon) {
            document.getElementById('centerLat').value = lat.toFixed(5);
            document.getElementById('centerLon').value = lon.toFixed(5);
            window.mapView.updateSearchRadiusCircle(lat, lon, parseFloat(searchRadius.value));
        },
        clockToleranceS: () => {
            const v = parseFloat(document.getElementById('clockTolInput').value);
            return Number.isFinite(v) && v >= 0 ? v : 2;
        },
        enterLiveMode() {
            currentScenarioId = null;
            currentSavedAs = null;
            clockInput.value = '';
            clockNote.textContent = 'Live mode time-stamps the microphone audio itself.';
        },
    };
    if (window.liveListener) window.liveListener.init();

    // Initial default scenario load: SFO
    await loadScenario('sfo_dual');
});
