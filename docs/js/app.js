/**
 * Main Application Controller for Doppler Acoustic Aircraft Geolocator
 */
document.addEventListener('DOMContentLoaded', async () => {
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
            clockInput.placeholder = scenario && scenario.recording_start_epoch
                ? `optional, e.g. ${new Date(scenario.recording_start_epoch * 1000).toISOString()}`
                : 'ISO 8601 or Unix epoch';
            resetResults();
            // Detect real Doppler ridges in the recording (falls back to a synthetic seed).
            const found = await runAutoDetect(true);
            if (!found && flightResp && flightResp.flights && flightResp.flights.length > 0) {
                autoSeedInitialTrace(flightResp.flights[0], f0);
            }
        } catch (err) {
            alert(`Error loading scenario: ${err.message}`);
        } finally {
            hideLoading();
        }
    }

    // Auto-seed trace points along the ground-truth curve for immediate demo
    function autoSeedInitialTrace(flight, f0) {
        if (!window.audioAnalyzer.duration) return;
        const dur = window.audioAnalyzer.duration;
        const cpaTime = dur / 2.0;

        // Realistic S-curve equation: f(t) = f0 - delta * tanh((t - t_cpa) / tau)
        const delta = (flight.speeds ? flight.speeds[0] : 170.0) / 343.2 * f0 * 0.9;
        const tau = 4.2;

        const points = [];
        const nPts = 16;
        for (let i = 0; i < nPts; i++) {
            const t = 2.0 + (i / (nPts - 1)) * (dur - 4.0);
            // S-curve frequency drop
            const f = f0 - delta * Math.tanh((t - cpaTime) / tau);
            points.push({ time: t, freq: f });
        }
        window.tracer.loadSynthesizedPoints(points, f0);
    }

    async function runAutoDetect(quiet = false) {
        if (!window.audioAnalyzer.audioBuffer) {
            if (!quiet) alert('Load an audio recording first.');
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
            } catch (e) {
                console.warn('Server audio cache notice:', e);
            }
            discoveryStatus.classList.remove('warn');
            discoveryStatus.innerHTML = 'Audio loaded. Click <strong>Auto-Detect Ridges</strong> or trace manually.';
        } catch (err) {
            alert(`Failed to load audio: ${err.message}`);
        } finally {
            hideLoading();
        }
    }

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
            });

            if (resp && resp.flights && resp.flights.length > 0) {
                window.mapView.displayFlightTrajectories(resp.flights);
                document.getElementById('mapOverlayBadge').innerHTML = `
                    <span class="status-dot"></span> Found ${resp.count} flight(s) in sector
                `;
            } else {
                alert('No flight trajectories found in this sector.');
            }
        } catch (e) {
            alert(`Flight query error: ${e.message}`);
        } finally {
            hideLoading();
        }
    });

    // 8. Main Geolocate Button
    geolocateBtn.addEventListener('click', async () => {
        const events = window.tracer.getEvents();
        if (events.length === 0) {
            alert('Trace or auto-detect at least one Doppler event (3+ points) and make sure it is ticked.');
            return;
        }
        const fitF0 = document.getElementById('fitF0Check').checked;
        const f0 = parseFloat(document.getElementById('f0Input').value);
        if (!fitF0 && (!f0 || f0 <= 0)) {
            alert('Please specify a valid fundamental frequency f0, or enable "Fit f0 per event".');
            return;
        }
        let clock = null;
        try { clock = parseClock(clockInput.value); } catch (e) { alert(e.message); return; }
        const tempVal = document.getElementById('tempInput').value;

        const lat = parseFloat(document.getElementById('centerLat').value);
        const lon = parseFloat(document.getElementById('centerLon').value);
        const radius = parseFloat(searchRadius.value);
        const gridRes = parseInt(document.getElementById('gridResInput').value, 10);

        showLoading(`Jointly matching ${events.length} event(s) across ${gridRes * gridRes} candidate positions (shared clock)...`);
        try {
            const payload = {
                center_lat: lat,
                center_lon: lon,
                radius_km: radius,
                events: events.map(e => fitF0 ? e : Object.assign({}, e, { f0 })),
                f0: fitF0 ? undefined : f0,
                fit_f0: fitF0,
                scenario_id: currentScenarioId,
                grid_resolution: gridRes,
                sensitivity: document.getElementById('sensitivityCheck').checked,
            };
            if (clock !== null) {
                payload.recording_start_epoch = clock;
                payload.clock_tolerance_s = parseFloat(document.getElementById('clockTolInput').value) || 0;
            }
            if (tempVal !== '') payload.temperature_c = parseFloat(tempVal);

            const result = await window.apiClient.geolocateObserver(payload, (stage, frac) => {
                loadingMessage.textContent = `Client physics: ${stage} ${Math.round(frac * 100)}%`;
            });
            displayGeolocationResults(result);
        } catch (err) {
            alert(`Geolocation error: ${err.message}`);
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
            return `<tr>
                <td>${e.event_id}</td><td>${e.group_id || ''}</td>
                <td>${e.matched ? (e.callsign || e.flight_id) : '<span class="bad">unmatched</span>'}</td>
                <td>${e.f0_hz != null ? e.f0_hz.toFixed(1) : '--'}</td>
                <td${bad(e.rmse_hz, 3)}>${e.rmse_hz != null ? e.rmse_hz.toFixed(2) : '--'}</td>
                <td${bad(e.outlier_fraction, 0.25)}>${e.outlier_fraction != null ? (e.outlier_fraction * 100).toFixed(0) + '%' : '--'}</td>
                <td${e.support_fraction < 0.95 ? ' class="bad"' : ''}>${e.support_fraction != null ? (e.support_fraction * 100).toFixed(0) + '%' : '--'}</td>
                <td${e.drift_at_bound ? ' class="bad"' : ''}>${e.drift_fraction != null ? (e.drift_fraction * 100).toFixed(2) + '%' : '--'}</td>
                <td>${fit ? `${Math.round(fit.speed_mps)} m/s, ${(fit.cpa_slant_m / 1000).toFixed(2)} km` : '--'}</td>
            </tr>`;
        }).join('');
        const checks = ((result.decision && result.decision.checks) || []).map(c =>
            `<span class="check-pill ${c.passed ? 'pass' : 'fail'}">${c.passed ? '\u2713' : '\u2717'} ${c.check.replace(/_/g, ' ')}</span>`).join('');
        host.innerHTML = `
            <table>
                <thead><tr><th>Event</th><th>Group</th><th>Flight</th><th>f&#8320; Hz</th><th>RMSE Hz</th><th>Outliers</th><th>Support</th><th>Drift</th><th>Flyover fit</th></tr></thead>
                <tbody>${rows}</tbody>
            </table>
            <div class="checks">${checks}</div>`;
    }

    function displayGeolocationResults(result) {
        if (result.status !== 'success') {
            displayDecision(result.decision);
            document.getElementById('eventResults').innerHTML = `<p class="small-text">${result.message || 'No match.'}</p>`;
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

        const supported = decision && decision.status === 'supported';
        const dotColor = supported ? '#10b981' : '#f59e0b';
        document.getElementById('mapOverlayBadge').innerHTML = `
            <span class="status-dot" style="background:${dotColor}; box-shadow:0 0 8px ${dotColor};"></span>
            ${supported ? 'SUPPORTED' : 'INCONCLUSIVE'}: ${best.lat.toFixed(4)}, ${best.lon.toFixed(4)}
        `;

        if (window.innerWidth < 1200) {
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

    // Initial default scenario load: SFO
    await loadScenario('sfo_dual');
});
