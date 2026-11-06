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

    let currentScenarioId = 'sfo_ual421';
    let availableScenarios = [];

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
                opt.textContent = `${sc.name} (${sc.aircraft_type})`;
                if (sc.id === 'sfo_ual421') opt.selected = true;
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
        let audioUrl = `/static/samples/${scenarioId}.wav`;

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
                const primaryFlight = flightResp.flights[0];
                window.mapView.displayFlightTrajectory(primaryFlight);

                // Auto-populate an initial representative trace so user can click Geolocate immediately
                autoSeedInitialTrace(primaryFlight, f0);
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
            // Upload to server cache
            window.apiClient.uploadAudio(file).catch(e => console.warn('Server audio cache notice:', e));
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
                window.mapView.displayFlightTrajectory(resp.flights[0]);
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
        const tracedPoints = window.tracer.getTracedCurve();
        if (tracedPoints.length < 3) {
            alert('Please trace at least 3 points of the Doppler frequency curve on the spectrogram.');
            return;
        }

        const f0 = parseFloat(document.getElementById('f0Input').value);
        if (!f0 || f0 <= 0) {
            alert('Please specify a valid fundamental frequency f0.');
            return;
        }

        const lat = parseFloat(document.getElementById('centerLat').value);
        const lon = parseFloat(document.getElementById('centerLon').value);
        const radius = parseFloat(searchRadius.value);
        const gridRes = parseInt(document.getElementById('gridResInput').value, 10);

        showLoading(`Evaluating 3D Doppler acoustic models across ${gridRes * gridRes} candidate positions...`);

        try {
            const result = await window.apiClient.geolocateObserver({
                center_lat: lat,
                center_lon: lon,
                radius_km: radius,
                f0: f0,
                traced_curve: tracedPoints,
                scenario_id: currentScenarioId,
                grid_resolution: gridRes,
            });

            // Update UI with results
            displayGeolocationResults(result);

        } catch (err) {
            alert(`Geolocation error: ${err.message}`);
        } finally {
            hideLoading();
        }
    });

    function displayGeolocationResults(result) {
        const best = result.best_location;
        const flight = result.best_flight;

        // Telemetry values
        document.getElementById('matchScoreNum').textContent = best.score.toFixed(1);
        document.getElementById('resLatLon').textContent = `${best.lat.toFixed(5)}, ${best.lon.toFixed(5)}`;
        document.getElementById('resFlightId').textContent = `${flight.callsign || flight.flight_id}`;
        document.getElementById('resAircraftType').textContent = flight.aircraft_type || 'Commercial Jet';
        document.getElementById('resRmse').textContent = `${best.rmse_hz.toFixed(2)} Hz RMSE`;
        document.getElementById('resTimeShift').textContent = `${best.time_offset_sec.toFixed(2)} s`;
        document.getElementById('resTotalTested').textContent = `${result.heatmap.length} points`;

        // Render Heatmap on Map
        window.mapView.displayHeatmap(result.heatmap, best, flight);

        if (result.flight_trajectory) {
            window.mapView.displayFlightTrajectory(result.flight_trajectory);
        }

        // Render Comparison Curve Chart
        window.mapView.renderComparisonChart(result.curve_comparison, result.f0_hz);

        // Update map overlay badge
        document.getElementById('mapOverlayBadge').innerHTML = `
            <span class="status-dot" style="background:#ff0055; box-shadow:0 0 8px #ff0055;"></span>
            MIC MATCH: ${best.score.toFixed(1)}% | ${best.lat.toFixed(4)}, ${best.lon.toFixed(4)}
        `;

        // Scroll smoothly to results card on mobile/narrow screens
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
    await loadScenario('sfo_ual421');
});
