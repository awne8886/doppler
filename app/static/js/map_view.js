/**
 * Leaflet.js Mapping & Acoustic Heatmap Visualization Engine.
 * Renders spatial heatmaps, flight trajectories, and comparison curves.
 */
const esc = (v) => window.DopplerUtil.escapeHtml(v);
const num = (v, d = 0) => window.DopplerUtil.fmtNum(v, d);

class MapView {
    constructor() {
        this.map = null;
        this.heatLayer = null;
        this.gridMarkersLayer = null;
        this.flightLayer = null;
        this.radiusCircle = null;
        this.hottestMarker = null;
        this.altLayer = null;

        this.comparisonCanvas = document.getElementById('comparisonChart');
        this.chartCtx = this.comparisonCanvas.getContext('2d');

        this.initMap();
    }

    initMap() {
        // Configurable key (js/config.js or the server's DOPPLER_CARTO_KEY); omitted when empty.
        const cartoKey = (window.DOPPLER_CONFIG && window.DOPPLER_CONFIG.cartoKey) || '';
        const keyParam = cartoKey ? `?key=${encodeURIComponent(cartoKey)}` : '';

        // 1. CARTO Voyager (Default with User API Key)
        const cartoVoyager = L.tileLayer(
            `https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png${keyParam}`,
            {
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
                subdomains: 'abcd',
                maxZoom: 19,
            }
        );

        // 2. CARTO Dark Matter (with User API Key)
        const cartoDark = L.tileLayer(
            `https://{s}.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}.png${keyParam}`,
            {
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
                subdomains: 'abcd',
                maxZoom: 19,
            }
        );

        // 3. CARTO Positron (with User API Key)
        const cartoPositron = L.tileLayer(
            `https://{s}.basemaps.cartocdn.com/rastertiles/light_all/{z}/{x}/{y}.png${keyParam}`,
            {
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
                subdomains: 'abcd',
                maxZoom: 19,
            }
        );

        // Initialize Leaflet map centered at SFO
        this.map = L.map('map', {
            center: [37.5585, -122.2711],
            zoom: 11,
            zoomControl: true,
            layers: [cartoVoyager],
        });

        // Add Layer Switcher Control
        const baseLayers = {
            "CARTO Voyager": cartoVoyager,
            "CARTO Dark": cartoDark,
            "CARTO Positron": cartoPositron,
        };
        L.control.layers(baseLayers, null, { position: 'bottomleft' }).addTo(this.map);

        this.gridMarkersLayer = L.layerGroup().addTo(this.map);
        this.flightLayer = L.layerGroup().addTo(this.map);
        this.altLayer = L.layerGroup().addTo(this.map);
        this.ambiguityLayer = L.layerGroup().addTo(this.map);

        // Map click updates search center if user clicks holding Alt or Shift
        this.map.on('click', (e) => {
            if (e.originalEvent.shiftKey || e.originalEvent.altKey) {
                document.getElementById('centerLat').value = e.latlng.lat.toFixed(5);
                document.getElementById('centerLon').value = e.latlng.lng.toFixed(5);
                this.updateSearchRadiusCircle(e.latlng.lat, e.latlng.lng, parseFloat(document.getElementById('searchRadius').value));
            }
        });
    }

    setCenter(lat, lon, zoom = 11) {
        this.map.setView([lat, lon], zoom);
    }

    updateSearchRadiusCircle(lat, lon, radiusKm) {
        if (this.radiusCircle) {
            this.map.removeLayer(this.radiusCircle);
        }

        this.radiusCircle = L.circle([lat, lon], {
            radius: radiusKm * 1000,
            color: '#06b6d4',
            dashArray: '5, 8',
            weight: 1.5,
            fillColor: '#06b6d4',
            fillOpacity: 0.05,
        }).addTo(this.map);
    }

    displayHeatmap(heatmapPoints, bestLocation, flightInfo, extras = {}) {
        this.altLayer.clearLayers();
        heatmapPoints = (heatmapPoints || []).filter(p => p.rmse !== null && p.rmse !== undefined);
        // Clear previous layers
        if (this.heatLayer) {
            this.map.removeLayer(this.heatLayer);
            this.heatLayer = null;
        }
        this.gridMarkersLayer.clearLayers();

        if (!heatmapPoints || heatmapPoints.length === 0) return;

        // 1. Leaflet Heat Layer (continuous gradient)
        if (typeof L.heatLayer === 'function') {
            const heatData = heatmapPoints.map(p => [
                p.lat,
                p.lon,
                Math.max(0.01, (p.score / 100.0) ** 1.8), // Enhanced contrast
            ]);

            this.heatLayer = L.heatLayer(heatData, {
                radius: 28,
                blur: 20,
                maxZoom: 15,
                max: 1.0,
                gradient: {
                    0.1: '#002244',
                    0.3: '#0055ff',
                    0.5: '#00ffff',
                    0.7: '#ffff00',
                    0.9: '#ff3300',
                    1.0: '#ffffff',
                },
            }).addTo(this.map);
        }

        // 2. Interactive grid markers for high scoring points
        // Sort points by score descending
        const sorted = [...heatmapPoints].sort((a, b) => b.score - a.score);
        const topPoints = sorted.slice(0, 45); // Add clickable circles for top hot points

        for (const pt of topPoints) {
            const markerColor = pt.score > 85 ? '#ff3300' : pt.score > 60 ? '#ffff00' : '#00ffff';
            const circle = L.circleMarker([pt.lat, pt.lon], {
                radius: pt.score > 85 ? 6 : 4,
                color: markerColor,
                fillColor: markerColor,
                fillOpacity: 0.8,
                weight: 1,
            });

            circle.bindPopup(`
                <div style="font-family:'JetBrains Mono', monospace; font-size:12px; color:#111;">
                    <strong>Candidate Acoustic Observer</strong><br>
                    <strong>Coords:</strong> ${num(pt.lat, 5)}, ${num(pt.lon, 5)}<br>
                    <strong>Fit Score:</strong> <span style="color:#d97706; font-weight:bold;">${num(pt.score, 2)}%</span><br>
                    <strong>Robust RMSE:</strong> ${num(pt.rmse, 2)} Hz<br>
                    <strong>Matched Flight:</strong> ${esc(flightInfo.callsign || flightInfo.flight_id)}
                </div>
            `);
            this.gridMarkersLayer.addLayer(circle);
        }

        // 3. Competing optima and sensitivity radius
        const decision = extras.decision;
        for (const alt of extras.alternatives || []) {
            const tie = decision && (decision.checks || []).some(c => c.check === 'ambiguity' &&
                (c.near_tied_alternatives || []).some(t => Math.abs(t.lat - alt.lat) < 1e-5 && Math.abs(t.lon - alt.lon) < 1e-5));
            L.circleMarker([alt.lat, alt.lon], {
                radius: tie ? 9 : 6, color: tie ? '#f59e0b' : '#94a3b8', weight: 2, fillOpacity: 0, dashArray: tie ? null : '3,3',
            }).bindPopup(`
                <div style="font-family:'JetBrains Mono', monospace; font-size:12px; color:#111;">
                    <strong>${tie ? 'Near-tied alternative' : 'Competing local optimum'}</strong><br>
                    ${num(alt.lat, 5)}, ${num(alt.lon, 5)}<br>
                    Robust RMSE: ${num(alt.robust_rmse_hz, 3)} Hz<br>
                    ${num(alt.distance_from_best_m / 1000, 2)} km from best
                </div>`).addTo(this.altLayer);
        }
        if (bestLocation && bestLocation.uncertainty_radius_m) {
            L.circle([bestLocation.lat, bestLocation.lon], {
                radius: Math.max(10, bestLocation.uncertainty_radius_m),
                color: decision && decision.status === 'supported' ? '#10b981' : '#f59e0b',
                weight: 1.5, dashArray: '4,4', fillOpacity: 0.08,
            }).bindTooltip(`Sensitivity radius ${num(bestLocation.uncertainty_radius_m)} m (engineering envelope, uncalibrated)`).addTo(this.altLayer);
        }

        // 4. Mark Hottest Point (Estimated microphone location)
        if (bestLocation) {
            if (this.hottestMarker) {
                this.map.removeLayer(this.hottestMarker);
            }

            const micIcon = L.divIcon({
                className: 'hottest-mic-marker',
                html: `
                    <div style="
                        width: 32px; height: 32px;
                        background: radial-gradient(circle, #ff0055 0%, #ff3300 100%);
                        border: 2px solid #ffffff;
                        border-radius: 50%;
                        display: flex; align-items: center; justify-content: center;
                        box-shadow: 0 0 16px #ff0055, 0 0 25px rgba(255, 0, 85, 0.8);
                        animation: pulse-ring 1.8s infinite;
                        cursor: pointer;
                    ">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.5">
                            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                            <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                            <line x1="12" y1="19" x2="12" y2="23"></line>
                            <line x1="8" y1="23" x2="16" y2="23"></line>
                        </svg>
                    </div>
                `,
                iconSize: [32, 32],
                iconAnchor: [16, 16],
            });

            this.hottestMarker = L.marker([bestLocation.lat, bestLocation.lon], { icon: micIcon })
                .addTo(this.map)
                .bindPopup(`
                    <div style="font-family:'Space Grotesk', sans-serif; font-size:13px; color:#111;">
                        <h4 style="color:#e11d48; margin-bottom:4px;">&#127908; ${decision && decision.status === 'supported' ? 'SUPPORTED MICROPHONE LOCATION' : 'BEST-RANKED HYPOTHESIS (INCONCLUSIVE)'}</h4>
                        <strong>Coords:</strong> ${num(bestLocation.lat, 6)}, ${num(bestLocation.lon, 6)}<br>
                        <strong>Fit Score:</strong> ${num(bestLocation.score, 2)}%<br>
                        <strong>Worst Event RMSE:</strong> ${num(bestLocation.rmse_hz, 3)} Hz<br>
                        <strong>Matched Flight:</strong> ${esc(flightInfo.callsign || flightInfo.flight_id)} (${esc(flightInfo.aircraft_type || 'unknown type')})
                    </div>
                `)
                .openPopup();
        }
    }

    /** Single-flyover mirror ambiguity: mirror point, connecting line and (free clock) along-track ridges. */
    displayAmbiguity(best, amb, freeClock) {
        this.ambiguityLayer.clearLayers();
        L.circleMarker([amb.mirror.lat, amb.mirror.lon], { radius: 9, color: '#f59e0b', weight: 2, dashArray: '3,3', fillOpacity: 0.1 })
            .bindPopup(`<div style="font-family:'JetBrains Mono', monospace; font-size:12px; color:#111;">
                <strong>Mirror ambiguity</strong><br>${num(amb.mirror.lat, 5)}, ${num(amb.mirror.lon, 5)}<br>
                ${num(amb.distance_m / 1000, 2)} km from the best point, across the flight track.<br>
                One straight flyover sounds the same on both sides of its track.</div>`)
            .addTo(this.ambiguityLayer);
        L.polyline([[best.lat, best.lon], [amb.mirror.lat, amb.mirror.lon]], { color: '#f59e0b', weight: 1.5, dashArray: '2,6' })
            .addTo(this.ambiguityLayer);
        if (freeClock) {
            for (const ridge of amb.ridges) {
                L.polyline(ridge.map(p => [p.lat, p.lon]), { color: '#f59e0b', weight: 6, opacity: 0.25 })
                    .bindTooltip('Ambiguity ridge: with a free recording clock the observer can also slide along the track.')
                    .addTo(this.ambiguityLayer);
            }
        }
    }

    clearAmbiguity() {
        if (this.ambiguityLayer) this.ambiguityLayer.clearLayers();
    }

    displayFlightTrajectories(trajectories) {
        this.flightLayer.clearLayers();
        const colors = ['#f59e0b', '#a3e635', '#f472b6', '#c084fc', '#60a5fa'];
        (trajectories || []).forEach((t, i) => this.displayFlightTrajectory(t, colors[i % colors.length], false));
    }

    displayFlightTrajectory(trajectory, color = '#f59e0b', clear = true) {
        color = /^#[0-9a-fA-F]{3,8}$/.test(color) ? color : '#f59e0b';
        if (clear) this.flightLayer.clearLayers();
        if (!trajectory || !trajectory.latitudes || trajectory.latitudes.length === 0) return;

        const pathPoints = [];
        for (let i = 0; i < trajectory.latitudes.length; i++) {
            pathPoints.push([trajectory.latitudes[i], trajectory.longitudes[i]]);
        }

        // Draw flight path polyline
        const polyline = L.polyline(pathPoints, {
            color: color,
            weight: 3,
            opacity: 0.85,
            dashArray: '8, 6',
        }).addTo(this.flightLayer);

        // Add start & end markers
        const startPoint = pathPoints[0];
        const endPoint = pathPoints[pathPoints.length - 1];

        // Aircraft icon for the flight
        const planeIcon = L.divIcon({
            className: 'plane-marker',
            html: `
                <div style="
                    width: 28px; height: 28px;
                    background: #1e293b;
                    border: 2px solid ${color};
                    border-radius: 50%;
                    display: flex; align-items: center; justify-content: center;
                    box-shadow: 0 0 10px rgba(245, 158, 11, 0.6);
                ">
                    <span style="color:${color}; font-size:16px;">&#9992;</span>
                </div>
            `,
            iconSize: [28, 28],
            iconAnchor: [14, 14],
        });

        const planeMarker = L.marker(endPoint, { icon: planeIcon }).addTo(this.flightLayer);
        planeMarker.bindPopup(`
            <div style="font-family:'JetBrains Mono', monospace; font-size:12px; color:#111;">
                <strong>${esc(trajectory.callsign || trajectory.flight_id)}</strong><br>
                ${esc(trajectory.aircraft_type || 'Aircraft')}<br>
                <strong>Speed:</strong> ${trajectory.speeds && trajectory.speeds.length ? `${num(trajectory.speeds[0])} m/s` : 'not reported'}<br>
                <strong>Altitude:</strong> ${trajectory.altitudes && trajectory.altitudes.length ? `${num(trajectory.altitudes[0])} m` : 'not reported'}
                ${trajectory.altitude_basis ? `(${esc(trajectory.altitude_basis)})` : ''}<br>
                <strong>Source:</strong> ${esc(trajectory.data_source || 'unknown')}
            </div>
        `);
    }

    renderComparisonChart(comparisonPoints, f0) {
        const canvas = this.comparisonCanvas;
        const ctx = this.chartCtx;
        const width = canvas.width;
        const height = canvas.height;
        ctx.clearRect(0, 0, width, height);
        if (!comparisonPoints || comparisonPoints.length === 0) return;

        const padL = 50, padR = 20, padT = 20, padB = 30;
        const plotW = width - padL - padR;
        const plotH = height - padT - padB;

        // Group by event so separate flyovers are drawn as separate curves.
        const byEvent = new Map();
        for (const p of comparisonPoints) {
            const k = p.event_id || 'E1';
            if (!byEvent.has(k)) byEvent.set(k, []);
            byEvent.get(k).push(p);
        }
        for (const arr of byEvent.values()) arr.sort((a, b) => a.time - b.time);

        const freqs = [];
        for (const p of comparisonPoints) {
            if (p.traced_freq > 0) freqs.push(p.traced_freq);
            if (p.modeled_freq > 0) freqs.push(p.modeled_freq);
        }
        const fMinRaw = Math.min(...freqs), fMaxRaw = Math.max(...freqs);
        const padF = Math.max(20, 0.05 * (fMaxRaw - fMinRaw));
        const minF = fMinRaw - padF, maxF = fMaxRaw + padF;
        const times = comparisonPoints.map(p => p.time);
        const minT = Math.min(...times), maxT = Math.max(...times);
        const dt = Math.max(1.0, maxT - minT);
        const tToX = (t) => padL + ((t - minT) / dt) * plotW;
        const fToY = (f) => padT + (1.0 - (f - minF) / (maxF - minF)) * plotH;

        ctx.strokeStyle = '#1e293b';
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i <= 4; i++) {
            const fVal = minF + (i / 4) * (maxF - minF);
            const y = fToY(fVal);
            ctx.moveTo(padL, y);
            ctx.lineTo(width - padR, y);
            ctx.fillStyle = '#64748b';
            ctx.font = '10px "JetBrains Mono", monospace';
            ctx.fillText(`${Math.round(fVal)}Hz`, 4, y + 3);
        }
        ctx.stroke();

        if (f0 >= minF && f0 <= maxF) {
            const yF0 = fToY(f0);
            ctx.strokeStyle = 'rgba(148, 163, 184, 0.4)';
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(padL, yF0);
            ctx.lineTo(width - padR, yF0);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        const drawLine = (pts, key, color, w) => {
            ctx.strokeStyle = color;
            ctx.lineWidth = w;
            ctx.beginPath();
            let pen = false;
            for (const p of pts) {
                const v = p[key];
                if (v === null || v === undefined || !(v > 0)) { pen = false; continue; }
                const x = tToX(p.time), y = fToY(v);
                if (!pen) { ctx.moveTo(x, y); pen = true; } else ctx.lineTo(x, y);
            }
            ctx.stroke();
        };
        for (const [id, pts] of byEvent.entries()) {
            drawLine(pts, 'modeled_freq', '#f59e0b', 2.5);
            drawLine(pts, 'traced_freq', '#22d3ee', 1.5);
            const stride = Math.max(1, Math.ceil(pts.length / 30));
            ctx.fillStyle = '#22d3ee';
            for (let i = 0; i < pts.length; i += stride) {
                ctx.beginPath();
                ctx.arc(tToX(pts[i].time), fToY(pts[i].traced_freq), 2.5, 0, Math.PI * 2);
                ctx.fill();
            }
            if (byEvent.size > 1) {
                ctx.fillStyle = '#e2e8f0';
                ctx.font = '10px "JetBrains Mono", monospace';
                ctx.fillText(id, tToX(pts[0].time) + 3, fToY(pts[0].traced_freq) - 6);
            }
        }

        ctx.fillStyle = '#64748b';
        ctx.font = '10px "JetBrains Mono", monospace';
        for (let i = 0; i <= 4; i++) {
            const tVal = minT + (i / 4) * dt;
            ctx.fillText(`${tVal.toFixed(1)}s`, tToX(tVal) - 10, height - 8);
        }
    }
}

window.mapView = new MapView();
