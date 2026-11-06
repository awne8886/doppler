/**
 * Leaflet.js Mapping & Acoustic Heatmap Visualization Engine.
 * Renders spatial heatmaps, flight trajectories, and comparison curves.
 */
class MapView {
    constructor() {
        this.map = null;
        this.heatLayer = null;
        this.gridMarkersLayer = null;
        this.flightLayer = null;
        this.radiusCircle = null;
        this.hottestMarker = null;

        this.comparisonCanvas = document.getElementById('comparisonChart');
        this.chartCtx = this.comparisonCanvas.getContext('2d');

        this.initMap();
    }

    initMap() {
        const cartoKey = 'cb1_53g1_1_fd641ece496f7c5c49a5afbe';

        // 1. CARTO Voyager (Default with User API Key)
        const cartoVoyager = L.tileLayer(
            `https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=${cartoKey}`,
            {
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
                subdomains: 'abcd',
                maxZoom: 19,
            }
        );

        // 2. CARTO Dark Matter (with User API Key)
        const cartoDark = L.tileLayer(
            `https://{s}.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}.png?key=${cartoKey}`,
            {
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
                subdomains: 'abcd',
                maxZoom: 19,
            }
        );

        // 3. CARTO Positron (with User API Key)
        const cartoPositron = L.tileLayer(
            `https://{s}.basemaps.cartocdn.com/rastertiles/light_all/{z}/{x}/{y}.png?key=${cartoKey}`,
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

    displayHeatmap(heatmapPoints, bestLocation, flightInfo) {
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
                    <strong>Coords:</strong> ${pt.lat.toFixed(5)}, ${pt.lon.toFixed(5)}<br>
                    <strong>Match Score:</strong> <span style="color:#d97706; font-weight:bold;">${pt.score}%</span><br>
                    <strong>Least Squares RMSE:</strong> ${pt.rmse} Hz<br>
                    <strong>Matched Flight:</strong> ${flightInfo.callsign || flightInfo.flight_id}
                </div>
            `);
            this.gridMarkersLayer.addLayer(circle);
        }

        // 3. Mark Hottest Point (Estimated microphone location)
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
                        <h4 style="color:#e11d48; margin-bottom:4px;">&#127908; ESTIMATED MICROPHONE LOCATION</h4>
                        <strong>Coords:</strong> ${bestLocation.lat.toFixed(6)}, ${bestLocation.lon.toFixed(6)}<br>
                        <strong>Confidence Match:</strong> ${bestLocation.score}%<br>
                        <strong>Residual RMSE:</strong> ${bestLocation.rmse_hz} Hz<br>
                        <strong>Matched Flight:</strong> ${flightInfo.callsign || flightInfo.flight_id} (${flightInfo.aircraft_type})
                    </div>
                `)
                .openPopup();
        }
    }

    displayFlightTrajectory(trajectory) {
        this.flightLayer.clearLayers();
        if (!trajectory || !trajectory.latitudes || trajectory.latitudes.length === 0) return;

        const pathPoints = [];
        for (let i = 0; i < trajectory.latitudes.length; i++) {
            pathPoints.push([trajectory.latitudes[i], trajectory.longitudes[i]]);
        }

        // Draw flight path polyline
        const polyline = L.polyline(pathPoints, {
            color: '#f59e0b',
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
                    border: 2px solid #f59e0b;
                    border-radius: 50%;
                    display: flex; align-items: center; justify-content: center;
                    box-shadow: 0 0 10px rgba(245, 158, 11, 0.6);
                ">
                    <span style="color:#f59e0b; font-size:16px;">&#9992;</span>
                </div>
            `,
            iconSize: [28, 28],
            iconAnchor: [14, 14],
        });

        const planeMarker = L.marker(endPoint, { icon: planeIcon }).addTo(this.flightLayer);
        planeMarker.bindPopup(`
            <div style="font-family:'JetBrains Mono', monospace; font-size:12px; color:#111;">
                <strong>${trajectory.callsign || trajectory.flight_id}</strong><br>
                ${trajectory.aircraft_type || 'Aircraft'}<br>
                <strong>Speed:</strong> ${trajectory.speeds ? trajectory.speeds[0] : 170} m/s<br>
                <strong>Altitude:</strong> ${trajectory.altitudes ? trajectory.altitudes[0] : 1200} m
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

        // Margins
        const padL = 50, padR = 20, padT = 20, padB = 30;
        const plotW = width - padL - padR;
        const plotH = height - padT - padB;

        const times = comparisonPoints.map(p => p.time);
        const tracedFreqs = comparisonPoints.map(p => p.traced_freq);
        const modeledFreqs = comparisonPoints.map(p => p.modeled_freq);

        const allFreqs = [...tracedFreqs, ...modeledFreqs, f0].filter(f => f > 0);
        const minF = Math.min(...allFreqs) - 50;
        const maxF = Math.max(...allFreqs) + 50;
        const minT = Math.min(...times);
        const maxT = Math.max(...times);
        const dt = Math.max(1.0, maxT - minT);

        const tToX = (t) => padL + ((t - minT) / dt) * plotW;
        const fToY = (f) => padT + (1.0 - (f - minF) / (maxF - minF)) * plotH;

        // Draw axes and grid lines
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
            ctx.fillText(`${Math.round(fVal)}Hz`, 8, y + 3);
        }
        ctx.stroke();

        // Draw f0 line
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

        // Draw Modeled Curve (Amber)
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        for (let i = 0; i < comparisonPoints.length; i++) {
            const x = tToX(comparisonPoints[i].time);
            const y = fToY(comparisonPoints[i].modeled_freq);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Draw Traced Curve (Cyan with dots)
        ctx.strokeStyle = '#22d3ee';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        for (let i = 0; i < comparisonPoints.length; i++) {
            const x = tToX(comparisonPoints[i].time);
            const y = fToY(comparisonPoints[i].traced_freq);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Draw points for traced curve
        for (let i = 0; i < comparisonPoints.length; i++) {
            const x = tToX(comparisonPoints[i].time);
            const y = fToY(comparisonPoints[i].traced_freq);
            ctx.fillStyle = '#22d3ee';
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();
        }

        // Time axis ticks
        ctx.fillStyle = '#64748b';
        ctx.font = '10px "JetBrains Mono", monospace';
        for (let i = 0; i <= 4; i++) {
            const tVal = minT + (i / 4) * dt;
            const x = tToX(tVal);
            ctx.fillText(`${tVal.toFixed(1)}s`, x - 10, height - 8);
        }
    }
}

window.mapView = new MapView();
