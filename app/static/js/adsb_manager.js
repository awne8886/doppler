/**
 * Client-Side ADS-B Flight Trajectory Manager.
 * Provides curated scenarios and realistic dynamic trajectory generation for GitHub Pages.
 */

class ClientFlightManager {
    constructor() {
        this.scenarios = this.initCuratedScenarios();
    }

    generateTrajectory(flightId, callsign, aircraftType, startLat, startLon, startAlt, headingDeg, speedMps, verticalRateMps, durationSec = 60.0, startEpoch = 1700000000.0, f0 = 800.0, gtMic = { lat: 0, lon: 0 }) {
        const R_earth = 6371000.0;
        const numPoints = Math.floor(durationSec) + 1;
        const headingRad = (headingDeg * Math.PI) / 180.0;

        const vx = speedMps * Math.sin(headingRad);
        const vy = speedMps * Math.cos(headingRad);
        const vz = verticalRateMps;

        const timestamps = [];
        const latitudes = [];
        const longitudes = [];
        const altitudes = [];
        const speeds = [];
        const headings = [];

        const startLatRad = (startLat * Math.PI) / 180.0;

        for (let i = 0; i < numPoints; i++) {
            const t = i * 1.0;
            const dx = vx * t;
            const dy = vy * t;
            const dz = vz * t;

            const dLatRad = dy / R_earth;
            const latRad = startLatRad + dLatRad;
            const latDeg = (latRad * 180.0) / Math.PI;

            const latMid = (startLatRad + latRad) / 2.0;
            const dLonRad = dx / (R_earth * Math.cos(latMid));
            const lonDeg = startLon + (dLonRad * 180.0) / Math.PI;

            timestamps.push(startEpoch + t);
            latitudes.push(Math.round(latDeg * 1e6) / 1e6);
            longitudes.push(Math.round(lonDeg * 1e6) / 1e6);
            altitudes.push(Math.round((startAlt + dz) * 10) / 10);
            speeds.push(Math.round(speedMps * 10) / 10);
            headings.push(headingDeg);
        }

        return {
            flight_id: flightId,
            icao24: flightId.toLowerCase().slice(0, 6),
            callsign,
            aircraft_type: aircraftType,
            f0_hz: f0,
            ground_truth_mic: gtMic,
            start_epoch: startEpoch,
            duration_sec: durationSec,
            timestamps,
            latitudes,
            longitudes,
            altitudes,
            speeds,
            headings,
        };
    }

    initCuratedScenarios() {
        return {
            sfo_ual421: {
                id: 'sfo_ual421',
                name: 'San Francisco Bay Area - UAL421 (B737)',
                description: 'United Airlines Boeing 737 on approach to KSFO passing directly over Foster City.',
                search_center: { lat: 37.5585, lon: -122.2711 },
                recommended_radius_km: 10.0,
                f0_target: 840.0,
                sample_audio_url: './samples/sfo_ual421.wav',
                trajectory: this.generateTrajectory(
                    'UAL421', 'UAL421', 'Boeing 737-800',
                    37.5100, -122.2000, 1450.0,
                    305.0, 168.0, -7.5, 60.0, 1700000000.0, 840.0,
                    { lat: 37.5585, lon: -122.2711 }
                ),
            },
            lhr_baw282: {
                id: 'lhr_baw282',
                name: 'London Heathrow - BAW282 (A350)',
                description: 'British Airways A350 descending towards EGLL runway 27L over South West London.',
                search_center: { lat: 51.4620, lon: -0.3080 },
                recommended_radius_km: 12.0,
                f0_target: 680.0,
                sample_audio_url: './samples/lhr_baw282.wav',
                trajectory: this.generateTrajectory(
                    'BAW282', 'BAW282', 'Airbus A350-900',
                    51.4300, -0.1600, 1200.0,
                    280.0, 155.0, -6.0, 60.0, 1700000000.0, 680.0,
                    { lat: 51.4620, lon: -0.3080 }
                ),
            },
            jfk_dal410: {
                id: 'jfk_dal410',
                name: 'New York JFK - DAL410 (A321)',
                description: 'Delta Air Lines A321 climbing on departure over Queens/Long Island.',
                search_center: { lat: 40.6900, lon: -73.7800 },
                recommended_radius_km: 10.0,
                f0_target: 910.0,
                sample_audio_url: './samples/jfk_dal410.wav',
                trajectory: this.generateTrajectory(
                    'DAL410', 'DAL410', 'Airbus A321',
                    40.6400, -73.7780, 600.0,
                    25.0, 185.0, 12.0, 60.0, 1700000000.0, 910.0,
                    { lat: 40.6900, lon: -73.7800 }
                ),
            },
            sfo_dual: (() => {
                const gt = { lat: 37.5585, lon: -122.2711 };
                const epoch = 1700003600.0;
                const flights = [
                    this.generateTrajectory('UAL1714', 'UAL1714', 'Boeing 737-900', 37.509314, -122.200289, 1250.0,
                        305.0, 165.0, -3.0, 120.0, epoch - 32.0, 820.0, gt),
                    this.generateTrajectory('SWA2286', 'SWA2286', 'Boeing 737-700', 37.482178, -122.32051, 380.0,
                        35.0, 135.0, 6.0, 120.0, epoch - 18.0, 590.0, gt),
                    this.generateTrajectory('ASA331', 'ASA331', 'Airbus A320', 37.662529, -122.288682, 3000.0,
                        160.0, 190.0, 0.0, 120.0, epoch + 90.0, 700.0, gt),
                ];
                return {
                    id: 'sfo_dual',
                    name: 'Foster City - Two Flyovers (UAL1714 + SWA2286)',
                    description: 'Two crossing flyovers in one 75 s recording with a shared clock; an unrelated ASA331 pass is included as a decoy.',
                    search_center: gt,
                    recommended_radius_km: 8.0,
                    f0_target: 820.0,
                    audio_duration_sec: 75.0,
                    recording_start_epoch: epoch,
                    sample_audio_url: './samples/sfo_dual.wav',
                    trajectory: flights[0],
                    flights,
                };
            })(),
        };
    }

    /** Epoch of the first audio sample of a single-flyover sample (window centred on CPA arrival). */
    recordingStartEpoch(traj, gt, duration = 25.0, c = 343.2) {
        const R = 6371000.0, toRad = Math.PI / 180.0;
        const la0 = gt.lat * toRad, lo0 = gt.lon * toRad;
        let best = Infinity, iBest = 0;
        for (let i = 0; i < traj.timestamps.length; i++) {
            const la = traj.latitudes[i] * toRad, lo = traj.longitudes[i] * toRad;
            const x = R * (lo - lo0) * Math.cos((la + la0) / 2), y = R * (la - la0), z = traj.altitudes[i];
            const d = Math.sqrt(x * x + y * y + z * z);
            if (d < best) { best = d; iBest = i; }
        }
        const tObs = (traj.timestamps[iBest] - traj.timestamps[0]) + best / c;
        return traj.timestamps[0] + Math.max(0, tObs - duration / 2);
    }

    getScenarioFlights(sc) {
        return sc.flights || [sc.trajectory];
    }

    getScenarios() {
        return Object.values(this.scenarios).map(sc => ({
            id: sc.id,
            name: sc.name,
            description: sc.description,
            search_center: sc.search_center,
            recommended_radius_km: sc.recommended_radius_km,
            f0_target: sc.f0_target,
            callsign: sc.trajectory.callsign,
            aircraft_type: sc.trajectory.aircraft_type,
            sample_audio_url: sc.sample_audio_url,
            n_flights: this.getScenarioFlights(sc).length,
            audio_duration_sec: sc.audio_duration_sec || 25.0,
            recording_start_epoch: sc.recording_start_epoch || this.recordingStartEpoch(sc.trajectory, sc.search_center),
        }));
    }

    fetchOrGenerateFlights(centerLat, centerLon, radiusKm, scenarioId = null) {
        if (scenarioId && this.scenarios[scenarioId]) {
            return this.getScenarioFlights(this.scenarios[scenarioId]);
        }

        // Check if close to curated scenario
        for (const sc of Object.values(this.scenarios)) {
            const dLat = (centerLat - sc.search_center.lat) * 111.0;
            const dLon = (centerLon - sc.search_center.lon) * 111.0 * Math.cos((centerLat * Math.PI) / 180.0);
            if (Math.hypot(dLat, dLon) < 15.0) {
                return this.getScenarioFlights(sc);
            }
        }

        // Dynamically synthesize candidate flight passing near center
        const hdg = 290.0;
        const rad = (hdg * Math.PI) / 180.0;
        const startDist = radiusKm * 750.0;
        const dx = -startDist * Math.sin(rad);
        const dy = -startDist * Math.cos(rad);

        const sLat = centerLat + (dy / 6371000.0) * (180.0 / Math.PI);
        const sLon = centerLon + (dx / (6371000.0 * Math.cos((centerLat * Math.PI) / 180.0))) * (180.0 / Math.PI);

        const traj = this.generateTrajectory(
            'ADS01', 'SKY772', 'Commercial Jet (ADS-B Sim)',
            sLat, sLon, 1600.0,
            hdg, 175.0, -3.0, 60.0, 1700000000.0, 800.0,
            { lat: centerLat, lon: centerLon }
        );

        return [traj];
    }
}

window.clientFlights = new ClientFlightManager();
