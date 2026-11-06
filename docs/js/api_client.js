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

        // Client-side flight retrieval
        const flights = window.clientFlights.fetchOrGenerateFlights(
            params.center_lat,
            params.center_lon,
            params.radius_km,
            params.scenario_id
        );
        return { count: flights.length, flights };
    }

    async geolocateObserver(payload) {
        const hasBackend = await this.checkBackend();
        if (hasBackend) {
            try {
                const resp = await fetch(`${this.baseUrl}/api/geolocate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                if (resp.ok) return await resp.json();
            } catch (e) {
                console.warn('Backend geolocation failed, falling back to client physics:', e);
            }
        }

        // Execute physics engine in browser!
        const flights = window.clientFlights.fetchOrGenerateFlights(
            payload.center_lat,
            payload.center_lon,
            payload.radius_km,
            payload.scenario_id
        );

        if (!flights || flights.length === 0) {
            throw new Error('No flight trajectory found.');
        }

        const primaryFlight = flights[0];
        const gridRes = payload.grid_resolution || 21;

        const evalResult = window.clientPhysics.evaluateGrid(
            payload.center_lat,
            payload.center_lon,
            payload.radius_km,
            primaryFlight,
            payload.traced_curve,
            payload.f0,
            gridRes
        );

        evalResult.flight_trajectory = {
            flight_id: primaryFlight.flight_id,
            callsign: primaryFlight.callsign,
            aircraft_type: primaryFlight.aircraft_type,
            latitudes: primaryFlight.latitudes,
            longitudes: primaryFlight.longitudes,
            altitudes: primaryFlight.altitudes,
            speeds: primaryFlight.speeds,
        };

        return {
            status: 'success',
            best_flight: {
                flight_id: evalResult.flight_id,
                callsign: evalResult.callsign,
                aircraft_type: evalResult.aircraft_type,
            },
            best_location: evalResult.best_location,
            heatmap: evalResult.heatmap,
            curve_comparison: evalResult.curve_comparison,
            flight_trajectory: evalResult.flight_trajectory,
            f0_hz: evalResult.f0_hz,
            mode: 'client_physics',
        };
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
