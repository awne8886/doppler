/**
 * API Client for Doppler Acoustic Aircraft Geolocator backend.
 */
class ApiClient {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
    }

    async getScenarios() {
        const resp = await fetch(`${this.baseUrl}/api/scenarios`);
        if (!resp.ok) throw new Error(`Failed to load scenarios (${resp.status})`);
        return await resp.json();
    }

    async searchFlights(params) {
        const resp = await fetch(`${this.baseUrl}/api/flights/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `Flight search failed (${resp.status})`);
        }
        return await resp.json();
    }

    async geolocateObserver(payload) {
        const resp = await fetch(`${this.baseUrl}/api/geolocate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `Geolocation evaluation failed (${resp.status})`);
        }
        return await resp.json();
    }

    async uploadAudio(file) {
        const formData = new FormData();
        formData.append('file', file);

        const resp = await fetch(`${this.baseUrl}/api/audio/upload`, {
            method: 'POST',
            body: formData,
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `Upload failed (${resp.status})`);
        }
        return await resp.json();
    }
}

window.apiClient = new ApiClient();
