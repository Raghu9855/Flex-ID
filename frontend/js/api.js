/**
 * FLEX-ID — Native Fetch API Client
 * Clean, lightweight, zero-dependency communication with FastAPI REST backend
 */

const API_BASE = '';

class ApiClient {
  constructor(baseUrl = API_BASE) {
    this.baseUrl = baseUrl;
  }

  async _request(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const headers = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      ...options.headers,
    };

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), options.timeout || 30000);

    try {
      const response = await fetch(url, {
        ...options,
        headers,
        signal: controller.signal,
      });

      clearTimeout(timeout);

      const data = await response.json().catch(() => null);

      if (!response.ok) {
        const errorMsg = data?.detail || data?.message || `HTTP ${response.status}: ${response.statusText}`;
        const error = new Error(errorMsg);
        error.status = response.status;
        error.data = data;
        throw error;
      }

      return data;
    } catch (err) {
      clearTimeout(timeout);
      if (err.name === 'AbortError') {
        throw new Error('Request timed out. The backend service may be starting up.');
      }
      throw err;
    }
  }

  // System & Health
  async getHealth() {
    return this._request('/health');
  }

  // Models
  async getModels() {
    return this._request('/api/v1/models');
  }

  async getModelDetails(version) {
    return this._request(`/api/v1/models/${encodeURIComponent(version)}`);
  }

  async activateModel(version) {
    return this._request(`/api/v1/models/${encodeURIComponent(version)}/activate`, {
      method: 'POST',
    });
  }

  // Predictions
  async predictSingle(features, modelVersion = null) {
    const payload = { features };
    if (modelVersion) payload.model_version = modelVersion;
    return this._request('/api/v1/predictions', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async predictBatch(records, modelVersion = null) {
    const payload = {
      records: records.map(r => ({ features: r })),
    };
    if (modelVersion) payload.model_version = modelVersion;
    return this._request('/api/v1/predictions/batch', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async getPredictionRecord(id) {
    return this._request(`/api/v1/predictions/${encodeURIComponent(id)}`);
  }

  // Experiments
  async getExperiments() {
    return this._request('/api/v1/experiments');
  }

  async getExperimentDetails(id) {
    return this._request(`/api/v1/experiments/${encodeURIComponent(id)}`);
  }

  async getExperimentMetrics(id) {
    return this._request(`/api/v1/experiments/${encodeURIComponent(id)}/metrics`);
  }

  async triggerExperiment(strategy = 'fedprox', rounds = 10, clients = 4, attackType = null) {
    const payload = {
      name: `${strategy.toUpperCase()} ${attackType ? `[${attackType}]` : '[Clean]'} - Web Triggered`,
      strategy,
      rounds,
      clients,
      attack_type: attackType,
    };
    return this._request('/api/v1/experiments', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  // Explainability (SHAP)
  async getExplanation(features, modelVersion = null, topK = 5) {
    const payload = {
      features,
      top_k: topK,
    };
    if (modelVersion) payload.model_version = modelVersion;
    return this._request('/api/v1/explanations', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }
}

// Export singleton instance
window.api = new ApiClient();
