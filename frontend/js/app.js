/**
 * FLEX-ID — Apple-Level Interactive Application Logic
 * Pure Vanilla JavaScript · Zero Dependencies · State, Views, Visualizers & Handlers
 */

// 29 Canonical Features of CSE-CIC-IDS2018
const CANONICAL_FEATURES = [
  "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
  "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Mean",
  "Bwd Pkt Len Max", "Bwd Pkt Len Mean", "Flow Byts/s", "Flow Pkts/s",
  "Flow IAT Mean", "Flow IAT Max", "Fwd IAT Mean", "Bwd IAT Mean",
  "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s",
  "Pkt Len Mean", "Pkt Len Max", "Pkt Len Var", "SYN Flag Cnt",
  "RST Flag Cnt", "ACK Flag Cnt", "Init Fwd Win Byts", "Init Bwd Win Byts"
];

// Realistic Telemetry Presets
const FEATURE_PRESETS = {
  benign: {
    name: "Standard HTTPS Web Traffic",
    desc: "Legitimate outbound encrypted browser session on port 443 with balanced TCP window parameters.",
    values: [
      443.0, 6.0, 125000.0, 10.0, 8.0, 500.0, 1200.0, 200.0, 50.0,
      400.0, 150.0, 13600.0, 144.0, 8500.0, 25000.0, 12000.0, 15000.0,
      200.0, 160.0, 80.0, 64.0, 94.4, 400.0, 15200.0, 0.0,
      0.0, 1.0, 26883.0, 26883.0
    ]
  },
  dos_hulk: {
    name: "DoS Attacks - Hulk",
    desc: "Volumetric HTTP request flood targeting web application exhaust ports with abnormally high forward packet rates.",
    values: [
      80.0, 6.0, 850000.0, 48.0, 2.0, 4520.0, 110.0, 512.0, 94.1,
      55.0, 55.0, 5440.0, 58.8, 17700.0, 145000.0, 18000.0, 5000.0,
      960.0, 40.0, 56.4, 2.3, 92.6, 512.0, 18400.0, 0.0,
      0.0, 1.0, 8192.0, 255.0
    ]
  },
  ddos_hoic: {
    name: "DDoS Attack - HOIC",
    desc: "High Orbit Ion Cannon distributed denial of service attack pattern exhibiting synchronized high byte rates.",
    values: [
      80.0, 6.0, 1450000.0, 120.0, 0.0, 24000.0, 0.0, 200.0, 200.0,
      0.0, 0.0, 16551.0, 82.7, 12000.0, 80000.0, 12000.0, 0.0,
      2400.0, 0.0, 82.7, 0.0, 200.0, 200.0, 0.0, 0.0,
      0.0, 1.0, 64240.0, 0.0
    ]
  },
  portscan: {
    name: "Infiltration / Port Reconnaissance",
    desc: "Low-latency reconnaissance probe testing service availability across protected network boundaries.",
    values: [
      22.0, 6.0, 1200.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 2500.0, 600.0, 1200.0, 1200.0, 0.0,
      40.0, 20.0, 1666.6, 833.3, 0.0, 0.0, 0.0, 1.0,
      0.0, 0.0, 1024.0, 0.0
    ]
  }
};

// Empirical Benchmark Data (CSE-CIC-IDS2018 144K Test Flows)
const VERIFIED_BENCHMARKS = {
  no_attack: {
    fedprox: { acc: 90.93, bal_acc: 74.01, macro_f1: 0.5899, weighted_f1: 0.9109, mcc: 0.8074, roc_auc: 0.9812 },
    fedavg:  { acc: 81.47, bal_acc: 60.38, macro_f1: 0.4350, weighted_f1: 0.8382, mcc: 0.6275, roc_auc: 0.9496 }
  },
  under_attack: {
    fedprox: { acc: 88.82, bal_acc: 63.60, macro_f1: 0.5186, weighted_f1: 0.8495, mcc: 0.7295, roc_auc: 0.9731 },
    fedavg:  { acc: 85.49, bal_acc: 56.67, macro_f1: 0.4422, weighted_f1: 0.8577, mcc: 0.6809, roc_auc: 0.9792 }
  }
};

// 10-Round Convergence Learning Curves
const CONVERGENCE_SERIES = {
  rounds: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  fedprox_acc: [72.1, 78.4, 83.2, 86.0, 87.9, 89.1, 89.8, 90.3, 90.7, 90.93],
  fedavg_acc:  [68.4, 72.8, 75.3, 77.9, 79.1, 80.0, 80.6, 81.0, 81.2, 81.47],
  fedprox_loss: [0.65, 0.48, 0.38, 0.31, 0.26, 0.23, 0.20, 0.18, 0.17, 0.16],
  fedavg_loss:  [0.72, 0.59, 0.50, 0.44, 0.40, 0.38, 0.36, 0.34, 0.33, 0.32]
};

// Application State
const state = {
  currentView: 'overview',
  theme: localStorage.getItem('flex-id-theme') || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'),
  models: [],
  activeModel: 'fedprox-v1',
  recentPredictions: [],
  benchmarkMode: 'no_attack',
};

// ==============================================================================
// Theme & Navigation
// ==============================================================================
function initTheme() {
  document.documentElement.setAttribute('data-theme', state.theme);
  updateThemeButton();
}

function toggleTheme() {
  state.theme = state.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', state.theme);
  localStorage.setItem('flex-id-theme', state.theme);
  updateThemeButton();
  renderConvergenceChart();
}

function updateThemeButton() {
  const btn = document.getElementById('theme-toggle-btn');
  if (!btn) return;
  btn.innerHTML = state.theme === 'dark'
    ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>`
    : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>`;
  btn.setAttribute('aria-label', `Switch to ${state.theme === 'dark' ? 'light' : 'dark'} mode`);
}

function navigateTo(viewId) {
  const views = document.querySelectorAll('.content-view');
  const navItems = document.querySelectorAll('.nav-item');
  const pageTitle = document.getElementById('page-title');

  const titles = {
    overview: 'System Overview & Benchmarks',
    predictions: 'Network Intrusion Detection',
    experiments: 'Federated Learning Experiments',
    explanations: 'Model Explainability (SHAP)',
    models: 'Registered Model Checkpoints'
  };

  views.forEach(v => v.classList.remove('active'));
  navItems.forEach(n => n.classList.remove('active'));

  const targetView = document.getElementById(`view-${viewId}`);
  const targetNav = document.querySelector(`.nav-item[data-view="${viewId}"]`);

  if (targetView) targetView.classList.add('active');
  if (targetNav) targetNav.classList.add('active');
  if (pageTitle && titles[viewId]) pageTitle.textContent = titles[viewId];

  state.currentView = viewId;
  window.location.hash = viewId;

  // Trigger lazy renders
  if (viewId === 'overview' || viewId === 'experiments') {
    renderConvergenceChart();
  }
}

// Toast Notifications
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = 'toast';

  const icons = {
    success: `<span style="color: var(--success);">&#x2713;</span>`,
    danger: `<span style="color: var(--danger);">&#x26A0;</span>`,
    info: `<span style="color: var(--accent);">&#x2139;</span>`,
  };

  toast.innerHTML = `${icons[type] || icons.info} <span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(8px)';
    toast.style.transition = 'all 200ms ease';
    setTimeout(() => toast.remove(), 200);
  }, 4000);
}

// ==============================================================================
// 1. Overview View
// ==============================================================================
async function loadOverviewData() {
  try {
    const health = await window.api.getHealth();
    const statusDot = document.getElementById('global-status-dot');
    const statusText = document.getElementById('global-status-text');

    if (health.status === 'healthy') {
      if (statusDot) statusDot.className = 'status-dot';
      if (statusText) statusText.textContent = `Online (${health.active_model || 'fedprox-v1'})`;
    } else {
      if (statusDot) statusDot.className = 'status-dot danger';
      if (statusText) statusText.textContent = 'Degraded';
    }

    state.activeModel = health.active_model || 'fedprox-v1';
  } catch (err) {
    const statusDot = document.getElementById('global-status-dot');
    const statusText = document.getElementById('global-status-text');
    if (statusDot) statusDot.className = 'status-dot danger';
    if (statusText) statusText.textContent = 'Disconnected';
  }

  // Load models for dropdowns
  try {
    const modelsData = await window.api.getModels();
    state.models = modelsData.models || [];
    populateModelSelectors();
  } catch (err) {
    console.warn('Could not load models list:', err);
  }
}

function renderBenchmarkCards() {
  const data = VERIFIED_BENCHMARKS[state.benchmarkMode];
  if (!data) return;

  const accEl = document.getElementById('kpi-accuracy');
  const f1El = document.getElementById('kpi-f1');
  const deltaEl = document.getElementById('kpi-delta');

  if (accEl) accEl.textContent = `${data.fedprox.acc.toFixed(2)}%`;
  if (f1El) f1El.textContent = data.fedprox.macro_f1.toFixed(4);
  if (deltaEl) {
    const diff = (data.fedprox.acc - data.fedavg.acc).toFixed(2);
    deltaEl.textContent = `+${diff}% vs FedAvg`;
  }
}

// ==============================================================================
// 2. Prediction Experience
// ==============================================================================
function populateFeaturesForm(presetKey = 'benign') {
  const container = document.getElementById('features-inputs-container');
  if (!container) return;

  const preset = FEATURE_PRESETS[presetKey];
  const values = preset ? preset.values : new Array(29).fill(0);

  container.innerHTML = '';
  CANONICAL_FEATURES.forEach((featureName, idx) => {
    const val = values[idx] !== undefined ? values[idx] : 0.0;
    const field = document.createElement('div');
    field.className = 'feature-field';
    field.innerHTML = `
      <label title="${featureName}">${featureName}</label>
      <input type="number" step="any" id="feat-${idx}" name="${featureName}" value="${val}" required />
    `;
    container.appendChild(field);
  });

  const descEl = document.getElementById('preset-description');
  if (descEl && preset) {
    descEl.textContent = preset.desc;
  }
}

function getEnteredFeatures() {
  const features = [];
  for (let i = 0; i < 29; i++) {
    const input = document.getElementById(`feat-${i}`);
    features.push(input ? parseFloat(input.value) || 0.0 : 0.0);
  }
  return features;
}

async function handleSinglePrediction(e) {
  e.preventDefault();
  const btn = document.getElementById('btn-run-prediction');
  const resultsContainer = document.getElementById('prediction-result-card');
  const features = getEnteredFeatures();
  const modelSelect = document.getElementById('prediction-model-select');
  const modelVersion = modelSelect ? modelSelect.value : state.activeModel;

  if (btn) {
    btn.disabled = true;
    btn.textContent = 'Classifying flow...';
  }

  try {
    const res = await window.api.predictSingle(features, modelVersion);
    renderPredictionResult(res);
    showToast(`Traffic analyzed: ${res.predicted_class}`, res.is_intrusion ? 'danger' : 'success');
  } catch (err) {
    showToast(`Classification failed: ${err.message}`, 'danger');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = 'Execute Detection';
    }
  }
}

function renderPredictionResult(res) {
  const card = document.getElementById('prediction-result-card');
  if (!card) return;

  card.style.display = 'block';

  const badgeClass = res.is_intrusion ? 'badge-danger' : 'badge-success';
  const statusIcon = res.is_intrusion ? '&#x26A0; Intrusion Detected' : '&#x2713; Normal Operation';

  const confPct = (res.confidence * 100).toFixed(2);

  let probsHtml = '';
  if (res.class_probabilities) {
    const sorted = Object.entries(res.class_probabilities)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);

    probsHtml = sorted.map(([cls, prob]) => {
      const p = (prob * 100).toFixed(1);
      const isTop = cls === res.predicted_class;
      return `
        <div style="margin-bottom: 8px;">
          <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;">
            <span style="font-weight: ${isTop ? '600' : '400'}; color: var(--text-primary);">${cls}</span>
            <span class="tabular-nums" style="color: var(--text-secondary);">${p}%</span>
          </div>
          <div style="height: 6px; background: var(--bg-tertiary); border-radius: var(--radius-pill); overflow: hidden;">
            <div style="height: 100%; width: ${p}%; background: ${isTop ? (res.is_intrusion ? 'var(--danger)' : 'var(--success)') : 'var(--text-tertiary)'}; border-radius: var(--radius-pill);"></div>
          </div>
        </div>
      `;
    }).join('');
  }

  card.innerHTML = `
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">
      <span class="badge ${badgeClass}" style="font-size: 14px; padding: 6px 14px;">${statusIcon}</span>
      <span class="latency-badge" style="font-size: 12px; color: var(--text-tertiary);">Inference: ${res.latency_ms.toFixed(2)} ms</span>
    </div>

    <div style="margin-bottom: 20px;">
      <h3 style="font-size: 24px; font-weight: 700; color: var(--text-primary); margin-bottom: 4px;">${res.predicted_class}</h3>
      <p style="font-size: 13.5px; color: var(--text-secondary);">Model: <strong>${res.model_version}</strong> · Softmax Confidence: <strong>${confPct}%</strong></p>
    </div>

    <div style="margin-bottom: 24px;">
      <h4 style="font-size: 12px; font-weight: 600; text-transform: uppercase; color: var(--text-tertiary); margin-bottom: 10px;">Top Predicted Class Probabilities</h4>
      ${probsHtml}
    </div>

    <div style="display: flex; gap: 10px; border-top: 1px solid var(--border-subtle); padding-top: 16px;">
      <button class="btn btn-secondary btn-sm" onclick="navigateToExplanationFromPrediction()">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><path d="M12 16v-4"></path><path d="M12 8h.01"></path></svg>
        Inspect Feature Attributions (SHAP)
      </button>
    </div>
  `;

  // Store for explanation transition
  window._lastPredictionFeatures = getEnteredFeatures();
  window._lastPredictionModel = res.model_version;
}

function navigateToExplanationFromPrediction() {
  if (window._lastPredictionFeatures) {
    navigateTo('explanations');
    loadShapExplanation(window._lastPredictionFeatures, window._lastPredictionModel);
  }
}

// ==============================================================================
// 3. Experiments & Convergence Curves (Native SVG)
// ==============================================================================
async function loadExperiments() {
  const tbody = document.getElementById('experiments-table-body');
  if (!tbody) return;

  try {
    const data = await window.api.getExperiments();
    const exps = data.experiments || [];

    if (exps.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-tertiary); padding: 24px;">No experiment runs recorded.</td></tr>`;
      return;
    }

    tbody.innerHTML = exps.map(exp => {
      const statusBadge = exp.status === 'COMPLETED'
        ? `<span class="badge badge-success">Completed</span>`
        : `<span class="badge badge-warning">Running</span>`;

      return `
        <tr>
          <td style="font-weight: 600;">${exp.name}</td>
          <td><span class="badge badge-accent">${exp.strategy.toUpperCase()}</span></td>
          <td class="tabular-nums">${exp.total_rounds}</td>
          <td class="tabular-nums">${exp.clients_count}</td>
          <td>${statusBadge}</td>
          <td class="tabular-nums" style="color: var(--text-secondary); font-size: 12px;">${new Date(exp.created_at).toLocaleTimeString()}</td>
        </tr>
      `;
    }).join('');
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--danger); padding: 20px;">Failed to load experiments.</td></tr>`;
  }
}

function renderConvergenceChart() {
  const container = document.getElementById('convergence-chart-svg');
  if (!container) return;

  const isDark = state.theme === 'dark';
  const gridColor = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)';
  const textColor = isDark ? '#a1a1a6' : '#6e6e73';
  const fedproxColor = '#0071e3';
  const fedavgColor = isDark ? '#a1a1a6' : '#86868b';

  const w = 700;
  const h = 240;
  const pad = { top: 20, right: 30, bottom: 30, left: 50 };

  const series = CONVERGENCE_SERIES;
  const minAcc = 65;
  const maxAcc = 95;

  const getX = (round) => pad.left + ((round - 1) / 9) * (w - pad.left - pad.right);
  const getY = (val) => pad.top + (1 - (val - minAcc) / (maxAcc - minAcc)) * (h - pad.top - pad.bottom);

  // Build SVG Path
  const fedproxPoints = series.rounds.map((r, i) => `${getX(r)},${getY(series.fedprox_acc[i])}`).join(' L ');
  const fedavgPoints = series.rounds.map((r, i) => `${getX(r)},${getY(series.fedavg_acc[i])}`).join(' L ');

  container.innerHTML = `
    <svg viewBox="0 0 ${w} ${h}" style="width: 100%; height: 100%;">
      <!-- Grid Lines -->
      ${[70, 80, 90].map(v => `
        <line x1="${pad.left}" y1="${getY(v)}" x2="${w - pad.right}" y2="${getY(v)}" stroke="${gridColor}" stroke-dasharray="3,3" />
        <text x="${pad.left - 8}" y="${getY(v) + 4}" fill="${textColor}" font-size="11" text-anchor="end">${v}%</text>
      `).join('')}

      <!-- X-axis labels -->
      ${series.rounds.map(r => `
        <text x="${getX(r)}" y="${h - 8}" fill="${textColor}" font-size="11" text-anchor="middle">R${r}</text>
      `).join('')}

      <!-- FedAvg Curve -->
      <path d="M ${fedavgPoints}" fill="none" stroke="${fedavgColor}" stroke-width="2" stroke-dasharray="4,4" />
      ${series.rounds.map((r, i) => `
        <circle cx="${getX(r)}" cy="${getY(series.fedavg_acc[i])}" r="3.5" fill="${fedavgColor}" />
      `).join('')}

      <!-- FedProx Curve -->
      <path d="M ${fedproxPoints}" fill="none" stroke="${fedproxColor}" stroke-width="2.5" />
      ${series.rounds.map((r, i) => `
        <circle cx="${getX(r)}" cy="${getY(series.fedprox_acc[i])}" r="4" fill="${fedproxColor}" />
      `).join('')}
    </svg>
  `;
}

// ==============================================================================
// 4. Explainable AI (SHAP)
// ==============================================================================
async function loadShapExplanation(features = null, modelVersion = null) {
  const container = document.getElementById('shap-attributions-container');
  const summaryEl = document.getElementById('shap-summary-text');
  if (!container) return;

  container.innerHTML = `<div style="text-align: center; color: var(--text-tertiary); padding: 30px;">Computing feature attributions...</div>`;

  const featArray = features || FEATURE_PRESETS.dos_hulk.values;
  const version = modelVersion || state.activeModel;

  try {
    const res = await window.api.getExplanation(featArray, version, 8);

    if (summaryEl) {
      summaryEl.textContent = res.summary;
    }

    const items = res.top_contributing_features || [];
    const maxVal = Math.max(...items.map(x => Math.abs(x.importance)), 0.01);

    container.innerHTML = items.map(item => {
      const isPos = item.direction === 'positive';
      const pct = Math.min(100, (Math.abs(item.importance) / maxVal) * 100);

      return `
        <div class="attribution-bar-row">
          <div class="attribution-label" title="${item.feature}">${item.feature}</div>
          <div class="attribution-track">
            <div class="attribution-fill ${isPos ? 'positive' : 'negative'}" style="width: ${pct}%;"></div>
          </div>
          <div class="attribution-val">${isPos ? '+' : '-'}${Math.abs(item.importance).toFixed(3)}</div>
        </div>
      `;
    }).join('');

    renderGlobalShapBaseline(res.global_importance_baseline);
  } catch (err) {
    container.innerHTML = `<div style="color: var(--danger); padding: 20px;">Failed to generate explanation: ${err.message}</div>`;
  }
}

function renderGlobalShapBaseline(baseline) {
  const tbody = document.getElementById('global-shap-table-body');
  if (!tbody || !baseline) return;

  const rows = Object.entries(baseline)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8);

  tbody.innerHTML = rows.map(([feat, score], i) => `
    <tr>
      <td class="tabular-nums" style="color: var(--text-tertiary);">${i + 1}</td>
      <td style="font-weight: 500;">${feat}</td>
      <td class="tabular-nums" style="text-align: right;">${score.toFixed(1)}</td>
    </tr>
  `).join('');
}

// ==============================================================================
// 5. Models Management
// ==============================================================================
async function loadModelsView() {
  const container = document.getElementById('models-grid-container');
  if (!container) return;

  try {
    const data = await window.api.getModels();
    const models = data.models || [];

    container.innerHTML = models.map(m => {
      const isActive = m.model_version === state.activeModel;
      return `
        <div class="card" style="${isActive ? 'border-color: var(--accent);' : ''}">
          <div class="card-header">
            <div>
              <div style="font-size: 16px; font-weight: 600;">${m.model_version}</div>
              <div style="font-size: 12px; color: var(--text-tertiary);">Strategy: ${m.strategy.toUpperCase()}</div>
            </div>
            ${isActive ? '<span class="badge badge-accent">Active</span>' : ''}
          </div>

          <div style="margin: 16px 0; font-size: 13px; color: var(--text-secondary); line-height: 1.6;">
            <div>Parameters: <strong class="tabular-nums">52,941</strong></div>
            <div>Architecture: <strong>DNN (256-128-64-13)</strong></div>
            <div>Verified Test Acc: <strong class="tabular-nums">${(m.accuracy * 100).toFixed(2)}%</strong></div>
            <div>Federated Rounds: <strong class="tabular-nums">${m.rounds}</strong></div>
          </div>

          <div style="border-top: 1px solid var(--border-subtle); padding-top: 14px;">
            ${isActive
              ? '<span class="badge badge-success">Currently Serving Traffic</span>'
              : `<button class="btn btn-secondary btn-sm" onclick="handleActivateModel('${m.model_version}')">Set as Default Active</button>`
            }
          </div>
        </div>
      `;
    }).join('');
  } catch (err) {
    container.innerHTML = `<div style="color: var(--danger); padding: 20px;">Could not load models list.</div>`;
  }
}

async function handleActivateModel(version) {
  try {
    await window.api.activateModel(version);
    state.activeModel = version;
    showToast(`Model ${version} set as active default`, 'success');
    loadModelsView();
    loadOverviewData();
  } catch (err) {
    showToast(`Activation failed: ${err.message}`, 'danger');
  }
}

function populateModelSelectors() {
  const selects = ['prediction-model-select', 'explanation-model-select'];
  selects.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = state.models.map(m => `
      <option value="${m.model_version}" ${m.model_version === state.activeModel ? 'selected' : ''}>
        ${m.model_version} (${m.strategy.toUpperCase()})
      </option>
    `).join('');
  });
}

// ==============================================================================
// Global Initialization & Listeners
// ==============================================================================
document.addEventListener('DOMContentLoaded', () => {
  initTheme();

  // Navigation Click Handlers
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const view = item.getAttribute('data-view');
      if (view) navigateTo(view);
    });
  });

  // Theme Toggle Button
  const themeBtn = document.getElementById('theme-toggle-btn');
  if (themeBtn) themeBtn.addEventListener('click', toggleTheme);

  // Preset Selector Buttons
  document.querySelectorAll('.btn-preset').forEach(btn => {
    btn.addEventListener('click', () => {
      const presetKey = btn.getAttribute('data-preset');
      document.querySelectorAll('.btn-preset').forEach(b => b.classList.remove('btn-primary'));
      btn.classList.add('btn-primary');
      populateFeaturesForm(presetKey);
    });
  });

  // Prediction Form Handler
  const predForm = document.getElementById('prediction-form');
  if (predForm) predForm.addEventListener('submit', handleSinglePrediction);

  // Benchmark Tab Toggle (Clean vs Under Attack)
  document.querySelectorAll('.btn-benchmark-mode').forEach(btn => {
    btn.addEventListener('click', () => {
      const mode = btn.getAttribute('data-mode');
      document.querySelectorAll('.btn-benchmark-mode').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.benchmarkMode = mode;
      renderBenchmarkCards();
    });
  });

  // Trigger SHAP Explanation Button
  const btnExplain = document.getElementById('btn-run-explanation');
  if (btnExplain) {
    btnExplain.addEventListener('click', () => {
      const featSelect = document.getElementById('explanation-preset-select');
      const key = featSelect ? featSelect.value : 'dos_hulk';
      loadShapExplanation(FEATURE_PRESETS[key].values);
    });
  }

  // Handle URL hash routing
  const initialHash = window.location.hash.replace('#', '') || 'overview';
  navigateTo(initialHash);

  // Load Initial API Data
  populateFeaturesForm('benign');
  loadOverviewData();
  renderBenchmarkCards();
  loadExperiments();
  loadModelsView();
});
