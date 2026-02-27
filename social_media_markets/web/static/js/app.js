/* ═══════════════════════════════════════════════════════════════
   Social Media Prediction Markets — Frontend Application
   ═══════════════════════════════════════════════════════════════ */

// ── State ─────────────────────────────────────────────────────

const state = {
  demoData: {},        // cached demo data per scenario
  currentScenario: {
    predict: 'linear',
    growth: 'linear',
    simulate: 'exponential',
    edge: 'exponential',
  },
  charts: {},          // Chart.js instances
};

// ── Chart Color Palette ───────────────────────────────────────

const COLORS = {
  blue:      'rgba(59, 130, 246, 1)',
  blueFill:  'rgba(59, 130, 246, 0.15)',
  purple:    'rgba(139, 92, 246, 1)',
  green:     'rgba(16, 185, 129, 1)',
  greenFill: 'rgba(16, 185, 129, 0.1)',
  red:       'rgba(239, 68, 68, 1)',
  redFill:   'rgba(239, 68, 68, 0.1)',
  amber:     'rgba(245, 158, 11, 1)',
  amberFill: 'rgba(245, 158, 11, 0.1)',
  cyan:      'rgba(6, 182, 212, 1)',
  cyanFill:  'rgba(6, 182, 212, 0.1)',
  gray:      'rgba(148, 163, 184, 0.5)',
  grayFill:  'rgba(148, 163, 184, 0.05)',
  white:     'rgba(241, 245, 249, 0.8)',
};

const MODEL_COLORS = {
  linear:      COLORS.blue,
  exponential: COLORS.green,
  logistic:    COLORS.purple,
  viral:       COLORS.red,
  plateau:     COLORS.amber,
  declining:   COLORS.cyan,
};

// ── Chart.js Global Config ────────────────────────────────────

Chart.defaults.color = 'rgba(148, 163, 184, 0.8)';
Chart.defaults.borderColor = 'rgba(30, 42, 69, 0.5)';
Chart.defaults.font.family = "'Inter', -apple-system, sans-serif";
Chart.defaults.font.size = 11;

// ── Tab Navigation ────────────────────────────────────────────

document.querySelectorAll('.nav-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(`panel-${tab.dataset.tab}`).classList.add('active');
  });
});

// ── Scenario Pill Selection ───────────────────────────────────

document.querySelectorAll('.scenario-pills').forEach(container => {
  container.querySelectorAll('.scenario-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      container.querySelectorAll('.scenario-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      const tabId = container.id.replace('-scenarios', '');
      const mapTab = { predict: 'predict', growth: 'growth', sim: 'simulate', edge: 'edge' };
      state.currentScenario[mapTab[tabId] || tabId] = pill.dataset.scenario;
    });
  });
});

// ── API Helper ────────────────────────────────────────────────

async function apiCall(endpoint, body) {
  const resp = await fetch(`/api/${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.error || `API error: ${resp.status}`);
  }
  return resp.json();
}

async function getDemoData(scenario) {
  if (state.demoData[scenario]) return state.demoData[scenario];
  const resp = await apiCall('demo-data', { scenario });
  state.demoData[scenario] = resp.data;
  return resp.data;
}

// ── Loading Helpers ───────────────────────────────────────────

function showLoading(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add('active');
}

function hideLoading(id) {
  const el = document.getElementById(id);
  if (el) el.classList.remove('active');
}

// ── Number Formatting ─────────────────────────────────────────

function fmtNum(n) {
  if (n === null || n === undefined) return '--';
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(1) + 'B';
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toFixed(0);
}

function fmtPct(n) {
  if (n === null || n === undefined) return '--%';
  return (n * 100).toFixed(1) + '%';
}

// ── Gauge Animation ───────────────────────────────────────────

function setGauge(value) {
  const circumference = 2 * Math.PI * 85;
  const arc = document.getElementById('gauge-arc');
  const valEl = document.getElementById('gauge-value');

  const offset = circumference * (1 - value);
  arc.style.strokeDasharray = circumference;
  arc.style.strokeDashoffset = offset;

  // Color based on probability
  if (value >= 0.7) {
    arc.style.stroke = COLORS.green;
  } else if (value >= 0.4) {
    arc.style.stroke = COLORS.blue;
  } else if (value >= 0.2) {
    arc.style.stroke = COLORS.amber;
  } else {
    arc.style.stroke = COLORS.red;
  }

  valEl.textContent = fmtPct(value);
  valEl.style.color = arc.style.stroke;
}

// ══════════════════════════════════════════════════════════════
// PREDICT TAB
// ══════════════════════════════════════════════════════════════

async function runPredict() {
  const btn = document.getElementById('btn-predict');
  btn.disabled = true;
  showLoading('predict-loading');

  try {
    const scenario = state.currentScenario.predict;
    const data = await getDemoData(scenario);

    const body = {
      entity_name: document.getElementById('predict-entity').value || 'Entity',
      platform: document.getElementById('predict-platform').value,
      metric_type: document.getElementById('predict-metric').value,
      target_value: parseFloat(document.getElementById('predict-target').value) || 100000,
      deadline_days: parseInt(document.getElementById('predict-deadline').value) || null,
      market_price: parseFloat(document.getElementById('predict-market').value) || null,
      historical_data: data,
      n_simulations: 5000,
    };

    const result = await apiCall('predict', body);

    // Update stats
    document.getElementById('stat-probability').textContent = fmtPct(result.probability);
    document.getElementById('stat-ci').textContent =
      `${fmtPct(result.confidence_interval[0])} - ${fmtPct(result.confidence_interval[1])}`;

    const edgeEl = document.getElementById('stat-edge');
    if (result.edge && result.edge !== 0) {
      const edgeSign = result.edge > 0 ? '+' : '';
      edgeEl.textContent = edgeSign + fmtPct(result.edge);
      edgeEl.className = 'stat-value ' + (result.edge > 0.05 ? 'positive' : result.edge < -0.05 ? 'negative' : 'neutral');
    } else {
      edgeEl.textContent = 'N/A';
      edgeEl.className = 'stat-value';
    }

    document.getElementById('stat-pattern').textContent =
      result.growth_pattern ? result.growth_pattern.charAt(0).toUpperCase() + result.growth_pattern.slice(1) : '--';

    // Update gauge
    setGauge(result.probability);

    // Update chart
    renderPredictChart(data);

    // Update reasoning
    const parts = result.reasoning.split(' | ').filter(Boolean);
    const reasoningHtml = parts.map(p => `<div style="margin-bottom: 6px;">&bull; ${p}</div>`).join('');
    document.getElementById('predict-reasoning').innerHTML = reasoningHtml || 'No detailed reasoning available.';

  } catch (err) {
    console.error(err);
    document.getElementById('predict-reasoning').innerHTML =
      `<span style="color: var(--accent-red);">Error: ${err.message}</span>`;
  } finally {
    btn.disabled = false;
    hideLoading('predict-loading');
  }
}

function renderPredictChart(data) {
  const ctx = document.getElementById('predict-chart');
  if (state.charts.predict) state.charts.predict.destroy();

  const labels = data.map(d => d.date);
  const values = data.map(d => d.value);

  state.charts.predict = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Observed',
        data: values,
        borderColor: COLORS.blue,
        backgroundColor: COLORS.blueFill,
        fill: true,
        borderWidth: 2,
        pointRadius: 0,
        pointHitRadius: 8,
        tension: 0.3,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(17, 24, 39, 0.95)',
          borderColor: 'rgba(59, 130, 246, 0.3)',
          borderWidth: 1,
          padding: 12,
          callbacks: {
            label: ctx => `${ctx.dataset.label}: ${fmtNum(ctx.parsed.y)}`,
          },
        },
      },
      scales: {
        x: {
          ticks: { maxTicksLimit: 8, maxRotation: 0 },
          grid: { display: false },
        },
        y: {
          ticks: { callback: v => fmtNum(v) },
          grid: { color: 'rgba(30, 42, 69, 0.3)' },
        },
      },
    },
  });
}

// ══════════════════════════════════════════════════════════════
// GROWTH CURVES TAB
// ══════════════════════════════════════════════════════════════

async function runGrowthFit() {
  showLoading('growth-loading');

  try {
    const scenario = state.currentScenario.growth;
    const data = await getDemoData(scenario);

    const result = await apiCall('fit', { data_points: data, entity_name: 'demo' });
    const fits = result.fits;

    // Render chart
    renderGrowthChart(data, fits);

    // Render model list
    renderGrowthModelList(fits);

    // Render table
    renderGrowthTable(fits);

    // Best badge
    if (fits.length > 0) {
      document.getElementById('growth-best-badge').innerHTML =
        `<span class="badge badge-green">Best: ${fits[0].pattern}</span>`;
    }

  } catch (err) {
    console.error(err);
  } finally {
    hideLoading('growth-loading');
  }
}

function renderGrowthChart(data, fits) {
  const ctx = document.getElementById('growth-chart');
  if (state.charts.growth) state.charts.growth.destroy();

  const datasets = [{
    label: 'Observed Data',
    data: data.map(d => ({ x: d.date, y: d.value })),
    borderColor: COLORS.white,
    backgroundColor: 'rgba(241, 245, 249, 0.05)',
    fill: false,
    borderWidth: 2,
    pointRadius: 1,
    pointHitRadius: 6,
    tension: 0.1,
    order: 10,
  }];

  fits.forEach((fit, i) => {
    const color = MODEL_COLORS[fit.pattern] || COLORS.gray;
    datasets.push({
      label: `${fit.pattern} (R²=${fit.r_squared.toFixed(3)})`,
      data: fit.prediction.dates.map((d, j) => ({ x: d, y: fit.prediction.values[j] })),
      borderColor: color,
      backgroundColor: 'transparent',
      borderWidth: i === 0 ? 3 : 1.5,
      borderDash: i === 0 ? [] : [6, 4],
      pointRadius: 0,
      tension: 0.3,
      order: i,
    });
  });

  state.charts.growth = new Chart(ctx, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          position: 'top',
          labels: { usePointStyle: true, padding: 16 },
        },
        tooltip: {
          backgroundColor: 'rgba(17, 24, 39, 0.95)',
          borderColor: 'rgba(59, 130, 246, 0.3)',
          borderWidth: 1,
          padding: 12,
          callbacks: {
            label: ctx => `${ctx.dataset.label}: ${fmtNum(ctx.parsed.y)}`,
          },
        },
      },
      scales: {
        x: {
          type: 'category',
          labels: fits[0]?.prediction?.dates || data.map(d => d.date),
          ticks: { maxTicksLimit: 8, maxRotation: 0 },
          grid: { display: false },
        },
        y: {
          ticks: { callback: v => fmtNum(v) },
          grid: { color: 'rgba(30, 42, 69, 0.3)' },
        },
      },
    },
  });
}

function renderGrowthModelList(fits) {
  const container = document.getElementById('growth-model-list');
  container.innerHTML = fits.map((fit, i) => {
    const color = MODEL_COLORS[fit.pattern] || COLORS.gray;
    const badgeClass = i === 0 ? 'badge-green' : 'badge-blue';
    return `
      <div class="signal-row" style="border-left: 3px solid ${color};">
        <div class="signal-content">
          <div class="signal-name" style="color: ${color};">
            ${fit.pattern.charAt(0).toUpperCase() + fit.pattern.slice(1)}
            ${i === 0 ? '<span class="badge badge-green" style="margin-left: 8px;">Best</span>' : ''}
          </div>
          <div class="signal-explanation">
            R² = ${fit.r_squared.toFixed(4)} &middot; AIC = ${fit.aic.toFixed(1)} &middot; &#x3c3; = ${fmtNum(fit.residual_std)}
          </div>
        </div>
      </div>`;
  }).join('');
}

function renderGrowthTable(fits) {
  const tbody = document.querySelector('#growth-table tbody');
  tbody.innerHTML = fits.map(fit => `
    <tr>
      <td><span class="badge badge-purple">${fit.pattern}</span></td>
      <td class="mono">${fit.r_squared.toFixed(4)}</td>
      <td class="mono">${fit.aic.toFixed(1)}</td>
      <td class="mono">${fmtNum(fit.residual_std)}</td>
      <td class="mono" style="font-size: 0.75rem;">
        ${Object.entries(fit.parameters).map(([k,v]) => `${k}=${v.toFixed(3)}`).join(', ')}
      </td>
    </tr>
  `).join('');
}

// ══════════════════════════════════════════════════════════════
// SIMULATE TAB
// ══════════════════════════════════════════════════════════════

async function runSimulation() {
  showLoading('sim-loading');

  try {
    const scenario = state.currentScenario.simulate;
    const data = await getDemoData(scenario);
    const horizon = parseInt(document.getElementById('sim-horizon').value) || 365;
    const nSim = parseInt(document.getElementById('sim-count').value) || 2000;
    const target = parseFloat(document.getElementById('sim-target').value) || null;

    const result = await apiCall('simulate', {
      data_points: data,
      entity_name: 'demo',
      horizon_days: horizon,
      n_simulations: nSim,
      target_value: target,
    });

    renderSimChart(data, result);
    renderSimStats(result);

  } catch (err) {
    console.error(err);
  } finally {
    hideLoading('sim-loading');
  }
}

function renderSimChart(historicalData, simResult) {
  const ctx = document.getElementById('sim-chart');
  if (state.charts.sim) state.charts.sim.destroy();

  const dates = simResult.dates;
  const p5 = simResult.percentiles['5'];
  const p25 = simResult.percentiles['25'];
  const p50 = simResult.percentiles['50'];
  const p75 = simResult.percentiles['75'];
  const p95 = simResult.percentiles['95'];

  const allLabels = historicalData.map(d => d.date).concat(dates);

  const historicalValues = historicalData.map(d => d.value);
  const pad = new Array(dates.length).fill(null);
  const histPad = new Array(historicalData.length).fill(null);

  const datasets = [
    {
      label: 'Historical',
      data: historicalValues.concat(pad),
      borderColor: COLORS.white,
      backgroundColor: 'transparent',
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.3,
      order: 0,
    },
    {
      label: 'P50 (Median)',
      data: histPad.concat(p50),
      borderColor: COLORS.blue,
      backgroundColor: 'transparent',
      borderWidth: 2.5,
      pointRadius: 0,
      tension: 0.3,
      order: 1,
    },
    {
      label: 'P25-P75',
      data: histPad.concat(p75),
      borderColor: 'transparent',
      backgroundColor: COLORS.blueFill,
      fill: '+1',
      borderWidth: 0,
      pointRadius: 0,
      tension: 0.3,
      order: 3,
    },
    {
      label: 'P25',
      data: histPad.concat(p25),
      borderColor: COLORS.blue,
      backgroundColor: 'transparent',
      borderWidth: 1,
      borderDash: [4, 4],
      pointRadius: 0,
      tension: 0.3,
      order: 2,
    },
    {
      label: 'P5-P95',
      data: histPad.concat(p95),
      borderColor: 'transparent',
      backgroundColor: 'rgba(59, 130, 246, 0.06)',
      fill: '+1',
      borderWidth: 0,
      pointRadius: 0,
      tension: 0.3,
      order: 5,
    },
    {
      label: 'P5',
      data: histPad.concat(p5),
      borderColor: 'rgba(59, 130, 246, 0.3)',
      backgroundColor: 'transparent',
      borderWidth: 1,
      borderDash: [2, 4],
      pointRadius: 0,
      tension: 0.3,
      order: 4,
    },
  ];

  // Add target line if present
  const target = parseFloat(document.getElementById('sim-target').value);
  if (target && !isNaN(target)) {
    datasets.push({
      label: `Target: ${fmtNum(target)}`,
      data: new Array(allLabels.length).fill(target),
      borderColor: COLORS.red,
      borderWidth: 2,
      borderDash: [8, 4],
      pointRadius: 0,
      fill: false,
      order: 0,
    });
  }

  state.charts.sim = new Chart(ctx, {
    type: 'line',
    data: { labels: allLabels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          position: 'top',
          labels: { usePointStyle: true, padding: 14, filter: item => !item.text.startsWith('P25-') && !item.text.startsWith('P5-') },
        },
        tooltip: {
          backgroundColor: 'rgba(17, 24, 39, 0.95)',
          borderColor: 'rgba(59, 130, 246, 0.3)',
          borderWidth: 1,
          padding: 12,
          callbacks: {
            label: ctx => {
              if (ctx.parsed.y === null) return null;
              return `${ctx.dataset.label}: ${fmtNum(ctx.parsed.y)}`;
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { maxTicksLimit: 10, maxRotation: 0 },
          grid: { display: false },
        },
        y: {
          ticks: { callback: v => fmtNum(v) },
          grid: { color: 'rgba(30, 42, 69, 0.3)' },
        },
      },
    },
  });

  // Target badge
  if (simResult.target_probability !== undefined) {
    document.getElementById('sim-target-badge').innerHTML =
      `<span class="badge ${simResult.target_probability > 0.5 ? 'badge-green' : 'badge-amber'}">
        P(target) = ${fmtPct(simResult.target_probability)}
      </span>`;
  } else {
    document.getElementById('sim-target-badge').innerHTML = '';
  }
}

function renderSimStats(result) {
  const container = document.getElementById('sim-stats');
  const p50Final = result.percentiles['50'].slice(-1)[0];
  const p5Final = result.percentiles['5'].slice(-1)[0];
  const p95Final = result.percentiles['95'].slice(-1)[0];

  let html = `
    <div class="signal-row">
      <div class="signal-content">
        <div class="signal-name" style="color: var(--accent-blue-light);">Median Final Value</div>
        <div class="signal-explanation">${fmtNum(p50Final)}</div>
      </div>
    </div>
    <div class="signal-row">
      <div class="signal-content">
        <div class="signal-name" style="color: var(--text-secondary);">90% Range</div>
        <div class="signal-explanation">${fmtNum(p5Final)} &mdash; ${fmtNum(p95Final)}</div>
      </div>
    </div>
    <div class="signal-row">
      <div class="signal-content">
        <div class="signal-name" style="color: var(--accent-purple);">Growth Pattern</div>
        <div class="signal-explanation">${result.growth_pattern}</div>
      </div>
    </div>`;

  if (result.viral_stats) {
    html += `
    <div class="signal-row">
      <div class="signal-content">
        <div class="signal-name" style="color: var(--accent-red-light);">Viral Probability</div>
        <div class="signal-explanation">${fmtPct(result.viral_stats.viral_probability)} per event &middot; &#x3b1; = ${result.viral_stats.power_law_alpha?.toFixed(2) || '--'}</div>
      </div>
    </div>`;
  }

  if (result.target_probability !== undefined) {
    html += `
    <div class="signal-row" style="border-left: 3px solid var(--accent-green);">
      <div class="signal-content">
        <div class="signal-name" style="color: var(--accent-green-light);">Target Probability</div>
        <div class="signal-explanation">${fmtPct(result.target_probability)}</div>
      </div>
    </div>`;
  }

  container.innerHTML = html;
}

// ══════════════════════════════════════════════════════════════
// EDGE DETECTOR TAB
// ══════════════════════════════════════════════════════════════

async function runEdge() {
  showLoading('edge-loading');

  try {
    const scenario = state.currentScenario.edge;
    const data = await getDemoData(scenario);

    const result = await apiCall('edge', {
      entity_name: document.getElementById('edge-entity').value || 'Entity',
      target_value: parseFloat(document.getElementById('edge-target').value) || 200000,
      market_price: parseFloat(document.getElementById('edge-market').value) || 0.5,
      data_points: data,
    });

    renderEdgeResults(result);

  } catch (err) {
    console.error(err);
  } finally {
    hideLoading('edge-loading');
  }
}

function renderEdgeResults(result) {
  // Stats
  document.getElementById('edge-stat-model').textContent = fmtPct(result.model_probability);
  document.getElementById('edge-stat-market').textContent = fmtPct(result.market_price);

  const netEl = document.getElementById('edge-stat-net');
  const sign = result.net_edge > 0 ? '+' : '';
  netEl.textContent = sign + fmtPct(result.net_edge);
  netEl.className = 'stat-value ' + (
    result.net_edge > 0.05 ? 'positive' : result.net_edge < -0.05 ? 'negative' : 'neutral'
  );

  // Recommendation banner
  const recContainer = document.getElementById('edge-recommendation');
  const icons = { buy: '\u2705', sell: '\u274c', pass: '\u23f8\ufe0f' };
  const labels = { buy: 'BUY — Market underprices this outcome', sell: 'SELL — Market overprices this outcome', pass: 'PASS — No actionable edge detected' };

  recContainer.innerHTML = `
    <div class="recommendation-banner ${result.recommendation}">
      <div class="recommendation-icon">${icons[result.recommendation] || ''}</div>
      <div class="recommendation-text">
        <div class="recommendation-label">Recommendation</div>
        <div class="recommendation-action">${labels[result.recommendation] || result.recommendation.toUpperCase()}</div>
      </div>
      <span class="badge ${result.has_actionable_edge ? 'badge-green' : 'badge-amber'}">
        ${result.has_actionable_edge ? 'Actionable' : 'No Edge'}
      </span>
    </div>`;

  // Signals
  const sigContainer = document.getElementById('edge-signals');
  if (result.signals.length === 0) {
    sigContainer.innerHTML = `
      <div style="text-align: center; color: var(--text-muted); padding: 40px 0;">
        No edge signals detected for this configuration.
      </div>`;
    return;
  }

  sigContainer.innerHTML = result.signals.map(sig => {
    const icon = sig.direction === 'under' ? '\u2b06\ufe0f' : '\u2b07\ufe0f';
    const dirColor = sig.direction === 'under' ? 'var(--accent-green-light)' : 'var(--accent-red-light)';
    return `
      <div class="signal-row">
        <div class="signal-icon">${icon}</div>
        <div class="signal-content">
          <div class="signal-name" style="color: ${dirColor};">
            ${sig.name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
            <span class="badge ${sig.direction === 'under' ? 'badge-green' : 'badge-red'}" style="margin-left: 8px;">
              ${sig.direction === 'under' ? 'Underpriced' : 'Overpriced'}
            </span>
          </div>
          <div class="signal-explanation">${sig.explanation}</div>
          <div style="margin-top: 6px; font-size: 0.75rem; color: var(--text-muted);">
            Magnitude: ${(sig.magnitude * 100).toFixed(1)}% &middot; Confidence: ${(sig.confidence * 100).toFixed(0)}%
          </div>
        </div>
      </div>`;
  }).join('');
}

// ── Auto-run on page load ─────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  // Run initial prediction with default data
  runPredict();
});
