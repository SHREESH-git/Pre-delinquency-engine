import './style.css';
import { createRiskDistributionChart, createRiskTrendChart } from './charts';

// --- Constants ---
const API_BASE = 'http://localhost:8002/api';
// const REFRESH_INTERVAL = 30000; // Removed unused variable

// --- State Management ---
interface AppState {
  activeTab: string;
  selectedCustomerId: string | null;
  isLoading: boolean;
  error: string | null;
  data: any;
  customerProfile: any | null;
  customerSignals: any[] | null;
  streamStatus: {
    status: string;
    messages_ingested: number;
    active_topic: string | null;
  };
  predictionResult: any | null;
  simulationData: any[] | null;
}

const state: AppState = {
  activeTab: 'overview',
  selectedCustomerId: null,
  isLoading: false,
  error: null,
  data: null,
  customerProfile: null,
  customerSignals: null,
  streamStatus: {
    status: 'stopped',
    messages_ingested: 0,
    active_topic: null
  },
  predictionResult: null,
  simulationData: null
};

async function init() {
  try {
    const app = document.querySelector<HTMLDivElement>('#app');
    if (app) app.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100vh; font-family: sans-serif; color: #64748b;">Initializing EarlyShield...</div>';

    await loadTabData();
    await fetchPriorityAlerts(); // Fetch sidebar alerts
    renderApp();
    setupEventListeners();
    initChartsForTab(state.activeTab);
    startStatusPolling();
  } catch (err: any) {
    console.error('Initialization failed:', err);
    const app = document.querySelector<HTMLDivElement>('#app');
    if (app) {
      app.innerHTML = `
  < div style = "display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; font-family: sans-serif; color: #ef4444; padding: 20px; text-align: center;" >
    <h2 style="margin-bottom: 12px;" > App Failed to Load </h2>
      < p style = "color: #64748b; margin-bottom: 24px;" > ${err.message || 'Unknown initialization error'} </p>
        < button onclick = "window.location.reload()" style = "padding: 10px 20px; background: #3b82f6; color: #fff; border: none; border-radius: 8px; cursor: pointer; font-weight: 600;" > Reload Dashboard </button>
          </div>
            `;
    }
  }
}

async function fetchPriorityAlerts() {
  try {
    const res = await fetch(`${API_BASE}/alerts/priority`);
    if (res.ok) {
      (state as any).priorityAlerts = await res.json();
    }
  } catch (err) {
    console.warn('Failed to fetch priority alerts');
  }
}

async function loadTabData() {
  state.isLoading = true;
  state.error = null;
  renderApp();

  let endpoint = '';
  switch (state.activeTab) {
    case 'overview': endpoint = '/dashboard/overview'; break;
    case 'portfolio': endpoint = '/portfolio/overview'; break;
    case 'operations': endpoint = '/operations'; break;
    case 'model': endpoint = '/model/metrics'; break;
    default: endpoint = '/dashboard/overview';
  }

  try {
    const response = await fetch(`${API_BASE}${endpoint} `);
    if (!response.ok) throw new Error('Failed to fetch data');
    state.data = await response.json();
  } catch (err: any) {
    state.error = err.message;
  } finally {
    state.isLoading = false;
    renderApp();
  }
}

async function searchCustomer(customerId: string) {
  state.isLoading = true;
  state.error = null;
  state.customerProfile = null;
  state.customerSignals = null;
  renderApp();

  try {
    const [profRes, sigRes] = await Promise.all([
      fetch(`${API_BASE} /customers/${customerId} `),
      fetch(`${API_BASE} /customers/${customerId}/stress-signals`)
    ]);

    if (!profRes.ok) throw new Error('Customer not found');

    state.customerProfile = await profRes.json();
    state.customerSignals = await sigRes.json();
    state.selectedCustomerId = customerId;

    showToast('Customer profile loaded', 'success');

    // Auto-trigger prediction
    await runInference(customerId);

  } catch (err: any) {
    state.error = err.message;
    showToast(`Error: ${err.message}`, 'error');
  } finally {
    state.isLoading = false;
    renderApp();
  }
}

async function runInference(customerId: string) {
  try {
    const predRes = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ customer_id: customerId })
    });

    if (predRes.ok) {
      state.predictionResult = await predRes.json();
      showToast('ML Inference complete', 'success');
    }
  } catch (err) {
    showToast('Failed to run inference', 'error');
  } finally {
    renderApp();
  }
}

async function fetchSimulationData(customerId: string) {
  try {
    const res = await fetch(`${API_BASE}/debug/synthetic-data/${customerId}`);
    if (res.ok) {
      state.simulationData = await res.json();
      showModal('Synthetic History Verification', `
        <p style="margin-bottom: 16px;">The following records were fetched from the <strong>Synthetic ML Data Engine</strong> to feed the feature engineering pipeline.</p>
        <pre>${JSON.stringify(state.simulationData, null, 2)}</pre>
      `);
    }
  } catch (err) {
    showToast('Failed to fetch synthetic data', 'error');
  }
}

async function fetchAlertDetails(customerId: string) {
  try {
    const res = await fetch(`${API_BASE}/details/alert/${customerId}`);
    if (res.ok) {
      const data = await res.json();
      showModal(data.title, `
            <div style="margin-bottom: 20px;">
                <h4 style="color: var(--risk-high); margin-bottom: 8px;">Reason for Alert</h4>
                <p>${data.reason}</p>
            </div>
            <div style="margin-bottom: 20px;">
                <h4 style="margin-bottom: 8px;">Risk Drivers</h4>
                <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                    ${data.drivers.map((d: any) => `
                        <div style="padding: 8px 12px; background: ${d.color}22; border: 1px solid ${d.color}; border-radius: 6px; font-size: 12px; color: ${d.color}; font-weight: 600;">
                            ${d.name}: ${d.value}
                        </div>
                    `).join('')}
                </div>
            </div>
            <div style="margin-bottom: 20px;">
                <h4 style="margin-bottom: 8px;">Action Recommendation</h4>
                <div style="padding: 12px; background: rgba(59, 130, 246, 0.05); border-left: 4px solid var(--accent-blue); font-size: 13px;">
                    ${data.recommendation}
                </div>
            </div>
        `);
    }
  } catch (err) {
    showToast('Failed to fetch alert details', 'error');
  }
}

async function fetchProductDetails(productType: string) {
  try {
    const res = await fetch(`${API_BASE}/details/product/${productType}`);
    if (res.ok) {
      const data = await res.json();
      showModal(data.title, `
                <div class="kpi-grid" style="grid-template-columns: repeat(2, 1fr); gap: 16px; margin-bottom: 24px;">
                    ${Object.entries(data.metrics).map(([k, v]) => `
                        <div style="padding: 16px; background: var(--surface-tertiary); border-radius: 12px;">
                            <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase;">${k}</div>
                            <div style="font-size: 18px; font-weight: 800;">${v}</div>
                        </div>
                    `).join('')}
                </div>
                <div>
                    <h4 style="margin-bottom: 12px;">Portfolio Health Breakdown</h4>
                    <div style="display: flex; height: 12px; border-radius: 6px; overflow: hidden; margin-bottom: 12px;">
                         <div style="width: 92%; background: var(--risk-low);"></div>
                         <div style="width: 5%; background: var(--risk-medium);"></div>
                         <div style="width: 3%; background: var(--risk-high);"></div>
                    </div>
                    <div style="display: flex; gap: 20px; font-size: 12px;">
                        ${data.breakdown.map((b: any) => `
                            <div style="display: flex; align-items: center; gap: 6px;">
                                <div style="width: 8px; height: 8px; border-radius: 50%; background: ${b.label === 'Standard' ? 'var(--risk-low)' : b.label === 'Sub-standard' ? 'var(--risk-medium)' : 'var(--risk-high)'};"></div>
                                <span>${b.label}: ${b.value}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `);
    }
  } catch (err) {
    showToast('Failed to fetch product performance', 'error');
  }
}

function showToast(message: string, type: 'success' | 'error' | 'info' = 'info') {
  let container = document.querySelector('.toast-container');
  if (!container) {
    container = document.createElement('div');
    container.className = 'toast-container';
    document.body.appendChild(container);
  }

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <i class="ph-bold ${type === 'success' ? 'ph-check-circle' : type === 'error' ? 'ph-warning-circle' : 'ph-info'}" style="font-size: 20px;"></i>
    <span>${message}</span>
  `;

  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(100%)';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

function showModal(title: string, content: string) {
  const overlay = document.createElement('div');
  overlay.className = 'modal-overlay';
  overlay.innerHTML = `
    <div class="modal">
      <div class="modal-header">
        <h3>${title}</h3>
        <button class="btn-close modal-close-trigger">&times;</button>
      </div>
      <div class="modal-content">
        ${content}
      </div>
      <div class="modal-footer">
        <button class="btn btn-primary modal-close-trigger" style="padding: 10px 20px;">Close</button>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);
}

function renderApp() {
  const app = document.querySelector<HTMLDivElement>('#app');
  if (!app) return;

  app.innerHTML = `
    <div class="app-container">
      ${renderSidebar()}
      ${renderMain()}
    </div>
  `;
}

function setupEventListeners() {
  document.addEventListener('click', async (e) => {
    const target = e.target as HTMLElement;

    // Nav Click
    const navItem = target.closest('.nav-item');
    if (navItem) {
      e.preventDefault();
      const tab = navItem.getAttribute('data-tab');
      if (tab && tab !== state.activeTab) {
        state.activeTab = tab;
        state.data = null; // Clear data when switching tabs to avoid map() crashes
        await loadTabData();
        initChartsForTab(tab);
      }
      return;
    }

    // Stream Controls
    if (target.classList.contains('start-stream-btn')) {
      await controlStream('start');
    }
    if (target.classList.contains('stop-stream-btn')) {
      await controlStream('stop');
    }

    // Search Click
    if (target.classList.contains('search-btn')) {
      const input = document.querySelector<HTMLInputElement>('#custSearchInput');
      if (input && input.value.trim()) {
        await searchCustomer(input.value.trim());
      }
    }

    // Run ML Inference (same button as search-btn in profile)
    if (target.classList.contains('run-inference-btn')) {
      await runInference(state.selectedCustomerId!);
    }

    // CSV Customer EL/PD Prediction
    if (target.id === 'csvPredictBtn') {
      const input = document.querySelector<HTMLInputElement>('#csvCustInput');
      if (input && input.value.trim()) {
        await predictCsvCustomer(input.value.trim());
      }
    }

    // View Synthetic Data
    if (target.classList.contains('view-data-btn')) {
      await fetchSimulationData(state.selectedCustomerId!);
    }

    // Close Modal
    if (target.classList.contains('modal-close-trigger')) {
      const modal = document.querySelector('.modal-overlay');
      if (modal) modal.remove();
    }

    // Detail Popups: Priority Alerts
    const stressItem = (target as HTMLElement).closest('.stress-item');
    if (stressItem) {
      const customerId = (stressItem as HTMLElement).dataset.customerId;
      if (customerId) await fetchAlertDetails(customerId);
    }

    // Detail Popups: Loan Product Performance
    const productRow = (target as HTMLElement).closest('.product-row');
    if (productRow) {
      const product = (productRow as HTMLElement).dataset.product;
      if (product) await fetchProductDetails(product);
    }

    // Detail Popups: At-Risk Customer
    const atRiskRow = (target as HTMLElement).closest('.at-risk-row');
    if (atRiskRow) {
      const customerId = (atRiskRow as HTMLElement).dataset.customerId;
      if (customerId) await searchCustomer(customerId); // Reuse searchCustomer for profile view
    }
  });

  // Enter key for search
  document.addEventListener('keypress', async (e) => {
    if (e.key === 'Enter') {
      const custInput = document.querySelector<HTMLInputElement>('#custSearchInput');
      if (custInput && custInput === document.activeElement && custInput.value.trim()) {
        await searchCustomer(custInput.value.trim());
      }
      const csvInput = document.querySelector<HTMLInputElement>('#csvCustInput');
      if (csvInput && csvInput === document.activeElement && csvInput.value.trim()) {
        await predictCsvCustomer(csvInput.value.trim());
      }
    }
  });
}

async function predictCsvCustomer(customerId: string) {
  const btn = document.querySelector<HTMLButtonElement>('#csvPredictBtn');
  if (btn) { btn.disabled = true; btn.textContent = 'Running...'; }
  try {
    const response = await fetch(`${API_BASE}/predict/customer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ customer_id: customerId })
    });
    if (!response.ok) {
      const err = await response.json();
      alert(`Error: ${err.detail || 'Customer not found'}`);
      return;
    }
    (state as any).csvPredictionResult = await response.json();
    renderApp();
  } catch (err: any) {
    alert(`Request failed: ${err.message}`);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Run Inference'; }
  }
}

function initChartsForTab(tab: string) {
  setTimeout(() => {
    if (tab === 'overview' && state.data) {
      const d = state.data;

      // Map backend data to chart formats
      if (d.portfolio_type) {
        createRiskDistributionChart('riskDistChart', {
          labels: d.portfolio_type.labels,
          data: d.portfolio_type.data
        });
      }

      if (d.portfolio_growth) {
        createRiskTrendChart('riskTrendChart', {
          labels: d.portfolio_growth.labels,
          datasets: [
            {
              label: 'Portfolio Growth (Cr)',
              data: d.portfolio_growth.data,
              borderColor: '#3b82f6',
              backgroundColor: 'rgba(59, 130, 246, 0.08)',
              borderWidth: 3,
              pointRadius: 4,
              pointBackgroundColor: '#fff',
              pointBorderColor: '#3b82f6',
              pointBorderWidth: 2,
              fill: true,
              tension: 0.4
            }
          ]
        });
      }
    }
  }, 0);
}

function renderSidebar() {
  const alerts = (state as any).priorityAlerts || []; // Use fetched alerts

  return `
    <aside class="sidebar">
      <div class="logo-section">
        <h2>EarlyShield</h2>
        <p>Risk Protection Portal</p>
      </div>
      
      <nav class="nav-group">
        <a href="#" class="nav-item ${state.activeTab === 'overview' ? 'active' : ''}" data-tab="overview">
          <i class="ph-bold ph-squares-four"></i>
          <span>Overview</span>
        </a>
        <a href="#" class="nav-item ${state.activeTab === 'portfolio' ? 'active' : ''}" data-tab="portfolio">
          <i class="ph-bold ph-chart-line-up"></i>
          <span>Portfolio Analytics</span>
        </a>
        <a href="#" class="nav-item ${state.activeTab === 'operations' ? 'active' : ''}" data-tab="operations">
          <i class="ph-bold ph-shield-check"></i>
          <span>Safety Monitoring</span>
        </a>
        <a href="#" class="nav-item ${state.activeTab === 'customer' ? 'active' : ''}" data-tab="customer">
          <i class="ph-bold ph-users"></i>
          <span>Customer Safety</span>
        </a>
        <a href="#" class="nav-item ${state.activeTab === 'model' ? 'active' : ''}" data-tab="model">
          <i class="ph-bold ph-cpu"></i>
          <span>AI Health</span>
        </a>
      </nav>

      <div class="stress-panel">
        <div class="stress-header">
           <i class="ph-bold ph-warning-circle"></i> Priority Alerts
        </div>
        <div class="stress-list">
          ${alerts.length > 0 ? alerts.map((a: any) => `
              <div class="stress-item" data-customer-id="${a.id}">
                <div style="display:flex; justify-content:space-between; width:100%">
                   <span class="stress-label">${a.id}</span>
                   <span class="stress-badge" style="background: var(--risk-${(a.risk || 'High').toLowerCase()});">
                     ${a.risk}
                   </span>
                </div>
                <div style="font-size: 10px; color: var(--text-tertiary); margin-top: 4px;">
                   Action: ${a.action}
                </div>
              </div>
            `).join('') : `
              <div style="padding: 10px; color: var(--text-tertiary); font-style: italic; font-size: 11px;">
                No priority alerts
              </div>
            `}
        </div>
      </div>
      
      <div style="margin-top: auto; padding: 20px; border-top: 1px solid var(--bg-secondary);">
         <div style="display: flex; align-items: center; gap: 10px; font-size: 12px; color: var(--text-tertiary); margin-bottom: 12px;">
            <div class="stream-indicator" style="width: 8px; height: 8px; border-radius: 50%; background: ${state.streamStatus.status === 'running' ? '#22c55e' : '#64748b'};"></div>
            <span>Kafka Ingestion</span>
         </div>
         <div class="message-count" style="font-size: 11px; color: #64748b; margin-bottom: 16px;">Messages: ${state.streamStatus.messages_ingested}</div>
         <div style="display: flex; gap: 8px;">
            <button class="start-stream-btn" style="flex: 1; padding: 6px; font-size: 11px; font-weight: 700; border-radius: 6px; cursor: pointer; border: 1px solid #22c55e; background: transparent; color: #22c55e;" ${state.streamStatus.status === 'running' ? 'disabled' : ''}>START</button>
            <button class="stop-stream-btn" style="flex: 1; padding: 6px; font-size: 11px; font-weight: 700; border-radius: 6px; cursor: pointer; border: 1px solid #ef4444; background: transparent; color: #ef4444;" ${state.streamStatus.status === 'stopped' ? 'disabled' : ''}>STOP</button>
         </div>
      </div>
    </aside>
  `;
}

function renderMain() {
  if (state.isLoading && !state.data && !state.customerProfile) {
    return `<main class="main-content"><div style="display: flex; align-items: center; justify-content: center; height: 100vh;">Loading...</div></main>`;
  }

  let content = '';
  switch (state.activeTab) {
    case 'overview': content = renderOverview(); break;
    case 'portfolio': content = renderPortfolio(); break;
    case 'operations': content = renderOperations(); break;
    case 'customer': content = renderCustomer(); break;
    case 'model': content = renderModel(); break;
    default: content = renderOverview();
  }

  return `
    <main class="main-content">
      <header class="top-bar">
        <div class="search-box" style="display: flex; align-items: center; background: var(--surface-secondary); padding: 10px 16px; border-radius: var(--radius-md); border: 1px solid var(--border-light); width: 400px;">
          <i class="ph-bold ph-magnifying-glass" style="color: var(--text-tertiary); margin-right: 12px;"></i>
          <input type="text" placeholder="Search safety metrics, customers, or alerts..." style="border: none; background: transparent; outline: none; width: 100%; font-size: 14px;">
        </div>
        <div class="user-actions" style="display: flex; align-items: center; gap: 20px;">
          <button class="icon-btn" style="background: none; border: none; font-size: 20px; color: var(--text-secondary); cursor: pointer;"><i class="ph-bold ph-bell"></i></button>
          <div class="user-profile" style="display: flex; align-items: center; gap: 12px;">
            <div class="avatar" style="width: 36px; height: 36px; border-radius: 50%; background: var(--accent-gold); color: var(--text-primary); display: flex; align-items: center; justify-content: center; font-weight: 700; border: 2px solid #fff; box-shadow: var(--shadow-sm);">AD</div>
            <span style="font-weight: 600; color: var(--text-primary);">Admin</span>
          </div>
        </div>
      </header>

      <section class="content-scroll">
        ${state.error ? `<div style="padding: 16px; background: rgba(239, 68, 68, 0.1); border: 1px solid var(--risk-high); border-radius: 8px; color: var(--risk-high); margin-bottom: 24px;">${state.error}</div>` : ''}
        ${content}
      </section>
    </main>
  `;
}

function renderOverview() {
  const d = state.data || {};
  return `
    <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 40px;">
      <div>
        <h1 style="font-size: 32px; color: var(--text-primary)">Welcome back to EarlyShield</h1>
        <p style="color: var(--text-secondary); font-size: 16px; margin-top: 4px;">Here is an overview of your current safety metrics</p>
      </div>
      <div style="display: flex; gap: 16px;">
         <button class="btn btn-secondary" style="padding: 12px 24px; border-radius: var(--radius-md); border: 1px solid var(--border-light); background: #fff; font-weight: 600; font-size: 14px; cursor: pointer; transition: all 0.2s;">Download Summary</button>
         <button class="btn btn-primary" style="padding: 12px 24px; border-radius: var(--radius-md); background: var(--accent-blue); color: #fff; border: none; font-weight: 600; font-size: 14px; cursor: pointer; transition: all 0.2s;">Set Alert Rules</button>
      </div>
    </div>

    <div class="kpi-grid">
      <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: start;">
          <div>
            <p class="kpi-label">Safety Score</p>
            <h2 class="kpi-value">${d.safetyScore || '84.2'}</h2>
          </div>
          <span class="trend trend-up">${d.safetyTrend || '+3.2%'}</span>
        </div>
        <p style="font-size: 12px; color: var(--text-tertiary); margin-top: 12px;">Average score across your entire portfolio</p>
      </div>
      <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: start;">
          <div>
            <p class="kpi-label">Protected Accounts</p>
            <h2 class="kpi-value">${d.activeProtected || '24,582'}</h2>
          </div>
          <span class="trend trend-up">${d.accountsTrend || '+5.8%'}</span>
        </div>
        <p style="font-size: 12px; color: var(--text-tertiary); margin-top: 12px;">Currently being monitored by EarlyShield</p>
      </div>
      <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: start;">
          <div>
            <p class="kpi-label">Intervention Success</p>
            <h2 class="kpi-value">${d.successRate || '76.3%'}</h2>
          </div>
          <span class="trend trend-up">${d.successTrend || '+4.1%'}</span>
        </div>
        <p style="font-size: 12px; color: var(--text-tertiary); margin-top: 12px;">Rate of successful risk mitigations</p>
      </div>
    </div>

    <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 32px; margin-bottom: 40px;">
      <div class="card">
        <h3 style="margin-bottom: 8px;">Safety Trends</h3>
        <p style="font-size: 13px; color: var(--text-secondary); margin-bottom: 24px;">How your risk levels are changing over time</p>
        <div style="height: 300px;">
          <canvas id="riskTrendChart"></canvas>
        </div>
      </div>
      <div class="card">
        <h3 style="margin-bottom: 8px;">Risk Distribution</h3>
        <p style="font-size: 13px; color: var(--text-secondary); margin-bottom: 24px;">Portfolio break-up by risk levels</p>
        <div style="height: 300px;">
          <canvas id="riskDistChart"></canvas>
        </div>
      </div>
    </div>

    <div class="card">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px;">
        <h3 style="font-size: 18px;">Recent High-Alert Segments</h3>
        <button class="btn btn-secondary" style="padding: 8px 16px; font-size: 13px; background: #fff; border: 1px solid var(--border-light); border-radius: 8px; font-weight: 600; cursor: pointer;">View All Alerts</button>
      </div>
      <table class="data-table">
        <thead>
          <tr>
            <th style="text-align: left; padding-bottom: 12px; color: var(--text-secondary); font-size: 13px;">Segment Name</th>
            <th style="text-align: left; padding-bottom: 12px; color: var(--text-secondary); font-size: 13px;">Safety Score</th>
            <th style="text-align: left; padding-bottom: 12px; color: var(--text-secondary); font-size: 13px;">Alert Level</th>
            <th style="text-align: left; padding-bottom: 12px; color: var(--text-secondary); font-size: 13px;">Intervention Status</th>
          </tr>
        </thead>
        <tbody>
          ${(d.recentAlerts || []).map((alert: any) => `
            <tr class="at-risk-row" data-customer-id="${alert.segment.includes('101') ? '101' : alert.segment.includes('102') ? '102' : '103'}">
              <td>${alert.segment}</td>
              <td><strong>${parseInt(alert.score).toFixed(1)}</strong></td>
              <td><span class="risk-badge" style="background: var(--risk-${alert.level.toLowerCase()}); color: #fff; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 700;">${alert.level}</span></td>
              <td><span style="color: ${alert.status === 'High Priority' ? 'var(--risk-critical)' : 'var(--accent-blue)'}; font-weight: 600;">${alert.status}</span></td>
            </tr>
          `).join('') || `
            <tr><td colspan="4" style="text-align: center; color: var(--text-tertiary);">No recent alerts</td></tr>
          `}
        </tbody>
      </table>
    </div>
  `;
}

function renderPortfolio() {
  const d = state.data || {};
  return `
    <div style="margin-bottom: 40px;">
      <h1 style="font-size: 32px; color: var(--text-primary)">Portfolio Analysis</h1>
      <p style="color: var(--text-secondary); font-size: 16px; margin-top: 4px;">Real data from ${d.dataSource || 'dataset'}</p>
    </div>

    <div class="kpi-grid">
      <div class="card">
        <p class="kpi-label">Portfolio Value</p>
        <h2 class="kpi-value">${d.totalValue || '₹845.2Cr'}</h2>
        <p style="font-size: 12px; color: var(--text-tertiary); margin-top: 12px;">Total EMI obligations across all products</p>
      </div>
      <div class="card">
        <p class="kpi-label">Healthy Loans</p>
        <h2 class="kpi-value">${d.healthyCount || '—'}</h2>
        <p style="font-size: 12px; color: var(--text-tertiary); margin-top: 12px;">${d.healthyPercent || '—'} of total portfolio (risk level ≤ 1)</p>
      </div>
      <div class="card">
        <p class="kpi-label">Avg Safety Score</p>
        <h2 class="kpi-value">${d.avgScore || '—'}</h2>
        <p style="font-size: 12px; color: var(--text-tertiary); margin-top: 12px;">Target: Above 75.0</p>
      </div>
    </div>

    <div class="card" style="margin-bottom: 32px;">
      <h3 style="margin-bottom: 24px;">Loan Product Performance <span style="font-size: 12px; font-weight: 400; color: var(--text-tertiary); margin-left: 8px;">All 90,000 records</span></h3>
      <table class="data-table">
        <thead>
          <tr>
            <th>Product Type</th>
            <th>Active Loans</th>
            <th>Total Value</th>
            <th>Avg Ticket</th>
            <th>Safety Score</th>
            <th>Default Rate</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          ${(d.productPerformance || []).map((p: any) => `
            <tr class="product-row" data-product="${p.type}">
              <td><strong>${p.type}</strong></td>
              <td>${p.active}</td>
              <td>${p.value}</td>
              <td>${p.ticket}</td>
              <td><span style="color: var(--risk-${p.risk.toLowerCase()}); font-weight: 700;">${p.score}</span></td>
              <td><strong style="color: ${parseFloat(p.defaultRate) > 10 ? 'var(--risk-high)' : 'var(--risk-low)'}">${p.defaultRate}</strong></td>
              <td><span class="risk-badge" style="background: var(--risk-${p.risk.toLowerCase()}); color: #fff; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 700;">${p.status}</span></td>
            </tr>
          `).join('') || `<tr><td colspan="7">No performance data</td></tr>`}
        </tbody>
      </table>
    </div>

    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 32px; margin-bottom: 32px;">
      <div class="card">
        <h3 style="margin-bottom: 20px;">Region Breakdown <span style="font-size: 12px; font-weight: 400; color: var(--text-tertiary);">All 90k records</span></h3>
        <table class="data-table">
          <thead><tr><th>Region</th><th>Count</th><th>Default Rate</th><th>Avg Risk</th></tr></thead>
          <tbody>
            ${(d.regionBreakdown || []).map((r: any) => `
              <tr>
                <td><strong>${r.tier}</strong></td>
                <td>${r.count}</td>
                <td style="color: ${parseFloat(r.defaultRate) > 10 ? 'var(--risk-high)' : 'var(--risk-low)'}; font-weight: 700;">${r.defaultRate}</td>
                <td>${r.avgRisk}</td>
              </tr>
            `).join('') || '<tr><td colspan="4">No data</td></tr>'}
          </tbody>
        </table>
      </div>
      <div class="card">
        <h3 style="margin-bottom: 20px;">Customer Segment <span style="font-size: 12px; font-weight: 400; color: var(--text-tertiary);">All 90k records</span></h3>
        <table class="data-table">
          <thead><tr><th>Segment</th><th>Count</th><th>Default Rate</th><th>Avg Income</th></tr></thead>
          <tbody>
            ${(d.segmentBreakdown || []).map((s: any) => `
              <tr>
                <td><strong>${s.segment}</strong></td>
                <td>${s.count}</td>
                <td style="color: ${parseFloat(s.defaultRate) > 10 ? 'var(--risk-high)' : 'var(--risk-low)'}; font-weight: 700;">${s.defaultRate}</td>
                <td>${s.avgIncome}</td>
              </tr>
            `).join('') || '<tr><td colspan="4">No data</td></tr>'}
          </tbody>
        </table>
      </div>
    </div>
  `;
}

function renderOperations() {
  const d = state.data || [];
  return `
    <div style="margin-bottom: 40px;">
      <h1 style="font-size: 32px; color: var(--text-primary)">Safety Monitoring</h1>
      <p style="color: var(--text-secondary); font-size: 16px; margin-top: 4px;">Top 10 highest-risk customers from stratified sample</p>
    </div>

    <div class="card">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 32px;">
        <h3>At-Risk Customer Portfolio <span style="font-size: 12px; font-weight: 400; color: var(--text-tertiary);">Sorted by risk level & EMI ratio</span></h3>
        <div style="display: flex; gap: 12px;">
           <button class="btn btn-secondary" style="padding: 8px 16px; background: #fff; border: 1px solid var(--border-light); border-radius: 8px; font-weight: 600; cursor: pointer;">Filter</button>
           <button class="btn btn-primary" style="padding: 8px 16px; background: var(--accent-blue); color: #fff; border: none; border-radius: 8px; font-weight: 600; cursor: pointer;">Bulk Action</button>
        </div>
      </div>
      <table class="data-table">
        <thead>
          <tr>
            <th>Customer ID</th>
            <th>Product</th>
            <th>Region</th>
            <th>Safety Score</th>
            <th>EMI/Income</th>
            <th>Salary Delay</th>
            <th>Action</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          ${Array.isArray(d) ? d.map((c: any) => `
            <tr style="cursor: pointer;">
              <td><strong>${c.id}</strong></td>
              <td>${c.product || ''}</td>
              <td>${c.region || ''}</td>
              <td><span style="color: var(--risk-${(c.risk || 'low').toLowerCase()}); font-weight: 700;">${c.score || 0}</span></td>
              <td><strong style="color: ${parseFloat(c.emiRatio) > 40 ? 'var(--risk-high)' : 'var(--text-primary)'}">${c.emiRatio || '—'}</strong></td>
              <td style="color: ${parseInt(c.salaryDelay) > 5 ? 'var(--risk-high)' : 'var(--text-primary)'}">${c.salaryDelay || '—'}</td>
              <td>${c.action || ''}</td>
              <td><span class="risk-badge" style="background: var(--risk-${(c.statusRisk || c.risk || 'low').toLowerCase()}); color: #fff; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 700;">${c.status || ''}</span></td>
            </tr>
          `).join('') : `<tr><td colspan="8" style="text-align: center; padding: 40px; color: var(--text-tertiary);">No at-risk customers found</td></tr>`}
        </tbody>
      </table>
    </div>
  `;
}

function renderCustomer() {
  const p = state.customerProfile;
  const s = state.customerSignals || [];
  const pr = (state as any).csvPredictionResult;

  return `
    <div style="margin-bottom: 40px;">
      <h1 style="font-size: 32px; color: var(--text-primary)">Customer Safety Profile</h1>
      <p style="color: var(--text-secondary); font-size: 16px; margin-top: 4px;">Analyze individual safety factors and behavioral trends</p>
    </div>

    <!-- CSV Customer EL/PD Prediction -->
    <div class="card" style="margin-bottom: 32px; border-left: 4px solid var(--accent-blue);">
      <h3 style="margin-bottom: 8px; display: flex; align-items: center; gap: 8px; color: var(--accent-blue);">
        <i class="ph-bold ph-brain"></i> Predict EL &amp; PD — Dataset Customer
      </h3>
      <p style="font-size: 13px; color: var(--text-secondary); margin-bottom: 20px;">Enter a customer ID from the CSV dataset (e.g. <code>CUST0000000</code>) to run real ML inference</p>
      <div style="display: flex; gap: 12px; margin-bottom: 20px;">
        <input type="text" id="csvCustInput" placeholder="e.g. CUST0000000" style="flex: 1; padding: 12px 16px; border-radius: var(--radius-md); border: 1px solid var(--border-light); outline: none; font-family: monospace; font-size: 14px;">
        <button id="csvPredictBtn" style="padding: 12px 28px; background: var(--accent-blue); color: #fff; border: none; border-radius: var(--radius-md); font-weight: 700; font-size: 14px; cursor: pointer; transition: opacity 0.2s;">Run Inference</button>
      </div>
      ${pr ? `
        <div style="background: var(--surface-secondary); border-radius: 12px; padding: 24px;">
          <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
            <div style="display: flex; gap: 24px;">
              <div>
                <div style="font-size: 12px; color: var(--text-tertiary); margin-bottom: 4px;">Customer</div>
                <div style="font-size: 16px; font-weight: 700; font-family: monospace;">${pr.customer_id}</div>
              </div>
            </div>
            <span class="risk-badge" style="background: var(--risk-${pr.risk_level.toLowerCase().replace(' risk', '').replace(' ', '-')}); color: #fff; padding: 8px 18px; border-radius: 10px; font-size: 13px; font-weight: 800; letter-spacing: 0.5px;">${pr.risk_level}</span>
          </div>
          <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 20px;">
            <div style="text-align: center; padding: 16px; background: #fff; border-radius: 10px; border: 1px solid var(--border-light);">
              <div style="font-size: 11px; color: var(--text-tertiary); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">PD Score</div>
              <div style="font-size: 28px; font-weight: 800; color: var(--text-primary);">${(pr.probability * 100).toFixed(1)}%</div>
            </div>
            <div style="text-align: center; padding: 16px; background: #fff; border-radius: 10px; border: 1px solid var(--border-light);">
              <div style="font-size: 11px; color: var(--text-tertiary); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">Expected Loss</div>
              <div style="font-size: 28px; font-weight: 800; color: var(--risk-high);">₹${pr.expected_loss.toLocaleString()}</div>
            </div>
            <div style="text-align: center; padding: 16px; background: #fff; border-radius: 10px; border: 1px solid var(--border-light);">
              <div style="font-size: 11px; color: var(--text-tertiary); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">LGD</div>
              <div style="font-size: 28px; font-weight: 800; color: var(--text-primary);">${(pr.lgd * 100).toFixed(1)}%</div>
            </div>
          </div>
          <div style="font-size: 12px; color: var(--text-tertiary); margin-bottom: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Key Feature Drivers</div>
          <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px;">
            ${(pr.key_features || []).map((f: any) => `
              <div style="padding: 10px 14px; background: #fff; border-radius: 8px; border: 1px solid var(--border-light);">
                <div style="font-size: 11px; color: var(--text-tertiary); margin-bottom: 4px;">${f.label}</div>
                <div style="font-size: 14px; font-weight: 700;">${f.value}</div>
              </div>
            `).join('')}
          </div>
          ${pr.actual_risk_level >= 0 ? `
            <div style="margin-top: 16px; padding: 12px 16px; background: rgba(59,130,246,0.06); border-radius: 8px; font-size: 12px; color: var(--text-secondary);">
              <strong>Actual (CSV):</strong> Risk Level ${pr.actual_risk_level} &nbsp;|&nbsp; Default Observed: ${pr.actual_default === 1 ? '<span style="color:var(--risk-high);font-weight:700;">Yes</span>' : '<span style="color:var(--risk-low);font-weight:700;">No</span>'}
            </div>
          ` : ''}
        </div>
      ` : `
        <div style="padding: 20px; text-align: center; color: var(--text-tertiary); font-style: italic; background: var(--surface-secondary); border-radius: 10px;">
          Enter a CSV customer ID above and click <strong>Run Inference</strong> to see EL &amp; PD results
        </div>
      `}
    </div>

    <div class="card" style="margin-bottom: 40px;">
      <h3 style="margin-bottom: 24px;">Customer Lookup</h3>
      <div style="display: flex; gap: 16px;">
        <input type="text" id="custSearchInput" placeholder="Enter Customer ID (e.g. 101)" value="${state.selectedCustomerId || ''}" style="flex: 1; padding: 12px 16px; border-radius: var(--radius-md); border: 1px solid var(--border-light); outline: none;">
        <button class="btn btn-primary search-btn" style="padding: 12px 24px; background: var(--accent-blue); color: #fff; border: none; border-radius: var(--radius-md); font-weight: 600; cursor: pointer;">Search Profile</button>
      </div>
    </div>
    
    ${!p ? `
      <div style="display: flex; align-items: center; justify-content: center; height: 100px; color: var(--text-tertiary);">
         Search a customer ID (101 to 105) to view detailed analysis
      </div>
    ` : `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 32px; margin-bottom: 32px;">
        <div class="card">
          <h3>Overview: ${p.name}</h3>
          <p style="color: var(--text-secondary); margin-bottom: 24px;">Account details and current safety standings</p>
          <div style="display: flex; flex-direction: column; gap: 16px;">
             <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-subtle); padding-bottom: 12px;">
               <span style="color: var(--text-secondary);">Loan Type</span>
               <strong>${p.loanType}</strong>
             </div>
             <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-subtle); padding-bottom: 12px;">
               <span style="color: var(--text-secondary);">Outstanding</span>
               <strong>${p.outstanding}</strong>
             </div>
             <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-subtle); padding-bottom: 12px;">
               <span style="color: var(--text-secondary);">Monthly EMI</span>
               <strong>${p.emi}</strong>
             </div>
             <div style="display: flex; justify-content: space-between;">
               <span style="color: var(--text-secondary);">Safety Level</span>
               <span class="risk-badge" style="background: var(--risk-${p.riskLevel.toLowerCase().replace(' ', '-')}); color: #fff; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 700;">${p.riskLevel}</span>
             </div>
          </div>
        <div class="card">
          <h3>Risk Drivers</h3>
          <p style="color: var(--text-secondary); margin-bottom: 24px;">Contributing safety factors (ML Analysis)</p>
          <div style="display: flex; flex-direction: column; gap: 24px;">
             ${p.drivers.map((d: any) => `
               <div>
                 <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                   <span style="font-size: 14px; font-weight: 600;">${d.name}</span>
                   <span style="color: ${d.value.toString().startsWith('-') || d.percent > 70 ? 'var(--risk-high)' : 'var(--text-primary)'}; font-weight: 700;">${d.value}</span>
                 </div>
                 <div style="height: 8px; background: var(--surface-tertiary); border-radius: 4px; overflow: hidden;">
                   <div style="width: ${d.percent}%; height: 100%; background: ${d.color === 'danger' ? 'var(--risk-critical)' : d.color === 'warning' ? 'var(--risk-medium)' : 'var(--risk-low)'};"></div>
                 </div>
               </div>
             `).join('')}
          </div>
        </div>
      </div>

      <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 32px; margin-bottom: 32px;">
        <div class="card" style="border-left: 4px solid var(--accent-blue);">
          <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 20px;">
            <div>
              <h3 style="color: var(--accent-blue); display: flex; align-items: center; gap: 8px;">
                <i class="ph-bold ph-brain"></i> Real-time AI Risk Prediction
              </h3>
              <p style="font-size: 13px; color: var(--text-secondary);">Hybrid ML/DL Ensemble Output</p>
            </div>
            ${state.predictionResult ? `<span class="risk-badge" style="background: var(--risk-${state.predictionResult.risk_level.toLowerCase().replace(' ', '-')}); color: #fff; padding: 6px 14px; border-radius: 8px; font-size: 12px; font-weight: 800; letter-spacing: 0.5px;">${state.predictionResult.risk_level}</span>` : ''}
          </div>

          ${state.predictionResult ? `
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px;">
              <div style="text-align: center; padding: 16px; background: rgba(59, 130, 246, 0.05); border-radius: 12px;">
                <div style="font-size: 12px; color: var(--text-tertiary); margin-bottom: 4px;">Probability (PD)</div>
                <div style="font-size: 24px; font-weight: 800; color: var(--text-primary);">${(state.predictionResult.probability * 100).toFixed(1)}%</div>
              </div>
              <div style="text-align: center; padding: 16px; background: rgba(59, 130, 246, 0.05); border-radius: 12px;">
                <div style="font-size: 12px; color: var(--text-tertiary); margin-bottom: 4px;">Exposure (EAD)</div>
                <div style="font-size: 24px; font-weight: 800; color: var(--text-primary);">₹${(state.predictionResult.ead / 1000).toFixed(1)}K</div>
              </div>
              <div style="text-align: center; padding: 16px; background: rgba(59, 130, 246, 0.05); border-radius: 12px;">
                <div style="font-size: 12px; color: var(--text-tertiary); margin-bottom: 4px;">Expected Loss (EL)</div>
                <div style="font-size: 24px; font-weight: 800; color: var(--risk-high);">₹${state.predictionResult.expected_loss.toLocaleString()}</div>
              </div>
            </div>
            <div style="margin-top: 20px; font-size: 12px; color: var(--text-tertiary); display: flex; align-items: center; gap: 6px;">
              <i class="ph-bold ph-info"></i> Calculated as: PD (${(state.predictionResult.probability * 100).toFixed(1)}%) × LGD (${(state.predictionResult.lgd * 100).toFixed(1)}%) × EAD (₹${state.predictionResult.ead.toLocaleString()})
            </div>
          ` : `
            <div style="display: flex; align-items: center; justify-content: center; height: 100px; color: var(--text-tertiary); font-style: italic;">
              Prediction pending... Click "Run Inference" to update
            </div>
          `}
        </div>

        <div class="card" style="display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);">
          <i class="ph-bold ph-magic-wand" style="font-size: 32px; color: var(--accent-blue); margin-bottom: 12px;"></i>
          <h4 style="margin-bottom: 8px;">Trigger Recalculation</h4>
          <p style="font-size: 12px; color: var(--text-tertiary); margin-bottom: 20px;">Force update risk score using latest behavioral data</p>
          <button class="btn btn-primary run-inference-btn" style="width: 100%; padding: 10px; background: var(--accent-blue); margin-bottom: 12px;">Run ML Inference</button>
          <button class="btn btn-secondary view-data-btn" style="width: 100%; padding: 10px; background: #fff; border: 1px solid var(--border-light); font-size: 12px; font-weight: 600; cursor: pointer;">View Synthetic Data</button>
        </div>
      </div>

      <div class="card">
        <h3 style="margin-bottom: 24px;">Behavioral Stress signals (Real-time)</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px;">
           ${s.map((signal: any) => `
             <div style="padding: 16px; border: 1px solid var(--border-light); border-radius: 12px; background: var(--surface-secondary);">
                <div style="font-size: 12px; color: var(--text-tertiary); margin-bottom: 8px;">${signal.label}</div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                   <strong>${signal.value}</strong>
                   <span style="font-size: 10px; font-weight: 700; color: var(--risk-${signal.status.toLowerCase()});">${signal.status}</span>
                </div>
             </div>
           `).join('')}
        </div>
      </div>
    `}
  `;
}


function startStatusPolling() {
  setInterval(async () => {
    try {
      const response = await fetch(`${API_BASE}/streams/status`);
      if (response.ok) {
        state.streamStatus = await response.json();
        updateStreamUI();
      }
    } catch (err) {
      console.error('Polling failed', err);
    }
  }, 5000);
}

function updateStreamUI() {
  const statusIndicator = document.querySelector<HTMLElement>('.stream-indicator');
  const countDisplay = document.querySelector('.message-count');
  const startBtn = document.querySelector<HTMLButtonElement>('.start-stream-btn');
  const stopBtn = document.querySelector<HTMLButtonElement>('.stop-stream-btn');

  if (statusIndicator) {
    statusIndicator.style.background = state.streamStatus.status === 'running' ? '#22c55e' : '#64748b';
    if (state.streamStatus.status === 'running') {
      statusIndicator.classList.add('pulse');
    } else {
      statusIndicator.classList.remove('pulse');
    }
  }

  if (countDisplay) {
    countDisplay.textContent = `Messages: ${state.streamStatus.messages_ingested}`;
  }

  if (startBtn && stopBtn) {
    startBtn.disabled = state.streamStatus.status === 'running';
    stopBtn.disabled = state.streamStatus.status === 'stopped';
  }
}

async function controlStream(action: 'start' | 'stop') {
  try {
    const response = await fetch(`${API_BASE}/streams/${action}`, { method: 'POST' });
    if (response.ok) {
      // Immediately fetch status to update UI
      const statusRes = await fetch(`${API_BASE}/streams/status`);
      if (statusRes.ok) {
        state.streamStatus = await statusRes.json();
        updateStreamUI();
      }
    }
  } catch (err) {
    console.error(`Failed to ${action} stream`, err);
  }
}

function renderModel() {
  const d = state.data || {};
  const riskDist = d.riskDistribution || { labels: [], data: [] };
  return `
    <div style="margin-bottom: 40px;">
      <h1 style="font-size: 32px; color: var(--text-primary)">AI Health Monitoring</h1>
      <p style="color: var(--text-secondary); font-size: 16px; margin-top: 4px;">Real metrics from ${d.dataSource || 'dataset'}</p>
    </div>

    <div class="kpi-grid">
      <div class="card">
        <p class="kpi-label">Non-Default Rate</p>
        <h2 class="kpi-value">${d.accuracy || '—'}</h2>
        <p style="font-size: 12px; color: var(--text-tertiary); margin-top: 12px;">Across ${d.totalSampled || '—'} records</p>
      </div>
      <div class="card">
        <p class="kpi-label">Default Rate</p>
        <h2 class="kpi-value" style="color: var(--risk-high);">${d.defaultRate || '—'}</h2>
        <p style="font-size: 12px; color: var(--text-tertiary); margin-top: 12px;">Observed loan defaults</p>
      </div>
      <div class="card">
        <p class="kpi-label">High-Risk Customers</p>
        <h2 class="kpi-value">${d.highRiskCount || '—'}</h2>
        <p style="font-size: 12px; color: var(--text-tertiary); margin-top: 12px;">${d.highRiskPct || '—'} of portfolio (risk level ≥ 2)</p>
      </div>
    </div>

    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 32px; margin-bottom: 32px;">
      <div class="card">
        <h3 style="margin-bottom: 20px;">Risk Level Distribution <span style="font-size: 12px; font-weight: 400; color: var(--text-tertiary);">All 90k records</span></h3>
        <div style="display: flex; flex-direction: column; gap: 16px;">
          ${riskDist.labels.map((label: string, i: number) => {
    const count = riskDist.data[i] || 0;
    const total = riskDist.data.reduce((a: number, b: number) => a + b, 0) || 1;
    const pct = Math.round(count / total * 100);
    const colors = ['var(--risk-low)', 'var(--risk-medium)', 'var(--risk-high)', 'var(--risk-critical)'];
    return `
              <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                  <span style="font-size: 13px; font-weight: 600;">${label}</span>
                  <span style="font-size: 13px; color: var(--text-secondary);">${count.toLocaleString()} (${pct}%)</span>
                </div>
                <div style="height: 8px; background: var(--surface-tertiary); border-radius: 4px; overflow: hidden;">
                  <div style="width: ${pct}%; height: 100%; background: ${colors[i]}; border-radius: 4px;"></div>
                </div>
              </div>
            `;
  }).join('')}
        </div>
      </div>
      <div class="card">
        <h3 style="margin-bottom: 20px;">Model Configuration</h3>
        <div style="display: flex; flex-direction: column; gap: 16px;">
          <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-subtle); padding-bottom: 12px;">
            <span style="color: var(--text-secondary);">Avg Risk Score</span>
            <strong>${d.avgRiskScore || '—'}</strong>
          </div>
          <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-subtle); padding-bottom: 12px;">
            <span style="color: var(--text-secondary);">F1 Score</span>
            <strong>${d.f1 || '0.865'}</strong>
          </div>
          <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-subtle); padding-bottom: 12px;">
            <span style="color: var(--text-secondary);">Last Retrained</span>
            <strong>${d.lastRetrained || 'Feb 1, 2026'}</strong>
          </div>
          <div style="display: flex; justify-content: space-between;">
            <span style="color: var(--text-secondary);">PSI (Drift)</span>
            <strong>0.12 (Monitor)</strong>
          </div>
        </div>
      </div>
    </div>
  `;
}

init();
