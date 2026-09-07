/**
 * CityFlow AI — Municipal Operations Dashboard Script
 * Incident management, evidence inspection, and repair progress workflow for RoadGuard reports.
 */

let allReports = [];
let currentCityId = 'hyderabad';

const API_BASE = window.location.origin.includes('localhost') || window.location.origin.includes('127.0.0.1')
  ? window.location.origin
  : 'http://127.0.0.1:8000';

document.addEventListener('DOMContentLoaded', async () => {
  initEventListeners();
  await loadCities();
  await loadReports();
  // Auto-poll every 5 seconds to sync with citizen reports
  setInterval(loadReports, 5000);
});

function initEventListeners() {
  document.getElementById('btnRefreshMuni').addEventListener('click', loadReports);

  document.getElementById('muniCitySelect').addEventListener('change', async (e) => {
    currentCityId = e.target.value;
    try {
      await fetch(`${API_BASE}/api/city/switch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ city_id: currentCityId })
      });
      await loadReports();
    } catch (err) {
      console.error('Error switching municipal jurisdiction:', err);
    }
  });

  document.getElementById('filterStatus').addEventListener('change', renderTable);
  document.getElementById('filterSeverity').addEventListener('change', renderTable);

  document.getElementById('btnCloseModal').addEventListener('click', () => {
    document.getElementById('evidenceModal').style.display = 'none';
  });

  document.getElementById('evidenceModal').addEventListener('click', (e) => {
    if (e.target.id === 'evidenceModal') {
      document.getElementById('evidenceModal').style.display = 'none';
    }
  });
}

async function loadCities() {
  try {
    const res = await fetch(`${API_BASE}/api/cities`);
    if (!res.ok) return;
    const cities = await res.json();
    const select = document.getElementById('muniCitySelect');
    if (cities.length > 0) {
      select.innerHTML = '';
      cities.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c.city_id;
        opt.textContent = `🇮🇳 ${c.city_name} (${c.authority.split('(')[1]?.replace(')', '') || c.authority})`;
        select.appendChild(opt);
      });
      select.value = currentCityId;
    }
  } catch (err) {
    console.error('Failed to load cities catalog:', err);
  }
}

async function loadReports() {
  try {
    const res = await fetch(`${API_BASE}/api/roadguard/reports?city_id=${currentCityId}`);
    if (!res.ok) throw new Error('Failed to fetch RoadGuard reports');
    allReports = await res.json();

    updateKpis(allReports);
    renderTable();
  } catch (err) {
    console.error('Error loading reports:', err);
  }
}

function updateKpis(reports) {
  const total = reports.length;
  const critical = reports.filter(r => r.severity === 'CRITICAL' && r.status !== 'RESOLVED').length;
  const review = reports.filter(r => r.status === 'UNDER_REVIEW').length;
  const repairing = reports.filter(r => r.status === 'REPAIRING').length;
  const resolved = reports.filter(r => r.status === 'RESOLVED').length;

  document.getElementById('muniKpiTotal').textContent = total;
  document.getElementById('muniKpiCritical').textContent = critical;
  document.getElementById('muniKpiReview').textContent = review;
  document.getElementById('muniKpiRepairing').textContent = repairing;
  document.getElementById('muniKpiResolved').textContent = resolved;
}

function renderTable() {
  const statusFilter = document.getElementById('filterStatus').value;
  const severityFilter = document.getElementById('filterSeverity').value;

  let filtered = allReports;
  if (statusFilter) {
    filtered = filtered.filter(r => r.status === statusFilter);
  }
  if (severityFilter) {
    filtered = filtered.filter(r => r.severity === severityFilter);
  }

  document.getElementById('reportsCountText').textContent = filtered.length;

  const tbody = document.getElementById('muniTableBody');
  tbody.innerHTML = '';

  if (filtered.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="9" style="text-align:center; padding: 40px; color: var(--text-muted);">
          <i class="fa-solid fa-clipboard-check" style="font-size:32px; margin-bottom:10px; display:block; color:var(--accent-green);"></i>
          No incident reports matching active filter criteria.
        </td>
      </tr>
    `;
    return;
  }

  filtered.forEach(r => {
    const tr = document.createElement('tr');

    const severityClass = r.severity === 'CRITICAL' ? 'badge-red' : (r.severity === 'HIGH' ? 'badge-amber' : 'badge-cyan');
    const statusClass = `status-${r.status}`;

    tr.innerHTML = `
      <td style="font-family:var(--font-mono); font-weight:700; color:var(--accent-cyan);">
        #${r.id}
      </td>
      <td>
        <img src="${r.image_url}" alt="Evidence" class="thumb-evidence" onclick="openEvidenceModal('${r.id}')" title="Click to inspect evidence" />
      </td>
      <td>
        <strong style="color:#fff;">${r.damage_type}</strong>
        <div style="font-size:11px; color:var(--text-muted);">${r.title}</div>
      </td>
      <td>
        <span class="badge ${severityClass}">${r.severity}</span>
      </td>
      <td>
        <strong>${r.zone_name}</strong>
        <div style="font-size:10px; color:var(--text-muted); font-family:var(--font-mono);">${r.lat.toFixed(4)}, ${r.lng.toFixed(4)}</div>
      </td>
      <td style="font-size:11px; color:var(--text-muted); font-family:var(--font-mono);">
        ${r.reported_at}
      </td>
      <td>
        <span class="status-badge-stepper ${statusClass}">
          <i class="fa-solid fa-circle-dot" style="font-size:8px;"></i> ${r.status.replace('_', ' ')}
        </span>
      </td>
      <td style="min-width:140px;">
        <div style="display:flex; justify-content:space-between; font-size:10px; margin-bottom:3px;">
          <span>${r.repair_progress_pct || 0}%</span>
          <span style="color:var(--text-muted);">${r.assigned_crew?.split(' ')[0] || 'Assigned'}</span>
        </div>
        <div style="background:rgba(255,255,255,0.08); height:6px; border-radius:3px; overflow:hidden;">
          <div style="background:${getProgressColor(r.status)}; width:${r.repair_progress_pct || 10}%; height:100%;"></div>
        </div>
      </td>
      <td>
        ${renderActionButtons(r)}
      </td>
    `;

    tbody.appendChild(tr);
  });
}

function getProgressColor(status) {
  if (status === 'RESOLVED') return 'var(--accent-green)';
  if (status === 'REPAIRING') return 'var(--accent-cyan)';
  if (status === 'UNDER_REVIEW') return 'var(--accent-amber)';
  return 'var(--accent-red)';
}

function renderActionButtons(report) {
  if (report.status === 'REPORTED') {
    return `
      <button class="action-btn-sm btn-review" onclick="advanceStatus('${report.id}', 'UNDER_REVIEW')">
        <i class="fa-solid fa-magnifying-glass"></i> Review
      </button>
    `;
  } else if (report.status === 'UNDER_REVIEW') {
    return `
      <button class="action-btn-sm btn-repair" onclick="advanceStatus('${report.id}', 'REPAIRING')">
        <i class="fa-solid fa-truck-pickup"></i> Dispatch Crew
      </button>
    `;
  } else if (report.status === 'REPAIRING') {
    return `
      <button class="action-btn-sm btn-resolve" onclick="advanceStatus('${report.id}', 'RESOLVED')">
        <i class="fa-solid fa-check"></i> Mark Resolved
      </button>
    `;
  } else {
    return `
      <span style="font-size:11px; color:var(--accent-green);"><i class="fa-solid fa-circle-check"></i> Closed</span>
    `;
  }
}

async function advanceStatus(reportId, newStatus) {
  try {
    let action = '';
    let crew = '';
    let progress = null;

    if (newStatus === 'UNDER_REVIEW') {
      action = 'Civil Engineering Inspector dispatched to verify road integrity.';
      crew = 'Municipal Ward Inspector Unit';
      progress = 35;
    } else if (newStatus === 'REPAIRING') {
      action = 'Road Crew and Cold-Mix Asphalt Recycler dispatched on site.';
      crew = 'Municipal Rapid Road Repair Crew #4';
      progress = 75;
    } else if (newStatus === 'RESOLVED') {
      action = 'Asphalt resurfacing completed and road safety verified.';
      crew = 'Work Order Completed & Closed';
      progress = 100;
    }

    const res = await fetch(`${API_BASE}/api/roadguard/reports/${reportId}/status`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        status: newStatus,
        municipal_action: action,
        assigned_crew: crew,
        repair_progress_pct: progress
      })
    });

    if (!res.ok) throw new Error('Status update failed');
    await loadReports();
  } catch (err) {
    alert('Failed to update report status: ' + err.message);
  }
}

function openEvidenceModal(reportId) {
  const r = allReports.find(x => x.id === reportId);
  if (!r) return;

  document.getElementById('modalTicketTitle').textContent = `Work Order #${r.id}: ${r.title}`;
  document.getElementById('modalImg').src = r.image_url;
  document.getElementById('modalType').textContent = r.damage_type;
  document.getElementById('modalConf').textContent = `${r.confidence_score}%`;
  document.getElementById('modalLoc').textContent = `${r.zone_name} (${r.lat.toFixed(4)}, ${r.lng.toFixed(4)})`;
  document.getElementById('modalTime').textContent = r.reported_at;
  document.getElementById('modalDesc').textContent = r.description || 'No additional citizen notes provided.';
  document.getElementById('modalCrew').textContent = `${r.assigned_crew} — ${r.municipal_action}`;

  document.getElementById('evidenceModal').style.display = 'flex';
}
