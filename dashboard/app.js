/**
 * CityFlow AI — Centralized Smart City Intelligence Dashboard Application.
 * Full WebSocket real-time telemetry, interactive Leaflet road network map,
 * Smart Traffic & AI Prediction, 3 Red Zones, Saved Places, RoadGuard Citizen Reporting,
 * CCTV AI Monitoring Demo, and CITYFLOW AI Voice Assistant.
 */

// Global State
let map;
let zoneMarkers = {};
let roadLines = [];
let activeRoutePolyline = null;
let ambulanceMarkers = {};
let signalMarkers = {};
let redZoneLayers = [];
let roadGuardMarkers = [];
let allZonesData = [];
let ws = null;
let reconnectTimer = null;
let currentCityId = 'hyderabad';
let savedPlaces = {};
let isListening = false;
let recognition = null;
let cctvInterval = null;

const API_BASE = window.location.origin.includes('localhost') || window.location.origin.includes('127.0.0.1')
  ? window.location.origin
  : 'http://127.0.0.1:8000';

const WS_URL = API_BASE.replace('http', 'ws') + '/ws/live';

// ==========================================================
// 1. INITIALIZATION & LEAFLET MAP SETUP
// ==========================================================
document.addEventListener('DOMContentLoaded', async () => {
  initMap();
  initTabNavigation();
  initEventListeners();
  await loadCitiesCatalog();
  await loadSavedPlaces();
  await loadInitialCityData();
  await loadTopRedZones();
  await loadRoadGuardReports();
  checkEmergencyRoadAlerts();
  initCctvDemo();
  initVoiceAssistant();
  connectWebSocket();
});

function initMap() {
  map = L.map('cityMap', {
    zoomControl: false,
    attributionControl: false
  }).setView([17.4435, 78.3772], 14);

  L.control.zoom({ position: 'topright' }).addTo(map);

  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    maxZoom: 19,
    subdomains: 'abcd'
  }).addTo(map);
}

// ==========================================================
// 2. DATA LOADING & MULTI-CITY LOGIC
// ==========================================================
async function loadCitiesCatalog() {
  try {
    const res = await fetch(`${API_BASE}/api/cities`);
    if (!res.ok) return;
    const cities = await res.json();
    const select = document.getElementById('citySelector');
    if (select && cities.length > 0) {
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
    console.error('Error loading cities catalog:', err);
  }
}

async function loadInitialCityData(cityId = null) {
  try {
    const url = cityId ? `${API_BASE}/api/city?city_id=${cityId}` : `${API_BASE}/api/city`;
    const resCity = await fetch(url);
    if (resCity.ok) {
      const cityInfo = await resCity.json();
      currentCityId = cityInfo.city_id || currentCityId;
      document.getElementById('topAuthoritySubtitle').textContent = `${cityInfo.authority} • ${cityInfo.city_name}`;
      if (document.getElementById('citySelector')) document.getElementById('citySelector').value = currentCityId;

      if (cityInfo.center && map) {
        map.setView([cityInfo.center.lat, cityInfo.center.lng], cityInfo.center.zoom || 14);
      }
    }

    const res = await fetch(`${API_BASE}/api/zones`);
    if (!res.ok) throw new Error('Failed to load zones');
    allZonesData = await res.json();

    populateZoneDropdowns(allZonesData);
    renderRoadNetwork(allZonesData);
    renderZoneMarkers(allZonesData);
    logEvent(`Loaded ${allZonesData.length} traffic zones for active metro`, 'info');

    // Also load module initial states
    fetchSignals();
    fetchParking();
    fetchTransit();
    fetchLogistics();
    fetchWaste();
    fetchFlood();
    fetchIncidents();
    loadTopRedZones();
    loadRoadGuardReports();
  } catch (err) {
    console.error('Initialization error:', err);
    logEvent('Warning: Central server connection pending...', 'alert');
  }
}

function renderRoadNetwork(zones) {
  roadLines.forEach(line => map.removeLayer(line));
  roadLines = [];

  const zoneMap = {};
  zones.forEach(z => { zoneMap[z.id] = z; });
  const drawnEdges = new Set();

  zones.forEach(zone => {
    (zone.connections || []).forEach(conn => {
      const target = zoneMap[conn.to];
      if (target) {
        const edgeKey = [zone.id, target.id].sort().join('--');
        if (!drawnEdges.has(edgeKey)) {
          drawnEdges.add(edgeKey);

          const isCongested = (zone.congestion_factor > 0.6 || target.congestion_factor > 0.6);
          const isBlocked = (zone.is_blocked || target.is_blocked);
          
          let color = '#2a3b5c';
          let dashArray = null;
          let weight = 4;

          if (isBlocked) {
            color = '#ff4b2b';
            dashArray = '6, 6';
            weight = 5;
          } else if (isCongested) {
            color = '#f6d365';
          }

          const line = L.polyline([[zone.lat, zone.lng], [target.lat, target.lng]], {
            color: color,
            weight: weight,
            opacity: 0.65,
            dashArray: dashArray
          }).addTo(map);

          roadLines.push(line);
        }
      }
    });
  });
}

function renderZoneMarkers(zones) {
  zones.forEach(zone => {
    let color = '#38ef7d'; // default green
    if (zone.is_blocked) {
      color = '#ff4b2b'; // blocked red
    } else if (zone.flood_level_cm > 10) {
      color = '#3b82f6'; // flooded blue
    } else if (zone.density_color) {
      color = `rgb(${zone.density_color[0]}, ${zone.density_color[1]}, ${zone.density_color[2]})`;
    }

    const popupHtml = `
      <div style="font-family: 'Inter', sans-serif; min-width: 180px; color: #111;">
        <strong style="font-size: 13px;">${zone.name}</strong><br/>
        <span style="font-size: 11px; color: #666;">Type: ${zone.type.toUpperCase()} | ${zone.camera_id}</span>
        <hr style="margin: 6px 0; border: none; border-top: 1px solid #ddd;"/>
        <div style="font-size: 12px; margin-bottom: 2px;">
          <strong>Vehicles:</strong> ${zone.vehicle_count}
        </div>
        <div style="font-size: 12px; margin-bottom: 2px;">
          <strong>Status:</strong> <span style="font-weight: 700;">${zone.density_label}</span>
        </div>
        <div style="font-size: 12px; margin-bottom: 2px;">
          <strong>Avg Speed:</strong> ${zone.avg_speed_kmh} km/h
        </div>
        ${zone.signal_id ? `<div style="font-size: 12px; color: #ff9800;"><strong>Signal:</strong> ${zone.signal_state} (${zone.signal_timer_sec}s)</div>` : ''}
        ${zone.is_blocked ? `<div style="font-size: 12px; color: #d32f2f; font-weight: bold;">⛔ ${zone.blockage_reason}</div>` : ''}
      </div>
    `;

    if (zoneMarkers[zone.id]) {
      zoneMarkers[zone.id].setStyle({ color: color, fillColor: color });
      zoneMarkers[zone.id].setPopupContent(popupHtml);
    } else {
      const marker = L.circleMarker([zone.lat, zone.lng], {
        radius: 12,
        color: color,
        weight: 2,
        fillColor: color,
        fillOpacity: 0.75
      }).addTo(map);

      marker.bindPopup(popupHtml);
      zoneMarkers[zone.id] = marker;
    }
  });
}

function populateZoneDropdowns(zones) {
  const selects = [
    document.getElementById('smartFromZone'),
    document.getElementById('smartToZone'),
    document.getElementById('routeOrigin'),
    document.getElementById('routeDest'),
    document.getElementById('ambPickupZone'),
    document.getElementById('ambHospZone'),
    document.getElementById('accidentZoneSelect'),
    document.getElementById('parkingDestSelect'),
    document.getElementById('reportZoneSelect'),
    document.getElementById('saveHomeSelect'),
    document.getElementById('saveWorkSelect'),
    document.getElementById('saveCustomZoneSelect')
  ];

  selects.forEach(s => { if (s) s.innerHTML = ''; });

  zones.forEach(z => {
    const opt = document.createElement('option');
    opt.value = z.id;
    opt.textContent = `${z.name} (${z.density_label})`;

    selects.forEach(s => {
      if (s) s.appendChild(opt.cloneNode(true));
    });
  });

  const smartFrom = document.getElementById('smartFromZone');
  const smartTo = document.getElementById('smartToZone');
  if (smartFrom && smartFrom.options.length > 0) smartFrom.selectedIndex = 0;
  if (smartTo && smartTo.options.length > 2) smartTo.selectedIndex = 2;

  const originSelect = document.getElementById('routeOrigin');
  const destSelect = document.getElementById('routeDest');
  if (originSelect && originSelect.options.length > 0) originSelect.selectedIndex = 0;
  if (destSelect && destSelect.options.length > 3) destSelect.selectedIndex = 3;

  const ambPickup = document.getElementById('ambPickupZone');
  const ambHosp = document.getElementById('ambHospZone');
  if (ambPickup && ambPickup.options.length > 1) ambPickup.selectedIndex = 1;
  if (ambHosp) {
    const hospOpt = Array.from(ambHosp.options).find(o => o.value.includes('hospital'));
    if (hospOpt) hospOpt.selected = true;
  }
}

// ==========================================================
// 3. SAVED PLACES MANAGEMENT
// ==========================================================
async function loadSavedPlaces() {
  try {
    const res = await fetch(`${API_BASE}/api/saved-places`);
    if (!res.ok) return;
    savedPlaces = await res.json();
    renderSavedPlacesChips();
  } catch (err) {
    console.error('Error loading saved places:', err);
  }
}

function renderSavedPlacesChips() {
  const container = document.getElementById('homeSavedPlacesChips');
  if (!container) return;
  container.innerHTML = '';

  if (savedPlaces.home) {
    const chip = createPlaceChip(savedPlaces.home.name, 'fa-house', () => {
      applySavedPlaceToInputs(savedPlaces.home.zone_id);
    });
    container.appendChild(chip);
  }

  if (savedPlaces.work) {
    const chip = createPlaceChip(savedPlaces.work.name, 'fa-briefcase', () => {
      applySavedPlaceToInputs(savedPlaces.work.zone_id);
    });
    container.appendChild(chip);
  }

  (savedPlaces.custom || []).forEach(p => {
    const chip = createPlaceChip(p.name, p.icon || 'fa-location-dot', () => {
      applySavedPlaceToInputs(p.zone_id);
    });
    container.appendChild(chip);
  });
}

function createPlaceChip(name, icon, onClick) {
  const btn = document.createElement('button');
  btn.className = 'place-chip';
  btn.innerHTML = `<i class="fa-solid ${icon}"></i> ${name}`;
  btn.addEventListener('click', onClick);
  return btn;
}

function applySavedPlaceToInputs(zoneId) {
  const fromSelect = document.getElementById('smartFromZone');
  const toSelect = document.getElementById('smartToZone');

  if (!fromSelect.value || fromSelect.value === zoneId) {
    fromSelect.value = zoneId;
  } else {
    toSelect.value = zoneId;
  }
  showTraffic();
}

function openSavedPlacesModal() {
  const modal = document.getElementById('savedPlacesModal');
  if (!modal) return;

  if (savedPlaces.home && document.getElementById('saveHomeSelect')) {
    document.getElementById('saveHomeSelect').value = savedPlaces.home.zone_id;
  }
  if (savedPlaces.work && document.getElementById('saveWorkSelect')) {
    document.getElementById('saveWorkSelect').value = savedPlaces.work.zone_id;
  }

  renderCustomPlacesListInModal();
  modal.style.display = 'flex';
}

function renderCustomPlacesListInModal() {
  const list = document.getElementById('customPlacesList');
  if (!list) return;
  list.innerHTML = '';

  (savedPlaces.custom || []).forEach(p => {
    const item = document.createElement('div');
    item.style.cssText = 'display:flex; justify-content:space-between; align-items:center; background:rgba(255,255,255,0.05); padding:6px 10px; border-radius:6px; font-size:12px;';
    item.innerHTML = `
      <div><i class="fa-solid ${p.icon || 'fa-location-dot'} text-cyan"></i> <strong>${p.name}</strong> (${p.zone_name})</div>
      <button onclick="deleteCustomPlace('${p.id}')" style="background:none; border:none; color:#ff6b6b; cursor:pointer;"><i class="fa-solid fa-trash"></i></button>
    `;
    list.appendChild(item);
  });
}

async function addCustomPlace() {
  const nameInput = document.getElementById('customPlaceNameInput');
  const zoneSelect = document.getElementById('saveCustomZoneSelect');
  const name = nameInput.value.trim();
  const zoneId = zoneSelect.value;

  if (!name || !zoneId) {
    alert('Please enter a place name and select a zone.');
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/api/saved-places`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type: 'CUSTOM', name: name, zone_id: zoneId })
    });
    if (!res.ok) {
      const err = await res.json();
      alert(err.detail || 'Failed to add custom place');
      return;
    }
    nameInput.value = '';
    await loadSavedPlaces();
    renderCustomPlacesListInModal();
  } catch (e) {
    alert('Error adding custom place');
  }
}

async function deleteCustomPlace(placeId) {
  try {
    await fetch(`${API_BASE}/api/saved-places/${placeId}`, { method: 'DELETE' });
    await loadSavedPlaces();
    renderCustomPlacesListInModal();
  } catch (e) {
    console.error(e);
  }
}

async function savePlacesSubmit() {
  const homeZone = document.getElementById('saveHomeSelect').value;
  const workZone = document.getElementById('saveWorkSelect').value;

  try {
    if (homeZone) {
      await fetch(`${API_BASE}/api/saved-places`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: 'HOME', name: 'Home', zone_id: homeZone })
      });
    }
    if (workZone) {
      await fetch(`${API_BASE}/api/saved-places`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: 'WORK', name: 'Work', zone_id: workZone })
      });
    }
    await loadSavedPlaces();
    document.getElementById('savedPlacesModal').style.display = 'none';
    logEvent('Saved places updated successfully', 'success');
  } catch (e) {
    alert('Error saving places');
  }
}

// ==========================================================
// 4. SMART TRAFFIC, AI PREDICTION & 3 RED ZONES
// ==========================================================
async function showTraffic() {
  const fromZone = document.getElementById('smartFromZone').value;
  const toZone = document.getElementById('smartToZone').value;

  if (!fromZone || !toZone) return;
  if (fromZone === toZone) {
    alert('Please choose two different origin and destination areas.');
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/api/route?from=${fromZone}&to=${toZone}`);
    if (!res.ok) throw new Error('No path found');
    const data = await res.json();

    const resultCard = document.getElementById('smartTrafficResultCard');
    resultCard.style.display = 'block';

    document.getElementById('smartEta').textContent = data.estimated_time_min;
    document.getElementById('smartDistance').textContent = data.total_distance_km;
    document.getElementById('smartDelaySaved').textContent = Math.round(data.congestion_penalty_sec);

    // Calculate average path congestion
    const pathZones = data.path.map(zid => allZonesData.find(z => z.id === zid)).filter(Boolean);
    const avgCong = pathZones.length > 0
      ? Math.round((pathZones.reduce((acc, z) => acc + (z.congestion_factor || 0.2), 0) / pathZones.length) * 100)
      : 45;

    const meterBar = document.getElementById('smartTrafficMeterBar');
    const statusBadge = document.getElementById('smartTrafficStatusBadge');

    meterBar.style.width = `${Math.min(99, Math.max(15, avgCong))}%`;
    if (avgCong > 75) {
      statusBadge.className = 'badge badge-red';
      statusBadge.textContent = `${avgCong}% HEAVY 🔴`;
    } else if (avgCong > 45) {
      statusBadge.className = 'badge badge-amber';
      statusBadge.textContent = `${avgCong}% MODERATE`;
    } else {
      statusBadge.className = 'badge badge-green';
      statusBadge.textContent = `${avgCong}% SMOOTH`;
    }

    // Draw Route on Map
    if (activeRoutePolyline) map.removeLayer(activeRoutePolyline);
    const latlngs = data.path.map(zid => {
      const zone = allZonesData.find(z => z.id === zid);
      return zone ? [zone.lat, zone.lng] : null;
    }).filter(Boolean);

    activeRoutePolyline = L.polyline(latlngs, {
      color: '#00f2fe',
      weight: 6,
      opacity: 0.95
    }).addTo(map);

    map.fitBounds(activeRoutePolyline.getBounds(), { padding: [40, 40] });

    // AI Traffic Prediction
    fetchAiTrafficPrediction(fromZone, toZone);
    // Check Emergency Alerts
    checkEmergencyRoadAlerts(data.path.join(','));
  } catch (err) {
    console.error('Error showing traffic:', err);
  }
}

async function fetchAiTrafficPrediction(fromZone, toZone) {
  try {
    const res = await fetch(`${API_BASE}/api/traffic/predict?from=${fromZone}&to=${toZone}`);
    if (!res.ok) return;
    const data = await res.json();

    const p15 = data.predictions.find(p => p.horizon_min === 15);
    const p30 = data.predictions.find(p => p.horizon_min === 30);
    const p60 = data.predictions.find(p => p.horizon_min === 60);

    if (p15) {
      document.getElementById('pred15Val').textContent = `${p15.congestion_pct}%`;
      document.getElementById('pred15Delta').textContent = `${p15.delta >= 0 ? '+' : ''}${p15.delta}%`;
    }
    if (p30) {
      document.getElementById('pred30Val').textContent = `${p30.congestion_pct}%`;
      document.getElementById('pred30Delta').textContent = `+${p30.delta}% 🔴`;
    }
    if (p60) {
      document.getElementById('pred60Val').textContent = `${p60.congestion_pct}%`;
      document.getElementById('pred60Delta').textContent = `${p60.delta >= 0 ? '+' : ''}${p60.delta}%`;
    }

    if (data.recommendation) {
      document.getElementById('aiRecommendationText').innerHTML = `<i class="fa-solid fa-lightbulb text-amber"></i> ${data.recommendation}`;
    }
  } catch (err) {
    console.error('Error in AI prediction:', err);
  }
}

async function loadTopRedZones() {
  try {
    const res = await fetch(`${API_BASE}/api/traffic/red-zones`);
    if (!res.ok) return;
    const redZones = await res.json();

    // Clear previous red zone markers
    redZoneLayers.forEach(l => map.removeLayer(l));
    redZoneLayers = [];

    const container = document.getElementById('homeRedZonesList');
    if (container) container.innerHTML = '';

    redZones.forEach((rz, idx) => {
      // 1. Leaflet Radar Circle on Map
      const circle = L.circle([rz.lat, rz.lng], {
        radius: 400,
        color: '#ff4b2b',
        weight: 2,
        fillColor: '#ff4b2b',
        fillOpacity: 0.22,
        dashArray: '4, 4'
      }).addTo(map);

      const marker = L.circleMarker([rz.lat, rz.lng], {
        radius: 11,
        color: '#ff4b2b',
        fillColor: '#ff2020',
        fillOpacity: 0.9,
        weight: 3
      }).addTo(map);

      marker.bindPopup(`
        <div style="font-family:'Inter', sans-serif; color:#111; min-width:180px;">
          <strong style="color:#d32f2f;">🔴 RED ZONE #${rz.rank}: ${rz.name}</strong><br/>
          <span style="font-size:11px; color:#555;">Congestion: <strong>${rz.congestion_pct}%</strong> | Cam: ${rz.camera_id}</span>
          <hr style="margin:5px 0; border:none; border-top:1px solid #eee;"/>
          <div style="font-size:11px; color:#222;">${rz.bypass_advice}</div>
        </div>
      `);

      redZoneLayers.push(circle, marker);

      // 2. Add to Sidebar Card List
      if (container) {
        const item = document.createElement('div');
        item.className = 'redzone-card-item';
        item.innerHTML = `
          <div>
            <div class="redzone-title">${idx + 1}. ${rz.name}</div>
            <div class="redzone-sub">${rz.bypass_advice}</div>
          </div>
          <span class="badge badge-red">${rz.congestion_pct}% LOAD</span>
        `;
        container.appendChild(item);
      }
    });
  } catch (err) {
    console.error('Error loading top red zones:', err);
  }
}

// ==========================================================
// 5. ROADGUARD CITIZEN REPORTING & REPAIR STATUS
// ==========================================================
let selectedDamagePhotoUrl = 'https://images.unsplash.com/photo-1515162816999-a0c47dc192f7?w=600&auto=format&fit=crop&q=80';
let selectedDamageType = 'Severe Pothole (Grade 4 Hazard)';

async function loadRoadGuardReports() {
  try {
    const res = await fetch(`${API_BASE}/api/roadguard/reports?city_id=${currentCityId}`);
    if (!res.ok) return;
    const reports = await res.json();

    const countBadge = document.getElementById('roadGuardCountBadge');
    if (countBadge) countBadge.textContent = `${reports.length} Active`;

    const listContainer = document.getElementById('roadGuardReportsList');
    if (!listContainer) return;
    listContainer.innerHTML = '';

    // Clear previous damage map pins
    roadGuardMarkers.forEach(m => map.removeLayer(m));
    roadGuardMarkers = [];

    reports.forEach(r => {
      // 1. Add Marker on Map
      const isCritical = r.severity === 'CRITICAL';
      const mColor = isCritical ? '#ff4b2b' : (r.severity === 'HIGH' ? '#f59e0b' : '#38ef7d');

      const hazardIcon = L.divIcon({
        className: 'custom-hazard-icon',
        html: `<div style="background:${mColor}; color:white; width:26px; height:26px; border-radius:50%; display:flex; align-items:center; justify-content:center; box-shadow:0 0 12px ${mColor}; font-size:12px;"><i class="fa-solid fa-triangle-exclamation"></i></div>`,
        iconSize: [26, 26],
        iconAnchor: [13, 13]
      });

      const mapPin = L.marker([r.lat, r.lng], { icon: hazardIcon }).addTo(map);
      mapPin.bindPopup(`
        <div style="font-family:'Inter', sans-serif; color:#111; min-width:180px;">
          <strong style="color:#d32f2f;"><i class="fa-solid fa-triangle-exclamation"></i> ${r.damage_type}</strong><br/>
          <span style="font-size:11px; color:#555;">#${r.id} | Status: <strong>${r.status}</strong></span>
          <hr style="margin:5px 0; border:none; border-top:1px solid #eee;"/>
          <div style="font-size:12px; margin-bottom:4px;">${r.title}</div>
          <div style="font-size:10px; color:#666;">${r.zone_name} • ${r.reported_at}</div>
        </div>
      `);
      roadGuardMarkers.push(mapPin);

      // 2. Sidebar Report Card with Lifecycle Stepper
      const card = document.createElement('div');
      card.className = 'list-card';
      card.innerHTML = `
        <div class="card-top">
          <span class="card-title">#${r.id}: ${r.damage_type}</span>
          <span class="badge ${isCritical ? 'badge-red' : 'badge-amber'}">${r.severity}</span>
        </div>
        <div style="display:flex; gap:10px; margin: 6px 0;">
          <img src="${r.image_url}" style="width:60px; height:60px; border-radius:6px; object-fit:cover; border:1px solid var(--border-subtle);" />
          <div style="flex:1; font-size:11px; color:var(--text-secondary);">
            <div><strong>Location:</strong> ${r.zone_name}</div>
            <div><strong>AI Confidence:</strong> <span class="text-green">${r.confidence_score}%</span></div>
            <div style="color:var(--text-muted); font-size:10px; margin-top:2px;">${r.description}</div>
          </div>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:6px; padding-top:6px; border-top:1px solid rgba(255,255,255,0.06);">
          <span class="status-badge-stepper status-${r.status}">
            <i class="fa-solid fa-circle-dot" style="font-size:8px;"></i> ${r.status.replace('_', ' ')}
          </span>
          <span style="font-size:10px; color:var(--accent-cyan); font-family:var(--font-mono);">${r.repair_progress_pct || 10}% Fixed</span>
        </div>
      `;
      listContainer.appendChild(card);
    });
  } catch (err) {
    console.error('Error loading RoadGuard reports:', err);
  }
}

async function submitCitizenReport() {
  const title = document.getElementById('reportTitle').value.trim();
  const desc = document.getElementById('reportDesc').value.trim();
  const zoneId = document.getElementById('reportZoneSelect').value;

  if (!title) {
    alert('Please enter a hazard title.');
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/api/roadguard/report`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: title,
        description: desc || 'Citizen reported surface damage.',
        zone_id: zoneId,
        photo_url: selectedDamagePhotoUrl,
        damage_type: selectedDamageType,
        city_id: currentCityId
      })
    });

    if (!res.ok) throw new Error('Submission failed');
    const data = await res.json();

    document.getElementById('citizenReportModal').style.display = 'none';
    logEvent(`📸 CITIZEN REPORT FILED: #${data.report.id} (${data.report.damage_type}) — AI Severity: ${data.report.severity}`, 'alert');

    await loadRoadGuardReports();
    switchTab('roadguard');
  } catch (err) {
    alert('Error submitting report: ' + err.message);
  }
}

async function checkEmergencyRoadAlerts(pathZoneIds = '') {
  try {
    const url = pathZoneIds
      ? `${API_BASE}/api/roadguard/alerts?route_zones=${pathZoneIds}`
      : `${API_BASE}/api/roadguard/alerts`;
    const res = await fetch(url);
    if (!res.ok) return;
    const data = await res.json();

    const banner = document.getElementById('emergencyRoadAlert');
    if (!banner) return;

    if (data.active_alerts && data.active_alerts.length > 0) {
      const first = data.active_alerts[0];
      document.getElementById('roadAlertTitle').textContent = `🚨 ${first.severity} COMMUTE HAZARD: ${first.damage_type}`;
      document.getElementById('roadAlertDesc').textContent = `${first.title} near ${first.zone_name}. Municipal repair unit active.`;
      banner.style.display = 'flex';
    } else {
      banner.style.display = 'none';
    }
  } catch (err) {
    console.error('Error checking emergency alerts:', err);
  }
}

// ==========================================================
// 6. CCTV AI MONITORING DEMO STREAM
// ==========================================================
function initCctvDemo() {
  const canvas = document.getElementById('cctvCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  let frameCount = 0;
  let vehicles = [
    { x: 50, y: 120, vx: 2.5, w: 45, h: 25, label: 'CAR 82%' },
    { x: 180, y: 150, vx: 1.8, w: 70, h: 32, label: 'BUS 94%' },
    { x: 300, y: 110, vx: 3.2, w: 35, h: 18, label: 'TAXI 88%' }
  ];

  if (cctvInterval) clearInterval(cctvInterval);

  cctvInterval = setInterval(() => {
    frameCount++;
    // Dark road background
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Road lanes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, 90); ctx.lineTo(canvas.width, 90);
    ctx.moveTo(0, 180); ctx.lineTo(canvas.width, 180);
    ctx.stroke();

    // Dashed center lane line
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.setLineDash([12, 10]);
    ctx.beginPath();
    ctx.moveTo(0, 135); ctx.lineTo(canvas.width, 135);
    ctx.stroke();
    ctx.setLineDash([]);

    // Pothole Detection Bounding Box on Road
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    ctx.strokeRect(160, 105, 55, 30);
    ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
    ctx.fillRect(160, 105, 55, 30);
    ctx.fillStyle = '#ef4444';
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.fillText('POTHOLE (CONF 96%)', 160, 100);

    // Vehicles with bounding boxes
    vehicles.forEach(v => {
      v.x += v.vx;
      if (v.x > canvas.width) v.x = -v.w;

      ctx.strokeStyle = '#00f2fe';
      ctx.lineWidth = 2;
      ctx.strokeRect(v.x, v.y, v.w, v.h);

      ctx.fillStyle = 'rgba(0, 242, 254, 0.15)';
      ctx.fillRect(v.x, v.y, v.w, v.h);

      ctx.fillStyle = '#00f2fe';
      ctx.font = '9px JetBrains Mono, monospace';
      ctx.fillText(v.label, v.x, v.y - 4);
    });

    // Scanline effect
    const scanY = (frameCount * 3) % canvas.height;
    ctx.fillStyle = 'rgba(0, 242, 254, 0.12)';
    ctx.fillRect(0, scanY, canvas.width, 3);
  }, 33);
}

// ==========================================================
// 7. CITYFLOW AI VOICE ASSISTANT
// ==========================================================
function initVoiceAssistant() {
  const micBtn = document.getElementById('voiceMicButton');
  const navTrigger = document.getElementById('btnToggleVoiceAssistant');
  const closeBtn = document.getElementById('btnCloseVoiceCard');
  const sendBtn = document.getElementById('btnSendVoiceText');
  const fallbackInput = document.getElementById('voiceFallbackInput');

  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRec) {
    recognition = new SpeechRec();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-IN';

    recognition.onstart = () => {
      isListening = true;
      micBtn?.classList.add('listening');
      const wave = document.getElementById('voiceWaveAnimation');
      if (wave) wave.style.display = 'block';
      updateVoiceTranscript('Listening... Speak now...');
    };

    recognition.onresult = (event) => {
      const speechText = event.results[0][0].transcript;
      updateVoiceTranscript(`"${speechText}"`);
      handleVoiceQuery(speechText);
    };

    recognition.onerror = () => {
      stopVoiceListening();
      updateVoiceTranscript('Could not detect audio. Try the quick prompt buttons below.');
    };

    recognition.onend = () => {
      stopVoiceListening();
    };
  }

  function toggleVoice() {
    const card = document.getElementById('voiceTranscriptCard');
    if (!card) return;

    if (card.style.display === 'none' || !card.style.display) {
      card.style.display = 'block';
      if (recognition && !isListening) {
        try { recognition.start(); } catch (e) { }
      }
    } else {
      card.style.display = 'none';
      stopVoiceListening();
    }
  }

  micBtn?.addEventListener('click', toggleVoice);
  navTrigger?.addEventListener('click', toggleVoice);
  closeBtn?.addEventListener('click', () => {
    document.getElementById('voiceTranscriptCard').style.display = 'none';
    stopVoiceListening();
  });

  sendBtn?.addEventListener('click', () => {
    const query = fallbackInput.value.trim();
    if (query) {
      handleVoiceQuery(query);
      fallbackInput.value = '';
    }
  });

  fallbackInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      const query = fallbackInput.value.trim();
      if (query) {
        handleVoiceQuery(query);
        fallbackInput.value = '';
      }
    }
  });
}

function stopVoiceListening() {
  isListening = false;
  document.getElementById('voiceMicButton')?.classList.remove('listening');
  const wave = document.getElementById('voiceWaveAnimation');
  if (wave) wave.style.display = 'none';
}

function updateVoiceTranscript(text) {
  const el = document.getElementById('voiceTranscriptText');
  if (el) el.textContent = text;
}

function speakVoiceResponse(text) {
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.05;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  }
}

async function handleVoiceQuery(query) {
  const q = query.toLowerCase();
  const card = document.getElementById('voiceTranscriptCard');
  if (card) card.style.display = 'block';

  if (q.includes('traffic on my route') || q.includes('traffic on route') || q.includes('commute')) {
    updateVoiceTranscript('Analyzing your saved commute route traffic...');
    if (savedPlaces.home && savedPlaces.work) {
      document.getElementById('smartFromZone').value = savedPlaces.home.zone_id;
      document.getElementById('smartToZone').value = savedPlaces.work.zone_id;
      await showTraffic();
      const eta = document.getElementById('smartEta').textContent;
      const resp = `Traffic on your commute from ${savedPlaces.home.name} to ${savedPlaces.work.name} is currently estimated at ${eta} minutes.`;
      updateVoiceTranscript(resp);
      speakVoiceResponse(resp);
    } else {
      await showTraffic();
      const eta = document.getElementById('smartEta').textContent;
      const resp = `Current route travel time is estimated at ${eta} minutes.`;
      updateVoiceTranscript(resp);
      speakVoiceResponse(resp);
    }
  } else if (q.includes('fastest route') || q.includes('fast route')) {
    const from = document.getElementById('smartFromZone').value;
    const to = document.getElementById('smartToZone').value;
    const resp = `Opening the fast route navigator bypassing all 3 active red zones.`;
    updateVoiceTranscript(resp);
    speakVoiceResponse(resp);
    setTimeout(() => {
      window.location.href = `fast-route.html?from=${from}&to=${to}&city=${currentCityId}`;
    }, 1200);
  } else if (q.includes('road problem') || q.includes('hazard') || q.includes('pothole')) {
    try {
      const res = await fetch(`${API_BASE}/api/roadguard/alerts`);
      const data = await res.json();
      if (data.active_alerts && data.active_alerts.length > 0) {
        const first = data.active_alerts[0];
        const resp = `Caution! A ${first.damage_type} is reported near ${first.zone_name}. A municipal repair crew has been dispatched.`;
        updateVoiceTranscript(resp);
        speakVoiceResponse(resp);
        switchTab('roadguard');
      } else {
        const resp = `Good news! No severe road problems or damage alerts are currently reported on your active corridor.`;
        updateVoiceTranscript(resp);
        speakVoiceResponse(resp);
      }
    } catch (e) {
      updateVoiceTranscript('Unable to check road hazards right now.');
    }
  } else if (q.includes('red zone') || q.includes('bottleneck')) {
    try {
      const res = await fetch(`${API_BASE}/api/traffic/red-zones`);
      const redZones = await res.json();
      const names = redZones.map(r => r.name).join(', ');
      const resp = `The 3 active high-traffic red zones in ${currentCityId.toUpperCase()} are ${names}.`;
      updateVoiceTranscript(resp);
      speakVoiceResponse(resp);
    } catch (e) {
      updateVoiceTranscript('Could not fetch red zones.');
    }
  } else if (q.includes('switch to') || q.includes('city') || q.includes('bengaluru') || q.includes('mumbai') || q.includes('delhi')) {
    let target = 'hyderabad';
    if (q.includes('bengaluru') || q.includes('bangalore')) target = 'bengaluru';
    else if (q.includes('mumbai')) target = 'mumbai';
    else if (q.includes('delhi')) target = 'delhi';
    else if (q.includes('pune')) target = 'pune';
    else if (q.includes('chennai')) target = 'chennai';

    try {
      await fetch(`${API_BASE}/api/city/switch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ city_id: target })
      });
      await loadInitialCityData(target);
      const resp = `Switched jurisdiction to ${target.toUpperCase()}. Road network and telemetry refreshed.`;
      updateVoiceTranscript(resp);
      speakVoiceResponse(resp);
    } catch (e) {
      updateVoiceTranscript('Failed to switch city.');
    }
  } else {
    const resp = `I heard: "${query}". You can ask about route traffic, the fastest route, or road hazard alerts.`;
    updateVoiceTranscript(resp);
    speakVoiceResponse(resp);
  }
}

// ==========================================================
// 8. WEBSOCKET REAL-TIME STREAMING
// ==========================================================
function connectWebSocket() {
  const pill = document.getElementById('wsStatusPill');
  const text = document.getElementById('wsStatusText');
  const dot = pill?.querySelector('.status-dot');

  try {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      if (text) text.textContent = 'CONNECTED • LIVE';
      if (dot) dot.className = 'status-dot connected';
      logEvent('Centralized WebSocket stream established successfully', 'success');
      if (reconnectTimer) {
        clearInterval(reconnectTimer);
        reconnectTimer = null;
      }
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        handleLiveServerMessage(msg);
      } catch (err) {
        console.error('Error parsing live WS payload:', err);
      }
    };

    ws.onclose = () => {
      if (text) text.textContent = 'OFFLINE (RETRYING)';
      if (dot) dot.className = 'status-dot pulsing';
      scheduleReconnect();
    };

    ws.onerror = () => {
      ws.close();
    };
  } catch (e) {
    scheduleReconnect();
  }
}

function scheduleReconnect() {
  if (!reconnectTimer) {
    reconnectTimer = setInterval(connectWebSocket, 3000);
  }
}

function handleLiveServerMessage(msg) {
  if (msg.type === 'CITY_STATE_UPDATE' || msg.type === 'INITIAL_CITY_SNAPSHOT') {
    if (msg.stats) {
      document.getElementById('kpiVehicles').textContent = msg.stats.total_active_vehicles || 0;
      document.getElementById('kpiCongestion').textContent = (msg.stats.congestion_rate_pct || 0) + '%';
      document.getElementById('kpiSpeed').textContent = (msg.stats.city_avg_speed_kmh || 0) + ' km/h';
      document.getElementById('kpiAmbulances').textContent = `${msg.stats.active_ambulances || 0} Active`;
      document.getElementById('kpiIncidents').textContent = msg.stats.open_incidents || 0;
    }

    if (msg.zones) {
      allZonesData = msg.zones;
      renderZoneMarkers(msg.zones);
      renderRoadNetwork(msg.zones);
    }

    if (msg.ambulances) {
      renderAmbulancesOnMap(msg.ambulances);
      renderAmbulanceList(msg.ambulances);
    }

    if (msg.signals) renderSignalsList(msg.signals);
    if (msg.accidents) renderAccidentsList(msg.accidents);
    if (msg.incidents) renderIncidentsList(msg.incidents);
  }
}

// ==========================================================
// 9. MAP OVERLAYS: AMBULANCES & SIGNALS
// ==========================================================
function renderAmbulancesOnMap(ambulances) {
  const banner = document.getElementById('activeEmergencyBanner');
  const activeCount = ambulances.filter(a => a.status !== 'ARRIVED').length;

  if (banner) {
    if (activeCount > 0) {
      banner.style.display = 'flex';
      const first = ambulances.find(a => a.status !== 'ARRIVED');
      document.getElementById('emergencyBannerTitle').textContent = `${first.unit_name} Active`;
      document.getElementById('emergencyBannerDesc').textContent = `En route to ${first.hospital_zone_id.replace('zone_', '')} | Progress: ${first.progress_pct}%`;
    } else {
      banner.style.display = 'none';
    }
  }

  const currentIds = new Set(ambulances.map(a => a.id));
  Object.keys(ambulanceMarkers).forEach(id => {
    if (!currentIds.has(id)) {
      map.removeLayer(ambulanceMarkers[id]);
      delete ambulanceMarkers[id];
    }
  });

  ambulances.forEach(amb => {
    if (amb.status === 'ARRIVED') return;

    if (ambulanceMarkers[amb.id]) {
      ambulanceMarkers[amb.id].setLatLng([amb.current_lat, amb.current_lng]);
    } else {
      const ambIcon = L.divIcon({
        className: 'custom-amb-icon',
        html: `<div style="background:#ff4b2b; color:white; width:30px; height:30px; border-radius:50%; display:flex; align-items:center; justify-content:center; box-shadow:0 0 16px #ff4b2b; font-size:14px;"><i class="fa-solid fa-truck-medical"></i></div>`,
        iconSize: [30, 30],
        iconAnchor: [15, 15]
      });

      const m = L.marker([amb.current_lat, amb.current_lng], { icon: ambIcon }).addTo(map);
      m.bindPopup(`<strong>${amb.unit_name}</strong><br/>Status: ${amb.status}`);
      ambulanceMarkers[amb.id] = m;
    }
  });
}

// ==========================================================
// 10. MODULE ACTIONS & REST CALLS
// ==========================================================
async function calculateRoute() {
  const origin = document.getElementById('routeOrigin').value;
  const dest = document.getElementById('routeDest').value;
  const emergency = document.getElementById('routeEmergencyMode').checked;

  if (!origin || !dest) return;
  if (origin === dest) {
    alert('Please pick two different origin and destination zones.');
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/api/route?from=${origin}&to=${dest}&emergency=${emergency}`);
    if (!res.ok) throw new Error('No path found');
    const data = await res.json();

    document.getElementById('routeResultsCard').style.display = 'block';
    document.getElementById('resDistance').textContent = data.total_distance_km;
    document.getElementById('resDuration').textContent = data.estimated_time_min;
    document.getElementById('resDelaySaved').textContent = Math.round(data.congestion_penalty_sec);

    const stepsDiv = document.getElementById('routeTurnByTurn');
    stepsDiv.innerHTML = '';
    (data.steps || []).forEach((s, idx) => {
      const el = document.createElement('div');
      el.className = 'turn-step-item';
      el.innerHTML = `
        <span style="font-weight:700; color:var(--accent-cyan);">${idx + 1}.</span>
        <span>${s.from_name} ➔ ${s.to_name}</span>
        <span style="margin-left:auto; font-family:var(--font-mono);">${s.distance_km}km (${Math.round(s.estimated_time_sec)}s)</span>
      `;
      stepsDiv.appendChild(el);
    });

    if (activeRoutePolyline) map.removeLayer(activeRoutePolyline);
    const latlngs = data.path.map(zid => {
      const zone = allZonesData.find(z => z.id === zid);
      return zone ? [zone.lat, zone.lng] : null;
    }).filter(Boolean);

    activeRoutePolyline = L.polyline(latlngs, {
      color: emergency ? '#ff4b2b' : '#00f2fe',
      weight: 6,
      opacity: 0.9,
      dashArray: emergency ? '8, 8' : null
    }).addTo(map);

    map.fitBounds(activeRoutePolyline.getBounds(), { padding: [40, 40] });
    logEvent(`Optimal route calculated: ${data.total_distance_km}km | ${data.estimated_time_min} mins`, 'info');
  } catch (err) {
    alert('Could not find viable route between zones.');
  }
}

async function dispatchAmbulance() {
  const pickup = document.getElementById('ambPickupZone').value;
  const hosp = document.getElementById('ambHospZone').value;

  try {
    const res = await fetch(`${API_BASE}/api/ambulance/dispatch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pickup_zone_id: pickup, hospital_zone_id: hosp })
    });
    const mission = await res.json();
    logEvent(`🚨 EMERGENCY DISPATCH: ${mission.unit_name} en route to ${mission.pickup_zone_id}`, 'alert');
    switchTab('ambulance');
  } catch (e) {
    alert('Failed to dispatch ambulance.');
  }
}

function renderAmbulanceList(ambulances) {
  const list = document.getElementById('activeAmbulancesList');
  if (!list) return;

  if (ambulances.length === 0) {
    list.innerHTML = '<div class="empty-state">No active emergency missions</div>';
    return;
  }

  list.innerHTML = '';
  ambulances.forEach(a => {
    const el = document.createElement('div');
    el.className = 'list-card';
    el.innerHTML = `
      <div class="card-top">
        <span class="card-title">${a.unit_name}</span>
        <span class="badge ${a.status === 'ARRIVED' ? 'badge-green' : 'badge-red'}">${a.status}</span>
      </div>
      <div class="card-desc">Destination: ${a.hospital_zone_id} | Elapsed: ${a.elapsed_sec}s</div>
      <div style="background:rgba(255,255,255,0.06); height:6px; border-radius:3px; overflow:hidden; margin:4px 0;">
        <div style="background:var(--accent-red); width:${a.progress_pct}%; height:100%;"></div>
      </div>
    `;
    list.appendChild(el);
  });
}

async function fetchSignals() {
  try {
    const res = await fetch(`${API_BASE}/api/signals`);
    if (!res.ok) return;
    const signals = await res.json();
    renderSignalsList(signals);
  } catch (e) { }
}

function renderSignalsList(signals) {
  const list = document.getElementById('signalsList');
  if (!list) return;

  list.innerHTML = '';
  signals.forEach(s => {
    const el = document.createElement('div');
    el.className = 'list-card';
    const phaseClass = s.current_phase === 'GREEN' ? 'badge-green' : (s.current_phase === 'YELLOW' ? 'badge-amber' : 'badge-red');
    el.innerHTML = `
      <div class="card-top">
        <span class="card-title">${s.intersection_name}</span>
        <span class="badge ${phaseClass}">${s.current_phase} (${s.timer_sec}s)</span>
      </div>
      <div class="card-desc">Queue Demand: <strong>${s.queue_length} vehicles</strong> | Green Split: ${s.green_split_sec}s</div>
    `;
    list.appendChild(el);
  });
}

async function optimizeSignals() {
  try {
    const res = await fetch(`${API_BASE}/api/signals/optimize`, { method: 'POST' });
    const data = await res.json();
    logEvent(`AI Signal Engine: Dynamically rebalanced ${data.adjustments?.length || 0} intersections`, 'success');
    fetchSignals();
  } catch (e) {
    alert('Failed to optimize signals.');
  }
}

async function triggerAccident() {
  const zoneId = document.getElementById('accidentZoneSelect').value;
  try {
    const res = await fetch(`${API_BASE}/api/accidents/trigger`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ zone_id: zoneId, severity: 'CRITICAL', vehicles_involved: 2 })
    });
    const acc = await res.json();
    logEvent(`💥 ACCIDENT REPORTED: ${acc.description} at ${acc.zone_name}`, 'alert');
    fetchAccidents();
  } catch (e) {
    alert('Failed to simulate accident.');
  }
}

async function fetchAccidents() {
  try {
    const res = await fetch(`${API_BASE}/api/accidents`);
    if (!res.ok) return;
    const data = await res.json();
    renderAccidentsList(data.active || []);
  } catch (e) { }
}

function renderAccidentsList(accidents) {
  const list = document.getElementById('accidentsList');
  if (!list) return;

  if (accidents.length === 0) {
    list.innerHTML = '<div class="empty-state">No active collision incidents</div>';
    return;
  }

  list.innerHTML = '';
  accidents.forEach(a => {
    const el = document.createElement('div');
    el.className = 'list-card';
    el.innerHTML = `
      <div class="card-top">
        <span class="card-title">${a.zone_name}</span>
        <span class="badge badge-red">${a.severity}</span>
      </div>
      <div class="card-desc">${a.description} | Vehicles Involved: ${a.vehicles_involved}</div>
      <button class="action-btn btn-green mt-2" onclick="resolveAccident('${a.id}')">Clear Incident</button>
    `;
    list.appendChild(el);
  });
}

async function resolveAccident(id) {
  try {
    await fetch(`${API_BASE}/api/accidents/${id}/resolve`, { method: 'POST' });
    logEvent(`Incident #${id} resolved. Road cleared.`, 'success');
    fetchAccidents();
  } catch (e) { }
}

async function fetchParking() {
  try {
    const res = await fetch(`${API_BASE}/api/parking`);
    if (!res.ok) return;
    const list = await res.json();
    const container = document.getElementById('parkingFacilitiesList');
    if (!container) return;
    container.innerHTML = '';
    list.forEach(p => {
      const el = document.createElement('div');
      el.className = 'list-card';
      el.innerHTML = `
        <div class="card-top">
          <span class="card-title">${p.name}</span>
          <span class="badge badge-green">${p.available_spots} Available</span>
        </div>
        <div class="card-desc">Capacity: ${p.total_spots} | Occupancy: ${p.occupancy_rate_pct}% | Rate: ₹${p.hourly_rate_inr}/h</div>
      `;
      container.appendChild(el);
    });
  } catch (e) { }
}

async function findParking() {
  const dest = document.getElementById('parkingDestSelect').value;
  try {
    const res = await fetch(`${API_BASE}/api/parking/recommend?destination=${dest}`);
    const data = await res.json();
    alert(`Nearest Parking: ${data.recommended_facility.name} (${data.recommended_facility.available_spots} spots available).`);
  } catch (e) {
    alert('No nearby parking found.');
  }
}

async function fetchTransit() {
  try {
    const res = await fetch(`${API_BASE}/api/transit`);
    if (!res.ok) return;
    const list = await res.json();
    const container = document.getElementById('transitLinesList');
    if (!container) return;
    container.innerHTML = '';
    list.forEach(t => {
      const el = document.createElement('div');
      el.className = 'list-card';
      el.innerHTML = `
        <div class="card-top">
          <span class="card-title">${t.line_name} (${t.vehicle_id})</span>
          <span class="badge badge-cyan">${t.status}</span>
        </div>
        <div class="card-desc">Delay: ${t.delay_minutes} min | Current Stop: ${t.current_zone_id}</div>
      `;
      container.appendChild(el);
    });
  } catch (e) { }
}

async function optimizeTransit() {
  try {
    const res = await fetch(`${API_BASE}/api/transit/optimize`, { method: 'POST' });
    const data = await res.json();
    logEvent(`Transit Optimizer: Evaluated ${data.recommendations?.length || 0} line detours`, 'info');
    fetchTransit();
  } catch (e) { }
}

async function fetchLogistics() {
  try {
    const res = await fetch(`${API_BASE}/api/logistics`);
    if (!res.ok) return;
    const list = await res.json();
    const container = document.getElementById('logisticsFleetList');
    if (!container) return;
    container.innerHTML = '';
    list.forEach(v => {
      const el = document.createElement('div');
      el.className = 'list-card';
      el.innerHTML = `
        <div class="card-top">
          <span class="card-title">${v.vehicle_id} (${v.vehicle_type})</span>
          <span class="badge badge-blue">Deliveries: ${v.completed_deliveries}/${v.assigned_stops.length}</span>
        </div>
        <div class="card-desc">Fuel Saved: ${v.fuel_saved_liters}L | CO2 Cut: ${v.co2_prevented_kg}kg</div>
      `;
      container.appendChild(el);
    });
  } catch (e) { }
}

async function optimizeLogistics() {
  try {
    const res = await fetch(`${API_BASE}/api/logistics/optimize`, { method: 'POST' });
    const data = await res.json();
    logEvent(`Freight VRP: Multi-stop delivery routes optimized across fleet`, 'success');
    fetchLogistics();
  } catch (e) { }
}

async function fetchWaste() {
  try {
    const res = await fetch(`${API_BASE}/api/waste`);
    if (!res.ok) return;
    const list = await res.json();
    const container = document.getElementById('wasteBinsList');
    if (!container) return;
    container.innerHTML = '';
    list.forEach(b => {
      const el = document.createElement('div');
      el.className = 'list-card';
      el.innerHTML = `
        <div class="card-top">
          <span class="card-title">Bin #${b.bin_id} (${b.location_name})</span>
          <span class="badge ${b.is_overflowing ? 'badge-red' : (b.fill_level_pct > 75 ? 'badge-amber' : 'badge-green')}">${b.fill_level_pct}%</span>
        </div>
        <div class="card-desc">Type: ${b.waste_type} | IoT Sensor Status: Active</div>
      `;
      container.appendChild(el);
    });
  } catch (e) { }
}

async function planWasteCollection() {
  try {
    const res = await fetch(`${API_BASE}/api/waste/collect`, { method: 'POST' });
    const plan = await res.json();
    logEvent(`Waste Tour Planned: Scheduled pickup for ${plan.bins_to_collect?.length || 0} full containers`, 'info');
    fetchWaste();
  } catch (e) { }
}

async function fetchFlood() {
  try {
    const res = await fetch(`${API_BASE}/api/flood`);
    if (!res.ok) return;
    const sensors = await res.json();
    const container = document.getElementById('floodSensorsList');
    if (!container) return;
    container.innerHTML = '';
    sensors.forEach(s => {
      const el = document.createElement('div');
      el.className = 'list-card';
      el.innerHTML = `
        <div class="card-top">
          <span class="card-title">${s.zone_name}</span>
          <span class="badge ${s.alert_level === 'CRITICAL' ? 'badge-red' : 'badge-blue'}">${s.water_depth_cm} cm</span>
        </div>
        <div class="card-desc">Catchment Runoff: ${s.flow_rate_lps} L/s | Inundation Risk: ${s.alert_level}</div>
      `;
      container.appendChild(el);
    });
  } catch (e) { }
}

async function applyRainSim() {
  const val = document.getElementById('rainSlider').value;
  try {
    await fetch(`${API_BASE}/api/flood/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rainfall_rate_mmh: parseFloat(val) })
    });
    logEvent(`Urban Storm Simulation: Precipitation set to ${val} mm/h`, 'alert');
    fetchFlood();
  } catch (e) { }
}

async function clearFlood() {
  try {
    await fetch(`${API_BASE}/api/flood/clear`, { method: 'POST' });
    document.getElementById('rainSlider').value = 0;
    document.getElementById('rainValText').textContent = '0 mm/h';
    logEvent('Urban drainage active. Water levels returned to normal.', 'success');
    fetchFlood();
  } catch (e) { }
}

async function fetchIncidents() {
  try {
    const res = await fetch(`${API_BASE}/api/incidents`);
    if (!res.ok) return;
    const data = await res.json();
    renderIncidentsList(data.open || []);
  } catch (e) { }
}

function renderIncidentsList(incidents) {
  const list = document.getElementById('incidentsQueueList');
  if (!list) return;

  if (incidents.length === 0) {
    list.innerHTML = '<div class="empty-state">No open multi-agency incidents</div>';
    return;
  }

  list.innerHTML = '';
  incidents.forEach(inc => {
    const el = document.createElement('div');
    el.className = 'list-card';
    el.innerHTML = `
      <div class="card-top">
        <span class="card-title">${inc.title}</span>
        <span class="badge ${inc.severity_level === 'CRITICAL' ? 'badge-red' : 'badge-amber'}">P${inc.triage_priority}</span>
      </div>
      <div class="card-desc">${inc.description}</div>
      <div style="font-size:10px; color:var(--text-muted); margin-top:4px;">Auto Actions: ${(inc.auto_actions || []).join(' • ')}</div>
    `;
    list.appendChild(el);
  });
}

// ==========================================================
// 11. UI HELPERS & TAB SWITCHING
// ==========================================================
function initTabNavigation() {
  const tabs = document.querySelectorAll('.nav-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.getAttribute('data-tab');
      switchTab(target);
    });
  });
}

function switchTab(tabKey) {
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.module-panel').forEach(p => p.classList.remove('active'));

  const activeTab = document.querySelector(`.nav-tab[data-tab="${tabKey}"]`);
  const activePanel = document.getElementById(`panel-${tabKey}`);

  if (activeTab) activeTab.classList.add('active');
  if (activePanel) activePanel.classList.add('active');
}

function initEventListeners() {
  // City Selector
  document.getElementById('citySelector')?.addEventListener('change', async (e) => {
    const targetCity = e.target.value;
    try {
      await fetch(`${API_BASE}/api/city/switch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ city_id: targetCity })
      });
      await loadInitialCityData(targetCity);
    } catch (err) {
      console.error('Error switching city:', err);
    }
  });

  // Smart Traffic Controls
  document.getElementById('btnShowTraffic')?.addEventListener('click', showTraffic);
  document.getElementById('btnViewFastRouteDirect')?.addEventListener('click', () => {
    const from = document.getElementById('smartFromZone').value;
    const to = document.getElementById('smartToZone').value;
    window.location.href = `fast-route.html?from=${from}&to=${to}&city=${currentCityId}`;
  });

  // Saved Places Modal
  document.getElementById('btnManageSavedPlacesModal')?.addEventListener('click', openSavedPlacesModal);
  document.getElementById('btnCloseSavedModal')?.addEventListener('click', () => {
    document.getElementById('savedPlacesModal').style.display = 'none';
  });
  document.getElementById('btnAddCustomPlaceBtn')?.addEventListener('click', addCustomPlace);
  document.getElementById('btnSavePlacesSubmit')?.addEventListener('click', savePlacesSubmit);

  // RoadGuard Modal & Presets
  document.getElementById('btnOpenReportModal')?.addEventListener('click', () => {
    document.getElementById('citizenReportModal').style.display = 'flex';
  });
  document.getElementById('btnCloseReportModal')?.addEventListener('click', () => {
    document.getElementById('citizenReportModal').style.display = 'none';
  });

  document.querySelectorAll('.btn-preset-photo').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.btn-preset-photo').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      selectedDamagePhotoUrl = btn.getAttribute('data-img');
      selectedDamageType = btn.getAttribute('data-type');
      document.getElementById('reportPhotoImg').src = selectedDamagePhotoUrl;
      document.getElementById('reportTitle').value = `${selectedDamageType} on Commute Corridor`;
    });
  });

  document.getElementById('btnAutoLocation')?.addEventListener('click', () => {
    const notice = document.getElementById('autoGpsNotice');
    if ('geolocation' in navigator) {
      notice.style.display = 'block';
      notice.textContent = 'Acquiring GPS fix...';
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          notice.textContent = `📍 GPS Fixed: ${pos.coords.latitude.toFixed(4)}, ${pos.coords.longitude.toFixed(4)}`;
        },
        () => {
          notice.textContent = `📍 Localized to nearest junction: ${allZonesData[0]?.name || 'City Center'}`;
        }
      );
    } else {
      notice.style.display = 'block';
      notice.textContent = `📍 Location locked: ${allZonesData[0]?.name || 'Hi-Tech Junction'}`;
    }
  });

  document.getElementById('btnSubmitCitizenReport')?.addEventListener('click', submitCitizenReport);
  document.getElementById('btnDismissRoadAlert')?.addEventListener('click', () => {
    document.getElementById('emergencyRoadAlert').style.display = 'none';
  });

  // Standard Module Handlers
  document.getElementById('btnCalculateRoute')?.addEventListener('click', calculateRoute);
  document.getElementById('btnDispatchAmbulance')?.addEventListener('click', dispatchAmbulance);
  document.getElementById('btnOptimizeSignals')?.addEventListener('click', optimizeSignals);
  document.getElementById('btnTriggerAccident')?.addEventListener('click', triggerAccident);
  document.getElementById('btnFindParking')?.addEventListener('click', findParking);
  document.getElementById('btnOptimizeTransit')?.addEventListener('click', optimizeTransit);
  document.getElementById('btnOptimizeLogistics')?.addEventListener('click', optimizeLogistics);
  document.getElementById('btnPlanWasteCollection')?.addEventListener('click', planWasteCollection);
  document.getElementById('btnApplyRain')?.addEventListener('click', applyRainSim);
  document.getElementById('btnClearFlood')?.addEventListener('click', clearFlood);

  // Quick Action Header Buttons
  document.getElementById('btnQuickOptimize')?.addEventListener('click', () => {
    optimizeSignals();
    optimizeTransit();
    optimizeLogistics();
  });

  document.getElementById('rainSlider')?.addEventListener('input', (e) => {
    document.getElementById('rainValText').textContent = `${e.target.value} mm/h`;
  });

  const simSlider = document.getElementById('simIntensitySlider');
  simSlider?.addEventListener('input', (e) => {
    const val = e.target.value;
    document.getElementById('sliderIntensityVal').textContent = `${val}x`;
    fetch(`${API_BASE}/api/simulation/traffic?multiplier=${val}`, { method: 'POST' });
  });
}

function logEvent(text, type = 'info') {
  const container = document.getElementById('eventLogContainer');
  if (!container) return;

  const now = new Date();
  const timeStr = now.toTimeString().split(' ')[0];

  const entry = document.createElement('div');
  entry.className = `ticker-entry ${type}`;
  entry.innerHTML = `<span class="time">[${timeStr}]</span> ${text}`;

  container.prepend(entry);
  if (container.children.length > 50) {
    container.removeChild(container.lastChild);
  }
}
