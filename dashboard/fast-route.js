/**
 * CityFlow AI — Fast Route & Congestion Bypass Navigator JS
 * High-performance multi-route comparison, 3 Red Zones avoidance, and interactive Leaflet map.
 */

let map;
let zoneData = [];
let routeLayers = {};
let redZoneMarkers = [];
let currentRoutes = [];
let selectedRouteId = 'route_fastest';
let savedPlaces = {};

const API_BASE = window.location.origin.includes('localhost') || window.location.origin.includes('127.0.0.1')
  ? window.location.origin
  : 'http://127.0.0.1:8000';

document.addEventListener('DOMContentLoaded', async () => {
  initMap();
  initEventListeners();
  await loadSavedPlaces();
  await loadCities();
  await loadCurrentCity();
});

function initMap() {
  map = L.map('fastRouteMap', {
    zoomControl: false,
    attributionControl: false
  }).setView([17.4435, 78.3772], 14);

  L.control.zoom({ position: 'bottomright' }).addTo(map);

  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    maxZoom: 19,
    subdomains: 'abcd'
  }).addTo(map);
}

function initEventListeners() {
  document.getElementById('btnRunFastRoute').addEventListener('click', runRouteCalculation);

  document.getElementById('citySelectFast').addEventListener('change', async (e) => {
    const cityId = e.target.value;
    try {
      await fetch(`${API_BASE}/api/city/switch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ city_id: cityId })
      });
      await loadCurrentCity(cityId);
    } catch (err) {
      console.error('Error switching city:', err);
    }
  });
}

async function loadCities() {
  try {
    const res = await fetch(`${API_BASE}/api/cities`);
    if (!res.ok) return;
    const cities = await res.json();
    const select = document.getElementById('citySelectFast');
    if (cities.length > 0) {
      select.innerHTML = '';
      cities.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c.city_id;
        opt.textContent = `🇮🇳 ${c.city_name} (${c.authority.split('(')[1]?.replace(')', '') || c.authority})`;
        select.appendChild(opt);
      });
    }
  } catch (err) {
    console.error('Failed to load cities catalog:', err);
  }
}

async function loadSavedPlaces() {
  try {
    const res = await fetch(`${API_BASE}/api/saved-places`);
    if (!res.ok) return;
    savedPlaces = await res.json();
    renderSavedPlacesChips();
  } catch (err) {
    console.error('Failed to load saved places:', err);
  }
}

function renderSavedPlacesChips() {
  const container = document.getElementById('savedPlacesChipsContainer');
  if (!container) return;
  container.innerHTML = '';

  if (savedPlaces.home) {
    const chip = createChip(savedPlaces.home.name, 'fa-house', () => {
      applySavedPlace(savedPlaces.home.zone_id);
    });
    container.appendChild(chip);
  }

  if (savedPlaces.work) {
    const chip = createChip(savedPlaces.work.name, 'fa-briefcase', () => {
      applySavedPlace(savedPlaces.work.zone_id);
    });
    container.appendChild(chip);
  }

  (savedPlaces.custom || []).forEach(p => {
    const chip = createChip(p.name, p.icon || 'fa-location-dot', () => {
      applySavedPlace(p.zone_id);
    });
    container.appendChild(chip);
  });
}

function createChip(name, icon, onClick) {
  const btn = document.createElement('button');
  btn.className = 'place-chip';
  btn.innerHTML = `<i class="fa-solid ${icon}"></i> ${name}`;
  btn.addEventListener('click', onClick);
  return btn;
}

function applySavedPlace(zoneId) {
  const fromSelect = document.getElementById('fastFromZone');
  const toSelect = document.getElementById('fastToZone');

  // If From is empty or already matches, set To; else set From
  if (!fromSelect.value || fromSelect.value === zoneId) {
    fromSelect.value = zoneId;
  } else {
    toSelect.value = zoneId;
  }
}

async function loadCurrentCity(specificCityId = null) {
  try {
    const url = specificCityId ? `${API_BASE}/api/city?city_id=${specificCityId}` : `${API_BASE}/api/city`;
    const res = await fetch(url);
    if (!res.ok) return;
    const city = await res.json();

    document.getElementById('activeAuthorityBadge').textContent = city.authority || city.city_name;
    document.getElementById('citySelectFast').value = city.city_id;

    if (city.center) {
      map.setView([city.center.lat, city.center.lng], city.center.zoom || 13);
    }

    // Load zones
    const zRes = await fetch(`${API_BASE}/api/zones`);
    zoneData = await zRes.json();
    populateZoneSelects(zoneData);

    // Load 3 Red Zones
    await loadTopRedZones();

    // Check URL params for pre-fills
    const params = new URLSearchParams(window.location.search);
    const fromParam = params.get('from');
    const toParam = params.get('to');

    if (fromParam && document.getElementById('fastFromZone').querySelector(`option[value="${fromParam}"]`)) {
      document.getElementById('fastFromZone').value = fromParam;
    }
    if (toParam && document.getElementById('fastToZone').querySelector(`option[value="${toParam}"]`)) {
      document.getElementById('fastToZone').value = toParam;
    }

    // Auto-run if From and To exist
    runRouteCalculation();
  } catch (err) {
    console.error('Error loading city data:', err);
  }
}

function populateZoneSelects(zones) {
  const fromSelect = document.getElementById('fastFromZone');
  const toSelect = document.getElementById('fastToZone');

  fromSelect.innerHTML = '';
  toSelect.innerHTML = '';

  zones.forEach(z => {
    const opt = document.createElement('option');
    opt.value = z.id;
    opt.textContent = `${z.name} (${z.density_label})`;
    fromSelect.appendChild(opt.cloneNode(true));
    toSelect.appendChild(opt);
  });

  if (fromSelect.options.length > 0) fromSelect.selectedIndex = 0;
  if (toSelect.options.length > 3) toSelect.selectedIndex = 3;
}

async function loadTopRedZones() {
  try {
    const res = await fetch(`${API_BASE}/api/traffic/red-zones`);
    if (!res.ok) return;
    const redZones = await res.json();

    // Clear previous red zone markers
    redZoneMarkers.forEach(m => map.removeLayer(m));
    redZoneMarkers = [];

    const listEl = document.getElementById('fastRedZonesList');
    listEl.innerHTML = '';

    redZones.forEach((rz, idx) => {
      // 1. Plot Pulsing Red Zone Radar on Map
      const circle = L.circle([rz.lat, rz.lng], {
        radius: 450,
        color: '#ff4b2b',
        weight: 2,
        fillColor: '#ff4b2b',
        fillOpacity: 0.28,
        dashArray: '5, 5'
      }).addTo(map);

      const marker = L.circleMarker([rz.lat, rz.lng], {
        radius: 12,
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

      redZoneMarkers.push(circle, marker);

      // 2. Add to Sidebar List
      const item = document.createElement('div');
      item.className = 'redzone-item';
      item.innerHTML = `
        <div>
          <strong>${idx + 1}. ${rz.name}</strong>
          <div style="color:rgba(255,255,255,0.6); font-size:10px;">${rz.bypass_advice}</div>
        </div>
        <span class="badge badge-red" style="font-weight:700;">${rz.congestion_pct}% LOAD</span>
      `;
      listEl.appendChild(item);
    });
  } catch (err) {
    console.error('Failed to load top red zones:', err);
  }
}

async function runRouteCalculation() {
  const fromZone = document.getElementById('fastFromZone').value;
  const toZone = document.getElementById('fastToZone').value;

  if (!fromZone || !toZone) return;
  if (fromZone === toZone) {
    alert('Please select two different zones for route optimization.');
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/api/routes/alternatives?from=${fromZone}&to=${toZone}`);
    if (!res.ok) throw new Error('Failed to compute routes');
    const data = await res.json();

    currentRoutes = data.routes || [];
    renderRouteCards(currentRoutes);
    selectRoute(currentRoutes[0]?.route_id || 'route_fastest');
  } catch (err) {
    console.error('Routing calculation error:', err);
  }
}

function renderRouteCards(routes) {
  const container = document.getElementById('routeOptionsContainer');
  container.innerHTML = '';

  routes.forEach((r, idx) => {
    const card = document.createElement('div');
    card.className = `route-card-option ${r.route_id === selectedRouteId ? 'selected' : ''}`;
    card.id = `card-${r.route_id}`;

    const isFastest = idx === 0;

    card.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
        <span style="font-weight:700; font-size:13px; color:${r.color};">
          <i class="fa-solid fa-route"></i> ${r.title}
        </span>
        <span class="badge ${isFastest ? 'badge-cyan' : 'badge-amber'}" style="font-size:10px;">${r.badge}</span>
      </div>
      <div style="display:flex; justify-content:space-between; align-items:baseline;">
        <span style="font-size:18px; font-weight:800; color:#fff;">${r.estimated_time_min} mins</span>
        <span style="font-size:12px; color:var(--text-muted); font-family:var(--font-mono);">${r.total_distance_km} km</span>
      </div>
      <div style="font-size:11px; color:var(--accent-green); margin-top:4px;">
        <i class="fa-solid fa-circle-check"></i> ${isFastest ? `Saves ${r.delay_saved_sec}s vs gridlock` : 'Alternative bypass corridor'}
      </div>
    `;

    card.addEventListener('click', () => {
      selectRoute(r.route_id);
    });

    container.appendChild(card);
  });
}

function selectRoute(routeId) {
  selectedRouteId = routeId;
  const route = currentRoutes.find(r => r.route_id === routeId) || currentRoutes[0];
  if (!route) return;

  // Highlight Card
  document.querySelectorAll('.route-card-option').forEach(c => c.classList.remove('selected'));
  document.getElementById(`card-${routeId}`)?.classList.add('selected');

  // Update Stat Tiles
  document.getElementById('statDuration').textContent = route.estimated_time_min;
  document.getElementById('statDistance').textContent = route.total_distance_km;
  document.getElementById('statSaved').textContent = route.delay_saved_sec;

  // Render Turn Steps
  const turnContainer = document.getElementById('fastTurnSteps');
  turnContainer.innerHTML = '';
  (route.steps || []).forEach((step, idx) => {
    const el = document.createElement('div');
    el.className = 'turn-step-item';
    el.innerHTML = `
      <span style="font-weight:700; color:${route.color};">${idx + 1}.</span>
      <span>${step.from_name} ➔ ${step.to_name}</span>
      <span style="margin-left:auto; font-family:var(--font-mono); font-size:11px; color:#888;">
        ${step.distance_km}km (${Math.round(step.estimated_time_sec)}s)
      </span>
    `;
    turnContainer.appendChild(el);
  });

  // Render Polylines on Leaflet Map
  Object.values(routeLayers).forEach(l => map.removeLayer(l));
  routeLayers = {};

  // Draw other routes with subtle opacity
  currentRoutes.forEach(r => {
    if (r.route_id !== routeId) {
      const coords = r.path.map(zid => {
        const z = zoneData.find(zone => zone.id === zid);
        return z ? [z.lat, z.lng] : null;
      }).filter(Boolean);

      if (coords.length > 1) {
        const poly = L.polyline(coords, {
          color: r.color,
          weight: 4,
          opacity: 0.35,
          dashArray: '6, 6'
        }).addTo(map);
        routeLayers[r.route_id] = poly;
      }
    }
  });

  // Draw active selected route brightly
  const activeCoords = route.path.map(zid => {
    const z = zoneData.find(zone => zone.id === zid);
    return z ? [z.lat, z.lng] : null;
  }).filter(Boolean);

  if (activeCoords.length > 1) {
    const activePoly = L.polyline(activeCoords, {
      color: route.color,
      weight: 7,
      opacity: 0.95
    }).addTo(map);
    routeLayers[route.route_id] = activePoly;
    map.fitBounds(activePoly.getBounds(), { padding: [50, 50] });
  }
}
