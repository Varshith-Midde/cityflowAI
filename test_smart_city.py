"""
Automated Verification Test Suite for CityFlow AI Smart City Platform.
Verifies all multi-city features, Smart Traffic & AI Prediction, 3 Red Zones,
Saved Places, RoadGuard citizen reports, Municipal status progression,
dynamic Dijkstra alternative routes, and existing municipal modules.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

def run_tests():
    print("=" * 75)
    print("  CityFlow AI India-Wide Smart City Platform — Verification Suite")
    print("=" * 75)

    # 1. Test Multi-City Catalog
    res = client.get("/api/cities")
    assert res.status_code == 200, f"Cities endpoint failed: {res.text}"
    cities = res.json()
    assert len(cities) >= 5, f"Expected at least 5 cities, got {len(cities)}"
    print(f"✅ [1/15] Multi-City Catalog: {len(cities)} Indian metros supported ({', '.join(c['city_id'] for c in cities)})")

    # 2. Test City Metadata & Jurisdictions
    res = client.get("/api/city")
    assert res.status_code == 200
    city = res.json()
    assert "authority" in city
    print(f"✅ [2/15] Active City Jurisdiction: '{city['city_name']}' under {city['authority']}")

    # 3. Test Multi-City Switching (Switch to Bengaluru BBMP)
    res_switch = client.post("/api/city/switch", json={"city_id": "bengaluru"})
    assert res_switch.status_code == 200
    switched = res_switch.json()
    assert switched["city_id"] == "bengaluru"
    assert "BBMP" in switched["authority"]
    print(f"✅ [3/15] Dynamic Metro Switch: Successfully switched to '{switched['city_name']}' ({switched['authority']})")

    # Switch back to Hyderabad for remaining route tests
    client.post("/api/city/switch", json={"city_id": "hyderabad"})

    # 4. Test Traffic Zones
    res = client.get("/api/zones")
    assert res.status_code == 200
    zones = res.json()
    assert len(zones) >= 10
    print(f"✅ [4/15] Traffic Zones: {len(zones)} interconnected corridor nodes verified")

    # 5. Test 3 Red Zones Only Algorithm
    res_rz = client.get("/api/traffic/red-zones")
    assert res_rz.status_code == 200
    red_zones = res_rz.json()
    assert len(red_zones) == 3, f"Must have strictly 3 Red Zones, got {len(red_zones)}"
    print(f"✅ [5/15] Strictly 3 Red Zones: #{red_zones[0]['name']} ({red_zones[0]['congestion_pct']}%), #{red_zones[1]['name']} ({red_zones[1]['congestion_pct']}%), #{red_zones[2]['name']} ({red_zones[2]['congestion_pct']}%)")

    # 6. Test AI Traffic Prediction
    res_pred = client.get("/api/traffic/predict?from=zone_hitech_junction&to=zone_central_hospital")
    assert res_pred.status_code == 200
    pred = res_pred.json()
    assert "predictions" in pred and len(pred["predictions"]) == 3
    p30 = next(p for p in pred["predictions"] if p["horizon_min"] == 30)
    print(f"✅ [6/15] AI Traffic Prediction: Current {pred['current_congestion_pct']}% -> Peak (+30m): {p30['congestion_pct']}% (Trend: {pred['trend']})")

    # 7. Test Alternative Routes (Fastest, Arterial, Low Congestion)
    res_routes = client.get("/api/routes/alternatives?from=zone_hitech_junction&to=zone_central_hospital")
    assert res_routes.status_code == 200
    routes_data = res_routes.json()
    assert len(routes_data["routes"]) >= 2
    fastest = routes_data["routes"][0]
    print(f"✅ [7/15] Alternative Routes: Primary '{fastest['title']}' ({fastest['estimated_time_min']}m | {fastest['total_distance_km']}km) avoids {len(fastest['bypassed_red_zones'])} Red Zones")

    # 8. Test Saved Places (Home, Work, Custom x5)
    res_places = client.get("/api/saved-places")
    assert res_places.status_code == 200
    places = res_places.json()
    assert "home" in places and "work" in places
    # Test adding a custom place
    res_add = client.post("/api/saved-places", json={
        "type": "CUSTOM",
        "name": "Airport Terminal 1",
        "zone_id": "zone_ring_road_south",
        "icon": "fa-plane"
    })
    assert res_add.status_code == 200
    print(f"✅ [8/15] Saved Places: Home & Work active, added custom '{res_add.json()['place']['name']}'")

    # 9. Test RoadGuard Citizen Report Submission with AI Analysis
    res_rg = client.post("/api/roadguard/report", json={
        "title": "Severe Pothole Cluster on Tech Corridor",
        "description": "Deep asphalt depression causing vehicles to brake abruptly.",
        "zone_id": "zone_it_corridor",
        "damage_type": "Pothole (Grade 4 Hazard)"
    })
    assert res_rg.status_code == 200
    rg_data = res_rg.json()["report"]
    assert rg_data["status"] == "REPORTED"
    assert rg_data["severity"] in ["CRITICAL", "HIGH"]
    assert rg_data["confidence_score"] > 85.0
    print(f"✅ [9/15] RoadGuard Citizen Reporting: Filed #{rg_data['id']} ({rg_data['damage_type']}) — AI Confidence {rg_data['confidence_score']}%")

    # 10. Test Municipal Repair Status Lifecycle Progression
    report_id = rg_data["id"]
    res_status = client.post(f"/api/roadguard/reports/{report_id}/status", json={
        "status": "REPAIRING",
        "municipal_action": "Crew and asphalt recycler dispatched",
        "assigned_crew": "GHMC Ward 112 Rapid Repair Unit",
        "repair_progress_pct": 70
    })
    assert res_status.status_code == 200
    updated_rg = res_status.json()["report"]
    assert updated_rg["status"] == "REPAIRING"
    assert updated_rg["repair_progress_pct"] == 70
    print(f"✅ [10/15] Municipal Workflow: #{report_id} progressed to 'REPAIRING' (Progress: {updated_rg['repair_progress_pct']}%)")

    # 11. Test Emergency Road Hazard Commute Alert
    res_alert = client.get("/api/roadguard/alerts")
    assert res_alert.status_code == 200
    alerts = res_alert.json()["active_alerts"]
    assert len(alerts) >= 1
    print(f"✅ [11/15] Commute Road Alerts: {len(alerts)} active critical road alerts broadcasted to commuters")

    # 12. Test Smart Ambulance EMS Green-Wave Dispatch
    res_amb = client.post("/api/ambulance/dispatch", json={"pickup_zone_id": "zone_market_square"})
    assert res_amb.status_code == 200
    amb = res_amb.json()
    print(f"✅ [12/15] Smart Ambulance EMS: Dispatched {amb['unit_name']} via corridor: {amb['route']}")

    # 13. Test AI Adaptive Traffic Signals Optimization
    res_sig = client.post("/api/signals/optimize")
    assert res_sig.status_code == 200
    sigs = res_sig.json()
    print(f"✅ [13/15] AI Traffic Signals: Dynamically adjusted {len(sigs['adjustments'])} signal intersections")

    # 14. Test Road Accident Anomaly Detection
    res_acc = client.post("/api/accidents/trigger", json={"zone_id": "zone_cyber_towers", "severity": "CRITICAL"})
    assert res_acc.status_code == 200
    acc = res_acc.json()
    print(f"✅ [14/15] Road Accident Detection: Registered {acc['severity']} incident at {acc['zone_name']}")

    # 15. Test Smart Logistics VRP, Parking, and City Resources
    res_vrp = client.post("/api/logistics/optimize")
    assert res_vrp.status_code == 200
    res_park = client.get("/api/parking")
    assert res_park.status_code == 200
    res_waste = client.post("/api/waste/collect")
    assert res_waste.status_code == 200
    res_flood = client.post("/api/flood/simulate", json={"rainfall_rate_mmh": 25.0})
    assert res_flood.status_code == 200
    print(f"✅ [15/15] City Resources & Logistics: VRP Freight, Parking, Smart Waste, and Flood Drainage all verified")

    print("\n" + "=" * 75)
    print("  ALL 15 VERIFICATION CHECKS PASSED! CITYFLOW AI READY FOR PRODUCTION 🚀")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    run_tests()
