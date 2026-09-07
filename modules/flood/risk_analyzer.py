"""
AI-Based Urban Flood Warning & Water-Level Monitoring System.
Combines simulated meteorological precipitation, drainage permeability ratings,
and ultrasound water-level sensors to predict flood inundation risks along road corridors.
"""
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from city_model import CityRoadGraph, TrafficZone


@dataclass
class FloodSensorNode:
    id: str
    zone_id: str
    zone_name: str
    current_water_level_cm: float = 0.0
    critical_threshold_cm: float = 25.0
    drainage_permeability: float = 0.50  # 0.0 (poor) to 1.0 (excellent)
    rainfall_rate_mmh: float = 0.0
    risk_level: str = "LOW"  # LOW, MODERATE, HIGH, CRITICAL

    def update_simulation(self, rain_rate: float) -> None:
        """Simulates water accumulation based on rainfall and drainage capacity."""
        self.rainfall_rate_mmh = rain_rate
        # Inflow vs drainage outflow
        net_inflow_cm = (rain_rate * 0.15) - (self.drainage_permeability * 2.5)
        self.current_water_level_cm = max(0.0, min(120.0, self.current_water_level_cm + net_inflow_cm))

        if self.current_water_level_cm >= self.critical_threshold_cm:
            self.risk_level = "CRITICAL"
        elif self.current_water_level_cm >= (self.critical_threshold_cm * 0.6):
            self.risk_level = "HIGH"
        elif self.current_water_level_cm >= (self.critical_threshold_cm * 0.3):
            self.risk_level = "MODERATE"
        else:
            self.risk_level = "LOW"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "water_level_cm": round(self.current_water_level_cm, 1),
            "critical_threshold_cm": self.critical_threshold_cm,
            "rainfall_rate_mmh": round(self.rainfall_rate_mmh, 1),
            "risk_level": self.risk_level,
            "is_road_impassable": self.current_water_level_cm >= self.critical_threshold_cm
        }


class UrbanFloodWarningSystem:
    """Manages city water sensors, automated flood detection, and hazard warnings."""

    def __init__(self, city_graph: CityRoadGraph):
        self.graph = city_graph
        self.sensors: Dict[str, FloodSensorNode] = {}
        self.active_rainfall_event: bool = False
        self.current_citywide_rain_mmh: float = 0.0
        self._init_sensors()

    def _init_sensors(self) -> None:
        """Sets up telemetry sensor nodes at known low-elevation catchment corridors."""
        flood_prone_zones = [
            ("SENS-FL-01", "zone_lake_road", 0.35, 20.0),
            ("SENS-FL-02", "zone_residential_avenue", 0.65, 30.0),
            ("SENS-FL-03", "zone_it_corridor", 0.50, 25.0)
        ]
        for sid, zid, drainage, thresh in flood_prone_zones:
            zone = self.graph.get_zone(zid)
            zname = zone.name if zone else zid
            self.sensors[sid] = FloodSensorNode(
                id=sid,
                zone_id=zid,
                zone_name=zname,
                drainage_permeability=drainage,
                critical_threshold_cm=thresh
            )

    def set_rainfall(self, rain_rate_mmh: float) -> List[Dict[str, Any]]:
        """Simulates storm precipitation intensity across the urban zone network."""
        self.current_citywide_rain_mmh = rain_rate_mmh
        self.active_rainfall_event = rain_rate_mmh > 5.0
        alerts = []

        for sensor in self.sensors.values():
            sensor.update_simulation(rain_rate_mmh)
            zone = self.graph.get_zone(sensor.zone_id)
            if zone:
                zone.flood_level_cm = sensor.current_water_level_cm
                zone.flood_risk = sensor.risk_level

                if sensor.risk_level == "CRITICAL":
                    zone.is_blocked = True
                    zone.blockage_reason = f"Severe Water Inundation ({round(sensor.current_water_level_cm, 1)}cm)"
                    alerts.append({
                        "sensor_id": sensor.id,
                        "zone_id": zone.id,
                        "zone_name": zone.name,
                        "level": "CRITICAL",
                        "water_cm": sensor.current_water_level_cm,
                        "message": f"Flooding emergency at {zone.name}! Road closed to non-emergency transit."
                    })
                elif sensor.risk_level == "HIGH":
                    alerts.append({
                        "sensor_id": sensor.id,
                        "zone_id": zone.id,
                        "zone_name": zone.name,
                        "level": "WARNING",
                        "water_cm": sensor.current_water_level_cm,
                        "message": f"Urban runoff accumulating at {zone.name}. Reduce vehicle speed."
                    })

        return alerts

    def clear_flood(self, sensor_id: Optional[str] = None) -> None:
        """Resets water level sensors and unblocks zone."""
        targets = [self.sensors[sensor_id]] if sensor_id and sensor_id in self.sensors else self.sensors.values()
        for s in targets:
            s.current_water_level_cm = 0.0
            s.risk_level = "LOW"
            zone = self.graph.get_zone(s.zone_id)
            if zone:
                zone.flood_level_cm = 0.0
                zone.flood_risk = "LOW"
                if "Water Inundation" in zone.blockage_reason:
                    zone.is_blocked = False
                    zone.blockage_reason = ""

    def get_all(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.sensors.values()]
