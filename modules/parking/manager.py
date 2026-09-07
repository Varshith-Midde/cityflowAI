"""
Intelligent Smart Parking Management & Prediction Engine.
Monitors parking facility occupancy, predicts demand surges,
and guides riders to nearest open parking slots.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from city_model import CityRoadGraph, TrafficZone


@dataclass
class ParkingFacility:
    id: str
    zone_id: str
    name: str
    total_spots: int
    occupied_spots: int
    ev_charging_spots: int = 15
    ev_occupied_spots: int = 6
    hourly_rate_inr: float = 40.0

    @property
    def available_spots(self) -> int:
        return max(0, self.total_spots - self.occupied_spots)

    @property
    def occupancy_pct(self) -> float:
        if self.total_spots == 0:
            return 0.0
        return (self.occupied_spots / self.total_spots) * 100.0

    @property
    def status(self) -> str:
        pct = self.occupancy_pct
        if pct >= 95.0:
            return "FULL"
        elif pct >= 75.0:
            return "LIMITED"
        elif pct >= 40.0:
            return "MODERATE"
        else:
            return "AVAILABLE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "zone_id": self.zone_id,
            "name": self.name,
            "total_spots": self.total_spots,
            "occupied_spots": self.occupied_spots,
            "available_spots": self.available_spots,
            "occupancy_pct": round(self.occupancy_pct, 1),
            "status": self.status,
            "ev_charging_spots": self.ev_charging_spots,
            "ev_available": max(0, self.ev_charging_spots - self.ev_occupied_spots),
            "hourly_rate_inr": self.hourly_rate_inr
        }


class SmartParkingManager:
    """Manages parking facilities across the smart city."""

    def __init__(self, city_graph: CityRoadGraph):
        self.graph = city_graph
        self.facilities: Dict[str, ParkingFacility] = {}
        self._init_facilities()

    def _init_facilities(self) -> None:
        """Configures default parking lots based on city zones."""
        for zid, zone in self.graph.zones.items():
            if zone.type == "parking" or zone.parking_total_spots > 0:
                total = zone.parking_total_spots or 100
                occupied = int(total * 0.55)
                fac = ParkingFacility(
                    id=f"PARK-{zid.replace('zone_', '').upper()}",
                    zone_id=zid,
                    name=f"{zone.name} Parking",
                    total_spots=total,
                    occupied_spots=occupied
                )
                self.facilities[fac.id] = fac
                zone.parking_total_spots = total
                zone.parking_occupied_spots = occupied

    def update_from_camera(self, zone_id: str, detected_parked_count: int) -> Optional[ParkingFacility]:
        """Updates occupancy from computer vision camera detection."""
        for fac in self.facilities.values():
            if fac.zone_id == zone_id:
                fac.occupied_spots = min(fac.total_spots, max(0, detected_parked_count))
                zone = self.graph.get_zone(zone_id)
                if zone:
                    zone.parking_occupied_spots = fac.occupied_spots
                return fac
        return None

    def find_nearest_available(self, destination_zone_id: str, min_spaces: int = 5) -> Optional[Dict[str, Any]]:
        """Finds closest parking garage to the rider's destination with available capacity."""
        candidates = [
            f for f in self.facilities.values()
            if f.available_spots >= min_spaces
        ]
        if not candidates:
            return None

        best_fac = None
        best_route = None
        min_time = float("inf")

        for fac in candidates:
            route = self.graph.calculate_route(destination_zone_id, fac.zone_id)
            if route and route.estimated_time_sec < min_time:
                min_time = route.estimated_time_sec
                best_fac = fac
                best_route = route

        if not best_fac:
            best_fac = candidates[0]

        return {
            "facility": best_fac.to_dict(),
            "route_from_destination": best_route.to_dict() if best_route else None
        }

    def get_all(self) -> List[Dict[str, Any]]:
        return [f.to_dict() for f in self.facilities.values()]
