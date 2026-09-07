"""
Smart Logistics & Delivery Fleet Multi-Stop Route Optimization Engine.
Minimizes travel time, fuel burn, and idle carbon emissions for commercial delivery fleets
by dynamically sequencing drop-offs based on real-time traffic congestion.
"""
import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from city_model import CityRoadGraph


@dataclass
class DeliveryPackage:
    id: str
    destination_zone_id: str
    recipient: str
    priority: str = "NORMAL"  # EXPRESS, NORMAL
    weight_kg: float = 5.0
    status: str = "IN_TRANSIT"  # PENDING, IN_TRANSIT, DELIVERED


@dataclass
class DeliveryVehicle:
    id: str
    name: str
    vehicle_type: str  # EV_CARGO_VAN, DIESEL_TRUCK, ELECTRIC_SCOOTER
    home_hub_zone_id: str = "zone_logistics_park"
    current_zone_id: str = "zone_logistics_park"
    assigned_packages: List[DeliveryPackage] = field(default_factory=list)
    optimized_path: List[str] = field(default_factory=list)
    total_distance_km: float = 0.0
    fuel_saved_liters: float = 0.0
    co2_reduced_kg: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "vehicle_type": self.vehicle_type,
            "current_zone_id": self.current_zone_id,
            "packages_count": len(self.assigned_packages),
            "optimized_path": self.optimized_path,
            "total_distance_km": round(self.total_distance_km, 2),
            "fuel_saved_liters": round(self.fuel_saved_liters, 2),
            "co2_reduced_kg": round(self.co2_reduced_kg, 2)
        }


class SmartLogisticsOptimizer:
    """Solves multi-stop delivery routes and tracks commercial freight fleet."""

    def __init__(self, city_graph: CityRoadGraph):
        self.graph = city_graph
        self.fleet: Dict[str, DeliveryVehicle] = {}
        self._init_fleet()

    def _init_fleet(self) -> None:
        """Initializes sample delivery fleet at Logistics Distribution Hub."""
        v1 = DeliveryVehicle(
            id="LOG-01",
            name="CityFlow Green EV Van 01",
            vehicle_type="EV_CARGO_VAN",
            assigned_packages=[
                DeliveryPackage(id="PKG-101", destination_zone_id="zone_market_square", recipient="Retail Mart"),
                DeliveryPackage(id="PKG-102", destination_zone_id="zone_it_corridor", recipient="Tech HQ"),
                DeliveryPackage(id="PKG-103", destination_zone_id="zone_residential_avenue", recipient="Hilltop Apts")
            ]
        )
        self.fleet[v1.id] = v1

        v2 = DeliveryVehicle(
            id="LOG-02",
            name="Rapid Express Van 02",
            vehicle_type="DIESEL_TRUCK",
            assigned_packages=[
                DeliveryPackage(id="PKG-201", destination_zone_id="zone_central_hospital", recipient="Medical Supplies", priority="EXPRESS"),
                DeliveryPackage(id="PKG-202", destination_zone_id="zone_metro_station", recipient="Metro Parcel Lockers")
            ]
        )
        self.fleet[v2.id] = v2

        # Optimize their routes initially
        self.optimize_all_fleet_routes()

    def plan_multistop_route(self, origin_zone: str, stop_zones: List[str]) -> Dict[str, Any]:
        """
        Solves optimal multi-stop sequence using a Greedy Nearest-Neighbor heuristic
        factoring in live congestion edge costs from CityRoadGraph.
        """
        if not stop_zones:
            return {"sequence": [origin_zone], "total_distance_km": 0.0, "total_time_sec": 0.0}

        unvisited = list(stop_zones)
        curr = origin_zone
        sequence = [curr]
        full_path = [curr]
        total_dist = 0.0
        total_time = 0.0

        while unvisited:
            best_next = None
            best_route = None
            min_cost = float("inf")

            for target in unvisited:
                route = self.graph.calculate_route(curr, target, avoid_flooded=True)
                if route and route.estimated_time_sec < min_cost:
                    min_cost = route.estimated_time_sec
                    best_next = target
                    best_route = route

            if not best_next:
                best_next = unvisited.pop(0)
                sequence.append(best_next)
                curr = best_next
            else:
                unvisited.remove(best_next)
                sequence.append(best_next)
                if best_route:
                    # Append intermediate path steps
                    full_path.extend(best_route.path[1:])
                    total_dist += best_route.total_distance_km
                    total_time += best_route.estimated_time_sec
                curr = best_next

        # Return to hub
        return_route = self.graph.calculate_route(curr, origin_zone, avoid_flooded=True)
        if return_route:
            full_path.extend(return_route.path[1:])
            total_dist += return_route.total_distance_km
            total_time += return_route.estimated_time_sec

        # Fuel & CO2 savings estimation vs naive gridlock routing
        fuel_saved = total_dist * 0.08  # ~8% fuel saved by congestion bypass
        co2_saved = fuel_saved * 2.31   # kg CO2 per liter

        return {
            "stop_sequence": sequence,
            "full_path": full_path,
            "total_distance_km": round(total_dist, 2),
            "estimated_time_min": round(total_time / 60.0, 1),
            "fuel_saved_liters": round(fuel_saved, 2),
            "co2_reduced_kg": round(co2_saved, 2)
        }

    def optimize_all_fleet_routes(self) -> List[Dict[str, Any]]:
        results = []
        for v_id, vehicle in self.fleet.items():
            destinations = [pkg.destination_zone_id for pkg in vehicle.assigned_packages]
            plan = self.plan_multistop_route(vehicle.home_hub_zone_id, destinations)
            vehicle.optimized_path = plan["full_path"]
            vehicle.total_distance_km = plan["total_distance_km"]
            vehicle.fuel_saved_liters = plan["fuel_saved_liters"]
            vehicle.co2_reduced_kg = plan["co2_reduced_kg"]
            results.append({"vehicle_id": v_id, "plan": plan})
        return results

    def get_all(self) -> List[Dict[str, Any]]:
        return [v.to_dict() for v in self.fleet.values()]
