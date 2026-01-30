"""
Maintenance Cost Calculator
Calculates predicted maintenance costs based on vehicle condition and predictions
"""

import numpy as np

class MaintenanceCostCalculator:
    """Advanced maintenance cost calculator with predictive modeling"""
    
    def __init__(self, vehicle_type="EV", region="US"):
        self.vehicle_type = vehicle_type
        self.region = region
        
        # Base cost database (in USD)
        self.base_costs = {
            "EV": {
                "oil_change": 0,
                "brake_pads": 350,
                "brake_rotors": 600,
                "tire_replacement": 800,
                "tire_rotation": 50,
                "battery_inspection": 150,
                "battery_replacement": 8000,
                "coolant_flush": 120,
                "cabin_filter": 40,
                "wiper_blades": 60,
                "alignment": 100,
                "air_filter": 0,
                "transmission_fluid": 0,
            },
            "ICE": {
                "oil_change": 80,
                "brake_pads": 300,
                "brake_rotors": 500,
                "tire_replacement": 700,
                "tire_rotation": 40,
                "air_filter": 50,
                "fuel_filter": 70,
                "spark_plugs": 200,
                "coolant_flush": 150,
                "transmission_fluid": 180,
                "cabin_filter": 35,
                "battery_inspection": 100,
                "battery_replacement": 500,
            },
            "Hybrid": {
                "oil_change": 70,
                "brake_pads": 320,
                "brake_rotors": 550,
                "tire_replacement": 750,
                "tire_rotation": 45,
                "air_filter": 45,
                "fuel_filter": 65,
                "spark_plugs": 180,
                "coolant_flush": 140,
                "transmission_fluid": 200,
                "cabin_filter": 40,
                "battery_inspection": 120,
                "battery_replacement": 3000,
            },
        }
        
        self.regional_multipliers = {
            "US": 1.0,
            "EU": 1.15,
            "UK": 1.20,
            "CN": 0.70,
        }
        
        self.labor_rates = {"US": 120, "EU": 95, "UK": 110, "CN": 45}
    
    def calculate_oil_change_cost(self, km_until_change, oil_viscosity=5.0):
        """Calculate oil change cost"""
        if self.vehicle_type == "EV":
            return None
        
        base_cost = self.base_costs[self.vehicle_type].get("oil_change", 80)
        if oil_viscosity >= 5.0:
            base_cost += 30
        
        if km_until_change < 500:
            urgency, multiplier = "Critical", 1.2
        elif km_until_change < 1500:
            urgency, multiplier = "High", 1.1
        else:
            urgency, multiplier = "Medium", 1.0
        
        days_until = int(km_until_change / 50)
        total_cost = base_cost * multiplier * self.regional_multipliers[self.region]
        
        return {
            "service": "Oil Change",
            "cost": round(total_cost, 2),
            "urgency": urgency,
            "days_until": days_until,
            "km_remaining": km_until_change,
        }
    
    def calculate_brake_cost(self, brake_thickness_mm):
        """Calculate brake service cost"""
        services = []
        base_pad_cost = self.base_costs[self.vehicle_type].get("brake_pads", 300)
        base_rotor_cost = self.base_costs[self.vehicle_type].get("brake_rotors", 500)
        labor_rate = self.labor_rates[self.region]
        
        if brake_thickness_mm < 3:
            cost = (base_pad_cost + base_rotor_cost + labor_rate * 2.5) * self.regional_multipliers[self.region]
            services.append({
                "service": "Brake Pads & Rotors Replacement",
                "cost": round(cost, 2),
                "urgency": "Critical",
                "days_until": 7,
            })
        elif brake_thickness_mm < 5:
            cost = (base_pad_cost + labor_rate * 1.5) * self.regional_multipliers[self.region]
            services.append({
                "service": "Brake Pads Replacement",
                "cost": round(cost, 2),
                "urgency": "High",
                "days_until": 21,
            })
        elif brake_thickness_mm < 7:
            cost = (base_pad_cost * 0.5 + labor_rate * 1.0) * self.regional_multipliers[self.region]
            services.append({
                "service": "Brake Inspection",
                "cost": round(cost, 2),
                "urgency": "Medium",
                "days_until": 60,
            })
        
        return services
    
    def calculate_tire_cost(self, tire_tread_mm, tire_pressure):
        """Calculate tire service cost"""
        services = []
        tire_replacement_cost = self.base_costs[self.vehicle_type].get("tire_replacement", 700)
        tire_rotation_cost = self.base_costs[self.vehicle_type].get("tire_rotation", 40)
        labor_rate = self.labor_rates[self.region]
        
        if tire_pressure < 28:
            cost = (50 + labor_rate * 0.5) * self.regional_multipliers[self.region]
            services.append({
                "service": "Tire Pressure Check & Repair",
                "cost": round(cost, 2),
                "urgency": "Critical",
                "days_until": 2,
            })
        elif tire_pressure < 30:
            cost = 30 * self.regional_multipliers[self.region]
            services.append({
                "service": "Tire Inflation",
                "cost": round(cost, 2),
                "urgency": "High",
                "days_until": 7,
            })
        
        if tire_tread_mm < 2:
            cost = (tire_replacement_cost + labor_rate * 2) * self.regional_multipliers[self.region]
            services.append({
                "service": "Tire Replacement (Urgent)",
                "cost": round(cost, 2),
                "urgency": "Critical",
                "days_until": 7,
            })
        elif tire_tread_mm < 3:
            cost = (tire_replacement_cost + labor_rate * 2) * self.regional_multipliers[self.region]
            services.append({
                "service": "Tire Replacement (Soon)",
                "cost": round(cost, 2),
                "urgency": "High",
                "days_until": 30,
            })
        
        return services
    
    def calculate_battery_cost(self, soh_percentage):
        """Calculate battery-related costs (for EVs and Hybrids)"""
        if self.vehicle_type not in ["EV", "Hybrid"]:
            return []
        
        services = []
        base_inspection = self.base_costs[self.vehicle_type].get("battery_inspection", 150)
        base_replacement = self.base_costs[self.vehicle_type].get("battery_replacement", 8000)
        
        if soh_percentage < 70:
            cost = base_replacement * self.regional_multipliers[self.region]
            services.append({
                "service": "Battery Pack Replacement",
                "cost": round(cost, 2),
                "urgency": "Critical",
                "days_until": 30,
            })
        elif soh_percentage < 80:
            cost = base_inspection * self.regional_multipliers[self.region]
            services.append({
                "service": "Battery Health Assessment",
                "cost": round(cost, 2),
                "urgency": "Medium",
                "days_until": 60,
            })
        
        return services
    
    def calculate_coolant_cost(self, km_until_change):
        """Calculate coolant service cost"""
        if km_until_change > 20000:
            return []
        
        services = []
        cost = self.base_costs[self.vehicle_type].get("coolant_flush", 120) * self.regional_multipliers[self.region]
        
        if km_until_change < 5000:
            urgency, days = "Critical", 14
        elif km_until_change < 10000:
            urgency, days = "High", 30
        else:
            urgency, days = "Medium", 60
        
        services.append({
            "service": "Coolant System Flush",
            "cost": round(cost, 2),
            "urgency": urgency,
            "days_until": days,
        })
        
        return services
    
    def calculate_air_filter_cost(self, km_until_change):
        """Calculate air filter cost"""
        if self.vehicle_type == "EV" or km_until_change > 10000:
            return []
        
        services = []
        cost = self.base_costs[self.vehicle_type].get("air_filter", 50) * self.regional_multipliers[self.region]
        
        if km_until_change < 2000:
            urgency, days = "High", 14
        else:
            urgency, days = "Medium", 30
        
        services.append({
            "service": "Air Filter Replacement",
            "cost": round(cost, 2),
            "urgency": urgency,
            "days_until": days,
        })
        
        return services
    
    def calculate_transmission_cost(self, km_until_change):
        """Calculate transmission service cost"""
        if self.vehicle_type == "EV" or km_until_change > 30000:
            return []
        
        services = []
        cost = self.base_costs[self.vehicle_type].get("transmission_fluid", 180) * self.regional_multipliers[self.region]
        
        if km_until_change < 10000:
            urgency, days = "High", 30
        else:
            urgency, days = "Medium", 60
        
        services.append({
            "service": "Transmission Fluid Change",
            "cost": round(cost, 2),
            "urgency": urgency,
            "days_until": days,
        })
        
        return services
    
    def calculate_total_maintenance_costs(self, predictions):
        """Calculate comprehensive maintenance costs"""
        all_services = []
        
        # Oil change
        if predictions.get("oil_life"):
            oil_service = self.calculate_oil_change_cost(predictions["oil_life"])
            if oil_service:
                all_services.append(oil_service)
        
        # Brakes
        if predictions.get("brake_pad_thickness"):
            brake_services = self.calculate_brake_cost(predictions["brake_pad_thickness"])
            all_services.extend(brake_services)
        
        # Tires
        if predictions.get("tire_tread_depth") and predictions.get("tire_pressure"):
            tire_services = self.calculate_tire_cost(
                predictions["tire_tread_depth"],
                predictions["tire_pressure"]
            )
            all_services.extend(tire_services)
        
        # Battery
        if predictions.get("battery_soh"):
            battery_services = self.calculate_battery_cost(predictions["battery_soh"])
            all_services.extend(battery_services)
        
        # Coolant
        if predictions.get("coolant_life"):
            coolant_services = self.calculate_coolant_cost(predictions["coolant_life"])
            all_services.extend(coolant_services)
        
        # Air filter
        if predictions.get("air_filter_life"):
            filter_services = self.calculate_air_filter_cost(predictions["air_filter_life"])
            all_services.extend(filter_services)
        
        # Transmission
        if predictions.get("transmission_life"):
            trans_services = self.calculate_transmission_cost(predictions["transmission_life"])
            all_services.extend(trans_services)
        
        # Calculate projections
        costs_30_days = sum(s["cost"] for s in all_services if s.get("days_until", 999) <= 30)
        costs_90_days = sum(s["cost"] for s in all_services if s.get("days_until", 999) <= 90)
        
        # Routine annual costs vary by vehicle type
        routine_annual = {
            "EV": 300,
            "Hybrid": 600,
            "ICE": 800
        }.get(self.vehicle_type, 500)
        
        costs_annual = costs_90_days * 1.5 + routine_annual
        
        return {
            "services": sorted(all_services, 
                             key=lambda x: {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}[x["urgency"]]),
            "summary": {
                "30_days": round(costs_30_days, 2),
                "90_days": round(costs_90_days, 2),
                "annual_estimate": round(costs_annual, 2),
                "total_services": len(all_services),
                "critical_count": sum(1 for s in all_services if s["urgency"] == "Critical"),
            }
        }