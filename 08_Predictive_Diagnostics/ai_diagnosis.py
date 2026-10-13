"""
BeyondTech - AI Diagnosis Module
Generates a plain-English maintenance diagnosis using OpenAI GPT-4o-mini.
Results are cached per unique vehicle + sensor state to avoid repeated API calls.
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def build_sensor_summary(predictions: dict, vehicle_data: dict, vehicle_type: str) -> str:
    """Format the most relevant sensor readings into a compact string for the prompt."""
    lines = []

    if vehicle_type == "EV":
        if "battery_soh" in predictions:
            lines.append(f"Battery SoH: {predictions['battery_soh']:.1f}%")
        if "ev_range" in predictions:
            lines.append(f"EV Range: {predictions['ev_range']:.0f} km")
        fc = vehicle_data.get("Fast_Charge_Percentage", vehicle_data.get("fast_charge_pct"))
        if fc is not None:
            lines.append(f"Fast Charge usage: {float(fc):.0f}%")
        dod = vehicle_data.get("Average_Depth_of_Discharge")
        if dod is not None:
            lines.append(f"Avg Depth of Discharge: {float(dod):.0f}%")
    else:
        if "oil_life" in predictions:
            lines.append(f"Oil life remaining: {predictions['oil_life']:.0f} km")
        if "transmission_life" in predictions:
            lines.append(f"Transmission fluid life: {predictions['transmission_life']:.0f} km")

    if "tire_tread" in predictions:
        lines.append(f"Tire tread: {predictions['tire_tread']:.1f} mm")
    if "brake_thickness" in predictions:
        lines.append(f"Brake pad thickness: {predictions['brake_thickness']:.1f} mm")
    if "air_filter_life" in predictions:
        lines.append(f"Air filter life: {predictions['air_filter_life']:.0f} km")
    if "coolant_life" in predictions:
        lines.append(f"Coolant life: {predictions['coolant_life']:.0f} km")

    harsh = vehicle_data.get("Harsh_Braking_Events")
    if harsh is not None:
        lines.append(f"Harsh braking events/month: {float(harsh):.0f}")
    dist = vehicle_data.get("Total_Distance")
    if dist is not None:
        lines.append(f"Total distance: {float(dist):,.0f} km")
    style_map = {0: "Conservative", 1: "Normal", 2: "Aggressive"}
    style = vehicle_data.get("Driving_Style")
    if style is not None:
        lines.append(f"Driving style: {style_map.get(int(style), 'Normal')}")

    return "\n".join(lines)


@st.cache_data(show_spinner=False)
def get_ai_diagnosis(vehicle_name: str, vehicle_type: str,
                     critical_components: tuple, sensor_summary: str) -> str:
    """
    Call GPT-4o-mini to generate a 2-3 sentence plain-English diagnosis.
    Cached per unique (vehicle, critical components, sensor values) combination.
    Returns None if the API key is missing or the call fails.
    """
    if not OPENAI_API_KEY:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        prompt = (
            f"You are an expert automotive technician AI for a predictive maintenance platform.\n\n"
            f"Vehicle: {vehicle_name} ({vehicle_type})\n"
            f"Critical components: {', '.join(critical_components)}\n"
            f"Current sensor readings:\n{sensor_summary}\n\n"
            f"In 2-3 concise sentences explain: (1) why these components are critical based on "
            f"the sensor data, and (2) what the fleet manager should do immediately. "
            f"Be specific, reference the actual values, and keep it actionable. "
            f"No bullet points — flowing prose only."
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=160,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None
