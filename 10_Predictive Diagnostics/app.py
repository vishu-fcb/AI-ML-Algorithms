"""
BeyondTech - Predictive Maintenance Dashboard
Complete Streamlit application with all predictive maintenance features
"""

import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import os
from maintenance_cost_calculator import MaintenanceCostCalculator
from vehicle_data import vehicles_data, feature_mappings
import marketplace
from streamlit_folium import st_folium

# Page configuration
st.set_page_config(
    page_title="BeyondTech - Predictive Maintenance",
    layout="wide",
   # page_icon="🚗",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 100%);
        color: white;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #00C853 0%, #00E676 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #00E676 0%, #00C853 100%);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 200, 83, 0.4);
    }
    .metric-card {
        background: rgba(30, 30, 30, 0.8);
        border-radius: 15px;
        padding: 20px;
        border-left: 5px solid #00C853;
        margin: 10px 0;
    }
    .alert-high {
        background: rgba(255, 68, 68, 0.2);
        border-left-color: #FF4444 !important;
    }
    .alert-medium {
        background: rgba(255, 179, 102, 0.2);
        border-left-color: #FFB366 !important;
    }
    h1, h2, h3 {
        color: white !important;
    }
    .vehicle-card {
        background: rgba(30, 30, 30, 0.6);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        border: 2px solid transparent;
        transition: all 0.3s;
    }
    .vehicle-card:hover {
        border-color: #00C853;
        transform: scale(1.02);
    }
    /* ── Forecast controls: pill-style radio buttons ── */
    div[data-testid="stRadio"] > div {
        gap: 6px !important;
    }
    div[data-testid="stRadio"] label {
        background-color: rgba(25, 25, 45, 0.95) !important;
        color: white !important;
        padding: 5px 16px !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        cursor: pointer !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        transition: all 0.15s !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
        background-color: white !important;
        color: black !important;
        border-color: white !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) p,
    div[data-testid="stRadio"] label:has(input:checked) span {
        color: black !important;
    }
    /* Hide the radio circle — keep only the label text */
    div[data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    """Load all trained ML models"""
    models = {}
    model_files = {
        'ev_range': 'Trained Model/ev_range_model.pkl',
        'oil_life': 'Trained Model/oil_life_model.pkl',
        'tire_wear': 'Trained Model/tire_wear_model.pkl',
        'brake_pad': 'Trained Model/brake_pad_model.pkl',
        'battery_degradation': 'Trained Model/battery_degradation_model.pkl',
        'coolant_health': 'Trained Model/coolant_health_model.pkl',
        'air_filter': 'Trained Model/air_filter_model.pkl',
        'transmission_health': 'Trained Model/transmission_health_model.pkl',
    }
    
    for name, filepath in model_files.items():
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                models[name] = pickle.load(f)
    
    return models


def calculate_health_score(predictions, vehicle_type="EV"):
    """
    Calculate overall vehicle health score (0-100) with realistic thresholds
    
    IMPROVEMENTS:
    - Fixed battery double-counting issue
    - Realistic safety thresholds for tires and brakes
    - Non-linear scoring that properly penalizes critical conditions
    - Critical condition penalty for multiple issues
    """
    
    component_scores = {}
    
    # ============================================================================
    # 1. BATTERY/RANGE HEALTH (35% for EVs, 30% for ICE/Hybrid)
    # ============================================================================
    battery_score = 0
    
    if vehicle_type == "EV":
        # For EVs, combine range and SoH properly (no double counting)
        if 'ev_range' in predictions and 'battery_soh' in predictions:
            range_score = calculate_range_score(predictions['ev_range'])
            soh_score = (predictions['battery_soh'] / 100) * 100
            battery_score = (range_score + soh_score) / 2
        elif 'ev_range' in predictions:
            battery_score = calculate_range_score(predictions['ev_range'])
        elif 'battery_soh' in predictions:
            battery_score = (predictions['battery_soh'] / 100) * 100
        else:
            battery_score = 100
    else:
        if 'battery_soh' in predictions:
            battery_score = (predictions['battery_soh'] / 100) * 100
        else:
            battery_score = 100
    
    component_scores['battery'] = battery_score
    
    # ============================================================================
    # 2. OIL LIFE (20% for ICE/Hybrid)
    # ============================================================================
    if vehicle_type != "EV" and 'oil_life' in predictions:
        oil_score = calculate_oil_score(predictions['oil_life'])
        component_scores['oil'] = oil_score
    else:
        component_scores['oil'] = None
    
    # ============================================================================
    # 3. TIRE HEALTH (20% for EVs, 15% for ICE/Hybrid)
    # ============================================================================
    if 'tire_tread' in predictions:
        tire_score = calculate_tire_score(predictions['tire_tread'])
        component_scores['tire'] = tire_score
    else:
        component_scores['tire'] = 100
    
    # ============================================================================
    # 4. BRAKE HEALTH (25% for EVs, 20% for ICE/Hybrid)
    # ============================================================================
    if 'brake_thickness' in predictions:
        brake_score = calculate_brake_score(predictions['brake_thickness'])
        component_scores['brake'] = brake_score
    else:
        component_scores['brake'] = 100
    
    # ============================================================================
    # 5. OTHER SYSTEMS (20% for EVs, 15% for ICE/Hybrid)
    # ============================================================================
    other_scores = []
    
    if 'coolant_life' in predictions:
        other_scores.append(calculate_coolant_score(predictions['coolant_life']))
    if 'air_filter_life' in predictions:
        other_scores.append(calculate_air_filter_score(predictions['air_filter_life']))
    if 'transmission_life' in predictions and vehicle_type != "EV":
        other_scores.append(calculate_transmission_score(predictions['transmission_life']))
    
    component_scores['other'] = sum(other_scores) / len(other_scores) if other_scores else 100
    
    # ============================================================================
    # FINAL SCORE CALCULATION
    # ============================================================================
    if vehicle_type == "EV":
        weights = {'battery': 0.35, 'tire': 0.20, 'brake': 0.25, 'other': 0.20}
    else:
        weights = {'battery': 0.30, 'oil': 0.20, 'tire': 0.15, 'brake': 0.20, 'other': 0.15}
    
    total_score = sum(component_scores[c] * w for c, w in weights.items() if component_scores.get(c) is not None)
    
    # Apply critical condition penalty
    critical_penalty = calculate_critical_penalty(predictions)
    total_score = max(0, total_score - critical_penalty)
    
    return round(min(total_score, 100), 1)


def calculate_range_score(range_km):
    """Score EV range based on practical usability"""
    if range_km >= 300:
        return 100
    elif range_km >= 200:
        return 80 + ((range_km - 200) / 100) * 20
    elif range_km >= 100:
        return 60 + ((range_km - 100) / 100) * 20
    elif range_km >= 50:
        return 30 + ((range_km - 50) / 50) * 30
    else:
        return max(0, (range_km / 50) * 30)


def calculate_oil_score(oil_km):
    """Score oil life based on maintenance urgency"""
    if oil_km >= 4000:
        return 100
    elif oil_km >= 2000:
        return 80 + ((oil_km - 2000) / 2000) * 20
    elif oil_km >= 1000:
        return 60 + ((oil_km - 1000) / 1000) * 20
    elif oil_km >= 500:
        return 30 + ((oil_km - 500) / 500) * 30
    else:
        return max(0, (oil_km / 500) * 30)


def calculate_tire_score(tread_mm):
    """Score tire condition based on safety standards (legal min: 1.6mm, safe: 3mm)"""
    if tread_mm >= 6:
        return 100
    elif tread_mm >= 4:
        return 80 + ((tread_mm - 4) / 2) * 20
    elif tread_mm >= 3:
        return 60 + ((tread_mm - 3) / 1) * 20
    elif tread_mm >= 2:
        return 30 + ((tread_mm - 2) / 1) * 30
    else:
        return max(0, (tread_mm / 2) * 30)


def calculate_brake_score(thickness_mm):
    """Score brake pad condition based on safety standards (min safe: 3mm)"""
    if thickness_mm >= 8:
        return 100
    elif thickness_mm >= 6:
        return 80 + ((thickness_mm - 6) / 2) * 20
    elif thickness_mm >= 4:
        return 60 + ((thickness_mm - 4) / 2) * 20
    elif thickness_mm >= 3:
        return 30 + ((thickness_mm - 3) / 1) * 30
    else:
        return max(0, (thickness_mm / 3) * 30)


def calculate_coolant_score(coolant_km):
    """Score coolant condition"""
    if coolant_km >= 40000:
        return 100
    elif coolant_km >= 20000:
        return 80 + ((coolant_km - 20000) / 20000) * 20
    elif coolant_km >= 10000:
        return 60 + ((coolant_km - 10000) / 10000) * 20
    elif coolant_km >= 5000:
        return 40 + ((coolant_km - 5000) / 5000) * 20
    else:
        return max(20, (coolant_km / 5000) * 40)


def calculate_air_filter_score(filter_km):
    """Score air filter condition"""
    if filter_km >= 15000:
        return 100
    elif filter_km >= 10000:
        return 80 + ((filter_km - 10000) / 5000) * 20
    elif filter_km >= 5000:
        return 60 + ((filter_km - 5000) / 5000) * 20
    elif filter_km >= 2000:
        return 40 + ((filter_km - 2000) / 3000) * 20
    else:
        return max(20, (filter_km / 2000) * 40)


def calculate_transmission_score(trans_km):
    """Score transmission fluid condition"""
    if trans_km >= 60000:
        return 100
    elif trans_km >= 40000:
        return 80 + ((trans_km - 40000) / 20000) * 20
    elif trans_km >= 20000:
        return 60 + ((trans_km - 20000) / 20000) * 20
    elif trans_km >= 10000:
        return 40 + ((trans_km - 10000) / 10000) * 20
    else:
        return max(20, (trans_km / 10000) * 40)


def calculate_critical_penalty(predictions):
    """Apply additional penalty if multiple critical conditions exist"""
    critical_count = 0
    
    if 'tire_tread' in predictions and predictions['tire_tread'] < 2.0:
        critical_count += 1
    if 'brake_thickness' in predictions and predictions['brake_thickness'] < 3.0:
        critical_count += 1
    if 'ev_range' in predictions and predictions['ev_range'] < 50:
        critical_count += 1
    if 'oil_life' in predictions and predictions['oil_life'] < 500:
        critical_count += 1
    
    if critical_count == 0:
        return 0
    elif critical_count == 1:
        return 5
    elif critical_count == 2:
        return 15
    else:
        return 25

def get_alert_level(value, thresholds):
    """Return alert level and color based on thresholds"""
    if value <= thresholds['critical']:
        return 'Critical', '#FF4444'
    elif value <= thresholds['warning']:
        return 'Warning', '#FFB366'
    else:
        return 'Good', '#00C853'

def create_gauge_chart(value, title, max_value=100):
    """Create a gauge chart for metrics"""
    color = '#00C853' if value > 70 else '#FFB366' if value > 40 else '#FF4444'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16, 'color': 'white'}},
        gauge={
            'axis': {'range': [None, max_value], 'tickcolor': 'white'},
            'bar': {'color': color},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 2,
            'bordercolor': 'gray',
            'steps': [
                {'range': [0, max_value * 0.4], 'color': 'rgba(255, 68, 68, 0.3)'},
                {'range': [max_value * 0.4, max_value * 0.7], 'color': 'rgba(255, 179, 102, 0.3)'},
                {'range': [max_value * 0.7, max_value], 'color': 'rgba(0, 200, 83, 0.3)'}
            ],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=180,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig

def create_degradation_forecast(predictions, vehicle_data, vehicle_type):
    """90-day forward-looking component health degradation forecast."""

    def _h2r(h, a):
        """hex colour → rgba string"""
        r, g, b = int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)
        return f"rgba({r},{g},{b},{a})"

    days = np.arange(0, 91)

    # Daily driving distance estimate
    avg_spd  = float(vehicle_data.get("Average_Speed", vehicle_data.get("Driving_Speed", 60)))
    daily_km = float(np.clip(avg_spd * 1.5, 25, 200))

    # Composite intensity factor (braking + style)
    harsh     = float(vehicle_data.get("Harsh_Braking_Events", 50))
    style     = float(vehicle_data.get("Driving_Style", 1))
    intensity = (0.5 + style * 0.35) * (0.7 + harsh / 200.0)

    components = []

    # ── Tire Tread ──────────────────────────────────────────────────────────────
    if "tire_tread" in predictions:
        cur = float(predictions["tire_tread"])
        lo, hi, warn = 1.6, 8.2, 3.0
        rate = 0.000130 * intensity                         # mm / km
        raw  = np.maximum(lo - 0.3, cur - rate * daily_km * days)
        hlth = np.clip((raw - lo) / (hi - lo) * 100, 0, 100)
        components.append(dict(name="Tires", health=hlth, raw=raw, unit="mm", fmt=".1f",
                               warn_pct=(warn-lo)/(hi-lo)*100, color="#00E5FF"))

    # ── Brake Pads ──────────────────────────────────────────────────────────────
    if "brake_thickness" in predictions:
        cur = float(predictions["brake_thickness"])
        lo, hi, warn = 3.0, 12.0, 4.0
        mtn  = float(vehicle_data.get("Mountain_Driving_Percent", 10))
        rate = 0.000065 * intensity * (1 + mtn / 120.0)    # mm / km
        raw  = np.maximum(lo - 0.2, cur - rate * daily_km * days)
        hlth = np.clip((raw - lo) / (hi - lo) * 100, 0, 100)
        components.append(dict(name="Brake Pads", health=hlth, raw=raw, unit="mm", fmt=".1f",
                               warn_pct=(warn-lo)/(hi-lo)*100, color="#FF6D00"))

    # ── Battery SoH ─────────────────────────────────────────────────────────────
    if "battery_soh" in predictions:
        cur = float(predictions["battery_soh"])
        lo, hi, warn = 60.0, 100.0, 80.0
        fc   = float(vehicle_data.get("Fast_Charge_Percentage", 30))
        dod  = float(vehicle_data.get("Average_Depth_of_Discharge", 60))
        rate = 0.011 + fc * 0.00025 + dod * 0.00010        # % / day
        raw  = np.maximum(lo, cur - rate * days)
        hlth = np.clip((raw - lo) / (hi - lo) * 100, 0, 100)
        components.append(dict(name="Battery SoH", health=hlth, raw=raw, unit="%", fmt=".1f",
                               warn_pct=(warn-lo)/(hi-lo)*100, color="#69FF47"))

    # ── Oil Life (ICE / Hybrid only) ────────────────────────────────────────────
    if "oil_life" in predictions and vehicle_type != "EV":
        cur = float(predictions["oil_life"])
        hi, warn = 10500.0, 1500.0
        raw  = np.maximum(0.0, cur - daily_km * days)
        hlth = np.clip(raw / hi * 100, 0, 100)
        components.append(dict(name="Oil Life", health=hlth, raw=raw, unit="km", fmt=".0f",
                               warn_pct=warn/hi*100, color="#FFD600"))

    # ── Air Filter ──────────────────────────────────────────────────────────────
    if "air_filter_life" in predictions:
        cur = float(predictions["air_filter_life"])
        hi, warn = 25000.0, 2000.0
        aqi  = float(vehicle_data.get("Air_Quality_Index", 60))
        rate = daily_km * (1.0 + max(0, aqi - 50) / 180.0)
        raw  = np.maximum(0.0, cur - rate * days)
        hlth = np.clip(raw / hi * 100, 0, 100)
        components.append(dict(name="Air Filter", health=hlth, raw=raw, unit="km", fmt=".0f",
                               warn_pct=warn/hi*100, color="#E040FB"))

    # ── Coolant ─────────────────────────────────────────────────────────────────
    if "coolant_life" in predictions:
        cur = float(predictions["coolant_life"])
        hi, warn = 70000.0, 5000.0
        hl   = float(vehicle_data.get("Heavy_Load_Percentage", 15))
        rate = daily_km * (1.0 + hl / 180.0)
        raw  = np.maximum(0.0, cur - rate * days)
        hlth = np.clip(raw / hi * 100, 0, 100)
        components.append(dict(name="Coolant", health=hlth, raw=raw, unit="km", fmt=".0f",
                               warn_pct=warn/hi*100, color="#40C4FF"))

    # ── Build figure ─────────────────────────────────────────────────────────────
    fig = go.Figure()

    # Health zone background bands
    fig.add_hrect(y0=70, y1=101, fillcolor="rgba(0,200,83,0.06)",   line_width=0, layer="below")
    fig.add_hrect(y0=40, y1=70,  fillcolor="rgba(255,179,102,0.07)", line_width=0, layer="below")
    fig.add_hrect(y0=0,  y1=40,  fillcolor="rgba(255,68,68,0.09)",   line_width=0, layer="below")

    # Zone boundary lines
    fig.add_hline(y=70, line=dict(color="rgba(0,200,83,0.25)",    width=1, dash="dot"))
    fig.add_hline(y=40, line=dict(color="rgba(255,68,68,0.25)",   width=1, dash="dot"))

    # Zone labels on right axis
    for y, label, color in [(85, "GOOD", "rgba(0,200,83,0.45)"),
                             (55, "WARN", "rgba(255,179,102,0.55)"),
                             (20, "CRIT", "rgba(255,68,68,0.55)")]:
        fig.add_annotation(x=90, y=y, text=label, xanchor="right",
                           font=dict(size=9, color=color), showarrow=False)

    # Today marker
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.5)", width=2))
    fig.add_annotation(x=1, y=99, text="TODAY", xanchor="left",
                       font=dict(size=9, color="rgba(255,255,255,0.6)"), showarrow=False)

    annotation_offsets = {}   # spread overlapping annotations

    for comp in components:
        hlth  = comp["health"]
        color = comp["color"]
        fill  = _h2r(color, 0.07)

        # Gradient fill: separate trace for filled area
        fig.add_trace(go.Scatter(
            x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([hlth, np.zeros(len(days))]),
            fill="toself", fillcolor=fill,
            line=dict(width=0), showlegend=False,
            hoverinfo="skip",
        ))

        # Main line
        fig.add_trace(go.Scatter(
            x=days, y=hlth,
            mode="lines",
            name=comp["name"],
            line=dict(color=color, width=2.5, shape="spline", smoothing=0.5),
            hovertemplate=(
                f"<b>{comp['name']}</b><br>"
                "Day %{x}<br>"
                f"Health: %{{y:.1f}}%<br>"
                f"Value: %{{customdata:{comp['fmt']}}}{comp['unit']}"
                "<extra></extra>"
            ),
            customdata=comp["raw"],
        ))

        # Service-due crossing marker
        warn_h = comp["warn_pct"]
        cross_day = next(
            (i for i in range(len(hlth)-1) if hlth[i] >= warn_h > hlth[i+1]),
            None
        )
        if cross_day is not None and cross_day <= 89:
            fig.add_vline(x=cross_day,
                          line=dict(color=_h2r(color, 0.5), width=1.5, dash="dash"))

            # Stagger overlapping annotations
            ay_offset = annotation_offsets.get(cross_day, -38)
            annotation_offsets[cross_day] = ay_offset - 28

            fig.add_annotation(
                x=cross_day, y=warn_h,
                text=f"<b>{comp['name']}</b><br>⚠ Day {cross_day}",
                showarrow=True, arrowhead=2, arrowsize=0.9,
                arrowcolor=color,
                font=dict(color=color, size=10),
                bgcolor="rgba(10,10,20,0.85)",
                bordercolor=color, borderwidth=1,
                borderpad=4,
                ax=28, ay=ay_offset,
            )

    fig.update_layout(
        title=dict(
            text="90-Day Component Health Forecast",
            font=dict(size=17, color="white"),
            x=0.01,
        ),
        paper_bgcolor="rgba(12,12,22,0.95)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="sans-serif"),
        xaxis=dict(
            title="Days from Today",
            range=[-1, 91],
            gridcolor="rgba(255,255,255,0.05)",
            tickcolor="rgba(255,255,255,0.2)",
            zeroline=False,
            tickvals=[0, 15, 30, 45, 60, 75, 90],
            ticktext=["Now","15d","30d","45d","60d","75d","90d"],
        ),
        yaxis=dict(
            title="Health (%)",
            range=[-2, 102],
            gridcolor="rgba(255,255,255,0.05)",
            tickcolor="rgba(255,255,255,0.2)",
            zeroline=False,
            ticksuffix="%",
        ),
        legend=dict(
            bgcolor="rgba(20,20,35,0.85)",
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
            orientation="h",
            yanchor="bottom", y=1.03,
            xanchor="left",  x=0,
            font=dict(size=11),
        ),
        hovermode="x unified",
        height=420,
        margin=dict(l=10, r=30, t=90, b=40),
    )

    return fig


def create_gauges_row(health_score, batt_health, brake_health, tire_health, second_label="Battery Health"):
    """Render all 4 KPI gauges as one single figure — immune to zoom misalignment."""
    from plotly.subplots import make_subplots

    values = [health_score, batt_health, brake_health, tire_health]
    titles = ["Overall Health", second_label, "Brake Health", "Tire Health"]
    colors = ['#00C853' if v > 70 else '#FFB366' if v > 40 else '#FF4444' for v in values]

    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "indicator"}] * 4],
        horizontal_spacing=0.04,
    )

    for i, (val, title, color) in enumerate(zip(values, titles, colors), start=1):
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=round(val, 1),
                number={"font": {"size": 28, "color": "white"}, "suffix": "%"},
                title={"text": title, "font": {"size": 13, "color": "#AAAAAA"}},
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": "rgba(255,255,255,0.3)",
                        "tickfont": {"size": 9, "color": "rgba(255,255,255,0.4)"},
                        "nticks": 5,
                    },
                    "bar": {"color": color, "thickness": 0.7},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 1,
                    "bordercolor": "rgba(100,100,100,0.4)",
                    "steps": [
                        {"range": [0,  40], "color": "rgba(255,68,68,0.18)"},
                        {"range": [40, 70], "color": "rgba(255,179,102,0.18)"},
                        {"range": [70,100], "color": "rgba(0,200,83,0.18)"},
                    ],
                    "threshold": {
                        "line": {"color": color, "width": 3},
                        "thickness": 0.85,
                        "value": round(val, 1),
                    },
                },
            ),
            row=1, col=i,
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        height=210,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    return fig


def create_component_forecast(component_key, predictions, vehicle_data, days=90, auto_y=True):
    """Single-component 90-day forecast with in-chart range and Y-scale controls."""

    def _h2r(h, a):
        r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
        return f"rgba({r},{g},{b},{a})"

    x = np.arange(0, 91)   # full 90-day index array (distinct from the `days` parameter)
    avg_spd  = float(vehicle_data.get("Average_Speed", vehicle_data.get("Driving_Speed", 60)))
    daily_km = float(np.clip(avg_spd * 1.5, 25, 200))
    harsh    = float(vehicle_data.get("Harsh_Braking_Events", 50))
    style    = float(vehicle_data.get("Driving_Style", 1))
    intensity = (0.5 + style * 0.35) * (0.7 + harsh / 200.0)

    # ── Component specs ────────────────────────────────────────────────────────
    specs = {
        "tire_tread": dict(
            label="Tire Tread", color="#00E5FF", unit="mm", fmt=".2f",
            lo=1.6, hi=8.2, warn=3.0, crit=2.0,
            warn_label="Replace Soon (3 mm)", crit_label="Legal Min (1.6 mm)",
        ),
        "brake_thickness": dict(
            label="Brake Pad", color="#FF6D00", unit="mm", fmt=".2f",
            lo=3.0, hi=12.0, warn=4.0, crit=3.0,
            warn_label="Replace Soon (4 mm)", crit_label="Unsafe (3 mm)",
        ),
        "battery_soh": dict(
            label="Battery SoH", color="#69FF47", unit="%", fmt=".2f",
            lo=60.0, hi=100.0, warn=80.0, crit=70.0,
            warn_label="Degraded (80%)", crit_label="Critical (70%)",
        ),
        "oil_life": dict(
            label="Oil Life", color="#FFD600", unit="km", fmt=".0f",
            lo=0.0, hi=10500.0, warn=1500.0, crit=500.0,
            warn_label="Service Soon (1 500 km)", crit_label="Overdue (500 km)",
        ),
        "air_filter_life": dict(
            label="Air Filter", color="#E040FB", unit="km", fmt=".0f",
            lo=0.0, hi=25000.0, warn=2000.0, crit=500.0,
            warn_label="Replace Soon (2 000 km)", crit_label="Clogged (500 km)",
        ),
        "coolant_life": dict(
            label="Coolant", color="#40C4FF", unit="km", fmt=".0f",
            lo=0.0, hi=70000.0, warn=5000.0, crit=1000.0,
            warn_label="Change Soon (5 000 km)", crit_label="Critical (1 000 km)",
        ),
        "transmission_life": dict(
            label="Transmission Fluid", color="#FF80AB", unit="km", fmt=".0f",
            lo=0.0, hi=120000.0, warn=10000.0, crit=2000.0,
            warn_label="Change Soon (10 000 km)", crit_label="Critical (2 000 km)",
        ),
    }

    if component_key not in specs or component_key not in predictions:
        return go.Figure()

    s   = specs[component_key]
    cur = float(predictions[component_key])
    lo, hi, warn, crit = s["lo"], s["hi"], s["warn"], s["crit"]

    # ── Degradation projection ─────────────────────────────────────────────────
    if component_key == "tire_tread":
        rate = 0.000130 * intensity
        raw  = np.maximum(lo, cur - rate * daily_km * x)

    elif component_key == "brake_thickness":
        mtn  = float(vehicle_data.get("Mountain_Driving_Percent", 10))
        rate = 0.000065 * intensity * (1 + mtn / 120.0)
        raw  = np.maximum(lo, cur - rate * daily_km * x)

    elif component_key == "battery_soh":
        fc   = float(vehicle_data.get("Fast_Charge_Percentage", 30))
        dod  = float(vehicle_data.get("Average_Depth_of_Discharge", 60))
        rate = 0.011 + fc * 0.00025 + dod * 0.00010
        raw  = np.maximum(lo, cur - rate * x)

    elif component_key == "oil_life":
        raw = np.maximum(lo, cur - daily_km * x)

    elif component_key == "air_filter_life":
        aqi  = float(vehicle_data.get("Air_Quality_Index", 60))
        rate = daily_km * (1.0 + max(0, aqi - 50) / 180.0)
        raw  = np.maximum(lo, cur - rate * x)

    elif component_key == "coolant_life":
        hl   = float(vehicle_data.get("Heavy_Load_Percentage", 15))
        rate = daily_km * (1.0 + hl / 180.0)
        raw  = np.maximum(lo, cur - rate * x)

    elif component_key == "transmission_life":
        raw = np.maximum(lo, cur - daily_km * x)

    else:
        raw = np.maximum(lo, cur - (daily_km * 0.001) * x)

    # ── Y values: always plot in raw units (mm / % / km) ──────────────────────
    color = s["color"]
    fig   = go.Figure()

    # Background health zones (in raw units)
    fig.add_hrect(y0=warn, y1=hi * 1.05, fillcolor="rgba(0,200,83,0.06)",    line_width=0, layer="below")
    fig.add_hrect(y0=crit, y1=warn,       fillcolor="rgba(255,179,102,0.08)", line_width=0, layer="below")
    fig.add_hrect(y0=lo,   y1=crit,       fillcolor="rgba(255,68,68,0.10)",   line_width=0, layer="below")

    # Warn / crit threshold lines
    fig.add_hline(y=warn, line=dict(color="rgba(255,179,102,0.6)", width=1.5, dash="dash"))
    fig.add_hline(y=crit, line=dict(color="rgba(255,68,68,0.6)",   width=1.5, dash="dash"))

    # Threshold labels
    fig.add_annotation(x=91, y=warn, text=s["warn_label"], xanchor="right",
                       font=dict(size=10, color="rgba(255,179,102,0.8)"),
                       showarrow=False, xref="x", yref="y")
    fig.add_annotation(x=91, y=crit, text=s["crit_label"], xanchor="right",
                       font=dict(size=10, color="rgba(255,68,68,0.8)"),
                       showarrow=False, xref="x", yref="y")

    # Gradient fill under line
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([raw, np.full(len(x), lo)]),
        fill="toself", fillcolor=_h2r(color, 0.08),
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))

    # Main degradation line
    fig.add_trace(go.Scatter(
        x=x, y=raw,
        mode="lines",
        name=s["label"],
        line=dict(color=color, width=3, shape="spline", smoothing=0.4),
        hovertemplate=f"Day %{{x}}<br><b>{s['label']}: %{{y:{s['fmt']}}} {s['unit']}</b><extra></extra>",
    ))

    # Current value dot
    fig.add_trace(go.Scatter(
        x=[0], y=[cur],
        mode="markers+text",
        marker=dict(color=color, size=12, symbol="circle",
                    line=dict(color="white", width=2)),
        text=[f"  Now: {cur:{s['fmt']}} {s['unit']}"],
        textposition="middle right",
        textfont=dict(color="white", size=11),
        showlegend=False, hoverinfo="skip",
    ))

    # Service-due crossing marker
    cross_warn = next((i for i in range(len(raw)-1) if raw[i] >= warn > raw[i+1]), None)
    cross_crit = next((i for i in range(len(raw)-1) if raw[i] >= crit > raw[i+1]), None)

    for cross_day, cross_y, cross_color, cross_label in [
        (cross_warn, warn, "rgba(255,179,102,0.9)", "⚠ Service Due"),
        (cross_crit, crit, "rgba(255,68,68,0.9)",   "🚨 Critical"),
    ]:
        if cross_day is not None and cross_day <= 89:
            fig.add_vline(x=cross_day,
                          line=dict(color=cross_color, width=1.5, dash="dot"))
            fig.add_annotation(
                x=cross_day, y=cross_y,
                text=f"<b>{cross_label}</b><br>Day {cross_day}",
                showarrow=True, arrowhead=2, arrowsize=0.9,
                arrowcolor=cross_color,
                font=dict(color=cross_color, size=11),
                bgcolor="rgba(10,10,20,0.85)",
                bordercolor=cross_color, borderwidth=1, borderpad=5,
                ax=40, ay=-45,
            )

    # Today marker
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.45)", width=2))

    # ── Y range ───────────────────────────────────────────────────────────────
    y_pad     = (max(raw) - min(raw)) * 0.12 + (hi - lo) * 0.03
    y_lo_auto = max(lo - y_pad * 0.5,  lo - (hi - lo) * 0.05)
    y_hi_auto = min(hi * 1.04,          max(raw) + y_pad)
    y_range   = [y_lo_auto, y_hi_auto] if auto_y else [lo - (hi - lo) * 0.02, hi * 1.04]

    # ── X tick labels clipped to requested day range ──────────────────────────
    all_ticks = [0, 15, 30, 45, 60, 75, 90]
    all_labels = ["Now", "15d", "30d", "45d", "60d", "75d", "90d"]
    tick_vals  = [t for t in all_ticks  if t <= days]
    tick_text  = [l for t, l in zip(all_ticks, all_labels) if t <= days]

    fig.update_layout(
        paper_bgcolor="rgba(12,12,22,0.95)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="sans-serif"),
        height=400,
        margin=dict(l=15, r=120, t=30, b=50),
        showlegend=False,
        hovermode="x unified",
        xaxis=dict(
            title="Days from Today",
            range=[-1, days + 1],
            gridcolor="rgba(255,255,255,0.05)",
            tickcolor="rgba(255,255,255,0.2)",
            zeroline=False,
            tickvals=tick_vals,
            ticktext=tick_text,
        ),
        yaxis=dict(
            title=f"{s['label']} ({s['unit']})",
            range=y_range,
            gridcolor="rgba(255,255,255,0.05)",
            tickcolor="rgba(255,255,255,0.2)",
            zeroline=False,
        ),
    )

    return fig


def generate_recommendations(predictions, vehicle_type):
    """Generate maintenance recommendations"""
    recommendations = []
    
    # Battery range
    if 'ev_range' in predictions and predictions['ev_range'] < 100:
        recommendations.append({
            'icon': '🔋',
            'priority': 'High',
            'title': 'Low Battery Range',
            'message': f'Only {predictions["ev_range"]:.0f} km remaining.',
            'action': 'Charge immediately'
        })
    
    # Oil change
    if 'oil_life' in predictions and predictions['oil_life'] < 1000:
        recommendations.append({
            'icon': '🛢️',
            'priority': 'High',
            'title': 'Oil Change Due Soon',
            'message': f'{predictions["oil_life"]:.0f} km until oil change.',
            'action': 'Schedule service'
        })
    
    # Tire wear
    if 'tire_tread' in predictions and predictions['tire_tread'] < 3:
        recommendations.append({
            'icon': '🛞',
            'priority': 'Critical',
            'title': 'Tire Replacement Needed',
            'message': f'Tread depth: {predictions["tire_tread"]:.1f}mm (min: 1.6mm)',
            'action': 'Replace tires immediately'
        })
    
    # Brake pads
    if 'brake_thickness' in predictions and predictions['brake_thickness'] < 4:
        recommendations.append({
            'icon': '🔧',
            'priority': 'High',
            'title': 'Brake Pad Replacement',
            'message': f'Pad thickness: {predictions["brake_thickness"]:.1f}mm',
            'action': 'Schedule brake service'
        })
    
    # Battery health
    if 'battery_soh' in predictions and predictions['battery_soh'] < 80:
        recommendations.append({
            'icon': '⚡',
            'priority': 'Medium',
            'title': 'Battery Degradation',
            'message': f'Battery health: {predictions["battery_soh"]:.0f}%',
            'action': 'Consider battery assessment'
        })
    
    if not recommendations:
        recommendations.append({
            'icon': '✅',
            'priority': 'Low',
            'title': 'All Systems Normal',
            'message': 'Vehicle is in good condition',
            'action': 'Continue regular maintenance'
        })
    
    return sorted(recommendations, key=lambda x: {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}[x['priority']])

def make_predictions(models, vehicle_data):
    """Make all predictions for a vehicle, skipping models irrelevant to vehicle type."""
    vehicle_type = vehicle_data.get('type', 'EV')
    predictions = {}

    # Models that only apply to specific vehicle types
    ev_only_models  = {'ev_range', 'battery_degradation'}
    ice_only_models = {'oil_life', 'transmission_health'}

    for model_name, features in feature_mappings.items():
        if vehicle_type == 'ICE' and model_name in ev_only_models:
            continue
        if vehicle_type == 'EV' and model_name in ice_only_models:
            continue
        if model_name in models:
            try:
                X = pd.DataFrame([{k: vehicle_data[k] for k in features}])
                pred = models[model_name].predict(X)[0]
                predictions[model_name] = pred
            except Exception as e:
                st.error(f"Error predicting {model_name}: {e}")
    
    # Map to friendly names
    result = {}
    if 'ev_range' in predictions:
        result['ev_range'] = predictions['ev_range']
    if 'oil_life' in predictions:
        result['oil_life'] = predictions['oil_life']
    if 'tire_wear' in predictions:
        result['tire_tread'] = predictions['tire_wear']
        result['tire_pressure'] = vehicle_data.get('Tire_Pressure', 32)
    if 'brake_pad' in predictions:
        result['brake_thickness'] = predictions['brake_pad']
    if 'battery_degradation' in predictions:
        result['battery_soh'] = predictions['battery_degradation']
    if 'coolant_health' in predictions:
        result['coolant_life'] = predictions['coolant_health']
    if 'air_filter' in predictions:
        result['air_filter_life'] = predictions['air_filter']
    if 'transmission_health' in predictions:
        result['transmission_life'] = predictions['transmission_health']
    
    return result

def display_maintenance_costs(predictions, vehicle_type, region="US"):
    """Display comprehensive maintenance cost analysis"""
    calculator = MaintenanceCostCalculator(vehicle_type=vehicle_type, region=region)
    cost_results = calculator.calculate_total_maintenance_costs(predictions)
    
    st.markdown("---")
    st.markdown("### 💰 Estimated Maintenance Costs")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Next 30 Days",
            f"${cost_results['summary']['30_days']:,.0f}",
            delta=f"{cost_results['summary']['critical_count']} critical" if cost_results['summary']['critical_count'] > 0 else None,
            delta_color="inverse"
        )
    
    with col2:
        st.metric("Next 90 Days", f"${cost_results['summary']['90_days']:,.0f}")
    
    with col3:
        st.metric("Annual Estimate", f"${cost_results['summary']['annual_estimate']:,.0f}")
    
    with col4:
        st.metric("Total Services", cost_results['summary']['total_services'])
    
    # Detailed breakdown
    if cost_results['services']:
        st.markdown("---")
        st.markdown("### 📋 Detailed Cost Breakdown")
        
        for service in cost_results['services']:
            urgency_colors = {
                "Critical": "#FF4444",
                "High": "#FFB366",
                "Medium": "#FFA726",
                "Low": "#00C853"
            }
            
            color = urgency_colors[service['urgency']]
            days_text = f"Due in {service['days_until']} days" if service.get('days_until') else "Schedule soon"
            
            st.markdown(f"""
            <div style="background: rgba(30, 30, 30, 0.6); border-left: 5px solid {color}; 
                        border-radius: 10px; padding: 15px; margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div style="flex: 1;">
                        <h4 style="margin: 0; color: white;">{service['service']}</h4>
                        <p style="margin: 5px 0; color: #888; font-size: 12px;">⏰ {days_text}</p>
                    </div>
                    <div style="text-align: right;">
                        <p style="margin: 0; font-size: 24px; font-weight: bold; color: {color};">
                            ${service['cost']:,.2f}
                        </p>
                        <span style="background: {color}; color: white; padding: 4px 12px; 
                                     border-radius: 15px; font-size: 11px; font-weight: bold;">
                            {service['urgency']}
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_marketplace_panel(recommendations: list, vehicle_data: dict):
    """
    Show a Parts & Workshop Marketplace panel when at least one component is Critical.
    Parts prices are static estimates; workshop map uses Google Maps Places API if key is set.
    """
    if not any(r['priority'] == 'Critical' for r in recommendations):
        return

    with st.expander("🛒 Parts & Workshop Marketplace", expanded=False):
        st.markdown(
            "Parts prices shown in EUR (estimated). "
            "Nearby workshops based on your vehicle's home city."
        )
        marketplace.render_parts_table(vehicle_data['type'])

        st.markdown("---")
        st.markdown("#### Nearby Workshops")

        if marketplace.GOOGLE_MAPS_API_KEY:
            workshop_map = marketplace.render_workshop_map(vehicle_data['name'])
            if workshop_map:
                st_folium(workshop_map, width="100%", height=450, returned_objects=[])
            else:
                st.warning("Could not load workshop locations for this vehicle.")
        else:
            st.info(
                "Set `GOOGLE_MAPS_API_KEY` in a `.env` file in the app directory "
                "to see nearby workshops on an interactive map."
            )


@st.cache_data
def compute_fleet_stats(_models):
    """Compute active alert count and average health score across all vehicles."""
    scores = []
    alert_count = 0
    for vin, data in vehicles_data.items():
        preds = make_predictions(_models, data)
        score = calculate_health_score(preds, data['type'])
        scores.append(score)
        if score < 70 or calculate_critical_penalty(preds) > 0:
            alert_count += 1
    avg = round(sum(scores) / len(scores), 1) if scores else 0.0
    return alert_count, avg


def main():
    """Main application"""

    # Load models early so sidebar stats can use them
    models = load_models()

    if not models:
        st.error("⚠️ No models found! Please train models first by running: `python AI_prediction_model.py`")
        return

    # Sidebar
    with st.sidebar:
        st.markdown("### 🚗 BeyondTech")
        st.markdown("**Predictive Maintenance Platform**")
        st.markdown("---")

        if st.button("🏠 Dashboard Home"):
            st.session_state.selected_vehicle = None

        st.markdown("---")
        st.markdown("**Region**")
        st.session_state["region"] = st.selectbox(
            "Pricing Region",
            options=["US", "EU", "UK", "CN"],
            format_func=lambda r: {"US": "🇺🇸 United States", "EU": "🇪🇺 Europe", "UK": "🇬🇧 United Kingdom", "CN": "🇨🇳 China"}[r],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("**Quick Stats**")
        _active_alerts, _avg_health = compute_fleet_stats(models)
        st.metric("Total Vehicles", len(vehicles_data))
        st.metric("Active Alerts", _active_alerts)
        st.metric("Avg Health Score", _avg_health)
        
        st.markdown("---")
        st.markdown("**About**")
        st.info("BeyondTech uses machine learning to predict maintenance needs before failures occur.")

        # Live Simulation Panel — only visible when a vehicle detail view is active
        if st.session_state.get("selected_vehicle"):
            _vin = st.session_state.selected_vehicle
            _base = vehicles_data[_vin]

            st.markdown("---")
            st.markdown("### Simulation Controls")
            st.caption("Pick a scenario or tune the sliders below.")

            # ── Scenario Preset Buttons ──
            st.markdown("**Quick Scenarios**")
            _is_ev_sim = _base['type'] == 'EV'
            _SIM_KEYS = ["sim_Ambient_Temperature", "sim_Driving_Speed", "sim_Total_Distance",
                         "sim_Harsh_Braking_Events", "sim_Driving_Style", "sim_Air_Quality_Index",
                         "sim_SoC", "sim_Fast_Charge_Percentage",
                         "sim_Engine_RPM", "sim_Distance_Since_Last_Change"]

            def _apply_preset(values: dict):
                for k in _SIM_KEYS:
                    st.session_state.pop(k, None)
                for k, v in values.items():
                    st.session_state[k] = v

            if _is_ev_sim:
                _presets = {
                    "❄️ Winter":     {"sim_SoC": 45, "sim_Ambient_Temperature": -8,  "sim_Driving_Speed": 52,  "sim_Harsh_Braking_Events": 35,  "sim_Driving_Style": 0, "sim_Fast_Charge_Percentage": 25, "sim_Air_Quality_Index": 55},
                    "🛣️ Highway":    {"sim_SoC": 82, "sim_Ambient_Temperature": 22,  "sim_Driving_Speed": 118, "sim_Harsh_Braking_Events": 12,  "sim_Driving_Style": 0, "sim_Fast_Charge_Percentage": 20, "sim_Air_Quality_Index": 40},
                    "🏎️ Aggressive": {"sim_SoC": 35, "sim_Ambient_Temperature": 28,  "sim_Driving_Speed": 115, "sim_Harsh_Braking_Events": 175, "sim_Driving_Style": 2, "sim_Fast_Charge_Percentage": 65, "sim_Air_Quality_Index": 88},
                    "💀 Neglected":  {"sim_SoC": 28, "sim_Ambient_Temperature": 32,  "sim_Driving_Speed": 95,  "sim_Total_Distance": 148000, "sim_Harsh_Braking_Events": 165, "sim_Driving_Style": 2, "sim_Fast_Charge_Percentage": 80, "sim_Air_Quality_Index": 130},
                }
            else:
                _presets = {
                    "❄️ Winter":     {"sim_Engine_RPM": 2000, "sim_Ambient_Temperature": -8,  "sim_Driving_Speed": 52,  "sim_Harsh_Braking_Events": 35,  "sim_Driving_Style": 0, "sim_Distance_Since_Last_Change": 9000,  "sim_Air_Quality_Index": 55},
                    "🛣️ Highway":    {"sim_Engine_RPM": 2800, "sim_Ambient_Temperature": 22,  "sim_Driving_Speed": 118, "sim_Harsh_Braking_Events": 12,  "sim_Driving_Style": 0, "sim_Distance_Since_Last_Change": 4000,  "sim_Air_Quality_Index": 40},
                    "🏎️ Aggressive": {"sim_Engine_RPM": 4000, "sim_Ambient_Temperature": 28,  "sim_Driving_Speed": 115, "sim_Harsh_Braking_Events": 175, "sim_Driving_Style": 2, "sim_Distance_Since_Last_Change": 10000, "sim_Air_Quality_Index": 88},
                    "💀 Neglected":  {"sim_Engine_RPM": 3500, "sim_Ambient_Temperature": 32,  "sim_Driving_Speed": 95,  "sim_Total_Distance": 148000, "sim_Harsh_Braking_Events": 165, "sim_Driving_Style": 2, "sim_Distance_Since_Last_Change": 14000, "sim_Air_Quality_Index": 130},
                }

            col_a, col_b = st.columns(2)
            _preset_items = list(_presets.items())
            for _col, (_label, _vals) in zip([col_a, col_a, col_b, col_b], _preset_items):
                with _col:
                    _key = "preset_" + _label.split()[1].lower()
                    if st.button(_label, key=_key, use_container_width=True):
                        _apply_preset(_vals)
                        st.rerun()

            st.markdown("---")
            st.caption("Or adjust manually:")

            if st.button("↺ Reset to Baseline", key="sim_reset", use_container_width=True):
                for _k in _SIM_KEYS:
                    st.session_state.pop(_k, None)

            if _is_ev_sim:
                st.slider("Battery SoC %", 20, 100,
                          value=st.session_state.get("sim_SoC", int(_base["SoC"])),
                          key="sim_SoC")
            else:
                st.slider("Engine RPM", 600, 5000,
                          value=st.session_state.get("sim_Engine_RPM", int(_base["Engine_RPM"])),
                          step=100, key="sim_Engine_RPM")

            st.slider("Ambient Temperature °C", -10, 45,
                      value=st.session_state.get("sim_Ambient_Temperature",
                                                  int(_base["Ambient_Temperature"])),
                      key="sim_Ambient_Temperature")

            st.slider("Driving Speed km/h", 30, 130,
                      value=st.session_state.get("sim_Driving_Speed",
                                                  int(_base["Driving_Speed"])),
                      key="sim_Driving_Speed")

            st.slider("Total Distance km", 0, 150000,
                      value=st.session_state.get("sim_Total_Distance",
                                                  int(_base["Total_Distance"])),
                      step=500, key="sim_Total_Distance")

            st.slider("Harsh Braking Events", 0, 200,
                      value=st.session_state.get("sim_Harsh_Braking_Events",
                                                  int(_base["Harsh_Braking_Events"])),
                      key="sim_Harsh_Braking_Events")

            st.selectbox("Driving Style",
                         options=[0, 1, 2],
                         format_func=lambda x: {0: "Conservative", 1: "Normal", 2: "Aggressive"}[x],
                         index=st.session_state.get("sim_Driving_Style",
                                                     int(_base["Driving_Style"])),
                         key="sim_Driving_Style")

            if _is_ev_sim:
                st.slider("Fast Charge %", 0, 100,
                          value=st.session_state.get("sim_Fast_Charge_Percentage",
                                                      int(_base["Fast_Charge_Percentage"])),
                          key="sim_Fast_Charge_Percentage")
            else:
                st.slider("Km Since Last Oil Change", 0, 15000,
                          value=st.session_state.get("sim_Distance_Since_Last_Change",
                                                      int(_base["Distance_Since_Last_Change"])),
                          step=250, key="sim_Distance_Since_Last_Change")

            st.slider("Air Quality Index", 20, 150,
                      value=st.session_state.get("sim_Air_Quality_Index",
                                                  int(_base["Air_Quality_Index"])),
                      key="sim_Air_Quality_Index")
    
    # Initialize session state
    if "selected_vehicle" not in st.session_state:
        st.session_state.selected_vehicle = None
    
    # Main dashboard
    if st.session_state.selected_vehicle is None:
        st.title(" BeyondTech - Vehicle Fleet Dashboard")
        st.markdown("### Monitor and predict maintenance needs across your entire fleet")
        st.markdown("---")
        
        # Fleet overview
        cols = st.columns(min(len(vehicles_data), 5))
        for idx, (vin, data) in enumerate(vehicles_data.items()):
            with cols[idx % 5]:
                predictions = make_predictions(models, data)
                health_score = calculate_health_score(predictions, data['type'])
                
                status_color = '#00C853' if health_score > 80 else '#FFB366' if health_score > 60 else '#FF4444'
                
                st.markdown(f"""
                <div class="vehicle-card">
                    <h4 style="margin:0;">{data['name']}</h4>
                    <p style="font-size:11px; color:#888; margin:5px 0;">{vin[:18]}...</p>
                    <div style="display:flex; justify-content:space-between; margin-top:10px;">
                        <div>
                            <p style="margin:0; font-size:12px; color:#888;">Health</p>
                            <p style="margin:0; font-size:24px; font-weight:bold; color:{status_color};">{health_score}</p>
                        </div>
                        <div style="text-align:right;">
                            <p style="margin:0; font-size:12px; color:#888;">Type</p>
                            <p style="margin:0; font-size:16px; font-weight:bold; color:#00C853;">{data['type']}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"View Details", key=f"btn_{vin}"):
                    st.session_state.selected_vehicle = vin
                    st.rerun()
    
    else:
        # Vehicle detail view
        vehicle = st.session_state.selected_vehicle

        # Clear simulation state when switching to a different vehicle
        if st.session_state.get("_last_sim_vehicle") != vehicle:
            for _k in ["sim_SoC", "sim_Ambient_Temperature", "sim_Driving_Speed",
                       "sim_Total_Distance", "sim_Harsh_Braking_Events",
                       "sim_Driving_Style", "sim_Fast_Charge_Percentage",
                       "sim_Air_Quality_Index",
                       "sim_Engine_RPM", "sim_Distance_Since_Last_Change"]:
                st.session_state.pop(_k, None)
            st.session_state["_last_sim_vehicle"] = vehicle

        vehicle_data = vehicles_data[vehicle]

        # Build live_data: baseline from vehicle_data, overridden by simulator sliders
        _is_ev = vehicle_data['type'] == 'EV'
        _sim_keys = {
            "Ambient_Temperature":  "sim_Ambient_Temperature",
            "Driving_Speed":        "sim_Driving_Speed",
            "Average_Speed":        "sim_Driving_Speed",  # same slider drives both features
            "Total_Distance":       "sim_Total_Distance",
            "Harsh_Braking_Events": "sim_Harsh_Braking_Events",
            "Driving_Style":        "sim_Driving_Style",
            "Air_Quality_Index":    "sim_Air_Quality_Index",
        }
        if _is_ev:
            _sim_keys["SoC"]                    = "sim_SoC"
            _sim_keys["Fast_Charge_Percentage"] = "sim_Fast_Charge_Percentage"
        else:
            _sim_keys["Engine_RPM"]                  = "sim_Engine_RPM"
            _sim_keys["Distance_Since_Last_Change"]  = "sim_Distance_Since_Last_Change"
        _overrides = {
            dk: st.session_state[sk]
            for dk, sk in _sim_keys.items()
            if sk in st.session_state
        }
        live_data = {**vehicle_data, **_overrides}

        # ── Total Distance: derive all "since last service" fields + age-based fields ──
        if "Total_Distance" in _overrides:
            sim_dist = _overrides["Total_Distance"]
            KM_PER_MONTH = 1500  # assumed average driving rate

            # Service interval derivations (last service point is fixed history)
            _last_brake  = vehicle_data["Total_Distance"] - vehicle_data["Distance_Since_Last_Replacement"]
            live_data["Distance_Since_Last_Replacement"] = max(0, sim_dist - _last_brake)

            _last_filter = vehicle_data["Total_Distance"] - vehicle_data["Distance_Since_Filter_Change"]
            live_data["Distance_Since_Filter_Change"] = max(0, sim_dist - _last_filter)

            if vehicle_data.get("Distance_Since_Last_Change", 0) > 0:  # oil — ICE/Hybrid only
                _last_oil = vehicle_data["Total_Distance"] - vehicle_data["Distance_Since_Last_Change"]
                live_data["Distance_Since_Last_Change"] = max(0, sim_dist - _last_oil)

            if vehicle_data.get("Distance_Since_Fluid_Change", 0) > 0:  # transmission fluid
                _last_trans = vehicle_data["Total_Distance"] - vehicle_data["Distance_Since_Fluid_Change"]
                live_data["Distance_Since_Fluid_Change"] = max(0, sim_dist - _last_trans)

            # Age-based features scale with time-on-road
            months_delta = (sim_dist - vehicle_data["Total_Distance"]) / KM_PER_MONTH
            live_data["Tire_Age_Months"]    = max(1, int(vehicle_data["Tire_Age_Months"]    + months_delta))
            live_data["Battery_Age_Months"] = max(1, int(vehicle_data["Battery_Age_Months"] + months_delta))
            live_data["Coolant_Age_Months"] = max(1, int(vehicle_data["Coolant_Age_Months"] + months_delta))

            # Charge cycles accumulate with distance (avg ~300 km per full cycle for EVs)
            if vehicle_data.get("Total_Charge_Cycles", 0) > 0:
                live_data["Total_Charge_Cycles"] = max(1, int(sim_dist / 300))

        # ── Ambient Temperature: battery and engine temps track ambient ──
        if "Ambient_Temperature" in _overrides:
            sim_temp = _overrides["Ambient_Temperature"]
            # Battery thermal management keeps pack warmer in cold, cooler in heat
            if sim_temp < 0:
                batt_offset = 15   # heating system active
            elif sim_temp > 25:
                batt_offset = 5    # active cooling
            else:
                batt_offset = 8    # mild conditions
            live_data["Battery_Temperature"]     = sim_temp + batt_offset
            live_data["Battery_Temperature_Avg"] = sim_temp + batt_offset - 2
            # Wider temperature extremes = larger seasonal swing seen by tires
            live_data["Temperature_Range"] = max(10, int(abs(sim_temp - 20) * 1.5 + 10))
            # Engine temperature (ICE/Hybrid only — thermostat keeps it ~90°C but cold starts shift avg)
            if vehicle_data.get("Engine_Temperature", 0) > 0:
                live_data["Engine_Temperature"]     = max(60, 90 + (sim_temp - 15) * 0.2)
                live_data["Engine_Temperature_Avg"] = max(60, 88 + (sim_temp - 15) * 0.2)
                live_data["Engine_Temperature_Max"] = max(70, 98 + (sim_temp - 15) * 0.15)

        # ── Driving Speed: affects brake temps, high-speed %, city driving %, idle time ──
        if "Driving_Speed" in _overrides:
            sim_speed = _overrides["Driving_Speed"]
            base_speed = vehicle_data["Driving_Speed"]
            speed_delta = sim_speed - base_speed

            # High-speed % rises above 80 km/h
            live_data["High_Speed_Percentage"] = int(np.clip((sim_speed - 60) * 1.2, 0, 95))
            # City/urban driving is inversely proportional to speed
            live_data["City_Driving_Percentage"]  = int(np.clip(90 - (sim_speed - 30) * 0.75, 10, 90))
            live_data["Urban_Driving_Percentage"] = int(np.clip(95 - (sim_speed - 30) * 0.70, 10, 90))
            # Idle time is higher at lower speeds (city stop-and-go)
            live_data["Idle_Time_Percentage"] = int(np.clip(35 - (sim_speed - 30) * 0.35, 5, 35))
            # Brake temperature rises with higher-speed stops
            live_data["Brake_Temperature_Avg"] = max(
                60, vehicle_data["Brake_Temperature_Avg"] + speed_delta * 1.5
            )
            # Gear shifts increase at lower speeds (ICE/Hybrid with manual-style shifting)
            if vehicle_data.get("Gear_Shifts_Per_100km", 0) > 0:
                live_data["Gear_Shifts_Per_100km"] = int(
                    np.clip(vehicle_data["Gear_Shifts_Per_100km"] * (75 / max(sim_speed, 35)), 50, 400)
                )

        # ── Harsh Braking: brake events per 100 km and brake temperature both scale up ──
        if "Harsh_Braking_Events" in _overrides:
            sim_harsh = _overrides["Harsh_Braking_Events"]
            base_harsh = max(vehicle_data["Harsh_Braking_Events"], 1)
            ratio = sim_harsh / base_harsh
            live_data["Brake_Events_Per_100km"] = int(
                np.clip(vehicle_data["Brake_Events_Per_100km"] * ratio, 10, 350)
            )
            # Cumulative: adds on top of any speed-derived brake temp already set
            harsh_temp_delta = (sim_harsh - base_harsh) * 0.7
            live_data["Brake_Temperature_Avg"] = max(
                60, live_data["Brake_Temperature_Avg"] + harsh_temp_delta
            )

        # ── Driving Style: affects acceleration events, depth-of-discharge, high-speed % ──
        if "Driving_Style" in _overrides:
            sim_style = _overrides["Driving_Style"]
            accel_mult  = {0: 0.50, 1: 1.0, 2: 2.20}[sim_style]
            live_data["Harsh_Acceleration_Events"] = int(
                np.clip(vehicle_data["Harsh_Acceleration_Events"] * accel_mult, 3, 200)
            )
            # Aggressive drivers discharge battery deeper before charging
            discharge_shift = {0: -15, 1: 0, 2: +20}[sim_style]
            live_data["Average_Depth_of_Discharge"] = int(
                np.clip(vehicle_data["Average_Depth_of_Discharge"] + discharge_shift, 20, 95)
            )
            # Don't overwrite High_Speed_Percentage if Driving_Speed slider already set it
            if "Driving_Speed" not in _overrides:
                hs_shift = {0: -12, 1: 0, 2: +22}[sim_style]
                live_data["High_Speed_Percentage"] = int(
                    np.clip(vehicle_data["High_Speed_Percentage"] + hs_shift, 0, 90)
                )

        # ── Fast Charge %: battery temperature and temperature range increase ──
        if "Fast_Charge_Percentage" in _overrides:
            sim_fc   = _overrides["Fast_Charge_Percentage"]
            base_fc  = vehicle_data["Fast_Charge_Percentage"]
            fc_delta = sim_fc - base_fc
            # Fast charging generates heat; adds on top of any ambient-derived temp
            current_batt_avg = live_data.get("Battery_Temperature_Avg",
                                             vehicle_data["Battery_Temperature_Avg"])
            live_data["Battery_Temperature_Avg"] = round(
                np.clip(current_batt_avg + fc_delta * 0.15, 5, 50), 1
            )
            # Wider thermal swings from repeated fast charge/cool cycles
            live_data["Battery_Temperature_Range"] = round(
                np.clip(vehicle_data["Battery_Temperature_Range"] + fc_delta * 0.08, 5, 40), 1
            )

        # ── Air Quality Index: dusty road % and engine air flow degrade together ──
        if "Air_Quality_Index" in _overrides:
            sim_aqi = _overrides["Air_Quality_Index"]
            # Dusty road % correlates with particulate-heavy air
            if sim_aqi < 50:
                live_data["Dusty_Road_Percentage"] = max(2,  int(sim_aqi * 0.10))
            elif sim_aqi < 100:
                live_data["Dusty_Road_Percentage"] = int(5 + (sim_aqi - 50) * 0.30)
            else:
                live_data["Dusty_Road_Percentage"] = int(20 + (sim_aqi - 100) * 0.40)
            # Engine air flow falls as filter clogs faster in dirty air
            live_data["Engine_Air_Flow"] = int(np.clip(100 - (sim_aqi - 35) * 0.22, 50, 100))

        # ── SoC: battery voltage tracks state-of-charge via linear cell chemistry model ──
        if "SoC" in _overrides:
            sim_soc = _overrides["SoC"]
            # Lithium-ion open-circuit voltage: ~350V at 20% SoC, ~420V at 100% SoC
            live_data["Battery_Voltage"] = round(350 + (sim_soc / 100) * 70, 1)

        # ── Engine RPM (ICE only): raises engine temp and accelerates oil degradation ──
        if "Engine_RPM" in _overrides:
            sim_rpm = _overrides["Engine_RPM"]
            base_rpm = max(vehicle_data["Engine_RPM"], 1)
            rpm_ratio = sim_rpm / base_rpm
            # Higher RPM = hotter engine
            live_data["Engine_Temperature"]     = int(np.clip(vehicle_data["Engine_Temperature"] * (0.8 + rpm_ratio * 0.25), 60, 115))
            live_data["Engine_Temperature_Avg"] = int(np.clip(vehicle_data["Engine_Temperature_Avg"] * (0.8 + rpm_ratio * 0.25), 60, 110))
            live_data["Engine_Temperature_Max"] = int(np.clip(vehicle_data["Engine_Temperature_Max"] * (0.85 + rpm_ratio * 0.20), 70, 120))

        # ── Distance Since Last Oil Change (ICE only) ──
        if "Distance_Since_Last_Change" in _overrides:
            live_data["Distance_Since_Last_Change"] = _overrides["Distance_Since_Last_Change"]

        _sim_active = any(live_data[dk] != vehicle_data[dk] for dk in live_data if dk in vehicle_data)

        # Back button
        if st.button("← Back to Fleet"):
            st.session_state.selected_vehicle = None
            st.rerun()
        
        st.title(f"🚗 {vehicle_data['name']}")
        st.markdown(f"**VIN:** {vehicle} | **Type:** {vehicle_data['type']}")
        if _sim_active:
            st.markdown(
                '<span style="background:#FF6F00; color:white; padding:4px 14px; '
                'border-radius:20px; font-size:13px; font-weight:700; letter-spacing:1px;">'
                'LIVE SIMULATION</span>',
                unsafe_allow_html=True
            )

        # Make predictions
        predictions = make_predictions(models, live_data)
        health_score = calculate_health_score(predictions, vehicle_data['type'])
        
        # Health metrics — single figure so zoom never breaks alignment
        st.markdown("---")
        if vehicle_data['type'] == 'EV':
            second_val   = predictions.get('battery_soh', vehicle_data.get('SoH', 90))
            second_label = "Battery Health"
        else:
            # Engine health: how much oil life remains as a % of full interval (10 500 km)
            second_val   = min(100.0, (predictions.get('oil_life', 5000) / 10500) * 100)
            second_label = "Engine Health"
        brake_health = min((predictions.get('brake_thickness', 10) / 12) * 100, 100)
        tire_health  = min((predictions.get('tire_tread', 6) / 8) * 100, 100)
        st.plotly_chart(
            create_gauges_row(health_score, second_val, brake_health, tire_health, second_label),
            use_container_width=True,
        )
        
        # Key predictions
        st.markdown("---")
        st.markdown("### 📊 Predictive Insights")
        
        cols = st.columns(3)
        
        if vehicle_data['type'] == 'EV' and 'ev_range' in predictions:
            with cols[0]:
                range_val = predictions['ev_range']
                range_status, range_color = get_alert_level(range_val, {'critical': 100, 'warning': 150})
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {range_color};">
                    <h4>🔋 Battery Range</h4>
                    <p style="font-size:36px; font-weight:bold; color:{range_color}; margin:5px 0;">
                        {range_val:.0f} km
                    </p>
                    <p style="color:#888;">Status: <span style="color:{range_color};">{range_status}</span></p>
                </div>
                """, unsafe_allow_html=True)

        if vehicle_data['type'] != 'EV' and 'oil_life' in predictions:
            with cols[0]:
                oil_val0 = predictions['oil_life']
                oil_s0, oil_c0 = get_alert_level(oil_val0, {'critical': 500, 'warning': 1500})
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {oil_c0};">
                    <h4>🛢️ Oil Life</h4>
                    <p style="font-size:36px; font-weight:bold; color:{oil_c0}; margin:5px 0;">
                        {oil_val0:.0f} km
                    </p>
                    <p style="color:#888;">Status: <span style="color:{oil_c0};">{oil_s0}</span></p>
                </div>
                """, unsafe_allow_html=True)

        if vehicle_data['type'] == 'EV' and 'battery_soh' in predictions:
            with cols[1]:
                soh_val = predictions['battery_soh']
                soh_status, soh_color = get_alert_level(soh_val, {'critical': 70, 'warning': 80})
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {soh_color};">
                    <h4>⚡ Battery Health</h4>
                    <p style="font-size:36px; font-weight:bold; color:{soh_color}; margin:5px 0;">
                        {soh_val:.1f}%
                    </p>
                    <p style="color:#888;">Status: <span style="color:{soh_color};">{soh_status}</span></p>
                </div>
                """, unsafe_allow_html=True)

        if vehicle_data['type'] != 'EV' and 'transmission_life' in predictions:
            with cols[1]:
                trans_val = predictions['transmission_life']
                trans_status, trans_color = get_alert_level(trans_val, {'critical': 2000, 'warning': 10000})
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {trans_color};">
                    <h4>⚙️ Transmission Fluid</h4>
                    <p style="font-size:36px; font-weight:bold; color:{trans_color}; margin:5px 0;">
                        {trans_val:.0f} km
                    </p>
                    <p style="color:#888;">Status: <span style="color:{trans_color};">{trans_status}</span></p>
                </div>
                """, unsafe_allow_html=True)
        
        if 'tire_tread' in predictions:
            with cols[2]:
                tire_val = predictions['tire_tread']
                tire_status = 'Critical' if tire_val < 2 else 'Warning' if tire_val < 3 else 'Good'
                tire_color = '#FF4444' if tire_val < 2 else '#FFB366' if tire_val < 3 else '#00C853'
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {tire_color};">
                    <h4>🛞 Tire Tread</h4>
                    <p style="font-size:36px; font-weight:bold; color:{tire_color}; margin:5px 0;">
                        {tire_val:.1f} mm
                    </p>
                    <p style="color:#888;">Status: <span style="color:{tire_color};">{tire_status}</span></p>
                </div>
                """, unsafe_allow_html=True)
        
        # Trends — tabbed per component
        st.markdown("---")
        st.markdown("### 📈 90-Day Component Health Forecast")

        _tab_defs = [
            ("🛞 Tires",       "tire_tread"),
            ("🔧 Brakes",      "brake_thickness"),
            ("🔋 Battery",     "battery_soh"),
            ("💨 Air Filter",  "air_filter_life"),
            ("💧 Coolant",     "coolant_life"),
            ("🛢️ Oil",         "oil_life"),
            ("⚙️ Transmission","transmission_life"),
        ]
        _available = [(label, key) for label, key in _tab_defs if key in predictions]

        # Critical thresholds — used to decide alert-vs-chart for each component
        _crit_specs = {
            "tire_tread":        dict(crit=2.0,    warn=3.0,    unit="mm", fmt=".1f", label="Tire Tread",         action="Replace tyres immediately"),
            "brake_thickness":   dict(crit=3.0,    warn=4.0,    unit="mm", fmt=".1f", label="Brake Pad",          action="Replace brake pads immediately"),
            "battery_soh":       dict(crit=70.0,   warn=80.0,   unit="%",  fmt=".1f", label="Battery SoH",        action="Schedule battery assessment"),
            "oil_life":          dict(crit=500.0,  warn=1500.0, unit="km", fmt=".0f", label="Oil Life",           action="Change oil immediately"),
            "air_filter_life":   dict(crit=500.0,  warn=2000.0, unit="km", fmt=".0f", label="Air Filter",         action="Replace air filter immediately"),
            "coolant_life":      dict(crit=1000.0, warn=5000.0, unit="km", fmt=".0f", label="Coolant",            action="Flush and replace coolant"),
            "transmission_life": dict(crit=2000.0, warn=10000.0,unit="km", fmt=".0f", label="Transmission Fluid", action="Change transmission fluid immediately"),
        }

        if _available:
            _tabs = st.tabs([label for label, _ in _available])
            for _tab, (_, _key) in zip(_tabs, _available):
                with _tab:
                    _cur_val = float(predictions.get(_key, 9999))
                    _cs      = _crit_specs.get(_key, {})
                    _is_crit = _cs and _cur_val <= _cs["crit"]
                    _is_warn = _cs and not _is_crit and _cur_val <= _cs["warn"]

                    if _is_crit:
                        # ── Already past critical threshold — show alert card ──
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, rgba(180,0,0,0.25) 0%, rgba(80,0,0,0.40) 100%);
                            border: 2px solid #FF4444;
                            border-radius: 16px;
                            padding: 48px 32px;
                            text-align: center;
                            margin: 16px 0;
                        ">
                            <div style="font-size: 60px; margin-bottom: 12px;">🚨</div>
                            <h2 style="color:#FF4444; margin:0 0 8px 0; font-size:26px; letter-spacing:1px;">
                                IMMEDIATE SERVICE REQUIRED
                            </h2>
                            <p style="color:#ffaaaa; font-size:15px; margin:0 0 32px 0;">
                                <b>{_cs['label']}</b> has already crossed the critical safety threshold
                            </p>
                            <div style="display:flex; justify-content:center; gap:32px; flex-wrap:wrap; margin-bottom:32px;">
                                <div style="background:rgba(0,0,0,0.35); border-radius:12px; padding:18px 28px; min-width:140px;">
                                    <p style="color:#888; margin:0 0 6px 0; font-size:11px; letter-spacing:1px;">CURRENT VALUE</p>
                                    <p style="color:#FF4444; margin:0; font-size:34px; font-weight:700; line-height:1;">
                                        {_cur_val:{_cs['fmt']}} {_cs['unit']}
                                    </p>
                                </div>
                                <div style="background:rgba(0,0,0,0.35); border-radius:12px; padding:18px 28px; min-width:140px;">
                                    <p style="color:#888; margin:0 0 6px 0; font-size:11px; letter-spacing:1px;">CRITICAL THRESHOLD</p>
                                    <p style="color:#FFB366; margin:0; font-size:34px; font-weight:700; line-height:1;">
                                        {_cs['crit']:{_cs['fmt']}} {_cs['unit']}
                                    </p>
                                </div>
                            </div>
                            <div style="background:rgba(255,68,68,0.15); border:1px solid rgba(255,68,68,0.4);
                                        border-radius:10px; padding:14px 24px; display:inline-block;">
                                <p style="color:#FF6666; margin:0; font-size:14px; font-weight:600;">
                                    → {_cs['action']}
                                </p>
                            </div>
                            <p style="color:#555; margin:28px 0 0 0; font-size:12px;">
                                Degradation forecast is not available — component already requires immediate attention.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    elif _is_warn:
                        # ── Past warning but not yet critical — show amber banner + chart ──
                        st.markdown(f"""
                        <div style="
                            background: rgba(255,179,102,0.12);
                            border: 1px solid rgba(255,179,102,0.5);
                            border-radius: 10px;
                            padding: 14px 20px;
                            margin-bottom: 12px;
                            display: flex;
                            align-items: center;
                            gap: 14px;
                        ">
                            <span style="font-size:24px;">⚠️</span>
                            <div>
                                <b style="color:#FFB366;">{_cs['label']} is in the warning zone</b>
                                <span style="color:#ccc; font-size:13px; margin-left:12px;">
                                    Current: {_cur_val:{_cs['fmt']}} {_cs['unit']} &nbsp;|&nbsp;
                                    Warning threshold: {_cs['warn']:{_cs['fmt']}} {_cs['unit']}
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        _ctl1, _ctl2, _ = st.columns([3, 3, 8])
                        with _ctl1:
                            _xdays = st.radio("Time Range", [30, 60, 90], index=2, horizontal=True, key=f"xrange_{_key}")
                        with _ctl2:
                            _yscale = st.radio("Y Scale", ["Auto", "Full"], index=0, horizontal=True, key=f"yscale_{_key}")
                        st.plotly_chart(
                            create_component_forecast(_key, predictions, live_data, days=_xdays, auto_y=(_yscale == "Auto")),
                            use_container_width=True,
                        )

                    else:
                        # ── Healthy — show chart as normal ──
                        _ctl1, _ctl2, _ = st.columns([3, 3, 8])
                        with _ctl1:
                            _xdays = st.radio("Time Range", [30, 60, 90], index=2, horizontal=True, key=f"xrange_{_key}")
                        with _ctl2:
                            _yscale = st.radio("Y Scale", ["Auto", "Full"], index=0, horizontal=True, key=f"yscale_{_key}")
                        st.plotly_chart(
                            create_component_forecast(_key, predictions, live_data, days=_xdays, auto_y=(_yscale == "Auto")),
                            use_container_width=True,
                        )
        
        # Recommendations
        st.markdown("---")
        st.markdown("### 🔔 Maintenance Recommendations")
        
        recommendations = generate_recommendations(predictions, vehicle_data['type'])
        
        for rec in recommendations:
            priority_color = '#FF4444' if rec['priority'] == 'Critical' else '#FFB366' if rec['priority'] == 'High' else '#FFA726' if rec['priority'] == 'Medium' else '#00C853'
            
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {priority_color};">
                <div style="display:flex; justify-content:space-between; align-items:start;">
                    <div style="flex:1;">
                        <h4 style="margin:0;">{rec['icon']} {rec['title']}</h4>
                        <p style="margin:5px 0; color:#ccc;">{rec['message']}</p>
                        <p style="margin:5px 0; font-weight:600; color:{priority_color};">→ {rec['action']}</p>
                    </div>
                    <span style="background:{priority_color}; color:white; padding:5px 15px; border-radius:20px; font-size:12px; font-weight:bold;">
                        {rec['priority']}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Cost analysis
        display_maintenance_costs(predictions, vehicle_data['type'], region=st.session_state.get("region", "US"))

        # Parts & Workshop Marketplace (shown only when a component is Critical)
        render_marketplace_panel(recommendations, vehicle_data)

if __name__ == "__main__":
    main()