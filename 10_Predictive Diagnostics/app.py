"""
BeyondTech - Predictive Maintenance Dashboard
Complete Streamlit application with all predictive maintenance features
"""

import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import os
from maintenance_cost_calculator import MaintenanceCostCalculator
from vehicle_data import vehicles_data, feature_mappings

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

def create_trend_chart(vehicle_name):
    """Create historical trend chart (simulated data)"""
    from datetime import datetime, timedelta
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    np.random.seed(hash(vehicle_name) % 2**32)
    battery_range = 250 + np.cumsum(np.random.randn(30) * 5)
    health_score = 85 + np.cumsum(np.random.randn(30) * 0.5)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=dates, y=battery_range,
            mode='lines+markers',
            name='Battery Range (km)',
            line=dict(color='#00C853', width=2),
            marker=dict(size=5)
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates, y=health_score,
            mode='lines+markers',
            name='Health Score',
            line=dict(color='#FFB366', width=2),
            marker=dict(size=5)
        ),
        secondary_y=True
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(
        title_text="Battery Range (km)",
        gridcolor='rgba(255,255,255,0.1)',
        title_font=dict(color='#00C853'),
        secondary_y=False
    )
    fig.update_yaxes(
        title_text="Health Score",
        title_font=dict(color='#FFB366'),
        secondary_y=True
    )
    
    fig.update_layout(
        title='30-Day Trends',
        paper_bgcolor='rgba(30,30,30,0.8)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        hovermode='x unified',
        height=280,
        margin=dict(l=10, r=10, t=40, b=10)
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
    """Make all predictions for a vehicle"""
    predictions = {}
    
    for model_name, features in feature_mappings.items():
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

def display_maintenance_costs(predictions, vehicle_type):
    """Display comprehensive maintenance cost analysis"""
    calculator = MaintenanceCostCalculator(vehicle_type=vehicle_type, region="US")
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

def main():
    """Main application"""
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🚗 BeyondTech")
        st.markdown("**Predictive Maintenance Platform**")
        st.markdown("---")
        
        if st.button("🏠 Dashboard Home"):
            st.session_state.selected_vehicle = None
        
        st.markdown("---")
        st.markdown("**Quick Stats**")
        st.metric("Total Vehicles", len(vehicles_data))
        st.metric("Active Alerts", "5")
        st.metric("Avg Health Score", "86.3")
        
        st.markdown("---")
        st.markdown("**About**")
        st.info("BeyondTech uses machine learning to predict maintenance needs before failures occur.")
    
    # Load models
    models = load_models()
    
    if not models:
        st.error("⚠️ No models found! Please train models first by running: `python AI_prediction_model.py`")
        return
    
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
        vehicle_data = vehicles_data[vehicle]
        
        # Back button
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("← Back to Fleet"):
                st.session_state.selected_vehicle = None
                st.rerun()
        
        st.title(f"🚗 {vehicle_data['name']}")
        st.markdown(f"**VIN:** {vehicle} | **Type:** {vehicle_data['type']}")
        
        # Make predictions
        predictions = make_predictions(models, vehicle_data)
        health_score = calculate_health_score(predictions, vehicle_data['type'])
        
        # Health metrics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.plotly_chart(create_gauge_chart(health_score, "Health Score"), use_container_width=True)
        
        with col2:
            batt_health = predictions.get('battery_soh', vehicle_data.get('SoH', 90))
            st.plotly_chart(create_gauge_chart(batt_health, "Battery Health"), use_container_width=True)
        
        with col3:
            brake_health = min((predictions.get('brake_thickness', 10) / 12) * 100, 100)
            st.plotly_chart(create_gauge_chart(brake_health, "Brake Health"), use_container_width=True)
        
        with col4:
            tire_health = min((predictions.get('tire_tread', 6) / 8) * 100, 100)
            st.plotly_chart(create_gauge_chart(tire_health, "Tire Health"), use_container_width=True)
        
        # Key predictions
        st.markdown("---")
        st.markdown("### 📊 Predictive Insights")
        
        cols = st.columns(3)
        
        if 'ev_range' in predictions:
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
        
        if 'oil_life' in predictions and vehicle_data['type'] != "EV":
            with cols[1]:
                oil_val = predictions['oil_life']
                oil_status, oil_color = get_alert_level(oil_val, {'critical': 500, 'warning': 1500})
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {oil_color};">
                    <h4>🛢️ Oil Life</h4>
                    <p style="font-size:36px; font-weight:bold; color:{oil_color}; margin:5px 0;">
                        {oil_val:.0f} km
                    </p>
                    <p style="color:#888;">Status: <span style="color:{oil_color};">{oil_status}</span></p>
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
        
        # Trends
        st.markdown("---")
        st.plotly_chart(create_trend_chart(vehicle_data['name']), use_container_width=True)
        
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
        display_maintenance_costs(predictions, vehicle_data['type'])

if __name__ == "__main__":
    main()