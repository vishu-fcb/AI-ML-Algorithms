import streamlit as st
import pandas as pd
import pickle

# Load models from pickle files
@st.cache_resource
def load_models():
    with open("Trained Model/battery_model.pkl", "rb") as f:
        model_ev = pickle.load(f)
    with open("Trained Model/oil_model.pkl", "rb") as f:
        model_oil = pickle.load(f)
    return model_ev, model_oil

# Hardcoded vehicle data inputs
vehicles_data = {
    "VIN 1WTSTUXX001111111": {
        "SoC": 85, "SoH": 90, "Battery_Voltage": 400, "Battery_Temperature": 25,
        "Driving_Speed": 60, "Load_Weight": 1500, "Ambient_Temperature": 20,
        "Engine_Temperature": 90, "Engine_RPM": 3000, "Distance_Since_Last_Change": 8000,
        "Oil_Viscosity": 5, "Idle_Time": 15
    },
    "VIN 1WTSTUXX001111112": {
        "SoC": 50, "SoH": 85, "Battery_Voltage": 390, "Battery_Temperature": 30,
        "Driving_Speed": 80, "Load_Weight": 1400, "Ambient_Temperature": 18,
        "Engine_Temperature": 95, "Engine_RPM": 3200, "Distance_Since_Last_Change": 10000,
        "Oil_Viscosity": 4.5, "Idle_Time": 20
    },
    "VIN 1WTSTUXX001111113": {
        "SoC": 65, "SoH": 88, "Battery_Voltage": 395, "Battery_Temperature": 22,
        "Driving_Speed": 70, "Load_Weight": 1600, "Ambient_Temperature": 21,
        "Engine_Temperature": 88, "Engine_RPM": 3100, "Distance_Since_Last_Change": 6000,
        "Oil_Viscosity": 5, "Idle_Time": 10
    },
    "VIN 1WTSTUXX001111114": {
        "SoC": 90, "SoH": 92, "Battery_Voltage": 410, "Battery_Temperature": 24,
        "Driving_Speed": 50, "Load_Weight": 1450, "Ambient_Temperature": 19,
        "Engine_Temperature": 85, "Engine_RPM": 2800, "Distance_Since_Last_Change": 9000,
        "Oil_Viscosity": 5.2, "Idle_Time": 18
    },
    "VIN 1WTSTUXX00111111": {
        "SoC": 40, "SoH": 80, "Battery_Voltage": 380, "Battery_Temperature": 28,
        "Driving_Speed": 75, "Load_Weight": 1550, "Ambient_Temperature": 22,
        "Engine_Temperature": 100, "Engine_RPM": 3300, "Distance_Since_Last_Change": 11000,
        "Oil_Viscosity": 4.8, "Idle_Time": 25
    },
}

def main():
    st.set_page_config(page_title="Vehicle Predictive Maintenance", layout="wide", page_icon="🚗")

    st.markdown(
        """
        <style>
        .main {
            background-color: #121212;
            color: white;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .stButton>button {
            background-color: #007ACC;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 18px;
            margin: 10px;
        }
        .stButton>button:hover {
            background-color: #005A9E;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🚗 Vehicle Predictive Maintenance Dashboard")

    model_ev, model_oil = load_models()

    if "selected_vehicle" not in st.session_state:
        st.session_state.selected_vehicle = None

    if st.session_state.selected_vehicle is None:
        st.subheader("Select a Vehicle to View Predictions")
        cols = st.columns(len(vehicles_data))
        for idx, vehicle in enumerate(vehicles_data.keys()):
            with cols[idx]:
                if st.button(vehicle):
                    st.session_state.selected_vehicle = vehicle
                    return

    else:
        vehicle = st.session_state.selected_vehicle
        st.markdown(f"### Details and Predictions for **{vehicle}**")

        vehicle_data = vehicles_data[vehicle]

        ev_features = ['SoC', 'SoH', 'Battery_Voltage', 'Battery_Temperature', 'Driving_Speed', 'Load_Weight', 'Ambient_Temperature']
        oil_features = ['Engine_Temperature', 'Engine_RPM', 'Load_Weight', 'Distance_Since_Last_Change', 'Oil_Viscosity', 'Ambient_Temperature', 'Idle_Time']

        X_ev = pd.DataFrame([{k: vehicle_data[k] for k in ev_features}])
        X_oil = pd.DataFrame([{k: vehicle_data[k] for k in oil_features}])

        pred_range = model_ev.predict(X_ev)[0]
        pred_oil_change = model_oil.predict(X_oil)[0]

        col1, col2 = st.columns([2, 3])

        with col1:
            st.write("#### Input Parameters")
            for k, v in vehicle_data.items():
                st.write(f"**{k}**: {v}")

            if st.button("← Back to Vehicle Selection"):
                st.session_state.selected_vehicle = None
                return

        with col2:
            st.write("#### Predictions")

            st.markdown(f"""
            <div style="background-color:#222; padding:20px; border-radius:10px; text-align:center;">
                <h3 style="color:#00aaff; margin-bottom:5px;">Battery Range (km)</h3>
                <p style="font-size:48px; font-weight:bold; color:#00aaff; margin-top:0;">{pred_range:.1f} km</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background-color:#222; padding:20px; border-radius:10px; text-align:center; margin-top:20px;">
                <h3 style="color:#ffaa00; margin-bottom:5px;">Oil Change Remaining (km)</h3>
                <p style="font-size:48px; font-weight:bold; color:#ffaa00; margin-top:0;">{pred_oil_change:.1f} km</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
