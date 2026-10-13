"""
VED (Vehicle Energy Dataset) Processor  EV Range Model Retraining
==================================================================
Transforms real OBD-II telemetry from 383 vehicles (University of Michigan)
into a training dataset for the EV Range prediction model.

Dataset:
  Paper : "Vehicle Energy Dataset (VED), A Large-scale Dataset for Vehicle
           Energy Consumption Research", IEEE Trans. ITS, 2020
  Repo  : https://github.com/gsoh/VED

Verified VED column names (actual dataset)
------------------------------------------
  Dynamic CSVs : DayNum, VehId, Trip, Timestamp(ms), Latitude[deg],
                 Longitude[deg], Vehicle Speed[km/h], OAT[DegC],
                 HV Battery Current[A], HV Battery SOC[%],
                 HV Battery Voltage[V], Heater Power[Watts],
                 Air Conditioning Power[Watts]
  Static ICE/HEV xlsx : VehId, Vehicle Type, Generalized_Weight
  Static PHEV/EV xlsx : VehId, EngineType,   Generalized_Weight

How to set up (if running fresh)
---------------------------------
  1. VED_Data/VED_Static_ICE_HEV.xlsx    already downloaded
  2. VED_Data/VED_Static_PHEV_EV.xlsx    already downloaded
  3. VED_Data/dynamic/*.csv              already extracted

Output
------
  Augmented_Datasets/ev_range_augmented.csv
  Auto-loaded by AI_prediction_model.py when present.
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

warnings.filterwarnings("ignore")

#  Configuration 

VED_DIR         = "VED_Data"
DYNAMIC_DIR     = os.path.join(VED_DIR, "dynamic")
STATIC_ICE_HEV  = os.path.join(VED_DIR, "VED_Static_ICE_HEV.xlsx")
STATIC_PHEV_EV  = os.path.join(VED_DIR, "VED_Static_PHEV_EV.xlsx")
OUTPUT_FILE     = "Augmented_Datasets/ev_range_augmented.csv"

# Engine type labels that have meaningful HV Battery signals
EV_TYPES = {"HEV", "PHEV", "EV", "BEV", "HYBRID"}

# Reference battery capacity to scale range into full-EV territory (Wh)
# VED HEVs have small packs (6-15 kWh); we scale output to a 60 kWh EV
REFERENCE_BATTERY_WH = 60_000

MIN_SAMPLES = 3_000
np.random.seed(42)


#  Helpers 

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi    = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def trip_distance_km(lats, lons):
    if len(lats) < 2:
        return 0.0
    return sum(
        haversine_km(lats[i], lons[i], lats[i + 1], lons[i + 1])
        for i in range(len(lats) - 1)
    )


#  Data Loading 

def load_static_data():
    """
    Merge both VED static Excel files into one DataFrame indexed by VehId.
    Returns columns: weight_kg, engine_type
    """
    dfs = []

    # ICE & HEV file  column is 'Vehicle Type'
    if os.path.exists(STATIC_ICE_HEV):
        df = pd.read_excel(STATIC_ICE_HEV)
        df = df.rename(columns={"Vehicle Type": "EngineType"})
        df["EngineType"] = df["EngineType"].str.strip().str.upper()
        dfs.append(df[["VehId", "EngineType", "Generalized_Weight"]])
    else:
        print(f"     {STATIC_ICE_HEV} not found  skipping ICE/HEV vehicles")

    # PHEV & EV file  column is 'EngineType'
    if os.path.exists(STATIC_PHEV_EV):
        df = pd.read_excel(STATIC_PHEV_EV)
        df["EngineType"] = df["EngineType"].str.strip().str.upper()
        dfs.append(df[["VehId", "EngineType", "Generalized_Weight"]])
    else:
        print(f"     {STATIC_PHEV_EV} not found  skipping PHEV/EV vehicles")

    if not dfs:
        raise FileNotFoundError("No static data files found in VED_Data/")

    static = pd.concat(dfs, ignore_index=True).drop_duplicates("VehId")
    static["weight_kg"] = pd.to_numeric(static["Generalized_Weight"], errors="coerce").fillna(3500) * 0.453592   # lb to kg
    static = static.set_index("VehId")

    ev_count  = static[static["EngineType"].isin(EV_TYPES)].shape[0]
    print(f"   Vehicles loaded: {len(static)} total | {ev_count} HEV/PHEV/EV")
    return static


def load_dynamic_files():
    """Load and concatenate all dynamic CSV files from DYNAMIC_DIR."""
    files = sorted(glob.glob(os.path.join(DYNAMIC_DIR, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {DYNAMIC_DIR}")

    print(f"   Found {len(files)} weekly CSV files  loading...")
    chunks = []
    for f in files:
        try:
            chunk = pd.read_csv(f, low_memory=False)
            chunks.append(chunk)
        except Exception as e:
            print(f"     Skipped {os.path.basename(f)}: {e}")

    df = pd.concat(chunks, ignore_index=True)
    print(f"   Total raw rows: {len(df):,}")
    return df


#  Trip Processing 

def process_trips(dynamic_df, static_df):
    """
    Aggregate raw OBD-II time-series into one feature vector per trip.

    Real signals used directly:
      HV Battery SOC[%]      SoC
      HV Battery Voltage[V]  Battery_Voltage
      Vehicle Speed[km/h]    Driving_Speed
      OAT[DegC]              Ambient_Temperature
      Generalized_Weight     Load_Weight (converted to kg)

    Derived:
      Battery_Temperature    Ambient + thermal model (heater/AC/speed)
      SoH                    Aging proxy from cumulative trip count
      Range_Left_km          SoC  (60 kWh / real Wh/km efficiency)
    """

    # Filter to EV-type vehicles only
    ev_ids = static_df[static_df["EngineType"].isin(EV_TYPES)].index
    df = dynamic_df[dynamic_df["VehId"].isin(ev_ids)].copy()
    print(f"   HEV/PHEV/EV rows: {len(df):,} across {df['VehId'].nunique()} vehicles")

    # Drop rows with no battery SOC or voltage
    df = df.dropna(subset=["HV Battery SOC[%]", "HV Battery Voltage[V]"])
    print(f"   After HV battery filter: {len(df):,} rows remaining")

    if df.empty:
        return pd.DataFrame()

    # Pre-compute per-vehicle trip counts for SoH aging proxy
    trips_per_veh = df.groupby("VehId")["Trip"].nunique()
    max_trips     = max(trips_per_veh.max(), 1)

    records = []
    skipped = 0
    grouped = df.groupby(["VehId", "Trip"])
    total   = len(grouped)

    print(f"\n   Processing {total:,} (VehId, Trip) groups...")

    for idx, ((veh_id, trip_id), grp) in enumerate(grouped):

        if idx % 1000 == 0 and idx > 0:
            print(f"   ... {idx:,}/{total:,} | valid: {len(records):,}")

        grp = grp.sort_values("Timestamp(ms)")

        soc_vals  = grp["HV Battery SOC[%]"].dropna().values.astype(float)
        volt_vals = grp["HV Battery Voltage[V]"].dropna().values.astype(float)

        if len(soc_vals) < 10:
            skipped += 1
            continue

        soc_start    = soc_vals[0]
        soc_end      = soc_vals[-1]
        soc_mean     = float(np.mean(soc_vals))
        soc_consumed = soc_start - soc_end   # positive when discharging

        if soc_consumed < 1.0:
            skipped += 1
            continue

        # GPS trip distance
        dist_km = 0.0
        lat_vals = grp["Latitude[deg]"].dropna().values.astype(float)
        lon_vals = grp["Longitude[deg]"].dropna().values.astype(float)
        if len(lat_vals) >= 2:
            dist_km = trip_distance_km(lat_vals.tolist(), lon_vals.tolist())

        if dist_km < 0.5:
            skipped += 1
            continue

        # Real energy consumed (Wh) from Voltage  |Current| integrated over time
        wh_consumed = None
        curr_col = "HV Battery Current[A]"
        if curr_col in grp.columns:
            volt_arr = grp["HV Battery Voltage[V]"].values.astype(float)
            curr_arr = grp[curr_col].values.astype(float)
            time_arr = grp["Timestamp(ms)"].values.astype(float)

            # Negative current = discharging; positive = charging/regen
            power_w  = volt_arr * (-curr_arr)          # positive when discharging
            dt_hours = np.diff(time_arr) / (1000.0 * 3600.0)
            avg_pwr  = (power_w[:-1] + power_w[1:]) / 2.0
            wh_consumed = float(np.sum(avg_pwr * dt_hours))

        # Efficiency (Wh/km)  real measured or SoC-based fallback
        if wh_consumed and wh_consumed > 10:
            eff = wh_consumed / dist_km
        else:
            eff = (soc_consumed / 100.0 * REFERENCE_BATTERY_WH) / dist_km

        eff = float(np.clip(eff, 100.0, 600.0))

        # Range remaining at mean SoC with this real efficiency
        range_left_km = float(np.clip(
            (soc_mean / 100.0) * REFERENCE_BATTERY_WH / eff,
            10.0, 500.0
        ))

        # Feature aggregates from real signals
        avg_volt  = float(np.mean(volt_vals))
        avg_speed = float(grp["Vehicle Speed[km/h]"].mean()) \
                    if "Vehicle Speed[km/h]" in grp.columns else 60.0
        avg_temp  = float(grp["OAT[DegC]"].mean()) \
                    if "OAT[DegC]" in grp.columns else 15.0
        avg_speed = float(np.clip(avg_speed, 0.0, 150.0))
        avg_temp  = float(np.clip(avg_temp, -30.0, 50.0))

        # Battery temperature: ambient + internal heating model
        heat_w = float(grp["Heater Power[Watts]"].mean()) \
                 if "Heater Power[Watts]" in grp.columns else 0.0
        ac_w   = float(grp["Air Conditioning Power[Watts]"].mean()) \
                 if "Air Conditioning Power[Watts]" in grp.columns else 0.0
        bat_temp = avg_temp + 10.0 \
                   + (avg_speed / 130.0) * 12.0 \
                   + (heat_w / 1000.0) * 2.0 \
                   - (ac_w / 600.0)
        bat_temp = float(np.clip(bat_temp, -10.0, 60.0))

        # SoH: aging proxy from cumulative vehicle trips
        total_trips = int(trips_per_veh.get(veh_id, 1))
        soh = 100.0 - (total_trips / max_trips) * 15.0
        soh = float(np.clip(soh + np.random.normal(0, 1.5), 75.0, 100.0))

        # Load weight from real static data
        weight_kg = float(static_df.loc[veh_id, "weight_kg"]) \
                    if veh_id in static_df.index else 1500.0
        weight_kg = float(np.clip(weight_kg, 800.0, 3500.0))

        records.append({
            "SoC":                 round(soc_mean,    2),
            "SoH":                 round(soh,         2),
            "Battery_Voltage":     round(avg_volt,    2),
            "Battery_Temperature": round(bat_temp,    2),
            "Driving_Speed":       round(avg_speed,   2),
            "Load_Weight":         round(weight_kg,   1),
            "Ambient_Temperature": round(avg_temp,    2),
            "Range_Left_km":       round(range_left_km, 2),
        })

    print(f"\n    Valid trips: {len(records):,}  |  Skipped: {skipped:,}")
    return pd.DataFrame(records)


#  Main 

def main():
    print("=" * 65)
    print("  VED Data Processor  EV Range Model (Real OBD-II Data)")
    print("=" * 65)

    print("\n Loading static vehicle data...")
    try:
        static_df = load_static_data()
    except FileNotFoundError as e:
        print(f"  {e}")
        return

    print("\n Loading dynamic telemetry data...")
    try:
        dynamic_df = load_dynamic_files()
    except FileNotFoundError as e:
        print(f"  {e}")
        return

    print("\n  Aggregating per-trip features from real OBD-II signals...")
    result_df = process_trips(dynamic_df, static_df)

    if result_df.empty:
        print("\n  No valid trips produced. Check HV Battery data availability.")
        return

    if len(result_df) < MIN_SAMPLES:
        print(f"\n  Only {len(result_df):,} samples  model may underfit.")

    print("\n Output Dataset Summary:")
    print("-" * 55)
    print(result_df.describe().round(2).to_string())
    print("-" * 55)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    result_df.to_csv(OUTPUT_FILE, index=False)

    print(f"""
  Saved  {OUTPUT_FILE}
    Rows    : {len(result_df):,}
    Columns : {list(result_df.columns)}

  Next  retrain EV Range model on real data:
    python AI_prediction_model.py
    Look for:  Using augmented data: {len(result_df):,} samples
""")


if __name__ == "__main__":
    main()
