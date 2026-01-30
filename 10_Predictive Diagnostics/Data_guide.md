# 📊 Generated Data Guide

## Overview
The realistic data generator creates **25,000 samples per model** using physics-based scenarios.

## Key Improvements
- ✅ Driver profiles (conservative/aggressive/highway/city)
- ✅ Time-based degradation
- ✅ Seasonal patterns
- ✅ Realistic correlations
- ✅ Failure scenarios

## File Location
```
Generated_output_data/PredictiveMaintenance_Realistic_25K.xlsx
```

## Data Quality
- **Size**: 25,000 samples per model (200,000 total)
- **Accuracy Improvement**: +1.3% average vs 5K random data
- **Realism**: Physics-based relationships, not random

## Excel File Structure

The Excel file contains **8 sheets**:

| Sheet # | Sheet Name | Purpose | Samples | Features |
|---------|------------|---------|---------|----------|
| 1 | **EV_Range_Data** | EV battery range prediction | 5000 | 7 + target |
| 2 | **Oil_Life_Data** | Oil change prediction | 5000 | 7 + target |
| 3 | **Tire_Wear_Data** | Tire tread depth prediction | 5000 | 9 + target |
| 4 | **Brake_Pad_Data** | Brake pad life prediction | 5000 | 9 + target |
| 5 | **Battery_Degradation_Data** | Battery health prediction | 5000 | 9 + target |
| 6 | **Coolant_Health_Data** | Coolant system prediction | 5000 | 9 + target |
| 7 | **Air_Filter_Data** | Air filter replacement | 5000 | 8 + target |
| 8 | **Transmission_Health_Data** | Transmission fluid change | 5000 | 8 + target |

**Total Data**: 40,000 training samples across 8 models

---

## Sheet Details

### 1️⃣ EV_Range_Data

**Purpose**: Predict remaining driving range for electric vehicles

**Features** (7):
- `SoC` - State of Charge (%)
- `SoH` - State of Health (%)
- `Battery_Voltage` - Voltage (V)
- `Battery_Temperature` - Temperature (°C)
- `Driving_Speed` - Speed (km/h)
- `Load_Weight` - Weight (kg)
- `Ambient_Temperature` - Outside temp (°C)

**Target**:
- `Range_Left_km` - Remaining range (10-500 km)

---

### 2️⃣ Oil_Life_Data

**Purpose**: Predict when oil change is needed

**Features** (7):
- `Engine_Temperature` - Engine temp (°C)
- `Engine_RPM` - RPM
- `Load_Weight` - Weight (kg)
- `Distance_Since_Last_Change` - km since last oil change
- `Oil_Viscosity` - Oil quality index
- `Ambient_Temperature` - Outside temp (°C)
- `Idle_Time` - Idle time percentage (%)

**Target**:
- `Oil_Change_In_km` - km until oil change (0-10000 km)

---

### 3️⃣ Tire_Wear_Data

**Purpose**: Predict tire tread depth and replacement needs

**Features** (9):
- `Total_Distance` - Total km driven
- `Average_Speed` - Typical speed (km/h)
- `Tire_Pressure` - Pressure (PSI)
- `Load_Weight` - Weight (kg)
- `Road_Type` - 0=Highway, 1=City, 2=Mixed
- `Alignment_Score` - Wheel alignment (%)
- `Tire_Age_Months` - Age in months
- `Harsh_Braking_Events` - Hard braking count
- `Temperature_Range` - Operating temp range (°C)

**Target**:
- `Tire_Tread_Depth_mm` - Tread depth (1.6-8.0 mm)
  - Legal minimum: 1.6mm
  - Replace at: <3mm

---

### 4️⃣ Brake_Pad_Data

**Purpose**: Predict brake pad thickness and service timing

**Features** (9):
- `Total_Distance` - Total km driven
- `Distance_Since_Last_Replacement` - km since last brake service
- `Average_Speed` - Speed (km/h)
- `Brake_Events_Per_100km` - Braking frequency
- `Load_Weight` - Weight (kg)
- `Driving_Style` - 0=Gentle, 1=Normal, 2=Aggressive
- `Mountain_Driving_Percent` - Hill driving (%)
- `Regenerative_Braking` - 0=No, 1=Yes (EV)
- `Brake_Temperature_Avg` - Brake temp (°C)

**Target**:
- `Brake_Pad_Thickness_mm` - Pad thickness (2-12 mm)
  - Minimum safe: 3mm
  - Replace at: <4mm

---

### 5️⃣ Battery_Degradation_Data

**Purpose**: Predict EV battery State of Health over time

**Features** (9):
- `Battery_Age_Months` - Age in months
- `Total_Charge_Cycles` - Number of charge cycles
- `Fast_Charge_Percentage` - % fast charges
- `Average_Depth_of_Discharge` - Typical discharge (%)
- `Battery_Temperature_Avg` - Average temp (°C)
- `Battery_Temperature_Range` - Temp fluctuation (°C)
- `Total_Distance` - Total km driven
- `Idle_Time_Percentage` - % idle time
- `High_Speed_Percentage` - % time over 100km/h

**Target**:
- `Battery_SoH_Percentage` - Battery health (60-100%)
  - Excellent: >90%
  - Replace: <70%

---

### 6️⃣ Coolant_Health_Data

**Purpose**: Predict coolant system service needs

**Features** (9):
- `Coolant_Age_Months` - Age in months
- `Engine_Temperature_Avg` - Average temp (°C)
- `Engine_Temperature_Max` - Peak temp (°C)
- `Coolant_Level` - Level (%)
- `Total_Distance` - Total km
- `Heavy_Load_Percentage` - % under heavy load
- `Ambient_Temperature` - Outside temp (°C)
- `Idle_Time_Percentage` - % idle time
- `Coolant_pH` - pH level (optimal: 9.0)

**Target**:
- `Coolant_Change_In_km` - km until change (0-60000 km)

---

### 7️⃣ Air_Filter_Data

**Purpose**: Predict air filter replacement timing

**Features** (8):
- `Distance_Since_Filter_Change` - km since last change
- `Air_Quality_Index` - Local air quality (AQI)
- `Dusty_Road_Percentage` - % dusty roads
- `Urban_Driving_Percentage` - % city driving
- `Engine_Air_Flow` - Flow efficiency (%)
- `Idle_Time_Percentage` - % idle time
- `Humidity_Avg` - Average humidity (%)
- `Filter_Type` - 0=Standard, 1=High-Performance

**Target**:
- `Filter_Change_In_km` - km until change (0-25000 km)

---

### 8️⃣ Transmission_Health_Data

**Purpose**: Predict transmission fluid change needs

**Features** (8):
- `Total_Distance` - Total km
- `Distance_Since_Fluid_Change` - km since last change
- `Transmission_Temperature` - Temp (°C)
- `Gear_Shifts_Per_100km` - Shift frequency
- `Harsh_Acceleration_Events` - Hard acceleration count
- `Towing_Percentage` - % time towing
- `City_Driving_Percentage` - % city driving
- `Transmission_Type` - 0=Manual, 1=Auto, 2=CVT

**Target**:
- `Transmission_Fluid_Change_In_km` - km until change (0-100000 km)

---

## Viewing the Data

### Method 1: Using Excel
```bash
# Navigate to the file
cd "10_Predictive Diagnostics/Generated_output_data"

# Open in Excel (Windows)
start PredictiveMaintenance_Complete.xlsx

# Open in Excel (Mac)
open PredictiveMaintenance_Complete.xlsx
```

### Method 2: Using Python Script
```bash
# View all data with statistics
python data_viewer.py

# Export to CSV files
python data_viewer.py export

# Compare features across models
python data_viewer.py compare
```

### Method 3: Using Python Code
```python
import pandas as pd

# Load specific sheet
df = pd.read_excel(
    "Generated_output_data/PredictiveMaintenance_Complete.xlsx",
    sheet_name="EV_Range_Data"
)

# View first few rows
print(df.head())

# View statistics
print(df.describe())

# List all sheets
excel_file = pd.ExcelFile("Generated_output_data/PredictiveMaintenance_Complete.xlsx")
print(excel_file.sheet_names)
```

---

## Data Quality

### Characteristics
- **Size**: 5000 samples per model (40,000 total)
- **Format**: Clean numerical data, no missing values
- **Distribution**: Realistic ranges based on real-world vehicle data
- **Correlations**: Features have realistic relationships with targets
- **Noise**: Random noise added to simulate real-world variability

### Validation
All generated data includes:
- ✅ Realistic value ranges
- ✅ Proper correlations between features
- ✅ Statistical noise for model robustness
- ✅ Edge cases (critical values)
- ✅ Clipping to prevent unrealistic outliers

---

## Regenerating Data

If you need fresh data or different parameters:

```bash
# Generate new data (5000 samples per model)
python synthetic_data_generator.py

# Or modify in code:
generator = MaintenanceDataGenerator(
    n_samples=10000,  # Change sample size
    random_state=123  # Change random seed
)
generator.generate_all_data()
```

---

## Using the Data

### For Model Training
```python
# The AI_prediction_model.py script automatically uses this data
python AI_prediction_model.py
```

### For Custom Analysis
```python
import pandas as pd

# Load data
df = pd.read_excel(
    "Generated_output_data/PredictiveMaintenance_Complete.xlsx",
    sheet_name="Tire_Wear_Data"
)

# Split features and target
X = df.drop(columns=["Tire_Tread_Depth_mm"])
y = df["Tire_Tread_Depth_mm"]

# Your analysis here...
```

---

## File Management

### Backup Data
```bash
# Create backup
cp Generated_output_data/PredictiveMaintenance_Complete.xlsx \
   Generated_output_data/PredictiveMaintenance_Complete_BACKUP.xlsx
```

### Check File Size
```bash
# Windows
dir Generated_output_data\PredictiveMaintenance_Complete.xlsx

# Mac/Linux
ls -lh Generated_output_data/PredictiveMaintenance_Complete.xlsx
```

Expected size: **~2-3 MB**

---

## Troubleshooting

### File Not Found
```
❌ Error: File not found
```
**Solution**: Run `python synthetic_data_generator.py`

### Excel Can't Open File
```
❌ Excel error: File is corrupted
```
**Solution**: 
1. Delete the file
2. Run `python synthetic_data_generator.py` again
3. Ensure openpyxl is installed: `pip install openpyxl`

### Wrong Column Names
```
❌ KeyError: Column not found
```
**Solution**: Regenerate data - column names have been updated

---

## Data Privacy

⚠️ **Important**: This is **synthetic (fake) data** generated algorithmically.
- No real vehicle data
- Safe to share and use for demos
- No privacy concerns

---

## Next Steps

1. ✅ Generate data: `python synthetic_data_generator.py`
2. ✅ View data: `python data_viewer.py`
3. ✅ Train models: `python AI_prediction_model.py`
4. ✅ Run dashboard: `streamlit run app.py`

---

**Questions?** Check the main README.md or examine the code in `synthetic_data_generator.py`