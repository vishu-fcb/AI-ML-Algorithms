# 📊 Data Guide - BeyondTech Predictive Maintenance

Complete guide to data sources, structure, and generation for the BeyondTech predictive maintenance system.

---

## 🎯 Overview

BeyondTech uses a **hybrid data approach** that combines:

1. **Real Public Research Data** (50,000 samples) - For 2 models
2. **Physics-Based Synthetic Data** (200,000 samples) - For 6 models

This approach provides both **credibility** (real data) and **flexibility** (synthetic data).

---

## 📁 Data Architecture

```
Data Pipeline
├── Augmented_Datasets/                # Real research data (COMMIT to Git)
│   ├── nasa_oil_life_augmented.csv        # 25K samples, 3.9 MB
│   ├── battery_degradation_augmented.csv  # 25K samples, 4.4 MB
│   └── uci_automobile_augmented.csv       # 5K samples, 1.4 MB
│
└── Generated_output_data/             # Synthetic data (DO NOT commit)
    └── PredictiveMaintenance_Realistic_25K.xlsx  # 200K samples, 21 MB
        ├── EV_Range_Data                    # 25,000 samples
        ├── Oil_Life_Data                    # 25,000 samples
        ├── Tire_Wear_Data                   # 25,000 samples
        ├── Brake_Pad_Data                   # 25,000 samples
        ├── Battery_Degradation_Data         # 25,000 samples
        ├── Coolant_Health_Data              # 25,000 samples
        ├── Air_Filter_Data                  # 25,000 samples
        └── Transmission_Health_Data         # 25,000 samples
```

---

## 🔬 Real Data Sources (Augmented Datasets)

### 1. NASA Oil Life (nasa_oil_life_augmented.csv)

**Origin:** NASA Turbofan Engine Degradation Dataset (C-MAPSS)  
**Samples:** 25,000  
**Size:** 3.9 MB  
**Used by:** Oil Life Prediction Model

**What it contains:**
- Real engine sensor measurements from aircraft turbofan engines
- Adapted to vehicle oil life prediction
- Temperature, RPM, load, and time-series degradation patterns

**Why it's credible:**
- Industry-standard benchmark dataset
- Used in 100+ published research papers
- From NASA's Prognostics Center of Excellence

**Training Result:** R² = 0.9799 (98.0% accuracy)

---

### 2. Battery Degradation (battery_degradation_augmented.csv)

**Origin:** Public battery cycling research datasets  
**Samples:** 25,000  
**Size:** 4.4 MB  
**Used by:** Battery Degradation Model

**What it contains:**
- Real Li-ion battery charge/discharge cycles
- Temperature effects on degradation
- Calendar aging and cycle aging patterns
- State of Health (SoH) measurements

**Why it's credible:**
- From academic battery research laboratories
- Real experimental data from battery cycling tests
- Validated thermal and electrochemical effects

**Training Result:** R² = 0.9248 (92.5% accuracy)

---

### 3. UCI Automobile (uci_automobile_augmented.csv)

**Origin:** UCI Machine Learning Repository  
**Samples:** 5,205  
**Size:** 1.4 MB  
**Used by:** Feature engineering and validation

**What it contains:**
- Real vehicle specifications and characteristics
- Make, model, engine type, dimensions, performance metrics
- Used for validating feature importance and relationships

---

## 🧪 Synthetic Data Generation

### Command
```bash
python synthetic_data_generator.py
```

### Output
```
Generated_output_data/PredictiveMaintenance_Realistic_25K.xlsx
Size: ~21 MB
Total samples: 200,000 (25,000 per model × 8 models)
Generation time: 3-5 minutes
```

---

## 📊 Synthetic Data Structure

### Excel File Organization

| Sheet # | Name | Samples | Features | Target | Data Source |
|---------|------|---------|----------|--------|-------------|
| 1 | **EV_Range_Data** | 25,000 | 7 | Range_Left_km | Synthetic |
| 2 | **Oil_Life_Data** | 25,000 | 7 | Oil_Change_In_km | **NASA Data** |
| 3 | **Tire_Wear_Data** | 25,000 | 9 | Tire_Tread_Depth_mm | Synthetic |
| 4 | **Brake_Pad_Data** | 25,000 | 9 | Brake_Pad_Thickness_mm | Synthetic |
| 5 | **Battery_Degradation_Data** | 25,000 | 9 | Battery_SoH_Percentage | **NASA Data** |
| 6 | **Coolant_Health_Data** | 25,000 | 9 | Coolant_Change_In_km | Synthetic |
| 7 | **Air_Filter_Data** | 25,000 | 8 | Filter_Change_In_km | Synthetic |
| 8 | **Transmission_Health_Data** | 25,000 | 8 | Transmission_Fluid_Change_In_km | Synthetic |

**Note:** Even though the Excel file contains Oil_Life_Data and Battery_Degradation_Data sheets, the training script will use the augmented CSV files from `Augmented_Datasets/` instead.

---

## 🎨 Data Generation Features

### 1. Driver Profile Distribution

| Profile | Percentage | Characteristics |
|---------|-----------|-----------------|
| **Conservative** | 30% | Gentle acceleration, lower speeds, careful braking |
| **Aggressive** | 20% | Hard acceleration, high speeds, frequent hard braking |
| **Highway** | 30% | Consistent high-speed, minimal braking, long distances |
| **City** | 20% | Stop-and-go traffic, frequent braking, lower speeds |

### 2. Physics-Based Degradation Models

#### Temperature Effects (Battery, Coolant, Brake)
- **Arrhenius equation** for thermal degradation
- **U-shaped efficiency curve** (optimal at 20-25°C)
- Cold weather: -30-40% efficiency
- Hot weather: -15-20% efficiency

#### Mechanical Wear (Tire, Brake, Transmission)
- **Distance-dependent:** Linear wear rate
- **Load-dependent:** Heavier vehicles wear faster
- **Alignment effects:** Poor alignment increases tire wear
- **Pressure effects:** Under/over-inflation accelerates wear

#### Chemical Degradation (Oil, Battery, Coolant)
- **Oxidation rates:** Temperature-dependent
- **pH degradation:** Coolant loses alkalinity over time
- **Viscosity breakdown:** Oil degrades with heat and contamination

### 3. Environmental Factors

- **Seasonal Temperature:** Winter/summer variations (-10°C to 45°C)
- **Air Quality:** AQI index (20-150) affects filter degradation
- **Terrain:** Mountain vs flat driving (affects brakes, transmission)
- **Climate:** Humidity (30-80%) and dust exposure

### 4. Time-Based Aging

- **Calendar aging:** Components degrade even when idle
- **Cycle aging:** Degradation from use (charge cycles, brake events)
- **Cumulative stress:** Long-term effects accumulate

### 5. Realistic Failure Scenarios

| Scenario | Percentage | Purpose |
|----------|-----------|---------|
| **Critical** | 10-15% | Near-failure cases for safety training |
| **Warning** | 20-25% | Maintenance soon needed |
| **Good** | 60-70% | Healthy component operation |

---

## 📋 Detailed Model Data Specifications

### 1️⃣ EV Range Prediction

**Features (7):**
```
SoC                    : State of Charge (20-100%)
SoH                    : State of Health (70-100%)
Battery_Voltage        : Voltage (360-420 V)
Battery_Temperature    : Battery temp (-5 to 45°C)
Driving_Speed          : Current speed (0-130 km/h)
Load_Weight            : Vehicle load (1200-1900 kg)
Ambient_Temperature    : Outside temp (-10 to 40°C)
```

**Target:**
```
Range_Left_km          : Remaining range (10-550 km)
```

**Physics:**
- Temperature affects battery chemistry (Arrhenius equation)
- Speed increases aerodynamic drag (proportional to speed²)
- Load increases energy consumption (linear relationship)
- Battery health reduces total capacity

**Training Result:** R² = 0.9852 (98.5% accuracy)

---

### 2️⃣ Oil Life Prediction

**Features (7):**
```
Engine_Temperature         : Engine temp (70-110°C)
Engine_RPM                 : Revolutions per minute (800-6000)
Load_Weight                : Vehicle load (1200-2400 kg)
Distance_Since_Last_Change : km since last oil change (0-15000)
Oil_Viscosity              : Oil quality grade (5.0, 5.5, 6.0)
Ambient_Temperature        : Outside temp (-10 to 45°C)
Idle_Time                  : % of time spent idling (0-40%)
```

**Target:**
```
Oil_Change_In_km           : Kilometers until oil change (0-10500 km)
```

**Physics:**
- High temperature = exponential oxidation rate
- High RPM = increased contamination
- Heavy load = more stress on lubricant
- Idle time = fuel dilution and moisture

**Data Source:** NASA Turbofan Dataset (25,000 samples)  
**Training Result:** R² = 0.9799 (98.0% accuracy)

---

### 3️⃣ Tire Wear Prediction

**Features (9):**
```
Total_Distance          : Total km driven (0-80000)
Average_Speed           : Typical speed (30-120 km/h)
Tire_Pressure           : Pressure in PSI (26-38)
Load_Weight             : Vehicle load (1200-2100 kg)
Road_Type               : 0=Highway, 1=City, 2=Mixed
Alignment_Score         : Wheel alignment quality (60-100%)
Tire_Age_Months         : Age in months (0-60)
Harsh_Braking_Events    : Hard braking count (10-120)
Temperature_Range       : Operating temp range (12-38°C)
```

**Target:**
```
Tire_Tread_Depth_mm     : Tread depth (1.6-8.2 mm)
```

**Safety Thresholds:**
- Legal minimum: 1.6mm
- Recommended replacement: <3mm
- New tire: 8mm

**Training Result:** R² = 0.9699 (97.0% accuracy)

---

### 4️⃣ Brake Pad Prediction

**Features (9):**
```
Total_Distance                 : Total km driven (0-80000)
Distance_Since_Last_Replacement : km since brake service (0-80000)
Average_Speed                  : Speed (20-130 km/h)
Brake_Events_Per_100km         : Braking frequency (15-250)
Load_Weight                    : Vehicle load (1200-2200 kg)
Driving_Style                  : 0=Gentle, 1=Normal, 2=Aggressive
Mountain_Driving_Percent       : Hill driving % (0-60%)
Regenerative_Braking           : 0=No (ICE), 1=Yes (EV/Hybrid)
Brake_Temperature_Avg          : Average brake temp (50-280°C)
```

**Target:**
```
Brake_Pad_Thickness_mm         : Pad thickness (2-12 mm)
```

**Safety Thresholds:**
- Minimum safe: 3mm
- Recommended service: <4mm
- New pad: 12mm

**Regenerative Braking Impact:**
- EVs: 50% slower brake wear
- Hybrids: 30% slower brake wear

**Training Result:** R² = 0.9644 (96.4% accuracy)

---

### 5️⃣ Battery Degradation Prediction

**Features (9):**
```
Battery_Age_Months             : Age in months (0-120)
Total_Charge_Cycles            : Charge cycle count (0-2000)
Fast_Charge_Percentage         : % of fast charges (0-80%)
Average_Depth_of_Discharge     : Typical discharge depth (20-95%)
Battery_Temperature_Avg        : Average temp (15-40°C)
Battery_Temperature_Range      : Temp fluctuation (5-35°C)
Total_Distance                 : Total km driven (0-250000)
Idle_Time_Percentage           : % idle time (0-30%)
High_Speed_Percentage          : % time over 100km/h (0-55%)
```

**Target:**
```
Battery_SoH_Percentage         : State of Health (60-100%)
```

**Health Categories:**
- Excellent: >90%
- Good: 80-90%
- Degraded: 70-80%
- Replace: <70%

**Physics:**
- Calendar aging: ~0.15% per month
- Cycle aging: ~0.015% per cycle
- Fast charging accelerates degradation
- Temperature stress is non-linear

**Data Source:** Battery Research Dataset (25,000 samples)  
**Training Result:** R² = 0.9248 (92.5% accuracy)

---

### 6️⃣ Coolant Health Prediction

**Features (9):**
```
Coolant_Age_Months         : Age in months (0-72)
Engine_Temperature_Avg     : Average engine temp (70-110°C)
Engine_Temperature_Max     : Peak temp (80-120°C)
Coolant_Level              : Level percentage (75-100%)
Total_Distance             : Total km driven (0-250000)
Heavy_Load_Percentage      : % under load (0-50%)
Ambient_Temperature        : Outside temp (-10 to 45°C)
Idle_Time_Percentage       : % idle (0-30%)
Coolant_pH                 : pH level (8.0-10.0, optimal 9.0)
```

**Target:**
```
Coolant_Change_In_km       : km until coolant change (0-70000 km)
```

**Service Interval:** 60,000 km or 3 years (whichever comes first)

**Training Result:** R² = 0.9949 (99.5% accuracy) - **World-class**

---

### 7️⃣ Air Filter Prediction

**Features (8):**
```
Distance_Since_Filter_Change : km since last change (0-30000)
Air_Quality_Index            : Local AQI (20-150)
Dusty_Road_Percentage        : % dusty roads (0-40%)
Urban_Driving_Percentage     : % city driving (30-80%)
Engine_Air_Flow              : Flow efficiency (75-100%)
Idle_Time_Percentage         : % idle (0-30%)
Humidity_Avg                 : Average humidity (30-80%)
Filter_Type                  : 0=Standard, 1=High-Performance
```

**Target:**
```
Filter_Change_In_km          : km until filter change (0-25000 km)
```

**Service Interval:** 15,000-25,000 km (depends on air quality)

**Training Result:** R² = 0.9929 (99.3% accuracy) - **World-class**

---

### 8️⃣ Transmission Health Prediction

**Features (8):**
```
Total_Distance                : Total km driven (0-250000)
Distance_Since_Fluid_Change   : km since last change (0-120000)
Transmission_Temperature      : Operating temp (60-130°C)
Gear_Shifts_Per_100km         : Shift frequency (50-300)
Harsh_Acceleration_Events     : Hard acceleration count (10-90)
Towing_Percentage             : % time towing (0-20%)
City_Driving_Percentage       : % city driving (30-80%)
Transmission_Type             : 0=Manual, 1=Automatic, 2=CVT
```

**Target:**
```
Transmission_Fluid_Change_In_km : km until fluid change (0-120000 km)
```

**Service Intervals:**
- Manual: 80,000-120,000 km
- Automatic: 60,000-100,000 km
- CVT: 50,000-80,000 km

**Training Result:** R² = 0.9858 (98.6% accuracy)

---

## 🔄 How Training Uses the Data

### Hybrid Training Logic

When you run `python AI_prediction_model.py`, the script:

```python
For each model:
    1. Check if augmented CSV exists in Augmented_Datasets/
       ├─ If YES: Use REAL data (Oil Life, Battery Degradation)
       └─ If NO: Use synthetic data from Excel
    
    2. Load data and split 80/20 train/test
    
    3. Train XGBoost model (150 trees, depth 5)
    
    4. Validate with 5-fold cross-validation
    
    5. Save model to Trained_Model/
```

### Data Priority

```
Training Priority (Highest to Lowest):
1. Augmented CSV (real data) in Augmented_Datasets/
2. Synthetic Excel sheet in Generated_output_data/
```

**Example:**
- **Oil Life Model:** Uses `nasa_oil_life_augmented.csv` (REAL)
- **Tire Wear Model:** Uses synthetic `Tire_Wear_Data` sheet (SYNTHETIC)

---

## 📊 Data Quality Metrics

### ✅ Realistic Correlations
- Distance → wear (positive, linear)
- Temperature → degradation (exponential)
- Load → stress (positive, linear)
- Time → aging (calendar effect)

### ✅ Multi-Modal Distributions
- Not purely random or normal
- Multiple peaks representing different scenarios
- Gamma distributions for realistic skew

### ✅ Edge Case Coverage
- 10-15% critical samples (near-failure)
- 20-25% warning samples (maintenance soon)
- 60-70% normal operation
- Ensures model robustness

### ✅ Physical Constraints
- All values within realistic ranges
- No impossible combinations
- Respects engineering limits

---

## 🔍 Viewing the Data

### Option 1: Excel
```bash
# Windows: Double-click the file
# Mac: open Generated_output_data/PredictiveMaintenance_Realistic_25K.xlsx
# Linux: libreoffice Generated_output_data/PredictiveMaintenance_Realistic_25K.xlsx
```

### Option 2: Python Script
```bash
python data_viewer.py

# Options:
# python data_viewer.py export   # Export sheets to individual CSVs
# python data_viewer.py compare  # Compare features across models
```

### Option 3: Python Code
```python
import pandas as pd

# Load a specific sheet
df = pd.read_excel(
    "Generated_output_data/PredictiveMaintenance_Realistic_25K.xlsx",
    sheet_name="EV_Range_Data"
)

# View statistics
print(df.describe())

# View first rows
print(df.head())

# Check correlations
print(df.corr())
```

---

## 🔄 Regenerating Data

### When to Regenerate

**Recommended:**
- Experimenting with different random seeds
- Testing model sensitivity to data variations
- Creating larger/smaller datasets

**Not Necessary:**
- Just for normal training
- After every model update
- For production deployment

### How to Regenerate

```bash
# Generate with default settings (25K per model)
python synthetic_data_generator.py

# This will overwrite the existing Excel file
```

### Modifying Generation Parameters

Edit `synthetic_data_generator.py`:

```python
# Change sample size
generator = RealisticMaintenanceDataGenerator(
    n_samples=50000,  # Change from 25000
    random_state=42   # Change for different randomization
)
```

---

## 🚨 Important Notes

### Git and Large Files

⚠️ **DO NOT commit generated files to Git:**

**DO commit:**
- ✅ `synthetic_data_generator.py` (the generator script)
- ✅ `Augmented_Datasets/*.csv` (real data - relatively small)
- ✅ `.gitkeep` files (preserve folder structure)

**DO NOT commit:**
- ❌ `PredictiveMaintenance_Realistic_25K.xlsx` (~21 MB)
- ❌ Trained model `.pkl` files (~50 MB each)
- ❌ Generated reports and plots

### Data Privacy

✅ **All data is safe to share:**
- Synthetic data: 100% generated, no real vehicles
- Augmented data: Public research datasets
- No personal information
- No proprietary vehicle data

---

## 📚 Related Documentation

- **README.md:** Project overview and quick start
- **requirements.txt:** Python dependencies
- **vehicle_data.py:** Sample vehicle configurations
- **AI_prediction_model.py:** Training script

---

## 🎓 Key Takeaways

### Why Hybrid Approach?

**Real Data Advantages:**
- ✅ Credibility and validation
- ✅ Industry-standard benchmarks
- ✅ Proven degradation patterns

**Synthetic Data Advantages:**
- ✅ Larger sample sizes (25K vs 5K)
- ✅ Controlled failure scenarios
- ✅ Complete feature coverage
- ✅ Customizable distributions

**Combined Benefits:**
- ✅ Best of both worlds
- ✅ 2 models with real data credibility
- ✅ 6 models with synthetic flexibility
- ✅ Total: 250,000 training samples

### Why Physics-Based Generation?

**Better than random data:**
- ✅ Realistic correlations encoded
- ✅ Domain knowledge built-in
- ✅ Edge cases systematically included

**Better than purely real data:**
- ✅ Much larger sample sizes
- ✅ Balanced failure scenarios
- ✅ Complete feature sets
- ✅ No missing values

---

**For detailed code implementation, see `synthetic_data_generator.py` with extensive inline comments.**