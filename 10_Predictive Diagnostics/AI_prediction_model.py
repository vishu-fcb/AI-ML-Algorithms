"""
AI Prediction Model Training
Trains all predictive maintenance models using XGBoost
NOW USES: Augmented public datasets + synthetic data (hybrid approach)
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import joblib
import os
from datetime import datetime

class MaintenanceModelTrainer:
    """Train and evaluate all predictive maintenance models"""
    
    def __init__(self, 
                 synthetic_data_file="Generated_output_data/PredictiveMaintenance_Realistic_25K.xlsx",
                 augmented_data_dir="Augmented_Datasets",
                 public_data_dir="Public_Datasets"):
        self.synthetic_data_file = synthetic_data_file
        self.augmented_data_dir = augmented_data_dir
        self.public_data_dir = public_data_dir
        self.models = {}
        self.metrics = {}
        self.data_sources = {}  # Track which data source was used
        
        # Create directories
        os.makedirs("Trained Model", exist_ok=True)
        os.makedirs("Model_Reports", exist_ok=True)
        
    def load_best_available_data(self, augmented_file, synthetic_sheet, feature_cols, target_col):
        """
        Load the best available data source in priority order:
        1. Augmented dataset (if exists)
        2. Synthetic dataset (fallback)
        
        Returns: (dataframe, source_description)
        """
        # Try augmented dataset first
        augmented_path = f"{self.augmented_data_dir}/{augmented_file}"
        
        if os.path.exists(augmented_path):
            try:
                df = pd.read_csv(augmented_path)
                
                # Verify it has the required columns
                required_cols = feature_cols + [target_col]
                if all(col in df.columns for col in required_cols):
                    source = f"Augmented ({len(df):,} samples from public datasets)"
                    print(f"   📥 Using augmented data: {len(df):,} samples")
                    return df[required_cols], source
                else:
                    print(f"   ⚠️  Augmented file missing columns, using synthetic")
            except Exception as e:
                print(f"   ⚠️  Error loading augmented data: {e}")
        
        # Fallback to synthetic
        print(f"   📥 Using synthetic data (augmented not available)")
        df = pd.read_excel(self.synthetic_data_file, sheet_name=synthetic_sheet)
        source = f"Synthetic ({len(df):,} samples)"
        return df, source
        
    def train_model(self, X_train, y_train):
        """Train an XGBoost model"""
        model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test):
        """Evaluate model performance"""
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        metrics = {
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'cv_score': np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='r2'))
        }
        return metrics
    
    def train_ev_range_model(self):
        """Train EV Range prediction model"""
        print("\n🔋 Training EV Range Model...")
        
        feature_cols = ['SoC', 'SoH', 'Battery_Voltage', 'Battery_Temperature', 
                       'Driving_Speed', 'Load_Weight', 'Ambient_Temperature']
        target_col = 'Range_Left_km'
        
        # Try to use augmented data, fallback to synthetic
        df, source = self.load_best_available_data(
            'ev_range_augmented.csv',  # Not generated yet, will use synthetic
            'EV_Range_Data',
            feature_cols,
            target_col
        )
        
        X = df[feature_cols]
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = self.train_model(X_train, y_train)
        metrics = self.evaluate_model(model, X_train, X_test, y_train, y_test)
        
        self.models['ev_range'] = model
        self.metrics['ev_range'] = metrics
        self.data_sources['ev_range'] = source
        
        joblib.dump(model, "Trained Model/ev_range_model.pkl")
        print(f"   ✓ R²: {metrics['test_r2']:.4f} | RMSE: {metrics['test_rmse']:.2f} km")
        print(f"   📊 Data source: {source}")
        return model, metrics
    
    def train_oil_life_model(self):
        """Train Oil Life prediction model - USES NASA AUGMENTED DATA"""
        print("\n🛢️  Training Oil Life Model...")
        
        feature_cols = ['Engine_Temperature', 'Engine_RPM', 'Load_Weight', 
                       'Distance_Since_Last_Change', 'Oil_Viscosity', 
                       'Ambient_Temperature', 'Idle_Time']
        target_col = 'Oil_Change_In_km'
        
        # Try NASA augmented data first
        df, source = self.load_best_available_data(
            'nasa_oil_life_augmented.csv',  # ✅ This exists!
            'Oil_Life_Data',
            feature_cols,
            target_col
        )
        
        X = df[feature_cols]
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = self.train_model(X_train, y_train)
        metrics = self.evaluate_model(model, X_train, X_test, y_train, y_test)
        
        self.models['oil_life'] = model
        self.metrics['oil_life'] = metrics
        self.data_sources['oil_life'] = source
        
        joblib.dump(model, "Trained Model/oil_life_model.pkl")
        print(f"   ✓ R²: {metrics['test_r2']:.4f} | RMSE: {metrics['test_rmse']:.2f} km")
        print(f"   📊 Data source: {source}")
        return model, metrics
    
    def train_tire_wear_model(self):
        """Train Tire Wear prediction model"""
        print("\n🛞 Training Tire Wear Model...")
        
        feature_cols = ['Total_Distance', 'Average_Speed', 'Tire_Pressure', 
                       'Load_Weight', 'Road_Type', 'Alignment_Score', 
                       'Tire_Age_Months', 'Harsh_Braking_Events', 'Temperature_Range']
        target_col = 'Tire_Tread_Depth_mm'
        
        df, source = self.load_best_available_data(
            'tire_wear_augmented.csv',
            'Tire_Wear_Data',
            feature_cols,
            target_col
        )
        
        X = df[feature_cols]
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = self.train_model(X_train, y_train)
        metrics = self.evaluate_model(model, X_train, X_test, y_train, y_test)
        
        self.models['tire_wear'] = model
        self.metrics['tire_wear'] = metrics
        self.data_sources['tire_wear'] = source
        
        joblib.dump(model, "Trained Model/tire_wear_model.pkl")
        print(f"   ✓ R²: {metrics['test_r2']:.4f} | RMSE: {metrics['test_rmse']:.3f} mm")
        print(f"   📊 Data source: {source}")
        return model, metrics
    
    def train_brake_pad_model(self):
        """Train Brake Pad Life prediction model"""
        print("\n🔧 Training Brake Pad Model...")
        
        feature_cols = ['Total_Distance', 'Distance_Since_Last_Replacement', 
                       'Average_Speed', 'Brake_Events_Per_100km', 'Load_Weight', 
                       'Driving_Style', 'Mountain_Driving_Percent', 
                       'Regenerative_Braking', 'Brake_Temperature_Avg']
        target_col = 'Brake_Pad_Thickness_mm'
        
        df, source = self.load_best_available_data(
            'brake_pad_augmented.csv',
            'Brake_Pad_Data',
            feature_cols,
            target_col
        )
        
        X = df[feature_cols]
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = self.train_model(X_train, y_train)
        metrics = self.evaluate_model(model, X_train, X_test, y_train, y_test)
        
        self.models['brake_pad'] = model
        self.metrics['brake_pad'] = metrics
        self.data_sources['brake_pad'] = source
        
        joblib.dump(model, "Trained Model/brake_pad_model.pkl")
        print(f"   ✓ R²: {metrics['test_r2']:.4f} | RMSE: {metrics['test_rmse']:.3f} mm")
        print(f"   📊 Data source: {source}")
        return model, metrics
    
    def train_battery_degradation_model(self):
        """Train Battery Degradation prediction model - USES AUGMENTED BATTERY DATA"""
        print("\n⚡ Training Battery Degradation Model...")
        
        feature_cols = ['Battery_Age_Months', 'Total_Charge_Cycles', 
                       'Fast_Charge_Percentage', 'Average_Depth_of_Discharge', 
                       'Battery_Temperature_Avg', 'Battery_Temperature_Range', 
                       'Total_Distance', 'Idle_Time_Percentage', 
                       'High_Speed_Percentage']
        target_col = 'Battery_SoH_Percentage'
        
        # Try battery augmented data
        df, source = self.load_best_available_data(
            'battery_degradation_augmented.csv',  # ✅ This exists!
            'Battery_Degradation_Data',
            feature_cols,
            target_col
        )
        
        X = df[feature_cols]
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = self.train_model(X_train, y_train)
        metrics = self.evaluate_model(model, X_train, X_test, y_train, y_test)
        
        self.models['battery_degradation'] = model
        self.metrics['battery_degradation'] = metrics
        self.data_sources['battery_degradation'] = source
        
        joblib.dump(model, "Trained Model/battery_degradation_model.pkl")
        print(f"   ✓ R²: {metrics['test_r2']:.4f} | RMSE: {metrics['test_rmse']:.2f} %")
        print(f"   📊 Data source: {source}")
        return model, metrics
    
    def train_coolant_health_model(self):
        """Train Coolant Health prediction model"""
        print("\n❄️  Training Coolant Health Model...")
        
        feature_cols = ['Coolant_Age_Months', 'Engine_Temperature_Avg', 
                       'Engine_Temperature_Max', 'Coolant_Level', 
                       'Total_Distance', 'Heavy_Load_Percentage', 
                       'Ambient_Temperature', 'Idle_Time_Percentage', 'Coolant_pH']
        target_col = 'Coolant_Change_In_km'
        
        df, source = self.load_best_available_data(
            'coolant_health_augmented.csv',
            'Coolant_Health_Data',
            feature_cols,
            target_col
        )
        
        X = df[feature_cols]
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = self.train_model(X_train, y_train)
        metrics = self.evaluate_model(model, X_train, X_test, y_train, y_test)
        
        self.models['coolant_health'] = model
        self.metrics['coolant_health'] = metrics
        self.data_sources['coolant_health'] = source
        
        joblib.dump(model, "Trained Model/coolant_health_model.pkl")
        print(f"   ✓ R²: {metrics['test_r2']:.4f} | RMSE: {metrics['test_rmse']:.2f} km")
        print(f"   📊 Data source: {source}")
        return model, metrics
    
    def train_air_filter_model(self):
        """Train Air Filter prediction model"""
        print("\n🌬️  Training Air Filter Model...")
        
        feature_cols = ['Distance_Since_Filter_Change', 'Air_Quality_Index', 
                       'Dusty_Road_Percentage', 'Urban_Driving_Percentage', 
                       'Engine_Air_Flow', 'Idle_Time_Percentage', 
                       'Humidity_Avg', 'Filter_Type']
        target_col = 'Filter_Change_In_km'
        
        df, source = self.load_best_available_data(
            'air_filter_augmented.csv',
            'Air_Filter_Data',
            feature_cols,
            target_col
        )
        
        X = df[feature_cols]
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = self.train_model(X_train, y_train)
        metrics = self.evaluate_model(model, X_train, X_test, y_train, y_test)
        
        self.models['air_filter'] = model
        self.metrics['air_filter'] = metrics
        self.data_sources['air_filter'] = source
        
        joblib.dump(model, "Trained Model/air_filter_model.pkl")
        print(f"   ✓ R²: {metrics['test_r2']:.4f} | RMSE: {metrics['test_rmse']:.2f} km")
        print(f"   📊 Data source: {source}")
        return model, metrics
    
    def train_transmission_model(self):
        """Train Transmission Health prediction model"""
        print("\n⚙️  Training Transmission Health Model...")
        
        feature_cols = ['Total_Distance', 'Distance_Since_Fluid_Change', 
                       'Transmission_Temperature', 'Gear_Shifts_Per_100km', 
                       'Harsh_Acceleration_Events', 'Towing_Percentage', 
                       'City_Driving_Percentage', 'Transmission_Type']
        target_col = 'Transmission_Fluid_Change_In_km'
        
        df, source = self.load_best_available_data(
            'transmission_health_augmented.csv',
            'Transmission_Health_Data',
            feature_cols,
            target_col
        )
        
        X = df[feature_cols]
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = self.train_model(X_train, y_train)
        metrics = self.evaluate_model(model, X_train, X_test, y_train, y_test)
        
        self.models['transmission_health'] = model
        self.metrics['transmission_health'] = metrics
        self.data_sources['transmission_health'] = source
        
        joblib.dump(model, "Trained Model/transmission_health_model.pkl")
        print(f"   ✓ R²: {metrics['test_r2']:.4f} | RMSE: {metrics['test_rmse']:.2f} km")
        print(f"   📊 Data source: {source}")
        return model, metrics
    
    def check_available_datasets(self):
        """Check which augmented datasets are available"""
        print("\n🔍 Checking for available datasets...")
        print("-" * 70)
        
        augmented_files = [
            ('NASA Oil Life', 'nasa_oil_life_augmented.csv'),
            ('Battery Degradation', 'battery_degradation_augmented.csv'),
            ('UCI Automobile', 'uci_automobile_augmented.csv'),
        ]
        
        found = []
        missing = []
        
        for name, filename in augmented_files:
            path = f"{self.augmented_data_dir}/{filename}"
            if os.path.exists(path):
                size = os.path.getsize(path) / (1024 * 1024)  # MB
                df = pd.read_csv(path)
                found.append(f"✅ {name:25s} ({len(df):6,} samples, {size:.1f} MB)")
            else:
                missing.append(f"⚠️  {name:25s} (will use synthetic)")
        
        if found:
            print("\n📊 Available Augmented Datasets:")
            for item in found:
                print(f"   {item}")
        
        if missing:
            print("\n⚠️  Missing Augmented Datasets (using synthetic):")
            for item in missing:
                print(f"   {item}")
        
        print()
    
    def train_all_models(self):
        """Train all maintenance prediction models"""
        print("\n" + "="*70)
        print("🚀 BeyondTech - HYBRID MODEL TRAINING")
        print("   Using Augmented Public Datasets + Synthetic Data")
        print("="*70)
        
        # Check available datasets
        self.check_available_datasets()
        
        # Train all models
        print("\n📊 TRAINING MODELS:")
        print("="*70)
        
        self.train_ev_range_model()
        self.train_oil_life_model()
        self.train_tire_wear_model()
        self.train_brake_pad_model()
        self.train_battery_degradation_model()
        self.train_coolant_health_model()
        self.train_air_filter_model()
        self.train_transmission_model()
        
        # Print summary
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE - SUMMARY")
        print("="*70 + "\n")
        
        summary_data = []
        for name, metrics in self.metrics.items():
            summary_data.append({
                'Model': name.replace('_', ' ').title(),
                'Test R²': f"{metrics['test_r2']:.4f}",
                'RMSE': f"{metrics['test_rmse']:.2f}",
                'CV Score': f"{metrics['cv_score']:.4f}",
                'Data Source': self.data_sources.get(name, 'Unknown')
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Count data sources
        augmented_count = sum(1 for source in self.data_sources.values() if 'Augmented' in source)
        synthetic_count = len(self.data_sources) - augmented_count
        
        print(f"\n📊 Data Source Breakdown:")
        print(f"   Models using AUGMENTED data: {augmented_count}/8")
        print(f"   Models using SYNTHETIC data:  {synthetic_count}/8")
        
        if augmented_count > 0:
            print(f"\n✅ CREDIBILITY BOOST!")
            print(f"   Using {augmented_count} augmented datasets from public research!")
            print(f"   Total training samples: {sum(int(s.split('(')[1].split()[0].replace(',', '')) for s in self.data_sources.values() if 'Augmented' in s):,}")
        
        print("\n📦 Saved Models:")
        for name in self.models.keys():
            print(f"   ✓ Trained Model/{name}_model.pkl")
        
        print("\n🎉 All models are ready for deployment!")
        print("   Run: streamlit run app.py")
        print("="*70 + "\n")
        
        return self.models, self.metrics


if __name__ == "__main__":
    print("\n BeyondTech - Hybrid Model Training\n")
    
    trainer = MaintenanceModelTrainer()
    models, metrics = trainer.train_all_models()