"""
BeyondTech AI Model - Phase 1 Critical Benchmarking Analysis
Comprehensive evaluation script for production readiness

Analyses Included:
1. Feature Importance Analysis
2. Residual Analysis (3 plots per model)
3. Detailed Cross-Validation (k=3, 5, 10)
4. Error Breakdown by Scenario (Critical/Warning/Good)
5. Statistical Summary Report

Run this after training models to validate production readiness.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import pickle
import os
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ModelBenchmarking:
    """Comprehensive benchmarking suite for predictive maintenance models"""
    
    def __init__(self, 
                 data_file="Generated_output_data/PredictiveMaintenance_Realistic_25K.xlsx",
                 model_dir="Trained Model",
                 output_dir="Benchmarking_Reports"):
        self.data_file = data_file
        self.model_dir = model_dir
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            'ev_range': {
                'file': 'ev_range_model.pkl',
                'sheet': 'EV_Range_Data',
                'features': ['SoC', 'SoH', 'Battery_Voltage', 'Battery_Temperature', 
                           'Driving_Speed', 'Load_Weight', 'Ambient_Temperature'],
                'target': 'Range_Left_km',
                'unit': 'km',
                'thresholds': {'critical': 50, 'warning': 150}
            },
            'oil_life': {
                'file': 'oil_life_model.pkl',
                'sheet': 'Oil_Life_Data',
                'features': ['Engine_Temperature', 'Engine_RPM', 'Load_Weight', 
                           'Distance_Since_Last_Change', 'Oil_Viscosity', 
                           'Ambient_Temperature', 'Idle_Time'],
                'target': 'Oil_Change_In_km',
                'unit': 'km',
                'thresholds': {'critical': 500, 'warning': 1500}
            },
            'tire_wear': {
                'file': 'tire_wear_model.pkl',
                'sheet': 'Tire_Wear_Data',
                'features': ['Total_Distance', 'Average_Speed', 'Tire_Pressure', 
                           'Load_Weight', 'Road_Type', 'Alignment_Score', 
                           'Tire_Age_Months', 'Harsh_Braking_Events', 'Temperature_Range'],
                'target': 'Tire_Tread_Depth_mm',
                'unit': 'mm',
                'thresholds': {'critical': 2.0, 'warning': 3.0}
            },
            'brake_pad': {
                'file': 'brake_pad_model.pkl',
                'sheet': 'Brake_Pad_Data',
                'features': ['Total_Distance', 'Distance_Since_Last_Replacement', 
                           'Average_Speed', 'Brake_Events_Per_100km', 'Load_Weight', 
                           'Driving_Style', 'Mountain_Driving_Percent', 
                           'Regenerative_Braking', 'Brake_Temperature_Avg'],
                'target': 'Brake_Pad_Thickness_mm',
                'unit': 'mm',
                'thresholds': {'critical': 3.0, 'warning': 5.0}
            },
            'battery_degradation': {
                'file': 'battery_degradation_model.pkl',
                'sheet': 'Battery_Degradation_Data',
                'features': ['Battery_Age_Months', 'Total_Charge_Cycles', 
                           'Fast_Charge_Percentage', 'Average_Depth_of_Discharge', 
                           'Battery_Temperature_Avg', 'Battery_Temperature_Range', 
                           'Total_Distance', 'Idle_Time_Percentage', 
                           'High_Speed_Percentage'],
                'target': 'Battery_SoH_Percentage',
                'unit': '%',
                'thresholds': {'critical': 70, 'warning': 80}
            },
            'coolant_health': {
                'file': 'coolant_health_model.pkl',
                'sheet': 'Coolant_Health_Data',
                'features': ['Coolant_Age_Months', 'Engine_Temperature_Avg', 
                           'Engine_Temperature_Max', 'Coolant_Level', 
                           'Total_Distance', 'Heavy_Load_Percentage', 
                           'Ambient_Temperature', 'Idle_Time_Percentage', 'Coolant_pH'],
                'target': 'Coolant_Change_In_km',
                'unit': 'km',
                'thresholds': {'critical': 5000, 'warning': 15000}
            },
            'air_filter': {
                'file': 'air_filter_model.pkl',
                'sheet': 'Air_Filter_Data',
                'features': ['Distance_Since_Filter_Change', 'Air_Quality_Index', 
                           'Dusty_Road_Percentage', 'Urban_Driving_Percentage', 
                           'Engine_Air_Flow', 'Idle_Time_Percentage', 
                           'Humidity_Avg', 'Filter_Type'],
                'target': 'Filter_Change_In_km',
                'unit': 'km',
                'thresholds': {'critical': 2000, 'warning': 8000}
            },
            'transmission_health': {
                'file': 'transmission_health_model.pkl',
                'sheet': 'Transmission_Health_Data',
                'features': ['Total_Distance', 'Distance_Since_Fluid_Change', 
                           'Transmission_Temperature', 'Gear_Shifts_Per_100km', 
                           'Harsh_Acceleration_Events', 'Towing_Percentage', 
                           'City_Driving_Percentage', 'Transmission_Type'],
                'target': 'Transmission_Fluid_Change_In_km',
                'unit': 'km',
                'thresholds': {'critical': 10000, 'warning': 30000}
            },
        }
        
        self.results = {}
    
    def load_model_and_data(self, model_name):
        """Load trained model and corresponding data"""
        config = self.model_configs[model_name]
        
        # Load model
        model_path = f"{self.model_dir}/{config['file']}"
        if not os.path.exists(model_path):
            print(f"[WARNING] Model not found: {model_path}")
            return None, None, None, None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load data
        df = pd.read_excel(self.data_file, sheet_name=config['sheet'])
        X = df[config['features']]
        y = df[config['target']]
        
        return model, X, y, config
    
    def analyze_feature_importance(self, model, feature_names, model_name):
        """Analyze and plot feature importance"""
        print(f"\n{'='*80}")
        print(f"FEATURE IMPORTANCE ANALYSIS: {model_name.replace('_', ' ').title()}")
        print('='*80)
        
        # Get feature importance
        importance = model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importance)[::-1]
        
        # Print ranking
        print("\nFeature Ranking:")
        print(f"{'Rank':<6} {'Feature':<35} {'Importance':<12} {'% of Total'}")
        print('-'*80)
        
        total_importance = importance.sum()
        cumulative = 0
        
        for rank, idx in enumerate(indices, 1):
            pct = (importance[idx] / total_importance) * 100
            cumulative += pct
            print(f"{rank:<6} {feature_names[idx]:<35} {importance[idx]:<12.4f} {pct:>6.2f}%")
            
            # Mark features that contribute 80% of importance
            if cumulative >= 80 and cumulative - pct < 80:
                print(f"       └─ Top features contributing 80% of predictive power")
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Bar plot
        plt.subplot(1, 2, 1)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
        plt.barh(range(len(feature_names)), importance[indices], color=colors)
        plt.yticks(range(len(feature_names)), [feature_names[i] for i in indices])
        plt.xlabel('Importance Score', fontsize=11, fontweight='bold')
        plt.title(f'Feature Importance - {model_name.replace("_", " ").title()}', 
                 fontsize=12, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Cumulative importance
        plt.subplot(1, 2, 2)
        cumulative_importance = np.cumsum(importance[indices]) / total_importance * 100
        plt.plot(range(1, len(feature_names) + 1), cumulative_importance, 
                marker='o', linewidth=2, markersize=8, color='#2E86AB')
        plt.axhline(y=80, color='r', linestyle='--', linewidth=2, label='80% threshold')
        plt.axhline(y=95, color='orange', linestyle='--', linewidth=2, label='95% threshold')
        plt.xlabel('Number of Features', fontsize=11, fontweight='bold')
        plt.ylabel('Cumulative Importance (%)', fontsize=11, fontweight='bold')
        plt.title('Cumulative Feature Importance', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/{model_name}_feature_importance.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store results
        return {
            'importance_scores': dict(zip(feature_names, importance)),
            'top_3_features': [feature_names[i] for i in indices[:3]],
            'features_for_80pct': np.sum(cumulative_importance <= 80) + 1
        }
    
    def analyze_residuals(self, y_true, y_pred, model_name, unit):
        """Comprehensive residual analysis with 3 plots"""
        print(f"\n{'='*80}")
        print(f"RESIDUAL ANALYSIS: {model_name.replace('_', ' ').title()}")
        print('='*80)
        
        residuals = y_true - y_pred
        
        # Calculate statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        print(f"\nResidual Statistics:")
        print(f"  Mean:               {mean_residual:>10.4f} {unit}")
        print(f"  Std Deviation:      {std_residual:>10.4f} {unit}")
        print(f"  Min Error:          {np.min(residuals):>10.4f} {unit}")
        print(f"  Max Error:          {np.max(residuals):>10.4f} {unit}")
        print(f"  Median Abs Error:   {np.median(np.abs(residuals)):>10.4f} {unit}")
        
        # Normality test
        _, p_value = stats.normaltest(residuals)
        print(f"\nNormality Test (Anderson-Darling):")
        print(f"  p-value: {p_value:.4f}")
        if p_value > 0.05:
            print(f"  [OK] Residuals are normally distributed (p > 0.05)")
        else:
            print(f"  [WARNING] Residuals may not be normally distributed (p < 0.05)")
        
        # Create 3-plot visualization
        fig = plt.figure(figsize=(18, 5))
        
        # Plot 1: Residuals vs Predicted
        ax1 = plt.subplot(1, 3, 1)
        plt.scatter(y_pred, residuals, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        plt.axhline(y=mean_residual, color='g', linestyle='-', linewidth=1.5, 
                   label=f'Mean: {mean_residual:.2f}')
        plt.axhline(y=2*std_residual, color='orange', linestyle=':', linewidth=1.5, 
                   label=f'±2σ: {2*std_residual:.2f}')
        plt.axhline(y=-2*std_residual, color='orange', linestyle=':', linewidth=1.5)
        plt.xlabel(f'Predicted Values ({unit})', fontsize=11, fontweight='bold')
        plt.ylabel(f'Residuals ({unit})', fontsize=11, fontweight='bold')
        plt.title('Residual Plot\n(Should show random scatter)', fontsize=12, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Histogram of Residuals
        ax2 = plt.subplot(1, 3, 2)
        plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        plt.axvline(x=mean_residual, color='g', linestyle='-', linewidth=1.5, 
                   label=f'Mean: {mean_residual:.2f}')
        plt.xlabel(f'Residual Error ({unit})', fontsize=11, fontweight='bold')
        plt.ylabel('Frequency', fontsize=11, fontweight='bold')
        plt.title('Residual Distribution\n(Should be bell-shaped)', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Plot 3: Q-Q Plot
        ax3 = plt.subplot(1, 3, 3)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot\n(Check Normality)', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/{model_name}_residual_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Check for patterns
        issues = []
        if abs(mean_residual) > std_residual / 2:
            issues.append("Mean residual not close to zero (possible bias)")
        if p_value < 0.05:
            issues.append("Residuals not normally distributed")
        
        if issues:
            print(f"\n[WARNING] Potential Issues Detected:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"\n[OK] No major issues detected in residual analysis")
        
        return {
            'mean_residual': mean_residual,
            'std_residual': std_residual,
            'normality_p_value': p_value,
            'issues': issues
        }
    
    def detailed_cross_validation(self, model, X, y, model_name):
        """Perform cross-validation with different k values"""
        print(f"\n{'='*80}")
        print(f"DETAILED CROSS-VALIDATION: {model_name.replace('_', ' ').title()}")
        print('='*80)
        
        k_values = [3, 5, 10]
        cv_results = {}
        
        print(f"\n{'K-Fold':<10} {'Mean R²':<15} {'Std Dev':<15} {'Min R²':<15} {'Max R²'}")
        print('-'*80)
        
        for k in k_values:
            kfold = KFold(n_splits=k, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
            
            cv_results[f'{k}-fold'] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max(),
                'scores': scores
            }
            
            print(f"{k}-Fold    {scores.mean():<15.4f} {scores.std():<15.4f} "
                  f"{scores.min():<15.4f} {scores.max():<15.4f}")
        
        # Stability check
        print(f"\nStability Analysis:")
        for k in k_values:
            std = cv_results[f'{k}-fold']['std']
            if std < 0.005:
                status = "[EXCELLENT]"
            elif std < 0.01:
                status = "[GOOD]"
            elif std < 0.02:
                status = "[ACCEPTABLE]"
            else:
                status = "[HIGH VARIANCE]"
            
            print(f"  {k}-Fold: Std = {std:.4f} - {status}")
        
        # Visualization
        plt.figure(figsize=(12, 6))
        
        # Box plot
        plt.subplot(1, 2, 1)
        data_to_plot = [cv_results[f'{k}-fold']['scores'] for k in k_values]
        bp = plt.boxplot(data_to_plot, labels=[f'{k}-Fold' for k in k_values],
                        patch_artist=True, showmeans=True)
        
        # Color boxes
        colors = ['lightblue', 'lightgreen', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.ylabel('R² Score', fontsize=11, fontweight='bold')
        plt.title('Cross-Validation Score Distribution', fontsize=12, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Score comparison
        plt.subplot(1, 2, 2)
        x = range(len(k_values))
        means = [cv_results[f'{k}-fold']['mean'] for k in k_values]
        stds = [cv_results[f'{k}-fold']['std'] for k in k_values]
        
        plt.errorbar(x, means, yerr=stds, fmt='o-', linewidth=2, markersize=10,
                    capsize=5, capthick=2, color='#2E86AB')
        plt.xticks(x, [f'{k}-Fold' for k in k_values])
        plt.ylabel('Mean R² Score', fontsize=11, fontweight='bold')
        plt.title('Mean Score with Standard Deviation', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/{model_name}_cross_validation.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return cv_results
    
    def error_breakdown_by_scenario(self, y_true, y_pred, model_name, thresholds, unit):
        """Analyze performance on critical/warning/good scenarios"""
        print(f"\n{'='*80}")
        print(f"ERROR BREAKDOWN BY SCENARIO: {model_name.replace('_', ' ').title()}")
        print('='*80)
        
        # For models where lower is worse (tire, brake, battery, oil that's about to change)
        # we need to invert the logic
        if model_name in ['tire_wear', 'brake_pad', 'battery_degradation']:
            critical_mask = y_true <= thresholds['critical']
            warning_mask = (y_true > thresholds['critical']) & (y_true <= thresholds['warning'])
            good_mask = y_true > thresholds['warning']
        else:
            # For models where lower is worse (km until service)
            critical_mask = y_true <= thresholds['critical']
            warning_mask = (y_true > thresholds['critical']) & (y_true <= thresholds['warning'])
            good_mask = y_true > thresholds['warning']
        
        scenarios = {
            'Critical': critical_mask,
            'Warning': warning_mask,
            'Good': good_mask
        }
        
        results = {}
        
        print(f"\n{'Scenario':<15} {'Count':<10} {'% Total':<12} {'R² Score':<15} {'RMSE ({unit})':<20} {'MAE ({unit})'}")
        print('-'*100)
        
        for scenario_name, mask in scenarios.items():
            if mask.sum() > 0:
                y_scenario = y_true[mask]
                pred_scenario = y_pred[mask]
                
                r2 = r2_score(y_scenario, pred_scenario)
                rmse = np.sqrt(mean_squared_error(y_scenario, pred_scenario))
                mae = mean_absolute_error(y_scenario, pred_scenario)
                count = mask.sum()
                pct = (count / len(y_true)) * 100
                
                results[scenario_name] = {
                    'count': count,
                    'percentage': pct,
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae
                }
                
                print(f"{scenario_name:<15} {count:<10} {pct:<12.1f} {r2:<15.4f} {rmse:<20.4f} {mae:<15.4f}")
            else:
                print(f"{scenario_name:<15} {'0':<10} {'0.0':<12} {'N/A':<15} {'N/A':<20} {'N/A'}")
        
        # Check for issues
        print(f"\nScenario Performance Assessment:")
        
        if 'Critical' in results:
            if results['Critical']['r2'] < 0.85:
                print(f"  >> Critical scenario R^2 is low ({results['Critical']['r2']:.4f})")
                print(f"     Model may struggle with edge cases - consider more critical samples")
            else:
                print(f"  >> Critical scenario R^2 is good ({results['Critical']['r2']:.4f})")
        
        if 'Good' in results and 'Critical' in results:
            r2_diff = results['Good']['r2'] - results['Critical']['r2']
            if r2_diff > 0.05:
                print(f"  >> Large R^2 gap between Good and Critical scenarios ({r2_diff:.4f})")
                print(f"     Model performs better on easy cases")
            else:
                print(f"  >> Consistent performance across scenarios (R^2 diff: {r2_diff:.4f})")
        
        # Visualization
        if len(results) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # R² comparison
            ax1 = axes[0]
            scenarios_list = list(results.keys())
            r2_scores = [results[s]['r2'] for s in scenarios_list]
            colors_map = {'Critical': '#FF6B6B', 'Warning': '#FFB366', 'Good': '#4ECDC4'}
            colors = [colors_map.get(s, 'gray') for s in scenarios_list]
            
            bars = ax1.bar(scenarios_list, r2_scores, color=colors, edgecolor='black', linewidth=1.5)
            ax1.set_ylabel('R² Score', fontsize=11, fontweight='bold')
            ax1.set_title('R² Score by Scenario', fontsize=12, fontweight='bold')
            ax1.set_ylim([min(r2_scores) - 0.02, 1.0])
            ax1.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, score in zip(bars, r2_scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
            
            # Sample distribution
            ax2 = axes[1]
            counts = [results[s]['count'] for s in scenarios_list]
            percentages = [results[s]['percentage'] for s in scenarios_list]
            
            bars = ax2.bar(scenarios_list, counts, color=colors, edgecolor='black', linewidth=1.5)
            ax2.set_ylabel('Sample Count', fontsize=11, fontweight='bold')
            ax2.set_title('Sample Distribution by Scenario', fontsize=12, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            
            # Add percentage labels
            for bar, count, pct in zip(bars, counts, percentages):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/plots/{model_name}_scenario_breakdown.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        return results
    
    def benchmark_model(self, model_name):
        """Run complete benchmarking suite for a single model"""
        print(f"\n\n{'#'*80}")
        print(f"# BENCHMARKING: {model_name.replace('_', ' ').upper()}")
        print(f"{'#'*80}")
        
        # Load model and data
        model, X, y, config = self.load_model_and_data(model_name)
        
        if model is None:
            print(f"[SKIP] Skipping {model_name} - model not found")
            return None
        
        # IMPORTANT: Evaluate on FULL dataset, not a new train/test split
        # The model was already trained with a specific split, so we evaluate
        # on all available data to get comprehensive metrics
        # (Cross-validation will test generalization separately)
        
        # Make predictions on full dataset
        y_pred_full = model.predict(X)
        
        # Calculate basic metrics on full dataset
        r2 = r2_score(y, y_pred_full)
        rmse = np.sqrt(mean_squared_error(y, y_pred_full))
        mae = mean_absolute_error(y, y_pred_full)
        
        print(f"\nBasic Metrics (evaluated on full dataset):")
        print(f"  Note: R^2 on full dataset shows overall fit")
        print(f"  Note: Cross-validation (below) tests generalization")
        print(f"  R^2 Score:  {r2:.4f}")
        print(f"  RMSE:      {rmse:.4f} {config['unit']}")
        print(f"  MAE:       {mae:.4f} {config['unit']}")
        
        # Run analyses
        results = {
            'model_name': model_name,
            'basic_metrics': {
                'r2': r2,
                'rmse': rmse,
                'mae': mae
            }
        }
        
        # 1. Feature Importance
        results['feature_importance'] = self.analyze_feature_importance(
            model, config['features'], model_name
        )
        
        # 2. Residual Analysis (on full dataset)
        results['residual_analysis'] = self.analyze_residuals(
            y, y_pred_full, model_name, config['unit']
        )
        
        # 3. Cross-Validation (this tests generalization properly)
        results['cross_validation'] = self.detailed_cross_validation(
            model, X, y, model_name
        )
        
        # 4. Error Breakdown by Scenario (on full dataset)
        results['scenario_breakdown'] = self.error_breakdown_by_scenario(
            y, y_pred_full, model_name, config['thresholds'], config['unit']
        )
        
        self.results[model_name] = results
        
        return results
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print(f"\n\n{'='*80}")
        print("COMPREHENSIVE BENCHMARKING SUMMARY REPORT")
        print('='*80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Models Analyzed: {len(self.results)}")
        
        # Create summary table
        summary_data = []
        
        for model_name, results in self.results.items():
            metrics = results['basic_metrics']
            cv = results['cross_validation']['5-fold']
            fi = results['feature_importance']
            
            summary_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'R² Score': f"{metrics['r2']:.4f}",
                'RMSE': f"{metrics['rmse']:.2f}",
                'MAE': f"{metrics['mae']:.2f}",
                'CV Mean': f"{cv['mean']:.4f}",
                'CV Std': f"{cv['std']:.4f}",
                'Top Feature': fi['top_3_features'][0],
                'Features (80%)': fi['features_for_80pct']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        print(f"\n{summary_df.to_string(index=False)}")
        
        # Overall statistics
        print(f"\n{'='*80}")
        print("OVERALL STATISTICS")
        print('='*80)
        
        all_r2 = [r['basic_metrics']['r2'] for r in self.results.values()]
        all_cv_std = [r['cross_validation']['5-fold']['std'] for r in self.results.values()]
        
        print(f"\nR² Scores:")
        print(f"  Average:    {np.mean(all_r2):.4f}")
        print(f"  Min:        {np.min(all_r2):.4f}")
        print(f"  Max:        {np.max(all_r2):.4f}")
        print(f"  Std Dev:    {np.std(all_r2):.4f}")
        
        print(f"\nCross-Validation Stability:")
        print(f"  Avg Std:    {np.mean(all_cv_std):.4f}")
        print(f"  Max Std:    {np.max(all_cv_std):.4f}")
        
        # Count issues
        total_issues = sum(len(r['residual_analysis']['issues']) for r in self.results.values())
        
        print(f"\nQuality Assessment:")
        print(f"  Models with R^2 >= 0.98:   {sum(1 for r2 in all_r2 if r2 >= 0.98)}/{len(all_r2)}")
        print(f"  Models with R^2 >= 0.96:   {sum(1 for r2 in all_r2 if r2 >= 0.96)}/{len(all_r2)}")
        print(f"  Models with R^2 >= 0.92:   {sum(1 for r2 in all_r2 if r2 >= 0.92)}/{len(all_r2)}")
        print(f"  Models with CV std < 0.01: {sum(1 for std in all_cv_std if std < 0.01)}/{len(all_cv_std)}")
        print(f"  Total residual issues:   {total_issues}")
        
        if all(r2 >= 0.92 for r2 in all_r2) and all(std < 0.02 for std in all_cv_std):
            print(f"\n[PRODUCTION READY] All models meet quality thresholds!")
        else:
            print(f"\n[REVIEW NEEDED] Some models may need improvement")
        
        # Save to file
        report_file = f"{self.output_dir}/benchmarking_summary_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("BEYONDTECH AI MODEL BENCHMARKING REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(summary_df.to_string(index=False))
            f.write("\n\n" + "="*80 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("="*80 + "\n")
            f.write(f"\nAverage R-squared: {np.mean(all_r2):.4f}\n")
            f.write(f"Models >= 0.98: {sum(1 for r2 in all_r2 if r2 >= 0.98)}/{len(all_r2)}\n")
            f.write(f"Models >= 0.96: {sum(1 for r2 in all_r2 if r2 >= 0.96)}/{len(all_r2)}\n")
            f.write(f"Models >= 0.92: {sum(1 for r2 in all_r2 if r2 >= 0.92)}/{len(all_r2)}\n")
            f.write(f"Avg CV Std: {np.mean(all_cv_std):.4f}\n")
            f.write(f"Total Issues: {total_issues}\n")
        
        print(f"\n[REPORT] Summary report saved to: {report_file}")
        
        return summary_df
    
    def run_all_benchmarks(self):
        """Run benchmarking for all models"""
        print("="*80)
        print("BEYONDTECH AI MODEL - PHASE 1 CRITICAL BENCHMARKING")
        print("="*80)
        print(f"\nAnalyses to be performed:")
        print("  1. Feature Importance Analysis")
        print("  2. Residual Analysis (3 plots)")
        print("  3. Detailed Cross-Validation (k=3, 5, 10)")
        print("  4. Error Breakdown by Scenario")
        print(f"\nOutput Directory: {self.output_dir}")
        print(f"Total Models: {len(self.model_configs)}")
        
        input("\nPress Enter to begin benchmarking...")
        
        # Benchmark each model
        for model_name in self.model_configs.keys():
            try:
                self.benchmark_model(model_name)
            except Exception as e:
                print(f"\n[ERROR] Error benchmarking {model_name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Generate summary
        if self.results:
            summary_df = self.generate_summary_report()
            
            # Save summary CSV
            summary_df.to_csv(f"{self.output_dir}/benchmarking_summary.csv", index=False)
            print(f"[INFO] Summary CSV saved to: {self.output_dir}/benchmarking_summary.csv")
        
        print(f"\n{'='*80}")
        print("BENCHMARKING COMPLETE!")
        print('='*80)
        print(f"\n[INFO] All results saved to: {self.output_dir}/")
        print(f"[INFO] Plots generated: {self.output_dir}/plots/")
        print(f"\nGenerated files:")
        print(f"  - benchmarking_summary_report.txt")
        print(f"  - benchmarking_summary.csv")
        print(f"  - plots/ (24 visualization files)")
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print("\n[START] BeyondTech - Phase 1 Critical Benchmarking Analysis\n")
    
    # Initialize benchmarking
    benchmarker = ModelBenchmarking(
        data_file="Generated_output_data/PredictiveMaintenance_Realistic_25K.xlsx",
        model_dir="Trained Model",
        output_dir="Benchmarking_Reports"
    )
    
    # Run all benchmarks
    benchmarker.run_all_benchmarks()
    
    print("[COMPLETE] Benchmarking analysis complete!")
    print("           Review the reports in Benchmarking_Reports/ directory\n")