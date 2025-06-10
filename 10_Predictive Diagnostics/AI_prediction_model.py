""" # main.py

from data_preprocessing import load_and_preprocess_data
from model import train_and_save_model
from visualization import plot_confusion_matrix, plot_feature_importance, plot_precision_recall_curve

def main():
    # Load and preprocess data
    X_train_resampled, X_test, y_train_resampled, y_test, target_mapping = load_and_preprocess_data('Predictive Diagnostics/OBD_dataset.csv')

    # Train and/or load the model and evaluate it
    model, y_pred, report = train_and_save_model(X_train_resampled, y_train_resampled, X_test, y_test)

    # Display classification report
    print("Updated Classification Report:")
    print(report)

    # Visualizations
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(model, ['ENGINE_RPM', 'SPEED', 'ENGINE_COOLANT_TEMP', 'INTAKE_MANIFOLD_PRESSURE', 'AIR_INTAKE_TEMP', 'MAF', 'BAROMETRIC_PRESSURE(KPA)'])
    plot_precision_recall_curve(y_test, y_pred, target_mapping)

if __name__ == "__main__":
    main()
 """


import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import joblib

def main():
    # Load data from Excel
    df_ev = pd.read_excel("Generated_output_data/PredictiveMaintenance_Combined.xlsx", sheet_name="EV_Range_Data")
    df_oil = pd.read_excel("Generated_output_data/PredictiveMaintenance_Combined.xlsx", sheet_name="Oil_Life_Data")

    # ------------ MODEL 1: EV Range ------------
    X_ev = df_ev.drop(columns=["Range_Left_km"])
    y_ev = df_ev["Range_Left_km"]

    X_train_ev, X_test_ev, y_train_ev, y_test_ev = train_test_split(X_ev, y_ev, test_size=0.2, random_state=42)

    model_ev = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
    model_ev.fit(X_train_ev, y_train_ev)
    y_pred_ev = model_ev.predict(X_test_ev)

    r2_ev = r2_score(y_test_ev, y_pred_ev)
    rmse_ev = np.sqrt(mean_squared_error(y_test_ev, y_pred_ev))
    print(f"[🔋 EV Range]     R²: {r2_ev:.2f} | RMSE: {rmse_ev:.2f}")

    # ------------ MODEL 2: Oil Life ------------
    X_oil = df_oil.drop(columns=["Oil_Change_In_km"])
    y_oil = df_oil["Oil_Change_In_km"]

    X_train_oil, X_test_oil, y_train_oil, y_test_oil = train_test_split(X_oil, y_oil, test_size=0.2, random_state=42)

    model_oil = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
    model_oil.fit(X_train_oil, y_train_oil)
    y_pred_oil = model_oil.predict(X_test_oil)

    r2_oil = r2_score(y_test_oil, y_pred_oil)
    rmse_oil = np.sqrt(mean_squared_error(y_test_oil, y_pred_oil))
    print(f"[🛢️ Oil Life]     R²: {r2_oil:.2f} | RMSE: {rmse_oil:.2f}")

    # Save models
    joblib.dump(model_ev, "Trained Model/battery_model.pkl")
    joblib.dump(model_oil, "Trained Model/oil_model.pkl")
    print("[✅] Models saved: battery_model.pkl and oil_model.pkl")

if __name__ == "__main__":
    main()
