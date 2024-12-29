# model.py

import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

MODEL_PATH = 'trained_model.pkl'  # Path to save/load the trained model

def train_and_save_model(X_train_resampled, y_train_resampled, X_test, y_test):
    """
    Trains and saves the model if not already saved. If the model is saved, it loads the model and makes predictions.
    """
    # Check if a trained model already exists
    if os.path.exists(MODEL_PATH):
        print("Loading pre-trained model...")
        model = joblib.load(MODEL_PATH)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
    else:
        # Hyperparameter tuning using GridSearchCV
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
        }
        rf = RandomForestClassifier(random_state=42, class_weight="balanced")
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=2)
        grid_search.fit(X_train_resampled, y_train_resampled)

        # Best parameters from GridSearchCV
        best_params = grid_search.best_params_

        # Train the model with the best parameters
        model = RandomForestClassifier(**best_params, random_state=42, class_weight="balanced")
        model.fit(X_train_resampled, y_train_resampled)

        # Save the trained model for future use
        joblib.dump(model, MODEL_PATH)

        # Predictions
        y_pred = model.predict(X_test)

        # Classification report
        report = classification_report(y_test, y_pred)

    return model, y_pred, report
