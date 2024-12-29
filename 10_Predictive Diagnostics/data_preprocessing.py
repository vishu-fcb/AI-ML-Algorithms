# data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    """
    Loads and preprocesses the dataset, applies scaling, and splits it into training and testing sets.
    Returns preprocessed data and target mapping.
    """
    # Load the OBD-II dataset
    dataset = pd.read_csv(file_path, low_memory=False)

    # Select relevant columns for clustering and predictive diagnostics
    selected_features = [
        'ENGINE_RPM', 'SPEED', 'ENGINE_COOLANT_TEMP', 
        'INTAKE_MANIFOLD_PRESSURE', 'AIR_INTAKE_TEMP',
        'MAF', 'BAROMETRIC_PRESSURE(KPA)'
    ]

    dataset['TROUBLE_CODES'] = dataset['TROUBLE_CODES'].fillna('NO_FAULT')

    # Ensure selected columns are numeric
    for feature in selected_features:
        dataset[feature] = pd.to_numeric(dataset[feature], errors='coerce')

    # Handle missing values by filling them with the column mean
    dataset[selected_features] = dataset[selected_features].fillna(dataset[selected_features].mean())

    # Apply scaling to the selected features
    scaler = StandardScaler()
    dataset[selected_features] = scaler.fit_transform(dataset[selected_features])

    # Encode target variable
    target_mapping = {label: idx for idx, label in enumerate(dataset['TROUBLE_CODES'].unique())}
    dataset['TROUBLE_CODES'] = dataset['TROUBLE_CODES'].map(target_mapping)

    # Train/test split
    X = dataset[selected_features]
    y = dataset['TROUBLE_CODES']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Handle class imbalance using SMOTEENN
    smoteenn = SMOTEENN(smote=SMOTE(k_neighbors=3), random_state=42)
    X_train_resampled, y_train_resampled = smoteenn.fit_resample(X_train, y_train)

    return X_train_resampled, X_test, y_train_resampled, y_test, target_mapping
