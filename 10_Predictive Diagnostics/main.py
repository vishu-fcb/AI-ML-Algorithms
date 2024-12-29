# main.py

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
