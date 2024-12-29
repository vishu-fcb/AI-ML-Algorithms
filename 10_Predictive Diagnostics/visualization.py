# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, auc

def plot_confusion_matrix(y_test, y_pred):
    """
    Plots the confusion matrix for the given predictions and true values.
    """
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()

def plot_feature_importance(model, selected_features):
    """
    Plots the feature importance of the trained model.
    """
    feature_importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=selected_features, palette="viridis")
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.show()

def plot_precision_recall_curve(y_test, y_pred, target_mapping):
    """
    Plots precision-recall curves for each class.
    """
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(target_mapping.keys()):
        y_test_bin = (y_test == i).astype(int)
        y_score_bin = (y_pred == i).astype(int)
        precision, recall, _ = precision_recall_curve(y_test_bin, y_score_bin)
        plt.plot(recall, precision, label=f"{label} (AUC = {auc(recall, precision):.2f})")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.grid()
    plt.show()
