# Artificial Neural Network (ANN)

**Category:** Deep Learning

---

## What It Does

This notebook builds a **feedforward ANN** using TensorFlow/Keras to predict customer churn — whether a bank customer will leave or stay. The network learns complex non-linear patterns in customer demographics and account activity that simpler models would miss.

---

## Network Architecture

| Layer | Units | Activation |
|---|---|---|
| Input + Hidden 1 | 6 | ReLU |
| Hidden 2 | 6 | ReLU |
| Output | 1 | Sigmoid |

- **Optimizer:** Adam
- **Loss:** Binary Cross-Entropy
- **Batch size:** 32 · **Epochs:** 100

---

## When To Use It

- Binary classification with many features and complex interactions
- When traditional ML models underfit the data
- For tabular data, ANNs work well but often don't outperform tree-based ensembles — XGBoost (notebook 27) is a strong competitor
- Scale data before training — ANNs are sensitive to feature magnitude

---

## Dataset

**File:** `Churn_Modelling.csv`

10,000 bank customers with 14 features.

| Key Features | Description |
|---|---|
| CreditScore | Customer credit score |
| Geography | France / Germany / Spain (one-hot encoded) |
| Age, Balance, Tenure | Account characteristics |
| NumOfProducts, IsActiveMember | Product and engagement metrics |
| Exited | 1 = churned, 0 = stayed — target variable |

---

## How to Run

```bash
pip install numpy pandas tensorflow scikit-learn
jupyter notebook artificial_neural_network.ipynb
```

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

- Training accuracy curve over 100 epochs
- Confusion matrix and accuracy on the test set
- Single-row prediction example — "Will this specific customer churn?"
