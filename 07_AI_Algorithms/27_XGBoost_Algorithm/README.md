# XGBoost

**Category:** Ensemble Learning (Boosting)

---

## What It Does

XGBoost (Extreme Gradient Boosting) builds an ensemble of **weak learners (shallow trees) sequentially**, where each new tree corrects the errors of all previous trees. It uses second-order gradient information (Newton's method) and regularisation to train faster and more accurately than traditional Gradient Boosting.

XGBoost is one of the most powerful algorithms for structured/tabular data and has won more Kaggle competitions than any other single algorithm.

---

## When To Use It

- Structured/tabular classification and regression
- When you need state-of-the-art performance with reasonable training speed
- When Random Forest or Gradient Boosting doesn't give good enough results
- Built-in regularisation (L1/L2) prevents overfitting better than vanilla Gradient Boosting
- Handles missing values natively

**XGBoost vs Random Forest:** Random Forest uses bagging (parallel trees, vote). XGBoost uses boosting (sequential trees, error correction). XGBoost typically achieves higher accuracy; Random Forest is simpler to tune.

---

## Dataset

**File:** `Data.csv`

Classification dataset with multiple features. The target is a binary class label.

---

## How to Run

```bash
pip install numpy pandas scikit-learn xgboost
jupyter notebook xg_boost.ipynb
```

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

- Accuracy on the test set
- K-Fold cross-validation score (mean ± standard deviation)
- The final benchmark to compare against all other algorithms in this collection
