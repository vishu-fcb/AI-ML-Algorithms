# Model Selection — Grid Search & K-Fold Cross Validation

**Category:** Model Selection & Evaluation

---

## What It Does

This folder contains two notebooks that address **how to evaluate and tune** a trained model — not a new algorithm per se, but essential techniques used with every algorithm:

1. **K-Fold Cross Validation** — evaluates a model's generalisation ability by testing it on K different data splits, giving a more reliable accuracy estimate than a single train/test split.

2. **Grid Search** — systematically searches over a grid of hyperparameter combinations (using cross-validation at each step) to find the optimal configuration.

---

## Why These Matter

A single train/test split can give misleadingly optimistic or pessimistic accuracy depending on which samples end up in the test set. K-Fold solves this by averaging over K splits.

Grid Search solves the problem of manual hyperparameter tuning — instead of guessing values, it exhaustively tests every combination and picks the best.

---

## When To Use Them

- **K-Fold:** Always — use it instead of a single split when evaluating any model
- **Grid Search:** When you need to tune hyperparameters (C and gamma for SVM, max_depth and n_estimators for Random Forest, etc.)
- Use `GridSearchCV` from scikit-learn — it combines both techniques automatically

---

## Dataset

**File:** `Social_Network_Ads.csv`

| Column | Description |
|---|---|
| Age | User age |
| EstimatedSalary | Estimated annual salary (USD) |
| Purchased | 1 = bought, 0 = did not — target variable |

The same dataset used across all classification notebooks — here used to demonstrate optimal SVM tuning via Grid Search.

---

## How to Run

```bash
pip install numpy pandas matplotlib scikit-learn
# Run either notebook:
jupyter notebook k_fold_cross_validation.ipynb
jupyter notebook Grid_search.ipynb
```

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

**K-Fold:** Mean accuracy and standard deviation across K folds — a confidence interval for model performance.

**Grid Search:** Best hyperparameter combination and the best cross-validated accuracy achieved.
