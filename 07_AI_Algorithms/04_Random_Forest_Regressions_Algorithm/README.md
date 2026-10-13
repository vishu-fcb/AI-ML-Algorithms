# Random Forest Regression

**Category:** Regression

---

## What It Does

Random Forest Regression builds an **ensemble of Decision Trees** — each trained on a random bootstrap sample of the data with a random subset of features. The final prediction is the average across all trees.

This bagging approach reduces variance and overfitting compared to a single Decision Tree, making it one of the most reliable off-the-shelf regression algorithms.

---

## When To Use It

- When you need a powerful non-linear model with minimal tuning
- When a single Decision Tree overfits
- When feature importances are useful (Random Forest naturally ranks features)
- Slower than a single tree — trade-off between accuracy and speed

---

## Dataset

**File:** `Position_Salaries.csv`

| Column | Description |
|---|---|
| Position | Job title |
| Level | Numeric level (1–10) |
| Salary | Annual salary (USD) — target variable |

The same small dataset used for Decision Tree, Polynomial, and SVR — making direct model comparisons easy.

---

## How to Run

```bash
pip install numpy pandas matplotlib scikit-learn
jupyter notebook random_forest_regression.ipynb
```

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

- High-resolution prediction curve showing how the ensemble smooths the step-wise Decision Tree output
- Prediction for a specific position level
- Comparison plot against individual tree predictions
