# Decision Tree Regression

**Category:** Regression

---

## What It Does

Decision Tree Regression splits the training data into regions based on feature thresholds, assigning the mean output value of each region as the prediction. Unlike linear models, it captures **non-linear relationships** and step-wise patterns without requiring feature scaling.

The tree is built by recursively choosing the split that minimises variance in the output — resulting in a piecewise constant function.

---

## When To Use It

- When the relationship between input and output is non-linear
- When you need an **interpretable, visualisable** model
- As a building block for ensemble methods (Random Forest, Gradient Boosting)
- Avoid when data is very noisy — Decision Trees overfit easily without pruning

---

## Dataset

**File:** `Position_Salaries.csv`

| Column | Description |
|---|---|
| Position | Job title (CEO, Manager, etc.) |
| Level | Numeric level (1–10) |
| Salary | Annual salary (USD) — target variable |

10 rows — a deliberately small dataset that shows how Decision Trees can fit complex curves.

---

## How to Run

```bash
pip install numpy pandas matplotlib scikit-learn
jupyter notebook decision_tree_regression.ipynb
```

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

- Scatter plot of actual salaries vs Decision Tree predictions
- Visual comparison with other regression models at different resolutions
- Prediction for a specific position level
