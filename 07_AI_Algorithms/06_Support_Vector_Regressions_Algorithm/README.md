# Support Vector Regression (SVR)

**Category:** Regression

---

## What It Does

Support Vector Regression applies the SVM principle to regression tasks. Instead of fitting a line that minimises residuals, SVR finds a function that stays within an **epsilon tube** around the data points. Only points outside the tube (support vectors) influence the model.

This makes SVR **robust to outliers** and effective on small datasets with complex non-linear patterns when used with a kernel (RBF in this notebook).

---

## When To Use It

- When the dataset is small and the relationship is non-linear
- When robustness to outliers is important
- Requires **feature scaling** — SVR is sensitive to the scale of input features
- Not ideal for large datasets (training time scales poorly)

---

## Dataset

**File:** `Position_Salaries.csv`

| Column | Description |
|---|---|
| Position | Job title |
| Level | Numeric level (1–10) |
| Salary | Annual salary (USD) — target variable |

This notebook shows why feature scaling is essential for SVR — both X and Y are standardised using `StandardScaler` before training and then inverse-transformed for output.

---

## How to Run

```bash
pip install numpy pandas matplotlib scikit-learn
jupyter notebook support_vector_regression.ipynb
```

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

- SVR prediction curve vs actual data points
- Demonstration of feature scaling (StandardScaler) applied to both input and output
- Prediction for a specific position level (inverse-scaled back to USD)
