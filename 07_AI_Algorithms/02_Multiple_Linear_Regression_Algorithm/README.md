# Multiple Linear Regression

**Category:** Regression

---

## What It Does

Multiple Linear Regression extends Simple Linear Regression to model the relationship between **multiple independent variables** and a single continuous output. It fits a hyperplane `y = b0 + b1*x1 + b2*x2 + ... + bn*xn` to minimise the sum of squared residuals.

This notebook also demonstrates **Backward Elimination** — an automatic feature selection technique that removes statistically insignificant predictors (p-value > 0.05) step by step.

---

## When To Use It

- When output depends on several input features simultaneously
- When the relationship is approximately linear
- As a baseline before trying more complex models
- Requires checking assumptions: linearity, independence of errors, homoscedasticity, no multicollinearity

---

## Dataset

**File:** `50_Startups.csv`

| Column | Description |
|---|---|
| R&D Spend | R&D budget (USD) |
| Administration | Admin costs (USD) |
| Marketing Spend | Marketing budget (USD) |
| State | US state (categorical) |
| Profit | Annual profit (USD) — target variable |

50 startup companies. Categorical State column is one-hot encoded before training.

---

## How to Run

```bash
pip install numpy pandas matplotlib scikit-learn statsmodels
jupyter notebook multiple_linear_regression.ipynb
```

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

- Trained model with coefficients for each feature
- Predicted vs actual profit comparison
- Backward Elimination result showing which features are significant
