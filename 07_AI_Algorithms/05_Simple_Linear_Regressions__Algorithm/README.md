# Simple Linear Regression

**Category:** Regression

---

## What It Does

Simple Linear Regression models the relationship between **one independent variable** and one continuous output by fitting a straight line `y = b0 + b1*x` that minimises the sum of squared residuals (Ordinary Least Squares).

This is the foundational starting point for all regression — the simplest possible predictive model.

---

## When To Use It

- When there is a single continuous predictor variable
- When the relationship appears linear on a scatter plot
- As a baseline before adding more features or complexity
- Assumes: linearity, independence of errors, constant variance, normality of residuals

---

## Dataset

**File:** `Salary_Data.csv`

| Column | Description |
|---|---|
| YearsExperience | Years of professional experience |
| Salary | Annual salary (USD) — target variable |

30 samples. A classic beginner dataset with a strong linear relationship between experience and salary.

---

## How to Run

```bash
pip install numpy pandas matplotlib scikit-learn
jupyter notebook simple_linear_regression.ipynb
```

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

- Scatter plot with the fitted regression line
- Training set and test set visualisations
- Predicted vs actual salary comparison
