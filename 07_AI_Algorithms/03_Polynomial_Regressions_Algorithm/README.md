# Polynomial Regression

**Category:** Regression

---

## What It Does

Polynomial Regression fits a **curved (non-linear) relationship** between the input and output by introducing polynomial features (x², x³, etc.) while still using Linear Regression internally. It is a flexible upgrade to Simple Linear Regression when the data follows a curve rather than a straight line.

This notebook fits both a Linear and a Polynomial model to the same dataset, letting you directly compare how each handles a curved distribution.

---

## When To Use It

- When a scatter plot shows a clear curved trend (quadratic, cubic, etc.)
- When Linear Regression underfits the data
- Choose the polynomial degree carefully — too high a degree causes overfitting

---

## Dataset

**File:** `Position_Salaries.csv`

| Column | Description |
|---|---|
| Position | Job title |
| Level | Numeric level (1–10) |
| Salary | Annual salary (USD) — target variable |

10 rows. Salaries grow non-linearly with level, making this ideal for demonstrating why polynomial features outperform a straight line here.

---

## How to Run

```bash
pip install numpy pandas matplotlib scikit-learn
jupyter notebook polynomial_regression.ipynb
```

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

- Side-by-side plots: Linear Regression vs Polynomial Regression fit
- Prediction for a specific position level using both models
- Visual illustration of underfitting (linear) vs good fit (polynomial)
