# Random Forest Classification

**Category:** Classification

---

## What It Does

Random Forest Classification builds an **ensemble of Decision Trees** — each trained on a bootstrap sample of the data with a random feature subset. Final predictions are made by **majority vote** across all trees.

This ensemble approach dramatically reduces the overfitting that single Decision Trees suffer from while retaining their non-linear expressiveness.

---

## When To Use It

- General-purpose, high-accuracy classification with minimal tuning
- When a single Decision Tree overfits
- When feature importance ranking is needed
- Works well on both numerical and categorical features
- The `n_estimators` parameter controls the number of trees — more trees = more stable but slower

---

## Dataset

**File:** `Social_Network_Ads.csv`

| Column | Description |
|---|---|
| Age | User age |
| EstimatedSalary | Estimated annual salary (USD) |
| Purchased | 1 = bought, 0 = did not — target variable |

400 samples. Compare the smooth, complex boundary here against the single Decision Tree (notebook 07) to see how ensembling smooths the decision surface.

---

## How to Run

```bash
pip install numpy pandas matplotlib scikit-learn
jupyter notebook random_forest_classification.ipynb
```

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

- Decision boundary visualisation — smoother and more complex than a single tree
- Confusion matrix and accuracy score
- Typically the highest accuracy among the 7 classifiers in this collection
