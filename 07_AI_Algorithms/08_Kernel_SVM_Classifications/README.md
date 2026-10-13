# Kernel SVM Classification

**Category:** Classification

---

## What It Does

Kernel SVM uses the **kernel trick** to implicitly map data into a higher-dimensional space where it becomes linearly separable — without ever computing the transformation explicitly. The RBF (Radial Basis Function) kernel is the most popular and is used in this notebook.

The result is a flexible, non-linear decision boundary that often outperforms linear classifiers on real-world data.

---

## When To Use It

- When classes are not linearly separable in the original feature space
- When you have a moderate-sized dataset (SVMs scale poorly to very large datasets)
- Requires feature scaling — SVMs maximise the margin, which is affected by feature magnitude
- The `C` parameter controls the trade-off between a smooth boundary and correct classification

---

## Dataset

**File:** `Social_Network_Ads.csv`

| Column | Description |
|---|---|
| Age | User age |
| EstimatedSalary | Estimated annual salary (USD) |
| Purchased | 1 = bought, 0 = did not — target variable |

400 samples. Compare the smooth, curved decision boundary here against the rectangular boundary of Decision Trees (notebook 07) on the same data.

---

## How to Run

```bash
pip install numpy pandas matplotlib scikit-learn
jupyter notebook kernel_svm.ipynb
```

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

- Non-linear decision boundary plot (training and test sets)
- Confusion matrix and accuracy score
- The characteristic smooth, curved boundary that kernel SVMs produce
