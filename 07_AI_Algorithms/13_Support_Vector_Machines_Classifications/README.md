# Support Vector Machine (SVM) Classification

**Category:** Classification

---

## What It Does

SVM finds the **maximum-margin hyperplane** that best separates two classes. The decision boundary is determined only by the training points closest to it — the **support vectors** — making the model compact and resistant to noise from distant points.

This notebook uses a **linear kernel**, showing SVM's simplest form. Compare with notebook 08 (Kernel SVM) to see how adding a kernel enables non-linear boundaries.

---

## When To Use It

- When classes are linearly separable (linear kernel) or nearly so
- For high-dimensional data — SVMs work well in spaces where dimensions >> samples (e.g. text)
- When a clean, maximum-margin boundary matters
- Kernel SVM (notebook 08) handles non-linear cases — use that when a straight boundary doesn't fit

---

## Dataset

**File:** `Social_Network_Ads.csv`

| Column | Description |
|---|---|
| Age | User age |
| EstimatedSalary | Estimated annual salary (USD) |
| Purchased | 1 = bought, 0 = did not — target variable |

400 samples. This is the same dataset as notebooks 07–12 — running all of them lets you visually compare decision boundaries across all classification algorithms.

---

## How to Run

```bash
pip install numpy pandas matplotlib scikit-learn
jupyter notebook support_vector_machine.ipynb
```

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

- Linear decision boundary plot — the maximum-margin line between the two classes
- Confusion matrix and accuracy score
