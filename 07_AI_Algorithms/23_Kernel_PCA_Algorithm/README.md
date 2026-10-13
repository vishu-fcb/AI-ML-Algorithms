# Kernel PCA — Dimensionality Reduction

**Category:** Dimensionality Reduction

---

## What It Does

Kernel PCA extends standard PCA by applying the **kernel trick** to capture non-linear structure in high-dimensional data. It maps features into a higher-dimensional space using a kernel function (RBF in this notebook) and then performs PCA in that space — allowing it to find non-linear principal components.

This notebook reduces wine features to 2 principal components and trains a classifier on the reduced data to demonstrate that the reduced representation retains discriminative information.

---

## When To Use It

- When standard PCA fails because the data's structure is non-linear
- High-dimensional datasets where visualisation in 2D or 3D is needed
- As a preprocessing step before classification or clustering
- More computationally expensive than linear PCA — use on small-to-medium datasets

**PCA vs Kernel PCA vs LDA:** PCA (unsupervised, linear), Kernel PCA (unsupervised, non-linear), LDA (supervised, maximises class separation). Run notebooks 23–25 on the same Wine dataset to compare directly.

---

## Dataset

**File:** `Wine.csv`

178 wine samples with 13 chemical features, classified into 3 wine types.

Used in notebooks 23, 24, and 25 — running all three lets you compare the effect of PCA, LDA, and Kernel PCA on the same data.

---

## How to Run

```bash
pip install numpy pandas matplotlib scikit-learn
jupyter notebook Kernel_pca.ipynb
```

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

- 2D scatter plot showing the three wine classes after Kernel PCA compression
- Classification accuracy on the reduced 2-feature representation
