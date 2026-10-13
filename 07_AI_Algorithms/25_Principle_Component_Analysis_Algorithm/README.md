# Principal Component Analysis (PCA)

**Category:** Dimensionality Reduction (Unsupervised)

---

## What It Does

PCA finds the directions of **maximum variance** in the data and projects it onto a lower-dimensional space (principal components). The first component captures the most variance, the second captures the most remaining variance, and so on.

This notebook compresses 13 wine features down to 2 principal components, then trains a classifier on the compressed data.

---

## When To Use It

- Reducing high-dimensional data for visualisation (2D or 3D)
- Removing correlated features (multicollinearity) before regression
- As a preprocessing step to speed up downstream algorithms
- When labels are not available (fully unsupervised) — use LDA instead when labels exist
- Requires feature scaling — PCA is sensitive to feature magnitude

**Explained variance:** Always check the cumulative explained variance ratio to decide how many components to keep (typically retain 95% of variance).

---

## Dataset

**File:** `Wine.csv`

178 wine samples with 13 chemical features, classified into 3 wine types.

Same dataset as LDA (notebook 24) and Kernel PCA (notebook 23) — run all three to compare dimensionality reduction approaches.

---

## How to Run

```bash
pip install numpy pandas matplotlib scikit-learn
jupyter notebook Principal_component_analysis.ipynb
```

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

- Explained variance ratio per component
- 2D scatter plot of wine classes after PCA compression
- Classification accuracy on the 2 principal components
