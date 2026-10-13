# Linear Discriminant Analysis (LDA)

**Category:** Dimensionality Reduction (Supervised)

---

## What It Does

LDA is a **supervised** dimensionality reduction technique that finds the axes that **maximise class separation** — not just variance (as PCA does). It projects data onto a lower-dimensional subspace where the classes are most distinguishable.

Since LDA uses class labels, it is often a better preprocessing step than PCA when the goal is classification.

---

## When To Use It

- Before classification — LDA's reduced features are optimised for separating classes
- Multi-class problems (LDA produces at most C-1 components for C classes)
- When you want both dimensionality reduction AND interpretable class boundaries
- LDA assumes Gaussian class distributions and equal covariances — check these assumptions on your data

**PCA vs LDA:** PCA maximises overall variance (ignores labels). LDA maximises between-class variance relative to within-class variance. LDA typically outperforms PCA as a classification preprocessor when its assumptions hold.

---

## Dataset

**File:** `Wine.csv`

178 wine samples with 13 chemical features, classified into 3 wine types.

With 3 classes, LDA produces at most 2 discriminant components — a natural 2D projection.

---

## How to Run

```bash
pip install numpy pandas matplotlib scikit-learn
jupyter notebook Linear_discriminant_analysis.ipynb
```

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

- 2D scatter plot of wine classes after LDA — classes are more cleanly separated than PCA on the same data
- Classification accuracy on the 2 LDA components
