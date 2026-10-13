# K-Nearest Neighbors (KNN) Classification

**Category:** Classification

---

## What It Does

K-Nearest Neighbors is a **lazy, instance-based** classifier — it makes no assumptions about the data distribution and stores all training samples. At prediction time, it finds the K closest training points (by Euclidean distance) and assigns the majority class among them.

Simple, intuitive, and surprisingly effective for low-dimensional data.

---

## When To Use It

- When the decision boundary is irregular and hard to parameterise
- When interpretability at the local level matters ("this point is similar to these neighbours")
- For small-to-medium datasets — KNN has no training phase but slow prediction as data grows
- Requires feature scaling — distance-based, so feature magnitudes matter
- Avoid for high-dimensional data (curse of dimensionality)

**Choosing K:** Small K → complex boundary, overfits. Large K → smooth boundary, underfits. Use cross-validation.

---

## Dataset

**File:** `Social_Network_Ads.csv`

| Column | Description |
|---|---|
| Age | User age |
| EstimatedSalary | Estimated annual salary (USD) |
| Purchased | 1 = bought, 0 = did not — target variable |

400 samples. Feature scaling is applied before training (mandatory for KNN).

---

## How to Run

```bash
pip install numpy pandas matplotlib scikit-learn
jupyter notebook k_nearest_neighbors.ipynb
```

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

- Decision boundary visualisation showing the "blobby", locally-adaptive boundary KNN produces
- Confusion matrix and accuracy score
