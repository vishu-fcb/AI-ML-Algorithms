# K-Means Clustering

**Category:** Clustering (Unsupervised Learning)

---

## What It Does

K-Means partitions data into **K clusters** by iteratively assigning each point to its nearest centroid and then updating centroids to the mean of their assigned points. The process repeats until cluster assignments stabilise.

It is the most widely used clustering algorithm — fast, simple, and effective for well-separated, roughly spherical clusters.

---

## When To Use It

- When you know (or can estimate) the number of clusters
- For large datasets — K-Means scales well (O(n))
- When clusters are roughly equal in size and shape
- Avoid when clusters have irregular shapes or widely different densities — try DBSCAN instead

**Choosing K:** Use the **Elbow Method** — plot inertia (within-cluster sum of squares) vs K and look for the "elbow" where adding more clusters gives diminishing returns.

---

## Dataset

**File:** `Mall_Customers.csv`

| Column | Description |
|---|---|
| CustomerID | Unique ID |
| Gender | Male / Female |
| Age | Customer age |
| Annual Income (k$) | Annual income in thousands USD |
| Spending Score (1-100) | Mall-assigned spending behaviour score |

200 customers. The Elbow Method is applied to find the optimal number of customer segments.

---

## How to Run

```bash
pip install numpy pandas matplotlib scikit-learn
jupyter notebook k_means_clustering.ipynb
```

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

- Elbow Method plot — visually identifies the optimal K (typically 5 for this dataset)
- Cluster scatter plot with colour-coded segments and centroid markers
- Comparison with Hierarchical Clustering (notebook 14)
