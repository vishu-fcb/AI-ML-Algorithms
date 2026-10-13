# Hierarchical Clustering

**Category:** Clustering (Unsupervised Learning)

---

## What It Does

Hierarchical Clustering builds a **tree of clusters (dendrogram)** by iteratively merging the two closest data points or clusters (agglomerative approach). The optimal number of clusters is determined by cutting the dendrogram at the largest vertical gap — no need to specify K upfront.

This makes it particularly useful for **exploratory analysis** when you don't know how many groups exist.

---

## When To Use It

- When you don't know the number of clusters in advance
- When understanding the **hierarchy of relationships** between groups is valuable
- For smaller datasets — agglomerative clustering scales as O(n³)
- Use the dendrogram to choose the number of clusters objectively

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

200 mall customers. The task is to discover natural customer segments based on income and spending behaviour — classic retail analytics.

---

## How to Run

```bash
pip install numpy pandas matplotlib scikit-learn scipy
jupyter notebook hierarchical_clustering.ipynb
```

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

- Dendrogram showing the full merge hierarchy — the optimal cut point is visually identified
- Cluster scatter plot with colour-coded customer segments
- Compare with K-Means (notebook 15) on the same data
