# Naive Bayes Classification

**Category:** Classification

---

## What It Does

Naive Bayes applies **Bayes' Theorem** with the "naive" assumption that all features are conditionally independent given the class. Despite this simplification rarely holding in practice, Naive Bayes is fast, effective, and surprisingly competitive — especially for text classification and high-dimensional data.

The Gaussian variant used here assumes features follow a normal distribution within each class.

---

## When To Use It

- Text classification (spam detection, sentiment analysis) — Naive Bayes is a classic here
- When training data is limited — it needs very few examples per class
- When training speed is critical — the fastest classifier in this collection
- Less effective when features are strongly correlated (violates the independence assumption)

---

## Dataset

**File:** `Social_Network_Ads.csv`

| Column | Description |
|---|---|
| Age | User age |
| EstimatedSalary | Estimated annual salary (USD) |
| Purchased | 1 = bought, 0 = did not — target variable |

400 samples. Running Naive Bayes on this dataset clearly shows the smooth, elliptical decision boundary it produces — characteristic of the Gaussian assumption.

---

## How to Run

```bash
pip install numpy pandas matplotlib scikit-learn
jupyter notebook naive_bayes.ipynb
```

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

- Decision boundary visualisation showing the smooth elliptical boundary of Gaussian Naive Bayes
- Confusion matrix and accuracy score
