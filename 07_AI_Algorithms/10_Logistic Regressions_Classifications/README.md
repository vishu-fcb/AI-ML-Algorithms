# Logistic Regression Classification

**Category:** Classification

---

## What It Does

Logistic Regression is a **linear classifier** that models the probability of a binary outcome using the logistic (sigmoid) function. Despite the name, it is a classification algorithm — it outputs a probability between 0 and 1 and classifies based on a threshold (typically 0.5).

It is fast, interpretable, and works well when classes are roughly linearly separable.

---

## When To Use It

- Binary classification problems with a roughly linear decision boundary
- When you need a **probabilistic output** (not just class labels)
- When model interpretability is important — coefficients show the direction and strength of each feature's effect
- As a strong baseline before trying more complex classifiers

---

## Dataset

**File:** `Social_Network_Ads.csv`

| Column | Description |
|---|---|
| Age | User age |
| EstimatedSalary | Estimated annual salary (USD) |
| Purchased | 1 = bought, 0 = did not — target variable |

400 samples. Because the classes here are not perfectly linearly separable, Logistic Regression will show clear misclassifications — which is useful for understanding its limitations.

---

## How to Run

```bash
pip install numpy pandas matplotlib scikit-learn
jupyter notebook logistic_regression.ipynb
```

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

- Linear decision boundary plot — compare its straight line with the curves from SVM and KNN on the same data
- Confusion matrix and accuracy score
