# Decision Tree Classification

**Category:** Classification

---

## What It Does

Decision Tree Classification learns a series of **if-then rules** by recursively splitting the feature space at the threshold that best separates the classes (using Gini impurity or entropy). The result is a tree of decisions that can be fully visualised and interpreted.

---

## When To Use It

- When model explainability is important (the decision path can be traced step by step)
- When the data has non-linear class boundaries
- As a building block for Random Forest and Gradient Boosted Trees
- Prone to overfitting on small datasets — use `max_depth` to control tree size

---

## Dataset

**File:** `Social_Network_Ads.csv`

| Column | Description |
|---|---|
| Age | User age |
| EstimatedSalary | Estimated annual salary (USD) |
| Purchased | 1 = bought the product, 0 = did not — target variable |

400 social media users. The task is to predict whether a user purchased a product based on demographic features.

This same dataset is used across all 7 classification algorithm notebooks (07–13), making it easy to compare decision boundaries visually.

---

## How to Run

```bash
pip install numpy pandas matplotlib scikit-learn
jupyter notebook decision_tree_classification.ipynb
```

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

- Decision boundary visualisation (training set and test set)
- Confusion matrix and accuracy score
- The characteristic "rectangular" decision boundary that Decision Trees produce
