# Apriori Algorithm — Association Rule Learning

**Category:** Association Rule Learning (Unsupervised Learning)

---

## What It Does

Apriori discovers **frequent itemsets** and **association rules** in transactional data — "customers who buy X also tend to buy Y". It works by iteratively finding itemsets that appear together more often than a minimum support threshold, then generating rules from those itemsets.

Classic application: **market basket analysis** — understanding what products are frequently bought together to inform shelf placement, promotions, and recommendations.

---

## When To Use It

- Market basket analysis and product recommendation
- Any domain where you want to discover "if A then B" patterns in transactions
- When you want **interpretable, human-readable rules** (not black-box predictions)
- Computationally expensive on datasets with many unique items — use Eclat or FP-Growth for large catalogues

---

## Key Metrics

| Metric | Meaning |
|---|---|
| **Support** | How often the itemset appears across all transactions |
| **Confidence** | P(B \| A) — if a customer buys A, how likely are they to buy B? |
| **Lift** | How much more likely B is when A is bought vs by chance (lift > 1 = positive association) |

---

## Dataset

**File:** `Market_Basket_Optimisation.csv`

7,500 grocery transactions. Each row is one customer's basket — a comma-separated list of products purchased. The task is to find which products are frequently bought together.

---

## How to Run

```bash
pip install numpy pandas matplotlib apyori
jupyter notebook Apriori.ipynb
```

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

- List of discovered association rules sorted by lift
- Support, confidence, and lift values for each rule
- Example: `{burger, cooking_oil} → {mineral_water}` with high lift
