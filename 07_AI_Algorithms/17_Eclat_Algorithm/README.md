# Eclat Algorithm — Association Rule Learning

**Category:** Association Rule Learning (Unsupervised Learning)

---

## What It Does

Eclat (Equivalence Class Transformation) discovers **frequent itemsets** in transactional data — the same goal as Apriori, but with a different internal strategy. Instead of scanning the full database repeatedly, Eclat uses a **vertical data format** (storing transaction IDs per item) and finds frequent itemsets by intersecting sets.

This makes Eclat faster than Apriori on dense datasets.

---

## When To Use It

- Same use cases as Apriori: market basket analysis, co-purchase patterns
- When speed is more important than generating full confidence/lift metrics — Eclat computes support only (not confidence or lift by default)
- Better than Apriori when the dataset is dense (many items frequently co-occur)
- Apriori is preferred when full rule metrics (confidence, lift) are needed

---

## Apriori vs Eclat

| Feature | Apriori | Eclat |
|---|---|---|
| Data format | Horizontal (rows = transactions) | Vertical (columns = items) |
| Metrics | Support, Confidence, Lift | Support only |
| Speed | Slower on dense data | Faster on dense data |
| Interpretability | Full rules with lift | Frequent itemsets |

---

## Dataset

**File:** `Market_Basket_Optimisation.csv`

7,500 grocery transactions — same as the Apriori notebook, allowing direct comparison.

---

## How to Run

```bash
pip install numpy pandas matplotlib apyori
jupyter notebook Eclat.ipynb
```

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

- Frequent itemsets sorted by support
- Comparison with Apriori results on the same dataset
