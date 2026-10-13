# Thompson Sampling — Reinforcement Learning

**Category:** Reinforcement Learning (Multi-Armed Bandit)

---

## What It Does

Thompson Sampling is a **Bayesian exploration-exploitation strategy** for the multi-armed bandit problem. Instead of maintaining point estimates of reward probabilities, it maintains a Beta distribution for each arm and at each round samples from those distributions — naturally exploring more when uncertain and exploiting when confident.

Applied here to **ad selection**: given 10 different ads, which one should be shown to maximise total clicks?

---

## When To Use It

- A/B testing and online ad optimisation
- Clinical trial design (minimise patients receiving inferior treatment)
- Any sequential decision problem where you need to balance trying new options vs exploiting known good ones
- Thompson Sampling typically **outperforms UCB** because its probabilistic exploration is more efficient — especially early in the process

---

## Dataset

**File:** `Ads_CTR_Optimisation.csv`

10,000 simulated rounds × 10 ads. Each cell is 1 (user clicked) or 0 (user did not click). This simulates a real click stream where only one ad is shown per round.

---

## How to Run

```bash
pip install numpy pandas matplotlib
jupyter notebook thompson_sampling.ipynb
```

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

- Total reward accumulated over 10,000 rounds
- Histogram showing how often each ad was selected (the best ad should dominate)
- Comparison with Upper Confidence Bound (notebook 19) — Thompson Sampling typically converges faster
