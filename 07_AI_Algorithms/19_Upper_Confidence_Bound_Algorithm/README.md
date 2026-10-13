# Upper Confidence Bound (UCB) — Reinforcement Learning

**Category:** Reinforcement Learning (Multi-Armed Bandit)

---

## What It Does

UCB is a **deterministic exploration-exploitation strategy** for the multi-armed bandit problem. At each round, it selects the arm with the highest **upper confidence bound** — a combination of estimated reward and an exploration bonus that decreases as an arm is tried more often.

Applied here to **ad selection**: which of 10 ads should be shown at each round to maximise cumulative clicks?

---

## When To Use It

- Ad optimisation, A/B testing, recommendation systems
- When you want a mathematically principled, parameter-free exploration strategy
- UCB guarantees logarithmic regret — provably near-optimal over time
- Compare with Thompson Sampling (notebook 18) — UCB is deterministic while Thompson Sampling is probabilistic

---

## UCB Formula

At round `t`, select the arm `i` that maximises:

```
reward_estimate_i + sqrt( 1.5 * log(t) / n_i )
```

Where `n_i` is the number of times arm `i` has been selected. The second term shrinks as `n_i` grows — the algorithm automatically stops exploring once it's confident.

---

## Dataset

**File:** `Ads_CTR_Optimisation.csv`

10,000 simulated rounds × 10 ads. Same as Thompson Sampling (notebook 18) — run both to compare which converges faster to the best ad.

---

## How to Run

```bash
pip install numpy pandas matplotlib
jupyter notebook upper_confidence_bound.ipynb
```

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

- Total reward accumulated over 10,000 rounds
- Histogram of selections per ad — the best ad dominates as rounds increase
