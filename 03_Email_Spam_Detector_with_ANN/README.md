# Email Spam Detector with ANN — Deep Learning

> Classify spam emails using an Artificial Neural Network — a deep learning alternative to the SVM approach in project 02.

---

## Overview

This project solves the same spam detection problem as [02_Email_Spam_Detector](../02_Email_Spam_Detector/) but replaces the SVM with a **TensorFlow/Keras Artificial Neural Network (ANN)**.

The NLP preprocessing pipeline is identical (Bag of Words, stemming, stopword removal). The key difference is the classifier: a 3-layer ANN trained with binary cross-entropy and the Adam optimiser over 200 epochs.

This makes it a direct apples-to-apples comparison of classical ML (SVM) vs deep learning (ANN) on the same dataset and feature set.

---

## Dataset

**File:** `spam.csv` (Windows-1252 encoding)

| Column | Description |
|---|---|
| v1 | Label — `spam` or `ham` (label-encoded to 0/1) |
| v2 | Raw email/SMS text |

**Size:** 5,572 labelled messages

**Source:** UCI SMS Spam Collection Dataset

---

## Algorithm & Architecture

### NLP Pipeline (same as project 02)

```
Raw text → clean → lowercase → remove stopwords → Porter stemming
        → Bag of Words (top 1,500 features)
```

### ANN Architecture

| Layer | Units | Activation |
|---|---|---|
| Input + Hidden 1 | 6 | ReLU |
| Hidden 2 | 6 | ReLU |
| Output | 1 | Sigmoid |

- **Optimizer:** Adam
- **Loss:** Binary Cross-Entropy
- **Epochs:** 200 · **Batch size:** 100

---

## How to Run

### Prerequisites

```
Python 3.8+
numpy, pandas, tensorflow, scikit-learn, nltk, chardet
```

### Install dependencies

```bash
pip install numpy pandas tensorflow scikit-learn nltk chardet
```

### Run

```bash
cd 03_Email_Spam_Detector_with_ANN
python spam_detector_ANN.py
```

Training takes a couple of minutes (200 epochs). At the end, enter any email text at the prompt to get a real-time spam/ham prediction.

---

## Results

| Metric | Value |
|---|---|
| Accuracy | ~97-98% (printed at runtime) |
| Classifier | ANN — 2 hidden layers |
| Train / Test split | 80% / 20% |

Compare with project 02 (SVM) to see which approach performs better on this specific dataset.

---

## Tech Stack

`Python` · `TensorFlow/Keras` · `NumPy` · `Pandas` · `scikit-learn` · `NLTK` · `chardet`
