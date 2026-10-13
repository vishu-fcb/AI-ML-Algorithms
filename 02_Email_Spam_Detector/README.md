# Email Spam Detector — SVM + NLP

> Classify emails as spam or legitimate using a Support Vector Machine trained on a Bag-of-Words representation.

---

## Overview

This project builds an email spam classifier using classic **Natural Language Processing (NLP)** and a **Support Vector Machine (SVM)** with an RBF kernel.

Raw email text is cleaned, stemmed, and converted into a numerical Bag-of-Words feature matrix, which the SVM then uses for binary classification (spam vs ham). The script ends with an interactive prompt so you can test any message in real time.

> **Why SVM over Naive Bayes?** Naive Bayes was tested first and achieved only ~82% accuracy on this dataset. SVM with an RBF kernel significantly outperformed it.

---

## Dataset

**File:** `spam.csv` (Windows-1252 encoding)

| Column | Description |
|---|---|
| v1 | Label — `spam` or `ham` |
| v2 | Raw email/SMS text |

**Size:** 5,572 labelled messages

**Source:** UCI SMS Spam Collection Dataset

---

## Algorithm & Pipeline

```
Raw text
   → Remove non-alphabetic characters
   → Lowercase
   → Remove stopwords (except "not")
   → Porter Stemming
   → Bag of Words (top 1,500 features)
   → SVM (kernel=RBF)
   → spam / ham prediction
```

---

## How to Run

### Prerequisites

```
Python 3.8+
numpy, pandas, scikit-learn, nltk, chardet
```

### Install dependencies

```bash
pip install numpy pandas scikit-learn nltk chardet
```

### Run

```bash
cd 02_Email_Spam_Detector
python spam_detector.py
```

On first run, NLTK will download the `stopwords` corpus automatically.

At the end, the script prompts you to enter any email text and tells you whether it is spam or legitimate.

---

## Results

| Metric | Value |
|---|---|
| Accuracy | ~98% (printed at runtime) |
| Classifier | SVM — RBF kernel |
| Train / Test split | 75% / 25% |

A confusion matrix is printed showing true positives, false positives, true negatives, and false negatives.

---

## Tech Stack

`Python` · `NumPy` · `Pandas` · `scikit-learn` · `NLTK` · `chardet`
