# Natural Language Processing (NLP) — Bag of Words

**Category:** Natural Language Processing

---

## What It Does

This notebook implements a complete NLP pipeline to classify restaurant reviews as positive or negative. Raw text is cleaned, tokenised, and converted into a **Bag of Words (BoW)** numerical representation, which is then fed into a Naive Bayes classifier.

It covers the core NLP preprocessing steps that appear in almost every text classification task.

---

## NLP Pipeline

```
Raw review text
   → Remove non-alphabetic characters
   → Lowercase
   → Tokenise (split into words)
   → Remove stopwords (except "not")
   → Porter Stemming (e.g. "loved" → "love")
   → Bag of Words (CountVectorizer — top 1,500 features)
   → Naive Bayes Classifier
   → Positive / Negative prediction
```

---

## When To Use It

- Short-text sentiment analysis (reviews, tweets, survey responses)
- Document categorisation
- As a baseline NLP pipeline before trying TF-IDF, word embeddings (Word2Vec), or transformer models (BERT)
- BoW ignores word order — suitable for short texts where context is less critical

---

## Dataset

**File:** `Restaurant_Reviews.tsv`

1,000 restaurant reviews labelled as positive (1) or negative (0).

Tab-separated values (TSV format).

---

## How to Run

```bash
pip install numpy pandas matplotlib scikit-learn nltk
jupyter notebook Natural_language_processing.ipynb
```

On first run, NLTK downloads the `stopwords` corpus automatically.

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

- Confusion matrix and accuracy score
- Cleaned corpus sample showing what text looks like after each preprocessing step
