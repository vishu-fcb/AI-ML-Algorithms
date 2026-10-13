# Sentiment Analysis — Twitter NLP

> Classify tweet sentiment as positive or negative using a comprehensive NLP preprocessing pipeline and a Stacking Ensemble classifier (~90% accuracy).

---

## Overview

This project performs **binary sentiment classification** on Twitter data. It demonstrates a production-grade NLP pipeline that handles all the messiness of real social media text: retweets, @mentions, emojis, URLs, hashtags, repeated characters, and contractions.

Two classification approaches are compared:
- **Logistic Regression** with GridSearchCV hyperparameter tuning
- **Stacking Ensemble** (Logistic Regression + Random Forest) — ~90% accuracy

---

## Dataset

**File:** `tweet_data.csv`

| Column | Description |
|---|---|
| sentiment | `positive` or `negative` — target variable |
| tweet_text | Raw tweet content |

---

## NLP Pipeline

The preprocessing chain handles every common issue found in tweet text:

```
Raw tweet
   → Remove RT (retweet marker)
   → Replace @mentions with "twitteruser"
   → Demojize emojis to text  (👍 → :thumbs_up:)
   → Remove URLs
   → Remove hashtag symbols
   → Lowercase
   → Collapse repeated characters (LOOOOK → LOOK)
   → Fix contractions (we'd → we would)
   → Remove repeated punctuation
   → Tokenize + remove stopwords (keep "not")
   → Snowball stemming
   → Bag of Words (bigrams, min_df=2)
```

---

## Models

### Logistic Regression + GridSearchCV

Hyperparameter grid:
- `C`: [0.1, 1, 10] — regularisation strength
- `solver`: [lbfgs, liblinear]
- Cross-validation: 5-fold

### Stacking Ensemble

| Base Estimator | Details |
|---|---|
| Logistic Regression | Best params from GridSearchCV |
| Random Forest | 100 trees |
| Meta-Estimator | Logistic Regression |

---

## How to Run

### Prerequisites

```
Python 3.8+
pandas, numpy, matplotlib, scikit-learn, nltk, wordcloud, emoji, contractions, tensorflow
```

### Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn nltk wordcloud emoji contractions tensorflow
```

### Run

```bash
cd 06_Sentiment_Analysis
python Sentimental_Analysis.py
```

The script generates word cloud visualisations for positive and negative tweets, then trains and evaluates both models, printing accuracy and a full classification report.

---

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression (tuned) | ~85% |
| Stacking Ensemble | ~90% |

A pie chart shows the sentiment distribution in the dataset, and word clouds visualise the most frequent positive and negative words.

---

## Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `scikit-learn` · `NLTK` · `WordCloud` · `emoji` · `contractions`
