# Stock Market Prediction — Linear Regression

> Predict Tesla's closing stock price using historical trading data and a multi-feature Linear Regression model.

---

## Overview

This project demonstrates how to apply **Multiple Linear Regression** to financial time-series data. Given a day's opening price, high, low, and trading volume, the model predicts the adjusted closing price for Tesla (TSLA).

The project covers the full ML pipeline: data loading, feature engineering (extracting day-of-year from dates), train/test split, model training, evaluation, and future-date prediction.

---

## Dataset

**File:** `TESLA.csv`

| Column | Description |
|---|---|
| Date | Trading date |
| Open | Opening price (USD) |
| High | Daily high price (USD) |
| Low | Daily low price (USD) |
| Close | Closing price (USD) |
| Adj Close | Adjusted closing price — **target variable** |
| Volume | Number of shares traded |

**Source:** Historical Tesla stock data (available on Yahoo Finance / Kaggle)

---

## Algorithm

**Multiple Linear Regression** — learns a linear relationship between multiple input features and a continuous output.

Features fed to the model:
- `day_of_year` (derived from the Date column)
- `Open`, `High`, `Low`, `Volume`

---

## How to Run

### Prerequisites

```
Python 3.8+
numpy, pandas, matplotlib, scikit-learn
```

### Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

### Run

```bash
cd 01_Stock_Market_Prediction
python stock_market_predictor_linear_regression.py
```

The script will:
1. Plot the overall stock price trend over time
2. Train the Linear Regression model on 80% of the data
3. Print MAE, MSE, and R² score on the test set
4. Make a sample future-date prediction

---

## Results

| Metric | Value |
|---|---|
| MAE | Mean Absolute Error (printed at runtime) |
| MSE | Mean Squared Error (printed at runtime) |
| R² | R-squared coefficient (printed at runtime) |

The model also outputs side-by-side predicted vs actual values and a sample prediction for a custom future date.

---

## Tech Stack

`Python` · `NumPy` · `Pandas` · `Matplotlib` · `scikit-learn`
