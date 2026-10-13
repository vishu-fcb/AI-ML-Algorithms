# AI & ML Algorithms — Portfolio

> A hands-on collection of AI and Machine Learning projects — from foundational algorithms to production-ready applications — built in Python.

---

## Projects

| # | Project | Algorithm / Approach | Key Libraries |
|---|---|---|---|
| [01](01_Stock_Market_Prediction/) | Stock Market Prediction | Multiple Linear Regression | scikit-learn, pandas |
| [02](02_Email_Spam_Detector/) | Email Spam Detector | SVM + NLP (Bag of Words) | scikit-learn, NLTK |
| [03](03_Email_Spam_Detector_with_ANN/) | Email Spam Detector with ANN | ANN (TensorFlow) + NLP | TensorFlow, NLTK |
| [04](04_Traffic_Sign_Detection_CNN/) | Traffic Sign Detection | CNN — 43-class image classification | TensorFlow, OpenCV |
| [05](05_Image_Classification_CIFAR_CNN/) | Image Classification (CIFAR-10) | CNN — 10-class image classification | TensorFlow |
| [06](06_Sentiment_Analysis/) | Sentiment Analysis | NLP + Stacking Ensemble (LR + RF) | scikit-learn, NLTK |
| [07](07_AI_Algorithms/) | AI Algorithms Reference Library | 27 algorithms across all major ML categories | scikit-learn, TensorFlow, XGBoost |
| [08](08_Predictive_Diagnostics/) | Predictive Maintenance Platform | XGBoost (8 models), Streamlit dashboard | XGBoost, Streamlit, OpenAI |
| [09](09_LLMs/) | LLM Applications | RAG chatbot, dual-model assistant, agentic CV bot | LangChain, OpenAI, Gradio |

---

## Project Highlights

### 01 — Stock Market Prediction
Linear Regression applied to Tesla stock data. Features: opening price, daily high/low, volume, day-of-year. Outputs MAE, MSE, R² and predicts future closing prices.

### 02 — Email Spam Detector
NLP pipeline (stemming, stopword removal, Bag of Words) + SVM classifier. Achieves ~98% accuracy on 5,572 emails. Interactive real-time prediction at the end.

### 03 — Email Spam Detector with ANN
Same NLP pipeline as project 02 but replaces SVM with a TensorFlow ANN (2 hidden layers). Direct comparison: classical ML vs deep learning on the same problem.

### 04 — Traffic Sign Detection (CNN)
CNN classifies 43 German traffic sign types (~95% accuracy). Data augmentation applied during training. Dataset: GTSRB from Kaggle.

### 05 — CIFAR-10 Image Classification (CNN)
CNN trained on CIFAR-10 (10 object classes, 60K images). Uses L2 regularisation, Dropout, and ReduceLROnPlateau. No pre-trained weights — trained from scratch.

### 06 — Sentiment Analysis
Production-grade tweet NLP pipeline: retweet removal, emoji handling, contraction fixing, Snowball stemming. Stacking ensemble (Logistic Regression + Random Forest) achieves ~90% accuracy.

### 07 — AI Algorithms Reference Library
27 self-contained Jupyter notebooks covering every major ML category — Regression, Classification, Clustering, Association Rules, Reinforcement Learning, NLP, Deep Learning, Dimensionality Reduction, and Model Selection. Each notebook pairs an algorithm with a real dataset and includes a README.

### 08 — BeyondTech Predictive Maintenance Platform
Full-stack predictive maintenance system: 8 XGBoost models (96.5% avg accuracy), Streamlit dashboard, live prediction simulator with 8 sliders, 90-day health forecasts, and a marketplace with AI diagnosis (GPT-4o-mini) + Google Maps workshop search. Trained on real NASA + OBD-II data.

### 09 — LLM Applications
Three LLM application patterns: RAG chatbot (LangChain + ChromaDB), dual-model Q&A assistant (OpenAI + local Ollama), and an agentic CV bot with tool use and a background Gemini quality evaluator.

---

## Skills Demonstrated

| Category | Technologies |
|---|---|
| Machine Learning | Linear/Polynomial/SVR/RF Regression, SVM, KNN, Naive Bayes, Decision Trees, Random Forest, XGBoost |
| Deep Learning | ANN, CNN (TensorFlow/Keras) |
| Unsupervised Learning | K-Means, Hierarchical Clustering, PCA, LDA, Kernel PCA |
| Reinforcement Learning | UCB, Thompson Sampling |
| NLP | Bag of Words, stemming, tokenisation, Bag-of-Ngrams, sentiment analysis |
| LLM Applications | RAG, tool use, LLM-as-judge, vector stores, agent patterns |
| Data & Tools | Pandas, NumPy, Matplotlib, scikit-learn, Streamlit, Gradio, LangChain |

---

## Setup

### Shared environment (root .venv)

A shared virtual environment exists at the repo root for the algorithm notebooks:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install numpy pandas matplotlib scikit-learn tensorflow xgboost jupyter
```

### Projects with their own environments

Projects 08 and 09 have their own `requirements.txt` — see their individual READMEs for setup instructions.

---

## Repository Structure

```
AI-ML-Algorithms/
├── 01_Stock_Market_Prediction/         Linear Regression on Tesla stock data
├── 02_Email_Spam_Detector/             SVM spam classifier
├── 03_Email_Spam_Detector_with_ANN/    ANN spam classifier
├── 04_Traffic_Sign_Detection_CNN/      CNN — 43 traffic sign classes
├── 05_Image_Classification_CIFAR_CNN/  CNN — CIFAR-10 (10 classes)
├── 06_Sentiment_Analysis/              Tweet sentiment (stacking ensemble)
├── 07_AI_Algorithms/                   27 algorithm notebooks (full ML spectrum)
├── 08_Predictive_Diagnostics/          Full-stack predictive maintenance platform
└── 09_LLMs/                            LLM applications (RAG, agents, tools)
```
