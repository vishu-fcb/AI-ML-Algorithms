# AI Algorithms Reference Library

> 27 self-contained Jupyter notebooks covering the full spectrum of classical and modern ML algorithms — each paired with a real dataset and ready to run.

---

## Overview

This folder is a hands-on reference library for AI/ML algorithms. Each subfolder contains:
- One Jupyter notebook implementing the algorithm end-to-end
- The dataset it runs on
- A README explaining the algorithm, when to use it, and what the notebook produces

The notebooks progress from fundamentals (linear regression) through advanced techniques (XGBoost, dimensionality reduction), covering every major algorithm category.

---

## Algorithm Index

### Regression

| # | Algorithm | Dataset | Notebook |
|---|---|---|---|
| 01 | Decision Tree Regression | Position_Salaries.csv | [decision_tree_regression.ipynb](01_Decision_Tree_Regression_Algorithms/) |
| 02 | Multiple Linear Regression | 50_Startups.csv | [multiple_linear_regression.ipynb](02_Multiple_Linear_Regression_Algorithm/) |
| 03 | Polynomial Regression | Position_Salaries.csv | [polynomial_regression.ipynb](03_Polynomial_Regressions_Algorithm/) |
| 04 | Random Forest Regression | Position_Salaries.csv | [random_forest_regression.ipynb](04_Random_Forest_Regressions_Algorithm/) |
| 05 | Simple Linear Regression | Salary_Data.csv | [simple_linear_regression.ipynb](05_Simple_Linear_Regressions__Algorithm/) |
| 06 | Support Vector Regression | Position_Salaries.csv | [support_vector_regression.ipynb](06_Support_Vector_Regressions_Algorithm/) |

### Classification

| # | Algorithm | Dataset | Notebook |
|---|---|---|---|
| 07 | Decision Tree Classification | Social_Network_Ads.csv | [decision_tree_classification.ipynb](07_Decision_Tree_Classifications/) |
| 08 | Kernel SVM Classification | Social_Network_Ads.csv | [kernel_svm.ipynb](08_Kernel_SVM_Classifications/) |
| 09 | K-Nearest Neighbors | Social_Network_Ads.csv | [k_nearest_neighbors.ipynb](09_K-NNN_Classifications/) |
| 10 | Logistic Regression | Social_Network_Ads.csv | [logistic_regression.ipynb](10_Logistic%20Regressions_Classifications/) |
| 11 | Naive Bayes | Social_Network_Ads.csv | [naive_bayes.ipynb](11_Naive_Bayes_Classifications/) |
| 12 | Random Forest Classification | Social_Network_Ads.csv | [random_forest_classification.ipynb](12_Random_Forest_Classifications/) |
| 13 | Support Vector Machines | Social_Network_Ads.csv | [support_vector_machine.ipynb](13_Support_Vector_Machines_Classifications/) |

### Clustering

| # | Algorithm | Dataset | Notebook |
|---|---|---|---|
| 14 | Hierarchical Clustering | Mall_Customers.csv | [hierarchical_clustering.ipynb](14_Hierachical_Clustering/) |
| 15 | K-Means Clustering | Mall_Customers.csv | [k_means_clustering.ipynb](15_K-Means_Clustering/) |

### Association Rule Learning

| # | Algorithm | Dataset | Notebook |
|---|---|---|---|
| 16 | Apriori | Market_Basket_Optimisation.csv | [Apriori.ipynb](16_Apriori_Algorithm/) |
| 17 | Eclat | Market_Basket_Optimisation.csv | [Eclat.ipynb](17_Eclat_Algorithm/) |

### Reinforcement Learning

| # | Algorithm | Dataset | Notebook |
|---|---|---|---|
| 18 | Thompson Sampling | Ads_CTR_Optimisation.csv | [thompson_sampling.ipynb](18_Thompson_Sampling_Algorithm/) |
| 19 | Upper Confidence Bound | Ads_CTR_Optimisation.csv | [upper_confidence_bound.ipynb](19_Upper_Confidence_Bound_Algorithm/) |

### Natural Language Processing

| # | Algorithm | Dataset | Notebook |
|---|---|---|---|
| 20 | NLP — Bag of Words | Restaurant_Reviews.tsv | [Natural_language_processing.ipynb](20_NLP_Algorithm/) |

### Deep Learning

| # | Algorithm | Dataset | Notebook |
|---|---|---|---|
| 21 | Artificial Neural Network | Churn_Modelling.csv | [artificial_neural_network.ipynb](21_Artificial_Neural_Network_Algorithm/) |
| 22 | Convolutional Neural Network | Image data | [convolutional_neural_network.ipynb](22_Convolutional_Neural_Network_Algorithm/) |

### Dimensionality Reduction

| # | Algorithm | Dataset | Notebook |
|---|---|---|---|
| 23 | Kernel PCA | Wine.csv | [Kernel_pca.ipynb](23_Kernel_PCA_Algorithm/) |
| 24 | Linear Discriminant Analysis (LDA) | Wine.csv | [Linear_discriminant_analysis.ipynb](24_Linear_Discriminant_Analysis_Algorithm/) |
| 25 | Principal Component Analysis (PCA) | Wine.csv | [Principal_component_analysis.ipynb](25_Principle_Component_Analysis_Algorithm/) |

### Model Selection & Boosting

| # | Algorithm | Dataset | Notebook |
|---|---|---|---|
| 26 | Model Selection (Grid Search + K-Fold) | Social_Network_Ads.csv | [Grid_search.ipynb](26_Model_Selection_Algorithm/) |
| 27 | XGBoost | Data.csv | [xg_boost.ipynb](27_XGBoost_Algorithm/) |

---

## How to Run Any Notebook

### Prerequisites

```
Python 3.8+
Jupyter Notebook or JupyterLab
```

### Install common dependencies

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras xgboost
pip install jupyter
```

Some notebooks require additional packages — see the individual README in each subfolder.

### Launch

```bash
jupyter notebook
```

Navigate to the relevant subfolder and open the `.ipynb` file. Run all cells with **Kernel → Restart & Run All**.

---

## Datasets Quick Reference

| Dataset | Used in | Description |
|---|---|---|
| `Position_Salaries.csv` | 01, 03, 04, 06 | Position level (1–10) vs salary |
| `50_Startups.csv` | 02 | R&D, Admin, Marketing spend vs profit |
| `Salary_Data.csv` | 05 | Years of experience vs salary |
| `Social_Network_Ads.csv` | 07–13, 26 | Age, estimated salary vs purchase decision |
| `Mall_Customers.csv` | 14, 15 | Customer age, income, spending score |
| `Market_Basket_Optimisation.csv` | 16, 17 | Grocery transaction data |
| `Ads_CTR_Optimisation.csv` | 18, 19 | Ad click-through rates across campaigns |
| `Restaurant_Reviews.tsv` | 20 | Reviews labelled liked/not liked |
| `Churn_Modelling.csv` | 21 | Bank customer features vs churn |
| `Wine.csv` | 23, 24, 25 | Wine chemical properties + type |
| `Data.csv` | 27 | General classification dataset |
