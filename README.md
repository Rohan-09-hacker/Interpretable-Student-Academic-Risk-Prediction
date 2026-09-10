# Interpretable Student Academic Risk Prediction

**Course**: CSE2267 – Machine Learning Techniques  
**Institution**: Presidency University, Bengaluru  
**Assessment**: CA-III (Research Paper)  
**Team Size**: Maximum 4 members

---

## 📌 Project Overview

This project focuses on building **interpretable machine learning models** to predict students who are at academic risk at an early stage.  

While many existing models achieve high accuracy, they act as black boxes and provide little actionable insight for educators. Our work aims to combine strong predictive performance with clear, human-understandable explanations using **SHAP (SHapley Additive exPlanations)**.

### Objectives
1. Develop accurate models to identify at-risk students.
2. Provide both global and local interpretability using SHAP.
3. Compare inherently interpretable models with powerful ensemble methods.
4. Analyze key risk factors that influence student academic performance.
5. (Optional) Examine fairness across demographic groups.

---

## 🗂️ Dataset

We plan to use one of the following publicly available datasets:

- Student Performance Dataset (UCI / Kaggle)
- Other relevant academic performance datasets from Kaggle

**Target Variable**: Binary classification  
- `At-Risk` (Final grade below threshold / Fail)  
- `Not At-Risk`

---

## 🛠️ Methodology

1. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical features
   - Feature scaling
   - Stratified train-test split

2. **Feature Engineering**
   - Creating meaningful academic and behavioral features
   - Correlation analysis and feature selection

3. **Models**
   - Interpretable: Logistic Regression, Decision Tree
   - Ensemble: Random Forest, XGBoost, LightGBM

4. **Explainability**
   - Global feature importance using SHAP summary plots
   - Local explanations for individual students using SHAP force/waterfall plots

5. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score, AUC-ROC
   - Confusion Matrix
   - SHAP-based insights

---

## 💻 Tech Stack

- **Language**: Python 3.10+
- **Libraries**:
  - pandas, numpy
  - scikit-learn
  - xgboost / lightgbm
  - shap
  - matplotlib, seaborn
  - jupyter

---
