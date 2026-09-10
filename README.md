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
1. **Early Identification**: Develop accurate classification models to identify at-risk students ($G3 < 10$).
2. **Interpretability with SHAP**: Provide both global feature importance (summary/beeswarm) and local student explanations (force/waterfall).
3. **Model Benchmarking**: Compare inherently interpretable models (Logistic Regression, Decision Trees) with powerful ensemble methods (Random Forest, XGBoost, LightGBM).
4. **Risk Factor Analysis**: Uncover key behavioral, academic, and socio-economic risk factors that drive student failure.
5. **Fairness Analysis (Optional)**: Examine model behavior and fairness across demographic subgroups.

---

## 📊 Project Milestones & Current Progress (~35% Complete)

| Phase | Milestone | Status | Details |
|---|---|---|---|
| **Phase 1** | **Project Scaffolding & Git Setup** | ✅ Completed | Repository structured, GitHub connected, dependencies configured |
| **Phase 2** | **Data Acquisition & Audit** | ✅ Completed | UCI Student Performance dataset ingested (`student-mat.csv`, `student-por.csv`) |
| **Phase 3** | **Exploratory Data Analysis (EDA)** | ✅ Completed | `01_EDA.ipynb` executed with full visual outputs and 5 exported figures |
| **Phase 4** | **Data Preprocessing Pipeline** | ✅ Completed | `src/data_preprocessing.py` & `02_Preprocessing_Pipeline.ipynb` (Train: 316, Test: 79) |
| **Phase 5** | **Model Training & Benchmarking** | ⏳ Next | Logistic Regression, Decision Trees, Random Forest, XGBoost, LightGBM |
| **Phase 6** | **XAI & SHAP Interpretability** | ⏳ Planned | TreeExplainer, Beeswarm plots, Waterfall plots, Actionable educator rules |
| **Phase 7** | **Evaluation & Results Synthesis** | ⏳ Planned | Accuracy, Recall, PR-AUC, Confusion matrices in `results/` |
| **Phase 8** | **Research Paper (CA-III Writeup)** | ⏳ Planned | Comprehensive IEEE/Springer format paper draft in `paper/` |

---

## 🗂️ Dataset Details

We utilize the **UCI Student Performance Dataset** (Mathematics course, $N=395$, 33 original features):

* **Target Variable**: Binary classification:
  * `At-Risk (1)`: Final grade $G3 < 10$ (32.9% of students, $N=130$)
  * `Not At-Risk (0)`: Final grade $G3 \ge 10$ (67.1% of students, $N=265$)
* **Engineered Features**:
  * `parent_edu_avg`: Mean of mother's and father's educational attainment.
  * `study_freetime_ratio`: Ratio of weekly study hours to free time.
  * `alcohol_index`: Composite metric of workday and weekend alcohol consumption.
  * `high_absences`: Binary indicator for students with severe absenteeism ($>10$ days).
  * `has_prior_failures`: Early-warning flag for past course failures.

---

## 📁 Repository Structure

```text
Interpretable-Student-Academic-Risk-Prediction/
├── data/
│   ├── student-mat.csv           # Raw mathematics course dataset (UCI)
│   ├── student-por.csv           # Raw Portuguese course dataset (UCI)
│   ├── student.txt               # UCI feature descriptions and metadata
│   └── processed/
│       ├── train.csv             # Processed & scaled training set (N=316, 44 features)
│       └── test.csv              # Processed & scaled testing set (N=79, 44 features)
├── notebooks/
│   ├── 01_EDA.ipynb              # Executed EDA with visual distributions & correlation heatmap
│   └── 02_Preprocessing_Pipeline.ipynb  # End-to-end preprocessing demonstration
├── src/
│   └── data_preprocessing.py     # Modular preprocessing & feature engineering pipeline
├── results/
│   └── figures/
│       ├── 01_target_distribution.png
│       ├── 02_correlation_matrix.png
│       ├── 03_risk_by_failures.png
│       ├── 04_risk_by_studytime.png
│       └── 05_support_and_aspirations.png
├── paper/                        # Research paper draft (CA-III)
├── requirements.txt              # Project dependencies
├── .gitignore                    # Version control ignore rules
└── README.md                     # Project documentation
```

---

## 🛠️ Getting Started

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Run Preprocessing Pipeline
```powershell
python src/data_preprocessing.py
```

### 3. Launch Notebooks
```powershell
jupyter notebook notebooks/01_EDA.ipynb
```
