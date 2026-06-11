# Explainable ML for Predicting Nursing Licensure Examination Failure in Ghana

**An XGBoost-SHAP Approach**

---

## Overview

This repository contains the full code pipeline for the MSc Health Informatics thesis:

> **"Explainable Machine Learning for Predicting Nursing Licensure Examination Failure in Ghana: An XGBoost-SHAP Approach"**

**Author:** Miller Prince Bortey | ID: 22388461 | Supervision Code: 289
**Supervisor:** Dr. Eric Opoku Osei | Department of Computer Science, KNUST, Ghana
**Programme:** Master of Health Informatics (MHI) | 2025–2026
**Ethics:** Approved by HuSSREC, KNUST (Ref: _______________)

---

## Research Summary

Ghana's Nursing and Midwifery Council Licensure Examination (NMC-LE) has a national first-attempt pass rate of approximately 50%. This study builds an explainable machine learning model using **XGBoost + SHAP** to predict which nursing trainees are at risk of failing the NMC-LE, using routinely collected academic records — enabling nurse educators to target remediation before the high-stakes examination.

**Target variable:** Binary — Fail = 1 (failed ≥1 of 6 theory papers on first attempt) | Pass = 0

**Predictors:** WASSCE entry grades, continuous assessment scores, programme CGPA, mock examination scores, programme type, age band, gender

**Theory:** Astin's Input–Environment–Output (I-E-O) model

---

## Repository Structure

```
xgboost-shap-nursing-licensure/
│
├── notebooks/
│   └── 01_pipeline_and_experiments.ipynb   # Full Colab notebook
│
├── src/
│   ├── 01_synthetic_data.py                # Generate synthetic pilot data
│   ├── 02_preprocessing.py                 # Preprocessing pipeline
│   ├── 03_split.py                         # Train/val/test splitting
│   ├── 04_baselines.py                     # LogReg, RF, LightGBM
│   ├── 05_xgboost_shap.py                  # XGBoost training + SHAP
│   └── 06_evaluation.py                    # DeLong, calibration, fairness
│
├── config.yaml                             # Hyperparameters (logged here)
├── requirements.txt
└── README.md
```

---

## Hypotheses

| # | Hypothesis |
|---|---|
| H1 | XGBoost significantly outperforms Logistic Regression (DeLong, p < 0.05) |
| H2 | Academic indicators (CGPA, mock scores) > demographic variables in SHAP importance |
| H3 | SHAP explanations rated positively by educators (understandability, trust, actionability) |
| H4 | Model demonstrates acceptable discrimination (AUC-PR) and calibration (Brier score) |

---

## Evaluation Metrics

- **Primary:** AUC-PR (Precision-Recall AUC — imbalanced classification)
- AUC-ROC, F1-weighted, Precision, Recall
- DeLong test for AUC comparison between models
- Brier score + calibration curve (reliability diagram)
- Fairness: FNR, FPR, Equal Opportunity Difference by gender / programme / institution

---

## Target Journals

1. **Computers & Education: Artificial Intelligence** (Elsevier) — primary
2. **Nurse Education Today** (Elsevier) — backup

---

## Setup

```bash
# Clone the repository
git clone https://github.com/yhmiller/xgboost-shap-nursing-licensure.git
cd xgboost-shap-nursing-licensure

# Install dependencies
pip install -r requirements.txt
```

**Google Colab:** Upload `notebooks/01_pipeline_and_experiments.ipynb` or run cells directly.

---

## Data

Primary data is collected from Ghanaian Nursing Training Colleges under HuSSREC ethics approval. The anonymised dataset will be deposited on Kaggle upon publication.

A synthetic pilot dataset (generated with SDV for pipeline testing only) is used during development. **No synthetic data is used for model training or evaluation.**

---

## Reproducibility

All stochastic steps use `random_state=42`. Hyperparameters are logged in `config.yaml`. Final code will be archived on Zenodo with a citable DOI.

---

## Acknowledgements

Study conducted under the KNUST CANDO Lab Research Programme, supervised by Dr. Eric Opoku Osei. Ethics clearance granted by the Humanities and Social Sciences Research Ethics Committee (HuSSREC), KNUST.
