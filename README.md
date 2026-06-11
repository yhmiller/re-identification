# Explainable ML for Predicting Nursing Licensure Examination Failure in Ghana

**An XGBoost-SHAP Approach**

---

## Overview

Code pipeline for the MSc Health Informatics thesis:

> **"Explainable Machine Learning for Predicting Nursing Licensure Examination Failure in Ghana: An XGBoost-SHAP Approach"**

**Author:** Prince Bortey Miller | ID: 22388461 | Supervision Code: 289
**Supervisor:** Dr. Eric Opoku Osei | Department of Computer Science, KNUST, Ghana
**Programme:** Master of Health Informatics (MHI) | 2025–2026
**Ethics:** HuSSREC, KNUST (Ref: _______________)

---

## Research Summary

Ghana's NMC Licensure Examination has a national first-attempt pass rate of ~50%. This study builds an explainable ML model (**XGBoost + SHAP**) to predict which nursing trainees are at risk of failing, using routinely collected academic records — enabling targeted remediation before the examination. The engineering contribution is **E-XGBoost**: SHAP-guided feature pruning compared against the locked baseline on identical splits (Wilcoxon signed-rank).

**Target:** Fail = 1 (failed ≥1 of 6 theory papers, first attempt) | Pass = 0
**Predictors (pre-exam only):** WASSCE entry grades, continuous assessment, programme CGPA, mock scores, programme type, age band, gender
**Theory:** Astin's Input–Environment–Output (I-E-O) model

---

## Repository Structure

```
xgboost-shap-nursing-licensure/
│
├── data/                                  # NOT committed (see .gitignore)
│   ├── raw_college_records.xlsx           # raw file from college — never share
│   └── anonymised_records.csv             # output of Stage 0
│
├── run.sh                                 # one command: set up env + run everything
├── run_all.py                             # runs Stage 1 + Stage 2 in one process
├── app.sh                                 # one command: launch the educator app
│
├── notebooks/
│   ├── 00_data_preparation_and_eda.py     # Stage 0: clean → anonymise → EDA
│   ├── 01_pipeline_and_experiments.py     # Stage 1: baselines + XGBoost + SHAP + eval
│   └── 02_model_engineering.py            # Stage 2: E-XGBoost vs baseline
│
├── app/                                   # educator risk-screening web app
│   ├── app.py                             # Streamlit UI
│   ├── inference.py                       # load model bundle + score records
│   └── example_students.csv              # sample upload to try the app
│
├── results/                               # all auto-generated when stages run
│   ├── eda_*.csv / eda_*.png
│   ├── shap_*.png, roc_pr_curves.png, calibration_curves.png
│   ├── baseline_metrics.json, engineered_metrics.json
│   ├── comparison_table.csv               # Table 1 of the Results chapter
│   ├── inference_bundle.pkl               # model + preprocessor + transform (for the app)
│   └── config.yaml                        # hyperparameters — written by Stage 1, Cell 19
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Quick start (one command)

```bash
git clone https://github.com/yhmiller/xgboost-shap-nursing-licensure.git
cd xgboost-shap-nursing-licensure
./run.sh
```

`run.sh` sets up the Python environment on first use (it needs Python 3.11 —
on macOS: `brew install python@3.11`), then runs Stage 1 + Stage 2 end-to-end.
Every figure, table and metric is written to `results/`. Re-running is safe; the
setup step is skipped once the environment exists.

## Run order

| Stage | File | When |
|---|---|---|
| 0 | `notebooks/00_data_preparation_and_eda.py` | When the college delivers data (edit PREP-CELL 2 mapping first) |
| 1+2 | `run_all.py` (runs both notebooks in one process) | Now on synthetic data; re-run on real data after editing Stage 1 Cell 6 |

For interactive exploration, each `.py` file is also organised into numbered
cells (`# CELL N` / `# ENG-CELL N`) to paste into Jupyter/Colab.

## Educator screening app

For nurse educators (no coding needed) — screen a class and see who is at risk:

```bash
./run.sh     # once, to train the model
./app.sh     # launch the app in your browser
```

Upload a spreadsheet of student records (download the blank template from the
app, or try `app/example_students.csv`). The app returns a ranked at-risk list
and, for any student, the top factors behind their risk estimate. It loads
`results/inference_bundle.pkl`, so re-running `./run.sh` on real data updates the
app automatically — no app changes needed.

> Predictions are decision support, not verdicts. The current model is trained on
> synthetic data; results are illustrative until trained on real records.

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

AUC-PR (primary), AUC-ROC, F1-weighted, Precision, Recall · DeLong test (AUC comparison) · McNemar (label disagreement) · Brier + calibration curves · Fairness: FNR, FPR, Equal Opportunity Difference · Wilcoxon signed-rank for the baseline-vs-E-XGBoost comparison.

## Data & Privacy

Primary data from Ghanaian Nursing Training Colleges under HuSSREC approval. Raw records are **never committed** (enforced via `.gitignore`); only the SHA-256-anonymised dataset is retained. Synthetic data (SDV) is used for pipeline testing only — never in reported results. Anonymised dataset deposited on Kaggle upon publication; code archived on Zenodo.

## Target Journals

1. **Computers & Education: Artificial Intelligence** (Elsevier) — primary
2. **Nurse Education Today** (Elsevier) — backup

## Reproducibility

`random_state=42` everywhere; pinned versions in `requirements.txt`; final hyperparameters auto-logged to `results/config.yaml` by Stage 1.