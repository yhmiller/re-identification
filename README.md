# E-XGBoost: SHAP-Guided Feature Pruning for Nursing Licensure Failure Prediction

**Explainable Machine Learning for Predicting Nursing Licensure
Examination Failure in Ghana: An XGBoost-SHAP Approach**

MSc Thesis Project | Health Informatics |
Kwame Nkrumah University of Science and Technology (KNUST), Ghana

**Author:** Prince Bortey Miller (Student ID: 22388461)
**Supervisor:** Dr. Eric Opoku Osei, Department of Computer Science, KNUST
**Year:** 2026

---

## Overview

Ghana's Nursing and Midwifery Council Licensure Examination (NMC-LE) has a
first-attempt pass rate near 50%, so roughly half of each graduating cohort is
delayed from entering practice. This study engineers **E-XGBoost**, an XGBoost
classifier whose feature set is reduced by SHAP attribution rather than manual
selection, to identify at-risk trainees from academic records the college
already collects.

The engineering operation is **Operation R (Remove)**: train a baseline on all
36 predictors, rank inputs by mean absolute SHAP value, keep the smallest
subset explaining 95% of total attribution, then retrain on identical folds and
hyperparameters. The feature set is the only thing that changes, so any
difference in performance is attributable to the pruning.

Alongside the analysis pipeline, the repository ships a **Streamlit screening
app** for nurse educators. Upload a class spreadsheet, get a ranked at-risk
list and per-student SHAP explanations. No coding required.

> **Status.** The numbers below come from a synthetic pilot corpus used to
> verify the pipeline end to end. Real college records are still being linked
> to licensure outcomes. No result in this repository is a finding about any
> real cohort.

---

## Three Contributions

1. **Original field data** — the first Ghanaian nursing-licensure dataset
   assembled for machine learning, collected under a data sharing agreement
   with a nursing training college and anonymised at source.

2. **XGBoost with SHAP explainability** — per-student attributions that tell a
   tutor which subject scores drive a given risk estimate, not just that risk
   is high.

3. **E-XGBoost** — SHAP-guided feature pruning compared against a locked
   baseline on identical cross-validation folds, tested with the Wilcoxon
   signed-rank test and reported with Cohen's d.

---

## Key Results (synthetic pilot corpus, not study findings)

| Model | Features | AUC-ROC | AUC-PR | Wilcoxon p | Cohen's d |
|---|---|---|---|---|---|
| Baseline XGBoost | 36 | 0.7894 ± 0.0215 | 0.6312 ± 0.0447 | n/a | n/a |
| E-XGBoost (pruned) | 19 | 0.8030 ± 0.0211 | 0.6441 ± 0.0412 | 0.0625 | 1.430 |

Pruning removed 47% of predictors with no loss on any metric, and cut mean
training time per fold by roughly a quarter.

Note on the p-value: with five paired folds the smallest two-sided p the
Wilcoxon signed-rank test can return is 0.0625, which occurs when every fold
favours the same model. That is what happened here. The test cannot reach
p < 0.05 at this fold count regardless of effect size, so the real-data
analysis uses ten folds.

---

## Feature schema (36 predictors)

All predictors are available **before** the trainee sits the examination, which
is what keeps the model free of target leakage. Defined once in
[`nmcle_schema.py`](nmcle_schema.py).

| Group | Count | Columns |
|---|---|---|
| Raw numeric | 13 | `programme_cgpa`, `ca_<subject>` (×6), `mock_<subject>` (×6) |
| Engineered numeric | 21 | `ca_avg`, `mock_avg`, `ca_mock_gap`, `ca_consistency`, `mock_consistency`, `n_weak_ca_subjects`, `n_weak_mock_subjects`, `min_ca_score`, `min_mock_score`, `weak_ca_<subject>` (×6), `weak_mock_<subject>` (×6) |
| Categorical | 2 | `age_band`, `gender` |

The six NMC-LE theory papers are medical-surgical, mental health, paediatric,
public health, obstetric, and pharmacology.

**Not predictors.** `programme_type`, `region`, and `cohort_year` are collected
but excluded from the model. The cohort is a single programme at a single
college, so the first two are constant. They serve as fairness strata and drive
the temporal split instead.

**No WASSCE.** The college registry does not hold entry aggregates, so
`wassce_aggregate` is absent from the schema.

Which features E-XGBoost prunes is **not** fixed in code. It is decided at
runtime from the SHAP ranking, and every run prints the kept and pruned sets.

---

## Repository structure

```
xgboost-shap-nursing-licensure/
├── run.sh                        One command: build env + run the analysis
├── app.sh                        One command: launch the educator app
├── run_all.py                    Runs Stage 1 and Stage 2 in one process
│
├── nmcle_schema.py               Feature schema, single source of truth
├── baseline_xgboost.py           Baselines, grid search, seed variance, ablation
├── engineered_xgboost.py         SHAP pruning and E-XGBoost retraining
├── model_comparison.py           Wilcoxon comparison table, Cohen's d
├── stats_validation.py           DeLong, McNemar, fairness, class metrics
├── synthetic_data.py             Pilot generator and SDV field replica
├── prediction.py                 Bundle load/save and record scoring
│
├── notebooks/                    Stages, also pasteable into Colab cell by cell
│   ├── 00_data_preparation_and_eda.py   Clean, exclude, anonymise, EDA
│   ├── 01_pipeline_and_experiments.py   Baselines, XGBoost, SHAP, evaluation
│   ├── 02_model_engineering.py          E-XGBoost against the locked baseline
│   └── 03_hybrid_transfer.py            Set-aside public-data approach
│
├── app/                          Educator risk-screening web app
│   ├── app.py                    Streamlit UI
│   ├── labels.py                 Plain-English feature names
│   └── example_records.csv       Fabricated sample upload
│
├── scripts/                      Manuscript tooling (Markdown to docx, PDF merge)
├── docs/                         Technical documentation and project TODO
├── data/                         Never committed. See Data and Privacy below.
└── results/                      Generated. Never committed.
```

---

## Setup

Requires **Python 3.11**. The pinned libraries (`sdv==1.11.0` in particular) do
not support 3.12 or later.

```bash
brew install python@3.11          # macOS
git clone https://github.com/yhmiller/xgboost-shap-nursing-licensure.git
cd xgboost-shap-nursing-licensure
./run.sh
```

`run.sh` builds the `ml_env` virtual environment on first use, installs
dependencies, then runs Stage 1 and Stage 2 end to end. Re-running is safe; the
setup step is skipped once the environment exists. Add `--verbose` for full
logs, otherwise everything is written to `results/analysis.log`.

### The educator app

```bash
./run.sh     # once, to train and export the model bundle
./app.sh     # launch in your browser
```

Upload a spreadsheet, or download the blank template from the sidebar, or try
`app/example_records.csv`. The app loads `results/real/inference_bundle.pkl` if
it exists and falls back to `results/synthetic/inference_bundle.pkl`, so
retraining on real data updates the app with no code change.

Predictions are decision support, not verdicts.

---

## Run order

| Stage | File | When to run |
|---|---|---|
| 0 | `notebooks/00_data_preparation_and_eda.py` | When the college delivers records. Edit `COLUMN_MAPPING` and `TARGET_RULE` in PREP-CELL 2 first. |
| 1 + 2 | `run_all.py` (via `./run.sh`) | Now on synthetic data. Re-run on real data after pointing Stage 1 CELL 6 at `anonymised_records.csv`. |

Stage 2 must run in the same process as Stage 1, which is why `run_all.py`
exists: it inherits Stage 1's fitted preprocessor and data in memory rather
than rebuilding them.

---

## Methods summary

- **Splits:** stratified 70/15/15, plus a temporal split training on earlier
  cohorts and testing on the latest.
- **Imbalance:** a failure rate between 35% and 65% is treated as natural and
  left alone. Outside that range, `scale_pos_weight` is applied.
- **Primary metric:** AUC-PR, since accuracy misleads under imbalance.
- **Comparison:** five-fold stratified CV on shared folds, Wilcoxon
  signed-rank, Cohen's d for paired samples.
- **Also reported:** DeLong for AUC comparison, McNemar for label
  disagreement, Brier score and reliability curves for calibration, and
  FNR/FPR/equal-opportunity difference across gender, programme, and region.
- **Reproducibility:** `random_state=42` throughout, pinned versions, tuned
  hyperparameters written to `results/**/config.yaml`, and a run manifest
  recording the git commit behind every reported figure.

Full write-up in
[docs/E-XGBoost-technical-documentation.md](docs/E-XGBoost-technical-documentation.md).

---

## Data and privacy

Primary data come from a Ghanaian nursing training college under a data
sharing agreement. **No participant data is in this repository and none has
ever been committed.** `.gitignore` excludes `data/`, all spreadsheet formats,
`results/`, model bundles, and the manuscript and ethics directories.

College index numbers are hashed with SHA-256 in Stage 0 and the plaintext
identifiers are discarded before any analytical step. The linkage table that
maps index numbers to study identifiers is the only artefact connecting a real
person to a licensure outcome; it is held on encrypted storage outside this
repository and is never distributed.

Synthetic data is used for pipeline testing only and never appears in a
reported study result.

The anonymised dataset will be deposited publicly after ethics clearance.

---

## Ethics

Submitted to the KNUST Humanities and Social Sciences Research Ethics Committee
(HuSSREC). Approval reference pending committee issuance. Survey participants
gave written informed consent; responses without affirmed consent were excluded
before analysis.

---

## Requirements

```
python 3.11
xgboost==2.0.3
scikit-learn==1.4.2
shap==0.44.1
sdv==1.11.0
lightgbm
pandas
numpy
scipy
matplotlib
streamlit
cloudpickle
```

Full list in [requirements.txt](requirements.txt).

---

## Citation

```
Miller, P. B. (2026). Explainable Machine Learning for Predicting Nursing
Licensure Examination Failure in Ghana: An XGBoost-SHAP Approach
[MSc thesis]. Kwame Nkrumah University of Science and Technology.
```

---

## License

Shared for academic and research purposes under Creative Commons Attribution
4.0 International (CC BY 4.0). See [LICENSE](LICENSE).
