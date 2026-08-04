# E-XGBoost: SHAP-Guided Feature Pruning for Health Professions Licensure Failure Prediction

**Explainable Machine Learning for Predicting Health Professions Licensure
Examination Failure in Ghana: An XGBoost-SHAP Approach**

MSc Thesis Project | Health Informatics |
Kwame Nkrumah University of Science and Technology (KNUST), Ghana

**Author:** Prince Bortey Miller (Student ID: 22388461)
**Supervisor:** Dr. Eric Opoku Osei, Department of Computer Science, KNUST
**Year:** 2026

---

## Examining body

This study's cohort (Environmental Health, Occupational Health & Safety, and
Occupational Therapy at the Accra School of Hygiene) sits the **Allied Health
Professions Council** licensure examination under the Health Professions
Regulatory Bodies Act, 2013 (Act 857) — **not** the Nursing and Midwifery
Council examination.

| | NMC-LE (nursing) | AHPC-LE (this study) |
|---|---|---|
| Pass mark | 50% | **60%** |
| Sat after | Completion of training | **Internship / national service** |
| Published pass rates | Cited in the literature | **Pass lists only, no denominator** |

Ghanaian nursing licensure studies (Amankwaa et al. 2015 and others) are cited
as **adjacent-profession** evidence for which predictors matter, never as a
source for the failure prevalence.

---

## Overview

Failing a health professions licensure examination at first attempt delays a
graduate from entering practice and imposes repeat costs on trainee and
institution alike. This study engineers **E-XGBoost**, an XGBoost classifier
whose feature set is reduced by SHAP attribution rather than manual selection,
to identify at-risk trainees from academic records the college already
collects.

The study population is **Accra School of Hygiene, Korle-Bu**: 110 students
across Environmental Health, Occupational Health and Safety, and Occupational
Therapy, cohorts 2021 and 2022. These are allied health professions regulated
by Ghana's Allied Health Professions Council.

E-XGBoost is a modified version of XGBoost in which the predictor set is the
only altered component: train a baseline on all 36 predictors, rank inputs by
mean absolute SHAP value, keep the smallest subset explaining 95% of total
attribution, then retrain on identical folds and hyperparameters. Because
nothing else changes, any difference in performance is attributable to the
reduction.

Alongside the analysis pipeline, the repository ships a **Streamlit screening
app** for tutors. Upload a class spreadsheet, get a ranked at-risk list and
per-student SHAP explanations. No coding required.

> **Status.** The numbers below come from a synthetic pilot corpus used to
> verify the pipeline end to end. Real college records are still being linked
> to licensure outcomes. No result in this repository is a finding about any
> real cohort.

---

## Three Contributions

1. **Original field data.** The first Ghanaian health professions licensure
   dataset assembled for machine learning, collected under a data sharing
   agreement and carrying no student identifiers at any point.

2. **XGBoost with SHAP explainability.** Per-student attributions that tell a
   tutor which subject scores drive a given risk estimate, not just that risk
   is high.

3. **E-XGBoost.** SHAP-guided feature pruning compared against a locked
   baseline on identical cross-validation folds, tested with the Wilcoxon
   signed-rank test and reported with Cohen's d.

---

## Key Results (synthetic pilot corpus, not study findings)

Feature selection is nested inside every fold, so the ranking model never sees
held-out labels. 5 folds x 5 repeats = 25 paired observations.

| Model | Features | AUC-ROC | AUC-PR | Wilcoxon p | Cohen's d |
|---|---|---|---|---|---|
| No-skill reference | n/a | 0.5000 | 0.3467 | n/a | n/a |
| Baseline XGBoost | 36 | 0.8377 +/- 0.0307 | 0.7447 +/- 0.0481 | n/a | n/a |
| E-XGBoost (nested) | 21 | 0.8372 +/- 0.0305 | 0.7458 +/- 0.0430 | 0.895 | +0.074 |

**The honest verdict is equivalence, not improvement.** E-XGBoost matches the
baseline while using 15 fewer predictors. Every effect size is negligible and
nothing reaches significance.

An earlier version ranked features on the full dataset before cross-validating,
which let selection see held-out labels. That reported a +0.0057 AUC-PR gain.
Nesting the selection reduces it to +0.0011, so **81% of the apparent
improvement was selection bias** (Vabalas et al., 2019). Both arms are still
run, and the gap between them is reported as a result.

A gain-ranked arm runs on identical folds as a comparator (Wang et al., 2024):
it reaches AUC-PR 0.7464 but retains 29.4 features on average against 21.0 for
SHAP, so SHAP prunes harder for equivalent discrimination.

**Read AUC-PR against prevalence, not against 0.5.** A classifier that ignores
every predictor scores the positive-class prevalence, here 0.3467. Every run
prints the no-skill floors and the PR figure draws the prevalence line.

Calibration is fitted on the validation fold and cannot change ranking:
Brier skill +0.280 to +0.337, AUC unchanged.

---

## Feature schema (36 predictors)

All predictors are available **before** the trainee sits the examination, which
is what keeps the model free of target leakage. Defined once in
[`schema.py`](schema.py).

| Group | Count | Columns |
|---|---|---|
| Raw | 14 | `cgpa`, `total_credits`, `gpa_sem1..6`, `n_courses`, `n_grade_A..E` |
| Engineered | 21 | `gpa_mean/min/max`, `gpa_consistency`, `gpa_trend`, `gpa_first_half`, `gpa_final_half`, `weak_sem1..6`, `n_weak_semesters`, `n_failed`, `fail_rate`, `prop_grade_A..E` |
| Categorical | 1 | `programme` (EH, OHS, OT) |

The college's records are semester-level, so the derived features apply
averaging, consistency, minimum and weakness measures to the semester
trajectory rather than to per-subject scores.

**Not predictors.** `cohort_year` is context only: it drives the temporal split
and never enters the model, since a model keyed on year cannot generalise to a
new cohort. `programme` is both a predictor and the fairness stratum, standing
in for the demographics the college cannot release.

**No demographics, no entry grades, no subject scores.** The registry holds
none of these, and student identifiers are withheld under the college's data
protection rules. Identifiers are generated deterministically instead; see
[docs/real-data-migration-findings.md](docs/real-data-migration-findings.md).

Which features E-XGBoost prunes is **not** fixed in code. It is decided per
fold from that fold's SHAP ranking, and `results/**/feature_stability.csv`
reports how often each feature survived across the 25 folds.

---

## Repository structure

```
xgboost-shap-nursing-licensure/
├── run.sh                        One command: build env + run the analysis
├── app.sh                        One command: launch the educator app
├── run_all.py                    Runs Stage 1 and Stage 2 in one process
│
├── schema.py                     Feature schema, single source of truth
├── real_data.py                  Loads the college dataset for the pipeline
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
├── diagrams/                     Mermaid sources for the manuscript figures
├── scripts/                      Data consolidation, simulation, manuscript tooling
│   ├── consolidate_real_data.py         Workbooks -> tidy dataset
│   ├── build_model_dataset.py           Tidy -> model-ready + generated IDs
│   ├── build_synthetic_alpha.py         Simulated corpus, not reportable
│   ├── run_option2_replication.py       Real against its fitted replica
│   └── sample_size_comparison.py        CI width against n
├── docs/                         Technical documentation and project TODO
├── data/                         Never committed. See Data and Privacy below.
└── results/                      Generated. Never committed.
```

`proposal/`, `local/`, `data/`, and `results/` are excluded from version control.
They hold participant records, ethics paperwork, manuscript drafts, and
regenerable artefacts respectively.

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

| Step | Command | Purpose |
|---|---|---|
| 1 | `scripts/consolidate_real_data.py` | Six college workbooks into one tidy dataset |
| 2 | `scripts/build_model_dataset.py` | Tidy dataset into model-ready features + generated IDs |
| 3 | `./run.sh` | Stage 1 and Stage 2 end to end |

`notebooks/00_data_preparation_and_eda.py` is **superseded** for this delivery.
It was written for a single spreadsheet with named columns to map; the college
released per-programme semester grade matrices instead, which steps 1 and 2
handle.

Stage 2 must run in the same process as Stage 1, which is why `run_all.py`
exists: it inherits Stage 1's fitted preprocessor and data in memory rather
than rebuilding them.

**The switch to real data is automatic.** Stage 1 checks for licensure
outcomes on every run. While they are absent it uses pilot data and writes to
`results/synthetic/`; once the `licensure_fail` column is populated it trains on
the real records, writes to `results/real/`, relabels every figure caption and
repoints the educator app. No flags, no edits.

---

---

## What is left

Current as of 31 July 2026. Ordered by what blocks what.

### Blocked on other people

| Item | Who | Blocks |
|---|---|---|
| **Licensure outcomes** for the 2021/22 cohort | College registrar | Everything. Expected Mon 3 Aug |
| **Confirm source row ordering** in the outcome file, in writing | College registrar | The outcome join. A silent off-by-one corrupts every downstream number without raising an error |
| **One examination or three?** EH, OHS and OT may sit different licensure exams under different councils | College registrar | Whether `licensure_fail` is one outcome or three, with 27, 22 and 10 students behind them. Schema question, not wording |
| **Hybrid-data exemption**, allowing a field-plus-replica design in place of public-plus-field | Dr. Osei | Marking scheme item A6 is an automatic fail without it. Methods §2.2.0 documents and justifies it but cannot grant it |
| **HuSSREC approval reference** | Ethics committee | The placeholder in Methods §2.4 |
| **Additional cohorts** 2018–2020 and 2023–2024 | College registrar | Precision, not performance. See the sample-size note below |
| **AHPC pass-rate source** | Public record at `ahpc.gov.gh` | Chapter 2 §2.2 and the Chapter 1 baseline figure |

### Before the outcomes arrive

- [ ] **Widen the regularisation grid.** `XGB_PARAM_GRID` tunes depth, learning rate and estimators only; `min_child_weight`, `reg_alpha` and `reg_lambda` sit at defaults. At an events-per-predictor ratio near 1.0 this is the single most appropriate lever, and the CAN-DO guide names it directly.
- [ ] **Tune the decision threshold** for sensitivity rather than leaving it at 0.50. A screening tool should favour recall, and the operating point is defensible in Methods.
- [ ] **Generate Figure 4** (precision-recall, baseline against E-XGBoost). `roc_pr_curves.png` plots the four reference classifiers instead, so it does not match its caption.
- [ ] **Generate Figure 5** (mean absolute SHAP, bar chart, top ten). `shap_beeswarm.png` is a different plot answering a different question.
- [ ] **Generate Table 1** (descriptive statistics of the continuous predictors). Produced by the superseded Stage 0; nothing in `results/` corresponds to it.
- [ ] **Render Figures 1–3** from `diagrams/*.mmd` at [mermaid.live](https://mermaid.live/) and export SVG.

### Once the outcomes land

- [ ] Run the two consolidation scripts, then `./run.sh`. The switch to `results/real/` is automatic.
- [ ] Run `scripts/run_option2_replication.py`. It blocks today with a clear message; all four steps are wired and smoke-tested.
- [ ] Report **events per predictor parameter** against Riley et al. (2019) and use it as the statistical justification for predictor reduction.
- [ ] Add a **PSI or KL distribution comparison** between the real corpus and its replica. Step 1 of the CAN-DO diagnostic, and it explains any divergence rather than leaving it to guesswork.
- [ ] Decide a **minimum stratum size** for reporting subgroup fairness. The alpha run produced an equal-opportunity difference of 0.333 on two male trainees; the real cohort is roughly 86% female, so the same problem will recur.

### Manuscripts

- [ ] **Methods is 3,336 words** against a 2,800 ceiling. A condensation deduction is accepted; moving §2.9–2.11 to supplementary brings the body to about 2,500 if an IEEE-length version is needed.
- [ ] **Results is ~1,380 words** against a 1,500 floor. Sections 3.1 and 3.9 expand naturally once real outcomes replace the current corpus.
- [ ] **Chapter 2 has not been started.** 55 verified references are in the pool against a target of 44–49; the work left is selection and argument, not accumulation. Draft §2.6 first, then §2.9.
- [ ] Draft in the **allied health framing from the first sentence**. Do not write nursing prose and retrofit it.
- [ ] Confirm the **merged supervisor feedback PDF** is present and named to match the `Miller_*` pattern. It was not in `proposal/` at the last check.

### Known limits, stated deliberately

Not tasks. Recorded so they are not rediscovered as problems.

- **Sample size is the dominant constraint.** At n=110 the 95% interval on AUC-ROC is roughly ±0.04, so differences below about 0.039 are not resolvable. The observed baseline-versus-E-XGBoost differences are +0.0014 to −0.0005, an order of magnitude below that. Larger cohorts would not raise discrimination; they would narrow the interval. Tripling n narrows it about 1.57×, and most of that comes from the first doubling.
- **Failure signal is weak.** Only 11 of 110 students carry any failing grade, so failure-derived features have almost no variance. SHAP prunes them, correctly.
- **H2 is not testable as written.** It compares academic indicators against demographics the college does not release. The ablation supports a reframe to dominance within the academic indicators.
- **Astin's Input block is unobserved.** No entry qualifications, no demographics. The study operationalises Environment and Output only, and Methods §2.1 says so.

---

## Manuscript figures

[`diagrams/`](diagrams/) holds mermaid sources for the Methods figures. Paste
either file into [mermaid.live](https://mermaid.live/) and export SVG.

| File | Figure |
|---|---|
| `figure1-exgboost-framework.mmd` | The three-phase E-XGBoost pipeline, with the locked baseline feeding both the pruning path and the paired comparison |
| `figure2-replication-design.mmd` | Pipeline replication across the institutional corpus and its synthetic replica |
| `figure3-baseline-vs-engineered.mmd` | Baseline against E-XGBoost, with the SHAP-guided predictor reduction highlighted |

Keeping the source in the repository rather than only an exported image means
the figures stay editable and reviewable in diffs.

---

## Methods summary

- **Splits:** stratified 70/15/15, plus a temporal split training on earlier
  cohorts and testing on the latest.
- **Imbalance:** a failure rate between 35% and 65% is treated as natural and
  left alone. Outside that range, `scale_pos_weight` is applied.
- **Primary metric:** AUC-PR, since accuracy misleads under imbalance. Every
  metric is reported against its no-skill floor, which is prevalence for
  AUC-PR and p(1-p) for the Brier score, not 0.5.
- **Comparison:** five-fold stratified CV on shared folds, Wilcoxon
  signed-rank, Cohen's d for paired samples.
- **Also reported:** DeLong for AUC comparison, McNemar for label
  disagreement, Brier score and reliability curves for calibration, and
  FNR/FPR/equal-opportunity difference. Programme is the only stratum the real
  records support; demographic strata are picked up automatically if they ever
  become available.
- **Reproducibility:** `random_state=42` throughout, pinned versions, tuned
  hyperparameters written to `results/**/config.yaml`, and a run manifest
  recording the git commit behind every reported figure.

Full write-up in
[docs/E-XGBoost-technical-documentation.md](docs/E-XGBoost-technical-documentation.md).

---

## Data and privacy

Primary data come from the Accra School of Hygiene under a data sharing
agreement. **No participant data is in this repository and none has
ever been committed.** `.gitignore` excludes `data/`, all spreadsheet formats,
`results/`, model bundles, and the manuscript and ethics directories.

The delivered workbooks contain no names, index numbers or demographic fields,
so there was nothing to hash. Study identifiers are instead generated
deterministically from non-identifying provenance as
`SOH-<cohort>-<programme>-<serial>`, stable across regenerations. The
consequence is that the college can align licensure outcomes only by source row
order, and that ordering must be confirmed in writing before any join.

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
Miller, P. B. (2026). Explainable Machine Learning for Predicting Health
Professions Licensure Examination Failure in Ghana: An XGBoost-SHAP
Approach [MSc thesis]. Kwame Nkrumah University of Science and Technology.
```

---

## License

Shared for academic and research purposes under Creative Commons Attribution
4.0 International (CC BY 4.0). See [LICENSE](LICENSE).
