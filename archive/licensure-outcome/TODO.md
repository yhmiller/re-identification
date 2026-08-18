# TODO

# Explainable Machine Learning for Predicting Allied Health Licensure Examination Failure in Ghana: An XGBoost-SHAP Approach


## Done / already implemented

### ✅ E-XGBoost (Contribution 3a — SHAP-guided feature pruning)
**Status: implemented, verified end-to-end, runs in `run_all.py`.** This is NOT
a to-do — it already exists in [../notebooks/02_model_engineering.py](../notebooks/02_model_engineering.py)
(ENG-CELL 1–10).

**What it actually does (plain English):**
1. Trains a normal XGBoost on **all** features and locks that as the baseline.
2. Uses **SHAP** to score how much each feature contributes to predictions,
   then sums the one-hot pieces back to each original input column.
3. **Removes** the low-contribution features — keeps only the ones that together
   explain 95% of the total SHAP importance, drops the long tail (Operation R).
4. Retrains XGBoost on this smaller feature set using the **same folds and
   hyperparameters**, so the only thing that changed is the feature set.
5. Compares the two with a **Wilcoxon signed-rank test** and writes the table.

In short: **a leaner XGBoost that uses fewer inputs but performs the same** —
its value is parsimony + cleaner SHAP explanations for allied health educators, NOT a
new algorithm. It is a *feature-level* contribution (changes the inputs, not the
model internals).

- Artefacts: `results/synthetic/baseline_metrics.json`,
  `results/synthetic/engineered_metrics.json`,
  `results/synthetic/comparison_table.csv` (Table 1) — results now split by
  dataset source, see "Codebase restructure" below.
- Last synthetic-data run: 39 → 21 features, every metric equal-or-better (all
  ns), training ~21% faster.
- Full write-up: [E-XGBoost-technical-documentation.md](E-XGBoost-technical-documentation.md).
- ⚠️ The current numbers are synthetic-data illustrative — re-run on real data
  (the real SHAP rankings decide which features are actually pruned).

---

## Codebase restructure: extract core logic into modules

**✅ Done.** `baseline_xgboost.py`, `engineered_xgboost.py`,
`model_comparison.py`, `stats_validation.py`, `synthetic_data.py`, and
`prediction.py` now exist at the repo root; `notebooks/01` and `02` were
rewritten as thin orchestration scripts that import and call them. Verified
by running `ml_env/bin/python run_all.py` end-to-end (populates
`results/synthetic/` unchanged in behaviour) and by booting
`streamlit run app/app.py` to confirm `prediction.py` (moved from
`app/inference.py`) still serves the app correctly.

**Problem (historical):** `notebooks/01_pipeline_and_experiments.py` and
`notebooks/02_model_engineering.py` are large cell-based scripts that mix
data loading, preprocessing, model training, statistics, and plotting in one
file each (`run_all.py` currently `runpy`-executes both in-process just to
share in-memory state between them). That worked for a single pipeline run,
but the upcoming Option 2 work (same pipeline, run once on real data and
once on synthetic data, then compared) will either duplicate hundreds of
lines into a `notebooks/04_...py`, or need the logic to be callable as
functions. Time to extract before that duplication happens, not after.

**Research:** the standard reference for this is
[Cookiecutter Data Science (DrivenData)](https://cookiecutter-data-science.drivendata.org/) —
the most widely adopted convention for structuring exactly this kind of
project. Its core opinion: notebooks stay as the narrative/orchestration
layer (what a reader follows top-to-bottom, what gets pasted into Colab),
while reusable logic — data loading, feature engineering, model training,
prediction, plotting — moves into plain importable modules that both the
notebooks and other consumers (like the Streamlit app) call into. Its
canonical layout is `dataset.py`, `features.py`, `modeling/train.py`,
`modeling/predict.py`, `plots.py`, `config.py`.

**Recommendation — adapt that pattern to this repo's own naming, don't
adopt the full CCDS scaffold wholesale.** This is a solo thesis codebase
that intentionally pastes cells into Colab (`schema.py` already notes
`__file__` is undefined in a pasted cell) — a deep `src/` package would
fight that workflow. Instead, keep the numbered `notebooks/*.py` files as
thin orchestration scripts, and pull their logic out into a flat set of
modules at the repo root, matching the names already proposed:

- [x] `baseline_xgboost.py` — the baseline model training block (LR / RF /
  LightGBM / XGBoost + grid search + seed variance + temporal validation +
  ablation), was CELL 8–11.7 and 18.5 of `01_pipeline_and_experiments.py`.
- [x] `engineered_xgboost.py` — the E-XGBoost SHAP-pruning workflow, was
  ENG-CELL 2–7 and 11 of `02_model_engineering.py`.
- [x] `model_comparison.py` — the Wilcoxon signed-rank comparison table and
  Cohen's d aggregation, was ENG-CELL 8–9 of `02_model_engineering.py`; also
  has `compare_real_vs_synthetic()` ready for Option 2 (not yet called from
  a notebook — needs real data first).
- [x] `stats_validation.py` — `evaluate_model`, `delong_test`,
  `mcnemar_test`, `fairness_metrics`, plus the new `class_metrics()` (see
  "Result tracking" below) and `cohens_d_paired()`. No bootstrap-CI code
  existed to move — the proposal mentions it but the pipeline has only ever
  reported 10-seed variance (see `proposal/Supervisor_Query_Methods.md`);
  still a gap if bootstrap CIs are wanted for real.
- [x] `prediction.py` — moved from `app/inference.py` (it was already
  generic, no real duplication to unify) plus `save_bundle()` /
  `resolve_bundle_path()` so `app/app.py` and both notebooks share one
  bundle load/save path, and the app auto-prefers `results/real/` once it
  exists.
- [x] `synthetic_data.py` — `generate_pilot_data()` (was CELL 4 of
  `01_pipeline_and_experiments.py`) plus `generate_field_replica()`, the
  Option 2 SDV generator — generic and working today, it just has nothing
  real to fit until the college data lands.

Each numbered notebook is now a short script that imports these and calls
them in order — still readable top-to-bottom for an examiner, still
pasteable into Colab, but the logic itself is reusable and independently
testable instead of copy-pasted for every new dataset run.

---

## When the real dataset arrives

### Data-handling: switch to Option 2 (pipeline replication)
Corrected per a closer read of the CAN-DO Lab dataset-handling guide (see
[`../proposal/Notes/CAN-DO_ DATASET_HANDLING.pdf`](../proposal/Notes/CAN-DO_%20DATASET_HANDLING.pdf)
and [`../proposal/Supervisor_Report_DataHandling.md`](../proposal/Supervisor_Report_DataHandling.md)):
the guide names nursing licensure failure directly as a topic with **no
closely-matched public dataset** — UCI Student Performance (Portuguese
secondary students) does not clear the "genuinely comparable" bar for a
health/professional-licensure topic. **Option 2 (framework/pipeline
replication) replaces the earlier Option 1 hybrid-transfer plan.** Pending
supervisor confirmation.

- [x] `synthetic_data.generate_field_replica()` now exists — a generic
  SDV `GaussianCopulaSynthesizer` generator that fits on whatever real
  dataframe it's given. It's correct code today; it just has nothing
  meaningful to fit until real records arrive.
- [ ] Once real data lands: call `generate_field_replica()` on it, run
  `baseline_xgboost` + `engineered_xgboost` unchanged on the replica (writes
  to `results/synthetic_replica/` or similar — a third results subfolder,
  distinct from the pilot `results/synthetic/`), tuned **independently**
  with the same protocol (not the same fitted hyperparameters copied
  across) — the guide's practical default for Option 2.
- [ ] Compare metrics and SHAP importance rankings side by side between the
  real-data run and the synthetic-data run; write this up as evidence the
  pipeline performs consistently, not that the model transfers across
  populations.
- [ ] Set aside `notebooks/03_hybrid_transfer.py` and the UCI public-dataset
  pretrain/fine-tune approach as a side note rather than the main approach
  (data stays in `data/public/` pending confirmation — do not delete).

### Performance tracking: baseline model on both datasets
**✅ Infrastructure done; real-data half still pending.** `results/` is now
split by `DATA_SOURCE` (`notebooks/01_pipeline_and_experiments.py` CELL 6 →
`RESULTS_DIR = results/<synthetic|real>/`, threaded through
`baseline_xgboost.py`, `engineered_xgboost.py`, and `run_all.py`) instead of
one flat directory that every run overwrote. Verified: a full `run_all.py`
run now populates `results/synthetic/` (baseline + E-XGBoost models, metrics,
plots, bundle) cleanly.

- [x] Baseline XGBoost (not just E-XGBoost) already runs and is scored on
  the synthetic/public side via `baseline_xgboost.py` — CELL 11 of
  `01_pipeline_and_experiments.py`.
- [x] Outputs land in `results/synthetic/`, separate from a future
  `results/real/` — confirmed neither can silently overwrite the other.
- [ ] Nothing yet exists in `results/real/` — that half needs the real
  college data (see "Data-handling: switch to Option 2" above).

### Result tracking, class metrics, and hypothesis testing
**✅ Mostly done.**

- [x] Run manifest — `run_all.py` now appends to `results/run_manifest.csv`
  after every run (timestamp, git commit, data source, model, feature
  count, headline metrics), for the baseline held-out-test result and both
  CV means (baseline + E-XGBoost). Confirmed populated by the verification
  run.
- [x] Class-wise metrics — `stats_validation.class_metrics()` (confusion
  matrix, per-class precision/recall/F1, sensitivity/specificity), called
  in `01_pipeline_and_experiments.py` CELL 11.3 right after the baseline
  XGBoost model is trained. Confirmed printing correctly (e.g. Sensitivity
  0.6087 / Specificity 0.6346 on the current synthetic run).
- [x] Hypothesis-testing suite consolidated into `stats_validation.py`
  (DeLong, McNemar, fairness) and `model_comparison.py` (Wilcoxon,
  Cohen's d) — one place for every test.
- [ ] `model_comparison.compare_real_vs_synthetic()` exists but isn't called
  from a notebook yet — needs the real-data CV results from Option 2 first.

### Add a model-level contribution ("Contribution 3b")
E-XGBoost (SHAP pruning) is a *feature-level* contribution. Add ONE genuine
*model-level* change on top of it — verified before/after vs the baseline,
fully reversible, like the shelved monotonic-constraints work.

Pick ONE of the three (full write-up with pros/cons/why in
[E-XGBoost-technical-documentation.md](E-XGBoost-technical-documentation.md) §5):

- [ ] **Monotonic constraints** — *recommended if simplest + interpretability.*
  Already implemented, verified, and backed up at
  [../backups/m_xgboost_monotonic_constraints.py.bak](../backups/m_xgboost_monotonic_constraints.py.bak)
  (has step-by-step restore instructions). Just reinstate and re-run on real data.

- [ ] **Focal-loss objective** — *strongest contribution; attacks class imbalance
  at the loss level.* Use the `imbalance-xgboost` package (Wang et al., 2020) so
  no hand-derived gradients. Must be able to explain the focusing parameter γ.

- [ ] **DART booster** — *small-data overfitting defence, one parameter*
  (`booster='dart'` + `rate_drop`). Weaker fit to the interpretability narrative.

Honorable mention: **probability calibration** (isotonic/Platt) so the app's risk
percentages are trustworthy — small, defensible, aligns with the educator-facing app.

NOTE: avoid plain false-negative weighting / weighted cross-entropy — it is
mathematically equivalent to `scale_pos_weight` (an existing parameter) and an
examiner can dismiss it.

### Methodology chapter writing
KNUST's *Guide for Preparation and Evaluation of Higher Degree Research
Thesis* (School of Graduate Studies, 2016) requires Chapter 3, "Approach and
Methodology," to cover: the research design adopted, the method used to
select the sample, the tools used to collect data, and the steps taken to
screen, combine, and reduce the data in preparation for testing and
validating the model. A draft already exists at
[`../proposal/Manuscript_Methods_Draft.md`](../proposal/Manuscript_Methods_Draft.md),
but Sections 3.2–3.3 describe the old Option 1 hybrid-transfer design and
need rewriting for the Option 2 correction above.

- [ ] Rewrite Section 3.2 (data acquisition) and Section 3.3 (fusion
  strategy) to describe Option 2 — pipeline replication on real vs.
  CTGAN/SDV-synthetic data — in place of the UCI pretrain/fine-tune design.
- [ ] Resolve the draft's flagged "two-track integration" question (E-XGBoost
  pruning the full 39-feature college schema vs. the transfer step's
  narrower 5-feature common schema) — this simplifies under Option 2, since
  the identical full pipeline now runs on both real and synthetic data
  rather than two different schemas.
- [ ] Check the chapter against the KNUST guide's required structure once
  rewritten: design → sampling → data-collection tools → screening/
  combination/reduction → prepared for testing and validation.
- [ ] Fill in the remaining `[➤ FILL ...]` placeholders (region, cohort
  years, N, exclusion counts, missingness, imbalance ratio) once real data
  is in hand.

### Re-run everything on real data
- [ ] Run Stage 0 (anonymisation + EDA) on the delivered records.
- [ ] Re-run `run_all.py` — the SHAP pruning result and any 3b numbers are
  illustrative on synthetic data; only the real cohort's rankings count.
