# ============================================================
# 01_pipeline_and_experiments.py
# Paste each section into a new Colab cell as indicated.
#
# This notebook is the NARRATIVE/ORCHESTRATION layer: it loads data, calls
# into the reusable modules at the repo root (baseline_xgboost.py,
# stats_validation.py, synthetic_data.py, prediction.py), and renders
# figures/tables. The actual model-training and statistics logic lives in
# those modules so it is reusable and independently testable — see
# docs/TODO.md, "Codebase restructure: extract core logic into modules".
#
# Project : Explainable ML for Predicting AHPC-LE Failure in Ghana
# Author  : Prince Bortey Miller | ID: 22388461 | KNUST
# Supervisor: Dr. Eric Opoku Osei
# ============================================================


# ─────────────────────────────────────────────────────────────
# CELL 1 — Install libraries
# ─────────────────────────────────────────────────────────────
"""
Run this cell ONCE at the start of each Colab session.
It installs the exact versions pinned in requirements.txt.
"""
# !pip install xgboost==2.0.3 lightgbm==4.1.0 scikit-learn==1.4.2 \
#              imbalanced-learn==0.12.3 shap==0.44.1 sdv==1.11.0 \
#              statsmodels==0.14.2 seaborn==0.13.2 --quiet


# ─────────────────────────────────────────────────────────────
# CELL 2 — Imports and global seeds
# ─────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd

# When run as a plain .py script (no Jupyter/Colab), force a non-interactive
# matplotlib backend so plt.show() never blocks waiting for a GUI window to be
# closed — that would hang a headless run forever. Figures are saved to results/
# either way; inline display still works under IPython where get_ipython exists.
import matplotlib
try:
    get_ipython()  # noqa: F821 — only defined inside IPython/Jupyter/Colab
except NameError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# macOS OpenMP guard — must run before `import shap`, which eagerly imports
# torch. LightGBM links Homebrew's libomp while torch bundles its own; whichever
# OpenMP runtime initialises first wins, and LightGBM loads its copy lazily on
# the first .fit(). Forcing that fit here, before torch loads, lets the two
# runtimes coexist instead of segfaulting at training time. No-op on Linux/Colab.
import lightgbm as lgb
lgb.LGBMClassifier(n_estimators=1).fit(np.zeros((4, 1)), [0, 1, 0, 1])

import shap
import xgboost as xgb
from sklearn.metrics import (RocCurveDisplay, PrecisionRecallDisplay,
                             brier_score_loss)
from sklearn.calibration import CalibrationDisplay
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ── Global reproducibility ──────────────────────────────────
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)

# Results are split by dataset source: results/synthetic/ or results/real/
# (see docs/TODO.md, "Performance tracking: baseline model on both datasets")
# — RESULTS_DIR is finalised once DATA_SOURCE is known, in CELL 6.
RESULTS_ROOT = "results"

print("✅ Imports and seeds set. numpy seed:", SEED)
print("   XGBoost:", xgb.__version__,
      "| LightGBM:", lgb.__version__,
      "| SHAP:", shap.__version__)


# ─────────────────────────────────────────────────────────────
# CELL 3 — Column schema (single source of truth: schema.py)
# The schema, engineered-column lists, and engineer_features all live in
# schema.py at the repo root, so subjects/columns never drift between
# notebooks. To change the exam papers or feature set, edit ONLY that file.
#
# run_all.py puts the repo root on sys.path. The block below also finds it for a
# direct `python notebooks/01_...py` run. In Colab, clone the repo and run from
# inside it so schema.py sits on the path.
# ─────────────────────────────────────────────────────────────
import sys

try:
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
except NameError:
    pass  # __file__ is undefined in a pasted Colab cell — rely on the cwd

from schema import (
    GPA_COLS, GRADE_COUNT_COLS,
    NUMERIC_COLS, CATEGORICAL_COLS, ALL_FEATURES, TARGET,
    engineer_features, ENGINEERED_NUMERIC, ALL_NUMERIC, ALL_FEATURES_V2,
)

# Reusable core-logic modules (repo root) — see docs/TODO.md for the map of
# which notebook cell each one used to be.
import synthetic_data
import real_data
import baseline_xgboost
import stats_validation
import prediction

print("✅ Schema defined.")
print(f"   Numeric features : {len(NUMERIC_COLS)}")
print(f"   Categorical features : {len(CATEGORICAL_COLS)}")
print(f"   Total features : {len(ALL_FEATURES)}")
print(f"   Target : '{TARGET}' (1 = fail, 0 = pass)")


# ─────────────────────────────────────────────────────────────
# CELL 4 — Generate synthetic pilot data (SDV)
# IMPORTANT: pipeline testing ONLY.
# This data is NEVER used for model training or evaluation.
# Swap in real anonymised data in CELL 6 and everything
# downstream runs unchanged.
# ─────────────────────────────────────────────────────────────

syn_df = synthetic_data.generate_pilot_data(
    GPA_COLS, GRADE_COUNT_COLS, TARGET, SEED,
    n_reference=30, n_rows=1000,
)

print("✅ Synthetic data generated.")
print(f"   Shape  : {syn_df.shape}")
print(f"   Columns: {list(syn_df.columns)}")
print("\n   Target distribution (synthetic generator output):")
print(syn_df[TARGET].value_counts(normalize=True).rename({0: "Pass", 1: "Fail"}))
print("\n   ⚠️  SYNTHETIC DATA — used for pipeline testing only.")
print("   Real data is loaded in CELL 6 when outcomes are available.")
print("\n   Preview:")
print(syn_df[["cgpa", "gpa_sem1", "gpa_sem6", "n_courses",
              "programme", TARGET]].head(5).to_string(index=False))


# ─────────────────────────────────────────────────────────────
# CELL 5 — Feature engineering
# engineer_features() and the engineered-column lists are defined in
# schema.py (imported in CELL 3). This cell just applies them.
# ─────────────────────────────────────────────────────────────

# Apply to synthetic data
syn_df = engineer_features(syn_df)
print("✅ Feature engineering applied.")

print(f"   Total features after engineering: {len(ALL_FEATURES_V2)}")
print(f"   Engineered features added: {len(ENGINEERED_NUMERIC)}")


# ─────────────────────────────────────────────────────────────
# CELL 6 — Select the data source
# The real college records are used automatically as soon as the licensure
# outcomes are populated in data/model_dataset_2021_2022.xlsx. Until then the
# pipeline falls back to pilot data so the machinery stays exercisable.
# EVERYTHING BELOW THIS CELL RUNS UNCHANGED either way.
#
# To force pilot data even when outcomes exist, set FORCE_PILOT = True.
# ─────────────────────────────────────────────────────────────
FORCE_PILOT = False

# Optional override for exercising the pipeline against a simulated corpus that
# carries fields the real extraction lacks (sex, age band, region). Set
# ALPHA_CORPUS=<path> in the environment. Results land in their own directory
# and every caption is labelled, so an alpha run can never be mistaken for or
# overwrite a real one.
ALPHA_CORPUS = os.environ.get("ALPHA_CORPUS")

# Optional override that keeps the REAL academic records but supplies a
# simulated licensure outcome from a registrar-format return workbook. Results
# land in their own per-prevalence directory and every caption is labelled, so
# a simulated run can never be mistaken for or overwrite a real one.
SYNTHETIC_OUTCOMES = os.environ.get("SYNTHETIC_OUTCOMES")

flow = real_data.summarise()
print(f"ℹ️  College dataset: {flow['records_total']} records | "
      f"outcomes supplied {flow['outcomes_supplied']} | "
      f"cohorts {flow['cohorts']} | programmes {flow['programmes']}")

if ALPHA_CORPUS:
    raw_df = engineer_features(pd.read_excel(ALPHA_CORPUS, sheet_name="SYNTHETIC_data"))
    DATA_SOURCE = "synthetic_alpha"
    print(f"⚠️  SIMULATED ALPHA CORPUS: {len(raw_df)} rows from {ALPHA_CORPUS}")
    print("⚠️  Outcomes here are GENERATED. Nothing from this run is reportable.")
elif SYNTHETIC_OUTCOMES:
    raw_df = real_data.load(outcomes_from=SYNTHETIC_OUTCOMES)
    provenance = raw_df.attrs.get("outcome_provenance", "UNKNOWN")
    prevalence = raw_df.attrs.get("outcome_prevalence", float("nan"))
    if provenance == real_data.REGISTRAR_PROVENANCE:
        # The registrar's genuine return carries OBSERVED outcomes, so it must
        # not land in a real_synthetic_outcome_pNNN/ directory: that NNN is an
        # assumed prevalence, and these outcomes are measured, not assumed.
        DATA_SOURCE = "real"
        print(f"✅ REGISTRAR RETURN: {len(raw_df)} students with OBSERVED outcomes "
              f"from {SYNTHETIC_OUTCOMES}")
        print(f"   Observed failure rate: {raw_df[TARGET].mean():.3f} (measured, not assumed).")
    elif pd.isna(prevalence):
        # Failing here rather than defaulting: a guessed prevalence would name
        # the results directory after a number nothing in the run supports.
        raise ValueError(
            f"{SYNTHETIC_OUTCOMES} declares record_type '{provenance}' but its "
            f"'{real_data.PROVENANCE_SHEET}' sheet supplies no usable "
            "'prevalence' field. A simulated run is filed under "
            "results/real_synthetic_outcome_pNNN/ and NNN comes from that "
            "field, so it cannot be defaulted without mislabelling the "
            "directory. Either add the prevalence to the workbook, or supply "
            "the registrar's genuine return, which needs no provenance sheet "
            "and runs as observed data.")
    else:
        DATA_SOURCE = f"real_synthetic_outcome_p{int(round(prevalence * 1000)):03d}"
        print(f"⚠️  REAL academic records with SIMULATED outcomes: {len(raw_df)} students")
        print(f"⚠️  Prevalence {prevalence} is an ASSUMPTION — no AHPC pass rate is published.")
        print("⚠️  Outcomes here are GENERATED. Nothing from this run is reportable.")
elif FORCE_PILOT:
    raw_df, DATA_SOURCE = syn_df.copy(), "synthetic"
    print("ℹ️  FORCE_PILOT set — using pilot data.")
else:
    try:
        raw_df = real_data.load()
        DATA_SOURCE = "real"
        print(f"✅ Using REAL college records: {len(raw_df)} students with outcomes.")
    except real_data.OutcomeNotSuppliedError as exc:
        raw_df, DATA_SOURCE = syn_df.copy(), "synthetic"
        print(f"ℹ️  {exc}")
        print("ℹ️  Falling back to SYNTHETIC pilot data (pipeline testing mode).")

# Figure captions derive their data-source label from DATA_SOURCE rather than
# hardcoding it, so a real-data run cannot silently ship plots captioned
# "SYNTHETIC DATA".
if DATA_SOURCE.startswith("real_synthetic_outcome"):
    SOURCE_NOTE = (f"Real academic records, SIMULATED outcomes "
                   f"(prevalence {raw_df.attrs.get('outcome_prevalence')}) — not reportable")
else:
    SOURCE_NOTE = {
        "synthetic":       "SYNTHETIC DATA — pipeline testing only",
        "synthetic_alpha": "SIMULATED ALPHA CORPUS — generated outcomes, not reportable",
    }.get(DATA_SOURCE, "Institutional cohort")

# Results land in results/<DATA_SOURCE>/ so a real run and a synthetic run
# never overwrite each other's artefacts (see docs/TODO.md, "Performance
# tracking: baseline model on both datasets").
RESULTS_DIR = os.path.join(RESULTS_ROOT, DATA_SOURCE)
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"   Results directory: {RESULTS_DIR}/")

# ── Validate required columns ───────────────────────────────────
missing = [c for c in ALL_FEATURES_V2 + [TARGET] if c not in raw_df.columns]
if missing:
    raise ValueError(f"❌ Missing columns in dataset: {missing}")

print(f"✅ Dataset loaded: {raw_df.shape[0]} rows × {raw_df.shape[1]} cols")
print(f"\n   Class prevalence:")
vc = raw_df[TARGET].value_counts()
print(f"   Pass (0): {vc.get(0, 0)} ({vc.get(0,0)/len(raw_df)*100:.1f}%)")
print(f"   Fail (1): {vc.get(1, 0)} ({vc.get(1,0)/len(raw_df)*100:.1f}%)")

# ── Imbalance decision note ──────────────────────────────────────
fail_rate = vc.get(1, 0) / len(raw_df)
if 0.35 <= fail_rate <= 0.65:
    print("\n   ✅ Classes roughly balanced — SMOTE may not be needed.")
    print("   Train on natural distribution first; test SMOTE as one strategy.")
else:
    print(f"\n   ⚠️  Class imbalance detected (fail rate={fail_rate:.2f}).")
    print("   Will compare SMOTE / scale_pos_weight / threshold tuning.")


# ─────────────────────────────────────────────────────────────
# CELL 7 — Train / validation / test split
# Two splits applied:
#   1. Stratified random split (70/15/15) — primary
#   2. Temporal split — if cohort year column exists (handled inside
#      baseline_xgboost.temporal_validation in CELL 11.7)
# ─────────────────────────────────────────────────────────────

# Drop predictors that take one value across the whole cohort. A constant
# column cannot be split on, contributes exactly zero SHAP attribution, and
# would otherwise be reported as a feature the model "uses". This matters here
# because programme and region were constant in the single-programme extraction
# the schema was originally written against.
_candidate = raw_df[ALL_FEATURES_V2]
ZERO_VARIANCE = [c for c in ALL_FEATURES_V2 if _candidate[c].nunique(dropna=True) <= 1]
MODEL_FEATURES = [c for c in ALL_FEATURES_V2 if c not in ZERO_VARIANCE]

if ZERO_VARIANCE:
    print(f"ℹ️  Dropped {len(ZERO_VARIANCE)} zero-variance predictor(s): {ZERO_VARIANCE}")
else:
    print("✅ No zero-variance predictors; all "
          f"{len(MODEL_FEATURES)} features retained.")

_missing_pct = (_candidate[MODEL_FEATURES].isna().mean() * 100)
print(f"   Maximum missingness across predictors: {_missing_pct.max():.2f}% "
      f"({int((_missing_pct > 0).sum())} of {len(MODEL_FEATURES)} affected)")

X = raw_df[MODEL_FEATURES]
y = raw_df[TARGET]

# ── Stratified random split ──────────────────────────────────
X_trainval, X_test,   y_trainval, y_test   = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=SEED
)
X_train,    X_val,    y_train,    y_val     = train_test_split(
    X_trainval, y_trainval, test_size=0.15/0.85,
    stratify=y_trainval, random_state=SEED
)

print("✅ Stratified split (70 / 15 / 15):")
print(f"   Train : {X_train.shape[0]} rows | Fail rate: {y_train.mean():.2f}")
print(f"   Val   : {X_val.shape[0]} rows   | Fail rate: {y_val.mean():.2f}")
print(f"   Test  : {X_test.shape[0]} rows  | Fail rate: {y_test.mean():.2f}")

if "cohort_year" in raw_df.columns:
    print(f"\nℹ️  cohort_year present — temporal validation runs in CELL 11.7.")
else:
    print("\nℹ️  No cohort_year column — temporal split skipped.")
    print("   Add cohort_year to real data for temporal validation.")


# ─────────────────────────────────────────────────────────────
# CELL 8 — Preprocessing pipeline (scikit-learn)
# Handles: imputation → scaling → encoding
# This is a reusable sklearn Pipeline object. (baseline_xgboost.build_preprocessor)
# ─────────────────────────────────────────────────────────────

# Built from the surviving features, not the full schema list: a predictor
# dropped for zero variance above is absent from X, and passing it here would
# make the ColumnTransformer ask for a column that no longer exists.
MODEL_NUMERIC = [c for c in ALL_NUMERIC if c in MODEL_FEATURES]
MODEL_CATEGORICAL = [c for c in CATEGORICAL_COLS if c in MODEL_FEATURES]
preprocessor = baseline_xgboost.build_preprocessor(MODEL_NUMERIC, MODEL_CATEGORICAL)

# Quick smoke test
X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc   = preprocessor.transform(X_val)
X_test_proc  = preprocessor.transform(X_test)

FEATURE_NAMES = baseline_xgboost.encoded_feature_names(
    preprocessor, MODEL_NUMERIC, MODEL_CATEGORICAL)

print("✅ Preprocessing pipeline built and fitted.")
print(f"   Input features  : {X_train.shape[1]}")
print(f"   Output features : {X_train_proc.shape[1]} (after OHE expansion)")
print(f"   Train rows      : {X_train_proc.shape[0]}")


# ─────────────────────────────────────────────────────────────
# CELL 9 — Baseline comparator models (Logistic Regression, RF, LightGBM)
# All use the SAME preprocessor from CELL 8. (baseline_xgboost.train_comparator_baselines)
# ─────────────────────────────────────────────────────────────

baseline_results, baseline_models, baseline_probas = baseline_xgboost.train_comparator_baselines(
    X_train_proc, y_train, X_val_proc, y_val, X_test_proc, y_test, SEED)
lr_proba, rf_proba, lgbm_proba = (baseline_probas["Logistic Regression"],
                                   baseline_probas["Random Forest"],
                                   baseline_probas["LightGBM"])

print("✅ Baseline models trained and evaluated on test set.")
stats_validation.print_metrics(baseline_results)

import joblib
joblib.dump(baseline_models["Logistic Regression"], os.path.join(RESULTS_DIR, "lr_baseline.pkl"))
joblib.dump(baseline_models["Random Forest"], os.path.join(RESULTS_DIR, "rf_baseline.pkl"))
joblib.dump(baseline_models["LightGBM"], os.path.join(RESULTS_DIR, "lgbm_baseline.pkl"))
print("\n   Models saved.")


# ─────────────────────────────────────────────────────────────
# CELL 10.5 — Hyperparameter grid search (XGBoost)
# Tune the core hyperparameters by N_CV_FOLDS-fold CV on the training split,
# scoring AUC-PR (the primary metric). The winners feed CELL 11.
# (baseline_xgboost.tune_xgboost)
# ─────────────────────────────────────────────────────────────

best_xgb_params, spw, grid = baseline_xgboost.tune_xgboost(X_train_proc, y_train, SEED)
print(f"ℹ️  scale_pos_weight = {spw}")
print("✅ Grid search complete.")
print(f"   Best params   : {best_xgb_params}")
print(f"   Best CV AUC-PR: {grid.best_score_:.4f}")


# ─────────────────────────────────────────────────────────────
# CELL 11 — XGBoost model (the baseline the whole project compares against)
# Final model uses the grid-search winners + early stopping on validation
# AUC-PR. (baseline_xgboost.train_xgboost)
# ─────────────────────────────────────────────────────────────

xgb_clf, xgb_res, xgb_proba, xgb_labels = baseline_xgboost.train_xgboost(
    X_train_proc, y_train, X_val_proc, y_val, X_test_proc, y_test,
    best_xgb_params, spw, SEED)

print("✅ XGBoost trained.")
print(f"   Best iteration: {xgb_clf.best_iteration}")
all_results = baseline_results + [xgb_res]
stats_validation.print_metrics(all_results)

# Probability calibration. The raw XGBoost score ranks well but is not a
# probability, so the risk percentage the educator app shows a tutor is not
# trustworthy without this. Isotonic needs a few hundred validation points;
# below that Platt scaling is the safer choice, so pick on validation size.
CALIBRATION_METHOD = "isotonic" if len(y_val) >= 200 else "sigmoid"
xgb_calibrated, calib_res, xgb_proba_calibrated = baseline_xgboost.calibrate(
    xgb_clf, X_val_proc, y_val, X_test_proc, y_test, method=CALIBRATION_METHOD)

print(f"\n✅ Probability calibration ({calib_res['method']}, fitted on validation fold):")
print(f"   No-skill Brier            : {calib_res['brier_no_skill']:.4f}")
print(f"   Brier before calibration  : {xgb_res['brier']:.4f} "
      f"(skill {1 - xgb_res['brier'] / calib_res['brier_no_skill']:+.3f})")
print(f"   Brier after  calibration  : {calib_res['brier']:.4f} "
      f"(skill {calib_res['brier_skill_score']:+.3f})")
print(f"   Ranking unchanged by design: AUC-ROC {calib_res['auc_roc']:.4f} | "
      f"AUC-PR {calib_res['auc_pr']:.4f}")

joblib.dump(xgb_clf, os.path.join(RESULTS_DIR, "xgboost_model.pkl"))
print("\n   Model saved → " + RESULTS_DIR + "/xgboost_model.pkl")

# Self-contained bundle for the educator screening app (Phase 2). prediction.save_bundle
# cloudpickles engineer_features WITH its referenced globals (CA_COLS, etc.), so the
# app reproduces the exact training-time transform: raw record → engineer →
# preprocess → predict. Re-running on real data regenerates this automatically.
INFERENCE_BUNDLE = {
    "model": xgb_clf,
    "preprocessor": preprocessor,
    "engineer_features": engineer_features,
    "feature_columns": MODEL_FEATURES,   # order the preprocessor expects
    "encoded_feature_names": FEATURE_NAMES,
    "raw_numeric_columns": NUMERIC_COLS,
    "categorical_columns": CATEGORICAL_COLS,
    "target": TARGET,
    "data_source": DATA_SOURCE,   # "synthetic" or "real" — drives the app banner
}
prediction.save_bundle(INFERENCE_BUNDLE, os.path.join(RESULTS_DIR, "inference_bundle.pkl"))
print("   Inference bundle saved → " + RESULTS_DIR + "/inference_bundle.pkl")


# ─────────────────────────────────────────────────────────────
# CELL 11.3 — Class-wise metrics (confusion matrix, sensitivity/specificity)
# Aggregate AUC-ROC/AUC-PR hide how the model does on Fail specifically —
# sensitivity (recall on Fail) is the number that matters here, since a
# false negative means a genuinely at-risk student is missed.
# (stats_validation.class_metrics)
# ─────────────────────────────────────────────────────────────

xgb_class_metrics = stats_validation.class_metrics(y_test.values, xgb_labels)
stats_validation.print_class_metrics(xgb_class_metrics, model_name="XGBoost (baseline)")


# ─────────────────────────────────────────────────────────────
# CELL 11.5 — Repeated-seed variance
# Retrain the tuned model under 10 random seeds and report the spread
# of test metrics — confirms results are not a single-seed fluke.
# (baseline_xgboost.repeated_seed_variance)
# ─────────────────────────────────────────────────────────────
SEED_RUNS = 10
seed_var_df = baseline_xgboost.repeated_seed_variance(
    X_train_proc, y_train, X_test_proc, y_test, best_xgb_params, spw, n_runs=SEED_RUNS)
seed_var_df.to_csv(os.path.join(RESULTS_DIR, "seed_variance.csv"), index=False)
print(f"✅ Repeated-seed variance ({SEED_RUNS} runs):")
print(f"   AUC-ROC: {seed_var_df['auc_roc'].mean():.4f} ± {seed_var_df['auc_roc'].std():.4f}")
print(f"   AUC-PR : {seed_var_df['auc_pr'].mean():.4f} ± {seed_var_df['auc_pr'].std():.4f}")
print("   Saved → " + RESULTS_DIR + "/seed_variance.csv")


# ─────────────────────────────────────────────────────────────
# CELL 11.7 — Temporal validation (train earlier cohorts → test latest)
# The honest deployment scenario: predict a future cohort from past ones.
# Runs only when cohort_year is present. (baseline_xgboost.temporal_validation)
# ─────────────────────────────────────────────────────────────
temporal_df = baseline_xgboost.temporal_validation(
    raw_df, X, y, preprocessor, best_xgb_params, spw, SEED)
if temporal_df is not None:
    temporal_df.to_csv(os.path.join(RESULTS_DIR, "temporal_validation.csv"), index=False)
    print("✅ Temporal validation (train past cohorts → test latest):")
    print(temporal_df.to_string(index=False))
    print("   Saved → " + RESULTS_DIR + "/temporal_validation.csv")
else:
    print("ℹ️  Temporal validation skipped (no cohort_year, or test cohort single-class).")


# ─────────────────────────────────────────────────────────────
# CELL 12 — SHAP: global feature importance (beeswarm plot)
# Uses TreeExplainer — exact Shapley values for XGBoost.
# ─────────────────────────────────────────────────────────────

explainer   = shap.TreeExplainer(xgb_clf)
shap_values = explainer.shap_values(X_test_proc)

# Convert to DataFrame for analysis
shap_df = pd.DataFrame(
    np.abs(shap_values),
    columns=FEATURE_NAMES
)
global_importance = shap_df.mean().sort_values(ascending=False)

print("✅ SHAP computed.")
print("\n   Top 10 most important features (mean |SHAP|):")
print(global_importance.head(10).to_string())

# Beeswarm plot
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_test_proc,
                  feature_names=FEATURE_NAMES,
                  plot_type="violin",
                  show=False)
plt.title("Global SHAP Feature Importance — AHPC-LE Failure Prediction\n"
          f"({SOURCE_NOTE})",
          fontsize=11, style="italic")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
plt.show()
print("   Plot saved → shap_beeswarm.png")


# ─────────────────────────────────────────────────────────────
# CELL 12.5 — SHAP dependence plots (top-3 predictors)
# How each top predictor's value relates to its SHAP impact on risk.
# ─────────────────────────────────────────────────────────────
import re, glob
# Dependence-plot filenames depend on which features rank top-3, which can shift
# between runs — clear stale ones so results/ only holds the current top-3.
for stale in glob.glob(os.path.join(RESULTS_DIR, "shap_dependence_*.png")):
    os.remove(stale)
top3_features = global_importance.head(3).index.tolist()
for feat in top3_features:
    plt.figure()
    shap.dependence_plot(feat, shap_values, X_test_proc,
                         feature_names=FEATURE_NAMES, show=False)
    plt.title(f"SHAP dependence — {feat}\n({SOURCE_NOTE})",
              fontsize=10, style="italic")
    plt.tight_layout()
    safe = re.sub(r"[^0-9a-zA-Z]+", "_", feat).strip("_")
    plt.savefig(os.path.join(RESULTS_DIR, f"shap_dependence_{safe}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
print(f"✅ SHAP dependence plots (top-3): {top3_features}")


# ─────────────────────────────────────────────────────────────
# CELL 13 — SHAP: per-student waterfall plot (5 at-risk cases)
# ─────────────────────────────────────────────────────────────

shap_explanation = explainer(X_test_proc)

# Find 5 actual failures with highest predicted probability
fail_idx = np.where((y_test.values == 1))[0]
if len(fail_idx) > 0:
    top5 = fail_idx[np.argsort(xgb_proba[fail_idx])[::-1][:5]]
    for rank, idx in enumerate(top5, 1):
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(shap_explanation[idx], show=False, max_display=12)
        plt.title(f"Student {rank} — Predicted Fail (p={xgb_proba[idx]:.3f})\n"
                  f"({SOURCE_NOTE})", fontsize=10, style="italic")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"shap_waterfall_student{rank}.png"),
                    dpi=150, bbox_inches="tight")
        plt.show()
    print(f"✅ Waterfall plots saved for {len(top5)} at-risk students.")
else:
    print("ℹ️  No fail cases in test set — regenerate synthetic data for testing.")


# ─────────────────────────────────────────────────────────────
# CELL 14 — Calibration: Brier score + reliability diagram
# Required for a decision-support tool (H4).
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0,1],[0,1],"k--", label="Perfectly calibrated", linewidth=1)

models_eval = [
    ("Logistic Regression", lr_proba),
    ("Random Forest",       rf_proba),
    ("LightGBM",            lgbm_proba),
    ("XGBoost",             xgb_proba),
]

for mname, proba in models_eval:
    bs = brier_score_loss(y_test, proba)
    disp = CalibrationDisplay.from_predictions(
        y_test, proba, n_bins=8,
        name=f"{mname} (Brier={bs:.4f})",
        ax=ax
    )

ax.set_title(f"Calibration Curves — All Models\n({SOURCE_NOTE})", style="italic")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "calibration_curves.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ Calibration curves saved → calibration_curves.png")

# A raw Brier score is uninterpretable without its no-skill floor: always
# predicting the base rate scores p(1-p), so at 30% prevalence a "good-looking"
# 0.21 is worth nothing. The skill score expresses the gain over that floor.
brier_no_skill = float(y_test.mean() * (1 - y_test.mean()))
print(f"\n   Brier scores (lower = better calibrated)")
print(f"   No-skill floor (always predict base rate): {brier_no_skill:.4f}")
for mname, proba in models_eval:
    brier = brier_score_loss(y_test, proba)
    skill = 1 - (brier / brier_no_skill)
    print(f"   {mname:<25}: {brier:.4f}   skill score {skill:+.3f}")


# ─────────────────────────────────────────────────────────────
# CELL 15 — ROC and Precision-Recall curves
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for mname, proba in models_eval:
    RocCurveDisplay.from_predictions(
        y_test, proba, name=mname, ax=axes[0], plot_chance_level=(mname=="XGBoost")
    )
    PrecisionRecallDisplay.from_predictions(
        y_test, proba, name=mname, ax=axes[1]
    )

# A precision-recall curve has no fixed chance line: a no-skill classifier sits
# at y = prevalence, not at 0.5. Without this reference an AUC-PR of 0.64 reads
# as mediocre when it is in fact roughly twice the no-skill floor.
no_skill = float(y_test.mean())
axes[1].axhline(no_skill, linestyle=":", color="0.35", linewidth=1.6,
                 label=f"No-skill classifier (prevalence = {no_skill:.3f})")
axes[1].legend(loc="upper right", fontsize=8)
axes[1].set_ylim(0, 1)

axes[0].set_title(f"ROC Curves\n({SOURCE_NOTE})", style="italic")
axes[1].set_title(f"Precision-Recall Curves — PRIMARY METRIC\n({SOURCE_NOTE})",
                   style="italic")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_pr_curves.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ ROC + PR curves saved → roc_pr_curves.png")
print(f"   No-skill reference lines — AUC-ROC 0.500 | AUC-PR {no_skill:.4f} "
      f"(prevalence) | Brier {no_skill * (1 - no_skill):.4f}")


# ─────────────────────────────────────────────────────────────
# CELL 16 — DeLong test (AUC comparison: XGBoost vs each baseline)
# Tests H1: XGBoost significantly outperforms Logistic Regression.
# DeLong = correct test for comparing AUC values.
# McNemar = for label disagreement only (see CELL 17).
# (stats_validation.delong_test)
# ─────────────────────────────────────────────────────────────

print("✅ DeLong test (H1 — XGBoost AUC vs baselines):")
print(f"   {'Comparison':<40} {'AUC A':>8} {'AUC B':>8} {'z':>8} {'p-value':>10} {'sig':>5}")
print("   " + "─" * 85)

comparisons = [
    ("XGBoost vs Logistic Regression", xgb_proba, lr_proba),
    ("XGBoost vs Random Forest",       xgb_proba, rf_proba),
    ("XGBoost vs LightGBM",            xgb_proba, lgbm_proba),
]

for label, pa, pb in comparisons:
    auc_a, auc_b, z, p = stats_validation.delong_test(y_test.values, pa, pb)
    sig = "✅" if p < 0.05 else "ns"
    print(f"   {label:<40} {auc_a:>8.4f} {auc_b:>8.4f} {z:>8.3f} {p:>10.4f} {sig:>5}")


# ─────────────────────────────────────────────────────────────
# CELL 17 — McNemar test (classification-label disagreement only)
# NOT for AUC comparison — for comparing which errors differ.
# (stats_validation.mcnemar_test)
# ─────────────────────────────────────────────────────────────

print("✅ McNemar test (classification-label disagreement — NOT AUC):")
print(f"   {'Comparison':<40} {'b':>6} {'c':>6} {'p-value':>10} {'sig':>5}")
print("   " + "─" * 65)
for label, pa, pb in comparisons:
    la = (pa >= 0.5).astype(int)
    lb = (pb >= 0.5).astype(int)
    p, b, c = stats_validation.mcnemar_test(y_test.values, la, lb,
                            label.split(" vs ")[0],
                            label.split(" vs ")[1])
    sig = "✅" if p < 0.05 else "ns"
    print(f"   {label:<40} {b:>6} {c:>6} {p:>10.4f} {sig:>5}")


# ─────────────────────────────────────────────────────────────
# CELL 18 — Fairness disaggregation
# FNR, FPR, Equal Opportunity Difference by group.
# FNR prioritised: a missed failing student gets no support.
# (stats_validation.fairness_metrics)
# ─────────────────────────────────────────────────────────────

# Disaggregate by whichever strata this cohort actually carries. The college
# releases no demographics under its data protection rules, so programme and
# cohort year are what remain. Any demographic column that appears later is
# picked up automatically.
fairness_cols = [c for c in ["programme", "cohort_year", "gender", "age_band", "region"]
                 if c in raw_df.columns]
test_groups = raw_df.loc[X_test.index, fairness_cols].reset_index(drop=True)
xgb_test_labels = (xgb_proba >= 0.5).astype(int)
y_test_arr = y_test.values

print("✅ Fairness disaggregation (FNR prioritised):")
for gcol in fairness_cols:
    print(f"\n   ── By {gcol} ──")
    fair_df = stats_validation.fairness_metrics(y_test_arr, xgb_test_labels, gcol, test_groups)
    print(fair_df.to_string(index=False))
    max_eod = fair_df["EqualOpportunityDiff"].max() if "EqualOpportunityDiff" in fair_df else 0
    if max_eod > 0.10:
        print(f"   ⚠️  Equal Opportunity Difference = {max_eod:.3f} (>0.10 — investigate bias)")
    else:
        print(f"   ✅ Equal Opportunity Difference = {max_eod:.3f} (within acceptable range)")


# ─────────────────────────────────────────────────────────────
# CELL 18.5 — Ablation study (feature-group contribution)
# Retrain the tuned model on reduced feature sets to show each group's
# contribution. N_CV_FOLDS-fold CV AUC-PR; preprocessing refitted inside each fold.
# (baseline_xgboost.run_ablation_study)
# ─────────────────────────────────────────────────────────────
ablation_df = baseline_xgboost.run_ablation_study(
    X, y, ALL_NUMERIC, CATEGORICAL_COLS, MODEL_FEATURES, best_xgb_params, spw, SEED)
ablation_df.to_csv(os.path.join(RESULTS_DIR, "ablation_table.csv"), index=False)
print(f"✅ Ablation study ({baseline_xgboost.N_CV_FOLDS}-fold CV AUC-PR):")
print(ablation_df.to_string(index=False))
print("   Saved → " + RESULTS_DIR + "/ablation_table.csv")


# ─────────────────────────────────────────────────────────────
# CELL 19 — Save config.yaml (all hyperparameters for GitHub)
# ─────────────────────────────────────────────────────────────

config = f"""# config.yaml
# XGBoost hyperparameters — AHPC-LE failure prediction
# Author: Prince Bortey Miller | KNUST 2026

model:
  name: XGBoost
  tuning: GridSearchCV ({baseline_xgboost.N_CV_FOLDS}-fold, scoring=AUC-PR)
  n_estimators: {best_xgb_params.get("n_estimators")}
  max_depth: {best_xgb_params.get("max_depth")}
  learning_rate: {best_xgb_params.get("learning_rate")}
  subsample: {best_xgb_params.get("subsample")}
  colsample_bytree: {best_xgb_params.get("colsample_bytree")}
  scale_pos_weight: {spw}
  early_stopping_rounds: 50
  eval_metric: aucpr
  random_state: {SEED}

split:
  train: 0.70
  val: 0.15
  test: 0.15
  stratify: true
  temporal: true   # train on earlier cohorts, test on latest

preprocessing:
  numeric_imputer: median
  scaler: MinMaxScaler
  categorical_imputer: most_frequent
  encoder: OneHotEncoder

evaluation:
  primary_metric: auc_pr
  auc_comparison_test: DeLong
  label_disagreement_test: McNemar
  calibration: BrierScore + CalibrationDisplay
  fairness_metrics: [FNR, FPR, EqualOpportunityDiff]
  class_metrics: [confusion_matrix, sensitivity, specificity]
  p_threshold: 0.05
"""

with open(os.path.join(RESULTS_DIR, "config.yaml"), "w") as f:
    f.write(config)
print("✅ " + RESULTS_DIR + "/config.yaml saved — commit this to GitHub.")


# ─────────────────────────────────────────────────────────────
# CELL 20 — End-to-end verification summary
# If you see all ✅ below, the pipeline is ready.
# Swap in real data in CELL 6 and re-run from CELL 6 onward.
# ─────────────────────────────────────────────────────────────

CLEAN_RUN = os.environ.get("CLEAN_RUN") == "1"

checks = [
    ("Libraries imported and seeds set",         True),
    ("Schema defined (features + target)",       True),
    ("Synthetic data generated (SDV)",           True),
    ("Feature engineering applied",              True),
    ("Preprocessing pipeline built",             True),
    ("Stratified 70/15/15 split",                True),
    ("Temporal split (cohort_year present)",
     "cohort_year" in raw_df.columns),
    ("Temporal validation (past → latest cohort)",
     temporal_df is not None),
    ("Baseline models trained + evaluated",      True),
    ("Hyperparameter grid search (AUC-PR)",      True),
    ("XGBoost trained + evaluated",              True),
    ("Class-wise metrics (confusion matrix, sensitivity/specificity)", True),
    ("Repeated-seed variance (10 runs)",         True),
    ("SHAP computed (global + per-student)",     True),
    ("SHAP dependence plots (top-3)",            True),
    ("Calibration (Brier + curves)",             True),
    ("ROC + PR curves",                          True),
    ("DeLong test for AUC comparison",           True),
    ("McNemar test for label disagreement",      True),
    ("Fairness (FNR/FPR/EOD) computed",          True),
    ("Ablation study (feature groups)",          True),
    ("config.yaml saved",                        True),
]

all_pass = True
checklist_lines = []
checklist_lines.append("=" * 60)
checklist_lines.append("END-TO-END PIPELINE VERIFICATION SUMMARY")
checklist_lines.append("=" * 60)

for label, status in checks:
    icon = "✅" if status else "⚠️ "
    if not status:
        all_pass = False
    checklist_lines.append(f"  {icon}  {label}")

checklist_lines.append("")
if all_pass:
    checklist_lines.append("🎉🎉 PIPELINE VERIFIED — ready for real data.🎉🎉")
    checklist_lines.append("")
    checklist_lines.append("NEXT STEP:")
    checklist_lines.append("  1. When real anonymised data arrives, update CELL 6:")
    checklist_lines.append("     Uncomment: raw_df = pd.read_csv(REAL_DATA_PATH)")
    checklist_lines.append("     Comment out: raw_df = syn_df.copy()")
    checklist_lines.append("  2. Re-run from CELL 6 onward — no other changes needed.")
    checklist_lines.append("  3. Check class prevalence output in CELL 6 to decide")
    checklist_lines.append("     whether SMOTE / scale_pos_weight is actually needed.")
else:
    checklist_lines.append("⚠️  Some checks failed — review cells above.")

checklist_content = "\n".join(checklist_lines)

if CLEAN_RUN:
    # Save the checklist details to results/<data_source>/verification_checklist.txt
    checklist_path = os.path.join(RESULTS_DIR, "verification_checklist.txt")
    with open(checklist_path, "w", encoding="utf-8") as f:
        f.write(checklist_content)

    if all_pass:
        print(f"\n🎉 Stage 1 pipeline verified successfully! (Verification checklist saved → {RESULTS_DIR}/verification_checklist.txt)")
    else:
        print(f"\n⚠️  Stage 1 pipeline verification checks had warnings. (Details saved → {RESULTS_DIR}/verification_checklist.txt)")
else:
    print(checklist_content)
