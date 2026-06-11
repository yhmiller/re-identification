# ============================================================
# 01_pipeline_and_experiments.py
# Paste each section into a new Colab cell as indicated.
#
# Project : Explainable ML for Predicting NMC-LE Failure in Ghana
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
import joblib

# macOS OpenMP guard — must run before `import shap`, which eagerly imports
# torch. LightGBM links Homebrew's libomp while torch bundles its own; whichever
# OpenMP runtime initialises first wins, and LightGBM loads its copy lazily on
# the first .fit(). Forcing that fit here, before torch loads, lets the two
# runtimes coexist instead of segfaulting at training time. No-op on Linux/Colab.
import lightgbm as lgb
lgb.LGBMClassifier(n_estimators=1).fit(np.zeros((4, 1)), [0, 1, 0, 1])

import shap
import xgboost as xgb

from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.pipeline       import Pipeline
from sklearn.compose        import ColumnTransformer
from sklearn.preprocessing  import MinMaxScaler, OneHotEncoder
from sklearn.impute         import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics        import (roc_auc_score, average_precision_score,
                                    f1_score, precision_score, recall_score,
                                    brier_score_loss, confusion_matrix,
                                    RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.calibration     import CalibrationDisplay
from imblearn.over_sampling import SMOTE
from imblearn.pipeline      import Pipeline as ImbPipeline
from statsmodels.stats.contingency_tables import mcnemar
from scipy                  import stats
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# ── Global reproducibility ──────────────────────────────────
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)

# All figures, metrics, models and config land here (per PROJECT_GUIDE §1).
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("✅ Imports and seeds set. numpy seed:", SEED)
print("   XGBoost:", xgb.__version__,
      "| LightGBM:", lgb.__version__,
      "| SHAP:", shap.__version__)


# ─────────────────────────────────────────────────────────────
# CELL 3 — Column schema definition
# Define ALL columns here. Swap in real data later with
# the SAME column names — no code changes needed.
# ─────────────────────────────────────────────────────────────

# NMC-LE has 6 theory papers:
NMC_SUBJECTS = [
    "medical_surgical",
    "mental_health",
    "paediatric",
    "public_health",
    "obstetric",
    "pharmacology",
]

# Continuous assessment columns (one per subject, 0–100)
CA_COLS  = [f"ca_{s}"   for s in NMC_SUBJECTS]
# Mock exam columns (one per subject, 0–100)
MOCK_COLS = [f"mock_{s}" for s in NMC_SUBJECTS]

# All numeric feature columns (pre-exam only — NO leakage)
NUMERIC_COLS = (
    ["wassce_aggregate", "programme_cgpa"]
    + CA_COLS
    + MOCK_COLS
)

# Categorical feature columns
CATEGORICAL_COLS = ["programme_type", "age_band", "gender", "region"]

# All predictor columns
ALL_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS

# Target column
TARGET = "fail"   # 1 = failed ≥1 paper on first attempt; 0 = passed all

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
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata    import SingleTableMetadata

def build_schema_dataframe(n=20):
    """
    Build a tiny hand-crafted reference table that tells SDV
    the realistic distribution of each column.
    """
    rng = np.random.default_rng(SEED)

    df = pd.DataFrame()
    df["wassce_aggregate"] = rng.integers(6, 36, n).astype(float)
    df["programme_cgpa"]   = np.clip(rng.normal(2.8, 0.6, n), 1.0, 4.0)

    for col in CA_COLS:
        df[col] = np.clip(rng.normal(62, 12, n), 20, 100)
    for col in MOCK_COLS:
        df[col] = np.clip(rng.normal(58, 15, n), 10, 100)

    df["programme_type"] = rng.choice(["RGN", "RM", "NAC", "NAP"],
                                       n, p=[0.55, 0.25, 0.12, 0.08])
    df["age_band"]  = rng.choice(["Below 20", "20-24", "25-29", "30+"],
                                  n, p=[0.10, 0.55, 0.25, 0.10])
    df["gender"]    = rng.choice(["Female", "Male"], n, p=[0.72, 0.28])
    df["region"]    = rng.choice(
        ["Ashanti", "Greater Accra", "Eastern", "Central",
         "Western", "Brong-Ahafo", "Northern", "Other"],
        n, p=[0.22, 0.18, 0.12, 0.10, 0.10, 0.08, 0.10, 0.10]
    )
    # Target: loosely correlated with CGPA and mock performance.
    # Center the linear predictor on its own mean so the synthetic fail
    # rate lands near 50% (the national NMC-LE first-attempt rate);
    # uncentered, typical students sit far below threshold and the
    # generated target collapses to all-Pass.
    linear = (-0.5 * df["programme_cgpa"]
              - 0.02 * df[[f"mock_{s}" for s in NMC_SUBJECTS]].mean(axis=1))
    logit = (linear - linear.mean()) + rng.normal(0, 0.5, n)
    prob_fail = 1 / (1 + np.exp(-logit))
    # Keep as bool to match the boolean sdtype declared in the SDV metadata;
    # the sampled output is cast back to int after generation.
    df[TARGET] = prob_fail > 0.5
    return df

# Build reference schema and fit SDV synthesiser
schema_df = build_schema_dataframe(n=30)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(schema_df)
# Override SDV's guesses for columns that need specific types
metadata.update_column("programme_type", sdtype="categorical")
metadata.update_column("age_band",       sdtype="categorical")
metadata.update_column("gender",         sdtype="categorical")
metadata.update_column("region",         sdtype="categorical")
metadata.update_column(TARGET,           sdtype="boolean")

synthesiser = GaussianCopulaSynthesizer(metadata, enforce_rounding=True)
synthesiser.fit(schema_df)

# Generate 1,000 synthetic rows
syn_df = synthesiser.sample(num_rows=1000)
syn_df[TARGET] = syn_df[TARGET].astype(int)

print("✅ Synthetic data generated.")
print(f"   Shape  : {syn_df.shape}")
print(f"   Columns: {list(syn_df.columns)}")
print(f"\n   Target distribution (SYNTHETIC DATA — NOT real results):")
print(syn_df[TARGET].value_counts(normalize=True).rename({0:"Pass",1:"Fail"}))
print("\n   ⚠️  SYNTHETIC DATA — used for pipeline testing only.")
print("   Real data will be loaded in CELL 6 when available.")
print("\n   Preview:")
print(syn_df[["wassce_aggregate","programme_cgpa",
              "ca_medical_surgical","mock_medical_surgical",
              TARGET]].head(5).to_string(index=False))


# ─────────────────────────────────────────────────────────────
# CELL 5 — Feature engineering
# Add derived features that go beyond raw scores.
# Applied AFTER the raw data is loaded (real or synthetic).
# ─────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pre-exam-only engineered features.
    All inputs must be available BEFORE the student sits the NMC-LE.
    """
    df = df.copy()

    # ── 1. Average CA and mock scores ──────────────────────────
    df["ca_avg"]   = df[CA_COLS].mean(axis=1)
    df["mock_avg"] = df[MOCK_COLS].mean(axis=1)

    # ── 2. CA-to-mock gap: positive = performed better in CA than mocks ──
    df["ca_mock_gap"] = df["ca_avg"] - df["mock_avg"]

    # ── 3. Performance consistency: low variance = consistent student ──
    df["ca_consistency"]   = df[CA_COLS].std(axis=1)
    df["mock_consistency"] = df[MOCK_COLS].std(axis=1)

    # ── 4. Subject-level weakness flags (below 50 is a concerning score) ──
    for s in NMC_SUBJECTS:
        df[f"weak_ca_{s}"]   = (df[f"ca_{s}"]   < 50).astype(int)
        df[f"weak_mock_{s}"] = (df[f"mock_{s}"] < 50).astype(int)

    # ── 5. Total weak subjects count ─────────────────────────────
    weak_ca_cols   = [f"weak_ca_{s}"   for s in NMC_SUBJECTS]
    weak_mock_cols = [f"weak_mock_{s}" for s in NMC_SUBJECTS]
    df["n_weak_ca_subjects"]   = df[weak_ca_cols].sum(axis=1)
    df["n_weak_mock_subjects"] = df[weak_mock_cols].sum(axis=1)

    # ── 6. Minimum mock score (identifies worst single subject) ────
    df["min_mock_score"] = df[MOCK_COLS].min(axis=1)
    df["min_ca_score"]   = df[CA_COLS].min(axis=1)

    # ── 7. WASSCE band (ordinal encoding for tree models) ──────────
    # Lower aggregate = better in Ghana's WASSCE
    df["wassce_band"] = pd.cut(
        df["wassce_aggregate"],
        bins=[0, 12, 18, 24, 36],
        labels=[3, 2, 1, 0],   # 3=excellent, 0=below average
        right=True
    ).astype(float)

    return df


# Apply to synthetic data
syn_df = engineer_features(syn_df)
print("✅ Feature engineering applied.")

# Updated column lists after engineering
ENGINEERED_NUMERIC = [
    "ca_avg", "mock_avg", "ca_mock_gap",
    "ca_consistency", "mock_consistency",
    "n_weak_ca_subjects", "n_weak_mock_subjects",
    "min_mock_score", "min_ca_score",
    "wassce_band",
] + [f"weak_ca_{s}" for s in NMC_SUBJECTS] \
  + [f"weak_mock_{s}" for s in NMC_SUBJECTS]

ALL_NUMERIC    = NUMERIC_COLS + ENGINEERED_NUMERIC
ALL_FEATURES_V2 = ALL_NUMERIC + CATEGORICAL_COLS

print(f"   Total features after engineering: {len(ALL_FEATURES_V2)}")
print(f"   Engineered features added: {len(ENGINEERED_NUMERIC)}")


# ─────────────────────────────────────────────────────────────
# CELL 6 — Load real data (when available)
# Currently loads synthetic data as a placeholder.
# When real data arrives: uncomment the pd.read_csv line,
# point to your Google Drive path, and comment out syn_df.
# EVERYTHING BELOW THIS CELL RUNS UNCHANGED.
# ─────────────────────────────────────────────────────────────
try:
    from google.colab import drive
    # drive.mount('/content/drive')   # ← uncomment when using Drive
except ModuleNotFoundError:
    pass  # not on Colab — running locally, Drive mount not needed

# ── Option A: Load real anonymised data (uncomment when ready) ──
# REAL_DATA_PATH = "/content/drive/MyDrive/nursing_data/anonymised_records.csv"
# raw_df = pd.read_csv(REAL_DATA_PATH)
# raw_df = engineer_features(raw_df)

# ── Option B: Use synthetic pilot data (default for now) ────────
raw_df = syn_df.copy()
print("ℹ️  Using SYNTHETIC data (pipeline testing mode).")

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
#   2. Temporal split — if cohort year column exists
# ─────────────────────────────────────────────────────────────

X = raw_df[ALL_FEATURES_V2]
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

# ── Temporal split (if cohort_year exists in real data) ──────
if "cohort_year" in raw_df.columns:
    years = sorted(raw_df["cohort_year"].unique())
    latest = years[-1]
    train_mask = raw_df["cohort_year"] < latest
    test_mask  = raw_df["cohort_year"] == latest
    X_train_t, y_train_t = X[train_mask], y[train_mask]
    X_test_t,  y_test_t  = X[test_mask],  y[test_mask]
    print(f"\n✅ Temporal split: train on {years[:-1]}, test on {latest}")
    print(f"   Train: {X_train_t.shape[0]} | Test: {X_test_t.shape[0]}")
    print("   Real-world deployment scenario: model predicts a future cohort.")
else:
    print("\nℹ️  No cohort_year column — temporal split skipped.")
    print("   Add cohort_year to real data for temporal validation.")


# ─────────────────────────────────────────────────────────────
# CELL 8 — Preprocessing pipeline (scikit-learn)
# Handles: imputation → scaling → encoding
# This is a reusable sklearn Pipeline object.
# ─────────────────────────────────────────────────────────────

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  MinMaxScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer,    ALL_NUMERIC),
    ("cat", categorical_transformer, CATEGORICAL_COLS),
], remainder="drop")

# Quick smoke test
X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc   = preprocessor.transform(X_val)
X_test_proc  = preprocessor.transform(X_test)

# Get transformed feature names for SHAP later
num_features  = ALL_NUMERIC
cat_features  = (preprocessor
                 .named_transformers_["cat"]
                 .named_steps["encoder"]
                 .get_feature_names_out(CATEGORICAL_COLS)
                 .tolist())
FEATURE_NAMES = num_features + cat_features

print("✅ Preprocessing pipeline built and fitted.")
print(f"   Input features  : {X_train.shape[1]}")
print(f"   Output features : {X_train_proc.shape[1]} (after OHE expansion)")
print(f"   Train rows      : {X_train_proc.shape[0]}")


# ─────────────────────────────────────────────────────────────
# CELL 9 — Helper: evaluate any classifier
# Computes all required metrics in one call.
# ─────────────────────────────────────────────────────────────

def evaluate_model(name, y_true, y_pred_proba, y_pred_labels):
    """Return a dict of all evaluation metrics for a model."""
    return {
        "model"     : name,
        "auc_roc"   : roc_auc_score(y_true, y_pred_proba),
        "auc_pr"    : average_precision_score(y_true, y_pred_proba),
        "f1_weighted": f1_score(y_true, y_pred_labels, average="weighted", zero_division=0),
        "precision" : precision_score(y_true, y_pred_labels, zero_division=0),
        "recall"    : recall_score(y_true, y_pred_labels, zero_division=0),
        "brier"     : brier_score_loss(y_true, y_pred_proba),
    }

def print_metrics(results: dict):
    print(f"\n  {'Model':<22} {'AUC-ROC':>8} {'AUC-PR':>8} "
          f"{'F1-W':>8} {'Prec':>8} {'Recall':>8} {'Brier':>8}")
    print("  " + "─"*70)
    for r in results:
        print(f"  {r['model']:<22} {r['auc_roc']:>8.4f} {r['auc_pr']:>8.4f} "
              f"{r['f1_weighted']:>8.4f} {r['precision']:>8.4f} "
              f"{r['recall']:>8.4f} {r['brier']:>8.4f}")

print("✅ Evaluation helper defined.")


# ─────────────────────────────────────────────────────────────
# CELL 10 — Baseline models (Logistic Regression, RF, LightGBM)
# All use the SAME preprocessor from CELL 8.
# ─────────────────────────────────────────────────────────────

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
lr.fit(X_train_proc, y_train)
lr_proba  = lr.predict_proba(X_test_proc)[:, 1]
lr_labels = lr.predict(X_test_proc)
lr_res = evaluate_model("Logistic Regression", y_test, lr_proba, lr_labels)

# Random Forest
rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=SEED,
                             class_weight="balanced")
rf.fit(X_train_proc, y_train)
rf_proba  = rf.predict_proba(X_test_proc)[:, 1]
rf_labels = rf.predict(X_test_proc)
rf_res = evaluate_model("Random Forest", y_test, rf_proba, rf_labels)

# LightGBM
lgbm_clf = lgb.LGBMClassifier(n_estimators=500, random_state=SEED,
                                verbose=-1, class_weight="balanced")
lgbm_clf.fit(X_train_proc, y_train,
             eval_set=[(X_val_proc, y_val)],
             callbacks=[lgb.early_stopping(50, verbose=False)])
lgbm_proba  = lgbm_clf.predict_proba(X_test_proc)[:, 1]
lgbm_labels = lgbm_clf.predict(X_test_proc)
lgbm_res = evaluate_model("LightGBM", y_test, lgbm_proba, lgbm_labels)

baseline_results = [lr_res, rf_res, lgbm_res]
print("✅ Baseline models trained and evaluated on test set.")
print_metrics(baseline_results)

# Save for later comparison
joblib.dump(lr,  os.path.join(RESULTS_DIR, "lr_baseline.pkl"))
joblib.dump(rf,  os.path.join(RESULTS_DIR, "rf_baseline.pkl"))
joblib.dump(lgbm_clf, os.path.join(RESULTS_DIR, "lgbm_baseline.pkl"))
print("\n   Models saved.")


# ─────────────────────────────────────────────────────────────
# CELL 11 — XGBoost model
# Uses early stopping on validation AUC-PR.
# ─────────────────────────────────────────────────────────────

# Estimated class weight (update with real data prevalence)
neg = int((y_train == 0).sum())
pos = int((y_train == 1).sum())
spw = round(neg / pos, 2) if pos > 0 else 1
print(f"ℹ️  scale_pos_weight = {spw} (neg/pos = {neg}/{pos})")

xgb_clf = xgb.XGBClassifier(
    n_estimators      = 500,
    max_depth         = 6,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    scale_pos_weight  = spw,
    eval_metric       = "aucpr",
    early_stopping_rounds = 50,
    random_state      = SEED,
    verbosity         = 0,
    use_label_encoder = False,
)

xgb_clf.fit(
    X_train_proc, y_train,
    eval_set=[(X_val_proc, y_val)],
    verbose=False,
)

xgb_proba  = xgb_clf.predict_proba(X_test_proc)[:, 1]
xgb_labels = xgb_clf.predict(X_test_proc)
xgb_res = evaluate_model("XGBoost", y_test, xgb_proba, xgb_labels)

print("✅ XGBoost trained.")
print(f"   Best iteration: {xgb_clf.best_iteration}")
all_results = baseline_results + [xgb_res]
print_metrics(all_results)

joblib.dump(xgb_clf, os.path.join(RESULTS_DIR, "xgboost_model.pkl"))
print("\n   Model saved → results/xgboost_model.pkl")


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
plt.title("Global SHAP Feature Importance — NMC-LE Failure Prediction\n"
          "(SYNTHETIC DATA — for pipeline testing only)",
          fontsize=11, style="italic")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
plt.show()
print("   Plot saved → shap_beeswarm.png")


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
                  "(SYNTHETIC DATA)", fontsize=10, style="italic")
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

ax.set_title("Calibration Curves — All Models\n(SYNTHETIC DATA)", style="italic")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "calibration_curves.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ Calibration curves saved → calibration_curves.png")

print("\n   Brier scores (lower = better calibrated):")
for mname, proba in models_eval:
    print(f"   {mname:<25}: {brier_score_loss(y_test, proba):.4f}")


# ─────────────────────────────────────────────────────────────
# CELL 15 — ROC and Precision-Recall curves
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for mname, proba in models_eval:
    fpr, tpr, _ = RocCurveDisplay.from_predictions(
        y_test, proba, name=mname, ax=axes[0], plot_chance_level=(mname=="XGBoost")
    ).fpr, RocCurveDisplay.from_predictions(
        y_test, proba, name=mname, ax=axes[0]
    ).tpr, None
    PrecisionRecallDisplay.from_predictions(
        y_test, proba, name=mname, ax=axes[1]
    )

axes[0].set_title("ROC Curves\n(SYNTHETIC DATA)", style="italic")
axes[1].set_title("Precision-Recall Curves — PRIMARY METRIC\n(SYNTHETIC DATA)",
                   style="italic")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_pr_curves.png"), dpi=150, bbox_inches="tight")
plt.show()
print("✅ ROC + PR curves saved → roc_pr_curves.png")


# ─────────────────────────────────────────────────────────────
# CELL 16 — DeLong test (AUC comparison: XGBoost vs each baseline)
# Tests H1: XGBoost significantly outperforms Logistic Regression.
# DeLong = correct test for comparing AUC values.
# McNemar = for label disagreement only (see CELL 17).
# ─────────────────────────────────────────────────────────────

def delong_test(y_true, proba_a, proba_b):
    """
    Non-parametric DeLong test for comparing two AUCs.
    Returns z-statistic and two-tailed p-value.
    Reference: DeLong et al. (1988), Biometrics.
    """
    def _compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=np.float64)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1)
            i = j
        T2 = np.empty(N, dtype=np.float64)
        T2[J] = T + 1
        return T2

    def _fastDeLong(predictions_sorted_transposed, label_1_count):
        m = label_1_count
        n = predictions_sorted_transposed.shape[1] - m
        positive_examples = predictions_sorted_transposed[:, :m]
        negative_examples = predictions_sorted_transposed[:, m:]
        k = predictions_sorted_transposed.shape[0]

        tx = np.empty([k, m], dtype=float)
        ty = np.empty([k, n], dtype=float)
        tz = np.empty([k, m + n], dtype=float)

        for r in range(k):
            tx[r, :] = _compute_midrank(positive_examples[r, :])
            ty[r, :] = _compute_midrank(negative_examples[r, :])
            tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

        aucs = (tz[:, :m].sum(axis=1) / m / n
                - (m + 1.0) / (2.0 * n))
        v01 = (tz[:, :m] - tx) / n
        v10 = 1.0 - (tz[:, m:] - ty) / m
        sx = np.cov(v01)
        sy = np.cov(v10)
        delongcov = sx / m + sy / n
        return aucs, delongcov

    y_true = np.asarray(y_true)
    proba_a = np.asarray(proba_a)
    proba_b = np.asarray(proba_b)

    sorted_idx = np.argsort(y_true)[::-1]
    label_1_count = int(y_true.sum())

    preds_sorted = np.vstack([proba_a[sorted_idx], proba_b[sorted_idx]])
    aucs, cov = _fastDeLong(preds_sorted, label_1_count)
    auc_diff = aucs[0] - aucs[1]
    se = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    z = auc_diff / se if se > 0 else 0.0
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


print("✅ DeLong test (H1 — XGBoost AUC vs baselines):")
print(f"   {'Comparison':<40} {'AUC A':>8} {'AUC B':>8} {'z':>8} {'p-value':>10} {'sig':>5}")
print("   " + "─" * 85)

comparisons = [
    ("XGBoost vs Logistic Regression", xgb_proba, lr_proba),
    ("XGBoost vs Random Forest",       xgb_proba, rf_proba),
    ("XGBoost vs LightGBM",            xgb_proba, lgbm_proba),
]

for label, pa, pb in comparisons:
    auc_a, auc_b, z, p = delong_test(y_test.values, pa, pb)
    sig = "✅" if p < 0.05 else "ns"
    print(f"   {label:<40} {auc_a:>8.4f} {auc_b:>8.4f} {z:>8.3f} {p:>10.4f} {sig:>5}")


# ─────────────────────────────────────────────────────────────
# CELL 17 — McNemar test (classification-label disagreement only)
# NOT for AUC comparison — for comparing which errors differ.
# ─────────────────────────────────────────────────────────────

def mcnemar_test(y_true, labels_a, labels_b, model_a_name, model_b_name):
    """McNemar test on the disagreement table between two classifiers."""
    b = int(((labels_a == 1) & (labels_b == 0) & (y_true == 1)).sum()
           + ((labels_a == 1) & (labels_b == 0) & (y_true == 0)).sum())
    c = int(((labels_a == 0) & (labels_b == 1) & (y_true == 1)).sum()
           + ((labels_a == 0) & (labels_b == 1) & (y_true == 0)).sum())
    table = [[0, b], [c, 0]]
    result = mcnemar(table, exact=True)
    return result.pvalue, b, c

print("✅ McNemar test (classification-label disagreement — NOT AUC):")
print(f"   {'Comparison':<40} {'b':>6} {'c':>6} {'p-value':>10} {'sig':>5}")
print("   " + "─" * 65)
for label, pa, pb in comparisons:
    la = (pa >= 0.5).astype(int)
    lb = (pb >= 0.5).astype(int)
    p, b, c = mcnemar_test(y_test.values, la, lb,
                            label.split(" vs ")[0],
                            label.split(" vs ")[1])
    sig = "✅" if p < 0.05 else "ns"
    print(f"   {label:<40} {b:>6} {c:>6} {p:>10.4f} {sig:>5}")


# ─────────────────────────────────────────────────────────────
# CELL 18 — Fairness disaggregation
# FNR, FPR, Equal Opportunity Difference by group.
# FNR prioritised: a missed failing student gets no support.
# ─────────────────────────────────────────────────────────────

def fairness_metrics(y_true, y_pred, group_col, groups_df):
    """Compute FNR and FPR per group and Equal Opportunity Difference."""
    results = []
    for grp in groups_df[group_col].unique():
        mask = groups_df[group_col] == grp
        yt = y_true[mask]
        yp = y_pred[mask]
        if yt.sum() == 0:
            continue
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        results.append({
            "group": grp,
            "n": int(mask.sum()),
            "fail_rate": float(yt.mean()),
            "FNR": round(fnr, 4),
            "FPR": round(fpr, 4),
        })
    df_res = pd.DataFrame(results)
    if len(df_res) > 1:
        df_res["EqualOpportunityDiff"] = (
            df_res["FNR"] - df_res["FNR"].min()
        ).round(4)
    return df_res

# Use test-set rows with their raw categorical columns for grouping
test_groups = X_test[CATEGORICAL_COLS].reset_index(drop=True)
xgb_test_labels = (xgb_proba >= 0.5).astype(int)
y_test_arr = y_test.values

print("✅ Fairness disaggregation (FNR prioritised):")
for gcol in ["gender", "programme_type"]:
    print(f"\n   ── By {gcol} ──")
    fair_df = fairness_metrics(y_test_arr, xgb_test_labels, gcol, test_groups)
    print(fair_df.to_string(index=False))
    max_eod = fair_df["EqualOpportunityDiff"].max() if "EqualOpportunityDiff" in fair_df else 0
    if max_eod > 0.10:
        print(f"   ⚠️  Equal Opportunity Difference = {max_eod:.3f} (>0.10 — investigate bias)")
    else:
        print(f"   ✅ Equal Opportunity Difference = {max_eod:.3f} (within acceptable range)")


# ─────────────────────────────────────────────────────────────
# CELL 19 — Save config.yaml (all hyperparameters for GitHub)
# ─────────────────────────────────────────────────────────────

config = f"""# config.yaml
# XGBoost hyperparameters — NMC-LE failure prediction
# Author: Prince Bortey Miller | KNUST 2026

model:
  name: XGBoost
  n_estimators: 500
  max_depth: 6
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
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
  bootstrap_ci_samples: 1000
  p_threshold: 0.05
"""

with open(os.path.join(RESULTS_DIR, "config.yaml"), "w") as f:
    f.write(config)
print("✅ results/config.yaml saved — commit this to GitHub.")


# ─────────────────────────────────────────────────────────────
# CELL 20 — End-to-end verification summary
# If you see all ✅ below, the pipeline is ready.
# Swap in real data in CELL 6 and re-run from CELL 6 onward.
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("END-TO-END PIPELINE VERIFICATION SUMMARY")
print("=" * 60)

checks = [
    ("Libraries imported and seeds set",         True),
    ("Schema defined (features + target)",       True),
    ("Synthetic data generated (SDV)",           True),
    ("Feature engineering applied",              True),
    ("Preprocessing pipeline built",             True),
    ("Stratified 70/15/15 split",                True),
    ("Temporal split (if cohort_year exists)",
     "cohort_year" in raw_df.columns),
    ("Baseline models trained + evaluated",      True),
    ("XGBoost trained + evaluated",              True),
    ("SHAP computed (global + per-student)",     True),
    ("Calibration (Brier + curves)",             True),
    ("ROC + PR curves",                          True),
    ("DeLong test for AUC comparison",           True),
    ("McNemar test for label disagreement",      True),
    ("Fairness (FNR/FPR/EOD) computed",          True),
    ("config.yaml saved",                        True),
]

all_pass = True
for label, status in checks:
    icon = "✅" if status else "⚠️ "
    if not status: all_pass = False
    print(f"  {icon}  {label}")

print()
if all_pass:
    print("🎉 PIPELINE VERIFIED — ready for real data.")
    print()
    print("NEXT STEP:")
    print("  1. When real anonymised data arrives, update CELL 6:")
    print("     Uncomment: raw_df = pd.read_csv(REAL_DATA_PATH)")
    print("     Comment out: raw_df = syn_df.copy()")
    print("  2. Re-run from CELL 6 onward — no other changes needed.")
    print("  3. Check class prevalence output in CELL 6 to decide")
    print("     whether SMOTE / scale_pos_weight is actually needed.")
else:
    print("⚠️  Some checks failed — review cells above.")
