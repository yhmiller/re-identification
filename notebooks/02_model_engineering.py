# ============================================================
# 02_model_engineering.py  —  E-XGBoost (SHAP-Guided Pruning)
#
# THE 3rd CONTRIBUTION: Model Engineering
# Operation R (Remove): prune low-SHAP features → leaner,
# equally/more accurate, more interpretable model.
#
# Engineered model name: E-XGBoost (Enhanced XGBoost)
# Author : Prince Bortey Miller | ID: 22388461 | KNUST
# Supervisor: Dr. Eric Opoku Osei
#
# HOW TO USE:
#   Run AFTER cells 1–8 of 01_pipeline_and_experiments.py
#   (those define the schema, load data, build the preprocessor).
#   Then paste each ENG-CELL below into a new Colab/Jupyter cell.
#
#   The baseline is run FIRST and LOCKED, then E-XGBoost,
#   then a formal comparison with Wilcoxon signed-rank test.
#
# This notebook is the NARRATIVE/ORCHESTRATION layer — the SHAP-pruning
# workflow itself lives in engineered_xgboost.py, and the comparison table
# in model_comparison.py (see docs/TODO.md, "Codebase restructure").
# ============================================================


# ─────────────────────────────────────────────────────────────
# ENG-CELL 1 — Engineering imports + reproducibility
# ─────────────────────────────────────────────────────────────
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

# Reuse SEED, preprocessor, X (raw features), y (target) from the
# main pipeline. If running this file standalone, define them here.
SEED = 42
np.random.seed(SEED)

# Find the repo root the same way 01_pipeline_and_experiments.py CELL 3 does,
# so `engineered_xgboost` etc. import cleanly even when this file is run on
# its own rather than chained after Stage 1 via run_all.py.
import sys
try:
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
except NameError:
    pass  # __file__ is undefined in a pasted Colab cell — rely on the cwd

import baseline_xgboost
import engineered_xgboost
import transfer_fusion
import model_comparison
import prediction
from baseline_xgboost import N_CV_FOLDS, N_CV_REPEATS  # single source of truth for the CV design

# Inherited from Stage 1 when run via run_all.py; falls back to
# results/synthetic so the metric/table paths resolve when this runs on its
# own (matches the results/<data_source>/ convention — see docs/TODO.md).
RESULTS_DIR = globals().get("RESULTS_DIR", os.path.join("results", "synthetic"))
os.makedirs(RESULTS_DIR, exist_ok=True)

# Use the SAME grid-search-tuned hyperparameters as Stage 1 so the engineering
# comparison is baseline-tuned vs pruned-tuned (only the feature set changes).
# Falls back to the documented defaults when run standalone without Stage 1.
XGB_PARAMS = globals().get("best_xgb_params", {
    "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8,
})

# We use the FULL feature matrix and target from the main pipeline.
# X = raw_df[ALL_FEATURES_V2]   (already defined in 01_..., Cell 7)
# y = raw_df[TARGET]
# preprocessor                  (already fitted in 01_..., Cell 8)

print("✅ Engineering module ready.")
print("   Operation: R (Remove) — SHAP-guided feature pruning")
print("   Baseline  : standard XGBoost (all features)")
print("   Engineered: E-XGBoost (low-SHAP features pruned)")


# ─────────────────────────────────────────────────────────────
# ENG-CELL 2 — Shared CV evaluator
# Both models are scored with the IDENTICAL StratifiedKFold object
# and identical preprocessing — the doc's "critical rule".
# (engineered_xgboost.run_cv)
# ─────────────────────────────────────────────────────────────

# One shared fold object — guarantees identical splits for both models.
# Repeated so the Wilcoxon test has enough paired observations to reach
# significance at all; see the N_CV_REPEATS note in baseline_xgboost.py.
SKF = RepeatedStratifiedKFold(n_splits=N_CV_FOLDS, n_repeats=N_CV_REPEATS,
                               random_state=SEED)
N_PAIRED_FOLDS = N_CV_FOLDS * N_CV_REPEATS

print(f"✅ Shared CV evaluator ready: {N_CV_FOLDS} folds x {N_CV_REPEATS} repeats "
      f"= {N_PAIRED_FOLDS} paired observations (identical splits for both models).")


# ─────────────────────────────────────────────────────────────
# ENG-CELL 3 — PHASE 1: Build and LOCK the baseline
# Standard XGBoost, ALL features. Run first, save, never touch again.
# ─────────────────────────────────────────────────────────────

ALL_FEATURE_LIST = MODEL_FEATURES   # from Stage 1, zero-variance predictors already removed

print("PHASE 1 — Baseline XGBoost (all features), locking results...")
baseline_cv_results = engineered_xgboost.run_cv(
    ALL_FEATURE_LIST, X, y, ALL_NUMERIC, CATEGORICAL_COLS, XGB_PARAMS, SKF, SEED, "BASELINE")

# Lock immediately to disk
json.dump(
    {k: [float(v) for v in vals] for k, vals in baseline_cv_results.items()},
    open(os.path.join(RESULTS_DIR, "baseline_metrics.json"), "w"), indent=2
)
print(f"✅ Baseline LOCKED → {RESULTS_DIR}/baseline_metrics.json ({len(ALL_FEATURE_LIST)} features)")


# ─────────────────────────────────────────────────────────────
# ENG-CELL 4 — Rank features by global SHAP importance
# Fit one XGBoost on the full data, compute SHAP, rank features.
# This ranking drives the pruning decision (the engineering step).
# (engineered_xgboost.rank_shap_importance)
# ─────────────────────────────────────────────────────────────

enc_importance = engineered_xgboost.rank_shap_importance(
    X, y, preprocessor, XGB_PARAMS, FEATURE_NAMES, SEED)

print("✅ SHAP importance computed (encoded features).")
print("\n   Top 10 encoded features:")
print(enc_importance.head(10).to_string())
print("\n   Bottom 10 encoded features (pruning candidates):")
print(enc_importance.tail(10).to_string())


# ─────────────────────────────────────────────────────────────
# ENG-CELL 5 — Map encoded SHAP back to ORIGINAL features
# One-hot expands a categorical into many columns; we sum their
# SHAP back to the parent feature so we prune whole input columns.
# (engineered_xgboost.map_to_original)
# ─────────────────────────────────────────────────────────────

orig_importance = engineered_xgboost.map_to_original(enc_importance, CATEGORICAL_COLS)
print("✅ SHAP importance mapped to original input features:")
print(orig_importance.to_string())

# Normalise to cumulative contribution
cum = orig_importance.cumsum() / orig_importance.sum()
print("\n   Cumulative contribution:")
for feat, c in cum.items():
    print(f"   {feat:<28} {c:.3f}")


# ─────────────────────────────────────────────────────────────
# ENG-CELL 6 — THE ENGINEERING DECISION [Operation R: Remove]
# Prune features contributing the least SHAP importance.
# Strategy: keep features that together explain ~95% of total
# SHAP importance; drop the long tail of near-zero contributors.
# (engineered_xgboost.select_pruned_features)
# ─────────────────────────────────────────────────────────────

# === ENGINEERING DECISION [R] ===========================================
# Standard XGBoost uses ALL collected features, including many that
# contribute almost nothing to the prediction (per SHAP).
# We REMOVE the lowest-SHAP features so the model is leaner and its
# explanations are cleaner for nurse educators — without losing accuracy.
# This single modification creates E-XGBoost.
KEEP_CUM_THRESHOLD = engineered_xgboost.KEEP_CUM_THRESHOLD   # keep features explaining 95% of SHAP mass
# ========================================================================

kept_features, pruned_features, cum = engineered_xgboost.select_pruned_features(
    orig_importance, ALL_FEATURE_LIST, KEEP_CUM_THRESHOLD)

# `pruned_features` = the features E-XGBoost REMOVES for the current data.
# This is the definitive "what we removed" list (data-dependent, not fixed).
# Full feature dictionary of all 39 candidates: see 01_pipeline_and_experiments.py.
print("✅ ENGINEERING DECISION applied (Operation R — Remove):")
print(f"   Threshold: keep features explaining {KEEP_CUM_THRESHOLD:.0%} of SHAP importance")
print(f"   KEPT    ({len(kept_features)}): {kept_features}")
print(f"   PRUNED  ({len(pruned_features)}): {pruned_features}")   # ← features removed


# ─────────────────────────────────────────────────────────────
# ENG-CELL 7 — PHASE 2: Run E-XGBoost on the pruned feature set
# Identical folds (same SKF), identical hyperparameters — only the
# feature set differs. This isolates the engineering effect.
# ─────────────────────────────────────────────────────────────

print("PHASE 2 — E-XGBoost (SHAP-pruned features)...")

# Selection is nested inside each fold, so the ranking model never sees the
# held-out labels. Vabalas et al. (2019): at small n, nesting the feature
# selection is what controls the optimism bias, more than nesting the
# hyperparameter search. This is the arm that produces the reported comparison.
engineered_cv_results, fold_selections = engineered_xgboost.run_nested_cv(
    X, y, ALL_NUMERIC, CATEGORICAL_COLS, ALL_FEATURE_LIST, XGB_PARAMS, SKF,
    SEED, KEEP_CUM_THRESHOLD, "E-XGBoost (nested selection)", ranking="shap")

# Gain-ranked arm on identical folds. Wang et al. (2024) report importance-based
# selection outperforming SHAP-based selection; that is only answerable with
# both measured the same way.
gain_cv_results, gain_selections = engineered_xgboost.run_nested_cv(
    X, y, ALL_NUMERIC, CATEGORICAL_COLS, ALL_FEATURE_LIST, XGB_PARAMS, SKF,
    SEED, KEEP_CUM_THRESHOLD, "Gain-pruned (comparator)", ranking="gain")

# The unnested run is retained for transparency: the gap between it and the
# nested run is the size of the selection bias at this sample size.
unnested_cv_results = engineered_xgboost.run_cv(
    kept_features, X, y, ALL_NUMERIC, CATEGORICAL_COLS, XGB_PARAMS, SKF, SEED,
    "E-XGBoost (unnested, biased)")

# ── Transfer fusion arms ────────────────────────────────────────────────
# The data-handling strategy is intermediate/transfer fusion: a generated source
# corpus is regenerated inside every fold from that fold's training partition,
# the early boosting rounds are fitted on it, and the remaining rounds are
# fitted on the field data. Both arms are run so the pruning comparison is not
# confounded by the fusion, and a field-only arm is retained so the fusion's own
# contribution is measured rather than assumed.
print("\nTRANSFER FUSION — pre-train on generated source, fine-tune on field...")
baseline_transfer_results = transfer_fusion.run_transfer_cv(
    ALL_FEATURE_LIST, X, y, ALL_NUMERIC, CATEGORICAL_COLS, XGB_PARAMS, SKF,
    SEED, TARGET, "BASELINE + transfer", engineered_xgboost._build_preprocessor)

engineered_transfer_results = transfer_fusion.run_transfer_cv(
    kept_features, X, y, ALL_NUMERIC, CATEGORICAL_COLS, XGB_PARAMS, SKF,
    SEED, TARGET, "E-XGBoost + transfer", engineered_xgboost._build_preprocessor)

json.dump({k: [float(v) for v in vals] for k, vals in baseline_transfer_results.items()},
          open(os.path.join(RESULTS_DIR, "baseline_transfer_metrics.json"), "w"), indent=2)
json.dump({k: [float(v) for v in vals] for k, vals in engineered_transfer_results.items()},
          open(os.path.join(RESULTS_DIR, "engineered_transfer_metrics.json"), "w"), indent=2)

fusion_contribution = model_comparison.summarise_with_ci(
    baseline_transfer_results, metrics=("auc_roc", "auc_pr"))
fusion_contribution.insert(0, "Arm", "Baseline + transfer fusion")
field_only = model_comparison.summarise_with_ci(
    baseline_cv_results, metrics=("auc_roc", "auc_pr"))
field_only.insert(0, "Arm", "Baseline, field data only")
fusion_table = pd.concat([field_only, fusion_contribution], ignore_index=True)
fusion_table.to_csv(os.path.join(RESULTS_DIR, "fusion_contribution.csv"), index=False)
print("✅ Fusion contribution table → fusion_contribution.csv")
print(fusion_table.to_string(index=False))

stability = engineered_xgboost.selection_stability(fold_selections, ALL_FEATURE_LIST)
stability.to_csv(os.path.join(RESULTS_DIR, "feature_stability.csv"), index=False)
always = int((stability.retention_rate == 1.0).sum())
never = int((stability.retention_rate == 0.0).sum())
print(f"✅ Feature-set stability across {len(fold_selections)} folds → feature_stability.csv")
print(f"   retained in every fold: {always} | never retained: {never} | "
      f"unstable: {len(stability) - always - never}")

json.dump(
    {k: [float(v) for v in vals] for k, vals in engineered_cv_results.items()},
    open(os.path.join(RESULTS_DIR, "engineered_metrics.json"), "w"), indent=2
)
json.dump(
    {k: [float(v) for v in vals] for k, vals in gain_cv_results.items()},
    open(os.path.join(RESULTS_DIR, "gain_pruned_metrics.json"), "w"), indent=2
)
json.dump(
    {k: [float(v) for v in vals] for k, vals in unnested_cv_results.items()},
    open(os.path.join(RESULTS_DIR, "unnested_metrics.json"), "w"), indent=2
)
print(f"✅ E-XGBoost results saved → {RESULTS_DIR}/engineered_metrics.json "
      f"(mean {np.mean(engineered_cv_results['n_features']):.1f} features across folds)")


# ─────────────────────────────────────────────────────────────
# ENG-CELL 8 — PHASE 3: Formal comparison + Wilcoxon signed-rank
# This table is the centrepiece of your Results section.
# (model_comparison.compare_baseline_vs_engineered)
# ─────────────────────────────────────────────────────────────

b = json.load(open(os.path.join(RESULTS_DIR, "baseline_metrics.json")))
e = json.load(open(os.path.join(RESULTS_DIR, "engineered_metrics.json")))

comparison = model_comparison.compare_baseline_vs_engineered(
    b, e, ALL_FEATURE_LIST, kept_features)
comparison.to_csv(os.path.join(RESULTS_DIR, "comparison_table.csv"), index=False)

print("=" * 78)
print("TABLE 1 - XGBoost (Baseline) vs E-XGBoost (Engineered)")
print(f"{N_CV_FOLDS}-fold stratified CV x {N_CV_REPEATS} repeats = {N_PAIRED_FOLDS} paired folds"
      f" | identical splits | Wilcoxon signed-rank test")
print("=" * 78)
print(comparison.to_string(index=False))
print(f"\n✅ Comparison table saved → {RESULTS_DIR}/comparison_table.csv")


# ─────────────────────────────────────────────────────────────
# ENG-CELL 9 — Auto-generate the Results paragraph for your paper
# (model_comparison.generate_results_paragraph)
# ─────────────────────────────────────────────────────────────

paragraph = model_comparison.generate_results_paragraph(
    b, e, ALL_FEATURE_LIST, kept_features, KEEP_CUM_THRESHOLD, N_PAIRED_FOLDS)
n_removed = len(ALL_FEATURE_LIST) - len(kept_features)

# Save to academic_summary.txt so it remains easily copy-pasteable without terminal clutter
summary_text = f"""RESULTS PARAGRAPH (paste into your paper, replace with final numbers):

{paragraph}

======================================================================
MODEL ENGINEERING — CONTRIBUTION 3 SUMMARY
======================================================================

  Operation applied : R (Remove) — SHAP-guided feature pruning
  Baseline model    : XGBoost, all {len(ALL_FEATURE_LIST)} features
  Engineered model  : E-XGBoost, {len(kept_features)} features
  Features pruned   : {n_removed}
  Comparison        : {N_CV_FOLDS}-fold CV x {N_CV_REPEATS} repeats ({N_PAIRED_FOLDS} paired folds), Wilcoxon test
  Artefacts         : baseline_metrics.json, engineered_metrics.json,
                      comparison_table.csv

  DEFENCE-DAY THREE CONTRIBUTIONS:
  1. Original field data — first Ghanaian nursing-licensure dataset
  2. Modern ML model — XGBoost + SHAP
  3. Engineered model — E-XGBoost (SHAP-guided pruning) vs baseline

  ⚠️  I will RUN this on REAL data once cleaned. On synthetic data the
     pruning result is illustrative only - the real SHAP rankings
     will differ and determine which features are actually pruned.
"""

summary_path = os.path.join(RESULTS_DIR, "academic_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary_text)

CLEAN_RUN = os.environ.get("CLEAN_RUN") == "1"
if CLEAN_RUN:
    print(f"✅ Academic results paragraph and defense summary saved → {RESULTS_DIR}/academic_summary.txt")
else:
    print(paragraph)
    print("=" * 70)
    print("MODEL ENGINEERING — CONTRIBUTION 3 SUMMARY")
    print("=" * 70)
    print(f"""
  Operation applied : R (Remove) — SHAP-guided feature pruning
  Baseline model    : XGBoost, all {len(ALL_FEATURE_LIST)} features
  Engineered model  : E-XGBoost, {len(kept_features)} features
  Features pruned   : {n_removed}
  Comparison        : {N_CV_FOLDS}-fold CV x {N_CV_REPEATS} repeats ({N_PAIRED_FOLDS} paired folds), Wilcoxon test
  Artefacts         : baseline_metrics.json, engineered_metrics.json,
                      comparison_table.csv

  DEFENCE-DAY THREE CONTRIBUTIONS:
  1. Original field data — first Ghanaian nursing-licensure dataset
  2. Modern ML model — XGBoost + SHAP
  3. Engineered model — E-XGBoost (SHAP-guided pruning) vs baseline

  ⚠️  I will RUN this on REAL data once cleaned. On synthetic data the
     pruning result is illustrative only - the real SHAP rankings
     will differ and determine which features are actually pruned.
""")


# ─────────────────────────────────────────────────────────────
# ENG-CELL 11 — Export the E-XGBoost inference bundle for the educator app
# So the app serves the REPORTED contribution (E-XGBoost), not the full-feature
# Stage 1 model. The deployed model mirrors Stage 1's production recipe (tuned
# params + scale_pos_weight + early stopping) but on the retained feature set.
# The raw upload schema is unchanged: engineer_features still needs all raw
# inputs to compute the retained engineered features; the model simply uses
# fewer of the resulting columns. This OVERWRITES the Stage 1 bundle on purpose.
# (engineered_xgboost.train_engineered_model)
# ─────────────────────────────────────────────────────────────

ex_spw = globals().get("spw", 1)
ex_model, ex_pre, ex_feature_names = engineered_xgboost.train_engineered_model(
    X_train, y_train, X_val, y_val, kept_features,
    ALL_NUMERIC, CATEGORICAL_COLS, XGB_PARAMS, ex_spw, SEED)

EX_BUNDLE = {
    "model": ex_model,
    "preprocessor": ex_pre,
    "engineer_features": engineer_features,
    "feature_columns": kept_features,
    "encoded_feature_names": ex_feature_names,
    "raw_numeric_columns": NUMERIC_COLS,
    "categorical_columns": CATEGORICAL_COLS,
    "target": TARGET,
    "data_source": globals().get("DATA_SOURCE", "synthetic"),
    "model_name": "E-XGBoost",
}
prediction.save_bundle(EX_BUNDLE, os.path.join(RESULTS_DIR, "inference_bundle.pkl"))
print(f"✅ E-XGBoost inference bundle saved → {RESULTS_DIR}/inference_bundle.pkl "
      f"({len(kept_features)} retained features; the app now serves E-XGBoost)")
