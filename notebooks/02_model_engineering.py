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
# ============================================================


# ─────────────────────────────────────────────────────────────
# ENG-CELL 1 — Engineering imports + reproducibility
# ─────────────────────────────────────────────────────────────
import os
import json
import time
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             average_precision_score)
from scipy.stats import wilcoxon

# Reuse SEED, preprocessor, X (raw features), y (target) from the
# main pipeline. If running this file standalone, define them here.
SEED = 42
np.random.seed(SEED)

# Inherited from Stage 1 when run via run_all.py; falls back to "results"
# so the metric/table paths resolve when this runs on its own.
RESULTS_DIR = globals().get("RESULTS_DIR", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

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
# ─────────────────────────────────────────────────────────────

# One shared fold object — guarantees identical splits for both models
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def run_cv(feature_subset, label):
    """
    Run 5-fold stratified CV on a given feature subset.
    Preprocessing is fitted INSIDE each fold (no leakage).
    Returns dict of per-fold metric lists + mean training time.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer

    # Split the subset into numeric vs categorical for the transformer
    num_sub = [c for c in feature_subset if c in ALL_NUMERIC]
    cat_sub = [c for c in feature_subset if c in CATEGORICAL_COLS]

    num_tf = Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("sc",  MinMaxScaler())])
    cat_tf = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                       ("oh",  OneHotEncoder(handle_unknown="ignore",
                                             sparse_output=False))])
    pre = ColumnTransformer([("num", num_tf, num_sub),
                             ("cat", cat_tf, cat_sub)], remainder="drop")

    Xsub = X[feature_subset]
    res = {"acc": [], "f1": [], "auc_roc": [], "auc_pr": [], "time": []}

    for tr_idx, te_idx in SKF.split(Xsub, y):
        Xtr, Xte = Xsub.iloc[tr_idx], Xsub.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

        Xtr_p = pre.fit_transform(Xtr)
        Xte_p = pre.transform(Xte)

        model = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=SEED, verbosity=0,
        )
        t0 = time.time()
        model.fit(Xtr_p, ytr)
        res["time"].append(time.time() - t0)

        preds = model.predict(Xte_p)
        proba = model.predict_proba(Xte_p)[:, 1]
        res["acc"].append(accuracy_score(yte, preds))
        res["f1"].append(f1_score(yte, preds, average="macro", zero_division=0))
        res["auc_roc"].append(roc_auc_score(yte, proba))
        res["auc_pr"].append(average_precision_score(yte, proba))

    print(f"   [{label}] {len(feature_subset)} features | "
          f"AUC-ROC {np.mean(res['auc_roc']):.4f} | "
          f"AUC-PR {np.mean(res['auc_pr']):.4f}")
    return res

print("✅ Shared CV evaluator defined (identical folds for both models).")


# ─────────────────────────────────────────────────────────────
# ENG-CELL 3 — PHASE 1: Build and LOCK the baseline
# Standard XGBoost, ALL features. Run first, save, never touch again.
# ─────────────────────────────────────────────────────────────

ALL_FEATURE_LIST = ALL_FEATURES_V2   # from main pipeline (numeric + categorical)

print("PHASE 1 — Baseline XGBoost (all features), locking results...")
baseline_results = run_cv(ALL_FEATURE_LIST, "BASELINE")

# Lock immediately to disk
json.dump(
    {k: [float(v) for v in vals] for k, vals in baseline_results.items()},
    open(os.path.join(RESULTS_DIR, "baseline_metrics.json"), "w"), indent=2
)
print(f"✅ Baseline LOCKED → results/baseline_metrics.json ({len(ALL_FEATURE_LIST)} features)")


# ─────────────────────────────────────────────────────────────
# ENG-CELL 4 — Rank features by global SHAP importance
# Fit one XGBoost on the full data, compute SHAP, rank features.
# This ranking drives the pruning decision (the engineering step).
# ─────────────────────────────────────────────────────────────

# Fit a model on the full preprocessed data to get SHAP rankings
X_full_proc = preprocessor.fit_transform(X)     # preprocessor from Cell 8
rank_model = xgb.XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=SEED, verbosity=0,
)
rank_model.fit(X_full_proc, y)

explainer = shap.TreeExplainer(rank_model)
shap_vals = explainer.shap_values(X_full_proc)

# Mean |SHAP| per ENCODED feature
enc_importance = pd.Series(
    np.abs(shap_vals).mean(axis=0), index=FEATURE_NAMES
).sort_values(ascending=False)

print("✅ SHAP importance computed (encoded features).")
print("\n   Top 10 encoded features:")
print(enc_importance.head(10).to_string())
print("\n   Bottom 10 encoded features (pruning candidates):")
print(enc_importance.tail(10).to_string())


# ─────────────────────────────────────────────────────────────
# ENG-CELL 5 — Map encoded SHAP back to ORIGINAL features
# One-hot expands a categorical into many columns; we sum their
# SHAP back to the parent feature so we prune whole input columns.
# ─────────────────────────────────────────────────────────────

def map_to_original(enc_series):
    """Aggregate encoded-feature SHAP importance back to source columns."""
    orig = {}
    for enc_name, val in enc_series.items():
        matched = None
        for cat in CATEGORICAL_COLS:
            if enc_name.startswith(cat + "_"):
                matched = cat
                break
        key = matched if matched else enc_name
        orig[key] = orig.get(key, 0.0) + val
    return pd.Series(orig).sort_values(ascending=False)

orig_importance = map_to_original(enc_importance)
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
# ─────────────────────────────────────────────────────────────

# === ENGINEERING DECISION [R] ===========================================
# Standard XGBoost uses ALL collected features, including many that
# contribute almost nothing to the prediction (per SHAP).
# We REMOVE the lowest-SHAP features so the model is leaner and its
# explanations are cleaner for nurse educators — without losing accuracy.
# This single modification creates E-XGBoost.
KEEP_CUM_THRESHOLD = 0.95   # keep features explaining 95% of SHAP mass
# ========================================================================

kept_features = cum[cum <= KEEP_CUM_THRESHOLD].index.tolist()
# Always keep at least the top feature even if it alone exceeds threshold
if len(kept_features) == 0:
    kept_features = [orig_importance.index[0]]
# Ensure we keep the feature that crosses the threshold too
if len(kept_features) < len(orig_importance):
    next_feat = cum.index[len(kept_features)]
    kept_features.append(next_feat)

pruned_features = [f for f in ALL_FEATURE_LIST if f not in kept_features]

print("✅ ENGINEERING DECISION applied (Operation R — Remove):")
print(f"   Threshold: keep features explaining {KEEP_CUM_THRESHOLD:.0%} of SHAP importance")
print(f"   KEPT    ({len(kept_features)}): {kept_features}")
print(f"   PRUNED  ({len(pruned_features)}): {pruned_features}")


# ─────────────────────────────────────────────────────────────
# ENG-CELL 7 — PHASE 2: Run E-XGBoost on the pruned feature set
# Identical folds (same SKF), identical hyperparameters — only the
# feature set differs. This isolates the engineering effect.
# ─────────────────────────────────────────────────────────────

print("PHASE 2 — E-XGBoost (SHAP-pruned features)...")
engineered_results = run_cv(kept_features, "E-XGBoost")

json.dump(
    {k: [float(v) for v in vals] for k, vals in engineered_results.items()},
    open(os.path.join(RESULTS_DIR, "engineered_metrics.json"), "w"), indent=2
)
print(f"✅ E-XGBoost results saved → results/engineered_metrics.json "
      f"({len(kept_features)} features)")


# ─────────────────────────────────────────────────────────────
# ENG-CELL 8 — PHASE 3: Formal comparison + Wilcoxon signed-rank
# This table is the centrepiece of your Results section.
# ─────────────────────────────────────────────────────────────

b = json.load(open(os.path.join(RESULTS_DIR, "baseline_metrics.json")))
e = json.load(open(os.path.join(RESULTS_DIR, "engineered_metrics.json")))

metrics = ["acc", "f1", "auc_roc", "auc_pr"]
metric_names = {"acc": "Accuracy", "f1": "Macro F1",
                "auc_roc": "AUC-ROC", "auc_pr": "AUC-PR"}
rows = []
for m in metrics:
    bvals, evals = np.array(b[m]), np.array(e[m])
    # Wilcoxon needs non-identical paired samples
    if np.allclose(bvals, evals):
        p = 1.0
    else:
        try:
            _, p = wilcoxon(bvals, evals)
        except ValueError:
            p = 1.0
    delta = evals.mean() - bvals.mean()
    rows.append({
        "Metric":   metric_names[m],
        "XGBoost (Baseline)": f"{bvals.mean():.4f} ± {bvals.std():.4f}",
        "E-XGBoost (Engineered)": f"{evals.mean():.4f} ± {evals.std():.4f}",
        "Δ Change": f"{delta:+.4f}",
        "p-value":  f"{p:.4f}",
        "Sig (p<0.05)": "Yes" if p < 0.05 else "ns",
        "Better?": "✓" if delta >= 0 else "✗",
    })

# Add efficiency rows
bt, et = np.array(b["time"]), np.array(e["time"])
rows.append({
    "Metric": "Training Time (s)",
    "XGBoost (Baseline)": f"{bt.mean():.3f} ± {bt.std():.3f}",
    "E-XGBoost (Engineered)": f"{et.mean():.3f} ± {et.std():.3f}",
    "Δ Change": f"{et.mean()-bt.mean():+.3f}s",
    "p-value": "—", "Sig (p<0.05)": "—",
    "Better?": "✓" if et.mean() <= bt.mean() else "~",
})
rows.append({
    "Metric": "Feature Count",
    "XGBoost (Baseline)": f"{len(ALL_FEATURE_LIST)}",
    "E-XGBoost (Engineered)": f"{len(kept_features)}",
    "Δ Change": f"{len(kept_features)-len(ALL_FEATURE_LIST)}",
    "p-value": "—", "Sig (p<0.05)": "—",
    "Better?": "✓ (leaner)",
})

comparison = pd.DataFrame(rows)
comparison.to_csv(os.path.join(RESULTS_DIR, "comparison_table.csv"), index=False)

print("=" * 78)
print("TABLE 1 — XGBoost (Baseline) vs E-XGBoost (Engineered)")
print("5-fold stratified CV | identical splits | Wilcoxon signed-rank test")
print("=" * 78)
print(comparison.to_string(index=False))
print("\n✅ Comparison table saved → results/comparison_table.csv")


# ─────────────────────────────────────────────────────────────
# ENG-CELL 9 — Auto-generate the Results paragraph for your paper
# ─────────────────────────────────────────────────────────────

auc_pr_delta = np.array(e["auc_pr"]).mean() - np.array(b["auc_pr"]).mean()
auc_roc_delta = np.array(e["auc_roc"]).mean() - np.array(b["auc_roc"]).mean()
n_removed = len(ALL_FEATURE_LIST) - len(kept_features)

paragraph = f"""
RESULTS PARAGRAPH (paste into your paper, replace with final numbers):

"Table 1 presents the comparative performance of the baseline XGBoost and the
proposed E-XGBoost across five-fold stratified cross-validation using identical
data splits. E-XGBoost was engineered by removing {n_removed} features that SHAP
global analysis identified as low-contribution, retaining the {len(kept_features)}
features that together account for {KEEP_CUM_THRESHOLD:.0%} of total SHAP importance.
Despite using fewer inputs, E-XGBoost achieved an AUC-PR change of {auc_pr_delta:+.4f}
and an AUC-ROC change of {auc_roc_delta:+.4f} relative to the baseline. This
demonstrates that SHAP-guided feature pruning produces a more parsimonious model
whose explanations are simpler for nurse educators to interpret, without sacrificing
predictive performance — positioning SHAP not only as a post-hoc explanation method
but as an active model-engineering instrument."

NOTE: If Δ is positive and p < 0.05 → report as a significant improvement.
      If Δ ≈ 0 (not significant) → report as 'equivalent performance with fewer
      features and improved interpretability', which is ALSO a valid contribution
      (the doc explicitly accepts efficiency/parsimony gains at equal accuracy).
"""
print(paragraph)


# ─────────────────────────────────────────────────────────────
# ENG-CELL 10 — Engineering summary (defence-day checklist)
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("MODEL ENGINEERING — CONTRIBUTION 3 SUMMARY")
print("=" * 70)
print(f"""
  Operation applied : R (Remove) — SHAP-guided feature pruning
  Baseline model    : XGBoost, all {len(ALL_FEATURE_LIST)} features
  Engineered model  : E-XGBoost, {len(kept_features)} features
  Features pruned   : {n_removed}
  Comparison        : 5-fold CV, identical splits, Wilcoxon test
  Artefacts         : baseline_metrics.json, engineered_metrics.json,
                      comparison_table.csv

  DEFENCE-DAY THREE CONTRIBUTIONS:
  1. Original field data — first Ghanaian nursing-licensure dataset
  2. Modern ML model — XGBoost + SHAP
  3. Engineered model — E-XGBoost (SHAP-guided pruning) vs baseline

  ⚠️  Run this on REAL data once collected. On synthetic data the
     pruning result is illustrative only — the real SHAP rankings
     will differ and determine which features are actually pruned.
""")
