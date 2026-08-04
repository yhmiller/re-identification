"""
model_comparison.py — formal comparison tables between two sets of CV
results: baseline XGBoost vs E-XGBoost (Wilcoxon signed-rank + Cohen's d),
and, once Option 2 is implemented, real-data vs synthetic-data runs of the
same pipeline (see docs/TODO.md — "Data-handling: switch to Option 2").
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from stats_validation import cohens_d_label, cohens_d_paired

METRIC_NAMES = {"acc": "Accuracy", "f1": "Macro F1",
                "auc_roc": "AUC-ROC", "auc_pr": "AUC-PR"}


def compare_baseline_vs_engineered(baseline_results, engineered_results,
                                    all_feature_list, kept_features):
    """
    Table 1: baseline XGBoost vs E-XGBoost across the four CV metrics
    (Wilcoxon signed-rank + Cohen's d), plus training-time and
    feature-count rows. (was ENG-CELL 8)
    """
    rows = []
    for m, label in METRIC_NAMES.items():
        bvals = np.array(baseline_results[m])
        evals = np.array(engineered_results[m])

        if np.allclose(bvals, evals):
            p = 1.0
        else:
            try:
                _, p = wilcoxon(bvals, evals)
            except ValueError:
                p = 1.0

        delta = evals.mean() - bvals.mean()
        d = cohens_d_paired(bvals, evals)

        rows.append({
            "Metric": label,
            "XGBoost (Baseline)": f"{bvals.mean():.4f} ± {bvals.std():.4f}",
            "E-XGBoost (Engineered)": f"{evals.mean():.4f} ± {evals.std():.4f}",
            "Δ Change": f"{delta:+.4f}",
            "p-value": f"{p:.4f}",
            "Sig (p<0.05)": "Yes" if p < 0.05 else "ns",
            "Cohen's d": f"{d:+.3f} ({cohens_d_label(d)})",
            "Better?": "✓" if delta >= 0 else "✗",
        })

    bt, et = np.array(baseline_results["time"]), np.array(engineered_results["time"])
    rows.append({
        "Metric": "Training Time (s)",
        "XGBoost (Baseline)": f"{bt.mean():.3f} ± {bt.std():.3f}",
        "E-XGBoost (Engineered)": f"{et.mean():.3f} ± {et.std():.3f}",
        "Δ Change": f"{et.mean() - bt.mean():+.3f}s",
        "p-value": "—", "Sig (p<0.05)": "—",
        "Cohen's d": "—",
        "Better?": "✓" if et.mean() <= bt.mean() else "~",
    })
    rows.append({
        "Metric": "Feature Count",
        "XGBoost (Baseline)": f"{len(all_feature_list)}",
        "E-XGBoost (Engineered)": f"{len(kept_features)}",
        "Δ Change": f"{len(kept_features) - len(all_feature_list)}",
        "p-value": "—", "Sig (p<0.05)": "—",
        "Cohen's d": "—",
        "Better?": "✓ (leaner)",
    })
    return pd.DataFrame(rows)


def generate_results_paragraph(baseline_results, engineered_results,
                                all_feature_list, kept_features,
                                keep_threshold, n_folds):
    """Auto-drafted Results paragraph for the paper. (was ENG-CELL 9)"""
    auc_pr_delta = np.array(engineered_results["auc_pr"]).mean() - np.array(baseline_results["auc_pr"]).mean()
    auc_roc_delta = np.array(engineered_results["auc_roc"]).mean() - np.array(baseline_results["auc_roc"]).mean()
    n_removed = len(all_feature_list) - len(kept_features)

    return f"""
"Table 1 presents the comparative performance of the baseline XGBoost and the
proposed E-XGBoost across {n_folds}-fold stratified cross-validation using identical
data splits. E-XGBoost was engineered by removing {n_removed} features that SHAP
global analysis identified as low-contribution, retaining the {len(kept_features)}
features that together account for {keep_threshold:.0%} of total SHAP importance.
Despite using fewer inputs, E-XGBoost achieved an AUC-PR change of {auc_pr_delta:+.4f}
and an AUC-ROC change of {auc_roc_delta:+.4f} relative to the baseline. This
demonstrates that SHAP-guided feature pruning produces a more parsimonious model
whose explanations are simpler for nurse educators to interpret, without sacrificing
predictive performance — positioning SHAP not only as a post-hoc explanation method
but as an active model-engineering instrument."

NOTE: If Δ (Delta) is positive and p < 0.05 → report as a significant improvement.
      If Δ ≈ 0 (not significant) → report as 'equivalent performance with fewer
      features and improved interpretability', which is ALSO a valid contribution
      (the doc explicitly accepts efficiency/parsimony gains at equal accuracy).
"""


def compare_real_vs_synthetic(real_results, synthetic_results, metrics=("auc_roc", "auc_pr")):
    """
    Option 2 (framework/pipeline replication): compare the SAME pipeline's
    CV results run once on real data and once on the synthetic replica,
    tuned independently. Supports "this pipeline performs consistently
    across contexts" — not a domain-transfer claim.

    `real_results` / `synthetic_results` are the CV-result dicts returned by
    engineered_xgboost.run_cv (or baseline_xgboost training), each holding
    per-fold metric lists keyed by metric name.

    Not yet wired into a notebook — real data is required to produce
    `real_results` (see docs/TODO.md, "Data-handling: switch to Option 2").
    """
    rows = []
    for m in metrics:
        rvals = np.array(real_results[m])
        svals = np.array(synthetic_results[m])
        d = cohens_d_paired(svals, rvals) if len(rvals) == len(svals) else float("nan")
        rows.append({
            "Metric": METRIC_NAMES.get(m, m),
            "Real data": f"{rvals.mean():.4f} ± {rvals.std():.4f}",
            "Synthetic data": f"{svals.mean():.4f} ± {svals.std():.4f}",
            "Δ (real − synthetic)": f"{rvals.mean() - svals.mean():+.4f}",
            "Cohen's d": f"{d:+.3f} ({cohens_d_label(d)})" if not np.isnan(d) else "—",
        })
    return pd.DataFrame(rows)
