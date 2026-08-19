"""
stats_validation.py — every statistical test and metric used to validate
and compare models in this project, in one place.

Used by notebooks/01_pipeline_and_experiments.py (evaluation, DeLong,
McNemar, fairness) and notebooks/02_model_engineering.py (Cohen's d for the
baseline vs E-XGBoost comparison). Also the home for the real-vs-synthetic
comparison introduced by the Option 2 data-handling plan (see docs/TODO.md).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
)
from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats


def evaluate_model(name, y_true, y_pred_proba, y_pred_labels):
    """Return a dict of all evaluation metrics for a model."""
    return {
        "model": name,
        "auc_roc": roc_auc_score(y_true, y_pred_proba),
        "auc_pr": average_precision_score(y_true, y_pred_proba),
        "f1_weighted": f1_score(
            y_true, y_pred_labels, average="weighted", zero_division=0
        ),
        "precision": precision_score(y_true, y_pred_labels, zero_division=0),
        "recall": recall_score(y_true, y_pred_labels, zero_division=0),
        "brier": brier_score_loss(y_true, y_pred_proba),
    }


def print_metrics(results: list):
    print(
        f"\n  {'Model':<22} {'AUC-ROC':>8} {'AUC-PR':>8} "
        f"{'F1-W':>8} {'Prec':>8} {'Recall':>8} {'Brier':>8}"
    )
    print("  " + "─" * 70)
    for r in results:
        print(
            f"  {r['model']:<22} {r['auc_roc']:>8.4f} {r['auc_pr']:>8.4f} "
            f"{r['f1_weighted']:>8.4f} {r['precision']:>8.4f} "
            f"{r['recall']:>8.4f} {r['brier']:>8.4f}"
        )


def class_metrics(y_true, y_pred, class_labels=("Pass", "Fail")):
    """
    Per-class precision/recall/F1, the confusion matrix, and sensitivity /
    specificity. Sensitivity (recall on Fail) is the headline number here:
    a false negative means a genuinely at-risk student is missed.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    sensitivity = recall[1]  # recall on Fail (class 1)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    per_class = {
        class_labels[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(2)
    }
    return {
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "per_class": per_class,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def print_class_metrics(results: dict, model_name=""):
    cm = results["confusion_matrix"]
    print(f"\n  Class-wise metrics{f' — {model_name}' if model_name else ''}:")
    print(
        f"  Confusion matrix — TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}"
    )
    for label, m in results["per_class"].items():
        print(
            f"  {label:<6} precision={m['precision']:.4f}  recall={m['recall']:.4f}  "
            f"f1={m['f1']:.4f}  n={m['support']}"
        )
    print(
        f"  Sensitivity (recall on Fail): {results['sensitivity']:.4f}  "
        f"| Specificity: {results['specificity']:.4f}"
    )


def delong_test(y_true, proba_a, proba_b):
    """
    Non-parametric DeLong test for comparing two AUCs.
    Returns (auc_a, auc_b, z, p) — z-statistic and two-tailed p-value.
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

        aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
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


def mcnemar_test(y_true, labels_a, labels_b, model_a_name, model_b_name):
    """McNemar test on the disagreement table between two classifiers."""
    b = int(
        ((labels_a == 1) & (labels_b == 0) & (y_true == 1)).sum()
        + ((labels_a == 1) & (labels_b == 0) & (y_true == 0)).sum()
    )
    c = int(
        ((labels_a == 0) & (labels_b == 1) & (y_true == 1)).sum()
        + ((labels_a == 0) & (labels_b == 1) & (y_true == 0)).sum()
    )
    table = [[0, b], [c, 0]]
    result = mcnemar(table, exact=True)
    return result.pvalue, b, c


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
        results.append(
            {
                "group": grp,
                "n": int(mask.sum()),
                "fail_rate": float(yt.mean()),
                "FNR": round(fnr, 4),
                "FPR": round(fpr, 4),
            }
        )
    df_res = pd.DataFrame(results)
    if len(df_res) > 1:
        df_res["EqualOpportunityDiff"] = (df_res["FNR"] - df_res["FNR"].min()).round(4)
    return df_res


def cohens_d_label(d):
    """Conventional magnitude bands for Cohen's d (Cohen, 1988)."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def cohens_d_paired(a_vals, b_vals):
    """
    Cohen's d for paired samples (d_z = mean difference / SD of the
    differences). Returns 0.0 when the differences have zero variance
    (identical paired samples).
    """
    diff = np.asarray(b_vals) - np.asarray(a_vals)
    sd_diff = diff.std(ddof=1)
    return float(diff.mean() / sd_diff) if sd_diff > 0 else 0.0


# ---------------------------------------------------------------------------
# Confirmatory analysis for the disclosure-risk study
# ---------------------------------------------------------------------------
#
# Two comparisons, deliberately different in construction because the units of
# independence differ.
#
# Risk is a property of a set of records, so records are resampled.
# Utility is estimated by cross-validation, so the folds are paired and the
# dependence between them is handled rather than resampled away.
#
# Neither effect size is standardised. Both metrics are bounded and directly
# interpretable, and dividing by a variance that repeated cross-validation makes
# ambiguous would import the dependence problem into the effect size without
# adding meaning. The raw difference in the metric's own units is the effect
# size, which is the preferred form when the units mean something.

import numpy as np  # noqa: E402
from scipy import stats as _stats  # noqa: E402

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260819


def subsample_difference(measure_a, measure_b, n_records, fraction=0.8,
                         n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Difference between two release measures, with a percentile interval.

    Subsampling without replacement, not the ordinary bootstrap.

    The ordinary bootstrap is wrong for uniqueness. Resampling with replacement
    duplicates records, and a duplicated record is no longer unique, so measured
    uniqueness falls. The fall is not symmetric between arms: an arm at 87%
    uniqueness has far more to lose than one at 13%, so the difference between
    them shrinks toward zero and the interval excludes the observed value. That
    was checked empirically before this function was written, and the observed
    difference of -0.7456 fell outside a bootstrap interval of [-0.31, -0.24].

    Subsampling without replacement creates no duplicates. Both arms are
    evaluated on the same subsample, so the comparison stays paired at record
    level.

    The cost is that uniqueness depends on set size, so the interval describes
    variability of the difference at `fraction` of the corpus rather than at
    full size. That is stated wherever the interval is reported. The observed
    difference is computed at the same fraction, averaged over the subsamples,
    so the point estimate and the interval describe the same quantity.
    """
    rng = np.random.default_rng(seed)
    m = max(2, int(round(fraction * n_records)))

    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.choice(n_records, size=m, replace=False)
        diffs[i] = measure_a(idx) - measure_b(idx)

    low, high = np.percentile(diffs, [2.5, 97.5])
    full = measure_a(np.arange(n_records)) - measure_b(np.arange(n_records))
    return {
        "difference_full_corpus": float(full),
        "difference_at_fraction": float(diffs.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "sd": float(diffs.std(ddof=1)),
        "subsample_fraction": fraction,
        "n_subsampled": m,
        "n_resamples": n_resamples,
        "excludes_zero": bool(low > 0 or high < 0),
    }


def paired_fold_difference(scores_a, scores_b, n_splits=5):
    """Paired difference across cross-validation folds, reported two ways.

    Both arms are scored on identical splits, so the folds pair naturally.

    The naive interval treats the fold scores as independent. They are not:
    training sets overlap heavily under repeated k-fold, so the naive variance
    understates uncertainty and the test is anti-conservative.

    The corrected interval inflates the variance by (1/J + n_test/n_train),
    following the standard treatment of resampling-induced dependence. For five
    folds repeated five times this multiplies the standard error by roughly 2.7.

    Both are returned. If the conclusion survives the correction, reporting the
    pair is stronger evidence than reporting either alone. The correction's fit
    to this design should be confirmed with a statistical adviser rather than
    assumed.
    """
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired comparison needs the same number of folds in each arm")

    d = a - b
    J = len(d)
    mean_d = float(d.mean())
    var_d = float(d.var(ddof=1))

    test_fraction = 1.0 / n_splits
    correction = 1.0 / J + test_fraction / (1.0 - test_fraction)

    out = {
        "n_folds": J,
        "mean_difference": mean_d,
        "sd_of_differences": float(np.sqrt(var_d)),
    }

    for label, factor in (("naive", 1.0 / J), ("corrected", correction)):
        se = float(np.sqrt(factor * var_d))
        t = mean_d / se if se > 0 else np.nan
        p = float(2 * _stats.t.sf(abs(t), df=J - 1)) if se > 0 else np.nan
        half = _stats.t.ppf(0.975, df=J - 1) * se
        out[f"{label}_se"] = se
        out[f"{label}_t"] = float(t)
        out[f"{label}_p"] = p
        out[f"{label}_ci_low"] = mean_d - half
        out[f"{label}_ci_high"] = mean_d + half

    out["se_inflation"] = (
        out["corrected_se"] / out["naive_se"] if out["naive_se"] else np.nan
    )
    return out
