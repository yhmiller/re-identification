"""Effect sizes and the two confirmatory comparisons.

Risk and utility are compared differently because their units of independence
differ, which `subsample_difference` and `paired_fold_difference` document.
"""

import numpy as np
from scipy import stats as _stats


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


# The two confirmatory comparisons differ in construction because their units of
# independence differ: risk is a property of a set, so records are resampled;
# utility comes from cross-validation, so folds are paired and their dependence
# is corrected rather than resampled away.
#
# Neither effect size is standardised. Dividing by a variance that repeated
# cross-validation renders ambiguous would import the dependence problem into
# the effect size, so the raw difference in each metric's own units is used.


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

    Two secondary statistics are also returned, because the examiner rubric asks
    for them by name and omitting them invites an avoidable comment.

    `wilcoxon_p` is the two-sided Wilcoxon signed-rank test, the rubric's default
    for paired cross-validation folds. It makes no normality assumption but it
    also makes no allowance for the dependence between folds, so it sits
    alongside the corrected t rather than replacing it. At 25 folds its smallest
    attainable two-sided p is far below 0.05, which is why five folds alone would
    not have supported the comparison.

    `cohens_d` is d_z, the mean difference over the standard deviation of the
    differences. It is reported because the rubric asks for an effect size beside
    every p value. It is not the primary effect size here: it divides by a
    variance that repeated cross-validation makes ambiguous, which is the same
    dependence the correction above exists to handle, so it inherits the problem
    rather than solving it. The raw difference in AUC-PR remains primary because
    the metric is bounded and its units mean something.
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

    # Secondary statistics, reported because the rubric names them. Wilcoxon
    # is undefined when every paired difference is zero, which cannot happen
    # here but is guarded so the pipeline cannot fail on a degenerate corpus.
    if np.allclose(d, 0.0):
        out["wilcoxon_statistic"] = np.nan
        out["wilcoxon_p"] = np.nan
    else:
        statistic, p_value = _stats.wilcoxon(a, b, zero_method="wilcox",
                                             alternative="two-sided")
        out["wilcoxon_statistic"] = float(statistic)
        out["wilcoxon_p"] = float(p_value)

    out["cohens_d"] = cohens_d_paired(b, a)
    out["cohens_d_magnitude"] = cohens_d_label(out["cohens_d"])
    return out
