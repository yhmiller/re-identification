"""The risk-utility frontier.

Phase 5 of the re-identification study. Every de-identification control costs
something. This module measures both sides of that trade on the same
configurations, so a custodian can read off what a given level of protection
costs the research the data is shared for.

Risk is disclosure risk from `disclosure_risk`: uniqueness, smallest
equivalence class, marketer risk.

Utility is the question a data custodian actually asks, which is not whether the
distributions still look right but whether the analysis still works. So utility
is the predictive performance of an XGBoost model trained on the protected data,
scored against its no-skill floor, under the fold design PROJECT_GUIDE settled:
five folds repeated across five seeds, giving 25 paired observations. Five folds
alone cannot support a claim, because the smallest two-sided p Wilcoxon can
return at n=5 is 0.0625.

The predictors are the semester sequence, which is exactly what the
de-identification protects. That is deliberate. Protecting a column the model
does not use would cost nothing and prove nothing.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier

import derivation_consistent as dc
import deidentify as di
import disclosure_risk as dr

N_CV_FOLDS = 5
N_CV_REPEATS = 5
BASE_SEED = 42

# Held fixed across every configuration. The only thing that varies between rows
# of the frontier is the data, so any difference in performance is attributable
# to the de-identification rather than to the model.
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "verbosity": 0,
}


def cross_validated_utility(X, y, seed=BASE_SEED, return_folds=False):
    """Mean AUC-ROC and AUC-PR over 5 folds repeated across 5 seeds.

    Both are reported against their no-skill floor, 0.5 for AUC-ROC and the
    prevalence for AUC-PR. Without that an AUC-PR of 0.74 at a prevalence of
    0.35 reads as far better than it is.

    `return_folds` adds the per-fold scores. The confirmatory analysis needs
    them: a mean cannot be paired, and both the equivalence test and the
    stability plot operate on the 25 individual observations.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    prevalence = float(y.mean())

    splitter = RepeatedStratifiedKFold(
        n_splits=N_CV_FOLDS, n_repeats=N_CV_REPEATS, random_state=seed
    )

    roc_scores, pr_scores = [], []
    for train_idx, test_idx in splitter.split(X, y):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        model = XGBClassifier(**XGB_PARAMS, random_state=seed)
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        roc_scores.append(roc_auc_score(y[test_idx], proba))
        pr_scores.append(average_precision_score(y[test_idx], proba))

    out = {
        "n_folds_scored": len(roc_scores),
        "prevalence": prevalence,
        "auc_roc": float(np.mean(roc_scores)) if roc_scores else np.nan,
        "auc_pr": float(np.mean(pr_scores)) if pr_scores else np.nan,
        "auc_roc_sd": float(np.std(roc_scores, ddof=1)) if len(roc_scores) > 1 else np.nan,
        "auc_pr_sd": float(np.std(pr_scores, ddof=1)) if len(pr_scores) > 1 else np.nan,
        "auc_roc_above_floor": float(np.mean(roc_scores)) - 0.5 if roc_scores else np.nan,
        "auc_pr_above_floor": float(np.mean(pr_scores)) - prevalence if pr_scores else np.nan,
    }
    if return_folds:
        out["fold_auc_roc"] = roc_scores
        out["fold_auc_pr"] = pr_scores
    return out


def outcome_follows_predictors(frame, semester_cols, predictors):
    """Records whose outcome grade lies strictly after every predictor column.

    The sensitive attribute is weak performance in each record's own last
    observed position, while the predictors are one fixed prefix of the
    sequence. A record observed no further than that prefix has its outcome
    sitting inside the predictors, so the label is a threshold on a column the
    recipient already holds and the model is not predicting anything. Such
    records can score well and say nothing, so they are kept out of every
    utility sample. Risk is a property of the released file and still uses
    every record.
    """
    columns = list(semester_cols)
    observed = frame[columns].notna().to_numpy()
    last_observed = observed.shape[1] - 1 - np.argmax(observed[:, ::-1], axis=1)
    last_predictor = max(columns.index(c) for c in predictors)
    return pd.Series(observed.any(axis=1) & (last_observed > last_predictor),
                     index=frame.index)


def derived_block(frame, released, predictors, mode):
    """The derived block a recipient models, over the predictor positions only.

    Deliberately not taken from `release.frame`: the release derives over the
    whole sequence, which includes the final observed position the sensitive
    attribute comes from, so modelling that block would leak the target. This
    recomputes over the predictors with the same source rule the release used.

    One function serves both call sites so the mode rule exists once.
    """
    if mode == dc.NONE:
        return None
    if mode == dc.SELECTIVE:
        protected = di.derive_features(released, predictors)
        original = di.derive_features(frame, predictors)
        free = [c for c in original.columns if c not in di.CONSTRAINING_COLUMNS]
        block = protected[
            [c for c in protected.columns if c in di.CONSTRAINING_COLUMNS]
        ].join(original[free])
        return block[list(original.columns)]
    origin = frame if mode == dc.BASELINE else released
    return di.derive_features(origin, predictors)


def release_matrix(frame, semester_cols, predictors, target, band_width=None,
                   suppress_k=None, derived_mode="none", leak_free=True):
    """The feature matrix and labels a recipient of this release could model.

    Extracted so every stage that scores a release builds it the same way. A
    second implementation is how the two arms end up differing in something
    other than the derivation source.

    `leak_free=False` keeps the records whose outcome sits inside the
    predictors. It exists only so the leak can be measured against the sample
    that excludes it.
    """
    release = dc.build(frame, semester_cols, band_width, suppress_k, derived_mode)
    released = release.frame

    features = released[predictors]
    derived = derived_block(frame, released, predictors, release.mode)
    if derived is not None:
        informative = [c for c in derived.columns if derived[c].nunique() > 1]
        features = features.join(derived[informative])

    usable = features.join(frame[[target]]).dropna()
    if leak_free:
        follows = outcome_follows_predictors(frame, semester_cols, predictors)
        usable = usable[follows.reindex(usable.index)]
    columns = [c for c in usable.columns if c != target]
    return usable[columns], usable[target], columns


def frontier_row(
    frame,
    semester_cols,
    predictors,
    target,
    label,
    band_width=None,
    suppress_k=None,
    derived_mode="none",
    return_folds=False,
    leak_free=True,
):
    """One point on the frontier: what this configuration protects, and costs.

    Risk uses every record of the release. Utility uses only the records whose
    outcome follows the predictors, so the two axes can rest on different
    numbers of records, and `n_modelled` says which sample the utility did.
    """
    release = dc.build(frame, semester_cols, band_width, suppress_k, derived_mode)
    released, suppressed = release.frame, release.records_suppressed

    risk = dr.risk_profile(released, release.visible_columns,
                           sensitive=target)

    # What the recipient actually receives: the generalised predictor columns
    # plus the derived block. Passing only the source columns would make the two
    # arms identical by construction, since those columns are byte-identical
    # between them.
    features = released[predictors]
    derived = derived_block(frame, released, predictors, release.mode)
    if derived is not None:
        informative = [c for c in derived.columns if derived[c].nunique() > 1]
        features = features.join(derived[informative])

    usable = features.join(frame[[target]]).dropna()
    if leak_free:
        follows = outcome_follows_predictors(frame, semester_cols, predictors)
        usable = usable[follows.reindex(usable.index)]
    feature_columns = [c for c in usable.columns if c != target]
    if len(usable) > 20 and usable[target].nunique() > 1:
        utility = cross_validated_utility(
            usable[feature_columns], usable[target], return_folds=return_folds
        )
    else:
        utility = {
            k: np.nan
            for k in (
                "prevalence", "auc_roc", "auc_pr", "auc_roc_sd", "auc_pr_sd",
                "auc_roc_above_floor", "auc_pr_above_floor",
            )
        }
        utility["n_folds_scored"] = 0
        if return_folds:
            utility["fold_auc_roc"] = []
            utility["fold_auc_pr"] = []

    return {
        "config": label,
        "band_width": band_width or 0.0,
        "suppress_k": suppress_k or 0,
        "derived_mode": release.mode,
        "records_suppressed": suppressed,
        "n_modelled": len(usable),
        "n_features": len(feature_columns),
        "prop_unique": risk.prop_unique,
        "min_class_size": risk.min_class_size,
        "marketer_risk": risk.marketer_risk,
        "prosecutor_risk": risk.prosecutor_risk,
        "min_l_diversity": risk.min_l_diversity,
        "max_t_closeness": risk.max_t_closeness,
        **utility,
    }
