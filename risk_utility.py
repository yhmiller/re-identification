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


def cross_validated_utility(X, y, seed=BASE_SEED):
    """Mean AUC-ROC and AUC-PR over 5 folds repeated across 5 seeds.

    Both are reported against their no-skill floor, 0.5 for AUC-ROC and the
    prevalence for AUC-PR. Without that an AUC-PR of 0.74 at a prevalence of
    0.35 reads as far better than it is.
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

    return {
        "n_folds_scored": len(roc_scores),
        "prevalence": prevalence,
        "auc_roc": float(np.mean(roc_scores)) if roc_scores else np.nan,
        "auc_pr": float(np.mean(pr_scores)) if pr_scores else np.nan,
        "auc_roc_above_floor": (
            float(np.mean(roc_scores)) - 0.5 if roc_scores else np.nan
        ),
        "auc_pr_above_floor": (
            float(np.mean(pr_scores)) - prevalence if pr_scores else np.nan
        ),
    }


def apply_release(frame, semester_cols, band_width, suppress_k, derived_mode):
    """Build the file a custodian would actually hand out.

    `derived_mode` is the decision Experiment B made consequential:
        none        publish no derived features
        leaky       derive from the originals, then publish alongside the bands
        safe        derive from the generalised values instead
    """
    released = frame.copy()
    suppressed = 0

    if band_width:
        released = di.generalise(released, semester_cols, band_width)

    if suppress_k and suppress_k > 1:
        released, suppressed = di.suppress(released, semester_cols, suppress_k)

    derived = None
    if derived_mode == "leaky":
        derived = di.derive_features(frame, semester_cols)
    elif derived_mode == "safe":
        derived = di.derive_features(released, semester_cols)

    if derived is not None:
        released = released.drop(columns=derived.columns, errors="ignore").join(derived)

    return released, suppressed


def frontier_row(
    frame,
    semester_cols,
    predictors,
    target,
    label,
    band_width=None,
    suppress_k=None,
    derived_mode="none",
):
    """One point on the frontier: what this configuration protects, and costs."""
    released, suppressed = apply_release(
        frame, semester_cols, band_width, suppress_k, derived_mode
    )

    # Risk is measured on everything the recipient can see.
    visible = list(semester_cols)
    if derived_mode != "none":
        visible += di.DERIVED_COLUMNS
    risk = dr.risk_profile(released, visible)

    # Utility uses only the predictors, which the model would actually consume.
    usable = released[predictors].join(frame[[target]]).dropna()
    utility = (
        cross_validated_utility(usable[predictors], usable[target])
        if len(usable) > 20 and usable[target].nunique() > 1
        else {
            "n_folds_scored": 0,
            "prevalence": np.nan,
            "auc_roc": np.nan,
            "auc_pr": np.nan,
            "auc_roc_above_floor": np.nan,
            "auc_pr_above_floor": np.nan,
        }
    )

    return {
        "config": label,
        "band_width": band_width or 0.0,
        "suppress_k": suppress_k or 0,
        "derived_mode": derived_mode,
        "records_suppressed": suppressed,
        "n_modelled": len(usable),
        "prop_unique": risk.prop_unique,
        "min_class_size": risk.min_class_size,
        "marketer_risk": risk.marketer_risk,
        **utility,
    }
