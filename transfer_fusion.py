"""Intermediate/transfer fusion: pre-train on a generated corpus, fine-tune on field data.

Why this exists
---------------
The reporting template requires two datasets joined by exactly one named fusion
mechanism. The CAN-DO guidance forbids only one of the four the template lists,
naive concatenation into a single training set, because that hides which
population a result applies to. Transfer fusion satisfies both: the corpora are
never pooled, and the mechanism is explicit.

The guidance also names this as the core engineering lever for small samples,
and gives the tree-model recipe directly: train on the source domain, then
continue boosting on the target domain, keeping the early rounds. XGBoost
supports exactly this through warm-started boosting, so the pre-trained trees
remain in the ensemble and the fine-tuning rounds are fitted on field data.

The disclosure this obliges
---------------------------
A generated corpus contributes to training. That must be stated in the Methods,
not buried. It is also why the source is regenerated inside every fold: the
synthesiser is fitted to the fold's training partition alone, so it never sees a
held-out label. Fitting it once on the full dataset would leak the test folds
into the source corpus, which is the same defect that inflated the pruning
comparison before selection was nested.

What it does and does not claim
-------------------------------
When the source corpus is a replica fitted to the field data, it carries no
information the field data does not already hold. Pre-training on it therefore
evaluates pipeline behaviour under a transfer protocol; it is not evidence of
external domain transfer, and the Methods says so. What it does supply is a
larger, smoother corpus for the early boosting rounds, which is a defensible
response to an events-per-predictor ratio near 1.0.
"""
import numpy as np
import xgboost as xgb
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             roc_auc_score)

import synthetic_data

# Of the total boosting rounds, this share is fitted on the source corpus and
# the remainder on the field data. Two thirds pre-train leaves enough capacity
# for the target domain to move the model while still benefiting from a smoother
# start. Fixed before the comparison runs; not tuned.
PRETRAIN_SHARE = 2 / 3

# The source corpus is generated at this multiple of the training partition.
# Larger than the target because the point of pre-training is a smoother
# gradient estimate, not a matched replica.
SOURCE_MULTIPLIER = 3


def generate_source(X_train, y_train, target_col, categorical_cols, seed,
                     multiplier=SOURCE_MULTIPLIER):
    """Fit the synthesiser on this fold's training partition and sample from it.

    Fitted per fold, never on the full dataset, so no held-out label reaches
    the source corpus.
    """
    frame = X_train.copy()
    frame[target_col] = y_train.values
    replica = synthetic_data.generate_field_replica(
        frame, target_col, categorical_cols, seed,
        n_rows=int(len(frame) * multiplier))
    return replica.drop(columns=[target_col]), replica[target_col]


def fit_transfer(X_source, y_source, X_target, y_target, params, seed,
                  pretrain_share=PRETRAIN_SHARE):
    """Boost on the source corpus, then continue boosting on the target.

    Returns the fine-tuned model. The source-trained trees stay in the ensemble;
    the fine-tuning rounds are fitted on field data only.
    """
    total = int(params.get("n_estimators", 300))
    n_pretrain = max(1, int(total * pretrain_share))
    n_finetune = max(1, total - n_pretrain)

    source_params = {**params, "n_estimators": n_pretrain}
    base = xgb.XGBClassifier(**source_params, eval_metric="logloss",
                              random_state=seed, verbosity=0)
    base.fit(X_source, y_source)

    target_params = {**params, "n_estimators": n_finetune}
    tuned = xgb.XGBClassifier(**target_params, eval_metric="logloss",
                               random_state=seed, verbosity=0)
    tuned.fit(X_target, y_target, xgb_model=base.get_booster())
    return tuned


def run_transfer_cv(feature_subset, X, y, all_numeric, categorical_cols,
                     params, skf, seed, target_col, label,
                     build_preprocessor):
    """Cross-validate the transfer-fusion pipeline with per-fold source generation.

    `build_preprocessor` is injected rather than imported so this module does
    not depend on the notebook's preprocessing choices.
    """
    Xsub = X[feature_subset]
    res = {"acc": [], "f1": [], "auc_roc": [], "auc_pr": [], "time": []}

    for fold, (tr_idx, te_idx) in enumerate(skf.split(Xsub, y)):
        Xtr, Xte = Xsub.iloc[tr_idx], Xsub.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

        X_src, y_src = generate_source(
            Xtr, ytr, target_col, categorical_cols, seed + fold)

        pre = build_preprocessor(feature_subset, all_numeric, categorical_cols)
        # Fitted on the target training partition, then applied to both corpora,
        # so the two share one feature space without the source defining it.
        Xtr_p = pre.fit_transform(Xtr)
        Xsrc_p = pre.transform(X_src[feature_subset])
        Xte_p = pre.transform(Xte)

        model = fit_transfer(Xsrc_p, y_src, Xtr_p, ytr, params, seed)

        preds = model.predict(Xte_p)
        proba = model.predict_proba(Xte_p)[:, 1]
        res["acc"].append(accuracy_score(yte, preds))
        res["f1"].append(f1_score(yte, preds, average="macro", zero_division=0))
        res["auc_roc"].append(roc_auc_score(yte, proba))
        res["auc_pr"].append(average_precision_score(yte, proba))
        res["time"].append(0.0)

    print(f"   [{label}] {len(feature_subset)} features | "
          f"AUC-ROC {np.mean(res['auc_roc']):.4f} | "
          f"AUC-PR {np.mean(res['auc_pr']):.4f}")
    return res
