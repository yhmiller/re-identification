"""
engineered_xgboost.py — E-XGBoost: SHAP-guided feature pruning (Operation R).

Ranks features by global SHAP importance, removes the low-contribution tail
(keeping the smallest subset that explains 95% of SHAP mass), and retrains
XGBoost on the pruned set with the identical CV folds and hyperparameters
used for the baseline — isolating the effect of the feature-set change.

Used by notebooks/02_model_engineering.py.
"""
import time

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

KEEP_CUM_THRESHOLD = 0.95   # keep features explaining 95% of SHAP mass


def _build_preprocessor(feature_subset, all_numeric, categorical_cols):
    """Median-impute and scale numerics, mode-impute and one-hot categoricals."""
    num_sub = [c for c in feature_subset if c in all_numeric]
    cat_sub = [c for c in feature_subset if c in categorical_cols]

    num_tf = Pipeline([("imp", SimpleImputer(strategy="median")),
                        ("sc", MinMaxScaler())])
    cat_tf = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore",
                                              sparse_output=False))])
    return ColumnTransformer([("num", num_tf, num_sub),
                               ("cat", cat_tf, cat_sub)], remainder="drop")


def _encoded_names(fitted_preprocessor):
    """Column names after encoding, with the ColumnTransformer prefix stripped
    so they match the raw feature names `map_to_original` expects."""
    return [name.split("__", 1)[-1]
            for name in fitted_preprocessor.get_feature_names_out()]


def run_cv(feature_subset, X, y, all_numeric, categorical_cols, xgb_params, skf, seed, label):
    """
    Stratified CV (fold count set by the caller's `skf`) on a given feature
    subset, preprocessing refit inside each fold (no leakage). Returns
    per-fold metric lists + times.
    (was ENG-CELL 2's run_cv, made explicit over its inputs)
    """
    pre = _build_preprocessor(feature_subset, all_numeric, categorical_cols)
    Xsub = X[feature_subset]
    res = {"acc": [], "f1": [], "auc_roc": [], "auc_pr": [], "time": []}

    for tr_idx, te_idx in skf.split(Xsub, y):
        Xtr, Xte = Xsub.iloc[tr_idx], Xsub.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

        Xtr_p = pre.fit_transform(Xtr)
        Xte_p = pre.transform(Xte)

        model = xgb.XGBClassifier(**xgb_params, eval_metric="logloss",
                                   random_state=seed, verbosity=0)
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


def run_nested_cv(X, y, all_numeric, categorical_cols, all_features,
                   xgb_params, skf, seed, keep_threshold, label,
                   ranking="shap"):
    """Score SHAP-pruned XGBoost with feature selection nested inside each fold.

    Why this exists
    ---------------
    `rank_shap_importance` ranks on the full dataset. Pruning from that ranking
    and then scoring on folds drawn from the same data lets feature selection
    see the labels of every held-out fold, which biases the comparison in favour
    of the engineered model. Vabalas et al. (2019) show that k-fold CV gives
    strongly biased estimates at small sample sizes and that nesting the
    *feature selection* is the part that matters most for controlling it. At
    n=110 that bias is not a technicality.

    Here the ranking model, the SHAP computation and the pruning decision all
    happen on the training fold alone, so the held-out fold is never seen by
    selection. The kept set therefore differs between folds; the per-fold
    selections are returned so feature-set stability can be reported, which is
    itself evidence about how reliable the ranking is at this sample size.

    `ranking` is "shap" (mean absolute SHAP) or "gain" (XGBoost's built-in
    importance), so the two can be compared on identical folds. Wang et al.
    (2024) report importance-based selection beating SHAP-based selection, and
    that claim is only answerable with both arms measured the same way.
    """
    res = {"acc": [], "f1": [], "auc_roc": [], "auc_pr": [], "time": [],
           "n_features": []}
    selections = []

    for tr_idx, te_idx in skf.split(X, y):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

        # Rank on the training fold only.
        rank_pre = _build_preprocessor(all_features, all_numeric, categorical_cols)
        Xtr_ranked = rank_pre.fit_transform(Xtr[all_features])
        enc_names = _encoded_names(rank_pre)

        rank_model = xgb.XGBClassifier(**xgb_params, eval_metric="logloss",
                                        random_state=seed, verbosity=0)
        rank_model.fit(Xtr_ranked, ytr)

        if ranking == "gain":
            booster_scores = rank_model.get_booster().get_score(importance_type="gain")
            enc_importance = pd.Series(
                [booster_scores.get(f"f{i}", 0.0) for i in range(len(enc_names))],
                index=enc_names).sort_values(ascending=False)
        else:
            shap_vals = shap.TreeExplainer(rank_model).shap_values(Xtr_ranked)
            enc_importance = pd.Series(
                np.abs(shap_vals).mean(axis=0), index=enc_names
            ).sort_values(ascending=False)

        orig_importance = map_to_original(enc_importance, categorical_cols)
        kept, _, _ = select_pruned_features(orig_importance, all_features,
                                             keep_threshold)
        selections.append(kept)

        # Retrain and score on the pruned set, preprocessing refit inside the fold.
        fold_pre = _build_preprocessor(kept, all_numeric, categorical_cols)
        Xtr_p = fold_pre.fit_transform(Xtr[kept])
        Xte_p = fold_pre.transform(Xte[kept])

        model = xgb.XGBClassifier(**xgb_params, eval_metric="logloss",
                                   random_state=seed, verbosity=0)
        t0 = time.time()
        model.fit(Xtr_p, ytr)
        res["time"].append(time.time() - t0)

        preds = model.predict(Xte_p)
        proba = model.predict_proba(Xte_p)[:, 1]
        res["acc"].append(accuracy_score(yte, preds))
        res["f1"].append(f1_score(yte, preds, average="macro", zero_division=0))
        res["auc_roc"].append(roc_auc_score(yte, proba))
        res["auc_pr"].append(average_precision_score(yte, proba))
        res["n_features"].append(len(kept))

    print(f"   [{label}] {np.mean(res['n_features']):.1f} features (mean across folds) | "
          f"AUC-ROC {np.mean(res['auc_roc']):.4f} | "
          f"AUC-PR {np.mean(res['auc_pr']):.4f}")
    return res, selections


def selection_stability(selections, all_features):
    """How often each feature survived pruning across folds."""
    n = len(selections)
    counts = {f: sum(f in sel for sel in selections) for f in all_features}
    return (pd.DataFrame({
        "feature": list(counts),
        "folds_retained": list(counts.values()),
        "retention_rate": [c / n for c in counts.values()],
    }).sort_values(["folds_retained", "feature"], ascending=[False, True])
       .reset_index(drop=True))


def rank_shap_importance(X, y, preprocessor, xgb_params, feature_names, seed):
    """
    Fit one XGBoost on the FULL data (refits `preprocessor` in place, same
    as the original ENG-CELL 4) and rank encoded features by mean |SHAP|.

    This full-data ranking is what the exported app bundle and the reported
    feature list use. It must NOT be used to produce the comparison numbers;
    `run_nested_cv` does that without leaking held-out labels into selection.
    """
    X_full_proc = preprocessor.fit_transform(X)
    rank_model = xgb.XGBClassifier(**xgb_params, eval_metric="logloss",
                                    random_state=seed, verbosity=0)
    rank_model.fit(X_full_proc, y)

    explainer = shap.TreeExplainer(rank_model)
    shap_vals = explainer.shap_values(X_full_proc)

    return pd.Series(
        np.abs(shap_vals).mean(axis=0), index=feature_names
    ).sort_values(ascending=False)


def map_to_original(enc_series, categorical_cols):
    """Aggregate encoded-feature SHAP importance back to source columns."""
    orig = {}
    for enc_name, val in enc_series.items():
        matched = None
        for cat in categorical_cols:
            if enc_name.startswith(cat + "_"):
                matched = cat
                break
        key = matched if matched else enc_name
        orig[key] = orig.get(key, 0.0) + val
    return pd.Series(orig).sort_values(ascending=False)


def select_pruned_features(orig_importance, all_features, keep_threshold=KEEP_CUM_THRESHOLD):
    """
    THE ENGINEERING DECISION [Operation R: Remove]. Keep the smallest set of
    features that together explain `keep_threshold` of total SHAP mass;
    everything else is pruned. Returns (kept_features, pruned_features, cum).
    """
    cum = orig_importance.cumsum() / orig_importance.sum()

    kept_features = cum[cum <= keep_threshold].index.tolist()
    if len(kept_features) == 0:
        kept_features = [orig_importance.index[0]]
    if len(kept_features) < len(orig_importance):
        next_feat = cum.index[len(kept_features)]
        kept_features.append(next_feat)

    pruned_features = [f for f in all_features if f not in kept_features]
    return kept_features, pruned_features, cum


def build_engineered_preprocessor(kept_features, all_numeric, categorical_cols):
    """Preprocessor scoped to only the retained features. (was part of ENG-CELL 11)"""
    ex_num = [c for c in kept_features if c in all_numeric]
    ex_cat = [c for c in kept_features if c in categorical_cols]

    transformers = []
    if ex_num:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler())]), ex_num))
    if ex_cat:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), ex_cat))
    return ColumnTransformer(transformers, remainder="drop"), ex_num, ex_cat


def train_engineered_model(X_train, y_train, X_val, y_val, kept_features,
                            all_numeric, categorical_cols, xgb_params, spw, seed):
    """
    Fit the final E-XGBoost model on the retained feature set. Returns
    (model, preprocessor, encoded_feature_names).
    """
    ex_pre, ex_num, ex_cat = build_engineered_preprocessor(
        kept_features, all_numeric, categorical_cols)
    ex_pre.fit(X_train[kept_features])
    ex_Xtr = ex_pre.transform(X_train[kept_features])
    ex_Xva = ex_pre.transform(X_val[kept_features])

    ex_cat_names = (
        ex_pre.named_transformers_["cat"].named_steps["encoder"]
        .get_feature_names_out(ex_cat).tolist() if ex_cat else []
    )
    ex_feature_names = ex_num + ex_cat_names

    ex_model = xgb.XGBClassifier(
        **xgb_params, scale_pos_weight=spw, eval_metric="aucpr",
        early_stopping_rounds=50, random_state=seed, verbosity=0,
        use_label_encoder=False,
    )
    ex_model.fit(ex_Xtr, y_train, eval_set=[(ex_Xva, y_val)], verbose=False)
    return ex_model, ex_pre, ex_feature_names
