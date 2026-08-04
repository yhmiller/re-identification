"""
baseline_xgboost.py — preprocessing, the baseline comparator models
(Logistic Regression, Random Forest, LightGBM), and the baseline XGBoost
model (grid search + final fit + repeated-seed variance + temporal
validation + ablation).

"Baseline" here means the standard XGBoost trained on ALL features — the
model E-XGBoost (engineered_xgboost.py) is compared against. Used by
notebooks/01_pipeline_and_experiments.py, and reused as-is for the Option 2
pipeline-replication run on synthetic/public data (see docs/TODO.md).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import average_precision_score, roc_auc_score

from stats_validation import evaluate_model

# Single source of truth for the CV fold count — used for the XGBoost grid
# search and ablation here, and imported by notebooks/02_model_engineering.py
# to build the SAME-fold SKF the baseline-vs-E-XGBoost comparison relies on.
#
# Kept at 5, not 10: the real college dataset will be small (300-800 records
# per the proposal). At 10 folds each test fold only holds ~30-80 rows and,
# at a ~15-30% fail rate, as few as 5-25 Fail cases — too little for a
# stable per-fold AUC-PR estimate. The 10-fold trial run showed exactly this:
# per-fold SD roughly doubled and AUC-PR's sign flipped vs baseline, without
# that flip being significant either way (p=0.92) — noise, not a real gain
# from more folds. 5-fold trades a touch more bias for meaningfully less
# variance, which matters more at this sample size.
N_CV_FOLDS = 5

XGB_PARAM_GRID = {
    "max_depth": [3, 6],
    "learning_rate": [0.05, 0.10],
    "n_estimators": [300, 500],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}


def build_preprocessor(all_numeric, categorical_cols):
    """Impute → scale numeric; impute → one-hot categorical. (was CELL 8)"""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, all_numeric),
        ("cat", categorical_transformer, categorical_cols),
    ], remainder="drop")


def encoded_feature_names(preprocessor, all_numeric, categorical_cols):
    cat_features = (preprocessor
                    .named_transformers_["cat"]
                    .named_steps["encoder"]
                    .get_feature_names_out(categorical_cols)
                    .tolist())
    return all_numeric + cat_features


def train_comparator_baselines(X_train_proc, y_train, X_val_proc, y_val,
                                X_test_proc, y_test, seed):
    """Logistic Regression, Random Forest, LightGBM. (was CELL 10)

    Returns (results: list[dict], models: dict[name, fitted estimator],
    probas: dict[name, np.ndarray]).
    """
    lr = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
    lr.fit(X_train_proc, y_train)
    lr_proba = lr.predict_proba(X_test_proc)[:, 1]
    lr_res = evaluate_model("Logistic Regression", y_test, lr_proba, lr.predict(X_test_proc))

    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed,
                                 class_weight="balanced")
    rf.fit(X_train_proc, y_train)
    rf_proba = rf.predict_proba(X_test_proc)[:, 1]
    rf_res = evaluate_model("Random Forest", y_test, rf_proba, rf.predict(X_test_proc))

    lgbm_clf = lgb.LGBMClassifier(n_estimators=500, random_state=seed,
                                   verbose=-1, class_weight="balanced")
    lgbm_clf.fit(X_train_proc, y_train,
                 eval_set=[(X_val_proc, y_val)],
                 callbacks=[lgb.early_stopping(50, verbose=False)])
    lgbm_proba = lgbm_clf.predict_proba(X_test_proc)[:, 1]
    lgbm_res = evaluate_model("LightGBM", y_test, lgbm_proba, lgbm_clf.predict(X_test_proc))

    results = [lr_res, rf_res, lgbm_res]
    models = {"Logistic Regression": lr, "Random Forest": rf, "LightGBM": lgbm_clf}
    probas = {"Logistic Regression": lr_proba, "Random Forest": rf_proba, "LightGBM": lgbm_proba}
    return results, models, probas


def tune_xgboost(X_train_proc, y_train, seed):
    """N_CV_FOLDS-fold grid search on AUC-PR. Returns (best_params, spw, grid). (was CELL 10.5)"""
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = round(neg / pos, 2) if pos > 0 else 1

    grid = GridSearchCV(
        xgb.XGBClassifier(scale_pos_weight=spw, eval_metric="aucpr",
                           random_state=seed, verbosity=0, use_label_encoder=False),
        XGB_PARAM_GRID,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed),
        n_jobs=1,
    )
    grid.fit(X_train_proc, y_train)
    return grid.best_params_, spw, grid


def train_xgboost(X_train_proc, y_train, X_val_proc, y_val, X_test_proc, y_test,
                   best_params, spw, seed):
    """Final XGBoost fit with early stopping on validation AUC-PR. (was CELL 11)

    Returns (model, result_dict, proba, labels).
    """
    xgb_clf = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=seed,
        verbosity=0,
        use_label_encoder=False,
    )
    xgb_clf.fit(X_train_proc, y_train, eval_set=[(X_val_proc, y_val)], verbose=False)

    xgb_proba = xgb_clf.predict_proba(X_test_proc)[:, 1]
    xgb_labels = xgb_clf.predict(X_test_proc)
    xgb_res = evaluate_model("XGBoost", y_test, xgb_proba, xgb_labels)
    return xgb_clf, xgb_res, xgb_proba, xgb_labels


def repeated_seed_variance(X_train_proc, y_train, X_test_proc, y_test,
                            best_params, spw, n_runs=10):
    """Retrain under N random seeds; report metric spread. (was CELL 11.5)"""
    rows = []
    for s in range(n_runs):
        m = xgb.XGBClassifier(**best_params, scale_pos_weight=spw,
                               eval_metric="aucpr", random_state=s,
                               verbosity=0, use_label_encoder=False)
        m.fit(X_train_proc, y_train)
        p = m.predict_proba(X_test_proc)[:, 1]
        rows.append({
            "seed": s,
            "auc_roc": round(roc_auc_score(y_test, p), 4),
            "auc_pr": round(average_precision_score(y_test, p), 4),
        })
    return pd.DataFrame(rows)


def temporal_validation(raw_df, X, y, preprocessor, best_params, spw, seed):
    """Train on earlier cohorts, test on the latest one. (was CELL 11.7)

    Returns a one-row DataFrame, or None if cohort_year is absent or the
    latest cohort is single-class.
    """
    if "cohort_year" not in raw_df.columns:
        return None

    years = sorted(raw_df["cohort_year"].unique())
    latest = years[-1]
    train_mask = raw_df["cohort_year"] < latest
    test_mask = raw_df["cohort_year"] == latest
    X_train_t, y_train_t = X[train_mask], y[train_mask]
    X_test_t, y_test_t = X[test_mask], y[test_mask]

    if y_test_t.nunique() <= 1:
        return None

    pre_t = clone(preprocessor)
    Xtr_t = pre_t.fit_transform(X_train_t)
    Xte_t = pre_t.transform(X_test_t)
    m_t = xgb.XGBClassifier(**best_params, scale_pos_weight=spw,
                             eval_metric="aucpr", random_state=seed,
                             verbosity=0, use_label_encoder=False)
    m_t.fit(Xtr_t, y_train_t)
    p_t = m_t.predict_proba(Xte_t)[:, 1]
    return pd.DataFrame([{
        "train_cohorts": ",".join(str(int(yr)) for yr in years[:-1]),
        "test_cohort": int(latest),
        "n_train": int(len(X_train_t)),
        "n_test": int(len(X_test_t)),
        "auc_roc": round(roc_auc_score(y_test_t, p_t), 4),
        "auc_pr": round(average_precision_score(y_test_t, p_t), 4),
    }])


def ablation_auc_pr(feature_subset, X, y, all_numeric, categorical_cols,
                     best_params, spw, seed):
    """N_CV_FOLDS-fold CV mean/std AUC-PR for one feature subset (no leakage). (was CELL 18.5)"""
    num_sub = [c for c in feature_subset if c in all_numeric]
    cat_sub = [c for c in feature_subset if c in categorical_cols]
    transformers = []
    if num_sub:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", MinMaxScaler())]), num_sub))
    if cat_sub:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
            cat_sub))
    pre = ColumnTransformer(transformers, remainder="drop")
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
    Xsub = X[feature_subset]
    scores = []
    for tr_idx, te_idx in skf.split(Xsub, y):
        Xtr = pre.fit_transform(Xsub.iloc[tr_idx])
        Xte = pre.transform(Xsub.iloc[te_idx])
        m = xgb.XGBClassifier(**best_params, scale_pos_weight=spw,
                               eval_metric="aucpr", random_state=seed,
                               verbosity=0, use_label_encoder=False)
        m.fit(Xtr, y.iloc[tr_idx])
        scores.append(average_precision_score(
            y.iloc[te_idx], m.predict_proba(Xte)[:, 1]))
    return float(np.mean(scores)), float(np.std(scores))


def run_ablation_study(X, y, all_numeric, categorical_cols, all_features,
                        best_params, spw, seed):
    """Run the fixed set of ablation variants and return a results table."""
    # "No demographics" answers whether age_band/gender earn their place as
    # predictors, or whether the academic indicators alone perform as well.
    variants = {
        "All features": all_features,
        "CGPA only": ["programme_cgpa"],
        "No mock scores": [f for f in all_features if "mock" not in f],
        "No demographics": [f for f in all_features
                             if f not in ("age_band", "gender")],
    }
    rows = []
    for name, feats in variants.items():
        mean_pr, std_pr = ablation_auc_pr(feats, X, y, all_numeric, categorical_cols,
                                           best_params, spw, seed)
        rows.append({
            "variant": name,
            "n_features": len(feats),
            "auc_pr_mean": round(mean_pr, 4),
            "auc_pr_std": round(std_pr, 4),
        })
    return pd.DataFrame(rows)
