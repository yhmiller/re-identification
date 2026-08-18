"""
03_hybrid_transfer.py — Hybrid data fusion via transfer learning.

Implements the Q1 template's Intermediate/Transfer fusion (Section 3):
PRE-TRAIN an XGBoost model on the public dataset, then FINE-TUNE (continue
boosting) on the college data. Both datasets are expressed in ONE shared
"common schema", because XGBoost continued training requires identical feature
columns across the two phases.

The public data NEVER appears in reported results; it only initialises the model.
All evaluation is on the held-out COLLEGE test split.

This script runs on SYNTHETIC college data so the fusion is verified before the
real records arrive. When real data is ready, replace make_synthetic_college()
with a load of the anonymised records and the same to_common_schema() applies.

Run:  ml_env/bin/python notebooks/03_hybrid_transfer.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

SEED = 42
REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_CSV = REPO_ROOT / "data" / "public" / "student_por_trimmed.csv"

# ── Common schema shared by BOTH datasets (the only columns the transfer sees) ──
NUMERIC_FEATURES = ["prior_score_1", "prior_score_2", "prior_failures"]
CATEGORICAL_FEATURES = ["age_band", "gender"]
COMMON_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "fail"

# Fixed category vocabularies guarantee identical encoded columns in both phases,
# regardless of which values happen to appear in either dataset.
AGE_BANDS = ["Below 20", "20-24", "25-29", "30+"]
GENDERS = ["Female", "Male"]

WEAK_SCORE = 50  # a subject score below this counts as "weak" (matches Stage 0)
SYNTH_COLLEGE_N = 250  # stand-in until the real records arrive
PRETRAIN_ROUNDS = 200
FINETUNE_ROUNDS = 100
XGB_PARAMS = dict(
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="aucpr",
    random_state=SEED,
    verbosity=0,
)


def make_synthetic_college(n):
    """A stand-in college dataset in the project's raw schema (6 CA + 6 mock)."""
    rng = np.random.default_rng(SEED)
    ca = np.clip(rng.normal(62, 12, (n, 6)), 20, 100)
    mock = np.clip(rng.normal(58, 15, (n, 6)), 10, 100)

    df = pd.DataFrame(
        {
            **{f"ca_{i+1}": ca[:, i] for i in range(6)},
            **{f"mock_{i+1}": mock[:, i] for i in range(6)},
        }
    )
    df["age_band"] = rng.choice(AGE_BANDS, n, p=[0.10, 0.55, 0.25, 0.10])
    df["gender"] = rng.choice(GENDERS, n, p=[0.72, 0.28])

    # Failure is driven by weak mock performance, centred near a ~50% rate.
    mock_avg = mock.mean(axis=1)
    logit = -(mock_avg - mock_avg.mean()) / 10 + rng.normal(0, 0.5, n)
    df[TARGET] = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)
    return df


def to_common_schema(college_df):
    """Map the college raw schema (6 CA + 6 mock + demographics) onto the
    common schema shared with the public data."""
    ca_cols = [f"ca_{i+1}" for i in range(6)]
    mock_cols = [f"mock_{i+1}" for i in range(6)]
    out = pd.DataFrame()
    out["prior_score_1"] = college_df[mock_cols].mean(axis=1)  # ~ mock_avg
    out["prior_score_2"] = college_df[ca_cols].mean(axis=1)  # ~ ca_avg
    out["prior_failures"] = (college_df[mock_cols] < WEAK_SCORE).sum(axis=1)
    out["age_band"] = college_df["age_band"]
    out["gender"] = college_df["gender"]
    out[TARGET] = college_df[TARGET].values
    return out


def build_preprocessor():
    numeric = Pipeline(
        [("imp", SimpleImputer(strategy="median")), ("sc", MinMaxScaler())]
    )
    categorical = Pipeline(
        [
            ("imp", SimpleImputer(strategy="most_frequent")),
            (
                "oh",
                OneHotEncoder(
                    categories=[AGE_BANDS, GENDERS],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)]
    )


def evaluate(model, X, y, label):
    proba = model.predict_proba(X)[:, 1]
    auc_roc = roc_auc_score(y, proba)
    auc_pr = average_precision_score(y, proba)
    print(f"   {label:<28} AUC-ROC {auc_roc:.4f} | AUC-PR {auc_pr:.4f}")
    return auc_roc, auc_pr


def main():
    public = pd.read_csv(PUBLIC_CSV)
    college = to_common_schema(make_synthetic_college(SYNTH_COLLEGE_N))
    print(f"Public (pre-train): {public.shape} | fail {public[TARGET].mean():.1%}")
    print(f"College (fine-tune): {college.shape} | fail {college[TARGET].mean():.1%}")

    # Preprocessor fitted on the SOURCE (public) domain; both phases transformed
    # with it so the feature matrices share identical columns.
    pre = build_preprocessor().fit(public[COMMON_FEATURES])
    X_pub = pre.transform(public[COMMON_FEATURES])
    y_pub = public[TARGET].values

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        college[COMMON_FEATURES],
        college[TARGET],
        test_size=0.30,
        stratify=college[TARGET],
        random_state=SEED,
    )
    Xc_train_p = pre.transform(Xc_train)
    Xc_test_p = pre.transform(Xc_test)

    # PRE-TRAIN on public.
    base = xgb.XGBClassifier(n_estimators=PRETRAIN_ROUNDS, **XGB_PARAMS)
    base.fit(X_pub, y_pub)

    # College-only baseline (no transfer) — the control.
    college_only = xgb.XGBClassifier(
        n_estimators=PRETRAIN_ROUNDS + FINETUNE_ROUNDS, **XGB_PARAMS
    )
    college_only.fit(Xc_train_p, yc_train)

    # FINE-TUNE: continue boosting the pre-trained model on the college data.
    hybrid = xgb.XGBClassifier(n_estimators=FINETUNE_ROUNDS, **XGB_PARAMS)
    hybrid.fit(Xc_train_p, yc_train, xgb_model=base.get_booster())

    print("\nEvaluation on the held-out COLLEGE test split:")
    evaluate(college_only, Xc_test_p, yc_test, "College-only (no transfer)")
    evaluate(hybrid, Xc_test_p, yc_test, "Hybrid (pre-train + fine-tune)")
    print(
        "\nNote: synthetic-college numbers are illustrative only; the real "
        "transfer benefit is measured once the college records arrive."
    )


if __name__ == "__main__":
    main()
