"""Single source of truth for the NMC-LE feature schema.

Notebooks 00 and 01 import from here so the schema is defined once and cannot
drift between them; notebook 02 inherits these names in-process via run_all.py.

The educator app does not import this module. `engineer_features` is serialised
by value into `results/<synthetic|real>/inference_bundle.pkl` via
`prediction.save_bundle()`, which calls
`cloudpickle.register_pickle_by_value(nmcle_schema)` at bundle-build time. That
lets the app reproduce the exact training-time transform without this file on
its path, so a retrained bundle changes the app's behaviour with no code edit.

To change the exam papers or the feature set, edit only this file.
"""
import pandas as pd

NMC_SUBJECTS = [
    "medical_surgical",
    "mental_health",
    "paediatric",
    "public_health",
    "obstetric",
    "pharmacology",
]

# Scores are percentages; below this a subject counts as a weakness.
WEAK_SCORE_THRESHOLD = 50

CA_COLS = [f"ca_{s}" for s in NMC_SUBJECTS]
MOCK_COLS = [f"mock_{s}" for s in NMC_SUBJECTS]

# wassce_aggregate is deliberately absent: the college registry does not hold
# WASSCE entry grades, so it can never be populated for the real cohort.
NUMERIC_COLS = ["programme_cgpa"] + CA_COLS + MOCK_COLS

# programme_type and region are collected but are not predictors. The cohort is
# a single programme at a single college, so both are constant and would yield
# meaningless SHAP values. They serve as fairness strata instead
# (see fairness_cols in 01_pipeline_and_experiments.py).
CATEGORICAL_COLS = ["age_band", "gender"]

ALL_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS

TARGET = "fail"  # 1 = failed at least one paper on first attempt


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features, all computable before the trainee sits the NMC-LE.

    Restricting inputs to pre-examination information is what keeps the model
    free of target leakage.
    """
    df = df.copy()

    df["ca_avg"] = df[CA_COLS].mean(axis=1)
    df["mock_avg"] = df[MOCK_COLS].mean(axis=1)

    # Positive gap means continuous assessment flattered the trainee relative
    # to their mock performance, which tends to precede a licensure failure.
    df["ca_mock_gap"] = df["ca_avg"] - df["mock_avg"]

    df["ca_consistency"] = df[CA_COLS].std(axis=1)
    df["mock_consistency"] = df[MOCK_COLS].std(axis=1)

    for s in NMC_SUBJECTS:
        df[f"weak_ca_{s}"] = (df[f"ca_{s}"] < WEAK_SCORE_THRESHOLD).astype(int)
        df[f"weak_mock_{s}"] = (df[f"mock_{s}"] < WEAK_SCORE_THRESHOLD).astype(int)

    df["n_weak_ca_subjects"] = df[[f"weak_ca_{s}" for s in NMC_SUBJECTS]].sum(axis=1)
    df["n_weak_mock_subjects"] = df[[f"weak_mock_{s}" for s in NMC_SUBJECTS]].sum(axis=1)

    df["min_mock_score"] = df[MOCK_COLS].min(axis=1)
    df["min_ca_score"] = df[CA_COLS].min(axis=1)

    return df


ENGINEERED_NUMERIC = [
    "ca_avg", "mock_avg", "ca_mock_gap",
    "ca_consistency", "mock_consistency",
    "n_weak_ca_subjects", "n_weak_mock_subjects",
    "min_mock_score", "min_ca_score",
] + [f"weak_ca_{s}" for s in NMC_SUBJECTS] \
  + [f"weak_mock_{s}" for s in NMC_SUBJECTS]

ALL_NUMERIC = NUMERIC_COLS + ENGINEERED_NUMERIC
ALL_FEATURES_V2 = ALL_NUMERIC + CATEGORICAL_COLS
