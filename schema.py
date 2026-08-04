"""Single source of truth for the licensure-failure feature schema.

Defines the raw column names the college supplies, the derived columns
`engineer_features` adds, and the prediction target. Notebooks 00 and 01 import
from here so the schema is defined once and cannot drift between them;
notebook 02 inherits these names in-process via run_all.py.

The educator app does not import this module. `engineer_features` is serialised
by value into `results/<source>/inference_bundle.pkl` via
`prediction.save_bundle()`, which calls
`cloudpickle.register_pickle_by_value(schema)` at bundle-build time. That lets
the app reproduce the exact training-time transform without this file on its
path, so a retrained bundle changes the app's behaviour with no code edit.

Feature design
--------------
The college's records are semester-level: per-semester GPA, credits, and letter
grade counts. There are no per-subject continuous-assessment or mock scores, so
the derived features apply the same averaging, consistency, minimum and
weakness measures to the semester trajectory instead of to subject scores.

To change the feature set, edit only this file.
"""
import numpy as np
import pandas as pd

N_SEMESTERS = 6
SEMESTERS = list(range(1, N_SEMESTERS + 1))

# A semester GPA below this counts as a weak semester. 2.0 is the college's own
# progression threshold, so a student under it was already at academic risk.
WEAK_GPA_THRESHOLD = 2.0

GPA_COLS = [f"gpa_sem{i}" for i in SEMESTERS]
GRADE_LETTERS = list("ABCDE")
GRADE_COUNT_COLS = [f"n_grade_{g}" for g in GRADE_LETTERS]

# Credits attempted per semester. Present in the real dataset, absent from pilot
# data and the alpha corpus, so every use must tolerate their absence.
CREDIT_COLS = [f"credits_sem{i}" for i in SEMESTERS]

# Grades A to D are passes; E is a fail. The college records no F.
PASS_LETTERS = set("ABCD")
FAIL_LETTERS = set("E")

# Raw columns as supplied by the college registry.
NUMERIC_COLS = ["cgpa", "total_credits"] + GPA_COLS + ["n_courses"] + GRADE_COUNT_COLS

# Programme now varies across the cohort (Environmental Health, Occupational
# Health and Safety, Occupational Therapy), so unlike the single-programme
# extraction it carries signal and is a predictor.
CATEGORICAL_COLS = ["programme"]

# cohort_year is context only: it drives the temporal split and is never a
# predictor, since a model keyed on year cannot generalise to a new cohort.
CONTEXT_COLS = ["cohort_year", "student_id"]

ALL_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS

# 1 = failed the AHPC licensure examination at first attempt. The examining body
# is the Allied Health Professions Council (Act 857, 2013), pass mark 60%, sat
# after internship or national service. Not the Nursing and Midwifery Council.
TARGET = "fail"


def _weighted_mean(values, weights):
    return (values * weights).sum(axis=1) / weights.sum(axis=1)


def _weighted_std(values, weights, mean):
    deviation = values.sub(mean, axis=0) ** 2
    return np.sqrt((deviation * weights).sum(axis=1) / weights.sum(axis=1))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features, all computable before the licensure examination.

    Restricting inputs to pre-examination academic records is what keeps the
    model free of target leakage.
    """
    df = df.copy()
    gpas = df[GPA_COLS]

    # Semester 4 is a low-credit term in this cohort (6 credits against 20-24
    # elsewhere), so an unweighted mean gives it 1/6 of the weight against its
    # 7.5% share of credits. Weighting by credits is what CGPA itself does.
    # Pilot data and the alpha corpus carry no per-semester credits, so both
    # measures fall back to unweighted there.
    has_credits = all(col in df.columns for col in CREDIT_COLS)
    if has_credits:
        weights = df[CREDIT_COLS].fillna(0.0)
        weights.columns = GPA_COLS          # align for the elementwise product
        df["gpa_mean"] = _weighted_mean(gpas, weights)
        df["gpa_consistency"] = _weighted_std(gpas, weights, df["gpa_mean"])
    else:
        df["gpa_mean"] = gpas.mean(axis=1)
        df["gpa_consistency"] = gpas.std(axis=1)

    # Extrema name the weakest and strongest semester; weighting an extremum is
    # not meaningful, so these stay unweighted. The semester-4 quantisation does
    # push gpa_max to 4.0 often, which is a known limitation.
    df["gpa_min"] = gpas.min(axis=1)
    df["gpa_max"] = gpas.max(axis=1)

    # Positive trend means the student was improving into their final year,
    # which matters more than the average for a candidate sitting soon.
    df["gpa_trend"] = df[GPA_COLS[-1]] - df[GPA_COLS[0]]
    df["gpa_final_half"] = df[GPA_COLS[3:]].mean(axis=1)
    df["gpa_first_half"] = df[GPA_COLS[:3]].mean(axis=1)

    for i in SEMESTERS:
        df[f"weak_sem{i}"] = (df[f"gpa_sem{i}"] < WEAK_GPA_THRESHOLD).astype(int)
    df["n_weak_semesters"] = df[[f"weak_sem{i}" for i in SEMESTERS]].sum(axis=1)

    df["n_failed"] = df["n_grade_E"]
    n_courses = df["n_courses"].replace(0, np.nan)
    df["fail_rate"] = df["n_failed"] / n_courses
    for g in GRADE_LETTERS:
        df[f"prop_grade_{g}"] = df[f"n_grade_{g}"] / n_courses

    return df


# gpa_mean is deliberately absent. Once credit-weighted it reduces to
# sum(grade_points) / sum(credits), which is exactly how the registry computes
# cgpa — measured correlation 1.000000 on the 2021/2022 cohort. Two identical
# predictors make SHAP split attribution between them arbitrarily, so only cgpa
# is carried. engineer_features still computes gpa_mean, because the workbook is
# read by humans who expect the column.
ENGINEERED_NUMERIC = [
    "gpa_min", "gpa_max", "gpa_consistency",
    "gpa_trend", "gpa_final_half", "gpa_first_half",
    "n_weak_semesters", "n_failed", "fail_rate",
] + [f"weak_sem{i}" for i in SEMESTERS] \
  + [f"prop_grade_{g}" for g in GRADE_LETTERS]

ALL_NUMERIC = NUMERIC_COLS + ENGINEERED_NUMERIC
ALL_FEATURES_V2 = ALL_NUMERIC + CATEGORICAL_COLS
