"""
inference.py — load the trained model bundle and score student records.

The bundle (results/inference_bundle.pkl) is produced by Stage 1 and carries the
model, the fitted preprocessor, and the exact engineer_features function used in
training — so scoring here reproduces the training-time transform precisely.
"""
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import shap

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_PATH = REPO_ROOT / "results" / "inference_bundle.pkl"

RISK_PROB_COLUMN = "fail_probability"
RANK_COLUMN = "risk_rank"
STUDENT_ID_COLUMN = "student_id"


def load_bundle(path=BUNDLE_PATH):
    """Load the inference bundle. Raises FileNotFoundError if Stage 1 hasn't run.

    Safe pickle use: this file is produced locally by our own Stage 1 run
    (results/inference_bundle.pkl), never fetched from an external/untrusted
    source. cloudpickle is required because the bundle holds the live
    engineer_features function, not just plain data.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No model bundle at {path}. Run ./run.sh (or run_all.py) first."
        )
    with open(path, "rb") as f:
        return cloudpickle.load(f)


def required_raw_columns(bundle):
    """The columns an uploaded spreadsheet must contain (excludes the target)."""
    return bundle["raw_numeric_columns"] + bundle["categorical_columns"]


def missing_columns(raw_df, bundle):
    """Return required columns absent from the uploaded data."""
    return [c for c in required_raw_columns(bundle) if c not in raw_df.columns]


def _to_model_matrix(raw_df, bundle):
    """raw records → engineered → ordered feature frame → preprocessed matrix."""
    engineered = bundle["engineer_features"](raw_df)
    ordered = engineered[bundle["feature_columns"]]
    return bundle["preprocessor"].transform(ordered)


def score_students(raw_df, bundle, threshold):
    """
    Score every record and return a frame sorted by fail probability (highest
    first), with a risk rank and an at-risk flag based on the threshold.
    """
    matrix = _to_model_matrix(raw_df, bundle)
    proba = bundle["model"].predict_proba(matrix)[:, 1]

    result = raw_df.copy()
    if STUDENT_ID_COLUMN not in result.columns:
        result.insert(0, STUDENT_ID_COLUMN, [f"row_{i}" for i in range(len(result))])
    result[RISK_PROB_COLUMN] = proba
    result["at_risk"] = proba >= threshold
    result = result.sort_values(RISK_PROB_COLUMN, ascending=False).reset_index(drop=True)
    result[RANK_COLUMN] = np.arange(1, len(result) + 1)
    return result


def explain_student(raw_row, bundle, top_n=8):
    """
    Return the top_n encoded features pushing one student's prediction, as a
    frame of (feature, shap_value) sorted by absolute contribution.
    """
    matrix = _to_model_matrix(raw_row, bundle)
    explainer = shap.TreeExplainer(bundle["model"])
    shap_values = explainer.shap_values(matrix)[0]

    contributions = pd.DataFrame({
        "feature": bundle["encoded_feature_names"],
        "shap_value": shap_values,
    })
    contributions["abs"] = contributions["shap_value"].abs()
    return (contributions.sort_values("abs", ascending=False)
            .head(top_n)
            .drop(columns="abs")
            .reset_index(drop=True))


def blank_template(bundle, n_rows=3):
    """An empty spreadsheet template with the exact columns the app expects."""
    columns = [STUDENT_ID_COLUMN] + required_raw_columns(bundle)
    return pd.DataFrame({c: [""] * n_rows for c in columns})
