"""Load the college's model-ready dataset for the pipeline.

`data/model_dataset_2021_2022.xlsx` is produced by
scripts/consolidate_real_data.py followed by scripts/build_model_dataset.py.
This module is the single place the pipeline reads it from, so adding a cohort
means re-running those two scripts and nothing else.

Adding cohorts
--------------
Drop the new workbooks into data/<range>/ , point SOURCE_DIR in
scripts/consolidate_real_data.py at them, and re-run both scripts. Student
identifiers stay stable for existing cohorts because they derive from cohort,
programme and source row rather than from row position in the combined file.
"""
from pathlib import Path

import pandas as pd

import schema

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = REPO_ROOT / "data" / "model_dataset_2021_2022.xlsx"
SHEET = "model_data"

# The build script writes the outcome under this name; the pipeline expects
# schema.TARGET. Renaming here keeps the Excel file readable to the registrar.
# Council-neutral: this cohort is examined by the Allied Health Professions
# Council, not the Nursing and Midwifery Council.
OUTCOME_SOURCE_COLUMN = "licensure_fail"


class OutcomeNotSuppliedError(RuntimeError):
    """Raised when the dataset has no licensure outcomes to train against."""


def load(path=None, require_outcome=True):
    """Return the college records as a modelling frame.

    Rows without a licensure outcome are dropped, since a record with no target
    cannot contribute to training or evaluation. The count of dropped rows is
    reported by `summarise` so it can go into the participant flow table.
    """
    path = Path(path) if path else DEFAULT_DATASET
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/consolidate_real_data.py then "
            "scripts/build_model_dataset.py first.")

    df = pd.read_excel(path, sheet_name=SHEET)
    if OUTCOME_SOURCE_COLUMN in df.columns:
        df = df.rename(columns={OUTCOME_SOURCE_COLUMN: schema.TARGET})

    if schema.TARGET not in df.columns:
        raise OutcomeNotSuppliedError(
            f"{path.name} has no '{OUTCOME_SOURCE_COLUMN}' column.")

    supplied = df[schema.TARGET].notna()
    if require_outcome and not supplied.any():
        raise OutcomeNotSuppliedError(
            f"{path.name} has {len(df)} records but no licensure outcomes yet. "
            f"The college must populate '{OUTCOME_SOURCE_COLUMN}' in the "
            "outcome_template sheet. Until then the pipeline runs on pilot data.")

    df = df[supplied].copy()
    df[schema.TARGET] = df[schema.TARGET].astype(int)

    missing = [c for c in schema.NUMERIC_COLS + schema.CATEGORICAL_COLS
               if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    return schema.engineer_features(df)


def summarise(path=None):
    """Counts for the participant flow table, without requiring outcomes."""
    path = Path(path) if path else DEFAULT_DATASET
    df = pd.read_excel(path, sheet_name=SHEET)
    outcome = df.get(OUTCOME_SOURCE_COLUMN)
    supplied = int(outcome.notna().sum()) if outcome is not None else 0
    return {
        "records_total": len(df),
        "outcomes_supplied": supplied,
        "outcomes_missing": len(df) - supplied,
        "cohorts": sorted(df["cohort_year"].dropna().astype(int).unique().tolist()),
        "programmes": sorted(df["programme"].dropna().unique().tolist()),
    }
