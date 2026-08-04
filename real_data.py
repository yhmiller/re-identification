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


class OutcomeJoinError(RuntimeError):
    """Raised when a supplied outcome return does not cover every record.

    A silent partial join would attach outcomes to the wrong students and
    quietly corrupt every downstream number, so it must fail loudly.
    """


OUTCOME_RETURN_SHEET = "OUTCOMES"
OUTCOME_RETURN_COLUMN = "licensure_outcome"
PROVENANCE_SHEET = "PROVENANCE_read_first"
REGISTRAR_PROVENANCE = "REGISTRAR_RETURN"

# Codes the registrar may return instead of a pass or fail. Each means "no
# first-attempt result", so the row cannot train or evaluate anything.
NON_NUMERIC_OUTCOMES = frozenset({"absent", "deferred", "withheld", "unknown"})


def _read_provenance(path):
    """Return (provenance, prevalence) for a registrar-format return workbook.

    A simulated return carries a provenance sheet recording how it was
    generated. The registrar's genuine return has no such sheet because there
    is nothing to record, so its absence identifies a real return rather than a
    malformed one, and its prevalence is unknown until the outcomes are read.
    """
    with pd.ExcelFile(path) as workbook:
        if PROVENANCE_SHEET not in workbook.sheet_names:
            return REGISTRAR_PROVENANCE, float("nan")
        provenance = pd.read_excel(workbook, sheet_name=PROVENANCE_SHEET)
    fields = provenance.set_index("field")["how it was produced"]
    return (fields.get("record_type", "UNKNOWN"),
            float(fields.get("prevalence", "nan")))


def _apply_outcome_return(df, outcomes_from):
    """Fill the outcome column from a registrar-format return workbook.

    Kept separate from `load` so the same join serves a genuine college return
    and a simulated one; only the path differs.
    """
    path = Path(outcomes_from)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")

    returned = pd.read_excel(path, sheet_name=OUTCOME_RETURN_SHEET)
    matched = df["student_id"].isin(returned["student_id"])
    if not matched.all():
        raise OutcomeJoinError(
            f"{path.name} covers {int(matched.sum())} of {len(df)} students. "
            "Every record must be present; a partial return would attach "
            "outcomes to the wrong students.")

    # A repeated row is a realistic registrar mistake, and there is no way to
    # tell which of the two outcomes is the right one, so it must fail here
    # rather than let pandas raise an indexing error out of the join below.
    repeated = returned["student_id"][returned["student_id"].duplicated()].unique()
    if len(repeated):
        raise OutcomeJoinError(
            f"{path.name} repeats {len(repeated)} student_id(s): "
            f"{sorted(repeated.tolist())}. Each student must appear exactly "
            "once; a repeated row leaves the correct outcome ambiguous.")

    lookup = returned.set_index("student_id")[OUTCOME_RETURN_COLUMN]
    raw = df["student_id"].map(lookup).astype(str).str.strip().str.lower()
    df[OUTCOME_SOURCE_COLUMN] = pd.to_numeric(
        raw.where(~raw.isin(NON_NUMERIC_OUTCOMES)), errors="coerce")

    provenance, prevalence = _read_provenance(path)
    df.attrs["outcome_provenance"] = provenance
    df.attrs["outcome_prevalence"] = prevalence
    return df


def load(path=None, require_outcome=True, outcomes_from=None):
    """Return the college records as a modelling frame.

    Rows without a licensure outcome are dropped, since a record with no target
    cannot contribute to training or evaluation. The count of dropped rows is
    reported by `summarise` so it can go into the participant flow table.

    `outcomes_from` supplies the outcome from a separate registrar-format
    return workbook rather than from the dataset itself. Pass the college's
    genuine return here when it arrives; until then a simulated return from
    scripts/build_synthetic_outcome_return.py exercises the same path.
    """
    path = Path(path) if path else DEFAULT_DATASET
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/consolidate_real_data.py then "
            "scripts/build_model_dataset.py first.")

    df = pd.read_excel(path, sheet_name=SHEET)
    if outcomes_from:
        df = _apply_outcome_return(df, outcomes_from)
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

    engineered = schema.engineer_features(df)
    engineered.attrs.update(df.attrs)
    return engineered


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
