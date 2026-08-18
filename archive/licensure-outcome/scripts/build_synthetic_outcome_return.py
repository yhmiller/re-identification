#!/usr/bin/env python
"""Generate simulated AHPC licensure outcomes in the registrar's return format.

WHAT THIS IS
------------
The 110 real academic records have no licensure outcome, so `real_data.load()`
raises and the pipeline falls back to pilot data. This produces what the
registrar's completed return would look like, so the pipeline can run end to
end against the real academic distribution.

The outcome is written to a SEPARATE return workbook, never into
data/model_dataset_2021_2022.xlsx. The primary dataset's outcome column stays
empty until the college supplies real results.

WHAT THIS IS NOT
----------------
Evidence. See outcome_model.py. Every outcome here is generated.

Prevalence is swept over outcome_model.PREVALENCES because no AHPC pass rate is
published. One workbook per prevalence, all sharing one seed and one logit, so
they differ in the outcome column and nothing else.

Usage:
    ml_env/bin/python scripts/build_synthetic_outcome_return.py
    ml_env/bin/python scripts/build_synthetic_outcome_return.py --calibrate
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import outcome_model            # noqa: E402
import schema                   # noqa: E402

SOURCE = REPO_ROOT / "data" / "model_dataset_2021_2022.xlsx"
REQUEST = REPO_ROOT / "data" / "OUTCOME_REQUEST_2021_2022.xlsx"
OUTPUT_STEM = "OUTCOME_REQUEST_2021_2022_SYNTHETIC_RETURN"

SEED = 42
EXAMINING_BODY = "Allied Health Professions Council"
EXAM_DIET = {2021: "Nov 2024", 2022: "Nov 2025"}

# Real registrar returns are not all clean passes and fails. Emitting a few
# non-numeric codes makes the artefact faithful to the format and exercises the
# drop-rows-without-outcome path in real_data.load. Set to 0 to disable.
N_NON_NUMERIC = 4
NON_NUMERIC_CODES = ["absent", "withheld", "deferred"]

CALIBRATION_GRID = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5)


def workbook_path(prevalence: float) -> Path:
    return REPO_ROOT / "data" / f"{OUTPUT_STEM}_p{int(round(prevalence * 1000)):03d}.xlsx"


def load_source():
    """The real academic records, engineered, plus the outcome template."""
    model = pd.read_excel(SOURCE, sheet_name="model_data")
    template = pd.read_excel(SOURCE, sheet_name="outcome_template")
    return schema.engineer_features(model), template


def choose_non_numeric_rows(engineered: pd.DataFrame) -> pd.Index:
    """Which rows get a non-numeric code.

    The incomplete record is always one of them: its null CGPA would produce a
    NaN logit, so it cannot receive a generated outcome at all.
    """
    incomplete = engineered.index[~engineered["has_complete_record"].astype(bool)]
    rng = np.random.default_rng(SEED)
    remaining = engineered.index.difference(incomplete)
    needed = max(N_NON_NUMERIC - len(incomplete), 0)
    drawn = rng.choice(remaining, size=needed, replace=False) if needed else []
    return pd.Index(list(incomplete) + list(drawn)).sort_values()


def measure_auc(engineered: pd.DataFrame, outcome: np.ndarray) -> float:
    """Repeated stratified CV, because at n=106 a single 5-fold split's AUC
    swings enough to pass or fail the gate on fold luck alone."""
    features = pd.get_dummies(engineered[schema.ALL_FEATURES_V2], drop_first=True)
    features = features.fillna(features.median(numeric_only=True))
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    return cross_val_score(GradientBoostingClassifier(random_state=SEED),
                           features, outcome, cv=splitter,
                           scoring="roc_auc").mean()


def calibrate(engineered: pd.DataFrame, numeric_rows: pd.Index):
    """Print the noise/AUC table used to set outcome_model.NOISE_SD_REAL_110."""
    subset = engineered.loc[numeric_rows]
    lo, hi = outcome_model.PLAUSIBLE_AUC
    print(f"Calibrating on {len(subset)} rows, target AUC {lo}-{hi}, "
          f"prevalence {outcome_model.PRIMARY_PREVALENCE}")
    print(f"{'noise_sd':>10}  {'AUC-ROC':>8}  verdict")
    for noise_sd in CALIBRATION_GRID:
        logit = outcome_model.risk_logit(subset, SEED, noise_sd)
        outcome = outcome_model.assign_outcome(logit, outcome_model.PRIMARY_PREVALENCE)
        auc = measure_auc(subset, outcome)
        verdict = "within" if lo <= auc <= hi else "OUTSIDE"
        print(f"{noise_sd:>10.2f}  {auc:>8.4f}  {verdict}")
    print("\nSet outcome_model.NOISE_SD_REAL_110 to the value nearest the middle "
          "of the band, then rerun without --calibrate.")


def build_sheets(template: pd.DataFrame, outcome_column: pd.Series,
                 prevalence: float, auc: float) -> dict:
    outcomes = template[["student_id", "cohort_year", "programme",
                         "source_workbook", "source_row_label"]].copy()
    outcomes["licensure_outcome"] = outcome_column.values
    outcomes["exam_diet"] = outcomes.cohort_year.map(EXAM_DIET)
    outcomes["examining_body"] = EXAMINING_BODY
    outcomes["notes"] = ""

    provenance = pd.DataFrame([
        ("record_type", "SYNTHETIC_OUTCOME", "Real academic records, generated outcome."),
        ("academic features", "REAL", "The 110 records supplied by the Accra School of Hygiene. Unmodified."),
        ("licensure_outcome", "GENERATED by outcome_model.generate()", "ASSUMPTION. Not observed."),
        ("examining body", EXAMINING_BODY, "Act 857 (2013), pass mark 60%. NOT the Nursing and Midwifery Council."),
        ("prevalence", f"{prevalence}", "ASSUMPTION. No AHPC pass rate is published: the Council releases pass lists with no candidates-presented denominator. Swept over 0.25/0.40/0.55; 0.40 is primary."),
        ("prevalence, what it is not", "not Amankwaa 26.1%", "That figure is that study's own 176-student two-college sample, and is a nursing figure. The same paper reports ~50% national nursing failure."),
        ("outcome coefficients", "Academic dominant, demographics absent", "The real records carry no demographic columns, so the demographic term is not applied here."),
        ("noise_sd", f"{outcome_model.NOISE_SD_REAL_110}", "Calibrated against these 110 records via --calibrate."),
        ("learnability", f"AUC-ROC {auc:.4f}", f"Repeated 5x5 stratified CV. Gate: must fall in {outcome_model.PLAUSIBLE_AUC}."),
        ("non-numeric rows", f"{N_NON_NUMERIC}", "Coded absent/withheld/deferred, as a real return would contain. Dropped at load."),
        ("H2 testable here", "NO", "No demographic columns exist in the real extraction."),
        ("reportable as findings", "NO", "Outcomes are generated, not observed."),
        ("generated", date.today().isoformat(), "scripts/build_synthetic_outcome_return.py"),
    ], columns=["field", "how it was produced", "note"])

    instructions = pd.read_excel(REQUEST, sheet_name="INSTRUCTIONS")
    banner = pd.DataFrame([
        ("*** SYNTHETIC ***", "This is NOT a registrar return. Every licensure_outcome "
                              "below was generated. See PROVENANCE_read_first."),
        ("", ""),
    ], columns=instructions.columns)

    return {
        "PROVENANCE_read_first": provenance,
        "INSTRUCTIONS": pd.concat([banner, instructions], ignore_index=True),
        "CODES": pd.read_excel(REQUEST, sheet_name="CODES"),
        "OUTCOMES": outcomes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true",
                        help="print the noise/AUC table and exit without writing")
    args = parser.parse_args()

    for path in (SOURCE, REQUEST):
        if not path.exists():
            sys.exit(f"Missing {path}. Run the consolidation scripts first.")

    engineered, template = load_source()
    if list(engineered.student_id) != list(template.student_id):
        sys.exit("model_data and outcome_template disagree on student_id order.")

    coded_rows = choose_non_numeric_rows(engineered)
    numeric_rows = engineered.index.difference(coded_rows)

    if args.calibrate:
        calibrate(engineered, numeric_rows)
        return

    # One logit for the whole sweep, so the workbooks differ only in the
    # threshold applied to it.
    subset = engineered.loc[numeric_rows]
    logit = outcome_model.risk_logit(subset, SEED, outcome_model.NOISE_SD_REAL_110)

    rng = np.random.default_rng(SEED)
    codes = rng.choice(NON_NUMERIC_CODES, size=len(coded_rows))

    lo, hi = outcome_model.PLAUSIBLE_AUC
    built = []
    for prevalence in outcome_model.PREVALENCES:
        outcome = outcome_model.assign_outcome(logit, prevalence)
        auc = measure_auc(subset, outcome)
        verdict = "within" if lo <= auc <= hi else "OUTSIDE"
        print(f"prevalence {prevalence:.2f}  realised {outcome.mean():.3f}  "
              f"AUC-ROC {auc:.4f} ({verdict} {lo}-{hi})")
        if not lo <= auc <= hi:
            sys.exit(f"Generated relationship is {'too strong' if auc > hi else 'too weak'} "
                     f"(AUC {auc:.3f}). Adjust NOISE_SD_REAL_110 via --calibrate and rerun. "
                     "Nothing was written.")

        column = pd.Series(index=engineered.index, dtype=object)
        column.loc[numeric_rows] = [str(v) for v in outcome]
        column.loc[coded_rows] = codes
        built.append((prevalence, column, auc))

    for prevalence, column, auc in built:
        sheets = build_sheets(template, column, prevalence, auc)
        path = workbook_path(prevalence)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=name, index=False)
        print(f"Wrote {path.relative_to(REPO_ROOT)}")

    print(f"\n{len(built)} workbooks | {len(numeric_rows)} trainable rows | "
          f"{len(coded_rows)} coded non-numeric")
    print("Outcomes are GENERATED. Nothing from a run on these is reportable.")


if __name__ == "__main__":
    main()
