import pandas as pd
import pytest

import outcome_model

PREVALENCES = outcome_model.PREVALENCES
NUMERIC_ROWS = 106
TOTAL_ROWS = 110


def outcomes_sheet(repo_root, prevalence):
    import scripts.build_synthetic_outcome_return as builder
    return pd.read_excel(builder.workbook_path(prevalence), sheet_name="OUTCOMES")


@pytest.mark.parametrize("prevalence", PREVALENCES)
def test_workbook_has_all_110_students(repo_root, prevalence):
    assert len(outcomes_sheet(repo_root, prevalence)) == TOTAL_ROWS


@pytest.mark.parametrize("prevalence", PREVALENCES)
def test_exactly_four_rows_are_non_numeric(repo_root, prevalence):
    sheet = outcomes_sheet(repo_root, prevalence)
    numeric = sheet.licensure_outcome.astype(str).isin(["0", "1"])
    assert (~numeric).sum() == 4


@pytest.mark.parametrize("prevalence", PREVALENCES)
def test_realised_prevalence_matches_target(repo_root, prevalence):
    sheet = outcomes_sheet(repo_root, prevalence)
    numeric = sheet[sheet.licensure_outcome.astype(str).isin(["0", "1"])]
    assert len(numeric) == NUMERIC_ROWS
    realised = numeric.licensure_outcome.astype(int).mean()
    assert abs(realised - prevalence) <= 0.02


@pytest.mark.parametrize("prevalence", PREVALENCES)
def test_examining_body_is_ahpc(repo_root, prevalence):
    sheet = outcomes_sheet(repo_root, prevalence)
    assert set(sheet.examining_body.unique()) == {"Allied Health Professions Council"}


def test_sweep_differs_only_in_the_outcome_column(repo_root):
    low = outcomes_sheet(repo_root, 0.25)
    high = outcomes_sheet(repo_root, 0.55)
    other = [c for c in low.columns if c != "licensure_outcome"]
    pd.testing.assert_frame_equal(low[other], high[other])


def test_sweep_is_monotone(repo_root):
    low = outcomes_sheet(repo_root, 0.25)
    high = outcomes_sheet(repo_root, 0.55)
    numeric = low.licensure_outcome.astype(str).isin(["0", "1"])
    lo = low.loc[numeric, "licensure_outcome"].astype(int).to_numpy()
    hi = high.loc[numeric, "licensure_outcome"].astype(int).to_numpy()
    assert (hi[lo == 1] == 1).all()


@pytest.mark.parametrize("prevalence", PREVALENCES)
def test_provenance_sheet_is_first_and_says_not_reportable(repo_root, prevalence):
    import scripts.build_synthetic_outcome_return as builder
    book = pd.ExcelFile(builder.workbook_path(prevalence))
    assert book.sheet_names[0] == "PROVENANCE_read_first"
    provenance = pd.read_excel(book, sheet_name="PROVENANCE_read_first")
    joined = provenance.astype(str).agg(" ".join, axis=1).str.cat(sep=" ")
    assert "reportable as findings" in joined
    assert "NO" in joined


@pytest.mark.parametrize("prevalence", PREVALENCES)
def test_student_ids_match_the_real_dataset(repo_root, prevalence, model_dataset):
    sheet = outcomes_sheet(repo_root, prevalence)
    assert list(sheet.student_id) == list(model_dataset.student_id)
