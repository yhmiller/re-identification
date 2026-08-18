import numpy as np
import pandas as pd
import pytest

import outcome_model


def make_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    cgpa = rng.normal(2.6, 0.67, n)
    return pd.DataFrame({
        "cgpa": cgpa,
        "gpa_trend": rng.normal(0.19, 0.61, n),
        "gpa_consistency": rng.normal(0.58, 0.27, n),
        "n_weak_semesters": rng.integers(0, 7, n),
        "fail_rate": rng.uniform(0, 0.46, n),
    })


def test_assign_outcome_hits_requested_prevalence():
    logit = np.linspace(-3, 3, 1000)
    outcome = outcome_model.assign_outcome(logit, 0.40)
    assert abs(outcome.mean() - 0.40) < 0.01


def test_assign_outcome_returns_int_zero_one():
    logit = np.linspace(-3, 3, 100)
    outcome = outcome_model.assign_outcome(logit, 0.40)
    assert set(np.unique(outcome)) <= {0, 1}


def test_risk_logit_is_deterministic_for_a_seed():
    frame = make_frame()
    first = outcome_model.risk_logit(frame, seed=42, noise_sd=2.5)
    second = outcome_model.risk_logit(frame, seed=42, noise_sd=2.5)
    np.testing.assert_array_equal(first, second)


def test_higher_cgpa_lowers_risk_when_noise_is_absent():
    frame = make_frame()
    logit = outcome_model.risk_logit(frame, seed=42, noise_sd=0.0)
    corr = np.corrcoef(frame.cgpa, logit)[0, 1]
    assert corr < -0.5


def test_sweep_shares_one_logit():
    frame = make_frame()
    logit = outcome_model.risk_logit(frame, seed=42, noise_sd=2.5)
    low = outcome_model.assign_outcome(logit, 0.25)
    high = outcome_model.assign_outcome(logit, 0.55)
    # Every row failing at the low prevalence must also fail at the high one.
    assert np.all(high[low == 1] == 1)


def test_demographic_term_applies_only_when_column_present():
    frame = make_frame()
    without = outcome_model.risk_logit(frame, seed=42, noise_sd=0.0)
    frame_with = frame.assign(gender="Male")
    with_male = outcome_model.risk_logit(frame_with, seed=42, noise_sd=0.0)
    assert np.allclose(with_male - without, outcome_model.COEF_DEMOGRAPHIC)


def test_prevalences_include_the_primary():
    assert outcome_model.PRIMARY_PREVALENCE in outcome_model.PREVALENCES
