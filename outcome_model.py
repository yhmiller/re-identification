"""The generative model behind every simulated licensure outcome in this project.

WHAT THIS IS
------------
One logistic model, imported by both `scripts/build_synthetic_alpha.py` and
`scripts/build_synthetic_outcome_return.py`, so the two synthetic artefacts
cannot drift apart and one citation covers both.

WHAT THIS IS NOT
----------------
Evidence. Every coefficient below is an assumption set by the author. A model
trained on generated outcomes recovers these coefficients, so no number derived
from them belongs in a Results chapter.

Prevalence
----------
No pass rate is published for the Allied Health Professions Council
examination: the Council releases pass lists with no candidates-presented
denominator. The Ghanaian nursing figures that were previously borrowed are
both from a different regulator and, in the case of the 26.1% once hardcoded
here, from a two-college convenience sample rather than a population (Amankwaa
et al. 2015 report 26.1% failing in their own 176-student sample, while the
same paper's introduction gives the national picture as 38.9% of 3,223 passing
in 2011, 51.8% of 2,439 in 2013, and a ~50% average).

Prevalence is therefore swept rather than asserted.
"""
import numpy as np
import pandas as pd

# Swept because no defensible point estimate exists. The low end sits near the
# old 0.261 assumption, the high end past the ~0.50 national nursing failure
# rate — justified because the AHPC pass mark is 60% against the NMC's 50%.
PREVALENCES = (0.25, 0.40, 0.55)
PRIMARY_PREVALENCE = 0.40

# OUTCOME MODEL COEFFICIENTS
# --------------------------
# Assumptions, but shaped by what the licensure-prediction literature reports:
#   - Cumulative academic performance dominates. Consistent across Hannaford et
#     al. (2021) and Oyler et al. (2025), and in the Ghanaian and Malawian
#     descriptive work (Amankwaa et al. 2015, CGPA AOR 15.27, 95% CI 6.28-27.11;
#     Mvula & Msosa 2023). Given the largest coefficient.
#   - Demographics are weak predictors. Amankwaa et al. found sociodemographic
#     characteristics and previous education had no influence at all. Given a
#     deliberately small coefficient, and applied only where the column exists.
#   - Trajectory and consistency carry real but secondary signal.
COEF_CGPA = -1.15           # stronger students fail less
COEF_TREND = -0.45          # improving trajectory protects
COEF_INCONSISTENCY = 0.40   # erratic semesters raise risk
COEF_WEAK_SEMESTERS = 0.35
COEF_FAIL_RATE = 0.30
COEF_DEMOGRAPHIC = 0.12

# The engineered predictors are algebraically related (cgpa, gpa_trend,
# gpa_consistency, n_weak_semesters and fail_rate all derive from the same six
# semester GPAs), so a model reconstructs the generating logit almost exactly
# unless noise dominates it. Both constants below were measured, not guessed.
#
# NOISE_SD_ALPHA, against 330 copula-sampled rows:
#     0.85 -> 0.956   1.5 -> 0.868   2.5 -> 0.784   3.5 -> 0.715   4.5 -> 0.681
NOISE_SD_ALPHA = 2.5

# NOISE_SD_REAL_110, against the 106 trainable real records at prevalence 0.40,
# measured by `build_synthetic_outcome_return.py --calibrate`:
#     0.5 -> 0.9417   1.0 -> 0.9256   1.5 -> 0.8745   2.0 -> 0.8096
#     2.5 -> 0.7528   3.0 -> 0.6711   3.5 -> 0.6720   4.0 -> 0.6282   4.5 -> 0.5645
# 2.5 is used, landing nearest 0.78, mid-band for published licensure models.
#
# That "nearest" call is close, not clean: 2.5 (|0.7528-0.78|=0.0272) beat 2.0
# (|0.8096-0.78|=0.0296) by only 0.0024. The grid itself is non-monotonic
# between 3.0 and 3.5 (0.6711 then 0.6720, a 0.0009 reversal against an
# otherwise decreasing trend), so a single-seed AUC at one grid point carries
# noise on roughly that same order — comparable to the margin that decided
# 2.5 over 2.0. Read the choice between 2.0 and 2.5 as within noise, not as a
# meaningful preference: 1.5, 2.0 and 2.5 all satisfy the plausibility gate,
# and 2.5 is a defensible pick, not a measured optimum.
NOISE_SD_REAL_110 = 2.5

# Published licensure-prediction models cluster in this band. A corpus that
# beats it is not a better corpus, it is an unrealistic one.
PLAUSIBLE_AUC = (0.68, 0.88)

PREDICTORS = ("cgpa", "gpa_trend", "gpa_consistency",
              "n_weak_semesters", "fail_rate")


def _standardise(series: pd.Series) -> pd.Series:
    sd = series.std()
    return (series - series.mean()) / (sd if sd else 1.0)


def risk_logit(engineered: pd.DataFrame, seed: int, noise_sd: float) -> np.ndarray:
    """The latent failure risk for each row, before any prevalence threshold.

    Separated from `assign_outcome` so a prevalence sweep can share one logit
    across every prevalence, making the resulting workbooks differ in the
    outcome column and nothing else.
    """
    missing = [c for c in PREDICTORS if c not in engineered.columns]
    if missing:
        raise KeyError(f"risk_logit needs engineered features, missing: {missing}")

    rng = np.random.default_rng(seed)
    z = {c: _standardise(engineered[c]) for c in PREDICTORS}

    linear = (COEF_CGPA * z["cgpa"]
              + COEF_TREND * z["gpa_trend"]
              + COEF_INCONSISTENCY * z["gpa_consistency"]
              + COEF_WEAK_SEMESTERS * z["n_weak_semesters"]
              + COEF_FAIL_RATE * z["fail_rate"])

    # The real 110 records carry no demographic columns, so this term applies
    # only to the alpha corpus. See the H2 warning in build_synthetic_alpha.py.
    if "gender" in engineered.columns:
        linear = linear + COEF_DEMOGRAPHIC * (engineered["gender"] == "Male").astype(float)

    return (linear + rng.normal(0, noise_sd, len(engineered))).to_numpy()


def assign_outcome(logit: np.ndarray, prevalence: float) -> np.ndarray:
    """Threshold a logit so the realised failure rate matches `prevalence`."""
    if not 0 < prevalence < 1:
        raise ValueError(f"prevalence must be in (0, 1), got {prevalence}")
    threshold = np.quantile(logit, 1 - prevalence)
    return (logit > threshold).astype(int)


def generate(engineered: pd.DataFrame, prevalence: float, seed: int,
             noise_sd: float) -> np.ndarray:
    """Convenience wrapper for callers that need only a single prevalence."""
    return assign_outcome(risk_logit(engineered, seed, noise_sd), prevalence)
