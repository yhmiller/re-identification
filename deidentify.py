"""Generalisation, suppression and the derived features computed over them.

Supports Phase 3 of the re-identification study, where the question is whether
features derived from a protected variable undo the protection.

The central object is a band. Generalising a GPA of 2.87 to a band width of 0.5
replaces it with the interval [2.5, 3.0), which is what a recipient of the
de-identified file actually sees. Every attack in `attack_models` starts from
those intervals and tries to narrow them back down.

`derive_features` is deliberately the single definition of the engineered
feature set, used three times over: once on the original values to reproduce
what the pipeline already publishes, once by the attacker to check consistency,
and once on generalised data for Experiment C. Having one function rather than
three keeps the comparison honest, because a difference in results cannot come
from a difference in how the features were computed.
"""

import numpy as np
import pandas as pd

WEAK_SEMESTER_THRESHOLD = 2.0


def band_edges(values, width):
    """Lower edge of the band each value falls into."""
    return np.floor(np.asarray(values, dtype=float) / width) * width


def generalise(frame, columns, width):
    """Replace each value with the lower edge of its band.

    Returns a copy. Missing values stay missing: a gap is not a band, and
    filling it here would invent data the recipient never received.
    """
    out = frame.copy()
    for col in columns:
        out[col] = np.where(out[col].notna(), band_edges(out[col], width), np.nan)
    return out


def band_intervals(frame, columns, width):
    """The (low, high) interval a recipient can infer for each generalised value.

    This is what the attacker starts from, so it is derived from the published
    band edge rather than from the original value.
    """
    low, high = {}, {}
    for col in columns:
        edge = band_edges(frame[col], width)
        low[col] = np.where(frame[col].notna(), edge, np.nan)
        high[col] = np.where(frame[col].notna(), edge + width, np.nan)
    return pd.DataFrame(low, index=frame.index), pd.DataFrame(high, index=frame.index)


def suppress(frame, columns, min_class_size, width=None):
    """Blank out records whose equivalence class is smaller than `min_class_size`.

    Suppression is the second lever after generalisation and is what makes a
    k-anonymity target reachable when banding alone cannot get there.
    """
    working = generalise(frame, columns, width) if width else frame
    sizes = working.groupby(list(columns), dropna=False)[columns[0]].transform("size")
    out = frame.copy()
    out.loc[sizes < min_class_size, list(columns)] = np.nan
    return out, int((sizes < min_class_size).sum())


def derive_features(frame, semester_cols, prefix=""):
    """The engineered feature set, computed from whatever values it is given.

    Mirrors the definitions in `schema.ENGINEERED_NUMERIC`. Computed with
    skipna so that students observed for fewer semesters still get features,
    which matters for the nursing stratum where intakes are observed at
    different levels.
    """
    sem = frame[semester_cols].astype(float)
    out = pd.DataFrame(index=frame.index)

    out[f"{prefix}gpa_min"] = sem.min(axis=1)
    out[f"{prefix}gpa_max"] = sem.max(axis=1)
    out[f"{prefix}gpa_mean"] = sem.mean(axis=1)
    # Population standard deviation, so a student with one observed semester
    # gets 0 rather than NaN.
    out[f"{prefix}gpa_consistency"] = sem.std(axis=1, ddof=0)

    half = len(semester_cols) // 2
    out[f"{prefix}gpa_first_half"] = sem[semester_cols[:half]].mean(axis=1)
    out[f"{prefix}gpa_final_half"] = sem[semester_cols[half:]].mean(axis=1)
    out[f"{prefix}gpa_trend"] = (
        out[f"{prefix}gpa_final_half"] - out[f"{prefix}gpa_first_half"]
    )
    out[f"{prefix}n_weak_semesters"] = (sem < WEAK_SEMESTER_THRESHOLD).sum(axis=1)
    return out


DERIVED_COLUMNS = [
    "gpa_min",
    "gpa_max",
    "gpa_mean",
    "gpa_consistency",
    "gpa_first_half",
    "gpa_final_half",
    "gpa_trend",
    "n_weak_semesters",
]

# The subset that constrains the raw values arithmetically. `n_weak_semesters`
# is a count and `gpa_consistency` is non-linear, so neither enters the interval
# propagation in `attack_models`; they are still published and still contribute
# to uniqueness, which is Experiment A's concern rather than Experiment B's.
CONSTRAINING_COLUMNS = [
    "gpa_min",
    "gpa_max",
    "gpa_mean",
    "gpa_first_half",
    "gpa_final_half",
]
