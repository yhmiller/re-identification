"""Disclosure risk measurement for student academic records.

Phase 1 and Phase 2 of the re-identification study. Everything here answers one
question: given a set of quasi-identifiers, how many students can be picked out
of the dataset, and which attributes are doing the picking.

Two design decisions worth stating up front, because they drive the numbers.

**Sampling fraction.** Most published re-identification work measures a sample
and must estimate population uniqueness from it, which introduces an assumption
that weakens the result. The Accra School of Hygiene extract is every student in
the 2021 and 2022 EH/OHS/OT cohorts, so the population *is* the dataset and the
sampling fraction is 1. Sample uniqueness therefore equals population uniqueness
exactly, and prosecutor and journalist risk coincide. `sampling_fraction` is
still a parameter so the assumption is visible and testable rather than buried.

**Precision is a risk control, not a formatting choice.** A GPA carried to two
decimals is far more identifying than one carried to one decimal. `precision`
rounds numeric quasi-identifiers before grouping, which is the crudest form of
generalisation and the baseline every later de-identification strategy is
measured against.

No function here returns record-level output. Everything is aggregate by
construction, so no re-identified student can leave this module.
"""

from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

# Equivalence classes at or below this size are conventionally treated as high
# risk in health data disclosure control. Reported, never enforced silently.
K_THRESHOLDS = (2, 3, 5, 10)


@dataclass
class RiskProfile:
    """Aggregate disclosure risk for one quasi-identifier set."""

    n_records: int
    n_quasi_identifiers: int
    quasi_identifiers: list
    precision: object
    sampling_fraction: float

    n_unique: int
    prop_unique: float
    n_equivalence_classes: int
    min_class_size: int
    median_class_size: float
    mean_class_size: float
    share_below_k: dict

    prosecutor_risk: float
    journalist_risk: float
    marketer_risk: float

    min_l_diversity: int = None
    max_t_closeness: float = None
    sensitive_attribute: str = None

    def as_row(self, label=None):
        row = asdict(self)
        row["quasi_identifiers"] = ", ".join(self.quasi_identifiers)
        row["share_below_k"] = "; ".join(
            f"k<{k}:{v:.3f}" for k, v in self.share_below_k.items()
        )
        if label is not None:
            row = {"label": label, **row}
        return row


def _prepare(frame, quasi_identifiers, precision=None):
    """Return just the quasi-identifier columns, numerics rounded to `precision`.

    Missing values are filled with a sentinel so that two records sharing a gap
    group together. Treating NaN as its own value would silently split classes
    and understate risk.
    """
    missing = [c for c in quasi_identifiers if c not in frame.columns]
    if missing:
        raise KeyError(f"quasi-identifiers not in frame: {missing}")

    qi = frame[list(quasi_identifiers)].copy()
    if precision is not None:
        for col in qi.select_dtypes(include=[np.number]).columns:
            qi[col] = qi[col].round(precision)
    for col in qi.columns:
        qi[col] = qi[col].astype(object).where(qi[col].notna(), "__MISSING__")
    return qi


def equivalence_class_sizes(frame, quasi_identifiers, precision=None):
    """Size of the equivalence class each record falls into, aligned to `frame`."""
    qi = _prepare(frame, quasi_identifiers, precision)
    return qi.groupby(list(qi.columns), dropna=False)[qi.columns[0]].transform("size")


def l_diversity(frame, quasi_identifiers, sensitive, precision=None):
    """Smallest number of distinct sensitive values in any equivalence class.

    l = 1 means at least one class is homogeneous: an adversary who narrows a
    target to that class learns the sensitive value without needing to identify
    the individual record. This is the hole k-anonymity alone does not close.
    """
    qi = _prepare(frame, quasi_identifiers, precision)
    joined = qi.assign(_sensitive=frame[sensitive].values)
    return int(
        joined.groupby(list(qi.columns), dropna=False)["_sensitive"].nunique().min()
    )


def t_closeness(frame, quasi_identifiers, sensitive, precision=None):
    """Largest total variation distance between a class and the overall distribution.

    Total variation rather than Earth Mover's, because the sensitive attributes
    in this study are categorical or binary and have no meaningful ordering.
    Higher means some class is more skewed than the dataset as a whole, so
    membership of that class is itself informative.
    """
    qi = _prepare(frame, quasi_identifiers, precision)
    joined = qi.assign(_sensitive=frame[sensitive].values)
    overall = joined["_sensitive"].value_counts(normalize=True)

    worst = 0.0
    for _, group in joined.groupby(list(qi.columns), dropna=False):
        local = group["_sensitive"].value_counts(normalize=True)
        aligned = local.reindex(overall.index, fill_value=0.0)
        worst = max(worst, 0.5 * float(np.abs(aligned - overall).sum()))
    return worst


def risk_profile(
    frame,
    quasi_identifiers,
    sensitive=None,
    precision=None,
    sampling_fraction=1.0,
):
    """Full aggregate risk profile for one quasi-identifier set.

    The three adversary models differ only in what the attacker is assumed to
    know and to want:

    - prosecutor: knows the target is in the dataset, so risk is driven by the
      smallest class and equals 1 / min_k.
    - journalist: does not know who is in the dataset, so risk is scaled by the
      sampling fraction. At a fraction of 1 it collapses onto prosecutor risk,
      which is the situation here.
    - marketer: wants many re-identifications and tolerates errors, so risk is
      the mean of 1/k across records.
    """
    qi = _prepare(frame, quasi_identifiers, precision)
    sizes = qi.groupby(list(qi.columns), dropna=False)[qi.columns[0]].transform("size")
    n_classes = qi.groupby(list(qi.columns), dropna=False).ngroups

    prosecutor = 1.0 / int(sizes.min())
    return RiskProfile(
        n_records=len(frame),
        n_quasi_identifiers=len(quasi_identifiers),
        quasi_identifiers=list(quasi_identifiers),
        precision=precision,
        sampling_fraction=sampling_fraction,
        n_unique=int((sizes == 1).sum()),
        prop_unique=float((sizes == 1).mean()),
        n_equivalence_classes=int(n_classes),
        min_class_size=int(sizes.min()),
        median_class_size=float(sizes.median()),
        mean_class_size=float(sizes.mean()),
        share_below_k={k: float((sizes < k).mean()) for k in K_THRESHOLDS},
        prosecutor_risk=prosecutor,
        journalist_risk=prosecutor * sampling_fraction,
        marketer_risk=float((1.0 / sizes).mean()),
        min_l_diversity=(
            l_diversity(frame, quasi_identifiers, sensitive, precision)
            if sensitive
            else None
        ),
        max_t_closeness=(
            t_closeness(frame, quasi_identifiers, sensitive, precision)
            if sensitive
            else None
        ),
        sensitive_attribute=sensitive,
    )


def profile_table(
    frame, named_sets, sensitive=None, precision=None, sampling_fraction=1.0
):
    """Risk profiles for several named quasi-identifier sets, as one table.

    `named_sets` maps a label to a list of columns. Sets referencing columns the
    frame does not have are skipped rather than raising, so the same comparison
    table can run across datasets with different schemas.
    """
    rows = []
    for label, qi in named_sets.items():
        present = [c for c in qi if c in frame.columns]
        if not present:
            continue
        rows.append(
            risk_profile(
                frame, present, sensitive, precision, sampling_fraction
            ).as_row(label)
        )
    return pd.DataFrame(rows)


def precision_sweep(
    frame, quasi_identifiers, precisions=(3, 2, 1, 0), sampling_fraction=1.0
):
    """How risk falls as numeric quasi-identifiers are rounded more coarsely.

    This is the simplest possible generalisation strategy and forms the floor
    that Mondrian and the optimisation arms in Phase 5 must beat.
    """
    rows = []
    for p in precisions:
        rows.append(
            risk_profile(
                frame,
                quasi_identifiers,
                precision=p,
                sampling_fraction=sampling_fraction,
            ).as_row(f"precision={p}")
        )
    return pd.DataFrame(rows)


def solo_identifying_power(frame, candidates, precision=None):
    """Uniqueness each attribute achieves on its own, ranked.

    Leave-one-out attribution is useless on this data: uniqueness saturates at
    100% with two attributes, so dropping any single one of a large set changes
    nothing and every marginal contribution comes back as zero. That saturation
    is itself the finding, but it means attribution has to be measured from
    below rather than from above. This ranks each attribute by what it achieves
    alone, which stays informative no matter how redundant the full set is.
    """
    rows = []
    for col in candidates:
        if col not in frame.columns:
            continue
        sizes = equivalence_class_sizes(frame, [col], precision)
        rows.append(
            {
                "attribute": col,
                "n_distinct_values": int(frame[col].nunique(dropna=False)),
                "prop_unique_alone": float((sizes == 1).mean()),
                "min_class_size_alone": int(sizes.min()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("prop_unique_alone", ascending=False)
        .reset_index(drop=True)
    )


def saturation_curve(frame, candidates, precision=None, max_size=6):
    """Uniqueness as attributes are added greedily, most identifying first.

    Answers the examiner's question directly: how few facts does somebody need
    before every student is singled out. Greedy rather than exhaustive because
    the exhaustive search over 20-odd candidates is combinatorially pointless
    once the curve saturates in the first two or three steps.
    """
    candidates = [c for c in candidates if c in frame.columns]
    chosen, rows = [], []

    for step in range(1, min(max_size, len(candidates)) + 1):
        best, best_prop = None, -1.0
        for col in candidates:
            if col in chosen:
                continue
            prop = float(
                (equivalence_class_sizes(frame, chosen + [col], precision) == 1).mean()
            )
            if prop > best_prop:
                best, best_prop = col, prop
        chosen.append(best)
        sizes = equivalence_class_sizes(frame, chosen, precision)
        rows.append(
            {
                "n_attributes": step,
                "added": best,
                "attributes": ", ".join(chosen),
                "prop_unique": best_prop,
                "min_class_size": int(sizes.min()),
            }
        )
        if best_prop >= 1.0:
            break
    return pd.DataFrame(rows)


def minimum_attribute_sets(frame, candidates, precision=None, max_size=3, target=1.0):
    """Smallest attribute combinations that reach `target` uniqueness.

    Answers the question an examiner will ask directly: how few things does
    somebody need to know about a student before they can find that student's
    row. Searches combinations up to `max_size`, stopping at the first size
    that reaches the target so the search stays tractable.
    """
    from itertools import combinations

    candidates = [c for c in candidates if c in frame.columns]
    found = []
    for size in range(1, max_size + 1):
        for combo in combinations(candidates, size):
            prop = float(
                (equivalence_class_sizes(frame, list(combo), precision) == 1).mean()
            )
            if prop >= target:
                found.append(
                    {
                        "n_attributes": size,
                        "attributes": ", ".join(combo),
                        "prop_unique": prop,
                    }
                )
        if found:
            break
    return pd.DataFrame(found)


def group_size_effect(frame, quasi_identifiers, group_cols, precision=None):
    """Risk within each natural group, against that group's size.

    Cohort size is a research variable in this study, not a nuisance. Small
    programmes should show higher within-group uniqueness than large ones, and
    this is where that gets measured rather than asserted.
    """
    rows = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        inner = [c for c in quasi_identifiers if c not in group_cols]
        if not inner or len(group) < 2:
            continue
        sizes = equivalence_class_sizes(group, inner, precision)
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "group_size": len(group),
                "prop_unique_within_group": float((sizes == 1).mean()),
                "min_class_size": int(sizes.min()),
            }
        )
    return pd.DataFrame(rows).sort_values("group_size").reset_index(drop=True)
