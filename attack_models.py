"""Attacks against generalised student records.

Experiment B of Phase 3. The question is narrow and answerable: if a custodian
generalises the semester GPAs to protect students, then publishes the derived
features alongside at full precision, can the original values be recovered?

The attacker is assumed to hold only what the custodian released, which is the
band each GPA falls into and the exact value of every derived feature. No side
knowledge, no auxiliary dataset. That makes this a lower bound on what a real
adversary could do, and it means any recovery it achieves is unarguable.

The method is interval propagation. Each generalised GPA starts as the interval
its band defines. The derived features then act as constraints that narrow those
intervals, applied repeatedly until nothing moves:

    minimum     every GPA is at least gpa_min, and if only one interval can
                contain gpa_min then that semester is pinned to it exactly
    maximum     the same argument from above
    means       gpa_mean, gpa_first_half and gpa_final_half each fix a sum, so
                any one value is bounded by that sum minus the others

An interval that collapses to a point is an exact recovery: the custodian
published a band, and the attacker recovered the number inside it.
"""

import numpy as np
import pandas as pd

EXACT_TOLERANCE = 1e-6
MAX_ITERATIONS = 50


def _narrow_to_sum(low, high, members, target_sum):
    """Tighten each member's interval given that the members sum to `target_sum`.

    Standard interval arithmetic: a value can be no smaller than the target
    minus everything the others could contribute at their maximum, and no
    larger than the target minus their minimum.
    """
    if len(members) == 0 or not np.isfinite(target_sum):
        return
    lo_sum, hi_sum = low[members].sum(), high[members].sum()
    for i in members:
        low[i] = max(low[i], target_sum - (hi_sum - high[i]))
        high[i] = min(high[i], target_sum - (lo_sum - low[i]))


def _pin_extreme(low, high, active, value, is_min):
    """If only one interval can hold the extreme, that semester is pinned to it."""
    if not np.isfinite(value):
        return
    for i in active:
        if is_min:
            low[i] = max(low[i], value)
        else:
            high[i] = min(high[i], value)

    candidates = [
        i
        for i in active
        if low[i] - EXACT_TOLERANCE <= value <= high[i] + EXACT_TOLERANCE
    ]
    if len(candidates) == 1:
        i = candidates[0]
        low[i] = high[i] = value


def reconstruct_row(band_low, band_high, derived, n_semesters):
    """Narrow one student's intervals as far as the derived features allow."""
    low = np.array(band_low, dtype=float)
    high = np.array(band_high, dtype=float)
    active = [i for i in range(n_semesters) if np.isfinite(low[i])]
    if not active:
        return low, high

    half = n_semesters // 2
    first = [i for i in active if i < half]
    final = [i for i in active if i >= half]

    for _ in range(MAX_ITERATIONS):
        before = (low.copy(), high.copy())

        _pin_extreme(low, high, active, derived.get("gpa_min", np.nan), is_min=True)
        _pin_extreme(low, high, active, derived.get("gpa_max", np.nan), is_min=False)

        for members, key in (
            (active, "gpa_mean"),
            (first, "gpa_first_half"),
            (final, "gpa_final_half"),
        ):
            value = derived.get(key, np.nan)
            if members and np.isfinite(value):
                _narrow_to_sum(low, high, members, value * len(members))

        if np.allclose(before[0], low, equal_nan=True) and np.allclose(
            before[1], high, equal_nan=True
        ):
            break

    return low, high


def reconstruct(frame, semester_cols, band_low, band_high, derived_frame):
    """Run the attack across every record.

    `band_low` and `band_high` are what the recipient sees. `derived_frame`
    carries the published derived features at full precision. Returns the
    narrowed intervals plus, for each record, how many semesters were recovered
    exactly.
    """
    n = len(semester_cols)
    lows, highs, exact_counts, observed_counts = [], [], [], []

    for idx in frame.index:
        lo, hi = reconstruct_row(
            band_low.loc[idx, semester_cols].to_numpy(dtype=float),
            band_high.loc[idx, semester_cols].to_numpy(dtype=float),
            derived_frame.loc[idx].to_dict(),
            n,
        )
        lows.append(lo)
        highs.append(hi)
        observed = np.isfinite(lo)
        observed_counts.append(int(observed.sum()))
        exact_counts.append(int(((hi - lo) <= EXACT_TOLERANCE)[observed].sum()))

    low_df = pd.DataFrame(lows, index=frame.index, columns=semester_cols)
    high_df = pd.DataFrame(highs, index=frame.index, columns=semester_cols)
    summary = pd.DataFrame(
        {
            "semesters_observed": observed_counts,
            "semesters_recovered_exactly": exact_counts,
        },
        index=frame.index,
    )
    return low_df, high_df, summary


def recovery_metrics(low_df, high_df, summary, band_width, truth=None):
    """Aggregate what the attack achieved. Never returns anything record-level."""
    widths = (high_df - low_df).to_numpy(dtype=float)
    observed = np.isfinite(widths)
    widths = widths[observed]

    total_observed = int(summary["semesters_observed"].sum())
    total_exact = int(summary["semesters_recovered_exactly"].sum())

    metrics = {
        "band_width": band_width,
        "values_observed": total_observed,
        "mean_interval_width_before": band_width,
        "mean_interval_width_after": float(np.mean(widths)) if widths.size else np.nan,
        "median_interval_width_after": (
            float(np.median(widths)) if widths.size else np.nan
        ),
        "narrowing_ratio": (
            float(np.mean(widths) / band_width) if widths.size else np.nan
        ),
        "values_recovered_exactly": total_exact,
        "prop_values_recovered_exactly": (
            total_exact / total_observed if total_observed else np.nan
        ),
        "prop_students_fully_recovered": float(
            (
                summary["semesters_recovered_exactly"] == summary["semesters_observed"]
            ).mean()
        ),
        "prop_students_partially_recovered": float(
            (summary["semesters_recovered_exactly"] > 0).mean()
        ),
    }

    # Sanity check rather than a result: a correct attack must never exclude the
    # true value from its own interval. Reported so a reviewer can see it held.
    if truth is not None:
        t = truth.to_numpy(dtype=float)
        inside = (t >= low_df.to_numpy() - EXACT_TOLERANCE) & (
            t <= high_df.to_numpy() + EXACT_TOLERANCE
        )
        valid = np.isfinite(t) & np.isfinite(low_df.to_numpy())
        metrics["containment_check"] = float(inside[valid].mean())

    return metrics


def point_estimates(low_df, high_df):
    """Midpoint of each recovered interval, for re-running uniqueness on it.

    The midpoint is the attacker's best single guess. Where an interval has
    collapsed the midpoint is the exact original value, so uniqueness computed
    on this frame is what the attacker can actually achieve.
    """
    return (low_df + high_df) / 2.0


# Linkage under simulated side knowledge. The attacker knows the target, knows
# roughly which group they were in, and remembers a grade imprecisely, so
# recollection is degraded by a tolerance rather than matched exactly: nobody
# remembers a classmate scored 2.87, they remember "around a 3".
#
# No real individual is identified. Targets are dataset rows, the attacker's
# knowledge is derived from those rows, and every metric returned is an
# aggregate over all targets.


def _band_width_for(column, release_width):
    """Band width applied to one column. A dict lets a release generalise some
    columns and leave others at full precision, which is exactly the leaky
    configuration Experiment B identified."""
    if isinstance(release_width, dict):
        return release_width.get(column, 0.0)
    return release_width


def _candidate_mask(released, remembered, spec, release_width=0.0):
    """Records consistent with what the attacker remembers.

    `remembered` comes from the true record, because an attacker recalls what
    they saw rather than what the custodian later published. `released` is what
    the custodian published. When the release is generalised, each released
    value is a band edge standing for the interval [edge, edge + width), so a
    match is an overlap between the attacker's recalled interval and that band,
    not a distance between two points.
    """
    mask = np.ones(len(released), dtype=bool)

    for column, tolerance in spec:
        truth = remembered[column]
        values = released[column]

        if tolerance is None:
            mask &= (values == truth).to_numpy()
        elif pd.isna(truth):
            # Knowing a value is absent is itself a clue.
            mask &= values.isna().to_numpy()
        else:
            recalled_low, recalled_high = truth - tolerance, truth + tolerance
            width = _band_width_for(column, release_width)
            band_low = values
            band_high = values + width
            overlaps = (band_low <= recalled_high + EXACT_TOLERANCE) & \
                       (band_high >= recalled_low - EXACT_TOLERANCE)
            mask &= overlaps.fillna(False).to_numpy()

    return mask


def linkage_attack(released, spec, label, truth=None, release_width=0.0,
                   sample_index=None):
    """Try to isolate each target from the released file using `spec`.

    `spec` is a list of (column, tolerance) pairs. A tolerance of None means the
    attacker recalls the value exactly, which is realistic for a categorical
    fact such as programme. A numeric tolerance means recall to within that
    margin.

    `truth` is the un-generalised frame the attacker's memory is drawn from. It
    defaults to `released`, which is only correct when the release is itself
    un-generalised.

    Returns aggregate metrics only. `prop_target_in_candidate_set` is a genuine
    result rather than a sanity check once the release is generalised: imprecise
    recall can exclude the right person, and an attacker who has excluded their
    target cannot identify them however small the candidate set becomes.
    """
    truth = released if truth is None else truth
    targets = truth if sample_index is None else truth.loc[sample_index]
    positions = {idx: i for i, idx in enumerate(released.index)}

    sizes, unique, correct, contained = [], 0, 0, 0

    for idx, remembered in targets.iterrows():
        mask = _candidate_mask(released, remembered, spec, release_width)
        size = int(mask.sum())
        sizes.append(size)

        own = positions[idx]
        if mask[own]:
            contained += 1
            if size == 1:
                correct += 1
        if size == 1:
            unique += 1

    sizes = np.asarray(sizes, dtype=float)
    n = len(sizes)
    return {
        "attack": label,
        "n_targets": n,
        "n_released": len(released),
        "mean_candidate_set": float(sizes.mean()),
        "median_candidate_set": float(np.median(sizes)),
        "prop_uniquely_isolated": unique / n,
        "prop_correctly_isolated": correct / n,
        "prop_narrowed_below_5": float((sizes < 5).mean()),
        "prop_target_in_candidate_set": contained / n,
    }


def missingness_signature(frame, columns):
    """Which of `columns` a record has, as a single string.

    Absence carries information. In the nursing stratum the positions a student
    occupies are a function of intake year, so the pattern of what is missing
    may identify without any value being read at all.
    """
    present = frame[columns].notna()
    return present.astype(int).astype(str).agg("".join, axis=1)
