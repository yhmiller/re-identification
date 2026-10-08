"""Diagnostic. Where each corpus's outcome position sits against its predictors.

The sensitive attribute is weak performance in the last observed position of a
record's sequence, chosen per record. The predictor set is a fixed prefix of
that sequence. The two separate only when every record is observed past the
prefix, which holds in allied health and in the public corpus. It fails in
nursing, where most modelled records stop at the second position, and that
position is released as a predictor, so for those records the label is a
threshold on a column the recipient already holds.

Writes three tables to `results/disclosure/`:

    leak_audit_positions.csv        where the outcome position falls, per corpus
    leak_audit_arm_comparison.csv   the nursing arm comparison on the full
                                    modelled sample against the leak-free subset
    leak_audit_group_composition.csv  which groups survive into the leak-free
                                      subset, since dropping records by sequence
                                      length is not neutral across intakes

Deliberately absent from `run_all.STAGES`. It measures the cost of a correction
that has not been approved, so it writes nothing that any reported table reads.

Run:
    ml_env/bin/python notebooks/14_outcome_leak_diagnostic.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import derivation_consistent as dc  # noqa: E402
import risk_utility as ru  # noqa: E402
import stats_validation as sv  # noqa: E402
import strata as st  # noqa: E402

RESULTS = ROOT / "results" / "disclosure"

ARMS = (dc.BASELINE, dc.PROPOSED)


def last_observed_position(frame, sequence):
    """Each record's own last observed sequence position, one-based.

    Matches `strata._weak_final`, which forward-fills before taking the final
    column, so this is the position the label is cut from.
    """
    observed = frame[sequence].notna()
    return observed.apply(
        lambda row: max([i + 1 for i, seen in enumerate(row) if seen], default=0),
        axis=1,
    )


def leak_free_index(stratum):
    """Records whose outcome position falls beyond the predictor prefix."""
    position = last_observed_position(stratum.frame, stratum.sequence)
    return position[position > len(stratum.predictors)].index


def modelled_index(stratum, band_width, mode=dc.BASELINE):
    """The records that survive listwise deletion into the utility model.

    The arm does not change which records survive, since both arms derive over
    the same predictor positions, so one arm is enough to establish the sample.
    """
    features, _, _ = ru.release_matrix(
        stratum.frame, stratum.sequence, stratum.predictors, st.SENSITIVE,
        band_width=band_width, derived_mode=mode,
    )
    return features.index


def position_audit(strata):
    """How many modelled records in each corpus can read their own label."""
    rows = []
    for name, stratum in strata.items():
        modelled = modelled_index(stratum, stratum.band_widths[1])
        position = last_observed_position(stratum.frame, stratum.sequence).loc[modelled]
        inside = position <= len(stratum.predictors)
        labels = stratum.frame.loc[modelled, st.SENSITIVE]
        rows.append({
            "stratum": name,
            "n_sequence_positions": len(stratum.sequence),
            "n_predictor_positions": len(stratum.predictors),
            "n_corpus": stratum.n,
            "n_modelled": len(modelled),
            "n_outcome_inside_predictors": int(inside.sum()),
            "proportion_inside": float(inside.mean()),
            "n_leak_free": int((~inside).sum()),
            "prevalence_modelled": float(labels.mean()),
            "prevalence_leak_free": (float(labels[~inside.values].mean())
                                     if (~inside).any() else np.nan),
            "last_position_counts": str(dict(position.value_counts().sort_index())),
        })
    return pd.DataFrame(rows)


def group_composition(stratum):
    """Which groups the leak-free subset keeps, and which it empties.

    Records are dropped for having a short sequence, which is a property of when
    a cohort enrolled, so the subset can lose a whole intake rather than thin
    every intake evenly.
    """
    modelled = modelled_index(stratum, stratum.band_widths[1])
    free = leak_free_index(stratum).intersection(modelled)
    group = stratum.frame[stratum.group]
    rows = []
    for name in sorted(group.loc[modelled].unique()):
        in_modelled = int((group.loc[modelled] == name).sum())
        in_free = int((group.loc[free] == name).sum())
        rows.append({
            "stratum": stratum.name,
            "group": name,
            "n_modelled": in_modelled,
            "n_leak_free": in_free,
            "proportion_retained": in_free / in_modelled if in_modelled else np.nan,
        })
    return pd.DataFrame(rows)


def _fold_scores(stratum, band_width, mode, keep=None):
    features, labels, _ = ru.release_matrix(
        stratum.frame, stratum.sequence, stratum.predictors, st.SENSITIVE,
        band_width=band_width, derived_mode=mode,
    )
    if keep is not None:
        rows = features.index.intersection(keep)
        features, labels = features.loc[rows], labels.loc[rows]
    scored = ru.cross_validated_utility(features, labels, return_folds=True)
    return scored["fold_auc_pr"], len(features), float(labels.mean())


def arm_comparison(stratum):
    """The arm gap on the full modelled sample and on the leak-free subset.

    The label is left exactly as the corpus defines it, threshold included, so
    the only thing that changes between the two samples is which records are
    scored. Recomputing the threshold on the subset would confound the leak
    with a different construct.
    """
    samples = (("all_modelled", None), ("leak_free", leak_free_index(stratum)))
    rows = []
    for band_width in stratum.band_widths:
        for sample, keep in samples:
            folds, counts, prevalence = {}, {}, {}
            for mode in ARMS:
                folds[mode], counts[mode], prevalence[mode] = _fold_scores(
                    stratum, band_width, mode, keep)
            paired = sv.paired_fold_difference(
                folds[dc.PROPOSED], folds[dc.BASELINE], n_splits=ru.N_CV_FOLDS)
            difference = np.array(folds[dc.PROPOSED]) - np.array(folds[dc.BASELINE])
            rows.append({
                "stratum": stratum.name,
                "band_width": band_width,
                "sample": sample,
                "n_modelled": counts[dc.BASELINE],
                "prevalence": prevalence[dc.BASELINE],
                "auc_pr_baseline": float(np.mean(folds[dc.BASELINE])),
                "auc_pr_proposed": float(np.mean(folds[dc.PROPOSED])),
                "mean_difference": paired["mean_difference"],
                "corrected_ci_low": paired["corrected_ci_low"],
                "corrected_ci_high": paired["corrected_ci_high"],
                "corrected_p": paired["corrected_p"],
                "wilcoxon_p": paired["wilcoxon_p"],
                "cohens_d": paired["cohens_d"],
                "cohens_d_magnitude": paired["cohens_d_magnitude"],
                "n_folds": paired["n_folds"],
                "folds_favouring_baseline": int((difference < 0).sum()),
            })
    return pd.DataFrame(rows)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    strata = st.load_all()

    positions = position_audit(strata)
    positions.to_csv(RESULTS / "leak_audit_positions.csv", index=False)
    print(f"\n{'stratum':<16}{'modelled':>10}{'inside':>8}{'share':>9}{'leak-free':>11}")
    for row in positions.itertuples():
        print(f"{row.stratum:<16}{row.n_modelled:>10}"
              f"{row.n_outcome_inside_predictors:>8}{row.proportion_inside:>9.1%}"
              f"{row.n_leak_free:>11}")

    affected = [name for name, stratum in strata.items()
                if len(leak_free_index(stratum)) < len(modelled_index(
                    stratum, stratum.band_widths[1]))]
    if not affected:
        print("\nno corpus has an outcome position inside its predictors")
        return

    composition = pd.concat([group_composition(strata[name]) for name in affected],
                            ignore_index=True)
    composition.to_csv(RESULTS / "leak_audit_group_composition.csv", index=False)
    print(f"\n{'group':<12}{'modelled':>10}{'leak-free':>11}{'retained':>10}")
    for row in composition.itertuples():
        print(f"{row.group:<12}{row.n_modelled:>10}{row.n_leak_free:>11}"
              f"{row.proportion_retained:>10.1%}")

    comparison = pd.concat([arm_comparison(strata[name]) for name in affected],
                           ignore_index=True)
    comparison.to_csv(RESULTS / "leak_audit_arm_comparison.csv", index=False)
    print(f"\n{'band':>6}{'sample':>14}{'n':>6}{'gap':>9}{'corrected CI':>22}"
          f"{'corr p':>9}{'wilcox p':>11}{'d_z':>8}{'folds':>8}")
    for row in comparison.itertuples():
        interval = f"[{row.corrected_ci_low:+.4f}, {row.corrected_ci_high:+.4f}]"
        print(f"{row.band_width:>6}{row.sample:>14}{row.n_modelled:>6}"
              f"{row.mean_difference:>+9.4f}{interval:>22}{row.corrected_p:>9.4f}"
              f"{row.wilcoxon_p:>11.2e}{row.cohens_d:>+8.3f}"
              f"{row.folds_favouring_baseline:>5}/{row.n_folds}")

    print(f"\nwrote 3 tables to {RESULTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
