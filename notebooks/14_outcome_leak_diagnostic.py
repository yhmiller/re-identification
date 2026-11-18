"""Stage 14. Where each corpus's outcome position sits against its predictors.

The sensitive attribute is weak performance in the last observed position of a
record's sequence, chosen per record. The predictor set is a fixed prefix of
that sequence. The two separate only when every record is observed past the
prefix, which holds in allied health and in the public corpus. It fails in
nursing, where most modelled records stop at the second position, and that
position is released as a predictor, so for those records the label is a
threshold on a column the recipient already holds.

Every utility sample in the pipeline now excludes those records, through
`risk_utility.outcome_follows_predictors`. This stage measures what that
exclusion does by scoring the same comparison on both samples, which is the
sensitivity analysis the corrected result is reported against.

Writes four tables to `results/disclosure/`:

    leak_audit_positions.csv          where the outcome position falls, per corpus
    leak_audit_arm_comparison.csv     the arm comparison on every modelled record
                                      against the leak-free records
    leak_audit_group_composition.csv  which groups survive the exclusion, since
                                      dropping records by sequence length is not
                                      neutral across intakes
    leak_audit_error_profile.csv      confusion counts per arm on both samples

Run:
    ml_env/bin/python notebooks/14_outcome_leak_diagnostic.py
"""

import importlib.util
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

# The two samples the comparison is scored on, keyed to release_matrix's
# `leak_free` argument.
SAMPLES = (("all_modelled", False), ("leak_free", True))


def _stage_10():
    """Stage 10's fit-and-score and error-profile routines, imported by path.

    The stage is a numbered script rather than a module. Reusing its routines
    keeps these counts on the protocol the Results already report.
    """
    spec = importlib.util.spec_from_file_location(
        "stage10", ROOT / "notebooks" / "10_error_and_curves.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def modelled_index(stratum, band_width, leak_free=False):
    """The records that survive into the utility model.

    The arm does not change which records survive, since both arms derive over
    the same predictor positions, so one arm is enough to establish the sample.
    """
    features, _, _ = ru.release_matrix(
        stratum.frame, stratum.sequence, stratum.predictors, st.SENSITIVE,
        band_width=band_width, derived_mode=dc.BASELINE, leak_free=leak_free,
    )
    return features.index


def position_audit(strata):
    """How many modelled records in each corpus can read their own label."""
    rows = []
    for name, stratum in strata.items():
        band = stratum.band_widths[1]
        modelled = modelled_index(stratum, band)
        clean = modelled_index(stratum, band, leak_free=True)
        position = last_observed_position(stratum.frame, stratum.sequence).loc[modelled]
        labels = stratum.frame.loc[modelled, st.SENSITIVE]
        inside = ~modelled.isin(clean)
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
            "prevalence_leak_free": (float(labels[~inside].mean())
                                     if (~inside).any() else np.nan),
            "last_position_counts": str(dict(position.value_counts().sort_index())),
        })
    return pd.DataFrame(rows)


def group_composition(stratum):
    """Which groups the leak-free sample keeps, and which it empties.

    Records are dropped for having a short sequence, which is a property of when
    a cohort enrolled, so the sample can lose a whole intake rather than thin
    every intake evenly.
    """
    band = stratum.band_widths[1]
    modelled = modelled_index(stratum, band)
    clean = modelled_index(stratum, band, leak_free=True)
    group = stratum.frame[stratum.group]
    rows = []
    for name in sorted(group.loc[modelled].unique()):
        in_modelled = int((group.loc[modelled] == name).sum())
        in_clean = int((group.loc[clean] == name).sum())
        rows.append({
            "stratum": stratum.name,
            "group": name,
            "n_modelled": in_modelled,
            "n_leak_free": in_clean,
            "proportion_retained": in_clean / in_modelled if in_modelled else np.nan,
        })
    return pd.DataFrame(rows)


def _fold_scores(stratum, band_width, mode, leak_free):
    features, labels, _ = ru.release_matrix(
        stratum.frame, stratum.sequence, stratum.predictors, st.SENSITIVE,
        band_width=band_width, derived_mode=mode, leak_free=leak_free,
    )
    scored = ru.cross_validated_utility(features, labels, return_folds=True)
    return scored["fold_auc_pr"], len(features), float(labels.mean())


def arm_comparison(stratum):
    """The arm gap on every modelled record and on the leak-free records.

    The label is left exactly as the corpus defines it, threshold included, so
    the only thing that changes between the two samples is which records are
    scored. Recomputing the threshold on the clean records would confound the
    leak with a different construct.
    """
    rows = []
    for band_width in stratum.band_widths:
        for sample, leak_free in SAMPLES:
            folds, counts, prevalence = {}, {}, {}
            for mode in ARMS:
                folds[mode], counts[mode], prevalence[mode] = _fold_scores(
                    stratum, band_width, mode, leak_free)
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


def error_profile(stratum, stage10):
    """Confusion counts per arm at the middle band width, on both samples."""
    rows = []
    for sample, leak_free in SAMPLES:
        for mode in dc.MODES:
            features, labels, _ = ru.release_matrix(
                stratum.frame, stratum.sequence, stratum.predictors, st.SENSITIVE,
                band_width=stratum.band_widths[1], derived_mode=mode,
                leak_free=leak_free)
            scored = stage10._fit_and_score(features, labels)
            profile = stage10._error_profile(scored["y"], scored["mean_proba"])
            rows.append({"stratum": stratum.name, "sample": sample,
                         "derived_mode": mode, "n_modelled": len(features),
                         **profile})
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

    affected = positions[positions.n_outcome_inside_predictors > 0].stratum.tolist()
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

    stage10 = _stage_10()
    errors = pd.concat([error_profile(strata[name], stage10) for name in affected],
                       ignore_index=True)
    errors.to_csv(RESULTS / "leak_audit_error_profile.csv", index=False)
    print("\nfalse positives and false negatives by arm:")
    for sample, group in errors.groupby("sample"):
        counts = {r.derived_mode: (int(r.false_positive), int(r.false_negative))
                  for r in group.itertuples()}
        print(f"  {sample:<14}{counts}")

    print(f"\nwrote 4 tables to {RESULTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
