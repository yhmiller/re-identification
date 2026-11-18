"""Stage 8. Confirmatory analysis.

Runs the comparisons fixed in NewDirection/analysis-specification.md. Nothing
here is chosen after seeing a result.

Two comparisons, built differently because the units of independence differ.

    Risk      A property of the released set, so records are the resampling
              unit. Subsampling without replacement, both arms evaluated on the
              same subsample so the comparison is paired at record level. The
              ordinary bootstrap is wrong here and was rejected on evidence;
              see stats_validation.subsample_difference.

    Utility   Estimated by cross-validation on identical splits, so folds pair
              naturally. Reported naive and with a correction for the dependence
              between overlapping training sets.

Neither effect size is standardised. Both metrics are bounded and interpretable
in their own units.

Run:
    ml_env/bin/python notebooks/08_confirmatory_tests.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import derivation_consistent as dc  # noqa: E402
import disclosure_risk as dr  # noqa: E402
import risk_utility as ru  # noqa: E402
import stats_validation as sv  # noqa: E402
import strata as st  # noqa: E402

RESULTS = ROOT / "results" / "disclosure"
RESULTS.mkdir(parents=True, exist_ok=True)

# The proposed arm is the comparison the specification fixed in advance. The
# selective arm was built after seeing the proposed arm's result, so it is run
# through the same two tests but written to its own files, and never merged into
# the confirmatory ones.
TREATMENTS = {dc.PROPOSED: "confirmatory", dc.SELECTIVE: "posthoc_selective"}


def risk_comparison(stratum, band_width, treatment=dc.PROPOSED):
    """Difference in uniqueness between the baseline arm and one treatment arm.

    Subsampling without replacement rather than the ordinary bootstrap, because
    duplicated records destroy uniqueness asymmetrically between arms. The full
    corpus estimate is reported beside the subsample estimate so that any drift
    from the reduced set size is visible.
    """
    base = dc.build(stratum.frame, stratum.sequence, band_width, None, dc.BASELINE)
    prop = dc.build(stratum.frame, stratum.sequence, band_width, None, treatment)

    def uniqueness(release):
        cols = release.visible_columns

        def measure(idx):
            subset = release.frame.iloc[idx]
            sizes = dr.equivalence_class_sizes(subset, cols)
            return float((sizes == 1).mean())

        return measure

    result = sv.subsample_difference(
        uniqueness(prop), uniqueness(base), n_records=len(stratum.frame)
    )

    # Record both arms, not only their difference. The utility side has always
    # reported auc_pr_baseline and auc_pr_proposed; the risk side reported the
    # difference alone, which meant a reader wanting the two absolute figures
    # had to reconstruct one from the other. Reconstructing it from the
    # post-attack uniqueness in the ablation table gives a different quantity
    # and would produce a negative proportion.
    every_record = np.arange(len(stratum.frame))
    result["prop_unique_baseline"] = uniqueness(base)(every_record)
    result[f"prop_unique_{treatment}"] = uniqueness(prop)(every_record)
    return result


def utility_comparison(stratum, band_width, treatment=dc.PROPOSED):
    """Paired per-fold AUC-PR difference between baseline and one treatment arm."""
    folds = {}
    for mode in (dc.BASELINE, treatment):
        row = ru.frontier_row(
            stratum.frame,
            stratum.sequence,
            stratum.predictors,
            st.SENSITIVE,
            mode,
            band_width=band_width,
            derived_mode=mode,
            return_folds=True,
        )
        folds[mode] = row
    result = sv.paired_fold_difference(
        folds[treatment]["fold_auc_pr"],
        folds[dc.BASELINE]["fold_auc_pr"],
        n_splits=ru.N_CV_FOLDS,
    )
    result["fold_differences"] = list(
        np.asarray(folds[treatment]["fold_auc_pr"])
        - np.asarray(folds[dc.BASELINE]["fold_auc_pr"])
    )
    result["auc_pr_baseline"] = folds[dc.BASELINE]["auc_pr"]
    result[f"auc_pr_{treatment}"] = folds[treatment]["auc_pr"]
    result["relative_loss"] = (
        result["mean_difference"] / folds[dc.BASELINE]["auc_pr"]
        if folds[dc.BASELINE]["auc_pr"]
        else np.nan
    )
    return result


def _report(width, risk, utility, treatment):
    print(f"\n  band {width:g}")
    print(f"    RISK      {treatment} minus baseline uniqueness")
    print(
        f"              full corpus {risk['difference_full_corpus']:+.4f}, "
        f"at {risk['subsample_fraction']:.0%} {risk['difference_at_fraction']:+.4f}"
    )
    print(
        f"              95% CI [{risk['ci_low']:+.4f}, {risk['ci_high']:+.4f}]  "
        f"{'excludes 0' if risk['excludes_zero'] else 'includes 0'}"
    )
    print(f"    UTILITY   {treatment} minus baseline AUC-PR")
    print(
        f"              {utility['mean_difference']:+.4f}  "
        f"({utility['relative_loss']:+.1%} of baseline "
        f"{utility['auc_pr_baseline']:.4f})"
    )
    print(
        f"              naive     CI [{utility['naive_ci_low']:+.4f}, "
        f"{utility['naive_ci_high']:+.4f}]  p {utility['naive_p']:.2e}"
    )
    print(
        f"              corrected CI [{utility['corrected_ci_low']:+.4f}, "
        f"{utility['corrected_ci_high']:+.4f}]  p {utility['corrected_p']:.2e}  "
        f"SE x{utility['se_inflation']:.2f}"
    )


def add_holm(utility_rows):
    """Holm-adjust the corrected p within each corpus.

    The family is one corpus's utility comparisons across its three band widths,
    which M16 declares. Risk comparisons carry percentile intervals and no p
    value, so they have nothing to adjust.
    """
    for stratum in {row["stratum"] for row in utility_rows}:
        family = [row for row in utility_rows if row["stratum"] == stratum]
        adjusted = sv.holm_adjust([row["corrected_p"] for row in family])
        for row, value in zip(family, adjusted):
            row["holm_p"] = float(value)


def compare(treatment):
    """Both confirmatory comparisons, for one treatment arm against baseline."""
    risk_rows, utility_rows = [], []
    for name, s in st.load_all().items():
        print(f"\n{'=' * 78}\n{name}: {s.n} records, {treatment} minus baseline"
              f"\n{'=' * 78}")
        for width in s.band_widths:
            r = risk_comparison(s, width, treatment)
            r.update(stratum=name, band_width=width, treatment=treatment)
            risk_rows.append(r)

            u = utility_comparison(s, width, treatment)
            u.update(stratum=name, band_width=width, treatment=treatment)
            utility_rows.append(u)

            _report(width, r, u, treatment)
    add_holm(utility_rows)
    return risk_rows, utility_rows


def write(prefix, risk_rows, utility_rows):
    # Per-fold differences go to their own long-format table. The stability
    # figure needs the distribution rather than the summary: a figure showing
    # only means would hide the corpus whose folds straddle zero.
    fold_rows = [
        {"stratum": u["stratum"], "band_width": u["band_width"],
         "fold": i, "difference": d}
        for u in utility_rows for i, d in enumerate(u["fold_differences"])
    ]
    pd.DataFrame(fold_rows).to_csv(
        RESULTS / f"{prefix}_fold_differences.csv", index=False
    )

    for rows, filename in ((risk_rows, f"{prefix}_risk.csv"),
                           (utility_rows, f"{prefix}_utility.csv")):
        frame = pd.DataFrame(rows).drop(columns=["fold_differences"], errors="ignore")
        lead = ["stratum", "band_width"]
        frame = frame[lead + [c for c in frame.columns if c not in lead]]
        frame.to_csv(RESULTS / filename, index=False)
    return 3


def main():
    written = 0
    for treatment, prefix in TREATMENTS.items():
        written += write(prefix, *compare(treatment))

    print(f"\nwrote {written} tables to {RESULTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
