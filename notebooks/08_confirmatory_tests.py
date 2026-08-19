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

import deidentify as di  # noqa: E402
import derivation_consistent as dc  # noqa: E402
import disclosure_risk as dr  # noqa: E402
import risk_utility as ru  # noqa: E402
import stats_validation as sv  # noqa: E402
import strata as st  # noqa: E402

RESULTS = ROOT / "results" / "disclosure"
RESULTS.mkdir(parents=True, exist_ok=True)


def risk_comparison(stratum, band_width):
    """Difference in uniqueness between the two derivation arms.

    Subsampling without replacement rather than the ordinary bootstrap, because
    duplicated records destroy uniqueness asymmetrically between arms. The full
    corpus estimate is reported beside the subsample estimate so that any drift
    from the reduced set size is visible.
    """
    base = dc.baseline(stratum.frame, stratum.sequence, band_width=band_width)
    prop = dc.proposed(stratum.frame, stratum.sequence, band_width=band_width)

    def uniqueness(release):
        cols = release.visible_columns

        def measure(idx):
            subset = release.frame.iloc[idx]
            sizes = dr.equivalence_class_sizes(subset, cols)
            return float((sizes == 1).mean())

        return measure

    return sv.subsample_difference(
        uniqueness(prop), uniqueness(base), n_records=len(stratum.frame)
    )


def utility_comparison(stratum, band_width):
    """Paired per-fold AUC-PR difference between the two derivation arms."""
    folds = {}
    for mode in (dc.BASELINE, dc.PROPOSED):
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
        folds[dc.PROPOSED]["fold_auc_pr"],
        folds[dc.BASELINE]["fold_auc_pr"],
        n_splits=ru.N_CV_FOLDS,
    )
    result["fold_differences"] = list(
        np.asarray(folds[dc.PROPOSED]["fold_auc_pr"])
        - np.asarray(folds[dc.BASELINE]["fold_auc_pr"])
    )
    result["auc_pr_baseline"] = folds[dc.BASELINE]["auc_pr"]
    result["auc_pr_proposed"] = folds[dc.PROPOSED]["auc_pr"]
    result["relative_loss"] = (
        result["mean_difference"] / folds[dc.BASELINE]["auc_pr"]
        if folds[dc.BASELINE]["auc_pr"]
        else np.nan
    )
    return result


def main():
    risk_rows, utility_rows = [], []

    for name, s in st.load_all().items():
        print(f"\n{'=' * 78}\n{name}: {s.n} records\n{'=' * 78}")

        for width in s.band_widths:
            r = risk_comparison(s, width)
            r.update(stratum=name, band_width=width)
            risk_rows.append(r)

            u = utility_comparison(s, width)
            u.update(stratum=name, band_width=width)
            utility_rows.append(u)

            print(f"\n  band {width:g}")
            print(f"    RISK      proposed minus baseline uniqueness")
            print(
                f"              full corpus {r['difference_full_corpus']:+.4f}, "
                f"at {r['subsample_fraction']:.0%} {r['difference_at_fraction']:+.4f}"
            )
            print(
                f"              95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  "
                f"{'excludes 0' if r['excludes_zero'] else 'includes 0'}"
            )
            print(f"    UTILITY   proposed minus baseline AUC-PR")
            print(
                f"              {u['mean_difference']:+.4f}  "
                f"({u['relative_loss']:+.1%} of baseline {u['auc_pr_baseline']:.4f})"
            )
            print(
                f"              naive     CI [{u['naive_ci_low']:+.4f}, "
                f"{u['naive_ci_high']:+.4f}]  p {u['naive_p']:.2e}"
            )
            print(
                f"              corrected CI [{u['corrected_ci_low']:+.4f}, "
                f"{u['corrected_ci_high']:+.4f}]  p {u['corrected_p']:.2e}  "
                f"SE x{u['se_inflation']:.2f}"
            )

    # Per-fold differences go to their own long-format table. The stability
    # figure needs the distribution rather than the summary: a figure showing
    # only means would hide the corpus whose folds straddle zero.
    fold_rows = [
        {"stratum": u["stratum"], "band_width": u["band_width"],
         "fold": i, "difference": d}
        for u in utility_rows for i, d in enumerate(u["fold_differences"])
    ]
    pd.DataFrame(fold_rows).to_csv(
        RESULTS / "confirmatory_fold_differences.csv", index=False
    )

    for rows, filename in (
        (risk_rows, "confirmatory_risk.csv"),
        (utility_rows, "confirmatory_utility.csv"),
    ):
        frame = pd.DataFrame(rows).drop(columns=["fold_differences"], errors="ignore")
        lead = ["stratum", "band_width"]
        frame = frame[lead + [c for c in frame.columns if c not in lead]]
        frame.to_csv(RESULTS / filename, index=False)

    print(f"\nwrote 2 tables to {RESULTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
