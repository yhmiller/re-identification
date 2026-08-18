"""Stage 5. The derived-feature experiments.

Phase 3 of the re-identification study, and its main contribution. Three
experiments, each answering one question about engineered features:

    A   Do derived features identify students on their own, with the raw
        semester sequence withheld? This tests the common belief that releasing
        summary statistics rather than records is privacy-protective.

    B   Does publishing derived features at full precision alongside generalised
        source variables reverse the protection? gpa_min and gpa_max are exact
        order statistics of the sequence just banded, so the suspicion is that
        they hand back the numbers the banding removed.

    C   Does recomputing the derived features from the generalised data instead
        of the originals close the hole, and what does it cost?

Run:
    ml_env/bin/python notebooks/05_derived_feature_experiments.py

Writes to results/disclosure/. Every output is aggregate.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import attack_models as am  # noqa: E402
import deidentify as di  # noqa: E402
import disclosure_risk as dr  # noqa: E402

RESULTS = ROOT / "results" / "disclosure"
RESULTS.mkdir(parents=True, exist_ok=True)

SEMESTERS = [f"gpa_sem{i}" for i in range(1, 7)]
BAND_WIDTHS = [0.25, 0.5, 1.0]


def load_strata():
    allied = pd.read_excel(ROOT / "data" / "model_dataset_2021_2022.xlsx")
    allied["group"] = (
        allied["programme"].astype(str) + "-" + allied["cohort_year"].astype(str)
    )

    nursing = pd.read_excel(
        ROOT / "data" / "consolidated_nursing.xlsx", sheet_name="students"
    )
    nursing["group"] = "NUR-" + nursing["intake_year"].astype(str)

    return {"allied_health": allied, "nursing": nursing}


def experiment_a(name, frame):
    """Derived features alone, raw sequence withheld."""
    derived = di.derive_features(frame, SEMESTERS)
    combined = frame[["group"]].join(derived)

    sets = {
        "raw semester sequence": (frame, SEMESTERS),
        "all 8 derived features": (combined, di.DERIVED_COLUMNS),
        "3 derived (min, max, trend)": (combined, ["gpa_min", "gpa_max", "gpa_trend"]),
        "2 derived (mean, consistency)": (combined, ["gpa_mean", "gpa_consistency"]),
        "gpa_trend alone": (combined, ["gpa_trend"]),
    }

    rows = []
    for label, (source, cols) in sets.items():
        profile = dr.risk_profile(source, cols)
        rows.append(
            {
                "stratum": name,
                "release": label,
                "n_attributes": len(cols),
                "prop_unique": profile.prop_unique,
                "min_class_size": profile.min_class_size,
                "n_equivalence_classes": profile.n_equivalence_classes,
            }
        )
    return pd.DataFrame(rows)


def experiment_b(name, frame):
    """Generalise the sequence, publish derived features at full precision, attack."""
    truth = frame[SEMESTERS]
    derived = di.derive_features(frame, SEMESTERS)[di.CONSTRAINING_COLUMNS]

    rows = []
    for width in BAND_WIDTHS:
        generalised = di.generalise(frame, SEMESTERS, width)
        low, high = di.band_intervals(frame, SEMESTERS, width)

        protected = dr.risk_profile(generalised, SEMESTERS).prop_unique
        original = dr.risk_profile(frame, SEMESTERS).prop_unique

        rec_low, rec_high, summary = am.reconstruct(
            frame, SEMESTERS, low, high, derived
        )
        metrics = am.recovery_metrics(rec_low, rec_high, summary, width, truth=truth)

        # What uniqueness does the attacker get back from the recovered values?
        recovered = frame[["group"]].join(am.point_estimates(rec_low, rec_high))
        attacked = dr.risk_profile(recovered, SEMESTERS).prop_unique

        rows.append(
            {
                "stratum": name,
                "band_width": width,
                "prop_unique_original": original,
                "prop_unique_after_generalisation": protected,
                "prop_unique_after_attack": attacked,
                "protection_reversed": (
                    (attacked - protected) / (original - protected)
                    if original > protected
                    else float("nan")
                ),
                **{k: v for k, v in metrics.items() if k != "band_width"},
            }
        )
    return pd.DataFrame(rows)


def experiment_c(name, frame):
    """Recompute derived features from the generalised data instead."""
    rows = []
    for width in BAND_WIDTHS:
        generalised = di.generalise(frame, SEMESTERS, width)

        leaky = di.derive_features(frame, SEMESTERS)  # from originals
        safe = di.derive_features(generalised, SEMESTERS)  # from bands

        leaky_release = generalised[["group"]].join(leaky)
        safe_release = generalised[["group"]].join(safe)

        cols = SEMESTERS + di.DERIVED_COLUMNS
        rows.append(
            {
                "stratum": name,
                "band_width": width,
                "prop_unique_derived_from_originals": dr.risk_profile(
                    leaky_release.join(generalised[SEMESTERS]), cols
                ).prop_unique,
                "prop_unique_derived_from_bands": dr.risk_profile(
                    safe_release.join(generalised[SEMESTERS]), cols
                ).prop_unique,
                # What the custodian gives up by recomputing: how far the safe
                # features drift from the ones the pipeline currently publishes.
                "mean_abs_error_gpa_min": float(
                    (safe["gpa_min"] - leaky["gpa_min"]).abs().mean()
                ),
                "mean_abs_error_gpa_mean": float(
                    (safe["gpa_mean"] - leaky["gpa_mean"]).abs().mean()
                ),
                "corr_gpa_trend": float(safe["gpa_trend"].corr(leaky["gpa_trend"])),
            }
        )
    return pd.DataFrame(rows)


def main():
    strata = load_strata()
    a_all, b_all, c_all = [], [], []

    for name, frame in strata.items():
        print(f"\n{'=' * 74}\n{name}: {len(frame)} students\n{'=' * 74}")

        a = experiment_a(name, frame)
        a_all.append(a)
        print("EXPERIMENT A  does a summary-only release protect anyone?")
        print(f"  {'release':<32}{'attrs':>7}{'unique':>9}{'min k':>7}")
        for _, r in a.iterrows():
            print(
                f"  {r['release']:<32}{r['n_attributes']:>7}"
                f"{r['prop_unique']:>8.1%}{r['min_class_size']:>7}"
            )

        b = experiment_b(name, frame)
        b_all.append(b)
        print("\nEXPERIMENT B  do derived features reverse generalisation?")
        print(
            f"  {'band':>6}{'original':>10}{'banded':>9}{'attacked':>10}"
            f"{'reversed':>10}{'exact':>8}{'check':>7}"
        )
        for _, r in b.iterrows():
            print(
                f"  {r['band_width']:>6.2f}{r['prop_unique_original']:>9.1%}"
                f"{r['prop_unique_after_generalisation']:>9.1%}"
                f"{r['prop_unique_after_attack']:>10.1%}"
                f"{r['protection_reversed']:>10.1%}"
                f"{r['prop_values_recovered_exactly']:>8.1%}"
                f"{r['containment_check']:>7.0%}"
            )

        c = experiment_c(name, frame)
        c_all.append(c)
        print("\nEXPERIMENT C  recomputing derived features from the bands")
        print(
            f"  {'band':>6}{'from originals':>16}{'from bands':>13}{'MAE min':>10}{'trend r':>9}"
        )
        for _, r in c.iterrows():
            print(
                f"  {r['band_width']:>6.2f}"
                f"{r['prop_unique_derived_from_originals']:>15.1%}"
                f"{r['prop_unique_derived_from_bands']:>13.1%}"
                f"{r['mean_abs_error_gpa_min']:>10.3f}"
                f"{r['corr_gpa_trend']:>9.3f}"
            )

    for frames, filename in (
        (a_all, "experiment_a_summary_release.csv"),
        (b_all, "experiment_b_reconstruction.csv"),
        (c_all, "experiment_c_consistent_protection.csv"),
    ):
        pd.concat(frames, ignore_index=True).to_csv(RESULTS / filename, index=False)

    print(f"\nwrote 3 tables to {RESULTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
