"""Stage 13. The close-out checks, as pipeline code rather than a loose script.

Four quantities reported in the manuscript were produced by a close-out script
run outside the repository, so no committed code stood behind them. Each is
recomputed here and written to `results/disclosure/`:

    ten root seeds rather than one, because fold variance is not seed variance
    and a direction that holds at one seed is not yet a direction
    the same comparison with the single most influential fold dropped
    nursing at three predictor positions, which tests whether sequence length
    explains the public corpus pattern
    the reconstruction and containment check against every release arm, and the
    classifier error profile per arm, which are the two Q21 validation legs

The error profile imports stage 10's own routine rather than reimplementing it,
so the counts are produced by the protocol R7 already reports.

Run:
    ml_env/bin/python notebooks/13_closeout.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import attack_models as am  # noqa: E402
import deidentify as di  # noqa: E402
import derivation_consistent as dc  # noqa: E402
import risk_utility as ru  # noqa: E402
import strata as st  # noqa: E402

RESULTS = ROOT / "results" / "disclosure"

# 42 is risk_utility.BASE_SEED and the seed every reported fit uses; the rest
# exist only to test whether the direction survives a different shuffle.
SEED_LIST = (42, 7, 2026, 101, 202, 303, 404, 505, 606, 707)


def _stage_10():
    """Stage 10's fit-and-score and error-profile routines.

    Imported by path because the stage is a numbered script rather than a
    module. Reused so the per-arm error counts come from the same protocol as
    the ones R7 reports, not from a second implementation of it.
    """
    spec = importlib.util.spec_from_file_location(
        "stage10", ROOT / "notebooks" / "10_error_and_curves.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _matrix(stratum, mode, predictors=None, band=None):
    return ru.release_matrix(
        stratum.frame, stratum.sequence, predictors or stratum.predictors,
        st.SENSITIVE, band_width=band or stratum.band_widths[1], derived_mode=mode,
    )


def seed_stability(strata):
    """The arm comparison at ten root seeds."""
    rows = []
    for name, stratum in strata.items():
        for seed in SEED_LIST:
            scored = {}
            for mode in (dc.BASELINE, dc.PROPOSED):
                X, y, _ = _matrix(stratum, mode)
                scored[mode] = (np.nan if len(X) <= 20 or y.nunique() < 2
                                else ru.cross_validated_utility(X, y, seed=seed)["auc_pr"])
            rows.append({
                "stratum": name,
                "band_width": stratum.band_widths[1],
                "seed": seed,
                "auc_pr_baseline": scored[dc.BASELINE],
                "auc_pr_proposed": scored[dc.PROPOSED],
                "delta": scored[dc.PROPOSED] - scored[dc.BASELINE],
            })
    return pd.DataFrame(rows)


def leave_one_fold_out(strata):
    """Whether the sign of the gap survives dropping its most influential fold."""
    rows = []
    for name, stratum in strata.items():
        folds = {}
        for mode in (dc.BASELINE, dc.PROPOSED):
            X, y, _ = _matrix(stratum, mode)
            if len(X) <= 20 or y.nunique() < 2:
                folds[mode] = np.array([])
                continue
            scored = ru.cross_validated_utility(X, y, seed=ru.BASE_SEED,
                                                return_folds=True)
            folds[mode] = np.array(scored["fold_auc_pr"])
        base, prop = folds[dc.BASELINE], folds[dc.PROPOSED]
        if not len(base) or len(base) != len(prop):
            continue
        differences = prop - base
        full = float(differences.mean())
        dropped = [float(np.delete(differences, i).mean())
                   for i in range(len(differences))]
        rows.append({
            "stratum": name,
            "n_folds": len(differences),
            "mean_difference": full,
            "min_leave_one_out": min(dropped),
            "max_leave_one_out": max(dropped),
            "sign_stable": bool(all(np.sign(d) == np.sign(full) for d in dropped)),
        })
    return pd.DataFrame(rows)


def nursing_three_positions(strata):
    """Nursing at two predictor positions against three.

    The sample is not held constant: requiring a third observed position is
    what shrinks it, so the row records n alongside the scores.
    """
    stratum = strata["nursing"]
    rows = []
    for label, predictors in (("two", list(stratum.sequence[:2])),
                              ("three", list(stratum.sequence[:3]))):
        for mode in (dc.BASELINE, dc.PROPOSED):
            X, y, _ = _matrix(stratum, mode, predictors=predictors)
            rows.append({
                "predictor_positions": label,
                "n_predictors": len(predictors),
                "derived_mode": mode,
                "n_modelled": len(X),
                "prevalence": float(y.mean()),
                "auc_pr": ru.cross_validated_utility(X, y, seed=ru.BASE_SEED)["auc_pr"],
            })
    return pd.DataFrame(rows)


def reconstruction_per_arm(strata):
    """Reconstruction and containment against every arm, not the baseline alone.

    Containment below 1.0 for a protected arm is not a defect of the procedure.
    The published constraining features are computed from the banded values
    there, so the premise the attack reasons from is false and its intervals
    collapse onto points that exclude the truth.
    """
    rows = []
    for name, stratum in strata.items():
        frame, sequence = stratum.frame, stratum.sequence
        truth = frame[sequence]
        for band in stratum.band_widths:
            low, high = di.band_intervals(frame, sequence, band)
            for mode in (dc.BASELINE, dc.PROPOSED, dc.SELECTIVE):
                release = dc.build(frame, sequence, band, None, mode)
                published = release.frame[
                    [c for c in di.CONSTRAINING_COLUMNS if c in release.frame]
                ]
                rec_low, rec_high, summary = am.reconstruct(
                    frame, sequence, low, high, published)
                metrics = am.recovery_metrics(rec_low, rec_high, summary, band,
                                              truth=truth)
                metrics.update(stratum=name, derived_mode=mode)
                rows.append(metrics)
    return pd.DataFrame(rows)


def error_profile_per_arm(strata, stage10):
    """Confusion counts per arm, the Q21 targeted-error leg.

    Q21 step 2 located the failure as over-flagging of students who are not
    weak, so the quantity it asks about is the classifier's false positive.
    """
    rows = []
    for name, stratum in strata.items():
        for mode in (dc.NONE, dc.BASELINE, dc.PROPOSED, dc.SELECTIVE):
            X, y, _ = _matrix(stratum, mode)
            scored = stage10._fit_and_score(X, y)
            profile = stage10._error_profile(scored["y"], scored["mean_proba"])
            rows.append({"stratum": name, "band_width": stratum.band_widths[1],
                         "derived_mode": mode, "n_modelled": len(X), **profile})
    return pd.DataFrame(rows)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    strata = st.load_all()
    stage10 = _stage_10()

    seeds = seed_stability(strata)
    seeds.to_csv(RESULTS / "closeout_seed_stability.csv", index=False)
    print(f"\n{'stratum':<16}{'seeds':>6}{'mean delta':>12}{'sd':>9}{'sign flips':>12}")
    for name, group in seeds.groupby("stratum"):
        deltas = group.delta.dropna()
        signs = np.sign(deltas)
        flips = int((signs != signs.iloc[0]).sum())
        print(f"{name:<16}{len(deltas):>6}{deltas.mean():>12.4f}"
              f"{deltas.std(ddof=1):>9.4f}{flips:>12}")

    folds = leave_one_fold_out(strata)
    folds.to_csv(RESULTS / "closeout_fold_leave_one_out.csv", index=False)
    print("\nsign survives dropping any one fold:",
          {r.stratum: r.sign_stable for r in folds.itertuples()})

    three = nursing_three_positions(strata)
    three.to_csv(RESULTS / "closeout_nursing_three_positions.csv", index=False)
    print("\nnursing predictor positions, AUC-PR and sample:")
    for row in three.itertuples():
        print(f"  {row.predictor_positions:<6}{row.derived_mode:<10}"
              f"auc_pr={row.auc_pr:.4f}  n={row.n_modelled}")

    arms = reconstruction_per_arm(strata)
    arms.to_csv(RESULTS / "v2_selective_reconstruction.csv", index=False)
    print("\ncontainment by arm:",
          {m: round(float(g.containment_check.mean()), 4)
           for m, g in arms.groupby("derived_mode")})

    errors = error_profile_per_arm(strata, stage10)
    errors.to_csv(RESULTS / "q21_targeted_error_selective.csv", index=False)
    print("\nfalse positives by arm:")
    for name, group in errors.groupby("stratum"):
        counts = {r.derived_mode: int(r.false_positive) for r in group.itertuples()}
        print(f"  {name:<16}{counts}")

    print(f"\nwrote 5 tables to {RESULTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
