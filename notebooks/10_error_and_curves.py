"""Stage 10. Curves, fold distributions, error profiles and cost, for Results.

The CAN-DO ML-RESULTS template requires four things the earlier stages do not
produce:

    precision-recall curves for both arms on the same axes
    the primary metric per fold for each arm, not only their difference
    normalised confusion matrices, per-class scores and error counts
    training time per fold, inference time and peak memory

**On timing.** Wall-clock measurements do not reproduce exactly between runs, so
`compute_cost.csv` is the one output of this pipeline that is not byte-identical
on a clean re-run. Fit times here are tens of milliseconds and their standard
deviations are a sizeable fraction of their means, so the table supports a
statement that the two arms cost the same order of magnitude and does not
support a claim that either is faster.

All four are computed here from the same splits, seeds and release construction
the confirmatory stage used, so a number here and a number in stage 08 describe
the same fitted models. `_check_against_stage_08` asserts that rather than
assuming it.

**On the threshold.** Methods M15 reports AUC-PR and AUC-ROC, both rank-based,
and applies no decision threshold to anything. An error profile cannot be
formed without one:
a confusion matrix is a thresholded object. A fixed 0.5 is therefore applied
here, identically to both arms, **for the error profile only**. No metric
reported anywhere else in the study depends on it, and M15 declares it.

**On pooling folds.** Under five repetitions of 5-fold cross-validation every
record is tested five times. Concatenating all 25 test partitions would give a
confusion matrix over 5n rows, which is not a corpus and cannot be read as one.
Each record's predicted probability is therefore averaged across its five
out-of-fold predictions first, giving one probability per record and a
confusion matrix of exactly n.

Run:
    ml_env/bin/python notebooks/10_error_and_curves.py
"""

import resource
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import derivation_consistent as dc  # noqa: E402
import risk_utility as ru  # noqa: E402
import strata as st  # noqa: E402

RESULTS = ROOT / "results" / "disclosure"
RESULTS.mkdir(parents=True, exist_ok=True)

# Applied to both arms, for the error profile only. Declared in Methods M15.
ERROR_THRESHOLD = 0.5

# One curve per corpus would be unreadable at nine band widths, so the reported
# curves use the middle width, which is the one stage 09 explains.
ARMS = (dc.BASELINE, dc.PROPOSED)


def _fit_and_score(X, y):
    """Out-of-fold probabilities, per-fold scores, and what the fitting cost.

    Mirrors risk_utility.cross_validated_utility exactly in splitter, seed and
    model, and additionally retains the predictions those folds produced.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    splitter = RepeatedStratifiedKFold(
        n_splits=ru.N_CV_FOLDS,
        n_repeats=ru.N_CV_REPEATS,
        random_state=ru.BASE_SEED,
    )

    # One column per repetition, so a record's five out-of-fold predictions can
    # be averaged without assuming which fold produced which.
    probabilities = np.full((len(y), ru.N_CV_REPEATS), np.nan)
    folds, fit_times, fit_cpu, predict_times = [], [], [], []

    # One discarded fit first. Without it the first timed fit carries XGBoost's
    # one-off setup and inflates the mean for whichever arm happens to run
    # first, which is a property of the running order rather than the release.
    XGBClassifier(**ru.XGB_PARAMS, random_state=ru.BASE_SEED).fit(X, y)

    tracemalloc.start()
    for index, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        repeat = index // ru.N_CV_FOLDS

        model = XGBClassifier(**ru.XGB_PARAMS, random_state=ru.BASE_SEED)
        # Wall clock and CPU time both recorded. Wall clock is what a user
        # waits and fixes the scale, but it is dominated by whatever
        # else the machine is doing: the same fit has measured 0.10 s and 0.49 s
        # on this hardware. CPU time counts cycles actually spent and is the
        # stable quantity, so it is what any comparison between arms uses.
        started, started_cpu = time.perf_counter(), time.process_time()
        model.fit(X[train_idx], y[train_idx])
        fit_times.append(time.perf_counter() - started)
        fit_cpu.append(time.process_time() - started_cpu)

        started = time.perf_counter()
        proba = model.predict_proba(X[test_idx])[:, 1]
        predict_times.append((time.perf_counter() - started) / len(test_idx))

        probabilities[test_idx, repeat] = proba
        folds.append(
            {
                "fold": len(folds),
                "auc_pr": average_precision_score(y[test_idx], proba),
                "auc_roc": roc_auc_score(y[test_idx], proba),
            }
        )
    peak_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    return {
        "y": y,
        "mean_proba": np.nanmean(probabilities, axis=1),
        "folds": pd.DataFrame(folds),
        "fit_seconds": np.array(fit_times),
        "fit_cpu_seconds": np.array(fit_cpu),
        "predict_seconds_per_instance": float(np.mean(predict_times)),
        # tracemalloc sees Python allocations only. XGBoost allocates its
        # tree structures in C++, so this is a floor on the true footprint and
        # is reported under a name that says so. The process high-water mark
        # is recorded separately, once, in main().
        "python_peak_alloc_mb": peak_bytes / 1024 / 1024,
    }


def _error_profile(y, proba):
    """Confusion counts and minority-class scores at the fixed threshold."""
    predicted = (proba >= ERROR_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=[1], zero_division=0
    )
    return {
        "threshold": ERROR_THRESHOLD,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        # Normalised by true class, which is what the R7 caption must state.
        "recall_negative": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "recall_positive": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "precision_positive": float(precision[0]),
        "f1_positive": float(f1[0]),
        "support_positive": int(support[0]),
    }


def _band_edge_profile(stratum, band, y, proba, mode):
    """Q21.2. How far the misclassified sit from the nearest band boundary.

    The examination's hypothesis: the proposed arm's extra false positives are
    records whose true grades sit near a band edge, where banding displaces them
    across the weak threshold and the displacement then propagates into every
    derived column at once.

    Emitted as group means only. Per-record distances are never written out; the
    aggregate-only rule that governs every other output governs this one.
    """
    predicted = (proba >= ERROR_THRESHOLD).astype(int)
    grades = stratum.frame[stratum.predictors].reindex(
        stratum.frame.index).to_numpy(dtype=float)
    # Distance to the nearest band edge, averaged over a record's predictors.
    offset = np.mod(grades, band)
    distance = np.nanmean(np.minimum(offset, band - offset), axis=1)
    distance = distance[: len(y)]

    groups = {
        "false positive": (y == 0) & (predicted == 1),
        "true negative": (y == 0) & (predicted == 0),
        "false negative": (y == 1) & (predicted == 0),
        "true positive": (y == 1) & (predicted == 1),
    }
    rows = []
    for label, mask in groups.items():
        selected = distance[mask]
        rows.append({
            "derived_mode": mode, "outcome_group": label,
            "n": int(mask.sum()),
            "mean_distance_to_band_edge": (
                float(np.nanmean(selected)) if mask.sum() else np.nan),
            "band_width": band,
        })
    return rows


def _check_against_stage_08(rows):
    """The per-fold means here must reproduce the confirmatory stage's means.

    Both stages fit the same model on the same splits, so a disagreement means
    one of them is not measuring the release it claims to. Checked rather than
    asserted in prose.
    """
    path = RESULTS / "confirmatory_utility.csv"
    if not path.exists():
        print("  stage 08 output absent, skipping cross-check")
        return
    stage_08 = pd.read_csv(path)
    checked = 0
    for row in rows:
        match = stage_08[
            (stage_08.stratum == row["stratum"])
            & (np.isclose(stage_08.band_width, row["band_width"]))
        ]
        if match.empty:
            continue
        column = f"auc_pr_{row['derived_mode']}"
        if column not in match:
            continue
        expected = float(match.iloc[0][column])
        assert abs(expected - row["auc_pr_mean"]) < 1e-9, (
            f"{row['stratum']} {row['derived_mode']}: stage 10 measured "
            f"{row['auc_pr_mean']:.6f} against stage 08's {expected:.6f}"
        )
        checked += 1
    print(f"  cross-checked {checked} means against stage 08")


def main():
    fold_rows, error_rows, curve_rows, cost_rows, summary = [], [], [], [], []
    edge_rows = []

    for name, stratum in st.load_all().items():
        band = stratum.band_widths[1]
        print(f"\n{'=' * 78}\n{name}: band {band:g}\n{'=' * 78}")

        for mode in ARMS:
            X, y, _ = ru.release_matrix(
                stratum.frame,
                stratum.sequence,
                stratum.predictors,
                st.SENSITIVE,
                band_width=band,
                derived_mode=mode,
            )
            if len(X) <= 20 or y.nunique() < 2:
                print(f"  {mode}: too few usable records")
                continue

            scored = _fit_and_score(X, y)
            folds = scored["folds"]
            folds.insert(0, "derived_mode", mode)
            folds.insert(0, "band_width", band)
            folds.insert(0, "stratum", name)
            fold_rows.append(folds)

            profile = _error_profile(scored["y"], scored["mean_proba"])
            profile.update(
                stratum=name,
                band_width=band,
                derived_mode=mode,
                n_records=len(scored["y"]),
            )
            error_rows.append(profile)

            precision, recall, _ = precision_recall_curve(
                scored["y"], scored["mean_proba"]
            )
            curve_rows.append(
                pd.DataFrame(
                    {
                        "stratum": name,
                        "band_width": band,
                        "derived_mode": mode,
                        "recall": recall,
                        "precision": precision,
                    }
                )
            )

            edge_rows.extend([
                {"stratum": name, **row}
                for row in _band_edge_profile(
                    stratum, band, scored["y"], scored["mean_proba"], mode)
            ])

            cost_rows.append(
                {
                    "stratum": name,
                    "band_width": band,
                    "derived_mode": mode,
                    "n_fits": len(scored["fit_seconds"]),
                    "fit_seconds_mean": float(scored["fit_seconds"].mean()),
                    "fit_seconds_sd": float(scored["fit_seconds"].std(ddof=1)),
                    "fit_cpu_mean": float(scored["fit_cpu_seconds"].mean()),
                    "fit_cpu_sd": float(scored["fit_cpu_seconds"].std(ddof=1)),
                    "predict_ms_per_instance": (
                        scored["predict_seconds_per_instance"] * 1000
                    ),
                    "python_peak_alloc_mb": scored["python_peak_alloc_mb"],
                }
            )

            summary.append(
                {
                    "stratum": name,
                    "band_width": band,
                    "derived_mode": mode,
                    "auc_pr_mean": float(folds.auc_pr.mean()),
                    "auc_pr_sd": float(folds.auc_pr.std(ddof=1)),
                    "auc_pr_min": float(folds.auc_pr.min()),
                    "auc_pr_max": float(folds.auc_pr.max()),
                }
            )

            print(
                f"  {mode}: AUC-PR {folds.auc_pr.mean():.3f} "
                f"± {folds.auc_pr.std(ddof=1):.3f} over {len(folds)} folds, "
                f"recall {profile['recall_positive']:.3f} at t={ERROR_THRESHOLD}, "
                f"{profile['false_negative']} false negatives"
            )

    # R2 describes how far apart the two curves run. Computed here rather than
    # read off the plot: an eyeballed crossing point is a number nothing checks.
    curves = pd.concat(curve_rows)
    grid = np.linspace(0.02, 0.99, 120)
    separation = []
    for corpus in curves.stratum.unique():
        interpolated = {}
        for mode in ARMS:
            arm = curves[(curves.stratum == corpus) & (curves.derived_mode == mode)]
            arm = arm.sort_values("recall")
            interpolated[mode] = np.interp(grid, arm.recall, arm.precision)
        gap = interpolated[dc.BASELINE] - interpolated[dc.PROPOSED]
        separation.append(
            {
                "stratum": corpus,
                "n_grid_points": len(grid),
                "mean_abs_precision_gap": float(np.abs(gap).mean()),
                "max_abs_precision_gap": float(np.abs(gap).max()),
                "share_grid_baseline_above": float((gap > 0).mean()),
            }
        )
    pd.DataFrame(separation).to_csv(RESULTS / "curve_separation.csv", index=False)

    print("\nCross-checking against the confirmatory stage")
    _check_against_stage_08(summary)

    pd.concat(fold_rows).to_csv(RESULTS / "fold_scores.csv", index=False)
    pd.DataFrame(error_rows).to_csv(RESULTS / "error_profile.csv", index=False)
    pd.concat(curve_rows).to_csv(RESULTS / "pr_curves.csv", index=False)
    # Process high-water mark for the whole stage. Unlike the per-arm figures
    # this includes XGBoost's native allocations, but it is monotonic across
    # the run and so cannot be attributed to one arm.
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    cost = pd.DataFrame(cost_rows)
    cost["process_peak_rss_mb"] = round(peak_rss_mb, 1)
    cost.to_csv(RESULTS / "compute_cost.csv", index=False)
    pd.DataFrame(summary).to_csv(RESULTS / "fold_summary.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(RESULTS / "band_edge_profile.csv", index=False)
    print(f"wrote 7 tables to {RESULTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
