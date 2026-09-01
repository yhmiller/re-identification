"""Stage 12. Sensitivity of the arm comparison to choices held fixed elsewhere.

Four axes, each answering a way the headline comparison could be an artefact of
one unvaried decision:

    the outcome threshold recomputed within each training fold rather than
    across the corpus, since a corpus-wide quantile lets a test-fold label
    depend on training-fold values
    the nine configurations re-run at two further root seeds, since every fold
    assignment and every ensemble descends from one seed and fold variance is
    not seed variance
    a reduced quasi-identifier configuration per corpus, since quasi-identifier
    selection is the largest single lever on uniqueness
    the minimum detectable difference per configuration, so a non-significant
    result can be read as a statement about resolution rather than about effect

Run:
    ml_env/bin/python notebooks/12_robustness.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import derivation_consistent as dc  # noqa: E402
import disclosure_risk as dr  # noqa: E402
import risk_utility as ru  # noqa: E402
import stats_validation as sv  # noqa: E402
import strata as st  # noqa: E402

RESULTS = ROOT / "results" / "disclosure"
EXTRA_SEEDS = (7, 2026)
ALPHA = 0.05


def _paired_fold_scores(X, y, seed):
    """Per-fold AUC-PR at one root seed."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
    splitter = RepeatedStratifiedKFold(
        n_splits=ru.N_CV_FOLDS, n_repeats=ru.N_CV_REPEATS, random_state=seed
    )
    scores = []
    for train, test in splitter.split(X, y):
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        model = XGBClassifier(**ru.XGB_PARAMS, random_state=seed)
        model.fit(X[train], y[train])
        scores.append(
            average_precision_score(y[test], model.predict_proba(X[test])[:, 1])
        )
    return np.array(scores)


def seed_sensitivity(stratum, name):
    """The arm comparison at three root seeds rather than one."""
    rows = []
    for band in stratum.band_widths:
        matrices = {}
        for mode in (dc.BASELINE, dc.PROPOSED):
            matrices[mode] = ru.release_matrix(
                stratum.frame,
                stratum.sequence,
                stratum.predictors,
                st.SENSITIVE,
                band_width=band,
                derived_mode=mode,
            )
        for seed in (ru.BASE_SEED, *EXTRA_SEEDS):
            scored = {}
            for mode in (dc.BASELINE, dc.PROPOSED):
                X, y, _ = matrices[mode]
                if len(X) <= 20 or y.nunique() < 2:
                    scored[mode] = np.array([])
                    continue
                scored[mode] = _paired_fold_scores(X, y, seed)
            base, prop = scored[dc.BASELINE], scored[dc.PROPOSED]
            if not len(base) or len(base) != len(prop):
                continue
            differences = prop - base
            corrected = sv.paired_fold_difference(list(base), list(prop))
            rows.append(
                {
                    "stratum": name,
                    "band_width": band,
                    "seed": seed,
                    "auc_pr_baseline": float(base.mean()),
                    "auc_pr_proposed": float(prop.mean()),
                    "mean_difference": float(differences.mean()),
                    "corrected_p": corrected["corrected_p"],
                    "significant": bool(corrected["corrected_p"] < ALPHA),
                    "folds_below_zero": int((differences < 0).sum()),
                    "n_folds": len(differences),
                }
            )
    return pd.DataFrame(rows)


def quasi_identifier_sensitivity(stratum, name):
    """Uniqueness under a reduced quasi-identifier set as well as the full one.

    The reduced set drops the derived block and keeps the released sequence,
    which is the configuration a cautious custodian would actually publish.
    """
    rows = []
    for band in stratum.band_widths:
        for mode in (dc.BASELINE, dc.PROPOSED, dc.SELECTIVE):
            release = dc.build(stratum.frame, stratum.sequence, band, None, mode)
            full = release.visible_columns
            reduced = list(stratum.sequence)
            for label, columns in (("full", full), ("sequence only", reduced)):
                profile = dr.risk_profile(
                    release.frame, columns, sensitive=st.SENSITIVE
                )
                rows.append(
                    {
                        "stratum": name,
                        "band_width": band,
                        "derived_mode": mode,
                        "qi_set": label,
                        "n_quasi_identifiers": len(columns),
                        "prop_unique": profile.prop_unique,
                        "min_class_size": profile.min_class_size,
                    }
                )
    return pd.DataFrame(rows)


def fold_internal_threshold(stratum, name):
    """The outcome threshold recomputed inside each training fold.

    The corpus-wide quantile of M8 lets a test-fold record's label depend on
    training-fold values. Recomputing the cut on the training fold and applying
    it to the held-out fold removes that dependence. If no verdict moves, the
    construct definition stands on evidence rather than on argument.
    """
    rows = []
    for band in stratum.band_widths:
        scored = {}
        for mode in (dc.BASELINE, dc.PROPOSED):
            X, _, _ = ru.release_matrix(
                stratum.frame,
                stratum.sequence,
                stratum.predictors,
                st.SENSITIVE,
                band_width=band,
                derived_mode=mode,
            )
            last = stratum.frame[stratum.sequence].ffill(axis=1).iloc[:, -1]
            last = last.reindex(X.index)
            if len(X) <= 20 or last.isna().all():
                scored[mode] = np.array([])
                continue
            values, labels = np.asarray(X, dtype=float), np.asarray(last, dtype=float)
            base_y = np.asarray(stratum.frame[st.SENSITIVE].reindex(X.index), dtype=int)
            splitter = RepeatedStratifiedKFold(
                n_splits=ru.N_CV_FOLDS,
                n_repeats=ru.N_CV_REPEATS,
                random_state=ru.BASE_SEED,
            )
            fold_scores = []
            for train, test in splitter.split(values, base_y):
                cut = np.nanquantile(labels[train], st.WEAK_QUANTILE)
                y_train = (labels[train] <= cut).astype(int)
                y_test = (labels[test] <= cut).astype(int)
                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    continue
                model = XGBClassifier(**ru.XGB_PARAMS, random_state=ru.BASE_SEED)
                model.fit(values[train], y_train)
                fold_scores.append(
                    average_precision_score(
                        y_test, model.predict_proba(values[test])[:, 1]
                    )
                )
            scored[mode] = np.array(fold_scores)
        base, prop = scored[dc.BASELINE], scored[dc.PROPOSED]
        if not len(base) or len(base) != len(prop):
            continue
        corrected = sv.paired_fold_difference(list(base), list(prop))
        rows.append(
            {
                "stratum": name,
                "band_width": band,
                "auc_pr_baseline": float(base.mean()),
                "auc_pr_proposed": float(prop.mean()),
                "mean_difference": float((prop - base).mean()),
                "corrected_p": corrected["corrected_p"],
                "significant": bool(corrected["corrected_p"] < ALPHA),
                "n_folds": len(base),
            }
        )
    return pd.DataFrame(rows)


def minimum_detectable_difference(path=RESULTS / "confirmatory_utility.csv"):
    """The smallest AUC-PR difference each design could have detected.

    Reported so that a non-significant result can be read as a statement about
    resolution rather than as evidence of no effect.
    """
    from scipy import stats

    utility = pd.read_csv(path)
    rows = []
    for row in utility.itertuples():
        critical = stats.t.ppf(1 - ALPHA / 2, df=row.n_folds - 1)
        rows.append(
            {
                "stratum": row.stratum,
                "band_width": row.band_width,
                "observed_difference": row.mean_difference,
                "corrected_se": row.corrected_se,
                "minimum_detectable_difference": critical * row.corrected_se,
                "detectable": abs(row.mean_difference) >= critical * row.corrected_se,
            }
        )
    return pd.DataFrame(rows)


def main():
    seeds, qi, folds = [], [], []
    for name, stratum in st.load_all().items():
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}", flush=True)
        s = seed_sensitivity(stratum, name)
        seeds.append(s)
        for row in s.itertuples():
            print(
                f"  band {row.band_width:<5g} seed {row.seed:<6} "
                f"diff={row.mean_difference:+.4f} p={row.corrected_p:.4f} "
                f"{'significant' if row.significant else 'not significant'}"
            )
        qi.append(quasi_identifier_sensitivity(stratum, name))
        folds.append(fold_internal_threshold(stratum, name))

    pd.concat(seeds, ignore_index=True).to_csv(
        RESULTS / "seed_sensitivity.csv", index=False
    )
    pd.concat(qi, ignore_index=True).to_csv(RESULTS / "qi_sensitivity.csv", index=False)
    pd.concat(folds, ignore_index=True).to_csv(
        RESULTS / "fold_internal_threshold.csv", index=False
    )
    minimum_detectable_difference().to_csv(
        RESULTS / "minimum_detectable_difference.csv", index=False
    )
    print(f"\nwrote 4 tables to {RESULTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
