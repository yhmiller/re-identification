"""Stage 9. Interpretability, for Methods M17 and Results R6.

The new CAN-DO Methods template requires a named interpretability method and,
for an engineered study, evidence that the engineered component actually
influences predictions rather than being ignored by the model.

For this study the engineered component is not a feature. It is the **source**
from which the derived features are computed, so "is it ignored" has a precise
form: do the published derived features carry weight in the model at all, and
does changing where they are derived from change what the model relies on?

Two things are therefore reported per corpus:

    1. Ranked mean absolute SHAP value for every published column, so the
       derived block can be located against the source columns.
    2. The same ranking under the baseline arm and under the proposed arm, so a
       shift in reliance is visible even where AUC-PR barely moves.

The second is the point. AUC-PR says whether the analysis survives; it does not
say whether the model still leans on the same things afterwards. A release that
holds performance while changing what drives it is a utility finding a single
scalar hides.

Explanations are computed per corpus and never averaged across corpora, as the
template requires and as the no-pooling rule has required throughout.

Run:
    ml_env/bin/python notebooks/09_interpretability.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import deidentify as di  # noqa: E402
import derivation_consistent as dc  # noqa: E402
import risk_utility as ru  # noqa: E402
import strata as st  # noqa: E402

RESULTS = ROOT / "results" / "disclosure"
RESULTS.mkdir(parents=True, exist_ok=True)

# Explanations are computed on held-out predictions, not on training data. A
# model explains its training set too well to be informative about what it
# relies on in use.
EXPLAIN_SEED = ru.BASE_SEED
EXPLAIN_TEST_FRACTION = 0.25


def _release_matrix(stratum, band_width, mode):
    """What the utility model receives, and its column names.

    Delegates to risk_utility.release_matrix so the explanation describes the
    release the study evaluated rather than a second reconstruction of it.
    """
    X, y, columns = ru.release_matrix(
        stratum.frame,
        stratum.sequence,
        stratum.predictors,
        st.SENSITIVE,
        band_width=band_width,
        derived_mode=mode,
    )
    return X, y.astype(int), columns


def explain(stratum, band_width, mode):
    """Mean absolute SHAP value per column, on a held-out partition."""
    import shap
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    X, y, columns = _release_matrix(stratum, band_width, mode)
    if len(X) < 40 or y.nunique() < 2:
        return None

    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=EXPLAIN_TEST_FRACTION,
        stratify=y,
        random_state=EXPLAIN_SEED,
    )
    model = XGBClassifier(**ru.XGB_PARAMS, random_state=EXPLAIN_SEED)
    model.fit(X_train, y_train)

    values = shap.TreeExplainer(model).shap_values(X_test)
    importance = np.abs(values).mean(axis=0)
    total = importance.sum()

    return (
        pd.DataFrame(
            {
                "column": columns,
                "mean_abs_shap": importance,
                "share_of_total": importance / total if total else np.nan,
                "block": [
                    "derived" if c in di.DERIVED_COLUMNS else "source" for c in columns
                ],
                "n_explained": len(X_test),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def main():
    import shap

    rows, summary = [], []
    for name, stratum in st.load_all().items():
        band = stratum.band_widths[1]
        print(f"\n{'=' * 78}\n{name}: band {band:g}\n{'=' * 78}")

        for mode in (dc.BASELINE, dc.PROPOSED):
            table = explain(stratum, band, mode)
            if table is None:
                print(f"  {mode}: too few usable records to explain")
                continue

            table.insert(0, "derived_mode", mode)
            table.insert(0, "band_width", band)
            table.insert(0, "stratum", name)
            rows.append(table)

            derived = table[table.block == "derived"]
            share = float(derived.share_of_total.sum())
            top = table.iloc[0]
            best_derived = derived.iloc[0] if len(derived) else None

            summary.append(
                {
                    "stratum": name,
                    "band_width": band,
                    "derived_mode": mode,
                    "n_columns": len(table),
                    "n_derived_columns": len(derived),
                    "derived_share_of_importance": share,
                    "top_column": top.column,
                    "top_column_block": top.block,
                    "top_derived_column": (
                        None if best_derived is None else best_derived.column
                    ),
                    "top_derived_rank": (
                        None if best_derived is None else int(derived.index[0]) + 1
                    ),
                    "n_explained": int(top.n_explained),
                }
            )

            print(
                f"  {mode}: derived block carries {share:.1%} of total "
                f"importance across {len(derived)} of {len(table)} columns"
            )
            for r in table.head(5).itertuples():
                print(f"      {r.column:<22} {r.mean_abs_shap:.4f}  ({r.block})")

    pd.concat(rows).to_csv(RESULTS / "interpretability_shap.csv", index=False)
    pd.DataFrame(summary).to_csv(RESULTS / "interpretability_summary.csv", index=False)
    print(f"\nshap {shap.__version__}")
    print(f"wrote 2 tables to {RESULTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
