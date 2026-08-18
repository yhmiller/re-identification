# Archived: licensure-outcome tooling

Moved here on 17 August 2026 when the thesis scope changed from predicting
licensure failure to measuring re-identification risk in the same records.

Everything in this folder existed to stand in for the licensure outcome that
the Accra School of Hygiene registrar never returned. `outcome_model.py`
generated synthetic outcomes at a chosen prevalence so the prediction pipeline
could run before real results arrived. The scripts built the return workbooks
and the prevalence sweep. The tests covered all of it.

The new study has **no dependent variable**. Nothing is predicted, so nothing
needs an outcome, so none of this is reachable from the new pipeline.

## What is here

| Path | Was |
|---|---|
| `outcome_model.py` | Synthetic licensure-outcome generator |
| `scripts/build_synthetic_outcome_return.py` | Built the simulated registrar return |
| `scripts/build_synthetic_alpha.py` | Built the alpha corpus |
| `scripts/compare_prevalence_sweep.py` | Compared runs across simulated prevalences |
| `tests/` | The four test modules covering the above |

## Recovering it

Nothing is deleted. Two routes back:

- The full pre-pivot codebase is tagged: `git checkout scope/licensure-prediction`
- These files individually: `git mv archive/licensure-outcome/<file> <original path>`

## What was deliberately left in place

`transfer_fusion.py` is still imported by `notebooks/02_model_engineering.py`,
which the new study reuses as its utility measurement instrument. Moving it
would break that notebook, so it stays.

`baseline_xgboost.py`, `engineered_xgboost.py`, `stats_validation.py`,
`model_comparison.py`, `schema.py` and `real_data.py` all stay for the same
reason: the new study measures how much model performance survives
de-identification, which means it needs a working model pipeline to measure
against.
