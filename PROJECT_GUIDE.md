# Project Guide

Thesis-side companion to the [README](README.md). The README covers what the
code is and how to run it; this file covers how the outputs map onto the
thesis, and the decisions already settled so they are not re-litigated.

Prince Bortey Miller | 22388461 | Supervision Code 289
Supervisor: Dr. Eric Opoku Osei | KNUST, 2026

---

## When the real dataset arrives

Only two files need editing.

1. Put the college file at `data/raw_college_records.xlsx`.
2. **Stage 0, PREP-CELL 2** — set `COLUMN_MAPPING` to the college's actual
   headers and set `TARGET_RULE`.
3. Run Stage 0. This produces `anonymised_records.csv` and the exclusion-flow
   table for the Methods chapter.
4. **Stage 1, CELL 6** — point `REAL_DATA_PATH` at `anonymised_records.csv`
   and comment out the synthetic generator.
5. Run `./run.sh`. Everything lands in `results/real/`, and the educator app
   switches to the real model automatically.

Outstanding before step 1 is possible:

- Licensure outcomes are held by the college but were not supplied linked to
  the registry extraction. A bridge key is needed.
- The course result forms carry no student identifier. Row ordering must be
  confirmed by the registrar in writing before any join, since a silent
  off-by-one would corrupt every downstream number without raising an error.
- How the college's per-course scores map onto the six NMC-LE papers is
  unresolved.

---

## Output to thesis chapter mapping

| Thesis section | Comes from |
|---|---|
| Ch. 3 Methodology, participant flow | Stage 0 exclusion table |
| Ch. 4 Results, descriptives | `eda_descriptives.csv`, fail-rate-by-group tables |
| Ch. 4 Results, model comparison | Stage 1 metrics table, `roc_pr_curves.png` |
| Ch. 4 Results, Table 1 | `comparison_table.csv` (baseline vs E-XGBoost, Δ, p) |
| Ch. 4 Results, explainability | `shap_beeswarm.png`, waterfall and dependence plots |
| Ch. 4 Results, calibration (H4) | `calibration_curves.png` plus Brier scores |
| Ch. 4 Results, fairness | FNR/FPR/EOD tables from Stage 1 |
| Ch. 5 Discussion, H1 | DeLong results |
| Ch. 5 Discussion, H2 | SHAP global ranking against demographics |
| Ch. 5 Discussion, H3 | Trainee and tutor survey Likert results |

---

## Decisions already settled

Do not re-open these without a reason.

- **Imbalance.** A 35–65% failure rate is a natural distribution and gets no
  resampling. Outside that band, `scale_pos_weight` applies.
- **Exclusions.** Absent, deferred, withheld, missing outcome, duplicate, and
  missing-core records are removed and counted.
- **Splits.** Stratified 70/15/15 plus a temporal split on `cohort_year`.
- **Fold count.** Five, not ten. At the expected sample size (300–800 records)
  ten folds leave too few failures per fold for a stable AUC-PR; a trial run
  roughly doubled per-fold SD without producing a significant difference.
  Rationale is recorded at the top of `baseline_xgboost.py`.
- **Engineering verdict.** Either "significant improvement" or "equivalent
  accuracy, leaner and more interpretable" is a valid claim. State whichever
  the real numbers support.
- **Statistical tests.** DeLong for AUC, McNemar for label disagreement,
  Wilcoxon for the engineering comparison, p < 0.05.
- **Synthetic data.** Pipeline testing only. Never in a reported result.
- **Module layout.** Logic lives in flat modules at the repo root, not a `src/`
  package, so the numbered notebooks stay pasteable into Colab.

---

## Status

- [x] Stage 0 built and tested on a mock college file (528 raw, 488 clean, 0 identifier leaks)
- [x] Stage 1 built and verified
- [x] Stage 2 built and verified end to end
- [x] Full chain runs via `./run.sh` on synthetic data
- [x] Educator app built and verified against the exported bundle
- [x] Modules extracted from the notebooks and independently importable
- [ ] HuSSREC approval reference
- [ ] Licensure outcomes linked to academic records
- [ ] Re-run on real data, results final, app serves the real model

---

## Contribution defence

Three claims, stated and then left alone:

1. Original field data, the first Ghanaian nursing-licensure ML dataset.
2. A modern explainable model, XGBoost with SHAP.
3. An engineered model, E-XGBoost, compared against the locked baseline on
   identical folds and Wilcoxon-tested.

On "did you engineer the model or just apply it?": E-XGBoost removes the
features contributing the bottom 5% of SHAP attribution, retrains on identical
splits with identical hyperparameters, and is Wilcoxon-tested against the
baseline. One change only, so any difference is attributable to the pruning.
