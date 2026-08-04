# Project Guide

Thesis-side companion to the [README](README.md). The README covers what the
code is and how to run it; this file covers how the outputs map onto the
thesis, and the decisions already settled so they are not re-litigated.

Prince Bortey Miller | 22388461 | Supervision Code 289
Supervisor: Dr. Eric Opoku Osei | KNUST, 2026

Two companion records carry the detail behind the decisions below:
[real-data-migration-findings.md](docs/real-data-migration-findings.md) is
authoritative on the data, and
[literature-review-findings.md](docs/literature-review-findings.md) is
authoritative on sources.

---

## Study population

The delivered records are **Accra School of Hygiene, Korle-Bu**: 110 students
across Environmental Health, Occupational Health and Safety, and Occupational
Therapy, cohorts 2021 and 2022. These are allied health professions regulated
by Ghana's Allied Health Professions Council, not the Nursing and Midwifery
Council.

The research question, the method and the engineering contribution are
unchanged by this. The title, the examining council and the population framing
are not. Draft in the allied-health framing from the first sentence rather than
writing nursing prose and retrofitting it.

---

## Getting real results

Data preparation is two scripts, not Stage 0.

```bash
ml_env/bin/python scripts/consolidate_real_data.py   # workbooks -> tidy dataset
ml_env/bin/python scripts/build_model_dataset.py     # tidy -> model-ready
./run.sh                                             # Stage 1 + Stage 2
```

The pipeline detects the outcome column automatically. While `licensure_fail` in
`data/model_dataset_2021_2022.xlsx` is empty it runs on pilot data and writes
to `results/synthetic/`; once the college populates it, the same command trains
on real records and writes to `results/real/`. No flags, no edits.

Adding cohorts means dropping new workbooks in, pointing `SOURCE_DIR` in
`consolidate_real_data.py` at them, and re-running both scripts. Student
identifiers derive from cohort, programme and source row, so existing ones stay
stable and outcomes already collected do not shift.

### Blocking

- **The outcome column is empty.** The college will supply it for 2021/22.
- **One examination or three?** EH, OHS and OT may sit different licensure
  examinations under different councils. If so the outcome is three variables,
  not one, with 27, 22 and 10 students behind them. This is a schema question,
  not a wording question, and it must be answered before the column is filled.
- **Row-order confirmation.** Identifiers are generated, so the college can
  only align outcomes by source row order. Get the registrar to confirm in
  writing that their result file preserves it. A silent off-by-one would
  corrupt every downstream number without raising an error.
- **An AHPC pass-rate source.** Blocks Chapter 2 §2.2 and the Chapter 1
  baseline figure. The council publishes pass lists at `ahpc.gov.gh`.

---

## Output to thesis chapter mapping

| Thesis section | Comes from |
|---|---|
| Ch. 3 Methodology, participant flow | `data_quality` sheet of the consolidated workbook |
| Ch. 3 Methodology, reporting adherence | TRIPOD+AI statement (Collins et al., 2024) |
| Ch. 4 Results, descriptives | `eda_descriptives.csv`, fail-rate-by-group tables |
| Ch. 4 Results, model comparison | Stage 1 metrics table, `roc_pr_curves.png` |
| Ch. 4 Results, Table 1 | `comparison_table.csv` (baseline vs E-XGBoost, Δ, p, d) |
| Ch. 4 Results, selection bias | nested vs unnested arms, `unnested_metrics.json` |
| Ch. 4 Results, ranking comparator | `gain_pruned_metrics.json` |
| Ch. 4 Results, feature stability | `feature_stability.csv` |
| Ch. 4 Results, explainability | `shap_beeswarm.png`, waterfall and dependence plots |
| Ch. 4 Results, calibration (H4) | `calibration_curves.png` plus Brier skill scores |
| Ch. 4 Results, fairness | FNR/FPR/EOD by programme from Stage 1 |
| Ch. 5 Discussion, H1 | DeLong results |
| Ch. 5 Discussion, H2 | see the note under Hypotheses below |
| Ch. 5 Discussion, H3 | Trainee and tutor survey Likert results |

---

## Decisions already settled

Do not re-open these without a reason.

- **Imbalance.** A 35–65% failure rate is a natural distribution and gets no
  resampling. Outside that band, `scale_pos_weight` applies.
- **Fold design.** Five folds repeated across five seeds, giving 25 paired
  observations. Five folds alone cannot support the claim: the smallest
  two-sided p Wilcoxon can return at n=5 is 0.0625, so significance is
  unreachable regardless of effect size. Repeating rather than raising the fold
  count keeps test folds the same size, which matters at n=110. Rationale is at
  the top of `baseline_xgboost.py`.
- **Feature selection is nested inside every fold.** Ranking on the full
  dataset and then cross-validating lets selection see held-out labels. On the
  pilot corpus that inflated the AUC-PR gain from +0.0011 to +0.0057, so 81% of
  the apparent improvement was bias (Vabalas et al., 2019). Both arms still
  run; the gap between them is a reported result.
- **E-XGBoost inherits the baseline's hyperparameters, and this was tested.** An
  arm that re-tuned on the pruned feature set under the same protocol (same
  grid, same scoring, tuned on the same training partition) selected a deeper,
  faster model and performed *worse*: AUC-ROC 0.7055 against 0.7202, AUC-PR
  0.5015 against 0.5253 on the simulated corpus. Freezing the hyperparameters
  therefore does not handicap the engineered model, it helps it, and the
  one-change rule costs nothing. The arm was removed from the pipeline to keep
  the comparison simple; this record is the evidence if an examiner asks
  whether the engineered model was disadvantaged by design.
- **Engineering verdict.** Equivalence, not improvement. E-XGBoost matches the
  baseline on 15 fewer predictors, with every effect size negligible. Claim
  parsimony, not performance.
- **Calibration.** Isotonic above 200 validation points, Platt below, fitted on
  the validation fold only. It cannot change ranking by construction. Justified
  by Van Calster et al. (2019): a well-discriminating but miscalibrated model
  cannot support an "act if risk exceeds X%" rule, which is exactly what the
  app does.
- **Metrics are reported against their no-skill floor.** 0.500 for AUC-ROC,
  prevalence for AUC-PR, p(1−p) for Brier. Without this an AUC-PR of 0.74 at a
  prevalence of 0.35 reads as far better than it is.
- **Statistical tests.** DeLong for AUC, McNemar for label disagreement,
  Wilcoxon with Cohen's d for the engineering comparison, α = 0.05. Report d
  alongside every p regardless of the significance outcome.
- **Synthetic data.** Pipeline testing and Option 2 replication only. Never
  merged into a training set to inflate n. The route to 300 records is more
  real cohorts.
- **Module layout.** Logic lives in flat modules at the repo root, not a `src/`
  package, so the numbered notebooks stay pasteable into Colab.

---

## Hypotheses, as they now stand

H2 as written compares academic indicators against demographic variables in
SHAP importance. **The college releases no demographics**, so there is nothing
to compare against and the hypothesis is not testable as stated.

The supported reframe, which the ablation already evidences: test dominance
*within* the academic indicators. Withholding the grade distribution costs
0.187 AUC-PR and CGPA alone reaches only 0.3899 against 0.7222 for the full
set. Put this to the supervisor rather than leaving H2 to fail silently.

---

## Status

- [x] Real records consolidated: 110 students, 5,095 grade records, 282 courses
- [x] Deterministic identifiers generated, no PII anywhere in the pipeline
- [x] Schema migrated to the semester-GPA feature set (36 features)
- [x] Pipeline auto-detects real data and falls back to pilot until outcomes land
- [x] Feature selection nested inside the CV loop
- [x] Gain-ranked comparator arm on identical folds
- [x] Probability calibration on the validation fold
- [x] App updated for the new schema, verified end to end
- [ ] HuSSREC approval reference
- [ ] Licensure outcomes supplied by the college
- [ ] One examination or three, confirmed by the registrar
- [ ] Additional cohorts towards n = 300
- [ ] Chapter 2 drafted in the allied-health framing
- [ ] Manuscripts reframed from nursing to allied health

---

## Contribution defence

Three claims, stated and then left alone:

1. Original field data, the first Ghanaian health-professions licensure dataset
   assembled for machine learning.
2. A modern explainable model, XGBoost with SHAP.
3. An engineered model, E-XGBoost, compared against a locked baseline on
   identical folds with selection nested inside every fold.

On "did you engineer the model or just apply it?": E-XGBoost removes the
features below the 95% cumulative SHAP threshold, retrains on identical splits
with identical hyperparameters and seeds, and is Wilcoxon-tested against the
baseline. One change only, so any difference is attributable to the pruning.

Three follow-ups worth rehearsing, because they are the ones an examiner who
knows the field will actually ask:

- **"Isn't SHAP-based feature selection already established?"** Yes. Powershap
  (Verhaeghe et al., 2023) and LLpowershap (Madakkatel & Hyppönen, 2024)
  formalise it, and Wang et al. (2024) report importance-based selection
  beating it. The claim is not invention. It is the first evaluation of it in
  health-professions licensure prediction, on a Sub-Saharan African cohort,
  under a severe sample-size constraint. The gain-ranked arm answers Wang
  directly: it needs 29.4 features to SHAP's 21.0 for the same discrimination.
- **"Why prune at all if it doesn't improve accuracy?"** Events per predictor
  parameter is roughly 1.0 at n=110 against 36 parameters. Riley et al. (2019)
  make reducing parameters the textbook response. Pruning is a methodological
  necessity here, not an interpretability preference.
- **"Did feature selection see your test data?"** No, and the repository shows
  the unnested arm alongside the nested one to quantify what it would have cost.
