# Project Guide

Thesis-side companion to the [README](README.md). The README covers what the
code is and how to run it; this file covers how the outputs map onto the thesis,
and the decisions already settled so they are not re-litigated.

Prince Bortey Miller | 22388461 | Supervision Code 289
Supervisor: Dr. Eric Opoku Osei | KNUST, 2026

Authoritative record of what has been measured:
[docs/disclosure-risk-findings.md](docs/disclosure-risk-findings.md).

---

## Scope

The study measures **re-identification risk in health professions education
records**, and asks whether the engineered features a machine learning pipeline
derives from those records undo the de-identification applied to their source.

It replaced an earlier study that predicted licensure failure from the same
records. That study could not proceed: the Accra School of Hygiene registrar
never returned the outcome column, and without a dependent variable nothing
could be estimated. The present study has no dependent variable at all, so the
same blockage cannot recur.

Nothing from the earlier scope was deleted. This repository is a clone of the
one that held it, so the history came across intact. The pre-pivot codebase is
at the tag `scope/licensure-prediction`. The outcome-simulation tooling, with
a README explaining what each file was for, is at
the tag `licensure-archive`. Read any of it with
`git show licensure-archive:archive/licensure-outcome/README.md`, or recover
the whole directory with `git checkout licensure-archive -- archive/`.

---

## Corpora

Three, deliberately never pooled. Grading scales, grade-point definitions and
sequence lengths all differ, so a merged corpus would confound every comparison
drawn from it. Specifications live in [strata.py](strata.py).

| | Role | n | Source |
|---|---|---|---|
| `allied_health` | study population | 110 | Accra School of Hygiene, EH/OHS/OT, 2021-2022 |
| `nursing` | study population | 566 | BSc Nursing, 2023/24 and 2024/25, levels 200-400 |
| `public` | replication corpus | 649 | UCI Student Performance (Cortez & Silva, 2008) |

**The public corpus is not part of the study population.** It is Portuguese
secondary school students in a language course: not health, not professional,
not tertiary. It is present to test whether the mechanism reproduces outside the
study population, and for no other purpose. No claim about health records, about
Ghanaian institutions or about Act 843 rests on it.

An earlier decision rejected this corpus as a comparator for licensure
prediction, because Portuguese secondary students do not clear the "genuinely
comparable" bar for a health professions topic. That decision stands. It
addressed a different use: transfer learning for prediction needs comparable
populations, whereas a leak in an order statistic does not. `gpa_min` is the
minimum of the sequence regardless of who the student is.

---

## Getting results

```bash
ml_env/bin/python scripts/consolidate_real_data.py        # allied health workbooks
ml_env/bin/python scripts/consolidate_nursing_data.py     # nursing workbooks
ml_env/bin/python notebooks/04_disclosure_risk.py         # Phases 1 and 2
ml_env/bin/python notebooks/05_derived_feature_experiments.py  # Phase 3
ml_env/bin/python notebooks/06_linkage_attack.py          # Phase 4
ml_env/bin/python notebooks/07_risk_utility_frontier.py   # Phase 5
ml_env/bin/python notebooks/08_confirmatory_tests.py      # confirmatory
ml_env/bin/python figures.py                              # figures
```

Outputs land in `results/disclosure/`. Every output is aggregate. No function in
`disclosure_risk`, `deidentify`, `attack_models` or `risk_utility` returns
record-level results, so no re-identified student can leave the pipeline.

---

## Identifiers

The nursing workbooks carry an Index No column and a Student Name column. They
are the only source in the project that ever did.

Names are dropped at parse time. Index numbers become HMAC-SHA256 digests keyed
on a salt in `local/.nursing_salt`, gitignored and generated once. An unsalted
hash would be worthless here: the identifier space is roughly six hundred values
of the form `NUR-YY-NNN`, enumerable in under a second.

`data/` is gitignored in full and no file under it has ever been tracked. An
assertion in the consolidation script fails the build if an identifier-shaped
column reaches an output.

---

## Output to thesis chapter mapping

| Thesis section | Comes from |
|---|---|
| Ch. 3 Methodology, corpora | `data_quality` and `provenance` sheets, `strata.py` |
| Ch. 3 Methodology, why not leave-one-out | the saturation argument in the findings |
| Ch. 4 Results, baseline risk | `baseline_risk_profiles.csv` |
| Ch. 4 Results, attribution | `solo_identifying_power.csv`, `saturation_curve.csv` |
| Ch. 4 Results, group size | `group_size_effect.csv` |
| Ch. 4 Results, generalisation floor | `precision_sweep.csv` |
| Ch. 4 Results, Experiments A, B, C | `experiment_a/b/c_*.csv` |
| Ch. 4 Results, linkage attack | `linkage_attack.csv` |
| Ch. 4 Results, risk-utility frontier | `risk_utility_frontier.csv` |
| Ch. 5 Discussion, recommended standard | the Pareto frontier |

---

## Decisions already settled

Do not re-open these without a reason.

- **Corpora are never pooled.** Replication across three, not one merged sample.
- **The public corpus is a replication, not a comparator.** Every table labels
  its role.
- **Band widths and recall tolerances are fractions of each corpus's grade
  range**, not absolute values. A band of 0.5 covers an eighth of a four-point
  GPA scale and a fortieth of a twenty-point scale, where it does not even merge
  adjacent integers. Fixing the absolute width made generalisation look
  ineffective on the public corpus when it had never been applied.
- **The sensitive attribute is the bottom 40% of each corpus**, chosen as a
  quantile rather than an absolute grade because the scales are not comparable
  and both l-diversity and t-closeness are prevalence sensitive. Integer grades
  tie, so the cut nearest 40% is used and the prevalence actually achieved is
  reported.
- **Sampling fraction is 1.** Every corpus is a complete population of its
  cohorts, so sample uniqueness equals population uniqueness and prosecutor and
  journalist risk coincide. Most published work must estimate this.
- **Attribution is measured from below, not by leave-one-out.** Uniqueness
  saturates at two attributes, so dropping any single member of a larger set
  changes nothing and every marginal contribution returns zero.
- **Fold design.** Five folds repeated across five seeds, giving 25 paired
  observations. Five folds alone cannot support a claim: the smallest two-sided
  p Wilcoxon can return at n=5 is 0.0625.
- **Metrics are reported against their no-skill floor.** 0.500 for AUC-ROC,
  prevalence for AUC-PR.
- **Imbalance.** A 35-65% positive rate is a natural distribution and gets no
  resampling.
- **Statistical tests.** Risk uses subsampling without replacement at 80%, not
  the ordinary bootstrap: duplicated records are not unique, so resampling with
  replacement changes the estimand rather than perturbing the estimator, and
  asymmetrically between arms. Utility uses paired folds, reported naive and
  with a dependence correction. Effect sizes are the raw differences in each
  metric's own units, unstandardised.
- **Equivalence testing was removed and delta left the hypotheses.** The arms
  differ, so an equivalence test at any defensible margin fails and conveys
  nothing the interval does not. delta survives only as a tolerance a custodian
  brings to the frontier.
- **The utility model receives what the recipient receives.** Generalised source
  columns plus the derived features published alongside them, with the derived
  features computed over the predictor positions only so the target cannot leak
  into them. Passing only the source columns makes the two derivation arms
  identical by construction and produces a finding that is arithmetic rather than
  evidence.
- **Module layout.** Logic lives in flat modules at the repo root, not a `src/`
  package, so the numbered notebooks stay pasteable into Colab.
- **Synthetic data.** Pipeline testing only. Never merged into a corpus to
  inflate n.

---

## Status

Analysis complete. Methods and Results drafted. Four sections still to write.

- [x] Both Ghanaian corpora consolidated, no PII anywhere in the pipeline
- [x] Public replication corpus wired in and labelled as such
- [x] Exploratory phases 1 to 5 complete on all three corpora
- [x] Engineered artefact given one home, `derivation_consistent.py`
- [x] Analysis specification frozen before the confirmatory stage
- [x] Confirmatory comparisons run: subsampling for risk, paired folds with a
      dependence correction for utility
- [x] Nine figure and table artefacts built, regenerated by one command
- [x] Findings documented through Stage 4
- [x] Methods section drafted, Stage 5. 16 subsections, 13 citations, 2,790 words
- [x] Results section drafted, Stage 6. 10 subsections, 4 citations, 1,641 words
- [x] Both sections audited against the marking scheme, Stage 7. Part A passes
- [x] Runtimes measured for the efficiency subsection
- [x] Title and two objectives adopted. The marking scheme closed the question
- [x] Scoping literature search run, and its findings propagated to every
      document that carried the superseded novelty claim
- [ ] **Ethics approval reference number.** `CHRPE/AP/XXX/26` caps the marking
      scheme at 70% against a 90% pass. The single highest-value action left
- [ ] Read the ten sources marked critical in `docs/literature/sources.md`.
      Every citation added from the search rests on an abstract
- [ ] Run the database searches in `docs/literature/protocol.md` before the
      word "systematic" is used anywhere
- [ ] Tell the supervisor the title changed, and why
- [ ] Adviser confirmation that the fold-dependence correction suits this design
- [ ] Discussion, Introduction and literature review, Abstract, Conclusion
- [ ] No tests cover the analysis modules
- [ ] Optional and last: prototype tool, written framework, differential privacy arm

Full plan: [docs/NewDirection/writing-plan.md](docs/NewDirection/writing-plan.md).

---

## Contribution defence

The engineered artefact is **derivation-consistent generalisation**, implemented
in [derivation_consistent.py](derivation_consistent.py): one modification to a
standard de-identification pipeline, in which features derived from a protected
variable are recomputed from the protected values rather than from the originals.

Only the derivation step differs between the two arms. Banding, suppression, the
risk metrics and the utility model are identical, so any difference in disclosure
risk is attributable to the derivation source and nothing else.

- **Baseline.** Standard generalisation. Bands the source variables, publishes
  derived features at original precision. This is what pipelines do today.
- **Proposed.** Recomputes the derived features from the bands.
- **Ablation.** Revert the derivation step. Protection collapses by 98.7% to
  100% across the study population, and by 67% to 89% on the replication corpus.

The claim is transfer, not novelty, and the scoping search in `docs/literature/`
settled that. Released statistics leak (Dinur and Nissim, 2003) and derivation
consistency is standing practice in tabular disclosure control, where agencies
run an additivity-restoring step after perturbation. Neither has been carried
into microdata generalisation for machine learning, and the tabular literature
does not ask what enforcing consistency costs an analyst because nobody there is
fitting a model to the release. This study carries it across and prices it.

The two arms differ in utility, and substantially. An earlier version of this
pipeline reported them as identical, which was an artefact: the utility model was
given only the generalised source columns, which are byte-identical between the
arms, so it never saw the derived features that are the only thing separating
them. Corrected, the model receives what a recipient actually receives.

Derived features computed from unprotected source values restore
approximately all of the analytical utility that generalisation removed, between
99.4% and 109.8% across the three corpora, and approximately all of the
disclosure risk it removed, between 98.7% and 100%. Those two figures agree
because it is the same information seen from two sides. Closing the leak
therefore has a price: recomputing the derived features from the protected
values costs 2.7 to 13.5 AUC-PR points depending on the corpus.

The baseline release is therefore not dominated. It sits on the frontier at the
high-utility, low-protection end. The contribution is the measured trade-off and
the mechanism behind it, not a free fix.
