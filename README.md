# Re-identification Risk in Health Professions Education Records

Do the features a machine learning pipeline derives from student grades undo the
de-identification applied to those grades?

Measurably, yes, and the leak is not free to close.

Publishing derived features at original precision alongside generalised source
variables restores **98.7% to 100%** of the disclosure risk the generalisation
removed, and **99.4% to 109.8%** of the analytical utility it removed. Those two
figures agree because it is the same information seen from two sides. The
generalisation is, for practical purposes, undone.

Recomputing the derived features from the protected values closes the leak, and
costs **2.7 to 13.5 AUC-PR points** depending on the corpus. That is a genuine
privacy-utility trade-off, not a free fix, and the frontier is what a custodian
chooses from.

Prince Bortey Miller | 22388461 | MSc Health Informatics, KNUST, 2026
Supervisor: Dr. Eric Opoku Osei

---

## What this is

Health training institutions share student academic records after deleting names
and index numbers, and treat what remains as anonymous. It is not. A student's
grade sequence is a fingerprint: in the allied health corpus, CGPA alone
identifies 96.4% of students, and any two attributes identify all of them.

The study measures that residual risk, tests whether generalisation reduces it,
and measures a transfer. That released statistics leak, and that a protected
quantity and its derivations must be treated consistently, are both established
in tabular disclosure control. Neither appears to have been carried into
microdata generalisation for machine learning, where features derived from
unprotected values are routinely published beside a generalised version of those
same values. This study measures that configuration and prices the fix.

The engineered contribution is **derivation-consistent generalisation**. One
modification: derived features are recomputed from the protected values rather
than from the originals. It closes the hole at no cost in model utility.

> **Scope note.** This replaced an earlier study predicting licensure failure
> from the same records, which could not proceed because the outcome column was
> never returned. Nothing was deleted. This repository was cloned from the
> earlier one, so the whole history came across: the pre-pivot codebase is at
> the tag `scope/licensure-prediction`, and the outcome tooling is at
> `f00df56:archive/licensure-outcome/`. Recover any of it with
> `git show <commit>:<path>`.

---

## Corpora

Three, never pooled. Scales, grade-point definitions and sequence lengths differ,
so merging them would confound every comparison.

| | Role | n | Source |
|---|---|---|---|
| `allied_health` | study population | 110 | Accra School of Hygiene, EH/OHS/OT, 2021-2022 |
| `nursing` | study population | 566 | BSc Nursing, 2023/24 and 2024/25, levels 200-400 |
| `public` | replication corpus | 649 | UCI Student Performance (Cortez & Silva, 2008) |

The public corpus is **not** part of the study population. It is Portuguese
secondary school students in a language course, present only to test whether the
mechanism reproduces outside the study population. No claim about health records
or about Ghanaian institutions rests on it. See [strata.py](strata.py).

---

## Headline findings

| | allied_health | nursing | public |
|---|---|---|---|
| Unique on full quasi-identifier set | 100.0% | 100.0% | 85.7% |
| Unique on CGPA alone | 96.4% | 36.2% | n/a |
| Attributes needed to reach 100% unique | 2 | 5 | 6 |
| Uncertainty removed by baseline derivation | 100% | 98.7-100% | 67-89% |
| Source values narrowed to a point | 20.9-33.9% | 34.1-62.8% | 74.1-93.7% |
| Utility restored by baseline derivation | 109.8% | 99.4% | 99.5% |
| Utility cost of closing the leak (AUC-PR) | -0.027 | -0.135 | -0.067 |

### What survives formal testing

Risk reduction is distinguishable from zero in **eight of nine** configurations.
Utility cost in **six of nine**: every nursing and every public configuration,
and none of the three allied health configurations, where corrected p values are
0.890, 0.361 and 0.854.

The allied health null is a power result at 109 modelled records, not evidence of
no cost. Its widest interval spans [-0.087, +0.033], consistent with a cost as
large as any observed elsewhere and equally consistent with none. The same
corpus proved too small for generalisation to protect at all, so one sample-size
story appears twice.

Monotonicity is not claimed. Across the evaluated configurations wider
generalisation generally produced greater risk reduction accompanied by greater
utility loss, with two exceptions.

Group size protects: Spearman **-0.893** across 11 groups of the study
population spanning 5 to 238 students, **-0.880** across all 13 groups spanning
5 to 423.

Uniqueness figures describe records within each defined institutional
population under a specified quasi-identifier configuration. They are an upper
bound on risk, not a re-identification probability.

Realisable risk is far below theoretical uniqueness. An attacker with plausible
imprecise recall isolates at most 40% of allied health students, and under 6%
once the sequence is banded.

Full record: [docs/disclosure-risk-findings.md](docs/disclosure-risk-findings.md).

---

## Setup

Requires **Python 3.11**. The pinned libraries do not support 3.12 or later.

```bash
brew install python@3.11          # macOS
git clone <remote URL>
cd re-identification
./run.sh                          # builds ml_env on first use, then runs everything
```

---

## Run order

`./run.sh` runs the five stages and rebuilds every figure. To run a stage on
its own, which is what the writing process actually needs, call it directly.
The corpus builders are separate because they read `data/` and only need
re-running when the source workbooks change.

```bash
ml_env/bin/python scripts/consolidate_real_data.py             # allied health
ml_env/bin/python scripts/consolidate_nursing_data.py          # nursing
ml_env/bin/python notebooks/04_disclosure_risk.py              # Phases 1 and 2
ml_env/bin/python notebooks/05_derived_feature_experiments.py  # Phase 3
ml_env/bin/python notebooks/06_linkage_attack.py               # Phase 4
ml_env/bin/python notebooks/07_risk_utility_frontier.py        # Phase 5
ml_env/bin/python notebooks/08_confirmatory_tests.py           # confirmatory tests
ml_env/bin/python figures.py                                   # figures
```

Outputs land in `results/disclosure/`, figures in
`results/disclosure/figures/`. Both are gitignored: every artefact is one
command away, so no stale figure can drift out of step with the table it came
from.

---

## Modules

| File | Role |
|---|---|
| [strata.py](strata.py) | the three corpora and their column roles |
| [disclosure_risk.py](disclosure_risk.py) | uniqueness, k-anonymity, l-diversity, t-closeness, adversary risk |
| [deidentify.py](deidentify.py) | banding, suppression, the single derived-feature definition |
| [attack_models.py](attack_models.py) | interval-propagation reconstruction, linkage under side knowledge |
| [derivation_consistent.py](derivation_consistent.py) | the named artefact: NONE, BASELINE, PROPOSED release modes |
| [risk_utility.py](risk_utility.py) | release configurations, cross-validated utility |
| [stats_validation.py](stats_validation.py) | subsampling for risk, paired fold comparison for utility |
| [figures.py](figures.py) | every manuscript figure, algorithm box, and Table 1 |

Logic lives in flat modules at the repo root so the numbered notebooks stay
pasteable into Colab.

---

## Data and privacy

Every output is aggregate. No function returns record-level results, so no
re-identified student can leave the pipeline.

The nursing workbooks are the only source that ever carried direct identifiers.
Names are dropped at parse time. Index numbers become HMAC-SHA256 digests keyed
on a salt in `local/.nursing_salt`, gitignored and generated once. An unsalted
hash would be worthless: the identifier space is roughly six hundred values of
the form `NUR-YY-NNN`, enumerable in under a second.

`data/` is gitignored in full and no file under it has ever been tracked. An
assertion in the consolidation script fails the build if an identifier-shaped
column reaches an output.

The attack experiments simulate side knowledge from within the corpora. **No
attempt is made to identify any real, named individual at any point**, and no
re-identified record is reported.

---

## Ethics

Approved by the KNUST Committee on Human Research, Publications and Ethics.
Reference number to be inserted.

The study processes records already held under an existing data-sharing
arrangement. No new data was collected. Findings go to the institutions before
publication, with remediation guidance.

---

## Requirements

```
python 3.11
xgboost==2.0.3
scikit-learn==1.4.2
shap==0.44.1
pandas
numpy
scipy
matplotlib
```

Full list in [requirements.txt](requirements.txt).

---

## Citation

```
Miller, P. B. (2026). Re-identification Risk in Ghanaian Health Professions
Education Records: Do Derived Features Undermine De-identification?
[MSc thesis]. Kwame Nkrumah University of Science and Technology.
```

---

## License

Shared for academic and research purposes under Creative Commons Attribution
4.0 International (CC BY 4.0). See [LICENSE](LICENSE).
