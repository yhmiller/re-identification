# Derivation-Consistent De-identification of Health Professions Education Records

Disclosure Risk and Analytical Utility Under Hybrid Public and Institutional
Academic Data

Do the features a machine learning pipeline derives from student grades undo the
de-identification applied to those grades?

Measurably, yes, and the leak is not free to close.

Publishing derived features at original precision alongside generalised source
variables restores **67% to 100%** of the disclosure risk the generalisation
removed, and **79% to 119%** of the analytical utility it removed. On the two
Ghanaian corpora the risk figure is 98.7% to 100%, so for the study population
the generalisation is, for practical purposes, undone.

Both quantities come back substantially, which is what one expects if the same
information drives both. They are **not** numerically equal configuration by
configuration and are not presented as though they were. The utility ratio has an
unstable denominator, and it exceeds 100% in the nursing corpus because the
baseline release carries eight derived features that the unprotected release does
not.

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
identifies 96.4% of students, and ten of the 105 attribute pairs identify all
of them.

The study measures that residual risk, tests whether generalisation reduces it,
and measures a transfer. That released statistics leak, and that a protected
quantity and its derivations must be treated consistently, are both established
in tabular disclosure control. Neither appears to have been carried into
microdata generalisation for machine learning, where features derived from
unprotected values are routinely published beside a generalised version of those
same values. This study measures that configuration and prices the fix.

The engineered contribution is **derivation-consistent generalisation**. One
modification: derived features are recomputed from the protected values rather
than from the originals. It closes the hole, and the cost of closing it is
measured rather than assumed away.

> **Scope note.** This replaced an earlier study predicting licensure failure
> from the same records, which could not proceed because the outcome column was
> never returned. Nothing was deleted. This repository was cloned from the
> earlier one, so the whole history came across: the pre-pivot codebase is at
> the tag `scope/licensure-prediction`, and the outcome tooling is at
> the tag `licensure-archive`. Recover any of it with
> `git show licensure-archive:archive/licensure-outcome/README.md`.

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
| Fewest attributes reaching 100% unique | 2 | 5 | not reached (67.8% at 6) |
| Uncertainty removed by baseline derivation | 100% | 98.7-100% | 67-88% |
| Source values narrowed to a point | 20.9-33.9% | 34.1-62.8% | 74.1-93.7% |
| Utility restored by baseline derivation | 79-80% | 115-119% | 93-97% |
| Utility cost of closing the leak (AUC-PR) | -0.027 | -0.135 | -0.067 |

### What survives formal testing

Risk reduction is distinguishable from zero in **eight of nine** configurations.
Utility cost in **six of nine**: every nursing and every public configuration,
and none of the three allied health configurations, where corrected p values are
0.890, 0.361 and 0.854.

The allied health null is a power result at 109 modelled records, not evidence of
no cost. Its interval at band 0.50 reaches -0.087, admitting a cost larger than
several observed elsewhere and equally admitting none. The same
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
imprecise recall isolates at most 40% of allied health students. Banding does
not reduce that figure, because the cumulative average is not among the
generalised columns. It does cut the grade-recall scenarios, from 20.9% to 5.5%.

Full record: `docs/disclosure-risk-findings.md`, which is kept local.

---

## Setup

Requires **Python 3.11**. The pinned libraries do not support 3.12 or later.

```bash
brew install python@3.11          # macOS
git clone git@github.com:yhmiller/re-identification.git
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
# scripts/ is held locally with the restricted data, not published
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

## Repository structure

```
re-identification/
│
├── strata.py                    corpus definitions, column roles, band widths
├── deidentify.py                banding, suppression, the 8 derived features
├── derivation_consistent.py     THE ARTEFACT: NONE / BASELINE / PROPOSED / SELECTIVE
├── disclosure_risk.py           uniqueness, k, l, t, adversary risk
├── attack_models.py             interval-propagation reconstruction, linkage
├── risk_utility.py              release configs, cross-validated utility
├── stats_validation.py          subsampling, paired folds, dependence correction
├── figures.py                   every figure, the algorithm box, Table 1
├── run_all.py  run.sh           one-command runner over the nine stages
│
├── notebooks/                   orchestration only, no logic
│   ├── 04_disclosure_risk.py            Phases 1 and 2, baseline risk
│   ├── 05_derived_feature_experiments.py Phase 3, Experiments A, B, C
│   ├── 06_linkage_attack.py             Phase 4, adversarial validation
│   ├── 07_risk_utility_frontier.py      Phase 5, the frontier
│   ├── 08_confirmatory_tests.py         pre-specified confirmatory comparisons
│   ├── 09_interpretability.py           SHAP importance under each arm
│   ├── 10_error_and_curves.py           curves, fold scores, errors, cost
│   ├── 11_gate_remediation.py           four arms, suppression, dominance
│   └── 12_robustness.py                 seed, threshold and QI sensitivity
│
├── scripts/      gitignored. Workbook consolidation and writing tools, held
│               with the restricted data they parse
│
├── docs/         gitignored. The written thesis: manuscript, literature
│               review, topic document, and the record of every measurement
├── tests/        gitignored. The suite that checks every reported number
│
├── data/         gitignored. Raw workbooks and the built corpora
├── local/        gitignored. The HMAC salt
├── proposal/     gitignored. Ethics paperwork, templates, marking schemes
├── results/      gitignored. Every table and figure, all reproducible
└── ml_env/       gitignored. Python 3.11 virtualenv
```

Eight directories are gitignored and must be carried by hand when the repository
is cloned somewhere new. `results/` regenerates from one command and `ml_env/`
rebuilds from `requirements.txt`, but `data/` and `local/` cannot be
reconstructed and are the only irreplaceable things here. `local/.nursing_salt`
in particular: lose it and no nursing pseudonym can ever be reproduced, which
makes every nursing result unrepeatable.

**What a clone of this repository can and cannot do.** It carries the analysis
modules, the numbered notebooks, the figure code and the runner, which is the
whole of the method under test. It does not carry the written thesis, the test
suite or the workbook consolidation scripts. The suite reads `docs/manuscript/`
and `results/` and so could not run from a clone in any case; the consolidation
scripts parse the specific institutional workbook layouts and do nothing without
the restricted data. Both are available on request.

---

## Architecture

Four layers, each depending only on the one above it. No cycles, and no module
in the analysis layer imports anything from a notebook.

```
     strata.py                     what the corpora are
         │
         ▼
     deidentify.py                 how a release is protected
         │
         ▼
  derivation_consistent.py         the one operation under test
         │
    ┌────┴─────────────┬──────────────────┐
    ▼                  ▼                  ▼
disclosure_risk.py  risk_utility.py  attack_models.py
    │                  │                  │
    └────────┬─────────┴──────────────────┘
             ▼
     stats_validation.py            does the difference survive testing
             │
             ▼
     notebooks/04-12  →  results/  →  figures.py  →  docs/manuscript/
```

**Why flat modules and not a `src/` package.** The numbered notebooks stay
pasteable into Colab, which matters because the examiners may want to run one.
A package layout would require an install step before anything ran.

**Why notebooks hold no logic.** Every notebook is orchestration: load a corpus,
call the modules, write a CSV. Anything a result depends on lives in a module
that another notebook can import, so no finding exists in only one place.

**The two-arm design is the whole architecture.** `derivation_consistent.py`
exposes three modes and `build()` differs between BASELINE and PROPOSED on one
line, the line that chooses the derivation source. Banding, suppression, the
risk metrics and the utility model are identical across arms. That is what makes
any measured difference attributable to the derivation source and nothing else,
and it is why the artefact has its own module rather than living as a flag
inside `risk_utility`.

**Where the data flows.** Raw workbooks in `data/` are consolidated once by the
`scripts/`, read by `strata.py`, protected by `deidentify.py`, released by
`derivation_consistent.py`, measured by the three analysis modules, tested by
`stats_validation.py`, and written to `results/disclosure/` as CSV. `figures.py`
reads only those CSVs, never the corpora, so a figure cannot disagree with the
table it came from.

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
Miller, P. B. (2026). Derivation-Consistent De-identification of Health
Professions Education Records: Disclosure Risk and Analytical Utility Under
Hybrid Public and Institutional Academic Data [MSc thesis]. Kwame Nkrumah
University of Science and Technology.
```

---

## License

Shared for academic and research purposes under Creative Commons Attribution
4.0 International (CC BY 4.0). See [LICENSE](LICENSE).
