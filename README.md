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
and examines a mechanism that has received limited empirical attention: features
computed from source variables before those variables are generalised continue to
disclose information about them afterwards.

The engineered contribution is **derivation-consistent generalisation**. One
modification: derived features are recomputed from the protected values rather
than from the originals. It closes the hole at no cost in model utility.

> **Scope note.** This replaced an earlier study predicting licensure failure
> from the same records, which could not proceed because the outcome column was
> never returned. Nothing was deleted: the pre-pivot codebase is tagged
> `scope/licensure-prediction` and the outcome tooling is in
> [archive/licensure-outcome/](archive/licensure-outcome/).

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
git clone https://github.com/yhmiller/xgboost-shap-nursing-licensure.git
cd xgboost-shap-nursing-licensure
./run.sh                          # builds ml_env on first use
```

---

## Run order

```bash
ml_env/bin/python scripts/consolidate_real_data.py             # allied health
ml_env/bin/python scripts/consolidate_nursing_data.py          # nursing
ml_env/bin/python notebooks/04_disclosure_risk.py              # Phases 1 and 2
ml_env/bin/python notebooks/05_derived_feature_experiments.py  # Phase 3
ml_env/bin/python notebooks/06_linkage_attack.py               # Phase 4
ml_env/bin/python notebooks/07_risk_utility_frontier.py        # Phase 5
```

Outputs land in `results/disclosure/`.

---

## Modules

| File | Role |
|---|---|
| [strata.py](strata.py) | the three corpora and their column roles |
| [disclosure_risk.py](disclosure_risk.py) | uniqueness, k-anonymity, l-diversity, t-closeness, adversary risk |
| [deidentify.py](deidentify.py) | banding, suppression, the single derived-feature definition |
| [attack_models.py](attack_models.py) | interval-propagation reconstruction, linkage under side knowledge |
| [risk_utility.py](risk_utility.py) | release configurations, cross-validated utility |
| `baseline_xgboost.py`, `engineered_xgboost.py` | retained from the earlier scope as the utility measurement instrument |

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
