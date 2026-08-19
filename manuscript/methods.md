# 2. Materials and Methods

*Derivation-Consistent De-identification of Health Professions Education Records: Disclosure Risk and Analytical Utility Under Hybrid Public and Institutional Academic Data*

> **UNFILLED PLACEHOLDER — MUST BE REPLACED BEFORE SUPERVISOR SUBMISSION.**
> The ethics approval reference in Section 2.4 reads `CHRPE/AP/XXX/26`. Approval
> is granted; the reference has not yet been recorded. Under the marking scheme
> any unfilled placeholder caps the score at 70%, so this draft cannot pass until
> the real reference replaces it. This is the only placeholder in the section.

**Study objectives.** (1) To develop a derivation-consistent generalisation
procedure for student academic records, in which features derived from a
generalised variable are recomputed from the protected values rather than from
the originals. (2) To quantify the disclosure risk and the analytical utility of
that procedure against standard generalisation, using hybrid public and
institutional academic data.

This study addressed an information pathway that de-identification practice
leaves open: features derived from a source variable at its original precision,
released alongside a generalised version of that same variable. A
derivation-consistent generalisation procedure was constructed, in which such
features are recomputed from the protected values. Within the F/M/R/S
engineering taxonomy this is a **Modify [M]** operation, altering one step of an
otherwise conventional generalisation pipeline rather than introducing a new
component. The procedure was evaluated on two corpora of Ghanaian health
professions education records and replicated on a public academic corpus.

## 2.1 Study design and system overview

The framework comprised four phases: corpus construction and de-identification;
release construction under a generalisation procedure with a configurable
derivation source; disclosure risk measurement and adversarial evaluation; and
analytical utility measurement. The architecture is illustrated in Figure 1.

The baseline was standard generalisation (Sweeney, 2002): source variables are
banded, and the derived features an analyst requires are published alongside them
at original precision. The single engineering contribution altered one step, the
source from which those features are derived. Banding, suppression, risk
metrics, model, hyperparameters and fold assignments were held identical, so any
measured difference is attributable to the derivation source alone.

![Figure 1](../results/disclosure/figures/figure_1_architecture.png)

*Figure 1. The four study phases. Institutional records and the public
replication corpus enter separately, pass through de-identification and release
construction, and are evaluated for disclosure risk and analytical utility before
the risk-utility characterisation.*

The study has no dependent variable. Disclosure risk is a property of a released
set rather than an outcome to be modelled, so no outcome data was required.

## 2.2 Research setting and data acquisition

### 2.2.1 Public dataset component

The Student Performance dataset (Cortez and Silva, 2008) was retrieved from the
UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/320/student+performance)
on 14 August 2026 under a Creative Commons Attribution 4.0 International licence
(DOI: 10.24432/C5TG7T). The Portuguese language course file comprised 649
student records described by 33 attributes, of which three are period grades on
a 0 to 20 scale and four are demographic attributes used as quasi-identifiers.
Integrity was confirmed by row and column count against the repository
description; the file contains no missing values.

This corpus is a **replication corpus, not part of the study population**: its
students are secondary school pupils in a language course. It is present to test
whether the mechanism reproduces beyond the study population, and carries the
study's only demographic quasi-identifiers. No claim about health records or
Ghanaian data protection law rests upon it.

### 2.2.2 Primary institutional record component

Two institutional record sets were obtained under an existing data-sharing
arrangement, following ethical approval described in Section 2.4. Both are
system-of-record academic transcripts rather than survey instruments.

The allied health set comprised 110 students of Environmental Health,
Occupational Health and Safety, and Occupational Therapy at the Accra School of
Hygiene, cohorts 2021 and 2022, described by 15 candidate quasi-identifiers: a
six-position grade sequence, a credit-weighted cumulative average, credit and
course counts, and grade counts. No demographic variables were released.

The nursing set comprised 566 BSc Nursing students, academic years 2023/24 and
2024/25, levels 200 to 400, from twelve score sheets and described by 18
candidate quasi-identifiers on an A to E scale with plus modifiers. No credit
hours are recorded, so the position grade point is an unweighted mean. Position 1
is level 200 semester 1, since level 100 is not on file, and no student is
observed in more than four of six positions.

All identifiers were irreversibly pseudonymised before analysis by the method
given in Section 2.4.

## 2.3 Hybrid data strategy: cross-validation replication

Of the four fusion strategies available (early, late, intermediate-transfer, and
cross-validation), **cross-validation replication** was selected, in the specific
form in which no corpus contributes to the fitting of anything applied to
another. The identical procedure was executed independently on each corpus and
reported separately. Figure 2 illustrates the design.

The mechanism is stated precisely rather than as combination. Each corpus C_i
produces its own release pair, risk estimates and utility estimates. No parameter
fitted on C_i is applied to C_j, no records are concatenated, and no pooled
corpus is formed at any point. The public corpus serves as the out-of-domain
replication, separating properties of the procedure from properties of a
setting.

![Figure 2](../results/disclosure/figures/figure_2_replication.png)

*Figure 2. Cross-validation replication. The identical procedure is applied
independently to each corpus and reported separately. There is no fusion point,
because the corpora are not commensurable.*

Early, late and intermediate-transfer fusion were rejected because the corpora
are not commensurable in grade scale, grade-point definition or sequence length,
so a pooled corpus would attribute to the procedure differences arising from the
measurement scales. The final inventory is three separate corpora of n = 110, 566
and 649, with sensitive-attribute prevalence 40.9%, 40.3% and 46.4% and
negative-to-positive class ratios 1.444:1, 1.482:1 and 1.156:1.

## 2.4 Ethical compliance and data governance

**Path A, full institutional ethics.** The study was approved by the KNUST
Committee on Human Research, Publications and Ethics (Reference:
`CHRPE/AP/XXX/26`) in accordance with the Declaration of Helsinki (World Medical
Association, 2013) and the Ghana Data Protection Act, 2012 (Act 843).

Records were already held under an institutional data-sharing arrangement and no
new data was collected. Written confirmation was obtained that the arrangement
extends to secondary use for this research question, since consent for an
outcome-prediction study does not automatically cover a disclosure-risk study.

The nursing score sheets were the only source carrying direct identifiers, an
index number and a student name. Names were discarded during parsing. Index
numbers were replaced by HMAC-SHA256 digests keyed on a 256-bit salt generated
once and held outside version control, truncated to 48 bits for the released key.
An unkeyed hash would offer no protection, the identifier space comprising
roughly six hundred values of known format (Narayanan and Shmatikov, 2008). The
digest is deterministic because the sheets are organised by year and level, so a
sequence can only be assembled by linking the same individual across two years.
An assertion fails the build if any identifier-shaped column reaches an output.

Processed data were held on an encrypted, access-controlled workstation. All
reported outputs are aggregate; no record-level result is produced, exported or
logged, and no attempt was made to identify any real, named individual.

## 2.5 Data preprocessing pipeline

The allied health workbooks were consolidated by locating grade-matrix headers,
reading per-course grade, credit and grade-point triplets, and reconciling them
against the final-summary sheet. The nursing sheets were parsed as repeating
seven-row blocks, with course codes from each header and scores two rows below.
Sequences were assembled by mapping each (level, semester) pair to a position
and, for nursing, linking students across academic years on their pseudonym.

Missing values were retained rather than imputed. Sequence-cell missingness was
0.9%, 53.8% and 0.0% across the three corpora, the nursing figure reflecting that
only two academic years were available. Imputing an absent position would
fabricate an observation a recipient would not possess and would alter the
equivalence-class structure risk is computed over, so missing positions propagate
to the risk calculation, where records sharing a gap are grouped rather than
separated.

For utility evaluation only, listwise deletion was applied to the predictor set,
retaining 109 of 110, 421 of 566 and 649 of 649 records.

No scaling, encoding or resampling was applied. Gradient-boosted trees handle
mixed feature scales natively (Chen and Guestrin, 2016), so no scaler is fitted
and the question of fold-wise scaler leakage does not arise. Class ratios of
1.444:1, 1.482:1 and 1.156:1 fall inside the band treated as natural for this
study, so no synthetic oversampling was applied and the class distribution of
every partition is the observed one.

## 2.6 Feature engineering

Eight features were derived from each grade sequence, reproducing the set a
machine learning pipeline over academic records conventionally constructs. For a
sequence **X** = (x₁ … x_p) with h = ⌊p/2⌋, the construction rules are given in
Table 1.

| Feature | Construction rule | Type | Constrains X |
|---|---|---|---|
| `gpa_min` | min(**X**) | Continuous | Yes, exact order statistic |
| `gpa_max` | max(**X**) | Continuous | Yes, exact order statistic |
| `gpa_mean` | (1/p) Σ xᵢ | Continuous | Yes, fixes Σ xᵢ |
| `gpa_first_half` | (1/h) Σ_{i≤h} xᵢ | Continuous | Yes, fixes a partial sum |
| `gpa_final_half` | (1/(p−h)) Σ_{i>h} xᵢ | Continuous | Yes, fixes a partial sum |
| `gpa_trend` | gpa_final_half − gpa_first_half | Continuous | No, linear combination of the above |
| `gpa_consistency` | √( (1/p) Σ (xᵢ − gpa_mean)² ) | Continuous | No, non-linear |
| `n_weak_positions` | \|{ i : xᵢ < 2.0 }\| | Integer | No, threshold count |

*Table 1. Derived feature dictionary. "Constrains X" marks the five features that
bound the source values arithmetically and are therefore available to the
reconstruction procedure of Section 2.11.*

One function computes this feature set wherever it is required, so a difference
in results cannot arise from a difference in how the features were computed.

## 2.7 Baseline model specification

The baseline is standard interval generalisation as implemented in established
disclosure-control practice (Sweeney, 2002), applied unmodified. For a source
value x and band width w, the published value is

    g_w(x) = ⌊ x / w ⌋ · w

so that a recipient can infer only

    x ∈ [ g_w(x), g_w(x) + w )

where ⌊·⌋ denotes the floor function and w > 0 is the band width. The baseline
release publishes g_w(**X**) together with the Table 1 features computed from the
original **X** at full precision, which is what a pipeline produces when
generalisation is applied to source columns and the derivation step is left
untouched.

Band widths were fractions of each corpus's grade range rather than absolute
values, at 0.0625, 0.125 and 0.25, giving 0.25, 0.5 and 1.0 on the four-point
scales and 1.25, 2.5 and 5.0 on the twenty-point scale. An absolute width is not
comparable across scales: 0.5 covers an eighth of a four-point range but a
fortieth of a twenty-point one, where it fails to merge adjacent integers and so
generalises nothing.

## 2.8 Model engineering contribution: derivation-consistent generalisation

The proposed procedure implements a **Modify [M]** operation on the release step
of the Section 2.7 baseline. The principle it applies is not new in itself:
tabular disclosure control already requires that a perturbed table and the
quantities derived from it be made consistent, through an additivity-restoring
step applied after perturbation (Australian Bureau of Statistics, 2013). The
operation defined here carries that requirement into microdata generalisation,
where it is not standard practice. 

Let **X** ∈ ℝ^{n×p} denote the source sequence matrix, g_w the elementwise
generalisation operator of Section 2.7 at band width w, and d : ℝ^{n×p} → ℝ^{n×8}
the derivation function whose components are given in Table 1. The two releases
are

    R_baseline = [ g_w(**X**) | d(**X**) ]
    R_proposed = [ g_w(**X**) | d( g_w(**X**) ) ]

where [ · | · ] denotes horizontal concatenation, **X** is the matrix of original
source values, g_w(**X**) is the generalised matrix published to the recipient,
and d(·) is the eight-column derived feature block. The releases differ in the
argument of d and in nothing else, publishing the same generalised columns and
the same derived feature names. Figure 3 contrasts the two constructions with the
modified step marked, and Algorithm 1 gives the procedure with that step
identified at line 3.

![Figure 3](../results/disclosure/figures/figure_3_baseline_vs_proposed.png)

*Figure 3. The two release constructions. Generalisation, concatenation and the
published column set are identical; the two arms differ only in the argument
supplied to the derivation function, marked [M].*

```
Algorithm 1: Derivation-consistent generalisation

Input:  source matrix X (n x p), band width w, suppression threshold k,
        derivation function d, derivation source mode m in {baseline, proposed}
Output: release R

1:  G <- g_w(X)                                band each source value
2:  G, n_suppressed <- suppress(G, k)          blank classes below k
3:  S <- X if m = baseline else G              <- [M] THE MODIFIED STEP
4:  D <- d(S)                                  derive the published features
5:  R <- [ G | D ]                             concatenate and release
6:  return R
```

*Algorithm 1. Only line 3 differs between the two arms. Lines 1, 2, 4 and 5 are
identical, so any measured difference is attributable to the derivation source
alone.*

## 2.9 Training configuration and computing environment

Experiments were executed on an Apple silicon workstation (arm64, 16 GB RAM)
under macOS, using Python 3.11.15 with numpy 1.26.4, pandas 2.1.4, scikit-learn
1.4.2, xgboost 2.0.3, scipy 1.13.0 and matplotlib 3.8.4.

The utility model was an XGBoost classifier (Chen and Guestrin, 2016) held at a
fixed configuration across every release and corpus, given in Table 2.

| Parameter | Value |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 3 |
| `learning_rate` | 0.05 |
| `subsample` | 0.9 |
| `colsample_bytree` | 0.9 |
| `objective` | binary logistic |
| `eval_metric` | logloss |
| `tree_method` | hist |
| Early stopping | not used; boosting rounds fixed at 300 |
| Regularisation | library defaults, unmodified |
| `random_state` | 42 |
| Resampling seed | 20260819 |

*Table 2. Model and protocol configuration, held identical across all releases
and corpora so that the only variable across evaluation rows is the data.*

Hyperparameters were not tuned per configuration. Tuning each release separately
would confound the effect of the release with the effect of the search.

## 2.10 Experimental design and reproducibility controls

Utility was estimated by five-fold stratified cross-validation repeated across
five independent seeds, giving 25 paired observations per configuration. Both
arms were scored on identical fold assignments. Five folds alone cannot support a
claim, since the smallest two-sided p a Wilcoxon test can return at n = 5 is
0.0625.

Fold assignment used seed 42; resampling used seed 20260819. Both are fixed in
source. All results are reported as mean ± standard deviation across the 25
folds, or as a point estimate with a 95% interval where the quantity is a single
property of a released set rather than a per-fold score.

## 2.11 Evaluation metrics

Disclosure risk metrics and their justifications are given in Table 3.

| Metric | Definition | Why it is used here |
|---|---|---|
| Proportion unique | Share of records alone in their equivalence class | Primary outcome; direct measure of singling-out risk |
| Prosecutor risk | 1 / smallest class size | Upper bound when the adversary knows the target is present, which holds for a classmate or lecturer |
| Marketer risk | Mean of 1 / class size | Appropriate where an adversary seeks many re-identifications and tolerates error |
| Minimum k | Smallest equivalence class size | Standard disclosure-control threshold measure (Sweeney, 2002) |
| l-diversity | Fewest distinct sensitive values in any class | k-anonymity permits homogeneous classes; l-diversity detects them (Machanavajjhala et al., 2007) |
| t-closeness | Largest class-to-population distributional distance | Detects classes whose sensitive distribution is itself informative (Li et al., 2007) |
| AUC-PR | Area under the precision-recall curve | Primary utility metric; more informative than AUC-ROC at the prevalences here (Saito and Rehmsmeier, 2015) |
| AUC-ROC | Area under the ROC curve | Secondary utility metric, reported against its 0.500 floor |

*Table 3. Evaluation metrics and the reason each is reported. Accuracy is not
used, being uninformative at these class ratios.*

Each corpus comprises every student in its cohorts rather than a sample, so the
sampling fraction is one, sample uniqueness equals population uniqueness without
an estimation step, and prosecutor and journalist risk coincide (El Emam et al.,
2011). This licenses statements about records within a defined institutional
population under a stated quasi-identifier configuration, not about Ghanaian
students in general.

The sensitive attribute is weak performance in each student's final observed
position, at the bottom 40% of each corpus. A quantile was used rather than an
absolute grade because the scales are not comparable and both l-diversity and
t-closeness are prevalence sensitive; integer grades tie, so the cut nearest the
target was applied and the prevalence achieved is reported.

## 2.12 Adversarial evaluation

A **reconstruction procedure** narrows the intervals a generalised release
establishes, using only the released bands and the published derived features.
Each value begins as the interval its band defines; the minimum and maximum bound
every value and pin one exactly where only one interval can contain them, and
each mean fixes a sum, permitting standard interval narrowing. Constraints are
applied to a fixed point, to 50 iterations, with widths below 1 × 10⁻⁶ treated as
exact. The attacker holds only what the custodian published, making the result a
lower bound. Reduction in interval width relative to band width is reported
first, with the proportion of values narrowed to a point as a subset of it. A
containment check verifies that the true value never falls outside the interval
produced.

A **linkage evaluation** measures isolation by an adversary holding partial,
imprecise knowledge (Narayanan and Shmatikov, 2008). Four scenarios were
specified, each tied to a person who plausibly holds that knowledge: a classmate,
a lecturer who marked the work, an employer holding application materials, and an
administrator processing credential verification. Recall was degraded by a
tolerance expressed as a fraction of the grade range, and matching was by
interval overlap between the recalled range and the published band rather than by
point distance, since an adversary recalls the value they saw rather than the
band published later.

Utility was the predictive performance of the Table 2 model trained on what a
recipient receives: the generalised source columns plus the derived features
published alongside. Passing only the source columns would make the arms
identical by construction, those columns being byte-identical between them.
Derived features were computed over the predictor positions only, since the
sensitive attribute derives from the final observed position.

## 2.13 Statistical analysis

The analysis specification was fixed before the confirmatory stage and is
archived with the repository. Earlier phases are reported as exploratory. No part
is described as pre-registered, since decisions taken after exploratory analysis
cannot be.

Risk differences were assessed by subsampling records without replacement at 80%,
2000 resamples, both arms on the same subsample, with 95% percentile intervals
(Efron and Tibshirani, 1993). An ordinary bootstrap was specified first,
implemented, executed and rejected: resampling with replacement duplicates
records, a duplicated record is not unique, and uniqueness falls asymmetrically
between arms as a result. Duplication does not perturb the estimator; it changes
the estimand, uniqueness being a property of the equivalence-class structure of a
set. A diagnostic comparing resample mean against observed value was specified in
advance and detected the failure.

Utility differences were assessed on the 25 paired per-fold AUC-PR scores. Folds
share training data under repeated cross-validation, so a naive paired analysis
understates uncertainty; every comparison is reported twice, naive and with the
variance correction for resampling-induced dependence (Nadeau and Bengio, 2003),
which inflated the standard error by a factor of 2.69 at this design.

Effect sizes are the raw differences in each metric's own units, both metrics
being bounded and directly interpretable. Standardising would divide by a
variance that repeated cross-validation renders ambiguous.

Equivalence testing was specified and removed. Two one-sided tests against a
margin formed part of the original specification while the arms were believed to
perform identically; the corrected utility evaluation showed that they do not, so
such a test fails at any defensible margin and conveys nothing the interval does
not.

Confidence was 95% throughout, alpha was 0.05, and Holm correction (Holm, 1979)
was applied within each family of comparisons, families being defined per corpus
and per outcome.

## 2.14 Ablation study

The ablation reverted the single modified step of Section 2.8, computing derived
features from the original rather than the generalised values while holding
banding, suppression, corpora, folds, seeds and model configuration constant.
This isolates the derivation source as the only active variable.

Reverting the step raised uniqueness in the nursing corpus at band width 0.5 from
0.129 to 0.875, a 0.746 increase in the primary risk metric, restoring 100.0% of
the protection generalisation had provided. Full ablation results across all
three corpora and all three band widths are reported in Section 3.7.

## 2.15 Reproducibility and open science artefacts

**Option 2, code and specification released, record-level data restricted.** The
source code, corpus construction scripts, frozen analysis specification and
figure generation code are available in the project repository at
https://github.com/yhmiller/xgboost-shap-nursing-licensure, to be archived on
Zenodo with a DOI issued on submission. Analysis modules are `strata.py`,
`disclosure_risk.py`, `deidentify.py`, `derivation_consistent.py`,
`attack_models.py`, `risk_utility.py`, `stats_validation.py` and `figures.py`,
executed by numbered notebooks 04 to 08.

Record-level data is not released: it is held under an agreement that does not
permit redistribution, and publishing it would contradict the finding this study
reports. The public replication corpus is openly available from the source in
Section 2.2.1, so the mechanism reproduces end to end without institutional
access. No cloud storage link is used for any artefact.

## 2.16 Supervisor verification checklist

| Item | State |
|---|---|
| Title, objectives and Section 2.8 all name the same operation, Modify [M] | Yes |
| Hybrid data present: public corpus and primary institutional records | Yes, Sections 2.2.1 and 2.2.2 |
| Fusion strategy named and justified | Yes, cross-validation replication, Section 2.3 |
| Exactly one ethics path selected | Yes, Path A, Section 2.4 |
| Exact preprocessing thresholds and class ratios stated | Yes, Section 2.5 |
| Ablation isolates the single contribution and states its effect | Yes, Section 2.14 |
| Seeds, split protocol and number of runs specified | Yes, Section 2.10 |
| No cloud storage link anywhere | Yes |
| All required figures and tables present and referenced before appearing | Yes, Figures 1 to 3, Algorithm 1, Tables 1 to 3 |
| No unfilled placeholder | **No. The ethics reference in Section 2.4 is `CHRPE/AP/XXX/26`** |

*Table 4. Self-audit against the marking scheme. One item is not satisfied and is
stated rather than ticked.*
