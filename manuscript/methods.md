# 2. Materials and Methods

*Re-identification Risk in Ghanaian Health Professions Education Records: Do Derived Features Undermine De-identification?*

> **Draft status.** Written against the current title. If the proposed title and
> two-objective structure are adopted, three places change together: this
> subtitle, the objectives, and the naming in Section 2.8. `[REF]` marks the
> ethics approval number, granted but not yet recorded, left visibly incomplete
> rather than filled with a plausible-looking placeholder.

This study addressed an information pathway that de-identification practice
leaves open: features derived from a source variable at its original precision,
released alongside a generalised version of that same variable. A
derivation-consistent generalisation procedure was constructed, in which such
features are recomputed from the protected values. Within the F/M/R/S
engineering taxonomy this is a Modify [M] operation, altering one step of an
otherwise conventional pipeline rather than introducing a new component. The
procedure was evaluated on two corpora of Ghanaian health professions education
records and replicated on a public academic corpus.

## 2.1 Study design and system overview

The framework comprised four phases: corpus construction and de-identification;
release construction under a generalisation procedure with a configurable
derivation source; disclosure risk measurement and adversarial evaluation; and
analytical utility measurement. Figure 1 illustrates the architecture.

The baseline was standard generalisation: source variables are banded, and the
derived features an analyst requires are published alongside them at original
precision. The contribution altered exactly one step, the source from which
those features are derived. Banding, suppression, risk metrics, model,
hyperparameters and fold assignments were held identical, so any measured
difference is attributable to the derivation source alone.

The study has no dependent variable. Disclosure risk is a property of a released
set rather than an outcome to be modelled, so no outcome data was required.

## 2.2 Research setting and corpora

Three corpora were used. Two constitute the study population; the third is a
replication corpus, labelled as such throughout. Their characteristics are
summarised in Table 1.

| | allied health | nursing | public |
|---|---|---|---|
| Role | study population | study population | replication |
| Students | 110 | 566 | 649 |
| Grade scale | 0 to 4 | 0 to 4 | 0 to 20 |
| Aggregate | credit-weighted CGPA | unweighted mean | none |
| Sequence positions | 6 | 6 | 3 |
| Positions observed, median | 6 | 2 | 3 |
| Quasi-identifiers | 15 | 18 | 8 |
| Demographics | 0 | 0 | 4 |
| Band widths | 0.25, 0.5, 1 | 0.25, 0.5, 1 | 1.25, 2.5, 5 |
| Sensitive prevalence | 40.9% | 40.3% | 46.4% |

*Table 1. Characteristics of the three corpora. Band widths are fractions of
each corpus's grade range, as described in Section 2.7.*

**Allied health.** Records for 110 students of Environmental Health,
Occupational Health and Safety, and Occupational Therapy at the Accra School of
Hygiene, cohorts 2021 and 2022: a six-semester grade-point sequence, a
credit-weighted cumulative average, credit and course counts, and counts of
grades A to E. No demographic variables were released.

**Nursing.** Records for 566 BSc Nursing students, academic years 2023/24 and
2024/25, levels 200 to 400, from twelve institutional score sheets. Grades use
an A to E scale with plus modifiers. No credit hours are recorded, so the
semester grade point is an unweighted mean, which differs from the allied health
corpus and is recorded in the data-quality report. Sequence position 1 is level
200 semester 1, year two of a four-year programme, since level 100 is not on
file, and no student is observed in more than four positions.

**Public, replication.** The UCI Student Performance dataset, Portuguese
language course (Cortez and Silva, 2008): 649 students, three period grades on a
0 to 20 scale, with demographic attributes.

This corpus is not part of the study population. Its students are secondary
school pupils in a language course, present only to test whether the mechanism
reproduces beyond that population, and no claim about health records, Ghanaian
institutions or Act 843 rests upon it. It carries the study's only demographic
quasi-identifiers, since neither Ghanaian institution released any.

## 2.3 Replication across corpora rather than data fusion

No fusion strategy was applied. The identical procedure was executed
independently on each corpus and reported separately, which is a replication
design rather than a fusion design.

Fusion was rejected because the corpora are not commensurable: grade scales
differ (0 to 4 against 0 to 20), grade-point definitions differ
(credit-weighted against unweighted), and sequence lengths differ (six positions
against three). A pooled corpus would attribute to the procedure differences
arising from the measurement scales. Replication across corpora spanning group
sizes from 5 to 423 students is also what makes the relationship between group
size and disclosure risk measurable. Figure 2 illustrates the design.

## 2.4 Ethical compliance and data governance

The study was approved by the KNUST Committee on Human Research, Publications
and Ethics (Reference: `[REF]`) in accordance with the Ghana Data Protection
Act, 2012 (Act 843). Records were already held under an institutional
data-sharing arrangement and no new data was collected. Written confirmation was
obtained that the arrangement extends to secondary use for this question, since
consent for an outcome-prediction study does not automatically cover a
disclosure-risk study.

The nursing score sheets were the only source carrying direct identifiers, an
index number and a student name. Names were discarded during parsing and never
entered any dataframe. Index numbers were replaced by HMAC-SHA256 digests keyed
on a salt generated once and held outside version control; an unkeyed hash would
offer no protection, since the identifier space comprises roughly six hundred
values of known format and is exhaustible by enumeration. The digest is
deterministic because the sheets are organised by academic year and level, so a
sequence can only be assembled by linking the same individual across two years.
An assertion fails the build if any identifier-shaped column reaches an output.

All reported outputs are aggregate; no record-level result is produced, exported
or logged. The adversarial evaluation simulates side knowledge from within each
corpus, and no attempt was made to identify any real, named individual.

## 2.5 Data preprocessing pipeline

The allied health workbooks were consolidated by locating grade-matrix headers,
reading per-course grade, credit and grade-point triplets, and reconciling them
against the final-summary sheet. The nursing sheets were parsed as repeating
seven-row blocks, with course codes from each header and scores two rows below.
Sequences were assembled by mapping each (level, semester) pair to a position
and, for nursing, linking students across years on their pseudonym.

No imputation, scaling or encoding was applied. Gradient-boosted trees handle
mixed scales natively, and imputing a missing position would fabricate an
observation a recipient would not possess. Missing positions are retained
throughout, including in the risk calculation, where records sharing a gap are
grouped rather than separated.

## 2.6 Derived feature specification

Eight features were derived from each sequence, reproducing the set a machine
learning pipeline over academic records conventionally constructs: minimum,
maximum, mean and population standard deviation; the means of the first and
second halves; their difference; and a count of positions below a
weak-performance threshold.

Five constrain the source values arithmetically and are therefore available to
the reconstruction procedure of Section 2.11. The minimum and maximum are exact
order statistics of the sequence, and each mean fixes a sum across it. The
standard deviation is non-linear and the weak-position count is a threshold
count, so neither enters interval propagation, though both are published and
both contribute to uniqueness.

One function computes this set wherever it is required, so a difference in
results cannot arise from a difference in how the features were computed.

## 2.7 Baseline specification: standard generalisation

The baseline released each source variable generalised to a band, together with
the Section 2.6 features computed from the original values at full precision.
This is what a pipeline produces when generalisation is applied to source
columns and the derivation step is left untouched.

Generalisation replaced each value with the lower edge of its band, so a value
*v* at width *w* is published as ⌊*v*/*w*⌋ × *w*, and the recipient can infer
only that the original lay within [⌊*v*/*w*⌋×*w*, ⌊*v*/*w*⌋×*w* + *w*).

Band widths were fractions of each corpus's grade range rather than absolute
values: 0.0625, 0.125 and 0.25, giving 0.25, 0.5 and 1.0 on the four-point
scales and 1.25, 2.5 and 5.0 on the twenty-point scale. Absolute widths are not
comparable across scales: 0.5 covers an eighth of a four-point range but a
fortieth of a twenty-point range, where it fails to merge adjacent integers and
so applies no generalisation at all.

## 2.8 Engineering contribution: derivation-consistent generalisation

The proposed procedure implements a Modify [M] operation on the release step,
stated in one sentence: where features are derived from a variable that has been
generalised, they are derived from the generalised values rather than the
originals.

Let **X** denote the source sequence, *g<sub>w</sub>* the generalisation
operator at width *w*, and *d* the derivation function of Section 2.6:

    R_baseline = [ g_w(X) | d(X) ]
    R_proposed = [ g_w(X) | d(g_w(X)) ]

The releases differ in the argument of *d* and in nothing else, publishing the
same generalised columns and the same derived feature names. Figure 3 contrasts
the two constructions with the modified step marked, and Algorithm 1 gives the
procedure with that step identified at line 3.

```
Algorithm 1: Derivation-consistent generalisation

Input:  source sequence X (n x p), band width w, suppression threshold k,
        derivation function d, derivation source mode m in {baseline, proposed}
Output: release R

1:  G <- generalise(X, w)                      band each source value
2:  G, n_suppressed <- suppress(G, k)          blank classes below k
3:  S <- X if m = baseline else G              <- [M] THE MODIFIED STEP
4:  D <- d(S)                                  derive the published features
5:  R <- [ G | D ]                             join and release
6:  return R
```

*Algorithm 1. Only line 3 differs between the two arms. Lines 1, 2, 4 and 5 are
identical, so any measured difference is attributable to the derivation source
alone.*

## 2.9 Training configuration and computing environment

Experiments ran on Python 3.11.15 (macOS, arm64) with numpy 1.26.4, pandas
2.1.4, scikit-learn 1.4.2, xgboost 2.0.3, scipy 1.13.0 and matplotlib 3.8.4.

The utility model was an XGBoost classifier at a fixed configuration across
every release and corpus: 300 estimators, maximum depth 3, learning rate 0.05,
subsample 0.9, column subsample 0.9, histogram tree method, logistic loss.
Hyperparameters were not tuned per configuration, so the only variable across
evaluation rows is the data.

## 2.10 Disclosure risk evaluation

Risk was measured over the columns a recipient can see, including the derived
features wherever published. The primary measure is the proportion of records
uniquely identifiable on the quasi-identifier configuration, with prosecutor
risk alongside. Secondary measures are the equivalence class size distribution, minimum class
size, share of records below k thresholds of 2, 3, 5 and 10, l-diversity and
t-closeness.

Each corpus comprises every student in its cohorts rather than a sample, so
sampling fraction is one, sample uniqueness equals population uniqueness without
an estimation step, and prosecutor and journalist risk coincide. This licenses
statements about records within a defined institutional population under a
stated quasi-identifier configuration, not statements about Ghanaian students in
general or about the probability that a particular individual can be identified.
Uniqueness and re-identification probability are distinct, and both are
reported.

The sensitive attribute is weak performance in each student's final observed
position, defined per corpus as the bottom 40%. A quantile was used rather than
an absolute grade because the scales are not comparable and both l-diversity and
t-closeness are prevalence sensitive; integer grades tie heavily, so the cut
nearest the target is applied and the prevalence achieved is reported.

Attribution was measured from below rather than by leave-one-out, which returns
zero for every attribute once uniqueness saturates at two. It reports what each
attribute achieves alone, with a greedy accumulation curve.

## 2.11 Adversarial evaluation

A **reconstruction procedure** narrows the intervals a generalised release
establishes, using only the released bands and the published derived features.
Each value begins as the interval its band defines; the minimum and maximum
bound every value and pin one exactly where only one interval can contain them,
and each mean fixes a sum, permitting interval narrowing. Constraints are applied
to a fixed point, to 50 iterations, with widths below 1 × 10⁻⁶ treated as exact.
The attacker holds only what the custodian published, making the result a lower
bound. Two quantities are reported, the first leading: the reduction in interval
width relative to the band width, and the proportion of values narrowed to a
single point. A containment check verifies the true value never falls outside
the interval produced.

A **linkage evaluation** measures isolation by an adversary holding partial,
imprecise knowledge. Four scenarios were specified, each tied to someone who
plausibly holds that knowledge: a classmate who sat the same courses, a lecturer
who marked the work, an employer holding application materials and an
approximate graduation year, and an administrator processing credential
verification. Recall was degraded by a tolerance expressed as a fraction of the
grade range, and matching was by interval overlap between the recalled range and
the published band rather than by point distance, because an adversary recalls
the value they saw rather than the band published later.

## 2.12 Analytical utility evaluation

Utility was the predictive performance of the Section 2.9 model trained on what
a recipient actually receives: the generalised source columns together with the
derived features published alongside. Passing only the source columns would make
the arms identical by construction, since those columns are byte-identical
between them, yielding an equivalence that is arithmetic rather than empirical.
Derived features were computed over the predictor positions only, because the
sensitive attribute derives from the final observed position and deriving over
the full sequence would leak the target.

The task is early-warning screening: weak performance in the final observed
position predicted from earlier positions. Five folds repeated across five seeds
gave 25 paired observations; five folds alone cannot support a claim, since the
smallest two-sided p a Wilcoxon test can return at n = 5 is 0.0625. Both arms
were scored on identical splits. AUC-PR is primary and AUC-ROC secondary, each
reported against its no-skill floor.

## 2.13 Statistical analysis

The analysis specification was fixed before this stage and is archived with the
repository. Earlier phases are reported as exploratory; the comparisons below
are confirmatory. No part is described as pre-registered, since decisions taken
after exploratory analysis cannot be.

**Risk.** Differences were assessed by subsampling records without replacement
at 80%, 2000 resamples, both arms on the same subsample, with 95% percentile
intervals. An ordinary bootstrap was specified first, implemented, run and
rejected. Resampling with replacement duplicates records, a duplicated record is
not unique, and uniqueness therefore falls under resampling, asymmetrically
between arms. Duplication does not perturb the estimator; it changes the
estimand, since uniqueness is a property of the equivalence-class structure of a
set. A diagnostic comparing resample mean against observed value was specified
in advance and detected the failure. Intervals describe variability at 80% of
each corpus, and both figures are reported.

**Utility.** Differences were assessed on the 25 paired per-fold AUC-PR scores.
Folds share training data under repeated cross-validation, so a naive paired
analysis understates uncertainty. Every comparison is reported twice, naive and
with a variance correction for resampling-induced dependence, which inflates the
standard error by 2.69 at this design.

**Effect sizes** are the raw differences in each metric's own units, both being
bounded and directly interpretable.

**Equivalence testing was specified and removed.** Two one-sided tests against a
margin were planned while the arms were believed to perform identically. The
corrected utility evaluation showed they do not, so such a test fails at any
defensible margin and conveys nothing the interval does not. The paired
difference and its interval are reported instead, allowing a reader holding a
different tolerance to apply it directly.

Confidence is 95%, alpha 0.05, with Holm correction within each family of
comparisons, families defined per corpus and per outcome.

## 2.14 Ablation

The ablation reverts the single modified step, computing derived features from
the original values while holding banding, suppression, corpora, folds, seeds
and model configuration constant. This isolates the derivation source as the
only active variable.

## 2.15 Reproducibility artefacts

The source code, corpus construction scripts, analysis specification and figure
generation code are available in the project repository. Analysis modules are
`strata.py`, `disclosure_risk.py`, `deidentify.py`, `derivation_consistent.py`,
`attack_models.py`, `risk_utility.py`, `stats_validation.py` and `figures.py`,
executed by numbered notebooks 04 to 08. Seeds are fixed at 42 for fold
assignment and 20260819 for resampling.

The record-level data cannot be shared. It comprises identifiable academic
records held under an agreement that does not permit redistribution, and
releasing it would contradict the finding this study reports. The public
replication corpus is openly available from its original source, so the
mechanism can be reproduced end to end without institutional access.
