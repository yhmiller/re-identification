# 3. Results and Analysis

> **Draft status.** Figures 1 to 3, Algorithm 1 and Tables 1 to 4 appear in
> Section 2. This section continues from Figure 4 and Table 5. All values are
> produced by `notebooks/04` to `08` and `figures.py`; none is transcribed by
> hand.

## 3.1 Descriptive summary of the corpora

Following the consolidation described in Section 2.5, the allied health corpus
comprised 110 students across three programmes and two cohorts, the nursing
corpus 566 students across two academic years, and the public replication corpus
649 students (Cortez and Silva, 2008). Corpus characteristics are given in Section 2.2.

The sensitive attribute, weak performance in the final observed sequence
position, was present in 40.9%, 40.3% and 46.4% of records respectively. Median
observed sequence positions were six for allied health, two for nursing and
three for the public corpus; the nursing median reflects that only two academic
years were available, so most students appear in a single year.

Utility evaluation used 109 allied health records, 421 nursing records and 649
public records after listwise deletion on the predictor set, with class
prevalence of 0.413, 0.428 and 0.464.

## 3.2 Baseline disclosure risk

Uniqueness under the unprotected release rose sharply with the number of
attributes an adversary was assumed to hold. Structural attributes alone
identified no records in any corpus, with minimum equivalence class sizes of 9,
5 and 226. Adding grade information changed this immediately.

In the allied health corpus, the cumulative grade point average alone identified
96.4% of records, taking 108 distinct values across 110 students. Two attributes
identified all 110. In the nursing corpus the same aggregate identified 36.2%,
taking 310 distinct values across 566 students, and the full quasi-identifier set
identified all records. In the public corpus the full set identified 85.7%,
while demographic attributes alone identified 1.8%.

Prosecutor risk, the reciprocal of the smallest equivalence class, was 0.111,
0.200 and 0.004 under structural attributes alone and 1.000 under the full
quasi-identifier set in all three corpora. Marketer risk, the mean of the
reciprocal class size, rose from 0.055, 0.009 and 0.003 to 1.000, 1.000 and
0.921 across the same contrast.

Equivalence class structure moved correspondingly. Under structural attributes
the three corpora formed 6, 5 and 2 classes with median sizes of 22, 146 and
423 records, and no record fell below k = 5. Under the full quasi-identifier set
they formed 110, 566 and 598 classes with median size 1, and the share of
records below k = 5 was 1.000, 1.000 and 1.000.

The cumulative average alone placed 96.4% of allied health records below k = 2
and 100% below k = 3. In the nursing corpus the same attribute placed 36.2%
below k = 2, 55.7% below k = 3 and 72.8% below k = 5.

Minimum l-diversity (Machanavajjhala et al., 2007) was 1 in every scenario
except structural-attributes-only in the allied health and public corpora, where
it was 2. Maximum t-closeness under the full quasi-identifier set was 0.591,
0.597 and 0.536.

Rounding the grade sequence reduced uniqueness only at the coarsest setting.
Allied health remained 100% unique at three, two and one decimal places, falling
to 66.4% at zero decimals; the public corpus was unaffected at every precision
tested, its grades already being integers.

## 3.3 Risk attribution

Attribution measured from below, as described in Section 2.10, ranked the
cumulative average first in both Ghanaian corpora. Greedy accumulation reached
100% uniqueness after two attributes in allied health (`cgpa`, `gpa_sem1`) and
five in nursing (`cgpa`, `gpa_sem2`, and three grade counts). In the public
corpus six attributes reached 67.8%.

Within-group uniqueness measured on a single rounded attribute varied inversely
with group size across 13 groups spanning 5 to 423 students, with a Spearman
correlation of -0.880. Restricted to the study population, 11 groups spanning 5
to 238 students gave -0.893. The smallest group, 5 students, was 40.0% unique;
the largest, 423 students, was 0.9% unique.

## 3.4 Baseline versus proposed release

Table 5 reports the primary comparison at the middle band width of each corpus.
Uniqueness is reported for the release as a whole. AUC-PR, the primary utility
metric on account of the class balance of the prediction task (Saito and
Rehmsmeier, 2015), is reported as the mean across 25 cross-validation folds with
its standard deviation, produced by the gradient-boosted tree model (Chen and
Guestrin, 2016) specified in Section 2.9.

| Corpus | Release | Unique | AUC-PR | AUC-PR SD |
|---|---|---|---|---|
| Allied health | Unprotected | 100.0% | 0.8321 | 0.0570 |
| | Generalised, no derived features | 92.7% | 0.8031 | 0.0649 |
| | Generalised + baseline derivation | 100.0% | 0.8264 | 0.0732 |
| | Generalised + proposed derivation | 92.7% | 0.7994 | 0.0741 |
| Nursing | Unprotected | 87.5% | 0.9081 | 0.0382 |
| | Generalised, no derived features | 12.9% | 0.8028 | 0.0561 |
| | Generalised + baseline derivation | 87.5% | 0.9284 | 0.0298 |
| | Generalised + proposed derivation | 12.9% | 0.7937 | 0.0533 |
| Public | Unprotected | 13.6% | 0.9749 | 0.0111 |
| | Generalised, no derived features | 2.2% | 0.9081 | 0.0212 |
| | Generalised + baseline derivation | 12.6% | 0.9702 | 0.0151 |
| | Generalised + proposed derivation | 2.2% | 0.9035 | 0.0267 |

*Table 5. Disclosure risk and analytical utility under four releases, at band
width 0.5 for the two Ghanaian corpora and 2.5 for the public corpus. AUC-PR is
the mean across 25 folds (5 folds × 5 seeds). No-skill floors are 0.413, 0.428
and 0.464 respectively.*

Uniqueness under the baseline derivation matched the unprotected release to
within 1.0 percentage point in all three corpora (100.0% against 100.0%, 87.5%
against 87.5%, 12.6% against 13.6%). Uniqueness under the proposed derivation
matched the generalised release exactly in all three (92.7%, 12.9% and 2.2%).

AUC-PR under the baseline derivation was within 0.005 of the unprotected release
in allied health and public, and exceeded it by 0.020 in nursing. AUC-PR under
the proposed derivation was within 0.005 of the generalised release in all three.

The four release states are shown in Figure 4.

*Figure 4. Records uniquely identifiable under four releases in each corpus:
unprotected, generalised, generalised with derived features computed from the
original values, and generalised with the same features recomputed from the
generalised values. Band width 0.5 for the Ghanaian corpora and 2.5 for the
public corpus.*

## 3.5 Statistical comparison

Table 6 reports the confirmatory comparisons specified in Section 2.13.

| Corpus | Band | Risk difference | 95% CI | Utility difference | 95% CI, corrected | p, corrected |
|---|---|---|---|---|---|---|
| Allied health | 0.25 | -0.018 | [-0.023, 0.000] | +0.002 | [-0.030, +0.035] | 0.890 |
| | 0.50 | -0.073 | [-0.109, -0.036] | -0.027 | [-0.087, +0.033] | 0.361 |
| | 1.00 | -0.545 | [-0.600, -0.482] | -0.007 | [-0.079, +0.066] | 0.854 |
| Nursing | 0.25 | -0.528 | [-0.550, -0.488] | -0.080 | [-0.115, -0.044] | 0.0001 |
| | 0.50 | -0.746 | [-0.773, -0.715] | -0.135 | [-0.189, -0.080] | <0.0001 |
| | 1.00 | -0.825 | [-0.863, -0.815] | -0.154 | [-0.202, -0.106] | <0.0001 |
| Public | 1.25 | -0.035 | [-0.050, -0.027] | -0.014 | [-0.027, -0.002] | 0.026 |
| | 2.50 | -0.105 | [-0.137, -0.100] | -0.067 | [-0.088, -0.045] | <0.0001 |
| | 5.00 | -0.103 | [-0.137, -0.106] | -0.134 | [-0.162, -0.107] | <0.0001 |

*Table 6. Proposed minus baseline derivation. Risk differences are in proportion
uniquely identifiable, with percentile intervals from 2000 subsamples at 80% of
each corpus. Utility differences are in AUC-PR across 25 paired folds, with
intervals and p values corrected for fold dependence. Effect sizes are the
differences themselves.*

Risk intervals excluded zero in eight of nine configurations. The exception was
allied health at band 0.25, where the upper bound was 0.000.

Utility intervals excluded zero in six of nine. All three nursing and all three
public configurations excluded zero; none of the three allied health
configurations did, with corrected p values of 0.890, 0.361 and 0.854. The
allied health interval at band 0.50 spanned [-0.087, +0.033].

The dependence correction inflated every standard error by a factor of 2.69. All
six configurations that excluded zero under the naive analysis also excluded zero
under the corrected analysis.

Across the evaluated configurations, wider generalisation was generally
accompanied by larger risk reduction and larger utility loss. Two exceptions
occurred: allied health utility did not order with band width, and public risk
was marginally smaller at band 5.00 than at band 2.50.

## 3.6 Cross-validation stability

Figure 6 presents the distribution of the 225 paired per-fold AUC-PR differences,
25 for each of nine configurations.

*Figure 6. Paired per-fold AUC-PR differences, proposed minus baseline
derivation, for three band widths in each corpus. Each point is one of 25 folds.
Distributions whose folds cross zero are shown in a separate colour from those
whose folds all fall below it.*

All 75 nursing folds and all 75 public folds fell below zero. Of the 75 allied
health folds, 43 fell below zero.

Per-configuration fold means were +0.002, -0.027 and -0.007 for allied health at
bands 0.25, 0.50 and 1.00; -0.080, -0.135 and -0.154 for nursing at the same
widths; and -0.014, -0.067 and -0.134 for the public corpus at bands 1.25, 2.50
and 5.00. Fold ranges were [-0.191, +0.102] for allied health, [-0.230, -0.024]
for nursing and [-0.184, -0.001] for public.

The three allied health distributions are the only ones in Figure 6 that cross
zero, and are marked accordingly.

## 3.7 Ablation

The ablation reverted the single modified step, computing derived features from
the original rather than the generalised values. Table 7 reports the effect on
uniqueness across all three band widths.

| Corpus | Band | Unprotected | After generalisation | After reversion | Protection reversed | Values recovered exactly |
|---|---|---|---|---|---|---|
| Allied health | 0.25 | 100.0% | 98.2% | 100.0% | 100.0% | 33.9% |
| | 0.50 | 100.0% | 92.7% | 100.0% | 100.0% | 25.7% |
| | 1.00 | 100.0% | 45.5% | 100.0% | 100.0% | 21.0% |
| Nursing | 0.25 | 87.5% | 34.6% | 87.5% | 100.0% | 62.8% |
| | 0.50 | 87.5% | 12.9% | 87.5% | 100.0% | 46.9% |
| | 1.00 | 87.5% | 4.6% | 86.4% | 98.7% | 34.1% |
| Public | 1.25 | 13.6% | 9.6% | 13.1% | 88.5% | 93.7% |
| | 2.50 | 13.6% | 2.2% | 12.2% | 87.8% | 83.8% |
| | 5.00 | 13.6% | 0.9% | 9.4% | 67.1% | 74.1% |

*Table 7. Ablation of the modified derivation step. Protection reversed is the
share of the uniqueness removed by generalisation that reverting the step
restores. Values recovered exactly is the proportion of source values narrowed to
a single point by the reconstruction procedure of Section 2.11.*

Protection reversed ranged from 67.1% to 100.0%. Mean interval width after
reconstruction ranged from 5.0% to 47.8% of the band width. The containment
check held at 100% in all nine configurations.

A summary-only release, publishing derived features with the source sequence
withheld, produced uniqueness of 100.0%, 86.2% and 10.5% in the three corpora
against 100.0%, 87.5% and 13.6% for the raw sequence. Two derived features alone
produced 100.0%, 84.1% and 6.3%. A single derived feature, `gpa_trend`, produced
98.2%, 43.1% and 0.6%.

## 3.8 Adversarial evaluation

Under simulated side knowledge, correct isolation was achieved for at most 40.0%
of allied health targets, 1.4% of nursing targets and 1.9% of public targets.

In the allied health corpus, group membership alone left a mean candidate set of
20.8 records and isolated no targets. Adding two grades at sharp recall isolated
20.9% of targets from the unprotected release and 5.5% from the generalised
release. Adding the cumulative average recalled to one decimal place isolated
40.0%, unchanged across all three release configurations.

In the nursing corpus, group membership alone left a mean candidate set of 173.8
records. Two grades at sharp recall reduced this to 34.4 against the unprotected
release and 70.6 against the generalised release, isolating 1.4% and 0.7% of
targets respectively.

In the public corpus, group membership alone left a mean candidate set of 354.4
records; adding four demographic attributes reduced this to 17.2 and isolated
1.9% of targets, the largest reduction achieved in that corpus by any scenario
tested.

Across all three corpora, generalising the sequence increased the mean candidate
set for every grade-recall scenario and left the cumulative-average scenario
unchanged, that attribute not being among the generalised columns.

Where derived features were published, recalling `gpa_trend` isolated 4.5%,
0.2% and 0.5% of targets in the three corpora, and recalling the minimum and
maximum together isolated 4.5%, 0.0% and 0.0%.

The missingness pattern of the sequence isolated 0.9%, 0.9% and 0.0% of targets.
The target was retained in the candidate set in 100% of all 72 linkage runs.

## 3.9 Risk-utility characterisation

Figure 5 plots risk reduction against retained utility for every configuration
in all three corpora.

*Figure 5. Disclosure risk reduction against analytical utility retained,
expressed as percentages of the unprotected release. Marker shape denotes corpus
and marker fill denotes the release type. The dotted line marks the utility of
the unprotected release.*

In the nursing corpus, generalisation at band 0.25, 0.50 and 1.00 reduced risk by
60.4%, 85.3% and 94.7% while retaining 95.0%, 86.5% and 81.3% of unprotected
AUC-PR. Adding suppression at k below 5 to band 0.50 reduced risk by 100.0% while
retaining 92.2%, using 339 of 421 records. In the public corpus, band 1.25, 2.50
and 5.00 reduced risk by 29.5%, 84.1% and 93.2% while retaining 98.7%, 93.1% and
85.8%.

In the allied health corpus, suppression at k below 3 and k below 5 suppressed
all 110 records at band 0.50, and the largest risk reduction achieved by any
non-empty configuration was 54.5%.

Configurations publishing derived features from the original values reduced risk
by 0.0% to 17.0% while retaining 96.5% to 102.2% of unprotected AUC-PR.

## 3.10 Computational cost

Constructing one release took 3.6 ms and computing a full risk profile over it
5.2 ms for a 566-record corpus. The reconstruction procedure over 566 records
completed in 0.17 s. The utility evaluation, comprising 25 model fits, took
2.66 s per configuration. The subsampling comparison at 2000 resamples was the
dominant cost of the confirmatory stage.

## References cited in this section

- Cortez, P. and Silva, A. (2008). Using data mining to predict secondary school student performance.
- Chen, T. and Guestrin, C. (2016). XGBoost: a scalable tree boosting system.
- Machanavajjhala, A., Kifer, D., Gehrke, J. and Venkitasubramaniam, M. (2007). l-diversity: privacy beyond k-anonymity.
- Saito, T. and Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets.
