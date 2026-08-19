# Analysis specification

Frozen 19 August 2026, before the statistical phase runs. Nothing in this file
is chosen after seeing a test result. Where a decision was made with knowledge of
earlier exploratory output, that is stated rather than hidden.

The purpose is to make the statistical phase boringly predetermined, so that the
write-up cannot become "we ran some tests and kept the ones that made sense
afterwards".

## Status of the earlier analysis

Phases 1 to 5 have already been run and their point estimates are known. That
work is **exploratory**. It established what the measurements look like and how
the pipeline behaves.

What follows is the **confirmatory** analysis. It restates the questions as
formal hypotheses, fixes the tests and their parameters in advance of running
them, and reports the result whatever it is.

The distinction is reported in the write-up in these terms. No part of this
specification is described as pre-registered, because a decision taken after
exploratory analysis cannot be.

## Specification

| Component | Decision |
|---|---|
| Primary privacy outcome | Proportion of records uniquely identifiable on the specified quasi-identifier configuration |
| Secondary privacy outcomes | Prosecutor risk; minimum equivalence class size; l-diversity; uncertainty reduction; exact recovery; attack success rate |
| Primary utility outcome | AUC-PR, reported against its no-skill floor (the prevalence) |
| Secondary utility outcome | AUC-ROC, reported against 0.500 |
| Risk comparison | Subsampling without replacement over records, 80% fraction, 2000 resamples |
| Risk statistic | Difference in proportion unique, with a 95% percentile confidence interval |
| Utility comparison | Paired difference in per-fold AUC-PR, baseline against proposed |
| Utility fold design | Five folds repeated across five seeds, 25 paired observations, identical splits in both arms |
| Fold dependence | Reported twice: naive, and with a variance correction for resampling-induced dependence. Adviser to confirm the correction suits this design |
| Effect size | The raw difference in each metric's own units. Not standardised, see below |
| Equivalence testing | Removed. See below |
| delta | Removed from hypothesis testing. Retained only as an illustrative input to the decision framework |
| Confidence level | 95% throughout |
| Alpha | 0.05, one-sided per TOST component |
| Multiple comparisons | Holm correction within each family. Families are defined below |
| Corpora | Three, tested separately. Never pooled |
| Random seeds | 42 for the fold splitter, 20260819 for the bootstrap |

## Why equivalence testing was removed

An equivalence test was specified while the arms were believed to perform
identically. The corrected comparison shows differences of -0.027, -0.135 and
-0.067, so an equivalence test at any defensible margin fails on every corpus. A
test designed to fail conveys nothing the point estimate does not.

The paired difference and its confidence interval are reported instead. That is
strictly more informative: a reader who holds a different tolerance can apply it
to the interval without needing the thesis to have guessed it.

delta survives only in the decision framework, where a custodian supplies their
own tolerance and reads off the configurations available to them. A value of 0.02
appears in the worked example and is labelled illustrative. It is not a threshold
the thesis tests against, so the question of when it was chosen no longer arises.

## Why effect sizes are not standardised

Both primary metrics are bounded and directly interpretable. A difference of
0.135 AUC-PR, or of 84.1 percentage points of uniqueness, means something on its
own.

Standardising divides by a variance that repeated cross-validation makes
ambiguous, which imports the dependence problem into the effect size without
adding interpretability, and invites an argument about the choice of denominator
that has no bearing on the finding. The raw difference in the metric's own units
is an unstandardised effect size, which is the preferred form when the units are
meaningful.

Reported alongside each difference: the confidence interval, and the loss as a
percentage of the baseline value.

## Hypotheses

**H1, risk.** The proposed derivation reduces the proportion of uniquely
identifiable records relative to the baseline derivation, at the same
generalisation width.

    H1_0: p_unique(proposed) - p_unique(baseline) >= 0
    H1_1: p_unique(proposed) - p_unique(baseline) <  0

Tested by paired bootstrap. Rejected if the 95% confidence interval for the
difference lies entirely below zero.

**H2, utility cost.** The proposed derivation changes analytical utility relative
to the baseline derivation.

    H2_0: AUC-PR(proposed) - AUC-PR(baseline) =  0
    H2_1: AUC-PR(proposed) - AUC-PR(baseline) != 0

This replaces an equivalence test as the primary utility hypothesis. Exploratory
analysis found differences of -0.027, -0.135 and -0.067 across the three corpora,
so the question is no longer whether the arms are equivalent but how large the
cost is and whether it is larger than the variability of the procedure.

**Acceptability is not tested.** Whether a given utility cost is worth a given
risk reduction is a custodian's judgement about their own operational
requirements, not a proposition about the data. The thesis supplies the frontier
and the intervals; the choice belongs to whoever is releasing the file.

**H2c, mechanism.** Exploratory and reported descriptively. The baseline
derivation restores the analytical utility that generalisation removed.
Measured as (baseline - proposed) / (unprotected - proposed), which was 109.8%,
99.4% and 99.5% across the three corpora. Reported alongside the corresponding
share of disclosure risk restored, because the claim is that these are the same
information.

**H3, generalisation.** Both of the above hold on the replication corpus as well
as on the study population. Reported separately, never pooled.

## Families for multiple-comparison correction

Holm within each family, not across families, because the families answer
different questions.

| Family | Members |
|---|---|
| Risk, per corpus | One comparison per generalisation width, 3 widths |
| Utility, per corpus | One comparison per generalisation width, 3 widths |

Nine risk tests and nine utility tests in total across three corpora. Correction
is applied within corpus and within family, so three members per correction.

## Correction incorporated into this specification

An earlier version of the utility comparison passed only the generalised source
columns to the model. Those columns are byte-identical between the two derivation
arms, so the model never received the derived features that are the sole
difference between them, and the arms produced identical scores by construction.

The corrected comparison gives the model what a recipient of each release
actually receives: the generalised source columns plus the derived features
published alongside them. Derived features are computed over the predictor
positions only, because the sensitive attribute derives from the final observed
position and deriving over the whole sequence would leak the target.

Four checks were run before the interpretation was accepted, and all passed on
all three corpora: identical predictor structure between arms; both releases
genuinely publish the derived features; identical usable rows, index and
prevalence; and every transformation row-wise, with no preprocessing fitted
across records inside the evaluation.

One residual dependency is disclosed rather than removed. The sensitive attribute
is defined by a quantile computed over the whole corpus. It is a label definition
rather than a feature, and it is identical across arms, so it cannot bias the
comparison between them. It is noted because it is the only operation in the
pipeline that uses information across records.

## delta in the decision framework

delta is no longer a margin the thesis tests against. It is the tolerance a
custodian brings to the frontier, and the framework reads configurations off
against it.

The worked example uses 0.02 AUC-PR, justified from the application: screening
acts on a ranked list, and at cohort sizes of roughly one hundred to six hundred
a difference of that size moves the composition of the flagged group by at most
one or two students, below the granularity at which institutions allocate
support. It is illustrative. A custodian with a different operational tolerance
substitutes their own and the framework answers accordingly.

Because delta enters no hypothesis, when it was chosen no longer matters.

## A resampling scheme rejected on evidence

The specification originally called for an ordinary bootstrap over records for
the risk comparison. It was implemented, run, and rejected.

Resampling with replacement duplicates records, and a duplicated record is no
longer unique, so measured uniqueness falls under resampling. The fall is not
symmetric between the arms: an arm at 87% uniqueness has far more to lose than
one at 13%. The difference therefore shrinks toward zero, and on the nursing
corpus the observed difference of -0.7456 fell outside a bootstrap interval of
[-0.3074, -0.2367].

The check that caught this was written into the procedure in advance: the
resample mean is compared against the observed value, and a discrepancy means
the resampling scheme is measuring something other than the quantity of
interest.

Subsampling without replacement at 80% replaces it. No duplicates are created,
both arms are evaluated on the same subsample, and the point estimate now sits
inside its own interval: -0.7443 at the subsample against -0.7456 on the full
corpus, interval [-0.7704, -0.7152].

The cost is that uniqueness depends on set size, so intervals describe
variability at 80% of each corpus. Both figures are reported wherever the
interval appears, so any drift from the reduced size is visible rather than
assumed away.

## Definitions fixed in advance

**Uncertainty reduction.** For one source value released as a band of width w,
the attacker's interval after propagation has width w'. Uncertainty reduction is
1 - mean(w') / w over all observed values. Reported as the primary leakage
measure.

**Exact recovery.** A source value is exactly recovered when its interval
collapses to width at or below 1e-6. Reported as a subset of uncertainty
reduction, never in place of it.

**Attack success.** A target is successfully isolated when the candidate set
consistent with the adversary's side knowledge contains exactly one record and
that record is the target's. Both conditions are required.

**Protection reversed.** (attacked - banded) / (original - banded), the share of
the uniqueness that generalisation removed which the derived features restore.
Undefined and reported as such when original equals banded.

**Quasi-identifier configuration.** Fixed per corpus in `strata.py` and not
varied during the confirmatory analysis. Sensitivity to that choice belongs to
the exploratory phase and is already reported.

## What is not tested

Stated so that absence is a decision rather than an oversight.

- No comparison against Mondrian, integer programming or differential privacy.
  Out of scope, argued in the proposal.
- No test of whether the three corpora differ from one another. They are not
  comparable and are never pooled.
- No hypothesis about absolute risk levels. Risk magnitude is descriptive and
  specific to each corpus.
- No causal claim about adversary behaviour. Attack success is measured under
  stated assumptions, not observed in the field.

## Outstanding before Stage 3 runs

1. Supervisor sign-off on delta = 0.02, or a different value with its own
   justification.
2. Confirmation that the exploratory and confirmatory framing is acceptable for
   the programme.
