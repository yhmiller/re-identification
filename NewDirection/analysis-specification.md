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
| Risk comparison | Paired bootstrap over records, 2000 resamples, baseline against proposed derivation |
| Risk test statistic | Difference in proportion unique, with a 95% percentile confidence interval |
| Risk effect size | Cliff's delta on the paired resample differences |
| Utility comparison, primary | Paired difference in per-fold AUC-PR, baseline against proposed |
| Utility comparison, secondary | Two one-sided tests against margin delta, same pairs |
| Fold-correlation handling | Variance correction for repeated cross-validation. Naive paired tests are anti-conservative because folds share data |
| Utility fold design | Five folds repeated across five seeds, 25 paired observations |
| Equivalence margin | delta = 0.02 AUC-PR, absolute. See justification below |
| Margin sensitivity | Conclusion re-reported across delta in 0.01 to 0.10, step 0.01 |
| Confidence level | 95% throughout |
| Alpha | 0.05, one-sided per TOST component |
| Multiple comparisons | Holm correction within each family. Families are defined below |
| Corpora | Three, tested separately. Never pooled |
| Random seeds | 42 for the fold splitter, 20260819 for the bootstrap |

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

**H2b, acceptability.** Secondary, and only meaningful per configuration. The
utility cost falls within a margin a custodian would accept.

    H2b_0a: difference <= -delta
    H2b_0b: difference >=  delta

Equivalence is concluded only if both one-sided nulls are rejected. Failing to
reject a difference is not equivalence and is never reported as such. Under
delta = 0.02 none of the three corpora is expected to pass, which is a result
rather than a failure: it says the protection is not utility-neutral.

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
| Risk, per corpus | One test per generalisation width, 3 widths |
| Utility, per corpus | One TOST per generalisation width, 3 widths |

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

## Justification of delta

The margin is a scientific assumption, not a statistical setting. It states how
much utility loss is practically unimportant for this application, so it is
justified from the application.

The utility task is early-warning screening: rank students by risk of weak
final-semester performance so that support can be directed. A custodian receiving
a protected release decides one thing, whether that release still supports the
screening use.

Screening acts on a ranked list. What matters is whether the protected release
puts materially different students near the top. At the cohort sizes this tool
would be used on, from roughly one hundred to six hundred students, a difference
of 0.02 AUC-PR moves the composition of the flagged group by at most one or two
students. Institutions allocate tutoring and remediation in whole students and
not in fractions of a ranking metric, so a difference below that threshold cannot
be acted upon even if it is real.

Two grounds this justification deliberately does not use:

- The observed difference between the arms. That is approximately zero, which
  makes the test easy to pass and would make a margin chosen on that basis
  circular.
- Convention. No established convention for AUC-PR equivalence margins in this
  setting was found, and inventing one would be worse than arguing from the
  application.

**Sensitivity rather than assertion.** The conclusion is re-reported across delta
from 0.01 to 0.10. The write-up states the range over which equivalence holds,
which is a demonstrated claim, in place of asserting that no plausible margin
would change it, which is not.

**Fold-noise check.** If the observed standard deviation of AUC-PR across the 25
folds exceeds 0.02 on any corpus, delta is raised to that standard deviation for
that corpus and the change is reported. Claiming equivalence within a margin
smaller than the noise of the measurement would be claiming a precision the
design does not have.

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
