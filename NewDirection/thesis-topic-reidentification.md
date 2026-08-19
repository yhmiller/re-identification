# Thesis topic: re-identification risk in health professions education records

Programme: MSc Health Informatics
Status: proposal draft, revised 19 August 2026 against external review
Compiled: 16 August 2026

## Title

**Adopted.**

Derivation-Consistent De-identification of Health Professions Education Records:
Disclosure Risk and Analytical Utility Under Hybrid Public and Institutional
Academic Data

Short title: Derivation-consistent de-identification of academic records

### Why it changed

The earlier title was a question: *Re-identification Risk in Ghanaian Health
Professions Education Records: Do Derived Features Undermine De-identification?*
It was carried while the framing was open.

The Q1 methods marking scheme settled it. Item A8 fails a draft outright when the
title and objectives do not name the F/M/R/S operation described in Section 8 of
the Methods. A question names no operation, so the earlier title failed the gate
regardless of the content beneath it. The title, the two objectives and Methods
Section 2.8 now all name the same Modify [M] operation.

"Anonymisation" was avoided throughout in favour of "de-identification". The
study measures *residual* risk and the proposed method reduces it without
eliminating it, so claiming anonymity would invite an argument about a word that
has no bearing on the contribution.

**This change was not a free choice and the supervisor was told it was open.**
It should be reported to them rather than left to be noticed.

## Summary

Health training institutions share student academic records with researchers,
accreditation bodies and regulators after removing names and identity numbers.
Removing direct identifiers does not by itself make a dataset anonymous:
residual risk depends on which attributes remain, what an adversary already
knows, and what the release can be linked to. Academic records retain
combinations such as programme, cohort year and a sequence of semester grades
that remain unique to individuals after every direct identifier is gone.

This study measures that residual risk, then examines a specific mechanism that
has received limited empirical attention: whether features derived from source
variables *before* those variables are generalised continue to disclose
information about them afterwards. It evaluates a derivation-consistent
alternative, in which the same features are recomputed from the generalised
values, and reports the disclosure risk and analytical utility of both.

## The proposition, in one paragraph

Generalisation can protect source variables while features calculated from the
original values continue to disclose information about those variables. This
study quantifies that leakage in health professions education records and
evaluates whether recomputing the derived features from the generalised values
reduces disclosure risk without materially reducing analytical utility.

Everything below serves that sentence. Where an earlier draft made the Ghanaian
risk estimates the novelty, they are now the empirical setting. Where it promised
a national standard, it now offers a proposed operational framework.

## 1. Background and problem statement

Ghana's health training institutions hold detailed academic records for every
nursing, midwifery and allied health student they train. Those records are shared
for institutional research, for accreditation, for workforce planning and lately
for machine learning projects.

The prevailing protection is direct identifier removal. Delete the name, delete
the index number, treat what remains as protected. That practice has been known
to be inadequate for more than two decades: re-identification risk depends on the
attributes that remain and on what an adversary can link them to, not on whether
a name column is present.

Academic records are well suited to this failure. A student's grade sequence
functions as a fingerprint, and programme and cohort year narrow the field before
the grades are consulted. An adversary needs modest side knowledge, of the kind a
classmate or lecturer would ordinarily hold, to reduce the candidate set sharply.

What makes this a health informatics problem rather than an education problem is
what the records attach to. Health professions student data links to licensure
outcomes, to fitness-to-practise information and to workforce registries. The
attribute at the end of that chain, whether a person passed their professional
licensure examination, is career-determining and, in a small professional
community, socially consequential.

Ghana's Data Protection Act, 2012 (Act 843) governs this processing, and the Data
Protection Commission frames personal data broadly enough to include individuals
recognisable indirectly through their characteristics. Whether Ghanaian health
training institutions have operational de-identification guidance is a question
this study does not attempt to answer, because answering it properly would
require a review of institutional and regulatory guidance that is outside its
scope. The study reports what the records disclose and what a specific procedure
does about it, and leaves the regulatory gap analysis to work equipped to do it.

## 2. Research gap

Two things are established and one is not.

**Established.** Removing identifiers does not produce anonymity, and residual
risk is measurable. Health data privacy research has developed the metrics, the
adversary models and the risk-utility framing this study uses.

**Established.** Information that looks non-identifying can still disclose after
anonymisation, and derived or transformed representations can leak about their
sources. The general concern is not new.

**Not established.** What has received limited empirical attention is the
specific combination this study addresses: features derived from source variables
at their original precision, released alongside a generalised version of those
same variables, in a setting where the derived features are exactly the ones a
machine learning pipeline produces. Two assumptions are commonly made and neither
is well evidenced. The first is that releasing only derived summaries, rather
than records, is protective. The second is that generalisation applied to a
source variable also protects variables computed from it.

This framing is deliberately narrower than an earlier draft, which asserted that
the failure mode "has not been examined anywhere" and that "no published work
quantifies it". Those claims are not defensible without a systematic search, and
a single counter-example would sink them. The claim now made is about limited
evidence for a specific combination of setting, operation and attack, which is
what the study can actually support.

A systematic literature search is required before submission, and its result may
narrow this claim further. That is expected rather than a risk.

## 3. Aim and objectives

**Aim.** To determine whether features derived from unprotected source values
undermine the generalisation applied to those values, and to evaluate a
derivation-consistent alternative in terms of disclosure risk and analytical
utility.

**Objectives.** Two, adopted. The Q1 methods template caps a manuscript at three
and prefers two, and its marking scheme fails a draft outright when the title and
objectives do not name the engineering operation described in Section 8, so this
was settled by the scheme rather than by preference.

1. To develop a derivation-consistent generalisation procedure for student
   academic records, in which features derived from a generalised variable are
   recomputed from the protected values rather than from the originals.
2. To quantify the disclosure risk and the analytical utility of that procedure
   against standard generalisation, using hybrid public and institutional
   academic data.

Each maps to one research question in Section 5 and to one part of the results.

### What the earlier six objectives became

An earlier draft carried six. Nothing measured under them is discarded; the
change is what the thesis is organised around, not what it reports.

| Earlier objective | Now |
|---|---|
| Measure baseline disclosure risk | Reported as context establishing the problem |
| Attribute risk to specific attributes | Supporting finding |
| Establish whether derived features form a disclosure channel | Folded into Objective 1, and after the literature search reported as replication rather than discovery |
| Validate risk against a simulated linkage attack | Supporting finding, and a bound on how far measured uniqueness is realisable |
| Characterise the risk-utility frontier | Folded into Objective 2 |
| Propose an operational framework | Moved to future work, Section 5 |

## 4. Outcomes, ranked

An earlier draft listed a dozen measures without saying which one the thesis
turns on. An examiner asking "what is the primary outcome" needs one answer.

**Primary outcome.** Disclosure risk under a predefined adversary model,
expressed as the proportion of records uniquely identifiable on a specified
quasi-identifier configuration, with prosecutor risk reported alongside.

**Secondary outcomes.**

1. Equivalence class structure: k-anonymity level, class size distribution, l-diversity, t-closeness.
2. Leakage from derived features: reduction in the uncertainty that generalisation introduces, and the proportion of source values narrowed to a point.
3. Attack success: proportion of targets isolated under justified side-knowledge scenarios.
4. Analytical utility: predictive performance retained on the protected release.

Everything else is diagnostic and belongs in supplementary material.

### Explainability, and which kind

Reporting that a dataset is 87.5% unique tells a custodian nothing they can act
on. Reporting *which attributes make it so* tells them what to protect. That
interpretive layer sits inside the secondary outcomes rather than becoming a
separate objective, because it answers "why" about the primary outcome rather
than asking a new question.

Two distinct things get called explainability and only one of them is central
here.

**Disclosure-risk explainability, central.** Why is this release risky, which
attributes carry the risk, how few does an adversary need, and what changes when
a protection configuration is applied. Phase 2 already produces this: identifying
power per attribute, the greedy accumulation curve, the group-size relationship,
and the interval-narrowing measurements that show which derived features leak and
by how much. It needs naming and surfacing, not building.

**Model explainability, secondary and repurposed.** SHAP over the utility model,
used for a question the utility metric cannot answer. AUC-PR says whether the
protected release still supports the analysis. It does not say whether the model
is relying on the same things. If generalisation shifts feature importance while
holding performance constant, the release preserves accuracy while changing what
drives it, which is a finding a custodian should hear and which a single scalar
hides. This reuses the existing SHAP infrastructure for a purpose the earlier
scope never had.

The explanatory work therefore attaches to proposed objective 2, which already
covers disclosure risk and analytical utility. It does not need a third
objective, and adding one would undo the tightening the two-objective structure
was for.

## 5. Research questions

Two, matching the two objectives one to one. The supervisor's guidance was to
carry two critical questions properly rather than six thinly, and the marking
scheme's cap of three objectives points the same way.

**RQ1.** Does publishing features derived from unprotected source values
undermine the protection that generalising those values provides, and by how
much?

*Answered by:* the reconstruction procedure, the uniqueness comparison between
release arms, and the ablation that reverts the derivation step. Maps to
Objective 1.

**RQ2.** What does derivation-consistent generalisation cost in analytical
utility, and what risk-utility choices does it leave a data custodian?

*Answered by:* the paired utility comparison, the confirmatory tests, and the
risk-utility frontier. Maps to Objective 2.

### Findings reported but not framed as questions

A thesis answers fewer questions than it reports findings. The following were
research questions in an earlier draft. They are retained in the Results because
they establish the problem RQ1 and RQ2 address, and because discarding measured
results would be waste, but they are not claims the thesis is organised around.

| Former question | Now |
|---|---|
| Proportion uniquely identifiable, and risk under three adversary models | Baseline context in Results, establishing that the problem exists |
| Which attributes drive the risk, and how few an adversary needs | Supporting finding. The CGPA result motivates the choice of quasi-identifiers |
| Whether derived features form an independent disclosure channel | Supporting finding, and after the literature search a replication of DeSIA (2025) in a new setting rather than a discovery |
| Group size against within-group uniqueness | Supporting finding. Strong on its own terms and worth reporting, but a different question from the one the thesis answers |

Demoting these is a presentational decision, not a retraction. Every number
stands and every one is reported.

### Future research directions

Deferred deliberately, each with the reason.

**A written de-identification framework for institutions.** The frontier supports
one, and the practical contribution of this thesis is the measurement that would
underpin it. Producing and validating institutional guidance is a separate piece
of work requiring engagement with regulators.

**Differential privacy as a comparator.** This study defends generalisation with
its known weaknesses. A comparison against a formal privacy mechanism is the
obvious next question and is out of scope here, argued in Section 10.

**Whether attack realism holds outside simulation.** Side knowledge is simulated
from within each corpus. Validating adversary assumptions against real behaviour
needs a study design this one does not have.

**Transfer to other derived-feature families.** The eight features here are the
conventional academic set. Whether the finding holds for time-series summaries,
embeddings or clinical derived variables is untested.

**The prototype decision-support tool.** Scoped in the writing plan, scheduled
after the written work, and droppable without loss to the thesis.

## 6. Conceptual framework

Three disclosure types are measured separately.

| Type | Definition | Relevance here |
|---|---|---|
| Identity disclosure | An adversary matches a record to a specific individual | The primary outcome |
| Attribute disclosure | An adversary learns a sensitive value without necessarily isolating a record | Applies where a whole group shares an outcome |
| Membership disclosure | An adversary learns that an individual appears in the dataset | Relevant where inclusion is itself sensitive |

Three adversary models yield three risk figures from the same data. The
prosecutor knows the target is in the dataset, so risk is driven by the smallest
equivalence class. The journalist does not know who is in the dataset, so risk
turns on uniqueness scaled by the sampling fraction. The marketer wants many
re-identifications and tolerates errors, so risk is averaged across records.

### Uniqueness is not re-identification probability

These are related and not the same, and conflating them is the easiest way to
overstate a result. Uniqueness says a record has no twin on a given attribute
set. Re-identification requires an adversary who holds those attribute values,
holds them accurately, and knows the target is present.

The study measures both, and they diverge sharply. In the allied health corpus
every record is unique on the full quasi-identifier set, while an adversary with
plausible imprecise recall isolates at most 40% of targets, and under 6% once the
sequence is generalised. Uniqueness is an upper bound; measured attack success is
a lower bound, since a real adversary may hold more than the simulation grants.
Both are reported, and neither is presented as the risk.

### What the sampling fraction claim does and does not license

Each corpus covers every student in its cohorts, so within that defined
population sample uniqueness equals population uniqueness and no estimation step
is needed. This is a genuine strength relative to studies that must infer
population uniqueness from a sample.

It licenses statements of the form: *this proportion of records was unique within
the defined institutional population under the specified quasi-identifier
configuration*. It does not license statements about Ghanaian students in
general, about other institutions, or about the probability that a particular
student can be identified by a particular adversary. Results are worded
accordingly throughout.

## 7. Data

Three corpora, never pooled. Full specifications in `strata.py`.

| | Role | n | Source |
|---|---|---|---|
| `allied_health` | study population | 110 | Accra School of Hygiene, EH/OHS/OT, 2021-2022 |
| `nursing` | study population | 566 | BSc Nursing, 2023/24 and 2024/25, levels 200-400 |
| `public` | replication corpus | 649 | UCI Student Performance (Cortez & Silva, 2008) |

The public corpus is not part of the study population. It is Portuguese secondary
school students in a language course, present to test whether the mechanism
reproduces outside the study population and for no other purpose. No claim about
health records, Ghanaian institutions or Act 843 rests on it.

Unit of analysis: the individual student record. No new data collection.

Sensitive attribute: weak performance in the final observed sequence position,
defined per corpus at the bottom 40%. The licensure outcome is not required.
There is no dependent variable, so the study does not depend on any outstanding
data request.

## 8. Method

### Phase 1. Baseline risk

Uniqueness, equivalence class size distribution, k-anonymity, share of records
below k thresholds, l-diversity, t-closeness, and prosecutor, journalist and
marketer risk, across alternative quasi-identifier definitions and precision
levels.

### Phase 2. Attribution

Identifying power of each attribute alone, and a greedy accumulation curve
showing how few attributes suffice. Leave-one-out attribution is not used:
uniqueness saturates at two attributes, so removing any single member of a larger
set changes nothing and every marginal contribution returns zero. Attribution is
therefore measured from below.

### Phase 3. The derived-feature experiments

The study's main contribution.

**Experiment A.** Measure uniqueness on the derived feature set alone, with the
source sequence withheld. Tests whether a summary-only release is protective.

**Experiment B.** Generalise the source sequence, publish the derived features at
original precision alongside, and measure how far an adversary can narrow the
generalised values using them.

The claim here is deliberately weaker than "reconstruct the originals". What the
derived features do is *reduce the uncertainty that generalisation introduced*.
`gpa_min` and `gpa_max` are exact order statistics, so releasing them discloses
that at least one source value equals a stated number; each mean fixes a sum
across the sequence. Together these constraints narrow the intervals the bands
established, and in some cases collapse an interval to a point.

Two quantities are therefore reported, and the first leads:

- **Uncertainty reduction.** Mean interval width after the attack against the
  band width before it. This is the honest primary measure.
- **Exact recovery.** The proportion of source values narrowed to a single value.
  A subset of the above, reported second.

A containment check confirms that the true value never falls outside the interval
the attack produces. A soundness error would surface there immediately.

**Experiment C.** Repeat with the derived features recomputed from the
generalised values. Quantify what that costs.

### Phase 4. Adversarial validation

Attack success under side knowledge, simulated from within each corpus and
degraded to reflect imprecise recall. Every scenario is tied to a person who
plausibly holds that knowledge, because an examiner will ask why an adversary
would know a third-semester grade.

| Adversary | Plausibly knows | Source of that knowledge |
|---|---|---|
| Classmate | Programme, cohort, approximate performance in shared courses | Sat the same courses; results are often discussed or posted |
| Lecturer or tutor | Programme, cohort, grades in courses they taught | Marked the work |
| Employer or placement supervisor | Programme, approximate graduation year, broad standing | Application materials, references, transcripts submitted voluntarily |
| Administrator at a second institution | Programme, cohort, transfer or verification records | Credential verification requests |

Recall is degraded by a tolerance expressed as a fraction of the grade range,
because nobody recalls that a classmate scored 2.87. Results are reported by
scenario, so risk can be read as a function of how much the adversary knows
rather than as a single number.

**Ethical guardrail.** Side knowledge is simulated from within the corpora. No
attempt is made to identify any real, named individual at any point. No
re-identified record is reported.

### Phase 5. Risk-utility frontier

Release configurations across generalisation width, suppression threshold and
derivation mode. Risk is the primary outcome; utility is predictive performance
under the settled fold design, reported against its no-skill floor. Configurations
that are dominated on both axes are identified as such.

### Phase 6. Framework and write-up

Convert the frontier into a proposed operational framework with worked examples
and a decision procedure, positioned as something that could inform institutional
guidance rather than as a standard.

## 9. Success criterion, prespecified

Defined before results are examined, so the conclusion is not fitted to what was
found.

> The proposed procedure succeeds if it reduces the primary disclosure risk
> outcome substantially relative to standard generalisation, while retaining
> analytical utility within a prespecified equivalence margin.

Risk reduction is tested with a paired bootstrap comparison, reporting the
difference with a confidence interval and an effect size.

Utility equivalence is tested with two one-sided tests against a margin **δ**,
because a non-significant difference test does not establish equivalence. **δ
must be fixed before the tests are run.** See `NewDirection/writing-plan.md` for
the options and the recommendation; it is a decision for the supervisor, not a
number to be discovered.

## 10. Tools

Reduced to what the study actually uses. An earlier draft listed five external
anonymisation packages and two optimisation frameworks, most of which were never
adopted. Listing tools that were not used is a reviewer invitation.

| Purpose | Tool |
|---|---|
| Risk metrics, attribution, attacks, release construction | Own implementation, `disclosure_risk.py`, `deidentify.py`, `attack_models.py`, `derivation_consistent.py` |
| Utility measurement | Existing XGBoost pipeline |
| Statistics | `scipy`, paired bootstrap and TOST |

Cross-validation of the risk metrics against an established package, most likely
ARX or `sdcMicro`, is worth doing once to show the implementation agrees with a
reference. That is a verification step, not a second methodology.

**Deliberately out of scope.** Mondrian partitioning, integer programming for
optimal generalisation, and a differential privacy arm. Each is defensible work
and none is necessary for the proposition. If time is short, the derived-feature
experiment is the thesis and these are the first things to drop. Differential
privacy is discussed as a comparator paradigm rather than implemented, and the
positioning is that this study addresses a different operational problem:
consistent transformation of derived features inside a generalisation workflow.

## 11. Expected contribution

Academic:

1. Quantified residual re-identification risk for a category of health data that
   has attracted little attention, in a setting with no published estimates.
2. An empirical treatment of derived-feature leakage under generalisation:
   a controlled baseline-against-proposed comparison, quantified effect on
   disclosure risk, tested against a simulated attack, with the utility cost
   measured.
3. Measured rather than estimated uniqueness within the defined population,
   avoiding the sampling-fraction assumption most comparable work must make.
4. Replication of the mechanism on an independent public corpus, separating what
   is a property of the arithmetic from what is a property of the setting.

Practical:

1. A proposed operational framework that institutions could adopt and that could
   inform future guidance.
2. A quantified answer to the question a custodian actually asks: how much can
   this be protected before it stops being useful?
3. A procedural recommendation: recompute derived features from protected data,
   rather than releasing them alongside generalised sources at original
   precision.
4. A prototype decision-support tool, if time permits after the written work.

### The prototype tool, and its hard constraint

A custodian choosing a release configuration currently has no way to see what
they are choosing. A prototype would let them select a configuration and read
back the risk, the drivers of that risk, the leakage from derived features, the
retained utility, and a recommendation against a stated risk threshold.

It is a demonstration of the findings, not a finding. The thesis must stand
without it, and it is scheduled after the written work rather than beside it. If
the schedule slips, it is the first thing to drop.

**One constraint is not negotiable, and the existing codebase gets it wrong.**
The tool from the previous scope accepts an uploaded spreadsheet of students and
renders per-student explanations. For a study arguing that academic records
disclose individuals, a tool that displays individuals would be self-defeating,
and an examiner would be right to say so.

The prototype is therefore aggregate by construction:

- It reports over a dataset, never over a record.
- No record-level output is rendered, exported or logged.
- Real student data stays inside the approved environment. Any public-facing
  demonstration runs on the replication corpus or on synthetic input.
- Uploaded data, if upload is supported at all, is summarised and discarded
  rather than displayed.

The existing app is a usable chassis for layout and deployment. Its per-student
explanation flow is not reusable and should not be carried across.

### On "isn't this obvious?"

The expected challenge, and worth rehearsing. The principle is intuitive once
stated. The contribution is not the intuition but the work around it: formalising
the operation, constructing a controlled comparison in which only the derivation
source differs, quantifying the effect on disclosure risk, testing it against a
simulated attack, measuring the utility cost, and showing the result holds across
corpora with different scales and structures. Intuitions that nobody has measured
are how pipelines end up publishing both columns.

## 12. Ethics and governance

Institutional ethics approval obtained; reference number to be recorded.

Written confirmation that the data-sharing arrangement covers secondary use for
this research question. Consent for a licensure-prediction study does not
automatically cover a disclosure-risk study.

No real individual is identified at any stage. Reported outputs are aggregate
only. No record-level results, no illustrative example re-identifications.

Responsible disclosure: findings reach the institutions before publication, with
remediation guidance. Data stays in its existing storage arrangement; the study
adds no new transfer or exposure.

## 13. Scope and limitations

Risk estimates are specific to these corpora, their sizes and their cohort
structures. What transfers is the mechanism, not the numbers.

Uniqueness and re-identification probability are distinct. Uniqueness figures are
upper bounds on risk; simulated attack success is a lower bound.

Side knowledge is assumed rather than observed. The scenarios are justified but
not validated against real adversary behaviour.

k-anonymity and its extensions have known weaknesses. Differential privacy is
discussed rather than implemented, which is a deliberate scope decision and a
limitation.

Utility is measured against one modelling task. A configuration that preserves
this model's performance may not preserve another analysis.

The sensitive attribute is an academic proxy, since no licensure outcome exists.
That limits the attribute-disclosure analysis specifically.

The regulatory gap is asserted only as far as the study can support. No review of
Ghanaian institutional guidance was conducted.

## 14. Fit

MSc Health Informatics. Health data privacy, disclosure control, secondary use
and data governance are core to the discipline.

CANDO "build and apply a model". A disclosure risk model, an inference attack
that narrows generalised intervals, a linkage attack under justified adversary
scenarios, and a release-construction procedure evaluated on a risk-utility
frontier. An existing predictive model is repurposed as the utility instrument.

Continuity. The dataset, the feature engineering and the XGBoost pipeline already
exist. The earlier modelling work becomes an instrument within the new study
rather than its subject, and the study has no dependent variable, so it does not
depend on the outstanding licensure outcome file.

## 15. Anchor literature

Verify every citation before use. These are positions to argue against, not
sources already read.

Foundational disclosure control:

- Sweeney, L. Simple Demographics Often Identify People Uniquely (2000); k-anonymity (2002)
- Machanavajjhala, A. et al. l-diversity (2007)
- Li, N. et al. t-closeness (2007)
- Dwork, C. Differential Privacy (2006)

Re-identification demonstrations:

- Narayanan, A. and Shmatikov, V. Robust De-anonymization of Large Sparse Datasets (2008)
- de Montjoye, Y.-A. et al. Unique in the Crowd (2013)
- Rocher, L., Hendrickx, J. and de Montjoye, Y.-A. Estimating the success of re-identifications in incomplete datasets using generative models (2019)

Health data risk assessment and the risk-utility framing:

- El Emam, K. et al. R-U policy frontiers for health data de-identification
- Work on assessing and minimising re-identification risk in research data derived from health care records
- Work on enabling realistic health data re-identification risk assessment through adversarial modelling
- Work on preventing unintended disclosure of personally identifiable data following anonymisation

Ghanaian legal framework:

- Data Protection Act, 2012 (Act 843), and Data Protection Commission guidance. Read directly and check for amendments.
- Cybersecurity Act, 2020 (Act 1038)

The claim to make, once the systematic search supports it:

- Limited empirical evidence addressing derived features computed at original
  precision from variables that are generalised in the same release, in a
  machine learning feature-engineering setting.

## 16. Outstanding before submission

1. Systematic literature search, to fix the novelty claim at whatever level the evidence supports.
2. Fix the equivalence margin δ, with justification, before running the tests.
3. Ethics approval reference number.
4. Cross-check the risk implementation against an established package once.
5. Decide whether the proposed title and two objectives are adopted, as one package with Methods section 8.
