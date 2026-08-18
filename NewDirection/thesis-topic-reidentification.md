# Thesis topic: re-identification risk in health professions education records

Programme: MSc Health Informatics
Status: proposal-ready draft
Compiled: 16 August 2026

## Title

Re-identification Risk in Ghanaian Health Professions Education Records: Do Derived Features Undermine De-identification?

Short title: Re-identification risk in Ghanaian nursing education records

## Summary

Health training institutions share student academic records with researchers, accreditation bodies and regulators after "anonymising" them by deleting names and identity numbers. This is not anonymisation. Academic records hold combinations of attributes such as programme, cohort year and a sequence of semester grades that remain unique to individuals after every direct identifier has been removed.

This study measures that risk on a real Ghanaian nursing student dataset. It then tests a failure mode that has not been examined anywhere: whether the engineered features machine learning practitioners derive from academic records undo the de-identification applied to the source data. The study closes by deriving a risk-utility frontier and proposing an operational de-identification standard that health training institutions can apply.

## 1. Background and problem statement

Ghana's health training institutions hold detailed academic records for every nursing, midwifery and allied health student they train. Those records are shared more often now than in the past: for institutional research, for accreditation, for workforce planning and lately for machine learning projects that predict academic and licensure outcomes.

The prevailing protection is direct identifier removal. Delete the name, delete the index number, treat what remains as anonymous. That practice has been known to be inadequate for more than two decades. Foundational work in health data showed that most of a population can be uniquely identified from a small number of ordinary attributes. The same work showed that an "anonymised" medical dataset could be re-identified by linking it to a publicly available list.

Academic records may be more exposed than the demographic data those demonstrations used. A student's six-semester grade sequence works as a fingerprint. The probability that two students at one institution share the same six semester GPAs is negligible. Programme and cohort year narrow the field before the grades are consulted at all. An adversary needs very little side knowledge, the kind a classmate, lecturer or prospective employer would ordinarily hold, to isolate a single record and read everything else in it.

What makes this a health informatics problem rather than an education problem is what the records attach to. Health professions student data links to licensure outcomes, to fitness-to-practise information and to health workforce registries. The sensitive attribute at the end of that chain, whether a person passed their professional licensure examination, is career-determining. In a small professional community it is socially consequential as well.

Ghana's Data Protection Act, 2012 (Act 843) governs this processing. No operational standard tells a Ghanaian health training institution what adequate de-identification looks like. Institutions are left applying a rule of thumb that the technical literature abandoned two decades ago.

## 2. Research gap

Three gaps converge here. The third is the novel one.

The first is that no measured risk exists for Ghanaian health data. Re-identification risk has been quantified extensively for North American and European health datasets. It has not been quantified for Ghanaian health data of any kind. No evidence-based de-identification standard exists for Ghanaian health institutions.

The second is that health professions education records are unexamined. Disclosure control research concentrates on clinical records, administrative health data and survey microdata. Student academic records in health training institutions fall outside the clinical privacy literature. They fall outside the learning analytics ethics literature too, which is largely normative rather than quantitative.

The third is that derived features are treated as harmless. Machine learning on academic records nearly always involves feature engineering: trends, minima, maxima, consistency measures, counts of weak periods. Two assumptions are made implicitly and neither has been tested. The first is that releasing only derived summary features, rather than raw records, protects privacy. The second is that de-identification applied to source variables also protects the variables computed from them.

Both assumptions are questionable. A derived feature computed at full precision from a generalised source variable can reveal the original value, so the derived feature reverses the protection. Any health data science team can fall into this. No published work quantifies it.

## 3. Aim and objectives

Aim: to quantify re-identification risk in Ghanaian health professions education records, establish whether derived features undermine de-identification of the source data and produce an evidence-based de-identification standard that preserves analytical utility.

Objectives:

1. Measure baseline disclosure risk in a real Ghanaian nursing student dataset under recognised risk metrics and adversary models.
2. Attribute that risk to specific attributes and attribute combinations.
3. Establish whether derived academic features form an independent disclosure channel and whether they defeat generalisation applied to their source variables.
4. Validate the theoretical risk through a simulated linkage attack under realistic side-knowledge assumptions.
5. Optimise de-identification to maximise retained analytical utility subject to a defined risk ceiling and characterise the resulting risk-utility frontier.
6. Turn the findings into an operational de-identification standard for Ghanaian health training institutions.

## 4. Research questions

RQ1. What proportion of records in a Ghanaian health professions education dataset are uniquely identifiable on quasi-identifiers alone? What is the disclosure risk under the prosecutor, journalist and marketer adversary models?

RQ2. Which attributes and attribute combinations drive that risk? How few attributes does an adversary need to isolate an individual?

RQ3. Do derived academic features such as trend, consistency, minima, maxima and weak-period counts form an independent re-identification channel when released without the raw grade sequence?

RQ4. Does releasing derived features at full precision defeat generalisation applied to their source variables? Can the original values be reconstructed?

RQ5. What generalisation and suppression configuration maximises retained analytical utility subject to a defined risk ceiling? What shape does the risk-utility trade-off take?

RQ6. What operational de-identification standard follows from the results? How does it compare with existing international standards?

## 5. Conceptual framework

Three disclosure types are measured separately.

| Type | Definition | Relevance here |
|---|---|---|
| Identity disclosure | An adversary matches a record to a specific individual | The core risk, since grade sequences are near-unique |
| Attribute disclosure | An adversary learns a sensitive value about an individual without necessarily isolating their record | Applies where a whole programme-cohort group shares an outcome |
| Membership disclosure | An adversary learns that an individual appears in the dataset | Relevant where inclusion is itself sensitive |

Three adversary models yield three different risk figures from the same data. The prosecutor knows the target is in the dataset, so risk sits at its maximum and is driven by the smallest equivalence class. The journalist does not know who is in the dataset but wants to re-identify anyone, so risk turns on uniqueness in the wider population. The marketer wants many re-identifications and tolerates errors, so risk is averaged across records.

One methodological point deserves stating early. Most re-identification studies work on a sample and must estimate population uniqueness from sample uniqueness, which introduces a sampling-fraction assumption that weakens their conclusions. If the dataset used here covers all students in the relevant cohorts at the institution, then the population is the dataset, the sampling fraction is one and sample uniqueness equals population uniqueness exactly. The risk figures are measured rather than estimated. That is a stronger position than most published work in this area occupies.

## 6. Data

Source: student academic records from a Ghanaian health training institution, already obtained and processed under an existing data-sharing arrangement. No new data collection is required.

Unit of analysis: the individual student record.

Variables available, per schema.py:

| Group | Fields |
|---|---|
| Context and direct | student_id, cohort_year, programme |
| Aggregate performance | cgpa, total_credits, n_courses |
| Semester sequence | gpa_sem1 to gpa_sem6 |
| Grade distribution | n_grade_A to n_grade_E |
| Derived features | gpa_min, gpa_max, gpa_consistency, gpa_trend, gpa_first_half, gpa_final_half, n_weak_semesters, n_failed, fail_rate |

Candidate quasi-identifier set: programme, cohort_year, gpa_sem1 to gpa_sem6, cgpa, n_courses, total_credits and the grade counts. Sensitivity analysis will test alternative quasi-identifier definitions, since that choice materially affects measured risk.

Sensitive attribute: academic failure indicators derivable from the record. The licensure outcome is not required. There is no dependent variable in this study, so it does not depend on any outstanding data request.

## 7. Methodology

### Phase 0. Governance (weeks 1 to 2)

Confirm in writing that the existing data-sharing arrangement covers secondary use for this research question. Obtain institutional ethics approval. Agree a responsible disclosure route with the institution so that findings reach them before publication.

### Phase 1. Baseline risk measurement (weeks 3 to 5)

Compute the following over the candidate quasi-identifier set: the proportion of records that are unique; the distribution of equivalence class sizes; the k-anonymity level achieved together with the share of records at k below 3 and k below 5; l-diversity and t-closeness with respect to the sensitive attribute; prosecutor, journalist and marketer risk.

Repeat across alternative quasi-identifier definitions and across precision levels for the continuous variables.

### Phase 2. Risk attribution (weeks 5 to 7)

Find the minimum attribute set sufficient to isolate an individual by measuring uniqueness across all attribute subsets up to a tractable size. Rank attributes by marginal contribution to disclosure risk. The expected finding is that programme, cohort year and a small number of semester GPAs will suffice.

### Phase 3. The derived-feature experiments (weeks 7 to 10)

This phase carries the study's main contribution.

Experiment A treats derived features as an independent channel. Measure uniqueness and k-anonymity on the derived feature set alone (gpa_trend, gpa_consistency, gpa_min, gpa_max, n_weak_semesters, fail_rate) with the raw semester sequence withheld. This tests the common assumption that releasing summaries rather than records protects privacy.

Experiment B sets derived features against generalisation. Apply k-anonymisation to the raw semester GPAs by banding them into progressively coarser intervals, then release the derived features at full precision alongside them. Attempt to reconstruct the original raw values from the derived features. Since gpa_min and gpa_max are exact order statistics of the source vector, this tests directly whether derived features reverse the protection applied to their source.

Experiment C establishes consistent protection. Repeat Experiment B with the derived features recomputed from the generalised data rather than from the originals, then quantify the utility difference. This settles the correct operational procedure.

### Phase 4. Adversarial validation (weeks 10 to 12)

Simulate a linkage attack under realistic side knowledge: the profile a classmate, lecturer or employer would plausibly hold, meaning programme, approximate cohort, one or two remembered grades and possibly a rank position. Measure attack success against theoretical risk to establish whether the calculated risk is realisable in practice.

An ethical guardrail applies throughout this phase. Side knowledge is simulated from within the dataset. No attempt is made to identify any real, named individual at any point. No re-identified record is reported.

### Phase 5. Optimisation and the risk-utility frontier (weeks 12 to 16)

Implement generalisation and suppression optimisation that maximises retained information subject to a risk constraint, using Mondrian partitioning with an integer programming formulation for comparison. Compare against a differential privacy alternative so that the study does not end up defending a single paradigm.

Utility is measured two ways. The information-theoretic measures are the discernibility metric, average equivalence class size and non-uniform entropy loss. The analytical measure is the existing XGBoost pipeline's predictive performance on protected data against original data. The second is the decisive one, because it answers the question a data custodian actually asks: would this de-identification break the research the data is shared for?

Produce the frontier with risk on one axis and retained model performance on the other.

### Phase 6. Standard and write-up (weeks 16 to 24)

Convert the frontier into a de-identification standard with worked examples and a decision procedure. Benchmark it against international comparators. Draft, revise, submit.

## 8. Tools

| Purpose | Tool |
|---|---|
| Risk metrics and anonymisation | ARX Data Anonymization Tool; sdcMicro (R); pycanon (Python) |
| Privacy attack evaluation | anonymeter; custom reconstruction attacks |
| Differential privacy comparison | diffprivlib or OpenDP |
| Utility measurement | Existing XGBoost pipeline (engineered_xgboost.py, baseline_xgboost.py) |
| Optimisation | Mondrian implementation; PuLP for the ILP formulation |

## 9. Expected contribution

Academic:

1. First quantified re-identification risk estimates for Ghanaian health data of any kind.
2. First treatment of health professions education records as a disclosure control problem.
3. Evidence on the derived-feature question: whether engineered features form an independent disclosure channel and whether they defeat generalisation of their source variables. This generalises well beyond education. Any health machine learning pipeline that derives features from protected variables faces the same exposure.
4. Measured rather than estimated population uniqueness, which avoids the sampling-fraction assumption that weakens most comparable work.

Practical:

1. An operational de-identification standard that Ghanaian health training institutions and their regulators could adopt.
2. A quantified answer to the question a data custodian actually asks: how much can we protect this before it stops being useful?
3. A procedural recommendation for health data science teams. Recompute derived features from protected data. Never release them alongside generalised sources at original precision.

## 10. Ethics and governance

Institutional ethics approval is required before Phase 1 begins.

Written confirmation is needed that the existing data-sharing arrangement extends to this research question. Consent obtained for a licensure prediction study does not automatically cover a disclosure risk study. This must be settled explicitly rather than assumed.

No real individual is identified at any stage. Attacks use side knowledge simulated from within the dataset.

Reported outputs are aggregate risk metrics only. No record-level results are published. No illustrative "example" re-identifications appear anywhere in the thesis.

Responsible disclosure applies. Findings go to the institution before publication, with remediation guidance.

Data stays in its existing storage arrangement throughout. The study adds no new transfer or exposure.

## 11. Scope and limitations

Single institution. Risk figures are specific to this dataset's size, programme mix and cohort structure. What transfers is the mechanism rather than the numbers.

Side knowledge is assumed rather than observed. Real adversaries may hold more than the simulation grants them, so measured attack success is a lower bound on real risk.

k-anonymity and its extensions have known weaknesses. The differential privacy comparison in Phase 5 addresses this rather than ignoring it.

Utility is measured against one modelling task. A de-identification setting that preserves XGBoost performance may not preserve performance for a different analysis. This is stated rather than glossed over.

No licensure outcome. The sensitive attribute is an academic proxy. That limits the attribute disclosure analysis specifically and is disclosed as such.

## 12. Fit

MSc Health Informatics. Health data privacy, disclosure control, secondary use of health data and data governance are core to the discipline. The study sits on the health informatics agenda rather than beside it.

CANDO "build and apply a model". Four models are built and applied: a disclosure risk model covering uniqueness, k-anonymity and adversary-specific risk; a reconstruction and inference attack model in Phase 3; a linkage attack model in Phase 4; a constrained optimisation model producing the risk-utility frontier in Phase 5. An existing predictive model is repurposed as the utility measurement instrument.

Continuity with prior work. The dataset, the feature engineering and the XGBoost pipeline already exist. Nothing built to date is discarded. The earlier modelling work becomes an instrument within the new study rather than its subject. This study has no dependent variable, so it does not depend on the outstanding licensure outcome file.

## 13. Anchor literature

Verify exact citations before use. These are the works the study should be positioned against.

Foundational disclosure control:

- Sweeney, L. Simple Demographics Often Identify People Uniquely (2000); k-anonymity: A Model for Protecting Privacy (2002)
- Machanavajjhala, A. et al. l-diversity: Privacy Beyond k-anonymity (2007)
- Li, N. et al. t-closeness: Privacy Beyond k-anonymity and l-diversity (2007)
- Dwork, C. Differential Privacy (2006)

Re-identification demonstrations:

- Narayanan, A. and Shmatikov, V. Robust De-anonymization of Large Sparse Datasets, on the Netflix Prize data (2008)
- de Montjoye, Y.-A. et al. Unique in the Crowd (2013)
- Rocher, L., Hendrickx, J. and de Montjoye, Y.-A. Estimating the success of re-identifications in incomplete datasets using generative models, Nature Communications (2019)

Health data de-identification practice:

- El Emam, K. Work on health data de-identification methodology and the prosecutor, journalist and marketer risk framework
- HIPAA Safe Harbor and Expert Determination standards; UK Anonymisation Network guidance, as international comparators

Ghanaian legal framework:

- Data Protection Act, 2012 (Act 843) and Data Protection Commission guidance. Read these directly and check for amendments.
- Cybersecurity Act, 2020 (Act 1038)

The gap to claim:

- Learning analytics ethics literature, which is largely normative rather than quantitative
- The absence of published quantitative disclosure risk work on health professions education records, together with the absence of any published treatment of derived-feature leakage

## 14. Immediate next steps

| Step | Action | Timing |
|---|---|---|
| 1 | Written confirmation that the data-sharing arrangement covers this research question | Week 1 |
| 2 | Ethics submission | Week 1 |
| 3 | Run a one-day uniqueness check on the existing dataset. What share of records is unique on programme, cohort year and the GPA sequence? | Week 1 |
| 4 | Install and validate ARX, sdcMicro and pycanon against a known benchmark | Week 2 |
| 5 | Confirm total cohort coverage. Is this all students, making the sampling fraction one? | Week 1 |

Step 3 is the go/no-go. If uniqueness turns out near-total as expected, the study has its headline finding in week one and everything else builds on it. If uniqueness is unexpectedly low, the emphasis shifts toward attribute disclosure and the derived-feature experiments, both of which remain intact either way.
