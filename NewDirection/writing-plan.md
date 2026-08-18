# Writing plan

Compiled 18 August 2026. Branch `disclosure-risk`.

Phases 1 to 5 of the analysis are complete and documented in
`docs/disclosure-risk-findings.md`. This plan covers everything between that
state and a submitted Methods and Results section written to the Q1 templates in
`proposal/Templates/`.

Read the two templates before starting any writing step. They are prescriptive,
not advisory, and several of their requirements are things the analysis does not
yet satisfy.

## What the templates demand that we do not yet have

| Requirement | Source | Status |
|---|---|---|
| A public dataset **and** primary field data | Methods template, called non-negotiable | Missing. Two institutional strata, no public arm. |
| Exactly one engineering operation, F/M/R/S | Methods §8 | Not framed. Reads as a study, not an artefact. |
| Title naming the artefact, matching §8 | Methods Part 0 | Current title is a question. Mismatch is a desk-reject. |
| Two objectives, each mapping to a subsection | Methods Part 0 | Currently six objectives. |
| p-value **and** effect size on the main comparison | Results §5 | No significance testing anywhere yet. |
| Figures at 300 dpi or vector | Results, placement rules | No figures produced at all. |
| Training and inference timings | Results §8 | Not measured. |
| Ethics reference number | Methods §4 | Approval granted; number not yet in any file. |

## Stage 1. Fold the public arm into the pipeline

The UCI Student Performance corpus (Cortez & Silva, 2008) is already in
`data/public/student-por.csv`: 649 records, 33 variables, a three-period grade
sequence G1 to G3, and real demographics.

It earns its place twice over. It satisfies the template's hybrid-data rule, and
it carries demographic quasi-identifiers that the Ghanaian data does not have at
all, because the college released no demographics. A spot check already
replicates the core finding: banding cuts uniqueness from 13.6% to 3.7%, the
reconstruction attack restores it to 12.6%, and 85.0% of grades are recovered
exactly with containment at 100%.

- [ ] Add a `load_public()` loader beside the two existing strata.
- [ ] Sequence columns are G1 to G3, so the derived-feature functions need to
      accept a three-element sequence. `derive_features` already generalises;
      confirm the first-half and final-half split behaves sensibly at length 3.
- [ ] Re-run notebooks 04 to 07 with three strata.
- [ ] Record the demographic quasi-identifier result separately. It answers a
      question the Ghanaian data cannot: what happens when age and address sit
      beside grades.

Note in writing: the public arm is a **replication**, not a pooled sample. The
grading scales and sequence lengths differ, so the three strata are never merged.

## Stage 2. Reframe as one engineering operation

The template permits Fabricate, Modify, Remove or Replace, and exactly one. The
work already done is one clean modification. It needs naming, not rebuilding.

| | |
|---|---|
| Baseline | Standard generalisation. Bands the source variables, publishes derived features at original precision. This is what pipelines do today. |
| Proposed, operation [M] | Derivation-consistent generalisation. Recomputes derived features from the protected values. |
| Ablation | Revert the derivation step to the original values. Protection collapses by 98.7% to 100%. |

Experiment C becomes the contribution. Experiment B becomes its ablation. That
is the baseline-versus-engineered spine both templates are built around, with no
new experiments required.

- [ ] Confirm the title and objectives with the supervisor before drafting.
      Title, objectives and §8 must agree.
- [ ] Adopt the proposed title in
      `NewDirection/thesis-topic-reidentification.md` only once agreed.
- [ ] Reduce six objectives to two.

## Stage 3. Statistical testing

The Results template requires a p-value and an effect size on the main
comparison. Nothing in the analysis currently produces either.

Two tests are needed, and they are not the same test.

**Risk side, a difference test.** Uniqueness under the leaky release against the
safe release. There is one number per configuration, so a distribution has to
come from bootstrap resampling of records. B = 2000 resamples, paired
differences, Wilcoxon signed-rank plus Cohen's d. Expect a very large effect.

**Utility side, an equivalence test.** The claim is that the two releases give
identical model utility, which is a claim of no difference. A non-significant
Wilcoxon does not establish that; absence of evidence is not evidence of absence.
The correct instrument is **TOST**, two one-sided tests, against a
pre-registered equivalence margin. Set the margin before running it, and justify
it as the smallest AUC-PR difference that would change a custodian's decision.

This distinction is worth a sentence in the Methods. An examiner who knows
statistics will ask how equivalence was established, and "p was not significant"
is the wrong answer.

- [ ] Implement bootstrap paired comparison in `stats_validation.py`.
- [ ] Implement TOST with a stated margin.
- [ ] Report both with effect sizes, per the settled convention of reporting d
      beside every p regardless of significance.

## Stage 4. Figures and tables

The Results template names seven required items and caps the total at five to
seven. Vector or 300 dpi, caption below, referenced in text before appearing,
no orphans.

| Item | Content | Source |
|---|---|---|
| Table 1 | Descriptive statistics of the three corpora | consolidation outputs |
| Figure 1 | System architecture | to draw |
| Figure 2 | Data fusion, three strata into one procedure | to draw |
| Figure 3 | Baseline against proposed, the one modification highlighted | to draw |
| Algorithm 1 | Derivation-consistent generalisation, the modified step marked | to write |
| Table 2 | Ablation, revert to source-derived features | `experiment_c_*.csv` |
| Table 3 | Main comparison, risk and utility, mean and SD | `risk_utility_frontier.csv` |
| Figure 4 | Risk-utility frontier, three strata | `risk_utility_frontier.csv` |
| Figure 5 | Cross-validation stability box plot across 25 folds | new run, needs per-fold scores retained |
| Figure 6 | Attribute identifying power, the SHAP analogue | `solo_identifying_power.csv` |
| Figure 7 | Reconstruction success by band width | `experiment_b_*.csv` |

That is eleven candidates for five to seven slots, so some move to supplementary
material. Figures 1 to 3 and Algorithm 1 belong to Methods; the rest to Results.

- [ ] `risk_utility.cross_validated_utility` currently returns means only.
      Retain per-fold scores so Figure 5 and the Wilcoxon are possible.
- [ ] Build a `figures.py` with one function per figure, 300 dpi, consistent
      palette, no seaborn defaults.
- [ ] Match the template's visual conventions: dashed blue-grey for baseline,
      solid green for proposed, value labels, no caption inside the image.

## Stage 5. Methods section

Target 1,800 to 2,200 words, 10 to 12 citations, fourteen subsections in the
template's fixed order. Past tense for what was done, present for standing facts,
never "I", always "we" or passive.

Subsections and what fills each:

1. Study design and system overview, plus Figure 1
2. Research setting and data acquisition, public and field components
3. Hybrid data strategy, replication across three strata rather than fusion
4. Ethical compliance and data governance, needs the approval reference number
5. Data preprocessing, consolidation, HMAC pseudonymisation, sequence assembly
6. Feature engineering, the eight derived features and their definitions
7. Baseline specification, standard generalisation
8. The engineering contribution, derivation-consistent generalisation, one
   operation, plus Figure 3 and Algorithm 1
9. Training configuration and computing environment
10. Experimental design and reproducibility, five folds by five seeds, fixed seeds
11. Evaluation metrics, disclosure risk and utility, each against its floor
12. Statistical analysis, bootstrap Wilcoxon and TOST
13. Ablation study, plus Table 2
14. Reproducibility artefacts, repository, and what cannot be shared

The template's own warning applies to §3. Our design is replication across
strata, not fusion. Say so plainly rather than forcing it into their fusion
vocabulary, and justify why merging would be wrong: the grading scales, GPA
definitions and sequence lengths differ.

## Stage 6. Results section

Target 1,500 to 2,000 words, 4 to 8 citations, five to seven figures and tables.

The rule the template calls the most common fatal error: **no discussion in
Results.** No "this demonstrates", no "likely due to", no "consistent with prior
work", no implications, no recommendations. Numbers and observations only.

Eight subsections: descriptive summary, baseline against proposed,
cross-validation stability, ablation, statistical significance, attribute
importance, error analysis, computational efficiency.

Two adaptations are needed because this is a disclosure study rather than a
classifier study. "Feature importance" becomes attribute identifying power, and
"error analysis" becomes attack success and containment. Both keep the template's
intent, which is to show where the method works and where it does not.

- [ ] Measure runtimes for §8. Nothing has been timed.

## Stage 7. Verification

Both templates end in a supervisor checklist. Run them literally before showing
anyone.

- [ ] Every figure referenced in text before it appears
- [ ] Every caption standalone and below the item
- [ ] No interpretation language anywhere in Results
- [ ] Every number as mean and SD, not a bare mean
- [ ] Significance reported with p and effect size together
- [ ] Word counts inside range for both sections
- [ ] Citations inside range for both sections
- [ ] Title, objectives and §8 all name the same single operation
- [ ] No result quoted that did not come from a run on real data

## Still open from the research side

Neither blocks writing, and both strengthen the Discussion.

- **Phase 6.** Convert the frontier into a written de-identification standard
  with worked examples. Short, and the practical contribution.
- **Differential privacy comparison.** Without it the study defends k-anonymity
  alone, which has known weaknesses. An examiner will raise it. Even a small
  comparison arm removes the objection.
- **Read `MOCK QUESTIONS_THESIS EXAMINERS & Q1 JOURNAL REVIEWER.pdf`** before
  drafting the Discussion, and answer each question the analysis can answer.

## Sequence

Stages 1 and 2 first, because everything downstream depends on the framing and
on the third stratum existing. Stage 3 next, since Table 3 cannot be written
without the tests. Stage 4 before Stage 6, since Results is written around the
figures. Stage 5 can run in parallel with Stage 4.

The one dependency outside our control is the ethics approval reference number
for Methods §4. Retrieve it now rather than leaving a placeholder that survives
to submission.
