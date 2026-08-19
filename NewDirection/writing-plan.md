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
| A public dataset **and** primary field data | Methods template, called non-negotiable | Done. Stage 1. |
| Exactly one engineering operation, F/M/R/S | Methods §8 | Done. Stage 2. `derivation_consistent.py`. |
| Title naming the artefact, matching §8 | Methods Part 0 | Drafted, awaiting the supervisor decision. |
| Two objectives, each mapping to a subsection | Methods Part 0 | Drafted, awaiting the supervisor decision. |
| p-value **and** effect size on the main comparison | Results §5 | Done. Stage 3. |
| Figures at 300 dpi or vector | Results, placement rules | Done. Stage 4. PNG and PDF. |
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

- [x] Add a `load_public()` loader beside the two existing strata.
- [x] Sequence columns are G1 to G3. `derive_features` handles a three-element
      sequence; the halves split one and two, so `gpa_trend` is asymmetric.
      Defensible, and stated in the write-up rather than left implicit.
- [x] Re-run notebooks 04 to 07 with three strata.
- [x] Record the demographic quasi-identifier result separately. Demographics
      narrow a linkage candidate set from 354 records to 17, which grades alone
      never achieve in that corpus.
- [x] Two bugs the third corpus exposed and both now fixed: band widths and
      recall tolerances were absolute rather than scale-relative, and the
      sensitive-attribute threshold broke on tied integer grades.

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

- [x] Consolidate the artefact into `derivation_consistent.py`, so the thing the
      title names has one home. `risk_utility` now delegates to it.
- [x] Draft two objectives alongside the existing six, not replacing them.
- [ ] Confirm the title and objectives with the supervisor before drafting.
      Title, objectives and §8 must agree.
- [ ] Adopt the proposed title and the two objectives together, only once
      agreed. Both are drafted and waiting in the topic document.

## Stage 3. Statistical testing. DONE

Completed 19 August 2026. Specification frozen in
`NewDirection/analysis-specification.md`, results in
`docs/disclosure-risk-findings.md`, tables in
`results/disclosure/confirmatory_*.csv`.

The plan changed twice while running, both times because a check failed rather
than because a result was unwelcome. Both changes are documented rather than
tidied away.

**Equivalence testing was removed.** TOST was specified while the two arms were
believed to perform identically. Correcting the utility model showed differences
of -0.027, -0.135 and -0.067, so an equivalence test at any defensible margin
fails everywhere and conveys nothing the point estimate does not. The paired
difference and its interval are reported instead, which is strictly more
informative because a reader with a different tolerance can apply it directly.

**delta left the hypotheses entirely.** It survives only as a tolerance a
custodian brings to the frontier, with 0.02 as a labelled illustrative default.
Because it enters no test, when it was chosen no longer matters, which dissolves
the pre-registration problem rather than arguing around it.

**The ordinary bootstrap was rejected on evidence.** Duplicated records are not
unique, so resampling with replacement changes the estimand rather than
perturbing the estimator, and asymmetrically between arms. Subsampling without
replacement at 80% replaced it. The diagnostic that caught it was built in
beforehand.

**Effect sizes are unstandardised.** Both metrics are bounded and interpretable
in their own units, and standardising would divide by a variance that repeated
cross-validation makes ambiguous.

- [x] Subsample-difference comparison for risk, in `stats_validation.py`
- [x] Paired fold comparison for utility, naive and dependence-corrected
- [x] Per-fold scores retained in `risk_utility`
- [x] `notebooks/08_confirmatory_tests.py` runs both across three corpora
- [ ] Adviser to confirm the fold-dependence correction suits this design

### What the numbers say, stated at the strength they support

Risk reduction is distinguishable from zero in eight of nine configurations.
Utility cost is distinguishable from zero in six of nine: every nursing and every
public configuration, and none of the three allied health configurations, where
corrected p values are 0.890, 0.361 and 0.854.

The allied health null is a power result at 109 modelled records, not evidence of
no cost, and it coheres with Phase 5 finding that same corpus too small to
protect at all.

Monotonicity is not claimed. Allied health utility does not order with band
width and public risk reverses between the two widest bands. The correct wording
is that wider generalisation generally produced greater risk reduction
accompanied by greater utility loss across the evaluated configurations.

An earlier summary of this stage claimed that no interval crossed zero on either
axis. That was wrong, and the correction is recorded here and in the findings
document rather than quietly amended.

## Stage 4. Figures. DONE

Stage 4 is communication, not new science. Nothing here should introduce another
layer of methodological complexity; the analysis is finished.

Four figures, each telling one story. The Results template caps figures and
tables at five to seven, so these four plus two tables fits with room to spare.

**Figure A, the mechanism.** The central contribution in one image.
Generalisation lowers risk; derived features computed from the originals raise it
again; recomputing them from the protected values holds the reduction. Baseline
against proposed on the same axis.

**Figure B, the frontier.** Risk reduction against utility retained, all
configurations, all three corpora, Pareto set marked. This is the decision
figure and the one a custodian would actually use.

**Figure C, stability.** Distribution of the paired per-fold differences, now
that the scores are retained. Shows the effect is not one lucky average, and
shows honestly that the allied health distribution straddles zero.

**Figure D, the explanation.** A flow diagram from generalisation through
derivation to the two outcomes, risk and utility, for one representative
configuration. Non-specialist readable, and the eventual basis of the prototype
tool's main screen.

Methods additionally needs Figure 1 (architecture), Figure 3 (baseline against
proposed, the one modification marked) and Algorithm 1. Table 1 describes the
corpora; Table 2 carries the confirmatory results.

- [x] Per-fold scores retained, and written to
      `results/disclosure/confirmatory_fold_differences.csv`, 225 rows
- [x] `figures.py`, one function per figure, 300 dpi PNG and PDF, no seaborn
- [x] Baseline blue-grey, proposed green, values labelled on the marks, no
      caption inside any image
- [x] Figure C shows every fold as a jittered point over the box, and colours
      the three allied health distributions differently because they straddle
      zero. The corpus that does not support the claim is the most visible thing
      in the figure

Figures are written to `results/disclosure/figures/`, which is gitignored along
with the rest of `results/`. `figures.py` is tracked, so any figure is one
command away and no stale image can drift out of step with the tables.

- [ ] Methods still needs Figure 1 (architecture), Figure 3 (baseline against
      proposed with the modification marked) and Algorithm 1. All three are
      diagrams of the procedure rather than plots of results, so they are drawn
      when Methods is drafted

### One caution for the figures

Do not let a diagram assert more than the experiment supports. The defensible
claim is that the results are consistent with derived features acting as an
information pathway that substantially reverses the privacy effect of
generalisation. Arrows implying proven identity of information would overstate
it, and the thesis does not need that claim.

## Stage 5. Methods section. DRAFTED

Target 1,800 to 2,200 words, 10 to 12 citations, fourteen subsections in the
template's fixed order. Past tense for what was done, present for standing facts,
never "I", always "we" or passive.

Everything the section describes has now been run, so no subsection needs to be
written in the future tense or left as a placeholder. Three things are still
outstanding and each has a defined handling:

| Outstanding | Handling in the draft |
|---|---|
| Ethics approval reference number | A visible `[REF]` marker, never a plausible-looking invented number |
| Adviser confirmation on the fold-dependence correction | Write the method as applied and report both corrected and naive, which is true regardless of the outcome |
| Title and two objectives | Draft against the current title. Switching later changes three places, all of them known |

Four decisions from the analysis belong in the Methods and would look like
evasions if they surfaced first in the Results:

1. The utility model receives what a recipient receives, and why passing only
   the source columns made the two arms identical by construction.
2. The ordinary bootstrap was rejected on evidence, with the diagnostic.
3. Equivalence testing was removed once the arms were found to differ.
4. Attribution is measured from below because uniqueness saturates at two
   attributes and leave-one-out returns zero everywhere.

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
12. Statistical analysis: subsampling for risk, paired folds with a
    dependence correction for utility, the rejected bootstrap and why
13. Ablation study, plus Table 2
14. Reproducibility artefacts, repository, and what cannot be shared

The template's own warning applies to §3. Our design is replication across
strata, not fusion. Say so plainly rather than forcing it into their fusion
vocabulary, and justify why merging would be wrong: the grading scales, GPA
definitions and sequence lengths differ.

### Draft state

`manuscript/methods.md`, 15 subsections, 2,420 words excluding headings.

Over the template's 1,800 to 2,200 best-fit band, inside the 1,500 to 2,500 it
quotes as the Elsevier norm, and well under the 2,800 at which it warns of a
desk request to shorten. Every one of the 22 factual claims in the draft was
verified against the code rather than transcribed from notes.

The excess is not padding. Six methodological decisions each need justifying
because each pre-empts a specific question: three corpora that are not pooled,
a resampling scheme rejected on evidence, an equivalence test specified and
removed, a corrected utility model, attribution measured from below, and a
governance section for the one corpus carrying real identifiers.

If the supervisor wants 2,200, cut in this order and lose about 220 words:

1. Section 2.11, the interval-narrowing mechanics, to two sentences. The detail
   belongs in the algorithm rather than the prose.
2. Section 2.15, reproducibility, to the module list and the data-sharing
   position.
3. Section 2.2, the per-corpus descriptions, once Table 1 carries the same
   figures.

Cut nothing from 2.13. The rejected bootstrap and the removed equivalence test
are the two places the section demonstrates that the analysis checked itself.

### Still needed alongside

- [x] Figure 1, architecture. Four phases end to end
- [x] Figure 2, replication across three corpora, no fusion point
- [x] Figure 3, baseline against proposed with the modified step marked
- [x] Algorithm 1, modification identified at line 3, text in the manuscript
      and a rendered box for layout
- [x] Table 1, corpus descriptives, generated from the corpora rather than typed
- [ ] Ethics reference number, replacing the visible `[REF]` marker

Placement rules verified: every figure and table is referenced in the text
before it appears, each carries a caption below it, none is an orphan, and no
invented ethics number appears anywhere in the file.

## Stage 6. Results section. DRAFTED

Target 1,500 to 2,000 words, 4 to 8 citations, five to seven figures and tables.

The rule the template calls the most common fatal error: **no discussion in
Results.** No "this demonstrates", no "likely due to", no "consistent with prior
work", no implications, no recommendations. Numbers and observations only.

Eight subsections: descriptive summary, baseline against proposed,
cross-validation stability, ablation, statistical significance, attribute
importance, error analysis, computational efficiency.

Two adaptations were made because this is a disclosure study rather than a
classifier study. "Feature importance" became attribute identifying power, and
"error analysis" became attack success and containment. Both keep the template's
intent, which is to show where the method works and where it does not.

### Draft state

`manuscript/results.md`, ten subsections, 1,573 words of prose.

Checklist run literally, all passing:

| Requirement | Status |
|---|---|
| Word count 1,500 to 2,000 | 1,573 |
| Figures and tables, 5 to 7 | 6 (Figures 4 to 6, Tables 2 to 4) |
| In-text citations, 4 to 8 | 4 |
| Every item referenced before it appears | yes |
| Every item captioned | yes |
| No interpretation language | no banned phrase present |
| Means reported with SD | yes |
| p reported with effect size | yes, effect size is the raw difference |
| No-skill floors stated | yes |

Writing it surfaced a Methods-Results inconsistency worth recording. Section 2.10
lists prosecutor risk and the share of records below k thresholds as reported
measures, and the first draft of Results gave neither. Both are now reported in
Section 3.2. Drafting the two sections against each other is what caught it; a
checklist alone would not have.

The explanation figure is deliberately not in Results. It carries an
interpretation of the mechanism, which belongs in the Discussion, and Results is
already at six items against a cap of seven.

- [x] Runtimes measured for the efficiency subsection: 3.6 ms to build a
      release, 5.2 ms for a risk profile, 0.17 s for the reconstruction attack
      over 566 records, 2.66 s for a full 25-fold utility evaluation. They belong
      in Results section 8 rather than Methods, per the template.

## Stage 7. Verification. PARTLY DONE

Both sections were audited against `proposal/MethodsSection/MARKING SCHEME_METHOD
SECTION.pdf`, which is a harder document than the templates: pass is 90% and it
states that no draft below 90% may go to the supervisor.

### What the audit forced

The first draft **failed Part A** on item A8, title and objectives not matching
the F/M/R/S operation described in Section 8. The title was a question and named
no operation. That settled a decision held open since Stage 2: under this scheme
the proposed title and the two-objective structure are not optional, they are
required to pass the gate. Both are now adopted.

Seven further items were missing and are now supplied:

| Rubric item | Was | Now |
|---|---|---|
| A6 public dataset provenance | citation only | source URL, retrieval date, licence, DOI |
| 3 fusion strategy | "no fusion applied" | cross-validation replication, named against the four-type taxonomy and justified |
| 4 ethics path | unstated | Path A named, with the Declaration of Helsinki and Act 843 |
| 5 preprocessing | qualitative | missingness as exact percentages, class ratios as numbers |
| 6 feature engineering | features named | Table 1 feature dictionary with the formula for all eight |
| 9 training configuration | prose | Table 2, every parameter itemised |
| 11 evaluation metrics | prose | Table 3, every metric with its one-line justification |
| 13 ablation | described | point-change stated numerically, 0.129 to 0.875 |
| 16 supervisor checklist | absent | Section 2.16, with the one unmet item stated rather than ticked |
| G4 citations | 1 | 12, inside the 10 to 14 target |

### Current state

| Gate | Methods | Results |
|---|---|---|
| Part A | PASS | n/a |
| G1 tense and voice | PASS | PASS |
| G2 placeholders | **FAIL, caps at 70%** | PASS |
| G3 length | 2,729 (deduction over 2,800) | 1,641, in range |
| G4 citations | 12, in range | 4, in range |
| Required items | all present | 6 of a 5 to 7 cap |
| No discussion in Results | n/a | PASS |

### The one thing blocking a pass

The ethics reference reads `CHRPE/AP/XXX/26`. Any unfilled placeholder caps the
score at 70% under G2, and the threshold is 90%. Nothing else in either section
now caps the score.

The placeholder is deliberately in the committee's own reference format so its
shape is right when the real number arrives, and it is flagged three times: in
the draft-status note at the head of the section, in Section 2.4 itself, and as
the single unmet row in the Section 2.16 self-audit. It is stated rather than
ticked there, because a self-audit that claims an item it does not satisfy is
worth less than one that admits the gap.

- [ ] Replace `CHRPE/AP/XXX/26` with the issued reference. This is the highest
      value action available and is the only remaining cap.
- [ ] Optional: trim Methods from 2,729 toward the 2,200 safe range. No deduction
      applies below 2,800, so this is polish rather than a requirement.

### Figures were referenced but not placed

Caught by inspection, not by the audit. The first audit checked rubric section 15
by testing whether the string "Figure 1" appeared in the file. That tests for a
reference, not for an image, so it reported a pass while both manuscripts
contained zero image embeds and the three Methods figures had no captions at all.
Section 15 would have scored 1 of 3.

Now fixed and verified properly: every figure is embedded, the file it points to
exists on disk, it carries a caption below it, and it is referenced in the text
before it appears. Both PNG at 300 dpi and PDF are produced for each.

Figure files were also renamed to carry their manuscript number. They had been
`figure_a` through `figure_d`, which meant anyone placing them had to remember
that a maps to 4. They are now `figure_4_mechanism` and so on, and the old names
were deleted rather than left as stale duplicates.

The general lesson, and it has now happened three times in this project: a check
that a command succeeded is not a check that it produced what it should. Test for
the artefact, not for the mention of it.

## Stage 7. Original template checklists

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

## Stage 8. The prototype tool, only after the writing

Optional, and scheduled last on purpose. The thesis must be complete and
defensible without it.

Two additions to the analysis it depends on, both small and both worth doing
regardless:

- [ ] SHAP over the utility model, on the unprotected release against each
      protected one. AUC-PR says whether the analysis survives; it does not say
      whether the model relies on the same features afterwards. A shift in
      importance at constant performance is a utility finding that the scalar
      hides. Reuses the existing SHAP infrastructure.
- [ ] A single function that turns a release configuration into the numbers a
      custodian needs: risk, drivers, leakage, retained utility, recommendation
      against a threshold. The notebooks already compute all five separately.

Then the tool itself:

- [ ] Aggregate by construction. Reports over a dataset, never over a record. No
      record-level output rendered, exported or logged.
- [ ] Real data stays in the approved environment. Any demonstration outside it
      runs on the replication corpus or on synthetic input.
- [ ] Reuse the existing Streamlit chassis for layout and deployment only. Its
      upload-and-explain-each-student flow is the opposite of what this study
      argues and must not be carried across.

If the schedule slips, drop this. It is the last thing in and the first thing
out.

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
