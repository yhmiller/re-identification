# Gap analysis

What the search did to the three gap claims. Written after the search, against
the protocol registered before it.

**Summary: one claim survives narrowed, one narrows materially, one must be
substantially reframed. None is destroyed, and the reframed version of the third
is stronger than the original.**

---

## G1. No measured re-identification risk for Ghanaian health data

**Survives, narrowed.**

What exists is legal and comparative rather than empirical. Ghana appears in
comparative studies of African health data regulation and in the DS-I Africa Law
project's analysis of how twelve jurisdictions treat de-identification. Act 843
is well documented as a legal instrument.

No empirical re-identification risk estimate for Ghanaian health data was found.
The nearest African analogue is a de-identification framework applied to a
paediatric cohort in Uganda, which uses k-anonymity and a risk threshold on real
records. That work does not weaken the Ghanaian claim but it does weaken any
suggestion that the approach is novel for African health data generally.

**Wording that the evidence supports:**

> No empirical re-identification risk estimates for Ghanaian health data were
> located. Comparable measurement has been undertaken elsewhere in the region,
> for example on a paediatric cohort in Uganda, so the contribution is the
> Ghanaian estimate rather than the application of the method to African health
> data.

---

## G2. Health professions education records are unexamined

**Narrows materially. The claim as drafted is not supportable.**

Education data has been studied as a disclosure-control problem, and one source
anticipates a specific finding of this study. Vatsalan et al. (2022) quantify
re-identification risk in education data and explicitly treat grades from
different exam attempts as quasi-identifiers whose combination indicates a
particular student. That is Phase 1's finding, published four years earlier on
other data. A Journal of Learning Analytics paper argues directly that
de-identification is insufficient for student privacy.

The thesis previously described the learning analytics ethics literature as
"largely normative rather than quantitative". That characterisation is wrong and
must be removed.

**Wording that the evidence supports:**

> Re-identification risk in education data has been quantified, and academic
> grades have been identified as quasi-identifiers in that literature. What
> remains unexamined is the health professions case specifically, where the
> record links to professional licensure and to workforce registration, and where
> the sensitive attribute at the end of the chain is career-determining.

---

## G3. Derived-feature leakage under generalisation

**Must be substantially reframed. Three adjacent literatures bear on it, and one
is close enough that ignoring it would be a serious omission.**

### What is already established

**Releasing statistics leaks.** Dinur and Nissim (2003) proved it, the result is
called the Fundamental Law of Information Recovery, and the US Census
demonstrated it against real published statistics. This study's reconstruction
procedure is a small special case of a well-founded family.

**Few aggregates are enough.** DeSIA (2025) shows attribute inference against a
*limited fixed* set of aggregates and concludes that aggregation alone does not
protect privacy even when few statistics are released. This study's Experiment A
reports the same conclusion on eight derived features. **Experiment A should be
presented as replication in a new setting, not as a novel finding.**

**Consistency is already a principle, in tabular official statistics.** The
cell-key method perturbs tables and then runs an additivity-restoring step,
because perturbation breaks the relationship between inner cells and margins.
Agencies enforce that the same cell appearing in different tables receives the
same perturbation. The underlying principle, that a protected quantity and the
quantities derived from it must be treated consistently or the inconsistency is
itself exploitable, is established practice.

### What the search did not find

No source was located that measures the specific configuration studied here:

1. **Generalisation of microdata**, not perturbation of tabular aggregates.
2. Derived features computed from the **original** values and released **in the
   same file** as the generalised source columns.
3. In a **machine learning feature-engineering** setting, where the derived
   features exist because a model wants them.
4. With the **utility cost of the fix measured**, so the trade-off is quantified
   rather than asserted.

Point 4 appears to be the clearest space. The official statistics literature
treats consistency as a correctness requirement to be engineered away, not as a
trade-off with a measurable price. No source was found that asks what consistent
derivation costs the analyst.

### The honest reframing

The original framing was that a failure mode had gone unexamined. The supportable
framing is that a principle established in one field has not been transferred to
another, and that its cost there has not been measured.

**Wording that the evidence supports:**

> That released statistics leak information about their source is established,
> and that a protected quantity and its derivations must be treated consistently
> is standing practice in tabular disclosure control. Neither result appears to
> have been carried into microdata generalisation for machine learning, where
> features derived from unprotected values are routinely published alongside a
> generalised version of those same values. This study measures that
> configuration, and measures what enforcing consistency costs the analyst, which
> the tabular literature does not address because it treats consistency as a
> correctness requirement rather than a trade-off.

This is a narrower claim and a better one. "Nobody has thought of this" invites a
single counter-example. "This is known there, not done here, and here is the
price" invites the examiner to check a transfer argument, which the study can
support.

---

## Effect on the contribution list

| Contribution as drafted | Status |
|---|---|
| First quantified risk estimates for Ghanaian health data | Keep, with the Uganda comparator cited |
| First treatment of health professions education records | Reword. Education data is studied; the health professions case is not |
| Evidence on derived-feature leakage | Reframe as transfer plus cost measurement, citing DeSIA and the cell-key literature |
| Measured rather than estimated uniqueness | Keep. Nothing found challenges it |
| Replication on an independent corpus | Keep |

## Effect on the method

Nothing in the search invalidates a method choice. Two additions are now
warranted:

1. The reconstruction procedure should be positioned against the linear
   reconstruction literature, since the three means are linear in the source
   values and that literature bounds what is recoverable.
2. The framework in Phase 6 should be positioned against the cell-key
   additivity-restoration precedent, which is the same idea in a different
   medium.
