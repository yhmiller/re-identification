# Shortlist — Health Informatics Candidates

**Compiled:** 14 August 2026 · **Revised:** 16 August 2026
**Scope narrowed:** supervisor has directed that the thesis stay within health (MSc Health Informatics). Agriculture, horticulture and export/trade topics are **out of scope and removed**.
**Full detail:** [Health tech & health education — all 14 topics](health-tech-education-thesis-topics.md)
**Archived, out of scope:** [Agriculture & horticulture](ghana-agriculture-thesis-topics.md) · [The five CANDO topics](cando-five-topics-assessment.md)

**Selection rule applied throughout:** every candidate's core model runs without any institution granting access to anything.

---

## ✅ Selected — S1. Re-identification risk in health professions education records

*Beyond Name Removal: Quantifying Re-identification Risk and the Privacy Cost of Derived Features in Ghanaian Health Professions Education Records*

**→ Full proposal-ready write-up: [thesis-topic-reidentification.md](thesis-topic-reidentification.md)**

**The problem.** Health training institutions "anonymise" student records by deleting names and index numbers, then share them. That is not anonymisation. A six-semester GPA sequence plus programme and cohort year is a fingerprint — near-unique per student. An adversary with ordinary knowledge (a classmate, a lecturer, an employer) can isolate a record and read the sensitive attribute at the end of the chain: licensure outcome.

**What you build.** Four models — a disclosure risk model, a reconstruction attack, a linkage attack, and a constrained optimisation producing a risk–utility frontier.

**Why it was chosen over everything else:**

1. **No dependent variable.** The blocked registrar file is an *outcome* file. This study has no outcome to predict, so it cannot be blocked by the thing that blocked the last one.
2. **Reuses the existing dataset and pipeline.** The XGBoost work becomes the *utility measurement instrument* — how much model performance survives each level of protection. Nothing built to date is discarded.
3. **A genuinely novel contribution that came out of your own repo.** Do derived features (`gpa_trend`, `gpa_min`, `gpa_consistency`, `n_weak_semesters`) leak? Released at full precision alongside generalised source variables, they may reverse the protection — `gpa_min` and `gpa_max` are exact order statistics of the vector you just banded. Nobody has tested this, and it generalises to any health ML pipeline.
4. **Measured, not estimated risk.** If the dataset covers all students in the relevant cohorts, sampling fraction is one and sample uniqueness *is* population uniqueness — a stronger position than most published work in this area.
5. **Squarely health informatics.** Privacy, disclosure control, governance and secondary use of health data are core to the discipline.

**Feasibility: 9/10** · Ethics approval required · No new data needed

**Week-one go/no-go:** what share of records is unique on `programme` + `cohort_year` + the GPA sequence? If near-total as expected, the headline finding exists in week one.

---

## Alternatives, if the selected topic is blocked

### B1. Federated learning for Ghanaian health data sovereignty
*Training Clinical Prediction Models Without Moving Patient Data*

**The problem.** Ghana's clinical data is trapped facility-by-facility. Pooling collides with the Data Protection Act, institutional reluctance, and the absence of an in-country hyperscale cloud. Federated learning dissolves the trade-off in principle — the model travels to the data. Whether it survives Ghanaian conditions (thin bandwidth, unreliable power, unequal facility sizes) is unstudied.

**What you build.** A working federated system (Flower) with client-side DP-SGD and secure aggregation; partitions calibrated to Ghana's real facility landscape; a factorial sweep over federation size, heterogeneity, privacy budget ε, bandwidth and dropout; a response-surface model; and membership-inference attacks that turn ε into a measured risk.

**Why it's strong.** Highest ceiling of any candidate. Nothing can block it. The skills (federated learning, differential privacy, secure aggregation, privacy attack evaluation) are among the most sought-after in health data engineering globally.

**Its real cost.** Three weeks of engineering before any science. Facilities are *calibrated to* published Ghanaian characteristics, not drawn from Ghanaian data.

**Note:** shares intellectual territory with the selected topic — both are privacy-preserving health data work. If S1 is blocked on governance grounds, this is the natural substitute.

**Feasibility: 9/10** · No ethics approval for the core experiment

---

### B2. Drone and road medical logistics network design
*A Location-Allocation Model for Drone and Road Resupply of Health Facilities Under Stockout Risk*

**The problem.** Ghana operates one of the world's largest medical drone delivery networks. It is internationally celebrated and essentially unmodelled in open literature.

**What you build.** Facilities as georeferenced demand nodes; drone and road travel-time matrices with wet- and dry-season variants; a capacitated maximal covering location problem (MILP) extended with an inventory-pooling layer; a stochastic variant plus discrete-event simulation under shocks.

**Why it's strong.** Broadest technical demonstration — geospatial engineering, MILP, stochastic optimisation, DES. Ghana is the global reference case, so the work travels. Data is almost entirely free geospatial.

**Its real cost.** Demand and cost parameters are estimated rather than observed. Uses none of your existing data.

**Feasibility: 8/10** · No ethics approval

---

### B3. Health workforce pipeline simulation
*From Classroom to Departure Lounge: Stock-and-Flow Simulation Under Training Expansion and Emigration Pressure*

**The problem.** Ghana simultaneously trains large numbers of health professionals, has qualified graduates awaiting posting, and loses experienced staff abroad. These are debated as three separate problems. They are stages of one feedback system: backlog raises emigration, emigration relieves backlog, relief reduces pressure to create posts, backlog rebuilds.

**What you build.** A stock-and-flow system dynamics model of the full pipeline, with the emigration flow **pinned from public UK register data** reporting registrants by country of training — converting the hardest parameter from an assumption into an estimate. Plus scenario simulation and leverage analysis.

**Why it's strong.** Reframes a live policy debate rather than joining it. System dynamics is nearly absent from African health workforce research.

**Its real cost.** Domestic transition rates are incompletely published; sensitivity analysis is mandatory. Foreign registers miss destinations without public registers.

**Note:** the closest of the alternatives to your existing domain — it models the training-to-workforce pipeline your dataset sits inside — without needing institutional records.

**Feasibility: 8/10** · No ethics approval

---

## Side by side

| | Core model | Data risk | Uses your data? | Demands from you | Feas. |
|---|---|---|---|---|---|
| **S1** ✅ | Disclosure risk + attacks + risk–utility optimisation | **None** | **Yes — directly** | Privacy methods, careful governance | **9** |
| **B1** | Federated learning + DP + factorial + response surface | None | Possible | **Engineering ability** | 9 |
| **B2** | MILP location-allocation + pooling + DES | Very low | No | Geospatial + OR modelling | 8 |
| **B3** | System dynamics + Markov cohort | Low | No | Systems thinking | 8 |

---

## Why S1 over the alternatives

The other three are all good topics. S1 wins on one decisive property the others lack: **it is the only candidate that turns the existing dataset and pipeline into an asset rather than sunk cost, while remaining immune to the missing licensure outcome.**

B1 has an equally high ceiling but starts from zero and demands three weeks of distributed-systems engineering before producing anything. B2 and B3 are strong and unblockable but use none of the work already done.

**The keeper argument:** the blockage was a missing dependent variable. S1 has no dependent variable. That is not a workaround — it is a research design that is structurally immune to the failure that has already cost one thesis.

---

## Before committing

| Candidate | The check to run first | Time |
|---|---|---|
| **S1** ✅ | Uniqueness on `programme` + `cohort_year` + GPA sequence. Confirm cohort coverage is total (sampling fraction = 1). Confirm the data-sharing arrangement covers this research question **in writing**. | 1 day + governance |
| **B1** | Start clinical-database credentialing (has a lead time). Confirm compute for the factorial sweep. | Start week 1 |
| **B2** | Confirm facility registry completeness and OSM road coverage for rural Ghana | 2 days |
| **B3** | Confirm public register granularity — Ghana-trained counts by year and cadre? | 1 day |

The governance check on S1 is the real gate, not the technical one. Consent for a licensure-prediction study does not automatically extend to a disclosure-risk study. **Settle it in writing before Phase 1, not after.**

---

## Verification standing

- No specific statistics are quoted in this shortlist or the source documents. Every figure entering the proposal must come from the primary source with a date attached.
- Legal instruments change and Ghana's data protection landscape is active — read Act 843 and Act 1038 directly and check for amendments.
- Dataset availability and licensing terms change. Confirm before building on them.
