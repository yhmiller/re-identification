# Thesis Topic Options — Health Tech & Health Education in Ghana

**Prepared:** 14 August 2026
**Constraint set:** must build *and apply* a model (CANDO); master's timeframe; R/Python and Stata available; primary data access uncertain; **clean replacement** for prior work — nothing here depends on health-training-institution records.
**Stated interests:** AI / clinical decision support · health systems operations · **health data protection, storage, isolation, interoperability, cloud residency** · health professions education *and* public health education.

---

## Read this first — three framing decisions

**1. Your "not over-explored" constraint eliminates most of the field.** Ghanaian health research is dominated by a handful of formats that are effectively saturated: technology-acceptance models (TAM/UTAUT) applied to mHealth and telemedicine, NHIS enrolment and renewal determinants, knowledge-attitude-practice surveys, DHS-based antenatal-care and skilled-delivery utilisation studies, malaria incidence prediction, and COVID-era vaccine-hesitancy work. **None of these appear below.** Every topic here was selected partly because a literature search will return thin results for Ghana specifically.

**2. Your prior thesis was blocked by a gatekeeper, and that should shape this one.** Waiting on an institution to return a spreadsheet is the single highest-frequency cause of thesis failure, and you have already paid that cost once. So every topic below is built so its **core model runs without any institution granting you anything**. Several run on entirely simulated or public data by design — not as a compromise, but because the research question genuinely doesn't require real patient records.

This has a hard consequence worth stating plainly: **most health professions education topics in Ghana are structurally gatekeeper-dependent** — student records, admission scores, exam outcomes, attrition logs all sit behind a registrar. That is why only *one* health-education topic makes the ranked list, and it is the one built on **public international registers** rather than institutional records. This is a deliberate exclusion, not an oversight.

**3. Anything touching patient data needs ethics approval, and that is a schedule item.** Ghana Health Service Ethics Review Committee, or an institutional IRB (Noguchi Memorial Institute, KNUST, University of Ghana). Realistic turnaround is **four to twelve weeks**, and it must be secured *before* data collection, not after. Topics below are marked for whether they need it. Several do not, which is a real advantage in a compressed timeline.

---

## Your sovereignty question, answered directly

> *"Can we build a system that isolates health records data from any public need and limits access to country-specific?"*

Yes — but the honest answer separates three things that get conflated in vendor marketing.

| What you actually want | The technical name | Achievable in Ghana? |
|---|---|---|
| Data physically stays in Ghana | **Data residency / localisation** | **Partially.** At the time of writing, no hyperscale cloud provider operates a region inside Ghana; the nearest are in South Africa. True in-country residency therefore means national/private data-centre infrastructure (NITA's facilities, or colocation), not a hyperscaler tick-box. **Verify current region availability before you write this** — providers open regions regularly. |
| Data is legally beyond foreign reach | **Data sovereignty** | **Harder than residency.** Residency ≠ sovereignty: a foreign-owned operator can be subject to foreign legal process regardless of where servers sit. This distinction is the single most under-examined point in African digital-health policy and is a genuine contribution if you make it rigorously. |
| Useful analysis happens *without* the data ever moving | **Federated analytics / privacy-preserving computation** | **Fully achievable, and this is the strong answer.** Models travel to the data instead of data travelling to the model. Records never leave the facility, let alone the country. |

The third row is where the thesis is. It converts a policy aspiration into an engineering-and-modelling problem with measurable outcomes, and it is genuinely under-researched for African health systems operating under real infrastructure constraints. Topics 1–5 below all sit in this space; **Topic 1 is my overall recommendation.**

**Legal instruments you must read directly (do not cite secondary summaries):** Ghana's Data Protection Act, 2012 (Act 843) and the Data Protection Commission's guidance; the Cybersecurity Act, 2020 (Act 1038) and the Cyber Security Authority's critical-information-infrastructure designations; the African Union Malabo Convention on Cyber Security and Personal Data Protection. **Check for amendments and new instruments before committing** — this area is legislatively active and anything written more than a year ago may be stale.

---

## Part 1 — The health data landscape, tiered by access risk

### Tier A — Free and public today (no gatekeeper)

| Source | What it gives you |
|---|---|
| **Ghana DHS (DHS Program)** | Nationally representative household health microdata, multiple waves, freely downloadable on registration. Ghana's most under-used analytical asset. |
| **Ghana MICS (UNICEF)** | Child and maternal health indicators, microdata |
| **GSS 2021 Population & Housing Census** | District-level population, housing, disability, WASH |
| **WHO Global Health Observatory + National Health Workforce Accounts** | Ghana health workforce stocks and densities by cadre |
| **UK NMC and GMC public registers** | Registrant counts **by country of training** — makes Ghanaian health-worker emigration directly observable without asking Ghana for anything |
| **IHME Global Burden of Disease** | Ghana burden, mortality, YLL by cause, subnational where available |
| **healthsites.io / OpenStreetMap / GRID3** | Health facility locations and attributes |
| **WorldPop** | Gridded population, 100m |
| **Malaria Atlas Project friction surface** | Travel-time/accessibility modelling inputs |
| **VIIRS night-time lights** | Electrification and grid-reliability proxy — free, monthly, and badly under-used for health infrastructure work |
| **OpenCelliD / ITU / GSMA coverage data** | Mobile and broadband coverage |
| **ERA5, CHIRPS, MERRA-2** | Temperature, rainfall, and aerosol/dust (dust matters for meningitis) |
| **WHO AFRO IDSR weekly bulletins** | Notifiable-disease surveillance counts |
| **MIMIC-IV, eICU, PhysioNet** | Public clinical databases — for method development and federated-learning simulation where Ghanaian records are unavailable |

### Tier B — Formal request, usually granted, but a queue you don't control

Ghana Health Service **DHIMS2** aggregate extracts; GHS Holistic Assessment and annual performance reports; NHIA annual reports and aggregate claims statistics; Ministry of Health sector performance reviews; Nursing & Midwifery Council and Medical & Dental Council register summaries; Data Protection Commission registration data; National Ambulance Service deployment records.

### Tier C — Relationship-dependent, may never arrive (never build a core model on these)

**LHIMS patient-level records** · NHIA claim-level data · Zipline operational logs · National Blood Service transaction records · hospital EHR extracts (Korle-Bu, KATH) · **health training institution student records** ← *this is the exact category that blocked your previous thesis. Nothing below depends on it.*

---

## Part 2 — The fourteen topics

### Cluster A — Health data sovereignty, privacy and interoperability

---

### Topic 1 — Federated learning for Ghanaian health facilities

**1. Proposed title**
*Training Clinical Prediction Models Without Moving Patient Data: A Federated Learning Architecture for Ghana's Health System Under Real Infrastructure and Data-Protection Constraints*

**2. Research problem / gap**
Ghana's clinical data is trapped. Records accumulate facility-by-facility in LHIMS deployments and paper systems, and the analytical value of pooling them is obvious — but pooling collides with the Data Protection Act, with institutional reluctance, and with the absence of an in-country hyperscale cloud. The result is a national dataset that exists in aggregate but can never be assembled. Federated learning dissolves this trade-off in principle: the model travels to each facility, trains locally, and only parameters are shared. In principle. Whether it works in *Ghanaian* conditions — where district hospitals have intermittent power, thin and expensive bandwidth, wildly unequal data volumes, and case-mix that differs sharply between a teaching hospital and a CHPS compound — is unstudied. The entire federated-learning literature assumes infrastructure Ghana does not have.

**3. Research questions**
1. How much predictive accuracy is lost when a clinical model is trained federatively across heterogeneous Ghanaian facilities instead of on pooled data?
2. How does that loss vary with the degree of non-IID data distribution — i.e. with realistic Ghanaian case-mix and facility-size imbalance?
3. What is the privacy–utility frontier when differential privacy is added, and what privacy budget is defensible under Act 843?
4. Is the approach viable under realistic Ghanaian bandwidth, latency and node-dropout conditions, and what is the communication cost per unit of accuracy gained?

**4. Variables**
- **Dependent:** model performance (AUC-ROC, AUC-PR, calibration slope/intercept, Brier score); communication cost (MB transferred, rounds to convergence); wall-clock training time under simulated bandwidth.
- **Independent / experimental factors:** federation size (K facilities); non-IID severity (Dirichlet concentration parameter α); facility-size imbalance ratio; aggregation algorithm (FedAvg / FedProx / SCAFFOLD); differential-privacy budget ε; node-dropout rate; bandwidth cap; client-participation fraction per round.

**5. Model to build**
A full **federated learning system plus a factorial evaluation experiment** — this is both an engineered artefact and a quantitative study, which is exactly what a "build and apply a model" requirement wants.

- *The system:* a federated training pipeline (Flower or NVFlare) with a central aggregation server and K simulated facility clients, each holding a private partition. Client-side **differential privacy** (DP-SGD) with a tunable ε. Secure aggregation so the server sees only summed updates.
- *The clinical model:* a task chosen to be genuinely useful and low-input — e.g. maternal deterioration risk, paediatric malaria-anaemia severity, or hospital readmission — implemented as logistic regression and as a small neural network, so you can separate "federation costs" from "model-class costs."
- *The experiment:* a designed factorial sweep across the factors above, with centralised training as the upper baseline and facility-local-only training as the lower baseline. The headline output is a **privacy–utility–bandwidth frontier**: for a given ε and bandwidth budget, how much accuracy can Ghana buy without a single record leaving a facility?
- *The architecture layer:* a deployment design mapped explicitly against Act 843 and Act 1038 obligations, with a threat model covering gradient-inversion, membership-inference and malicious-server attacks.

**6. Data required**
**No Ghanaian patient data is required for the core experiment.** You partition a public clinical database (MIMIC-IV, eICU, or a public African clinical dataset) into K synthetic "facilities" using a non-IID partition calibrated to *published* Ghanaian facility characteristics — bed counts, admission volumes, case-mix profiles, referral tiers — which are available from GHS reports and the facility registry. Ghanaian bandwidth and outage parameters come from ITU/GSMA and VIIRS-derived electrification measures. If a real de-identified Ghanaian extract later becomes available, it becomes a validation chapter, not a dependency.

**7. Ghanaian / external sources**
GHS facility registry and Holistic Assessment reports (facility tiering and volumes, for realistic partitioning); DHIMS2 aggregate service statistics (case-mix priors); ITU/GSMA and NCA data (bandwidth); VIIRS night-lights (power reliability); Data Protection Commission and Cyber Security Authority instruments (compliance mapping); MIMIC-IV/PhysioNet (the clinical corpus); Ministry of Health digital health strategy documents.

**8. Unit of analysis**
The facility (federation client) for the system; the experimental run (a factor combination) for the evaluation; the patient record within each client partition.

**9. Methodology**
Characterise Ghana's facility landscape from public reports → derive a realistic non-IID partition scheme → implement the federated pipeline → implement DP-SGD and secure aggregation → run the factorial sweep with repeated seeds → fit a **response-surface model** relating accuracy to ε, K, non-IID severity and bandwidth (so the results generalise beyond the specific points you tested) → run privacy attacks (membership inference) to empirically validate the theoretical ε → map the architecture to legal obligations → produce deployment recommendations with a costed reference architecture.

**10. Expected contribution**
*Academic:* the first empirical characterisation of federated learning viability under African health-system infrastructure constraints, with a privacy–utility–bandwidth frontier that is calibrated rather than assumed. The response-surface model is itself reusable by other low-resource health systems. *Practical:* a concrete answer, with numbers, to whether Ghana's Ministry of Health can get national-scale clinical AI without national-scale data pooling — and a reference architecture that satisfies Act 843 by construction rather than by policy promise.

**11. Feasibility: 9/10** · *No ethics approval required for the core experiment* (no human subjects — public de-identified data plus simulation).
Everything needed is downloadable. The work is computational and entirely within your control. The only real cost is compute time for the factorial sweep, which is manageable on a laptop for logistic models and cheap cloud GPU for the neural variant.

**12. Challenges / limitations**
- **The honest central limitation:** simulated facilities are not real facilities. You must be explicit that the partition is *calibrated to* published Ghanaian characteristics, not drawn from Ghanaian data, and you must not overclaim external validity. Handled well, this is a limitation; handled badly, it's a fatal criticism — so pre-empt it in the design chapter.
- Differential privacy budgets are notoriously hard to interpret in policy terms; the membership-inference attack experiments are what make ε meaningful, so do not cut them.
- Requires real engineering competence (distributed training, privacy libraries). This is a strength for demonstrating skill and a risk if you underestimate the build time. Budget three weeks for the pipeline before any experiments run.
- The legal-compliance chapter needs care: you are not a lawyer, so frame it as a requirements-mapping exercise, not legal advice.

---

### Topic 2 — Privacy-preserving record linkage under Ghanaian naming conventions

**1. Proposed title**
*Linking Patients Without Revealing Them: Privacy-Preserving Record Linkage for Ghana's Fragmented Health Records, and the Problem of Day-Name Frequency Skew*

**2. Research problem / gap**
The same Ghanaian patient exists as unlinked records at a teaching hospital, a district hospital, an NHIS claim and a CHPS register. Linking them would transform care continuity and research, but sharing identifiers between institutions is a privacy hazard and, increasingly, a legal one. Privacy-preserving record linkage solves this internationally by matching on encrypted representations. But every published PPRL evaluation assumes Western name distributions — and **Ghanaian naming breaks that assumption in a specific, measurable way**. Akan day-names (Kwame, Kwabena, Kofi, Ama, Akosua and their variants) concentrate an enormous share of the population into a small number of given names, producing a frequency distribution far more skewed than PPRL frequency-attack models anticipate. Nobody has quantified what this does to either linkage quality or attack resistance.

**3. Research questions**
1. What linkage precision and recall can Bloom-filter PPRL achieve on Ghanaian-structured names under realistic data-entry error?
2. How does day-name frequency concentration degrade linkage quality relative to a Western-distribution baseline?
3. Does that same concentration make Ghanaian PPRL encodings *more* vulnerable to frequency-based cryptanalytic attack — and by how much?
4. Which encoding and hardening strategies (salting, record-level Bloom filters, balanced encoding, differentially private variants) best restore both quality and attack resistance?

**4. Variables**
- **Dependent:** linkage precision, recall, F1; attack success rate (fraction of encodings correctly re-identified by a frequency attack); computational cost.
- **Independent:** name-frequency distribution (Ghanaian vs Western baseline vs synthetic skew levels); error type and rate (typographic, phonetic, transliteration, name-order swap, missing date of birth, nickname substitution); Bloom filter length and hash count; encoding scheme; blocking strategy; dataset size.

**5. Model to build**
A **PPRL system plus an adversarial evaluation model**.
- *Encoding and matching:* Bloom-filter-based encoding with Dice-coefficient similarity, plus record-level (CLK) variants; blocking via LSH for scalability.
- *A Ghanaian name-and-error generator:* a probabilistic model that produces realistic Ghanaian identity records — day-name distributions, surname distributions across Akan/Ewe/Ga/Dagbani/Northern naming systems, transliteration variance, and an error-injection model parameterised from published data-quality studies. **This generator is itself a reusable research output.**
- *Adversarial model:* a frequency-alignment attack that attempts to map encodings back to plaintext names, giving an empirical re-identification risk rather than a theoretical claim.
- *Optimisation:* select encoding parameters on the linkage-quality vs attack-resistance Pareto frontier.

**6. Data required**
**None that is confidential.** Name-frequency distributions can be constructed from public sources (published Ghanaian onomastics literature, public electoral or census name statistics, public directory data). Everything else is generated.

**7. Ghanaian / external sources**
Ghana Statistical Service census outputs (population structure); published Ghanaian onomastics and naming-convention literature; NIA Ghana Card documentation (identifier structure and coverage); Data Protection Commission guidance; NHIA and GHS documentation on current patient-identification practice; international PPRL benchmark datasets for comparison.

**8. Unit of analysis**
The record pair (matched / non-matched); the encoding configuration for the optimisation.

**9. Methodology**
Build the Ghanaian identity generator and validate its distributions against published statistics → generate paired datasets with controlled error injection → implement Bloom-filter PPRL and variants → measure linkage quality across error and skew conditions → implement and run the frequency attack → compute the Pareto frontier → recommend a hardened configuration → benchmark against a Western-distribution control to isolate the Ghana-specific effect.

**10. Expected contribution**
*Academic:* a genuinely novel finding — that a specific, well-documented feature of Ghanaian naming has measurable consequences for a widely deployed privacy technology — plus a reusable synthetic Ghanaian identity generator. This is the kind of narrow, sharp, defensible contribution that examiners like and that generalises to other cultures with concentrated naming distributions. *Practical:* a validated configuration for any Ghanaian health-information-exchange effort, and a documented warning about the default parameters everyone would otherwise copy from the literature.

**11. Feasibility: 9/10** · *No ethics approval required* (no human subjects).
This is the **safest topic in the entire document**. Every input is synthetic or public, the scope is tightly bounded, the method is well documented, and the result is publishable regardless of which direction it comes out. If your priority is certainty of completion, this is the one.

**12. Challenges / limitations**
- Synthetic identity data, however carefully calibrated, is not a real patient register; external validity rests on how well you justify the generator, so validate its distributions transparently.
- The topic is narrower than the others here — deep rather than broad. Some supervisors want breadth; check yours.
- Requires comfort with cryptographic hashing concepts, though not with cryptography research itself.
- Ghana Card adoption may eventually make PPRL less necessary for *new* records — address this directly by framing the contribution around the large legacy record stock and around cross-institution scenarios where sharing the national ID is itself undesirable.

---

### Topic 3 — Synthetic Ghanaian health records: the privacy–utility frontier

**1. Proposed title**
*Can Synthetic Data Unlock Ghanaian Health Research? Generative Modelling of Ghanaian Health Records and the Privacy–Utility Trade-off*

**2. Research problem / gap**
Ghanaian health researchers and developers cannot get data, and institutions cannot release it without legal exposure. Synthetic data — statistically realistic records corresponding to no real person — is the internationally favoured escape from this deadlock, and it is being promoted for African health systems largely on faith. No one has tested whether generative models actually preserve the analytical relationships that matter in Ghanaian health data, or whether the resulting synthetic records leak membership information about the real people who trained them.

**3. Research questions**
1. How faithfully do generative models reproduce the marginal and joint distributions of Ghanaian health data?
2. Do models trained on synthetic data reach conclusions consistent with models trained on real data — i.e. is analytical utility preserved?
3. What is the empirical privacy risk, measured by membership-inference and attribute-inference attack success?
4. Where does the privacy–utility frontier sit, and is there a configuration that is simultaneously useful and safe enough to satisfy Act 843?

**4. Variables**
- **Dependent:** statistical fidelity (marginal and pairwise distances, correlation-matrix divergence); downstream utility (train-on-synthetic-test-on-real performance gap; agreement of estimated regression coefficients and their significance); privacy (membership-inference AUC, attribute-inference accuracy, nearest-neighbour distance ratio).
- **Independent:** generator architecture (CTGAN, TVAE, tabular diffusion, Bayesian network, classification-and-regression-tree synthesis); differential-privacy budget; training-set size; proportion of rare categories; number of variables.

**5. Model to build**
A **generative modelling pipeline with a three-axis evaluation framework** — fidelity, utility, privacy — producing an explicit frontier. Critically, the utility axis is not generic: you replicate a set of *real published Ghanaian health analyses* on the synthetic data and test whether the substantive conclusions survive. That is a much stronger utility test than distributional distance, and it is what makes the thesis about Ghanaian health research rather than about GAN benchmarking.

**6. Data required**
A real Ghanaian health dataset to train on. **This is available publicly**: Ghana DHS microdata is nationally representative, rich in health variables, and downloadable on registration. Ghana MICS is a second option. Using DHS rather than clinical records removes the gatekeeper entirely while keeping the data genuinely Ghanaian.

**7. Ghanaian / external sources**
Ghana DHS (DHS Program) — primary training corpus; Ghana MICS (UNICEF); GSS census data for validation of population structure; DHIMS2 aggregates for facility-level cross-checks; published Ghanaian health analyses (for the replication-based utility test); Data Protection Commission guidance (for the disclosure-risk standard).

**8. Unit of analysis**
The record (individual/household); the generator configuration for the frontier analysis.

**9. Methodology**
Obtain and prepare DHS microdata → select a replication set of published Ghanaian analyses → train multiple generator classes, with and without differential privacy → evaluate fidelity, utility (including analysis replication) and privacy attacks → construct the three-axis frontier → issue configuration guidance → optionally release a synthetic Ghanaian health dataset as an open research output.

**10. Expected contribution**
*Academic:* the first rigorous Ghanaian evaluation of synthetic health data, with a utility test grounded in real research questions rather than benchmark metrics. *Practical:* potentially an openly publishable synthetic Ghanaian dataset — a durable contribution other researchers actually use, which is rare for a master's thesis.

**11. Feasibility: 8/10** · *Ethics: DHS data use agreement required, but this is a form, not a committee.*
Data is public and Ghanaian. Methods are well-supported in Python (SDV, synthcity). Marked at 8 rather than 9 because survey microdata differs structurally from clinical records, and you must be careful not to over-generalise from one to the other.

**12. Challenges / limitations**
- **State this limitation prominently:** DHS is survey data, not clinical records. Conclusions transfer only partially to EHR-type data with its longitudinal structure, missingness patterns and rare events. Frame the scope honestly rather than letting a reviewer find it.
- Membership-inference attacks are sensitive to implementation; use established attack libraries and report the attack's own assumptions.
- Generative models handle rare categories poorly, and rare categories are often the clinically interesting ones — quantify this rather than glossing it.
- Some reviewers regard synthetic data as insufficiently protective regardless of evidence; engage that position rather than ignoring it.

---

### Topic 4 — Re-identification risk and optimal de-identification policy for Ghana's routine health data

**1. Proposed title**
*How Anonymous Is "Anonymised"? Quantifying Re-identification Risk in Ghanaian Routine Health Data and Optimising the De-identification Trade-off*

**2. Research problem / gap**
Ghanaian institutions release "anonymised" health data by removing names and identifiers — a standard that computer science abandoned two decades ago. Quasi-identifiers (district, age, sex, occupation, facility, visit date, condition) can re-identify individuals with high probability, and in Ghana's sparser districts the risk is *higher* than in dense settings because uniqueness is easier to achieve. No one has measured this for Ghana, and there is no evidence-based standard for what de-identification is adequate under Act 843.

**3. Research questions**
1. What proportion of records in a typical Ghanaian routine health dataset are unique or near-unique on standard quasi-identifier combinations?
2. How does re-identification risk vary with district population density, facility catchment size and condition rarity?
3. What generalisation and suppression strategy minimises information loss subject to a defined risk ceiling?
4. What does an evidence-based de-identification standard for Ghana look like in operational terms?

**4. Variables**
- **Dependent:** re-identification risk metrics (proportion unique, journalist/prosecutor/marketer risk, k-anonymity level achieved, l-diversity, δ-presence); information loss (discernibility metric, average equivalence-class size, downstream analytical distortion).
- **Independent:** quasi-identifier set; generalisation hierarchy depth (age bands, geographic aggregation level, date precision); suppression threshold; district population density; condition prevalence.

**5. Model to build**
- *Risk model:* a population-uniqueness estimator that accounts for sampling fraction (using census denominators to move from sample uniqueness to population uniqueness — the step most applied work skips and gets wrong).
- *Optimisation model:* generalisation-and-suppression optimisation (Mondrian or an ILP formulation) that maximises retained information subject to a risk constraint, producing a **risk–utility frontier** per data-release scenario.
- *Adversarial validation:* a simulated linkage attack against public auxiliary data (census, electoral, social media-style attributes) to show that the theoretical risk is realisable.

**6. Data required**
Ghana DHS or MICS microdata as the record base (public); GSS census outputs as population denominators (public); optionally a DHIMS2 aggregate extract for facility-level realism (Tier B, not essential).

**7. Ghanaian / external sources**
Ghana DHS/MICS; GSS 2021 Census; Data Protection Commission guidance and any published de-identification standard; GHS data-sharing policy documents; international comparators (HIPAA Safe Harbor, UK Anonymisation Network standards) for benchmarking.

**8. Unit of analysis**
The individual record; the release scenario for the optimisation.

**9. Methodology**
Define realistic Ghanaian release scenarios → construct quasi-identifier sets → estimate sample and population uniqueness with census denominators → run the simulated linkage attack → implement the generalisation optimisation → derive the risk–utility frontier → convert into a concrete de-identification standard proposal with worked examples.

**10. Expected contribution**
*Academic:* first quantified re-identification risk estimates for Ghanaian health data, with the sparse-district finding likely to be counterintuitive and citable. *Practical:* an operational de-identification standard the Data Protection Commission or GHS could adopt — a rare case where a thesis output maps onto a policy instrument.

**11. Feasibility: 8/10** · *Ethics: data-use agreement only.*
Public data throughout, mature tooling (ARX, sdcMicro, Python implementations). Held at 8 because the optimisation component needs care to rise above a routine k-anonymity application — make the constrained optimisation and the adversarial validation central, not decorative.

**12. Challenges / limitations**
- k-anonymity and its descendants have known weaknesses; acknowledge and consider a differential-privacy comparison so the thesis isn't defending an outdated paradigm.
- Population-uniqueness estimation depends on the sampling-fraction assumption; run sensitivity.
- Real adversaries have auxiliary data you cannot simulate; your risk estimates are lower bounds and should be labelled as such.
- Overlaps conceptually with Topics 1–3; do not attempt more than one of them.

---

### Topic 5 — Health information exchange network design under budget constraint

**1. Proposed title**
*Which Hospitals Should Ghana Connect First? A Network Design Model for Sequencing National Health Information Exchange Investment*

**2. Research problem / gap**
Ghana is incrementally connecting facilities into interoperable health information systems, and the sequencing is driven by administrative convenience rather than by where connection yields the most linked patient care. Because the value of an exchange rises non-linearly with connectivity — a connected pair is worth far more than two connected singletons — sequencing decisions have large and largely unexamined consequences. No model exists for optimising this.

**3. Research questions**
1. What does Ghana's inter-facility patient referral and movement network look like, and how concentrated is it?
2. Given a fixed budget, which facilities should be connected, and in what order, to maximise the share of patient encounters that become linkable?
3. How much does an optimised sequence outperform the status quo and simple heuristics (largest-first, capital-first, region-by-region)?
4. How robust is the optimal sequence to uncertainty in referral flows and connection costs?

**4. Variables**
- **Decision variables:** binary connect/don't-connect per facility per budget period.
- **Objective:** maximise linkable patient encounters (or referral pairs covered).
- **Parameters:** facility encounter volumes, inter-facility referral probabilities, connection cost (varying with existing connectivity and electrification), budget per period, maintenance cost.

**5. Model to build**
A **multi-period budget-constrained network design / maximal coverage optimisation** (MILP, solved with PuLP or OR-Tools), preceded by a **gravity-style referral flow model** estimated from facility characteristics and travel time — since actual referral matrices are not published, you *estimate* the network rather than observing it, which is itself a modelling contribution. Then stochastic/robust optimisation over flow and cost uncertainty, and comparison against heuristic sequences.

**6. Data required**
Facility registry with locations, tiers and volumes; population distribution; travel-time surface; connectivity and electrification proxies; connection cost estimates.

**7. Ghanaian / external sources**
GHS facility registry and healthsites.io/GRID3 (locations, tiers); GHS Holistic Assessment reports (facility volumes); WorldPop (catchment populations); Malaria Atlas Project friction surface (travel time); OpenCelliD/NCA (connectivity); VIIRS (electrification); MoH digital health strategy (cost benchmarks and current deployment status); DHIMS2 aggregates for referral validation (Tier B).

**8. Unit of analysis**
Health facility; facility pair (for the referral network).

**9. Methodology**
Build the facility network with attributes → estimate the referral gravity model → formulate and solve the multi-period MILP → compare against heuristics and status quo → robustness analysis under flow and cost uncertainty → produce a ranked connection schedule with a map.

**10. Expected contribution**
*Academic:* applies network design optimisation to health information infrastructure, a combination that is essentially absent for African health systems; the estimated referral network is a reusable artefact. *Practical:* a defensible, budget-aware connection sequence — directly usable by MoH and demonstrably better than the heuristics currently in use.

**11. Feasibility: 8/10** · *No ethics approval required.*
Data is overwhelmingly Tier A. The MILP is well within reach. Marked down slightly because the referral flow model is *estimated rather than observed*, and you must defend that estimate carefully — validate it against whatever aggregate referral statistics GHS publishes.

**12. Challenges / limitations**
- The referral network is inferred, not measured; this is the load-bearing assumption and needs explicit sensitivity analysis.
- Connection cost data is weak; parameterise from ranges and report frontier sensitivity.
- Real sequencing decisions are political; be honest that the model optimises a technical objective the decision-maker may not hold.

---

### Cluster B — AI and clinical decision support

---

### Topic 6 — A clinical early-warning score that uses only what Ghana can actually measure

**1. Proposed title**
*Decision Support Under Measurement Scarcity: Deriving a Feature-Constrained Clinical Early-Warning Score for Ghanaian District Hospitals*

**2. Research problem / gap**
Early-warning scores in global use (NEWS2, qSOFA, MEOWS) were derived where continuous vital-sign monitoring and same-day laboratory results are routine. In a Ghanaian district hospital, several required inputs are unavailable, intermittently available, or available only after the clinical decision has already been made. Practitioners then either use the score with missing inputs — silently degrading it — or abandon it. Nobody has asked the correct question: what is the *best possible* score constructible from the inputs Ghanaian district hospitals genuinely have?

**3. Research questions**
1. Which clinical inputs are actually and reliably available at Ghanaian district-hospital level, and at what latency?
2. How much discriminative performance is lost when established scores are computed with realistic Ghanaian input availability?
3. What is the best-performing score derivable under an explicit feature-availability budget, and how does it compare with the full-input benchmark?
4. Can that score be expressed in a form clinically usable without a computer — integer points, a card, a simple app?

**4. Variables**
- **Dependent:** deterioration/mortality outcome; model AUC, sensitivity at clinically relevant specificity, calibration; net benefit (decision-curve analysis).
- **Independent:** the constrained feature set (vital signs measurable without lab support, basic point-of-care tests, demographics, presenting complaint, simple clinical signs); availability and latency parameters per feature.

**5. Model to build**
**Constrained-optimisation model derivation.** Rather than fitting the best model on all features, you fit the best model *subject to a feature-availability constraint* — a formulation with real methodological interest. Use interpretable-by-construction methods that yield an integer point score (risk-calibrated supersparse linear integer models, or a shallow optimal decision tree), because a black box is unusable at the bedside. Benchmark against full-feature gradient boosting to quantify the cost of the constraint, and against the established scores computed under realistic availability. Decision-curve analysis to show clinical net benefit, not just AUC.

**6. Data required**
Clinical outcome data with vital signs. **This is the topic's weak point.** Ghanaian clinical records are Tier C. The workable path is: derive and validate on a public clinical database restricted to the Ghana-available feature set, then characterise Ghanaian availability from *published* facility-capability surveys (WHO SARA / Service Provision Assessment-type instruments, GHS reports) — making the thesis about the *method and the availability constraint* rather than about Ghanaian patients specifically. Be upfront that this is a transportability study.

**7. Ghanaian / external sources**
MIMIC-IV / eICU / public African clinical datasets (derivation); WHO Service Availability and Readiness Assessment and GHS facility-capability reports (the availability constraint — this is the Ghanaian core); Ghana Health Service standard treatment guidelines; KATH/Korle-Bu record extracts if a relationship materialises (bonus validation only).

**8. Unit of analysis**
Patient admission episode.

**9. Methodology**
Characterise Ghanaian district-hospital measurement capability from public assessments → define the feature-availability constraint → derive the constrained score on public data with proper train/validation/test separation → benchmark against unconstrained and established scores → decision-curve analysis → produce a paper-usable scoring instrument → simulate performance under realistic missingness and delayed-lab scenarios.

**10. Expected contribution**
*Academic:* reframes clinical prediction for low-resource settings as a constrained-optimisation problem rather than a data-scarcity complaint — a framing that generalises well beyond Ghana. *Practical:* a bedside-usable instrument designed for the measurement reality of a Ghanaian district hospital.

**11. Feasibility: 6/10** · *Ethics: not required if using public data only; required immediately if any Ghanaian records are used.*
The method is strong and the framing is original. The score is held at 6 because the clinical data is not Ghanaian, which weakens the central claim, and because any attempt to fix that by obtaining real records reintroduces exactly the gatekeeper risk you are trying to escape.

**12. Challenges / limitations**
- **The principal weakness:** deriving on non-Ghanaian patients and asserting Ghanaian relevance. You can defend this as a transportability study, but you cannot make it disappear.
- Case-mix in Ghanaian district hospitals differs substantially from ICU-heavy public databases (far more malaria, anaemia, obstetric emergency, sepsis-with-late-presentation); this limits transfer and must be discussed at length.
- Without prospective validation, the score is a proposal, not a validated instrument — say so.

---

### Topic 7 — Do imported diagnostic AI tools work in Ghana?

**1. Proposed title**
*Imported Intelligence: External Validation, Calibration Drift and Fairness Auditing of Diagnostic AI Deployed in Ghanaian Settings*

**2. Research problem / gap**
Ghanaian health institutions are being offered — and in places adopting — AI diagnostic tools trained and validated almost entirely on North American, European and East Asian populations. The published performance figures accompanying these tools are not the performance they deliver in Ghana, because disease prevalence, imaging equipment, patient demographics and comorbidity profiles all differ. The gap between advertised and delivered performance is unmeasured, and the procurement decisions are being made anyway.

**3. Research questions**
1. How does a publicly available, pretrained diagnostic model perform on African/Ghanaian-relevant data compared with its reported metrics?
2. Is the degradation primarily in discrimination, in calibration, or in both — and does prevalence shift alone explain it?
3. Do recalibration and lightweight domain adaptation recover performance, and at what data cost?
4. Does the model exhibit differential performance across demographic subgroups, and what does a procurement-grade fairness audit require?

**4. Variables**
- **Dependent:** AUC, sensitivity, specificity, positive predictive value at operating threshold; calibration slope and intercept; subgroup performance gaps; net benefit.
- **Independent:** source vs target domain; recalibration method (Platt, isotonic, prevalence correction); adaptation method (threshold shift, last-layer fine-tuning, full fine-tuning); adaptation sample size; subgroup (sex, age band, comorbidity, equipment/source).

**5. Model to build**
An **external validation and recalibration framework** — the validation study is itself the model-building exercise, plus you build recalibration and adaptation models on top. Include a **sample-size curve**: how many local labelled cases does Ghana need to recover acceptable calibration? That number is directly actionable for a procurement decision and is the most useful thing the thesis can produce.

**6. Data required**
A pretrained public diagnostic model plus African-origin or African-relevant labelled data for the target domain.

**7. Ghanaian / external sources**
Public pretrained models (chest X-ray classifiers, cervical cytology, retinopathy); publicly available African imaging and clinical datasets; WHO and Ghana Health Service disease prevalence data (for prevalence-shift correction — the Ghanaian anchor); Ghana FDA and MoH medical device/AI procurement guidance; Korle-Bu/KATH imaging archives if a relationship materialises (Tier C, bonus).

**8. Unit of analysis**
The individual case/image; the subgroup for fairness analysis.

**9. Methodology**
Select model and task with genuine Ghanaian relevance → assemble the target dataset → evaluate out of the box → decompose degradation into discrimination vs calibration vs prevalence effects → apply recalibration and adaptation with increasing sample sizes → build the sample-size curve → run the subgroup fairness audit → translate into a procurement checklist.

**10. Expected contribution**
*Academic:* turns the widely asserted claim that imported AI underperforms in Africa into measured quantities, with a decomposition of *why*. *Practical:* a procurement-grade evaluation protocol and a concrete local-data requirement figure for Ghana FDA/MoH.

**11. Feasibility: 6/10** · *Ethics: required if any Ghanaian patient data is used.*
Methodologically clean and highly relevant. Held at 6 purely on data: genuinely African labelled datasets for the tasks that matter to Ghana are limited, and without them the "Ghana" in the title is thin. Verify dataset availability in week one before committing — this is a genuine go/no-go.

**12. Challenges / limitations**
- Available African datasets may not be Ghanaian; be precise about what you actually validated on.
- Imaging work requires GPU compute and non-trivial engineering time.
- Pretrained model availability for the highest-relevance tasks is not guaranteed.
- Findings will be model-specific and may not generalise across vendors; frame the *protocol* as the durable contribution.

---

### Topic 8 — Climate-driven meningitis early warning for northern Ghana

**1. Proposed title**
*Forecasting the Harmattan Epidemic: A Climate-Driven Spatio-Temporal Early-Warning Model for Bacterial Meningitis in Northern Ghana*

**2. Research problem / gap**
Ghana's northernmost regions sit within the African meningitis belt and experience seasonal epidemics tied to the Harmattan — dry, dusty, low-humidity conditions that damage the nasopharyngeal mucosa and drive transmission. Reactive vaccination requires early detection, but epidemic-threshold rules currently in use are simple case-count triggers that ignore the climatic signal entirely, despite that signal being observable weeks in advance and freely available from satellite products. Meningitis-belt forecasting research has concentrated on Niger, Burkina Faso and northern Nigeria; Ghana is comparatively unmodelled.

**3. Research questions**
1. What is the spatio-temporal structure of meningitis incidence across northern Ghanaian districts?
2. Which climatic drivers — dust/aerosol load, humidity, temperature, wind — predict district-level case counts, and at what lead time?
3. Does a climate-driven model detect epidemic onset earlier than the current case-count threshold rule, and with what false-alarm cost?
4. What alert threshold optimises the trade-off between early detection and unnecessary response mobilisation?

**4. Variables**
- **Dependent:** weekly district-level suspected/confirmed meningitis cases; binary epidemic-threshold crossing.
- **Independent:** aerosol optical depth / dust concentration, relative humidity, maximum temperature, wind speed and direction (all lagged 1–8 weeks); population density; prior-season incidence; vaccination campaign coverage; spatial neighbour effects; harmonic seasonal terms.

**5. Model to build**
A **hierarchical Bayesian spatio-temporal model** (negative binomial with structured spatial and temporal random effects, fitted with INLA), which is the appropriate class for sparse count data with spatial correlation — plus a gradient-boosting benchmark. Then, critically, an **alert-threshold optimisation**: convert the probabilistic forecast into a binary alert and select the threshold that optimises detection timeliness against false-alarm cost, evaluated with ROC and with a timeliness metric (weeks of lead time gained versus the status quo rule).

**6. Data required**
Weekly district-level surveillance counts over multiple seasons; gridded climate and aerosol data; population denominators; vaccination campaign history.

**7. Ghanaian / external sources**
WHO AFRO IDSR weekly bulletins and Ghana Health Service surveillance reports (case counts — partly public, partly Tier B); MERRA-2 and CAMS aerosol reanalysis (dust — free); ERA5 (humidity, temperature, wind — free); CHIRPS (rainfall — free); WorldPop and GSS census (denominators); WHO/GAVI records of MenAfriVac and other campaign coverage; published Ghanaian meningitis epidemiology for validation.

**8. Unit of analysis**
District × week.

**9. Methodology**
Assemble the district-week surveillance panel → extract lagged climate and dust covariates → exploratory spatio-temporal analysis → fit the hierarchical Bayesian model with lag selection → benchmark against ML and against a seasonal-naive rule → rolling-origin forecast evaluation → alert-threshold optimisation with timeliness scoring → produce an operational early-warning specification.

**10. Expected contribution**
*Academic:* first Ghana-specific climate-driven meningitis forecasting model; the timeliness-versus-false-alarm optimisation is a framing that most outbreak-prediction papers omit in favour of raw accuracy. *Practical:* an early-warning specification GHS could operationalise, with an explicit and defensible alert rule.

**11. Feasibility: 7/10** · *Ethics: likely exempt (aggregate surveillance data), but confirm.*
All climate and dust data is free and excellent. The score is held at 7 by surveillance data: district-week counts over enough seasons may require a GHS request, and historical bulletin completeness is uncertain. Check bulletin availability in week one — if the series is short or patchy, the model class has to simplify.

**12. Challenges / limitations**
- Surveillance under-reporting is substantial and spatially variable; you are modelling *reported* cases, and must say so.
- Epidemics are rare events, so effective sample size is small even with many district-weeks — this limits model complexity and widens uncertainty.
- Vaccination campaigns structurally break the historical relationship; handle as an intervention rather than a covariate.
- If the surveillance series is too short, the whole topic degrades; this is its single point of failure.

---

### Cluster C — Health systems operations

---

### Topic 9 — Drone and road medical logistics network design

**1. Proposed title**
*Designing Ghana's Medical Delivery Network: A Location-Allocation Model for Drone and Road Resupply of Health Facilities Under Stockout Risk*

**2. Research problem / gap**
Ghana operates one of the world's largest medical drone-delivery networks, serving thousands of facilities from a small number of distribution centres. It is internationally celebrated and almost entirely unmodelled in the open literature: the distribution centres' placement, the drone-versus-road division of labour, and the inventory-pooling logic that makes centralised drone stock cheaper than distributed facility stock have never been subjected to independent optimisation. Whether the deployed network is close to optimal — and where the next distribution centre should go — are open, answerable and highly consequential questions.

**3. Research questions**
1. Given Ghana's facility distribution, demand profile and terrain, where should medical drone distribution centres be sited to maximise population and facility coverage within a clinically meaningful delivery window?
2. How does the optimised network compare with the deployed one, and what does any gap imply?
3. For which product–facility combinations is drone resupply economically superior to road resupply, once inventory-pooling savings and stockout costs are counted?
4. How robust is the optimal configuration to demand uncertainty, range assumptions and cost parameters?

**4. Variables**
- **Decision variables:** distribution centre locations (binary), facility-to-centre assignment, drone fleet size per centre, product allocation to drone versus road channel.
- **Objective:** maximise covered demand within the delivery-time window, or minimise total cost including stockout penalty.
- **Parameters:** facility locations and tiers, catchment populations, product demand rates and variability, drone effective range, road travel times, per-delivery costs, holding costs, stockout penalty, capacity limits.

**5. Model to build**
A **maximal covering location problem with capacity constraints** (MILP, OR-Tools/PuLP/Gurobi academic licence), extended in two directions that lift it well above a textbook exercise:
- an **inventory-pooling layer** quantifying the safety-stock savings from centralising stock at a hub versus distributing it — this is the actual economic argument for drone delivery and is rarely modelled;
- a **stochastic/robust variant** handling demand uncertainty, plus **simulation** of the resulting network under demand shocks (an obstetric-haemorrhage surge, a road washout in the rainy season) to test resilience rather than just average performance.

Compare optimised against deployed against road-only baselines.

**6. Data required**
Facility locations and tiers; catchment populations; product demand estimates; travel-time surface; terrain; cost parameters; drone performance envelope.

**7. Ghanaian / external sources**
GHS facility registry, healthsites.io and GRID3 (facilities); WorldPop (catchments); Malaria Atlas Project friction surface and OSM road network (road travel times); SRTM (terrain); GHS Holistic Assessment and DHIMS2 aggregates (service volumes → demand proxies); published Ghanaian stockout-rate studies and Ministry of Health supply chain reports (demand variability and stockout costs); publicly reported distribution-centre locations and service claims (for the deployed-network comparison); National Blood Service reports (blood is a high-value drone payload).

**8. Unit of analysis**
Health facility (demand node); candidate distribution centre site; product category.

**9. Methodology**
Build the georeferenced facility-demand layer → construct drone (Euclidean, range-limited) and road (friction-surface) travel-time matrices → estimate product demand and variability from service volumes and published rates → formulate and solve the MILP → compare optimised vs deployed vs road-only → add the inventory-pooling economics → run stochastic scenarios and shock simulations → produce a siting recommendation with maps and a sensitivity analysis.

**10. Expected contribution**
*Academic:* the first independent optimisation study of Ghana's medical drone network, combining location-allocation with inventory pooling — a combination that is thin even in the general OR literature. Internationally visible: Ghana is *the* reference case, so the work travels. *Practical:* an evidence-based answer to where the next distribution centre should go and which products justify air resupply.

**11. Feasibility: 8/10** · *No ethics approval required.*
The geospatial and facility data is entirely Tier A, the optimisation is standard-toolkit, and the Ghanaian relevance is unarguable. Held at 8 only because demand and cost parameters are estimated rather than observed — which sensitivity analysis handles honestly.

**12. Challenges / limitations**
- Operator cost and operational data is Tier C and will almost certainly not be shared; parameterise from published figures and be transparent that the cost analysis is scenario-based.
- Comparing your optimum to the deployed network invites the criticism that the operator optimised against objectives you don't know (regulatory constraints, airspace, partnership terms). Address this directly and frame the comparison as "optimal under stated objectives," not as a verdict.
- Drone range and payload assumptions materially drive results — run wide sensitivity.
- Demand estimated from service volumes rather than measured consumption introduces error; validate against published stockout rates.

---

### Topic 10 — National blood supply: forecasting plus perishable inventory optimisation

**1. Proposed title**
*Matching a Perishable Supply to an Unpredictable Demand: Donor Forecasting and Inventory Optimisation for Ghana's National Blood Supply*

**2. Research problem / gap**
Ghana's blood supply is chronically short of need, and the shortage is not purely a donor-recruitment problem — it is an operations problem. Blood is perishable on a short clock, demand is driven by shock events (road traffic trauma, obstetric haemorrhage, paediatric severe malarial anaemia) with strong seasonality, supply is driven by donor behaviour with entirely different seasonality (academic terms, religious calendar, campaign timing), and the two are matched by a compatibility structure that permits substitution across blood groups in one direction only. This is a textbook-rich operations research problem and it is essentially unmodelled for Ghana.

**3. Research questions**
1. What are the seasonal and event-driven structures of blood demand and donation in Ghana, and how far out of phase are they?
2. How accurately can each be forecast, and at what horizon?
3. What collection, allocation and substitution policy minimises expected shortage and expiry jointly?
4. How much improvement over current practice is achievable without any increase in total donations — i.e. how much of the shortage is logistical rather than volumetric?

**4. Variables**
- **Dependent:** units collected, units transfused, units expired, shortage events, days of cover by blood group.
- **Independent / drivers:** month and week of year, academic calendar, religious calendar, campaign timing, road-traffic-accident seasonality, malaria transmission season, maternal delivery volumes, holiday effects.
- **Decision variables:** collection targets by group and period, allocation across facilities, substitution policy, reorder points.

**5. Model to build**
Two coupled components.
- *Forecasting:* separate demand and supply models (SARIMAX with calendar and event regressors, plus an ML benchmark), forecast by blood group.
- *Optimisation:* a **stochastic perishable inventory model** — the classic and genuinely hard version of the newsvendor problem, with age-dependent expiry, group-compatibility substitution (a directed compatibility graph), and multi-echelon structure (national centre → regional → facility). Solved by stochastic programming or by simulation-optimisation, and evaluated in a **discrete-event simulation** against current practice under shock scenarios.

**6. Data required**
Time series of collections and transfusions by blood group and location; expiry counts; facility demand distribution.

**7. Ghanaian / external sources**
National Blood Service Ghana annual reports and published statistics (aggregate — the accessible layer); WHO Global Database on Blood Safety (Ghana country data); published Ghanaian transfusion-practice and blood-shortage studies (demand drivers and group distribution); GHS facility volumes and DHIMS2 aggregates (demand proxies); GSS census (blood group distribution priors from published Ghanaian haematology literature); National Blood Service transaction-level data (Tier C — desirable, not required).

**8. Unit of analysis**
Blood group × period × facility/region.

**9. Methodology**
Assemble aggregate supply and demand series from published sources → build a calibrated simulation of the blood system where transaction data is unavailable → fit and validate forecasting models → formulate the perishable inventory optimisation with substitution → solve → evaluate in discrete-event simulation against current policy → stress-test under shock scenarios → quantify the shortage reduction achievable at constant donation volume.

**10. Expected contribution**
*Academic:* perishable inventory optimisation with group substitution applied to a West African blood system — a real gap. The "how much shortage is logistical rather than volumetric" question is the sharp, quotable finding. *Practical:* collection targets and allocation policy that reduce both shortage and wastage without requiring more donors — the most budget-friendly kind of recommendation.

**11. Feasibility: 7/10** · *Ethics: aggregate data, likely exempt; confirm if any facility-level records are used.*
The model is genuinely sophisticated and the problem is real and urgent. Held at 7 because transaction-level data is Tier C, so much of the work rests on a calibrated simulation rather than observed operations. That is defensible — simulation studies are a respected format — but you must be scrupulous about labelling what is measured and what is assumed.

**12. Challenges / limitations**
- **Central limitation:** without transaction data, the baseline against which you claim improvement is itself modelled. Be explicit and validate the simulation against every published aggregate you can find.
- Ghanaian blood-group distribution and transfusion practice parameters must come from published literature; verify each.
- Perishable inventory optimisation with substitution is computationally demanding; you may need heuristics, which should be benchmarked against exact solutions on reduced instances.
- Informal replacement-donor practice (families sourcing donors) is a large, poorly documented part of how Ghana's system actually works and is hard to model — acknowledge it.

---

### Topic 11 — Geographic accessibility and optimal siting of emergency health services

**1. Proposed title**
*Who Can Reach Care in Time? Modelling Geographic Accessibility and Optimising Emergency Service Placement Across Ghana*

**2. Research problem / gap**
Ghana has invested heavily in emergency response capacity and in CHPS-level primary care, but placement decisions are made administratively — by constituency, by district, by political unit — rather than by where placement saves the most time for the most people. Accessibility mapping exists for Ghana in descriptive form; what is missing is the optimisation step that converts a map of the problem into a ranked list of where to act.

**3. Research questions**
1. What proportion of Ghana's population lives within clinically meaningful travel time of emergency obstetric care, and how does that vary by region, season and terrain?
2. Where are the largest accessibility deficits once population is weighted by need rather than by area?
3. Given a fixed number of new facilities or ambulance stations, where should they go to maximise the population brought within the time window?
4. How much better is the optimised placement than the current administrative allocation?

**4. Variables**
- **Dependent:** population within travel-time thresholds (30/60/120 minutes); mean travel time to nearest capable facility; coverage gain per unit deployed.
- **Independent:** road network and surface type, terrain slope, land cover, seasonal road passability, facility locations and capability tier, population distribution, need weighting (birth rate, disease burden).
- **Decision variables:** siting of new facilities/stations.

**5. Model to build**
**Cost-distance accessibility modelling** (AccessMod-style friction surface, or a custom implementation in Python with `scikit-image`/`graph-tool`) producing travel-time surfaces under dry- and wet-season scenarios, followed by a **maximal covering location problem** (MILP) that selects new sites to maximise newly covered population subject to a budget. Extended with an equity-weighted objective, so you can show explicitly how the optimal siting changes when the objective shifts from raw coverage to equitable coverage — that comparison is the intellectual core and prevents the thesis from being a routine GIS exercise.

**6. Data required**
Road network, terrain, land cover, facility registry with capability tiers, gridded population, need indicators.

**7. Ghanaian / external sources**
OpenStreetMap (roads); SRTM/Copernicus DEM (terrain); Copernicus/ESA land cover; healthsites.io, GRID3 and the GHS facility registry (facilities and tiers); WorldPop (population); Ghana DHS (need indicators — birth rates, care-seeking); GSS 2021 Census (district demographics); National Ambulance Service deployment records (Tier B, for the status-quo comparison); CHIRPS (seasonal passability proxy).

**8. Unit of analysis**
Population grid cell (100m); candidate site; district for reporting.

**9. Methodology**
Build the friction surface with seasonal variants → compute travel-time surfaces from existing facilities → quantify coverage deficits with need weighting → define candidate sites → formulate and solve the MCLP under coverage and equity objectives → compare optimised against status quo and against administrative allocation rules → map and rank.

**10. Expected contribution**
*Academic:* the coverage-versus-equity objective comparison is the novel element and is under-explored in African facility-siting work; the seasonal accessibility variant (wet-season road impassability) is a second real contribution that most static accessibility studies ignore. *Practical:* a ranked, mapped siting recommendation directly usable by MoH and GHS.

**11. Feasibility: 9/10** · *No ethics approval required.*
Every input is free, public and well-documented, and the tooling is mature. This is the topic with the **lowest execution risk** in the operations cluster.

**12. Challenges / limitations**
- Accessibility modelling for Africa is a reasonably crowded field — **originality must come from the equity objective and the seasonal variant, not from the accessibility map itself.** If you cut those, the topic becomes ordinary.
- OSM road completeness varies across Ghana, particularly for rural tracks; assess and report coverage.
- Modelled travel times are not observed travel times; validate against any published field measurements you can find.
- Ignores facility quality and capacity — a facility within 30 minutes that cannot perform a caesarean is not access. Address by using capability tiers rather than raw facility counts.

---

### Topic 12 — The health workforce pipeline: training, employment and emigration

*(Dual-listed: this is the strongest **health education** topic here, and the only one that avoids institutional gatekeepers.)*

**1. Proposed title**
*From Classroom to Departure Lounge: A Stock-and-Flow Simulation of Ghana's Health Workforce Pipeline Under Training Expansion and Emigration Pressure*

**2. Research problem / gap**
Ghana presents a paradox that is discussed constantly and modelled almost never: it simultaneously trains large numbers of health professionals, has qualified graduates waiting for posting, and loses experienced staff to emigration. These three facts are usually treated as separate problems with separate policy responses — expand training, clear the employment backlog, restrict emigration — when they are stages of one system whose behaviour is determined by feedback and delay. Policies aimed at one stage predictably fail when the system is treated as three. Crucially, **the emigration flow is directly observable in public foreign registers**, which means this can be studied without Ghanaian institutional cooperation.

**3. Research questions**
1. What are the transition rates between pipeline stages — admission, graduation, licensure, employment, continued service, emigration, exit — for Ghana's major health cadres?
2. What is the observable trajectory of Ghanaian-trained health professionals entering foreign registers, and how does it relate to domestic training output and employment absorption?
3. Under plausible scenarios (training expansion, employment-quota change, emigration-policy shift, salary adjustment), what workforce density does Ghana reach, and when?
4. Which single intervention point yields the largest sustained workforce gain per unit of cost?

**4. Variables**
- **Stocks:** students in training by year and cadre; unlicensed graduates; licensed-unemployed; employed by region and tier; emigrated; exited.
- **Flows:** admission, progression, graduation, licensure pass, employment absorption, emigration, return migration, retirement, attrition.
- **Drivers:** training institution capacity, government employment quota and fiscal space, domestic-versus-destination wage ratio, destination-country recruitment policy, licensure requirements, regional posting incentives.

**5. Model to build**
A **stock-and-flow system dynamics model** of the full pipeline (Vensim, Stella, or `pysd`/`BPTK-Py` in Python), with an embedded **Markov cohort transition model** for individual-level progression, calibrated to observed stocks and flows. Then:
- **Scenario simulation** across policy levers;
- **Sensitivity and policy-lever analysis** to identify the highest-leverage intervention point;
- optionally a **discrete-choice or survival component** modelling emigration timing from public register entry data (how many years post-qualification do departures cluster?), which converts an assumed rate into an estimated one.

The feedback structure is what makes this a *model* rather than a projection spreadsheet: employment backlog raises emigration, emigration relieves backlog, relief reduces political pressure to expand posts, and the cycle repeats.

**6. Data required**
Time series of training admissions and graduations by cadre; licensure volumes; public-sector employment; foreign register entries by country of training; workforce stocks and densities.

**7. Ghanaian / external sources**
**UK Nursing and Midwifery Council and General Medical Council public registers — registrant counts by country of training** (the key enabler: Ghanaian emigration measured without asking Ghana); comparable public data from other destination regulators where available; **WHO National Health Workforce Accounts** (Ghana stocks and densities, public); WHO health workforce migration reporting and the WHO health workforce support and safeguards list; Ministry of Health Health Sector Annual Performance Review reports and Holistic Assessment reports (training output and employment — public PDFs); Nursing & Midwifery Council and Medical & Dental Council published register summaries (Tier B); GSS census (population denominators for density).

**8. Unit of analysis**
Cadre × year for the system model; the individual professional cohort for the Markov component.

**9. Methodology**
Map the pipeline structure from policy documents and literature → assemble stock and flow series from public sources → estimate transition rates, using foreign registers to pin the emigration flow → build and calibrate the system dynamics model → validate against historical workforce density → run policy scenarios → sensitivity and leverage analysis → cost-effectiveness comparison of intervention points.

**10. Expected contribution**
*Academic:* the first integrated dynamic model of Ghana's health workforce pipeline treating training, employment and emigration as one feedback system; methodologically distinctive, since system dynamics is rarely used in African health workforce research. *Practical:* a scenario tool for the Ministry of Health and the training councils, and a defensible answer to whether Ghana's problem is production, absorption or retention — a question currently argued on assertion.

**11. Feasibility: 8/10** · *No ethics approval required* (aggregate published data only).
The emigration data being publicly observable in foreign registers is what rescues this topic from the gatekeeper problem that kills most health-education research in Ghana. Held at 8 because domestic transition rates will need estimation and assumption where published series are incomplete.

**12. Challenges / limitations**
- Foreign registers capture only registered emigration to countries with public registers; migration to destinations without public register data is invisible and will bias flows downward. State this and bound it.
- Domestic series (admissions, employment absorption) are incomplete in public reports; calibration will rest partly on assumption, so sensitivity analysis is not optional.
- System dynamics models are only as good as their structure; validate the structure against literature and, ideally, expert review, and report structural sensitivity as well as parameter sensitivity.
- Scenario projections are conditional, not predictive — police your own language carefully.

---

### Cluster D — Health education and digital health equity

---

### Topic 13 — Digital health readiness: where does Ghana's e-health strategy meet reality?

**1. Proposed title**
*The Infrastructure Beneath the Strategy: Modelling Digital Health Readiness and Investment Priority Across Ghana's Health Facilities*

**2. Research problem / gap**
Ghana's digital health ambitions — electronic records, telemedicine, e-claims, drone dispatch, national health information exchange — all presuppose three things at the facility: reliable electricity, adequate connectivity, and staff capability. Those preconditions are unevenly distributed and, critically, are *correlated with each other and with deprivation*, meaning digital health investment risks widening the very inequities it is meant to close. No one has measured facility-level digital readiness across Ghana or modelled where investment would yield the greatest health-system return.

**3. Research questions**
1. What is the spatial distribution of digital health readiness across Ghana's health facilities, measured across electricity, connectivity and capability?
2. How strongly is readiness correlated with deprivation, remoteness and existing health-service performance?
3. Would a naive digital rollout widen or narrow inequity in effective health service access?
4. Given a fixed budget, where should enabling infrastructure be placed to maximise health-system benefit, and how does that differ under efficiency versus equity objectives?

**4. Variables**
- **Dependent:** a composite facility digital readiness score; service performance indicators.
- **Components/independent:** grid electrification and reliability (night-lights derived), mobile/broadband coverage and technology generation, facility tier and staffing, distance to urban centre, district deprivation, catchment population, existing system deployment status.
- **Decision variables (optimisation stage):** where to place connectivity or power investment.

**5. Model to build**
Three layers, each doing real work:
- *Measurement:* a **latent-variable readiness model** — item response theory or confirmatory factor analysis rather than an arbitrary weighted index — so the composite is estimated rather than asserted. This is the methodological differentiator; weighted-index papers are common and weak, latent-variable measurement models are defensible.
- *Explanation:* **spatial regression** (spatial lag/error, or geographically weighted regression) of readiness on deprivation and remoteness, with explicit testing for spatial autocorrelation.
- *Prescription:* a **budget-constrained investment allocation optimisation** comparing efficiency-maximising against equity-maximising objectives, producing two ranked investment lists and quantifying the trade-off between them.

**6. Data required**
Facility locations and attributes; electrification proxy; connectivity coverage; deprivation indicators; service performance data.

**7. Ghanaian / external sources**
GHS facility registry, healthsites.io, GRID3 (facilities); **VIIRS night-time lights** (electrification and reliability — free, monthly, and the single most under-used dataset for this question); OpenCelliD, ITU, NCA and GSMA coverage data (connectivity); GSS 2021 Census (district deprivation, household electricity and phone access); Ghana DHS (service utilisation); GHS Holistic Assessment reports (facility performance); MoH digital health strategy and LHIMS deployment status documents (current rollout); WHO SARA-type assessments where available.

**8. Unit of analysis**
Health facility; district for aggregate reporting.

**9. Methodology**
Georeference the facility inventory → extract electrification, connectivity and deprivation covariates at facility locations → estimate the latent readiness model and validate it → map readiness and test spatial autocorrelation → estimate the readiness–deprivation relationship → simulate a naive rollout and measure its distributional consequence → formulate and solve the investment allocation under both objectives → produce ranked investment maps.

**10. Expected contribution**
*Academic:* a measured, latent-variable account of digital health readiness for Ghana, plus quantified evidence on whether digital health investment is inequity-widening by default — a claim frequently made and rarely tested. *Practical:* two ranked investment lists (efficiency and equity) with an explicit statement of what choosing one costs in terms of the other, which is precisely the form a policy trade-off should take.

**11. Feasibility: 8/10** · *No ethics approval required.*
Data is entirely Tier A and unusually good — night-lights and coverage data give you real facility-level infrastructure measurement without asking anyone. Held at 8 because the "capability" dimension of readiness (staff skills) has no good public data source and will have to be proxied, which weakens one of the three pillars of the composite.

**12. Challenges / limitations**
- Staff digital capability is the weakest measured dimension; either proxy it transparently or restrict the composite to infrastructure and rename accordingly.
- Night-lights are a proxy for electrification, not a measurement of facility power reliability; validate against any published facility power surveys.
- Coverage maps are operator-reported and optimistic; acknowledge and, where possible, cross-check against crowdsourced measurements.
- Composite indices attract methodological criticism; the latent-variable approach and sensitivity to weighting choices are your defence — do not skip them.

---

### Topic 14 — Health misinformation in Ghanaian digital discourse

**1. Proposed title**
*Detecting Health Misinformation in Ghanaian Digital Discourse: A Code-Switched Twi–Pidgin–English Classification Model*

**2. Research problem / gap**
Health misinformation in Ghana does not circulate in standard English. It circulates in Ghanaian English, in Pidgin, in Twi, and — most commonly — in fluid code-switching between them within a single message, often with English-script Twi orthography that no standard tokeniser handles. Every deployed health-misinformation detection system is trained on monolingual standard-variety text and fails on this material. The consequence is that the health information environment of a country of over thirty million people is effectively unmonitorable with existing tools.

**3. Research questions**
1. What linguistic characteristics distinguish Ghanaian online health discourse from the standard-variety text on which detection models are trained?
2. How far do existing multilingual and English misinformation classifiers degrade on Ghanaian code-switched health content?
3. Can a classifier adapted with a modest annotated Ghanaian corpus recover acceptable performance, and how much annotated data is needed?
4. What are the dominant themes and claim structures of health misinformation circulating in Ghanaian digital spaces?

**4. Variables**
- **Dependent:** classification performance (F1, precision, recall by class); performance stratified by language mix and code-switching density.
- **Independent:** model class (multilingual transformer, African-language-adapted model, classical baselines); adaptation strategy (zero-shot, few-shot, fine-tuning, continued pretraining); training corpus size; code-switching density; language composition; text length.

**5. Model to build**
A **code-switched text classification pipeline** with: a language-identification stage capable of handling intra-sentence switching; adapted tokenisation for English-script Twi; fine-tuned transformer classifiers benchmarked against multilingual baselines; and a **data-efficiency curve** establishing how much annotated Ghanaian data is required for acceptable performance — the practically decisive number. Plus topic modelling of the corpus to characterise what Ghanaian health misinformation actually claims, which is a descriptive contribution of independent value.

**6. Data required**
A corpus of Ghanaian online health discourse, annotated for veracity.

**7. Ghanaian / external sources**
Publicly accessible Ghanaian online content (platform terms and API access must be checked *before* committing — access conditions have tightened considerably and this is a live risk); Ghanaian fact-checking organisations' published verdicts (e.g. Dubawa, GhanaFact) as a labelled seed corpus — **this is the key enabler, since it gives you ground-truth labels without doing all the annotation yourself**; Ghana Health Service and Ministry of Health official communications (authoritative counter-corpus); existing African NLP resources (Masakhane and related community datasets); WHO infodemic management materials.

**8. Unit of analysis**
The individual post/message; the claim.

**9. Methodology**
Assemble the corpus from accessible sources and fact-checker archives → develop annotation guidelines and annotate a subset with inter-annotator agreement reported → characterise code-switching and orthographic variation → benchmark existing classifiers zero-shot → fine-tune and adapt → build the data-efficiency curve → topic-model the corpus → error analysis focused on where code-switching breaks the models.

**10. Expected contribution**
*Academic:* a genuinely novel resource and result — Ghanaian code-switched health NLP is close to unexplored, and the annotated corpus would be a lasting community contribution. *Practical:* a monitoring capability for GHS/MoH health communication, and a quantified statement of what it costs to build one.

**11. Feasibility: 6/10** · *Ethics: required — human-subjects/social-media research review, and platform terms compliance.*
The originality is the highest in this document. The feasibility is not, for three concrete reasons: platform data access is expensive and restricted; annotation is slow, and doing it credibly requires more than one annotator; and ethics clearance for social media research adds weeks. **Verify data access in week one** — if the corpus cannot be assembled, nothing else in the topic survives.

**12. Challenges / limitations**
- **Data access is the make-or-break issue** and is outside your control. Mitigate by anchoring on fact-checker archives, which are publicly published, rather than on platform scraping.
- Annotation quality requires multiple annotators and reported agreement statistics; budget for this properly or the labels won't be defensible.
- Twi orthographic variation in informal writing is extreme, with no standard; this is both the interesting problem and a serious practical obstacle.
- Misinformation labelling involves contested judgement; grounding labels in published fact-checker verdicts rather than your own assessment is both methodologically and ethically safer.

---

## Part 3 — Ranked top 10

| Rank | Topic | Model class | Data risk | Ethics needed | Originality | Feasibility |
|---|---|---|---|---|---|---|
| **1** | **T1 — Federated learning for health data sovereignty** | Federated learning + DP + factorial experiment + response surface | **None** | **No** | **Very high** | **9/10** |
| **2** | **T9 — Drone/road medical logistics network design** | MILP location-allocation + inventory pooling + stochastic simulation | **Very low** | **No** | **Very high** | **8/10** |
| **3** | **T12 — Health workforce pipeline simulation** | System dynamics + Markov cohort + scenario/leverage analysis | **Low** | **No** | **High** | **8/10** |
| 4 | T2 — Privacy-preserving record linkage (Ghanaian names) | Bloom-filter PPRL + adversarial attack + Pareto optimisation | **None** | No | **Very high** | **9/10** |
| 5 | T13 — Digital health readiness & investment priority | Latent-variable measurement + spatial regression + allocation optimisation | Very low | No | High | 8/10 |
| 6 | T3 — Synthetic Ghanaian health records | Generative models + three-axis privacy–utility–fidelity frontier | Low | Agreement only | High | 8/10 |
| 7 | T4 — Re-identification risk & de-identification policy | Risk estimation + constrained generalisation optimisation | Low | Agreement only | Medium-high | 8/10 |
| 8 | T11 — Accessibility & emergency service siting | Cost-distance accessibility + MCLP under equity vs efficiency | Very low | No | Medium | 9/10 |
| 9 | T5 — Health information exchange network design | Gravity referral estimation + multi-period MILP | Low | No | High | 8/10 |
| 10 | T8 — Meningitis climate early warning | Hierarchical Bayesian spatio-temporal + alert threshold optimisation | Moderate | Likely exempt | Medium-high | 7/10 |

*Below the line:* **T10 (blood supply, 7/10)** — outstanding model, but the operational baseline has to be simulated rather than observed. **T6 (constrained early-warning score, 6/10)** and **T7 (imported diagnostic AI, 6/10)** — both excellent framings undermined by the absence of Ghanaian clinical data. **T14 (health misinformation NLP, 6/10)** — the most original topic here and the one most likely to fail on data access; choose it only if you confirm corpus access in week one and accept the annotation burden.

**Note on rank 4:** Topic 2 has the highest raw feasibility of anything in this document and near-zero failure risk. It sits at 4 rather than 1 only because its scope is narrower than the topics above it. **If your priority is certainty of finishing rather than ambition of scope, choose Topic 2 over any of the top three.**

---

## Part 4 — The top three, in detail

### 🥇 First — Topic 1: Federated learning for Ghanaian health data sovereignty

**What you build, concretely.**

*The artefact.* A working federated learning system: one aggregation server, K simulated facility clients, each holding a private data partition it never transmits. Client-side DP-SGD with tunable ε. Secure aggregation so the server observes only the sum of updates, never an individual facility's contribution. Built with Flower in Python.

*The Ghanaian calibration — this is what makes it a Ghana thesis rather than a generic ML thesis.* The K partitions are not random. They are constructed to mirror Ghana's actual facility landscape drawn from public GHS reports: a small number of large teaching hospitals holding a disproportionate share of records, many mid-sized district hospitals, and a long tail of small facilities — with case-mix differing systematically by tier. Non-IID severity is controlled by a Dirichlet parameter so you can vary it and measure the effect. Bandwidth and dropout parameters come from ITU/GSMA connectivity data and VIIRS-derived power reliability.

*The experiment.* A factorial sweep over federation size, non-IID severity, aggregation algorithm, privacy budget ε, bandwidth cap and dropout rate. Two baselines bound the result: centralised training above (what you'd get if you could pool, ignoring the law), and facility-local-only training below (what each facility can do alone today).

*The models you fit on the results.* A response-surface model relating accuracy to the experimental factors, so the findings generalise beyond the specific grid you ran. And an empirical privacy evaluation — membership-inference attacks run against the trained models — so that ε becomes a measured risk rather than a theoretical parameter.

**Inputs:** a public clinical database partitioned by a Ghana-calibrated scheme; Ghanaian facility characteristics from public GHS reports; connectivity and power parameters from ITU/GSMA and VIIRS; Act 843 and Act 1038 requirements for the compliance mapping.

**Outputs:** (a) the privacy–utility–bandwidth frontier — how much accuracy Ghana can obtain at a given privacy budget and bandwidth cost without a single record leaving a facility; (b) empirical attack-success rates validating the privacy claim; (c) a threshold finding on where federation stops being worth it (facility count too low, heterogeneity too high, bandwidth too thin); (d) a reference architecture mapped to Ghanaian law.

**What the thesis demonstrates.** That you can build a distributed system, not just fit a model; that you can design and analyse a factorial experiment rather than report a single accuracy number; that you understand privacy as something to be attacked and measured rather than asserted; and that you can connect a technical architecture to a legal instrument. It answers your own question — *can we isolate health records to in-country, restricted access?* — with numbers instead of an opinion. Very few master's theses do all four of those things.

---

### 🥈 Second — Topic 9: Drone and road medical logistics network design

**What you build, concretely.**

*The demand layer.* Every health facility in Ghana as a georeferenced demand node, with catchment population from WorldPop, service volume from GHS reports, and estimated consumption rates for a set of critical products — blood, oxytocin and other obstetric commodities, antivenom, vaccines — with demand variability parameterised from published stockout studies.

*The travel-time layers.* Two matrices. Drone: range-limited, terrain-aware, effectively Euclidean. Road: computed over a friction surface built from OSM roads, terrain and land cover, with wet-season and dry-season variants — because a road that exists in the dry season and doesn't in the wet is the entire reason air resupply has value.

*The optimisation.* A capacitated maximal covering location problem selecting distribution centre sites and facility assignments to maximise demand covered within a clinically meaningful window. Layered with an inventory-pooling component that quantifies the safety-stock savings from holding stock centrally rather than at every facility — this is the actual economic case for hub-based air delivery and it is rarely modelled. Then a stochastic variant over demand uncertainty, and a discrete-event simulation stress-testing the network under shock scenarios: an obstetric haemorrhage surge, a rainy-season road failure, a distribution centre outage.

**Inputs:** facility registry, gridded population, service volumes, product demand rates, OSM road network, terrain, land cover, drone performance envelope, cost parameters.

**Outputs:** an optimal distribution-centre configuration with maps; a quantified comparison against the deployed network and against road-only resupply; a product-by-product determination of when air resupply beats road once pooling and stockout costs are counted; a resilience assessment under shocks; and a recommendation for where the next distribution centre should go.

**What the thesis demonstrates.** Command of operations research applied to a real, functioning, internationally significant Ghanaian system. Geospatial data engineering, MILP formulation and solution, stochastic modelling, and discrete-event simulation — a genuinely broad technical demonstration. And the subject matter travels: Ghana is the world's reference case for medical drone delivery, so an independent optimisation study of it has an audience well beyond your examiners.

---

### 🥉 Third — Topic 12: The health workforce pipeline

**What you build, concretely.**

*The structure.* A stock-and-flow model where the stocks are people at each pipeline stage — in training, graduated-unlicensed, licensed-unemployed, employed by region and tier, emigrated, exited — and the flows are the transitions between them. Feedback loops are the point: an employment backlog raises emigration; emigration relieves the backlog; relief reduces the political pressure to create posts; the backlog rebuilds. Linear projection cannot represent this, which is precisely why the existing policy debate keeps producing interventions that don't hold.

*The estimation.* Transition rates estimated from published Ghanaian sources where available. The emigration flow — normally the hardest parameter and the one usually assumed — is instead *estimated from public foreign registers*, which report registrants by country of training. Optionally, a survival model on register entry timing tells you how many years after qualification departures cluster, converting an assumed constant rate into an estimated hazard.

*The application.* Scenario simulation across the real policy levers: expand training intake, expand the employment quota, adjust regional posting incentives, respond to a destination-country recruitment policy change. Then sensitivity and leverage analysis identifying which single intervention point produces the largest sustained gain per cedi.

**Inputs:** training admissions and graduation series, licensure volumes, public-sector employment data (all from public MoH and council reports); UK NMC/GMC and other public foreign register data; WHO National Health Workforce Accounts; census denominators.

**Outputs:** estimated stage-to-stage transition rates; a validated dynamic model reproducing Ghana's historical workforce trajectory; workforce density projections under each scenario with uncertainty bounds; and a ranked answer to whether Ghana's binding constraint is production, absorption or retention.

**What the thesis demonstrates.** System dynamics and Markov modelling — methods almost absent from African health workforce research, so the method itself is differentiating. More importantly, it shows you can identify a *system* where others see three separate problems, and can extract a hard parameter from a public foreign register when the domestic source is closed. That last move is the intellectual signature of the whole document: routing around a gatekeeper rather than waiting on one.

---

## Part 5 — The single strongest recommendation

> ### Topic 1 — *Federated Learning for Ghanaian Health Facilities: Training Clinical Models Without Moving Patient Data*

**Why this one.**

**1. It is the question you actually asked.** You asked whether a system can isolate health records to in-country, restricted access. This topic answers that with an implementation and a measured frontier rather than an architecture diagram. A thesis built on a question you raised yourself will be better written than one built on a question handed to you, and that shows.

**2. Nobody can block it.** The core experiment needs a public clinical database, public Ghanaian facility reports, public connectivity data, and your own compute. There is no registrar, no ethics committee bottleneck, no institution to chase. Given that a gatekeeper has already cost you one thesis, this should weigh more heavily than anything else on the list.

**3. It is unmistakably "build and apply a model" — several times over.** You build a distributed system; you build clinical prediction models inside it; you design a factorial experiment; you fit a response surface to the experimental results; you run adversarial privacy attacks against your own models. A supervisor asking "where is the model?" will not get halfway through your table of contents.

**4. It is genuinely unexplored.** Federated learning is a large literature and African digital health is a large literature, but the intersection — federated learning evaluated under *realistic* African infrastructure constraints, with bandwidth, dropout and facility-size imbalance treated as first-class experimental factors rather than footnotes — is close to empty. Your originality requirement is satisfied structurally, not by finding a novel angle on a crowded topic.

**5. The problem is real and current.** Ghana is actively deploying national health information systems and actively legislating on data protection. The question of whether clinical AI requires surrendering data sovereignty is live, is being decided now, and is being decided largely without evidence. A thesis that supplies that evidence is useful in a way most theses are not.

**6. It positions you unusually well afterwards.** Federated learning, differential privacy, secure aggregation, and privacy attack evaluation are among the most sought-after capabilities in health data engineering globally. The skills are transferable far beyond Ghana and far beyond health.

### Two checks before you commit

**Check one — compute.** Confirm you can run the factorial sweep. Logistic-regression federations run comfortably on a laptop; the neural variant wants a GPU. If GPU access is uncertain, restrict the clinical model to logistic regression and small networks, and say so in the scope — the federated-learning findings hold regardless of model class, and the constraint costs you little.

**Check two — the clinical corpus.** Confirm you can obtain a public clinical database with a credentialing process you can complete (some require a short training certificate and a data use agreement — obtainable, but not instant). **Start this in week one**, because it has a lead time and everything else waits on it.

**If either check fails,** switch to **Topic 2 (privacy-preserving record linkage)**. It sits in the same intellectual territory, needs no external data at all, has the highest completion certainty in this document, and contains a genuinely novel Ghana-specific finding in the day-name frequency-skew problem.

### Suggested first three weeks

| Week | Action |
|---|---|
| 1 | Start the clinical-database credentialing process (it has a lead time — do it first). In parallel, read Act 843 and Act 1038 directly and begin the compliance-requirements map. Pull GHS facility reports and characterise the facility-size and tier distribution. |
| 2 | Build the federated pipeline end to end with a trivial model and two clients, just to prove the plumbing. Simultaneously construct the Ghana-calibrated non-IID partition scheme from the facility characterisation. |
| 3 | Add DP-SGD and secure aggregation. Run the first small factorial sweep to validate the experimental harness and estimate total compute time. Lock the proposal. |

The instinct that matters here is the one in week two: **prove the plumbing before you invest in the science.** A federated pipeline that doesn't converge is a three-week debugging exercise, and you want to discover that in week two rather than week ten.

---

## Appendix — Verification notes

- **Legal instruments change and this area is active.** Read Act 843, Act 1038 and any subsequent amendments or new instruments in their current form before writing the compliance chapter. Do not rely on secondary summaries, including this one.
- **Cloud region availability changes.** The claim that no hyperscale provider operates a region inside Ghana was accurate to the best of my knowledge at the time of writing but must be re-verified — providers announce new regions regularly, and a new Ghanaian or West African region would materially change the data-residency argument in Topics 1, 3 and 4.
- **Institutional names and mandates change.** Verify current names, mandates and reporting arrangements for GHS, NHIA, NITA, the Data Protection Commission, the Cyber Security Authority and the professional councils before citing them.
- **No specific statistics are quoted in this document, deliberately.** Every figure you cite — workforce densities, stockout rates, emigration counts, facility numbers, prevalence — must come from the primary source with a date attached.
- **Public dataset availability and access terms change.** Confirm current availability and licensing for every dataset in the Tier A table before building a topic on it, particularly the clinical databases and social media sources.
- **Ethics timelines are real.** If your chosen topic needs GHS ERC or institutional IRB approval, submit in week one. Four to twelve weeks is normal, and it is not a queue you can compress by asking politely.
