# Sources

Screened 19 August 2026. **Nothing here has been read in full.** Entries record
what the abstract, landing page or search summary stated. Every source marked
**critical** must be read end to end before the thesis cites it.

Sources are grouped by the claim they bear on.

---

## A. Reconstruction and inference from released statistics

Bears on G3. This is the largest threat to the contribution as originally framed.

**Dinur, I. and Nissim, K. (2003). Revealing information while preserving
privacy.** *critical*
The founding result. Releasing statistics from a confidential source leaks
information about it, and sufficiently many sufficiently accurate answers expose
the underlying microdata. Noise of order √n is required to prevent
reconstruction. Later called the Fundamental Law of Information Recovery.
→ https://differentialprivacy.org/reconstruction-theory/

**Abowd, J. et al. (2023). Database reconstruction does compromise
confidentiality.** *PNAS.* *critical*
Demonstrates reconstruction against real published statistics. Internal US Census
testing found 2010 statistics susceptible to reconstruction affecting a large
share of the population.
→ https://www.pnas.org/doi/full/10.1073/pnas.2300976120

**Garfinkel, S., Abowd, J. and Martindale, C. Understanding database
reconstruction attacks on public data.** *ACM Queue.*
Accessible treatment of the same result, useful for the thesis introduction.
→ https://queue.acm.org/detail.cfm?id=3295691

**Mao, Y., Stevanoski, B. and de Montjoye, Y.-A. (2025). DeSIA: attribute
inference attacks against limited fixed aggregate statistics.** *critical, closest single source*
Attribute inference where only a **limited number of fixed** aggregates are
released. Reported conclusion: aggregation alone does not protect privacy even
when few aggregates are published. This is Experiment A's finding, stated
generally and demonstrated on census data.
→ https://arxiv.org/abs/2504.18497

**Generate-then-verify: reconstructing data from limited published statistics
(2025).** *critical*
Reconstruction from a small published statistic set, the same regime as this
study's eight derived features.
→ https://arxiv.org/pdf/2504.21199

**Kasiviswanathan, S., Rudelson, M. and Smith, A. The power of linear
reconstruction attacks.**
Theory for reconstruction from linear statistics. The means used here are linear
in the source values, so this bounds what the interval propagation can achieve.
→ https://arxiv.org/pdf/1210.2381

---

## B. Consistency of derived quantities under disclosure control

Bears on G3. **The closest conceptual prior art found.**

**Cell-key method and additivity restoration, official statistics.** *critical*
Perturbing a table breaks additivity: perturbed inner cells no longer sum to
perturbed margins. Agencies including the Australian Bureau of Statistics
implement an additivity-restoring module, and enforce that the same cell appearing
in different tables receives the **same** perturbation, explicitly to maintain
consistency. Inconsistent treatment of the same underlying quantity is treated as
a defect to be engineered away.
→ https://www.unece.org/fileadmin/DAM/stats/documents/ece/ces/ge.46/2013/Topic_1_ABS.pdf
→ cellKey R package: https://www.ajs.or.at/index.php/ajs/article/view/2131/891

**Statistics Iceland (2020). Methods of statistical disclosure control for
aggregate data.**
Survey of the same problem family.
→ http://hagstofan.s3.amazonaws.com/media/public/2020/e9ea7160-5032-4580-9297-7b3b3cb634da.pdf

**ONS / Government Analysis Function. Disclosure control guidance for microdata
from social surveys; Anonymisation and data confidentiality.**
National guidance on microdata release. Searched specifically for an explicit
rule on recomputing derived variables after protection; none was found, which is
recorded as a negative result rather than as an absence.
→ https://analysisfunction.civilservice.gov.uk/policy-store/anonymisation-and-data-confidentiality/

---

## C. Re-identification risk in education data

Bears on G2. **This group narrows the gap claim materially.**

**Vatsalan, D. et al. (2022). Privacy risk quantification in education data using
Markov model.** *British Journal of Educational Technology.* *critical*
Quantifies re-identification risk in education data, models uniqueness and
uniformity, and validates on real education datasets. Reported finding that even
where uniqueness risk is low, uniformity risk can be high. Explicitly treats
grades from different exam attempts as quasi-identifiers whose combination
indicates a particular student.
→ https://bera-journals.onlinelibrary.wiley.com/doi/full/10.1111/bjet.13223

**De-identification is insufficient to protect student privacy, or what can a
field trip reveal?** *Journal of Learning Analytics.* *critical*
Directly argues that removing identifiers does not protect students.
→ https://www.learning-analytics.info/index.php/JLA/article/view/7353

**Open data, private learners: a de-identified student activity and performance
dataset for learning analytics.** *Scientific Data.*
A worked de-identified educational release, useful as a comparator for the
proposed framework.
→ https://www.nature.com/articles/s41597-026-06821-3

---

## D. Health data de-identification: risk, utility and practice

Bears on the method and the framing.

**Sweeney, L. (2002). k-anonymity.** Foundational, already cited.

**Machanavajjhala, A. et al. (2007). l-diversity.** Foundational, already cited.

**Li, N. et al. (2007). t-closeness.** Foundational, already cited.

**El Emam, K. et al. R-U policy frontiers for health data de-identification.** *critical*
Establishes the risk-utility frontier framing this study adopts. Must be read
before the frontier is presented as the study's own framing.

**Evaluating the re-identification risk of a clinical study report anonymised
under EMA Policy 0070 and Health Canada regulations.**
Applied risk measurement against a regulatory standard.
→ https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7029478/

**Anonymization: the imperfect science of using data while preserving privacy.**
*Science Advances.*
Recent survey, useful for the introduction.
→ https://www.science.org/doi/10.1126/sciadv.adn7053

---

## E. African and Ghanaian health data governance

Bears on G1.

**Data protection legislation in Africa and pathways for enhancing compliance in
big data health research.**
→ https://www.researchgate.net/publication/384937773

**The regulation of health data sharing in Africa: a comparative study.** *PMC.*
Covers Ghana, Kenya, Nigeria, South Africa and Uganda. Records Ghana as having
formal data protection legislation from October 2012.
→ https://pmc.ncbi.nlm.nih.gov/articles/PMC10800019/

**DS-I Africa Law project.** Examines how de-identification, anonymisation and
pseudonymisation are treated by legislators across twelve English-speaking
African jurisdictions including Ghana.
→ https://academic.oup.com/jlb/article-pdf/12/1/lsae029/62572710/lsae029.pdf

**A proposed de-identification framework for a cohort of children presenting at a
health facility in Uganda.** *critical*
The nearest African empirical analogue found: applies k-anonymity and a
risk-threshold approach to real African health records. The closest comparator
for this study's method, and the strongest single challenge to G1 as stated.
→ https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9931294/

**DLA Piper. Data protection laws in Ghana.** Practitioner summary of Act 843.
→ https://www.dlapiperdataprotection.com/index.html?t=law&c=GH
