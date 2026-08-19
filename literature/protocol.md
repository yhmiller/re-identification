# Search protocol

Registered 19 August 2026, before the gap analysis was written.

## What this is, and what it is not

This is a **structured scoping search over the open web and open-access
repositories**. It follows a written protocol, logs every query, and applies
stated inclusion criteria, so it is reproducible and auditable.

It is **not** a systematic review in the PRISMA sense. No bibliographic database
was searched, no reference lists were snowballed to exhaustion, no second
reviewer screened, and no full texts were read end to end. Describing it as
systematic in the thesis would misstate what was done, in a study whose entire
argument is about claims outrunning evidence.

Section "Extension required before submission" states what must be added before
the word systematic can be used.

## Purpose

The thesis makes three gap claims. This search exists to test them, and to narrow
any that the evidence does not support.

| | Claim as drafted | Status to be tested |
|---|---|---|
| G1 | No measured re-identification risk exists for Ghanaian health data | Does any empirical estimate exist? |
| G2 | Health professions education records are unexamined as a disclosure-control problem | Has education data been studied this way? |
| G3 | Derived features computed at original precision, released alongside a generalised version of their source, have received limited empirical attention | Is this already covered by adjacent literatures? |

G3 is the contribution. It is the claim most exposed and receives the most
attention here.

## Research questions

- RQ-L1. Has the interaction between generalisation of a source variable and
  publication of features derived from that variable at original precision been
  studied empirically?
- RQ-L2. What is established about information leakage from released aggregate or
  summary statistics?
- RQ-L3. Is consistency between a protected variable and quantities derived from
  it an established requirement anywhere in disclosure control practice?
- RQ-L4. What empirical re-identification risk work exists on student or
  educational records?
- RQ-L5. What empirical re-identification risk work exists on health data in
  Ghana or comparable African jurisdictions?
- RQ-L6. What is established about the risk-utility trade-off in health data
  de-identification?

## Sources searched

Open web and open-access repositories reachable from it, including arXiv, PubMed
Central, journal open-access pages, national statistical institute
methodological publications, and regulatory guidance.

## Inclusion criteria

A source is included if it addresses at least one research question and is one of:
a peer-reviewed article, a preprint from a recognised group, an official
statistical agency methodological publication, or a national regulatory guidance
document.

## Exclusion criteria

Vendor marketing pages, undated blog posts, and general explainers that cite no
primary work. These appeared frequently in the results and were excluded on
sight, with the exclusion logged.

## Screening

Single reviewer. Title and abstract, or the equivalent landing-page description,
were screened against the research questions. Sources retained were recorded with
the question they bear on and their effect on the corresponding gap claim.

## Extension required before submission

The following must be completed before the thesis uses the word systematic, or
before it repeats any gap claim in absolute form.

- [ ] Database searches with recorded Boolean strings and result counts: Scopus,
      Web of Science, ACM Digital Library, IEEE Xplore, PubMed
- [ ] Backward and forward citation chasing on the five most central sources
      identified here
- [ ] Full-text reading of every source marked **critical** in `sources.md`
- [ ] A second screener, or a documented justification for single screening
- [ ] A PRISMA flow diagram with counts at each stage
