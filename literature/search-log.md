# Search log

All queries executed 19 August 2026 against the open web. Recorded verbatim so
the search is reproducible and so a reader can see what was and was not asked.

| # | Query | RQ | Outcome |
|---|---|---|---|
| 1 | derived features leak information generalized quasi-identifiers k-anonymity aggregate statistics inconsistency | L1, L3 | No direct hit on the mechanism. Returned general k-anonymity limitations material. |
| 2 | database reconstruction attack from aggregate statistics Dinur Nissim census differencing | L2 | **High yield.** Located the reconstruction-from-aggregates literature. |
| 3 | statistical disclosure control consistency derived variables recomputed after perturbation additivity tables | L3 | **High yield.** Located the cell-key and additivity-restoration literature in official statistics. |
| 4 | DeSIA attribute inference attacks limited fixed aggregate statistics 2025 | L1, L2 | **High yield.** Closest single source to Experiment A. |
| 5 | re-identification risk student educational records learning analytics quasi-identifiers grades empirical | L4 | **High yield.** Education records have been studied. G2 narrows. |
| 6 | privacy leakage engineered features machine learning pipeline anonymized dataset released alongside generalized attributes | L1 | Low yield. Returned ML data-leakage material, a different sense of leakage. |
| 7 | re-identification risk health data Africa Ghana data protection act anonymisation empirical study | L5 | Moderate. Legal and comparative work found; no Ghanaian empirical estimate. |
| 8 | medical nursing student academic transcript records privacy re-identification health professions education | L4 | Low yield. Returned FERPA and HIPAA compliance material, not risk measurement. |
| 9 | "derived variables" OR "derived attributes" recompute from anonymised data guidance disclosure control best practice | L3 | Moderate. National guidance located; no explicit derivation-consistency rule found. |

## Queries deliberately not yet run

These need database access and belong to the extension in the protocol.

- Boolean strings combining (generalisation OR k-anonymity) AND (derived OR
  engineered OR summary) AND (leakage OR inference OR reconstruction), fielded to
  title and abstract, in Scopus and ACM DL
- Citation chasing forward from Dinur and Nissim (2003) filtered to work on
  microdata rather than query answering
- Citation chasing forward from the cell-key literature into microdata release

## Note on search asymmetry

Queries 1, 6 and 9 targeted the exact mechanism and returned little. That is weak
evidence of a gap and strong evidence that the mechanism has no settled name. A
literature can exist under vocabulary the searcher does not know, and the absence
of a hit for a phrase invented by the searcher is close to uninformative. The
gap analysis treats it accordingly.
