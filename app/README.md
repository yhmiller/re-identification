# Release Risk Advisor

```bash
./app.sh              # default port 8501
./app.sh 8600         # a specific port
```

`app.sh` sets up the Python environment on first run, the same way `run.sh`
does, and warns before opening if `results/disclosure/` is empty so the Study
findings tab does not just look broken.

Decision support for someone about to **share a dataset** of student academic
records. It answers one question: *if I release it this way, what am I exposing
and what am I keeping?*

## What it will not do, and why

It reports on datasets, never on individuals. No per-student rows, no ranked
lists, no "here is why this student was flagged".

That is not caution, it is the thesis's own argument applied to its own tool.
This study exists to show that per-student output derived from academic records
can identify people. A tool that displayed it would demonstrate the harm the
study documents.

The writing plan states the rule:

> "Aggregate by construction. Reports over a dataset, never over a record. No
> record-level output rendered, exported or logged."

and warns against the shape of the earlier licensure prototype:

> "Its upload-and-explain-each-student flow is the opposite of what this study
> argues and must not be carried across."

That prototype is not reused. Only its layout conventions are.

`tests/test_release_report.py` enforces this: nothing per-record may cross
`release_report.build`, and the interface source is parsed to check that no
Streamlit table receives anything but an aggregate.

## Where the data comes from, and how to refresh it

Two sources, refreshed differently.

### 1. Study findings tab

Reads the CSVs in `results/disclosure/`. These are the measured findings of the
thesis.

**To update:** re-run the pipeline.

```bash
./run.sh                                              # everything
ml_env/bin/python notebooks/08_confirmatory_tests.py  # just the comparisons
```

The tab prints the timestamp of the run that produced what it is showing, so a
stale reading is visible rather than silent. Nothing is cached to disk by the
app; it reads the files each time it loads.

### 2. Assess a release tab

Computes live, from the corpora `strata.py` can load. No stored results are
involved, so it reflects whatever is in `data/` right now.

**Which corpora appear** is decided by `app/data_sources.py`:

| Corpus | Availability |
|---|---|
| Public replication | Always. Safe outside the approved environment. |
| Allied health | Only when `data/` holds it, so only inside the approved environment. |
| Nursing | Same. Also needs `local/.nursing_salt`. |

A restricted corpus whose files are absent does not appear at all, rather than
appearing and failing. That is deliberate: the plan says real data stays in the
approved environment, and a machine without the data is a machine outside it.

**To add a new dataset:** add a loader to `strata.py` beside the three existing
ones and an entry to `CORPUS_ACCESS` in `app/data_sources.py`. The app picks it
up with no other change.

## What it shows

1. **Risk.** Share of records uniquely identifiable, smallest look-alike group.
2. **Leakage.** How much protection publishing derived features from original
   values would give back.
3. **Utility retained.** How much of the analysis survives, measured by fitting
   a model to exactly what a recipient receives.
4. **Drivers.** Which *columns* carry the risk. One row per column, not per
   person.
5. **A verdict** against tolerances the custodian sets, with reasons.

All five come from `release_report.build`, which is the single function the
writing plan asks for in Stage 8.

## Limits to state whenever this is demonstrated

- These are **measured properties under the attacks tested**, not guarantees.
- **Uniqueness is a worst case.** A realistic attacker with imprecise knowledge
  does considerably worse, by a wide margin in these corpora.
- The thresholds are **illustrative defaults**, not standards.
- Small cohorts **cannot be protected by generalisation at any width tested**.
  For those the tool will keep saying no, which is the correct answer.

## Status

Stage 8 of the writing plan. **Optional, and scheduled last on purpose.** The
thesis is complete and defensible without it. If the schedule slips, this is the
first thing to drop.
