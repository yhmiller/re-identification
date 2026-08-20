"""Where the app gets its data, and the rules about which data it may touch.

Two sources, deliberately separated.

`STUDY_RESULTS` are the tables `./run.sh` writes to `results/disclosure/`. They
are the measured findings of the thesis. **To refresh them, re-run the
pipeline**; the app reads whatever is on disk and reports the file timestamp so
a stale reading is visible rather than silent.

`assessable_corpora()` are datasets the app may run a live assessment on. The
writing plan is explicit about which those are:

    "Real data stays in the approved environment. Any demonstration outside it
    runs on the replication corpus or on synthetic input."

So the public replication corpus is always offered. The two Ghanaian corpora are
offered **only** when their source files are present, which is to say only on a
machine already inside the approved environment. On any other machine they do
not appear at all, rather than appearing and failing.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results" / "disclosure"

# Corpora the app may assess live, and whether each may leave the approved
# environment. Restricted corpora are hidden entirely when their data is absent.
CORPUS_ACCESS = {
    "public": {
        "label": "Public replication corpus (Portuguese schools)",
        "restricted": False,
    },
    "allied_health": {
        "label": "Allied health (Accra School of Hygiene)",
        "restricted": True,
    },
    "nursing": {"label": "Nursing (BSc)", "restricted": True},
}


def study_results():
    """The thesis tables, with the timestamp of the run that produced them."""
    wanted = {
        "frontier": "risk_utility_frontier.csv",
        "risk": "confirmatory_risk.csv",
        "utility": "confirmatory_utility.csv",
        "ablation": "experiment_b_reconstruction.csv",
        "baseline_profiles": "baseline_risk_profiles.csv",
        "group_size": "group_size_effect.csv",
        "linkage": "linkage_attack.csv",
    }
    tables, newest = {}, None
    for key, name in wanted.items():
        path = RESULTS / name
        if not path.exists():
            continue
        tables[key] = pd.read_csv(path)
        stamp = datetime.fromtimestamp(path.stat().st_mtime)
        newest = stamp if newest is None else max(newest, stamp)
    return tables, newest


def results_are_available():
    return (RESULTS / "risk_utility_frontier.csv").exists()


def assessable_corpora():
    """Corpora this machine may run a live assessment on.

    Loading is attempted rather than assumed, so a restricted corpus whose files
    are missing is simply absent from the list.
    """
    import strata as st

    available = {}
    for name, loader in st.LOADERS.items():
        meta = CORPUS_ACCESS.get(name)
        if meta is None:
            continue
        try:
            available[name] = {**meta, "stratum": loader()}
        except Exception:
            continue  # restricted data not present on this machine
    return available
