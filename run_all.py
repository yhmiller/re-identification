#!/usr/bin/env python
"""One-command runner for the disclosure-risk analysis.

    ./run.sh              # sets up the environment first time, then runs this
    ml_env/bin/python run_all.py

Runs the five analysis stages in order, then rebuilds every manuscript figure
and table. Outputs land in results/disclosure/.

Each stage runs in its own process. They communicate through files rather than
through a shared namespace, so a stage can be re-run on its own without the
ones before it, which is what makes the notebooks usable during writing.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = ROOT / "notebooks"

STAGES = [
    ("04_disclosure_risk.py", "Uniqueness, k-anonymity, l-diversity, t-closeness"),
    ("05_derived_feature_experiments.py", "What the derived features add to risk"),
    ("06_linkage_attack.py", "Reconstruction and linkage under the adversary models"),
    ("07_risk_utility_frontier.py", "Risk and utility across both derivation arms"),
    ("08_confirmatory_tests.py", "Pre-specified confirmatory comparisons"),
]

RULE = "=" * 78


def run(script, description):
    print(f"\n{RULE}\n  {script}  —  {description}\n{RULE}", flush=True)
    result = subprocess.run([sys.executable, str(NOTEBOOKS / script)], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"{script} failed with exit code {result.returncode}")


def main():
    for script, description in STAGES:
        run(script, description)

    print(f"\n{RULE}\n  Rebuilding figures and tables\n{RULE}", flush=True)
    import figures
    figures.build_all()

    # Ends on the comparison the thesis is about, so a run answers "what did the
    # engineered arm do against the baseline" without anyone opening a CSV.
    import compare
    compare.main()

    print(f"\n{RULE}\n  Done. Everything is in results/disclosure/\n{RULE}", flush=True)


if __name__ == "__main__":
    main()
