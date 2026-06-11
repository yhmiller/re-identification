#!/usr/bin/env python
"""
run_all.py — one-command runner for the full analysis.

Runs Stage 1 (pipeline + XGBoost + SHAP + evaluation) and then Stage 2
(E-XGBoost engineering comparison) inside a SINGLE process, so Stage 2
inherits Stage 1's in-memory objects (schema, data, fitted preprocessor)
instead of failing on undefined names.

Usage:
    ml_env/bin/python run_all.py

Everything is written to results/. Stage 0 (raw-data anonymisation + EDA)
is run separately, only when the college delivers real data — see PROJECT_GUIDE.
"""
import runpy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
NOTEBOOKS = REPO_ROOT / "notebooks"
STAGE_1 = NOTEBOOKS / "01_pipeline_and_experiments.py"
STAGE_2 = NOTEBOOKS / "02_model_engineering.py"

BANNER = "=" * 70


def run_stage(script_path, label, init_globals=None):
    """Execute one stage as __main__ and return its final global namespace."""
    print(f"\n{BANNER}\n  {label}\n{BANNER}", flush=True)
    return runpy.run_path(str(script_path), init_globals=init_globals,
                          run_name="__main__")


def main():
    # results/ paths in the stages are relative; resolve them against the repo
    # root regardless of where the runner was invoked from.
    import os
    os.chdir(REPO_ROOT)

    stage1_globals = run_stage(STAGE_1, "STAGE 1 — Pipeline, XGBoost, SHAP, evaluation")
    run_stage(STAGE_2, "STAGE 2 — E-XGBoost engineering comparison",
              init_globals=stage1_globals)

    print(f"\n{BANNER}\n  DONE — all outputs are in results/\n{BANNER}", flush=True)


if __name__ == "__main__":
    main()
