#!/usr/bin/env python
"""
run_all.py — one-command runner for the full analysis.

Runs Stage 1 (pipeline + XGBoost + SHAP + evaluation) and then Stage 2
(E-XGBoost engineering comparison) inside a SINGLE process, so Stage 2
inherits Stage 1's in-memory objects (schema, data, fitted preprocessor)
instead of failing on undefined names.

Usage:
    ml_env/bin/python run_all.py [--verbose / -v]
    ml_env/bin/python run_all.py --synthetic-outcomes [PATH]
        Run on the REAL academic records with a SIMULATED licensure outcome
        from a registrar-format return workbook. Bare (no PATH) defaults to
        the primary prevalence-0.40 workbook. Results are labelled and
        isolated in results/real_synthetic_outcome_p<prevalence>/ — see
        notebooks/01_pipeline_and_experiments.py, CELL 6.

Everything is written to results/. Stage 0 (raw-data anonymisation + EDA)
is run separately, only when the college delivers real data — see PROJECT_GUIDE.
"""
import csv
import runpy
import subprocess
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

from baseline_xgboost import N_CV_FOLDS   # single source of truth for the fold count

REPO_ROOT = Path(__file__).resolve().parent
NOTEBOOKS = REPO_ROOT / "notebooks"
STAGE_1 = NOTEBOOKS / "01_pipeline_and_experiments.py"
STAGE_2 = NOTEBOOKS / "02_model_engineering.py"
RUN_MANIFEST = REPO_ROOT / "results" / "run_manifest.csv"
# Default workbook for a bare `--synthetic-outcomes` (no path given).
SYNTHETIC_OUTCOMES_DEFAULT = REPO_ROOT / "data" / "OUTCOME_REQUEST_2021_2022_SYNTHETIC_RETURN_p400.xlsx"
MANIFEST_FIELDS = ["timestamp", "git_commit", "data_source", "model",
                    "n_features", "auc_roc", "auc_pr", "f1_weighted",
                    "precision", "recall", "brier"]

BANNER = "=" * 70
TABLE_RULE = "=" * 78
RULE_WIDTHS = {"=" * 60, "=" * 70, "=" * 78}
PROGRESS_PREFIXES = ("✅", "⚠️", "ℹ️", "🎉", "[BASELINE]", "[E-XGBoost]", "PHASE")


def is_stage_banner(line):
    return ("STAGE 1 -" in line or "STAGE 2 —" in line
            or "DONE — all outputs" in line)


def is_rule(stripped):
    return stripped in RULE_WIDTHS


def is_metric_header(stripped):
    return (stripped.startswith("Model")
            and "AUC-ROC" in stripped and "AUC-PR" in stripped)


def is_progress_marker(stripped):
    return stripped.startswith(PROGRESS_PREFIXES)


def is_headline_metric(line):
    return "±" in line and ("AUC-ROC:" in line or "AUC-PR" in line)


def synthetic_outcomes_arg():
    """
    Return the return-workbook path for --synthetic-outcomes, or None if the
    flag was not passed. Bare `--synthetic-outcomes` (nothing follows it, or
    the next token is itself a flag) resolves to SYNTHETIC_OUTCOMES_DEFAULT;
    `--synthetic-outcomes <PATH>` uses the given path.
    """
    if "--synthetic-outcomes" not in sys.argv:
        return None
    idx = sys.argv.index("--synthetic-outcomes")
    has_path = idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-")
    return sys.argv[idx + 1] if has_path else str(SYNTHETIC_OUTCOMES_DEFAULT)


class CleanStdoutWrapper:
    def __init__(self, original_stdout, log_file):
        self.original_stdout = original_stdout
        self.log_file = log_file
        self.buffer = ""
        self.in_table = False
        self.in_baseline_table = False
        self.in_temporal_table = False

    def write(self, data):
        self.log_file.write(data)
        self.log_file.flush()

        self.buffer += data
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self.process_line(line)

    def flush(self):
        self.original_stdout.flush()
        self.log_file.flush()

    def emit(self, line):
        self.original_stdout.write(line + "\n")
        self.original_stdout.flush()

    def process_line(self, line):
        stripped = line.strip()

        if self.in_table:
            self.emit(line)
            if "Comparison table saved" in line:
                self.in_table = False
                self.emit(TABLE_RULE)
            return

        if self.in_baseline_table:
            self.emit(line)
            if stripped == "" or "saved" in stripped:
                self.in_baseline_table = False
            return

        if self.in_temporal_table:
            self.emit(line)
            self.in_temporal_table = False
            return

        if is_stage_banner(line):
            self.emit(line)
        elif is_rule(stripped):
            return
        elif "TABLE 1 -" in line:
            self.in_table = True
            self.emit(TABLE_RULE)
            self.emit(line)
        elif is_metric_header(stripped):
            self.in_baseline_table = True
            self.emit(line)
        elif "train_cohorts" in line and "test_cohort" in line:
            self.in_temporal_table = True
            self.emit(line)
        elif is_progress_marker(stripped) or is_headline_metric(line):
            self.emit(line)

    def finish(self):
        if self.buffer:
            self.process_line(self.buffer)
            self.buffer = ""


class CleanStderrWrapper:
    def __init__(self, original_stderr, log_file):
        self.original_stderr = original_stderr
        self.log_file = log_file

    def write(self, data):
        self.log_file.write(data)
        self.log_file.flush()
        if "Warning" in data or "Info" in data or "[LightGBM]" in data:
            return
        self.original_stderr.write(data)
        self.original_stderr.flush()

    def flush(self):
        self.original_stderr.flush()
        self.log_file.flush()


def run_stage(script_path, label, init_globals=None):
    """Execute one stage as __main__ and return its final global namespace."""
    print(f"\n{BANNER}\n  {label}\n{BANNER}", flush=True)
    return runpy.run_path(str(script_path), init_globals=init_globals,
                          run_name="__main__")


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _cv_mean_row(cv_results, model_label, n_features):
    """One manifest row from a CV-results dict (acc/f1/auc_roc/auc_pr/time lists)."""
    return {
        "model": model_label,
        "n_features": n_features,
        "auc_roc": round(float(sum(cv_results["auc_roc"]) / len(cv_results["auc_roc"])), 4),
        "auc_pr": round(float(sum(cv_results["auc_pr"]) / len(cv_results["auc_pr"])), 4),
    }


def append_run_manifest(stage1_globals, stage2_globals):
    """
    Append one row per model/run to results/run_manifest.csv — so results
    accumulate across runs (baseline / E-XGBoost / real / synthetic) instead
    of only ever showing the latest one. See docs/TODO.md, "Result tracking,
    class metrics, and hypothesis testing".
    """
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    git_commit = _git_commit()
    data_source = stage1_globals.get("DATA_SOURCE", "unknown")

    rows = []

    xgb_res = stage1_globals.get("xgb_res")
    if xgb_res:
        rows.append({
            "model": "XGBoost (baseline, held-out test)",
            "n_features": len(stage1_globals.get("ALL_FEATURES_V2", [])),
            "auc_roc": round(xgb_res["auc_roc"], 4),
            "auc_pr": round(xgb_res["auc_pr"], 4),
            "f1_weighted": round(xgb_res["f1_weighted"], 4),
            "precision": round(xgb_res["precision"], 4),
            "recall": round(xgb_res["recall"], 4),
            "brier": round(xgb_res["brier"], 4),
        })

    baseline_cv = stage2_globals.get("baseline_cv_results")
    if baseline_cv:
        rows.append(_cv_mean_row(baseline_cv, f"XGBoost (baseline, {N_CV_FOLDS}-fold CV mean)",
                                  len(stage2_globals.get("ALL_FEATURE_LIST", []))))

    engineered_cv = stage2_globals.get("engineered_cv_results")
    if engineered_cv:
        rows.append(_cv_mean_row(engineered_cv, f"E-XGBoost ({N_CV_FOLDS}-fold CV mean)",
                                  len(stage2_globals.get("kept_features", []))))

    if not rows:
        return

    RUN_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RUN_MANIFEST.exists()
    with open(RUN_MANIFEST, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            row["timestamp"] = timestamp
            row["git_commit"] = git_commit
            row["data_source"] = data_source
            writer.writerow(row)
    print(f"   Run manifest updated → {RUN_MANIFEST.relative_to(REPO_ROOT)} (+{len(rows)} rows)")


def main():
    # results/ paths in the stages are relative; resolve them against the repo
    # root regardless of where the runner was invoked from.
    os.chdir(REPO_ROOT)

    # Make schema.py (the shared feature schema) importable from the
    # notebooks, which runpy executes with notebooks/ — not the repo root — as
    # sys.path[0].
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    # Run on the REAL academic records with a SIMULATED licensure outcome
    # from a registrar-format return workbook, instead of the default real/
    # pilot-data selection in notebook 01, CELL 6. Read by the notebook via
    # os.environ["SYNTHETIC_OUTCOMES"]; resolved against REPO_ROOT because
    # main() has already chdir'd there, so a relative PATH still works.
    synthetic_outcomes = synthetic_outcomes_arg()
    if synthetic_outcomes:
        synthetic_outcomes_path = Path(synthetic_outcomes)
        if not synthetic_outcomes_path.is_absolute():
            synthetic_outcomes_path = REPO_ROOT / synthetic_outcomes_path
        os.environ["SYNTHETIC_OUTCOMES"] = str(synthetic_outcomes_path)

    if not verbose:
        os.makedirs("results", exist_ok=True)
        log_path = os.path.join("results", "analysis.log")
        log_file = open(log_path, "w", encoding="utf-8")

        print(BANNER)
        print("  RUNNING ANALYSIS PIPELINE (CLEAN MODE)")
        print("  Full verbose logs will be saved to: results/analysis.log")
        print("  To run in verbose mode, use: ./run.sh --verbose")
        print(BANNER)

        # Stage scripts branch on this to suppress their own chatty output.
        os.environ["CLEAN_RUN"] = "1"

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = CleanStdoutWrapper(original_stdout, log_file)
        sys.stderr = CleanStderrWrapper(original_stderr, log_file)

    try:
        stage1_globals = run_stage(STAGE_1, "STAGE 1 - Pipeline, XGBoost, SHAP, evaluation")
        stage2_globals = run_stage(STAGE_2, "STAGE 2 — E-XGBoost engineering comparison",
                  init_globals=stage1_globals)
        append_run_manifest(stage1_globals, stage2_globals)
    finally:
        if not verbose:
            sys.stdout.finish()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()

    print(f"\n{BANNER}\n  DONE — all outputs are in results/\n{BANNER}", flush=True)


if __name__ == "__main__":
    main()
