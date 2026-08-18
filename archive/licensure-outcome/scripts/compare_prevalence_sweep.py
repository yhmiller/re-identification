#!/usr/bin/env python
"""Compare the three simulated-outcome runs across the prevalence sweep.

Prevalence is an assumption: no Allied Health Professions Council pass rate is
published. So the useful question is not "what are the metrics" but "does the
answer change when the assumption changes". If the E-XGBoost pruned feature set
moves across the sweep, the pruning is prevalence-dependent, and a thesis that
reports one pruned set as a finding would be reporting an artefact of the
assumption instead.

Usage:
    ml_env/bin/python scripts/compare_prevalence_sweep.py
"""
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import outcome_model            # noqa: E402

RESULTS = REPO_ROOT / "results"
OUTPUT_DIR = RESULTS / "prevalence_sweep"
STABILITY = "feature_stability.csv"
BUNDLE = "inference_bundle.pkl"
METRIC_FILES = {"baseline": "baseline_metrics.json",
                "engineered": "engineered_metrics.json"}

# Unlike results/real_synthetic_outcome_pNNN/, this directory name carries no
# marker, so every row states its own provenance. The metrics below are derived
# from generated outcomes and are not reportable as findings.
RECORD_TYPE_COLUMN = "record_type"
SIMULATED_OUTCOME = "SIMULATED_OUTCOME"


def run_dir(prevalence):
    return RESULTS / f"real_synthetic_outcome_p{int(round(prevalence * 1000)):03d}"


def baseline_features(directory):
    """Every feature the baseline model saw, from the stability table."""
    return pd.read_csv(directory / STABILITY)["feature"].tolist()


def kept_features(directory):
    """The features E-XGBoost retained, from the bundle the app is served."""
    # Bundle is generated locally by this repo's own pipeline (prediction.save_bundle),
    # not an externally-sourced or untrusted artefact, so unpickling it is safe here.
    with open(directory / BUNDLE, "rb") as handle:
        return list(pickle.load(handle)["feature_columns"])


def mean_metrics(directory, prefix, filename):
    """Fold-wise metric lists collapsed to their mean."""
    path = directory / filename
    if not path.exists():
        return {}
    out = {}
    for name, value in json.loads(path.read_text()).items():
        if isinstance(value, list) and value:
            out[f"{prefix}_{name}"] = sum(value) / len(value)
        elif isinstance(value, (int, float)):
            out[f"{prefix}_{name}"] = value
    return out


def main():
    rows = []
    for prevalence in outcome_model.PREVALENCES:
        directory = run_dir(prevalence)
        if not directory.exists():
            sys.exit(f"Missing {directory}. Run run_all.py --synthetic-outcomes "
                     f"for prevalence {prevalence} first.")

        baseline = baseline_features(directory)
        kept = kept_features(directory)
        pruned = [f for f in baseline if f not in kept]

        row = {
            RECORD_TYPE_COLUMN: SIMULATED_OUTCOME,
            "prevalence": prevalence,
            "n_features_baseline": len(baseline),
            "n_features_engineered": len(kept),
            "kept_features": "|".join(sorted(kept)),
            "pruned_features": "|".join(sorted(pruned)),
        }
        for prefix, filename in METRIC_FILES.items():
            row.update(mean_metrics(directory, prefix, filename))
        rows.append(row)

    table = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / "sensitivity_table.csv"
    table.to_csv(destination, index=False)

    summary = table[["prevalence", "n_features_baseline",
                     "n_features_engineered", "baseline_auc_roc",
                     "engineered_auc_roc"]]
    print(summary.to_string(index=False))
    print(f"\nWrote {destination.relative_to(REPO_ROOT)}")

    kept_sets = {frozenset(r.split("|")) for r in table.kept_features}
    if len(kept_sets) == 1:
        print("\nThe E-XGBoost pruned feature set is STABLE across the sweep: "
              "the same features are retained at every prevalence.")
    else:
        union = set().union(*kept_sets)
        common = set.intersection(*(set(s) for s in kept_sets))
        print(f"\n⚠️  The pruned feature set CHANGES with prevalence. "
              f"{len(common)} features are retained at every prevalence; "
              f"{len(union - common)} are retained at some but not all: "
              f"{sorted(union - common)}")
        print("Report this as a sensitivity observation: the pruning decision "
              "depends on an assumption, not only on the data. The metrics "
              "behind it come from simulated outcomes and are not findings.")


if __name__ == "__main__":
    main()
