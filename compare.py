#!/usr/bin/env python
"""Baseline against engineered, side by side, in one command.

    ml_env/bin/python compare.py

The engineered artefact in this study is a release procedure, not a model. The
baseline is standard generalisation, which bands the source variables and then
publishes the derived features an analyst asked for at their original precision.
The engineered arm changes exactly one step: those features are recomputed from
the protected values. Everything else, banding, suppression, model,
hyperparameters and fold assignments, is held identical, so any difference
between the two columns below is attributable to the derivation source alone.

XGBoost appears here as the measuring instrument rather than the contribution.
It answers one question: how much of the analysis survives each release.

Reads the tables `run_all.py` produces. Run `./run.sh` first if they are absent.
"""

import sys
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results" / "disclosure"

CORPUS_LABEL = {
    "allied_health": "Allied health",
    "nursing": "Nursing",
    "public": "Public (replication)",
}
RULE = "=" * 78
THIN = "-" * 78


def _load():
    needed = [
        "confirmatory_risk.csv",
        "confirmatory_utility.csv",
        "risk_utility_frontier.csv",
    ]
    missing = [n for n in needed if not (RESULTS / n).exists()]
    if missing:
        raise SystemExit(
            f"missing {', '.join(missing)} in {RESULTS}. Run ./run.sh first."
        )
    return (pd.read_csv(RESULTS / n) for n in needed)


def _reference(frontier, stratum, band):
    """Unprotected and generalised-only AUC-PR, as context for the two arms."""
    rows = frontier[frontier.stratum == stratum]
    unprotected = rows[rows.config == "original, no protection"]
    generalised = rows[rows.config == f"band {band:g}"]
    return (
        float(unprotected.auc_pr.iloc[0]) if len(unprotected) else float("nan"),
        float(generalised.auc_pr.iloc[0]) if len(generalised) else float("nan"),
    )


def main():
    risk, utility, frontier = _load()

    print(f"\n{RULE}")
    print("  BASELINE  vs  ENGINEERED")
    print("  standard generalisation  vs  derivation-consistent generalisation")
    print("  Modify [M]: one step changes, the source the features are derived from")
    print(RULE)

    for stratum in ("allied_health", "nursing", "public"):
        u = utility[utility.stratum == stratum].sort_values("band_width")
        r = risk[risk.stratum == stratum].sort_values("band_width")
        if u.empty:
            continue

        print(f"\n{CORPUS_LABEL[stratum]}")
        print(THIN)
        print(
            f"  {'band':>5}  {'DISCLOSURE RISK (prop. unique)':<34}"
            f"{'ANALYTICAL UTILITY (AUC-PR)':<30}"
        )
        print(
            f"  {'':>5}  {'baseline':>9} {'engineered':>11} {'change':>10}   "
            f"{'baseline':>9} {'engineered':>11} {'change':>9}"
        )

        for _, ur in u.iterrows():
            rr = r[r.band_width == ur.band_width]
            if rr.empty:
                continue
            rr = rr.iloc[0]

            # Both arms are recorded directly. An earlier version added the
            # difference to the ablation table's post-attack uniqueness, which
            # is a different quantity, and produced a negative proportion.
            print(
                f"  {ur.band_width:>5g}  {rr.prop_unique_baseline:>9.3f} "
                f"{rr.prop_unique_proposed:>11.3f} "
                f"{rr.difference_full_corpus:>+10.3f}   "
                f"{ur.auc_pr_baseline:>9.3f} {ur.auc_pr_proposed:>11.3f} "
                f"{ur.mean_difference:>+9.3f}"
            )

        print()
        for _, ur in u.iterrows():
            rr = r[r.band_width == ur.band_width]
            if rr.empty:
                continue
            rr = rr.iloc[0]
            risk_verdict = (
                "protection improves"
                if bool(rr.excludes_zero)
                else "no detectable change"
            )
            cost_verdict = (
                f"costs {abs(ur.mean_difference):.3f} AUC-PR"
                if ur.corrected_p < 0.05
                else "cost not distinguishable from zero"
            )
            print(
                f"    band {ur.band_width:<5g} {risk_verdict:<22} "
                f"{cost_verdict:<38} (p={ur.corrected_p:.3f}, d={ur.cohens_d:+.2f})"
            )

    print(f"\n{RULE}")
    print("  WHAT THIS SHOWS")
    print(RULE)
    n_risk = int(risk.excludes_zero.sum())
    n_util = int((utility.corrected_p < 0.05).sum())
    worst = utility.loc[utility.mean_difference.idxmin()]
    print(f"""
  The engineered arm cuts disclosure risk in {n_risk} of {len(risk)} configurations.
  It costs analytical utility in {n_util} of {len(utility)}, the largest being
  {abs(worst.mean_difference):.3f} AUC-PR ({CORPUS_LABEL[worst.stratum]}, band {worst.band_width:g}).

  So the baseline is not a mistake to correct. It sits at the high-utility,
  low-protection end of a real trade-off, and the engineered arm buys protection
  with utility. The frontier in results/disclosure/risk_utility_frontier.csv is
  what a custodian actually chooses from.

  Why the baseline leaks: its published features are exact order statistics and
  means of the values that were just banded, so they constrain those values back
  towards the numbers the bands were meant to hide. The engineered arm's features
  are a deterministic function of the columns already published, so a recipient
  can recompute them and learns nothing new.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
