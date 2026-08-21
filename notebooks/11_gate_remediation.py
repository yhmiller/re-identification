"""Stage 11. The GATE-3 and GATE-4 remediation sweep.

The testbed examination failed two desk-reject gates. This stage produces the
measurements those gates require, in one pass so that nothing is compared across
different builds.

What it adds over the existing frontier:

    G4.3  a fourth release arm, SELECTIVE, and a dominance test over every arm
    G4.4  suppression at k = 0, 3 and 5 for every arm rather than one
    Q3.1  the full declared risk metric set per configuration, so minimum k,
          marketer risk, l-diversity and t-closeness stop being declared in
          Methods and absent from Results
    Q20.3 the reconstruction procedure and containment check run against the
          new arm, not only against the baseline

**On the selective arm.** Table 5 of the manuscript partitions the eight derived
features by whether each constrains the source arithmetically. Five do. The
selective arm recomputes those five from the protected values and leaves the
other three at original precision, which should close the arithmetic pathway
while keeping features the reconstruction cannot exploit.

The success criterion is fixed here, before the run, and is the one the gate
demands: **uniqueness at or below the proposed arm, AUC-PR inside the baseline
arm's dependence-corrected interval, and not dominated by the withhold arm on
both axes.**

There is a specific reason to expect it may fail in two of the three corpora.
The "does not constrain" classification assumed a single derivation source.
Under a mixed source `gpa_trend` at original precision is the exact difference
between two source values, and where the derived block is computed over only two
predictor positions, that difference plus the published bands pins the pair to a
one-dimensional family. Whether that happens is measured rather than assumed.

Run:
    ml_env/bin/python notebooks/11_gate_remediation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import attack_models as am  # noqa: E402
import deidentify as di  # noqa: E402
import derivation_consistent as dc  # noqa: E402
import risk_utility as ru  # noqa: E402
import strata as st  # noqa: E402

RESULTS = ROOT / "results" / "disclosure"
RESULTS.mkdir(parents=True, exist_ok=True)

ARMS = (dc.NONE, dc.BASELINE, dc.PROPOSED, dc.SELECTIVE)
SUPPRESSION = (0, 3, 5)

# Fixed before the run. Reported whether or not it is met.
SUCCESS_CRITERION = (
    "uniqueness at or below the proposed arm; AUC-PR inside the baseline arm's "
    "dependence-corrected interval; not dominated by the withhold arm"
)


def sweep(stratum, name):
    """Every arm at every band width and suppression threshold."""
    rows = []
    for band in stratum.band_widths:
        for k in SUPPRESSION:
            for mode in ARMS:
                row = ru.frontier_row(
                    stratum.frame,
                    stratum.sequence,
                    stratum.predictors,
                    st.SENSITIVE,
                    f"band {band:g} k{k} {mode}",
                    band_width=band,
                    suppress_k=k or None,
                    derived_mode=mode,
                )
                row["stratum"] = name
                rows.append(row)
    return pd.DataFrame(rows)


def dominance(frame):
    """Which arms are dominated, on risk and utility together.

    An arm is dominated when another arm at the same band width and suppression
    threshold achieves uniqueness no higher and AUC-PR no lower, with at least
    one strictly better. This is the test GATE-4 turns on.
    """
    out = []
    for (band, k), group in frame.groupby(["band_width", "suppress_k"]):
        group = group.dropna(subset=["auc_pr"])
        for row in group.itertuples():
            better = group[
                (group.prop_unique <= row.prop_unique)
                & (group.auc_pr >= row.auc_pr)
                & (group.derived_mode != row.derived_mode)
                & ((group.prop_unique < row.prop_unique) | (group.auc_pr > row.auc_pr))
            ]
            out.append(
                {
                    "stratum": row.stratum,
                    "band_width": band,
                    "suppress_k": k,
                    "derived_mode": row.derived_mode,
                    "prop_unique": row.prop_unique,
                    "auc_pr": row.auc_pr,
                    "dominated": bool(len(better)),
                    "dominated_by": ",".join(sorted(better.derived_mode)) or "",
                }
            )
    return pd.DataFrame(out)


def attack_selective(stratum, name):
    """Run the reconstruction and containment check against the selective arm.

    The constraining features are published from the protected values, so the
    interval propagation should gain nothing from them. Whether the free
    features leak instead is the open question.
    """
    frame, seq = stratum.frame, stratum.sequence
    truth = frame[seq]
    rows = []
    for band in stratum.band_widths:
        low, high = di.band_intervals(frame, seq, band)
        for mode in (dc.BASELINE, dc.PROPOSED, dc.SELECTIVE):
            release = dc.build(frame, seq, band, None, mode)
            published = release.frame[
                [c for c in di.CONSTRAINING_COLUMNS if c in release.frame]
            ]
            rec_low, rec_high, summary = am.reconstruct(
                frame, seq, low, high, published
            )
            metrics = am.recovery_metrics(rec_low, rec_high, summary, band, truth=truth)
            rows.append(
                {
                    "stratum": name,
                    "band_width": band,
                    "derived_mode": mode,
                    **{k: v for k, v in metrics.items() if k != "band_width"},
                }
            )
    return pd.DataFrame(rows)


def main():
    sweeps, attacks = [], []
    for name, stratum in st.load_all().items():
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}", flush=True)
        frame = sweep(stratum, name)
        sweeps.append(frame)
        shown = frame[frame.suppress_k == 0]
        for row in shown.itertuples():
            print(
                f"  band {row.band_width:<5g} {row.derived_mode:<10s} "
                f"unique={row.prop_unique:.4f} minK={row.min_class_size:<4} "
                f"auc_pr={row.auc_pr if not np.isnan(row.auc_pr) else float('nan'):.4f}"
            )
        attacks.append(attack_selective(stratum, name))

    everything = pd.concat(sweeps, ignore_index=True)
    everything.to_csv(RESULTS / "arm_sweep.csv", index=False)

    verdicts = pd.concat(
        [dominance(g) for _, g in everything.groupby("stratum")], ignore_index=True
    )
    verdicts.to_csv(RESULTS / "dominance.csv", index=False)

    pd.concat(attacks, ignore_index=True).to_csv(
        RESULTS / "selective_reconstruction.csv", index=False
    )

    print(f"\n{'=' * 78}\nDominance, at k = 0\n{'=' * 78}")
    for row in verdicts[verdicts.suppress_k == 0].itertuples():
        flag = (
            f"DOMINATED by {row.dominated_by}" if row.dominated else "on the frontier"
        )
        print(
            f"  {row.stratum:<14s} band {row.band_width:<5g} "
            f"{row.derived_mode:<10s} {flag}"
        )

    undominated = verdicts[
        (~verdicts.dominated) & (verdicts.derived_mode == dc.PROPOSED)
    ]
    print(
        f"\nProposed arm undominated in {len(undominated)} of "
        f"{len(verdicts[verdicts.derived_mode == dc.PROPOSED])} configurations"
    )
    print(f"Success criterion, fixed before the run: {SUCCESS_CRITERION}")
    print(f"wrote 3 tables to {RESULTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
