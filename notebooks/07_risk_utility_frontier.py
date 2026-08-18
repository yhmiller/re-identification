"""Stage 7. The risk-utility frontier.

Phase 5 of the re-identification study. Sweeps de-identification configurations
and measures, on each, what it protects and what it costs.

Utility is deliberately a prediction task the protected columns feed. Early
semester GPAs predict weak performance in the final observed semester, which is
the early-warning question the original pipeline was built for. Protecting the
predictors is supposed to hurt; the point is to find out by how much, and where
the curve bends.

Non-circularity matters here. The target is derived from the final observed
semester, so that semester is excluded from the predictors, as are CGPA and the
grade counts, both of which are computed over the whole sequence including it.

Run:
    ml_env/bin/python notebooks/07_risk_utility_frontier.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import risk_utility as ru  # noqa: E402
import strata as st  # noqa: E402

RESULTS = ROOT / "results" / "disclosure"
RESULTS.mkdir(parents=True, exist_ok=True)

TARGET = st.SENSITIVE

ROLES = {"allied_health": "study population",
         "nursing": "study population",
         "public": "replication corpus"}




def configs_for(stratum):
    """Release configurations, with band widths scaled to the corpus."""
    narrow, mid, wide = stratum.band_widths
    return [
        ("original, no protection", None, None, "none"),
        (f"band {narrow:g}", narrow, None, "none"),
        (f"band {mid:g}", mid, None, "none"),
        (f"band {wide:g}", wide, None, "none"),
        (f"band {mid:g} + suppress k<3", mid, 3, "none"),
        (f"band {mid:g} + suppress k<5", mid, 5, "none"),
        (f"band {mid:g} + derived from originals", mid, None, "leaky"),
        (f"band {mid:g} + derived from bands", mid, None, "safe"),
        (f"band {wide:g} + derived from originals", wide, None, "leaky"),
        (f"band {wide:g} + derived from bands", wide, None, "safe"),
    ]


def main():
    all_rows = []

    for name, stratum in st.load_all().items():
        frame = stratum.frame
        predictors = stratum.predictors
        SEMESTERS = stratum.sequence
        print(
            f"\n{'=' * 86}\n{name} ({ROLES[name]}): {len(frame)} students, "
            f"predictors {', '.join(predictors)}\n{'=' * 86}"
        )
        print(
            f"  {'configuration':<38}{'unique':>8}{'min k':>7}"
            f"{'AUC-ROC':>9}{'AUC-PR':>8}{'PR-floor':>10}{'n':>6}"
        )

        rows = []
        for label, width, suppress_k, mode in configs_for(stratum):
            row = ru.frontier_row(
                frame,
                SEMESTERS,
                predictors,
                TARGET,
                label,
                band_width=width,
                suppress_k=suppress_k,
                derived_mode=mode,
            )
            row["stratum"] = name
            # Suppression that empties the file is not protection, it is
            # deletion. Reported as such rather than as a risk of zero.
            row["fully_suppressed"] = row["records_suppressed"] >= len(frame)
            rows.append(row)

            if row["fully_suppressed"]:
                print(f"  {label:<38}{'all records suppressed':>40}")
            else:
                print(
                    f"  {label:<38}{row['prop_unique']:>7.1%}"
                    f"{row['min_class_size']:>7}"
                    f"{row['auc_roc']:>9.3f}{row['auc_pr']:>8.3f}"
                    f"{row['auc_pr_above_floor']:>+10.3f}{row['n_modelled']:>6}"
                )

        frame_rows = pd.DataFrame(rows)
        baseline = frame_rows.iloc[0]
        frame_rows["auc_pr_retained"] = frame_rows["auc_pr"] / baseline["auc_pr"]
        frame_rows["risk_reduction"] = (
            baseline["prop_unique"] - frame_rows["prop_unique"]
        ) / baseline["prop_unique"]
        all_rows.append(frame_rows)

        usable = frame_rows[~frame_rows["fully_suppressed"]]
        print(f"\n  {'configuration':<38}{'risk cut':>10}{'utility kept':>14}")
        for _, r in usable.iloc[1:].iterrows():
            print(
                f"  {r['config']:<38}{r['risk_reduction']:>9.1%}"
                f"{r['auc_pr_retained']:>14.1%}"
            )

        # A configuration is dominated if another cuts at least as much risk
        # while keeping at least as much utility, and beats it on one of the
        # two. Dominated configurations are strictly worse choices, so the
        # frontier is what remains.
        candidates = usable.iloc[1:]
        frontier = []
        for _, r in candidates.iterrows():
            dominated = any(
                (o["risk_reduction"] >= r["risk_reduction"])
                and (o["auc_pr_retained"] >= r["auc_pr_retained"])
                and (
                    (o["risk_reduction"] > r["risk_reduction"])
                    or (o["auc_pr_retained"] > r["auc_pr_retained"])
                )
                for _, o in candidates.iterrows()
            )
            if not dominated:
                frontier.append(r)

        frame_rows["on_frontier"] = frame_rows["config"].isin(
            [r["config"] for r in frontier]
        )

        print(
            f"\n  Pareto frontier ({len(frontier)} of {len(candidates)} configurations):"
        )
        for r in sorted(frontier, key=lambda x: -x["risk_reduction"]):
            print(
                f"    {r['config']:<38}{r['risk_reduction']:>8.1%} risk cut, "
                f"{r['auc_pr_retained']:>6.1%} utility kept"
            )

        dominated_leaky = [
            r["config"]
            for _, r in candidates.iterrows()
            if r["derived_mode"] == "leaky"
            and r["config"] not in [f["config"] for f in frontier]
        ]
        if dominated_leaky:
            print(
                f"    dominated, publishing derived features from originals: "
                f"{len(dominated_leaky)} configuration(s)"
            )

    out = pd.concat(all_rows, ignore_index=True)
    lead = ["stratum", "config"]
    out = out[lead + [c for c in out.columns if c not in lead]]
    out.to_csv(RESULTS / "risk_utility_frontier.csv", index=False)
    print(f"\nwrote risk_utility_frontier.csv to {RESULTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
