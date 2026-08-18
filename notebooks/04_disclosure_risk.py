"""Stage 4. Baseline disclosure risk and risk attribution.

Phases 1 and 2 of the re-identification study, run across all three corpora.
Measures how many students can be singled out, and which attributes are
responsible.

Two corpora are the study population, Ghanaian health professions education
records. The third is a cross-domain replication corpus and is labelled as such
in every table. See `strata.py` for why the distinction is kept explicit and why
the three are never pooled.

Run:
    ml_env/bin/python notebooks/04_disclosure_risk.py

Writes to results/disclosure/. Every output is aggregate. No record-level result
is produced or saved, so nothing here can identify a student.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import disclosure_risk as dr  # noqa: E402
import strata as st  # noqa: E402

RESULTS = ROOT / "results" / "disclosure"
RESULTS.mkdir(parents=True, exist_ok=True)

# Every corpus is a complete population of its cohorts rather than a sample, so
# sample uniqueness equals population uniqueness and prosecutor and journalist
# risk coincide.
SAMPLING_FRACTION = 1.0

ROLES = {"allied_health": "study population",
         "nursing": "study population",
         "public": "replication corpus"}


def scenarios_for(s):
    """Escalating adversary knowledge, from structural facts to full detail."""
    seq = s.sequence
    scenarios = {
        "structural only": s.structural,
        "+ one grade": s.structural + seq[:1],
        "+ two grades": s.structural + seq[:2],
        "+ full sequence": s.structural + seq,
        "sequence alone": seq,
        "full candidate QI set": s.quasi_identifiers(),
    }
    if "cgpa" in s.frame.columns:
        scenarios["+ cgpa"] = s.structural + ["cgpa"]
        scenarios["cgpa alone"] = ["cgpa"]
    if s.grade_counts:
        scenarios["+ grade counts"] = s.structural + s.grade_counts
    # Only the replication corpus carries demographics. The Ghanaian colleges
    # released none, so this question can be asked of one corpus only.
    if s.demographics:
        scenarios["demographics only"] = s.structural + s.demographics
        scenarios["demographics + sequence"] = s.structural + s.demographics + seq
    return scenarios


def run_stratum(s):
    print(f"\n{'=' * 78}\n{s.name}  ({ROLES[s.name]}): {s.n} students, "
          f"{s.scale}\n  '{st.SENSITIVE}' prevalence "
          f"{s.frame[st.SENSITIVE].mean():.1%}\n{'=' * 78}")

    profiles = dr.profile_table(s.frame, scenarios_for(s), sensitive=st.SENSITIVE,
                                sampling_fraction=SAMPLING_FRACTION)
    profiles.insert(0, "role", ROLES[s.name])
    profiles.insert(0, "stratum", s.name)

    print("PHASE 1  risk by adversary knowledge")
    print(f"  {'scenario':<26}{'unique':>9}{'min k':>7}{'classes':>9}{'marketer':>10}{'l':>4}")
    for _, r in profiles.iterrows():
        print(f"  {r['label']:<26}{r['prop_unique']:>8.1%}{r['min_class_size']:>7}"
              f"{r['n_equivalence_classes']:>9}{r['marketer_risk']:>10.3f}"
              f"{r['min_l_diversity']:>4.0f}")

    full_qi = s.quasi_identifiers()

    solo = dr.solo_identifying_power(s.frame, full_qi)
    solo.insert(0, "stratum", s.name)
    print("\nPHASE 2  identifying power of each attribute alone (top 5)")
    for _, r in solo.head(5).iterrows():
        print(f"  {r['attribute']:<20} distinct {r['n_distinct_values']:>4}   "
              f"unique alone {r['prop_unique_alone']:>6.1%}")

    curve = dr.saturation_curve(s.frame, full_qi)
    curve.insert(0, "stratum", s.name)
    print("\nPHASE 2  uniqueness as attributes accumulate (greedy)")
    for _, r in curve.iterrows():
        print(f"  {r['n_attributes']} attribute(s): +{r['added']:<16} "
              f"unique {r['prop_unique']:>6.1%}")

    sweep = dr.precision_sweep(s.frame, s.structural + s.sequence,
                               sampling_fraction=SAMPLING_FRACTION)
    sweep.insert(0, "stratum", s.name)
    print("\nPHASE 1  rounding the grade sequence")
    for _, r in sweep.iterrows():
        print(f"  {r['label']:<14} unique {r['prop_unique']:>6.1%}   "
              f"min k {r['min_class_size']:>3}")

    anchor = "cgpa" if "cgpa" in s.frame.columns else s.sequence[-1]
    effect = dr.group_size_effect(s.frame, [s.group, anchor], [s.group], precision=1)
    effect.insert(0, "anchor", anchor)
    effect.insert(0, "stratum", s.name)
    print(f"\nPHASE 2  within-group uniqueness on {anchor} at 1dp, by group size")
    for _, r in effect.iterrows():
        print(f"  {str(r[s.group]):<12} n={int(r['group_size']):>4}  "
              f"unique {r['prop_unique_within_group']:>6.1%}  "
              f"min k {int(r['min_class_size'])}")

    return profiles, solo, curve, sweep, effect


def main():
    collected = {k: [] for k in ("profiles", "solo", "curve", "sweep", "effect")}

    for s in st.load_all().values():
        for key, frame in zip(collected, run_stratum(s)):
            collected[key].append(frame)

    names = {"profiles": "baseline_risk_profiles", "solo": "solo_identifying_power",
             "curve": "saturation_curve", "sweep": "precision_sweep",
             "effect": "group_size_effect"}
    for key, frames in collected.items():
        pd.concat(frames, ignore_index=True).to_csv(
            RESULTS / f"{names[key]}.csv", index=False)

    # Does a bigger cohort protect anyone? Pooled across corpora because the
    # relationship is between group size and uniqueness, which is scale-free,
    # not between grades that differ from corpus to corpus.
    effect = pd.concat(collected["effect"], ignore_index=True)
    corr = effect["group_size"].corr(effect["prop_unique_within_group"], method="spearman")
    print(f"\n{'=' * 78}\nCROSS-CORPUS  group size against within-group uniqueness\n{'=' * 78}")
    print(f"  groups: {len(effect)}   size range: {int(effect['group_size'].min())} "
          f"to {int(effect['group_size'].max())}   Spearman: {corr:+.3f}")

    study_only = effect[effect["stratum"] != "public"]
    corr_study = study_only["group_size"].corr(
        study_only["prop_unique_within_group"], method="spearman")
    print(f"  study population only: {len(study_only)} groups, Spearman {corr_study:+.3f}")

    print(f"\nwrote {len(names)} tables to {RESULTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
