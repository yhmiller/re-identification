"""Stage 4. Baseline disclosure risk and risk attribution, both strata.

Phases 1 and 2 of the re-identification study. Measures how many students can
be singled out of the Ghanaian health professions records, and which attributes
are responsible.

Two strata, deliberately kept separate rather than pooled:

    allied_health   Accra School of Hygiene, EH / OHS / OT, cohorts 2021-2022,
                    110 students, credit-weighted CGPA, grades A-E.
    nursing         BSc Nursing, academic years 2023/24 and 2024/25, levels
                    200-400, 566 students, unweighted GPA, grades A to E with
                    plus modifiers.

They are not pooled because the grading scales, the GPA definitions and the
cohort structures differ. Comparing them is more informative than merging them:
the strata bracket a wide range of group sizes, from 5 to 238, which is what
makes the group-size question answerable at all.

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

RESULTS = ROOT / "results" / "disclosure"
RESULTS.mkdir(parents=True, exist_ok=True)

# Both extracts are complete populations of their cohorts, so sample uniqueness
# equals population uniqueness and prosecutor and journalist risk coincide.
SAMPLING_FRACTION = 1.0

SEMESTERS = [f"gpa_sem{i}" for i in range(1, 7)]

# Weak performance in the final observed semester. Chosen over failure counts,
# which are too rare to model, and over CGPA thresholds, which are circular
# because CGPA is itself a quasi-identifier.
SENSITIVE = "weak_final"

# Defined as the bottom 40% of each stratum rather than an absolute GPA cut-off.
# The strata grade differently: median final-semester GPA is 2.667 in allied
# health against 2.958 in nursing, so one absolute threshold means two different
# things. A shared cut-off of 2.5 puts allied health at 41.8% and nursing at
# 10.2%, and l-diversity and t-closeness are both prevalence-sensitive, so that
# gap alone would drive the cross-stratum comparison. Fixing the quantile fixes
# prevalence by construction and leaves the comparison to the structure of the
# data. 40% also sits inside the 35-65% band PROJECT_GUIDE treats as natural,
# so no resampling applies.
WEAK_QUANTILE = 0.40


def _weak_final(series):
    return (series < series.quantile(WEAK_QUANTILE)).astype(int)


def load_allied_health():
    d = pd.read_excel(ROOT / "data" / "model_dataset_2021_2022.xlsx")
    d["group"] = d["programme"].astype(str) + "-" + d["cohort_year"].astype(str)
    d[SENSITIVE] = _weak_final(d["gpa_sem6"])
    grade_counts = [f"n_grade_{g}" for g in "ABCDE"]
    return d, grade_counts, ["programme", "cohort_year"]


def load_nursing():
    d = pd.read_excel(
        ROOT / "data" / "consolidated_nursing.xlsx", sheet_name="students"
    )
    d["group"] = "NUR-" + d["intake_year"].astype(str)
    # Intakes are observed at different levels, so each student's final position
    # differs. The last non-null semester is the right anchor, not a fixed column.
    d[SENSITIVE] = _weak_final(d[SEMESTERS].ffill(axis=1).iloc[:, -1])
    grade_counts = [c for c in d.columns if c.startswith("n_grade_")]
    return d, grade_counts, ["intake_year"]


STRATA = {"allied_health": load_allied_health, "nursing": load_nursing}


def scenarios_for(structural, grade_counts):
    """Escalating adversary knowledge, from structural facts to full detail."""
    return {
        "structural only": structural,
        "+ cgpa": structural + ["cgpa"],
        "+ one semester GPA": structural + ["gpa_sem1"],
        "+ two semester GPAs": structural + ["gpa_sem1", "gpa_sem2"],
        "+ full semester sequence": structural + SEMESTERS,
        "+ grade counts": structural + grade_counts,
        "full candidate QI set": structural
        + ["cgpa", "n_courses"]
        + SEMESTERS
        + grade_counts,
        "cgpa alone": ["cgpa"],
        "semester sequence alone": SEMESTERS,
    }


def run_stratum(name, loader):
    d, grade_counts, structural = loader()
    print(
        f"\n{'=' * 72}\n{name}: {len(d)} students, "
        f"'{SENSITIVE}' prevalence {d[SENSITIVE].mean():.1%}\n{'=' * 72}"
    )

    scenarios = scenarios_for(structural, grade_counts)
    profiles = dr.profile_table(
        d, scenarios, sensitive=SENSITIVE, sampling_fraction=SAMPLING_FRACTION
    )
    profiles.insert(0, "stratum", name)

    print("PHASE 1  risk by adversary knowledge")
    print(
        f"  {'scenario':<28}{'unique':>9}{'min k':>7}{'classes':>9}{'marketer':>10}{'l':>4}"
    )
    for _, r in profiles.iterrows():
        print(
            f"  {r['label']:<28}{r['prop_unique']:>8.1%}{r['min_class_size']:>7}"
            f"{r['n_equivalence_classes']:>9}{r['marketer_risk']:>10.3f}"
            f"{r['min_l_diversity']:>4.0f}"
        )

    full_qi = scenarios["full candidate QI set"]

    solo = dr.solo_identifying_power(d, full_qi)
    solo.insert(0, "stratum", name)
    print("\nPHASE 2  identifying power of each attribute alone (top 6)")
    for _, r in solo.head(6).iterrows():
        print(
            f"  {r['attribute']:<20} distinct {r['n_distinct_values']:>4}   "
            f"unique alone {r['prop_unique_alone']:>6.1%}"
        )

    curve = dr.saturation_curve(d, full_qi)
    curve.insert(0, "stratum", name)
    print("\nPHASE 2  uniqueness as attributes accumulate (greedy)")
    for _, r in curve.iterrows():
        print(
            f"  {r['n_attributes']} attribute(s): +{r['added']:<16} "
            f"unique {r['prop_unique']:>6.1%}"
        )

    sweep = dr.precision_sweep(
        d, structural + SEMESTERS, sampling_fraction=SAMPLING_FRACTION
    )
    sweep.insert(0, "stratum", name)
    print("\nPHASE 1  rounding the semester sequence")
    for _, r in sweep.iterrows():
        print(
            f"  {r['label']:<14} unique {r['prop_unique']:>6.1%}   "
            f"min k {r['min_class_size']:>3}"
        )

    effect = dr.group_size_effect(d, ["group", "cgpa"], ["group"], precision=1)
    effect.insert(0, "stratum", name)
    print("\nPHASE 2  within-group uniqueness on cgpa at 1dp, by group size")
    for _, r in effect.iterrows():
        print(
            f"  {str(r['group']):<12} n={int(r['group_size']):>4}  "
            f"unique {r['prop_unique_within_group']:>6.1%}  "
            f"min k {int(r['min_class_size'])}"
        )

    return profiles, solo, curve, sweep, effect


def main():
    collected = {k: [] for k in ("profiles", "solo", "curve", "sweep", "effect")}

    for name, loader in STRATA.items():
        for key, frame in zip(collected, run_stratum(name, loader)):
            collected[key].append(frame)

    names = {
        "profiles": "baseline_risk_profiles",
        "solo": "solo_identifying_power",
        "curve": "saturation_curve",
        "sweep": "precision_sweep",
        "effect": "group_size_effect",
    }
    for key, frames in collected.items():
        pd.concat(frames, ignore_index=True).to_csv(
            RESULTS / f"{names[key]}.csv", index=False
        )

    # The headline cross-stratum comparison: does a bigger cohort protect anyone?
    effect = pd.concat(collected["effect"], ignore_index=True)
    print(
        f"\n{'=' * 72}\nCROSS-STRATUM  group size against within-group uniqueness\n{'=' * 72}"
    )
    corr = effect["group_size"].corr(
        effect["prop_unique_within_group"], method="spearman"
    )
    print(
        f"  groups compared: {len(effect)}  "
        f"size range: {int(effect['group_size'].min())} to {int(effect['group_size'].max())}"
    )
    print(f"  Spearman correlation: {corr:+.3f}")

    print(f"\nwrote {len(names)} tables to {RESULTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
