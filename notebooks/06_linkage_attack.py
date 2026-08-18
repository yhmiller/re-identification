"""Stage 6. Linkage under simulated side knowledge.

Phase 4 of the re-identification study. Experiment B gave the attacker only the
released file. This gives them what a classmate, lecturer or employer plausibly
holds: they know who they are looking for, they know the group, and they recall
one or two grades imprecisely.

Each scenario is run against three release configurations, so the question is
not only "can a target be isolated" but "does de-identification stop it":

    raw                 the file as the pipeline currently produces it
    banded              semester GPAs generalised to width 0.5
    banded + derived    the same, with derived features published alongside,
                        which is the configuration Experiment B showed to be
                        self-defeating

Run:
    ml_env/bin/python notebooks/06_linkage_attack.py

Ethical guardrail: side knowledge is simulated from within the dataset. No
attempt is made to identify any real, named individual, and every figure below
is an aggregate over all targets.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import attack_models as am  # noqa: E402
import deidentify as di  # noqa: E402
import disclosure_risk as dr  # noqa: E402

RESULTS = ROOT / "results" / "disclosure"
RESULTS.mkdir(parents=True, exist_ok=True)

SEMESTERS = [f"gpa_sem{i}" for i in range(1, 7)]
BAND_WIDTH = 0.5

# What an attacker plausibly recalls. A grade remembered "to within half a
# point" is generous to the target; "to within a quarter" is the sharper
# recollection of someone who saw a transcript rather than heard a number.
LOOSE, SHARP = 0.5, 0.25


def load_strata():
    allied = pd.read_excel(ROOT / "data" / "model_dataset_2021_2022.xlsx")
    allied["group"] = (
        allied["programme"].astype(str) + "-" + allied["cohort_year"].astype(str)
    )

    nursing = pd.read_excel(
        ROOT / "data" / "consolidated_nursing.xlsx", sheet_name="students"
    )
    nursing["group"] = "NUR-" + nursing["intake_year"].astype(str)

    return {"allied_health": allied, "nursing": nursing}


def releases(frame):
    """The three configurations a custodian might hand out.

    Each carries the band width it was generalised at, because the attacker
    matches a recalled interval against a published band rather than against a
    point. A raw release has width zero.
    """
    derived = di.derive_features(frame, SEMESTERS)
    banded = di.generalise(frame, SEMESTERS, BAND_WIDTH)
    with_derived = banded.drop(columns=derived.columns, errors="ignore").join(derived)

    # Only the semester columns are generalised. The derived features go out at
    # full precision, which is the configuration Experiment B showed to be
    # self-defeating, so their band width is zero.
    banded_widths = {c: BAND_WIDTH for c in SEMESTERS}

    return {
        "raw": (frame, {}),
        "banded": (banded, banded_widths),
        "banded + derived": (with_derived, banded_widths),
    }


SCENARIOS = [
    ("group only", [("group", None)]),
    ("group + 1 grade, loose recall", [("group", None), ("gpa_sem1", LOOSE)]),
    ("group + 1 grade, sharp recall", [("group", None), ("gpa_sem1", SHARP)]),
    ("group + 2 grades, loose recall",
     [("group", None), ("gpa_sem1", LOOSE), ("gpa_sem2", LOOSE)]),
    ("group + 2 grades, sharp recall",
     [("group", None), ("gpa_sem1", SHARP), ("gpa_sem2", SHARP)]),
    ("group + cgpa to 1dp", [("group", None), ("cgpa", 0.05)]),
    ("2 grades, no group", [("gpa_sem1", SHARP), ("gpa_sem2", SHARP)]),
]


def derived_scenarios(release_name):
    """Only meaningful where the derived features were actually published."""
    if release_name != "banded + derived":
        return []
    return [
        ("group + gpa_trend recalled", [("group", None), ("gpa_trend", SHARP)]),
        ("gpa_min + gpa_max recalled",
         [("gpa_min", SHARP), ("gpa_max", SHARP)]),
    ]


def run_stratum(name, frame):
    print(f"\n{'=' * 78}\n{name}: {len(frame)} students\n{'=' * 78}")
    rows = []

    # The attacker recalls true values, including true derived features, so the
    # truth frame carries both. Recomputed rather than taken from the source so
    # that both strata use one definition.
    truth = frame.drop(columns=di.DERIVED_COLUMNS, errors="ignore").join(
        di.derive_features(frame, SEMESTERS)
    )

    for release_name, (released, width) in releases(frame).items():
        print(f"\n  release: {release_name}")
        print(f"    {'scenario':<32}{'mean set':>10}{'isolated':>10}"
              f"{'correct':>9}{'kept':>7}")

        for label, spec in SCENARIOS + derived_scenarios(release_name):
            if any(c not in released.columns or c not in truth.columns
                   for c, _ in spec):
                continue
            metrics = am.linkage_attack(released, spec, label,
                                        truth=truth, release_width=width)
            metrics.update(stratum=name, release=release_name)
            rows.append(metrics)
            print(f"    {label:<32}{metrics['mean_candidate_set']:>10.1f}"
                  f"{metrics['prop_uniquely_isolated']:>10.1%}"
                  f"{metrics['prop_correctly_isolated']:>9.1%}"
                  f"{metrics['prop_target_in_candidate_set']:>7.0%}")

    # Does absence identify? Which sequence positions a student occupies is a
    # function of intake year, so the pattern of gaps may single people out
    # without any value being read.
    signature = am.missingness_signature(frame, SEMESTERS)
    holder = pd.DataFrame({"missingness": signature, "group": frame["group"]})
    profile = dr.risk_profile(holder, ["missingness"])
    combined = dr.risk_profile(holder, ["missingness", "group"])
    print(f"\n  missingness pattern alone: {profile.prop_unique:.1%} unique, "
          f"{profile.n_equivalence_classes} distinct patterns, "
          f"min k {profile.min_class_size}")
    print(f"  missingness + group:       {combined.prop_unique:.1%} unique, "
          f"min k {combined.min_class_size}")

    rows.append({"stratum": name, "release": "n/a",
                 "attack": "missingness pattern alone",
                 "n_targets": len(frame), "n_released": len(frame),
                 "mean_candidate_set": float("nan"),
                 "median_candidate_set": float("nan"),
                 "prop_uniquely_isolated": profile.prop_unique,
                 "prop_correctly_isolated": profile.prop_unique,
                 "prop_narrowed_below_5": float("nan"),
                 "prop_target_in_candidate_set": 1.0})

    return pd.DataFrame(rows)


def main():
    frames = [run_stratum(name, frame) for name, frame in load_strata().items()]
    out = pd.concat(frames, ignore_index=True)

    lead = ["stratum", "release", "attack"]
    out = out[lead + [c for c in out.columns if c not in lead]]
    out.to_csv(RESULTS / "linkage_attack.csv", index=False)

    # Retention is a result once the release is generalised. An attacker whose
    # imprecise recall has excluded the right person cannot identify them
    # however small the candidate set becomes.
    kept = out.loc[out["release"] == "raw", "prop_target_in_candidate_set"].dropna()
    banded = out.loc[out["release"].str.startswith("banded"),
                     "prop_target_in_candidate_set"].dropna()
    print(f"\n{'=' * 78}")
    print(f"target retained in candidate set: raw {kept.mean():.0%}, "
          f"generalised {banded.mean():.0%} (min {banded.min():.0%})")
    print(f"wrote linkage_attack.csv to {RESULTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
