"""The three corpora the study runs on, and their column specifications.

Every notebook from Stage 4 onward loads its data here rather than carrying its
own copy, so a change to a stratum takes effect everywhere at once.

The three are deliberately never pooled. Grading scales, grade-point
definitions and sequence lengths all differ, so a merged corpus would confound
every comparison drawn from it. They are run as replications instead, and the
spread of group sizes across them, from 5 to 423, is what makes the group-size
question answerable at all.

Two of the three are the study population. The third is not, and the difference
matters enough to state before the code.

    Study population        allied_health and nursing. Ghanaian health
                            professions education records. Every claim about
                            health data, about Ghanaian institutions and about
                            Act 843 rests on these two and only these two.

    Replication corpus      public. Portuguese secondary school students in a
                            language course. Not health, not professional, not
                            tertiary. It is present to test whether the
                            mechanism reproduces outside the study population,
                            and for no other purpose.

An earlier decision in this project rejected the public corpus as a comparator
for licensure prediction, on the grounds that Portuguese secondary students do
not clear the "genuinely comparable" bar for a health professions topic
(docs/TODO.md). That decision stands and is not reversed here, because it
addressed a different use. Transfer learning for prediction needs comparable
populations; a leak in an order statistic does not. gpa_min is the minimum of
the sequence whether the student sits in Kumasi or Coimbra, so the mechanism
replicates or fails to replicate independently of who the students are.

The replication corpus is therefore never merged with the study population,
never used to fit anything applied to it, and never cited in support of a claim
about health records. Reported findings that mention it say which corpus they
came from.

    allied_health   Accra School of Hygiene, EH / OHS / OT, cohorts 2021-2022.
                    110 students, credit-weighted CGPA, grades A to E, six
                    semesters of a three-year programme.

    nursing         BSc Nursing, academic years 2023/24 and 2024/25, levels 200
                    to 400. 566 students, unweighted GPA, grades A to E with
                    plus modifiers, six sequence positions of which any one
                    student has at most four.

    public          UCI Student Performance, Portuguese language course
                    (Cortez & Silva, 2008). 649 students, three period grades
                    on a 0 to 20 scale.

The public corpus earns its place twice. It is an independent replication on
data nobody in this project assembled, and it is the only corpus here carrying
demographic quasi-identifiers. The Ghanaian colleges released no demographics at
all, so questions about age or address sitting beside grades can only be asked
of this one.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

# The bottom 40% of each corpus. Defined as a quantile rather than an absolute
# grade because the three scales are not comparable: a cut-off that marks the
# weakest 40% in one corpus marks the weakest 10% in another, and both
# l-diversity and t-closeness are prevalence-sensitive.
WEAK_QUANTILE = 0.40
SENSITIVE = "weak_final"

# Band widths are expressed as a fraction of each corpus's grade range, not as
# absolute values. A band of 0.5 covers an eighth of a four-point GPA scale but
# only a fortieth of a twenty-point scale, where it does not even merge adjacent
# integers. Fixing the absolute width would have made generalisation look
# ineffective on the public corpus when in fact it was never applied.
BAND_FRACTIONS = (0.0625, 0.125, 0.25)


@dataclass
class Stratum:
    """One corpus plus the column roles the analysis needs."""

    name: str
    frame: pd.DataFrame
    sequence: list
    structural: list
    grade_counts: list
    group: str
    scale: str
    scale_max: float
    demographics: list = field(default_factory=list)
    # Sequence positions the utility model may use. The last observed position
    # is excluded because the sensitive attribute derives from it, and so are
    # CGPA and the grade counts, which are computed over the whole sequence
    # including it. Nursing stops at two because coverage thins after that.
    predictors: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.frame)

    @property
    def band_widths(self):
        """Generalisation widths equivalent across corpora of different scales."""
        return [round(f * self.scale_max, 4) for f in BAND_FRACTIONS]

    def quasi_identifiers(self):
        """Everything an adversary could match on, for the full-set profile."""
        extra = [c for c in ("cgpa", "n_courses") if c in self.frame.columns]
        return (
            self.structural
            + self.demographics
            + extra
            + self.sequence
            + self.grade_counts
        )


def _weak_final(frame, sequence):
    """Weak performance in the last observed position of the sequence.

    Forward-filling across the sequence then taking the final column picks each
    student's own last observed value, which matters for nursing where intakes
    are observed at different levels and no fixed column is the last one.

    The threshold is the cut nearest the target prevalence rather than the
    quantile itself. Integer grades tie heavily: in the public corpus the 40th
    percentile of G3 is 11, and 104 students score exactly 11, so cutting below
    it gives 30.4% while cutting at or below it gives 46.4%. Neither is 40% and
    no threshold is, so the honest move is to pick the closest and report the
    prevalence actually achieved rather than the one intended.
    """
    last = frame[sequence].ffill(axis=1).iloc[:, -1]
    candidates = sorted(last.dropna().unique())
    best, best_gap = None, None
    for cut in candidates:
        for indicator in ((last < cut), (last <= cut)):
            gap = abs(indicator.mean() - WEAK_QUANTILE)
            if best_gap is None or gap < best_gap:
                best, best_gap = indicator, gap
    return best.astype(int)


def load_allied_health():
    d = pd.read_excel(ROOT / "data" / "model_dataset_2021_2022.xlsx")
    sequence = [f"gpa_sem{i}" for i in range(1, 7)]
    d["group"] = d["programme"].astype(str) + "-" + d["cohort_year"].astype(str)
    d[SENSITIVE] = _weak_final(d, sequence)
    return Stratum(
        name="allied_health",
        frame=d,
        sequence=sequence,
        structural=["programme", "cohort_year"],
        grade_counts=[f"n_grade_{g}" for g in "ABCDE"],
        group="group",
        scale="credit-weighted CGPA, grades A to E",
        scale_max=4.0,
        predictors=sequence[:5],
    )


def load_nursing():
    d = pd.read_excel(
        ROOT / "data" / "consolidated_nursing.xlsx", sheet_name="students"
    )
    sequence = [f"gpa_sem{i}" for i in range(1, 7)]
    d["group"] = "NUR-" + d["intake_year"].astype(str)
    d[SENSITIVE] = _weak_final(d, sequence)
    return Stratum(
        name="nursing",
        frame=d,
        sequence=sequence,
        structural=["intake_year"],
        grade_counts=[c for c in d.columns if c.startswith("n_grade_")],
        group="group",
        scale="unweighted mean grade point, grades A to E with plus modifiers",
        scale_max=4.0,
        predictors=sequence[:2],
    )


def load_public():
    """UCI Student Performance, Portuguese course (Cortez & Silva, 2008).

    G1 to G3 are the three period grades on a 0 to 20 scale. There are no grade
    count columns, because the corpus records period grades rather than a
    per-course transcript.
    """
    d = pd.read_csv(ROOT / "data" / "public" / "student-por.csv", sep=";")
    sequence = ["G1", "G2", "G3"]
    d["group"] = d["school"].astype(str)
    d[SENSITIVE] = _weak_final(d, sequence)
    return Stratum(
        name="public",
        frame=d,
        sequence=sequence,
        structural=["school"],
        grade_counts=[],
        group="group",
        scale="period grades on a 0 to 20 scale",
        scale_max=20.0,
        demographics=["sex", "age", "address", "famsize"],
        predictors=sequence[:2],
    )


LOADERS = {
    "allied_health": load_allied_health,
    "nursing": load_nursing,
    "public": load_public,
}


def load_all(names=None):
    """Every stratum, or the named subset, in a stable order."""
    names = names or list(LOADERS)
    return {name: LOADERS[name]() for name in names}
