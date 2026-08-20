"""Release Risk Advisor: what a custodian is exposing, and what they are keeping.

    ml_env/bin/streamlit run app/app.py

Who this is for
---------------
A registrar, data officer or programme lead who is about to share a file of
student academic records and wants to know what that release discloses.

What it deliberately does not do
--------------------------------
It never reports on an individual student. No per-record table, no ranked list,
no "here is why this student was flagged". That restriction is not squeamishness,
it is the thesis's own argument turned on its author: this study exists to show
that per-student output derived from academic records is re-identifying. A tool
that displayed it would demonstrate the harm the study documents.

The writing plan states the rule directly:

    "Aggregate by construction. Reports over a dataset, never over a record. No
    record-level output rendered, exported or logged."

and warns against the shape of the earlier prototype:

    "Its upload-and-explain-each-student flow is the opposite of what this study
    argues and must not be carried across."

Every number this app shows is a property of a dataset. `release_report.build`
enforces that at the boundary, and a test asserts the return type carries nothing
per-record.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP_DIR))

import data_sources as ds  # noqa: E402
import derivation_consistent as dc  # noqa: E402
import release_report as rr  # noqa: E402
import strata as strata_module  # noqa: E402

TITLE = "Release Risk Advisor"
SUBTITLE = "What this release discloses, and what analytical value it keeps"

RESEARCHER = "Prince Bortey Miller"
RESEARCHER_EMAIL = "prince@princemiller.com"
RESEARCHER_SITE = "https://www.princemiller.com"
SUPERVISOR = "Dr. Eric Opoku Osei"
INSTITUTION = "Kwame Nkrumah University of Science and Technology"
DEPARTMENT = "Department of Computer Science"
PROGRAMME = "MSc Health Informatics, 2026"

THESIS_TITLE = (
    "Derivation-Consistent De-identification of Health Professions Education "
    "Records: Disclosure Risk and Analytical Utility Under Hybrid Public and "
    "Institutional Academic Data"
)

ABOUT_MD = f"""### {TITLE}

Decision support for releasing a dataset of student academic records. It reports
what a release discloses and what analytical value it keeps, **for the file as a
whole and never for an individual student**.

Built from the MSc thesis:

> *{THESIS_TITLE}*

The thesis shows that removing names and index numbers does not anonymise an
academic transcript, that blurring grades is undone by publishing summary
features computed from the unblurred values, and that computing those features
from the protected values instead holds the protection at a measured cost in
analytical utility. This tool prices that trade-off for a dataset you are about
to share.

---

**Researcher**
{RESEARCHER}
[{RESEARCHER_EMAIL}](mailto:{RESEARCHER_EMAIL}) | [princemiller.com]({RESEARCHER_SITE})

**Supervisor**
{SUPERVISOR}

**Institution**
{INSTITUTION}
{DEPARTMENT}
{PROGRAMME}

---

Decision support only. The figures are measured properties of a dataset under
the attacks tested, not guarantees.
"""

VERDICT_STYLE = {
    rr.VERDICT_RELEASE: ("✅", "success"),
    rr.VERDICT_REVIEW: ("⚠️", "warning"),
    rr.VERDICT_HOLD: ("⛔", "error"),
}


def _header():
    st.set_page_config(page_title=TITLE, page_icon="🔐", layout="wide")
    st.title(TITLE)
    st.caption(SUBTITLE)
    with st.sidebar:
        st.markdown(f"### {TITLE}")
        st.markdown(
            "Decision support for **releasing a dataset**. It reports on the "
            "file as a whole and never on an individual record."
        )
        st.divider()
        st.markdown(
            f"**{RESEARCHER}**  \n"
            f"[{RESEARCHER_EMAIL}](mailto:{RESEARCHER_EMAIL})  \n"
            f"{PROGRAMME}  \n"
            f"Supervisor: {SUPERVISOR}"
        )
        with st.expander("About this tool"):
            st.markdown(ABOUT_MD)
        st.caption(f"Made by {RESEARCHER} | {INSTITUTION}")


def _no_individual_output_notice():
    st.info(
        "**This tool reports on datasets, never on individuals.** It shows no "
        "per-student rows, no ranked lists and no individual explanations. "
        "That is the point: this study exists to show that per-student output "
        "derived from academic records can identify people.",
        icon="🔐",
    )


# ---------------------------------------------------------------------------
# Tab 1: assess a release configuration
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _cached_report(
    corpus,
    band_width,
    derived_mode,
    suppress_k,
    risk_threshold,
    utility_tolerance,
    with_utility,
):
    stratum = strata_module.LOADERS[corpus]()
    report = rr.build(
        stratum.frame,
        stratum.sequence,
        stratum.predictors,
        strata_module.SENSITIVE,
        corpus=corpus,
        band_width=band_width,
        suppress_k=suppress_k or None,
        derived_mode=derived_mode,
        risk_threshold=risk_threshold,
        utility_tolerance=utility_tolerance,
        with_utility=with_utility,
    )
    # Cache a plain payload, not the dataclass, so Streamlit can hash it.
    return report.as_row(), report.drivers, rr.plain_english(report)


def _assess_tab(corpora):
    if not corpora:
        st.warning("No corpus on this machine can be assessed.")
        return

    st.subheader("1. Choose what you are releasing")
    left, right = st.columns([2, 3])

    with left:
        name = st.selectbox(
            "Dataset",
            list(corpora),
            format_func=lambda k: corpora[k]["label"],
        )
        stratum = corpora[name]["stratum"]
        if corpora[name]["restricted"]:
            st.caption(
                "Restricted corpus. Assess it only inside the approved " "environment."
            )

        widths = stratum.band_widths
        band = st.select_slider(
            "Generalisation band width",
            options=widths,
            value=widths[1],
            help="How coarsely each grade is blurred. Wider bands protect more "
            "and preserve less detail.",
        )
        mode = st.radio(
            "Derived features are computed from",
            [dc.PROPOSED, dc.BASELINE, dc.NONE],
            format_func={
                dc.PROPOSED: "the protected values  (derivation-consistent)",
                dc.BASELINE: "the original values  (what pipelines do today)",
                dc.NONE: "not published at all",
            }.get,
            help="This is the single choice this study is about.",
        )
        suppress = st.number_input(
            "Suppress groups smaller than k",
            min_value=0,
            max_value=20,
            value=0,
            help="0 disables suppression.",
        )

    with right:
        st.markdown("**Your tolerances**")
        st.caption(
            "Illustrative defaults, not standards. A custodian brings their own."
        )
        # Sliders work in whole percent and convert, because Streamlit applies
        # the format string to the raw value: a 0.05 proportion under "%.0f%%"
        # renders as "0%", which is unreadable.
        risk_threshold = (
            st.slider(
                "Most records that may be uniquely identifiable",
                0,
                50,
                int(rr.DEFAULT_RISK_THRESHOLD * 100),
                1,
                format="%d%%",
            )
            / 100
        )
        utility_tolerance = (
            st.slider(
                "Most analytical value you are willing to lose",
                0,
                50,
                int(rr.DEFAULT_UTILITY_TOLERANCE * 100),
                1,
                format="%d%%",
            )
            / 100
        )
        with_utility = st.checkbox(
            "Measure analytical value (slower, fits 50 models)", value=True
        )

    if not st.button("Assess this release", type="primary"):
        return

    with st.spinner("Measuring risk and utility..."):
        row, drivers, sentences = _cached_report(
            name,
            band,
            mode,
            int(suppress),
            risk_threshold,
            utility_tolerance,
            with_utility,
        )

    st.divider()
    st.subheader("2. What this release discloses")

    icon, kind = VERDICT_STYLE[row["verdict"]]
    getattr(st, kind)(f"{icon}  **{row['verdict'].upper()}**")
    for reason in row["reasons"].split("; "):
        st.markdown(f"- {reason}")

    a, b, c, d = st.columns(4)
    a.metric(
        "Uniquely identifiable",
        f"{row['prop_unique']:.1%}",
        help="Share of records that no other record matches.",
    )
    b.metric(
        "Smallest look-alike group",
        f"{row['min_class_size']}",
        help="How many records the most exposed person hides among. " "1 means alone.",
    )
    c.metric(
        "Protection given back by the leak",
        f"{row['leakage']:.0%}",
        help="How much of the generalisation's protection publishing "
        "derived features from original values would restore.",
    )
    if not pd.isna(row["auc_pr_retained"]):
        d.metric(
            "Analytical value kept",
            f"{row['auc_pr_retained']:.0%}",
            help="How much of the unprotected analysis survives.",
        )

    st.divider()
    st.subheader("3. In plain English")
    for line in sentences:
        st.markdown(f"- {line}")

    st.divider()
    st.subheader("4. What is driving the risk")
    st.caption(
        "One row per **column**, not per person. Read it as: if someone knew "
        "only this column about a student, how often would that alone single "
        "them out."
    )
    top = drivers.head(10).copy()
    top["prop_unique_alone"] = top["prop_unique_alone"].map("{:.1%}".format)
    top = top.rename(
        columns={
            "attribute": "Column",
            "n_distinct_values": "Distinct values",
            "prop_unique_alone": "Singles out, alone",
            "min_class_size_alone": "Smallest group",
        }
    )
    st.dataframe(top, use_container_width=True, hide_index=True)

    st.caption(
        "**What this is not.** These are measured properties of this dataset "
        "under the attacks tested, not a guarantee. An attacker with "
        "information this study did not simulate could do better."
    )


# ---------------------------------------------------------------------------
# Tab 2: the study's own measured results
# ---------------------------------------------------------------------------


def _findings_tab():
    if not ds.results_are_available():
        st.warning("No results on disk. Run `./run.sh` first.")
        return

    tables, stamp = ds.study_results()
    st.caption(
        f"Read from `results/disclosure/`, last rebuilt {stamp:%d %b %Y %H:%M}. "
        "Re-run `./run.sh` to refresh."
    )

    st.subheader("Baseline against engineered")
    risk, utility = tables["risk"], tables["utility"]
    merged = risk.merge(utility, on=["stratum", "band_width"])
    view = pd.DataFrame(
        {
            "Corpus": merged.stratum,
            "Band": merged.band_width,
            "Unique, baseline": merged.prop_unique_baseline.map("{:.1%}".format),
            "Unique, engineered": merged.prop_unique_proposed.map("{:.1%}".format),
            "AUC-PR, baseline": merged.auc_pr_baseline.round(3),
            "AUC-PR, engineered": merged.auc_pr_proposed.round(3),
            "Utility change": merged.mean_difference.round(3),
            "Real difference?": merged.corrected_p.map(
                lambda p: "yes" if p < 0.05 else "not detectable"
            ),
        }
    )
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.caption(
        "The engineered arm cuts risk in "
        f"{int(risk.excludes_zero.sum())} of {len(risk)} configurations and "
        f"costs measurable analytical value in "
        f"{int((utility.corrected_p < 0.05).sum())} of {len(utility)}. "
        "That is the trade-off, not a free fix."
    )

    st.subheader("Cohort size protects")
    groups = tables["group_size"].sort_values("group_size")
    st.caption(
        "Each point is a programme cohort. Smaller cohorts are more exposed, "
        "and no band width tested protects the smallest of them."
    )
    st.scatter_chart(
        groups.rename(
            columns={
                "group_size": "Students in cohort",
                "prop_unique_within_group": "Share unique",
            }
        ),
        x="Students in cohort",
        y="Share unique",
    )

    st.subheader("What a realistic attacker achieves")
    linkage = tables["linkage"]
    worst = linkage.groupby("stratum").prop_correctly_isolated.max() * 100
    st.caption(
        "Uniqueness is a worst case. With imprecise recall, the best any "
        "simulated attacker managed was: "
        + ", ".join(f"{s} {v:.1f}%" for s, v in worst.items())
        + "."
    )


# ---------------------------------------------------------------------------
# Tab 3: how to read any of this
# ---------------------------------------------------------------------------


def _guide_tab():
    st.markdown("""
### What the numbers mean

**Uniquely identifiable.** The share of records that no other record matches on
the released columns. If a record is unique, anyone who knows those values can
find it. Lower is better.

**Smallest look-alike group.** How many records the most exposed person is
hidden among. 1 means they are alone. This is the *k* in k-anonymity.

**Protection given back by the leak.** Generalisation lowers risk. Publishing
derived features computed from the original values raises it again. This is how
much of the reduction that puts back. High numbers mean the generalisation is
being undone.

**Analytical value kept.** How much of the analysis survives, as a share of the
unprotected file. Measured by fitting a model to what a recipient actually
receives and comparing performance. Note that the floor is not zero: on this
task, guessing scores around 0.43 to 0.46, so a score of 0.79 is well above
chance.

### The one decision this tool is about

A file blurs its grades to protect students, then publishes summary columns,
the minimum, maximum, average and trend. The question is what those summaries
are calculated from.

- **From the original values.** They are exact statistics of the numbers you
  just blurred, so they constrain those numbers back. The blurring is
  substantially undone.
- **From the protected values.** They can be recalculated by the recipient from
  the columns already published, so they add nothing. The blurring holds.

The second option costs analytical value. That is the trade-off this tool prices.

### Limits you should hold onto

- These are **measured properties under the attacks tested**, not guarantees.
- **Uniqueness is a worst case**, not a probability of re-identification. A
  realistic attacker with imprecise knowledge does considerably worse.
- The thresholds are **illustrative defaults**, not standards. Your institution
  or ethics committee sets the real ones.
- Small cohorts **cannot be protected by generalisation at any width tested**.
  For those, this tool will keep saying no, and that is the correct answer.
""")


def main():
    _header()
    _no_individual_output_notice()
    assess, findings, guide = st.tabs(
        ["Assess a release", "Study findings", "How to read this"]
    )
    with assess:
        _assess_tab(ds.assessable_corpora())
    with findings:
        _findings_tab()
    with guide:
        _guide_tab()


if __name__ == "__main__":
    main()
