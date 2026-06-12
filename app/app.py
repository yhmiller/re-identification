"""
app.py — NMC-LE risk screening for nurse educators.

Upload a spreadsheet of student records, get a ranked at-risk list with a
per-student explanation of why the model flagged them.

Launch:  ml_env/bin/streamlit run app/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inference  # noqa: E402 — local module, added to path above
import labels      # noqa: E402 — plain-English names + glossary

DEFAULT_THRESHOLD = 0.50
PROB_PERCENT_FORMAT = "{:.0%}"

# ── Project / authorship metadata (shown in the About modal + bylines) ──
APP_TITLE = "NMC-LE Risk Screening"
RESEARCHER = "Prince Bortey Miller"
RESEARCHER_ID = "22388461"
RESEARCHER_EMAIL = "prince@princemiller.com"
RESEARCHER_SITE = "https://www.princemiller.com"
SUPERVISOR = "Dr. Eric Opoku Osei"
INSTITUTION = "KNUST — Dept. of Computer Science"
PROGRAMME = "MSc Health Informatics, 2025-2026"

ABOUT_MD = f"""### {APP_TITLE}

Explainable machine learning for predicting **Nursing & Midwifery Council
Licensure Examination (NMC-LE)** failure in Ghana — an **XGBoost + SHAP**
approach. It screens nursing trainees for first-attempt failure risk so
educators can target remediation early. Decision support only — not a verdict.

**Researcher:** {RESEARCHER} (ID {RESEARCHER_ID})
[{RESEARCHER_EMAIL}](mailto:{RESEARCHER_EMAIL}) · [www.princemiller.com]({RESEARCHER_SITE})

**Supervisor:** {SUPERVISOR}
**Institution:** {INSTITUTION}
**Programme:** {PROGRAMME}
"""

MENU_ITEMS = {
    "About": ABOUT_MD,
    "Get help": RESEARCHER_SITE,
    "Report a bug": None,
}


HIDE_BRANDING_CSS = """
    <style>
    footer {visibility: hidden; height: 0;}
    [data-testid="stStatusWidget"] {visibility: hidden;}
    </style>
"""


@st.cache_resource
def get_bundle():
    return inference.load_bundle()


def read_upload(uploaded_file):
    """Read an uploaded .csv or .xlsx into a DataFrame."""
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def render_header():
    st.title("NMC-LE Risk Screening")
    st.caption(
        "Identify nursing trainees at risk of failing the licensure exam, "
        "so remediation can start early. Predictions are model estimates, not "
        "verdicts — use them alongside your own judgement."
    )


def render_data_warning(bundle):
    """Show an illustrative-only banner until the model is trained on real data."""
    if bundle.get("data_source", "synthetic") != "real":
        st.warning(
            "oen**Demonstration mode** — this model is trained on **synthetic "
            "pilot data**. The risk scores are illustrative only and must not be "
            "used for real student decisions. The banner disappears automatically "
            "once the model is retrained on the real college dataset.",
            icon="⚠️",
        )


def render_sidebar(bundle):
    st.sidebar.header("Settings")
    threshold = st.sidebar.slider(
        "At-risk threshold (fail probability)",
        min_value=0.05, max_value=0.95, value=DEFAULT_THRESHOLD, step=0.05,
        help="Students at or above this probability are flagged at-risk.",
    )
    st.sidebar.download_button(
        "Download blank template (.csv)",
        inference.blank_template(bundle).to_csv(index=False),
        file_name="student_records_template.csv",
        mime="text/csv",
    )
    with st.sidebar.expander("What columns must my file have?"):
        st.caption("One row per student. Column names must match exactly:")
        for col, meaning in labels.COLUMN_GLOSSARY.items():
            st.markdown(f"- **`{col}`** — {meaning}")
        st.caption("The 6 papers are: medical_surgical, mental_health, "
                   "paediatric, public_health, obstetric, pharmacology.")

    st.sidebar.divider()
    st.sidebar.caption(
        f"**{RESEARCHER}** · {INSTITUTION}  \n"
        f"Supervisor: {SUPERVISOR}  \n"
        f"[www.princemiller.com]({RESEARCHER_SITE})"
    )
    return threshold


def render_summary(scored):
    at_risk = int(scored["at_risk"].sum())
    total = len(scored)
    col1, col2, col3 = st.columns(3)
    col1.metric("Students screened", total)
    col2.metric("Flagged at-risk", at_risk)
    col3.metric("At-risk rate", PROB_PERCENT_FORMAT.format(at_risk / total) if total else "—")
    st.caption(
        '"At-risk" means the model estimates a fail probability at or above your '
        "sidebar threshold — these are the students to prioritise for remediation."
    )


def render_results_table(scored, bundle):
    display = scored.copy()
    display[inference.RISK_PROB_COLUMN] = display[inference.RISK_PROB_COLUMN].map(
        PROB_PERCENT_FORMAT.format
    )
    display["at_risk"] = display["at_risk"].map({True: "Yes", False: "No"})
    shown = [inference.RANK_COLUMN, inference.STUDENT_ID_COLUMN,
             inference.RISK_PROB_COLUMN, "at_risk"] + bundle["categorical_columns"]
    display = display[shown].rename(columns=labels.RESULT_HEADERS)
    st.dataframe(display, use_container_width=True, hide_index=True)

    with st.expander("What do these columns mean?"):
        for col, meaning in labels.METRIC_GLOSSARY.items():
            st.markdown(f"- **{col}** — {meaning}")

    st.download_button(
        "Download full risk report (.csv)",
        scored.to_csv(index=False),
        file_name="nmc_risk_report.csv",
        mime="text/csv",
    )


def _strength_word(value, max_abs):
    """Qualitative size of a factor relative to the strongest one shown."""
    ratio = abs(value) / max_abs if max_abs else 0
    if ratio >= 0.66:
        return "strong"
    if ratio >= 0.33:
        return "moderate"
    return "slight"


def render_explanation(scored, bundle):
    st.subheader("Why was a student flagged?")
    ids = scored[inference.STUDENT_ID_COLUMN].tolist()
    chosen = st.selectbox("Select a student", ids)

    row = scored[scored[inference.STUDENT_ID_COLUMN] == chosen]
    prob = float(row[inference.RISK_PROB_COLUMN].iloc[0])
    flagged = "at risk" if bool(row["at_risk"].iloc[0]) else "not at risk"
    st.markdown(
        f"**{chosen}** has an estimated **{PROB_PERCENT_FORMAT.format(prob)}** "
        f"chance of failing — currently **{flagged}**. The factors behind this:"
    )

    raw_row = row[inference.required_raw_columns(bundle)]
    contributions = inference.explain_student(raw_row, bundle)
    contributions["label"] = contributions["feature"].map(labels.humanize_feature)
    max_abs = contributions["shap_value"].abs().max()

    raises = contributions[contributions["shap_value"] > 0].sort_values(
        "shap_value", ascending=False)
    lowers = contributions[contributions["shap_value"] < 0].sort_values(
        "shap_value")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔺 Raises failure risk**")
        if raises.empty:
            st.caption("None among the top factors.")
        for _, r in raises.iterrows():
            st.markdown(f"- {r['label']} — *{_strength_word(r['shap_value'], max_abs)}*")
    with col2:
        st.markdown("**🔻 Lowers failure risk**")
        if lowers.empty:
            st.caption("None among the top factors.")
        for _, r in lowers.iterrows():
            st.markdown(f"- {r['label']} — *{_strength_word(r['shap_value'], max_abs)}*")

    with st.expander("See the contribution chart"):
        st.caption("SHAP values - how much each input pushed this student's risk "
                   "up (positive) or down (negative) versus the average student.")
        st.bar_chart(contributions.set_index("label")["shap_value"])


def render_footer():
    st.divider()
    st.caption(
        f"**{APP_TITLE}** · {RESEARCHER} (ID {RESEARCHER_ID}) · "
        f"Supervisor: {SUPERVISOR} · {INSTITUTION}, {PROGRAMME} · "
        f"[www.princemiller.com]({RESEARCHER_SITE}) · "
        f"[{RESEARCHER_EMAIL}](mailto:{RESEARCHER_EMAIL})"
    )


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🩺", layout="wide",
                       menu_items=MENU_ITEMS)
    st.markdown(HIDE_BRANDING_CSS, unsafe_allow_html=True)
    render_header()

    try:
        bundle = get_bundle()
    except FileNotFoundError as err:
        st.error(str(err))
        st.stop()

    render_data_warning(bundle)
    threshold = render_sidebar(bundle)

    uploaded = st.file_uploader("Upload student records (.csv or .xlsx)",
                                type=["csv", "xlsx"])
    if uploaded is None:
        st.info("Upload a spreadsheet to begin, or download the template from the sidebar.")
    else:
        raw_df = read_upload(uploaded)
        missing = inference.missing_columns(raw_df, bundle)
        if missing:
            st.error(f"The file is missing required columns: {missing}")
        else:
            scored = inference.score_students(raw_df, bundle, threshold)
            render_summary(scored)
            render_results_table(scored, bundle)
            render_explanation(scored, bundle)

    # footer with the byline
    render_footer()


if __name__ == "__main__":
    main()
