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

DEFAULT_THRESHOLD = 0.50
PROB_PERCENT_FORMAT = "{:.0%}"


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
    st.sidebar.markdown("**Required columns:**")
    st.sidebar.write(inference.required_raw_columns(bundle))
    return threshold


def render_summary(scored):
    at_risk = int(scored["at_risk"].sum())
    total = len(scored)
    col1, col2, col3 = st.columns(3)
    col1.metric("Students screened", total)
    col2.metric("Flagged at-risk", at_risk)
    col3.metric("At-risk rate", PROB_PERCENT_FORMAT.format(at_risk / total) if total else "—")


def render_results_table(scored, bundle):
    display = scored.copy()
    display[inference.RISK_PROB_COLUMN] = display[inference.RISK_PROB_COLUMN].map(
        PROB_PERCENT_FORMAT.format
    )
    shown = [inference.RANK_COLUMN, inference.STUDENT_ID_COLUMN,
             inference.RISK_PROB_COLUMN, "at_risk"] + bundle["categorical_columns"]
    st.dataframe(display[shown], use_container_width=True, hide_index=True)
    st.download_button(
        "Download full risk report (.csv)",
        scored.to_csv(index=False),
        file_name="nmc_risk_report.csv",
        mime="text/csv",
    )


def render_explanation(scored, bundle):
    st.subheader("Why was a student flagged?")
    ids = scored[inference.STUDENT_ID_COLUMN].tolist()
    chosen = st.selectbox("Select a student", ids)

    row = scored[scored[inference.STUDENT_ID_COLUMN] == chosen]
    prob = float(row[inference.RISK_PROB_COLUMN].iloc[0])
    st.write(f"Estimated fail probability: **{PROB_PERCENT_FORMAT.format(prob)}**")

    raw_row = row[inference.required_raw_columns(bundle)]
    contributions = inference.explain_student(raw_row, bundle)
    contributions = contributions.set_index("feature")["shap_value"]
    st.caption("Top factors. Bars to the right raise risk; to the left lower it.")
    st.bar_chart(contributions)


def main():
    st.set_page_config(page_title="NMC-LE Risk Screening", page_icon="🩺", layout="wide")
    render_header()

    try:
        bundle = get_bundle()
    except FileNotFoundError as err:
        st.error(str(err))
        st.stop()

    threshold = render_sidebar(bundle)

    uploaded = st.file_uploader("Upload student records (.csv or .xlsx)",
                                type=["csv", "xlsx"])
    if uploaded is None:
        st.info("Upload a spreadsheet to begin, or download the template from the sidebar.")
        return

    raw_df = read_upload(uploaded)
    missing = inference.missing_columns(raw_df, bundle)
    if missing:
        st.error(f"The file is missing required columns: {missing}")
        return

    scored = inference.score_students(raw_df, bundle, threshold)
    render_summary(scored)
    render_results_table(scored, bundle)
    render_explanation(scored, bundle)


if __name__ == "__main__":
    main()
