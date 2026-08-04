"""
app.py — licensure risk screening for allied health tutors.

Upload a spreadsheet of student records, get a ranked at-risk list with a
per-student explanation of why the model flagged them.

Launch:  ml_env/bin/streamlit run app/app.py
"""
import glob
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for prediction
sys.path.insert(0, str(Path(__file__).resolve().parent))          # app/, for labels
import prediction  # noqa: E402 — repo-root module, added to path above
import labels      # noqa: E402 — plain-English names + glossary

DEFAULT_THRESHOLD = 0.50
PROB_PERCENT_FORMAT = "{:.0%}"

# ── Project / authorship metadata (shown in the About modal + bylines) ──
APP_TITLE = "Licensure Risk Screening"
RESEARCHER = "Prince Bortey Miller"
# RESEARCHER_ID = "22388461"
RESEARCHER_EMAIL = "prince@princemiller.com"
RESEARCHER_SITE = "https://www.princemiller.com"
SUPERVISOR = "Dr. Eric Opoku Osei"
INSTITUTION = "KNUST — Dept. of Computer Science"
PROGRAMME = "MSc Health Informatics, 2025-2026"

ABOUT_MD = f"""### {APP_TITLE}

Explainable machine learning for predicting **health professions licensure
examination** failure in Ghana - an **XGBoost + SHAP** approach. Built on
Accra School of Hygiene records across Environmental Health, Occupational
Health and Safety, and Occupational Therapy. It screens trainees for
first-attempt failure risk so tutors can target remediation early.
Decision support only - not a verdict.

**Researcher:** {RESEARCHER}

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
    return prediction.load_bundle()


def read_upload(uploaded_file):
    """Read an uploaded .csv or .xlsx into a DataFrame."""
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def render_header():
    st.title("Licensure Risk Screening")
    st.caption(
        "Identify trainees at risk of failing the licensure examination, "
        "so remediation can start early. Predictions are model estimates, not "
        "verdicts - use them alongside your own judgement."
    )


def render_data_warning(bundle):
    """Show an illustrative-only banner until the model is trained on real data."""
    if bundle.get("data_source", "synthetic") != "real":
        st.warning(
            "**Trying to train the model** :::  **synthetic & public data**"
            " The Progress: *IT WORKS!.* I'm still trying to improve the model performance "
            "",
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
        prediction.blank_template(bundle).to_csv(index=False),
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
    display[prediction.RISK_PROB_COLUMN] = display[prediction.RISK_PROB_COLUMN].map(
        PROB_PERCENT_FORMAT.format
    )
    display["at_risk"] = display["at_risk"].map({True: "Yes", False: "No"})
    shown = [prediction.RANK_COLUMN, prediction.STUDENT_ID_COLUMN,
             prediction.RISK_PROB_COLUMN, "at_risk"] + bundle["categorical_columns"]
    display = display[shown].rename(columns=labels.RESULT_HEADERS)
    st.dataframe(display, use_container_width=True, hide_index=True)

    with st.expander("What do these columns mean?"):
        for col, meaning in labels.METRIC_GLOSSARY.items():
            st.markdown(f"- **{col}** — {meaning}")

    st.download_button(
        "Download full risk report (.csv)",
        scored.to_csv(index=False),
        file_name="ahpc_risk_report.csv",
        mime="text/csv",
    )


def _strength_word(value, max_abs):
    """How much a factor weighs, relative to the strongest one shown."""
    ratio = abs(value) / max_abs if max_abs else 0
    if ratio >= 0.66:
        return "major factor"
    if ratio >= 0.33:
        return "moderate factor"
    return "minor factor"


def _join_factors(frame, limit=2):
    """A readable 'A and B' phrase from the top factor labels (case preserved
    so domain acronyms like CA / CGPA / WASSCE stay intact)."""
    items = frame["label"].head(limit).tolist()
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _natural_summary(chosen, prob, at_risk, raises, lowers):
    """A plain-language sentence a tutor could read aloud."""
    pct = PROB_PERCENT_FORMAT.format(prob)
    top_raise = _join_factors(raises)
    top_lower = lowers["label"].iloc[0] if not lowers.empty else None

    if at_risk:
        text = (
            f"Mainly because of **{top_raise or 'this student’s overall profile'}**, "
            f"the model predicts **{chosen} is likely to fail** at least one of the "
            f"six papers at the first sitting — about a **{pct}** chance."
        )
        if top_lower:
            text += (f" Their **{top_lower}** counts in their favour, but not enough "
                     f"to offset the risk.")
        text += f" Consider prioritising {chosen} for early remediation."
    else:
        text = (
            f"**{chosen}**’s overall profile"
            + (f", helped by **{top_lower}**," if top_lower else "")
            + f" keeps the estimated first-sitting failure risk **low at {pct}**, so the "
              f"model does **not** flag them."
        )
        if top_raise:
            text += f" If anything could push the risk up, watch **{top_raise}**."
    return text


def render_explanation(scored, bundle):
    st.subheader("Why was a student flagged?")
    ids = scored[prediction.STUDENT_ID_COLUMN].tolist()
    chosen = st.selectbox("Select a student", ids)

    row = scored[scored[prediction.STUDENT_ID_COLUMN] == chosen]
    prob = float(row[prediction.RISK_PROB_COLUMN].iloc[0])
    at_risk = bool(row["at_risk"].iloc[0])
    st.markdown(
        f"**{chosen}** has an estimated **{PROB_PERCENT_FORMAT.format(prob)}** chance of "
        f"failing — currently **{'at risk' if at_risk else 'not at risk'}**. "
        "The factors behind this:"
    )

    raw_row = row[prediction.required_raw_columns(bundle)]
    contributions = prediction.explain_student(raw_row, bundle)
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

    st.info(_natural_summary(chosen, prob, at_risk, raises, lowers), icon="🩺")
    st.caption("A *major factor* weighs heavily in this estimate; a *minor factor* "
               "nudges it only slightly. These are the model's reasons, not a verdict — "
               "always combine with your own knowledge of the student.")

    with st.expander("See the contribution chart"):
        st.caption("SHAP values — how much each input pushed this student's risk "
                   "up (positive) or down (negative) versus the average student.")
        st.bar_chart(contributions.set_index("label")["shap_value"])


# Matches whichever bundle got loaded (real once it exists, else synthetic)
# so the comparison table and figures shown always agree with the served model.
RESULTS_DIR = prediction.resolve_bundle_path().parent

# ── Model-insight side drawer ──
INSIGHT_STATE_KEY = "show_model_insight"
INSIGHT_DRAWER_KEY = "model-insight-drawer"  # → DOM class `st-key-model-insight-drawer`
DRAWER_WIDTH_PX = 820

# Position the keyed container as a fixed right-side panel and dim the page behind
# it. The drawer holds native Streamlit widgets (st.image, st.caption), so it is a
# real container styled by its key — not raw HTML with base64 images.
DRAWER_CSS = f"""
    <style>
    .insight-overlay {{
        position: fixed;
        inset: 0;
        background: rgba(15, 23, 42, 0.45);
        z-index: 9998;
        animation: insightFade .2s ease-out;
    }}
    .st-key-{INSIGHT_DRAWER_KEY} {{
        position: fixed;
        top: 0;
        right: 0;
        width: {DRAWER_WIDTH_PX}px;
        max-width: 92vw;
        height: 100vh;
        background: #ffffff;
        color: #0f172a;
        box-shadow: -8px 0 28px rgba(15, 23, 42, .18);
        z-index: 9999;
        overflow-y: auto;
        overflow-x: hidden;
        padding: 4rem 1.75rem 3rem;
        animation: insightSlideIn .28s cubic-bezier(.16, 1, .3, 1);
    }}
    /* Keep wide figures inside the drawer. Inside a fixed panel, use_container_width
       mis-measures and the image blocks become flex items with min-width:auto, so
       they refuse to shrink below the image's intrinsic width and overflow. Reset
       min-width and cap every wrapper (and the <img>) to the drawer width. */
    .st-key-{INSIGHT_DRAWER_KEY} [data-testid="stElementContainer"],
    .st-key-{INSIGHT_DRAWER_KEY} [data-testid="stFullScreenFrame"],
    .st-key-{INSIGHT_DRAWER_KEY} [data-testid="stImageContainer"],
    .st-key-{INSIGHT_DRAWER_KEY} [data-testid="stImage"] {{
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
    }}
    .st-key-{INSIGHT_DRAWER_KEY} [data-testid="stImage"] img {{
        width: 100% !important;
        height: auto;
    }}
    @keyframes insightSlideIn {{
        from {{ transform: translateX(100%); }}
        to {{ transform: translateX(0); }}
    }}
    @keyframes insightFade {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    </style>
"""

# Baseline-vs-pruned comparison table (produced by Stage 1) — the headline
# E-XGBoost result: does SHAP pruning preserve performance on a leaner feature set?
COMPARISON_TABLE = "comparison_table.csv"

# Cohort-wide figures (produced by Stage 1) with plain-English captions, mirroring
# the "Seeing the model think" section of the project portfolio.
MODEL_FIGURES = [
    ("shap_beeswarm.png", "What drives risk across the whole cohort",
     "Each dot is one student. Features are ranked by how much they sway the "
     "prediction — academic signals like mock scores and CGPA outweigh "
     "demographic ones."),
    ("roc_pr_curves.png", "How well it tells pass from fail",
     "ROC and precision–recall curves, XGBoost against every baseline. AUC-PR is "
     "the headline metric — it stays honest on the imbalanced failure class."),
    ("calibration_curves.png", "Can you trust the percentages?",
     "A 70% risk score should mean roughly 70% of such students actually fail. "
     "The calibration curve and Brier score check the probabilities are meaningful."),
]


def open_model_insight():
    st.session_state[INSIGHT_STATE_KEY] = True


def close_model_insight():
    st.session_state[INSIGHT_STATE_KEY] = False


def _render_comparison_table():
    """TABLE 1 — baseline XGBoost vs the SHAP-pruned E-XGBoost."""
    path = RESULTS_DIR / COMPARISON_TABLE
    if not path.exists():
        return
    st.markdown("**TABLE 1 - XGBoost (Baseline) vs E-XGBoost (Engineered)**")
    st.caption("SHAP-guided pruning keeps only the features that jointly explain 95% "
               "of the model's decisions. The test: does the leaner model match the "
               "full one? Scores are mean ± SD over 10-fold cross-validation × 10 seeds.")
    st.table(pd.read_csv(path))
    st.caption("Δ Change is E-XGBoost minus baseline; p-values are Wilcoxon signed-rank "
               "on the paired per-fold scores, with Cohen's *d* effect size. Here the "
               "pruned model matches the baseline on every metric while using far fewer "
               "features — the parsimony result E-XGBoost is meant to show.")


def _render_insight_body():
    """Cohort-wide figures mirroring the portfolio's 'Seeing the model think' section."""
    st.caption("Cohort-wide views of the model behind the scores above. These are "
               "illustrative - generated on the current synthetic model.")

    _render_comparison_table()

    for fname, title, desc in MODEL_FIGURES:
        path = RESULTS_DIR / fname
        if path.exists():
            st.markdown(f"**{title}**")
            st.caption(desc)
            st.image(str(path), use_container_width=True)

    dependence = sorted(glob.glob(str(RESULTS_DIR / "shap_dependence_*.png")))
    if dependence:
        st.markdown("**How a top factor shapes risk**")
        st.caption("Dependence plots show the shape of a relationship — for "
                   "example, how falling mock scores push failure risk upward.")
        for path in dependence:
            st.image(path, use_container_width=True)

    st.caption("The per-student breakdown above ('Why was a student flagged?') "
               "is the same model, zoomed in to a single trainee.")


def render_model_insight():
    """A right-side drawer mirroring the portfolio's 'Seeing the model think' section."""
    st.session_state.setdefault(INSIGHT_STATE_KEY, False)

    st.button("🧠 Seeing the model think - how it reaches these scores? Click Mee!",
              on_click=open_model_insight)

    if not st.session_state[INSIGHT_STATE_KEY]:
        return

    st.markdown(DRAWER_CSS, unsafe_allow_html=True)
    st.markdown('<div class="insight-overlay"></div>', unsafe_allow_html=True)

    with st.container(key=INSIGHT_DRAWER_KEY):
        st.markdown("### Seeing the model think")
        st.button("✕ Close", on_click=close_model_insight, key="close_insight_drawer")
        _render_insight_body()


def render_footer():
    st.divider()
    st.caption(
        # f"**{APP_TITLE}** · {RESEARCHER} (ID {RESEARCHER_ID}) · "
        f"**{APP_TITLE}** · {RESEARCHER} · "
        f"Supervisor: {SUPERVISOR} \n "
        f"{INSTITUTION}, {PROGRAMME} · "
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
        missing = prediction.missing_columns(raw_df, bundle)
        if missing:
            st.error(f"The file is missing required columns: {missing}")
        else:
            scored = prediction.score_students(raw_df, bundle, threshold)
            render_summary(scored)
            render_results_table(scored, bundle)
            render_explanation(scored, bundle)

    render_model_insight()
    render_footer()


if __name__ == "__main__":
    main()
