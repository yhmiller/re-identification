"""
labels.py — plain-English names and glossary for the screening app.

The model uses short machine column names (gpa_sem1, prop_grade_D,
programme_OHS). Tutors should never see those. Every name shown in the UI is
translated here so the output reads in ordinary language.

Hur et al. (2025) found raw SHAP output is hard for practitioners to act on and
that clinician-friendly phrasing changes decision behaviour. This module is the
study's response to that; the tutor survey tests whether it works.
"""

# Programme codes → display titles
PROGRAMME_LABELS = {
    "EH":  "Environmental Health",
    "OHS": "Occupational Health and Safety",
    "OT":  "Occupational Therapy",
}

# One-hot encoded categorical prefixes → display labels
CATEGORICAL_LABELS = {
    "programme": "Programme",
}

# Exact-match labels for raw + engineered features
FEATURE_LABELS = {
    "cgpa":             "Cumulative GPA",
    "total_credits":    "Total credits attempted",
    "n_courses":        "Courses taken",

    "gpa_mean":         "Average semester GPA",
    "gpa_min":          "Weakest semester GPA",
    "gpa_max":          "Strongest semester GPA",
    "gpa_consistency":  "Semester-to-semester variability",
    "gpa_trend":        "Change from first to final semester",
    "gpa_first_half":   "Average GPA, semesters 1 to 3",
    "gpa_final_half":   "Average GPA, semesters 4 to 6",

    "n_weak_semesters": "Semesters below 2.0",
    "n_failed":         "Failed courses",
    "fail_rate":        "Proportion of courses failed",

    "n_grade_A":    "Number of A grades",
    "n_grade_B":    "Number of B grades",
    "n_grade_C":    "Number of C grades",
    "n_grade_D":    "Number of D grades",
    "n_grade_E":    "Number of E grades",
    "prop_grade_A": "Share of grades at A",
    "prop_grade_B": "Share of grades at B",
    "prop_grade_C": "Share of grades at C",
    "prop_grade_D": "Share of grades at D",
    "prop_grade_E": "Share of grades at E",
}

# Plain-English meaning of each required upload column (sidebar help)
COLUMN_GLOSSARY = {
    "cgpa":          "Cumulative GPA across the programme (0.0–4.0).",
    "total_credits": "Total credits attempted across all six semesters.",
    "gpa_sem1..6":   "Grade point average for each semester (0.0–4.0).",
    "n_courses":     "Total number of graded courses on the record.",
    "n_grade_A..E":  "How many A, B, C, D and E grades the student earned.",
    "programme":     "Programme code: EH, OHS or OT.",
}

# Plain-English meaning of each results-table column
METRIC_GLOSSARY = {
    "Rank":             "Ordering by estimated risk (1 = highest risk).",
    "Fail probability": "The model's calibrated estimate of the chance this "
                        "student fails the licensure examination at first "
                        "attempt. Calibrated on held-out data, so 0.70 means "
                        "roughly seven in ten similar students failed.",
    "At risk?":         "Flagged when the fail probability is at or above the "
                        "threshold you set in the sidebar.",
}

# Friendly headers for the results table
RESULT_HEADERS = {
    "risk_rank":        "Rank",
    "student_id":       "Student",
    "fail_probability": "Fail probability",
    "at_risk":          "At risk?",
    "programme":        "Programme",
    "cohort_year":      "Cohort",
}


def humanize_feature(name):
    """Translate a model feature name into a plain-English label."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]

    if name.startswith("gpa_sem"):
        return f"Semester {name[len('gpa_sem'):]} GPA"
    if name.startswith("weak_sem"):
        return f"Semester {name[len('weak_sem'):]} below 2.0"

    for col, label in CATEGORICAL_LABELS.items():
        if name.startswith(col + "_"):
            value = name[len(col) + 1:]
            return f"{label}: {PROGRAMME_LABELS.get(value, value)}"

    return name.replace("_", " ").strip().capitalize()
