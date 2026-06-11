"""
labels.py — plain-English names and glossary for the screening app.

The model uses short machine column names (ca_medical_surgical, mock_avg,
age_band_30+). Nurse educators should never see those. Every name shown in the
UI is translated here so the output reads in ordinary language.
"""

# NMC-LE theory papers → display titles
SUBJECT_LABELS = {
    "medical_surgical": "Medical-Surgical",
    "mental_health":    "Mental Health",
    "paediatric":       "Paediatric",
    "public_health":    "Public Health",
    "obstetric":        "Obstetric",
    "pharmacology":     "Pharmacology",
}

# One-hot encoded categorical prefixes → display labels
CATEGORICAL_LABELS = {
    "programme_type": "Programme",
    "age_band":       "Age band",
    "gender":         "Gender",
}

# Exact-match labels for raw + engineered features
FEATURE_LABELS = {
    "wassce_aggregate":     "WASSCE aggregate (entry grade — lower is better)",
    "programme_cgpa":       "Programme CGPA",
    "ca_avg":               "Average continuous-assessment score",
    "mock_avg":             "Average mock-exam score",
    "ca_mock_gap":          "Gap between CA and mock averages",
    "ca_consistency":       "Consistency across CA subjects",
    "mock_consistency":     "Consistency across mock subjects",
    "n_weak_ca_subjects":   "Number of weak CA subjects",
    "n_weak_mock_subjects": "Number of weak mock subjects",
    "min_mock_score":       "Lowest single mock-exam score",
    "min_ca_score":         "Lowest single continuous-assessment score",
    "wassce_band":          "WASSCE band (entry-grade tier)",
}

# Plain-English meaning of each required upload column (sidebar help)
COLUMN_GLOSSARY = {
    "wassce_aggregate": "WASSCE entry aggregate (6 = best, 36 = weakest).",
    "programme_cgpa":   "Cumulative GPA for the programme (e.g. 1.0–4.0).",
    "ca_<subject>":     "Continuous-assessment score (0–100) for each of the 6 papers.",
    "mock_<subject>":   "Internal mock-exam score (0–100) for each of the 6 papers.",
    "programme_type":   "Programme: RGN, RM, NAC or NAP.",
    "age_band":         "Age group: Below 20, 20-24, 25-29, 30+.",
    "gender":           "Female or Male.",
}

# Plain-English meaning of each results-table column
METRIC_GLOSSARY = {
    "Rank":             "Ordering by estimated risk (1 = highest risk).",
    "Fail probability": "The model's estimated chance the student fails at least "
                        "one of the six theory papers on the first attempt.",
    "At risk?":         "Flagged when the fail probability is at or above the "
                        "threshold you set in the sidebar.",
}

# Friendly headers for the results table
RESULT_HEADERS = {
    "risk_rank":        "Rank",
    "student_id":       "Student",
    "fail_probability": "Fail probability",
    "at_risk":          "At risk?",
    "programme_type":   "Programme",
    "age_band":         "Age band",
    "gender":           "Gender",
}


def humanize_feature(name):
    """Translate a model feature name into a plain-English label."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]

    for prefix, text in (("weak_ca_", "Weak CA"), ("weak_mock_", "Weak mock")):
        subject = name[len(prefix):]
        if name.startswith(prefix) and subject in SUBJECT_LABELS:
            return f"{text} in {SUBJECT_LABELS[subject]} (score below 50)"

    for prefix, text in (("ca_", "Continuous assessment"), ("mock_", "Mock exam")):
        subject = name[len(prefix):]
        if name.startswith(prefix) and subject in SUBJECT_LABELS:
            return f"{text}: {SUBJECT_LABELS[subject]}"

    for col, label in CATEGORICAL_LABELS.items():
        if name.startswith(col + "_"):
            return f"{label}: {name[len(col) + 1:]}"

    return name.replace("_", " ").strip().capitalize()
