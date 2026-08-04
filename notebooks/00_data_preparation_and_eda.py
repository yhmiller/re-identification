# ============================================================
# 00_data_preparation_and_eda.py
# STAGE 0 — Raw college records → anonymised, model-ready CSV
#           + Exploratory Data Analysis for the thesis
#
# Run this FIRST when the college hands you the raw data file.
# Output: anonymised_records.csv  → loaded by Cell 6 of
#         01_pipeline_and_experiments.py
#
# Author : Prince Bortey Miller | ID: 22388461 | KNUST
# Ethics : HuSSREC, KNUST — anonymisation per approved protocol
# ============================================================


# ─────────────────────────────────────────────────────────────
# PREP-CELL 1 — Imports
# ─────────────────────────────────────────────────────────────
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
sns.set_style("whitegrid")

print("✅ Stage 0 ready — data preparation + EDA")


# ─────────────────────────────────────────────────────────────
# PREP-CELL 2 — CONFIG: edit this block to match the college file
# This is the ONLY cell you should need to edit.
# ─────────────────────────────────────────────────────────────

# Path to the raw file the college gives you (Excel or CSV)
RAW_DATA_PATH = "raw_college_records.xlsx"      # ← EDIT

# Map the COLLEGE'S column names → the pipeline's schema names.
# Left side = exactly what appears in their file (check spelling!).
# Right side = pipeline schema. Fill in after you see the real file.
COLUMN_MAPPING = {
    # identity (will be hashed then dropped)
    "Index Number":            "student_id",
    # predictors
    "WASSCE Aggregate":        "wassce_aggregate",
    "CGPA":                    "programme_cgpa",
    "CA Medical Surgical":     "ca_medical_surgical",
    "CA Mental Health":        "ca_mental_health",
    "CA Paediatric":           "ca_paediatric",
    "CA Public Health":        "ca_public_health",
    "CA Obstetric":            "ca_obstetric",
    "CA Pharmacology":         "ca_pharmacology",
    "Mock Medical Surgical":   "mock_medical_surgical",
    "Mock Mental Health":      "mock_mental_health",
    "Mock Paediatric":         "mock_paediatric",
    "Mock Public Health":      "mock_public_health",
    "Mock Obstetric":          "mock_obstetric",
    "Mock Pharmacology":       "mock_pharmacology",
    "Programme":               "programme_type",
    "Age":                     "age_raw",          # converted to age_band below
    "Sex":                     "gender",
    "Region":                  "region",
    "Year Group":              "cohort_year",
    # outcome
    "Result Status":           "result_status",     # e.g. PASS/FAIL/ABSENT/WITHHELD
    "Papers Failed":           "papers_failed",     # count, if provided
}

# Values in result_status that mean the candidate has NO valid
# first-attempt outcome → EXCLUDED per the proposal's cleaning rule
EXCLUDE_STATUSES = ["ABSENT", "DEFERRED", "WITHHELD", "INCOMPLETE", "PENDING"]

# How to derive the binary target (pick ONE that matches their file):
#   "status"  → fail = 1 if result_status == "FAIL"
#   "papers"  → fail = 1 if papers_failed >= 1
TARGET_RULE = "status"                              # ← EDIT if needed

# Subjects and per-subject columns come from the single source of truth,
# nmcle_schema.py at the repo root (edit exam papers there, not here).
import os
import sys

try:
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
except NameError:
    pass  # __file__ is undefined in a pasted Colab cell — rely on the cwd

from nmcle_schema import NMC_SUBJECTS, CA_COLS, MOCK_COLS

print("✅ Config set. Edit COLUMN_MAPPING after inspecting the real file.")


# ─────────────────────────────────────────────────────────────
# PREP-CELL 3 — Load raw file and standardise columns
# ─────────────────────────────────────────────────────────────

if RAW_DATA_PATH.endswith((".xlsx", ".xls")):
    raw = pd.read_excel(RAW_DATA_PATH)
else:
    raw = pd.read_csv(RAW_DATA_PATH)

print(f"Loaded raw file: {raw.shape[0]} rows × {raw.shape[1]} cols")
print(f"Raw columns: {list(raw.columns)}")

# Trim whitespace in headers, apply mapping
raw.columns = [str(c).strip() for c in raw.columns]
unmapped = [c for c in COLUMN_MAPPING if c not in raw.columns]
if unmapped:
    print(f"⚠️  Mapping keys not found in file (fix COLUMN_MAPPING): {unmapped}")
df = raw.rename(columns=COLUMN_MAPPING)

# Standardise string cells: strip + uppercase status/categories
for col in ["result_status", "programme_type", "gender", "region"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.upper()

# Coerce numerics
for col in ["wassce_aggregate", "programme_cgpa", "papers_failed",
            "age_raw", "cohort_year"] + CA_COLS + MOCK_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("✅ Columns standardised.")


# ─────────────────────────────────────────────────────────────
# PREP-CELL 4 — Apply EXCLUSION rules (proposal §4.2 cleaning rule)
# Candidates with no valid first-attempt outcome are removed.
# Every exclusion is counted and reported (needed for the thesis
# participant-flow description).
# ─────────────────────────────────────────────────────────────

n0 = len(df)
exclusion_log = {"initial_rows": n0}

# 1) No outcome recorded at all
mask_no_outcome = df["result_status"].isna() | (df["result_status"] == "NAN")
exclusion_log["missing_outcome"] = int(mask_no_outcome.sum())
df = df[~mask_no_outcome]

# 2) Absent / deferred / withheld / incomplete
mask_excluded = df["result_status"].isin(EXCLUDE_STATUSES)
exclusion_log["absent_deferred_withheld"] = int(mask_excluded.sum())
df = df[~mask_excluded]

# 3) Duplicate student IDs (keep first occurrence)
dups = df.duplicated(subset="student_id", keep="first")
exclusion_log["duplicate_ids"] = int(dups.sum())
df = df[~dups]

# 4) Missing the CORE predictors (CGPA + at least one mock score)
mask_core = df["programme_cgpa"].isna() | df[MOCK_COLS].isna().all(axis=1)
exclusion_log["missing_core_predictors"] = int(mask_core.sum())
df = df[~mask_core]

exclusion_log["final_rows"] = len(df)

print("✅ Exclusion rules applied (report this table in the thesis):")
for k, v in exclusion_log.items():
    print(f"   {k:<28}: {v}")

if len(df) < 300:
    print(f"\n⚠️  WARNING: only {len(df)} records — below the 300 minimum")
    print("   in the proposal's inclusion criteria. Approach an additional")
    print("   college, or document this as a limitation.")
else:
    print(f"\n✅ {len(df)} records ≥ 300 minimum — inclusion criterion met.")


# ─────────────────────────────────────────────────────────────
# PREP-CELL 5 — Derive the binary target + age bands
# ─────────────────────────────────────────────────────────────

if TARGET_RULE == "status":
    df["fail"] = (df["result_status"] == "FAIL").astype(int)
elif TARGET_RULE == "papers":
    df["fail"] = (df["papers_failed"].fillna(0) >= 1).astype(int)
else:
    raise ValueError("TARGET_RULE must be 'status' or 'papers'")

# Age band from raw age (matches the questionnaire bands)
if "age_raw" in df.columns and df["age_raw"].notna().any():
    df["age_band"] = pd.cut(
        df["age_raw"], bins=[0, 19, 24, 29, 200],
        labels=["Below 20", "20-24", "25-29", "30+"]
    ).astype(str)
elif "age_band" not in df.columns:
    df["age_band"] = "UNKNOWN"

prev = df["fail"].mean()
print(f"✅ Target derived (rule: {TARGET_RULE}).")
print(f"   Class prevalence — Fail: {prev:.1%} | Pass: {1-prev:.1%}")
if 0.35 <= prev <= 0.65:
    print("   → Roughly balanced. Imbalance handling likely UNNECESSARY")
    print("     (matches proposal: train on natural distribution first).")
else:
    print("   → Skewed. Compare SMOTE / scale_pos_weight / threshold tuning")
    print("     and select by validation AUC-PR (proposal Phase 4 rule).")


# ─────────────────────────────────────────────────────────────
# PREP-CELL 6 — ANONYMISE (SHA-256) and drop identifiers
# This implements the HuSSREC-approved protocol exactly:
# index numbers → salted SHA-256 pseudonyms; no names/contacts kept.
# ─────────────────────────────────────────────────────────────

# A fixed study salt: prevents anyone re-hashing known index numbers
# to re-identify rows. Keep this string PRIVATE (do not publish).
STUDY_SALT = "MILLER-22388461-NMCLE-2026"          # ← keep private
# #TODO - Read from .env - untracked file

def pseudonymise(value: str) -> str:
    return hashlib.sha256((STUDY_SALT + str(value)).encode()).hexdigest()[:16]

df["pseudo_id"] = df["student_id"].apply(pseudonymise)

# Drop every identifying / non-model column
DROP_COLS = ["student_id", "age_raw", "result_status", "papers_failed"]
extra_identifiers = [c for c in df.columns
                     if any(k in c.lower() for k in
                            ["name", "phone", "email", "contact", "address"])]
df = df.drop(columns=[c for c in DROP_COLS + extra_identifiers
                      if c in df.columns])

print("✅ Anonymisation complete.")
print(f"   pseudo_id sample: {df['pseudo_id'].iloc[0]}")
print(f"   Dropped: {DROP_COLS + extra_identifiers}")
print(f"   Remaining columns: {list(df.columns)}")


# ─────────────────────────────────────────────────────────────
# PREP-CELL 7 — Save the model-ready dataset
# ─────────────────────────────────────────────────────────────

OUT_PATH = "anonymised_records.csv"
df.to_csv(OUT_PATH, index=False)
print(f"✅ Saved → {OUT_PATH}  ({df.shape[0]} rows × {df.shape[1]} cols)")
print("   Next: point Cell 6 of 01_pipeline_and_experiments.py at this file.")
print("   Store ONLY this anonymised file in Google Drive; delete/secure the raw file.")


# ═════════════════════════════════════════════════════════════
# PART B — EXPLORATORY DATA ANALYSIS (for thesis results chapter)
# Produces the descriptive tables and figures examiners expect
# BEFORE any modelling results.
# ═════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# EDA-CELL 1 — Descriptive statistics table
# ─────────────────────────────────────────────────────────────

numeric_cols = ["wassce_aggregate", "programme_cgpa"] + CA_COLS + MOCK_COLS
numeric_cols = [c for c in numeric_cols if c in df.columns]

desc = df[numeric_cols].describe().T[["count", "mean", "std", "min", "max"]]
desc = desc.round(2)
desc.to_csv("eda_descriptives.csv")
print("✅ Descriptive statistics (Table for thesis):")
print(desc.to_string())


# ─────────────────────────────────────────────────────────────
# EDA-CELL 2 — Missingness report
# ─────────────────────────────────────────────────────────────

miss = (df.isna().mean() * 100).round(1).sort_values(ascending=False)
miss = miss[miss > 0]
if len(miss):
    print("⚠️  Columns with missing values (% missing):")
    print(miss.to_string())
    miss.to_csv("eda_missingness.csv")
else:
    print("✅ No missing values after cleaning.")


# ─────────────────────────────────────────────────────────────
# EDA-CELL 3 — Outcome by group (prevalence tables)
# ─────────────────────────────────────────────────────────────

print("✅ Failure rate by group (report in thesis):")
for gcol in ["programme_type", "gender", "age_band", "cohort_year"]:
    if gcol in df.columns:
        tab = (df.groupby(gcol)["fail"]
                 .agg(n="count", fail_rate="mean")
                 .round(3))
        print(f"\n   ── by {gcol} ──")
        print(tab.to_string())
        tab.to_csv(f"eda_failrate_by_{gcol}.csv")


# ─────────────────────────────────────────────────────────────
# EDA-CELL 4 — Score distributions by outcome (figure)
# ─────────────────────────────────────────────────────────────

df["_mock_avg"] = df[MOCK_COLS].mean(axis=1)
candidate_cols = [("programme_cgpa", "Programme CGPA"),
                  ("wassce_aggregate", "WASSCE Aggregate"),
                  ("_mock_avg", "Mock Exam Average")]
plot_cols = [(c, t) for c, t in candidate_cols if c in df.columns]

fig, axes = plt.subplots(1, len(plot_cols), figsize=(5.4 * len(plot_cols), 4.5),
                          squeeze=False)
axes = axes.ravel()

for ax, (col, title) in zip(axes, plot_cols):
    for val, lab, color in [(0, "Pass", "#2E7D32"), (1, "Fail", "#C62828")]:
        ax.hist(df.loc[df["fail"] == val, col].dropna(),
                bins=20, alpha=0.55, label=lab, color=color)
    ax.set_title(f"{title} by Licensure Outcome")
    ax.set_xlabel(title); ax.set_ylabel("Count"); ax.legend()

plt.tight_layout()
plt.savefig("eda_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
df = df.drop(columns=["_mock_avg"])
print("✅ Distribution figure saved → eda_distributions.png")


# ─────────────────────────────────────────────────────────────
# EDA-CELL 5 — Correlation heatmap (numeric predictors + target)
# ─────────────────────────────────────────────────────────────

corr_cols = numeric_cols + ["fail"]
corr = df[corr_cols].corr().round(2)

plt.figure(figsize=(12, 9))
sns.heatmap(corr, cmap="RdBu_r", center=0, annot=False,
            linewidths=0.4, cbar_kws={"label": "Pearson r"})
plt.title("Correlation Matrix — Predictors and Licensure Failure")
plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=150, bbox_inches="tight")
plt.show()

top_corr = corr["fail"].drop("fail").abs().sort_values(ascending=False).head(8)
print("✅ Correlation heatmap saved → eda_correlation.png")
print("\n   Strongest raw correlations with failure:")
print(top_corr.to_string())


# ─────────────────────────────────────────────────────────────
# EDA-CELL 6 — Stage-0 summary
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STAGE 0 COMPLETE — DATA READY FOR MODELLING")
print("=" * 60)
print(f"""
  Records (final)   : {len(df)}
  Fail prevalence   : {df['fail'].mean():.1%}
  Output dataset    : anonymised_records.csv
  EDA artefacts     : eda_descriptives.csv, eda_missingness.csv,
                      eda_failrate_by_*.csv, eda_distributions.png,
                      eda_correlation.png
  Exclusion log     : report the PREP-CELL 4 table in your thesis

  NEXT:
   1. Open 01_pipeline_and_experiments.py → Cell 6
   2. Set REAL_DATA_PATH = "anonymised_records.csv" and uncomment
   3. Run Cells 6→20 (baselines, XGBoost, SHAP, calibration, fairness)
   4. Run 02_model_engineering.py (E-XGBoost vs baseline)
""")
