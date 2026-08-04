"""
synthetic_data.py — every synthetic-data generator used in this project.

Two distinct uses, kept in one file so they don't drift apart:

1. `generate_pilot_data` — a hand-crafted-distribution SDV synthesiser used
   ONLY to exercise the pipeline before real records arrive. Never used for
   training or evaluation results (see notebooks/01_pipeline_and_experiments.py).

2. `generate_field_replica` — the Option 2 data-handling generator (see
   docs/TODO.md, "Data-handling: switch to Option 2"). It fits SDV's
   GaussianCopulaSynthesizer on the REAL field data's own distribution and
   samples a synthetic replica, so the same pipeline can be run once on real
   data and once on the replica and the two compared (model_comparison.py's
   compare_real_vs_synthetic). This is generic and works today — it just has
   nothing meaningful to fit until the real college data lands.
"""
import numpy as np
import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata


PROGRAMMES = ["EH", "OHS", "OT"]
GRADE_LETTERS = list("ABCDE")

# Observed in the Accra School of Hygiene 2021 and 2022 extraction (n = 110).
# The pilot generator matches these so a pipeline-testing run behaves like the
# real corpus rather than like an arbitrary distribution.
REAL_CGPA_MEAN, REAL_CGPA_SD = 2.63, 0.67
REAL_COURSES_MEAN, REAL_COURSES_SD = 46, 6
REAL_GRADE_MIX = [0.305, 0.218, 0.311, 0.156, 0.010]


def build_schema_dataframe(gpa_cols, grade_count_cols, target, seed, n=20):
    """A small reference table describing each column's realistic distribution.

    SDV learns the joint distribution from this table, so it only has to be
    representative in shape, not large. Pilot (non-real) data only.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame()

    # A latent ability per student keeps the semester GPAs correlated with each
    # other and with CGPA, which an independent draw per column would not.
    ability = np.clip(rng.normal(REAL_CGPA_MEAN, REAL_CGPA_SD, n), 0.4, 4.0)
    for i, col in enumerate(gpa_cols):
        drift = 0.05 * i  # students tend to improve slightly across the programme
        df[col] = np.clip(ability + drift + rng.normal(0, 0.45, n), 0.0, 4.0)

    df["cgpa"] = np.clip(df[gpa_cols].mean(axis=1) + rng.normal(0, 0.08, n), 0.0, 4.0)
    df["total_credits"] = np.clip(rng.normal(115, 10, n), 80, 160).round()
    df["n_courses"] = np.clip(
        rng.normal(REAL_COURSES_MEAN, REAL_COURSES_SD, n), 25, 55).round()

    # Grade counts must sum to n_courses, and stronger students earn more A/B.
    for j, col in enumerate(grade_count_cols):
        tilt = (ability - REAL_CGPA_MEAN) / REAL_CGPA_SD * (1.5 - j) * 0.05
        share = np.clip(REAL_GRADE_MIX[j] + tilt, 0.001, None)
        df[col] = share
    shares = df[grade_count_cols].div(df[grade_count_cols].sum(axis=1), axis=0)
    for col in grade_count_cols:
        df[col] = (shares[col] * df["n_courses"]).round()

    df["programme"] = rng.choice(PROGRAMMES, n, p=[0.45, 0.40, 0.15])
    df["cohort_year"] = rng.choice([2021, 2022], n)

    # Target: weaker and less consistent students fail more often. Centred on
    # its own mean so the pilot fail rate lands near the middle rather than
    # collapsing to all-pass.
    linear = -1.4 * ability + 0.6 * df[gpa_cols].std(axis=1)
    logit = (linear - linear.mean()) + rng.normal(0, 0.5, n)
    df[target] = (1 / (1 + np.exp(-logit))) > 0.5
    return df


def generate_pilot_data(gpa_cols, grade_count_cols, target, seed,
                         n_reference=30, n_rows=1000):
    """Sample pilot records from a synthesiser fitted on the reference table.

    PIPELINE TESTING ONLY. Never used for training or evaluation results.
    """
    schema_df = build_schema_dataframe(gpa_cols, grade_count_cols, target,
                                        seed, n=n_reference)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(schema_df)
    metadata.update_column("programme", sdtype="categorical")
    metadata.update_column("cohort_year", sdtype="categorical")
    metadata.update_column(target, sdtype="boolean")

    synthesiser = GaussianCopulaSynthesizer(metadata, enforce_rounding=True)
    synthesiser.fit(schema_df)

    syn_df = synthesiser.sample(num_rows=n_rows)
    syn_df[target] = syn_df[target].astype(int)
    syn_df["cohort_year"] = syn_df["cohort_year"].astype(int)
    # Grade counts are resampled independently by the copula, so restore the
    # invariant the downstream proportion features depend on.
    syn_df["n_courses"] = syn_df[grade_count_cols].sum(axis=1)
    return syn_df


def generate_field_replica(real_df, target, categorical_cols, seed, n_rows=None):
    """
    Option 2 data-handling generator: fit SDV's GaussianCopulaSynthesizer on
    the REAL field data's own distribution and sample a synthetic replica of
    the same shape, for pipeline replication (run the identical downstream
    pipeline on both and compare — see model_comparison.compare_real_vs_synthetic).

    Uses SDV rather than CTGAN to stay consistent with `generate_pilot_data`
    above and avoid a second synthesiser library; swap in CTGANSynthesizer
    here if higher-fidelity categorical modelling is needed later.

    `n_rows` defaults to len(real_df) so the replica matches the real
    dataset's size — the point is pipeline replication, not augmentation.
    """
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_df)
    for col in categorical_cols:
        metadata.update_column(col, sdtype="categorical")

    synthesiser = GaussianCopulaSynthesizer(metadata, enforce_rounding=True)
    synthesiser.fit(real_df)

    replica = synthesiser.sample(num_rows=n_rows or len(real_df))
    if pd.api.types.is_bool_dtype(real_df[target]) or set(real_df[target].unique()) <= {0, 1}:
        replica[target] = replica[target].astype(int)
    return replica
