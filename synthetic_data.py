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


def build_schema_dataframe(ca_cols, mock_cols, nmc_subjects, target, seed, n=20):
    """
    A tiny hand-crafted reference table that tells SDV the realistic
    distribution of each column, for pilot (non-real) data only.
    (was 01_pipeline_and_experiments.py CELL 4's build_schema_dataframe)
    """
    rng = np.random.default_rng(seed)

    df = pd.DataFrame()
    df["wassce_aggregate"] = rng.integers(6, 36, n).astype(float)
    df["programme_cgpa"] = np.clip(rng.normal(2.8, 0.6, n), 1.0, 4.0)

    for col in ca_cols:
        df[col] = np.clip(rng.normal(62, 12, n), 20, 100)
    for col in mock_cols:
        df[col] = np.clip(rng.normal(58, 15, n), 10, 100)

    df["programme_type"] = rng.choice(["RGN", "RM", "NAC", "NAP"],
                                       n, p=[0.55, 0.25, 0.12, 0.08])
    df["age_band"] = rng.choice(["Below 20", "20-24", "25-29", "30+"],
                                 n, p=[0.10, 0.55, 0.25, 0.10])
    df["gender"] = rng.choice(["Female", "Male"], n, p=[0.72, 0.28])

    # Context columns — NOT model predictors. region drives fairness
    # disaggregation; cohort_year drives the temporal split.
    df["region"] = rng.choice(
        ["Ashanti", "Greater Accra", "Eastern", "Central",
         "Western", "Brong-Ahafo", "Northern", "Other"],
        n, p=[0.22, 0.18, 0.12, 0.10, 0.10, 0.08, 0.10, 0.10])
    df["cohort_year"] = rng.choice([2021, 2022, 2023, 2024], n)

    # Target: loosely correlated with CGPA and mock performance. Centered on
    # its own mean so the synthetic fail rate lands near 50% (the national
    # NMC-LE first-attempt rate) instead of collapsing to all-Pass.
    linear = (-0.5 * df["programme_cgpa"]
              - 0.02 * df[[f"mock_{s}" for s in nmc_subjects]].mean(axis=1))
    logit = (linear - linear.mean()) + rng.normal(0, 0.5, n)
    prob_fail = 1 / (1 + np.exp(-logit))
    df[target] = prob_fail > 0.5   # boolean, to match the SDV metadata sdtype
    return df


def generate_pilot_data(ca_cols, mock_cols, nmc_subjects, target, seed,
                         n_reference=30, n_rows=1000):
    """
    Fit a GaussianCopulaSynthesizer on the hand-crafted reference table and
    sample `n_rows` pilot records. PIPELINE TESTING ONLY — never used for
    training or evaluation results.
    """
    schema_df = build_schema_dataframe(ca_cols, mock_cols, nmc_subjects, target,
                                        seed, n=n_reference)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(schema_df)
    metadata.update_column("programme_type", sdtype="categorical")
    metadata.update_column("age_band", sdtype="categorical")
    metadata.update_column("gender", sdtype="categorical")
    metadata.update_column("region", sdtype="categorical")
    metadata.update_column("cohort_year", sdtype="categorical")
    metadata.update_column(target, sdtype="boolean")

    synthesiser = GaussianCopulaSynthesizer(metadata, enforce_rounding=True)
    synthesiser.fit(schema_df)

    syn_df = synthesiser.sample(num_rows=n_rows)
    syn_df[target] = syn_df[target].astype(int)
    syn_df["cohort_year"] = syn_df["cohort_year"].astype(int)
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
