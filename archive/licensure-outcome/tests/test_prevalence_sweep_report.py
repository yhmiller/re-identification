import pandas as pd

import outcome_model
from scripts.compare_prevalence_sweep import RECORD_TYPE_COLUMN, SIMULATED_OUTCOME

TABLE = "results/prevalence_sweep/sensitivity_table.csv"


def test_every_row_is_marked_as_a_simulated_outcome(repo_root):
    table = pd.read_csv(repo_root / TABLE)
    assert (table[RECORD_TYPE_COLUMN] == SIMULATED_OUTCOME).all()


def test_the_simulated_outcome_marker_is_the_first_column(repo_root):
    table = pd.read_csv(repo_root / TABLE)
    assert table.columns[0] == RECORD_TYPE_COLUMN


def test_table_has_one_row_per_prevalence(repo_root):
    table = pd.read_csv(repo_root / TABLE)
    assert sorted(table.prevalence) == sorted(outcome_model.PREVALENCES)


def test_engineered_feature_count_is_below_baseline(repo_root):
    table = pd.read_csv(repo_root / TABLE)
    assert (table.n_features_engineered < table.n_features_baseline).all()


def test_every_row_names_the_features_it_kept(repo_root):
    table = pd.read_csv(repo_root / TABLE)
    assert table.kept_features.str.len().gt(0).all()


def test_every_row_names_the_features_it_pruned(repo_root):
    table = pd.read_csv(repo_root / TABLE)
    assert table.pruned_features.str.len().gt(0).all()


def test_kept_and_pruned_partition_the_baseline_set(repo_root):
    table = pd.read_csv(repo_root / TABLE)
    for row in table.itertuples():
        kept = set(row.kept_features.split("|"))
        pruned = set(row.pruned_features.split("|"))
        assert not kept & pruned
        assert len(kept) + len(pruned) == row.n_features_baseline


def test_mean_auc_roc_is_reported_for_each_prevalence(repo_root):
    table = pd.read_csv(repo_root / TABLE)
    assert table.baseline_auc_roc.between(0, 1).all()
