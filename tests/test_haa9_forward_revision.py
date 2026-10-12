"""Independent fixtures and persisted-data audits for forward reanalysis."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wr_haa9_forward_revision import apply_cutoff, calibrated_cutoff, capacity_allocation, fold_id, operating_metrics, score_metrics


def test_cutoff_expected_training_workload_including_ties():
    scores = np.array([0., 0., 1., 2., 2., 2., 4., 7.])
    for target in (.05, .1, .15, .2, .3, .5, .9):
        threshold, tie = calibrated_cutoff(scores, target)
        selected = apply_cutoff(scores, threshold, tie)
        assert np.isclose(selected.mean(), target)
        for unique in np.unique(scores):
            assert len(np.unique(selected[scores == unique])) == 1


def test_score_metrics_match_expanded_bootstrap_records_with_ties():
    y = np.array([0., 1., 0., 1., 1., 0.])
    score = np.array([0., 1., 1., 2., 2., 3.])
    counts = np.array([[1., 1., 1., 1., 1., 1.], [2., 1., 0., 3., 1., 2.]])
    results = score_metrics(y, score, counts, False)
    for i, count in enumerate(counts):
        indices = np.repeat(np.arange(len(y)), count.astype(int))
        assert np.isclose(results["AUROC"][i], roc_auc_score(y[indices], score[indices]))
        assert np.isclose(results["average_precision"][i], average_precision_score(y[indices], score[indices]))


def test_operating_metrics_keep_negative_outcomes_and_fractional_ties():
    y = np.array([1., 1., 0., 0.])
    selected = np.array([1., .5, .5, 0.])
    metrics = operating_metrics(y, selected, np.ones((1, 4)))
    assert metrics["sensitivity"][0] == .75
    assert metrics["positive_predictive_value"][0] == .75
    assert metrics["followup_fraction"][0] == .5
    assert metrics["false_positive_rate"][0] == .25


def test_persisted_forward_chronology_and_fold_partition():
    p = pd.read_parquet(ROOT / "wr_results/haa9_revision/forward_out_of_fold_predictions.parquet")
    excluded = pd.read_parquet(ROOT / "wr_results/haa9_revision/forward_chronology_excluded_systems.parquet")
    assert len(p) == 2636 and p.PWSID.is_unique
    assert p.outcome.sum() == 420 and (1 - p.outcome).sum() == 2216
    assert (p.predictor_end < p.outcome_start).all()
    assert len(excluded) == 57 and (excluded.predictor_end >= excluded.outcome_start).all()
    assert len(p) + len(excluded) == 2693
    assert set(p.PWSID).isdisjoint(set(excluded.PWSID))
    assert (p.PWSID.map(fold_id) == p.outer_fold).all()
    assert (p.outcome == p.late_HAA9.gt(60)).all()


def test_persisted_training_scalar_cutoffs_reconstructed_without_test_labels():
    p = pd.read_parquet(ROOT / "wr_results/haa9_revision/forward_out_of_fold_predictions.parquet")
    cutoffs = pd.read_parquet(ROOT / "wr_results/haa9_revision/comparator_training_cutoffs.parquet")
    for row in cutoffs.loc[cutoffs.strategy.isin(["scalar_HAA5", "scalar_HAA9"])].itertuples():
        train = p.outer_fold.ne(row.outer_fold)
        col = "early_" + row.strategy.split("_")[-1]
        expected, tie = calibrated_cutoff(p.loc[train, col].to_numpy(), row.target_training_workload)
        assert expected == row.cutoff and tie == row.cutoff_tie_probability
        selected = apply_cutoff(p.loc[~train, col].to_numpy(), expected, tie)
        saved = p.loc[~train, f"selection_probability__{row.strategy}_workload_{row.target_training_workload:.2f}"].to_numpy()
        np.testing.assert_allclose(selected, saved, atol=1e-14)


def test_persisted_training_scalers_have_no_evaluation_observations():
    p = pd.read_parquet(ROOT / "wr_results/haa9_revision/forward_out_of_fold_predictions.parquet")
    fit = pd.read_parquet(ROOT / "wr_results/haa9_revision/forward_fitted_coefficients.parquet")
    for row in fit.itertuples():
        x = np.log1p(p.loc[p.outer_fold.ne(row.outer_fold), row.feature].to_numpy())
        assert np.isclose(x.mean(), row.training_log1p_mean)
        assert np.isclose(x.std(ddof=0), row.training_log1p_sd)
    for (fold, strategy), g in fit.groupby(["outer_fold", "strategy"]):
        subset = p.loc[p.outer_fold.eq(fold)]
        z = (np.log1p(subset[g.feature].to_numpy()) - g.training_log1p_mean.to_numpy()) / g.training_log1p_sd.to_numpy()
        logit = z @ g.coefficient.to_numpy() + g.intercept.iloc[0]
        np.testing.assert_allclose(1 / (1 + np.exp(-logit)), subset[f"score__{strategy}"], atol=1e-12)


def test_common_support_windows_and_nested_endpoints():
    w = pd.read_parquet(ROOT / "wr_results/haa9_revision/common_support_four_quarter_windows.parquet")
    assert len(w) == 11135 and w.PWSID.nunique() == 2487
    assert w.n_quarters.eq(4).all()
    assert all(pd.Period(end, freq="Q").ordinal - pd.Period(start, freq="Q").ordinal == 3
               for start, end in zip(w.first_quarter, w.last_quarter))
    for endpoint in ("maximum", "OEL_form", "LRAA_form"):
        assert (w[f"{endpoint}_HAA9"] >= w[f"{endpoint}_HAA5"] - 1e-9).all()
    for panel in ("HAA5", "HAA9"):
        assert (w[f"maximum_{panel}"] >= w[f"OEL_form_{panel}"] - 1e-9).all()
        assert (w[f"maximum_{panel}"] >= w[f"LRAA_form_{panel}"] - 1e-9).all()


def test_exact_capacity_is_label_free_deterministic_and_equal_across_scores():
    folds = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    ids = np.array([f"PWS{x}" for x in range(8)])
    for q in (.05, .1, .15, .2, .3, .5, .9):
        a = capacity_allocation(np.ones(8), folds, ids, q)
        b = capacity_allocation(np.arange(8), folds, ids, q)
        assert a.sum() == b.sum() == 2 * int(np.floor(4*q))
        assert np.array_equal(a, capacity_allocation(np.ones(8), folds, ids, q))


def test_saved_exact_capacity_is_the_declared_top_n_policy():
    p = pd.read_parquet(ROOT / "wr_results/haa9_revision/forward_out_of_fold_predictions.parquet")
    allocated = pd.read_parquet(ROOT / "wr_results/haa9_revision/forward_exact_capacity_allocations.parquet")
    assert np.array_equal(p.PWSID, allocated.PWSID)
    for name in allocated.columns[3:]:
        score_name, q = name.rsplit("_capacity_", 1)
        expected = capacity_allocation(p[f"score__{score_name}"].to_numpy(), p.outer_fold.to_numpy(), p.PWSID.to_numpy(), float(q))
        np.testing.assert_array_equal(expected, allocated[name])
    for q in (.05, .1, .15, .2, .3):
        names = [c for c in allocated if c.endswith(f"_capacity_{q:.2f}")]
        assert len(set(allocated[names].sum())) == 1
