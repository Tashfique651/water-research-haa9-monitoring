"""Frozen retrospective HAA5/HAA9 screening and common-support comparisons.

Run after installing requirements. This module never edits authoritative input
observations. See forward_retrospective_protocol.md for the pre-fit declaration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "wr_results" / "haa9_revision"
WORKLOADS = (0.05, 0.10, 0.15, 0.20, 0.30)
SPECIES5 = ("MCAA", "DCAA", "TCAA", "MBAA", "DBAA")
SPECIES9 = SPECIES5 + ("BCAA", "BDCAA", "CDBAA", "TBAA")
SEED = 20260919
B = 60.0


def save(frame: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / f"{name}.csv", index=False, lineterminator="\n")
    frame.to_parquet(OUT / f"{name}.parquet", index=False)


def fold_id(pwsid: str) -> int:
    token = hashlib.sha256(f"wr-forward-retrospective-v1|{pwsid}".encode()).digest()
    return int.from_bytes(token[:8], "big") % 5


def calibrated_cutoff(scores: np.ndarray, target: float) -> tuple[float, float]:
    """Cutoff and common tie probability with exact expected training workload."""
    values, counts = np.unique(scores, return_counts=True)
    values, counts = values[::-1], counts[::-1]
    desired = target * len(scores)
    j = int(np.searchsorted(np.cumsum(counts), desired, side="left"))
    above = int(counts[:j].sum())
    return float(values[j]), float(np.clip((desired - above) / counts[j], 0, 1))


def apply_cutoff(scores: np.ndarray, cutoff: float, tie_probability: float) -> np.ndarray:
    return (scores > cutoff).astype(float) + (scores == cutoff) * tie_probability


def capacity_allocation(scores: np.ndarray, folds: np.ndarray, ids: np.ndarray, target: float) -> np.ndarray:
    """Label-free top-N allocation within an outer evaluation batch."""
    allocated = np.zeros(len(scores), dtype=float)
    hashes = np.array([hashlib.sha256(f"wr-capacity-ties-v1|{value}".encode()).hexdigest() for value in ids])
    for fold in np.unique(folds):
        where = np.where(folds == fold)[0]
        slots = int(np.floor(target * len(where)))
        ranked = np.lexsort((hashes[where], -scores[where]))
        allocated[where[ranked[:slots]]] = 1
    return allocated


def load_cohort() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    events = pd.read_parquet(ROOT / "data/processed/ucmr4/paired_haa_events.parquet")
    events = events.loc[events.finite_grid_system_eligible.fillna(False)].copy()
    events["CollectionDate"] = pd.to_datetime(events.CollectionDate, format="mixed", errors="raise")
    rounds = events.groupby(["PWSID", "SampleEventCode"], sort=True).agg(
        first_date=("CollectionDate", "min"), last_date=("CollectionDate", "max"),
        n_samples=("SampleID", "size"), HAA5=("HAA5", "max"), HAA9=("HAA9", "max")
    ).reset_index()
    nround = rounds.groupby("PWSID").size()
    four = set(nround.index[nround.eq(4)])
    rounds = rounds.loc[rounds.PWSID.isin(four)].sort_values(
        ["PWSID", "first_date", "SampleEventCode"]
    ).copy()
    rounds["round_order"] = rounds.groupby("PWSID").cumcount()
    early = rounds.loc[rounds.round_order.lt(2)].groupby("PWSID").agg(
        predictor_start=("first_date", "min"), predictor_end=("last_date", "max"),
        early_HAA5=("HAA5", "max"), early_HAA9=("HAA9", "max"),
        early_samples=("n_samples", "sum")
    )
    late = rounds.loc[rounds.round_order.ge(2)].groupby("PWSID").agg(
        outcome_start=("first_date", "min"), outcome_end=("last_date", "max"),
        late_HAA5=("HAA5", "max"), late_HAA9=("HAA9", "max"),
        late_samples=("n_samples", "sum")
    )
    cohort = early.join(late, how="inner").reset_index()
    cohort["strict_date_separation"] = cohort.predictor_end < cohort.outcome_start
    save(cohort.loc[~cohort.strict_date_separation], "forward_chronology_excluded_systems")
    cohort = cohort.loc[cohort.strict_date_separation].sort_values("PWSID").reset_index(drop=True)
    cohort["outcome"] = cohort.late_HAA9.gt(B).astype(int)
    cohort["outer_fold"] = cohort.PWSID.map(fold_id)
    early_keys = rounds.loc[rounds.round_order.lt(2), ["PWSID", "SampleEventCode"]]
    early_events = events.merge(early_keys, on=["PWSID", "SampleEventCode"], validate="many_to_one")
    early_events = early_events.loc[early_events.PWSID.isin(cohort.PWSID)].copy()
    species = pd.read_parquet(ROOT / "results/additional/ucmr4_species_event_panel.parquet")
    columns = [f"{s}_lower_ug_L" for s in SPECIES9]
    species_for_join = species[["PWSID", "SampleID", "CollectionDate", "SampleEventCode", *columns]].rename(
        columns={"CollectionDate": "species_collection_date", "SampleEventCode": "species_event_code"})
    joined = early_events.merge(species_for_join,
                                on=["PWSID", "SampleID"], how="left", validate="one_to_one")
    if joined[columns].isna().any().any():
        raise ValueError("Incomplete species mapping; do not silently change model cohort")
    species_dates = pd.to_datetime(joined.species_collection_date, format="mixed", errors="raise")
    assert (species_dates == joined.CollectionDate).all()
    assert (joined.species_event_code == joined.SampleEventCode).all()
    species_h5_error = float(np.max(np.abs(joined[[f"{s}_lower_ug_L" for s in SPECIES5]].sum(axis=1) - joined.HAA5)))
    species_h9_error = float(np.max(np.abs(joined[columns].sum(axis=1) - joined.HAA9)))
    joined["HAA5_species_minus_official"] = joined[[f"{s}_lower_ug_L" for s in SPECIES5]].sum(axis=1) - joined.HAA5
    joined["HAA9_species_minus_official"] = joined[columns].sum(axis=1) - joined.HAA9
    disagreement = joined[["HAA5_species_minus_official", "HAA9_species_minus_official"]].abs().max(axis=1).gt(1e-9)
    # Preserve the archive's previously documented three small reconciliation
    # differences. Official totals remain outcomes/scalar predictors; do not
    # alter species values or drop systems to force exact agreement.
    save(joined.loc[disagreement, ["PWSID", "SampleID", "CollectionDate", "HAA5", "HAA9",
                                  "HAA5_species_minus_official", "HAA9_species_minus_official"]],
         "forward_species_sum_reconciliation_differences")
    feature = joined.groupby("PWSID")[columns].max().rename(
        columns={f"{s}_lower_ug_L": f"early_{s}" for s in SPECIES9}
    )
    cohort = cohort.join(feature, on="PWSID", validate="one_to_one")
    assert cohort.PWSID.is_unique and cohort.strict_date_separation.all()
    assert (cohort.early_HAA9 >= cohort.early_HAA5 - 1e-9).all()
    audit = {"eligible_complete_grid_PWS": int(events.PWSID.nunique()),
             "exact_four_round_PWS": len(four), "strict_chronological_PWS": len(cohort),
             "chronology_excluded_PWS": len(four) - len(cohort),
             "later_HAA9_positive_PWS": int(cohort.outcome.sum()),
             "later_HAA9_negative_PWS": int((1 - cohort.outcome).sum()),
             "early_species_mapped_samples": len(joined),
             "species_dates_and_event_codes_match": True,
             "species_sum_max_error_HAA5_ug_l": species_h5_error,
             "species_sum_max_error_HAA9_ug_l": species_h9_error,
             "species_sum_nonmatching_early_samples_gt_1e_9": int(disagreement.sum()),
             "species_censoring_rule": "detected concentration; zero for nondetects (released lower realization)",
             "predictor_support": "all samples in first two administrative groups",
             "outcome_support": "all samples in last two administrative groups",
             "strict_separation": "latest early date strictly before earliest late date"}
    return cohort, events, audit


def operating_metrics(y: np.ndarray, p: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    positives = weights @ y
    negatives = weights.sum(axis=1) - positives
    tp = weights @ (p * y)
    fp = weights @ (p * (1 - y))
    selected = tp + fp
    return {"sensitivity": tp / positives, "missed_positive_fraction": 1 - tp / positives,
            "positive_predictive_value": np.divide(tp, selected, out=np.full_like(tp, np.nan), where=selected > 0),
            "false_positive_rate": fp / negatives, "followup_fraction": selected / weights.sum(axis=1),
            "missed_positive_count": positives - tp}


def score_metrics(y: np.ndarray, score: np.ndarray, weights: np.ndarray, probability: bool) -> dict[str, np.ndarray]:
    """Weighted exact-tie AUROC/AP; bootstrap counts are case weights."""
    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    start = np.r_[0, np.where(np.diff(sorted_score) != 0)[0] + 1]
    group_pos = np.add.reduceat(weights[:, order] * y[order], start, axis=1)
    group_neg = np.add.reduceat(weights[:, order] * (1 - y[order]), start, axis=1)
    cum_pos, cum_neg = np.cumsum(group_pos, axis=1), np.cumsum(group_neg, axis=1)
    total_pos, total_neg = cum_pos[:, -1], cum_neg[:, -1]
    auc = (group_neg * (cum_pos - 0.5 * group_pos)).sum(axis=1) / (total_pos * total_neg)
    precision = np.divide(cum_pos, cum_pos + cum_neg, out=np.zeros_like(cum_pos), where=(cum_pos + cum_neg) > 0)
    ap = (group_pos * precision).sum(axis=1) / total_pos
    result = {"AUROC": auc, "average_precision": ap}
    if probability:
        clipped = np.clip(score, 1e-12, 1 - 1e-12)
        result["Brier_score"] = weights @ ((score - y) ** 2) / weights.sum(axis=1)
        result["log_loss"] = weights @ (-(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))) / weights.sum(axis=1)
    return result


def interval_row(name: str, metric: str, draws: np.ndarray) -> dict:
    lo, hi = np.nanquantile(draws[1:], [0.025, 0.975])
    return {"strategy": name, "metric": metric, "estimate": float(draws[0]),
            "ci_lower": float(lo), "ci_upper": float(hi)}


def forward(cohort: pd.DataFrame, bootstraps: int) -> dict:
    y = cohort.outcome.to_numpy(float)
    n = len(cohort)
    folds = cohort.outer_fold.to_numpy()
    probabilities = {}
    cutoffs, coefficients, fold_audit = [], [], []
    score_names = ["scalar_HAA5", "scalar_HAA9", "logistic_HAA5_total", "logistic_HAA9_total",
                   "logistic_five_species", "logistic_nine_species"]
    oof_scores = {key: np.zeros(n) for key in score_names}
    model_specs = {"logistic_HAA5_total": ["early_HAA5"],
                   "logistic_HAA9_total": ["early_HAA9"],
                   "logistic_five_species": [f"early_{s}" for s in SPECIES5],
                   "logistic_nine_species": [f"early_{s}" for s in SPECIES9]}
    for f in range(5):
        train, test = folds != f, folds == f
        assert not set(cohort.loc[train, "PWSID"]) & set(cohort.loc[test, "PWSID"])
        fold_audit.append({"outer_fold": f, "training_PWS": int(train.sum()), "evaluation_PWS": int(test.sum()),
                           "training_positive": int(y[train].sum()), "evaluation_positive": int(y[test].sum()),
                           "PWS_overlap": 0, "transforms_fit_on": "outer training PWS only"})
        training_scores = {"scalar_HAA5": cohort.loc[train, "early_HAA5"].to_numpy(),
                           "scalar_HAA9": cohort.loc[train, "early_HAA9"].to_numpy()}
        testing_scores = {"scalar_HAA5": cohort.loc[test, "early_HAA5"].to_numpy(),
                          "scalar_HAA9": cohort.loc[test, "early_HAA9"].to_numpy()}
        for name, cols in model_specs.items():
            x = np.log1p(cohort[cols].to_numpy(float))
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs", max_iter=3000))
            model.fit(x[train], y[train])
            training_scores[name] = model.predict_proba(x[train])[:, 1]
            testing_scores[name] = model.predict_proba(x[test])[:, 1]
            scale = model.named_steps["standardscaler"]
            fitted = model.named_steps["logisticregression"]
            for j, col in enumerate(cols):
                coefficients.append({"outer_fold": f, "strategy": name, "feature": col,
                                     "training_log1p_mean": scale.mean_[j], "training_log1p_sd": scale.scale_[j],
                                     "coefficient": fitted.coef_[0, j], "intercept": fitted.intercept_[0]})
        for name in score_names:
            oof_scores[name][test] = testing_scores[name]
            for workload in WORKLOADS:
                policy = f"{name}_workload_{workload:.2f}"
                cutoff, tie = calibrated_cutoff(training_scores[name], workload)
                p = apply_cutoff(testing_scores[name], cutoff, tie)
                probabilities.setdefault(policy, np.zeros(n))[test] = p
                training_workload = apply_cutoff(training_scores[name], cutoff, tie).mean()
                assert np.isclose(training_workload, workload)
                cutoffs.append({"outer_fold": f, "strategy": name, "target_training_workload": workload,
                                "cutoff": cutoff, "cutoff_tie_probability": tie,
                                "achieved_training_workload": training_workload,
                                "achieved_evaluation_workload": p.mean()})
    for panel in ("HAA5", "HAA9"):
        probabilities[f"direct_{panel}_B60"] = cohort[f"early_{panel}"].gt(B).to_numpy(float)
    for w in WORKLOADS:
        gate = probabilities[f"scalar_HAA5_workload_{w:.2f}"]
        probabilities[f"staged_HAA5_gate_then_HAA9_B60_workload_{w:.2f}"] = gate * probabilities["direct_HAA9_B60"]
    for key, values in oof_scores.items():
        cohort[f"score__{key}"] = values
    for key, values in probabilities.items():
        cohort[f"selection_probability__{key}"] = values
    save(cohort, "forward_out_of_fold_predictions")
    save(pd.DataFrame(fold_audit), "forward_fold_audit")
    save(pd.DataFrame(cutoffs), "comparator_training_cutoffs")
    save(pd.DataFrame(coefficients), "forward_fitted_coefficients")
    rng = np.random.default_rng(SEED)
    weights = np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1/n), size=bootstraps)]).astype(float)
    operating, operating_draws = [], {}
    for key, p in probabilities.items():
        metrics = operating_metrics(y, p, weights)
        operating_draws[key] = metrics
        row = {"strategy": key, "PWS": n, "later_positive_PWS": int(y.sum())}
        for metric, draw in metrics.items():
            result = interval_row(key, metric, draw)
            row[metric] = result["estimate"]
            row[f"{metric}_ci_lower"], row[f"{metric}_ci_upper"] = result["ci_lower"], result["ci_upper"]
        if key.startswith("staged"):
            w = float(key.rsplit("_", 1)[1])
            row["broader_analysis_fraction"] = probabilities[f"scalar_HAA5_workload_{w:.2f}"].mean()
        operating.append(row)
    save(pd.DataFrame(operating), "forward_operating_points")
    scored, score_draws, calibrations = [], {}, []
    for key, score in oof_scores.items():
        metrics = score_metrics(y, score, weights, key.startswith("logistic"))
        score_draws[key] = metrics
        scored += [interval_row(key, metric, value) for metric, value in metrics.items()]
        if key.startswith("logistic"):
            bins = np.minimum((score * 10).astype(int), 9)
            for j in range(10):
                subset = bins == j
                calibrations.append({"strategy": key, "bin_lower": j/10, "bin_upper": (j+1)/10,
                                     "PWS": int(subset.sum()), "positive_PWS": int(y[subset].sum()),
                                     "mean_predicted_probability": float(score[subset].mean()) if subset.any() else np.nan,
                                     "observed_positive_fraction": float(y[subset].mean()) if subset.any() else np.nan})
    save(pd.DataFrame(scored), "forward_score_metrics")
    save(pd.DataFrame(calibrations), "forward_probability_calibration")
    contrast, contrast_specs = [], [("scalar_HAA9", "scalar_HAA5"),
                                   ("logistic_HAA9_total", "logistic_HAA5_total"),
                                   ("logistic_nine_species", "logistic_five_species")]
    for broad, narrow in contrast_specs:
        family = []
        for w in WORKLOADS:
            bkey, nkey = f"{broad}_workload_{w:.2f}", f"{narrow}_workload_{w:.2f}"
            for metric in operating_draws[bkey]:
                d = operating_draws[bkey][metric] - operating_draws[nkey][metric]
                row = interval_row(f"{broad}_minus_{narrow}", metric, d)
                row["target_training_workload"] = w
                contrast.append(row)
                if metric == "sensitivity":
                    family.append(d)
        family = np.array(family).T
        sup = np.quantile(np.max(np.abs(family[1:] - family[0]), axis=1), .95)
        for row in contrast:
            if row["strategy"] == f"{broad}_minus_{narrow}" and row["metric"] == "sensitivity":
                row["simultaneous_ci_lower"] = row["estimate"] - sup
                row["simultaneous_ci_upper"] = row["estimate"] + sup
        for metric in score_draws[broad]:
            d = score_draws[broad][metric] - score_draws[narrow][metric]
            row = interval_row(f"{broad}_minus_{narrow}", metric, d)
            row["target_training_workload"] = np.nan
            contrast.append(row)
    for metric in operating_draws["direct_HAA9_B60"]:
        contrast.append(interval_row("direct_HAA9_B60_minus_direct_HAA5_B60", metric,
                        operating_draws["direct_HAA9_B60"][metric] - operating_draws["direct_HAA5_B60"][metric]))
    save(pd.DataFrame(contrast), "comparator_paired_contrasts")
    # Declared after the primary forward fit: retain separately from its frozen results.
    capacity_points, capacity_draws, capacity_selection = [], {}, cohort[["PWSID", "outcome", "outer_fold"]].copy()
    for name, scores in oof_scores.items():
        for w in WORKLOADS:
            key = f"{name}_capacity_{w:.2f}"
            allocated = capacity_allocation(scores, folds, cohort.PWSID.to_numpy(), w)
            capacity_selection[key] = allocated.astype(int)
            metrics = operating_metrics(y, allocated, weights)
            capacity_draws[key] = metrics
            row = {"strategy": name, "target_capacity": w, "allocated_slots": int(allocated.sum()), "PWS": n}
            for metric, values in metrics.items():
                summary = interval_row(key, metric, values)
                row[metric], row[f"{metric}_ci_lower"], row[f"{metric}_ci_upper"] = summary["estimate"], summary["ci_lower"], summary["ci_upper"]
            capacity_points.append(row)
    save(pd.DataFrame(capacity_points), "forward_exact_capacity_operating_points")
    save(capacity_selection, "forward_exact_capacity_allocations")
    capacity_contrasts = []
    for broad, narrow in contrast_specs:
        family = np.array([capacity_draws[f"{broad}_capacity_{w:.2f}"]["sensitivity"] -
                           capacity_draws[f"{narrow}_capacity_{w:.2f}"]["sensitivity"] for w in WORKLOADS]).T
        sup = np.quantile(np.max(np.abs(family[1:] - family[0]), axis=1), .95)
        for j, w in enumerate(WORKLOADS):
            for metric in capacity_draws[f"{broad}_capacity_{w:.2f}"]:
                d = capacity_draws[f"{broad}_capacity_{w:.2f}"][metric] - capacity_draws[f"{narrow}_capacity_{w:.2f}"][metric]
                row = interval_row(f"{broad}_minus_{narrow}", metric, d)
                row["target_capacity"] = w
                if metric == "sensitivity":
                    row["simultaneous_ci_lower"], row["simultaneous_ci_upper"] = family[0, j] - sup, family[0, j] + sup
                capacity_contrasts.append(row)
    save(pd.DataFrame(capacity_contrasts), "comparator_exact_capacity_paired_contrasts")
    # Rank formula independently checked against library implementations.
    from sklearn.metrics import average_precision_score, roc_auc_score
    for score in oof_scores.values():
        checked = score_metrics(y, score, weights[:1], False)
        assert np.isclose(checked["AUROC"][0], roc_auc_score(y, score))
        assert np.isclose(checked["average_precision"][0], average_precision_score(y, score))
    return {"bootstraps": bootstraps, "seed": SEED, "workloads": list(WORKLOADS),
            "CI_interpretation": "PWS percentile resampling of out-of-fold records conditional on frozen fits/cutoffs; no retraining uncertainty",
            "simultaneous_band": "centered bootstrap sup absolute deviation across five declared sensitivity contrasts per comparator family",
            "staged_strategy": "hypothetical same-sample retained-laboratory reporting only; retention feasibility and cost unverified",
            "transforms": "log1p concentrations in ug/L, outer-training mean and SD",
            "model": "fixed L2 penalized logistic regression C=1, lbfgs, max_iter=3000",
            "exact_capacity_sensitivity": "post-protocol label-free top floor(q*n) out-of-fold scores within each evaluation fold; fixed allocation during bootstrap",
            "packages": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}}


def common_support(events: pd.DataFrame, bootstraps: int) -> dict:
    source = events.copy()
    source["quarter"] = source.CollectionDate.dt.to_period("Q")
    quarterly = source.groupby(["PWSID", "site_id", "quarter"], sort=True).agg(
        HAA5_mean=("HAA5", "mean"), HAA9_mean=("HAA9", "mean"),
        HAA5_max=("HAA5", "max"), HAA9_max=("HAA9", "max"),
        n_samples=("SampleID", "size")
    ).reset_index()
    window_rows = []
    for (pwsid, site), g in quarterly.groupby(["PWSID", "site_id"], sort=True):
        g = g.sort_values("quarter").reset_index(drop=True)
        ordinal = np.array([q.ordinal for q in g.quarter])
        for start in range(len(g) - 3):
            if not np.array_equal(ordinal[start:start+4], np.arange(ordinal[start], ordinal[start]+4)):
                continue
            block = g.iloc[start:start+4]
            row = {"PWSID": pwsid, "site_id": site, "first_quarter": str(block.quarter.iloc[0]),
                   "last_quarter": str(block.quarter.iloc[-1]), "n_quarters": 4,
                   "physical_samples": int(block.n_samples.sum())}
            for p in ("HAA5", "HAA9"):
                q = block[f"{p}_mean"].to_numpy(float)
                row[f"maximum_{p}"] = float(block[f"{p}_max"].max())
                row[f"OEL_form_{p}"] = float(max((q[0]+q[1]+2*q[2])/4, (q[1]+q[2]+2*q[3])/4))
                row[f"LRAA_form_{p}"] = float(q.mean())
            window_rows.append(row)
    windows = pd.DataFrame(window_rows)
    save(windows, "common_support_four_quarter_windows")
    cols = [f"{endpoint}_{p}" for endpoint in ("maximum", "OEL_form", "LRAA_form") for p in ("HAA5", "HAA9")]
    systems = windows.groupby("PWSID")[cols].max().reset_index()
    support = windows.groupby("PWSID").agg(qualifying_sites=("site_id", "nunique"), qualifying_windows=("site_id", "size"))
    systems = systems.join(support, on="PWSID")
    save(systems, "common_support_system_endpoints")
    reference = systems.maximum_HAA9.gt(B).to_numpy()
    rows, gap_rows, gap_draws = [], [], {}
    rng = np.random.default_rng(SEED + 35)
    weights = np.vstack([np.ones(len(systems)), rng.multinomial(len(systems), np.full(len(systems), 1/len(systems)), size=bootstraps)])
    for endpoint in ("maximum", "OEL_form", "LRAA_form"):
        ref_specific = systems[f"{endpoint}_HAA9"].gt(B).to_numpy()
        h5 = systems[f"{endpoint}_HAA5"].gt(B).to_numpy()
        h9 = ref_specific
        assert not np.any(h5 & ~h9)
        for panel, positive in [("HAA5", h5), ("HAA9", h9)]:
            for reference_name, ref in [("all_common_support_PWS", np.ones(len(systems), dtype=bool)),
                                        ("endpoint_specific_HAA9_positive", ref_specific),
                                        ("fixed_maximum_HAA9_positive", reference)]:
                denominator = weights @ ref.astype(float)
                draw = (weights @ (positive & ref).astype(float)) / denominator
                lo, hi = np.quantile(draw[1:], [.025, .975])
                rows.append({"endpoint": endpoint, "panel": panel, "reference": reference_name,
                             "common_support_PWS": len(systems), "reference_PWS": int(ref.sum()),
                             "positive_in_reference_PWS": int((positive & ref).sum()),
                             "capture_or_positive_fraction": float(draw[0]), "ci_lower": float(lo), "ci_upper": float(hi)})
        for reference_name, ref in [("all_common_support_PWS", np.ones(len(systems), dtype=bool)),
                                    ("endpoint_specific_HAA9_positive", ref_specific),
                                    ("fixed_maximum_HAA9_positive", reference)]:
            difference = ((h9 & ref).astype(float) - (h5 & ref).astype(float))
            draw = (weights @ difference) / (weights @ ref.astype(float))
            gap_draws[(endpoint, reference_name)] = draw
            row = interval_row(f"{endpoint}_HAA9_minus_HAA5", "positive_fraction_difference", draw)
            row["endpoint"], row["reference"], row["reference_PWS"] = endpoint, reference_name, int(ref.sum())
            gap_rows.append(row)
    save(pd.DataFrame(rows), "common_support_endpoint_summary")
    save(pd.DataFrame(gap_rows), "common_support_paired_panel_gaps")
    changes = []
    for reference_name in ("all_common_support_PWS", "fixed_maximum_HAA9_positive"):
        for endpoint in ("OEL_form", "LRAA_form"):
            difference = gap_draws[(endpoint, reference_name)] - gap_draws[("maximum", reference_name)]
            row = interval_row(f"{endpoint}_gap_minus_maximum_gap", "change_in_panel_gap", difference)
            row["reference"] = reference_name
            changes.append(row)
    save(pd.DataFrame(changes), "common_support_aggregation_gap_changes")
    return {"common_support_PWS": len(systems), "qualifying_PWS_sites": len(windows[["PWSID", "site_id"]].drop_duplicates()),
            "qualifying_four_quarter_windows": len(windows), "maximum_HAA9_positive_PWS": int(reference.sum()),
            "quarterly_input": "site arithmetic mean of eligible exact-matched samples",
            "OEL_form": "maximum of the two three-quarter OEL-form values within each common four-quarter window",
            "maximum": "physical-sample maximum using exactly the same site and four-quarter window support",
            "legal_status": "all OEL-form/LRAA-form values are research analogues, not compliance determinations"}


def self_tests() -> dict:
    # Whole tie set receives one probability; evaluation labels never enter this function.
    score = np.array([8., 8., 4., 1.])
    cut, tie = calibrated_cutoff(score, .25)
    assert cut == 8 and tie == .5
    assert np.isclose(apply_cutoff(score, cut, tie).mean(), .25)
    y = np.array([1., 0., 1., 0.])
    m = operating_metrics(y, np.array([1., 0., .5, 0.]), np.ones((1, 4)))
    assert m["sensitivity"][0] == .75 and m["false_positive_rate"][0] == 0
    from sklearn.metrics import average_precision_score, roc_auc_score
    rank = score_metrics(y, score, np.ones((1, 4)), False)
    assert np.isclose(rank["AUROC"][0], roc_auc_score(y, score))
    assert np.isclose(rank["average_precision"][0], average_precision_score(y, score))
    return {"tie_workload_fixture": True, "operating_metric_fixture": True, "rank_tie_fixture": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstraps", type=int, default=5000)
    args = parser.parse_args()
    tests = self_tests()
    cohort, events, audit = load_cohort()
    print(json.dumps(audit, indent=2), flush=True)
    settings = forward(cohort, args.bootstraps)
    common = common_support(events, args.bootstraps)
    report = {"forward_cohort": audit, "forward_settings": settings, "common_support": common, "self_tests": tests,
              "protocol_sha256": hashlib.sha256((OUT / "forward_retrospective_protocol.md").read_bytes()).hexdigest(),
              "postprotocol_capacity_sensitivity_sha256": hashlib.sha256((OUT / "forward_postprotocol_capacity_sensitivity.md").read_bytes()).hexdigest()}
    (OUT / "forward_common_support_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
