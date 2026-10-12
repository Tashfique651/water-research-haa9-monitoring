"""Analyses requested by the Water Research revision playbook.

This module is additive. It reads the frozen public-data products already in the
repository and writes only to ``wr_results/playbook_extension``. It does not
alter the released primary analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "ucmr4"
OUT_DEFAULT = ROOT / "wr_results" / "playbook_extension"
SPECIES_DIR = (
    ROOT
    / "data"
    / "raw"
    / "additional"
    / "peterson_haa"
    / "python"
    / "data"
    / "UCMR4 FOIA"
)
POPULATION_FILE = ROOT / "data" / "interim" / "extracted" / "SYR4_Water_system_table.txt"
THRESHOLD = 60.0
SEED = 20260914
BOOTSTRAPS = 5_000
SIGN_FLIPS = 100_000

SPECIES = {
    "monobromoacetic acid": "MBAA",
    "monochloroacetic acid": "MCAA",
    "dibromoacetic acid": "DBAA",
    "dichloroacetic acid": "DCAA",
    "bromochloroacetic acid": "BCAA",
    "tribromoacetic acid": "TBAA",
    "trichloroacetic acid": "TCAA",
    "chlorodibromoacetic acid": "CDBAA",
    "bromodichloroacetic acid": "BDCAA",
}
HAA5_SPECIES = frozenset(["MBAA", "MCAA", "DBAA", "DCAA", "TCAA"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    parser.add_argument("--sign-flips", type=int, default=SIGN_FLIPS)
    return parser.parse_args()


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def save_table(frame: pd.DataFrame, stem: str, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{stem}.csv"
    parquet_path = out / f"{stem}.parquet"
    frame.to_csv(csv_path, index=False)
    frame.to_parquet(parquet_path, index=False)
    return [csv_path, parquet_path]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = pd.read_parquet(PROCESSED / "paired_haa_events.parquet")
    events["CollectionDate"] = pd.to_datetime(events["CollectionDate"], errors="raise")
    events = events.loc[events["finite_grid_system_eligible"].fillna(False)].copy()
    full = pd.read_parquet(PROCESSED / "system_full_grid_panel_maxima.parquet")
    contributions = pd.read_parquet(PROCESSED / "system_contributions_primary_B60.parquet")
    indicators = pd.read_parquet(PROCESSED / "indicator_system_round_summary.parquet")
    return events, full, contributions, indicators


def build_round_units(events: pd.DataFrame) -> pd.DataFrame:
    units = (
        events.groupby(["PWSID", "SampleEventCode"], sort=False)
        .agg(
            HAA5=("HAA5", "max"),
            HAA6Br=("HAA6Br", "max"),
            HAA9=("HAA9", "max"),
            first_date=("CollectionDate", "min"),
            last_date=("CollectionDate", "max"),
            representative_date=("CollectionDate", "median"),
            physical_samples=("SampleID", "nunique"),
            sites=("site_id", "nunique"),
        )
        .reset_index()
    )
    units["month"] = units["representative_date"].dt.month.astype(int)
    units["warm_season"] = units["month"].between(5, 9)
    return units


def derived_metrics(p: np.ndarray) -> dict[str, float]:
    p00, p10, p01, p11 = [float(x) for x in p]
    delta = p11 - p10 - p01 + p00
    multiplicative = p11 * p00 / (p10 * p01) if p10 > 0 and p01 > 0 else np.nan
    reri = delta / p00 if p00 > 0 else np.nan
    attributable_proportion = delta / p11 if p11 > 0 else np.nan
    denominator = p10 + p01 - 2 * p00
    synergy = (p11 - p00) / denominator if denominator != 0 else np.nan
    return {
        "p00": p00,
        "p10": p10,
        "p01": p01,
        "p11": p11,
        "additive_interaction": delta,
        "multiplicative_interaction_ratio": multiplicative,
        "relative_excess_risk_due_to_interaction": reri,
        "attributable_proportion_due_to_interaction": attributable_proportion,
        "synergy_index": synergy,
    }


def bootstrap_metrics(
    values: np.ndarray,
    seed: int,
    reps: int,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        raise ValueError("Cannot bootstrap an empty analysis set")
    weights_array = None if weights is None else np.asarray(weights, dtype=float)
    if weights_array is None:
        point_p = values.mean(axis=0)
    else:
        point_p = np.average(values, axis=0, weights=weights_array)
    point = derived_metrics(point_p)
    names = list(point)
    draws = np.empty((reps, len(names)), dtype=float)
    rng = np.random.default_rng(seed)
    batch_size = max(1, min(200, 2_000_000 // n))
    cursor = 0
    while cursor < reps:
        batch = min(batch_size, reps - cursor)
        indices = rng.integers(0, n, size=(batch, n))
        sampled = values[indices]
        if weights_array is None:
            p_draws = sampled.mean(axis=1)
        else:
            sampled_weights = weights_array[indices]
            p_draws = (sampled * sampled_weights[:, :, None]).sum(axis=1) / sampled_weights.sum(axis=1)[:, None]
        for row_index, p in enumerate(p_draws):
            draws[cursor + row_index] = list(derived_metrics(p).values())
        cursor += batch
    limits = np.nanquantile(draws, [0.025, 0.975], axis=0)
    result: dict[str, float] = {}
    for index, name in enumerate(names):
        result[name] = float(point[name])
        result[f"{name}_ci_lower"] = float(limits[0, index])
        result[f"{name}_ci_upper"] = float(limits[1, index])
    return result


def wild_sign_flip_test(values: np.ndarray, seed: int, reps: int) -> dict[str, float | int | str]:
    values = np.asarray(values, dtype=float)
    observed = float(values.mean())
    centered = values - observed
    rng = np.random.default_rng(seed)
    extreme_two = 0
    extreme_upper = 0
    batch_size = max(1, min(1_000, 4_000_000 // len(values)))
    cursor = 0
    while cursor < reps:
        batch = min(batch_size, reps - cursor)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(batch, len(values)))
        null_means = (signs * centered[None, :]).mean(axis=1)
        extreme_two += int((np.abs(null_means) >= abs(observed)).sum())
        extreme_upper += int((null_means >= observed).sum())
        cursor += batch
    return {
        "observed_additive_interaction": observed,
        "systems": int(len(values)),
        "replicates": int(reps),
        "two_sided_p_value": (extreme_two + 1) / (reps + 1),
        "upper_tailed_p_value": (extreme_upper + 1) / (reps + 1),
        "method": "cluster wild sign-flip test of a zero mean system contribution",
        "assumption": "Rademacher sign exchangeability of centered system contributions under the null",
    }


def interaction_scale_analysis(
    contributions: pd.DataFrame,
    rounds: pd.DataFrame,
    reps: int,
    sign_flips: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = contributions.loc[contributions["y11"].eq(1)].copy()
    rows = []
    result = bootstrap_metrics(primary[["y00", "y10", "y01", "y11"]].to_numpy(), SEED + 1, reps)
    rows.append({"analysis": "primary_complete_observed_grid_B60", "systems": len(primary), **result})
    tests = [{"analysis": "primary_complete_observed_grid_B60", **wild_sign_flip_test(primary["interaction"], SEED + 2, sign_flips)}]

    ordered = rounds.sort_values(["PWSID", "representative_date", "SampleEventCode"]).copy()
    counts = ordered.groupby("PWSID").size()
    exact_ids = set(counts[counts.eq(4)].index)
    ordered = ordered.loc[ordered["PWSID"].isin(exact_ids)].copy()
    ordered["order"] = ordered.groupby("PWSID").cumcount()
    direction_specs = [
        ("late2_condition_early2_evaluation", {2, 3}, {0, 1}),
        ("early2_condition_late2_evaluation", {0, 1}, {2, 3}),
    ]
    detail = []
    for direction_index, (label, condition_orders, evaluation_orders) in enumerate(direction_specs):
        cond = ordered.loc[ordered["order"].isin(condition_orders)]
        condition_positive = cond.groupby("PWSID")["HAA9"].max().gt(THRESHOLD)
        evaluation = ordered.loc[ordered["order"].isin(evaluation_orders)].groupby("PWSID").agg(
            y00=("HAA5", lambda x: float(np.mean(x.to_numpy() > THRESHOLD))),
            y10=("HAA9", lambda x: float(np.mean(x.to_numpy() > THRESHOLD))),
            y01=("HAA5", lambda x: float(np.max(x.to_numpy()) > THRESHOLD)),
            y11=("HAA9", lambda x: float(np.max(x.to_numpy()) > THRESHOLD)),
        )
        evaluation = evaluation.loc[evaluation.index.intersection(condition_positive[condition_positive].index)].copy()
        evaluation["interaction"] = evaluation.y11 - evaluation.y10 - evaluation.y01 + evaluation.y00
        evaluation = evaluation.reset_index()
        evaluation.insert(1, "analysis", label)
        detail.append(evaluation)
        metrics = bootstrap_metrics(
            evaluation[["y00", "y10", "y01", "y11"]].to_numpy(),
            SEED + 10 + direction_index,
            reps,
        )
        rows.append({"analysis": label, "systems": len(evaluation), **metrics})
        tests.append({"analysis": label, **wild_sign_flip_test(evaluation["interaction"], SEED + 20 + direction_index, sign_flips)})
    return pd.DataFrame(rows), pd.DataFrame(tests), pd.concat(detail, ignore_index=True)


def summarize_policy_rows(frame: pd.DataFrame, reps: int) -> pd.DataFrame:
    output = []
    for index, (policy, group) in enumerate(frame.groupby("policy", sort=False)):
        metrics = bootstrap_metrics(group[["y00", "y10", "y01", "y11"]].to_numpy(), SEED + 100 + index, reps)
        output.append(
            {
                "policy": policy,
                "systems": len(group),
                "systems_with_warm_season_round": int(group["has_warm_round"].sum()),
                "systems_with_TOC_round": int(group["has_TOC_round"].sum()),
                "selection_is_outcome_informed": bool(group["selection_is_outcome_informed"].iloc[0]),
                **metrics,
            }
        )
    return pd.DataFrame(output)


def operational_round_policies(
    rounds: pd.DataFrame,
    full: pd.DataFrame,
    indicators: pd.DataFrame,
    reps: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positive_ids = set(full.loc[full.HAA9_full_max.gt(THRESHOLD), "PWSID"])
    source = rounds.loc[rounds.PWSID.isin(positive_ids)].copy()
    toc = indicators.loc[indicators.Contaminant.eq("total organic carbon")].copy()
    toc = toc[
        [
            "PWSID",
            "SampleEventCode",
            "value_lower_median_ug_l",
            "value_midpoint_median_ug_l",
            "value_upper_median_ug_l",
        ]
    ].rename(
        columns={
            "value_lower_median_ug_l": "toc_lower",
            "value_midpoint_median_ug_l": "toc_midpoint",
            "value_upper_median_ug_l": "toc_upper",
        }
    )
    source = source.merge(toc, on=["PWSID", "SampleEventCode"], how="left", validate="many_to_one")
    full_lookup = full.set_index("PWSID")
    rows = []
    for pwsid, group in source.groupby("PWSID", sort=False):
        group = group.sort_values(["representative_date", "SampleEventCode"])
        warm = group.loc[group.warm_season]
        warm_candidates = warm if len(warm) else group
        policy_candidates: list[tuple[str, pd.DataFrame, bool]] = [
            ("random_event_code_group", group, False),
            ("warm_season_May_to_September_else_random", warm_candidates, False),
            ("first_chronological_event_code_group", group.iloc[[0]], False),
        ]
        max_h9 = group.HAA9.max()
        policy_candidates.append(("best_observed_HAA9_round_oracle", group.loc[group.HAA9.eq(max_h9)], True))
        for toc_column, suffix in [("toc_midpoint", "midpoint"), ("toc_lower", "lower_bound"), ("toc_upper", "upper_bound")]:
            observed = group.loc[group[toc_column].notna()]
            if len(observed):
                maximum = observed[toc_column].max()
                selected = observed.loc[np.isclose(observed[toc_column], maximum)]
                policy_candidates.append((f"highest_TOC_{suffix}", selected, False))
        for policy, selected, outcome_informed in policy_candidates:
            rows.append(
                {
                    "PWSID": pwsid,
                    "policy": policy,
                    "selected_candidate_rounds": len(selected),
                    "observed_rounds": len(group),
                    "has_warm_round": bool(group.warm_season.any()),
                    "has_TOC_round": bool(group.toc_midpoint.notna().any()),
                    "selection_is_outcome_informed": outcome_informed,
                    "y00": float(np.mean(selected.HAA5.to_numpy() > THRESHOLD)),
                    "y10": float(np.mean(selected.HAA9.to_numpy() > THRESHOLD)),
                    "y01": float(full_lookup.loc[pwsid, "HAA5_full_max"] > THRESHOLD),
                    "y11": 1.0,
                }
            )
    detail = pd.DataFrame(rows)
    toc_ids = set(detail.loc[detail.policy.eq("highest_TOC_midpoint"), "PWSID"])
    matched_random = detail.loc[
        detail.policy.eq("random_event_code_group") & detail.PWSID.isin(toc_ids)
    ].copy()
    matched_random["policy"] = "random_event_code_group_TOC_complete_case"
    detail = pd.concat([detail, matched_random], ignore_index=True)
    return detail, summarize_policy_rows(detail, reps)


def round_policy_contrasts(detail: pd.DataFrame, reps: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(SEED + 150)
    policies = [
        policy
        for policy in detail.policy.unique()
        if policy not in {"random_event_code_group", "random_event_code_group_TOC_complete_case"}
    ]
    for policy in policies:
        baseline = (
            "random_event_code_group_TOC_complete_case"
            if policy.startswith("highest_TOC")
            else "random_event_code_group"
        )
        selected = detail.loc[detail.policy.eq(policy), ["PWSID", "y00", "y10"]]
        reference = detail.loc[detail.policy.eq(baseline), ["PWSID", "y00", "y10"]]
        paired = selected.merge(reference, on="PWSID", suffixes=("_policy", "_random"), validate="one_to_one")
        differences = np.column_stack(
            [
                paired.y00_policy - paired.y00_random,
                paired.y10_policy - paired.y10_random,
                (paired.y00_policy - paired.y00_random) - (paired.y10_policy - paired.y10_random),
            ]
        )
        indices = rng.integers(0, len(paired), size=(reps, len(paired)))
        draws = differences[indices].mean(axis=1)
        limits = np.quantile(draws, [0.025, 0.975], axis=0)
        rows.append(
            {
                "policy": policy,
                "baseline_policy": baseline,
                "paired_systems": len(paired),
                "HAA5_detection_difference": float(differences[:, 0].mean()),
                "HAA5_difference_ci_lower": float(limits[0, 0]),
                "HAA5_difference_ci_upper": float(limits[1, 0]),
                "HAA9_detection_difference": float(differences[:, 1].mean()),
                "HAA9_difference_ci_lower": float(limits[0, 1]),
                "HAA9_difference_ci_upper": float(limits[1, 1]),
                "additive_interaction_difference": float(differences[:, 2].mean()),
                "interaction_difference_ci_lower": float(limits[0, 2]),
                "interaction_difference_ci_upper": float(limits[1, 2]),
            }
        )
    return pd.DataFrame(rows)


def hypergeom_capture(total: int, positive: int, selected: int) -> float:
    selected = max(0, min(int(selected), int(total)))
    if selected == 0 or positive == 0:
        return 0.0
    if total - positive < selected:
        return 1.0
    return 1.0 - math.comb(total - positive, selected) / math.comb(total, selected)


def equivalence_frontier(events: pd.DataFrame, reps: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    physical = (
        events.groupby(["PWSID", "SampleID"], sort=False)
        .agg(HAA5=("HAA5", "max"), HAA9=("HAA9", "max"))
        .reset_index()
    )
    counts = physical.groupby("PWSID").agg(
        available_samples=("SampleID", "size"),
        HAA5_positive_samples=("HAA5", lambda x: int((x.to_numpy() > THRESHOLD).sum())),
        HAA9_positive_samples=("HAA9", lambda x: int((x.to_numpy() > THRESHOLD).sum())),
    )
    counts = counts.loc[counts.HAA9_positive_samples.gt(0)].copy()
    counts["HAA5_full_record_detected"] = counts.HAA5_positive_samples.gt(0).astype(float)
    max_samples = int(counts.available_samples.max())
    ks = np.arange(0, max_samples + 1)
    capture_matrix = np.empty((len(counts), len(ks)), dtype=float)
    for row_index, row in enumerate(counts.itertuples()):
        capture_matrix[row_index] = [
            hypergeom_capture(row.available_samples, row.HAA9_positive_samples, int(k)) for k in ks
        ]
    benchmark = counts.HAA5_full_record_detected.to_numpy(float)
    rng = np.random.default_rng(SEED + 200)
    boot_capture = np.empty((reps, len(ks)), dtype=float)
    boot_benchmark = np.empty(reps, dtype=float)
    n = len(counts)
    batch_size = max(1, min(128, 1_500_000 // n))
    cursor = 0
    while cursor < reps:
        batch = min(batch_size, reps - cursor)
        indices = rng.integers(0, n, size=(batch, n))
        boot_capture[cursor : cursor + batch] = capture_matrix[indices].mean(axis=1)
        boot_benchmark[cursor : cursor + batch] = benchmark[indices].mean(axis=1)
        cursor += batch
    mean_capture = capture_matrix.mean(axis=0)
    curve = pd.DataFrame(
        {
            "HAA9_samples_selected": ks,
            "HAA9_conditional_capture": mean_capture,
            "HAA9_capture_ci_lower": np.quantile(boot_capture, 0.025, axis=0),
            "HAA9_capture_ci_upper": np.quantile(boot_capture, 0.975, axis=0),
            "full_HAA5_record_capture": float(benchmark.mean()),
            "paired_difference_vs_full_HAA5": mean_capture - benchmark.mean(),
            "paired_difference_ci_lower": np.quantile(boot_capture - boot_benchmark[:, None], 0.025, axis=0),
            "paired_difference_ci_upper": np.quantile(boot_capture - boot_benchmark[:, None], 0.975, axis=0),
            "reference_positive_systems": n,
        }
    )
    minimum_k = int(curve.loc[curve.HAA9_conditional_capture.ge(benchmark.mean()), "HAA9_samples_selected"].min())

    ratios = np.round(np.arange(1.0, 10.0001, 0.01), 2)
    cost_matrix = np.empty((n, len(ratios)), dtype=float)
    for row_index, row in enumerate(counts.itertuples()):
        for col_index, ratio in enumerate(ratios):
            expected_k = row.available_samples / ratio
            lower = min(row.available_samples, int(math.floor(expected_k)))
            upper = min(row.available_samples, lower + 1)
            fraction = min(1.0, max(0.0, expected_k - lower))
            low_capture = hypergeom_capture(row.available_samples, row.HAA9_positive_samples, lower)
            high_capture = hypergeom_capture(row.available_samples, row.HAA9_positive_samples, upper)
            cost_matrix[row_index, col_index] = low_capture + fraction * (high_capture - low_capture)
    cost_mean = cost_matrix.mean(axis=0)
    frontier = pd.DataFrame(
        {
            "HAA9_to_HAA5_per_sample_cost_ratio": ratios,
            "equal_expected_budget_HAA9_capture": cost_mean,
            "full_HAA5_record_capture": benchmark.mean(),
            "difference": cost_mean - benchmark.mean(),
        }
    )
    eligible_ratios = frontier.loc[frontier.difference.ge(0), "HAA9_to_HAA5_per_sample_cost_ratio"]
    critical_ratio = float(eligible_ratios.max()) if len(eligible_ratios) else np.nan
    crossing_draws = np.empty(reps, dtype=float)
    rng = np.random.default_rng(SEED + 201)
    cursor = 0
    while cursor < reps:
        batch = min(batch_size, reps - cursor)
        indices = rng.integers(0, n, size=(batch, n))
        differences = cost_matrix[indices].mean(axis=1) - benchmark[indices].mean(axis=1)[:, None]
        for j in range(batch):
            valid = ratios[differences[j] >= 0]
            crossing_draws[cursor + j] = valid.max() if len(valid) else np.nan
        cursor += batch
    summary = {
        "reference_positive_systems": n,
        "mean_full_record_physical_samples": float(counts.available_samples.mean()),
        "median_full_record_physical_samples": float(counts.available_samples.median()),
        "full_HAA5_record_capture": float(benchmark.mean()),
        "minimum_integer_HAA9_samples_matching_full_HAA5_capture": minimum_k,
        "HAA9_capture_at_matching_k": float(curve.loc[curve.HAA9_samples_selected.eq(minimum_k), "HAA9_conditional_capture"].iloc[0]),
        "paired_difference_at_matching_k": float(curve.loc[curve.HAA9_samples_selected.eq(minimum_k), "paired_difference_vs_full_HAA5"].iloc[0]),
        "HAA9_capture_at_k4": float(curve.loc[curve.HAA9_samples_selected.eq(4), "HAA9_conditional_capture"].iloc[0]),
        "HAA9_capture_at_k4_ci_lower": float(curve.loc[curve.HAA9_samples_selected.eq(4), "HAA9_capture_ci_lower"].iloc[0]),
        "HAA9_capture_at_k4_ci_upper": float(curve.loc[curve.HAA9_samples_selected.eq(4), "HAA9_capture_ci_upper"].iloc[0]),
        "paired_difference_k4_minus_full_HAA5": float(curve.loc[curve.HAA9_samples_selected.eq(4), "paired_difference_vs_full_HAA5"].iloc[0]),
        "paired_difference_k4_ci_lower": float(curve.loc[curve.HAA9_samples_selected.eq(4), "paired_difference_ci_lower"].iloc[0]),
        "paired_difference_k4_ci_upper": float(curve.loc[curve.HAA9_samples_selected.eq(4), "paired_difference_ci_upper"].iloc[0]),
        "critical_HAA9_to_HAA5_cost_ratio": critical_ratio,
        "critical_cost_ratio_ci_lower": float(np.nanquantile(crossing_draws, 0.025)),
        "critical_cost_ratio_ci_upper": float(np.nanquantile(crossing_draws, 0.975)),
        "cost_interpretation": "Equal expected analytical budget, using system-specific observed sample counts and fractional randomization between adjacent integer HAA9 sample counts",
    }
    return counts.reset_index(), curve, frontier, summary


def resolve_population_table() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(POPULATION_FILE, sep="\t", dtype={"PWSID": str}, encoding="latin1")
    duplicate_rows = int(raw.PWSID.duplicated(keep=False).sum())
    raw = raw.sort_values(
        ["PWSID", "Adjusted Total Population Served", "Retail Population Served"],
        ascending=[True, False, False],
        na_position="last",
    ).drop_duplicates("PWSID", keep="first")
    keep = raw[
        [
            "PWSID",
            "Retail Population Served",
            "Adjusted Total Population Served",
            "Source Water Type",
            "State Code",
        ]
    ].rename(
        columns={
            "Retail Population Served": "retail_population_served",
            "Adjusted Total Population Served": "adjusted_total_population_served",
            "Source Water Type": "syr4_source_water_type",
            "State Code": "syr4_state_code",
        }
    )
    return keep, {
        "source_rows": int(len(raw) + duplicate_rows // 2),
        "duplicate_member_rows": duplicate_rows,
        "duplicate_resolution": "kept the row with the largest adjusted population, then largest retail population",
    }


def population_weighting(contributions: pd.DataFrame, population: pd.DataFrame, reps: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    merged = contributions.merge(population, on="PWSID", how="left", validate="many_to_one")
    ref = merged.loc[merged.y11.eq(1)].copy()
    rows = []
    schemes = [
        ("system_equal_complete_case", "adjusted_total_population_served", False, False),
        ("adjusted_population", "adjusted_total_population_served", True, False),
        ("retail_population", "retail_population_served", True, False),
        ("adjusted_population_capped_at_complete_case_p99", "adjusted_total_population_served", True, True),
    ]
    for index, (label, column, weighted, cap) in enumerate(schemes):
        subset = ref.loc[ref[column].notna() & ref[column].gt(0)].copy()
        weights = subset[column].to_numpy(float)
        if cap:
            weights = np.minimum(weights, np.quantile(weights, 0.99))
        metrics = bootstrap_metrics(
            subset[["y00", "y10", "y01", "y11"]].to_numpy(),
            SEED + 300 + index,
            reps,
            weights if weighted else None,
        )
        rows.append(
            {
                "weighting_scheme": label,
                "systems": len(subset),
                "covered_population": float(weights.sum()),
                "effective_sample_size": float(weights.sum() ** 2 / np.square(weights).sum()) if weighted else float(len(subset)),
                "largest_system_weight_share": float(weights.max() / weights.sum()) if weighted else 1 / len(subset),
                **metrics,
            }
        )
    coverage = {
        "reference_positive_systems": int(len(ref)),
        "adjusted_population_available_positive_systems": int(ref.adjusted_total_population_served.notna().sum()),
        "positive_system_coverage_fraction": float(ref.adjusted_total_population_served.notna().mean()),
        "eligible_systems": int(len(merged)),
        "adjusted_population_available_eligible_systems": int(merged.adjusted_total_population_served.notna().sum()),
        "eligible_system_coverage_fraction": float(merged.adjusted_total_population_served.notna().mean()),
        "interpretation": "Secondary complete-case sensitivity using EPA SYR4 system populations, not a population-complete UCMR4 estimate",
    }
    return merged, pd.DataFrame(rows), coverage


def source_group(value: str) -> str:
    if value in {"GW", "GU"}:
        return "groundwater_or_under_influence"
    if value == "SW":
        return "surface_water"
    return "mixed_or_multiple"


def build_mechanism_table(
    events: pd.DataFrame,
    full: pd.DataFrame,
    contributions: pd.DataFrame,
    indicators: pd.DataFrame,
    population: pd.DataFrame,
) -> pd.DataFrame:
    indicator_system = (
        indicators.groupby(["PWSID", "Contaminant"], sort=False)
        .agg(
            indicator_midpoint_ug_l=("value_midpoint_median_ug_l", "median"),
            indicator_lower_ug_l=("value_lower_median_ug_l", "median"),
            indicator_upper_ug_l=("value_upper_median_ug_l", "median"),
            indicator_rounds=("SampleEventCode", "nunique"),
        )
        .reset_index()
    )
    mid = indicator_system.pivot(index="PWSID", columns="Contaminant", values="indicator_midpoint_ug_l").rename(
        columns={"bromide": "bromide_midpoint_ug_l", "total organic carbon": "toc_midpoint_ug_l"}
    )
    composition = events.groupby("PWSID").agg(
        median_extra4_fraction=("f_extra", "median"),
        median_brominated_fraction=("f_brominated", "median"),
        median_extra4_ug_l=("extra4_ug_l", "median"),
        maximum_extra4_ug_l=("extra4_ug_l", "max"),
        EPA_region=("Region", lambda values: "|".join(sorted(set(values.dropna().astype(str))))),
    )
    round_means = events.groupby(["PWSID", "SampleEventCode"], sort=False).HAA9.mean().rename("HAA9_round_mean").reset_index()
    temporal = round_means.groupby("PWSID").agg(
        HAA9_round_mean=("HAA9_round_mean", "mean"),
        HAA9_round_sd=("HAA9_round_mean", "std"),
        HAA9_rounds_for_cv=("HAA9_round_mean", "size"),
    )
    temporal["haa9_round_cv_eligible"] = temporal.HAA9_rounds_for_cv.ge(2) & temporal.HAA9_round_mean.gt(0)
    temporal["haa9_round_cv"] = np.where(
        temporal.haa9_round_cv_eligible,
        temporal.HAA9_round_sd / temporal.HAA9_round_mean,
        np.nan,
    )
    table = contributions.merge(
        full[
            [
                "PWSID",
                "observed_round_count",
                "observed_site_count",
                "facility_water_type_profile",
                "size_profile",
                "state_profile",
            ]
        ],
        on="PWSID",
        how="left",
        suffixes=("", "_full"),
        validate="one_to_one",
    )
    table = table.merge(mid.reset_index(), on="PWSID", how="left", validate="one_to_one")
    table = table.merge(composition.reset_index(), on="PWSID", how="left", validate="one_to_one")
    table = table.merge(temporal.reset_index(), on="PWSID", how="left", validate="one_to_one")
    table = table.merge(population, on="PWSID", how="left", validate="many_to_one")
    table = table.loc[table.y11.eq(1)].copy()
    table["haa5_invisible"] = table.y01.eq(0).astype(int)
    table["both_required_any"] = table.cat_both_required.gt(0).astype(int)
    table["source_group"] = table.facility_water_type_profile.map(source_group)
    table["round_count_band"] = pd.cut(
        table.observed_round_count,
        bins=[0, 3, 4, np.inf],
        labels=["1_to_3", "4", "more_than_4"],
    ).astype(str)
    table["site_count_band"] = pd.cut(
        table.observed_site_count,
        bins=[0, 3, 5, np.inf],
        labels=["1_to_3", "4_to_5", "6_or_more"],
    ).astype(str)
    for column, label in [
        ("bromide_midpoint_ug_l", "bromide_quartile"),
        ("toc_midpoint_ug_l", "toc_quartile"),
        ("median_extra4_fraction", "extra4_fraction_quartile"),
        ("adjusted_total_population_served", "population_quartile"),
    ]:
        valid = table[column].notna()
        quartile_codes = pd.qcut(table.loc[valid, column], 4, labels=False, duplicates="drop")
        table[label] = "missing"
        table.loc[valid, label] = quartile_codes.map(lambda value: f"Q{int(value) + 1}").astype(str)
        table.loc[table[column].isna(), label] = "missing"
    return table


def logistic_models(table: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.api as sm

    rows = []
    specifications = [
        (
            "environmental_indicators",
            ["bromide_midpoint_ug_l", "toc_midpoint_ug_l"],
            ["source_group", "size_profile"],
        ),
        (
            "composition_descriptor",
            ["median_extra4_fraction"],
            ["source_group", "size_profile"],
        ),
    ]
    for outcome in ["haa5_invisible", "both_required_any"]:
        for spec, continuous, categorical in specifications:
            required = [outcome, *continuous, *categorical, "observed_round_count", "observed_site_count"]
            data = table[required].dropna().copy()
            design_parts = []
            labels = []
            scale_notes = {}
            for column in continuous:
                transformed = np.log2(data[column] + 1) if column.endswith("_ug_l") else data[column].astype(float)
                iqr = float(transformed.quantile(0.75) - transformed.quantile(0.25))
                if iqr <= 0:
                    iqr = 1.0
                name = f"{column}_per_IQR"
                design_parts.append(pd.Series((transformed - transformed.median()) / iqr, index=data.index, name=name))
                labels.append(name)
                scale_notes[name] = iqr
            for column in ["observed_round_count", "observed_site_count"]:
                transformed = np.log2(data[column].astype(float))
                name = f"{column}_per_doubling"
                design_parts.append(pd.Series(transformed, index=data.index, name=name))
                labels.append(name)
                scale_notes[name] = 1.0
            for column in categorical:
                reference = "surface_water" if column == "source_group" else "L"
                dummies = pd.get_dummies(data[column].astype(str), prefix=column, dtype=float)
                reference_column = f"{column}_{reference}"
                if reference_column in dummies:
                    dummies = dummies.drop(columns=reference_column)
                for name in dummies.columns:
                    design_parts.append(dummies[name])
                    labels.append(name)
                    scale_notes[name] = 1.0
            x = pd.concat(design_parts, axis=1).astype(float)
            x = sm.add_constant(x, has_constant="add")
            y = data[outcome].astype(float)
            try:
                fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(cov_type="HC0")
                for term in x.columns:
                    estimate = float(fit.params[term])
                    se = float(fit.bse[term])
                    rows.append(
                        {
                            "outcome": outcome,
                            "model": spec,
                            "term": term,
                            "analysis_systems": len(data),
                            "outcome_events": int(y.sum()),
                            "coefficient_log_odds": estimate,
                            "robust_standard_error": se,
                            "adjusted_odds_ratio": math.exp(estimate),
                            "odds_ratio_ci_lower": math.exp(estimate - 1.96 * se),
                            "odds_ratio_ci_upper": math.exp(estimate + 1.96 * se),
                            "p_value": float(fit.pvalues[term]),
                            "AIC": float(fit.aic),
                            "scaling": "IQR of log2(x+1)" if term.endswith("_ug_l_per_IQR") else ("IQR" if term.endswith("_per_IQR") else "indicator or log2 doubling"),
                            "scale_value": scale_notes.get(term, 1.0),
                        }
                    )
            except Exception as exc:
                rows.append(
                    {
                        "outcome": outcome,
                        "model": spec,
                        "term": "MODEL_FAILURE",
                        "analysis_systems": len(data),
                        "outcome_events": int(y.sum()),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return pd.DataFrame(rows)


def heterogeneity_summaries(table: pd.DataFrame, reps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    variables = [
        "source_group",
        "size_profile",
        "EPA_region",
        "round_count_band",
        "site_count_band",
        "bromide_quartile",
        "toc_quartile",
        "extra4_fraction_quartile",
        "population_quartile",
    ]
    rng = np.random.default_rng(SEED + 400)
    for variable in variables:
        for level, group in table.groupby(variable, dropna=False, sort=False):
            if len(group) < 20 or str(level) == "missing":
                continue
            values = group[["interaction", "haa5_invisible", "cat_both_required"]].to_numpy(float)
            indices = rng.integers(0, len(group), size=(reps, len(group)))
            draws = values[indices].mean(axis=1)
            limits = np.quantile(draws, [0.025, 0.975], axis=0)
            rows.append(
                {
                    "stratifier": variable,
                    "level": str(level),
                    "systems": len(group),
                    "additive_interaction": float(values[:, 0].mean()),
                    "interaction_ci_lower": float(limits[0, 0]),
                    "interaction_ci_upper": float(limits[1, 0]),
                    "HAA5_invisible_fraction": float(values[:, 1].mean()),
                    "HAA5_invisible_ci_lower": float(limits[0, 1]),
                    "HAA5_invisible_ci_upper": float(limits[1, 1]),
                    "both_required_probability": float(values[:, 2].mean()),
                    "both_required_ci_lower": float(limits[0, 2]),
                    "both_required_ci_upper": float(limits[1, 2]),
                }
            )
    valid_state_codes = {
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN",
        "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV",
        "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
        "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
    }
    map_source = table.loc[table.state_profile.isin(valid_state_codes)].copy()
    states = (
        map_source.groupby("state_profile", dropna=False)
        .agg(
            systems=("PWSID", "size"),
            additive_interaction=("interaction", "mean"),
            HAA5_invisible_fraction=("haa5_invisible", "mean"),
            both_required_probability=("cat_both_required", "mean"),
            adjusted_population_covered=("adjusted_total_population_served", "sum"),
        )
        .reset_index()
    )
    return pd.DataFrame(rows), states


def geography_coverage(table: pd.DataFrame) -> dict:
    valid_state_codes = {
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN",
        "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV",
        "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
        "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
    }
    valid = table.state_profile.isin(valid_state_codes)
    excluded = table.loc[~valid, "state_profile"].value_counts(dropna=False).to_dict()
    return {
        "reference_positive_systems": int(len(table)),
        "systems_with_valid_50_state_or_DC_code": int(valid.sum()),
        "valid_state_code_coverage_fraction": float(valid.mean()),
        "excluded_state_or_regional_codes": {str(key): int(value) for key, value in excluded.items()},
        "mapping_rule": "Only USPS codes for the 50 states and District of Columbia were retained. Numeric EPA regional or tribal codes and territories were not reassigned without a verified crosswalk.",
    }


def load_species_matrix() -> tuple[pd.DataFrame, dict]:
    pieces = []
    source_rows = 0
    for path in sorted(SPECIES_DIR.glob("*.xlsx")):
        frame = pd.read_excel(path, dtype={"PWSID": str, "SampleID": str})
        result_column = [c for c in frame.columns if str(c).startswith("AnalyticalResultValue")][0]
        source_rows += len(frame)
        piece = frame[["PWSID", "SampleID", "SampleEventCode", "Contaminant", "AnalyticalResultsSign", result_column]].copy()
        piece["species"] = piece.Contaminant.astype(str).str.lower().map(SPECIES)
        piece["value"] = pd.to_numeric(piece[result_column], errors="coerce").where(piece.AnalyticalResultsSign.eq("="), 0.0).fillna(0.0)
        piece["SampleEventCode"] = piece.SampleEventCode.astype(str).str.upper().str.replace(r"^SEH", "SE", regex=True)
        pieces.append(piece[["PWSID", "SampleID", "SampleEventCode", "species", "value"]])
    long = pd.concat(pieces, ignore_index=True)
    if long.species.isna().any():
        raise ValueError("Unexpected species name in FOIA workbooks")
    duplicates = int(long.duplicated(["SampleID", "species"]).sum())
    matrix = long.pivot(index=["PWSID", "SampleID", "SampleEventCode"], columns="species", values="value").reset_index()
    species_columns = sorted(SPECIES.values())
    complete = matrix[species_columns].notna().all(axis=1)
    matrix = matrix.loc[complete].copy()
    return matrix, {
        "source_rows": source_rows,
        "unique_events": int(long.SampleID.nunique()),
        "complete_nine_species_events": int(complete.sum()),
        "duplicate_species_event_rows": duplicates,
    }


def rolling_endpoint_scores(frame: pd.DataFrame, panels: list[str]) -> pd.DataFrame:
    source = frame.copy()
    source["quarter"] = source.CollectionDate.dt.to_period("Q")
    quarterly = source.groupby(["PWSID", "site_id", "quarter"], sort=False)[panels].mean().reset_index()
    rows = []
    for (pwsid, site_id), group in quarterly.groupby(["PWSID", "site_id"], sort=False):
        group = group.sort_values("quarter")
        ordinals = group.quarter.map(lambda x: x.ordinal).to_numpy()
        values = group[panels].to_numpy(float)
        for endpoint, width in [("OEL_analogue", 3), ("LRAA_analogue", 4)]:
            for start in range(0, len(group) - width + 1):
                if not np.array_equal(ordinals[start : start + width], np.arange(ordinals[start], ordinals[start] + width)):
                    continue
                window = values[start : start + width]
                scores = (window[0] + window[1] + 2 * window[2]) / 4 if endpoint == "OEL_analogue" else window.mean(axis=0)
                row = {"PWSID": pwsid, "site_id": site_id, "endpoint": endpoint, "end_quarter": str(group.quarter.iloc[start + width - 1])}
                row.update({panel: float(scores[j]) for j, panel in enumerate(panels)})
                rows.append(row)
    windows = pd.DataFrame(rows)
    if windows.empty:
        return windows
    maxima = windows.groupby(["PWSID", "endpoint"])[panels].max().reset_index()
    return maxima


def endpoint_and_panel_analysis(
    events: pd.DataFrame,
    species: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    joined = events.merge(
        species[["PWSID", "SampleID", "SampleEventCode", "BCAA"]],
        on=["PWSID", "SampleID", "SampleEventCode"],
        how="left",
        validate="one_to_one",
    )
    if joined.BCAA.isna().any():
        raise ValueError("Complete released HAA events did not map one-to-one to complete FOIA species events")
    joined["HAA5_plus_BCAA"] = joined.HAA5 + joined.BCAA
    panels = ["HAA5", "HAA5_plus_BCAA", "HAA6Br", "HAA9"]
    single = joined.groupby("PWSID")[panels].max().reset_index()
    single["endpoint"] = "single_sample_maximum"
    rolling = rolling_endpoint_scores(joined[["PWSID", "site_id", "CollectionDate", *panels]], panels)
    detail = pd.concat([single[["PWSID", "endpoint", *panels]], rolling], ignore_index=True, sort=False)
    rows = []
    for endpoint, group in detail.groupby("endpoint", sort=False):
        h9_positive = group.HAA9.gt(THRESHOLD)
        haa5_positive = group.HAA5.gt(THRESHOLD)
        for panel in panels:
            positive = group[panel].gt(THRESHOLD)
            rows.append(
                {
                    "endpoint": endpoint,
                    "panel": panel,
                    "supported_systems": len(group),
                    "positive_systems": int(positive.sum()),
                    "positive_fraction": float(positive.mean()),
                    "HAA9_benchmark_positive_systems": int(h9_positive.sum()),
                    "capture_among_HAA9_positive_systems": float(positive[h9_positive].mean()) if h9_positive.any() else np.nan,
                    "additional_positive_systems_vs_HAA5": int((positive & ~haa5_positive).sum()),
                }
            )
    qc = {
        "mapped_events": int(len(joined)),
        "single_sample_supported_systems": int(single.PWSID.nunique()),
        "OEL_analogue_supported_systems": int(rolling.loc[rolling.endpoint.eq("OEL_analogue"), "PWSID"].nunique()),
        "LRAA_analogue_supported_systems": int(rolling.loc[rolling.endpoint.eq("LRAA_analogue"), "PWSID"].nunique()),
        "quarterly_value": "arithmetic mean of complete UCMR4 samples at a site within a calendar quarter",
        "OEL_formula": "(Q[t-2] + Q[t-1] + 2*Q[t]) / 4",
        "warning": "Research analogues only. UCMR4 is not Stage 2 compliance monitoring and HAA9 has no 60 ug/L MCL.",
    }
    return detail, pd.DataFrame(rows), qc


def exact_five_species_panel_analysis(
    events: pd.DataFrame,
    full: pd.DataFrame,
    species: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    aligned = events.merge(species, on=["PWSID", "SampleID", "SampleEventCode"], how="inner", validate="one_to_one")
    if len(aligned) != len(events):
        raise ValueError("Species matrix does not fully cover finite-grid events")
    species_names = sorted(SPECIES.values())
    event_matrix = aligned[species_names].to_numpy(float)
    round_keys = aligned.PWSID.astype(str) + "|" + aligned.SampleEventCode.astype(str)
    round_codes, unique_rounds = pd.factorize(round_keys, sort=False)
    round_pwsids = pd.Series(unique_rounds).str.split("|", n=1).str[0].to_numpy()
    system_ids = full.PWSID.astype(str).to_numpy()
    system_map = {pwsid: index for index, pwsid in enumerate(system_ids)}
    round_system_codes = np.array([system_map[pwsid] for pwsid in round_pwsids], dtype=int)
    round_counts = np.bincount(round_system_codes, minlength=len(system_ids))
    h9_round_max = np.full(len(unique_rounds), -np.inf)
    np.maximum.at(h9_round_max, round_codes, aligned.HAA9.to_numpy(float))
    h9_round_positive = h9_round_max > THRESHOLD
    h9_round_probability = np.bincount(round_system_codes, weights=h9_round_positive, minlength=len(system_ids)) / round_counts
    ref = full.HAA9_full_max.to_numpy(float) > THRESHOLD
    rows = []
    for combination in itertools.combinations(species_names, 5):
        columns = [species_names.index(name) for name in combination]
        event_values = event_matrix[:, columns].sum(axis=1)
        pseudo_round_max = np.full(len(unique_rounds), -np.inf)
        np.maximum.at(pseudo_round_max, round_codes, event_values)
        pseudo_positive = pseudo_round_max > THRESHOLD
        pseudo_round_probability = np.bincount(round_system_codes, weights=pseudo_positive, minlength=len(system_ids)) / round_counts
        pseudo_full = np.zeros(len(system_ids), dtype=float)
        np.maximum.at(pseudo_full, round_system_codes, pseudo_round_max)
        p = np.array(
            [
                pseudo_round_probability[ref].mean(),
                h9_round_probability[ref].mean(),
                (pseudo_full[ref] > THRESHOLD).mean(),
                1.0,
            ]
        )
        rows.append(
            {
                "five_species_panel": ";".join(combination),
                "is_regulatory_HAA5": frozenset(combination) == HAA5_SPECIES,
                **derived_metrics(p),
            }
        )
    panels = pd.DataFrame(rows).sort_values("additive_interaction", ascending=False).reset_index(drop=True)
    panels["descending_interaction_rank"] = np.arange(1, len(panels) + 1)
    actual = panels.loc[panels.is_regulatory_HAA5].iloc[0]
    duplicate_panel = np.column_stack(
        [
            contributions_like(events, full, "HAA5", "HAA5")[name]
            for name in ["y00", "y10", "y01", "y11"]
        ]
    )
    nested_metrics = derived_metrics(duplicate_panel[ref].mean(axis=0))
    controls = pd.DataFrame(
        [
            {
                "control": "duplicate_HAA5_scope_for_A0_and_A1",
                "reference_cohort": "HAA9_full_grid_positive_at_60_ug_l",
                **nested_metrics,
            }
        ]
    )
    summary = {
        "same_size_panels_enumerated": int(len(panels)),
        "regulatory_HAA5_interaction": float(actual.additive_interaction),
        "regulatory_HAA5_descending_rank": int(actual.descending_interaction_rank),
        "regulatory_HAA5_percentile_among_same_size_panels": float((panels.additive_interaction <= actual.additive_interaction).mean()),
        "interpretation": "Exact same-size species-set sensitivity. Species are not exchangeable, so the rank is not a formal randomization p-value.",
        "nested_null_interaction": float(controls.iloc[0].additive_interaction),
    }
    return panels, controls, summary


def contributions_like(events: pd.DataFrame, full: pd.DataFrame, panel0: str, panel1: str) -> pd.DataFrame:
    rounds = events.groupby(["PWSID", "SampleEventCode"], sort=False).agg(
        panel0=(panel0, "max"), panel1=(panel1, "max")
    ).reset_index()
    system = rounds.groupby("PWSID").agg(
        y00=("panel0", lambda x: float(np.mean(x.to_numpy() > THRESHOLD))),
        y10=("panel1", lambda x: float(np.mean(x.to_numpy() > THRESHOLD))),
        y01=("panel0", lambda x: float(np.max(x.to_numpy()) > THRESHOLD)),
        y11=("panel1", lambda x: float(np.max(x.to_numpy()) > THRESHOLD)),
    ).reindex(full.PWSID)
    return system.reset_index(drop=True)


def threshold_sensitivity(rounds: pd.DataFrame, full: pd.DataFrame, reps: int) -> pd.DataFrame:
    base_ids = set(full.loc[full.HAA9_full_max.gt(THRESHOLD), "PWSID"])
    full_index = full.set_index("PWSID")
    rows = []
    for index, threshold in enumerate(np.arange(20.0, 100.1, 5.0)):
        system = rounds.groupby("PWSID").agg(
            y00=("HAA5", lambda x, b=threshold: float(np.mean(x.to_numpy() > b))),
            y10=("HAA9", lambda x, b=threshold: float(np.mean(x.to_numpy() > b))),
        )
        system["y01"] = full_index.loc[system.index, "HAA5_full_max"].gt(threshold).astype(float)
        system["y11"] = full_index.loc[system.index, "HAA9_full_max"].gt(threshold).astype(float)
        for cohort, ids in [
            ("fixed_B60_HAA9_positive", base_ids),
            ("threshold_specific_HAA9_positive", set(system.index[system.y11.eq(1)])),
        ]:
            subset = system.loc[system.index.intersection(ids)]
            metrics = bootstrap_metrics(subset[["y00", "y10", "y01", "y11"]].to_numpy(), SEED + 500 + index * 2 + (cohort.startswith("threshold")), reps)
            rows.append({"threshold_ug_l": threshold, "cohort": cohort, "systems": len(subset), **metrics})
    return pd.DataFrame(rows)


def write_report(
    out: Path,
    feasibility: pd.DataFrame,
    scales: pd.DataFrame,
    policies: pd.DataFrame,
    equivalence: dict,
    population: pd.DataFrame,
    panel_summary: pd.DataFrame,
    endpoint_qc: dict,
    species_qc: dict,
    negative_summary: dict,
) -> Path:
    primary = scales.loc[scales.analysis.eq("primary_complete_observed_grid_B60")].iloc[0]
    chrono = scales.loc[scales.analysis.eq("late2_condition_early2_evaluation")].iloc[0]
    random_row = policies.loc[policies.policy.eq("random_event_code_group")].iloc[0]
    warm_row = policies.loc[policies.policy.eq("warm_season_May_to_September_else_random")].iloc[0]
    toc_row = policies.loc[policies.policy.eq("highest_TOC_midpoint")].iloc[0]
    pop_row = population.loc[population.weighting_scheme.eq("adjusted_population")].iloc[0]
    single_h5 = panel_summary.loc[(panel_summary.endpoint.eq("single_sample_maximum")) & (panel_summary.panel.eq("HAA5"))].iloc[0]
    single_h9 = panel_summary.loc[(panel_summary.endpoint.eq("single_sample_maximum")) & (panel_summary.panel.eq("HAA9"))].iloc[0]
    feasibility_header = "| " + " | ".join(feasibility.columns) + " |"
    feasibility_rule = "| " + " | ".join(["---"] * len(feasibility.columns)) + " |"
    feasibility_rows = [
        "| " + " | ".join(str(value).replace("|", "/") for value in row) + " |"
        for row in feasibility.itertuples(index=False, name=None)
    ]
    feasibility_markdown = "\n".join([feasibility_header, feasibility_rule, *feasibility_rows])
    text = f"""# Water Research playbook analysis extension

Legacy numerical layer. Editorial recommendations below are historical and
superseded by the HAA9 revision in wr_results/haa9_revision/. In particular,
the four-sample mean crossing does not establish equivalence or superiority.
Use the complete paired budget family and current manuscript for interpretation.

## Scope and reproducibility

This additive analysis uses only frozen public data already present in the repository. The official EPA UCMR4 HAA5, HAA6Br, and HAA9 totals remain authoritative. The frozen Peterson et al. FOIA records supply simultaneous individual-species values for BCAA and the exact same-size panel sensitivity. The EPA SYR4 system table supplies a secondary, incomplete population-served sensitivity. All random procedures use seed {SEED}. Cluster bootstrap intervals use the public water system as the resampling unit.

## Feasibility decisions

{feasibility_markdown}

## Main findings

The complete-observed-grid estimand retained an additive interaction of {primary.additive_interaction:.4f} (95% bootstrap interval {primary.additive_interaction_ci_lower:.4f} to {primary.additive_interaction_ci_upper:.4f}). Interaction depended on scale. The multiplicative interaction ratio was {primary.multiplicative_interaction_ratio:.3f} ({primary.multiplicative_interaction_ratio_ci_lower:.3f} to {primary.multiplicative_interaction_ratio_ci_upper:.3f}), while the relative excess risk due to interaction was {primary.relative_excess_risk_due_to_interaction:.3f}. The chronological late-condition, early-evaluation estimate was {chrono.additive_interaction:.4f} ({chrono.additive_interaction_ci_lower:.4f} to {chrono.additive_interaction_ci_upper:.4f}). The wild sign-flip results are reported separately and require sign-exchangeability under the null.

The prespecified May-to-September selector produced HAA9 one-round detection of {warm_row.p10:.4f}, compared with {random_row.p10:.4f} under uniform round selection. The highest-TOC policy produced {toc_row.p10:.4f} among {int(toc_row.systems)} TOC-complete benchmark-positive systems. The TOC result is not hydraulically paired because UCMR4 indicators were joined only by system and event code.

At the physical-sample scale, {equivalence['minimum_integer_HAA9_samples_matching_full_HAA5_capture']} uniformly selected HAA9 samples achieved conditional capture {equivalence['HAA9_capture_at_matching_k']:.4f}, compared with {equivalence['full_HAA5_record_capture']:.4f} for the entire observed HAA5 record. At exactly four samples, the paired difference was {equivalence['paired_difference_k4_minus_full_HAA5']:.4f} (whole-system bootstrap interval {equivalence['paired_difference_k4_ci_lower']:.4f} to {equivalence['paired_difference_k4_ci_upper']:.4f}). The benchmark-positive systems had a mean of {equivalence['mean_full_record_physical_samples']:.2f} physical samples. Under the explicitly hypothetical equal-expected-budget calculation, the estimated HAA9-to-HAA5 per-sample cost-ratio crossing was {equivalence['critical_HAA9_to_HAA5_cost_ratio']:.2f} (bootstrap interval {equivalence['critical_cost_ratio_ci_lower']:.2f} to {equivalence['critical_cost_ratio_ci_upper']:.2f}). This is a design frontier, not a dollar cost-effectiveness estimate.

The adjusted-population complete-case estimate used {int(pop_row.systems)} benchmark-positive systems and {pop_row.covered_population:,.0f} covered population. Its additive interaction was {pop_row.additive_interaction:.4f} ({pop_row.additive_interaction_ci_lower:.4f} to {pop_row.additive_interaction_ci_upper:.4f}). Coverage is incomplete, so this result cannot be described as nationally population representative.

Across all complete finite-grid systems, the single-sample-maximum endpoint identified {int(single_h5.positive_systems)} HAA5-positive systems and {int(single_h9.positive_systems)} HAA9-positive systems. The rolling research analogues retained {endpoint_qc['OEL_analogue_supported_systems']} systems for the three-quarter operational-evaluation formula and {endpoint_qc['LRAA_analogue_supported_systems']} systems for four consecutive quarters. These are not compliance determinations.

The species audit mapped {species_qc['complete_nine_species_events']:,} complete nine-species events. Exact enumeration of {negative_summary['same_size_panels_enumerated']} five-species panels placed regulatory HAA5 at descending interaction rank {negative_summary['regulatory_HAA5_descending_rank']}; this is a composition sensitivity, not a species-exchangeability test. The duplicate-panel nested-null control returned additive interaction {negative_summary['nested_null_interaction']:.6f}.

## Interpretation limits

1. The complete-grid cohort is conditioned on an observed HAA9 maximum above 60 ug/L. It estimates conditional detection, not national prevalence or compliance.
2. UCMR4 event codes are administrative groups and can span more than one collection date. Warm-season selection uses the median collection date within each event-code group.
3. TOC and bromide are source-side indicators joined by PWSID and event code. They are not asserted to be hydraulically paired with a distribution-system HAA sample.
4. Population weighting covers only systems found in the SYR4 water-system table and is a complete-case sensitivity.
5. LRAA and operational-evaluation calculations require consecutive calendar quarters at a site, use quarterly arithmetic means, and are research analogues only.
6. Same-size panel enumeration preserves measured species but does not make species exchangeable. It supplies a design sensitivity rather than a formal causal or randomization null.
7. The highest-HAA9 round is an outcome-informed oracle and cannot be implemented prospectively.
"""
    path = out / "METHODS_RESULTS_LIMITATIONS.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    generated: list[Path] = []
    events, full, contributions, indicators = load_inputs()
    rounds = build_round_units(events)

    scales, sign_tests, chronological_detail = interaction_scale_analysis(contributions, rounds, args.bootstraps, args.sign_flips)
    generated += save_table(scales, "interaction_scale_metrics", out)
    generated += save_table(sign_tests, "interaction_wild_sign_flip_tests", out)
    generated += save_table(chronological_detail, "chronological_transfer_systems", out)

    policy_detail, policy_summary = operational_round_policies(rounds, full, indicators, args.bootstraps)
    policy_contrasts = round_policy_contrasts(policy_detail, args.bootstraps)
    generated += save_table(policy_detail, "round_policy_systems", out)
    generated += save_table(policy_summary, "round_policy_summary", out)
    generated += save_table(policy_contrasts, "round_policy_contrasts", out)

    equivalence_systems, equivalence_curve, cost_frontier, equivalence_summary = equivalence_frontier(events, args.bootstraps)
    generated += save_table(equivalence_systems, "physical_sample_equivalence_systems", out)
    generated += save_table(equivalence_curve, "physical_sample_equivalence_curve", out)
    generated += save_table(cost_frontier, "cost_ratio_frontier", out)
    equivalence_path = out / "equivalence_summary.json"
    equivalence_path.write_text(json.dumps(json_ready(equivalence_summary), indent=2), encoding="utf-8")
    generated.append(equivalence_path)

    population_table, population_qc = resolve_population_table()
    population_systems, population_summary, population_coverage = population_weighting(contributions, population_table, args.bootstraps)
    generated += save_table(population_systems, "population_weighting_systems", out)
    generated += save_table(population_summary, "population_weighting_summary", out)
    population_meta = {**population_qc, **population_coverage}
    population_meta_path = out / "population_weighting_coverage.json"
    population_meta_path.write_text(json.dumps(json_ready(population_meta), indent=2), encoding="utf-8")
    generated.append(population_meta_path)

    mechanism = build_mechanism_table(events, full, contributions, indicators, population_table)
    models = logistic_models(mechanism)
    heterogeneity, state_map = heterogeneity_summaries(mechanism, min(args.bootstraps, 2_000))
    geography_qc = geography_coverage(mechanism)
    generated += save_table(mechanism, "mechanism_systems", out)
    generated += save_table(models, "mechanism_logistic_models", out)
    generated += save_table(heterogeneity, "heterogeneity_summary", out)
    generated += save_table(state_map, "state_map_values", out)
    geography_path = out / "state_map_coverage.json"
    geography_path.write_text(json.dumps(json_ready(geography_qc), indent=2), encoding="utf-8")
    generated.append(geography_path)

    species, species_qc = load_species_matrix()
    endpoint_detail, endpoint_summary, endpoint_qc = endpoint_and_panel_analysis(events, species)
    generated += save_table(endpoint_detail, "regulation_shaped_endpoint_systems", out)
    generated += save_table(endpoint_summary, "regulation_shaped_endpoint_summary", out)

    five_panels, negative_controls, negative_summary = exact_five_species_panel_analysis(events, full, species)
    generated += save_table(five_panels, "exact_five_species_panel_sensitivity", out)
    generated += save_table(negative_controls, "negative_controls", out)

    threshold = threshold_sensitivity(rounds, full, args.bootstraps)
    generated += save_table(threshold, "fixed_B60_and_changing_cohort_thresholds", out)

    feasibility = pd.DataFrame(
        [
            {"analysis": "A1 interaction scales and inference", "status": "implemented", "recommendation": "KEEP IN MAIN TEXT", "qualification": "Report additive and multiplicative scales together. Wild sign-flip p-values are secondary and require sign exchangeability."},
            {"analysis": "A2 operational round policies", "status": "implemented", "recommendation": "KEEP WARM-SEASON RESULT IN MAIN; MOVE TOC AND ORACLE TO SI", "qualification": "Warm season is prespecified; TOC has partial, non-hydraulic event-code linkage; HAA9 oracle is nonprospective."},
            {"analysis": "A3 mechanism and heterogeneity", "status": "implemented", "recommendation": "KEEP CONCISE BROMIDE/TOC AND COMPOSITION RESULT IN MAIN; FULL MODELS IN SI", "qualification": "Bromide and TOC models are complete-case associations; composition predictors are descriptive."},
            {"analysis": "A4 equivalence and cost-ratio frontier", "status": "implemented", "recommendation": "KEEP FOUR-SAMPLE EQUIVALENCE IN MAIN; KEEP COST-RATIO FRONTIER AS SCENARIO SENSITIVITY", "qualification": "Uniform physical-sample selection and hypothetical relative costs, not observed dollars."},
            {"analysis": "A5 LRAA and OEL analogues plus panel candidates", "status": "implemented", "recommendation": "KEEP ONE COMPACT MAIN-TEXT PANEL; PUT FULL WINDOWS AND DISCLAIMERS IN SI", "qualification": "Research analogues only; consecutive-quarter support is incomplete; HAA5+BCAA uses frozen species data."},
            {"analysis": "A6 population weighting", "status": "implemented as sensitivity", "recommendation": "KEEP IN SI AND MENTION DIRECTION IN MAIN", "qualification": "EPA SYR4 population coverage is incomplete for UCMR4 systems."},
            {"analysis": "A7 map", "status": "data implemented", "recommendation": "DROP FROM MAIN UNLESS SPACE ALLOWS; USE IN SI", "qualification": "State-level values are supplied for plotting; small state counts require visual suppression or annotation."},
            {"analysis": "A8 fixed-B60 cohort threshold sensitivity", "status": "implemented", "recommendation": "KEEP FIXED-COHORT CURVE IN MAIN ROBUSTNESS FIGURE", "qualification": "Reported beside threshold-specific cohorts to distinguish estimand drift."},
            {"analysis": "A9 negative controls", "status": "implemented", "recommendation": "KEEP NESTED NULL IN SI; USE SAME-SIZE PANEL ENUMERATION AS SI SENSITIVITY", "qualification": "Duplicate-panel null is exact; same-size species enumeration is a sensitivity, not a formal exchangeability test."},
            {"analysis": "Highest-temperature round policy", "status": "not implemented", "recommendation": "DROP", "qualification": "No co-located temperature variable exists; calendar season is used without fabricated climate data."},
            {"analysis": "National population-representative inference", "status": "not supportable", "recommendation": "DROP NATIONAL CLAIM", "qualification": "The available population join covers only a subset of eligible systems."},
        ]
    )
    generated += save_table(feasibility, "analysis_feasibility", out)

    metadata = {
        "analysis_version": "wr-playbook-extension-v1",
        "seed": SEED,
        "bootstrap_replicates": args.bootstraps,
        "wild_sign_flip_replicates": args.sign_flips,
        "threshold_ug_l": THRESHOLD,
        "runtime_seconds": time.time() - started,
        "species_qc": species_qc,
        "endpoint_qc": endpoint_qc,
        "population_qc": population_meta,
        "state_map_qc": geography_qc,
        "equivalence_summary": equivalence_summary,
        "negative_control_summary": negative_summary,
    }
    metadata_path = out / "analysis_summary.json"
    metadata_path.write_text(json.dumps(json_ready(metadata), indent=2), encoding="utf-8")
    generated.append(metadata_path)

    report = write_report(
        out,
        feasibility,
        scales,
        policy_summary,
        equivalence_summary,
        population_summary,
        endpoint_summary,
        endpoint_qc,
        species_qc,
        negative_summary,
    )
    generated.append(report)
    manifest = pd.DataFrame(
        [
            {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(set(generated))
        ]
    )
    manifest_path = out / "generation_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(json.dumps(json_ready(metadata), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
