"""Water Research extensions for the DBP monitoring-design analysis.

This module leaves the frozen core analysis untouched. It adds the
round-semantics audit, fixed-support analyses, threshold uncertainty,
cross-fitted HAA5 screening, resource-equivalent scenarios, severity and
persistence summaries, and a disjoint-grid validation requested for the Water
Research version.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "ucmr4"
OUT = ROOT / "wr_results" / "core"
THRESHOLD = 60.0
THRESHOLDS = np.arange(20.0, 100.0 + 0.001, 5.0)
BOOTSTRAPS = 10_000
SEED = 20260914


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float


def save_table(frame: pd.DataFrame, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / f"{stem}.csv", index=False, lineterminator="\n")
    frame.to_parquet(OUT / f"{stem}.parquet", index=False)


def sha_fold(value: str, salt: str, folds: int = 5) -> int:
    token = hashlib.sha256(f"{salt}|{value}".encode("utf-8")).digest()
    return int.from_bytes(token[:8], "big") % folds


def mean_bootstrap(values: np.ndarray, seed: int, reps: int = BOOTSTRAPS) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    n, width = values.shape
    if n == 0:
        return np.full(width, np.nan), np.full((2, width), np.nan)
    rng = np.random.default_rng(seed)
    draws = np.empty((reps, width), dtype=float)
    batch = 250
    offset = 0
    while offset < reps:
        take = min(batch, reps - offset)
        idx = rng.integers(0, n, size=(take, n))
        draws[offset : offset + take] = values[idx].mean(axis=1)
        offset += take
    estimate = values.mean(axis=0)
    limits = np.quantile(draws, [0.025, 0.975], axis=0)
    return estimate, limits


def median_bootstrap(values: np.ndarray, seed: int, reps: int = BOOTSTRAPS) -> BootstrapInterval:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = np.empty(reps, dtype=float)
    batch = 250
    offset = 0
    while offset < reps:
        take = min(batch, reps - offset)
        idx = rng.integers(0, n, size=(take, n))
        draws[offset : offset + take] = np.median(values[idx], axis=1)
        offset += take
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return BootstrapInterval(float(np.median(values)), float(lo), float(hi))


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_complete_events = pd.read_parquet(PROCESSED / "paired_haa_events.parquet")
    all_complete_events["CollectionDate"] = pd.to_datetime(
        all_complete_events["CollectionDate"], errors="coerce"
    )
    all_complete_events["collection_year"] = all_complete_events["CollectionDate"].dt.year.astype("Int16")
    all_complete_events["collection_quarter"] = all_complete_events["CollectionDate"].dt.quarter.astype("Int8")
    events = all_complete_events.loc[
        all_complete_events["finite_grid_system_eligible"].fillna(False)
    ].copy()
    full = pd.read_parquet(PROCESSED / "system_full_grid_panel_maxima.parquet")
    contributions = pd.read_parquet(PROCESSED / "system_contributions_primary_B60.parquet")
    return events, full, contributions, all_complete_events


def aggregate_units(events: pd.DataFrame, definition: str) -> pd.DataFrame:
    if definition == "event_code":
        keys = ["PWSID", "SampleEventCode"]
    elif definition == "event_code_year":
        keys = ["PWSID", "SampleEventCode", "collection_year"]
    elif definition == "collection_date":
        keys = ["PWSID", "CollectionDate"]
    elif definition == "facility_event_code_year":
        keys = ["PWSID", "FacilityID", "SampleEventCode", "collection_year"]
    elif definition == "physical_sample":
        keys = ["PWSID", "SampleID", "CollectionDate", "site_id"]
    else:
        raise ValueError(f"Unknown unit definition: {definition}")
    units = (
        events.groupby(keys, dropna=False, sort=False)
        .agg(
            HAA5_max=("HAA5", "max"),
            HAA9_max=("HAA9", "max"),
            first_date=("CollectionDate", "min"),
            last_date=("CollectionDate", "max"),
            observed_dates=("CollectionDate", "nunique"),
            observed_facilities=("FacilityID", "nunique"),
            observed_sites=("site_id", "nunique"),
            physical_samples=("SampleID", "nunique"),
        )
        .reset_index()
    )
    units["definition"] = definition
    return units


def system_design_rows(units: pd.DataFrame, full: pd.DataFrame, threshold: float) -> pd.DataFrame:
    grouped = units.groupby("PWSID", sort=False)
    rows = grouped.agg(
        y00=("HAA5_max", lambda s: float(np.mean(s.to_numpy() > threshold))),
        y10=("HAA9_max", lambda s: float(np.mean(s.to_numpy() > threshold))),
        admissible_unit_count=("HAA9_max", "size"),
    ).reset_index()
    keep = full[
        [
            "PWSID",
            "HAA5_full_max",
            "HAA9_full_max",
            "facility_water_type_profile",
            "state_profile",
            "size_profile",
        ]
    ]
    rows = rows.merge(keep, on="PWSID", how="inner", validate="one_to_one")
    rows["y01"] = (rows["HAA5_full_max"] > threshold).astype(float)
    rows["y11"] = (rows["HAA9_full_max"] > threshold).astype(float)
    rows["interaction"] = rows["y11"] - rows["y10"] - rows["y01"] + rows["y00"]
    return rows


def summarize_design(
    rows: pd.DataFrame,
    label: str,
    seed: int,
    condition_on_reference: bool = True,
) -> dict[str, float | int | str]:
    source = rows.loc[rows["y11"].eq(1)].copy() if condition_on_reference else rows.copy()
    values = source[["y00", "y10", "y01", "y11", "interaction"]].to_numpy(float)
    estimate, limits = mean_bootstrap(values, seed)
    names = ["p00", "p10", "p01", "p11", "interaction"]
    result: dict[str, float | int | str] = {
        "analysis": label,
        "population": "reference_positive" if condition_on_reference else "all_eligible_systems",
        "n_eligible_systems": int(len(rows)),
        "n_reference_positive_systems": int(rows["y11"].sum()),
        "n_analysis_systems": int(len(source)),
    }
    for j, name in enumerate(names):
        result[name] = float(estimate[j])
        result[f"{name}_ci_lower"] = float(limits[0, j])
        result[f"{name}_ci_upper"] = float(limits[1, j])
    return result


def round_semantics(events: pd.DataFrame, full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit = aggregate_units(events, "event_code")
    audit["date_span_days"] = (audit["last_date"] - audit["first_date"]).dt.days
    audit["crosses_calendar_year"] = audit["first_date"].dt.year.ne(audit["last_date"].dt.year)
    audit["multi_date"] = audit["observed_dates"].gt(1)
    audit["span_gt_7_days"] = audit["date_span_days"].gt(7)
    audit["span_gt_30_days"] = audit["date_span_days"].gt(30)
    audit["multi_facility"] = audit["observed_facilities"].gt(1)
    audit["multi_site"] = audit["observed_sites"].gt(1)

    summary_rows: list[dict[str, object]] = []
    for field in [
        "multi_date",
        "span_gt_7_days",
        "span_gt_30_days",
        "crosses_calendar_year",
        "multi_facility",
        "multi_site",
    ]:
        count = int(audit[field].sum())
        summary_rows.append(
            {
                "metric": field,
                "count": count,
                "denominator": int(len(audit)),
                "fraction": count / len(audit),
            }
        )
    for field in ["date_span_days", "observed_dates", "observed_facilities", "observed_sites", "physical_samples"]:
        values = audit[field].astype(float)
        summary_rows.extend(
            [
                {"metric": f"{field}_median", "count": float(values.median()), "denominator": np.nan, "fraction": np.nan},
                {"metric": f"{field}_p95", "count": float(values.quantile(0.95)), "denominator": np.nan, "fraction": np.nan},
                {"metric": f"{field}_maximum", "count": float(values.max()), "denominator": np.nan, "fraction": np.nan},
            ]
        )
    summary = pd.DataFrame(summary_rows)

    sensitivity: list[dict[str, object]] = []
    for index, definition in enumerate(
        ["event_code", "event_code_year", "collection_date", "facility_event_code_year", "physical_sample"]
    ):
        units = aggregate_units(events, definition)
        rows = system_design_rows(units, full, THRESHOLD)
        result = summarize_design(rows, definition, SEED + 10 + index, True)
        result["total_units"] = int(len(units))
        result["median_units_per_system"] = float(rows["admissible_unit_count"].median())
        sensitivity.append(result)
    return audit, summary, pd.DataFrame(sensitivity)


def support_analyses(
    events: pd.DataFrame,
    full: pd.DataFrame,
    event_units: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    round_counts = event_units.groupby("PWSID").size().rename("round_count")
    site_rounds = events.groupby(["PWSID", "site_id"])["SampleEventCode"].nunique().rename("site_round_count")
    presence = site_rounds.reset_index().merge(round_counts.reset_index(), on="PWSID", how="left")
    stable = (
        presence.assign(stable=lambda d: d["site_round_count"].eq(d["round_count"]))
        .groupby("PWSID")
        .agg(stable_site_count=("stable", "sum"))
    )
    support = full[
        ["PWSID", "facility_water_type_profile", "HAA9_full_max", "HAA5_full_max", "state_profile"]
    ].copy()
    support = support.merge(round_counts, on="PWSID", how="left").merge(stable, on="PWSID", how="left")
    support["stable_site_count"] = support["stable_site_count"].fillna(0).astype(int)
    source = support["facility_water_type_profile"]
    support["source_aligned_support"] = np.where(
        source.eq("GW"), support["round_count"].ge(2), support["round_count"].ge(4)
    )
    support["four_code_groups"] = support["round_count"].ge(4)
    support["four_code_groups_stable_site"] = support["four_code_groups"] & support["stable_site_count"].ge(1)
    support["reference_positive_B60"] = support["HAA9_full_max"].gt(THRESHOLD)

    system_rows = system_design_rows(event_units, full, THRESHOLD)
    cohort_masks: dict[str, pd.Series] = {
        "all_panel_complete_observed_grids": pd.Series(True, index=support.index),
        "source_aligned_minimum_rounds": support["source_aligned_support"],
        "at_least_four_event_code_groups": support["four_code_groups"],
        "four_groups_with_repeated_site": support["four_code_groups_stable_site"],
    }
    estimates: list[dict[str, object]] = []
    support_index = support.set_index("PWSID")
    for j, (name, mask) in enumerate(cohort_masks.items()):
        ids = set(support.loc[mask.fillna(False), "PWSID"])
        subset = system_rows.loc[system_rows["PWSID"].isin(ids)]
        result = summarize_design(subset, name, SEED + 100 + j, True)
        estimates.append(result)
    return support, pd.DataFrame(estimates)


def threshold_uncertainty(event_units: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base_ids = set(full.loc[full["HAA9_full_max"].gt(THRESHOLD), "PWSID"])
    for j, threshold in enumerate(THRESHOLDS):
        system_rows = system_design_rows(event_units, full, float(threshold))
        changing = summarize_design(system_rows, f"B{threshold:g}", SEED + 200 + j, True)
        changing["threshold_ug_l"] = float(threshold)
        changing["cohort_definition"] = "threshold_specific_reference_positive"
        rows.append(changing)

        fixed = system_rows.loc[system_rows["PWSID"].isin(base_ids)].copy()
        fixed_result = summarize_design(
            fixed,
            f"B{threshold:g}_within_B60_reference_cohort",
            SEED + 300 + j,
            False,
        )
        fixed_result["threshold_ug_l"] = float(threshold)
        fixed_result["cohort_definition"] = "fixed_B60_reference_positive_systems"
        rows.append(fixed_result)
    return pd.DataFrame(rows)


def simultaneous_interaction_band(event_units: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    system_ids = full["PWSID"].astype(str).tolist()
    by_threshold: list[pd.Series] = []
    estimates: list[float] = []
    supports: list[int] = []
    for threshold in THRESHOLDS:
        rows = system_design_rows(event_units, full, float(threshold)).set_index("PWSID").reindex(system_ids)
        reference = rows["y11"].eq(1)
        supports.append(int(reference.sum()))
        interaction = rows["interaction"].where(reference)
        by_threshold.append(interaction)
        estimates.append(float(interaction.mean()))
    matrix = np.column_stack([s.to_numpy(float) for s in by_threshold])
    rng = np.random.default_rng(SEED + 400)
    boot = np.empty((BOOTSTRAPS, len(THRESHOLDS)), dtype=float)
    n = len(system_ids)
    batch = 100
    for start in range(0, BOOTSTRAPS, batch):
        stop = min(start + batch, BOOTSTRAPS)
        idx = rng.integers(0, n, size=(stop - start, n))
        sampled = matrix[idx]
        boot[start:stop] = np.nanmean(sampled, axis=1)
    estimates_array = np.asarray(estimates)
    pointwise = np.quantile(boot, [0.025, 0.975], axis=0)
    centered = boot - estimates_array[None, :]
    critical = float(np.quantile(np.nanmax(np.abs(centered), axis=1), 0.95))
    return pd.DataFrame(
        {
            "threshold_ug_l": THRESHOLDS,
            "n_reference_positive_systems": supports,
            "interaction": estimates,
            "pointwise_ci_lower": pointwise[0],
            "pointwise_ci_upper": pointwise[1],
            "simultaneous_ci_lower": estimates_array - critical,
            "simultaneous_ci_upper": estimates_array + critical,
            "simultaneous_absolute_critical_value": critical,
            "bootstrap_replicates": BOOTSTRAPS,
        }
    )


def probability_by_system(units: pd.DataFrame, column: str, threshold: float) -> pd.Series:
    return units.groupby("PWSID")[column].apply(lambda s: float(np.mean(s.to_numpy(float) > threshold)))


def screening_metrics(reference: np.ndarray, screen_probability: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=bool)
    screen_probability = np.asarray(screen_probability, dtype=float)
    positive_weight = float(screen_probability[reference].sum())
    negative_weight = float(screen_probability[~reference].sum())
    total_screen = positive_weight + negative_weight
    return {
        "sensitivity": positive_weight / reference.sum(),
        "followup_fraction": total_screen / len(reference),
        "false_positive_rate": negative_weight / (~reference).sum(),
        "positive_predictive_value": positive_weight / total_screen if total_screen else np.nan,
    }


def bootstrap_screening(reference: np.ndarray, screen_probability: np.ndarray, seed: int) -> dict[str, float]:
    reference = np.asarray(reference, dtype=bool)
    screen_probability = np.asarray(screen_probability, dtype=float)
    n = len(reference)
    rng = np.random.default_rng(seed)
    names = ["sensitivity", "followup_fraction", "false_positive_rate", "positive_predictive_value"]
    values = np.empty((BOOTSTRAPS, len(names)), dtype=float)
    batch = 250
    offset = 0
    while offset < BOOTSTRAPS:
        take = min(batch, BOOTSTRAPS - offset)
        idx = rng.integers(0, n, size=(take, n))
        ref = reference[idx]
        prob = screen_probability[idx]
        pos_n = ref.sum(axis=1)
        neg_n = (~ref).sum(axis=1)
        pos_weight = (prob * ref).sum(axis=1)
        neg_weight = (prob * (~ref)).sum(axis=1)
        total = pos_weight + neg_weight
        values[offset : offset + take, 0] = pos_weight / pos_n
        values[offset : offset + take, 1] = total / n
        values[offset : offset + take, 2] = neg_weight / neg_n
        values[offset : offset + take, 3] = np.divide(
            pos_weight,
            total,
            out=np.full(take, np.nan),
            where=total > 0,
        )
        offset += take
    point = screening_metrics(reference, screen_probability)
    result: dict[str, float] = {}
    for j, name in enumerate(names):
        lo, hi = np.nanquantile(values[:, j], [0.025, 0.975])
        result[name] = float(point[name])
        result[f"{name}_ci_lower"] = float(lo)
        result[f"{name}_ci_upper"] = float(hi)
    return result


def screening_analysis(event_units: pd.DataFrame, full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    system_ids = full["PWSID"].astype(str).tolist()
    reference = full.set_index("PWSID").reindex(system_ids)["HAA9_full_max"].to_numpy(float) > THRESHOLD
    direct_h9 = probability_by_system(event_units, "HAA9_max", THRESHOLD).reindex(system_ids).to_numpy(float)
    thresholds = np.round(np.arange(0.0, THRESHOLD + 0.001, 0.25), 2)
    curve_rows: list[dict[str, float]] = []
    probability_cache: dict[float, np.ndarray] = {}
    for threshold in thresholds:
        probability = probability_by_system(event_units, "HAA5_max", float(threshold)).reindex(system_ids).to_numpy(float)
        probability_cache[float(threshold)] = probability
        metrics = screening_metrics(reference, probability)
        curve_rows.append({"haa5_action_threshold_ug_l": float(threshold), **metrics})
    curve = pd.DataFrame(curve_rows)

    folds = np.array([sha_fold(pwsid, "wr-screen-crossfit-v1", 5) for pwsid in system_ids], dtype=int)
    selected = np.empty(len(system_ids), dtype=float)
    selected[:] = np.nan
    fold_rows: list[dict[str, object]] = []
    for fold in range(5):
        training = folds != fold
        evaluation = folds == fold
        direct_target = screening_metrics(reference[training], direct_h9[training])["sensitivity"]
        eligible: list[float] = []
        for threshold in thresholds:
            sensitivity = screening_metrics(reference[training], probability_cache[float(threshold)][training])["sensitivity"]
            if sensitivity >= direct_target - 1e-12:
                eligible.append(float(threshold))
        chosen = max(eligible) if eligible else float(thresholds.min())
        selected[evaluation] = probability_cache[chosen][evaluation]
        train_metrics = screening_metrics(reference[training], probability_cache[chosen][training])
        test_metrics = screening_metrics(reference[evaluation], probability_cache[chosen][evaluation])
        fold_rows.append(
            {
                "fold": fold,
                "training_systems": int(training.sum()),
                "evaluation_systems": int(evaluation.sum()),
                "training_reference_positive_systems": int(reference[training].sum()),
                "evaluation_reference_positive_systems": int(reference[evaluation].sum()),
                "direct_HAA9_training_sensitivity": direct_target,
                "selected_HAA5_action_threshold_ug_l": chosen,
                **{f"training_{k}": v for k, v in train_metrics.items()},
                **{f"evaluation_{k}": v for k, v in test_metrics.items()},
            }
        )

    operating_rows: list[dict[str, object]] = []
    policies = {
        "HAA5_at_60_one_event_code_group": probability_cache[60.0],
        "HAA9_at_60_one_event_code_group": direct_h9,
        "cross_fitted_HAA5_trigger": selected,
    }
    for j, (policy, probability) in enumerate(policies.items()):
        result = bootstrap_screening(reference, probability, SEED + 500 + j)
        result.update(
            {
                "policy": policy,
                "eligible_systems": int(len(system_ids)),
                "reference_positive_systems": int(reference.sum()),
                "evaluation_role": "cross_fitted_internal_evaluation" if "cross_fitted" in policy else "fixed_comparator",
            }
        )
        operating_rows.append(result)
    return curve, pd.DataFrame(fold_rows), pd.DataFrame(operating_rows)


def hypergeom_capture(total: int, positive: int, selected: int) -> float:
    if selected <= 0 or positive <= 0:
        return 0.0
    if selected >= total:
        return 1.0
    if total - positive < selected:
        return 1.0
    return 1.0 - math.comb(total - positive, selected) / math.comb(total, selected)


def resource_analysis(events: pd.DataFrame, full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    physical = aggregate_units(events, "physical_sample")
    counts = physical.groupby("PWSID").agg(
        available_samples=("HAA9_max", "size"),
        HAA5_positive_samples=("HAA5_max", lambda s: int((s > THRESHOLD).sum())),
        HAA9_positive_samples=("HAA9_max", lambda s: int((s > THRESHOLD).sum())),
    )
    counts = counts.merge(full.set_index("PWSID")[["HAA9_full_max"]], left_index=True, right_index=True, how="inner")
    counts = counts.loc[counts["HAA9_full_max"].gt(THRESHOLD)].copy()
    curves: list[dict[str, object]] = []
    for panel, column in [("HAA5", "HAA5_positive_samples"), ("HAA9", "HAA9_positive_samples")]:
        for k in range(1, 13):
            probabilities = np.array(
                [
                    hypergeom_capture(int(n), int(m), k)
                    for n, m in zip(counts["available_samples"], counts[column], strict=True)
                ],
                dtype=float,
            )
            point, ci = mean_bootstrap(probabilities, SEED + 600 + k + (0 if panel == "HAA5" else 50))
            curves.append(
                {
                    "panel": panel,
                    "selected_physical_samples": k,
                    "conditional_capture": float(point[0]),
                    "ci_lower": float(ci[0, 0]),
                    "ci_upper": float(ci[1, 0]),
                    "reference_positive_systems": int(len(counts)),
                    "sample_selection": "uniform_without_replacement_with_saturation",
                }
            )
    curve = pd.DataFrame(curves)

    lookup = curve.set_index(["panel", "selected_physical_samples"])["conditional_capture"].to_dict()
    surface: list[dict[str, object]] = []
    ratios = np.round(np.arange(0.0, 3.0 + 0.001, 0.05), 2)
    for budget in range(2, 13):
        n5 = budget
        capture5 = lookup[("HAA5", min(n5, 12))]
        for incremental_ratio in ratios:
            total_h9_cost = 1.0 + float(incremental_ratio)
            n9 = int(math.floor(budget / total_h9_cost + 1e-12))
            capture9 = 0.0 if n9 < 1 else lookup[("HAA9", min(n9, 12))]
            difference = capture9 - capture5
            if abs(difference) < 1e-12:
                winner = "tie"
            elif difference > 0:
                winner = "HAA9_reporting"
            else:
                winner = "additional_HAA5_samples"
            surface.append(
                {
                    "budget_HAA5_sample_equivalents": budget,
                    "incremental_HAA9_reporting_cost_per_sample": float(incremental_ratio),
                    "HAA5_samples": n5,
                    "HAA9_samples": n9,
                    "HAA5_capture": capture5,
                    "HAA9_capture": capture9,
                    "HAA9_minus_HAA5_capture": difference,
                    "preferred_design": winner,
                    "cost_interpretation": "scenario only; HAA5 sample collection plus analysis equals one unit",
                }
            )
    surface_frame = pd.DataFrame(surface)
    break_even = (
        surface_frame.loc[surface_frame["HAA9_minus_HAA5_capture"].ge(0)]
        .groupby("budget_HAA5_sample_equivalents", as_index=False)
        .agg(max_incremental_HAA9_cost_ratio=("incremental_HAA9_reporting_cost_per_sample", "max"))
    )
    break_even["reference_positive_systems"] = int(len(counts))
    return curve, surface_frame, break_even


def severity_persistence(
    events: pd.DataFrame,
    full: pd.DataFrame,
    contributions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positive_sample = events.assign(
        HAA9_positive=events["HAA9"].gt(THRESHOLD),
        HAA5_positive=events["HAA5"].gt(THRESHOLD),
    )
    sample_stats = positive_sample.groupby("PWSID").agg(
        observed_samples=("SampleID", "nunique"),
        HAA9_positive_samples=("HAA9_positive", "sum"),
        HAA5_positive_samples=("HAA5_positive", "sum"),
        HAA9_max=("HAA9", "max"),
        HAA5_max=("HAA5", "max"),
    )
    round_stats = (
        positive_sample.groupby(["PWSID", "SampleEventCode"])
        .agg(HAA9_round_positive=("HAA9_positive", "max"), HAA5_round_positive=("HAA5_positive", "max"))
        .groupby("PWSID")
        .agg(
            observed_rounds=("HAA9_round_positive", "size"),
            HAA9_positive_rounds=("HAA9_round_positive", "sum"),
            HAA5_positive_rounds=("HAA5_round_positive", "sum"),
        )
    )
    site_stats = (
        positive_sample.groupby(["PWSID", "site_id"])
        .agg(HAA9_site_positive=("HAA9_positive", "max"), HAA5_site_positive=("HAA5_positive", "max"))
        .groupby("PWSID")
        .agg(
            observed_sites=("HAA9_site_positive", "size"),
            HAA9_positive_sites=("HAA9_site_positive", "sum"),
            HAA5_positive_sites=("HAA5_site_positive", "sum"),
        )
    )
    severity = sample_stats.join(round_stats).join(site_stats)
    severity = severity.join(
        full.set_index("PWSID")[["facility_water_type_profile", "state_profile", "size_profile"]]
    )
    severity = severity.join(
        contributions.set_index("PWSID")[["y00", "y10", "y01", "cat_both_required", "interaction"]]
    )
    severity = severity.loc[severity["HAA9_max"].gt(THRESHOLD)].reset_index()
    severity["HAA9_margin_above_B60_ug_l"] = severity["HAA9_max"] - THRESHOLD
    severity["HAA9_positive_round_fraction"] = severity["HAA9_positive_rounds"] / severity["observed_rounds"]
    severity["HAA9_positive_site_fraction"] = severity["HAA9_positive_sites"] / severity["observed_sites"]
    severity["HAA5_panel_ceiling_miss"] = severity["HAA5_max"].le(THRESHOLD)
    severity["observed_recurrence_class"] = np.select(
        [severity["HAA9_positive_rounds"].eq(1), severity["HAA9_positive_rounds"].ge(2)],
        ["one_positive_event_code_group", "two_or_more_positive_event_code_groups"],
        default="none",
    )

    summaries: list[dict[str, object]] = []
    for group_name, group in severity.groupby("HAA5_panel_ceiling_miss"):
        label = "HAA5_full_grid_nonpositive" if group_name else "HAA5_full_grid_positive"
        summaries.append(
            {
                "group": label,
                "systems": int(len(group)),
                "median_HAA9_margin_ug_l": float(group["HAA9_margin_above_B60_ug_l"].median()),
                "q25_HAA9_margin_ug_l": float(group["HAA9_margin_above_B60_ug_l"].quantile(0.25)),
                "q75_HAA9_margin_ug_l": float(group["HAA9_margin_above_B60_ug_l"].quantile(0.75)),
                "median_positive_round_fraction": float(group["HAA9_positive_round_fraction"].median()),
                "systems_with_two_or_more_positive_rounds": int(group["HAA9_positive_rounds"].ge(2).sum()),
                "fraction_with_two_or_more_positive_rounds": float(group["HAA9_positive_rounds"].ge(2).mean()),
            }
        )
    return severity, pd.DataFrame(summaries)


def continuous_interaction(event_units: pd.DataFrame, full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unit = event_units.copy()
    unit["panel_gap"] = unit["HAA9_max"] - unit["HAA5_max"]
    round_gap = unit.groupby("PWSID")["panel_gap"].mean().rename("mean_round_panel_gap_ug_l")
    frame = full.set_index("PWSID")[["HAA5_full_max", "HAA9_full_max", "state_profile"]].join(round_gap)
    frame["full_grid_panel_gap_ug_l"] = frame["HAA9_full_max"] - frame["HAA5_full_max"]
    frame["continuous_interaction_ug_l"] = frame["full_grid_panel_gap_ug_l"] - frame["mean_round_panel_gap_ug_l"]
    frame = frame.reset_index()
    values = frame["continuous_interaction_ug_l"].to_numpy(float)
    mean, mean_ci = mean_bootstrap(values, SEED + 700)
    median = median_bootstrap(values, SEED + 701)
    positive = (values > 0).astype(float)
    prop, prop_ci = mean_bootstrap(positive, SEED + 702)
    lower_cut, upper_cut = np.quantile(values, [0.01, 0.99])
    winsorized = np.clip(values, lower_cut, upper_cut)
    trimmed = values[(values >= lower_cut) & (values <= upper_cut)]
    winsor_mean, winsor_ci = mean_bootstrap(winsorized, SEED + 703)
    trimmed_mean, trimmed_ci = mean_bootstrap(trimmed, SEED + 704)
    summary = pd.DataFrame(
        [
            {
                "eligible_systems": int(len(frame)),
                "mean_continuous_interaction_ug_l": float(mean[0]),
                "mean_ci_lower": float(mean_ci[0, 0]),
                "mean_ci_upper": float(mean_ci[1, 0]),
                "median_continuous_interaction_ug_l": median.estimate,
                "median_ci_lower": median.lower,
                "median_ci_upper": median.upper,
                "fraction_positive": float(prop[0]),
                "fraction_positive_ci_lower": float(prop_ci[0, 0]),
                "fraction_positive_ci_upper": float(prop_ci[1, 0]),
                "winsorized_1pct_mean_ug_l": float(winsor_mean[0]),
                "winsorized_1pct_mean_ci_lower": float(winsor_ci[0, 0]),
                "winsorized_1pct_mean_ci_upper": float(winsor_ci[1, 0]),
                "trimmed_1pct_each_tail_mean_ug_l": float(trimmed_mean[0]),
                "trimmed_1pct_each_tail_mean_ci_lower": float(trimmed_ci[0, 0]),
                "trimmed_1pct_each_tail_mean_ci_upper": float(trimmed_ci[1, 0]),
                "lower_1pct_cut_ug_l": float(lower_cut),
                "upper_99pct_cut_ug_l": float(upper_cut),
                "definition": "(HAA9 full-grid max - HAA5 full-grid max) - mean(HAA9 round max - HAA5 round max)",
            }
        ]
    )
    return frame, summary


def leave_one_state_out(contributions: pd.DataFrame) -> pd.DataFrame:
    reference = contributions.loc[contributions["y11"].eq(1)].copy()
    overall = float(reference["interaction"].mean())
    rows: list[dict[str, object]] = []
    for state, group in reference.groupby("state_profile"):
        remaining = reference.loc[reference["state_profile"].ne(state)]
        rows.append(
            {
                "excluded_state_code": state,
                "excluded_reference_positive_systems": int(len(group)),
                "remaining_reference_positive_systems": int(len(remaining)),
                "interaction_without_state": float(remaining["interaction"].mean()),
                "difference_from_overall": float(remaining["interaction"].mean() - overall),
                "overall_interaction": overall,
            }
        )
    return pd.DataFrame(rows).sort_values("interaction_without_state")


def full_grid_from_available_events(events: pd.DataFrame) -> pd.DataFrame:
    """Build system maxima from complete triplets without asserting a complete grid."""

    def profile(series: pd.Series) -> str:
        values = sorted({str(value) for value in series.dropna() if str(value)})
        return "+".join(values) if values else "Unknown"

    return (
        events.groupby("PWSID", sort=False)
        .agg(
            HAA5_full_max=("HAA5", "max"),
            HAA9_full_max=("HAA9", "max"),
            facility_water_type_profile=("FacilityWaterType", profile),
            state_profile=("State", profile),
            size_profile=("Size", profile),
        )
        .reset_index()
    )


def cohort_and_operator_sensitivity(
    eligible_events: pd.DataFrame,
    all_complete_events: pd.DataFrame,
    full: pd.DataFrame,
) -> pd.DataFrame:
    """Audit incomplete-system inclusion, date scope, equality and local margins."""

    rows: list[dict[str, object]] = []

    def add_analysis(events: pd.DataFrame, system_full: pd.DataFrame, label: str, seed: int) -> None:
        units = aggregate_units(events, "event_code")
        design = system_design_rows(units, system_full, THRESHOLD)
        result = summarize_design(design, label, seed, True)
        result["threshold_ug_l"] = THRESHOLD
        result["comparison_type"] = "cohort_or_date_scope"
        rows.append(result)

    add_analysis(eligible_events, full, "primary_panel_complete_observed_grid", SEED + 900)

    pre_2021 = eligible_events.loc[eligible_events["CollectionDate"].dt.year.le(2020)].copy()
    pre_2021_full = full_grid_from_available_events(pre_2021)
    add_analysis(pre_2021, pre_2021_full, "exclude_2021_complete_records", SEED + 901)

    available_full = full_grid_from_available_events(all_complete_events)
    add_analysis(
        all_complete_events,
        available_full,
        "include_systems_with_any_available_complete_triplets",
        SEED + 902,
    )

    primary_units = aggregate_units(eligible_events, "event_code")
    for index, threshold in enumerate([59.0, 59.5, 60.0, 60.5, 61.0]):
        design = system_design_rows(primary_units, full, threshold)
        result = summarize_design(design, f"strict_greater_than_B{threshold:g}", SEED + 910 + index, True)
        result["threshold_ug_l"] = threshold
        result["comparison_type"] = "local_threshold_margin"
        rows.append(result)

    # At exactly 60, recalculate the four cells with >= to expose the small equality set.
    units = primary_units.copy()
    design = units.groupby("PWSID", sort=False).agg(
        y00=("HAA5_max", lambda s: float(np.mean(s.to_numpy() >= THRESHOLD))),
        y10=("HAA9_max", lambda s: float(np.mean(s.to_numpy() >= THRESHOLD))),
    ).reset_index()
    design = design.merge(
        full[["PWSID", "HAA5_full_max", "HAA9_full_max", "facility_water_type_profile", "state_profile", "size_profile"]],
        on="PWSID",
        how="inner",
        validate="one_to_one",
    )
    design["y01"] = design["HAA5_full_max"].ge(THRESHOLD).astype(float)
    design["y11"] = design["HAA9_full_max"].ge(THRESHOLD).astype(float)
    design["interaction"] = design["y11"] - design["y10"] - design["y01"] + design["y00"]
    result = summarize_design(design, "greater_than_or_equal_B60", SEED + 920, True)
    result["threshold_ug_l"] = THRESHOLD
    result["comparison_type"] = "operator"
    rows.append(result)
    return pd.DataFrame(rows)


def resource_unit_audit(
    events: pd.DataFrame,
    event_units: pd.DataFrame,
    full: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Translate an administrative event-code group into samples, dates and sites."""

    positive_ids = set(full.loc[full["HAA9_full_max"].gt(THRESHOLD), "PWSID"])
    selected = event_units.loc[event_units["PWSID"].isin(positive_ids)].copy()
    expected = selected.groupby("PWSID").agg(
        expected_samples_in_one_selected_group=("physical_samples", "mean"),
        expected_dates_in_one_selected_group=("observed_dates", "mean"),
        expected_sites_in_one_selected_group=("observed_sites", "mean"),
    )
    full_resource = events.loc[events["PWSID"].isin(positive_ids)].groupby("PWSID").agg(
        full_grid_physical_samples=("SampleID", "nunique"),
        full_grid_collection_dates=("CollectionDate", "nunique"),
        full_grid_sites=("site_id", "nunique"),
    )
    detail = expected.join(full_resource, how="inner").reset_index()
    summary_rows: list[dict[str, object]] = []
    for column in [
        "expected_samples_in_one_selected_group",
        "expected_dates_in_one_selected_group",
        "expected_sites_in_one_selected_group",
        "full_grid_physical_samples",
        "full_grid_collection_dates",
        "full_grid_sites",
    ]:
        values = detail[column].astype(float)
        summary_rows.append(
            {
                "resource_measure": column,
                "reference_positive_systems": int(len(values)),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "p95": float(values.quantile(0.95)),
                "maximum": float(values.max()),
            }
        )
    return detail, pd.DataFrame(summary_rows)


def expected_subgrid_capture(binary_grid: np.ndarray, requested_rounds: int, requested_sites: int) -> float:
    """Exact probability that a uniform round-by-site Cartesian subgrid contains a signal."""

    from itertools import combinations

    grid = np.asarray(binary_grid, dtype=bool)
    n_rounds, n_sites = grid.shape
    selected_rounds = min(requested_rounds, n_rounds)
    selected_sites = min(requested_sites, n_sites)
    total_site_sets = math.comb(n_sites, selected_sites)
    miss_probability = 0.0
    row_sets = list(combinations(range(n_rounds), selected_rounds))
    for rows in row_sets:
        positive_columns = int(grid[list(rows), :].any(axis=0).sum())
        negative_columns = n_sites - positive_columns
        miss_probability += (
            math.comb(negative_columns, selected_sites) / total_site_sets
            if negative_columns >= selected_sites
            else 0.0
        )
    miss_probability /= len(row_sets)
    return 1.0 - miss_probability


def joint_round_site_surface(
    events: pd.DataFrame,
    full: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Exact site-by-round capture over rectangular observed event-code grids."""

    cells = (
        events.groupby(["PWSID", "SampleEventCode", "site_id"], sort=False)
        .agg(HAA5=("HAA5", "max"), HAA9=("HAA9", "max"))
        .reset_index()
    )
    audit = cells.groupby("PWSID").agg(
        observed_rounds=("SampleEventCode", "nunique"),
        observed_sites=("site_id", "nunique"),
        observed_cells=("site_id", "size"),
    )
    audit["expected_rectangular_cells"] = audit["observed_rounds"] * audit["observed_sites"]
    audit["missing_cells"] = audit["expected_rectangular_cells"] - audit["observed_cells"]
    audit["rectangular_observed_grid"] = audit["missing_cells"].eq(0)
    audit = audit.join(full.set_index("PWSID")[["HAA9_full_max", "facility_water_type_profile", "state_profile"]])
    audit["reference_positive_B60"] = audit["HAA9_full_max"].gt(THRESHOLD)
    rectangular_ids = audit.index[audit["rectangular_observed_grid"] & audit["reference_positive_B60"]]

    labels: list[tuple[str, int, int]] = []
    for panel in ["HAA5", "HAA9"]:
        for rounds in range(1, 5):
            for sites in range(1, 9):
                labels.append((panel, rounds, sites))
    probabilities = np.empty((len(rectangular_ids), len(labels)), dtype=float)
    saturation_round = np.empty((len(rectangular_ids), len(labels)), dtype=bool)
    saturation_site = np.empty((len(rectangular_ids), len(labels)), dtype=bool)
    detail_rows: list[dict[str, object]] = []
    for row_index, pwsid in enumerate(rectangular_ids):
        group = cells.loc[cells["PWSID"].eq(pwsid)]
        rounds = sorted(group["SampleEventCode"].astype(str).unique())
        sites = sorted(group["site_id"].astype(str).unique())
        grids = {
            panel: (
                group.pivot(index="SampleEventCode", columns="site_id", values=panel)
                .reindex(index=rounds, columns=sites)
                .to_numpy(float)
                > THRESHOLD
            )
            for panel in ["HAA5", "HAA9"]
        }
        for column_index, (panel, requested_rounds, requested_sites) in enumerate(labels):
            probability = expected_subgrid_capture(grids[panel], requested_rounds, requested_sites)
            probabilities[row_index, column_index] = probability
            saturation_round[row_index, column_index] = requested_rounds >= len(rounds)
            saturation_site[row_index, column_index] = requested_sites >= len(sites)
            detail_rows.append(
                {
                    "PWSID": pwsid,
                    "panel": panel,
                    "requested_rounds": requested_rounds,
                    "requested_sites": requested_sites,
                    "available_rounds": len(rounds),
                    "available_sites": len(sites),
                    "round_dimension_saturated": requested_rounds >= len(rounds),
                    "site_dimension_saturated": requested_sites >= len(sites),
                    "conditional_capture_probability": probability,
                }
            )
    point, limits = mean_bootstrap(probabilities, SEED + 930)
    surface_rows: list[dict[str, object]] = []
    for column_index, (panel, rounds, sites) in enumerate(labels):
        surface_rows.append(
            {
                "panel": panel,
                "requested_rounds": rounds,
                "requested_sites": sites,
                "conditional_capture": float(point[column_index]),
                "ci_lower": float(limits[0, column_index]),
                "ci_upper": float(limits[1, column_index]),
                "reference_positive_rectangular_systems": int(len(rectangular_ids)),
                "fraction_round_dimension_saturated": float(saturation_round[:, column_index].mean()),
                "fraction_site_dimension_saturated": float(saturation_site[:, column_index].mean()),
                "selection": "uniform_without_replacement_exact_Cartesian_subgrid_with_saturation",
            }
        )
    return audit.reset_index(), pd.DataFrame(detail_rows), pd.DataFrame(surface_rows)


def chronological_disjoint_validation(
    event_units: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Condition and evaluate on disjoint early/late halves of exactly four-round systems."""

    units = event_units.sort_values(["PWSID", "first_date", "SampleEventCode"]).copy()
    four_ids = units.groupby("PWSID").size()
    four_ids = set(four_ids[four_ids.eq(4)].index)
    units = units.loc[units["PWSID"].isin(four_ids)].copy()
    units["chronological_order"] = units.groupby("PWSID").cumcount()
    direction_specs = [
        ("late2_conditions_early2_evaluates", {2, 3}, {0, 1}),
        ("early2_conditions_late2_evaluates", {0, 1}, {2, 3}),
    ]
    all_detail: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    for direction_index, (direction, reference_orders, evaluation_orders) in enumerate(direction_specs):
        reference_units = units.loc[units["chronological_order"].isin(reference_orders)]
        evaluation_units = units.loc[units["chronological_order"].isin(evaluation_orders)]
        for threshold_index, threshold in enumerate(THRESHOLDS):
            reference_positive = reference_units.groupby("PWSID")["HAA9_max"].max().gt(threshold)
            evaluation = evaluation_units.groupby("PWSID").agg(
                y00=("HAA5_max", lambda s, b=threshold: float(np.mean(s.to_numpy() > b))),
                y10=("HAA9_max", lambda s, b=threshold: float(np.mean(s.to_numpy() > b))),
                y01=("HAA5_max", lambda s, b=threshold: float(np.max(s.to_numpy()) > b)),
                y11=("HAA9_max", lambda s, b=threshold: float(np.max(s.to_numpy()) > b)),
                evaluation_rounds=("HAA9_max", "size"),
            )
            ids = reference_positive.index[reference_positive]
            evaluation = evaluation.loc[evaluation.index.intersection(ids)].copy()
            evaluation["interaction"] = evaluation["y11"] - evaluation["y10"] - evaluation["y01"] + evaluation["y00"]
            values = evaluation[["y00", "y10", "y01", "y11", "interaction"]].to_numpy(float)
            point, limits = mean_bootstrap(values, SEED + 950 + direction_index * 100 + threshold_index)
            row: dict[str, object] = {
                "direction": direction,
                "threshold_ug_l": float(threshold),
                "condition_positive_systems": int(len(evaluation)),
                "eligible_exactly_four_round_systems": int(len(four_ids)),
                "reference_rounds": "late two" if reference_orders == {2, 3} else "early two",
                "evaluation_rounds": "early two" if evaluation_orders == {0, 1} else "late two",
            }
            for column_index, name in enumerate(["p00", "p10", "p01", "p11", "interaction"]):
                row[name] = float(point[column_index])
                row[f"{name}_ci_lower"] = float(limits[0, column_index])
                row[f"{name}_ci_upper"] = float(limits[1, column_index])
            threshold_rows.append(row)
            if threshold == THRESHOLD:
                detail = evaluation.reset_index()
                detail.insert(1, "direction", direction)
                detail.insert(2, "threshold_ug_l", threshold)
                all_detail.append(detail)
                summary_rows.append(row)
    return pd.concat(all_detail, ignore_index=True), pd.DataFrame(summary_rows), pd.DataFrame(threshold_rows)


def disjoint_grid_validation(event_units: pd.DataFrame, full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    units = event_units.sort_values(["PWSID", "first_date", "SampleEventCode"]).copy()
    units["within_system_order"] = units.groupby("PWSID").cumcount()
    units["round_fold"] = units["within_system_order"] % 2
    eligible_ids = units.groupby("PWSID")["round_fold"].nunique()
    eligible_ids = set(eligible_ids[eligible_ids.eq(2)].index)
    units = units.loc[units["PWSID"].isin(eligible_ids)].copy()

    system_direction_rows: list[dict[str, object]] = []
    for reference_fold, evaluation_fold in [(0, 1), (1, 0)]:
        ref = units.loc[units["round_fold"].eq(reference_fold)]
        eval_units = units.loc[units["round_fold"].eq(evaluation_fold)]
        ref_positive = ref.groupby("PWSID")["HAA9_max"].max().gt(THRESHOLD)
        eval_system = eval_units.groupby("PWSID").agg(
            y00=("HAA5_max", lambda s: float(np.mean(s.to_numpy() > THRESHOLD))),
            y10=("HAA9_max", lambda s: float(np.mean(s.to_numpy() > THRESHOLD))),
            y01=("HAA5_max", lambda s: float(np.max(s.to_numpy()) > THRESHOLD)),
            y11=("HAA9_max", lambda s: float(np.max(s.to_numpy()) > THRESHOLD)),
            evaluation_rounds=("HAA9_max", "size"),
        )
        eval_system = eval_system.loc[eval_system.index.intersection(ref_positive[ref_positive].index)].copy()
        eval_system["interaction"] = eval_system["y11"] - eval_system["y10"] - eval_system["y01"] + eval_system["y00"]
        for pwsid, row in eval_system.iterrows():
            system_direction_rows.append(
                {
                    "PWSID": pwsid,
                    "reference_fold": reference_fold,
                    "evaluation_fold": evaluation_fold,
                    **row.to_dict(),
                }
            )
    detail = pd.DataFrame(system_direction_rows)
    summary_rows: list[dict[str, object]] = []
    for label, subset in [("pooled_directions", detail), ("reference_fold_0", detail.loc[detail["reference_fold"].eq(0)]), ("reference_fold_1", detail.loc[detail["reference_fold"].eq(1)])]:
        values = subset[["y00", "y10", "y01", "y11", "interaction"]].to_numpy(float)
        point, ci = mean_bootstrap(values, SEED + 800 + len(summary_rows))
        row: dict[str, object] = {
            "analysis": label,
            "system_direction_rows": int(len(subset)),
            "unique_systems": int(subset["PWSID"].nunique()),
            "interpretation": "reference positivity established in the opposite alternating observed-round fold",
        }
        for j, name in enumerate(["p00", "p10", "p01", "p11", "interaction"]):
            row[name] = float(point[j])
            row[f"{name}_ci_lower"] = float(ci[0, j])
            row[f"{name}_ci_upper"] = float(ci[1, j])
        summary_rows.append(row)
    return detail, pd.DataFrame(summary_rows)


def write_notes(summary: dict[str, object]) -> None:
    text = f"""# Water Research analysis extensions

The primary UCMR4 records remain the released group totals. The one-round unit is now described as a PWSID by SampleEventCode administrative event-code group, because {summary['round_multi_date_count']:,} of {summary['round_groups']:,} such groups span more than one collection date. Exact-date, event-code-year, facility-event and physical-sample analyses are reported as sensitivities rather than silently substituted for the released unit.

The cross-fitted HAA5 trigger was selected independently within each training fold to match the training-fold sensitivity of direct one-round HAA9 reporting for the finite full-grid HAA9 reference. It was evaluated only in the held-out fold. This is internal cross-fitting on an already inspected archive, not external or prospective validation. Follow-up fraction is a system-equal expected fraction of random event-code group realizations.

Resource-equivalent analyses use one physical sample with HAA5 collection and analysis as one relative unit. HAA9 uses the same EPA analytical methods in UCMR4, so the incremental HAA9 term represents an unknown reporting or analytical-processing increment rather than a nine-fifths chemical-count assumption. Results are scenario frontiers, not dollar costs or cost-effectiveness estimates.

The disjoint-grid analysis establishes HAA9 reference positivity in one alternating observed-round fold and evaluates all four cells in the other fold, then reverses the folds. Unlike the primary conditional factorial, its evaluation p11 is empirical rather than fixed to one.
"""
    with (OUT / "METHODS_RESULTS_EXTENSIONS.md").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events, full, contributions, all_complete_events = load_inputs()
    event_units = aggregate_units(events, "event_code")

    audit, audit_summary, round_sensitivity = round_semantics(events, full)
    support, support_estimates = support_analyses(events, full, event_units)
    threshold_results = threshold_uncertainty(event_units, full)
    threshold_band = simultaneous_interaction_band(event_units, full)
    screening_curve, screening_folds, screening_operating = screening_analysis(event_units, full)
    resource_curve, resource_surface, break_even = resource_analysis(events, full)
    severity, severity_summary = severity_persistence(events, full, contributions)
    continuous_detail, continuous_summary = continuous_interaction(event_units, full)
    state_influence = leave_one_state_out(contributions)
    disjoint_detail, disjoint_summary = disjoint_grid_validation(event_units, full)
    chronological_detail, chronological_summary, chronological_threshold = chronological_disjoint_validation(
        event_units
    )
    cohort_operator = cohort_and_operator_sensitivity(events, all_complete_events, full)
    resource_audit_detail, resource_audit_summary = resource_unit_audit(events, event_units, full)
    joint_grid_audit, joint_surface_detail, joint_surface = joint_round_site_surface(events, full)

    tables = {
        "round_semantics_audit": audit,
        "round_semantics_summary": audit_summary,
        "round_definition_sensitivity": round_sensitivity,
        "support_systems": support,
        "fixed_support_factorial": support_estimates,
        "threshold_factorial_with_intervals": threshold_results,
        "threshold_interaction_simultaneous_band": threshold_band,
        "screening_curve": screening_curve,
        "screening_crossfit_folds": screening_folds,
        "screening_operating_points": screening_operating,
        "resource_capture_curve": resource_curve,
        "resource_scenario_surface": resource_surface,
        "resource_break_even": break_even,
        "severity_persistence_systems": severity,
        "severity_persistence_summary": severity_summary,
        "continuous_interaction_systems": continuous_detail,
        "continuous_interaction_summary": continuous_summary,
        "leave_one_state_out": state_influence,
        "disjoint_grid_system_directions": disjoint_detail,
        "disjoint_grid_summary": disjoint_summary,
        "chronological_disjoint_B60_systems": chronological_detail,
        "chronological_disjoint_B60_summary": chronological_summary,
        "chronological_disjoint_threshold_curve": chronological_threshold,
        "cohort_operator_margin_sensitivity": cohort_operator,
        "resource_unit_systems": resource_audit_detail,
        "resource_unit_summary": resource_audit_summary,
        "joint_round_site_grid_audit": joint_grid_audit,
        "joint_round_site_system_probabilities": joint_surface_detail,
        "joint_round_site_capture_surface": joint_surface,
    }
    for stem, frame in tables.items():
        save_table(frame, stem)

    multi_date_row = audit_summary.loc[audit_summary["metric"].eq("multi_date")].iloc[0]
    b60_band = threshold_band.loc[threshold_band["threshold_ug_l"].eq(60.0)].iloc[0]
    crossfit = screening_operating.loc[screening_operating["policy"].eq("cross_fitted_HAA5_trigger")].iloc[0]
    direct = screening_operating.loc[screening_operating["policy"].eq("HAA9_at_60_one_event_code_group")].iloc[0]
    summary = {
        "analysis_date": "2026-09-14",
        "eligible_systems": int(len(full)),
        "reference_positive_systems_B60": int(full["HAA9_full_max"].gt(THRESHOLD).sum()),
        "round_groups": int(len(audit)),
        "round_multi_date_count": int(multi_date_row["count"]),
        "round_multi_date_fraction": float(multi_date_row["fraction"]),
        "B60_interaction": float(b60_band["interaction"]),
        "B60_simultaneous_ci": [float(b60_band["simultaneous_ci_lower"]), float(b60_band["simultaneous_ci_upper"])],
        "crossfit_HAA5_trigger_sensitivity": float(crossfit["sensitivity"]),
        "crossfit_HAA5_trigger_followup_fraction": float(crossfit["followup_fraction"]),
        "direct_HAA9_one_round_sensitivity": float(direct["sensitivity"]),
        "continuous_interaction": continuous_summary.iloc[0].to_dict(),
        "disjoint_grid": disjoint_summary.to_dict(orient="records"),
        "chronological_disjoint_B60": chronological_summary.to_dict(orient="records"),
        "rectangular_observed_grid_systems": int(joint_grid_audit["rectangular_observed_grid"].sum()),
        "rectangular_reference_positive_systems": int(
            (joint_grid_audit["rectangular_observed_grid"] & joint_grid_audit["reference_positive_B60"]).sum()
        ),
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed": SEED,
    }
    with (OUT / "wr_core_summary.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    write_notes(summary)
    print(json.dumps(summary, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)))


if __name__ == "__main__":
    main()
