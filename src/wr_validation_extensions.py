#!/usr/bin/env python3
"""Water Research validation extensions for the DBP monitoring study.

This script is deliberately isolated from the frozen core release. It
writes only to ``wr_results/validation`` and addresses four claim-matched
validation questions:

1. exact-site versus same-date SYR4 HAA5-TTHM pairing;
2. possible UCMR4-SYR4 record overlap and exclusion sensitivities;
3. quarter-capture curves from all valid single-analyte SYR4 records; and
4. threshold-safe recovery of ICR ``-999`` totals using documented MRL bounds.

The ICR recovery never treats ``-999`` as a measured zero. A zero-valued helper
is used only as a binary coding device after the documented upper bound has
proved that the result cannot exceed 60 ug/L.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyodbc
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata


SEED = 20260914
BOOTSTRAP_REPS = 10_000
PRIMARY_OVERLAP_TOLERANCE_UG_L = 0.05
OVERLAP_TOLERANCES_UG_L = (0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0)
PRIMARY_ICR_TYPES = ("AVG", "AVG1", "AVG2", "MAX", "NFC")
ALL_DS_ICR_TYPES = ("ENT", "DSE", "AVG", "AVG1", "AVG2", "MAX", "SDS", "NFC")
HAA5_COMPONENTS = ("MCAA", "DCAA", "TCAA", "MBAA", "DBAA")
HAA9_COMPONENTS = HAA5_COMPONENTS + ("BCAA", "BDCAA", "CDBAA", "TBAA")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root (default: parent of src)",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_value(payload), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_table(frame: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(stem.with_suffix(".csv"), index=False, lineterminator="\n")
    frame.to_parquet(stem.with_suffix(".parquet"), index=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cluster_bootstrap_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    clusters: np.ndarray,
    reps: int,
    seed: int,
) -> tuple[float, float, float, int]:
    frame = pd.DataFrame(
        {
            "cluster": pd.Series(clusters, dtype="string"),
            "num": np.asarray(numerator, dtype=float),
            "den": np.asarray(denominator, dtype=float),
        }
    )
    agg = frame.groupby("cluster", sort=True, observed=True)[["num", "den"]].sum()
    values = agg.to_numpy(dtype=float)
    n_clusters = len(values)
    den = values[:, 1].sum()
    point = float(values[:, 0].sum() / den) if den > 0 else np.nan
    if n_clusters < 2 or reps <= 0:
        return point, np.nan, np.nan, n_clusters
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=float)
    batch_size = max(1, min(128, 2_000_000 // n_clusters))
    cursor = 0
    while cursor < reps:
        batch = min(batch_size, reps - cursor)
        indices = rng.integers(0, n_clusters, size=(batch, n_clusters), dtype=np.int32)
        num = values[indices, 0].sum(axis=1)
        den = values[indices, 1].sum(axis=1)
        draws[cursor : cursor + batch] = np.divide(
            num, den, out=np.full(batch, np.nan), where=den > 0
        )
        cursor += batch
    finite = draws[np.isfinite(draws)]
    lo, hi = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
    return point, float(lo), float(hi), n_clusters


def fixed_rank_cluster_bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    reps: int,
    seed: int,
) -> tuple[float, float, float, int, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    clusters = np.asarray(clusters).astype(str)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, clusters = x[ok], y[ok], clusters[ok]
    if len(x) < 3:
        return np.nan, np.nan, np.nan, len(x), len(np.unique(clusters))
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    point = float(np.corrcoef(rx, ry)[0, 1])
    frame = pd.DataFrame(
        {
            "cluster": clusters,
            "n": 1.0,
            "sx": rx,
            "sy": ry,
            "sxx": rx * rx,
            "syy": ry * ry,
            "sxy": rx * ry,
        }
    )
    columns = ["n", "sx", "sy", "sxx", "syy", "sxy"]
    values = frame.groupby("cluster", sort=True)[columns].sum().to_numpy(dtype=float)
    n_clusters = len(values)
    if n_clusters < 2 or reps <= 0:
        return point, np.nan, np.nan, len(x), n_clusters
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=float)
    batch_size = max(1, min(96, 1_500_000 // n_clusters))
    cursor = 0
    while cursor < reps:
        batch = min(batch_size, reps - cursor)
        indices = rng.integers(0, n_clusters, size=(batch, n_clusters), dtype=np.int32)
        totals = values[indices].sum(axis=1)
        n, sx, sy, sxx, syy, sxy = (totals[:, i] for i in range(6))
        cov = sxy - sx * sy / n
        vx = sxx - sx * sx / n
        vy = syy - sy * sy / n
        draws[cursor : cursor + batch] = np.divide(
            cov,
            np.sqrt(vx * vy),
            out=np.full(batch, np.nan),
            where=(vx > 0) & (vy > 0),
        )
        cursor += batch
    finite = draws[np.isfinite(draws)]
    lo, hi = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
    return point, float(lo), float(hi), len(x), n_clusters


def cluster_bootstrap_matrix_mean(
    matrix: np.ndarray,
    clusters: np.ndarray,
    reps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Bootstrap means for several columns using one set of system resamples."""
    matrix = np.asarray(matrix, dtype=float)
    ids = pd.Series(clusters, dtype="string")
    columns = [f"m{i}" for i in range(matrix.shape[1])]
    frame = pd.DataFrame(matrix, columns=columns)
    frame.insert(0, "cluster", ids.to_numpy())
    frame["den"] = 1.0
    agg = frame.groupby("cluster", sort=True, observed=True)[columns + ["den"]].sum()
    values = agg.to_numpy(dtype=float)
    n_clusters = len(values)
    point = matrix.mean(axis=0) if len(matrix) else np.full(matrix.shape[1], np.nan)
    if n_clusters < 2 or reps <= 0:
        return point, np.full(len(point), np.nan), np.full(len(point), np.nan), n_clusters
    rng = np.random.default_rng(seed)
    draws = np.empty((reps, matrix.shape[1]), dtype=float)
    batch_size = max(1, min(64, 1_000_000 // n_clusters))
    cursor = 0
    while cursor < reps:
        batch = min(batch_size, reps - cursor)
        indices = rng.integers(0, n_clusters, size=(batch, n_clusters), dtype=np.int32)
        totals = values[indices].sum(axis=1)
        draws[cursor : cursor + batch] = totals[:, :-1] / totals[:, -1, None]
        cursor += batch
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
    return point, lo, hi, n_clusters


def finite_grid_capture(n: int, m: int, k: int) -> float:
    if n <= 0 or m <= 0 or k <= 0:
        return 0.0
    if k >= n or n - m < k:
        return 1.0
    return 1.0 - math.comb(n - m, k) / math.comb(n, k)


def interval_status(detect: pd.Series, value: pd.Series, dl: pd.Series, threshold: float) -> pd.Series:
    detected = detect.eq(1) & value.ge(0) & np.isfinite(value)
    censored = detect.eq(0)
    upper_known = censored & dl.gt(0) & np.isfinite(dl)
    status = pd.Series("indeterminate", index=detect.index, dtype="string")
    status.loc[detected & value.gt(threshold)] = "above"
    status.loc[detected & value.le(threshold)] = "below"
    status.loc[upper_known & dl.le(threshold)] = "below"
    return status


def load_single_analyte(path: Path, analyte: str, threshold: float) -> tuple[pd.DataFrame, dict]:
    usecols = [
        "PWSID",
        "STATE_CODE",
        "SIX_YEAR_ID",
        "SAMPLE_ID",
        "SAMPLE_COLLECTION_DATE",
        "WATER_FACILITY_ID",
        "SAMPLING_POINT_ID",
        "SAMPLE_TYPE_CODE",
        "DETECT",
        "VALUE",
        "UNIT",
        "DETECTION_LIMIT_VALUE",
        "DETECTION_LIMIT_UNIT",
    ]
    raw = pd.read_csv(
        path,
        sep="\t",
        encoding="cp1252",
        encoding_errors="replace",
        dtype="string",
        usecols=usecols,
        low_memory=False,
    )
    result = pd.DataFrame(index=raw.index)
    result["analyte"] = analyte
    result["pwsid"] = raw["PWSID"].str.strip()
    result["state_code"] = raw["STATE_CODE"].str.strip()
    result["six_year_id"] = raw["SIX_YEAR_ID"].str.strip()
    result["sample_id"] = raw["SAMPLE_ID"].str.strip()
    result["sample_date"] = pd.to_datetime(
        raw["SAMPLE_COLLECTION_DATE"], format="%d-%b-%y", errors="coerce"
    )
    result["water_facility_id"] = raw["WATER_FACILITY_ID"].str.strip()
    result["sampling_point_id"] = raw["SAMPLING_POINT_ID"].str.strip()
    result["sample_type_code"] = raw["SAMPLE_TYPE_CODE"].str.strip()
    result["detect"] = pd.to_numeric(raw["DETECT"], errors="coerce").astype("Int8")
    value = pd.to_numeric(raw["VALUE"], errors="coerce")
    unit = raw["UNIT"].fillna("").str.strip().str.upper()
    result["value_ug_l"] = np.where(unit.eq("MG/L"), 1000.0 * value, value)
    dl = pd.to_numeric(raw["DETECTION_LIMIT_VALUE"], errors="coerce")
    dl_unit = raw["DETECTION_LIMIT_UNIT"].fillna("").str.strip().str.upper()
    result["dl_ug_l"] = np.where(
        dl_unit.eq("MG/L"),
        1000.0 * dl,
        np.where(dl_unit.eq("UG/L"), dl, np.nan),
    )
    result["threshold_status"] = interval_status(
        result["detect"], result["value_ug_l"], result["dl_ug_l"], threshold
    )
    result["year"] = result["sample_date"].dt.year.astype("Int16")
    result["quarter"] = result["sample_date"].dt.quarter.astype("Int8")
    valid_id = result["pwsid"].notna() & result["pwsid"].ne("") & result["six_year_id"].notna()
    valid_date = result["sample_date"].notna()
    valid_detect = result["detect"].isin([0, 1])
    valid_measurement = (
        (result["detect"].eq(1) & result["value_ug_l"].ge(0) & np.isfinite(result["value_ug_l"]))
        | result["detect"].eq(0)
    )
    valid = valid_id & valid_date & valid_detect & valid_measurement
    duplicate_ids = int(result["six_year_id"].duplicated(keep=False).sum())
    if duplicate_ids:
        raise ValueError(f"{analyte}: SIX_YEAR_ID is not unique ({duplicate_ids} row members)")
    audit = {
        "analyte": analyte,
        "source_file": path.name,
        "raw_rows": int(len(result)),
        "valid_single_analyte_rows": int(valid.sum()),
        "invalid_rows": int((~valid).sum()),
        "unique_six_year_ids": int(result["six_year_id"].nunique()),
        "duplicate_six_year_id_row_members": duplicate_ids,
        "systems": int(result.loc[valid, "pwsid"].nunique()),
        "date_min": result.loc[valid, "sample_date"].min(),
        "date_max": result.loc[valid, "sample_date"].max(),
        "detected_rows": int((valid & result["detect"].eq(1)).sum()),
        "nondetect_rows": int((valid & result["detect"].eq(0)).sum()),
        "classifiable_rows": int((valid & result["threshold_status"].isin(["above", "below"])).sum()),
        "indeterminate_rows": int((valid & result["threshold_status"].eq("indeterminate")).sum()),
        "threshold_ug_l": threshold,
    }
    return result.loc[valid].reset_index(drop=True), audit


def pairing_metrics(
    paired: pd.DataFrame,
    cohort_name: str,
    reps: int,
    seed: int,
) -> tuple[list[dict], pd.DataFrame]:
    rows: list[dict] = []
    both_detected = paired["haa5_detect"].eq(1) & paired["tthm_detect"].eq(1)
    rho, rho_lo, rho_hi, n_detected, rho_systems = fixed_rank_cluster_bootstrap_spearman(
        paired.loc[both_detected, "haa5_value_ug_l"].to_numpy(),
        paired.loc[both_detected, "tthm_value_ug_l"].to_numpy(),
        paired.loc[both_detected, "pwsid"].to_numpy(),
        reps,
        seed,
    )
    rows.append(
        {
            "cohort": cohort_name,
            "metric": "spearman_rho_both_detected",
            "estimate": rho,
            "cluster_bootstrap_ci_low": rho_lo,
            "cluster_bootstrap_ci_high": rho_hi,
            "n_pairs": n_detected,
            "n_systems": rho_systems,
            "bootstrap_reps": reps,
        }
    )
    joint = paired["haa5_threshold_status"].isin(["above", "below"]) & paired[
        "tthm_threshold_status"
    ].isin(["above", "below"])
    equal = paired.loc[joint, "haa5_threshold_status"].eq(
        paired.loc[joint, "tthm_threshold_status"]
    )
    agreement, lo, hi, n_systems = cluster_bootstrap_ratio(
        equal.astype(float).to_numpy(),
        np.ones(int(joint.sum()), dtype=float),
        paired.loc[joint, "pwsid"].to_numpy(),
        reps,
        seed + 1,
    )
    h_above = paired.loc[joint, "haa5_threshold_status"].eq("above").mean()
    t_above = paired.loc[joint, "tthm_threshold_status"].eq("above").mean()
    expected = h_above * t_above + (1 - h_above) * (1 - t_above)
    kappa = (agreement - expected) / (1 - expected) if expected < 1 else np.nan
    rows.extend(
        [
            {
                "cohort": cohort_name,
                "metric": "threshold_status_agreement",
                "estimate": agreement,
                "cluster_bootstrap_ci_low": lo,
                "cluster_bootstrap_ci_high": hi,
                "n_pairs": int(joint.sum()),
                "n_systems": n_systems,
                "bootstrap_reps": reps,
            },
            {
                "cohort": cohort_name,
                "metric": "cohen_kappa_descriptive",
                "estimate": kappa,
                "cluster_bootstrap_ci_low": np.nan,
                "cluster_bootstrap_ci_high": np.nan,
                "n_pairs": int(joint.sum()),
                "n_systems": int(paired.loc[joint, "pwsid"].nunique()),
                "bootstrap_reps": 0,
            },
            {
                "cohort": cohort_name,
                "metric": "all_linked_pairs",
                "estimate": float(len(paired)),
                "cluster_bootstrap_ci_low": np.nan,
                "cluster_bootstrap_ci_high": np.nan,
                "n_pairs": int(len(paired)),
                "n_systems": int(paired["pwsid"].nunique()),
                "bootstrap_reps": 0,
            },
        ]
    )
    cells = (
        paired.loc[joint]
        .groupby(["haa5_threshold_status", "tthm_threshold_status"], observed=True)
        .agg(n_pairs=("sample_id", "size"), n_systems=("pwsid", "nunique"))
        .reset_index()
    )
    cells.insert(0, "cohort", cohort_name)
    cells["fraction_jointly_classifiable"] = cells["n_pairs"] / int(joint.sum())
    return rows, cells


def build_overlap_ledger(
    ucmr: pd.DataFrame,
    syr_haa5: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set], dict[str, set], dict]:
    u = ucmr.copy().reset_index(drop=True)
    u["ucmr_row_id"] = np.arange(len(u), dtype=np.int64)
    u["sample_date"] = pd.to_datetime(u["CollectionDate"], errors="coerce")
    u["ucmr_event_key"] = (
        u[["PWSID", "FacilityID", "SamplePointID", "CollectionDate", "SampleEventCode", "SampleID"]]
        .fillna("")
        .astype(str)
        .agg("|".join, axis=1)
    )
    s = syr_haa5.copy().reset_index(drop=True)
    s["syr_row_id"] = np.arange(len(s), dtype=np.int64)
    s["syr_observed_value"] = np.where(s["detect"].eq(1), s["value_ug_l"], np.nan)
    ucols = [
        "ucmr_row_id",
        "ucmr_event_key",
        "PWSID",
        "sample_date",
        "SampleID",
        "FacilityID",
        "SamplePointID",
        "SampleEventCode",
        "HAA5",
    ]
    scols = [
        "syr_row_id",
        "pwsid",
        "sample_date",
        "six_year_id",
        "sample_id",
        "water_facility_id",
        "sampling_point_id",
        "sample_type_code",
        "detect",
        "value_ug_l",
        "syr_observed_value",
        "threshold_status",
    ]
    ledger = u[ucols].merge(
        s[scols],
        left_on=["PWSID", "sample_date"],
        right_on=["pwsid", "sample_date"],
        how="inner",
        validate="many_to_many",
    )
    ledger.rename(
        columns={
            "PWSID": "pwsid_ucmr",
            "SampleID": "ucmr_sample_id",
            "FacilityID": "ucmr_facility_id",
            "SamplePointID": "ucmr_sampling_point_id",
            "SampleEventCode": "ucmr_sample_event_code",
            "HAA5": "ucmr_haa5_ug_l",
            "six_year_id": "syr_six_year_id",
            "sample_id": "syr_sample_id",
            "water_facility_id": "syr_water_facility_id",
            "sampling_point_id": "syr_sampling_point_id",
            "sample_type_code": "syr_sample_type_code",
            "detect": "syr_detect",
            "value_ug_l": "syr_haa5_ug_l",
            "threshold_status": "syr_threshold_status",
        },
        inplace=True,
    )
    ledger.drop(columns=["pwsid"], inplace=True)
    ledger["absolute_difference_ug_l"] = (
        ledger["ucmr_haa5_ug_l"] - ledger["syr_observed_value"]
    ).abs()
    ledger["candidate_pair_id"] = np.arange(len(ledger), dtype=np.int64)

    tolerance_rows: list[dict] = []
    match_event_sets: dict[str, set] = {}
    match_syr_sets: dict[str, set] = {}
    detected_ledger = ledger[ledger["absolute_difference_ug_l"].notna()].copy()
    grouped = list(detected_ledger.groupby(["pwsid_ucmr", "sample_date"], sort=True))
    for tol in OVERLAP_TOLERANCES_UG_L:
        label = f"tol_{str(tol).replace('.', 'p')}_ug_l"
        ledger[label] = False
        selected_ids: list[int] = []
        for _, group in grouped:
            u_ids = np.sort(group["ucmr_row_id"].unique())
            s_ids = np.sort(group["syr_row_id"].unique())
            u_lookup = {value: i for i, value in enumerate(u_ids)}
            s_lookup = {value: i for i, value in enumerate(s_ids)}
            difference = np.full((len(u_ids), len(s_ids)), np.inf, dtype=float)
            pair_id: dict[tuple[int, int], int] = {}
            for row in group.itertuples(index=False):
                i = u_lookup[row.ucmr_row_id]
                j = s_lookup[row.syr_row_id]
                if row.absolute_difference_ug_l < difference[i, j]:
                    difference[i, j] = row.absolute_difference_ug_l
                    pair_id[(i, j)] = row.candidate_pair_id
            dummy_cost = max(tol + 1.0, 1.0)
            cost = np.full((len(u_ids), len(s_ids) + len(u_ids)), dummy_cost, dtype=float)
            allowed = difference <= tol + 1e-12
            cost[:, : len(s_ids)] = np.where(allowed, difference, dummy_cost * 1_000_000)
            # Small deterministic row-specific offsets prevent arbitrary dummy ties.
            for i in range(len(u_ids)):
                cost[i, len(s_ids) + i] = dummy_cost + i * 1e-12
            row_indices, col_indices = linear_sum_assignment(cost)
            for i, j in zip(row_indices, col_indices):
                if j < len(s_ids) and allowed[i, j]:
                    selected_ids.append(pair_id[(i, j)])
        selected = set(selected_ids)
        ledger.loc[ledger["candidate_pair_id"].isin(selected), label] = True
        matched = ledger[ledger[label]]
        match_event_sets[label] = set(matched["ucmr_event_key"])
        match_syr_sets[label] = set(
            zip(matched["pwsid_ucmr"].astype(str), matched["syr_sample_id"].astype(str))
        )
        tolerance_rows.append(
            {
                "tolerance_ug_l": tol,
                "one_to_one_matches": int(len(matched)),
                "matched_ucmr_events": int(matched["ucmr_event_key"].nunique()),
                "matched_syr4_records": int(matched["syr_six_year_id"].nunique()),
                "matched_systems": int(matched["pwsid_ucmr"].nunique()),
                "fraction_of_same_date_ucmr_events": (
                    matched["ucmr_event_key"].nunique() / ledger["ucmr_event_key"].nunique()
                ),
                "primary_tolerance": bool(abs(tol - PRIMARY_OVERLAP_TOLERANCE_UG_L) < 1e-12),
            }
        )
    summary = pd.DataFrame(tolerance_rows)
    audit = {
        "same_pwsid_date_candidate_rows": int(len(ledger)),
        "same_pwsid_date_ucmr_events": int(ledger["ucmr_event_key"].nunique()),
        "same_pwsid_date_syr4_records": int(ledger["syr_six_year_id"].nunique()),
        "same_pwsid_date_systems": int(ledger["pwsid_ucmr"].nunique()),
        "candidate_date_min": ledger["sample_date"].min(),
        "candidate_date_max": ledger["sample_date"].max(),
        "primary_tolerance_ug_l": PRIMARY_OVERLAP_TOLERANCE_UG_L,
        "interpretation": "candidate shared records, not proof of identical provenance",
    }
    return ledger, summary, match_event_sets, match_syr_sets, audit


def quarter_capture(
    records: pd.DataFrame,
    analyte: str,
    threshold: float,
    cohort: str,
    reps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = records[["pwsid", "sample_date", "year", "quarter", "threshold_status"]].copy()
    source["_above"] = source["threshold_status"].eq("above").astype("int8")
    source["_below"] = source["threshold_status"].eq("below").astype("int8")
    source["_indeterminate"] = source["threshold_status"].eq("indeterminate").astype("int8")
    quarter = (
        source.groupby(["pwsid", "year", "quarter"], observed=True, sort=False)
        .agg(
            n_records=("_above", "size"),
            n_above=("_above", "sum"),
            n_below=("_below", "sum"),
            n_indeterminate=("_indeterminate", "sum"),
        )
        .reset_index()
    )
    quarter["quarter_status"] = np.where(
        quarter["n_above"].gt(0),
        "above",
        np.where(quarter["n_indeterminate"].eq(0), "below", "indeterminate"),
    )
    quarter["analyte"] = analyte
    quarter["threshold_ug_l"] = threshold
    quarter["cohort"] = cohort
    annual = (
        quarter.assign(
            _above=quarter["quarter_status"].eq("above").astype("int8"),
            _indeterminate=quarter["quarter_status"].eq("indeterminate").astype("int8"),
        )
        .groupby(["pwsid", "year"], observed=True, sort=False)
        .agg(
            n_available_quarters=("quarter", "nunique"),
            n_above_quarters=("_above", "sum"),
            n_indeterminate_quarters=("_indeterminate", "sum"),
            n_records=("n_records", "sum"),
        )
        .reset_index()
    )
    annual["eligible_ge3_complete"] = annual["n_available_quarters"].ge(3) & annual[
        "n_indeterminate_quarters"
    ].eq(0)
    annual["full_grid_positive"] = annual["n_above_quarters"].gt(0)
    annual["analyte"] = analyte
    annual["threshold_ug_l"] = threshold
    annual["cohort"] = cohort
    curves: list[dict] = []
    targets: list[dict] = []
    eligible = annual[annual["eligible_ge3_complete"] & annual["full_grid_positive"]]
    for n_grid, group in eligible.groupby("n_available_quarters", observed=True, sort=True):
        n_grid = int(n_grid)
        k_values = list(range(1, n_grid + 1))
        matrix = np.column_stack(
            [
                group["n_above_quarters"].map(
                    lambda positives, k=k: finite_grid_capture(n_grid, int(positives), k)
                )
                for k in k_values
            ]
        )
        point, lo, hi, n_systems = cluster_bootstrap_matrix_mean(
            matrix,
            group["pwsid"].to_numpy(),
            reps,
            seed + 100 * n_grid,
        )
        local: list[dict] = []
        for j, k in enumerate(k_values):
            row = {
                "cohort": cohort,
                "analyte": analyte,
                "threshold_ug_l": threshold,
                "n_available_quarters": n_grid,
                "quarters_sampled_k": k,
                "mean_exact_capture_probability": point[j],
                "cluster_bootstrap_ci_low": lo[j],
                "cluster_bootstrap_ci_high": hi[j],
                "n_system_years": int(len(group)),
                "n_systems": n_systems,
                "bootstrap_reps": reps,
            }
            curves.append(row)
            local.append(row)
        for target in (0.5, 0.9):
            hits = [r for r in local if r["mean_exact_capture_probability"] >= target]
            targets.append(
                {
                    "cohort": cohort,
                    "analyte": analyte,
                    "n_available_quarters": n_grid,
                    "target_probability": target,
                    "minimum_k": hits[0]["quarters_sampled_k"] if hits else np.nan,
                    "status": "reached" if hits else "not reached on supported grid",
                    "n_system_years": int(len(group)),
                    "n_systems": int(group["pwsid"].nunique()),
                }
            )
    return quarter, annual, pd.DataFrame(curves), pd.DataFrame(targets)


def paired_records(pairs: pd.DataFrame, prefix: str, strict_site: bool) -> pd.DataFrame:
    use = pairs[pairs["analysis_included"]].copy()
    if strict_site:
        use = use[use["facility_match"] & use["sampling_point_match"]]
    result = pd.DataFrame(
        {
            "pwsid": use["pwsid"],
            "sample_date": pd.to_datetime(use["sample_date"], errors="coerce"),
            "year": use["year"],
            "quarter": use["quarter"],
            "threshold_status": use[f"{prefix}_threshold_status"],
        }
    )
    return result.reset_index(drop=True)


def factorial_contributions_from_events(events: pd.DataFrame, scenario: str) -> pd.DataFrame:
    grid = events[events["finite_grid_system_eligible"]].copy()
    round_table = (
        grid.groupby(["PWSID", "SampleEventCode"], observed=True, sort=False)
        .agg(HAA5_max=("HAA5", "max"), HAA9_max=("HAA9", "max"), n_events=("SampleID", "size"))
        .reset_index()
    )
    full = (
        round_table.groupby("PWSID", observed=True, sort=False)
        .agg(
            HAA5_full_max=("HAA5_max", "max"),
            HAA9_full_max=("HAA9_max", "max"),
            n_rounds=("SampleEventCode", "nunique"),
            n_events=("n_events", "sum"),
        )
        .reset_index()
    )
    rows: list[dict] = []
    for pwsid, group in round_table.groupby("PWSID", sort=True):
        maxima = full[full["PWSID"].eq(pwsid)].iloc[0]
        y00 = group["HAA5_max"].gt(60).to_numpy(dtype=float)
        y10 = group["HAA9_max"].gt(60).to_numpy(dtype=float)
        y01 = float(maxima["HAA5_full_max"] > 60)
        y11 = float(maxima["HAA9_full_max"] > 60)
        rows.append(
            {
                "scenario": scenario,
                "PWSID": pwsid,
                "n_rounds": int(len(group)),
                "n_events": int(maxima["n_events"]),
                "y00": float(y00.mean()),
                "y10": float(y10.mean()),
                "y01": y01,
                "y11": y11,
                "panel_gain_at_M0": float(y10.mean() - y00.mean()),
                "design_gain_at_HAA5": float(y01 - y00.mean()),
                "combined_gain": float(y11 - y00.mean()),
                "interaction": float(y11 - y10.mean() - y01 + y00.mean()),
            }
        )
    return pd.DataFrame(rows)


def factorial_sensitivity_summary(
    contributions: pd.DataFrame,
    scenario: str,
    reps: int,
    seed: int,
) -> list[dict]:
    reference = contributions[contributions["y11"].eq(1)].copy()
    rows: list[dict] = []
    metrics = (
        "y00",
        "y10",
        "y01",
        "y11",
        "panel_gain_at_M0",
        "design_gain_at_HAA5",
        "combined_gain",
        "interaction",
    )
    for j, metric in enumerate(metrics):
        if len(reference):
            point, lo, hi, n_systems = cluster_bootstrap_ratio(
                reference[metric].to_numpy(),
                np.ones(len(reference)),
                reference["PWSID"].to_numpy(),
                reps,
                seed + j,
            )
        else:
            point, lo, hi, n_systems = np.nan, np.nan, np.nan, 0
        rows.append(
            {
                "scenario": scenario,
                "conditioning": "HAA9_M1_reference_positive",
                "metric": metric,
                "estimate": point,
                "cluster_bootstrap_ci_low": lo,
                "cluster_bootstrap_ci_high": hi,
                "eligible_systems": int(len(contributions)),
                "reference_positive_systems": int(len(reference)),
                "n_systems_estimate": n_systems,
                "bootstrap_reps": reps,
            }
        )
    return rows


def read_icr_mrls(access_path: Path) -> pd.DataFrame:
    connection = pyodbc.connect(
        rf"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={access_path.resolve()}"
    )
    cursor = connection.cursor()
    cursor.execute("SELECT ANAL_ID, CNCNTCDE, MRL_MSR, ANALNAME, ABBRNAME, AUX1NAME FROM TUXANLYT")
    columns = [item[0].upper() for item in cursor.description]
    rows = [tuple(row) for row in cursor.fetchall()]
    connection.close()
    frame = pd.DataFrame.from_records(rows, columns=columns)
    frame["AUX1NAME"] = frame["AUX1NAME"].astype("string").str.strip()
    frame["ABBRNAME"] = frame["ABBRNAME"].astype("string").str.strip()
    frame["ANALNAME"] = frame["ANALNAME"].astype("string").str.strip()
    frame["CNCNTCDE"] = frame["CNCNTCDE"].astype("string").str.strip()
    frame["MRL_MSR"] = pd.to_numeric(frame["MRL_MSR"], errors="coerce")
    wanted = set(HAA9_COMPONENTS)
    result = frame[frame["AUX1NAME"].isin(wanted)].copy()
    result["in_HAA5"] = result["AUX1NAME"].isin(HAA5_COMPONENTS)
    result["in_HAA9"] = result["AUX1NAME"].isin(HAA9_COMPONENTS)
    result.sort_values("ANAL_ID", inplace=True)
    return result.reset_index(drop=True)


def recover_icr_minus999(
    joined: pd.DataFrame,
    mrls: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    adjusted = joined.copy()
    audit_rows: list[dict] = []
    event_rows: list[pd.DataFrame] = []
    for total, components in (("HAA5", HAA5_COMPONENTS), ("HAA9", HAA9_COMPONENTS)):
        mrl = mrls.set_index("AUX1NAME").loc[list(components), "MRL_MSR"]
        mrl_sum = float(mrl.sum())
        if not mrl.notna().all() or mrl_sum >= 60:
            raise ValueError(f"Documented {total} MRL bound is unavailable or not below 60 ug/L")
        value = pd.to_numeric(adjusted[total], errors="coerce")
        component_values = adjusted[list(components)].apply(pd.to_numeric, errors="coerce")
        total_minus999 = value.eq(-999)
        components_all_minus999 = component_values.eq(-999).all(axis=1)
        recoverable = total_minus999 & components_all_minus999
        inconsistent = total_minus999 & ~components_all_minus999
        if inconsistent.any():
            raise ValueError(f"{total}: {int(inconsistent.sum())} -999 totals conflict with components")
        adjusted.loc[recoverable, total] = 0.0
        event = adjusted.loc[
            recoverable,
            ["EVENT_ID", "ICRPWSID", "SAMP_PER", "SAMP_QTR", "EVNTTYPE"],
        ].copy()
        event["analyte"] = total
        event["reported_total"] = -999.0
        event["documented_mrl_sum_upper_bound_ug_l"] = mrl_sum
        event["definitely_at_or_below_60"] = True
        event["threshold_binary_helper_value"] = 0.0
        event["helper_is_concentration_imputation"] = False
        event_rows.append(event)
        scopes = {
            "all_dbp": pd.Series(True, index=adjusted.index),
            "primary_distribution": adjusted["EVNTTYPE"].isin(PRIMARY_ICR_TYPES),
            "all_ds_sensitivity": adjusted["EVNTTYPE"].isin(ALL_DS_ICR_TYPES),
        }
        for scope, keep in scopes.items():
            audit_rows.append(
                {
                    "scope": scope,
                    "analyte": total,
                    "component_count": len(components),
                    "documented_component_mrl_sum_ug_l": mrl_sum,
                    "threshold_ug_l": 60.0,
                    "reported_minus999_events": int((keep & total_minus999).sum()),
                    "component_verified_minus999_events": int((keep & recoverable).sum()),
                    "inconsistent_minus999_events": int((keep & inconsistent).sum()),
                    "recovered_definitely_below_events": int((keep & recoverable).sum()),
                    "recovery_rule": "total and every named component are -999; sum of component MRLs is below 60 ug/L",
                }
            )
    events = pd.concat(event_rows, ignore_index=True)
    return adjusted, pd.DataFrame(audit_rows), events


def tag(frame: pd.DataFrame, analysis_version: str) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "analysis_version", analysis_version)
    return result


def tag_icr(frame: pd.DataFrame, analysis_version: str) -> pd.DataFrame:
    result = tag(frame, analysis_version)
    if analysis_version == "mrl_bounded_recovery":
        handling = "verified -999 totals classified at or below 60 using component-MRL upper bounds"
        if "threshold_definition" in result.columns:
            result["threshold_definition"] = (
                ">60 ug/L; verified -999 totals are below via MRL bounds; null remains unsupported"
            )
    else:
        handling = "-999 totals excluded from numeric threshold support"
    result.insert(1, "minus999_handling", handling)
    result.insert(2, "binary_helper_is_concentration_imputation", False)
    return result


def methods_note(summary: dict) -> str:
    site = summary["syr4_pairing"]
    overlap = summary["cross_program_overlap"]
    icr = summary["icr_minus999_recovery"]
    capture = summary["syr4_single_analyte_capture"]
    ucmr = summary["ucmr4_overlap_exclusion"]["factorial_headlines"]
    return f"""# Water Research validation extensions

## Scope

This analysis is isolated from the frozen core outputs. It creates only files under `wr_results/validation`. All thresholds are analytical signal benchmarks, not compliance determinations.

## Exact-site SYR4 pairing

HAA5 and TTHM records were first linked on exact PWSID and SAMPLE_ID, as in the prior analysis. The strict-site primary cohort additionally required identical collection date, WATER_FACILITY_ID and SAMPLING_POINT_ID. The same-date all-pairs cohort was retained as a sensitivity. This resolved {site['sampling_point_mismatch_pairs']:,} same-date links with nonmatching sampling points by excluding them from the strict-site primary cohort. The mismatches were retained in a row-level audit, not silently relabelled.

Detected HAA5-TTHM association used Spearman correlation. Threshold agreement used only pairs whose censoring intervals were classifiable at 60 and 80 ug/L. Ninety-five percent intervals used {summary['metadata']['bootstrap_reps']:,} whole-system resamples. The strict-site cohort contained {site['strict_site_pairs']:,} pairs in {site['strict_site_systems']:,} systems; its detected-pair rho was {site['strict_site_rho']:.6f} and threshold agreement was {site['strict_site_agreement']:.6f}. The all-pairs sensitivity contained {site['same_date_pairs']:,} pairs, with rho {site['same_date_rho']:.6f} and agreement {site['same_date_agreement']:.6f}.

## UCMR4-SYR4 overlap

Complete UCMR4 HAA triplets were joined to all valid SYR4 HAA5 records on exact PWSID and collection date. Concentration differences were evaluated with one-to-one minimum-cost matching at 0, 0.001, 0.01, 0.05, 0.1, 0.5 and 1.0 ug/L. The primary candidate definition used <=0.05 ug/L, which accommodates rounding to the nearest 0.1 ug/L without asserting record identity. Same-day concurrence and concentration similarity are candidate overlap, not proof of identical provenance.

There were {overlap['same_date_ucmr_events']:,} UCMR4 events in {overlap['same_date_systems']:,} systems with at least one SYR4 HAA5 record on the same date. At the primary tolerance, one-to-one matching identified {overlap['primary_matches']:,} candidate event pairs in {overlap['primary_match_systems']:,} systems. Robustness analyses removed matched event identifiers, all concurrent dates, matched systems, and all systems with a concurrent date. The exact paired-association and UCMR4 factorial results for each exclusion are in the output tables.

The original UCMR4 interaction was {ucmr['original_complete_grid']['interaction']:.6f}. It was {ucmr['exclude_primary_candidate_events_recomputed']['interaction']:.6f} after removing primary candidate events, {ucmr['exclude_primary_candidate_systems']['interaction']:.6f} after removing their systems, {ucmr['exclude_any_concurrent_date_events_recomputed']['interaction']:.6f} after removing every same-date UCMR4 event, and {ucmr['exclude_any_concurrent_date_systems']['interaction']:.6f} after removing every system with any same-date record.

## Full-record SYR4 quarter capture

Quarter-capture curves were rebuilt separately for HAA5 and TTHM from every valid single-analyte record keyed by the publisher-unique SIX_YEAR_ID, including records without SAMPLE_ID. Detected values were point observations. Nondetects were bounded from zero to a positive unit-normalized detection limit; unknown or unsupported bounds were indeterminate. A quarter was above threshold if any record was above, below if all records were definitely below, and otherwise indeterminate. System-years required at least three observed, classifiable quarters and at least one above-threshold quarter. Capture for k of N quarters was enumerated exactly without replacement and averaged with equal weight per system-year. The prior paired-only design was retained in both same-date and exact-site forms.

The all-record inputs contained {capture['HAA5_valid_rows']:,} valid HAA5 records and {capture['TTHM_valid_rows']:,} valid TTHM records. Curves and target crossings are reported without replacing the full-record analysis with the paired subset.

For four-quarter, reference-positive system-years, one randomly selected quarter captured {capture['full_record_four_quarter_capture']['HAA5']['capture_k1']:.6f} of HAA5 signals and {capture['full_record_four_quarter_capture']['TTHM']['capture_k1']:.6f} of TTHM signals. Two quarters increased those probabilities to {capture['full_record_four_quarter_capture']['HAA5']['capture_k2']:.6f} and {capture['full_record_four_quarter_capture']['TTHM']['capture_k2']:.6f}, respectively.

## ICR -999 recovery

The ICR guide states that -999 denotes a result below the minimum reporting level, that below-MRL components are replaced by zero only when calculating totals, and that HAA5 or HAA9 equals -999 when all named components are below their MRLs. MRLs were read directly from TUXANLYT in the supplied Access database. The HAA5 component MRLs sum to {icr['HAA5_mrl_sum_ug_l']:.1f} ug/L and the HAA9 component MRLs sum to {icr['HAA9_mrl_sum_ug_l']:.1f} ug/L, both below 60 ug/L. Every recovered total also had all required components coded -999.

For threshold classification only, verified -999 totals were coded with a binary helper value that cannot exceed 60 ug/L. This is not a concentration imputation and is never used in concentration summaries. The analysis recovered {icr['HAA5_recovered_events']:,} HAA5 and {icr['HAA9_recovered_events']:,} HAA9 event states. Original exclusion and MRL-bounded recovery results are supplied side by side for temporal support, capture and the matched factorial.

In the primary ICR scope, complete six-quarter support increased from {icr['primary_full_six_quarter_systems']['HAA5_original']} to {icr['primary_full_six_quarter_systems']['HAA5_recovered']} systems for HAA5 and from {icr['primary_full_six_quarter_systems']['HAA9_original']} to {icr['primary_full_six_quarter_systems']['HAA9_recovered']} for HAA9. The complete-six-quarter matched factorial increased from {icr['matched_factorial_complete_six_quarters']['eligible_systems_original']} to {icr['matched_factorial_complete_six_quarters']['eligible_systems_recovered']} eligible systems and from {icr['matched_factorial_complete_six_quarters']['reference_positive_systems_original']} to {icr['matched_factorial_complete_six_quarters']['reference_positive_systems_recovered']} HAA9-reference-positive systems.

## Reproducibility

All tables are supplied as CSV and Parquet. The JSON summary records software, thresholds, rules and exact headline counts. The manifest records SHA-256 hashes for every generated output. Random seeds are fixed; finite-grid probabilities and one-to-one overlap assignments are deterministic.
"""


def main() -> int:
    args = parse_args()
    if args.bootstrap_reps < 0:
        raise ValueError("--bootstrap-reps must be nonnegative")
    root = args.root.resolve()
    out = root / "wr_results" / "validation"
    out.mkdir(parents=True, exist_ok=True)

    pair_path = root / "data" / "processed" / "validation" / "syr4_exact_pairs.parquet"
    ucmr_path = root / "data" / "processed" / "ucmr4" / "paired_haa_events.parquet"
    icr_path = root / "data" / "processed" / "validation" / "icr_dbp_sample_wqp_residual_joined.parquet"
    haa5_path = root / "data" / "interim" / "extracted" / "SYR4_HAA5.txt"
    tthm_path = root / "data" / "interim" / "extracted" / "SYR4_TTHM.txt"
    access_path = root / "data" / "interim" / "extracted" / "DBPICR_Aux1.accdb"
    required = [pair_path, ucmr_path, icr_path, haa5_path, tthm_path, access_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    print("Loading SYR4 paired cohort and full HAA5 archive...", flush=True)
    pairs = pd.read_parquet(pair_path)
    same_date = pairs[pairs["analysis_included"]].copy()
    strict_site = same_date[
        same_date["facility_match"].fillna(False) & same_date["sampling_point_match"].fillna(False)
    ].copy()
    strict_site_type = strict_site[strict_site["sample_type_match"].fillna(False)].copy()
    mismatch_ledger = same_date[~same_date["sampling_point_match"].fillna(False)].copy()
    mismatch_ledger["pwsid_prefix"] = mismatch_ledger["pwsid"].str[:2]
    save_table(mismatch_ledger, out / "syr4_sampling_point_mismatch_ledger")

    syr_haa5, haa5_audit = load_single_analyte(haa5_path, "HAA5", 60.0)
    ucmr = pd.read_parquet(ucmr_path)
    print("Building UCMR4-SYR4 overlap ledger...", flush=True)
    ledger, tolerance_summary, match_event_sets, match_syr_sets, overlap_audit = build_overlap_ledger(
        ucmr, syr_haa5
    )
    save_table(ledger, out / "ucmr4_syr4_overlap_ledger")
    save_table(tolerance_summary, out / "ucmr4_syr4_overlap_tolerance_summary")

    primary_label = f"tol_{str(PRIMARY_OVERLAP_TOLERANCE_UG_L).replace('.', 'p')}_ug_l"
    primary_overlap = ledger[ledger[primary_label]].copy()
    primary_event_keys = set(primary_overlap["ucmr_event_key"])
    primary_syr_keys = set(
        zip(primary_overlap["pwsid_ucmr"].astype(str), primary_overlap["syr_sample_id"].astype(str))
    )
    primary_systems = set(primary_overlap["pwsid_ucmr"].astype(str))
    concurrent_event_keys = set(ledger["ucmr_event_key"].astype(str))
    concurrent_systems = set(ledger["pwsid_ucmr"].astype(str))
    concurrent_dates = set(
        zip(ledger["pwsid_ucmr"].astype(str), pd.to_datetime(ledger["sample_date"]).dt.date)
    )

    print("Estimating exact-site and overlap-exclusion pairing metrics...", flush=True)
    pair_key = list(zip(strict_site["pwsid"].astype(str), strict_site["sample_id"].astype(str)))
    pair_date_key = list(
        zip(strict_site["pwsid"].astype(str), pd.to_datetime(strict_site["sample_date"]).dt.date)
    )
    cohorts = {
        "same_date_all_pairs": same_date,
        "strict_site_primary": strict_site,
        "strict_site_plus_sample_type": strict_site_type,
        "strict_site_exclude_primary_candidate_events": strict_site[
            ~pd.Series(pair_key, index=strict_site.index).isin(primary_syr_keys)
        ],
        "strict_site_exclude_any_concurrent_dates": strict_site[
            ~pd.Series(pair_date_key, index=strict_site.index).isin(concurrent_dates)
        ],
        "strict_site_exclude_primary_candidate_systems": strict_site[
            ~strict_site["pwsid"].astype(str).isin(primary_systems)
        ],
        "strict_site_exclude_any_concurrent_systems": strict_site[
            ~strict_site["pwsid"].astype(str).isin(concurrent_systems)
        ],
    }
    metric_rows: list[dict] = []
    cell_frames: list[pd.DataFrame] = []
    for index, (name, cohort) in enumerate(cohorts.items()):
        rows, cells = pairing_metrics(cohort, name, args.bootstrap_reps, args.seed + 1_000 * index)
        metric_rows.extend(rows)
        cell_frames.append(cells)
    pairing_metric_table = pd.DataFrame(metric_rows)
    pairing_cells = pd.concat(cell_frames, ignore_index=True)
    save_table(pairing_metric_table, out / "syr4_strict_site_and_overlap_sensitivity_metrics")
    save_table(pairing_cells, out / "syr4_strict_site_and_overlap_concordance_cells")

    print("Rebuilding quarter-capture curves from all single-analyte records...", flush=True)
    quarter_frames: list[pd.DataFrame] = []
    annual_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    target_frames: list[pd.DataFrame] = []
    q, a, c, t = quarter_capture(
        syr_haa5,
        "HAA5",
        60.0,
        "all_valid_single_analyte_records",
        args.bootstrap_reps,
        args.seed + 20_000,
    )
    quarter_frames.append(q)
    annual_frames.append(a)
    curve_frames.append(c)
    target_frames.append(t)
    syr_tthm, tthm_audit = load_single_analyte(tthm_path, "TTHM", 80.0)
    q, a, c, t = quarter_capture(
        syr_tthm,
        "TTHM",
        80.0,
        "all_valid_single_analyte_records",
        args.bootstrap_reps,
        args.seed + 30_000,
    )
    quarter_frames.append(q)
    annual_frames.append(a)
    curve_frames.append(c)
    target_frames.append(t)
    for analyte, prefix, threshold, local_seed in (
        ("HAA5", "haa5", 60.0, args.seed + 40_000),
        ("TTHM", "tthm", 80.0, args.seed + 50_000),
    ):
        for strict, cohort_name, offset in (
            (False, "paired_same_date_sensitivity", 0),
            (True, "paired_strict_site_sensitivity", 500),
        ):
            records = paired_records(pairs, prefix, strict)
            q, a, c, t = quarter_capture(
                records,
                analyte,
                threshold,
                cohort_name,
                args.bootstrap_reps,
                local_seed + offset,
            )
            quarter_frames.append(q)
            annual_frames.append(a)
            curve_frames.append(c)
            target_frames.append(t)
    quarter_grid = pd.concat(quarter_frames, ignore_index=True)
    annual_support = pd.concat(annual_frames, ignore_index=True)
    capture_curves = pd.concat(curve_frames, ignore_index=True)
    capture_targets = pd.concat(target_frames, ignore_index=True)
    save_table(quarter_grid, out / "syr4_quarter_grid_extended")
    save_table(annual_support, out / "syr4_system_year_support_extended")
    save_table(capture_curves, out / "syr4_quarter_capture_curves_extended")
    save_table(capture_targets, out / "syr4_quarter_capture_targets_extended")
    input_audit = pd.DataFrame([haa5_audit, tthm_audit])
    save_table(input_audit, out / "syr4_single_analyte_input_audit")

    print("Running UCMR4 candidate-overlap exclusion sensitivities...", flush=True)
    baseline_contrib = pd.read_parquet(
        root / "data" / "processed" / "ucmr4" / "system_contributions_primary_B60.parquet"
    )
    baseline_slim = baseline_contrib[
        [
            "PWSID",
            "admissible_unit_count",
            "y00",
            "y10",
            "y01",
            "y11",
            "panel_gain_at_M0",
            "design_gain_at_HAA5",
            "combined_gain",
            "interaction",
        ]
    ].rename(columns={"admissible_unit_count": "n_rounds"})
    baseline_slim.insert(0, "scenario", "original_complete_grid")
    baseline_slim["n_events"] = np.nan
    system_primary = baseline_slim[~baseline_slim["PWSID"].astype(str).isin(primary_systems)].copy()
    system_primary["scenario"] = "exclude_primary_candidate_systems"
    system_concurrent = baseline_slim[
        ~baseline_slim["PWSID"].astype(str).isin(concurrent_systems)
    ].copy()
    system_concurrent["scenario"] = "exclude_any_concurrent_date_systems"
    ucmr_work = ucmr.copy()
    ucmr_work["sample_date"] = pd.to_datetime(ucmr_work["CollectionDate"], errors="coerce")
    ucmr_work["ucmr_event_key"] = (
        ucmr_work[
            ["PWSID", "FacilityID", "SamplePointID", "CollectionDate", "SampleEventCode", "SampleID"]
        ]
        .fillna("")
        .astype(str)
        .agg("|".join, axis=1)
    )
    recomputed_baseline = factorial_contributions_from_events(
        ucmr_work, "original_recomputed_for_validation"
    )
    event_primary = factorial_contributions_from_events(
        ucmr_work[~ucmr_work["ucmr_event_key"].isin(primary_event_keys)],
        "exclude_primary_candidate_events_recomputed",
    )
    event_concurrent = factorial_contributions_from_events(
        ucmr_work[~ucmr_work["ucmr_event_key"].isin(concurrent_event_keys)],
        "exclude_any_concurrent_date_events_recomputed",
    )
    all_factorial = pd.concat(
        [baseline_slim, system_primary, system_concurrent, event_primary, event_concurrent],
        ignore_index=True,
        sort=False,
    )
    save_table(all_factorial, out / "ucmr4_overlap_exclusion_system_contributions")
    factorial_rows: list[dict] = []
    for j, (scenario, group) in enumerate(all_factorial.groupby("scenario", sort=False)):
        factorial_rows.extend(
            factorial_sensitivity_summary(group, scenario, args.bootstrap_reps, args.seed + 60_000 + 100 * j)
        )
    factorial_summary = pd.DataFrame(factorial_rows)
    save_table(factorial_summary, out / "ucmr4_overlap_exclusion_factorial_metrics")

    print("Recovering threshold-safe ICR -999 states...", flush=True)
    icr = pd.read_parquet(icr_path)
    mrls = read_icr_mrls(access_path)
    save_table(mrls, out / "icr_component_mrls")
    adjusted_icr, recovery_audit, recovered_events = recover_icr_minus999(icr, mrls)
    save_table(recovery_audit, out / "icr_minus999_recovery_audit")
    save_table(recovered_events, out / "icr_minus999_recovered_event_states")
    # Reuse the frozen analysis functions without writing to their output directories.
    sys.path.insert(0, str(root / "src"))
    import validation_analysis as va

    original_capture = va.icr_capture_analysis(icr, args.bootstrap_reps, args.seed + 70_000)
    recovered_capture = va.icr_capture_analysis(
        adjusted_icr, args.bootstrap_reps, args.seed + 80_000
    )
    capture_names = (
        "icr_temporal_system_grids_comparison",
        "icr_temporal_support_comparison",
        "icr_temporal_capture_curve_comparison",
        "icr_capture_targets_comparison",
    )
    for name, original_frame, recovered_frame in zip(capture_names, original_capture, recovered_capture):
        save_table(
            pd.concat(
                [
                    tag_icr(original_frame, "original_minus999_excluded"),
                    tag_icr(recovered_frame, "mrl_bounded_recovery"),
                ],
                ignore_index=True,
                sort=False,
            ),
            out / name,
        )
    original_factorial = va.icr_matched_factorial(icr, args.bootstrap_reps, args.seed + 90_000)
    recovered_factorial = va.icr_matched_factorial(
        adjusted_icr, args.bootstrap_reps, args.seed + 100_000
    )
    factorial_names = (
        "icr_matched_factorial_systems_comparison",
        "icr_matched_factorial_support_comparison",
        "icr_matched_factorial_cells_comparison",
        "icr_matched_factorial_contrasts_comparison",
        "icr_matched_factorial_repair_fractions_comparison",
    )
    for name, original_frame, recovered_frame in zip(
        factorial_names, original_factorial, recovered_factorial
    ):
        save_table(
            pd.concat(
                [
                    tag_icr(original_frame, "original_minus999_excluded"),
                    tag_icr(recovered_frame, "mrl_bounded_recovery"),
                ],
                ignore_index=True,
                sort=False,
            ),
            out / name,
        )

    def metric_value(cohort: str, metric: str) -> float:
        match = pairing_metric_table[
            pairing_metric_table["cohort"].eq(cohort) & pairing_metric_table["metric"].eq(metric)
        ]
        return float(match.iloc[0]["estimate"])

    factorial_headlines: dict[str, dict] = {}
    for scenario, group in factorial_summary.groupby("scenario", sort=False):
        metric_lookup = group.set_index("metric")["estimate"]
        factorial_headlines[scenario] = {
            "eligible_systems": int(group["eligible_systems"].iloc[0]),
            "reference_positive_systems": int(group["reference_positive_systems"].iloc[0]),
            "p00": float(metric_lookup["y00"]),
            "p10": float(metric_lookup["y10"]),
            "p01": float(metric_lookup["y01"]),
            "p11": float(metric_lookup["y11"]),
            "interaction": float(metric_lookup["interaction"]),
        }

    full_four = capture_curves[
        capture_curves["cohort"].eq("all_valid_single_analyte_records")
        & capture_curves["n_available_quarters"].eq(4)
    ]
    four_quarter_headlines = {
        analyte: {
            f"capture_k{int(row.quarters_sampled_k)}": float(row.mean_exact_capture_probability)
            for row in group.itertuples(index=False)
        }
        for analyte, group in full_four.groupby("analyte", sort=True)
    }

    original_support = original_capture[1]
    recovered_support = recovered_capture[1]
    original_factor_support = original_factorial[1]
    recovered_factor_support = recovered_factorial[1]

    def support_value(frame: pd.DataFrame, analyte: str, column: str) -> int:
        match = frame[
            frame["scope"].eq("primary_distribution")
            & frame["analyte"].eq(analyte)
            & frame["time_scale"].eq("quarterly")
        ]
        return int(match.iloc[0][column])

    def complete_factor_value(frame: pd.DataFrame, column: str) -> int:
        match = frame[frame["cohort"].eq("quarterly_complete_six_quarters")]
        return int(match.iloc[0][column])

    primary_tol_row = tolerance_summary[tolerance_summary["primary_tolerance"]].iloc[0]
    mrl_lookup = mrls.set_index("AUX1NAME")["MRL_MSR"]
    summary = {
        "metadata": {
            "analysis": "Water Research claim-matched validation extensions",
            "seed": args.seed,
            "bootstrap_reps": args.bootstrap_reps,
            "threshold_operator": "strictly greater than",
            "thresholds_ug_l": {"HAA5": 60.0, "TTHM": 80.0, "ICR_HAA9": 60.0},
            "primary_overlap_tolerance_ug_l": PRIMARY_OVERLAP_TOLERANCE_UG_L,
            "software": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "pyodbc": pyodbc.version,
            },
        },
        "syr4_pairing": {
            "same_date_pairs": int(len(same_date)),
            "same_date_systems": int(same_date["pwsid"].nunique()),
            "sampling_point_mismatch_pairs": int(len(mismatch_ledger)),
            "sampling_point_mismatch_systems": int(mismatch_ledger["pwsid"].nunique()),
            "sampling_point_mismatch_state_prefixes": mismatch_ledger["pwsid_prefix"].value_counts().to_dict(),
            "strict_site_pairs": int(len(strict_site)),
            "strict_site_systems": int(strict_site["pwsid"].nunique()),
            "strict_site_plus_sample_type_pairs": int(len(strict_site_type)),
            "same_date_rho": metric_value("same_date_all_pairs", "spearman_rho_both_detected"),
            "same_date_agreement": metric_value("same_date_all_pairs", "threshold_status_agreement"),
            "strict_site_rho": metric_value("strict_site_primary", "spearman_rho_both_detected"),
            "strict_site_agreement": metric_value("strict_site_primary", "threshold_status_agreement"),
            "exclude_primary_candidate_systems_rho": metric_value(
                "strict_site_exclude_primary_candidate_systems", "spearman_rho_both_detected"
            ),
            "exclude_primary_candidate_systems_agreement": metric_value(
                "strict_site_exclude_primary_candidate_systems", "threshold_status_agreement"
            ),
            "exclude_any_concurrent_systems_rho": metric_value(
                "strict_site_exclude_any_concurrent_systems", "spearman_rho_both_detected"
            ),
            "exclude_any_concurrent_systems_agreement": metric_value(
                "strict_site_exclude_any_concurrent_systems", "threshold_status_agreement"
            ),
        },
        "cross_program_overlap": {
            "same_date_candidate_rows": overlap_audit["same_pwsid_date_candidate_rows"],
            "same_date_ucmr_events": overlap_audit["same_pwsid_date_ucmr_events"],
            "same_date_syr4_records": overlap_audit["same_pwsid_date_syr4_records"],
            "same_date_systems": overlap_audit["same_pwsid_date_systems"],
            "primary_tolerance_ug_l": PRIMARY_OVERLAP_TOLERANCE_UG_L,
            "primary_matches": int(primary_tol_row["one_to_one_matches"]),
            "primary_match_systems": int(primary_tol_row["matched_systems"]),
            "interpretation": "candidate shared records, not proof of identical provenance",
            "tolerance_results": tolerance_summary.to_dict(orient="records"),
        },
        "syr4_single_analyte_capture": {
            "HAA5_valid_rows": haa5_audit["valid_single_analyte_rows"],
            "HAA5_systems": haa5_audit["systems"],
            "TTHM_valid_rows": tthm_audit["valid_single_analyte_rows"],
            "TTHM_systems": tthm_audit["systems"],
            "curve_rows": int(len(capture_curves)),
            "target_rows": int(len(capture_targets)),
            "cohorts": sorted(capture_curves["cohort"].unique().tolist()),
            "full_record_four_quarter_capture": four_quarter_headlines,
        },
        "ucmr4_overlap_exclusion": {
            "primary_candidate_systems": int(len(primary_systems)),
            "any_concurrent_date_systems": int(len(concurrent_systems)),
            "primary_candidate_events": int(len(primary_event_keys)),
            "any_concurrent_date_events": int(len(concurrent_event_keys)),
            "factorial_scenarios": sorted(factorial_summary["scenario"].unique().tolist()),
            "factorial_headlines": factorial_headlines,
        },
        "icr_minus999_recovery": {
            "HAA5_mrl_sum_ug_l": float(mrl_lookup.loc[list(HAA5_COMPONENTS)].sum()),
            "HAA9_mrl_sum_ug_l": float(mrl_lookup.loc[list(HAA9_COMPONENTS)].sum()),
            "HAA5_recovered_events": int((recovered_events["analyte"] == "HAA5").sum()),
            "HAA9_recovered_events": int((recovered_events["analyte"] == "HAA9").sum()),
            "all_component_consistency_failures": int(recovery_audit["inconsistent_minus999_events"].sum()),
            "coding_rule": "zero helper only for binary <=60 classification; never a concentration imputation",
            "primary_full_six_quarter_systems": {
                "HAA5_original": support_value(original_support, "HAA5", "full_six_quarter_systems"),
                "HAA5_recovered": support_value(recovered_support, "HAA5", "full_six_quarter_systems"),
                "HAA9_original": support_value(original_support, "HAA9", "full_six_quarter_systems"),
                "HAA9_recovered": support_value(recovered_support, "HAA9", "full_six_quarter_systems"),
            },
            "matched_factorial_complete_six_quarters": {
                "eligible_systems_original": complete_factor_value(original_factor_support, "eligible_systems"),
                "eligible_systems_recovered": complete_factor_value(recovered_factor_support, "eligible_systems"),
                "reference_positive_systems_original": complete_factor_value(
                    original_factor_support, "reference_positive_systems"
                ),
                "reference_positive_systems_recovered": complete_factor_value(
                    recovered_factor_support, "reference_positive_systems"
                ),
            },
        },
        "baseline_validation": {},
    }

    # Validate key frozen baselines exactly or to floating-point tolerance.
    frozen_metrics = pd.read_csv(root / "results" / "validation" / "syr4_spearman_correlations.csv")
    frozen_rho = float(frozen_metrics.loc[frozen_metrics["analysis"].eq("both_detected"), "spearman_rho"].iloc[0])
    frozen_agree = pd.read_csv(root / "results" / "validation" / "syr4_concordance_metrics.csv")
    frozen_agreement = float(
        frozen_agree.loc[frozen_agree["metric"].eq("threshold_status_agreement"), "estimate"].iloc[0]
    )
    same_rho = summary["syr4_pairing"]["same_date_rho"]
    same_agreement = summary["syr4_pairing"]["same_date_agreement"]
    frozen_curve = pd.read_csv(
        root / "results" / "validation" / "syr4_random_quarter_capture_curve.csv"
    )
    paired_curve = capture_curves[
        capture_curves["cohort"].eq("paired_same_date_sensitivity")
    ]
    curve_compare = frozen_curve.merge(
        paired_curve,
        on=["analyte", "threshold_ug_l", "n_available_quarters", "quarters_sampled_k"],
        suffixes=("_frozen", "_extended"),
        validate="one_to_one",
    )
    recomputed_compare = baseline_slim.merge(
        recomputed_baseline,
        on="PWSID",
        suffixes=("_frozen", "_recomputed"),
        validate="one_to_one",
    )
    ucmr_recompute_max_error = max(
        float(
            (
                recomputed_compare[f"{metric}_frozen"]
                - recomputed_compare[f"{metric}_recomputed"]
            )
            .abs()
            .max()
        )
        for metric in ("y00", "y10", "y01", "y11", "interaction")
    )
    summary["baseline_validation"] = {
        "same_date_pair_count_matches_frozen": bool(len(same_date) == 527_296),
        "same_date_rho_matches_frozen": bool(abs(same_rho - frozen_rho) <= 1e-12),
        "same_date_agreement_matches_frozen": bool(abs(same_agreement - frozen_agreement) <= 1e-12),
        "ucmr_complete_event_count_matches_frozen": bool(len(ucmr) == 63_432),
        "ucmr_event_recomputation_systems_match_frozen": bool(
            len(recomputed_baseline) == len(baseline_slim) == len(recomputed_compare)
        ),
        "ucmr_event_recomputation_max_cell_error_le_1e-12": bool(
            ucmr_recompute_max_error <= 1e-12
        ),
        "icr_row_count_matches_frozen": bool(len(icr) == 18_214),
        "paired_capture_curve_rows_match_frozen": bool(
            len(frozen_curve) == len(paired_curve) == len(curve_compare)
        ),
        "paired_capture_curve_points_match_frozen": bool(
            (
                curve_compare["mean_exact_capture_probability_frozen"]
                - curve_compare["mean_exact_capture_probability_extended"]
            )
            .abs()
            .max()
            <= 1e-12
        ),
    }
    if not all(summary["baseline_validation"].values()):
        raise AssertionError(f"Frozen baseline validation failed: {summary['baseline_validation']}")

    write_json(summary, out / "validation_extensions_summary.json")
    note_path = out / "METHODS_AND_RESULTS_NOTE.md"
    note_path.write_text(methods_note(summary), encoding="utf-8")

    generated = sorted(
        path for path in out.iterdir() if path.is_file() and path.name not in {"validation_extension_manifest.csv", "validation_extension_manifest.parquet"}
    )
    manifest = pd.DataFrame(
        [
            {
                "file": path.name,
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "format": path.suffix.lstrip("."),
            }
            for path in generated
        ]
    )
    save_table(manifest, out / "validation_extension_manifest")
    print(json.dumps(json_value(summary), ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
