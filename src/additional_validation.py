#!/usr/bin/env python3
"""Historical and state-data paired HAA5/TTHM replication in EPA SYR3 and California CLIP 2025.

The analysis keeps the two releases separate, pairs analytes on release-specific
exact keys, handles non-detects as concentration intervals, and uses whole-system
bootstrap resampling for primary uncertainty estimates. Threshold crossings are
descriptive sample-level benchmark signals, not compliance determinations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import scipy
from matplotlib.ticker import PercentFormatter
from scipy.stats import rankdata, spearmanr


DEFAULT_REPS = 10_000
DEFAULT_SEED = 20260913
HAA5_THRESHOLD = 60.0
TTHM_THRESHOLD = 80.0


@dataclass(frozen=True)
class Paths:
    root: Path
    syr3_haa: Path
    syr3_tthm: Path
    california: Path
    audit_dir: Path
    validation_dir: Path
    validation_pairs: Path
    out: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def make_paths(root: Path) -> Paths:
    root = root.resolve()
    syr3 = root / "data" / "raw" / "additional" / "syr3"
    return Paths(
        root=root,
        syr3_haa=syr3 / "syr3_haa" / "HAA5.txt",
        syr3_tthm=syr3 / "syr3_thm" / "TTHM.txt",
        california=root / "data" / "raw" / "additional" / "california" / "clip_sql_2025.csv",
        audit_dir=root / "data" / "interim" / "additional_audit",
        validation_dir=root / "results" / "validation",
        validation_pairs=root / "data" / "processed" / "validation" / "syr4_exact_pairs.parquet",
        out=root / "results" / "additional",
    )


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NA:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(obj, path: Path) -> None:
    path.write_text(json.dumps(json_value(obj), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_table(df: pd.DataFrame, out: Path, name: str) -> None:
    frame = df.copy()
    frame.to_csv(out / f"{name}.csv", index=False, lineterminator="\n", encoding="utf-8")
    frame.to_parquet(out / f"{name}.parquet", index=False)


def compact_number(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:,}"


def classify_intervals(frame: pd.DataFrame, prefix: str, threshold: float) -> None:
    detect = pd.to_numeric(frame[f"{prefix}_detect"], errors="coerce")
    value = pd.to_numeric(frame[f"{prefix}_value_ug_l"], errors="coerce")
    limit = pd.to_numeric(frame[f"{prefix}_limit_ug_l"], errors="coerce")
    observed = detect.eq(1) & value.ge(0) & np.isfinite(value)
    censored = detect.eq(0)
    finite_limit = censored & limit.gt(0) & np.isfinite(limit)

    lower = pd.Series(np.nan, index=frame.index, dtype=float)
    upper = pd.Series(np.nan, index=frame.index, dtype=float)
    midpoint = pd.Series(np.nan, index=frame.index, dtype=float)
    lower.loc[observed] = value.loc[observed]
    upper.loc[observed] = value.loc[observed]
    midpoint.loc[observed] = value.loc[observed]
    lower.loc[censored] = 0.0
    upper.loc[finite_limit] = limit.loc[finite_limit]
    midpoint.loc[finite_limit] = limit.loc[finite_limit] / 2.0

    status = pd.Series("indeterminate", index=frame.index, dtype="string")
    status.loc[lower.gt(threshold)] = "above"
    status.loc[upper.le(threshold)] = "below"
    frame[f"{prefix}_lower_ug_l"] = lower
    frame[f"{prefix}_upper_ug_l"] = upper
    frame[f"{prefix}_midpoint_ug_l"] = midpoint
    frame[f"{prefix}_threshold_status"] = status


def cluster_bootstrap_ratios(
    frame: pd.DataFrame,
    cluster_col: str,
    specifications: dict[str, tuple[np.ndarray, np.ndarray]],
    reps: int,
    seed: int,
) -> dict[str, dict]:
    """Bootstrap several ratios of totals with whole-system resampling."""
    work = pd.DataFrame({"cluster": frame[cluster_col].astype(str).to_numpy()})
    column_names: list[str] = []
    for metric, (numerator, denominator) in specifications.items():
        ncol, dcol = f"{metric}__num", f"{metric}__den"
        work[ncol] = np.asarray(numerator, dtype=float)
        work[dcol] = np.asarray(denominator, dtype=float)
        column_names.extend([ncol, dcol])
    grouped = work.groupby("cluster", sort=True)[column_names].sum()
    matrix = grouped.to_numpy(dtype=float)
    n_clusters = len(matrix)

    results: dict[str, dict] = {}
    for idx, metric in enumerate(specifications):
        numerator = float(matrix[:, 2 * idx].sum())
        denominator = float(matrix[:, 2 * idx + 1].sum())
        results[metric] = {
            "estimate": numerator / denominator if denominator > 0 else np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "numerator": numerator,
            "denominator": denominator,
            "n_systems": n_clusters,
            "bootstrap_reps": reps,
        }

    if reps <= 0 or n_clusters < 2:
        return results

    rng = np.random.default_rng(seed)
    draws = {metric: np.empty(reps, dtype=float) for metric in specifications}
    batch = max(1, min(96, 1_500_000 // n_clusters))
    cursor = 0
    while cursor < reps:
        size = min(batch, reps - cursor)
        indices = rng.integers(0, n_clusters, size=(size, n_clusters), dtype=np.int32)
        totals = matrix[indices].sum(axis=1)
        for idx, metric in enumerate(specifications):
            numerator = totals[:, 2 * idx]
            denominator = totals[:, 2 * idx + 1]
            draws[metric][cursor : cursor + size] = np.divide(
                numerator,
                denominator,
                out=np.full(size, np.nan),
                where=denominator > 0,
            )
        cursor += size

    for metric, values in draws.items():
        finite = values[np.isfinite(values)]
        if len(finite):
            low, high = np.quantile(finite, [0.025, 0.975])
            results[metric]["ci_low"] = float(low)
            results[metric]["ci_high"] = float(high)
    return results


def fixed_rank_cluster_bootstrap_spearman(
    x: Iterable[float],
    y: Iterable[float],
    clusters: Iterable[str],
    reps: int,
    seed: int,
) -> dict:
    """Spearman point estimate with a fixed-global-rank system bootstrap."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    clusters = np.asarray(clusters).astype(str)
    valid = np.isfinite(x) & np.isfinite(y) & pd.notna(clusters)
    x, y, clusters = x[valid], y[valid], clusters[valid]
    n_pairs = len(x)
    n_systems = len(np.unique(clusters))
    if n_pairs < 3:
        return {
            "spearman_rho": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_pairs": n_pairs,
            "n_systems": n_systems,
            "bootstrap_reps": reps,
        }

    rank_x = rankdata(x, method="average")
    rank_y = rankdata(y, method="average")
    point = float(np.corrcoef(rank_x, rank_y)[0, 1])
    work = pd.DataFrame(
        {
            "cluster": clusters,
            "n": 1.0,
            "sx": rank_x,
            "sy": rank_y,
            "sxx": rank_x * rank_x,
            "syy": rank_y * rank_y,
            "sxy": rank_x * rank_y,
        }
    )
    matrix = work.groupby("cluster", sort=True)[["n", "sx", "sy", "sxx", "syy", "sxy"]].sum().to_numpy()
    if reps <= 0 or n_systems < 2:
        return {
            "spearman_rho": point,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_pairs": n_pairs,
            "n_systems": n_systems,
            "bootstrap_reps": reps,
        }

    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=float)
    batch = max(1, min(96, 1_500_000 // n_systems))
    cursor = 0
    while cursor < reps:
        size = min(batch, reps - cursor)
        indices = rng.integers(0, n_systems, size=(size, n_systems), dtype=np.int32)
        totals = matrix[indices].sum(axis=1)
        n, sx, sy, sxx, syy, sxy = (totals[:, i] for i in range(6))
        covariance = sxy - sx * sy / n
        variance_x = sxx - sx * sx / n
        variance_y = syy - sy * sy / n
        draws[cursor : cursor + size] = np.divide(
            covariance,
            np.sqrt(variance_x * variance_y),
            out=np.full(size, np.nan),
            where=(variance_x > 0) & (variance_y > 0),
        )
        cursor += size
    finite = draws[np.isfinite(draws)]
    low, high = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
    return {
        "spearman_rho": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_pairs": n_pairs,
        "n_systems": n_systems,
        "bootstrap_reps": reps,
    }


def create_syr3_pairs(con: duckdb.DuckDBPyConnection, paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    h_path = sql_path(paths.syr3_haa)
    t_path = sql_path(paths.syr3_tthm)
    con.execute(
        f"CREATE VIEW syr3_h_raw AS SELECT * FROM read_csv_auto('{h_path}', delim='\\t', "
        "header=true, all_varchar=true, sample_size=100000)"
    )
    con.execute(
        f"CREATE VIEW syr3_t_raw AS SELECT * FROM read_csv_auto('{t_path}', delim='\\t', "
        "header=true, all_varchar=true, sample_size=100000)"
    )
    signature = (
        '"Sample Collection Date","Water Facility ID","Sampling Point ID","Sample Type Code",'
        '"Detect","Value","Detection Limit Value","Detection Limit Unit","Unit",'
        '"Source Water Type","System Type","Adjusted Total Population Served"'
    )
    for tag in ("h", "t"):
        con.execute(
            f'''CREATE TEMP TABLE syr3_{tag}_keys AS
                SELECT "PWSID" AS pwsid, "Sample ID" AS sample_id, COUNT(*) AS n_rows,
                       COUNT(DISTINCT hash({signature})) AS n_signatures
                FROM syr3_{tag}_raw
                WHERE "PWSID" IS NOT NULL AND trim("PWSID") <> ''
                  AND "Sample ID" IS NOT NULL AND trim("Sample ID") <> ''
                GROUP BY ALL'''
        )
        con.execute(
            f'''CREATE TEMP TABLE syr3_{tag}_clean AS
                SELECT r.*
                FROM syr3_{tag}_raw r
                JOIN syr3_{tag}_keys k
                  ON r."PWSID" = k.pwsid AND r."Sample ID" = k.sample_id
                WHERE k.n_signatures = 1
                QUALIFY row_number() OVER (
                    PARTITION BY r."PWSID", r."Sample ID"
                    ORDER BY try_cast(r."Six Year ID" AS BIGINT), r."Six Year ID"
                ) = 1'''
        )

    input_rows: list[dict] = []
    for tag, analyte in (("h", "HAA5"), ("t", "TTHM")):
        raw = con.execute(
            f'''SELECT COUNT(*), COUNT(DISTINCT "PWSID"),
                       COUNT(*) FILTER (
                         WHERE "PWSID" IS NULL OR trim("PWSID") = ''
                            OR "Sample ID" IS NULL OR trim("Sample ID") = ''),
                       COUNT(*) FILTER (WHERE try_cast("Detect" AS INTEGER) = 1),
                       COUNT(*) FILTER (WHERE try_cast("Detect" AS INTEGER) = 0),
                       COUNT(*) FILTER (WHERE try_cast("Detect" AS INTEGER) NOT IN (0,1)
                                                OR "Detect" IS NULL)
                FROM syr3_{tag}_raw'''
        ).fetchone()
        keys = con.execute(
            f'''SELECT COUNT(*),
                       COUNT(*) FILTER (WHERE n_rows > 1 AND n_signatures = 1),
                       COUNT(*) FILTER (WHERE n_signatures > 1),
                       coalesce(SUM(n_rows) FILTER (WHERE n_signatures > 1), 0),
                       max(n_rows)
                FROM syr3_{tag}_keys'''
        ).fetchone()
        input_rows.append(
            {
                "dataset": "SYR3",
                "analyte": analyte,
                "raw_rows": int(raw[0]),
                "raw_systems": int(raw[1]),
                "missing_exact_key_rows": int(raw[2]),
                "detected_rows": int(raw[3]),
                "nondetect_rows": int(raw[4]),
                "invalid_or_missing_detect_flag_rows": int(raw[5]),
                "nonmissing_exact_keys": int(keys[0]),
                "consistent_duplicate_keys_collapsed": int(keys[1]),
                "conflicting_duplicate_keys_excluded": int(keys[2]),
                "rows_in_conflicting_keys": int(keys[3]),
                "maximum_key_multiplicity": int(keys[4]),
            }
        )

    con.execute(
        '''CREATE TEMP TABLE syr3_pairs AS
           SELECT
             h."PWSID"::VARCHAR AS pwsid,
             h."Sample ID"::VARCHAR AS sample_id,
             try_cast(h."Sample Collection Date" AS TIMESTAMP)::DATE AS haa5_date,
             try_cast(t."Sample Collection Date" AS TIMESTAMP)::DATE AS tthm_date,
             try_cast(h."Sample Collection Date" AS TIMESTAMP)::DATE =
               try_cast(t."Sample Collection Date" AS TIMESTAMP)::DATE AS date_match,
             coalesce(h."Water Facility ID", '') = coalesce(t."Water Facility ID", '') AS facility_match,
             coalesce(h."Sampling Point ID", '') = coalesce(t."Sampling Point ID", '') AS sampling_point_match,
             coalesce(h."Sample Type Code", '') = coalesce(t."Sample Type Code", '') AS sample_type_match,
             coalesce(h."State Code", t."State Code")::VARCHAR AS state_code,
             coalesce(h."Source Water Type", t."Source Water Type")::VARCHAR AS source_water_code,
             coalesce(h."System Type", t."System Type")::VARCHAR AS system_type,
             coalesce(try_cast(h."Adjusted Total Population Served" AS DOUBLE),
                      try_cast(t."Adjusted Total Population Served" AS DOUBLE)) AS population,
             h."Water Facility ID"::VARCHAR AS water_facility_id,
             h."Sampling Point ID"::VARCHAR AS sampling_point_id,
             h."Sample Type Code"::VARCHAR AS haa5_sample_type,
             t."Sample Type Code"::VARCHAR AS tthm_sample_type,
             try_cast(h."Detect" AS INTEGER) AS haa5_detect,
             try_cast(h."Value" AS DOUBLE) AS haa5_value_reported,
             h."Unit"::VARCHAR AS haa5_value_unit,
             CASE
               WHEN upper(trim(coalesce(h."Unit", ''))) = 'UG/L' THEN try_cast(h."Value" AS DOUBLE)
               WHEN upper(trim(coalesce(h."Unit", ''))) = 'MG/L' THEN 1000 * try_cast(h."Value" AS DOUBLE)
               ELSE NULL END AS haa5_value_ug_l,
             try_cast(h."Detection Limit Value" AS DOUBLE) AS haa5_limit_reported,
             h."Detection Limit Unit"::VARCHAR AS haa5_limit_unit,
             CASE
               WHEN try_cast(h."Detection Limit Value" AS DOUBLE) <= 0 THEN NULL
               WHEN upper(trim(coalesce(h."Detection Limit Unit", ''))) = 'MG/L'
                 THEN 1000 * try_cast(h."Detection Limit Value" AS DOUBLE)
               WHEN upper(trim(coalesce(h."Detection Limit Unit", ''))) IN ('', 'UG/L')
                 THEN try_cast(h."Detection Limit Value" AS DOUBLE)
               ELSE NULL END AS haa5_limit_ug_l,
             try_cast(t."Detect" AS INTEGER) AS tthm_detect,
             try_cast(t."Value" AS DOUBLE) AS tthm_value_reported,
             t."Unit"::VARCHAR AS tthm_value_unit,
             CASE
               WHEN upper(trim(coalesce(t."Unit", ''))) = 'UG/L' THEN try_cast(t."Value" AS DOUBLE)
               WHEN upper(trim(coalesce(t."Unit", ''))) = 'MG/L' THEN 1000 * try_cast(t."Value" AS DOUBLE)
               ELSE NULL END AS tthm_value_ug_l,
             try_cast(t."Detection Limit Value" AS DOUBLE) AS tthm_limit_reported,
             t."Detection Limit Unit"::VARCHAR AS tthm_limit_unit,
             CASE
               WHEN try_cast(t."Detection Limit Value" AS DOUBLE) <= 0 THEN NULL
               WHEN upper(trim(coalesce(t."Detection Limit Unit", ''))) = 'MG/L'
                 THEN 1000 * try_cast(t."Detection Limit Value" AS DOUBLE)
               WHEN upper(trim(coalesce(t."Detection Limit Unit", ''))) IN ('', 'UG/L')
                 THEN try_cast(t."Detection Limit Value" AS DOUBLE)
               ELSE NULL END AS tthm_limit_ug_l
           FROM syr3_h_clean h
           INNER JOIN syr3_t_clean t
             ON h."PWSID" = t."PWSID" AND h."Sample ID" = t."Sample ID"'''
    )
    pairs = con.execute("SELECT * FROM syr3_pairs ORDER BY pwsid, haa5_date, sample_id").fetchdf()
    pairs["sample_date"] = pd.to_datetime(pairs["haa5_date"], errors="coerce")
    pairs["year"] = pairs["sample_date"].dt.year.astype("Int16")
    pairs["month"] = pairs["sample_date"].dt.month.astype("Int8")
    classify_intervals(pairs, "haa5", HAA5_THRESHOLD)
    classify_intervals(pairs, "tthm", TTHM_THRESHOLD)

    audit = pd.DataFrame(
        [
            {"dataset": "SYR3", "metric": "conflict_resolved_exact_key_pairs", "count": len(pairs)},
            {"dataset": "SYR3", "metric": "same_date_primary_pairs", "count": int(pairs["date_match"].fillna(False).sum())},
            {"dataset": "SYR3", "metric": "date_mismatch_excluded", "count": int((~pairs["date_match"].fillna(False)).sum())},
            {"dataset": "SYR3", "metric": "facility_mismatch", "count": int((~pairs["facility_match"].fillna(False)).sum())},
            {"dataset": "SYR3", "metric": "sampling_point_mismatch", "count": int((~pairs["sampling_point_match"].fillna(False)).sum())},
            {"dataset": "SYR3", "metric": "sample_type_mismatch", "count": int((~pairs["sample_type_match"].fillna(False)).sum())},
            {"dataset": "SYR3", "metric": "primary_systems", "count": int(pairs.loc[pairs["date_match"].fillna(False), "pwsid"].nunique())},
        ]
    )

    field_qa = con.execute(
        '''SELECT 'SYR3' AS dataset,
                  CASE WHEN src = 'h' THEN 'HAA5' ELSE 'TTHM' END AS analyte,
                  audit_field, category, n
           FROM (
             SELECT 'h' AS src, 'detect_flag' AS audit_field,
                    coalesce("Detect", '<blank>') AS category, count(*) AS n
             FROM syr3_h_raw GROUP BY ALL
             UNION ALL
             SELECT 't', 'detect_flag', coalesce("Detect", '<blank>'), count(*)
             FROM syr3_t_raw GROUP BY ALL
             UNION ALL
             SELECT 'h', 'detected_result_unit', upper(coalesce("Unit", '<blank>')), count(*)
             FROM syr3_h_raw WHERE try_cast("Detect" AS INTEGER) = 1 GROUP BY ALL
             UNION ALL
             SELECT 't', 'detected_result_unit', upper(coalesce("Unit", '<blank>')), count(*)
             FROM syr3_t_raw WHERE try_cast("Detect" AS INTEGER) = 1 GROUP BY ALL
             UNION ALL
             SELECT 'h', 'nondetect_limit_unit', upper(coalesce("Detection Limit Unit", '<blank>')), count(*)
             FROM syr3_h_raw WHERE try_cast("Detect" AS INTEGER) = 0 GROUP BY ALL
             UNION ALL
             SELECT 't', 'nondetect_limit_unit', upper(coalesce("Detection Limit Unit", '<blank>')), count(*)
             FROM syr3_t_raw WHERE try_cast("Detect" AS INTEGER) = 0 GROUP BY ALL
             UNION ALL
             SELECT 'h', 'sample_type', coalesce("Sample Type Code", '<blank>'), count(*)
             FROM syr3_h_raw GROUP BY ALL
             UNION ALL
             SELECT 't', 'sample_type', coalesce("Sample Type Code", '<blank>'), count(*)
             FROM syr3_t_raw GROUP BY ALL
           ) q
           ORDER BY analyte, audit_field, n DESC'''
    ).fetchdf()
    return pairs, pd.DataFrame(input_rows), pd.concat([audit, field_qa.rename(columns={"n": "count"})], ignore_index=True, sort=False)


def create_california_pairs(con: duckdb.DuckDBPyConnection, paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ca_path = sql_path(paths.california)
    con.execute(
        f'''CREATE VIEW ca_raw AS
            SELECT * FROM read_csv_auto('{ca_path}', header=true, all_varchar=true,
                                        sample_size=200000)
            WHERE "Analyte Code" IN ('2456', '2950')'''
    )
    key_fields = ('"Water System Number"', '"PS Code"', '"Sample Date"', '"Lab Sample ID"')
    signature = (
        '"Result","Reporting Level","Less Than Reporting Level","Units of Measure",'
        '"Sample Type","Facility Type","Facility Water Type","MCL","Method"'
    )
    for tag, code in (("h", "2456"), ("t", "2950")):
        con.execute(
            f'''CREATE TEMP TABLE ca_{tag}_keys AS
                SELECT {key_fields[0]} AS pwsid, {key_fields[1]} AS ps_code,
                       {key_fields[2]} AS sample_date, {key_fields[3]} AS lab_sample_id,
                       COUNT(*) AS n_rows, COUNT(DISTINCT hash({signature})) AS n_signatures
                FROM ca_raw
                WHERE "Analyte Code" = '{code}'
                  AND {key_fields[0]} IS NOT NULL AND trim({key_fields[0]}) <> ''
                  AND {key_fields[1]} IS NOT NULL AND trim({key_fields[1]}) <> ''
                  AND {key_fields[2]} IS NOT NULL AND trim({key_fields[2]}) <> ''
                  AND {key_fields[3]} IS NOT NULL AND trim({key_fields[3]}) <> ''
                GROUP BY ALL'''
        )
        con.execute(
            f'''CREATE TEMP TABLE ca_{tag}_clean AS
                SELECT r.*
                FROM ca_raw r
                JOIN ca_{tag}_keys k
                  ON r."Water System Number" = k.pwsid
                 AND r."PS Code" = k.ps_code
                 AND r."Sample Date" = k.sample_date
                 AND r."Lab Sample ID" = k.lab_sample_id
                WHERE r."Analyte Code" = '{code}' AND k.n_signatures = 1
                QUALIFY row_number() OVER (
                  PARTITION BY r."Water System Number", r."PS Code", r."Sample Date", r."Lab Sample ID"
                  ORDER BY try_cast(r."Record ID#" AS BIGINT), r."Record ID#"
                ) = 1'''
        )

    input_rows: list[dict] = []
    for tag, code, analyte in (("h", "2456", "HAA5"), ("t", "2950", "TTHM")):
        raw = con.execute(
            f'''SELECT COUNT(*), COUNT(DISTINCT "Water System Number"),
                       COUNT(*) FILTER (
                         WHERE "Water System Number" IS NULL OR trim("Water System Number") = ''
                            OR "PS Code" IS NULL OR trim("PS Code") = ''
                            OR "Sample Date" IS NULL OR trim("Sample Date") = ''
                            OR "Lab Sample ID" IS NULL OR trim("Lab Sample ID") = ''),
                       COUNT(*) FILTER (WHERE upper(trim("Less Than Reporting Level")) = 'N'),
                       COUNT(*) FILTER (WHERE upper(trim("Less Than Reporting Level")) = 'Y'),
                       COUNT(*) FILTER (WHERE upper(trim("Less Than Reporting Level")) NOT IN ('N','Y')
                                                OR "Less Than Reporting Level" IS NULL)
                FROM ca_raw WHERE "Analyte Code" = '{code}' '''
        ).fetchone()
        keys = con.execute(
            f'''SELECT COUNT(*),
                       COUNT(*) FILTER (WHERE n_rows > 1 AND n_signatures = 1),
                       COUNT(*) FILTER (WHERE n_signatures > 1),
                       coalesce(SUM(n_rows) FILTER (WHERE n_signatures > 1), 0),
                       max(n_rows)
                FROM ca_{tag}_keys'''
        ).fetchone()
        input_rows.append(
            {
                "dataset": "California CLIP 2025",
                "analyte": analyte,
                "raw_rows": int(raw[0]),
                "raw_systems": int(raw[1]),
                "missing_exact_key_rows": int(raw[2]),
                "detected_rows": int(raw[3]),
                "nondetect_rows": int(raw[4]),
                "invalid_or_missing_detect_flag_rows": int(raw[5]),
                "nonmissing_exact_keys": int(keys[0]),
                "consistent_duplicate_keys_collapsed": int(keys[1]),
                "conflicting_duplicate_keys_excluded": int(keys[2]),
                "rows_in_conflicting_keys": int(keys[3]),
                "maximum_key_multiplicity": int(keys[4]),
            }
        )

    con.execute(
        '''CREATE TEMP TABLE ca_pairs AS
           SELECT
             h."Water System Number"::VARCHAR AS pwsid,
             h."PS Code"::VARCHAR AS sampling_point_id,
             h."Lab Sample ID"::VARCHAR AS sample_id,
             try_strptime(h."Sample Date", '%m/%d/%Y')::DATE AS sample_date,
             coalesce(h."Sample Type", '') = coalesce(t."Sample Type", '') AS sample_type_match,
             coalesce(h."Facility Type", '') = coalesce(t."Facility Type", '') AS facility_type_match,
             coalesce(h."Facility Water Type", '') = coalesce(t."Facility Water Type", '') AS facility_water_type_match,
             coalesce(h."Units of Measure", '') = coalesce(t."Units of Measure", '') AS unit_match,
             h."Sample Type"::VARCHAR AS haa5_sample_type,
             t."Sample Type"::VARCHAR AS tthm_sample_type,
             h."Facility Type"::VARCHAR AS haa5_facility_type,
             t."Facility Type"::VARCHAR AS tthm_facility_type,
             h."Facility Water Type"::VARCHAR AS facility_water_type,
             h."Principal County Served"::VARCHAR AS county,
             try_cast(h."Population Served" AS DOUBLE) AS population,
             CASE upper(trim(h."Less Than Reporting Level")) WHEN 'N' THEN 1 WHEN 'Y' THEN 0 ELSE NULL END AS haa5_detect,
             try_cast(h."Result" AS DOUBLE) AS haa5_value_reported,
             h."Units of Measure"::VARCHAR AS haa5_value_unit,
             CASE
               WHEN upper(trim(coalesce(h."Units of Measure", ''))) = 'UG/L' THEN try_cast(h."Result" AS DOUBLE)
               WHEN upper(trim(coalesce(h."Units of Measure", ''))) = 'MG/L' THEN 1000 * try_cast(h."Result" AS DOUBLE)
               ELSE NULL END AS haa5_value_ug_l,
             try_cast(h."Reporting Level" AS DOUBLE) AS haa5_limit_reported,
             h."Units of Measure"::VARCHAR AS haa5_limit_unit,
             CASE
               WHEN try_cast(h."Reporting Level" AS DOUBLE) <= 0 THEN NULL
               WHEN upper(trim(coalesce(h."Units of Measure", ''))) = 'UG/L' THEN try_cast(h."Reporting Level" AS DOUBLE)
               WHEN upper(trim(coalesce(h."Units of Measure", ''))) = 'MG/L' THEN 1000 * try_cast(h."Reporting Level" AS DOUBLE)
               ELSE NULL END AS haa5_limit_ug_l,
             CASE upper(trim(t."Less Than Reporting Level")) WHEN 'N' THEN 1 WHEN 'Y' THEN 0 ELSE NULL END AS tthm_detect,
             try_cast(t."Result" AS DOUBLE) AS tthm_value_reported,
             t."Units of Measure"::VARCHAR AS tthm_value_unit,
             CASE
               WHEN upper(trim(coalesce(t."Units of Measure", ''))) = 'UG/L' THEN try_cast(t."Result" AS DOUBLE)
               WHEN upper(trim(coalesce(t."Units of Measure", ''))) = 'MG/L' THEN 1000 * try_cast(t."Result" AS DOUBLE)
               ELSE NULL END AS tthm_value_ug_l,
             try_cast(t."Reporting Level" AS DOUBLE) AS tthm_limit_reported,
             t."Units of Measure"::VARCHAR AS tthm_limit_unit,
             CASE
               WHEN try_cast(t."Reporting Level" AS DOUBLE) <= 0 THEN NULL
               WHEN upper(trim(coalesce(t."Units of Measure", ''))) = 'UG/L' THEN try_cast(t."Reporting Level" AS DOUBLE)
               WHEN upper(trim(coalesce(t."Units of Measure", ''))) = 'MG/L' THEN 1000 * try_cast(t."Reporting Level" AS DOUBLE)
               ELSE NULL END AS tthm_limit_ug_l
           FROM ca_h_clean h
           INNER JOIN ca_t_clean t
             ON h."Water System Number" = t."Water System Number"
            AND h."PS Code" = t."PS Code"
            AND h."Sample Date" = t."Sample Date"
            AND h."Lab Sample ID" = t."Lab Sample ID"'''
    )
    pairs = con.execute("SELECT * FROM ca_pairs ORDER BY pwsid, sample_date, sample_id").fetchdf()
    pairs["sample_date"] = pd.to_datetime(pairs["sample_date"], errors="coerce")
    pairs["year"] = pairs["sample_date"].dt.year.astype("Int16")
    pairs["month"] = pairs["sample_date"].dt.month.astype("Int8")
    classify_intervals(pairs, "haa5", HAA5_THRESHOLD)
    classify_intervals(pairs, "tthm", TTHM_THRESHOLD)

    primary = (
        pairs["haa5_sample_type"].eq("RT")
        & pairs["tthm_sample_type"].eq("RT")
        & pairs["haa5_facility_type"].eq("DS")
        & pairs["tthm_facility_type"].eq("DS")
    )
    audit = pd.DataFrame(
        [
            {"dataset": "California CLIP 2025", "metric": "conflict_resolved_exact_event_pairs", "count": len(pairs)},
            {"dataset": "California CLIP 2025", "metric": "primary_routine_distribution_pairs", "count": int(primary.sum())},
            {"dataset": "California CLIP 2025", "metric": "nonprimary_exact_pairs", "count": int((~primary).sum())},
            {"dataset": "California CLIP 2025", "metric": "sample_type_mismatch", "count": int((~pairs["sample_type_match"].fillna(False)).sum())},
            {"dataset": "California CLIP 2025", "metric": "facility_type_mismatch", "count": int((~pairs["facility_type_match"].fillna(False)).sum())},
            {"dataset": "California CLIP 2025", "metric": "facility_water_type_mismatch", "count": int((~pairs["facility_water_type_match"].fillna(False)).sum())},
            {"dataset": "California CLIP 2025", "metric": "unit_mismatch", "count": int((~pairs["unit_match"].fillna(False)).sum())},
            {"dataset": "California CLIP 2025", "metric": "primary_systems", "count": int(pairs.loc[primary, "pwsid"].nunique())},
        ]
    )
    field_qa = con.execute(
        '''SELECT 'California CLIP 2025' AS dataset,
                  CASE WHEN "Analyte Code" = '2456' THEN 'HAA5' ELSE 'TTHM' END AS analyte,
                  audit_field, category, n
           FROM (
             SELECT "Analyte Code", 'censor_flag' AS audit_field,
                    coalesce("Less Than Reporting Level", '<blank>') AS category, count(*) AS n
             FROM ca_raw GROUP BY ALL
             UNION ALL
             SELECT "Analyte Code", 'result_unit', upper(coalesce("Units of Measure", '<blank>')), count(*)
             FROM ca_raw GROUP BY ALL
             UNION ALL
             SELECT "Analyte Code", 'sample_type', coalesce("Sample Type", '<blank>'), count(*)
             FROM ca_raw GROUP BY ALL
             UNION ALL
             SELECT "Analyte Code", 'facility_type', coalesce("Facility Type", '<blank>'), count(*)
             FROM ca_raw GROUP BY ALL
             UNION ALL
             SELECT "Analyte Code", 'nondetect_reporting_level_support',
                    CASE
                      WHEN try_cast("Reporting Level" AS DOUBLE) > 0 THEN 'positive_recognized_limit'
                      WHEN try_cast("Reporting Level" AS DOUBLE) = 0 THEN 'zero_limit'
                      ELSE 'missing_or_invalid_limit' END,
                    count(*)
             FROM ca_raw WHERE upper(trim("Less Than Reporting Level")) = 'Y' GROUP BY ALL
           ) q
           ORDER BY analyte, audit_field, n DESC'''
    ).fetchdf()
    return pairs, pd.DataFrame(input_rows), pd.concat([audit, field_qa.rename(columns={"n": "count"})], ignore_index=True, sort=False)


def concentration_summary(frame: pd.DataFrame, cohort_id: str, label: str, role: str) -> pd.DataFrame:
    rows: list[dict] = []
    for analyte, prefix, threshold in (("HAA5", "haa5", HAA5_THRESHOLD), ("TTHM", "tthm", TTHM_THRESHOLD)):
        series_map = {
            "detected_only": frame[f"{prefix}_value_ug_l"].where(frame[f"{prefix}_detect"].eq(1)),
            "lower_bound": frame[f"{prefix}_lower_ug_l"],
            "midpoint_limit_over_2": frame[f"{prefix}_midpoint_ug_l"],
            "upper_bound": frame[f"{prefix}_upper_ug_l"],
        }
        for method, values in series_map.items():
            valid = values.notna() & np.isfinite(values)
            x = values.loc[valid].astype(float)
            rows.append(
                {
                    "cohort_id": cohort_id,
                    "cohort_label": label,
                    "scope_role": role,
                    "analyte": analyte,
                    "censoring_summary": method,
                    "threshold_ug_l": threshold,
                    "n_pairs": int(valid.sum()),
                    "n_systems": int(frame.loc[valid, "pwsid"].nunique()),
                    "mean_ug_l": x.mean(),
                    "median_ug_l": x.median(),
                    "q25_ug_l": x.quantile(0.25),
                    "q75_ug_l": x.quantile(0.75),
                    "p95_ug_l": x.quantile(0.95),
                    "median_scaled_to_threshold": x.median() / threshold,
                    "p95_scaled_to_threshold": x.quantile(0.95) / threshold,
                }
            )
    return pd.DataFrame(rows)


def analyze_scope(
    frame: pd.DataFrame,
    cohort_id: str,
    label: str,
    role: str,
    rule: str,
    reps: int,
    seed: int,
) -> dict[str, object]:
    frame = frame.copy()
    n_pairs = len(frame)
    n_systems = int(frame["pwsid"].nunique())
    h_class = frame["haa5_threshold_status"].isin(["above", "below"])
    t_class = frame["tthm_threshold_status"].isin(["above", "below"])
    joint = h_class & t_class
    known_detect = frame["haa5_detect"].isin([0, 1]) & frame["tthm_detect"].isin([0, 1])
    detection_agree = known_detect & frame["haa5_detect"].eq(frame["tthm_detect"])
    threshold_agree = joint & frame["haa5_threshold_status"].eq(frame["tthm_threshold_status"])

    ratio_specs = {
        "detection_status_agreement": (detection_agree.to_numpy(float), known_detect.to_numpy(float)),
        "threshold_status_agreement": (threshold_agree.to_numpy(float), joint.to_numpy(float)),
        "haa5_above_rate_classifiable": (frame["haa5_threshold_status"].eq("above").to_numpy(float), h_class.to_numpy(float)),
        "tthm_above_rate_classifiable": (frame["tthm_threshold_status"].eq("above").to_numpy(float), t_class.to_numpy(float)),
    }
    ratios = cluster_bootstrap_ratios(frame, "pwsid", ratio_specs, reps, seed)

    metric_definitions = {
        "detection_status_agreement": "same detected or non-detect flag among pairs with both flags known",
        "threshold_status_agreement": "same above or below status among jointly classifiable interval pairs",
        "haa5_above_rate_classifiable": "HAA5 concentration interval definitely above 60 ug/L among classifiable HAA5 pairs",
        "tthm_above_rate_classifiable": "TTHM concentration interval definitely above 80 ug/L among classifiable TTHM pairs",
    }
    metric_rows: list[dict] = []
    for metric, values in ratios.items():
        metric_rows.append(
            {
                "cohort_id": cohort_id,
                "cohort_label": label,
                "scope_role": role,
                "metric": metric,
                **values,
                "unit": "proportion",
                "definition": metric_definitions[metric],
            }
        )
    if joint.sum():
        observed_agreement = float(threshold_agree.sum() / joint.sum())
        haa5_above = float(frame.loc[joint, "haa5_threshold_status"].eq("above").mean())
        tthm_above = float(frame.loc[joint, "tthm_threshold_status"].eq("above").mean())
        expected_agreement = haa5_above * tthm_above + (1 - haa5_above) * (1 - tthm_above)
        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement) if expected_agreement < 1 else np.nan
    else:
        kappa = np.nan
    metric_rows.append(
        {
            "cohort_id": cohort_id,
            "cohort_label": label,
            "scope_role": role,
            "metric": "cohen_kappa_descriptive",
            "estimate": kappa,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "numerator": np.nan,
            "denominator": int(joint.sum()),
            "n_systems": int(frame.loc[joint, "pwsid"].nunique()),
            "bootstrap_reps": 0,
            "unit": "coefficient",
            "definition": "chance-corrected agreement of HAA5 and TTHM threshold status among jointly classifiable pairs",
        }
    )

    correlation_rows: list[dict] = []
    detected = frame["haa5_detect"].eq(1) & frame["tthm_detect"].eq(1)
    detected &= frame["haa5_value_ug_l"].notna() & frame["tthm_value_ug_l"].notna()
    corr = fixed_rank_cluster_bootstrap_spearman(
        frame.loc[detected, "haa5_value_ug_l"],
        frame.loc[detected, "tthm_value_ug_l"],
        frame.loc[detected, "pwsid"],
        reps,
        seed + 101,
    )
    correlation_rows.append(
        {
            "cohort_id": cohort_id,
            "cohort_label": label,
            "scope_role": role,
            "analysis": "both_detected",
            **corr,
            "bootstrap_method": "whole-system resampling of fixed global midranks",
        }
    )

    midpoint = frame["haa5_midpoint_ug_l"].notna() & frame["tthm_midpoint_ug_l"].notna()
    midpoint_corr = fixed_rank_cluster_bootstrap_spearman(
        frame.loc[midpoint, "haa5_midpoint_ug_l"],
        frame.loc[midpoint, "tthm_midpoint_ug_l"],
        frame.loc[midpoint, "pwsid"],
        0,
        seed + 102,
    )
    correlation_rows.append(
        {
            "cohort_id": cohort_id,
            "cohort_label": label,
            "scope_role": role,
            "analysis": "finite_interval_midpoints_descriptive",
            **midpoint_corr,
            "bootstrap_method": "point estimate only; interval-midpoint sensitivity",
        }
    )

    system_medians = (
        frame.loc[detected, ["pwsid", "haa5_value_ug_l", "tthm_value_ug_l"]]
        .groupby("pwsid", as_index=False)
        .median(numeric_only=True)
    )
    median_corr = fixed_rank_cluster_bootstrap_spearman(
        system_medians["haa5_value_ug_l"],
        system_medians["tthm_value_ug_l"],
        system_medians["pwsid"],
        reps,
        seed + 103,
    )
    correlation_rows.append(
        {
            "cohort_id": cohort_id,
            "cohort_label": label,
            "scope_role": role,
            "analysis": "system_median_both_detected",
            **median_corr,
            "bootstrap_method": "whole-system resampling of system medians with fixed global midranks",
        }
    )

    cell_rows: list[dict] = []
    for h_status in ("below", "above"):
        for t_status in ("below", "above"):
            selected = joint & frame["haa5_threshold_status"].eq(h_status) & frame["tthm_threshold_status"].eq(t_status)
            cell_rows.append(
                {
                    "cohort_id": cohort_id,
                    "cohort_label": label,
                    "scope_role": role,
                    "haa5_status": h_status,
                    "tthm_status": t_status,
                    "n_pairs": int(selected.sum()),
                    "n_systems": int(frame.loc[selected, "pwsid"].nunique()),
                    "fraction_jointly_classifiable": float(selected.sum() / joint.sum()) if joint.sum() else np.nan,
                }
            )

    summary = {
        "cohort_id": cohort_id,
        "cohort_label": label,
        "scope_role": role,
        "cohort_rule": rule,
        "n_pairs": n_pairs,
        "n_systems": n_systems,
        "date_min": frame["sample_date"].min(),
        "date_max": frame["sample_date"].max(),
        "both_detected_pairs": int(detected.sum()),
        "haa5_classifiable_pairs": int(h_class.sum()),
        "tthm_classifiable_pairs": int(t_class.sum()),
        "jointly_classifiable_pairs": int(joint.sum()),
        "metrics": {row["metric"]: row for row in metric_rows},
        "correlations": {row["analysis"]: row for row in correlation_rows},
    }
    return {
        "summary": summary,
        "metrics": pd.DataFrame(metric_rows),
        "correlations": pd.DataFrame(correlation_rows),
        "concordance": pd.DataFrame(cell_rows),
        "concentration": concentration_summary(frame, cohort_id, label, role),
    }


def stratified_summary(frame: pd.DataFrame, cohort_id: str, dimension: str) -> pd.DataFrame:
    rows: list[dict] = []
    for level, group in frame.groupby(dimension, dropna=False, sort=True):
        detected = group["haa5_detect"].eq(1) & group["tthm_detect"].eq(1)
        detected &= group["haa5_value_ug_l"].notna() & group["tthm_value_ug_l"].notna()
        h_class = group["haa5_threshold_status"].isin(["above", "below"])
        t_class = group["tthm_threshold_status"].isin(["above", "below"])
        joint = h_class & t_class
        rho = (
            float(spearmanr(group.loc[detected, "haa5_value_ug_l"], group.loc[detected, "tthm_value_ug_l"]).statistic)
            if detected.sum() >= 3
            else np.nan
        )
        rows.append(
            {
                "cohort_id": cohort_id,
                "dimension": dimension,
                "level": str(level),
                "n_pairs": len(group),
                "n_systems": group["pwsid"].nunique(),
                "both_detected_pairs": int(detected.sum()),
                "spearman_rho_both_detected": rho,
                "haa5_classifiable_pairs": int(h_class.sum()),
                "haa5_above_rate_classifiable": group.loc[h_class, "haa5_threshold_status"].eq("above").mean(),
                "tthm_classifiable_pairs": int(t_class.sum()),
                "tthm_above_rate_classifiable": group.loc[t_class, "tthm_threshold_status"].eq("above").mean(),
                "jointly_classifiable_pairs": int(joint.sum()),
                "threshold_status_agreement": group.loc[joint, "haa5_threshold_status"].eq(
                    group.loc[joint, "tthm_threshold_status"]
                ).mean(),
            }
        )
    return pd.DataFrame(rows)


def load_syr4_reference(paths: Paths, reps: int, seed: int) -> dict[str, object]:
    summary_path = paths.validation_dir / "validation_summary.json"
    correlation_path = paths.validation_dir / "syr4_spearman_correlations.csv"
    concordance_path = paths.validation_dir / "syr4_concordance_metrics.csv"
    required = [summary_path, correlation_path, concordance_path, paths.validation_pairs]
    if any(not path.exists() for path in required):
        return {"available": False, "missing": [str(path) for path in required if not path.exists()]}

    summary = json.loads(summary_path.read_text(encoding="utf-8"))["syr4"]
    correlations = pd.read_csv(correlation_path)
    concordance_metrics = pd.read_csv(concordance_path)
    pairs = pd.read_parquet(
        paths.validation_pairs,
        columns=["pwsid", "analysis_included", "haa5_threshold_status", "tthm_threshold_status"],
    )
    pairs = pairs[pairs["analysis_included"].fillna(False)].copy()
    h_class = pairs["haa5_threshold_status"].isin(["above", "below"])
    t_class = pairs["tthm_threshold_status"].isin(["above", "below"])
    joint = h_class & t_class
    agree = joint & pairs["haa5_threshold_status"].eq(pairs["tthm_threshold_status"])
    ratios = cluster_bootstrap_ratios(
        pairs,
        "pwsid",
        {
            "threshold_status_agreement": (agree.to_numpy(float), joint.to_numpy(float)),
            "haa5_above_rate_classifiable": (pairs["haa5_threshold_status"].eq("above").to_numpy(float), h_class.to_numpy(float)),
            "tthm_above_rate_classifiable": (pairs["tthm_threshold_status"].eq("above").to_numpy(float), t_class.to_numpy(float)),
        },
        reps,
        seed,
    )
    cells: list[dict] = []
    for hs in ("below", "above"):
        for ts in ("below", "above"):
            mask = joint & pairs["haa5_threshold_status"].eq(hs) & pairs["tthm_threshold_status"].eq(ts)
            cells.append(
                {
                    "cohort_id": "SYR4_reference",
                    "cohort_label": "SYR4 2012 to 2019",
                    "scope_role": "reference",
                    "haa5_status": hs,
                    "tthm_status": ts,
                    "n_pairs": int(mask.sum()),
                    "n_systems": int(pairs.loc[mask, "pwsid"].nunique()),
                    "fraction_jointly_classifiable": float(mask.sum() / joint.sum()),
                }
            )
    both_detected = correlations.loc[correlations["analysis"].eq("both_detected")].iloc[0]
    agreement_row = concordance_metrics.loc[
        concordance_metrics["metric"].eq("threshold_status_agreement")
    ].iloc[0]
    return {
        "available": True,
        "summary": {
            "cohort_id": "SYR4_reference",
            "cohort_label": "SYR4 2012 to 2019",
            "scope_role": "reference",
            "cohort_rule": "existing same-date safe PWSID plus SAMPLE_ID primary validation cohort",
            "n_pairs": int(summary["same_date_primary_pairs"]),
            "n_systems": int(summary["primary_systems"]),
            "date_min": "2012-01-01",
            "date_max": "2019-12-31",
            "both_detected_pairs": int(summary["both_detected_pairs"]),
            "jointly_classifiable_pairs": int(summary["jointly_classifiable_pairs"]),
            "metrics": ratios,
            "correlations": {
                "both_detected": {
                    "spearman_rho": float(both_detected["spearman_rho"]),
                    "ci_low": float(both_detected["cluster_bootstrap_ci_low"]),
                    "ci_high": float(both_detected["cluster_bootstrap_ci_high"]),
                    "n_pairs": int(both_detected["n_pairs"]),
                    "n_systems": int(both_detected["n_systems"]),
                    "bootstrap_reps": int(both_detected["bootstrap_reps"]),
                }
            },
            "existing_agreement_check": {
                "estimate": float(agreement_row["estimate"]),
                "ci_low": float(agreement_row["ci_low"]),
                "ci_high": float(agreement_row["ci_high"]),
                "matches_recomputed": abs(float(agreement_row["estimate"]) - ratios["threshold_status_agreement"]["estimate"]) < 1e-12,
            },
        },
        "concordance": pd.DataFrame(cells),
    }


def component_qa(paths: Paths) -> pd.DataFrame:
    source = paths.audit_dir / "california_component_reconciliation.csv"
    if not source.exists():
        return pd.DataFrame()
    frame = pd.read_csv(source)
    frame.insert(0, "dataset", "California CLIP 2025")
    frame["complete_panel_fraction"] = frame["complete_component_panels"] / frame["reported_group_events"]
    frame["within_1_ug_l_fraction"] = frame["within_1_0_ug_L"] / frame["numeric_reported_group_with_complete_panel"]
    frame["interpretation"] = "reported group result remains primary; component sums are QA only"
    return frame


def qa_checks(
    syr3_pairs: pd.DataFrame,
    ca_pairs: pd.DataFrame,
    input_audit: pd.DataFrame,
    component: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict] = []

    def add(dataset: str, check: str, observed, expected: str, passed: bool, note: str) -> None:
        checks.append(
            {
                "dataset": dataset,
                "check": check,
                "observed": observed,
                "expected": expected,
                "status": "PASS" if passed else "REVIEW",
                "note": note,
            }
        )

    add("SYR3", "paired collection dates agree", int((~syr3_pairs["date_match"].fillna(False)).sum()), "0 mismatches", bool(syr3_pairs["date_match"].fillna(False).all()), "primary cohort requires matching dates")
    add("SYR3", "paired facilities agree", int((~syr3_pairs["facility_match"].fillna(False)).sum()), "0 mismatches", bool(syr3_pairs["facility_match"].fillna(False).all()), "exact sample IDs map to the same facility")
    add("SYR3", "paired sampling points agree", int((~syr3_pairs["sampling_point_match"].fillna(False)).sum()), "0 mismatches", bool(syr3_pairs["sampling_point_match"].fillna(False).all()), "exact sample IDs map to the same sampling point")
    add("SYR3", "paired sample types agree", int((~syr3_pairs["sample_type_match"].fillna(False)).sum()), "0 mismatches", bool(syr3_pairs["sample_type_match"].fillna(False).all()), "sample purpose is preserved")
    add("California CLIP 2025", "paired sample types agree", int((~ca_pairs["sample_type_match"].fillna(False)).sum()), "0 mismatches", bool(ca_pairs["sample_type_match"].fillna(False).all()), "strict state event key")
    add("California CLIP 2025", "paired facility types agree", int((~ca_pairs["facility_type_match"].fillna(False)).sum()), "0 mismatches", bool(ca_pairs["facility_type_match"].fillna(False).all()), "strict state event key")
    add("California CLIP 2025", "paired units agree", int((~ca_pairs["unit_match"].fillna(False)).sum()), "0 mismatches", bool(ca_pairs["unit_match"].fillna(False).all()), "all paired group results are normalized before analysis")

    for dataset, frame in (("SYR3", syr3_pairs), ("California CLIP 2025", ca_pairs)):
        for analyte, prefix in (("HAA5", "haa5"), ("TTHM", "tthm")):
            detected = frame[f"{prefix}_detect"].eq(1)
            missing_value = detected & (~np.isfinite(pd.to_numeric(frame[f"{prefix}_value_ug_l"], errors="coerce")))
            add(dataset, f"{analyte} detected rows have normalized numeric values", int(missing_value.sum()), "0 missing", not missing_value.any(), "detected concentrations without recognized units are not used")
            nondetect = frame[f"{prefix}_detect"].eq(0)
            supported = nondetect & pd.to_numeric(frame[f"{prefix}_limit_ug_l"], errors="coerce").gt(0)
            add(dataset, f"{analyte} nondetects with finite positive limit", int(supported.sum()), f"out of {int(nondetect.sum())} paired nondetects", True, "unsupported nondetects remain interval-indeterminate")

    if not component.empty:
        for _, row in component.iterrows():
            add(
                "California CLIP 2025",
                f"{row['group']} complete component panels",
                int(row["complete_component_panels"]),
                f"out of {int(row['reported_group_events'])} reported group events",
                True,
                "component completeness is a QA descriptor and does not replace reported group values",
            )

    duplicate_review = input_audit["conflicting_duplicate_keys_excluded"].sum()
    add("All", "conflicting duplicate keys excluded", int(duplicate_review), "documented exclusion count", True, "no conflicting key is resolved by first-row selection")
    return pd.DataFrame(checks)


def source_inventory(paths: Paths) -> pd.DataFrame:
    manifest = pd.read_csv(paths.audit_dir / "additional_dataset_manifest.csv")
    wanted = manifest[manifest["dataset"].isin(["EPA SYR3 haloacetic acids", "EPA SYR3 trihalomethanes", "California CLIP 2025"])].copy()
    source_rows = wanted[["dataset", "source_url", "version", "license", "path", "bytes", "sha256", "status"]].copy()

    def repository_relative(value: object) -> str:
        """Keep provenance portable by recording paths relative to the project root."""
        candidate = Path(str(value))
        try:
            return candidate.resolve().relative_to(paths.root).as_posix()
        except (OSError, ValueError):
            return candidate.as_posix()

    source_rows["path"] = source_rows["path"].map(repository_relative)
    exact_files = pd.DataFrame(
        [
            {
                "dataset": "EPA SYR3 HAA5 extracted member",
                "source_url": wanted.loc[wanted["dataset"].eq("EPA SYR3 haloacetic acids"), "source_url"].iloc[0],
                "version": "SYR3, 2006 to 2011",
                "license": "U.S. federal public data",
                "path": paths.syr3_haa.relative_to(paths.root).as_posix(),
                "bytes": paths.syr3_haa.stat().st_size,
                "sha256": sha256_file(paths.syr3_haa),
                "status": "analyzed extracted member",
            },
            {
                "dataset": "EPA SYR3 TTHM extracted member",
                "source_url": wanted.loc[wanted["dataset"].eq("EPA SYR3 trihalomethanes"), "source_url"].iloc[0],
                "version": "SYR3, 2006 to 2011",
                "license": "U.S. federal public data",
                "path": paths.syr3_tthm.relative_to(paths.root).as_posix(),
                "bytes": paths.syr3_tthm.stat().st_size,
                "sha256": sha256_file(paths.syr3_tthm),
                "status": "analyzed extracted member",
            },
        ]
    )
    return pd.concat([source_rows, exact_files], ignore_index=True)


def figure_data(primary_results: list[dict[str, object]], syr4_reference: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries = [result["summary"] for result in primary_results]
    if syr4_reference.get("available"):
        summaries.insert(1, syr4_reference["summary"])
    order = [summary["cohort_id"] for summary in summaries]

    support = pd.DataFrame(
        [
            {
                "cohort_id": summary["cohort_id"],
                "cohort_label": summary["cohort_label"],
                "n_pairs": summary["n_pairs"],
                "n_systems": summary["n_systems"],
                "both_detected_pairs": summary["both_detected_pairs"],
            }
            for summary in summaries
        ]
    )
    support["cohort_id"] = pd.Categorical(support["cohort_id"], order, ordered=True)
    support = support.sort_values("cohort_id")

    correlation = pd.DataFrame(
        [
            {
                "cohort_id": summary["cohort_id"],
                "cohort_label": summary["cohort_label"],
                **summary["correlations"]["both_detected"],
            }
            for summary in summaries
        ]
    )
    correlation["cohort_id"] = pd.Categorical(correlation["cohort_id"], order, ordered=True)
    correlation = correlation.sort_values("cohort_id")

    rates: list[dict] = []
    for summary in summaries:
        for analyte, metric in (("HAA5", "haa5_above_rate_classifiable"), ("TTHM", "tthm_above_rate_classifiable")):
            values = summary["metrics"][metric]
            rates.append(
                {
                    "cohort_id": summary["cohort_id"],
                    "cohort_label": summary["cohort_label"],
                    "analyte": analyte,
                    **values,
                }
            )
    rates_frame = pd.DataFrame(rates)
    rates_frame["cohort_id"] = pd.Categorical(rates_frame["cohort_id"], order, ordered=True)
    rates_frame = rates_frame.sort_values(["cohort_id", "analyte"])

    concordance = pd.concat([result["concordance"] for result in primary_results], ignore_index=True)
    if syr4_reference.get("available"):
        concordance = pd.concat([concordance, syr4_reference["concordance"]], ignore_index=True)
    concordance["cohort_id"] = pd.Categorical(concordance["cohort_id"], order, ordered=True)
    concordance = concordance.sort_values(["cohort_id", "haa5_status", "tthm_status"])
    return support, correlation, rates_frame, concordance


def make_figure(
    paths: Paths,
    support: pd.DataFrame,
    correlation: pd.DataFrame,
    rates: pd.DataFrame,
    concordance: pd.DataFrame,
) -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial Narrow", "Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.15), constrained_layout=False)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.90, bottom=0.125, hspace=0.52, wspace=0.40)
    fig.suptitle("Paired HAA5 and TTHM signals across monitoring archives", x=0.095, ha="left", fontsize=11, fontweight="bold")

    labels = [label.replace(" to ", "-") for label in support["cohort_label"]]
    short_labels = ["SYR3\n2006-2011", "SYR4\n2012-2019", "California\n2025 DS"] if len(labels) == 3 else labels
    x = np.arange(len(support))

    ax = axes[0, 0]
    bars = ax.bar(x, support["n_pairs"], width=0.62, color=["#4C78A8", "#9BA6B2", "#2A9D8F"][: len(x)], edgecolor="white", linewidth=0.7)
    ax.set_yscale("log")
    ax.set_ylabel("Exact paired samples (log scale)")
    ax.set_xticks(x, short_labels)
    ax.set_title("(a) Paired support", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for bar, pairs, systems in zip(bars, support["n_pairs"], support["n_systems"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.12,
            f"{compact_number(int(pairs))}\n{compact_number(int(systems))} systems",
            ha="center",
            va="bottom",
            fontsize=6.8,
            linespacing=1.15,
        )
    ax.set_ylim(max(100, support["n_pairs"].min() / 3), support["n_pairs"].max() * 3.8)

    ax = axes[0, 1]
    y = np.arange(len(correlation))[::-1]
    estimates = correlation["spearman_rho"].to_numpy(float)
    lows = correlation["ci_low"].to_numpy(float)
    highs = correlation["ci_high"].to_numpy(float)
    ax.errorbar(
        estimates,
        y,
        xerr=np.vstack([estimates - lows, highs - estimates]),
        fmt="o",
        markersize=5.5,
        color="#264653",
        ecolor="#264653",
        elinewidth=1.3,
        capsize=2.8,
        zorder=3,
    )
    ax.set_yticks(y, short_labels)
    ax.set_xlim(0.45, 0.85)
    ax.set_xlabel("Spearman rho, both detected")
    ax.set_title("(b) Paired concentration association", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#D9DEE3", linewidth=0.6)
    for yy, n in zip(y, correlation["n_pairs"]):
        ax.text(0.842, yy, f"n={compact_number(int(n))}", fontsize=6.6, ha="right", va="center", color="#374151")

    ax = axes[1, 0]
    colors = {"HAA5": "#2A9D8F", "TTHM": "#E9A23B"}
    offsets = {"HAA5": -0.10, "TTHM": 0.10}
    cohort_order = list(support["cohort_id"].astype(str))
    for analyte in ("HAA5", "TTHM"):
        subset = rates[rates["analyte"].eq(analyte)].copy()
        subset["cohort_key"] = subset["cohort_id"].astype(str)
        subset = subset.set_index("cohort_key").loc[cohort_order].reset_index()
        estimates = subset["estimate"].to_numpy(float)
        lows = subset["ci_low"].to_numpy(float)
        highs = subset["ci_high"].to_numpy(float)
        ax.errorbar(
            x + offsets[analyte],
            estimates,
            yerr=np.vstack([estimates - lows, highs - estimates]),
            fmt="o",
            markersize=5.4,
            color=colors[analyte],
            ecolor=colors[analyte],
            elinewidth=1.2,
            capsize=2.6,
            label=analyte,
        )
    ax.set_xticks(x, short_labels)
    ax.set_ylabel("Definitely-above sample fraction")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title("(c) Sample-level benchmark signals", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.6)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.set_ylim(bottom=0)

    ax = axes[1, 1]
    category_specs = [
        ("below", "below", "Both below", "#CBD5E1"),
        ("above", "below", "HAA5 only", "#2A9D8F"),
        ("below", "above", "TTHM only", "#E9A23B"),
        ("above", "above", "Both above", "#9B5DE5"),
    ]
    bottoms = np.zeros(len(x), dtype=float)
    concordance = concordance.copy()
    concordance["cohort_key"] = concordance["cohort_id"].astype(str)
    for h_status, t_status, category_label, color in category_specs:
        subset = concordance[
            concordance["haa5_status"].eq(h_status) & concordance["tthm_status"].eq(t_status)
        ].set_index("cohort_key")
        values = subset.loc[cohort_order, "fraction_jointly_classifiable"].to_numpy(float)
        ax.bar(x, values, width=0.65, bottom=bottoms, label=category_label, color=color, edgecolor="white", linewidth=0.5)
        bottoms += values
    ax.set_xticks(x, short_labels)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Jointly classifiable pairs")
    ax.set_title("(d) Threshold concordance", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.20), columnspacing=1.0, handlelength=1.2)

    for axis in axes.flat:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(width=0.7, length=3)

    fig.text(
        0.095,
        0.02,
        "Signals are single-sample concentrations >60 ug/L for HAA5 or >80 ug/L for TTHM. They are not compliance determinations.",
        ha="left",
        va="bottom",
        fontsize=7.0,
        color="#374151",
    )
    stem = paths.out / "FigS7_independent_paired_replication"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_methods_results(paths: Paths, summary: dict) -> None:
    syr3 = summary["cohorts"]["SYR3_primary"]
    ca = summary["cohorts"]["CA2025_primary_DS"]
    syr3_sens = summary["cohorts"]["SYR3_RT_only"]
    ca_sens = summary["cohorts"]["CA2025_all_exact"]
    h5_ca = summary["california_component_qa"]["HAA5"]
    thm_ca = summary["california_component_qa"]["TTHM"]

    def corr(cohort: dict) -> dict:
        return cohort["correlations"]["both_detected"]

    def metric(cohort: dict, name: str) -> dict:
        return cohort["metrics"][name]

    text = f"""# Additional historical and state-data validation: SYR3 and California CLIP 2025

## Methods

EPA Six-Year Review 3 (SYR3) HAA5 and TTHM group records were analyzed as a locked historical replication spanning 2006 to 2011. Identifiers were retained as strings. HAA5 and TTHM rows were joined only on the exact key `PWSID + SAMPLE_ID`; rows with a missing key were not matched. Within each analyte, repeated keys were collapsed only when all analysis-critical fields agreed, while conflicting duplicates were excluded. The primary cohort additionally required identical collection dates. A routine-sample-only subset (`Sample Type Code = RT` for both analytes) was retained as a sensitivity analysis.

California CLIP 2025 group records were joined on `Water System Number + PS Code + Sample Date + Lab Sample ID`. All four fields were complete and unique within each analyte after the duplicate audit. The primary state cohort required routine samples (`RT`) from distribution-system facilities (`DS`) for both analytes. The all-exact-pair cohort was retained as a sensitivity analysis because treatment-plant, well, and other facility samples are not interchangeable with distribution-system samples. The release covers a partial calendar year and was analyzed as a one-state external validation.

Detected values were normalized to ug/L and represented as point concentrations. A SYR3 non-detect was represented as `[0, reported detection limit]`; a positive missing-unit limit was treated as ug/L to match the prior SYR4 rule, while mg/L limits were multiplied by 1,000. California non-detects used the populated Reporting Level because DLR is blank for the HAA5 and TTHM group rows. Zero, missing, negative, or unrecognized-unit limits had an unknown upper bound. An interval was definitely above only when its lower bound was strictly greater than 60 ug/L for HAA5 or 80 ug/L for TTHM. It was definitely below when its upper bound was at or below the corresponding threshold. All other intervals were indeterminate.

Paired association was summarized with Spearman rho among pairs with both analytes detected. Ninety-five percent confidence intervals used {summary['metadata']['bootstrap_reps']:,} fixed-seed whole-system bootstrap resamples. Global midranks were fixed before cluster resampling to make the large historical bootstrap tractable. System-median and interval-midpoint estimates were retained as complementary analyses. Rates and agreement used ratios of cluster-resampled totals. Every threshold result is a sample-level benchmark signal. It is not a compliance determination, which requires the applicable regulatory averaging and confirmation framework.

## Results

SYR3 contributed {syr3['n_pairs']:,} same-date exact pairs from {syr3['n_systems']:,} systems. Both analytes were detected in {syr3['both_detected_pairs']:,} pairs. Detected-pair Spearman rho was {corr(syr3)['spearman_rho']:.3f} (95% system-bootstrap CI {corr(syr3)['ci_low']:.3f} to {corr(syr3)['ci_high']:.3f}). Of {syr3['jointly_classifiable_pairs']:,} jointly classifiable pairs, threshold-status agreement was {metric(syr3, 'threshold_status_agreement')['estimate']:.3f} (95% CI {metric(syr3, 'threshold_status_agreement')['ci_low']:.3f} to {metric(syr3, 'threshold_status_agreement')['ci_high']:.3f}). The HAA5 and TTHM definitely-above sample fractions were {metric(syr3, 'haa5_above_rate_classifiable')['estimate']:.3%} and {metric(syr3, 'tthm_above_rate_classifiable')['estimate']:.3%}, respectively. The routine-only sensitivity retained {syr3_sens['n_pairs']:,} pairs from {syr3_sens['n_systems']:,} systems and gave rho {corr(syr3_sens)['spearman_rho']:.3f}.

California contributed {ca_sens['n_pairs']:,} exact state-event pairs from {ca_sens['n_systems']:,} systems. The prespecified routine distribution-system cohort retained {ca['n_pairs']:,} pairs from {ca['n_systems']:,} systems, with {ca['both_detected_pairs']:,} pairs having both analytes detected. Detected-pair rho was {corr(ca)['spearman_rho']:.3f} (95% system-bootstrap CI {corr(ca)['ci_low']:.3f} to {corr(ca)['ci_high']:.3f}). Among {ca['jointly_classifiable_pairs']:,} jointly classifiable pairs, threshold-status agreement was {metric(ca, 'threshold_status_agreement')['estimate']:.3f} (95% CI {metric(ca, 'threshold_status_agreement')['ci_low']:.3f} to {metric(ca, 'threshold_status_agreement')['ci_high']:.3f}). The HAA5 and TTHM definitely-above sample fractions were {metric(ca, 'haa5_above_rate_classifiable')['estimate']:.3%} and {metric(ca, 'tthm_above_rate_classifiable')['estimate']:.3%}. The all-exact-pair sensitivity gave rho {corr(ca_sens)['spearman_rho']:.3f} and threshold agreement {metric(ca_sens, 'threshold_status_agreement')['estimate']:.3f}.

California component panels supported a separate internal audit. Complete components were present for {int(h5_ca['complete_component_panels']):,} of {int(h5_ca['reported_group_events']):,} HAA5 events ({h5_ca['complete_panel_fraction']:.2%}) and {int(thm_ca['complete_component_panels']):,} of {int(thm_ca['reported_group_events']):,} TTHM events ({thm_ca['complete_panel_fraction']:.2%}). Among numeric reported groups with complete panels, {int(h5_ca['within_1_0_ug_L']):,} HAA5 sums and {int(thm_ca['within_1_0_ug_L']):,} TTHM sums were within 1 ug/L of the reported group value. Reported group concentrations remained primary because the public extract does not resolve every rounding or revision difference.

## Interpretation limits

The results replicate paired concentration structure and sample-level benchmark signals across an older national compliance-monitoring release and a recent state release. They do not estimate individual exposure, causal effects, or legal compliance. SYR3 and SYR4 differ in calendar period, participation, and regulatory implementation. California represents one state and January through October 2025, not a national post-2019 cohort. Facility restriction reduces raw- and treatment-water mixing but does not establish that every sample represents the same served-water exposure condition.

## Supplementary Fig. 7 caption

**Supplementary Fig. 7 | Historical and state-data replication of paired HAA5 and TTHM monitoring signals.** (a) Exact paired-sample and system support in EPA SYR3, the existing SYR4 reference, and the California 2025 routine distribution-system cohort. (b) Spearman association among pairs with both analytes detected; points show cohort estimates and bars show 95% whole-system bootstrap confidence intervals. (c) Fractions of classifiable samples definitely above 60 ug/L for HAA5 or 80 ug/L for TTHM, with 95% whole-system bootstrap confidence intervals. (d) Composition of jointly classifiable pairs across the four HAA5/TTHM threshold-status cells. Threshold signals are single-sample descriptors and are not compliance determinations. DS, distribution system; HAA5, five regulated haloacetic acids; TTHM, total trihalomethanes.
"""
    (paths.out / "methods_results.md").write_text(text, encoding="utf-8")


def build_manifest(out: Path, owned_files: Iterable[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for name in sorted(set(owned_files), key=str.lower):
        path = out / name
        if not path.is_file():
            raise FileNotFoundError(f"Expected output was not created: {path}")
        rows.append(
            {
                "file": path.name,
                "relative_path": str(path.relative_to(out.parent.parent)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "format": path.suffix.lower().lstrip("."),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    if args.bootstrap_reps < 0:
        raise ValueError("--bootstrap-reps must be nonnegative")
    paths = make_paths(args.root)
    required = [paths.syr3_haa, paths.syr3_tthm, paths.california, paths.audit_dir / "additional_dataset_manifest.csv"]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(str(path) for path in missing))
    paths.out.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")
    syr3_pairs, syr3_input, syr3_qa = create_syr3_pairs(con, paths)
    ca_pairs, ca_input, ca_qa = create_california_pairs(con, paths)
    con.close()

    syr3_primary_frame = syr3_pairs[syr3_pairs["date_match"].fillna(False)].copy()
    syr3_rt_frame = syr3_primary_frame[
        syr3_primary_frame["haa5_sample_type"].eq("RT") & syr3_primary_frame["tthm_sample_type"].eq("RT")
    ].copy()
    ca_primary_mask = (
        ca_pairs["haa5_sample_type"].eq("RT")
        & ca_pairs["tthm_sample_type"].eq("RT")
        & ca_pairs["haa5_facility_type"].eq("DS")
        & ca_pairs["tthm_facility_type"].eq("DS")
    )
    ca_primary_frame = ca_pairs[ca_primary_mask].copy()

    scope_specs = [
        (
            syr3_primary_frame,
            "SYR3_primary",
            "SYR3 2006 to 2011",
            "primary replication",
            "conflict-resolved exact PWSID plus SAMPLE_ID pair with matching collection dates",
            args.bootstrap_reps,
            args.seed + 1000,
        ),
        (
            syr3_rt_frame,
            "SYR3_RT_only",
            "SYR3 2006 to 2011, routine only",
            "sensitivity",
            "SYR3 primary cohort with Sample Type Code RT for both analytes",
            0,
            args.seed + 2000,
        ),
        (
            ca_primary_frame,
            "CA2025_primary_DS",
            "California 2025 distribution",
            "primary external validation",
            "exact four-field state event key; both analytes routine RT and facility type DS",
            args.bootstrap_reps,
            args.seed + 3000,
        ),
        (
            ca_pairs,
            "CA2025_all_exact",
            "California 2025 all exact pairs",
            "sensitivity",
            "all conflict-resolved exact four-field state event pairs regardless of facility type",
            0,
            args.seed + 4000,
        ),
    ]
    analyses = [analyze_scope(*spec) for spec in scope_specs]
    by_id = {analysis["summary"]["cohort_id"]: analysis for analysis in analyses}
    primary_results = [by_id["SYR3_primary"], by_id["CA2025_primary_DS"]]

    input_audit = pd.concat([syr3_input, ca_input], ignore_index=True)
    pairing_and_field_audit = pd.concat([syr3_qa, ca_qa], ignore_index=True, sort=False)
    metrics = pd.concat([analysis["metrics"] for analysis in analyses], ignore_index=True)
    correlations = pd.concat([analysis["correlations"] for analysis in analyses], ignore_index=True)
    concordance = pd.concat([analysis["concordance"] for analysis in analyses], ignore_index=True)
    concentrations = pd.concat([analysis["concentration"] for analysis in analyses], ignore_index=True)
    strata = pd.concat(
        [
            stratified_summary(syr3_primary_frame, "SYR3_primary", "year"),
            stratified_summary(ca_primary_frame, "CA2025_primary_DS", "month"),
        ],
        ignore_index=True,
    )
    components = component_qa(paths)
    checks = qa_checks(syr3_pairs, ca_pairs, input_audit, components)
    sources = source_inventory(paths)

    syr4_reference = load_syr4_reference(paths, args.bootstrap_reps, args.seed + 5000)
    syr4_pairs = pd.read_parquet(
        paths.validation_pairs, columns=["pwsid", "analysis_split", "analysis_included"]
    )
    syr4_pairs = syr4_pairs[syr4_pairs["analysis_included"].fillna(False)]
    syr3_systems = set(syr3_primary_frame["pwsid"].astype(str).unique())
    syr4_systems = set(syr4_pairs["pwsid"].astype(str).unique())
    syr4_development = set(
        syr4_pairs.loc[syr4_pairs["analysis_split"].eq("development"), "pwsid"].astype(str).unique()
    )
    syr4_holdout = set(
        syr4_pairs.loc[syr4_pairs["analysis_split"].eq("holdout"), "pwsid"].astype(str).unique()
    )
    overlap = pd.DataFrame(
        [
            {
                "comparison": label,
                "syr3_primary_systems": len(syr3_systems),
                "comparison_systems": len(other),
                "overlapping_systems": len(syr3_systems & other),
                "overlap_fraction_of_syr3": len(syr3_systems & other) / len(syr3_systems),
            }
            for label, other in (
                ("SYR4 overall", syr4_systems),
                ("SYR4 development", syr4_development),
                ("SYR4 holdout", syr4_holdout),
            )
        ]
    )
    support_fig, correlation_fig, rates_fig, concordance_fig = figure_data(primary_results, syr4_reference)
    make_figure(paths, support_fig, correlation_fig, rates_fig, concordance_fig)

    tables = {
        "input_audit": input_audit,
        "pairing_and_field_audit": pairing_and_field_audit,
        "replication_metrics": metrics,
        "spearman_correlations": correlations,
        "threshold_concordance_cells": concordance,
        "censoring_concentration_summary": concentrations,
        "time_stratified_metrics": strata,
        "california_component_qa": components,
        "qa_checks": checks,
        "source_inventory": sources,
        "syr3_syr4_system_overlap": overlap,
        "figure_support_data": support_fig,
        "figure_correlation_data": correlation_fig,
        "figure_signal_rate_data": rates_fig,
        "figure_concordance_data": concordance_fig,
    }
    for name, table in tables.items():
        save_table(table, paths.out, name)

    component_summary = {
        str(row["group"]): row.to_dict() for _, row in components.iterrows()
    }
    summary = {
        "metadata": {
            "analysis_name": "SYR3 and California CLIP 2025 paired HAA5/TTHM replication",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "bootstrap_reps": args.bootstrap_reps,
            "seed": args.seed,
            "bootstrap_unit": "public water system",
            "threshold_operator": "strictly greater than",
            "thresholds_ug_l": {"HAA5": HAA5_THRESHOLD, "TTHM": TTHM_THRESHOLD},
            "legal_interpretation": "single-sample benchmark signals are not compliance determinations",
            "software": {
                "python": platform.python_version(),
                "duckdb": duckdb.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pyarrow": pyarrow.__version__,
                "scipy": scipy.__version__,
                "matplotlib": matplotlib.__version__,
            },
        },
        "cohorts": {analysis["summary"]["cohort_id"]: analysis["summary"] for analysis in analyses},
        "syr4_reference": syr4_reference.get("summary") if syr4_reference.get("available") else syr4_reference,
        "syr3_syr4_system_overlap": overlap.to_dict(orient="records"),
        "california_component_qa": component_summary,
        "qa_status_counts": checks["status"].value_counts().to_dict(),
        "source_files": sources.to_dict(orient="records"),
    }
    write_json(summary, paths.out / "additional_validation_summary.json")
    write_methods_results(paths, summary)

    owned_files = [
        *(f"{name}.{extension}" for name in tables for extension in ("csv", "parquet")),
        "FigS7_independent_paired_replication.svg",
        "FigS7_independent_paired_replication.pdf",
        "FigS7_independent_paired_replication.png",
        "additional_validation_summary.json",
        "methods_results.md",
    ]
    manifest = build_manifest(paths.out, owned_files)
    save_table(manifest, paths.out, "additional_validation_manifest")
    write_json(manifest.to_dict(orient="records"), paths.out / "additional_validation_manifest.json")

    compact = {
        "SYR3_primary": summary["cohorts"]["SYR3_primary"],
        "CA2025_primary_DS": summary["cohorts"]["CA2025_primary_DS"],
        "qa_status_counts": summary["qa_status_counts"],
        "output_files": int(len(manifest) + 3),
    }
    print(json.dumps(json_value(compact), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
