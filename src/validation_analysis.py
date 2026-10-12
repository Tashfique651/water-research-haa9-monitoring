#!/usr/bin/env python3
"""Cross-archive validation analyses for SYR4 and ICR DBP data.

The script is deliberately self-contained and deterministic.  It creates paired
SYR4 validation data, ICR Access-table extracts, tabular analyses, exact finite-
grid capture curves, cluster-bootstrap intervals, and a concise methods/results
record.  It does not create figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import numpy as np
import pandas as pd
import pyodbc
from scipy.stats import rankdata, spearmanr


SEED = 20260913
DEFAULT_BOOTSTRAP_REPS = 10_000
PRIMARY_ICR_TYPES = ("AVG", "AVG1", "AVG2", "MAX", "NFC")
ALL_DS_ICR_TYPES = ("ENT", "DSE", "AVG", "AVG1", "AVG2", "MAX", "SDS", "NFC")
ICR_TABLES = ("TUXDBP", "TUXSAMPLE", "TUXUTMON", "TUXWQP", "TUXDISFRES")
HAA5_COMPONENTS = ("MCAA", "DCAA", "TCAA", "MBAA", "DBAA")
HAA9_COMPONENTS = HAA5_COMPONENTS + ("BCAA", "BDCAA", "CDBAA", "TBAA")


@dataclass(frozen=True)
class Paths:
    root: Path
    input_dir: Path
    processed_dir: Path
    results_dir: Path


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def make_paths(root: Path) -> Paths:
    root = root.resolve()
    paths = Paths(
        root=root,
        input_dir=root / "data" / "interim" / "extracted",
        processed_dir=root / "data" / "processed" / "validation",
        results_dir=root / "results" / "validation",
    )
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    return paths


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _sanitize_json(value):
    if isinstance(value, dict):
        return {str(k): _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def write_json(obj, path: Path) -> None:
    path.write_text(
        json.dumps(_sanitize_json(obj), indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_table(df: pd.DataFrame, stem: Path) -> None:
    """Save each analysis table in open CSV and typed Parquet forms."""
    df.to_csv(stem.with_suffix(".csv"), index=False, lineterminator="\n")
    df.to_parquet(stem.with_suffix(".parquet"), index=False)


def stable_holdout(pwsid: str) -> str:
    token = hashlib.blake2b(str(pwsid).encode("utf-8"), digest_size=8, person=b"DBPVAL").digest()
    return "holdout" if int.from_bytes(token, "little") % 5 == 0 else "development"


def source_group(code: object) -> str:
    if pd.isna(code):
        return "Unknown"
    code = str(code).strip().upper()
    if code.startswith("GW"):
        return "Groundwater"
    if code.startswith("SW"):
        return "Surface water"
    if code.startswith("GU"):
        return "Groundwater under influence"
    return "Unknown"


def population_group(value: object) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "Unknown"
    if not np.isfinite(x) or x <= 0:
        return "Unknown"
    if x <= 500:
        return "≤500"
    if x <= 3_300:
        return "501–3,300"
    if x <= 10_000:
        return "3,301–10,000"
    if x <= 100_000:
        return "10,001–100,000"
    return ">100,000"


def cluster_bootstrap_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    clusters: np.ndarray,
    reps: int,
    seed: int,
) -> tuple[float, float, float, int]:
    """Ordinary nonparametric cluster bootstrap for a ratio of totals."""
    frame = pd.DataFrame(
        {
            "cluster": clusters.astype(str),
            "num": np.asarray(numerator, dtype=float),
            "den": np.asarray(denominator, dtype=float),
        }
    )
    agg = frame.groupby("cluster", sort=True)[["num", "den"]].sum()
    a = agg.to_numpy(dtype=float)
    n_clusters = len(a)
    point = float(a[:, 0].sum() / a[:, 1].sum()) if a[:, 1].sum() > 0 else np.nan
    if n_clusters < 2 or reps <= 0:
        return point, np.nan, np.nan, n_clusters
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=float)
    batch = max(1, min(128, 2_000_000 // n_clusters))
    cursor = 0
    while cursor < reps:
        b = min(batch, reps - cursor)
        idx = rng.integers(0, n_clusters, size=(b, n_clusters), dtype=np.int32)
        num = a[idx, 0].sum(axis=1)
        den = a[idx, 1].sum(axis=1)
        draws[cursor : cursor + b] = np.divide(num, den, out=np.full(b, np.nan), where=den > 0)
        cursor += b
    finite = draws[np.isfinite(draws)]
    if not len(finite):
        return point, np.nan, np.nan, n_clusters
    lo, hi = np.quantile(finite, [0.025, 0.975])
    return point, float(lo), float(hi), n_clusters


def fixed_rank_cluster_bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    reps: int,
    seed: int,
) -> tuple[float, float, float, int, int]:
    """Cluster bootstrap Pearson correlation of global midranks.

    Spearman's rho is Pearson correlation of ranks.  To make 10,000 whole-system
    resamples tractable at SYR4 scale, midranks are calculated once on the
    observed sample and their cluster sufficient statistics are resampled.  This
    preserves every observation within a resampled system and avoids treating
    repeated measurements as independent.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    clusters = np.asarray(clusters).astype(str)
    ok = np.isfinite(x) & np.isfinite(y) & pd.notna(clusters)
    x, y, clusters = x[ok], y[ok], clusters[ok]
    if len(x) < 3:
        return np.nan, np.nan, np.nan, len(x), len(np.unique(clusters))
    rx, ry = rankdata(x, method="average"), rankdata(y, method="average")
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
    agg = frame.groupby("cluster", sort=True)[["n", "sx", "sy", "sxx", "syy", "sxy"]].sum()
    a = agg.to_numpy(dtype=float)
    n_clusters = len(a)
    if n_clusters < 2 or reps <= 0:
        return point, np.nan, np.nan, len(x), n_clusters
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=float)
    batch = max(1, min(96, 1_500_000 // n_clusters))
    cursor = 0
    while cursor < reps:
        b = min(batch, reps - cursor)
        idx = rng.integers(0, n_clusters, size=(b, n_clusters), dtype=np.int32)
        totals = a[idx].sum(axis=1)
        n, sx, sy, sxx, syy, sxy = (totals[:, i] for i in range(6))
        cov = sxy - sx * sy / n
        vx = sxx - sx * sx / n
        vy = syy - sy * sy / n
        draws[cursor : cursor + b] = np.divide(
            cov,
            np.sqrt(vx * vy),
            out=np.full(b, np.nan),
            where=(vx > 0) & (vy > 0),
        )
        cursor += b
    finite = draws[np.isfinite(draws)]
    lo, hi = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
    return point, float(lo), float(hi), len(x), n_clusters


def finite_grid_capture(n: int, m: int, k: int) -> float:
    """P(at least one positive) when k of n occasions are sampled without replacement."""
    if n <= 0 or m <= 0 or k <= 0:
        return 0.0
    if k >= n:
        return 1.0
    if n - m < k:
        return 1.0
    return 1.0 - math.comb(n - m, k) / math.comb(n, k)


def classify_intervals(df: pd.DataFrame, prefix: str, threshold: float) -> None:
    detect = pd.to_numeric(df[f"{prefix}_detect"], errors="coerce")
    value = pd.to_numeric(df[f"{prefix}_value_ug_l"], errors="coerce")
    dl = pd.to_numeric(df[f"{prefix}_dl_ug_l"], errors="coerce")
    observed = detect.eq(1) & value.ge(0)
    censored = detect.eq(0)
    finite_dl = censored & dl.gt(0) & np.isfinite(dl)
    lower = pd.Series(np.nan, index=df.index, dtype=float)
    upper = pd.Series(np.nan, index=df.index, dtype=float)
    midpoint = pd.Series(np.nan, index=df.index, dtype=float)
    lower.loc[observed] = value.loc[observed]
    upper.loc[observed] = value.loc[observed]
    midpoint.loc[observed] = value.loc[observed]
    lower.loc[censored] = 0.0
    upper.loc[finite_dl] = dl.loc[finite_dl]
    midpoint.loc[finite_dl] = dl.loc[finite_dl] / 2.0
    status = pd.Series("indeterminate", index=df.index, dtype="string")
    status.loc[lower.gt(threshold)] = "above"
    status.loc[upper.le(threshold)] = "below"
    df[f"{prefix}_lower_ug_l"] = lower
    df[f"{prefix}_upper_ug_l"] = upper
    df[f"{prefix}_midpoint_ug_l"] = midpoint
    df[f"{prefix}_scaled_lower"] = lower / threshold
    df[f"{prefix}_scaled_upper"] = upper / threshold
    df[f"{prefix}_scaled_midpoint"] = midpoint / threshold
    df[f"{prefix}_threshold_status"] = status


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def prepare_water_systems(path: Path) -> tuple[pd.DataFrame, dict]:
    ws = pd.read_csv(path, sep="\t", encoding="cp1252", encoding_errors="replace", low_memory=False)
    ws["PWSID"] = ws["PWSID"].astype("string")
    ws["Adjusted Total Population Served"] = pd.to_numeric(
        ws["Adjusted Total Population Served"], errors="coerce"
    )
    ws["Retail Population Served"] = pd.to_numeric(ws["Retail Population Served"], errors="coerce")
    ws["population"] = ws["Adjusted Total Population Served"].where(
        ws["Adjusted Total Population Served"].gt(0), ws["Retail Population Served"]
    )
    duplicated = ws[ws["PWSID"].duplicated(keep=False)]
    conflict_groups = 0
    for _, group in duplicated.groupby("PWSID"):
        signatures = group[["Source Water Type", "population"]].fillna("<NA>").astype(str).drop_duplicates()
        conflict_groups += int(len(signatures) > 1)
    ws["_score"] = (
        ws["Source Water Type"].notna().astype(int) * 1_000_000
        + ws["population"].notna().astype(int) * 100_000
        + ws["population"].fillna(-1).clip(upper=99_999)
    )
    lookup = (
        ws.sort_values(["PWSID", "_score"], ascending=[True, False])
        .drop_duplicates("PWSID", keep="first")
        .rename(
            columns={
                "PWSID": "ws_pwsid",
                "Source Water Type": "ws_source_code",
                "population": "ws_population",
                "System Name": "ws_system_name",
                "State Code": "ws_state_code",
            }
        )[["ws_pwsid", "ws_source_code", "ws_population", "ws_system_name", "ws_state_code"]]
    )
    audit = {
        "rows": int(len(ws)),
        "unique_pwsids": int(ws["PWSID"].nunique()),
        "duplicated_pwsid_groups": int(duplicated["PWSID"].nunique()),
        "conflicting_source_or_population_groups": int(conflict_groups),
        "deduplication_rule": "prefer nonmissing source and population, then larger population",
    }
    return lookup, audit


def analyze_syr4(paths: Paths, reps: int, seed: int) -> tuple[dict, list[pd.DataFrame]]:
    con = duckdb.connect(database=":memory:")
    h_path = _sql_path(paths.input_dir / "SYR4_HAA5.txt")
    t_path = _sql_path(paths.input_dir / "SYR4_TTHM.txt")
    ws_lookup, ws_audit = prepare_water_systems(paths.input_dir / "SYR4_Water_system_table.txt")
    con.register("ws_lookup", ws_lookup)
    con.execute(
        f"CREATE VIEW h_raw AS SELECT * FROM read_csv_auto('{h_path}', delim='\\t', "
        "header=true, all_varchar=true, sample_size=100000)"
    )
    con.execute(
        f"CREATE VIEW t_raw AS SELECT * FROM read_csv_auto('{t_path}', delim='\\t', "
        "header=true, all_varchar=true, sample_size=100000)"
    )
    signature = (
        "SAMPLE_COLLECTION_DATE,WATER_FACILITY_ID,SAMPLING_POINT_ID,SAMPLE_TYPE_CODE,"
        "DETECT,VALUE,DETECTION_LIMIT_VALUE,DETECTION_LIMIT_UNIT,UNIT,SOURCE_WATER_TYPE,"
        "SYSTEM_TYPE,ADJUSTED_TOTAL_POPULATION_SERVED"
    )
    for tag in ("h", "t"):
        con.execute(
            f"""CREATE TEMP TABLE {tag}_keys AS
            SELECT PWSID, SAMPLE_ID, COUNT(*) AS n_rows,
                   COUNT(DISTINCT hash({signature})) AS n_signatures
            FROM {tag}_raw WHERE SAMPLE_ID IS NOT NULL
            GROUP BY PWSID, SAMPLE_ID"""
        )
        con.execute(
            f"""CREATE TEMP TABLE {tag}_clean AS
            SELECT r.* FROM {tag}_raw r
            JOIN {tag}_keys k USING (PWSID, SAMPLE_ID)
            WHERE k.n_signatures = 1
            QUALIFY row_number() OVER (
                PARTITION BY r.PWSID, r.SAMPLE_ID ORDER BY try_cast(r.SIX_YEAR_ID AS BIGINT)
            ) = 1"""
        )

    def input_stats(tag: str, analyte: str) -> dict:
        raw = con.execute(
            f"""SELECT COUNT(*) AS n_rows,
                       COUNT(*) FILTER (WHERE SAMPLE_ID IS NULL) AS missing_id_rows,
                       COUNT(DISTINCT PWSID) AS n_systems
                FROM {tag}_raw"""
        ).fetchone()
        keys = con.execute(
            f"""SELECT COUNT(*) AS n_keys,
                       COUNT(*) FILTER (WHERE n_rows > 1 AND n_signatures = 1) AS duplicate_consistent_keys,
                       COUNT(*) FILTER (WHERE n_signatures > 1) AS duplicate_conflicting_keys,
                       SUM(n_rows) FILTER (WHERE n_signatures > 1) AS conflicting_rows
                FROM {tag}_keys"""
        ).fetchone()
        return {
            "analyte": analyte,
            "raw_rows": int(raw[0]),
            "missing_sample_id_rows": int(raw[1]),
            "systems": int(raw[2]),
            "nonmissing_sample_keys": int(keys[0]),
            "consistent_duplicate_keys_collapsed": int(keys[1]),
            "conflicting_duplicate_keys_excluded": int(keys[2]),
            "rows_in_conflicting_keys": int(keys[3] or 0),
        }

    h_stats, t_stats = input_stats("h", "HAA5"), input_stats("t", "TTHM")
    con.execute(
        """CREATE TEMP TABLE paired_base AS
        SELECT
          h.PWSID::VARCHAR AS pwsid,
          h.SAMPLE_ID::VARCHAR AS sample_id,
          try_strptime(h.SAMPLE_COLLECTION_DATE, '%d-%b-%y')::DATE AS haa5_date,
          try_strptime(t.SAMPLE_COLLECTION_DATE, '%d-%b-%y')::DATE AS tthm_date,
          (h.SAMPLE_COLLECTION_DATE = t.SAMPLE_COLLECTION_DATE) AS date_match,
          (coalesce(h.WATER_FACILITY_ID,'') = coalesce(t.WATER_FACILITY_ID,'')) AS facility_match,
          (coalesce(h.SAMPLING_POINT_ID,'') = coalesce(t.SAMPLING_POINT_ID,'')) AS sampling_point_match,
          (coalesce(h.SAMPLE_TYPE_CODE,'') = coalesce(t.SAMPLE_TYPE_CODE,'')) AS sample_type_match,
          h.WATER_FACILITY_ID::VARCHAR AS water_facility_id,
          h.SAMPLING_POINT_ID::VARCHAR AS haa5_sampling_point_id,
          t.SAMPLING_POINT_ID::VARCHAR AS tthm_sampling_point_id,
          h.SAMPLE_TYPE_CODE::VARCHAR AS sample_type_code,
          coalesce(ws.ws_source_code,h.SOURCE_WATER_TYPE,t.SOURCE_WATER_TYPE)::VARCHAR AS source_water_code,
          coalesce(ws.ws_population,
                   try_cast(h.ADJUSTED_TOTAL_POPULATION_SERVED AS DOUBLE),
                   try_cast(t.ADJUSTED_TOTAL_POPULATION_SERVED AS DOUBLE)) AS population,
          coalesce(h.SYSTEM_TYPE,t.SYSTEM_TYPE)::VARCHAR AS system_type,
          try_cast(h.DETECT AS INTEGER) AS haa5_detect,
          try_cast(h.VALUE AS DOUBLE) AS haa5_value_ug_l,
          try_cast(h.DETECTION_LIMIT_VALUE AS DOUBLE) AS haa5_dl_reported,
          h.DETECTION_LIMIT_UNIT::VARCHAR AS haa5_dl_unit,
          CASE
            WHEN try_cast(h.DETECTION_LIMIT_VALUE AS DOUBLE) <= 0 THEN NULL
            WHEN upper(coalesce(h.DETECTION_LIMIT_UNIT,'')) = 'MG/L'
              THEN 1000 * try_cast(h.DETECTION_LIMIT_VALUE AS DOUBLE)
            WHEN upper(coalesce(h.DETECTION_LIMIT_UNIT,'')) IN ('','UG/L')
              THEN try_cast(h.DETECTION_LIMIT_VALUE AS DOUBLE)
            ELSE NULL END AS haa5_dl_ug_l,
          try_cast(t.DETECT AS INTEGER) AS tthm_detect,
          try_cast(t.VALUE AS DOUBLE) AS tthm_value_ug_l,
          try_cast(t.DETECTION_LIMIT_VALUE AS DOUBLE) AS tthm_dl_reported,
          t.DETECTION_LIMIT_UNIT::VARCHAR AS tthm_dl_unit,
          CASE
            WHEN try_cast(t.DETECTION_LIMIT_VALUE AS DOUBLE) <= 0 THEN NULL
            WHEN upper(coalesce(t.DETECTION_LIMIT_UNIT,'')) = 'MG/L'
              THEN 1000 * try_cast(t.DETECTION_LIMIT_VALUE AS DOUBLE)
            WHEN upper(coalesce(t.DETECTION_LIMIT_UNIT,'')) IN ('','UG/L')
              THEN try_cast(t.DETECTION_LIMIT_VALUE AS DOUBLE)
            ELSE NULL END AS tthm_dl_ug_l
        FROM h_clean h
        INNER JOIN t_clean t USING (PWSID, SAMPLE_ID)
        LEFT JOIN ws_lookup ws ON h.PWSID = ws.ws_pwsid"""
    )
    paired_all = con.execute("SELECT * FROM paired_base ORDER BY pwsid, sample_id, haa5_date").fetchdf()
    paired_all["sample_date"] = pd.to_datetime(paired_all["haa5_date"], errors="coerce")
    paired_all["year"] = paired_all["sample_date"].dt.year.astype("Int16")
    paired_all["month"] = paired_all["sample_date"].dt.month.astype("Int8")
    paired_all["quarter"] = paired_all["sample_date"].dt.quarter.astype("Int8")
    paired_all["source_water_group"] = paired_all["source_water_code"].map(source_group)
    paired_all["size_category"] = paired_all["population"].map(population_group)
    paired_all["analysis_split"] = paired_all["pwsid"].map(stable_holdout)
    paired_all["analysis_included"] = paired_all["date_match"].fillna(False)
    classify_intervals(paired_all, "haa5", 60.0)
    classify_intervals(paired_all, "tthm", 80.0)
    save_table(paired_all, paths.processed_dir / "syr4_exact_pairs")
    paired = paired_all[paired_all["analysis_included"]].copy()

    pair_audit = pd.DataFrame(
        [
            {"metric": "conflict_resolved_exact_key_pairs", "count": len(paired_all)},
            {"metric": "same_date_primary_pairs", "count": len(paired)},
            {"metric": "date_mismatch_excluded", "count": (~paired_all["date_match"]).sum()},
            {"metric": "facility_mismatch", "count": (~paired_all["facility_match"]).sum()},
            {"metric": "sampling_point_id_mismatch_retained", "count": (~paired_all["sampling_point_match"]).sum()},
            {"metric": "sample_type_mismatch", "count": (~paired_all["sample_type_match"]).sum()},
            {"metric": "primary_systems", "count": paired["pwsid"].nunique()},
        ]
    )
    save_table(pair_audit, paths.results_dir / "syr4_pairing_audit")

    distribution_rows: list[dict] = []
    for analyte, prefix, threshold in (("HAA5", "haa5", 60.0), ("TTHM", "tthm", 80.0)):
        methods = {
            "detected_only": paired[f"{prefix}_value_ug_l"].where(paired[f"{prefix}_detect"].eq(1)),
            "lower_bound": paired[f"{prefix}_lower_ug_l"],
            "midpoint_DL_over_2": paired[f"{prefix}_midpoint_ug_l"],
            "upper_bound": paired[f"{prefix}_upper_ug_l"],
        }
        for method, values in methods.items():
            ok = values.notna() & np.isfinite(values)
            v = values.loc[ok].astype(float) / threshold
            distribution_rows.append(
                {
                    "analyte": analyte,
                    "censoring_summary": method,
                    "threshold_ug_l": threshold,
                    "n_pairs": int(ok.sum()),
                    "n_systems": int(paired.loc[ok, "pwsid"].nunique()),
                    "mean_scaled": v.mean(),
                    "median_scaled": v.median(),
                    "q25_scaled": v.quantile(0.25),
                    "q75_scaled": v.quantile(0.75),
                    "p90_scaled": v.quantile(0.90),
                    "p95_scaled": v.quantile(0.95),
                }
            )
    paired_summary = pd.DataFrame(distribution_rows)
    save_table(paired_summary, paths.results_dir / "syr4_scaled_concentration_summary")

    ctab = (
        paired.groupby(["haa5_threshold_status", "tthm_threshold_status"], dropna=False)
        .agg(n_pairs=("sample_id", "size"), n_systems=("pwsid", "nunique"))
        .reset_index()
    )
    ctab["fraction_all_pairs"] = ctab["n_pairs"] / len(paired)
    save_table(ctab, paths.results_dir / "syr4_threshold_concordance_cells")
    joint = paired["haa5_threshold_status"].isin(["above", "below"]) & paired[
        "tthm_threshold_status"
    ].isin(["above", "below"])
    agreement = joint & paired["haa5_threshold_status"].eq(paired["tthm_threshold_status"])
    p_o = agreement.sum() / joint.sum() if joint.sum() else np.nan
    h_above = (paired.loc[joint, "haa5_threshold_status"] == "above").mean()
    t_above = (paired.loc[joint, "tthm_threshold_status"] == "above").mean()
    p_e = h_above * t_above + (1 - h_above) * (1 - t_above)
    kappa = (p_o - p_e) / (1 - p_e) if np.isfinite(p_e) and p_e < 1 else np.nan
    agree_point, agree_lo, agree_hi, agree_clusters = cluster_bootstrap_ratio(
        agreement.loc[joint].astype(float).to_numpy(),
        np.ones(int(joint.sum()), dtype=float),
        paired.loc[joint, "pwsid"].to_numpy(),
        reps,
        seed + 1,
    )
    concordance_metrics = pd.DataFrame(
        [
            {
                "metric": "jointly_classifiable_pairs",
                "estimate": int(joint.sum()),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n_pairs": len(paired),
                "n_systems": paired.loc[joint, "pwsid"].nunique(),
            },
            {
                "metric": "threshold_status_agreement",
                "estimate": agree_point,
                "ci_low": agree_lo,
                "ci_high": agree_hi,
                "n_pairs": int(joint.sum()),
                "n_systems": agree_clusters,
            },
            {
                "metric": "cohen_kappa_descriptive",
                "estimate": kappa,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n_pairs": int(joint.sum()),
                "n_systems": paired.loc[joint, "pwsid"].nunique(),
            },
        ]
    )
    save_table(concordance_metrics, paths.results_dir / "syr4_concordance_metrics")

    correlation_rows: list[dict] = []
    for method, hcol, tcol in (
        ("both_detected", "haa5_value_ug_l", "tthm_value_ug_l"),
        ("finite_interval_midpoints", "haa5_midpoint_ug_l", "tthm_midpoint_ug_l"),
    ):
        ok = paired[hcol].notna() & paired[tcol].notna()
        if method == "both_detected":
            ok &= paired["haa5_detect"].eq(1) & paired["tthm_detect"].eq(1)
        rho, lo, hi, n, nsys = fixed_rank_cluster_bootstrap_spearman(
            paired.loc[ok, hcol].to_numpy(),
            paired.loc[ok, tcol].to_numpy(),
            paired.loc[ok, "pwsid"].to_numpy(),
            reps,
            seed + (10 if method == "both_detected" else 11),
        )
        correlation_rows.append(
            {
                "analysis": method,
                "spearman_rho": rho,
                "cluster_bootstrap_ci_low": lo,
                "cluster_bootstrap_ci_high": hi,
                "n_pairs": n,
                "n_systems": nsys,
                "bootstrap_method": "whole-system resampling of fixed global midranks",
                "bootstrap_reps": reps,
            }
        )
    detected = paired[paired["haa5_detect"].eq(1) & paired["tthm_detect"].eq(1)]
    system_medians = detected.groupby("pwsid", as_index=False)[["haa5_value_ug_l", "tthm_value_ug_l"]].median()
    if len(system_medians) >= 3:
        rho = spearmanr(system_medians["haa5_value_ug_l"], system_medians["tthm_value_ug_l"]).statistic
        rr, lo, hi, n, nsys = fixed_rank_cluster_bootstrap_spearman(
            system_medians["haa5_value_ug_l"].to_numpy(),
            system_medians["tthm_value_ug_l"].to_numpy(),
            system_medians["pwsid"].to_numpy(),
            reps,
            seed + 12,
        )
        correlation_rows.append(
            {
                "analysis": "system_median_detected",
                "spearman_rho": float(rho),
                "cluster_bootstrap_ci_low": lo,
                "cluster_bootstrap_ci_high": hi,
                "n_pairs": n,
                "n_systems": nsys,
                "bootstrap_method": "whole-system resampling of system medians",
                "bootstrap_reps": reps,
            }
        )
    correlations = pd.DataFrame(correlation_rows)
    save_table(correlations, paths.results_dir / "syr4_spearman_correlations")

    pattern_rows: list[dict] = []
    dimensions = (("quarter", "quarter"), ("source_water", "source_water_group"), ("size", "size_category"), ("system_type", "system_type"))
    for dimension, column in dimensions:
        for level, group in paired.groupby(column, dropna=False, observed=True):
            hj = group["haa5_threshold_status"].isin(["above", "below"])
            tj = group["tthm_threshold_status"].isin(["above", "below"])
            both_det = group["haa5_detect"].eq(1) & group["tthm_detect"].eq(1)
            pattern_rows.append(
                {
                    "dimension": dimension,
                    "level": str(level),
                    "n_pairs": len(group),
                    "n_systems": group["pwsid"].nunique(),
                    "both_detected_pairs": int(both_det.sum()),
                    "median_haa5_scaled_detected": (group.loc[group["haa5_detect"].eq(1), "haa5_value_ug_l"] / 60).median(),
                    "median_tthm_scaled_detected": (group.loc[group["tthm_detect"].eq(1), "tthm_value_ug_l"] / 80).median(),
                    "haa5_classifiable_pairs": int(hj.sum()),
                    "haa5_definite_above_rate": group.loc[hj, "haa5_threshold_status"].eq("above").mean(),
                    "tthm_classifiable_pairs": int(tj.sum()),
                    "tthm_definite_above_rate": group.loc[tj, "tthm_threshold_status"].eq("above").mean(),
                }
            )
    patterns = pd.DataFrame(pattern_rows)
    save_table(patterns, paths.results_dir / "syr4_season_source_size_patterns")

    holdout_rows: list[dict] = []
    for split, group in paired.groupby("analysis_split"):
        det = group["haa5_detect"].eq(1) & group["tthm_detect"].eq(1)
        rho = spearmanr(group.loc[det, "haa5_value_ug_l"], group.loc[det, "tthm_value_ug_l"]).statistic
        jj = group["haa5_threshold_status"].isin(["above", "below"]) & group[
            "tthm_threshold_status"
        ].isin(["above", "below"])
        holdout_rows.append(
            {
                "split": split,
                "n_pairs": len(group),
                "n_systems": group["pwsid"].nunique(),
                "both_detected_n": int(det.sum()),
                "both_detected_systems": int(group.loc[det, "pwsid"].nunique()),
                "spearman_rho_both_detected": rho,
                "jointly_classifiable_n": int(jj.sum()),
                "jointly_classifiable_systems": int(group.loc[jj, "pwsid"].nunique()),
                "threshold_agreement": group.loc[jj, "haa5_threshold_status"].eq(
                    group.loc[jj, "tthm_threshold_status"]
                ).mean(),
                "haa5_above_rate_classifiable": group.loc[
                    group["haa5_threshold_status"].isin(["above", "below"]), "haa5_threshold_status"
                ].eq("above").mean(),
                "tthm_above_rate_classifiable": group.loc[
                    group["tthm_threshold_status"].isin(["above", "below"]), "tthm_threshold_status"
                ].eq("above").mean(),
            }
        )
    holdout = pd.DataFrame(holdout_rows)
    save_table(holdout, paths.results_dir / "syr4_deterministic_holdout")

    syq_rows: list[pd.DataFrame] = []
    curve_rows: list[dict] = []
    target_rows: list[dict] = []
    for analyte, prefix, threshold in (("HAA5", "haa5", 60.0), ("TTHM", "tthm", 80.0)):
        quarter_source = paired[["pwsid", "year", "quarter", f"{prefix}_threshold_status"]].copy()
        quarter_source["_above"] = quarter_source[f"{prefix}_threshold_status"].eq("above").astype("int8")
        quarter_source["_below"] = quarter_source[f"{prefix}_threshold_status"].eq("below").astype("int8")
        quarter_source["_indeterminate"] = quarter_source[f"{prefix}_threshold_status"].eq("indeterminate").astype("int8")
        quarter_grid = (
            quarter_source.groupby(["pwsid", "year", "quarter"], dropna=False, sort=False)
            .agg(
                n_pairs=("_above", "size"),
                n_above=("_above", "sum"),
                n_below=("_below", "sum"),
                n_indeterminate=("_indeterminate", "sum"),
            )
            .reset_index()
        )
        quarter_grid["quarter_status"] = np.where(
            quarter_grid["n_above"].gt(0),
            "above",
            np.where(quarter_grid["n_indeterminate"].eq(0), "below", "indeterminate"),
        )
        quarter_grid["_quarter_above"] = quarter_grid["quarter_status"].eq("above").astype("int8")
        quarter_grid["_quarter_indeterminate"] = quarter_grid["quarter_status"].eq("indeterminate").astype("int8")
        quarter_grid["analyte"] = analyte
        quarter_grid["threshold_ug_l"] = threshold
        syq_rows.append(quarter_grid)
        quarter_grid["_quarter_above"] = (quarter_grid["quarter_status"] == "above").astype("int8")
        quarter_grid["_quarter_indeterminate"] = (quarter_grid["quarter_status"] == "indeterminate").astype("int8")
        annual = (
            quarter_grid.groupby(["pwsid", "year"], dropna=False, sort=False)
            .agg(
                n_available_quarters=("quarter", "nunique"),
                n_above_quarters=("_quarter_above", "sum"),
                n_indeterminate_quarters=("_quarter_indeterminate", "sum"),
            )
            .reset_index()
        )
        annual["analyte"] = analyte
        annual["threshold_ug_l"] = threshold
        annual["eligible_ge3_complete"] = annual["n_available_quarters"].ge(3) & annual[
            "n_indeterminate_quarters"
        ].eq(0)
        annual["full_grid_positive"] = annual["n_above_quarters"].gt(0)
        syq_rows.append(annual.assign(record_type="system_year"))
        for n_grid, cohort in annual[
            annual["eligible_ge3_complete"] & annual["full_grid_positive"]
        ].groupby("n_available_quarters"):
            n_grid = int(n_grid)
            local_curves = []
            for k in range(1, n_grid + 1):
                probs = cohort["n_above_quarters"].map(lambda m: finite_grid_capture(n_grid, int(m), k)).to_numpy()
                point, lo, hi, nsys = cluster_bootstrap_ratio(
                    probs,
                    np.ones(len(probs)),
                    cohort["pwsid"].to_numpy(),
                    reps,
                    seed + 100 + int(threshold) + 10 * n_grid + k,
                )
                row = {
                    "analyte": analyte,
                    "threshold_ug_l": threshold,
                    "n_available_quarters": n_grid,
                    "quarters_sampled_k": k,
                    "mean_exact_capture_probability": point,
                    "cluster_bootstrap_ci_low": lo,
                    "cluster_bootstrap_ci_high": hi,
                    "n_system_years": len(cohort),
                    "n_systems": nsys,
                    "cohort": "≥3 quarters; every quarter classifiable; ≥1 definitely-above quarter",
                }
                curve_rows.append(row)
                local_curves.append(row)
            for target in (0.5, 0.9):
                hits = [r for r in local_curves if r["mean_exact_capture_probability"] >= target]
                target_rows.append(
                    {
                        "analyte": analyte,
                        "n_available_quarters": n_grid,
                        "target_probability": target,
                        "minimum_k": hits[0]["quarters_sampled_k"] if hits else np.nan,
                        "status": "reached" if hits else "not reached on supported grid",
                        "n_system_years": len(cohort),
                        "n_systems": cohort["pwsid"].nunique(),
                    }
                )
    syq_quarters = pd.concat([x for x in syq_rows if "quarter_status" in x.columns], ignore_index=True)
    syq_annual = pd.concat([x for x in syq_rows if "record_type" in x.columns], ignore_index=True)
    save_table(syq_quarters, paths.processed_dir / "syr4_system_year_quarter_grid")
    save_table(syq_annual, paths.results_dir / "syr4_system_year_support")
    syr4_curve = pd.DataFrame(curve_rows)
    syr4_targets = pd.DataFrame(target_rows)
    save_table(syr4_curve, paths.results_dir / "syr4_random_quarter_capture_curve")
    save_table(syr4_targets, paths.results_dir / "syr4_capture_targets")

    input_audit = pd.DataFrame([h_stats, t_stats])
    save_table(input_audit, paths.results_dir / "syr4_input_audit")
    summary = {
        "water_system_table": ws_audit,
        "inputs": [h_stats, t_stats],
        "conflict_resolved_exact_pairs": int(len(paired_all)),
        "same_date_primary_pairs": int(len(paired)),
        "primary_systems": int(paired["pwsid"].nunique()),
        "both_detected_pairs": int((paired["haa5_detect"].eq(1) & paired["tthm_detect"].eq(1)).sum()),
        "jointly_classifiable_pairs": int(joint.sum()),
        "threshold_agreement": float(agree_point),
        "threshold_agreement_ci": [agree_lo, agree_hi],
        "correlations": correlations.to_dict(orient="records"),
        "holdout": holdout.to_dict(orient="records"),
        "capture_curve_rows": int(len(syr4_curve)),
    }
    return summary, [input_audit, pair_audit, paired_summary, correlations, patterns, holdout, syr4_curve]


def read_access_table(connection: pyodbc.Connection, table: str) -> pd.DataFrame:
    cursor = connection.cursor()
    cursor.execute(f"SELECT * FROM [{table}]")
    columns = [item[0].upper() for item in cursor.description]
    rows: list[tuple] = []
    while True:
        batch = cursor.fetchmany(10_000)
        if not batch:
            break
        rows.extend(tuple(row) for row in batch)
    return pd.DataFrame.from_records(rows, columns=columns)


def expected_total(dbp: pd.DataFrame, components: Sequence[str]) -> pd.Series:
    values = dbp[list(components)].apply(pd.to_numeric, errors="coerce")
    invalid_negative = ((values < 0) & values.ne(-999)).any(axis=1)
    complete = values.notna().all(axis=1) & ~invalid_negative
    all_bmrl = values.eq(-999).all(axis=1)
    result = values.mask(values.eq(-999), 0.0).sum(axis=1).astype(float)
    result.loc[complete & all_bmrl] = -999.0
    result.loc[~complete] = np.nan
    return result


def identity_audit(dbp_joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    scopes = {
        "all_dbp": pd.Series(True, index=dbp_joined.index),
        "primary_distribution": dbp_joined["EVNTTYPE"].isin(PRIMARY_ICR_TYPES),
        "all_ds_sensitivity": dbp_joined["EVNTTYPE"].isin(ALL_DS_ICR_TYPES),
    }
    for total, comps in (("HAA5", HAA5_COMPONENTS), ("HAA9", HAA9_COMPONENTS)):
        expected = expected_total(dbp_joined, comps)
        reported = pd.to_numeric(dbp_joined[total], errors="coerce")
        comparable = expected.notna() & reported.notna()
        matches = comparable & (
            ((expected.eq(-999)) & reported.eq(-999))
            | (expected.ge(0) & reported.ge(0) & (expected - reported).abs().le(1e-9))
        )
        numeric_comparable = comparable & expected.ge(0) & reported.ge(0)
        for scope, keep in scopes.items():
            k = keep.fillna(False)
            diff = (expected.loc[k & numeric_comparable] - reported.loc[k & numeric_comparable]).abs()
            rows.append(
                {
                    "scope": scope,
                    "total": total,
                    "n_events": int(k.sum()),
                    "reported_present": int((k & reported.notna()).sum()),
                    "reported_numeric": int((k & reported.ge(0)).sum()),
                    "reported_minus999": int((k & reported.eq(-999)).sum()),
                    "component_identity_supported": int((k & expected.notna()).sum()),
                    "identity_matches": int((k & matches).sum()),
                    "identity_mismatches": int((k & comparable & ~matches).sum()),
                    "max_absolute_numeric_difference_ug_l": diff.max() if len(diff) else np.nan,
                    "tolerance_ug_l": 1e-9,
                }
            )
    return pd.DataFrame(rows)


def minus999_audit(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    analytical_tables = {"TUXDBP", "TUXWQP", "TUXDISFRES"}
    for table, frame in tables.items():
        for column in frame.columns:
            if not pd.api.types.is_numeric_dtype(frame[column]):
                continue
            x = pd.to_numeric(frame[column], errors="coerce")
            n_minus = int(x.eq(-999).sum())
            n_other_negative = int((x.lt(0) & ~x.eq(-999)).sum())
            if n_minus or n_other_negative or table in analytical_tables:
                rows.append(
                    {
                        "table": table,
                        "column": column,
                        "n_rows": len(frame),
                        "n_missing": int(x.isna().sum()),
                        "n_minus999": n_minus,
                        "n_other_negative": n_other_negative,
                        "n_nonnegative": int(x.ge(0).sum()),
                        "minus999_interpretation": (
                            "below minimum reporting level; do not treat as -999 or zero"
                            if table in analytical_tables
                            else "not assigned an analytical-result meaning by the user guide; treated as missing if encountered"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def icr_capture_analysis(joined: pd.DataFrame, reps: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grid_rows: list[pd.DataFrame] = []
    support_rows: list[dict] = []
    curve_rows: list[dict] = []
    target_rows: list[dict] = []
    scopes = {"primary_distribution": PRIMARY_ICR_TYPES, "all_ds_sensitivity": ALL_DS_ICR_TYPES}
    for scope, event_types in scopes.items():
        scoped = joined[joined["EVNTTYPE"].isin(event_types)].copy()
        for analyte in ("HAA5", "HAA9"):
            value = pd.to_numeric(scoped[analyte], errors="coerce")
            numeric = scoped[value.ge(0) & scoped["ICRPWSID"].notna()].copy()
            numeric["_value"] = pd.to_numeric(numeric[analyte], errors="coerce")
            numeric = numeric[numeric["SAMP_PER"].between(1, 18) & numeric["SAMP_QTR"].between(1, 6)]
            monthly = (
                numeric.groupby(["ICRPWSID", "SAMP_PER", "SAMP_QTR"], as_index=False)["_value"]
                .max()
                .rename(columns={"_value": "period_max_ug_l"})
            )
            monthly["positive_gt60"] = monthly["period_max_ug_l"].gt(60.0)
            quarterly = (
                monthly.groupby(["ICRPWSID", "SAMP_QTR"], as_index=False)
                .agg(period_max_ug_l=("period_max_ug_l", "max"))
            )
            quarterly["positive_gt60"] = quarterly["period_max_ug_l"].gt(60.0)
            quarters_by_system = quarterly.groupby("ICRPWSID")["SAMP_QTR"].agg(lambda s: set(map(int, s)))
            full_systems = quarters_by_system[quarters_by_system.map(lambda s: s == set(range(1, 7)))].index
            for scale, grid in (("monthly", monthly), ("quarterly", quarterly)):
                system_grid = (
                    grid[grid["ICRPWSID"].isin(full_systems)]
                    .groupby("ICRPWSID", as_index=False)
                    .agg(n_supported=("positive_gt60", "size"), n_positive=("positive_gt60", "sum"))
                )
                system_grid["scope"] = scope
                system_grid["analyte"] = analyte
                system_grid["time_scale"] = scale
                system_grid["threshold_ug_l"] = 60.0
                system_grid["ever_positive"] = system_grid["n_positive"].gt(0)
                grid_rows.append(system_grid)
                positive = system_grid[system_grid["ever_positive"]].copy()
                support_rows.append(
                    {
                        "scope": scope,
                        "analyte": analyte,
                        "time_scale": scale,
                        "scoped_dbp_events": len(scoped),
                        "numeric_analyte_events": len(numeric),
                        "minus999_events": int(value.eq(-999).sum()),
                        "systems_with_numeric_data": numeric["ICRPWSID"].nunique(),
                        "full_six_quarter_systems": len(system_grid),
                        "full_window_positive_systems": len(positive),
                        "min_supported_grid": system_grid["n_supported"].min() if len(system_grid) else np.nan,
                        "median_supported_grid": system_grid["n_supported"].median() if len(system_grid) else np.nan,
                        "max_supported_grid": system_grid["n_supported"].max() if len(system_grid) else np.nan,
                        "threshold_definition": ">60 µg/L; -999 and null are not numeric support",
                    }
                )
                if not len(positive):
                    for target in (0.5, 0.9):
                        target_rows.append(
                            {
                                "scope": scope,
                                "analyte": analyte,
                                "time_scale": scale,
                                "target_probability": target,
                                "minimum_k": np.nan,
                                "status": "no full-window positive systems",
                                "n_systems": 0,
                            }
                        )
                    continue
                max_k = int(positive["n_supported"].min())
                local: list[dict] = []
                for k in range(1, max_k + 1):
                    probs = positive.apply(
                        lambda r: finite_grid_capture(int(r["n_supported"]), int(r["n_positive"]), k), axis=1
                    ).to_numpy(dtype=float)
                    point, lo, hi, nsys = cluster_bootstrap_ratio(
                        probs,
                        np.ones(len(probs)),
                        positive["ICRPWSID"].to_numpy(),
                        reps,
                        seed + 1000 + 100 * list(scopes).index(scope) + 20 * (analyte == "HAA9") + 10 * (scale == "quarterly") + k,
                    )
                    row = {
                        "scope": scope,
                        "analyte": analyte,
                        "time_scale": scale,
                        "k_sampled_without_replacement": k,
                        "mean_exact_capture_probability": point,
                        "cluster_bootstrap_ci_low": lo,
                        "cluster_bootstrap_ci_high": hi,
                        "n_systems": nsys,
                        "minimum_system_grid_size": max_k,
                        "cohort": "numeric data in all six ICR quarters and ≥1 observed value >60 µg/L",
                    }
                    curve_rows.append(row)
                    local.append(row)
                for target in (0.5, 0.9):
                    hits = [r for r in local if r["mean_exact_capture_probability"] >= target]
                    sparse = len(positive) < 10
                    target_rows.append(
                        {
                            "scope": scope,
                            "analyte": analyte,
                            "time_scale": scale,
                            "target_probability": target,
                            "minimum_k": (
                                np.nan if sparse else (hits[0]["k_sampled_without_replacement"] if hits else np.nan)
                            ),
                            "descriptive_crossing_k": (
                                hits[0]["k_sampled_without_replacement"] if hits else np.nan
                            ),
                            "status": (
                                "insufficient support: fewer than 10 reference-positive systems"
                                if sparse
                                else ("reached" if hits else "not reached on supported grid")
                            ),
                            "n_systems": len(positive),
                        }
                    )
    return (
        pd.concat(grid_rows, ignore_index=True) if grid_rows else pd.DataFrame(),
        pd.DataFrame(support_rows),
        pd.DataFrame(curve_rows),
        pd.DataFrame(target_rows),
    )


def icr_matched_factorial(
    joined: pd.DataFrame, reps: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Matched HAA5/HAA9 x one-period/full-grid factorial at B=60 µg/L."""
    h5 = pd.to_numeric(joined["HAA5"], errors="coerce")
    h9 = pd.to_numeric(joined["HAA9"], errors="coerce")
    events = joined[
        joined["EVNTTYPE"].isin(PRIMARY_ICR_TYPES)
        & joined["ICRPWSID"].notna()
        & joined["SAMP_PER"].between(1, 18)
        & joined["SAMP_QTR"].between(1, 6)
        & h5.ge(0)
        & h9.ge(0)
    ][["EVENT_ID", "ICRPWSID", "SAMP_PER", "SAMP_QTR", "HAA5", "HAA9"]].copy()
    events["HAA5"] = pd.to_numeric(events["HAA5"], errors="coerce")
    events["HAA9"] = pd.to_numeric(events["HAA9"], errors="coerce")
    event_monotonicity_violations = int((events["HAA5"] > events["HAA9"] + 1e-9).sum())
    monthly = (
        events.groupby(["ICRPWSID", "SAMP_PER", "SAMP_QTR"], as_index=False, sort=False)
        .agg(HAA5=("HAA5", "max"), HAA9=("HAA9", "max"), n_paired_events=("EVENT_ID", "size"))
    )
    quarterly = (
        monthly.groupby(["ICRPWSID", "SAMP_QTR"], as_index=False, sort=False)
        .agg(HAA5=("HAA5", "max"), HAA9=("HAA9", "max"), n_paired_events=("n_paired_events", "sum"))
    )
    cohort_specs = (
        ("monthly_available_periods", monthly, "SAMP_PER", "at least two observed months with paired nonnegative panels"),
        ("quarterly_available_quarters", quarterly, "SAMP_QTR", "at least two observed quarters with paired nonnegative panels"),
        ("quarterly_complete_six_quarters", quarterly, "SAMP_QTR", "paired nonnegative panels in every ICR quarter 1–6"),
    )
    system_rows: list[dict] = []
    support_rows: list[dict] = []
    cell_rows: list[dict] = []
    contrast_rows: list[dict] = []
    repair_rows: list[dict] = []
    metric_seed = seed + 5_000

    def estimate(values: pd.Series, ids: pd.Series, local_seed: int) -> tuple[float, float, float, int]:
        return cluster_bootstrap_ratio(
            values.to_numpy(dtype=float), np.ones(len(values)), ids.to_numpy(), reps, local_seed
        )

    for cohort_name, grid, grid_column, rule in cohort_specs:
        if cohort_name == "quarterly_complete_six_quarters":
            sets = grid.groupby("ICRPWSID")[grid_column].agg(lambda s: set(map(int, s)))
            eligible_ids = sets[sets.map(lambda s: s == set(range(1, 7)))].index
        else:
            counts = grid.groupby("ICRPWSID")[grid_column].nunique()
            eligible_ids = counts[counts >= 2].index
        eligible = grid[grid["ICRPWSID"].isin(eligible_ids)].copy()
        local_rows: list[dict] = []
        for pwsid, group in eligible.groupby("ICRPWSID", sort=True):
            group = group.sort_values(grid_column)
            y00 = group["HAA5"].gt(60).to_numpy()
            y10 = group["HAA9"].gt(60).to_numpy()
            y01 = bool(y00.any())
            y11 = bool(y10.any())
            already = y00
            panel_only = (~y00) & y10 & (not y01)
            design_only = (~y00) & (~y10) & y01
            either_change = (~y00) & y10 & y01
            both_required = (~y00) & (~y10) & (not y01) & y11
            categories = np.vstack([already, panel_only, design_only, either_change, both_required]).mean(axis=1)
            row = {
                "cohort": cohort_name,
                "ICRPWSID": pwsid,
                "n_grid_periods": len(group),
                "n_paired_events": int(group["n_paired_events"].sum()),
                "p00_HAA5_M0": float(y00.mean()),
                "p10_HAA9_M0": float(y10.mean()),
                "p01_HAA5_M1": float(y01),
                "p11_HAA9_M1": float(y11),
                "interaction": float(y11 - y10.mean() - float(y01) + y00.mean()),
                "already_detected": float(categories[0]),
                "panel_only_repair": float(categories[1]),
                "design_only_repair": float(categories[2]),
                "either_change_repair": float(categories[3]),
                "both_changes_required": float(categories[4]),
                "reference_positive": y11,
            }
            local_rows.append(row)
            system_rows.append(row)
        local = pd.DataFrame(local_rows)
        ref = local[local["reference_positive"]].copy() if len(local) else local.copy()
        support_rows.append(
            {
                "cohort": cohort_name,
                "cohort_rule": rule,
                "paired_nonnegative_events_available": len(events),
                "systems_with_any_paired_event": events["ICRPWSID"].nunique(),
                "eligible_systems": len(local),
                "reference_positive_systems": len(ref),
                "reference_positive_fraction": len(ref) / len(local) if len(local) else np.nan,
                "min_grid_periods_eligible": local["n_grid_periods"].min() if len(local) else np.nan,
                "median_grid_periods_eligible": local["n_grid_periods"].median() if len(local) else np.nan,
                "max_grid_periods_eligible": local["n_grid_periods"].max() if len(local) else np.nan,
                "event_monotonicity_violations_HAA5_gt_HAA9": event_monotonicity_violations,
                "unknown_four_cell_outcomes": 0,
                "weighting": "equal weight per system; M0 uniform over that system's observed paired grid",
            }
        )
        cell_map = {
            "p00": "p00_HAA5_M0",
            "p10": "p10_HAA9_M0",
            "p01": "p01_HAA5_M1",
            "p11": "p11_HAA9_M1",
        }
        for conditioning, data in (("all_eligible", local), ("HAA9_M1_reference_positive", ref)):
            for cell, column in cell_map.items():
                metric_seed += 1
                point, lo, hi, nsys = estimate(data[column], data["ICRPWSID"], metric_seed) if len(data) else (np.nan, np.nan, np.nan, 0)
                cell_rows.append(
                    {
                        "cohort": cohort_name,
                        "conditioning": conditioning,
                        "cell": cell,
                        "panel": "HAA5" if cell in ("p00", "p01") else "HAA9",
                        "monitoring": "M0_uniform_one_period" if cell in ("p00", "p10") else "M1_full_finite_grid",
                        "probability": point,
                        "cluster_bootstrap_ci_low": lo,
                        "cluster_bootstrap_ci_high": hi,
                        "n_systems": nsys,
                        "definitionally_one": bool(conditioning == "HAA9_M1_reference_positive" and cell == "p11"),
                    }
                )
        contrasts = {
            "panel_gain_at_M0": ref["p10_HAA9_M0"] - ref["p00_HAA5_M0"],
            "design_gain_at_HAA5": ref["p01_HAA5_M1"] - ref["p00_HAA5_M0"],
            "combined_gain": ref["p11_HAA9_M1"] - ref["p00_HAA5_M0"],
            "interaction": ref["interaction"],
            "symmetric_scope_attribution": 0.5
            * ((ref["p10_HAA9_M0"] - ref["p00_HAA5_M0"]) + (ref["p11_HAA9_M1"] - ref["p01_HAA5_M1"])),
            "symmetric_design_attribution": 0.5
            * ((ref["p01_HAA5_M1"] - ref["p00_HAA5_M0"]) + (ref["p11_HAA9_M1"] - ref["p10_HAA9_M0"])),
        }
        for metric, values in contrasts.items():
            metric_seed += 1
            point, lo, hi, nsys = estimate(values, ref["ICRPWSID"], metric_seed) if len(ref) else (np.nan, np.nan, np.nan, 0)
            contrast_rows.append(
                {
                    "cohort": cohort_name,
                    "conditioning": "HAA9_M1_reference_positive",
                    "contrast": metric,
                    "estimate": point,
                    "cluster_bootstrap_ci_low": lo,
                    "cluster_bootstrap_ci_high": hi,
                    "n_systems": nsys,
                }
            )
        for category in (
            "already_detected",
            "panel_only_repair",
            "design_only_repair",
            "either_change_repair",
            "both_changes_required",
        ):
            metric_seed += 1
            point, lo, hi, nsys = estimate(ref[category], ref["ICRPWSID"], metric_seed) if len(ref) else (np.nan, np.nan, np.nan, 0)
            repair_rows.append(
                {
                    "cohort": cohort_name,
                    "conditioning": "HAA9_M1_reference_positive",
                    "repair_category": category,
                    "fraction": point,
                    "cluster_bootstrap_ci_low": lo,
                    "cluster_bootstrap_ci_high": hi,
                    "n_systems": nsys,
                }
            )
    return (
        pd.DataFrame(system_rows),
        pd.DataFrame(support_rows),
        pd.DataFrame(cell_rows),
        pd.DataFrame(contrast_rows),
        pd.DataFrame(repair_rows),
    )


def analyze_icr(paths: Paths, reps: int, seed: int) -> tuple[dict, list[pd.DataFrame]]:
    db_path = (paths.input_dir / "DBPICR_Aux1.accdb").resolve()
    connection = pyodbc.connect(
        f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={db_path};",
        autocommit=True,
    )
    tables = {name: read_access_table(connection, name) for name in ICR_TABLES}
    connection.close()
    for name, frame in tables.items():
        save_table(frame, paths.processed_dir / f"icr_{name.lower()}_extract")

    sentinel = minus999_audit(tables)
    save_table(sentinel, paths.results_dir / "icr_minus999_audit")
    dbp, sample, utmon, wqp, residual = (tables[name] for name in ICR_TABLES)
    event_audit = []
    for name, frame in (("TUXDBP", dbp), ("TUXSAMPLE", sample), ("TUXWQP", wqp), ("TUXDISFRES", residual)):
        event_audit.append(
            {
                "table": name,
                "rows": len(frame),
                "nonmissing_event_ids": int(frame["EVENT_ID"].notna().sum()),
                "unique_event_ids": int(frame["EVENT_ID"].nunique()),
                "duplicated_event_id_rows": int(frame["EVENT_ID"].duplicated(keep=False).sum()),
            }
        )
    event_audit_df = pd.DataFrame(event_audit)
    save_table(event_audit_df, paths.results_dir / "icr_event_id_audit")

    joined = dbp.merge(sample, on="EVENT_ID", how="left", validate="one_to_one", indicator="_sample_join")
    joined = joined.merge(wqp, on="EVENT_ID", how="left", validate="one_to_one", suffixes=("", "_WQP"))
    joined = joined.merge(residual, on="EVENT_ID", how="left", validate="one_to_one", suffixes=("", "_DISFRES"))
    utmon_keys = utmon.sort_values(["ICRPWSID", "SAMP_PER"]).drop_duplicates(["ICRPWSID", "SAMP_PER"])
    joined = joined.merge(
        utmon_keys,
        on=["ICRPWSID", "SAMP_PER"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_UTMON"),
        indicator="_utmon_join",
    )
    joined["PRIMARY_DISTRIBUTION"] = joined["EVNTTYPE"].isin(PRIMARY_ICR_TYPES)
    joined["ALL_DS_SENSITIVITY"] = joined["EVNTTYPE"].isin(ALL_DS_ICR_TYPES)
    joined["HAA5_RECOMPUTED"] = expected_total(joined, HAA5_COMPONENTS)
    joined["HAA9_RECOMPUTED"] = expected_total(joined, HAA9_COMPONENTS)
    joined["HAA5_NUMERIC"] = pd.to_numeric(joined["HAA5"], errors="coerce").where(
        pd.to_numeric(joined["HAA5"], errors="coerce").ge(0)
    )
    joined["HAA9_NUMERIC"] = pd.to_numeric(joined["HAA9"], errors="coerce").where(
        pd.to_numeric(joined["HAA9"], errors="coerce").ge(0)
    )
    save_table(joined, paths.processed_dir / "icr_dbp_sample_wqp_residual_joined")

    identities = identity_audit(joined)
    save_table(identities, paths.results_dir / "icr_haa_identity_audit")
    availability_rows: list[dict] = []
    scopes = {
        "all_dbp": pd.Series(True, index=joined.index),
        "primary_distribution": joined["PRIMARY_DISTRIBUTION"],
        "all_ds_sensitivity": joined["ALL_DS_SENSITIVITY"],
    }
    for scope, keep in scopes.items():
        for dimension, column in (("overall", None), ("event_type", "EVNTTYPE"), ("sample_period", "SAMP_PER"), ("sample_quarter", "SAMP_QTR")):
            subset = joined[keep.fillna(False)]
            groups: Iterable[tuple[object, pd.DataFrame]] = [("all", subset)] if column is None else subset.groupby(column, dropna=False)
            for level, group in groups:
                for analyte in ("HAA5", "HAA9"):
                    x = pd.to_numeric(group[analyte], errors="coerce")
                    availability_rows.append(
                        {
                            "scope": scope,
                            "dimension": dimension,
                            "level": str(level),
                            "analyte": analyte,
                            "n_events": len(group),
                            "n_systems": group["ICRPWSID"].nunique(),
                            "reported_present": int(x.notna().sum()),
                            "numeric_nonnegative": int(x.ge(0).sum()),
                            "minus999_below_mrl": int(x.eq(-999).sum()),
                            "numeric_gt60": int(x.gt(60).sum()),
                        }
                    )
    availability = pd.DataFrame(availability_rows)
    save_table(availability, paths.results_dir / "icr_haa5_haa9_availability")

    grids, support, curves, targets = icr_capture_analysis(joined, reps, seed)
    save_table(grids, paths.processed_dir / "icr_temporal_system_grids")
    save_table(support, paths.results_dir / "icr_temporal_support")
    save_table(curves, paths.results_dir / "icr_temporal_capture_curve")
    save_table(targets, paths.results_dir / "icr_capture_targets")
    factorial_systems, factorial_support, factorial_cells, factorial_contrasts, factorial_repairs = icr_matched_factorial(
        joined, reps, seed
    )
    save_table(factorial_systems, paths.processed_dir / "icr_matched_factorial_systems")
    save_table(factorial_support, paths.results_dir / "icr_matched_factorial_support")
    save_table(factorial_cells, paths.results_dir / "icr_matched_factorial_cells")
    save_table(factorial_contrasts, paths.results_dir / "icr_matched_factorial_contrasts")
    save_table(factorial_repairs, paths.results_dir / "icr_matched_factorial_repair_fractions")
    summary = {
        "table_rows": {name: int(len(frame)) for name, frame in tables.items()},
        "dbp_events_with_sample_match": int(joined["_sample_join"].eq("both").sum()),
        "dbp_events_without_sample_match": int(joined["_sample_join"].ne("both").sum()),
        "primary_event_types": list(PRIMARY_ICR_TYPES),
        "all_ds_sensitivity_event_types": list(ALL_DS_ICR_TYPES),
        "primary_dbp_events": int(joined["PRIMARY_DISTRIBUTION"].sum()),
        "all_ds_dbp_events": int(joined["ALL_DS_SENSITIVITY"].sum()),
        "identity_audit": identities.to_dict(orient="records"),
        "temporal_support": support.to_dict(orient="records"),
        "capture_targets": targets.to_dict(orient="records"),
        "matched_factorial_support": factorial_support.to_dict(orient="records"),
        "matched_factorial_cells": factorial_cells[
            factorial_cells["conditioning"].eq("HAA9_M1_reference_positive")
        ].to_dict(orient="records"),
        "matched_factorial_contrasts": factorial_contrasts.to_dict(orient="records"),
        "matched_factorial_repairs": factorial_repairs.to_dict(orient="records"),
    }
    return summary, [
        sentinel,
        event_audit_df,
        identities,
        availability,
        support,
        curves,
        targets,
        factorial_support,
        factorial_cells,
        factorial_contrasts,
        factorial_repairs,
    ]


def write_methods_results(paths: Paths, summary: dict, reps: int, seed: int) -> None:
    syr4 = summary["syr4"]
    icr = summary["icr"]
    corr = syr4["correlations"][0] if syr4["correlations"] else {}
    identity = {(r["scope"], r["total"]): r for r in icr["identity_audit"]}
    h5 = identity.get(("all_dbp", "HAA5"), {})
    h9 = identity.get(("all_dbp", "HAA9"), {})
    support_lines = []
    for row in icr["temporal_support"]:
        if row["scope"] == "primary_distribution":
            support_lines.append(
                f"- {row['analyte']} {row['time_scale']}: {row['full_six_quarter_systems']} systems had numeric support in all six quarters; "
                f"{row['full_window_positive_systems']} had at least one value >60 µg/L."
            )
    text = f"""# Cross-archive validation: methods and results

## Scope and non-interpretive guardrails

This analysis validates data structure, paired concentration patterns, and finite-grid sampling capture. It does **not** estimate legal compliance, individual exposure, health effects, or causal effects. A concentration equal to the stated threshold is classified as not above; the event definition is strictly `>60 µg/L` for HAA5 (and the non-regulatory HAA9 comparison) and `>80 µg/L` for TTHM.

## SYR4 pairing and censoring

HAA5 and TTHM were inner-joined on the exact composite key `PWSID + SAMPLE_ID`. Missing sample IDs were never matched. Within each analyte, duplicate keys were collapsed only when all analysis-critical fields agreed; conflicting keys were excluded. The retained primary cohort additionally required the HAA5 and TTHM collection dates to agree. Sampling-point database IDs were audited but not required because analyte-specific IDs can differ despite a matching sample key, date, and facility.

Detected values were used as point concentrations. Non-detects were represented as `[0, detection limit]`; positive limits reported in mg/L were multiplied by 1,000 and limits with a missing unit were assumed to be µg/L because all detected values in these analyte files were reported in µg/L. Zero, missing, negative, or unrecognized-unit limits have an unknown upper bound. A result was definitely above when its lower bound exceeded the threshold, definitely below when its upper bound was at or below the threshold, and otherwise indeterminate. Midpoint (`DL/2`), lower-bound, and upper-bound summaries are provided as sensitivity analyses, not replacements for the interval classification.

The two source files contained {syr4['inputs'][0]['raw_rows']:,} HAA5 and {syr4['inputs'][1]['raw_rows']:,} TTHM rows. Conflict-resolved exact-key pairing produced {syr4['conflict_resolved_exact_pairs']:,} pairs; {syr4['same_date_primary_pairs']:,} same-date pairs from {syr4['primary_systems']:,} systems entered primary analyses. There were {syr4['both_detected_pairs']:,} pairs with both analytes detected and {syr4['jointly_classifiable_pairs']:,} pairs with both threshold statuses classifiable. Threshold-status agreement was {syr4['threshold_agreement']:.3f} (95% system-cluster bootstrap CI {syr4['threshold_agreement_ci'][0]:.3f}–{syr4['threshold_agreement_ci'][1]:.3f}). Detected-pair Spearman rho was {corr.get('spearman_rho', float('nan')):.3f} (95% fixed-rank system-cluster bootstrap CI {corr.get('cluster_bootstrap_ci_low', float('nan')):.3f}–{corr.get('cluster_bootstrap_ci_high', float('nan')):.3f}). Source-water, population-size, system-type, and seasonal-quarter summaries are descriptive and retain their event and system denominators.

For the SYR4 finite-grid analysis, a system-year required at least three distinct sampled quarters and complete interval classification in every included quarter. A quarter was positive if any same-date exact pair was definitely above the analyte threshold. Within each 3- or 4-quarter grid and conditional on at least one positive quarter, capture for sampling `k` quarters without replacement was calculated exactly as `1 - C(Q-m,k)/C(Q,k)`.

## ICR extraction, identities, and event cohorts

The Access tables `TUXDBP`, `TUXSAMPLE`, `TUXUTMON`, `TUXWQP`, and `TUXDISFRES` were extracted without value substitution and written to CSV and Parquet. The ICR user guide's general rule GR012000140 defines `-999` in analytical results as below the minimum reporting level (MRL); it is never interpreted as a negative concentration. The WQP rules add a special averaged-sample convention for TOC, TOX, and UV-254: `-999` is retained when at least one initial/duplicate result is below MRL and the prescribed MRL-substitution average equals the MRL. For DBP totals, below-MRL constituents are replaced by zero during summation, while a total is coded `-999` when every constituent is below MRL (DBP011000071 for HAA5 and DBP011000131 for HAA9). Because analyte-specific MRLs are absent from these extracts, `-999` totals are excluded from numeric threshold support rather than silently set to zero.

The primary ICR distribution cohort is exactly `{', '.join(PRIMARY_ICR_TYPES)}`. The prespecified all-distribution-system sensitivity is exactly `{', '.join(ALL_DS_ICR_TYPES)}`. The primary cohort contains {icr['primary_dbp_events']:,} DBP events and the all-DS sensitivity contains {icr['all_ds_dbp_events']:,}.

The HAA5 identity was evaluable for {h5.get('component_identity_supported', 0):,} events and matched the five-component rule in {h5.get('identity_matches', 0):,}, with {h5.get('identity_mismatches', 0):,} mismatches. HAA9 was evaluable for {h9.get('component_identity_supported', 0):,} events and matched the nine-component rule in {h9.get('identity_matches', 0):,}, with {h9.get('identity_mismatches', 0):,} mismatches. HAA9 availability is reported explicitly by event type, sampling period, quarter, and cohort; it is not treated as if universally measured.

## ICR temporal capture

For each system and sampling period, the maximum nonnegative numeric analyte result within the specified event scope represented that period. Quarters were similarly represented by their maximum supported period result. A full-window system had numeric support in each of ICR quarters 1–6; missing and `-999` results did not count as support. Capture curves are conditional on a full-window system having at least one observed value strictly above 60 µg/L. For each system's finite supported grid, random-subset capture is the exact hypergeometric complement `1 - C(N-M,k)/C(N,k)`. Curves stop at the minimum supported grid size shared by the cohort. D50/D90 are reported only if the empirical curve reaches 0.50/0.90 on that grid; otherwise the output says `not reached on supported grid`.

{chr(10).join(support_lines)}

## Matched HAA5 × HAA9 factorial addendum

The compact ICR factorial uses only primary-distribution DBP events with paired nonnegative HAA5 and HAA9 at the same `EVENT_ID`. Within each system, events are collapsed to the maximum at each observed sampling period (monthly grid) or quarter (quarterly grid); both panels therefore share an identical finite grid. `M0` selects one grid period uniformly, evaluated by exact enumeration, and `M1` uses the complete observed grid. `p00`, `p10`, `p01`, and `p11` respectively denote HAA5/M0, HAA9/M0, HAA5/M1, and HAA9/M1 signals at the common research benchmark B=60 µg/L. The primary addendum is the available-period monthly cohort with at least two paired periods; available-quarter results and a strict paired six-quarter sensitivity are also reported. Conditional results use systems positive under HAA9/M1, so `p11=1` by definition and is not a claim of perfect physical or regulatory detection. The five exhaustive fractions are already detected, panel-only repair, design-only repair, either-change repair, and both-changes required; their sum is audited on the common reference-positive denominator. Interaction is `p11-p10-p01+p00`, a detection-scale interaction rather than biological synergy.

## Uncertainty and holdout

All reported bootstrap intervals use {reps:,} fixed-seed (`{seed}`) nonparametric whole-system resamples. For observation-level Spearman rho, global midranks were fixed before resampling and cluster sufficient statistics were resampled; the system-median analysis provides a complementary system-level check. SYR4's deterministic 20% system holdout was assigned by a stable cryptographic hash, preventing samples from the same system from crossing splits. An ICR holdout was not imposed because the full-window positive cohorts are small; exact finite-grid probabilities and explicit support counts are more informative there.

## Reproducibility

All cohort rules, thresholds, event-type sets, censoring decisions, bootstrap settings, and output checksums are recorded in `analysis_metadata.json`, `validation_summary.json`, and `output_manifest.csv`. Results are descriptive validation of these public monitoring data and should not be generalized beyond the observed systems, periods, locations, and reporting conventions.
"""
    (paths.results_dir / "methods_results.md").write_text(text, encoding="utf-8")


def build_manifest(paths: Paths) -> pd.DataFrame:
    rows = []
    for directory in (paths.processed_dir, paths.results_dir):
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.name.startswith("output_manifest"):
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            rows.append(
                {
                    "relative_path": path.relative_to(paths.root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    if args.bootstrap_reps < 0:
        raise ValueError("--bootstrap-reps must be nonnegative")
    paths = make_paths(args.root)
    required = (
        "SYR4_HAA5.txt",
        "SYR4_TTHM.txt",
        "SYR4_Water_system_table.txt",
        "DBPICR_Aux1.accdb",
    )
    missing = [name for name in required if not (paths.input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
    print("Running SYR4 validation...", flush=True)
    syr4_summary, _ = analyze_syr4(paths, args.bootstrap_reps, args.seed)
    print("Running ICR extraction and validation...", flush=True)
    icr_summary, _ = analyze_icr(paths, args.bootstrap_reps, args.seed)
    metadata = {
        "analysis_name": "DBP cross-archive historical validation",
        "seed": args.seed,
        "bootstrap_reps": args.bootstrap_reps,
        "software": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "duckdb": duckdb.__version__,
            "pyodbc": pyodbc.version,
        },
        "thresholds_ug_l": {"SYR4_HAA5": 60.0, "SYR4_TTHM": 80.0, "ICR_HAA5": 60.0, "ICR_HAA9_comparison": 60.0},
        "threshold_operator": "strictly greater than",
        "syr4_pair_key": ["PWSID", "SAMPLE_ID"],
        "syr4_primary_additional_rule": "same HAA5 and TTHM sample collection date",
        "syr4_duplicate_rule": "collapse only identical analysis-critical signatures; exclude conflicting keys",
        "syr4_censoring_rule": "non-detect interval [0, positive reported DL]; unknown upper bound if DL missing/zero/negative/unrecognized unit",
        "icr_primary_event_types": list(PRIMARY_ICR_TYPES),
        "icr_all_ds_sensitivity_event_types": list(ALL_DS_ICR_TYPES),
        "icr_minus999_rule": "below MRL analytical sentinel; never a numeric concentration; excluded from numeric support",
        "icr_minus999_semantics": {
            "general_rule_GR012000140": "-999 is entered for an analytical result below the minimum reporting level",
            "TUXDBP_components": "-999 denotes a below-MRL component; zero is used only inside the documented total calculation",
            "TUXDBP_HAA5_rule_DBP011000071": "reported total is -999 when all five HAA5 components are below MRL",
            "TUXDBP_HAA9_rule_DBP011000131": "reported total is -999 when all nine HAA9 components are below MRL",
            "TUXWQP_special_average_rules": "for TOC, TOX, and UV-254, -999 is used under the guide's initial/duplicate MRL-substitution averaging convention when the average equals the MRL",
            "analysis_handling": "never treated as -999 or silently as zero; excluded from numeric summaries and threshold-support denominators unless a documented total is recomputed for identity QA",
        },
        "icr_full_window_rule": "numeric support in each SAMP_QTR 1 through 6",
        "icr_factorial_rule": "same EVENT_ID paired nonnegative HAA5/HAA9; M0 uniform exact one-period enumeration; M1 complete finite paired grid; condition on HAA9/M1 >60",
        "capture_formula": "1 - combination(N-M,k)/combination(N,k), without replacement",
        "bootstrap_unit": "PWSID / ICRPWSID system cluster",
        "holdout_rule": "20% of SYR4 systems by stable blake2b hash modulo 5",
    }
    summary = {"syr4": syr4_summary, "icr": icr_summary, "metadata": metadata}
    write_json(metadata, paths.results_dir / "analysis_metadata.json")
    write_json(summary, paths.results_dir / "validation_summary.json")
    write_methods_results(paths, summary, args.bootstrap_reps, args.seed)
    manifest = build_manifest(paths)
    save_table(manifest, paths.results_dir / "output_manifest")
    write_json({"files": manifest.to_dict(orient="records")}, paths.results_dir / "output_manifest.json")
    print(json.dumps(_sanitize_json({"status": "complete", "summary": summary}), default=_json_default, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
