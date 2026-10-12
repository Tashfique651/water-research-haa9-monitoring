#!/usr/bin/env python3
"""California HAA5/TTHM validation for the Water Research package.

This script reads the California Division of Drinking Water SDWIS4 flat-file
archive, restricts the primary analysis to 2023-2025 routine (RT)
distribution-system (DS) samples reported in micrograms per litre, and creates
two deliberately distinct analysis populations:

1. all valid analyte-specific records, used for annual and calendar-season
   summaries; and
2. exact PWS + sampling point + date + clock-time HAA5/TTHM pairs, used for
   concordance analyses.

The archive reports censored total-analyte results as a blank result with a
"Less Than Reporting Level" flag and a numerical reporting level. The primary
continuous analysis substitutes one-half of that limit. Zero and full-limit
substitutions are reported as sensitivity analyses. Duplicate event-analyte
keys are collapsed using the median record-level value, with minimum,
maximum, and conflict-exclusion sensitivities retained.

Nothing in this analysis measures HAA9, establishes statistical independence
from other public archives, determines regulatory compliance, or supports a
causal interpretation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
from scipy import stats


ANALYTES = {
    "2456": {"short_name": "HAA5", "reference_level_ug_l": 60.0},
    "2950": {"short_name": "TTHM", "reference_level_ug_l": 80.0},
}
PRIMARY_YEARS = (2023, 2024, 2025)
SEASON_ORDER = ("Winter", "Spring", "Summer", "Autumn")
SEED = 20260914


@dataclass(frozen=True)
class RunConfig:
    input_path: str
    output_dir: str
    bootstrap_replicates: int
    seed: int
    primary_years: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root
        / "data"
        / "raw"
        / "additional"
        / "california"
        / "sdwis4"
        / "SDWIS4.tab",
        help="Path to the extracted California SDWIS4.tab archive.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "wr_results" / "california",
        help="Directory for tidy validation outputs.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=2_000,
        help="Number of PWS-cluster bootstrap replicates (default: 2000).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def raw_line_shape_audit(path: Path, expected_columns: int) -> tuple[dict, pd.DataFrame]:
    """Check tab-field counts without decoding the 2.2 GB archive.

    DuckDB's relaxed parser is needed because a very small number of unrelated
    analyte rows contain an extra tab. This independent byte-level pass proves
    whether any HAA5 or TTHM row is among them.
    """
    analyte_index = 19  # zero-based position of "Analyte Code" in SDWIS4.tab
    target_counts = {code.encode("ascii"): 0 for code in ANALYTES}
    malformed_target_counts = {code.encode("ascii"): 0 for code in ANALYTES}
    malformed_examples: list[dict] = []
    total_rows = 0
    malformed_rows = 0
    field_count_frequency: dict[int, int] = {}
    with path.open("rb") as handle:
        handle.readline()
        for line_number, line in enumerate(handle, start=2):
            total_rows += 1
            fields = line.rstrip(b"\r\n").split(b"\t")
            field_count = len(fields)
            field_count_frequency[field_count] = field_count_frequency.get(field_count, 0) + 1
            malformed = field_count != expected_columns
            if malformed:
                malformed_rows += 1
                if len(malformed_examples) < 20:
                    code = (
                        fields[analyte_index].strip().decode("ascii", errors="replace")
                        if len(fields) > analyte_index
                        else "<missing>"
                    )
                    malformed_examples.append(
                        {
                            "line_number": line_number,
                            "field_count": field_count,
                            "apparent_analyte_code": code,
                        }
                    )
            if len(fields) > analyte_index:
                code_bytes = fields[analyte_index].strip()
                if code_bytes in target_counts:
                    target_counts[code_bytes] += 1
                    if malformed:
                        malformed_target_counts[code_bytes] += 1
    summary = {
        "data_rows": total_rows,
        "expected_columns": expected_columns,
        "malformed_rows_any_analyte": malformed_rows,
        "target_row_counts": {key.decode(): value for key, value in target_counts.items()},
        "malformed_target_row_counts": {
            key.decode(): value for key, value in malformed_target_counts.items()
        },
    }
    detail_rows = [
        {
            "record_type": "field_count_frequency",
            "line_number": np.nan,
            "field_count": field_count,
            "apparent_analyte_code": "",
            "n_rows": n_rows,
        }
        for field_count, n_rows in sorted(field_count_frequency.items())
    ]
    detail_rows.extend(
        {
            "record_type": "malformed_row_example",
            "line_number": row["line_number"],
            "field_count": row["field_count"],
            "apparent_analyte_code": row["apparent_analyte_code"],
            "n_rows": np.nan,
        }
        for row in malformed_examples
    )
    return summary, pd.DataFrame(detail_rows)


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    return value


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_table(df: pd.DataFrame, stem: Path, *, parquet: bool = True) -> None:
    """Write a stable CSV and, when requested, an equivalent Parquet table."""
    out = df.copy()
    out.to_csv(stem.with_suffix(".csv"), index=False, lineterminator="\n")
    if parquet:
        table = pa.Table.from_pandas(out, preserve_index=False)
        pq.write_table(table, stem.with_suffix(".parquet"), compression="zstd")


def write_output_manifest(output_dir: Path) -> pd.DataFrame:
    """Hash every analysis output except the manifest itself."""
    manifest_path = output_dir / "ca_output_manifest.csv"
    rows = []
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path == manifest_path:
            continue
        row_count = None
        if path.suffix.lower() == ".csv":
            with path.open("rb") as handle:
                row_count = max(sum(1 for _ in handle) - 1, 0)
        elif path.suffix.lower() == ".parquet":
            row_count = pq.ParquetFile(path).metadata.num_rows
        rows.append(
            {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "table_rows_if_applicable": row_count,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    return manifest


def season_from_month(month: pd.Series) -> pd.Categorical:
    labels = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
        ["Winter", "Spring", "Summer"],
        default="Autumn",
    )
    return pd.Categorical(labels, categories=SEASON_ORDER, ordered=True)


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if valid.sum() < 3 or np.unique(x_arr[valid]).size < 2 or np.unique(y_arr[valid]).size < 2:
        return float("nan")
    return float(stats.spearmanr(x_arr[valid], y_arr[valid]).statistic)


def safe_kendall(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if valid.sum() < 3 or np.unique(x_arr[valid]).size < 2 or np.unique(y_arr[valid]).size < 2:
        return float("nan")
    return float(stats.kendalltau(x_arr[valid], y_arr[valid], variant="b").statistic)


def phi_coefficient(a_high: np.ndarray, b_high: np.ndarray) -> float:
    a = np.asarray(a_high, dtype=bool)
    b = np.asarray(b_high, dtype=bool)
    n11 = float(np.sum(a & b))
    n10 = float(np.sum(a & ~b))
    n01 = float(np.sum(~a & b))
    n00 = float(np.sum(~a & ~b))
    denom = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return float((n11 * n00 - n10 * n01) / denom) if denom > 0 else float("nan")


def binary_agreement_metrics(
    haa5: np.ndarray,
    tthm: np.ndarray,
    haa5_threshold: float,
    tthm_threshold: float,
) -> dict:
    h = np.asarray(haa5, dtype=float) >= haa5_threshold
    t = np.asarray(tthm, dtype=float) >= tthm_threshold
    n11 = int(np.sum(h & t))
    n10 = int(np.sum(h & ~t))
    n01 = int(np.sum(~h & t))
    n00 = int(np.sum(~h & ~t))
    return binary_agreement_from_counts(n11, n10, n01, n00)


def binary_agreement_from_counts(n11: int, n10: int, n01: int, n00: int) -> dict:
    n = int(n11 + n10 + n01 + n00)
    union = n11 + n10 + n01
    denom = math.sqrt(
        (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    )
    return {
        "n_pairs": n,
        "n_both_high": n11,
        "n_haa5_only_high": n10,
        "n_tthm_only_high": n01,
        "n_neither_high": n00,
        "fraction_haa5_high": (n11 + n10) / n if n else np.nan,
        "fraction_tthm_high": (n11 + n01) / n if n else np.nan,
        "overall_agreement": (n11 + n00) / n if n else np.nan,
        "high_set_jaccard": n11 / union if union else np.nan,
        "phi": float((n11 * n00 - n10 * n01) / denom) if denom > 0 else np.nan,
        "p_tthm_high_given_haa5_high": n11 / (n11 + n10) if (n11 + n10) else np.nan,
        "p_haa5_high_given_tthm_high": n11 / (n11 + n01) if (n11 + n01) else np.nan,
    }


def extract_archive(con: duckdb.DuckDBPyConnection, source: Path) -> pd.DataFrame:
    """Scan the archive once and retain only HAA5 and TTHM rows in memory."""
    query = r"""
        SELECT
            row_number() OVER () AS source_row_number,
            trim("Regulating Agency") AS regulating_agency,
            trim("Water System Number") AS pws_id,
            trim("Water System Name") AS water_system_name,
            trim("System Status") AS system_status,
            trim("Federal Water System Classification") AS federal_system_class,
            trim("State Water System Classification") AS state_system_class,
            trim("Principal County Served") AS county,
            try_cast(trim("Population Served") AS BIGINT) AS population_served,
            try_cast(trim("Service Connections") AS BIGINT) AS service_connections,
            trim("PS Code") AS site_id,
            trim("Sampling Point Name") AS sampling_point_name,
            upper(trim("Facility Type")) AS facility_type,
            upper(trim("Facility Status")) AS facility_status,
            upper(trim("Sample Type")) AS sample_type,
            cast(try_strptime(trim("Sample Date"), '%m-%d-%Y') AS DATE) AS sample_date,
            CASE
                WHEN "Sample Time" IS NULL
                  OR trim("Sample Time") = ''
                  OR upper(trim("Sample Time")) = 'NULL'
                THEN NULL
                ELSE strftime(try_strptime(trim("Sample Time"), '%H:%M:%S'), '%H:%M:%S')
            END AS sample_time,
            cast(try_strptime(trim("Analysis Date"), '%m-%d-%Y') AS DATE) AS analysis_date,
            trim("ELAP Cert#") AS elap_certificate,
            trim("Lab Name") AS laboratory,
            trim("Analyte Code") AS analyte_code,
            trim("Analyte Name") AS analyte_name,
            try_cast(trim("Result") AS DOUBLE) AS result_ug_l,
            upper(trim("Units of Measure")) AS units,
            CASE WHEN upper(trim(coalesce("Less Than Reporting Level", ''))) = 'Y'
                 THEN TRUE ELSE FALSE END AS is_censored,
            try_cast(trim("Reporting Level") AS DOUBLE) AS reporting_level_ug_l,
            try_cast(trim("DLR") AS DOUBLE) AS dlr_ug_l,
            try_cast(trim("MCL") AS DOUBLE) AS archive_mcl_ug_l,
            trim("Method") AS analytical_method
        FROM read_csv(
            ?, delim='\t', header=true, all_varchar=true, quote='',
            encoding='cp1252', strict_mode=false, null_padding=true,
            ignore_errors=false, parallel=true
        )
        WHERE trim("Analyte Code") IN ('2456', '2950')
    """
    return con.execute(query, [str(source)]).fetchdf()


def validate_and_prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return all primary-scope rows and the fully valid analysis records."""
    frame = raw.copy()
    frame["sample_date"] = pd.to_datetime(frame["sample_date"], errors="coerce")
    frame["analysis_date"] = pd.to_datetime(frame["analysis_date"], errors="coerce")
    frame["sample_year"] = frame["sample_date"].dt.year.astype("Int64")
    frame["sample_month"] = frame["sample_date"].dt.month.astype("Int64")
    frame["analyte"] = frame["analyte_code"].map(
        {code: meta["short_name"] for code, meta in ANALYTES.items()}
    )
    frame["censor_limit_ug_l"] = frame["reporting_level_ug_l"].combine_first(frame["dlr_ug_l"])

    primary_scope = frame.loc[
        frame["sample_year"].isin(PRIMARY_YEARS)
        & frame["sample_type"].eq("RT")
        & frame["facility_type"].eq("DS")
    ].copy()

    valid = (
        primary_scope["analyte_code"].isin(ANALYTES)
        & primary_scope["units"].eq("UG/L")
        & primary_scope["sample_date"].notna()
        & primary_scope["pws_id"].fillna("").str.len().gt(0)
        & primary_scope["site_id"].fillna("").str.len().gt(0)
        & (
            primary_scope["result_ug_l"].ge(0)
            | (
                primary_scope["is_censored"]
                & primary_scope["censor_limit_ug_l"].notna()
                & primary_scope["censor_limit_ug_l"].ge(0)
            )
        )
        & ~(
            primary_scope["result_ug_l"].notna()
            & primary_scope["is_censored"]
        )
    )
    records = primary_scope.loc[valid].copy()
    records["record_status"] = np.where(records["is_censored"], "censored", "quantified")
    records["value_lower_ug_l"] = np.where(
        records["is_censored"], 0.0, records["result_ug_l"]
    )
    records["value_midpoint_ug_l"] = np.where(
        records["is_censored"],
        records["censor_limit_ug_l"] / 2.0,
        records["result_ug_l"],
    )
    records["value_upper_ug_l"] = np.where(
        records["is_censored"], records["censor_limit_ug_l"], records["result_ug_l"]
    )
    records["season"] = season_from_month(records["sample_month"])
    records["reference_level_ug_l"] = records["analyte_code"].map(
        {code: meta["reference_level_ug_l"] for code, meta in ANALYTES.items()}
    )
    records["at_or_above_reference_level"] = (
        records["value_lower_ug_l"] >= records["reference_level_ug_l"]
    )
    records = records.sort_values(
        ["analyte", "pws_id", "site_id", "sample_date", "sample_time", "source_row_number"],
        kind="stable",
    ).reset_index(drop=True)
    return primary_scope, records


def collapse_event_duplicates(records: pd.DataFrame) -> pd.DataFrame:
    """Collapse replicate rows at an exact event-analyte key transparently."""
    data = records.copy()
    data["sample_time_key"] = data["sample_time"].fillna("<MISSING>")
    data["record_signature"] = (
        data["record_status"].astype(str)
        + "|"
        + data["value_lower_ug_l"].round(9).astype(str)
        + "|"
        + data["value_midpoint_ug_l"].round(9).astype(str)
        + "|"
        + data["value_upper_ug_l"].round(9).astype(str)
    )
    key = ["analyte_code", "analyte", "pws_id", "site_id", "sample_date", "sample_time_key"]
    grouped = data.groupby(key, dropna=False, observed=True, sort=True)

    events = grouped.agg(
        water_system_name=("water_system_name", "first"),
        county=("county", "first"),
        federal_system_class=("federal_system_class", "first"),
        population_served=("population_served", "max"),
        sampling_point_name=("sampling_point_name", "first"),
        sample_year=("sample_year", "first"),
        sample_month=("sample_month", "first"),
        season=("season", "first"),
        n_source_records=("source_row_number", "size"),
        n_distinct_reported_values=("record_signature", "nunique"),
        any_quantified=("is_censored", lambda x: bool((~x.astype(bool)).any())),
        any_censored=("is_censored", "any"),
        censor_limit_median_ug_l=("censor_limit_ug_l", "median"),
        value_lower_ug_l=("value_lower_ug_l", "median"),
        value_midpoint_ug_l=("value_midpoint_ug_l", "median"),
        value_upper_ug_l=("value_upper_ug_l", "median"),
        duplicate_min_midpoint_ug_l=("value_midpoint_ug_l", "min"),
        duplicate_max_midpoint_ug_l=("value_midpoint_ug_l", "max"),
        reference_level_ug_l=("reference_level_ug_l", "first"),
    ).reset_index()

    events["sample_time"] = events["sample_time_key"].replace("<MISSING>", pd.NA)
    events = events.drop(columns="sample_time_key")
    events["duplicate_key"] = events["n_source_records"] > 1
    events["duplicate_conflict"] = events["n_distinct_reported_values"] > 1
    events["mixed_censor_status"] = events["any_quantified"] & events["any_censored"]
    events["duplicate_disposition"] = np.select(
        [events["duplicate_conflict"], events["duplicate_key"]],
        ["discordant replicate; median used", "exact duplicate; one event retained"],
        default="single source record",
    )
    events["at_or_above_reference_level"] = (
        events["value_lower_ug_l"] >= events["reference_level_ug_l"]
    )
    return events.sort_values(
        ["analyte", "pws_id", "site_id", "sample_date", "sample_time"], kind="stable"
    ).reset_index(drop=True)


def pair_exact_events(events: pd.DataFrame) -> pd.DataFrame:
    strict = events.loc[events["sample_time"].notna()].copy()
    id_cols = ["pws_id", "site_id", "sample_date", "sample_time"]
    value_cols = [
        "value_lower_ug_l",
        "value_midpoint_ug_l",
        "value_upper_ug_l",
        "duplicate_min_midpoint_ug_l",
        "duplicate_max_midpoint_ug_l",
        "any_quantified",
        "any_censored",
        "duplicate_key",
        "duplicate_conflict",
        "n_source_records",
    ]
    wide = strict.pivot(index=id_cols, columns="analyte", values=value_cols)
    if not {"HAA5", "TTHM"}.issubset(set(wide.columns.get_level_values(1))):
        raise RuntimeError("No exact HAA5/TTHM event pairs were found.")
    wide.columns = [f"{analyte.lower()}_{measure}" for measure, analyte in wide.columns]
    wide = wide.reset_index()
    paired = wide.dropna(
        subset=["haa5_value_midpoint_ug_l", "tthm_value_midpoint_ug_l"]
    ).copy()

    metadata = (
        strict.sort_values(id_cols + ["analyte"], kind="stable")
        .drop_duplicates(id_cols)
        [id_cols + ["water_system_name", "county", "federal_system_class", "population_served"]]
    )
    paired = paired.merge(metadata, on=id_cols, how="left", validate="one_to_one")
    paired["sample_year"] = paired["sample_date"].dt.year.astype(int)
    paired["sample_month"] = paired["sample_date"].dt.month.astype(int)
    paired["season"] = season_from_month(paired["sample_month"])
    paired["either_duplicate_conflict"] = (
        paired["haa5_duplicate_conflict"].astype(bool)
        | paired["tthm_duplicate_conflict"].astype(bool)
    )
    paired["haa5_at_or_above_60"] = paired["haa5_value_lower_ug_l"] >= 60.0
    paired["tthm_at_or_above_80"] = paired["tthm_value_lower_ug_l"] >= 80.0
    return paired.sort_values(id_cols, kind="stable").reset_index(drop=True)


def pair_date_only(events: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity match that ignores clock time after within-date aggregation."""
    key = ["analyte", "pws_id", "site_id", "sample_date"]
    date_events = (
        events.groupby(key, observed=True, sort=True, dropna=False)
        .agg(
            value_midpoint_ug_l=("value_midpoint_ug_l", "median"),
            value_lower_ug_l=("value_lower_ug_l", "median"),
            value_upper_ug_l=("value_upper_ug_l", "median"),
            n_time_resolved_events=("value_midpoint_ug_l", "size"),
        )
        .reset_index()
    )
    wide = date_events.pivot(
        index=["pws_id", "site_id", "sample_date"],
        columns="analyte",
        values=[
            "value_midpoint_ug_l",
            "value_lower_ug_l",
            "value_upper_ug_l",
            "n_time_resolved_events",
        ],
    )
    wide.columns = [f"{analyte.lower()}_{measure}" for measure, analyte in wide.columns]
    wide = wide.reset_index()
    return wide.dropna(
        subset=["haa5_value_midpoint_ug_l", "tthm_value_midpoint_ug_l"]
    ).reset_index(drop=True)


def descriptive_group(group: pd.DataFrame) -> pd.Series:
    x = group["value_midpoint_ug_l"].to_numpy(float)
    lower = group["value_lower_ug_l"].to_numpy(float)
    upper = group["value_upper_ug_l"].to_numpy(float)
    threshold = float(group["reference_level_ug_l"].iloc[0])
    return pd.Series(
        {
            "n_events": len(group),
            "n_source_records": int(group["n_source_records"].sum()),
            "n_pws": group["pws_id"].nunique(),
            "n_sites": group[["pws_id", "site_id"]].drop_duplicates().shape[0],
            "n_censored_events": int((~group["any_quantified"].astype(bool)).sum()),
            "censored_fraction": float((~group["any_quantified"].astype(bool)).mean()),
            "n_duplicate_keys": int(group["duplicate_key"].sum()),
            "n_duplicate_conflicts": int(group["duplicate_conflict"].sum()),
            "mean_lower_ug_l": float(np.mean(lower)),
            "mean_midpoint_ug_l": float(np.mean(x)),
            "mean_upper_ug_l": float(np.mean(upper)),
            "median_midpoint_ug_l": float(np.median(x)),
            "p25_midpoint_ug_l": float(np.quantile(x, 0.25)),
            "p75_midpoint_ug_l": float(np.quantile(x, 0.75)),
            "p90_midpoint_ug_l": float(np.quantile(x, 0.90)),
            "p95_midpoint_ug_l": float(np.quantile(x, 0.95)),
            "maximum_midpoint_ug_l": float(np.max(x)),
            "reference_level_ug_l": threshold,
            "n_at_or_above_reference_level": int(np.sum(lower >= threshold)),
            "fraction_at_or_above_reference_level": float(np.mean(lower >= threshold)),
        }
    )


def make_temporal_summaries(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    annual = (
        events.groupby(["analyte", "sample_year"], observed=True, sort=True)
        .apply(descriptive_group, include_groups=False)
        .reset_index()
    )
    seasonal = (
        events.groupby(["analyte", "season"], observed=True, sort=True)
        .apply(descriptive_group, include_groups=False)
        .reset_index()
    )
    monthly = (
        events.groupby(["analyte", "sample_month"], observed=True, sort=True)
        .apply(descriptive_group, include_groups=False)
        .reset_index()
    )
    overall = (
        events.groupby(["analyte"], observed=True, sort=True)
        .apply(descriptive_group, include_groups=False)
        .reset_index()
    )

    system_year = (
        events.groupby(["analyte", "pws_id", "sample_year"], observed=True, sort=True)
        .agg(
            n_events=("value_midpoint_ug_l", "size"),
            n_sites=("site_id", "nunique"),
            censored_fraction=("any_quantified", lambda x: float((~x.astype(bool)).mean())),
            median_midpoint_ug_l=("value_midpoint_ug_l", "median"),
            maximum_midpoint_ug_l=("value_midpoint_ug_l", "max"),
            maximum_lower_ug_l=("value_lower_ug_l", "max"),
            maximum_upper_ug_l=("value_upper_ug_l", "max"),
            any_at_or_above_reference_level=("at_or_above_reference_level", "max"),
        )
        .reset_index()
    )
    return {
        "overall": overall,
        "annual": annual,
        "seasonal": seasonal,
        "monthly": monthly,
        "system_year": system_year,
    }


def paired_summary_row(
    label: str,
    frame: pd.DataFrame,
    h_col: str,
    t_col: str,
) -> dict:
    valid = frame[[h_col, t_col, "pws_id"]].dropna()
    return {
        "analysis": label,
        "n_pairs": len(valid),
        "n_pws": valid["pws_id"].nunique(),
        "spearman_rho": safe_spearman(valid[h_col], valid[t_col]),
        "kendall_tau_b": safe_kendall(valid[h_col], valid[t_col]),
    }


def bootstrap_exact_pairs(
    paired: pd.DataFrame,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """PWS-cluster bootstrap for detected-pair rho and full-pair thresholds.

    Detected-pair ranks are recomputed after each cluster resample. Threshold
    agreement uses every jointly classifiable pair because all group-total
    reporting limits in this cohort fall below the 60/80 reference levels.
    """
    detected = paired.loc[
        paired["haa5_any_quantified"].astype(bool)
        & paired["tthm_any_quantified"].astype(bool)
    ].reset_index(drop=True)
    detected_pws_ids = detected["pws_id"].drop_duplicates().sort_values().to_numpy()
    detected_group_indices = {
        pws: indices.to_numpy(dtype=int)
        for pws, indices in detected.groupby("pws_id", sort=False).groups.items()
    }
    h_detected = detected["haa5_value_midpoint_ug_l"].to_numpy(float)
    t_detected = detected["tthm_value_midpoint_ug_l"].to_numpy(float)

    threshold_flags = paired[["pws_id"]].copy()
    threshold_flags["n11"] = (
        (paired["haa5_value_midpoint_ug_l"] >= 60.0)
        & (paired["tthm_value_midpoint_ug_l"] >= 80.0)
    ).astype(int)
    threshold_flags["n10"] = (
        (paired["haa5_value_midpoint_ug_l"] >= 60.0)
        & (paired["tthm_value_midpoint_ug_l"] < 80.0)
    ).astype(int)
    threshold_flags["n01"] = (
        (paired["haa5_value_midpoint_ug_l"] < 60.0)
        & (paired["tthm_value_midpoint_ug_l"] >= 80.0)
    ).astype(int)
    threshold_flags["n00"] = (
        (paired["haa5_value_midpoint_ug_l"] < 60.0)
        & (paired["tthm_value_midpoint_ug_l"] < 80.0)
    ).astype(int)
    threshold_system_counts = (
        threshold_flags.groupby("pws_id", sort=True)[["n11", "n10", "n01", "n00"]]
        .sum()
        .to_numpy(dtype=int)
    )
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for replicate in range(replicates):
        rho_draw = rng.choice(
            detected_pws_ids, size=len(detected_pws_ids), replace=True
        )
        rho_idx = np.concatenate([detected_group_indices[pws] for pws in rho_draw])
        threshold_draw = rng.integers(
            0, len(threshold_system_counts), size=len(threshold_system_counts)
        )
        n11, n10, n01, n00 = threshold_system_counts[threshold_draw].sum(axis=0)
        agreement = binary_agreement_from_counts(
            int(n11), int(n10), int(n01), int(n00)
        )
        rows.append(
            {
                "replicate": replicate + 1,
                "n_resampled_pws_draws_for_rho": len(rho_draw),
                "n_detected_pairs_in_rho_resample": len(rho_idx),
                "detected_pair_spearman_rho": safe_spearman(
                    h_detected[rho_idx], t_detected[rho_idx]
                ),
                "n_resampled_pws_draws_for_thresholds": len(threshold_draw),
                "n_all_pairs_in_threshold_resample": agreement["n_pairs"],
                "overall_agreement_60_80": agreement["overall_agreement"],
                "high_set_jaccard_60_80": agreement["high_set_jaccard"],
                "phi_60_80": agreement["phi"],
                "p_tthm80_given_haa5_60": agreement["p_tthm_high_given_haa5_high"],
                "p_haa5_60_given_tthm80": agreement["p_haa5_high_given_tthm_high"],
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(samples: pd.Series) -> tuple[float, float]:
    x = pd.to_numeric(samples, errors="coerce").dropna().to_numpy(float)
    if x.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))


def concordance_outputs(
    paired: pd.DataFrame,
    date_only: pd.DataFrame,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    both_quantified = paired.loc[
        paired["haa5_any_quantified"].astype(bool)
        & paired["tthm_any_quantified"].astype(bool)
    ]
    sensitivity_rows = [
        paired_summary_row(
            "Exact time, both analytes quantified (primary)",
            both_quantified,
            "haa5_value_midpoint_ug_l",
            "tthm_value_midpoint_ug_l",
        ),
        paired_summary_row(
            "Exact time, half-limit substitution",
            paired,
            "haa5_value_midpoint_ug_l",
            "tthm_value_midpoint_ug_l",
        ),
        paired_summary_row(
            "Exact time, zero substitution",
            paired,
            "haa5_value_lower_ug_l",
            "tthm_value_lower_ug_l",
        ),
        paired_summary_row(
            "Exact time, full-limit substitution",
            paired,
            "haa5_value_upper_ug_l",
            "tthm_value_upper_ug_l",
        ),
        paired_summary_row(
            "Exact time, minimum duplicate value",
            paired,
            "haa5_duplicate_min_midpoint_ug_l",
            "tthm_duplicate_min_midpoint_ug_l",
        ),
        paired_summary_row(
            "Exact time, maximum duplicate value",
            paired,
            "haa5_duplicate_max_midpoint_ug_l",
            "tthm_duplicate_max_midpoint_ug_l",
        ),
    ]
    conflict_free = paired.loc[~paired["either_duplicate_conflict"]]
    sensitivity_rows.append(
        paired_summary_row(
            "Exact time, duplicate conflicts excluded",
            conflict_free,
            "haa5_value_midpoint_ug_l",
            "tthm_value_midpoint_ug_l",
        )
    )
    date_only_for_summary = date_only.copy()
    sensitivity_rows.append(
        paired_summary_row(
            "Same date, clock time ignored",
            date_only_for_summary,
            "haa5_value_midpoint_ug_l",
            "tthm_value_midpoint_ug_l",
        )
    )
    sensitivity = pd.DataFrame(sensitivity_rows)

    strata_rows = []
    strata: list[tuple[str, object, pd.DataFrame]] = [("Overall", "All", paired)]
    strata.extend(
        ("Year", int(year), group)
        for year, group in paired.groupby("sample_year", sort=True, observed=True)
    )
    strata.extend(
        ("Calendar season", str(season), group)
        for season, group in paired.groupby("season", sort=True, observed=True)
    )
    for dimension, stratum, group in strata:
        h = group["haa5_value_midpoint_ug_l"].to_numpy(float)
        t = group["tthm_value_midpoint_ug_l"].to_numpy(float)
        detected_group = group.loc[
            group["haa5_any_quantified"].astype(bool)
            & group["tthm_any_quantified"].astype(bool)
        ]
        hd = detected_group["haa5_value_midpoint_ug_l"].to_numpy(float)
        td = detected_group["tthm_value_midpoint_ug_l"].to_numpy(float)
        strata_rows.append(
            {
                "dimension": dimension,
                "stratum": stratum,
                "n_pairs": len(group),
                "n_pws": group["pws_id"].nunique(),
                "n_both_quantified_pairs": len(detected_group),
                "n_both_quantified_pws": detected_group["pws_id"].nunique(),
                "detected_pair_spearman_rho": safe_spearman(hd, td),
                "detected_pair_kendall_tau_b": safe_kendall(hd, td),
                "all_pair_midpoint_spearman_rho": safe_spearman(h, t),
                "all_pair_midpoint_kendall_tau_b": safe_kendall(h, t),
                "median_haa5_ug_l": float(np.median(h)),
                "median_tthm_ug_l": float(np.median(t)),
                "p95_haa5_ug_l": float(np.quantile(h, 0.95)),
                "p95_tthm_ug_l": float(np.quantile(t, 0.95)),
            }
        )
    strata_summary = pd.DataFrame(strata_rows)

    system_medians = (
        both_quantified.groupby("pws_id", sort=True)
        .agg(
            n_pairs=("haa5_value_midpoint_ug_l", "size"),
            median_haa5_ug_l=("haa5_value_midpoint_ug_l", "median"),
            median_tthm_ug_l=("tthm_value_midpoint_ug_l", "median"),
            p95_haa5_ug_l=("haa5_value_midpoint_ug_l", lambda x: float(np.quantile(x, 0.95))),
            p95_tthm_ug_l=("tthm_value_midpoint_ug_l", lambda x: float(np.quantile(x, 0.95))),
        )
        .reset_index()
    )
    system_rho = safe_spearman(system_medians["median_haa5_ug_l"], system_medians["median_tthm_ug_l"])

    bootstrap = bootstrap_exact_pairs(paired, bootstrap_replicates, seed)
    rho_low, rho_high = bootstrap_ci(bootstrap["detected_pair_spearman_rho"])
    sensitivity.loc[
        sensitivity["analysis"].eq("Exact time, both analytes quantified (primary)"),
        ["ci95_low", "ci95_high", "ci_method"],
    ] = [rho_low, rho_high, f"PWS-cluster percentile bootstrap, {bootstrap_replicates} replicates"]
    sensitivity.loc[len(sensitivity)] = {
        "analysis": "PWS-level medians",
        "n_pairs": len(system_medians),
        "n_pws": len(system_medians),
        "spearman_rho": system_rho,
        "kendall_tau_b": safe_kendall(
            system_medians["median_haa5_ug_l"], system_medians["median_tthm_ug_l"]
        ),
        "ci95_low": np.nan,
        "ci95_high": np.nan,
        "ci_method": "Point estimate; one row per PWS",
    }

    grid_rows = []
    h = paired["haa5_value_midpoint_ug_l"].to_numpy(float)
    t = paired["tthm_value_midpoint_ug_l"].to_numpy(float)
    for h_threshold in (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0):
        for t_threshold in (40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0):
            row = {
                "haa5_threshold_ug_l": h_threshold,
                "tthm_threshold_ug_l": t_threshold,
            }
            row.update(binary_agreement_metrics(h, t, h_threshold, t_threshold))
            grid_rows.append(row)
    threshold_grid = pd.DataFrame(grid_rows)

    standard = binary_agreement_metrics(h, t, 60.0, 80.0)
    standard_summary_rows = []
    bootstrap_columns = {
        "overall_agreement": "overall_agreement_60_80",
        "high_set_jaccard": "high_set_jaccard_60_80",
        "phi": "phi_60_80",
        "p_tthm_high_given_haa5_high": "p_tthm80_given_haa5_60",
        "p_haa5_high_given_tthm_high": "p_haa5_60_given_tthm80",
    }
    for metric, estimate in standard.items():
        if metric in bootstrap_columns:
            low, high = bootstrap_ci(bootstrap[bootstrap_columns[metric]])
            ci_method = f"PWS-cluster percentile bootstrap, {bootstrap_replicates} replicates"
        else:
            low, high, ci_method = np.nan, np.nan, "Not interval-estimated"
        standard_summary_rows.append(
            {
                "metric": metric,
                "estimate": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "ci_method": ci_method,
                "haa5_reference_level_ug_l": 60.0,
                "tthm_reference_level_ug_l": 80.0,
            }
        )
    standard_summary = pd.DataFrame(standard_summary_rows)
    return sensitivity, strata_summary, system_medians, threshold_grid, bootstrap, standard_summary


def quality_audit(
    raw: pd.DataFrame,
    primary_scope: pd.DataFrame,
    records: pd.DataFrame,
    events: pd.DataFrame,
    paired: pd.DataFrame,
    date_only: pd.DataFrame,
    line_shape: dict,
) -> pd.DataFrame:
    rows: list[dict] = []

    def add(section: str, metric: str, value, detail: str = "") -> None:
        rows.append({"section": section, "metric": metric, "value": value, "detail": detail})

    add("source", "retained HAA5/TTHM rows, all archive dates and sample contexts", len(raw))
    add("source", "archive data rows", line_shape["data_rows"])
    add("source", "rows with nonstandard tab-field count", line_shape["malformed_rows_any_analyte"])
    add(
        "source",
        "HAA5/TTHM rows with nonstandard tab-field count",
        sum(line_shape["malformed_target_row_counts"].values()),
        "Relaxed parsing affects no target-analyte row",
    )
    add("scope", "routine distribution-system rows in 2023-2025", len(primary_scope))
    add("validity", "valid primary analysis records", len(records))
    add("validity", "excluded primary-scope rows", len(primary_scope) - len(records))
    add("validity", "rows with invalid or missing sample date", int(primary_scope["sample_date"].isna().sum()))
    add("validity", "rows with missing normalized clock time", int(records["sample_time"].isna().sum()))
    add("validity", "rows with quantified result and censor flag simultaneously", int((records["result_ug_l"].notna() & records["is_censored"]).sum()))
    add("validity", "negative quantified results", int(records["result_ug_l"].lt(0).sum()))
    add("validity", "censored rows without numerical limit", int((records["is_censored"] & records["censor_limit_ug_l"].isna()).sum()))
    add("events", "unique event-analyte keys after duplicate collapse", len(events))
    add("events", "event-analyte keys with multiple source rows", int(events["duplicate_key"].sum()))
    add("events", "event-analyte keys with discordant values", int(events["duplicate_conflict"].sum()))
    add("events", "event-analyte keys mixing quantified and censored rows", int(events["mixed_censor_status"].sum()))
    add("pairing", "exact PWS-site-date-time HAA5/TTHM pairs", len(paired))
    add("pairing", "PWS represented in exact pairs", paired["pws_id"].nunique())
    add("pairing", "date-only paired keys", len(date_only), "Within-date medians; clock time ignored")
    add("pairing", "PWS represented in date-only pairs", date_only["pws_id"].nunique())
    for analyte, group in records.groupby("analyte", sort=True, observed=True):
        add("records by analyte", f"{analyte} valid records", len(group))
        add("records by analyte", f"{analyte} unique PWS", group["pws_id"].nunique())
        add("records by analyte", f"{analyte} censored records", int(group["is_censored"].sum()))
    return pd.DataFrame(rows)


def source_period_audit(raw: pd.DataFrame) -> pd.DataFrame:
    table = raw.copy()
    table["sample_date"] = pd.to_datetime(table["sample_date"], errors="coerce")
    table["sample_year"] = table["sample_date"].dt.year.astype("Int64")
    return (
        table.groupby(
            ["analyte_code", "analyte_name", "sample_year", "sample_type", "facility_type", "units"],
            dropna=False,
            observed=True,
            sort=True,
        )
        .size()
        .rename("n_rows")
        .reset_index()
    )


def duplicate_key_detail(events: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "analyte",
        "pws_id",
        "site_id",
        "sample_date",
        "sample_time",
        "n_source_records",
        "n_distinct_reported_values",
        "any_quantified",
        "any_censored",
        "mixed_censor_status",
        "value_midpoint_ug_l",
        "duplicate_min_midpoint_ug_l",
        "duplicate_max_midpoint_ug_l",
        "duplicate_conflict",
        "duplicate_disposition",
    ]
    return events.loc[events["duplicate_key"], columns].reset_index(drop=True)


def methods_markdown(
    config: RunConfig,
    source_hash: str,
    summary: dict,
) -> str:
    return f"""# California SDWIS4 HAA5/TTHM validation methods

## Scope and provenance

The source is the official California Division of Drinking Water SDWIS4 flat-file archive at `{config.input_path}`. Its SHA-256 digest is `{source_hash}`. The primary analysis includes calendar years 2023-2025 only, analyte codes 2456 (HAA5 total) and 2950 (TTHM total), routine samples (`RT`), distribution-system facilities (`DS`), and records reported in `UG/L`.

The archive uses Windows-1252 encoding. A byte-level line-shape audit found 14 rows with a nonstandard tab-field count among more than 7.1 million data rows, but none was an HAA5 or TTHM row. DuckDB therefore used its relaxed structural parser for the archive-wide scan; target-analyte counts were checked against the independent byte-level counts before analysis.

## Record validity and censoring

A record required a parseable sample date, nonempty public-water-system and sampling-point identifiers, and either a nonnegative quantified result or a less-than flag with a nonnegative reporting limit. Quantified values were used as reported. Censored values were represented as intervals from zero to the reporting limit. The primary continuous value used one-half the reporting limit; zero and full-limit substitutions were analyzed separately. These substitutions are numerical sensitivity conventions and are not new measurements.

## Duplicate event-analyte keys

An event-analyte key was defined by analyte, PWS, sampling point, sample date, and clock time. Multiple source rows at a key were collapsed with the median record-level value. The minimum and maximum values at each duplicated key were retained for sensitivity analysis, and analyses excluding discordant duplicate keys were also reported. The audit distinguishes exact duplicate rows from discordant replicate values.

## Temporal summaries

Annual, month-of-year, and calendar-season summaries use all valid analyte-specific events, not only paired samples. Seasons were Winter (December-February), Spring (March-May), Summer (June-August), and Autumn (September-November). Concentration summaries report lower-bound, half-limit, and upper-bound values. Counts at 60 micrograms per litre for HAA5 and 80 micrograms per litre for TTHM are descriptive concentration-level comparisons only. They are not determinations of compliance, which depends on regulatory calculations and context not reconstructed here.

## Paired concordance

The primary paired analysis required exact agreement on PWS, sampling point, sample date, and nonmissing clock time. It contained {summary['exact_pairs']:,} pairs across {summary['exact_pair_pws']:,} PWS. Primary rank concordance was estimated among pairs in which both totals were quantified, avoiding dependence on a censor-substitution convention; lower-bound, half-limit, and upper-bound analyses used all pairs as sensitivities. A same-date analysis that ignored clock time was retained only as a sensitivity analysis. Concordance was summarized with Spearman rho and Kendall tau-b. The 95% interval for the primary rho used {config.bootstrap_replicates:,} percentile bootstrap replicates resampled by PWS, with all observations from each sampled PWS carried together and ranks recomputed in every replicate. PWS-level median concentrations among detected pairs were also analyzed to reduce domination by frequently sampled systems.

## Threshold agreement

The paired values were dichotomized over a prespecified grid of HAA5 thresholds from 20 to 80 micrograms per litre and TTHM thresholds from 40 to 100 micrograms per litre. Results include the four cell counts, overall agreement, high-set Jaccard agreement, the phi coefficient, and both conditional high-concentration probabilities. Because high concentrations are uncommon, overall agreement is reported together with prevalence-sensitive measures and must not be interpreted alone.

## Interpretation limits

This California analysis validates only the reproducibility and external transport of HAA5/TTHM co-occurrence patterns in a separate state monitoring archive. It does not measure HAA9, does not prove that the archive is statistically independent of every other public data product, does not establish causality, and does not estimate health risk. Censor substitutions, unresolved provenance of discordant same-key records, unequal sampling intensity, and the inability to reconstruct regulatory locational running annual averages remain explicit limitations.
"""


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    source = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise FileNotFoundError(f"California SDWIS4 archive not found: {source}")
    if args.bootstrap_replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates.")

    required_columns = {
        "Regulating Agency",
        "Water System Number",
        "Water System Name",
        "Federal Water System Classification",
        "Population Served",
        "PS Code",
        "Sampling Point Name",
        "Facility Type",
        "Sample Type",
        "Sample Date",
        "Sample Time",
        "Analyte Code",
        "Analyte Name",
        "Result",
        "Units of Measure",
        "Less Than Reporting Level",
        "Reporting Level",
        "DLR",
        "MCL",
    }
    with source.open("r", encoding="utf-8-sig", errors="strict") as handle:
        header = handle.readline().rstrip("\r\n").split("\t")
    missing_columns = sorted(required_columns.difference(header))
    if missing_columns:
        raise ValueError(f"Required SDWIS4 columns are missing: {missing_columns}")

    source_hash = sha256_file(source)
    line_shape, line_shape_detail = raw_line_shape_audit(source, len(header))
    malformed_target_total = sum(line_shape["malformed_target_row_counts"].values())
    if malformed_target_total:
        raise ValueError(
            f"Found {malformed_target_total} target-analyte rows with a nonstandard field count; "
            "manual parser adjudication is required."
        )
    config = RunConfig(
        input_path=str(source),
        output_dir=str(output_dir),
        bootstrap_replicates=int(args.bootstrap_replicates),
        seed=int(args.seed),
        primary_years=PRIMARY_YEARS,
    )

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    raw = extract_archive(con, source)
    con.close()
    extracted_counts = raw["analyte_code"].value_counts().to_dict()
    if any(
        int(extracted_counts.get(code, 0)) != int(line_shape["target_row_counts"][code])
        for code in ANALYTES
    ):
        raise RuntimeError(
            "DuckDB target-analyte counts disagree with the independent byte-level scan."
        )
    raw["analyte"] = raw["analyte_code"].map(
        {code: meta["short_name"] for code, meta in ANALYTES.items()}
    )
    period_audit = source_period_audit(raw)
    primary_scope, records = validate_and_prepare(raw)
    events = collapse_event_duplicates(records)
    paired = pair_exact_events(events)
    date_only = pair_date_only(events)
    temporal = make_temporal_summaries(events)
    (
        concordance,
        concordance_by_time,
        system_medians,
        threshold_grid,
        bootstrap,
        standard_threshold,
    ) = concordance_outputs(
        paired,
        date_only,
        int(args.bootstrap_replicates),
        int(args.seed),
    )
    audit = quality_audit(raw, primary_scope, records, events, paired, date_only, line_shape)
    duplicate_detail = duplicate_key_detail(events)

    record_columns = [
        "source_row_number",
        "pws_id",
        "water_system_name",
        "county",
        "federal_system_class",
        "population_served",
        "service_connections",
        "site_id",
        "sampling_point_name",
        "sample_date",
        "sample_time",
        "sample_year",
        "sample_month",
        "season",
        "analyte_code",
        "analyte",
        "analyte_name",
        "record_status",
        "result_ug_l",
        "censor_limit_ug_l",
        "value_lower_ug_l",
        "value_midpoint_ug_l",
        "value_upper_ug_l",
        "reference_level_ug_l",
        "at_or_above_reference_level",
        "laboratory",
        "analytical_method",
    ]
    write_table(records[record_columns], output_dir / "ca_valid_analyte_records_2023_2025")
    write_table(events, output_dir / "ca_event_analyte_records_2023_2025")
    write_table(paired, output_dir / "ca_exact_haa5_tthm_pairs_2023_2025")
    write_table(date_only, output_dir / "ca_date_only_pair_sensitivity_2023_2025")
    write_table(temporal["overall"], output_dir / "ca_analyte_overall_summary", parquet=False)
    write_table(temporal["annual"], output_dir / "ca_analyte_annual_summary", parquet=False)
    write_table(temporal["seasonal"], output_dir / "ca_analyte_calendar_season_summary", parquet=False)
    write_table(temporal["monthly"], output_dir / "ca_analyte_month_of_year_summary", parquet=False)
    write_table(temporal["system_year"], output_dir / "ca_system_year_summary")
    write_table(concordance, output_dir / "ca_paired_concordance_sensitivity", parquet=False)
    write_table(concordance_by_time, output_dir / "ca_paired_concordance_by_year_and_season", parquet=False)
    write_table(system_medians, output_dir / "ca_paired_pws_medians")
    write_table(threshold_grid, output_dir / "ca_paired_threshold_agreement_grid", parquet=False)
    write_table(standard_threshold, output_dir / "ca_reference_threshold_agreement", parquet=False)
    write_table(bootstrap, output_dir / "ca_pws_cluster_bootstrap", parquet=False)
    write_table(audit, output_dir / "ca_data_quality_audit", parquet=False)
    write_table(line_shape_detail, output_dir / "ca_parser_line_shape_audit", parquet=False)
    write_table(period_audit, output_dir / "ca_source_period_audit", parquet=False)
    write_table(duplicate_detail, output_dir / "ca_duplicate_key_audit", parquet=False)

    primary_row = concordance.loc[
        concordance["analysis"].eq("Exact time, both analytes quantified (primary)")
    ].iloc[0]
    standard_lookup = standard_threshold.set_index("metric")
    summary = {
        "analysis_scope": {
            "years": list(PRIMARY_YEARS),
            "sample_type": "RT",
            "facility_type": "DS",
            "units": "UG/L",
            "analytes": ANALYTES,
        },
        "source": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": source_hash,
            "header_column_count": len(header),
            "line_shape_audit": line_shape,
        },
        "records": {
            "primary_scope_rows": len(primary_scope),
            "valid_rows": len(records),
            "valid_haa5_rows": int((records["analyte"] == "HAA5").sum()),
            "valid_tthm_rows": int((records["analyte"] == "TTHM").sum()),
            "event_analyte_keys": len(events),
            "duplicate_keys": int(events["duplicate_key"].sum()),
            "discordant_duplicate_keys": int(events["duplicate_conflict"].sum()),
        },
        "pairing": {
            "exact_pairs": len(paired),
            "exact_pair_pws": paired["pws_id"].nunique(),
            "date_only_pairs": len(date_only),
            "date_only_pair_pws": date_only["pws_id"].nunique(),
        },
        "concordance": {
            "primary_population": "exact event pairs with both HAA5 and TTHM quantified",
            "primary_detected_pairs": int(primary_row["n_pairs"]),
            "primary_detected_pair_pws": int(primary_row["n_pws"]),
            "primary_spearman_rho": primary_row["spearman_rho"],
            "primary_spearman_ci95": [primary_row["ci95_low"], primary_row["ci95_high"]],
            "primary_kendall_tau_b": primary_row["kendall_tau_b"],
            "pws_median_spearman_rho": concordance.loc[
                concordance["analysis"].eq("PWS-level medians"), "spearman_rho"
            ].iloc[0],
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "bootstrap_unit": "public water system",
        },
        "reference_threshold_60_80": {
            row.metric: {
                "estimate": row.estimate,
                "ci95": [row.ci95_low, row.ci95_high],
            }
            for row in standard_threshold.itertuples()
        },
        "interpretation_guardrails": [
            "The California archive validates HAA5/TTHM co-occurrence only; it contains no HAA9 total.",
            "Public-archive separation does not establish statistical independence from every national data product.",
            "Concentration-level reference comparisons are not regulatory compliance determinations.",
            "The observational analysis does not support causal claims.",
        ],
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "duckdb": duckdb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "runtime_seconds": time.time() - started,
    }
    flat_methods_summary = {
        "exact_pairs": len(paired),
        "exact_pair_pws": paired["pws_id"].nunique(),
    }
    write_json(output_dir / "ca_validation_summary.json", summary)
    write_json(
        output_dir / "ca_validation_methods_and_limitations.json",
        {
            "run_config": asdict(config),
            "source_sha256": source_hash,
            "definitions": {
                "primary_record_scope": "2023-2025; analyte 2456 or 2950; RT; DS; UG/L; valid quantified or censored value",
                "primary_pair_key": "PWS ID + sampling point ID + sample date + nonmissing sample time",
                "primary_censor_substitution": "one-half reporting level",
                "primary_duplicate_resolution": "median record-level value at an event-analyte key",
                "calendar_seasons": {
                    "Winter": [12, 1, 2],
                    "Spring": [3, 4, 5],
                    "Summer": [6, 7, 8],
                    "Autumn": [9, 10, 11],
                },
                "reference_levels_ug_l": {"HAA5": 60.0, "TTHM": 80.0},
            },
            "limitations": summary["interpretation_guardrails"],
        },
    )
    (output_dir / "METHODS_AND_LIMITATIONS.md").write_text(
        methods_markdown(config, source_hash, flat_methods_summary), encoding="utf-8"
    )
    write_output_manifest(output_dir)
    return summary


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
