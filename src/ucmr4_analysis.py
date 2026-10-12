#!/usr/bin/env python3
"""Reproducible UCMR4 HAA panel and finite-grid replay analysis.

The script reads the immutable EPA text extracts in data/interim/extracted and
writes tidy CSV/Parquet outputs to data/processed/ucmr4 and results/ucmr4.
It deliberately evaluates observed-grid chemical signals, not legal compliance,
causal effects, unobserved exposure, or continuous episode duration.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "interim" / "extracted"
PROCESSED_DIR = ROOT / "data" / "processed" / "ucmr4"
RESULTS_DIR = ROOT / "results" / "ucmr4"

INPUT_FILES = {
    "all": INPUT_DIR / "UCMR4_All.txt",
    "additional": INPUT_DIR / "UCMR4_HAA_AddtlDataElem.txt",
    "indicators": INPUT_DIR / "UCMR4_HAA_Indicators.txt",
    "zipcodes": INPUT_DIR / "UCMR4_ZIPCodes.txt",
}

PANELS = ["HAA5", "HAA6Br", "HAA9"]
EVENT_KEY = [
    "PWSID",
    "FacilityID",
    "SamplePointID",
    "CollectionDate",
    "SampleEventCode",
    "SampleID",
]
SITE_ROUND_KEY = ["PWSID", "FacilityID", "SamplePointID", "SampleEventCode"]
SITE_KEY = ["PWSID", "FacilityID", "SamplePointID"]
PRIMARY_THRESHOLD = 60.0
THRESHOLDS = np.arange(20.0, 100.0 + 0.001, 5.0)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260913
HOLDOUT_SALT = "UCMR4-HAA-PWS-HOLDOUT-v1"
TOL = 1e-8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_table(frame: pd.DataFrame, directory: Path, stem: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{stem}.csv"
    parquet_path = directory / f"{stem}.parquet"
    out = frame.copy()
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.to_csv(csv_path, index=False, encoding="utf-8", float_format="%.10g")
    out.to_parquet(parquet_path, index=False, compression="zstd")
    return [csv_path, parquet_path]


def safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def profile_value(values: Iterable[object]) -> str:
    clean = sorted({str(value).strip() for value in values if pd.notna(value) and str(value).strip()})
    if not clean:
        return "MISSING"
    if len(clean) == 1:
        return clean[0]
    return "MULTIPLE"


def numeric_summary(values: pd.Series) -> dict[str, float | int]:
    x = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if x.empty:
        return {
            "n": 0,
            "mean": float("nan"),
            "sd": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    q = x.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)) if x.size > 1 else float("nan"),
        "p05": float(q.loc[0.05]),
        "p25": float(q.loc[0.25]),
        "median": float(q.loc[0.5]),
        "p75": float(q.loc[0.75]),
        "p95": float(q.loc[0.95]),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def holdout_fields(pwsid: str) -> tuple[str, float]:
    digest = hashlib.sha256(f"{HOLDOUT_SALT}|{pwsid}".encode("utf-8")).hexdigest()
    score = int(digest[:16], 16) / float(2**64)
    return digest, score


def read_haa_events() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    columns = [
        "PWSID",
        "PWSName",
        "Size",
        "FacilityID",
        "FacilityName",
        "FacilityWaterType",
        "SamplePointID",
        "SamplePointName",
        "SamplePointType",
        "CollectionDate",
        "SampleID",
        "Contaminant",
        "MRL",
        "Units",
        "MethodID",
        "AnalyticalResultsSign",
        "AnalyticalResultValue",
        "SampleEventCode",
        "MonitoringRequirement",
        "Region",
        "State",
    ]
    raw = pd.read_csv(
        INPUT_FILES["all"], sep="\t", encoding="cp1252", dtype=str, usecols=columns
    )
    raw = raw.loc[raw["Contaminant"].isin(PANELS)].copy()
    raw["AnalyticalResultValue"] = pd.to_numeric(raw["AnalyticalResultValue"], errors="coerce")
    raw["CollectionDateParsed"] = pd.to_datetime(raw["CollectionDate"], errors="coerce")

    duplicated_panel_rows = int(raw.duplicated(EVENT_KEY + ["Contaminant"], keep=False).sum())
    if duplicated_panel_rows:
        raise ValueError(f"Found {duplicated_panel_rows} duplicate event-panel rows; exact pivot is unsafe")

    weak_key = [key for key in EVENT_KEY if key != "SampleID"]
    sampleid_conflicts = int((raw.groupby(weak_key, dropna=False)["SampleID"].nunique(dropna=False) > 1).sum())

    metadata_columns = [
        "PWSName",
        "Size",
        "FacilityName",
        "FacilityWaterType",
        "SamplePointName",
        "SamplePointType",
        "MonitoringRequirement",
        "Region",
        "State",
    ]
    conflict_counts = {
        column: int((raw.groupby(EVENT_KEY, dropna=False)[column].nunique(dropna=False) > 1).sum())
        for column in metadata_columns
    }
    if any(conflict_counts.values()):
        raise ValueError(f"Metadata conflicts within exact events: {conflict_counts}")

    meta = raw[EVENT_KEY + metadata_columns].drop_duplicates(EVENT_KEY).copy()
    values = raw.pivot(index=EVENT_KEY, columns="Contaminant", values="AnalyticalResultValue").reset_index()
    signs = raw.pivot(index=EVENT_KEY, columns="Contaminant", values="AnalyticalResultsSign").reset_index()
    signs.rename(columns={panel: f"{panel}_sign" for panel in PANELS}, inplace=True)
    events = values.merge(signs, on=EVENT_KEY, how="left", validate="one_to_one")
    events = events.merge(meta, on=EVENT_KEY, how="left", validate="one_to_one")
    events["CollectionDateParsed"] = pd.to_datetime(events["CollectionDate"], errors="coerce")
    events["complete_triplet"] = events[PANELS].notna().all(axis=1)
    events["missing_panels"] = events[PANELS].isna().apply(
        lambda row: "+".join(row.index[row].tolist()), axis=1
    )

    incomplete = events.loc[~events["complete_triplet"]].copy()
    incomplete_systems = set(incomplete["PWSID"])
    paired = events.loc[events["complete_triplet"]].copy()
    paired["finite_grid_system_eligible"] = ~paired["PWSID"].isin(incomplete_systems)
    paired["site_id"] = paired["FacilityID"].fillna("") + "::" + paired["SamplePointID"].fillna("")
    paired["extra4_ug_l"] = paired["HAA9"] - paired["HAA5"]
    paired["implied_overlap2_ug_l"] = paired["HAA5"] + paired["HAA6Br"] - paired["HAA9"]
    positive = paired["HAA9"] > 0
    paired["f_extra"] = np.where(positive, paired["extra4_ug_l"] / paired["HAA9"], np.nan)
    paired["f_brominated"] = np.where(positive, paired["HAA6Br"] / paired["HAA9"], np.nan)
    paired["f_haa5"] = np.where(positive, paired["HAA5"] / paired["HAA9"], np.nan)

    audit = {
        "haa_total_rows": int(len(raw)),
        "distinct_exact_events": int(len(events)),
        "complete_triplet_events": int(len(paired)),
        "incomplete_triplet_events": int(len(incomplete)),
        "systems_with_any_haa_total": int(events["PWSID"].nunique()),
        "systems_with_incomplete_grid": int(len(incomplete_systems)),
        "finite_grid_eligible_systems": int(events["PWSID"].nunique() - len(incomplete_systems)),
        "duplicate_event_panel_row_members": duplicated_panel_rows,
        "sample_id_conflicts_within_weak_event_key": sampleid_conflicts,
        "metadata_conflicts": conflict_counts,
        "non_equals_sign_rows": int((raw["AnalyticalResultsSign"] != "=").sum()),
        "missing_numeric_result_rows": int(raw["AnalyticalResultValue"].isna().sum()),
        "unparsed_collection_dates": int(raw["CollectionDateParsed"].isna().sum()),
    }
    return raw, paired, incomplete, audit


def read_residuals() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(INPUT_FILES["additional"], sep="\t", encoding="cp1252", dtype=str)
    residual = raw.loc[raw["AdditionalDataElement"].eq("DisinfectantResidual")].copy()
    residual["Response"] = residual["Response"].fillna("").str.strip()

    def collapse(group: pd.DataFrame) -> pd.Series:
        codes = sorted(set(group.loc[group["Response"].ne(""), "Response"]))
        return pd.Series(
            {
                "residual_code_set": "+".join(codes) if codes else np.nan,
                "residual_code_count": len(codes),
                "residual_source_rows": len(group),
                "residual_conflict": len(codes) > 1,
            }
        )

    collapsed = residual.groupby(SITE_ROUND_KEY, dropna=False, sort=False).apply(
        collapse, include_groups=False
    ).reset_index()
    audit = {
        "additional_data_rows": int(len(raw)),
        "residual_rows": int(len(residual)),
        "residual_unique_site_round_keys": int(len(collapsed)),
        "residual_conflicting_site_round_keys": int(collapsed["residual_conflict"].sum()),
        "residual_missing_response_rows": int(residual["Response"].eq("").sum()),
    }
    return collapsed, audit


def read_indicators() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_csv(INPUT_FILES["indicators"], sep="\t", encoding="cp1252", dtype=str)
    raw["result_ug_l"] = pd.to_numeric(raw["AnalyticalResultValue"], errors="coerce")
    raw["mrl_ug_l"] = pd.to_numeric(raw["MRL"], errors="coerce")
    raw["collection_date"] = pd.to_datetime(raw["CollectionDate"], errors="coerce")
    exact = raw["AnalyticalResultsSign"].eq("=")
    censored = raw["AnalyticalResultsSign"].eq("<")
    raw["value_lower_ug_l"] = np.where(exact, raw["result_ug_l"], np.where(censored, 0.0, np.nan))
    raw["value_upper_ug_l"] = np.where(exact, raw["result_ug_l"], np.where(censored, raw["mrl_ug_l"], np.nan))
    raw["value_midpoint_ug_l"] = np.where(
        exact, raw["result_ug_l"], np.where(censored, raw["mrl_ug_l"] / 2.0, np.nan)
    )
    raw["left_censored"] = censored

    grouped = raw.groupby(["PWSID", "SampleEventCode", "Contaminant"], dropna=False)
    round_summary = grouped.agg(
        source_sample_count=("SampleID", "size"),
        detected_sample_count=("left_censored", lambda x: int((~x).sum())),
        left_censored_sample_count=("left_censored", "sum"),
        value_lower_median_ug_l=("value_lower_ug_l", "median"),
        value_midpoint_median_ug_l=("value_midpoint_ug_l", "median"),
        value_upper_median_ug_l=("value_upper_ug_l", "median"),
        first_collection_date=("collection_date", "min"),
        last_collection_date=("collection_date", "max"),
        source_facility_count=("FacilityID", "nunique"),
        source_sample_point_count=("SamplePointID", "nunique"),
    ).reset_index()
    audit = {
        "indicator_rows": int(len(raw)),
        "indicator_systems": int(raw["PWSID"].nunique()),
        "indicator_system_round_analytes": int(len(round_summary)),
        "indicator_left_censored_rows": int(censored.sum()),
        "indicator_unknown_sign_rows": int((~(exact | censored)).sum()),
        "indicator_missing_mrl_rows": int(raw["mrl_ug_l"].isna().sum()),
        "indicator_unparsed_dates": int(raw["collection_date"].isna().sum()),
    }
    return raw, round_summary, audit


def read_zipcodes(haa_systems: set[str]) -> tuple[pd.DataFrame, dict]:
    zipcodes = pd.read_csv(INPUT_FILES["zipcodes"], sep="\t", encoding="cp1252", dtype=str)
    zipcodes["PWSID"] = zipcodes["PWSID"].str.strip()
    zipcodes["ZIPCODE"] = zipcodes["ZIPCODE"].str.strip()
    duplicate_rows = int(zipcodes.duplicated(["PWSID", "ZIPCODE"], keep=False).sum())
    zipcodes = zipcodes.drop_duplicates(["PWSID", "ZIPCODE"]).copy()
    zipcodes["valid_five_digit_zip"] = zipcodes["ZIPCODE"].str.fullmatch(r"\d{5}", na=False)
    zipcodes["in_haa_system_cohort"] = zipcodes["PWSID"].isin(haa_systems)
    audit = {
        "zip_rows_unique": int(len(zipcodes)),
        "zip_systems": int(zipcodes["PWSID"].nunique()),
        "haa_systems_with_zip": int(zipcodes.loc[zipcodes["in_haa_system_cohort"], "PWSID"].nunique()),
        "haa_systems_without_zip": int(len(haa_systems - set(zipcodes["PWSID"]))),
        "invalid_zip_rows": int((~zipcodes["valid_five_digit_zip"]).sum()),
        "duplicate_source_row_members": duplicate_rows,
    }
    return zipcodes, audit


def build_system_attributes(paired: pd.DataFrame, zipcodes: pd.DataFrame) -> pd.DataFrame:
    systems = pd.DataFrame({"PWSID": sorted(paired["PWSID"].unique())})
    for source_column, output_column in [
        ("Size", "size_profile"),
        ("FacilityWaterType", "facility_water_type_profile"),
        ("State", "state_profile"),
    ]:
        prof = paired.groupby("PWSID", sort=False)[source_column].agg(profile_value).rename(output_column)
        systems = systems.merge(prof, on="PWSID", how="left", validate="one_to_one")

    residual_rows = []
    for pwsid, group in paired.groupby("PWSID", sort=False):
        matched = group["residual_code_set"].notna()
        individual_codes: set[str] = set()
        for code_set in group.loc[matched, "residual_code_set"]:
            individual_codes.update(str(code_set).split("+"))
        if matched.sum() == 0:
            profile = "MISSING"
        elif matched.sum() < len(group):
            profile = "PARTIAL_COVERAGE"
        elif len(individual_codes) == 1:
            profile = next(iter(individual_codes))
        else:
            profile = "MULTIPLE"
        residual_rows.append(
            {
                "PWSID": pwsid,
                "residual_profile": profile,
                "residual_joined_events": int(matched.sum()),
                "paired_events": int(len(group)),
                "residual_join_coverage": float(matched.mean()),
                "residual_codes_observed": "+".join(sorted(individual_codes)) if individual_codes else np.nan,
                "residual_conflict_events": int(group["residual_conflict"].fillna(False).sum()),
            }
        )
    systems = systems.merge(pd.DataFrame(residual_rows), on="PWSID", how="left", validate="one_to_one")

    zip_summary = zipcodes.loc[zipcodes["in_haa_system_cohort"]].groupby("PWSID").agg(
        zip_code_count=("ZIPCODE", "nunique"),
        zip_codes=("ZIPCODE", lambda x: "+".join(sorted(set(x)))),
    ).reset_index()
    systems = systems.merge(zip_summary, on="PWSID", how="left", validate="one_to_one")
    systems["zip_code_count"] = systems["zip_code_count"].fillna(0).astype(int)
    holdout = systems["PWSID"].map(holdout_fields)
    systems["holdout_sha256"] = holdout.map(lambda value: value[0])
    systems["holdout_score"] = holdout.map(lambda value: value[1])
    systems["split"] = "development"
    n_holdout = int(round(0.20 * len(systems)))
    holdout_index = systems.sort_values(["holdout_sha256", "PWSID"]).index[:n_holdout]
    systems.loc[holdout_index, "split"] = "holdout"
    return systems


def composition_summary(paired: pd.DataFrame) -> pd.DataFrame:
    variables = ["HAA5", "HAA6Br", "HAA9", "extra4_ug_l", "implied_overlap2_ug_l", "f_extra", "f_brominated", "f_haa5"]
    rows = []
    for variable in variables:
        event_stats = numeric_summary(paired[variable])
        rows.append({"weighting": "event", "variable": variable, **event_stats})
        system_values = paired.groupby("PWSID", sort=False)[variable].mean()
        system_stats = numeric_summary(system_values)
        rows.append({"weighting": "system_equal_after_within_system_mean", "variable": variable, **system_stats})
    return pd.DataFrame(rows)


def identity_audit(paired: pd.DataFrame) -> pd.DataFrame:
    tests = {
        "nonnegative_HAA5": -paired["HAA5"],
        "nonnegative_HAA6Br": -paired["HAA6Br"],
        "nonnegative_HAA9": -paired["HAA9"],
        "HAA5_le_HAA9": paired["HAA5"] - paired["HAA9"],
        "HAA6Br_le_HAA9": paired["HAA6Br"] - paired["HAA9"],
        "HAA9_le_HAA5_plus_HAA6Br": paired["HAA9"] - paired["HAA5"] - paired["HAA6Br"],
    }
    rows = []
    for name, violation_magnitude in tests.items():
        magnitude = pd.to_numeric(violation_magnitude, errors="coerce")
        rows.append(
            {
                "check": name,
                "check_kind": "structural_identity",
                "events_tested": int(magnitude.notna().sum()),
                "flagged_count": int((magnitude > TOL).sum()),
                "max_positive_violation_ug_l": float(max(0.0, magnitude.max())),
            }
        )
    rows.append(
        {
            "check": "HAA9_nonpositive_fraction_denominator",
            "check_kind": "fraction_denominator_status",
            "events_tested": int(len(paired)),
            "flagged_count": int((paired["HAA9"] <= 0).sum()),
            "max_positive_violation_ug_l": float("nan"),
        }
    )
    return pd.DataFrame(rows)


def threshold_correspondence(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLDS:
        flags = pd.DataFrame(
            {
                "PWSID": paired["PWSID"],
                "h5": paired["HAA5"] > threshold,
                "h6": paired["HAA6Br"] > threshold,
                "h9": paired["HAA9"] > threshold,
            }
        )
        h9_n = int(flags["h9"].sum())
        h6_n = int(flags["h6"].sum())
        h9_system_n = int(flags.loc[flags["h9"], "PWSID"].nunique())
        rows.append(
            {
                "threshold_ug_l": threshold,
                "weighting": "event",
                "n_events": int(len(flags)),
                "n_systems": int(flags["PWSID"].nunique()),
                "reference_positive_events_HAA9": h9_n,
                "reference_positive_systems_HAA9": h9_system_n,
                "HAA9_positive_rate": float(flags["h9"].mean()),
                "HAA5_positive_rate": float(flags["h5"].mean()),
                "HAA6Br_positive_rate": float(flags["h6"].mean()),
                "HAA5_capture_given_HAA9_positive": safe_rate(int((flags["h5"] & flags["h9"]).sum()), h9_n),
                "HAA6Br_capture_given_HAA9_positive": safe_rate(int((flags["h6"] & flags["h9"]).sum()), h9_n),
                "HAA9_only_vs_HAA5_given_HAA9_positive": safe_rate(int((flags["h9"] & ~flags["h5"]).sum()), h9_n),
                "HAA5_positive_HAA9_negative_count": int((flags["h5"] & ~flags["h9"]).sum()),
                "HAA5_capture_given_HAA6Br_positive": safe_rate(int((flags["h5"] & flags["h6"]).sum()), h6_n),
                "HAA6Br_capture_given_HAA5_positive": safe_rate(int((flags["h5"] & flags["h6"]).sum()), int(flags["h5"].sum())),
            }
        )

        system_rows = []
        for _, group in flags.groupby("PWSID", sort=False):
            ref_n = int(group["h9"].sum())
            h6_ref_n = int(group["h6"].sum())
            h5_n = int(group["h5"].sum())
            system_rows.append(
                {
                    "HAA9_positive_rate": group["h9"].mean(),
                    "HAA5_positive_rate": group["h5"].mean(),
                    "HAA6Br_positive_rate": group["h6"].mean(),
                    "HAA5_capture_given_HAA9_positive": safe_rate(int((group["h5"] & group["h9"]).sum()), ref_n),
                    "HAA6Br_capture_given_HAA9_positive": safe_rate(int((group["h6"] & group["h9"]).sum()), ref_n),
                    "HAA9_only_vs_HAA5_given_HAA9_positive": safe_rate(int((group["h9"] & ~group["h5"]).sum()), ref_n),
                    "HAA5_capture_given_HAA6Br_positive": safe_rate(int((group["h5"] & group["h6"]).sum()), h6_ref_n),
                    "HAA6Br_capture_given_HAA5_positive": safe_rate(int((group["h5"] & group["h6"]).sum()), h5_n),
                    "HAA5_positive_HAA9_negative_count": int((group["h5"] & ~group["h9"]).sum()),
                }
            )
        sys = pd.DataFrame(system_rows)
        rows.append(
            {
                "threshold_ug_l": threshold,
                "weighting": "system_equal_after_within_system_rate",
                "n_events": int(len(flags)),
                "n_systems": int(len(sys)),
                "reference_positive_events_HAA9": h9_n,
                "reference_positive_systems_HAA9": h9_system_n,
                **{column: float(sys[column].mean()) for column in sys.columns},
            }
        )
    return pd.DataFrame(rows)


def make_grid_tables(paired: pd.DataFrame, attributes: pd.DataFrame):
    grid = paired.loc[paired["finite_grid_system_eligible"]].copy()
    round_table = grid.groupby(["PWSID", "SampleEventCode"], sort=False).agg(
        HAA5_max=("HAA5", "max"),
        HAA6Br_max=("HAA6Br", "max"),
        HAA9_max=("HAA9", "max"),
        observed_site_count=("site_id", "nunique"),
        observed_event_count=("SampleID", "nunique"),
        first_collection_date=("CollectionDateParsed", "min"),
        last_collection_date=("CollectionDateParsed", "max"),
    ).reset_index()
    site_table = grid.groupby(SITE_KEY + ["site_id"], sort=False).agg(
        HAA5_max=("HAA5", "max"),
        HAA6Br_max=("HAA6Br", "max"),
        HAA9_max=("HAA9", "max"),
        observed_round_count=("SampleEventCode", "nunique"),
        observed_event_count=("SampleID", "nunique"),
        first_collection_date=("CollectionDateParsed", "min"),
        last_collection_date=("CollectionDateParsed", "max"),
    ).reset_index()
    site_round_table = grid.groupby(SITE_ROUND_KEY + ["site_id"], sort=False).agg(
        HAA5_max=("HAA5", "max"),
        HAA6Br_max=("HAA6Br", "max"),
        HAA9_max=("HAA9", "max"),
        observed_event_count=("SampleID", "nunique"),
        first_collection_date=("CollectionDateParsed", "min"),
        last_collection_date=("CollectionDateParsed", "max"),
    ).reset_index()
    full_table = grid.groupby("PWSID", sort=False).agg(
        HAA5_full_max=("HAA5", "max"),
        HAA6Br_full_max=("HAA6Br", "max"),
        HAA9_full_max=("HAA9", "max"),
        observed_round_count=("SampleEventCode", "nunique"),
        observed_site_count=("site_id", "nunique"),
        observed_event_count=("SampleID", "nunique"),
        first_collection_date=("CollectionDateParsed", "min"),
        last_collection_date=("CollectionDateParsed", "max"),
    ).reset_index()
    full_table = full_table.merge(attributes, on="PWSID", how="left", validate="one_to_one")
    return grid, round_table, site_table, site_round_table, full_table


def make_system_contributions(
    units: pd.DataFrame, full: pd.DataFrame, threshold: float, design: str
) -> pd.DataFrame:
    merged = units[["PWSID", "HAA5_max", "HAA9_max"]].merge(
        full[[
            "PWSID",
            "HAA5_full_max",
            "HAA9_full_max",
            "split",
            "size_profile",
            "facility_water_type_profile",
            "residual_profile",
            "state_profile",
        ]],
        on="PWSID",
        how="inner",
        validate="many_to_one",
    )
    merged["y00"] = (merged["HAA5_max"] > threshold).astype(float)
    merged["y10"] = (merged["HAA9_max"] > threshold).astype(float)
    merged["y01"] = (merged["HAA5_full_max"] > threshold).astype(float)
    merged["y11"] = (merged["HAA9_full_max"] > threshold).astype(float)

    b00 = merged["y00"].astype(bool)
    b10 = merged["y10"].astype(bool)
    b01 = merged["y01"].astype(bool)
    b11 = merged["y11"].astype(bool)
    merged["cat_already_detected"] = (b00 & b11).astype(float)
    merged["cat_panel_only"] = (~b00 & b10 & ~b01 & b11).astype(float)
    merged["cat_design_only"] = (~b00 & ~b10 & b01 & b11).astype(float)
    merged["cat_either_change"] = (~b00 & b10 & b01 & b11).astype(float)
    merged["cat_both_required"] = (~b00 & ~b10 & ~b01 & b11).astype(float)

    metric_columns = [
        "y00",
        "y10",
        "y01",
        "y11",
        "cat_already_detected",
        "cat_panel_only",
        "cat_design_only",
        "cat_either_change",
        "cat_both_required",
    ]
    system = merged.groupby("PWSID", sort=False)[metric_columns].mean().reset_index()
    unit_count = merged.groupby("PWSID", sort=False).size().rename("admissible_unit_count").reset_index()
    system = system.merge(unit_count, on="PWSID", validate="one_to_one")
    system = system.merge(
        full[[
            "PWSID",
            "split",
            "size_profile",
            "facility_water_type_profile",
            "residual_profile",
            "state_profile",
        ]],
        on="PWSID",
        validate="one_to_one",
    )
    system["panel_gain_at_M0"] = system["y10"] - system["y00"]
    system["design_gain_at_HAA5"] = system["y01"] - system["y00"]
    system["combined_gain"] = system["y11"] - system["y00"]
    system["interaction"] = system["y11"] - system["y10"] - system["y01"] + system["y00"]
    system["scope_attribution"] = 0.5 * (
        (system["y10"] - system["y00"]) + (system["y11"] - system["y01"])
    )
    system["design_attribution"] = 0.5 * (
        (system["y01"] - system["y00"]) + (system["y11"] - system["y10"])
    )
    system["threshold_ug_l"] = threshold
    system["design"] = design

    order_violations = (
        (merged["y00"] > merged["y10"])
        | (merged["y00"] > merged["y01"])
        | (merged["y10"] > merged["y11"])
        | (merged["y01"] > merged["y11"])
    )
    system.attrs["unit_order_violations"] = int(order_violations.sum())
    return system


METRIC_COLUMNS = [
    "y00",
    "y10",
    "y01",
    "y11",
    "panel_gain_at_M0",
    "design_gain_at_HAA5",
    "combined_gain",
    "interaction",
    "scope_attribution",
    "design_attribution",
    "cat_already_detected",
    "cat_panel_only",
    "cat_design_only",
    "cat_either_change",
    "cat_both_required",
]


def summarize_contributions(
    contributions: pd.DataFrame,
    split: str = "all",
    population: str = "reference_positive",
) -> dict:
    data = contributions if split == "all" else contributions.loc[contributions["split"].eq(split)]
    n_all = int(len(data))
    n_ref = int((data["y11"] == 1).sum())
    if population == "reference_positive":
        evaluated = data.loc[data["y11"].eq(1)]
    elif population == "all_eligible":
        evaluated = data
    else:
        raise ValueError(population)
    summary = {
        "design": contributions["design"].iloc[0],
        "threshold_ug_l": float(contributions["threshold_ug_l"].iloc[0]),
        "split": split,
        "population": population,
        "n_eligible_systems": n_all,
        "n_reference_positive_systems": n_ref,
        "reference_positive_fraction": safe_rate(n_ref, n_all),
    }
    for column in METRIC_COLUMNS:
        summary[column] = float(evaluated[column].mean()) if len(evaluated) else float("nan")
    summary["repair_category_sum"] = float(
        sum(summary[column] for column in METRIC_COLUMNS if column.startswith("cat_"))
    )
    summary["attribution_sum_error"] = float(
        summary["scope_attribution"] + summary["design_attribution"] - summary["combined_gain"]
    )
    summary["interaction_category_identity_error"] = float(
        summary["interaction"] - (summary["cat_both_required"] - summary["cat_either_change"])
    )
    return summary


def bootstrap_primary(contributions: pd.DataFrame, split: str, seed: int) -> pd.DataFrame:
    data = contributions if split == "all" else contributions.loc[contributions["split"].eq(split)]
    data = data.reset_index(drop=True)
    n = len(data)
    y11 = data["y11"].to_numpy(float)
    conditional_columns = [
        "y00",
        "y10",
        "y01",
        "y11",
        "panel_gain_at_M0",
        "design_gain_at_HAA5",
        "combined_gain",
        "interaction",
        "scope_attribution",
        "design_attribution",
        "cat_already_detected",
        "cat_panel_only",
        "cat_design_only",
        "cat_either_change",
        "cat_both_required",
    ]
    unconditioned_columns = ["y00", "y10", "y01", "y11"]
    cond_matrix = data[conditional_columns].to_numpy(float) * y11[:, None]
    uncond_matrix = data[unconditioned_columns].to_numpy(float)
    names = ["reference_positive_fraction"]
    names += [f"conditional_{name}" for name in conditional_columns]
    names += [f"unconditional_{name}" for name in unconditioned_columns]
    samples = np.empty((BOOTSTRAP_REPLICATES, len(names)), dtype=float)
    rng = np.random.default_rng(seed)
    probabilities = np.repeat(1.0 / n, n)
    batch_size = 200
    offset = 0
    while offset < BOOTSTRAP_REPLICATES:
        batch = min(batch_size, BOOTSTRAP_REPLICATES - offset)
        weights = rng.multinomial(n, probabilities, size=batch)
        reference_denominator = weights @ y11
        samples[offset : offset + batch, 0] = reference_denominator / n
        conditional = weights @ cond_matrix
        conditional = np.divide(
            conditional,
            reference_denominator[:, None],
            out=np.full_like(conditional, np.nan),
            where=reference_denominator[:, None] > 0,
        )
        samples[offset : offset + batch, 1 : 1 + len(conditional_columns)] = conditional
        samples[offset : offset + batch, 1 + len(conditional_columns) :] = (weights @ uncond_matrix) / n
        offset += batch

    point = summarize_contributions(contributions, split=split, population="reference_positive")
    rows = []
    for index, name in enumerate(names):
        if name == "reference_positive_fraction":
            estimate = point[name]
        elif name.startswith("conditional_"):
            estimate = point[name.removeprefix("conditional_")]
        else:
            metric = name.removeprefix("unconditional_")
            estimate = summarize_contributions(
                contributions, split=split, population="all_eligible"
            )[metric]
        rows.append(
            {
                "split": split,
                "metric": name,
                "estimate": estimate,
                "ci_lower_2_5pct": float(np.nanquantile(samples[:, index], 0.025)),
                "ci_upper_97_5pct": float(np.nanquantile(samples[:, index], 0.975)),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "seed": seed,
                "n_eligible_systems": point["n_eligible_systems"],
                "n_reference_positive_systems": point["n_reference_positive_systems"],
            }
        )
    return pd.DataFrame(rows)


def hypergeom_capture(total: int, positive: int, selected: int) -> float:
    selected = min(selected, total)
    if total <= 0 or selected <= 0:
        return 0.0
    if positive <= 0:
        return 0.0
    if total - positive < selected:
        return 1.0
    return 1.0 - math.comb(total - positive, selected) / math.comb(total, selected)


def subset_capture_curve(
    units: pd.DataFrame,
    full: pd.DataFrame,
    dimension: str,
    threshold: float,
) -> pd.DataFrame:
    count_name = "observed_round_count" if dimension == "temporal_rounds" else "observed_site_count"
    budgets = range(1, int(full[count_name].max()) + 1)
    rows = []
    for signal_panel in ["HAA5", "HAA9"]:
        positive = units.assign(_positive=units[f"{signal_panel}_max"] > threshold).groupby("PWSID").agg(
            total_units=("_positive", "size"), positive_units=("_positive", "sum")
        ).reset_index()
        joined = full[["PWSID", "HAA5_full_max", "HAA9_full_max"]].merge(
            positive, on="PWSID", validate="one_to_one"
        )
        reference_panels = [signal_panel]
        if signal_panel == "HAA5":
            reference_panels.append("HAA9")
        for reference_panel in reference_panels:
            cohort = joined.loc[joined[f"{reference_panel}_full_max"] > threshold].copy()
            for budget in budgets:
                probabilities = cohort.apply(
                    lambda row: hypergeom_capture(
                        int(row["total_units"]), int(row["positive_units"]), budget
                    ),
                    axis=1,
                )
                rows.append(
                    {
                        "dimension": dimension,
                        "threshold_ug_l": threshold,
                        "signal_panel": signal_panel,
                        "reference_panel": reference_panel,
                        "selection_rule": "uniform random subset without replacement; use all if budget exceeds observed units",
                        "budget": budget,
                        "n_reference_positive_systems": int(len(cohort)),
                        "mean_capture_probability": float(probabilities.mean()),
                        "median_capture_probability": float(probabilities.median()),
                        "p25_capture_probability": float(probabilities.quantile(0.25)),
                        "p75_capture_probability": float(probabilities.quantile(0.75)),
                        "systems_using_all_observed_units": int((cohort["total_units"] <= budget).sum()),
                    }
                )
    return pd.DataFrame(rows)


def indicator_associations(paired: pd.DataFrame, indicator_round: pd.DataFrame) -> pd.DataFrame:
    haa_round = paired.groupby(["PWSID", "SampleEventCode"], sort=False).agg(
        HAA9_max=("HAA9", "max"),
        HAA5_max=("HAA5", "max"),
        f_extra_mean=("f_extra", "mean"),
        f_brominated_mean=("f_brominated", "mean"),
    ).reset_index()
    value_columns = [
        "value_lower_median_ug_l",
        "value_midpoint_median_ug_l",
        "value_upper_median_ug_l",
    ]
    wide = indicator_round.pivot(
        index=["PWSID", "SampleEventCode"], columns="Contaminant", values=value_columns
    )
    wide.columns = [f"{analyte}_{bound}" for bound, analyte in wide.columns]
    wide = wide.reset_index()
    joined = haa_round.merge(wide, on=["PWSID", "SampleEventCode"], how="inner", validate="one_to_one")
    pairs = [
        ("bromide", "f_brominated_mean"),
        ("bromide", "f_extra_mean"),
        ("bromide", "HAA9_max"),
        ("total organic carbon", "HAA9_max"),
        ("total organic carbon", "f_extra_mean"),
    ]
    rows = []
    bound_map = {
        "lower": "value_lower_median_ug_l",
        "midpoint": "value_midpoint_median_ug_l",
        "upper": "value_upper_median_ug_l",
    }
    for analyte, outcome in pairs:
        for bound_label, bound_column in bound_map.items():
            exposure = f"{analyte}_{bound_column}"
            for weighting in ["system_round", "system_equal_after_round_mean"]:
                data = joined[["PWSID", exposure, outcome]].dropna()
                if weighting == "system_equal_after_round_mean":
                    data = data.groupby("PWSID", sort=False)[[exposure, outcome]].mean().reset_index()
                if len(data) >= 3:
                    rho, pvalue = spearmanr(data[exposure], data[outcome], nan_policy="omit")
                else:
                    rho, pvalue = np.nan, np.nan
                rows.append(
                    {
                        "source_indicator": analyte,
                        "indicator_censoring_value": bound_label,
                        "distribution_outcome": outcome,
                        "weighting": weighting,
                        "n": int(len(data)),
                        "spearman_rho": float(rho),
                        "two_sided_p_value_descriptive": float(pvalue),
                        "join_level": "PWSID plus SampleEventCode; source and distribution sites are not asserted to be hydraulically paired",
                    }
                )
    return pd.DataFrame(rows)


def missingness_table(
    haa_raw: pd.DataFrame,
    paired: pd.DataFrame,
    indicators: pd.DataFrame,
    attributes: pd.DataFrame,
) -> pd.DataFrame:
    specs = [
        ("UCMR4_All_HAA_rows", haa_raw, ["PWSID", "FacilityID", "SamplePointID", "CollectionDate", "SampleID", "SampleEventCode", "Contaminant", "AnalyticalResultValue"]),
        ("paired_HAA_events", paired, ["HAA5", "HAA6Br", "HAA9", "residual_code_set"]),
        ("HAA_indicators", indicators, ["PWSID", "SampleEventCode", "Contaminant", "result_ug_l", "mrl_ug_l", "value_midpoint_ug_l"]),
        ("system_attributes", attributes, ["size_profile", "facility_water_type_profile", "residual_profile", "zip_codes"]),
    ]
    rows = []
    for dataset, frame, columns in specs:
        for column in columns:
            missing = frame[column].isna() | frame[column].astype(str).str.strip().eq("")
            rows.append(
                {
                    "dataset": dataset,
                    "field": column,
                    "rows": int(len(frame)),
                    "missing_count": int(missing.sum()),
                    "missing_fraction": float(missing.mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    input_manifest = {
        name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in INPUT_FILES.items()
    }

    haa_raw, paired, incomplete, pairing_audit = read_haa_events()
    residual, residual_audit = read_residuals()
    paired = paired.merge(residual, on=SITE_ROUND_KEY, how="left", validate="many_to_one")
    indicators, indicator_round, indicator_audit = read_indicators()
    zipcodes, zip_audit = read_zipcodes(set(paired["PWSID"]))
    attributes = build_system_attributes(paired, zipcodes)
    paired = paired.merge(
        attributes[["PWSID", "split"]], on="PWSID", how="left", validate="many_to_one"
    )

    grid, round_table, site_table, site_round_table, full_table = make_grid_tables(paired, attributes)
    composition = composition_summary(paired)
    identities = identity_audit(paired)
    correspondence = threshold_correspondence(paired)
    missingness = missingness_table(haa_raw, paired, indicators, attributes)
    indicator_assoc = indicator_associations(paired, indicator_round)

    primary_by_threshold = []
    contribution_by_threshold: dict[float, pd.DataFrame] = {}
    order_audit_rows = []
    for threshold in THRESHOLDS:
        contribution = make_system_contributions(round_table, full_table, threshold, "one_round")
        contribution_by_threshold[float(threshold)] = contribution
        order_audit_rows.append(
            {
                "threshold_ug_l": threshold,
                "design": "one_round",
                "unit_order_violations": contribution.attrs["unit_order_violations"],
            }
        )
        primary_by_threshold.append(summarize_contributions(contribution))
    four_cell_thresholds = pd.DataFrame(primary_by_threshold)

    c60 = contribution_by_threshold[PRIMARY_THRESHOLD]
    primary_rows = []
    for split in ["all", "development", "holdout"]:
        for population in ["reference_positive", "all_eligible"]:
            primary_rows.append(summarize_contributions(c60, split=split, population=population))
    four_cell_primary = pd.DataFrame(primary_rows)

    alternative_contributions = {
        "one_round": c60,
        "one_site": make_system_contributions(site_table, full_table, PRIMARY_THRESHOLD, "one_site"),
        "one_site_round": make_system_contributions(
            site_round_table, full_table, PRIMARY_THRESHOLD, "one_site_round"
        ),
    }
    alternative_rows = []
    repair_rows = []
    for design, contribution in alternative_contributions.items():
        order_audit_rows.append(
            {
                "threshold_ug_l": PRIMARY_THRESHOLD,
                "design": design,
                "unit_order_violations": contribution.attrs["unit_order_violations"],
            }
        )
        for split in ["all", "development", "holdout"]:
            summary = summarize_contributions(contribution, split=split)
            alternative_rows.append(summary)
            for category in [
                "cat_already_detected",
                "cat_panel_only",
                "cat_design_only",
                "cat_either_change",
                "cat_both_required",
            ]:
                repair_rows.append(
                    {
                        "design": design,
                        "threshold_ug_l": PRIMARY_THRESHOLD,
                        "split": split,
                        "category": category.removeprefix("cat_"),
                        "system_weighted_fraction": summary[category],
                        "n_reference_positive_systems": summary["n_reference_positive_systems"],
                    }
                )
    alternatives = pd.DataFrame(alternative_rows)
    repairs = pd.DataFrame(repair_rows)
    order_audit = pd.DataFrame(order_audit_rows).drop_duplicates()

    strata_rows = []
    for dimension, column in [
        ("Size", "size_profile"),
        ("FacilityWaterType_system_profile", "facility_water_type_profile"),
        ("DisinfectantResidual_system_profile", "residual_profile"),
    ]:
        for level in sorted(c60[column].dropna().unique()):
            subset = c60.loc[c60[column].eq(level)].copy()
            subset["split"] = "development"
            summary = summarize_contributions(subset, split="all")
            summary.update(
                {
                    "stratum_dimension": dimension,
                    "stratum_level": level,
                    "sparse_stratum_lt20_reference_systems": summary["n_reference_positive_systems"] < 20,
                }
            )
            strata_rows.append(summary)
    strata = pd.DataFrame(strata_rows)

    geography_rows = []
    for state in sorted(c60["state_profile"].dropna().unique()):
        subset = c60.loc[c60["state_profile"].eq(state)].copy()
        subset["split"] = "development"
        summary = summarize_contributions(subset, split="all")
        summary.update({"state": state, "sparse_lt10_reference_systems": summary["n_reference_positive_systems"] < 10})
        geography_rows.append(summary)
    geography = pd.DataFrame(geography_rows)

    temporal_capture = subset_capture_curve(round_table, full_table, "temporal_rounds", PRIMARY_THRESHOLD)
    spatial_capture = subset_capture_curve(site_table, full_table, "spatial_sites", PRIMARY_THRESHOLD)

    bootstrap = pd.concat(
        [
            bootstrap_primary(c60, "all", BOOTSTRAP_SEED),
            bootstrap_primary(c60, "development", BOOTSTRAP_SEED + 1),
            bootstrap_primary(c60, "holdout", BOOTSTRAP_SEED + 2),
        ],
        ignore_index=True,
    )

    qa_pairing_rows = [
        {
            "metric": key,
            "value": json.dumps(value, sort_keys=True) if isinstance(value, dict) else str(value),
        }
        for key, value in pairing_audit.items()
    ]
    qa_pairing_rows.extend(
        {
            "metric": f"incomplete_pattern_{pattern or 'none'}",
            "value": str(int(count)),
        }
        for pattern, count in incomplete["missing_panels"].value_counts(dropna=False).items()
    )
    qa_pairing = pd.DataFrame(qa_pairing_rows)

    residual_join_audit = {
        **residual_audit,
        "complete_triplet_events": int(len(paired)),
        "complete_triplet_events_with_exact_residual_join": int(paired["residual_code_set"].notna().sum()),
        "complete_triplet_event_residual_join_fraction": float(paired["residual_code_set"].notna().mean()),
        "systems_with_any_exact_residual_join": int(
            paired.loc[paired["residual_code_set"].notna(), "PWSID"].nunique()
        ),
    }

    paired_out_columns = EVENT_KEY + [
        "PWSName",
        "Size",
        "FacilityName",
        "FacilityWaterType",
        "SamplePointName",
        "SamplePointType",
        "Region",
        "State",
        "site_id",
        "HAA5",
        "HAA6Br",
        "HAA9",
        "extra4_ug_l",
        "implied_overlap2_ug_l",
        "f_extra",
        "f_brominated",
        "f_haa5",
        "residual_code_set",
        "residual_code_count",
        "residual_conflict",
        "finite_grid_system_eligible",
        "split",
    ]
    write_table(paired[paired_out_columns], PROCESSED_DIR, "paired_haa_events")
    write_table(incomplete, PROCESSED_DIR, "incomplete_haa_events")
    write_table(residual, PROCESSED_DIR, "residual_site_round")
    indicator_out = indicators[[
        "PWSID",
        "FacilityID",
        "SamplePointID",
        "SampleEventCode",
        "CollectionDate",
        "SampleID",
        "Contaminant",
        "AnalyticalResultsSign",
        "result_ug_l",
        "mrl_ug_l",
        "value_lower_ug_l",
        "value_midpoint_ug_l",
        "value_upper_ug_l",
        "left_censored",
    ]]
    write_table(indicator_out, PROCESSED_DIR, "indicator_records_clean")
    write_table(indicator_round, PROCESSED_DIR, "indicator_system_round_summary")
    write_table(zipcodes, PROCESSED_DIR, "system_zip_codes")
    write_table(attributes, PROCESSED_DIR, "system_attributes")
    write_table(round_table, PROCESSED_DIR, "system_round_panel_maxima")
    write_table(site_table, PROCESSED_DIR, "system_site_panel_maxima")
    write_table(site_round_table, PROCESSED_DIR, "system_site_round_panel_maxima")
    write_table(full_table, PROCESSED_DIR, "system_full_grid_panel_maxima")
    write_table(c60, PROCESSED_DIR, "system_contributions_primary_B60")

    write_table(qa_pairing, RESULTS_DIR, "qa_pairing")
    write_table(identities, RESULTS_DIR, "qa_nested_identities")
    write_table(order_audit, RESULTS_DIR, "qa_four_cell_order")
    write_table(missingness, RESULTS_DIR, "missingness")
    write_table(composition, RESULTS_DIR, "composition_summary")
    write_table(correspondence, RESULTS_DIR, "threshold_correspondence_20_100")
    write_table(four_cell_primary, RESULTS_DIR, "four_cell_primary_B60")
    write_table(four_cell_thresholds, RESULTS_DIR, "four_cell_threshold_sensitivity")
    write_table(alternatives, RESULTS_DIR, "alternative_designs_B60")
    write_table(repairs, RESULTS_DIR, "repair_categories_B60")
    write_table(strata, RESULTS_DIR, "four_cell_strata_B60")
    write_table(geography, RESULTS_DIR, "state_summary_B60")
    write_table(temporal_capture, RESULTS_DIR, "temporal_random_subset_capture_B60")
    write_table(spatial_capture, RESULTS_DIR, "spatial_random_subset_capture_B60")
    write_table(bootstrap, RESULTS_DIR, "bootstrap_system_cluster_B60")
    write_table(indicator_assoc, RESULTS_DIR, "indicator_associations")

    primary = summarize_contributions(c60)
    holdout = summarize_contributions(c60, split="holdout")
    composition_lookup = composition.set_index(["weighting", "variable"])
    event_extra = composition_lookup.loc[("event", "f_extra")].to_dict()
    system_extra = composition_lookup.loc[("system_equal_after_within_system_mean", "f_extra")].to_dict()
    metrics = {
        "analysis_scope": "Observed-grid UCMR4 HAA chemical-signal replay; not compliance, causality, exposure, or continuous duration",
        "parameters": {
            "primary_threshold_ug_l": PRIMARY_THRESHOLD,
            "threshold_grid_ug_l": THRESHOLDS.tolist(),
            "positive_operator": "strictly greater than threshold",
            "primary_M0": "one SampleEventCode round selected uniformly within PWS; exact enumeration",
            "M1": "all complete observed site-round events in the PWS finite grid",
            "system_weighting": "equal PWS weight; candidate-unit outcomes averaged within PWS first",
            "holdout": f"lowest round(20% of PWS) salted SHA-256 digests using salt {HOLDOUT_SALT}",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "input_manifest": input_manifest,
        "qa": {
            "pairing": pairing_audit,
            "residual_join": residual_join_audit,
            "indicators": indicator_audit,
            "zipcodes": zip_audit,
            "identity_and_denominator_flags": identities.set_index("check")["flagged_count"].to_dict(),
            "nested_identity_violation_count": int(
                identities.loc[
                    identities["check_kind"].eq("structural_identity"), "flagged_count"
                ].sum()
            ),
            "four_cell_order_violations": int(order_audit["unit_order_violations"].sum()),
        },
        "primary_B60_overall": primary,
        "primary_B60_holdout": holdout,
        "composition_f_extra": {
            "event_weighted": event_extra,
            "system_weighted": system_extra,
        },
        "split_counts_all_paired_systems": attributes["split"].value_counts().to_dict(),
        "split_counts_finite_grid_systems": full_table["split"].value_counts().to_dict(),
    }
    metrics_path = RESULTS_DIR / "analysis_metrics.json"
    metrics_path.write_text(json.dumps(json_ready(metrics), indent=2, sort_keys=True), encoding="utf-8")

    boot_lookup = bootstrap.loc[bootstrap["split"].eq("all")].set_index("metric")
    p00_ci = boot_lookup.loc["conditional_y00"]
    p10_ci = boot_lookup.loc["conditional_y10"]
    p01_ci = boot_lookup.loc["conditional_y01"]
    interaction_ci = boot_lookup.loc["conditional_interaction"]
    methods_results = f"""# UCMR4 HAA observed-grid analysis

## Methods

The analysis paired HAA5, HAA6Br and HAA9 only when PWSID, facility, sample point, collection date, SampleEventCode and SampleID matched exactly. All concentrations were reported as numeric equality results in the source extract. Same-event composition used every complete triplet. The finite-grid replay excluded every PWS with any incomplete three-total event, so the full reference grid did not silently discard unresolved panel results. HAA9 identity checks used a tolerance of {TOL:g} µg/L.

The primary endpoint was a chemical signal: at least one selected concentration strictly greater than {PRIMARY_THRESHOLD:g} µg/L. It is not a legal HAA9 standard or compliance determination. A0 was HAA5, A1 was HAA9, M0 was one SampleEventCode round selected uniformly without reference to its concentrations, and M1 was the full finite observed grid. All admissible M0 rounds were enumerated. Outcomes were averaged within PWS before PWS-level averaging. Thus systems with more rounds or sites did not receive more weight. The reference-positive population was defined by Y11=1. Alternative designs selected one fixed observed site across all its rounds, or one observed site-round, with the same within-system enumeration.

Temporal and spatial random-subset curves used exact hypergeometric probabilities. A budget larger than a system's observed universe used all of that system's observed units. The development/holdout label was set only from a salted SHA-256 hash of PWSID: the lowest rounded 20% of digests formed the holdout. Percentile 95% intervals came from {BOOTSTRAP_REPLICATES:,} PWS-cluster resamples (seed {BOOTSTRAP_SEED}); each resample preserved all candidate-round outcomes for a selected PWS.

Residual-disinfectant codes were joined exactly on PWSID, facility, sample point and SampleEventCode. Conflicting codes were retained as code sets. Source-water bromide and total organic carbon were joined only at PWSID–SampleEventCode for descriptive, non-hydraulically-paired associations. Nondetects were represented by 0–MRL bounds, with MRL/2 as a labeled midpoint sensitivity. Facility-water-type and residual strata use whole-system profiles; systems spanning values are labeled MULTIPLE, and partial residual coverage is labeled PARTIAL_COVERAGE.

## Results

The source contained {pairing_audit['distinct_exact_events']:,} distinct HAA events in {pairing_audit['systems_with_any_haa_total']:,} systems. Exact complete triplets were available for {pairing_audit['complete_triplet_events']:,} events. {pairing_audit['incomplete_triplet_events']:,} events in {pairing_audit['systems_with_incomplete_grid']:,} systems were incomplete; excluding those systems left {pairing_audit['finite_grid_eligible_systems']:,} systems for finite-grid replay. No complete triplet violated HAA5≤HAA9, HAA6Br≤HAA9, or HAA9≤HAA5+HAA6Br at the stated tolerance.

Among {primary['n_reference_positive_systems']:,} reference-positive systems at {PRIMARY_THRESHOLD:g} µg/L, the system-weighted primary cell probabilities were p00={primary['y00']:.4f} (95% cluster-bootstrap CI {p00_ci['ci_lower_2_5pct']:.4f}–{p00_ci['ci_upper_97_5pct']:.4f}), p10={primary['y10']:.4f} ({p10_ci['ci_lower_2_5pct']:.4f}–{p10_ci['ci_upper_97_5pct']:.4f}), p01={primary['y01']:.4f} ({p01_ci['ci_lower_2_5pct']:.4f}–{p01_ci['ci_upper_97_5pct']:.4f}), and p11=1 by reference definition. The panel gain at M0 was {primary['panel_gain_at_M0']:.4f}, the grid gain at HAA5 was {primary['design_gain_at_HAA5']:.4f}, and the combined gain was {primary['combined_gain']:.4f}. The interaction was {primary['interaction']:.4f} ({interaction_ci['ci_lower_2_5pct']:.4f}–{interaction_ci['ci_upper_97_5pct']:.4f}).

The mutually exclusive system-weighted repair fractions were already detected {primary['cat_already_detected']:.4f}, panel expansion sufficient while design expansion alone was insufficient {primary['cat_panel_only']:.4f}, design expansion sufficient while panel expansion alone was insufficient {primary['cat_design_only']:.4f}, either change alone sufficient {primary['cat_either_change']:.4f}, and both changes required {primary['cat_both_required']:.4f}. These fractions sum to {primary['repair_category_sum']:.4f} on the complete reference-positive denominator. The interaction identity error and attribution-sum error were both below {max(abs(primary['interaction_category_identity_error']), abs(primary['attribution_sum_error'])):.3g}.

Across complete same-event triplets with positive HAA9, the event-weighted median fraction outside HAA5 was {event_extra['median']:.4f} (IQR {event_extra['p25']:.4f}–{event_extra['p75']:.4f}); after averaging events within PWS first, the median system value was {system_extra['median']:.4f} (IQR {system_extra['p25']:.4f}–{system_extra['p75']:.4f}). These are measured mass fractions, not event probabilities or health-risk fractions.

Threshold, holdout, one-site, one-site-round, temporal/spatial subset, stratum, geographic and indicator sensitivity results are in the accompanying tidy tables. Stratum estimates with fewer than 20 reference-positive systems and state estimates with fewer than 10 are flagged. The observed-grid design cannot identify subgrid excursions, continuous event duration, unobserved household exposure, legal violations, or national prevalence.
"""
    (RESULTS_DIR / "METHODS_RESULTS.md").write_text(methods_results, encoding="utf-8")

    print(json.dumps(json_ready({
        "status": "complete",
        "paired_events": pairing_audit["complete_triplet_events"],
        "finite_grid_systems": pairing_audit["finite_grid_eligible_systems"],
        "reference_positive_systems_B60": primary["n_reference_positive_systems"],
        "p00": primary["y00"],
        "p10": primary["y10"],
        "p01": primary["y01"],
        "p11": primary["y11"],
        "combined_gain": primary["combined_gain"],
        "interaction": primary["interaction"],
        "results_dir": str(RESULTS_DIR),
    }), indent=2))


if __name__ == "__main__":
    main()
