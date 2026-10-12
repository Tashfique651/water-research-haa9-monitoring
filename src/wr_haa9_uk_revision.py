"""Audit the released UK group means without treating them as sample observations.

The public Cranfield release is an overlapping summary table, not the longitudinal
microdata needed for a sampling-policy validation. No threshold capture, bootstrap,
inferential test, weighted national mean, or synthetic individual records are made.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "haa9_revision"
OUT = ROOT / "wr_results" / "haa9_revision"
SPECIES = ["MCAA", "MBAA", "DCAA", "BCAA", "DBAA", "TCAA", "BDCAA", "DBCAA", "TBAA"]
HAA5 = ["MCAA", "MBAA", "DCAA", "DBAA", "TCAA"]
EXTRA4 = ["BCAA", "BDCAA", "DBCAA", "TBAA"]


def audit_recent_group_means() -> None:
    source = RAW / "20260407_HAA9_Survey_v2.extracted.csv"
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    assert fields is not None and len(fields) == 13
    assert len(rows) == 30, "Source change: reassess feasibility instead of silently proceeding."
    species_fields = {s: next(c for c in fields if c.startswith(f"Average of {s},")) for s in SPECIES}
    derived = []
    for idx, row in enumerate(rows, 2):
        values = {s: float(row[c]) for s, c in species_fields.items()}
        assert all(v >= 0 for v in values.values())
        h5 = sum(values[s] for s in HAA5)
        e4 = sum(values[s] for s in EXTRA4)
        h9 = h5 + e4
        derived.append({
            "source_csv_row": idx,
            "sample_type": row["Sample Type"], "source": row["Source "],
            "treatment": row["Treatment"], "disinfection": row["Disinfection"],
            **{s + "_group_mean_ug_L": values[s] for s in SPECIES},
            "sum_HAA5_group_means_ug_L": h5,
            "sum_Extra4_group_means_ug_L": e4,
            "sum_HAA9_group_means_ug_L": h9,
            "Extra4_fraction_of_summed_group_means": e4 / h9 if h9 else None,
            "unit_of_analysis": "published group mean; groups overlap",
        })
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "uk_published_group_mean_accounting.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(derived[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(derived)
    audit = {
        "dataset_doi": "10.57996/cran.ceres-2807",
        "dataset_version": "2, 2026-04-07; deposited 2026-04-08",
        "raw_representation": "Source-reader text extraction, not original byte-for-byte CSV",
        "source_extraction_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "encoding_note": "Source reader replaced micro sign in column headers with U+FFFD; numerical cells retained. Units confirmed from README as micrograms per litre. No raw-header repair was made.",
        "data_collection_start": "2020-01", "data_collection_end": "2025-05",
        "country": "United Kingdom", "overlap_with_UCMR4": "UK dataset, not the US UCMR4 programme; individual sample identity is not released",
        "n_data_rows": len(rows), "n_columns": len(fields), "n_species_mean_columns": 9,
        "has_individual_sample_records": False, "has_system_identifiers": False,
        "has_site_identifiers": False, "has_sample_collection_dates": False,
        "has_group_sample_counts": False, "has_reporting_limits": False,
        "has_repeated_observation_linkage": False,
        "supports_independent_sampling_policy_validation": False,
        "feasibility_decision": "Published composition context only. Do not treat group means as independent observations, compare their averages to screening benchmarks, infer exceedance rates, bootstrap them, or synthesize sample-level data.",
        "permitted_calculations": "Within each published row, sum the five and four specified species means. The resulting fraction is a ratio of group means, not a mean of sample-level fractions.",
        "censoring_boundary": "Unknown from released summary. No nondetect reconstruction or imputation.",
        "methods_from_readme": "Water-company LC-MS/MS methods reported UKAS-accredited or internally validated.",
        "license": "CC BY 4.0",
    }
    (OUT / "uk_data_feasibility_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(audit, indent=2))


def audit_historical_repeated_samples() -> None:
    """Feasibility only: historical UK tables contain no B60 reference positives."""
    source = RAW / "dwi_2011_table6_2_to6_5_digitized.csv"
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 80
    assert len({(r["round"], r["site"]) for r in rows}) == 80
    assert Counter(r["round"] for r in rows) == {"2": 20, "3": 20, "4": 20, "5": 20}
    derived = []
    for row in rows:
        if row["sampled"] == "not_sampled":
            assert all(row[s] == "ns" for s in SPECIES)
            continue
        sampled = datetime.strptime(row["sampled"], "%d/%m/%Y").date()
        analysed = datetime.strptime(row["analysed"], "%d/%m/%Y").date()
        hold_days = (analysed - sampled).days
        assert hold_days >= 0
        intervals = {
            s: (0.0, float(row[s][1:])) if row[s].startswith("<")
            else (float(row[s]), float(row[s])) for s in SPECIES
        }
        result = {
            "round": int(row["round"]), "site": row["site"],
            "sampled": sampled.isoformat(), "analysed": analysed.isoformat(),
            "collection_to_analysis_days": hold_days,
            "within_14_day_holding_screen": hold_days <= 14,
            "n_censored_species": sum(row[s].startswith("<") for s in SPECIES),
            "reported_HAA9_ug_L": None if row["HAA9_reported"] == "<lod" else float(row["HAA9_reported"]),
        }
        for label, species in [("HAA5", HAA5), ("Extra4", EXTRA4), ("HAA9", SPECIES)]:
            for i, bound in enumerate(["lower", "upper"]):
                result[f"{label}_{bound}_ug_L"] = round(sum(intervals[s][i] for s in species), 10)
        for bound in ["lower", "upper"]:
            assert abs(result[f"HAA9_{bound}_ug_L"] - result[f"HAA5_{bound}_ug_L"] - result[f"Extra4_{bound}_ug_L"]) < 1e-8
        for benchmark in [60, 80]:
            result[f"B{benchmark}_HAA9_certain_positive"] = result["HAA9_lower_ug_L"] > benchmark
            result[f"B{benchmark}_HAA9_possible_positive"] = result["HAA9_upper_ug_L"] > benchmark
        result["reported_minus_sum_of_detects_ug_L"] = None if result["reported_HAA9_ug_L"] is None else round(result["reported_HAA9_ug_L"] - result["HAA9_lower_ug_L"], 10)
        result["source_table"] = row["source_table"]
        result["source_printed_page"] = row["source_printed_page"]
        derived.append(result)
    with (OUT / "uk_dwi_2011_sample_accounting.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(derived[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(derived)
    within_hold = [r for r in derived if r["within_14_day_holding_screen"]]
    eligible_site_counts = Counter(r["site"] for r in within_hold)
    round_audit = []
    for sampling_round in [2, 3, 4, 5]:
        rr = [r for r in derived if r["round"] == sampling_round]
        round_audit.append({"round": sampling_round, "n_sampled": len(rr), "n_missing": 20 - len(rr),
                            "hold_day_min_from_table_dates": min(r["collection_to_analysis_days"] for r in rr),
                            "hold_day_max_from_table_dates": max(r["collection_to_analysis_days"] for r in rr),
                            "n_within_14_day_holding_screen": sum(r["within_14_day_holding_screen"] for r in rr)})
    audit = {
        "source": "WRc DWI70/2/253; UC8493/WD 0901, May 2011, Tables 6.2-6.5, printed pages 41-44; PDF pages 49-52",
        "source_url": "https://dwi-production-files.s3.eu-west-2.amazonaws.com/wp-content/uploads/2020/10/27111117/DWI70_2_253.pdf",
        "data_type": "Digitized published sample-level numerical facts, not original analytical-laboratory export",
        "digitized_csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "visual_QA": "All four table page images inspected against extracted CSV on 2026-09-19; anomalous published values retained, not silently corrected.",
        "geographic_scope": "20 anonymized UK drinking-water sites, A-T, selected to span source/quality/treatment characteristics; not a random national sample",
        "overlap_with_UCMR4": False,
        "n_scheduled_site_round_slots": len(rows), "n_actual_samples": len(derived), "n_sites": len({r["site"] for r in derived}),
        "collection_date_min": min(r["sampled"] for r in derived), "collection_date_max": max(r["sampled"] for r in derived),
        "n_within_14_day_holding_screen": len(within_hold), "n_sites_within_14_day_screen": len(eligible_site_counts),
        "sites_with_at_least_two_visits_within_14_day_screen": sorted(s for s, n in eligible_site_counts.items() if n >= 2),
        "rounds": round_audit,
        "max_reported_HAA9_ug_L_all_samples": max(r["reported_HAA9_ug_L"] for r in derived if r["reported_HAA9_ug_L"] is not None),
        "max_reconstructed_HAA9_upper_ug_L_all_samples": max(r["HAA9_upper_ug_L"] for r in derived),
        "max_reconstructed_HAA9_upper_ug_L_within_14_day_screen": max(r["HAA9_upper_ug_L"] for r in within_hold),
        "n_possible_B60_HAA9_positives_all_samples": sum(r["B60_HAA9_possible_positive"] for r in derived),
        "n_possible_B80_HAA9_positives_all_samples": sum(r["B80_HAA9_possible_positive"] for r in derived),
        "capture_sensitivity_and_superiority": None,
        "feasibility_decision": "No reference-positive samples even at the censoring upper bound for B60 or B80; threshold capture and panel advantage cannot be estimated. The 14-day screened subset has insufficient repeat support for a four-visit comparator and still no reference positives. This is a documented unsuccessful validation-source search, not evidence of zero or perfect sensitivity.",
        "holding_time_boundary": "Report method is based on EPA 557 and itself discusses its 14-day recommendation, but also reports stability tests to 68 days. Screen applies 14 days as a comparability condition, not a declaration of laboratory compliance or invalidity. Cold-chain, autosampler-residence time, UK method validation and potential source date errors are not independently verified.",
        "date_conflict": "Narrative holding ranges differ from dates printed in Tables 6.3-6.4; calculations use the explicit table dates without correction.",
        "censoring": "Shared species-wise intervals: censored value contributes [0, published limit]; detected values fixed at published rounded values. Upper bounds are conservative endpoints, not measured concentrations.",
        "rounding": "Published HAA9 totals retained separately; differences from sums of rounded species are not data corrections.",
        "license_boundary": "WRc report marked copyright 2011; full report and page images retained only as internal QA, excluded from public repository and release ZIP. Digitized numerical facts are attributed to source."
    }
    assert audit["n_possible_B60_HAA9_positives_all_samples"] == 0
    (OUT / "uk_dwi_2011_feasibility_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(audit, indent=2))


def main() -> None:
    audit_recent_group_means()
    audit_historical_repeated_samples()


if __name__ == "__main__":
    main()
