"""Acquire and verify the public data used in the DBP monitoring study.

The script downloads immutable source archives, checks their SHA-256 digests,
and extracts only the files required by the analyses. Existing verified files
are reused. It does not modify source data after extraction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EXTRACTED = ROOT / "data" / "interim" / "extracted"


@dataclass(frozen=True)
class Source:
    filename: str
    url: str
    sha256: str
    publisher: str
    coverage: str
    selected_members: tuple[tuple[str, str], ...]


SOURCES = (
    Source(
        "ucmr4-occurrence-data.zip",
        "https://www.epa.gov/system/files/other-files/2024-04/ucmr4-occurrence-data.zip",
        "932dd46139c1aef050d020054c546fea4178b97bcf12a4e455c050074d17b255",
        "US Environmental Protection Agency",
        "UCMR 4 occurrence data, 2018-2020, March 2024 revision",
        (
            ("UCMR4_All.txt", "UCMR4_All.txt"),
            ("UCMR4_HAA_AddtlDataElem.txt", "UCMR4_HAA_AddtlDataElem.txt"),
            ("UCMR4_HAA_Indicators.txt", "UCMR4_HAA_Indicators.txt"),
            ("UCMR4_ZIPCodes.txt", "UCMR4_ZIPCodes.txt"),
        ),
    ),
    Source(
        "syr4_haas.zip",
        "https://www.epa.gov/system/files/other-files/2024-03/syr4_haas.zip",
        "8bc74a7ad488928454f96681db933d7c7bf3ab894b9e274a9e9fe4b511ed9926",
        "US Environmental Protection Agency",
        "Fourth Six-Year Review compliance monitoring data, HAA5, 2012-2019",
        (("SYR4_HAAs/HALOACETIC ACIDS (HAA5).txt", "SYR4_HAA5.txt"),),
    ),
    Source(
        "syr4_thms.zip",
        "https://www.epa.gov/system/files/other-files/2024-03/syr4_thms.zip",
        "3b2a8a44b27a1e94e053d14c350fd7fad7d256d554ed2818de3317e06af81068",
        "US Environmental Protection Agency",
        "Fourth Six-Year Review compliance monitoring data, TTHM, 2012-2019",
        (("SYR4_THMs/TOTAL TRIHALOMETHANES (TTHM).txt", "SYR4_TTHM.txt"),),
    ),
    Source(
        "syr4_dbp_related-parameters.zip",
        "https://www.epa.gov/system/files/other-files/2024-03/syr4_dbp_related-parameters.zip",
        "565fbb6e6fa3befe312ea4b2aec74e93cdadd4eb33a4f690762ad3678e639615",
        "US Environmental Protection Agency",
        "Fourth Six-Year Review DBP-related parameters, 2012-2019",
        (("SYR4_DBP_Related Parameters/Paired TOC-Alkalinity.txt", "SYR4_Paired_TOC_Alkalinity.txt"),),
    ),
    Source(
        "syr4_disinfectant-residuals.zip",
        "https://www.epa.gov/system/files/other-files/2024-03/syr4_disinfectant-residuals.zip",
        "ef8545b0799cf2a72c883c8564919044dc4ec1a101b190507a16b8e0a248bcb2",
        "US Environmental Protection Agency",
        "Fourth Six-Year Review disinfectant residuals, 2012-2019",
        (),
    ),
    Source(
        "syr4_treatment.zip",
        "https://www.epa.gov/system/files/other-files/2024-03/syr4_treatment.zip",
        "37300249a7b2201ac5d143b99f5ff3984933b7ab48aa6234f6a9fceed239eeb3",
        "US Environmental Protection Agency",
        "Fourth Six-Year Review treatment and water-system tables",
        (("SYR4_Water_system_table.txt", "SYR4_Water_system_table.txt"),),
    ),
    Source(
        "dbpicr_aux1_updated.zip",
        "https://www.epa.gov/sites/default/files/2017-07/dbpicr_aux1_updated.zip",
        "d91454d34d535f8ce36905dfad4d9c9f40b3a64f912be1a3c7edcc62abf09328",
        "US Environmental Protection Agency",
        "Information Collection Rule auxiliary database, 1997-1998",
        (
            ("DBPICR_Aux1.accdb", "DBPICR_Aux1.accdb"),
            ("DBPICR_Aux1_User_Guide.pdf", "DBPICR_Aux1_User_Guide.pdf"),
        ),
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(source: Source, overwrite: bool) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    destination = RAW / source.filename
    if destination.exists() and not overwrite:
        actual = sha256(destination)
        if actual == source.sha256:
            return destination
        raise RuntimeError(f"Digest mismatch for existing file: {destination}")

    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(
        source.url,
        headers={"User-Agent": "DBP-monitoring-reproducibility/1.0"},
    )
    with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    actual = sha256(partial)
    if actual != source.sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded digest mismatch for {source.filename}: {actual}")
    partial.replace(destination)
    return destination


def extract(source: Source, archive: Path, overwrite: bool) -> None:
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        available = set(bundle.namelist())
        for member, target_name in source.selected_members:
            if member not in available:
                raise KeyError(f"Archive member not found: {member}")
            target = EXTRACTED / target_name
            if target.exists() and not overwrite:
                continue
            with bundle.open(member) as source_handle, target.open("wb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)


def write_manifest(rows: list[dict[str, str]]) -> None:
    manifest = RAW / "source_manifest.csv"
    fields = [
        "filename",
        "publisher",
        "coverage",
        "source_url",
        "sha256",
        "bytes",
        "verified_at_utc",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for source in SOURCES:
        archive = RAW / source.filename if args.verify_only else download(source, args.overwrite)
        if not archive.exists():
            raise FileNotFoundError(archive)
        actual = sha256(archive)
        if actual != source.sha256:
            raise RuntimeError(f"Digest mismatch for {source.filename}: {actual}")
        if not args.verify_only:
            extract(source, archive, args.overwrite)
        rows.append(
            {
                "filename": source.filename,
                "publisher": source.publisher,
                "coverage": source.coverage,
                "source_url": source.url,
                "sha256": actual,
                "bytes": str(archive.stat().st_size),
                "verified_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
        print(f"verified {source.filename} {actual}")
    write_manifest(rows)


if __name__ == "__main__":
    main()
