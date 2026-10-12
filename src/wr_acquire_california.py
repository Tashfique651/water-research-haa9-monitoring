"""Acquire and verify the California SDWIS4 archive used for validation."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "additional" / "california"
ARCHIVE = OUT / "SDWIS4.zip"
EXTRACTED = OUT / "sdwis4" / "SDWIS4.tab"
URL = "https://www.waterboards.ca.gov/drinking_water/certlic/drinkingwater/documents/edtlibrary/SDWIS4.zip"
ARCHIVE_SHA256 = "a4d8543490ff9d42ca180fcb5851c8d1e44793647531188350ee9e6fff2e1766"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists() and sha256(ARCHIVE) == ARCHIVE_SHA256:
        return
    temporary = ARCHIVE.with_suffix(".zip.part")
    request = urllib.request.Request(URL, headers={"User-Agent": "DBP-monitoring-reproducibility/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    observed = sha256(temporary)
    if observed != ARCHIVE_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"California archive checksum mismatch: {observed}")
    temporary.replace(ARCHIVE)


def extract() -> None:
    if EXTRACTED.exists():
        return
    EXTRACTED.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        members = [name for name in archive.namelist() if Path(name).name.lower() == "sdwis4.tab"]
        if len(members) != 1:
            raise RuntimeError(f"Expected one SDWIS4.tab member, found {len(members)}")
        member = archive.getinfo(members[0])
        destination = EXTRACTED.resolve()
        if EXTRACTED.parent.resolve() not in destination.parents:
            raise RuntimeError("Resolved extraction target is outside the intended directory")
        with archive.open(member) as source, EXTRACTED.open("wb") as output:
            shutil.copyfileobj(source, output, length=8 * 1024 * 1024)


def main() -> None:
    download()
    extract()
    print(f"Archive: {ARCHIVE} ({sha256(ARCHIVE)})")
    print(f"Extracted: {EXTRACTED} ({sha256(EXTRACTED)})")


if __name__ == "__main__":
    main()
