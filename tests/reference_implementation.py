"""Independent, compact reference calculations for monitoring-design tests.

This module intentionally does not import any production analysis module.  It is a
small executable specification of the estimands used by the project, suitable for
checking optimized implementations against brute-force enumeration.
"""

from __future__ import annotations

from itertools import combinations, product
from math import comb
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


PATHWAYS = (
    "already_detected",
    "panel_only",
    "design_only",
    "either_change",
    "both_required",
)


def hypergeometric_capture(total: int, positive: int, selected: int) -> float:
    """Probability that a uniform subset contains at least one positive unit."""

    if not all(isinstance(value, (int, np.integer)) for value in (total, positive, selected)):
        raise TypeError("total, positive, and selected must be integers")
    if total < 0 or positive < 0 or selected < 0:
        raise ValueError("counts must be nonnegative")
    if positive > total:
        raise ValueError("positive cannot exceed total")

    selected = min(selected, total)
    if total == 0 or selected == 0 or positive == 0:
        return 0.0
    if total - positive < selected:
        return 1.0
    return 1.0 - comb(total - positive, selected) / comb(total, selected)


def brute_force_capture(flags: Sequence[bool], selected: int) -> float:
    """Enumerate every equally likely subset and score any-positive capture."""

    if not isinstance(selected, (int, np.integer)):
        raise TypeError("selected must be an integer")
    if selected < 0:
        raise ValueError("selected must be nonnegative")

    values = tuple(bool(value) for value in flags)
    selected = min(selected, len(values))
    if selected == 0 or not values:
        return 0.0

    subsets = tuple(combinations(range(len(values)), selected))
    captured = sum(any(values[index] for index in subset) for subset in subsets)
    return captured / len(subsets)


def validate_nested_panels(
    haa5: Iterable[float], haa9: Iterable[float], tolerance: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite arrays after enforcing HAA5 <= HAA9 element by element."""

    h5 = np.asarray(tuple(haa5), dtype=float)
    h9 = np.asarray(tuple(haa9), dtype=float)
    if h5.ndim != 1 or h9.ndim != 1 or h5.shape != h9.shape:
        raise ValueError("HAA5 and HAA9 must be equal-length one-dimensional sequences")
    if h5.size == 0:
        raise ValueError("at least one observed unit is required")
    if not np.isfinite(h5).all() or not np.isfinite(h9).all():
        raise ValueError("panel concentrations must be finite")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    if np.any(h5 > h9 + tolerance):
        raise ValueError("nested-panel identity violated: HAA5 exceeds HAA9")
    return h5, h9


def monotone_pathway(y00: bool, y10: bool, y01: bool, y11: bool) -> str:
    """Classify one admissible Boolean 2 x 2 outcome into its unique pathway."""

    a, b, c, d = map(bool, (y00, y10, y01, y11))
    if a > b or a > c or b > d or c > d:
        raise ValueError("outcome is not monotone under nested panel and expanded design")
    if not d:
        return "reference_negative"
    if a:
        return "already_detected"
    if b and not c:
        return "panel_only"
    if c and not b:
        return "design_only"
    if b and c:
        return "either_change"
    return "both_required"


def admissible_reference_positive_patterns() -> list[tuple[bool, bool, bool, bool]]:
    """Enumerate the five monotone truth patterns with full HAA9 positivity."""

    patterns: list[tuple[bool, bool, bool, bool]] = []
    for pattern in product((False, True), repeat=4):
        try:
            pathway = monotone_pathway(*pattern)
        except ValueError:
            continue
        if pathway != "reference_negative":
            patterns.append(pattern)
    return patterns


def factorial_contrasts(y00: float, y10: float, y01: float, y11: float) -> dict[str, float]:
    """Calculate the four-cell gains and additive interaction."""

    values = np.asarray([y00, y10, y01, y11], dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("cell probabilities must be finite and in [0, 1]")
    return {
        "y00": float(y00),
        "y10": float(y10),
        "y01": float(y01),
        "y11": float(y11),
        "panel_gain_at_sparse_design": float(y10 - y00),
        "design_gain_for_haa5": float(y01 - y00),
        "combined_gain": float(y11 - y00),
        "interaction": float(y11 - y10 - y01 + y00),
    }


def factorial_from_units(
    haa5: Iterable[float], haa9: Iterable[float], threshold: float
) -> dict[str, float]:
    """Exact system-equal factorial estimand for uniform one-unit selection.

    The expanded design is the maximum over the same finite set of observed units.
    Positivity is deliberately strict (concentration > threshold).
    """

    h5, h9 = validate_nested_panels(haa5, haa9)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    sparse_h5 = h5 > threshold
    sparse_h9 = h9 > threshold
    full_h5 = bool(np.any(sparse_h5))
    full_h9 = bool(np.any(sparse_h9))

    result = factorial_contrasts(
        float(np.mean(sparse_h5)),
        float(np.mean(sparse_h9)),
        float(full_h5),
        float(full_h9),
    )

    pathway_counts = {name: 0 for name in PATHWAYS}
    reference_negative = 0
    for h5_flag, h9_flag in zip(sparse_h5, sparse_h9, strict=True):
        label = monotone_pathway(bool(h5_flag), bool(h9_flag), full_h5, full_h9)
        if label == "reference_negative":
            reference_negative += 1
        else:
            pathway_counts[label] += 1

    denominator = h5.size
    result.update({f"pathway_{name}": count / denominator for name, count in pathway_counts.items()})
    result["pathway_reference_negative"] = reference_negative / denominator
    result["pathway_sum"] = sum(pathway_counts.values()) / denominator
    result["n_units"] = float(denominator)
    return result


def aggregate_rounds(events: pd.DataFrame, definition: str) -> pd.DataFrame:
    """Aggregate exact rows under two independently specified round semantics."""

    required = {"PWSID", "SampleEventCode", "CollectionDate", "HAA5", "HAA9"}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    if definition not in {"event_code", "event_code_year"}:
        raise ValueError("definition must be 'event_code' or 'event_code_year'")

    data = events.loc[:, sorted(required)].copy()
    data["CollectionDate"] = pd.to_datetime(data["CollectionDate"], errors="raise")
    validate_nested_panels(data["HAA5"], data["HAA9"])

    keys = ["PWSID", "SampleEventCode"]
    if definition == "event_code_year":
        data["collection_year"] = data["CollectionDate"].dt.year.astype(int)
        keys.insert(1, "collection_year")

    aggregated = (
        data.groupby(keys, dropna=False, sort=True)
        .agg(
            HAA5_max=("HAA5", "max"),
            HAA9_max=("HAA9", "max"),
            physical_rows=("CollectionDate", "size"),
            distinct_dates=("CollectionDate", "nunique"),
            first_date=("CollectionDate", "min"),
            last_date=("CollectionDate", "max"),
        )
        .reset_index()
    )
    return aggregated.sort_values(keys, kind="stable").reset_index(drop=True)

