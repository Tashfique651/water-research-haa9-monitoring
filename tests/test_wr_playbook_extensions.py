from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wr_playbook_extensions import derived_metrics, hypergeom_capture


def test_derived_metrics_additive_identity() -> None:
    metrics = derived_metrics(np.array([0.20, 0.40, 0.55, 1.00]))
    assert math.isclose(metrics["additive_interaction"], 0.25)
    assert math.isclose(metrics["relative_excess_risk_due_to_interaction"], 1.25)
    assert math.isclose(metrics["attributable_proportion_due_to_interaction"], 0.25)


def test_duplicate_panel_has_zero_interaction() -> None:
    metrics = derived_metrics(np.array([0.3, 0.3, 0.7, 0.7]))
    assert math.isclose(metrics["additive_interaction"], 0.0, abs_tol=1e-12)
    assert math.isclose(metrics["multiplicative_interaction_ratio"], 1.0, abs_tol=1e-12)


def test_hypergeometric_capture_boundaries() -> None:
    assert hypergeom_capture(10, 2, 0) == 0.0
    assert hypergeom_capture(10, 0, 5) == 0.0
    assert hypergeom_capture(10, 2, 10) == 1.0
    expected = 1 - math.comb(8, 1) / math.comb(10, 1)
    assert math.isclose(hypergeom_capture(10, 2, 1), expected)
