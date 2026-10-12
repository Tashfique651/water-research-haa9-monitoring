"""Truth-table and brute-force checks for the independent specification."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from reference_implementation import (
    PATHWAYS,
    admissible_reference_positive_patterns,
    aggregate_rounds,
    brute_force_capture,
    factorial_from_units,
    hypergeometric_capture,
    monotone_pathway,
    validate_nested_panels,
)


def test_hypergeometric_formula_matches_every_small_brute_force_case() -> None:
    for total in range(0, 9):
        for positive in range(0, total + 1):
            flags = [True] * positive + [False] * (total - positive)
            for selected in range(0, total + 3):
                exact = hypergeometric_capture(total, positive, selected)
                brute = brute_force_capture(flags, selected)
                assert exact == pytest.approx(brute, abs=1e-15)


def test_hypergeometric_edges_and_invalid_counts() -> None:
    assert hypergeometric_capture(0, 0, 5) == 0.0
    assert hypergeometric_capture(5, 0, 3) == 0.0
    assert hypergeometric_capture(5, 2, 0) == 0.0
    assert hypergeometric_capture(5, 2, 5) == 1.0
    assert hypergeometric_capture(5, 5, 1) == 1.0
    with pytest.raises(ValueError):
        hypergeometric_capture(3, 4, 1)
    with pytest.raises(ValueError):
        hypergeometric_capture(3, 1, -1)


def test_nested_panel_identity_implies_four_cell_order() -> None:
    result = factorial_from_units([10.0, 65.0, 20.0], [15.0, 80.0, 70.0], 60.0)
    assert result["y00"] <= result["y10"] <= result["y11"]
    assert result["y00"] <= result["y01"] <= result["y11"]
    with pytest.raises(ValueError, match="HAA5 exceeds HAA9"):
        validate_nested_panels([12.0], [11.0])


def test_five_admissible_reference_positive_pathways_partition_truth_table() -> None:
    patterns = admissible_reference_positive_patterns()
    labels = [monotone_pathway(*pattern) for pattern in patterns]
    assert len(patterns) == 5
    assert set(labels) == set(PATHWAYS)
    assert sum(labels.count(name) for name in PATHWAYS) == 5
    for pattern in patterns:
        membership = [monotone_pathway(*pattern) == name for name in PATHWAYS]
        assert sum(membership) == 1


def test_identical_panels_have_zero_chemistry_gain_and_interaction() -> None:
    result = factorial_from_units([10.0, 70.0, 20.0, 80.0], [10.0, 70.0, 20.0, 80.0], 60.0)
    assert result["y10"] - result["y00"] == 0.0
    assert result["y11"] - result["y01"] == 0.0
    assert result["interaction"] == 0.0


def test_extra_four_only_signal_creates_panel_capture() -> None:
    result = factorial_from_units([10.0, 20.0, 30.0], [10.0, 80.0, 30.0], 60.0)
    assert result["y00"] == 0.0
    assert result["y10"] == pytest.approx(1 / 3)
    assert result["y01"] == 0.0
    assert result["y11"] == 1.0
    assert result["panel_gain_at_sparse_design"] == pytest.approx(1 / 3)
    assert result["interaction"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("haa5", "haa9", "expected"),
    [
        ([80.0, 80.0, 80.0], [90.0, 90.0, 90.0], 1.0),
        ([10.0, 10.0, 10.0], [20.0, 20.0, 20.0], 0.0),
    ],
)
def test_constant_rounds_have_no_design_or_interaction_gain(haa5, haa9, expected) -> None:
    result = factorial_from_units(haa5, haa9, 60.0)
    assert result["y00"] == expected
    assert result["y10"] == expected
    assert result["y01"] == expected
    assert result["y11"] == expected
    assert result["design_gain_for_haa5"] == 0.0
    assert result["interaction"] == 0.0


@pytest.mark.parametrize(
    ("haa5", "haa9"),
    [([80.0], [90.0]), ([10.0], [80.0]), ([10.0], [20.0])],
)
def test_one_round_edge_has_no_panel_by_design_interaction(haa5, haa9) -> None:
    result = factorial_from_units(haa5, haa9, 60.0)
    assert result["y00"] == result["y01"]
    assert result["y10"] == result["y11"]
    assert result["interaction"] == 0.0


def test_all_negative_system_has_no_reference_positive_pathway() -> None:
    result = factorial_from_units([0.0, 60.0, 12.0], [1.0, 60.0, 15.0], 60.0)
    assert result["y00"] == result["y10"] == result["y01"] == result["y11"] == 0.0
    assert result["pathway_sum"] == 0.0
    assert result["pathway_reference_negative"] == 1.0
    assert all(result[f"pathway_{name}"] == 0.0 for name in PATHWAYS)


def test_constructed_positive_interaction() -> None:
    result = factorial_from_units([0.0, 0.0], [80.0, 0.0], 60.0)
    assert result["interaction"] == pytest.approx(0.5)
    assert result["pathway_panel_only"] == pytest.approx(0.5)
    assert result["pathway_both_required"] == pytest.approx(0.5)


def test_constructed_negative_interaction() -> None:
    result = factorial_from_units([80.0, 0.0], [80.0, 80.0], 60.0)
    assert result["interaction"] == pytest.approx(-0.5)
    assert result["pathway_already_detected"] == pytest.approx(0.5)
    assert result["pathway_either_change"] == pytest.approx(0.5)


def test_reference_positive_pathway_fractions_sum_to_one() -> None:
    cases = [
        ([80.0, 0.0], [90.0, 0.0]),
        ([0.0, 0.0, 0.0], [0.0, 75.0, 0.0]),
        ([80.0, 0.0], [80.0, 80.0]),
    ]
    for haa5, haa9 in cases:
        result = factorial_from_units(haa5, haa9, 60.0)
        assert result["y11"] == 1.0
        assert math.fsum(result[f"pathway_{name}"] for name in PATHWAYS) == pytest.approx(1.0)


def test_strict_threshold_treats_equality_as_negative() -> None:
    result = factorial_from_units([60.0, 60.0001], [60.0, 60.0001], 60.0)
    assert result["y00"] == pytest.approx(0.5)
    assert result["y01"] == 1.0


def test_event_code_and_year_split_round_semantics_are_distinct() -> None:
    events = pd.DataFrame(
        {
            "PWSID": ["P1", "P1", "P1", "P2"],
            "SampleEventCode": ["E1", "E1", "E2", "E1"],
            "CollectionDate": ["2018-01-10", "2019-01-10", "2019-04-10", "2019-01-10"],
            "HAA5": [10.0, 70.0, 20.0, 5.0],
            "HAA9": [20.0, 80.0, 30.0, 6.0],
        }
    )
    code = aggregate_rounds(events, "event_code")
    code_year = aggregate_rounds(events, "event_code_year")

    assert len(code) == 3
    assert len(code_year) == 4
    p1e1 = code.loc[(code["PWSID"] == "P1") & (code["SampleEventCode"] == "E1")].iloc[0]
    assert p1e1["physical_rows"] == 2
    assert p1e1["distinct_dates"] == 2
    assert p1e1["HAA5_max"] == 70.0
    assert p1e1["HAA9_max"] == 80.0
    assert set(code_year.loc[code_year["PWSID"] == "P1", "collection_year"]) == {2018, 2019}


def test_nonmonotone_truth_patterns_are_rejected() -> None:
    with pytest.raises(ValueError, match="not monotone"):
        monotone_pathway(True, False, True, True)
    with pytest.raises(ValueError, match="not monotone"):
        monotone_pathway(False, True, False, False)

