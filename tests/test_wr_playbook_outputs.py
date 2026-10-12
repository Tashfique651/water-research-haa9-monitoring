from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "wr_results" / "playbook_extension"


def test_primary_and_transfer_estimands() -> None:
    frame = pd.read_csv(OUT / "interaction_scale_metrics.csv").set_index("analysis")
    primary = frame.loc["primary_complete_observed_grid_B60"]
    assert int(primary.systems) == 769
    assert math.isclose(primary.additive_interaction, 0.21228868660598182, abs_tol=1e-12)
    assert primary.additive_interaction_ci_lower > 0
    assert frame.loc["late2_condition_early2_evaluation", "additive_interaction"] > 0
    assert frame.loc["early2_condition_late2_evaluation", "additive_interaction"] > 0


def test_equivalence_is_paired_and_uncertain() -> None:
    summary = json.loads((OUT / "equivalence_summary.json").read_text(encoding="utf-8"))
    assert summary["minimum_integer_HAA9_samples_matching_full_HAA5_capture"] == 4
    assert math.isclose(
        summary["paired_difference_k4_minus_full_HAA5"],
        summary["HAA9_capture_at_k4"] - summary["full_HAA5_record_capture"],
        abs_tol=1e-12,
    )
    assert summary["paired_difference_k4_ci_lower"] < 0 < summary["paired_difference_k4_ci_upper"]
    assert summary["critical_cost_ratio_ci_lower"] < summary["critical_HAA9_to_HAA5_cost_ratio"] < summary["critical_cost_ratio_ci_upper"]


def test_endpoint_support_and_panel_order() -> None:
    frame = pd.read_csv(OUT / "regulation_shaped_endpoint_summary.csv")
    expected_support = {"single_sample_maximum": 4904, "OEL_analogue": 2683, "LRAA_analogue": 2487}
    for endpoint, support in expected_support.items():
        subset = frame.loc[frame.endpoint.eq(endpoint)].set_index("panel")
        assert subset.supported_systems.nunique() == 1
        assert int(subset.supported_systems.iloc[0]) == support
        assert subset.loc["HAA5", "positive_systems"] <= subset.loc["HAA5_plus_BCAA", "positive_systems"] <= subset.loc["HAA9", "positive_systems"]


def test_population_and_geography_coverage_are_explicit() -> None:
    population = json.loads((OUT / "population_weighting_coverage.json").read_text(encoding="utf-8"))
    assert population["reference_positive_systems"] == 769
    assert population["adjusted_population_available_positive_systems"] == 587
    geography = json.loads((OUT / "state_map_coverage.json").read_text(encoding="utf-8"))
    assert geography["systems_with_valid_50_state_or_DC_code"] == 709
    states = pd.read_csv(OUT / "state_map_values.csv", dtype={"state_profile": str})
    assert states.state_profile.str.fullmatch(r"[A-Z]{2}").all()
    assert not states.state_profile.isin(["PR", "VI", "GU", "MP"]).any()


def test_mechanism_cv_and_negative_control() -> None:
    mechanism = pd.read_parquet(OUT / "mechanism_systems.parquet")
    assert int(mechanism.haa9_round_cv_eligible.sum()) == 765
    assert mechanism.loc[mechanism.haa9_round_cv_eligible, "haa9_round_cv"].notna().all()
    control = pd.read_csv(OUT / "negative_controls.csv").iloc[0]
    assert math.isclose(control.additive_interaction, 0.0, abs_tol=1e-12)


def test_operational_policy_contrast() -> None:
    contrasts = pd.read_csv(OUT / "round_policy_contrasts.csv").set_index("policy")
    warm = contrasts.loc["warm_season_May_to_September_else_random"]
    assert warm.HAA9_detection_difference > 0
    assert warm.HAA9_difference_ci_lower > 0
