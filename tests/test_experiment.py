import numpy as np
import pytest

from src.data_generation import SimulationConfig, generate_telemetry
from src.experiment import (
    analyze_experiment,
    bootstrap_difference,
    experiment_diagnostics,
    minimum_detectable_effect,
    sample_ratio_mismatch,
    segment_analysis,
    two_proportion_ztest,
)


def test_z_test_detects_known_positive_effect():
    effect, low, high, p_value = two_proportion_ztest(
        [0] * 70 + [1] * 30,
        [0] * 55 + [1] * 45,
    )
    assert effect == pytest.approx(0.15)
    assert low > 0
    assert high > low
    assert p_value < 0.05


def test_bootstrap_is_seeded():
    args = ([1, 2, 3], [3, 4, 5])
    first = bootstrap_difference(
        *args,
        rng=np.random.default_rng(1),
        iterations=50,
    )
    second = bootstrap_difference(
        *args,
        rng=np.random.default_rng(1),
        iterations=50,
    )
    assert first == second


def test_assignment_diagnostics_are_valid():
    data = generate_telemetry(SimulationConfig(10_000, 42))
    srm = sample_ratio_mismatch(data)
    diagnostics = experiment_diagnostics(data)
    assert 0 <= srm["srm_p_value"] <= 1
    assert 0.47 < srm["treatment_share"] < 0.53
    assert diagnostics["d7_mde_80_power"] > 0
    assert minimum_detectable_effect(0.30, 5_000) == pytest.approx(
        diagnostics["d7_mde_80_power"],
        rel=0.12,
    )


def test_complete_experiment_and_segments():
    data = generate_telemetry(SimulationConfig(10_000, 42))
    result = analyze_experiment(data, bootstrap_iterations=50)
    assert set(result) == {
        "retained_d7",
        "retained_d1",
        "sessions_d1",
        "revenue_d7",
    }
    assert all(0 <= value["p_value"] <= 1 for value in result.values())
    segments = segment_analysis(data, ("platform",))
    assert set(segments.segment) == {"android", "ios"}
