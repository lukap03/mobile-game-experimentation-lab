import numpy as np
import pytest

from src.data_generation import SimulationConfig, generate_telemetry
from src.experiment import analyze_experiment, bootstrap_difference, segment_analysis, two_proportion_ztest


def test_z_test_detects_known_positive_effect():
    effect, low, high, p_value = two_proportion_ztest([0] * 70 + [1] * 30, [0] * 55 + [1] * 45)
    assert effect == pytest.approx(0.15)
    assert low > 0
    assert high > low
    assert p_value < 0.05


def test_bootstrap_is_seeded():
    args = ([1, 2, 3], [3, 4, 5])
    assert bootstrap_difference(*args, rng=np.random.default_rng(1), iterations=50) == bootstrap_difference(*args, rng=np.random.default_rng(1), iterations=50)


def test_complete_experiment_and_segments():
    data = generate_telemetry(SimulationConfig(10_000, 42))
    result = analyze_experiment(data, bootstrap_iterations=50)
    assert set(result) == {"retained_d7", "retained_d1", "sessions_d1", "revenue_d7"}
    assert all(0 <= value["p_value"] <= 1 for value in result.values())
    segments = segment_analysis(data, ("platform",))
    assert set(segments.segment) == {"android", "ios"}
