import pandas as pd
import pytest

from src.data_generation import SimulationConfig, generate_telemetry


def test_generator_is_deterministic_and_has_expected_schema():
    first = generate_telemetry(SimulationConfig(10_000, 7))
    second = generate_telemetry(SimulationConfig(10_000, 7))
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == first.player_id.nunique() == 10_000
    assert {"retained_d1", "retained_d7", "revenue_d7", "variant"}.issubset(first.columns)
    assert set(first.variant) == {"control", "treatment"}
    assert 0.47 < (first.variant == "treatment").mean() < 0.53


def test_minimum_population_is_enforced():
    with pytest.raises(ValueError, match="at least 10,000"):
        SimulationConfig(9_999)
