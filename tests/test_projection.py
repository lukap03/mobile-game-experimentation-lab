from src.data_generation import SimulationConfig, generate_telemetry
from src.projection import monte_carlo_projection


def test_projection_is_reproducible():
    data = generate_telemetry(SimulationConfig(10_000, 10))
    first = monte_carlo_projection(data, rollout_players=100_000, simulations=30, seed=4)
    second = monte_carlo_projection(data, rollout_players=100_000, simulations=30, seed=4)
    assert first == second
    assert 0 <= first["probability_positive_retention"] <= 1
    assert first["incremental_retained_players_ci_low"] <= first["incremental_retained_players_ci_high"]
