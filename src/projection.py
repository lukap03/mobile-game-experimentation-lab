"""Efficient uncertainty-aware business-impact projection."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _bootstrap_mean_difference(
    control: np.ndarray,
    treatment: np.ndarray,
    *,
    simulations: int,
    rng: np.random.Generator,
    batch_size: int = 128,
) -> np.ndarray:
    """Bootstrap mean differences in bounded-memory batches."""
    differences = np.empty(simulations)
    for start in range(0, simulations, batch_size):
        stop = min(start + batch_size, simulations)
        batch = stop - start
        control_indices = rng.integers(0, len(control), size=(batch, len(control)))
        treatment_indices = rng.integers(
            0, len(treatment), size=(batch, len(treatment))
        )
        differences[start:stop] = (
            treatment[treatment_indices].mean(axis=1)
            - control[control_indices].mean(axis=1)
        )
    return differences


def monte_carlo_projection(
    data: pd.DataFrame,
    *,
    rollout_players: int = 1_000_000,
    simulations: int = 2_000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Project rollout effects from bootstrap sampling uncertainty.

    Binary-retention bootstrap counts are sampled exactly from their empirical
    Bernoulli distributions. Revenue means are bootstrapped in memory-bounded
    batches to retain the skew and zero inflation of player-level revenue.
    """
    if rollout_players <= 0 or simulations <= 0:
        raise ValueError("rollout_players and simulations must be positive")

    control = data[data["variant"] == "control"]
    treatment = data[data["variant"] == "treatment"]
    if control.empty or treatment.empty:
        raise ValueError("Both control and treatment groups must be non-empty")

    rng = np.random.default_rng(seed)
    n_control, n_treatment = len(control), len(treatment)
    control_rate = float(control["retained_d7"].mean())
    treatment_rate = float(treatment["retained_d7"].mean())

    retention_delta = (
        rng.binomial(n_treatment, treatment_rate, simulations) / n_treatment
        - rng.binomial(n_control, control_rate, simulations) / n_control
    )
    revenue_delta = _bootstrap_mean_difference(
        control["revenue_d7"].to_numpy(),
        treatment["revenue_d7"].to_numpy(),
        simulations=simulations,
        rng=rng,
    )

    retained = retention_delta * rollout_players
    revenue = revenue_delta * rollout_players
    return {
        "rollout_players": int(rollout_players),
        "simulations": int(simulations),
        "incremental_retained_players_mean": float(retained.mean()),
        "incremental_retained_players_ci_low": float(
            np.quantile(retained, 0.025)
        ),
        "incremental_retained_players_ci_high": float(
            np.quantile(retained, 0.975)
        ),
        "probability_positive_retention": float((retained > 0).mean()),
        "incremental_revenue_mean": float(revenue.mean()),
        "incremental_revenue_ci_low": float(np.quantile(revenue, 0.025)),
        "incremental_revenue_ci_high": float(np.quantile(revenue, 0.975)),
        "probability_positive_revenue": float((revenue > 0).mean()),
    }
