"""Monte Carlo business-impact projection."""

from __future__ import annotations

import numpy as np
import pandas as pd


def monte_carlo_projection(data: pd.DataFrame, *, rollout_players: int = 1_000_000, simulations: int = 10_000, seed: int = 42) -> dict[str, float]:
    """Project rollout effects by bootstrapping players within variants."""
    if rollout_players <= 0 or simulations <= 0:
        raise ValueError("rollout_players and simulations must be positive")
    rng = np.random.default_rng(seed)
    control = data[data["variant"] == "control"]
    treatment = data[data["variant"] == "treatment"]
    retention_delta, revenue_delta = np.empty(simulations), np.empty(simulations)
    for i in range(simulations):
        c = control.iloc[rng.integers(0, len(control), len(control))]
        t = treatment.iloc[rng.integers(0, len(treatment), len(treatment))]
        retention_delta[i] = t["retained_d7"].mean() - c["retained_d7"].mean()
        revenue_delta[i] = t["revenue_d7"].mean() - c["revenue_d7"].mean()
    retained = retention_delta * rollout_players
    revenue = revenue_delta * rollout_players
    return {
        "rollout_players": int(rollout_players), "simulations": int(simulations),
        "incremental_retained_players_mean": float(retained.mean()),
        "incremental_retained_players_ci_low": float(np.quantile(retained, 0.025)),
        "incremental_retained_players_ci_high": float(np.quantile(retained, 0.975)),
        "probability_positive_retention": float((retained > 0).mean()),
        "incremental_revenue_mean": float(revenue.mean()),
        "incremental_revenue_ci_low": float(np.quantile(revenue, 0.025)),
        "incremental_revenue_ci_high": float(np.quantile(revenue, 0.975)),
        "probability_positive_revenue": float((revenue > 0).mean()),
    }
