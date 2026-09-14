"""Generate reproducible, entirely synthetic player telemetry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the synthetic population."""

    n_players: int = 20_000
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_players < 10_000:
            raise ValueError("n_players must be at least 10,000")


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def generate_telemetry(config: SimulationConfig = SimulationConfig()) -> pd.DataFrame:
    """Return one deterministic row per synthetic player.

    Treatment is assigned before behavioural and outcome variables. Correlated
    latent propensity creates realistic signal without exposing it as a model
    feature.
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_players
    variant = rng.choice(["control", "treatment"], n)
    treatment = (variant == "treatment").astype(int)
    platform = rng.choice(["ios", "android"], n, p=[0.43, 0.57])
    country = rng.choice(["US", "GB", "DE", "BR", "JP"], n, p=[0.35, 0.16, 0.16, 0.19, 0.14])
    channel = rng.choice(["organic", "paid_social", "search", "cross_promo"], n, p=[0.37, 0.27, 0.22, 0.14])
    age_band = rng.choice(["18-24", "25-34", "35-44", "45+"], n, p=[0.27, 0.38, 0.22, 0.13])
    install_date = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 28, n), unit="D")

    affinity = rng.normal(0, 1, n)
    tutorial_p = _sigmoid(0.55 + 0.35 * treatment + 0.62 * affinity - 0.18 * (channel == "paid_social"))
    tutorial = rng.binomial(1, tutorial_p)
    sessions_d1 = np.maximum(1, rng.poisson(0.78 + 0.50 * tutorial + 0.16 * treatment + 0.20 * np.maximum(affinity, -1)))
    playtime_d1 = np.maximum(2, rng.gamma(2.0 + 0.30 * tutorial, 6.1) + 2.0 * treatment + 2.2 * affinity)
    levels_d1 = np.maximum(1, rng.poisson(1.1 + 0.14 * sessions_d1 + 0.07 * playtime_d1))
    payer_p = _sigmoid(-3.25 + 0.48 * affinity + 0.12 * treatment + 0.18 * (country == "US"))
    payer = rng.binomial(1, payer_p)
    revenue_d1 = payer * rng.lognormal(2.05, 0.68, n)

    d1_logit = -0.72 + 0.16 * treatment + 0.66 * tutorial + 0.15 * affinity + 0.035 * playtime_d1
    retained_d1 = rng.binomial(1, _sigmoid(d1_logit))
    d7_logit = -1.33 + 0.16 * treatment + 0.54 * tutorial + 0.38 * retained_d1 + 0.12 * affinity + 0.018 * playtime_d1
    retained_d7 = rng.binomial(1, _sigmoid(d7_logit))
    later_spend = rng.binomial(1, _sigmoid(-3.0 + 0.75 * retained_d7 + 0.30 * affinity)) * rng.lognormal(2.10, 0.78, n)
    revenue_d7 = revenue_d1 + later_spend

    return pd.DataFrame(
        {
            "player_id": [f"P{i:07d}" for i in range(1, n + 1)],
            "install_date": install_date,
            "variant": variant,
            "platform": platform,
            "country": country,
            "acquisition_channel": channel,
            "age_band": age_band,
            "tutorial_completed": tutorial,
            "sessions_d1": sessions_d1,
            "playtime_minutes_d1": playtime_d1.round(2),
            "levels_completed_d1": levels_d1,
            "revenue_d1": revenue_d1.round(2),
            "retained_d1": retained_d1,
            "retained_d7": retained_d7,
            "churned_d7": 1 - retained_d7,
            "revenue_d7": revenue_d7.round(2),
        }
    )
