"""Frequentist and bootstrap analysis for the randomized experiment."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

METRICS = {
    "retained_d7": "binary",
    "retained_d1": "binary",
    "sessions_d1": "continuous",
    "revenue_d7": "continuous",
}


def two_proportion_ztest(control: Iterable[int], treatment: Iterable[int]) -> tuple[float, float, float, float]:
    """Return treatment-control uplift, CI bounds, and two-sided p-value."""
    c, t = np.asarray(control), np.asarray(treatment)
    pc, pt = c.mean(), t.mean()
    effect = pt - pc
    se_unpooled = np.sqrt(pc * (1 - pc) / len(c) + pt * (1 - pt) / len(t))
    pooled = (c.sum() + t.sum()) / (len(c) + len(t))
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / len(c) + 1 / len(t)))
    z = effect / se_pooled if se_pooled else 0.0
    p_value = float(2 * stats.norm.sf(abs(z)))
    return float(effect), float(effect - 1.96 * se_unpooled), float(effect + 1.96 * se_unpooled), p_value


def bootstrap_difference(
    control: Iterable[float], treatment: Iterable[float], *, rng: np.random.Generator, iterations: int = 2_000
) -> tuple[float, float]:
    """Bootstrap a percentile interval for a difference in means."""
    c, t = np.asarray(control), np.asarray(treatment)
    differences = np.empty(iterations)
    for i in range(iterations):
        differences[i] = rng.choice(t, len(t), replace=True).mean() - rng.choice(c, len(c), replace=True).mean()
    return tuple(float(x) for x in np.quantile(differences, [0.025, 0.975]))


def analyze_metric(data: pd.DataFrame, metric: str, *, seed: int = 42, bootstrap_iterations: int = 2_000) -> dict[str, float]:
    """Analyze one metric, choosing inference appropriate to its type."""
    if metric not in METRICS:
        raise ValueError(f"Unknown metric: {metric}")
    control = data.loc[data["variant"] == "control", metric].to_numpy()
    treatment = data.loc[data["variant"] == "treatment", metric].to_numpy()
    effect = float(treatment.mean() - control.mean())
    if METRICS[metric] == "binary":
        effect, ci_low, ci_high, p_value = two_proportion_ztest(control, treatment)
    else:
        test = stats.ttest_ind(treatment, control, equal_var=False)
        se = np.sqrt(treatment.var(ddof=1) / len(treatment) + control.var(ddof=1) / len(control))
        ci_low, ci_high, p_value = effect - 1.96 * se, effect + 1.96 * se, float(test.pvalue)
    boot_low, boot_high = bootstrap_difference(control, treatment, rng=np.random.default_rng(seed), iterations=bootstrap_iterations)
    return {
        "control": float(control.mean()), "treatment": float(treatment.mean()), "uplift": effect,
        "relative_uplift": float(effect / control.mean()) if control.mean() else 0.0,
        "ci_low": float(ci_low), "ci_high": float(ci_high), "p_value": p_value,
        "bootstrap_ci_low": boot_low, "bootstrap_ci_high": boot_high,
        "n_control": int(len(control)), "n_treatment": int(len(treatment)),
    }


def analyze_experiment(data: pd.DataFrame, *, seed: int = 42, bootstrap_iterations: int = 2_000) -> dict[str, dict[str, float]]:
    """Analyze primary, secondary, and guardrail metrics."""
    return {m: analyze_metric(data, m, seed=seed + i, bootstrap_iterations=bootstrap_iterations) for i, m in enumerate(METRICS)}


def segment_analysis(data: pd.DataFrame, dimensions: tuple[str, ...] = ("platform", "country", "acquisition_channel")) -> pd.DataFrame:
    """Return exploratory D7 effects and normal intervals for each segment."""
    rows = []
    for dimension in dimensions:
        for segment, frame in data.groupby(dimension, observed=True):
            result = analyze_metric(frame, "retained_d7", bootstrap_iterations=300)
            rows.append({"dimension": dimension, "segment": segment, **result})
    return pd.DataFrame(rows)
