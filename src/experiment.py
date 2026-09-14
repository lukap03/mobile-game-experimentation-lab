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


def _groups(data: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    control = data.loc[data["variant"] == "control", metric].to_numpy()
    treatment = data.loc[data["variant"] == "treatment", metric].to_numpy()
    if len(control) == 0 or len(treatment) == 0:
        raise ValueError("Both control and treatment groups must be non-empty")
    return control, treatment


def two_proportion_ztest(
    control: Iterable[int], treatment: Iterable[int]
) -> tuple[float, float, float, float]:
    """Return treatment-control uplift, Wald CI bounds, and two-sided p-value."""
    c, t = np.asarray(control), np.asarray(treatment)
    if len(c) == 0 or len(t) == 0:
        raise ValueError("Both samples must be non-empty")
    pc, pt = c.mean(), t.mean()
    effect = pt - pc
    se_unpooled = np.sqrt(pc * (1 - pc) / len(c) + pt * (1 - pt) / len(t))
    pooled = (c.sum() + t.sum()) / (len(c) + len(t))
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / len(c) + 1 / len(t)))
    z = effect / se_pooled if se_pooled else 0.0
    p_value = float(2 * stats.norm.sf(abs(z)))
    return (
        float(effect),
        float(effect - 1.96 * se_unpooled),
        float(effect + 1.96 * se_unpooled),
        p_value,
    )


def bootstrap_difference(
    control: Iterable[float],
    treatment: Iterable[float],
    *,
    rng: np.random.Generator,
    iterations: int = 2_000,
) -> tuple[float, float]:
    """Return a seeded percentile-bootstrap CI for a difference in means."""
    c, t = np.asarray(control), np.asarray(treatment)
    if len(c) == 0 or len(t) == 0:
        raise ValueError("Both samples must be non-empty")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    differences = np.empty(iterations)
    for i in range(iterations):
        differences[i] = (
            rng.choice(t, len(t), replace=True).mean()
            - rng.choice(c, len(c), replace=True).mean()
        )
    return tuple(float(x) for x in np.quantile(differences, [0.025, 0.975]))


def sample_ratio_mismatch(data: pd.DataFrame, expected_treatment_share: float = 0.5) -> dict[str, float | int]:
    """Check whether observed assignment is compatible with the planned split."""
    if not 0 < expected_treatment_share < 1:
        raise ValueError("expected_treatment_share must be between zero and one")
    treatment = int((data["variant"] == "treatment").sum())
    total = int(len(data))
    if total == 0:
        raise ValueError("Experiment data must be non-empty")
    test = stats.binomtest(treatment, total, expected_treatment_share, alternative="two-sided")
    return {
        "n_players": total,
        "n_treatment": treatment,
        "treatment_share": float(treatment / total),
        "srm_p_value": float(test.pvalue),
    }


def minimum_detectable_effect(
    baseline_rate: float,
    n_per_group: int,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Approximate two-sided MDE for an absolute difference in proportions."""
    if not 0 < baseline_rate < 1 or n_per_group < 2:
        raise ValueError("baseline_rate and n_per_group are invalid")
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    standard_error = np.sqrt(2 * baseline_rate * (1 - baseline_rate) / n_per_group)
    return float((z_alpha + z_power) * standard_error)


def experiment_diagnostics(data: pd.DataFrame) -> dict[str, float | int]:
    """Return assignment-health and approximate sensitivity diagnostics."""
    srm = sample_ratio_mismatch(data)
    control, treatment = _groups(data, "retained_d7")
    baseline = float(control.mean())
    return {
        **srm,
        "d7_baseline_rate": baseline,
        "d7_mde_80_power": minimum_detectable_effect(
            baseline, min(len(control), len(treatment))
        ),
    }


def analyze_metric(
    data: pd.DataFrame,
    metric: str,
    *,
    seed: int = 42,
    bootstrap_iterations: int = 2_000,
) -> dict[str, float | int]:
    """Analyze one metric, choosing inference appropriate to its type."""
    if metric not in METRICS:
        raise ValueError(f"Unknown metric: {metric}")
    control, treatment = _groups(data, metric)
    effect = float(treatment.mean() - control.mean())
    if METRICS[metric] == "binary":
        effect, ci_low, ci_high, p_value = two_proportion_ztest(control, treatment)
    else:
        test = stats.ttest_ind(treatment, control, equal_var=False)
        se = np.sqrt(
            treatment.var(ddof=1) / len(treatment)
            + control.var(ddof=1) / len(control)
        )
        ci_low, ci_high, p_value = (
            effect - 1.96 * se,
            effect + 1.96 * se,
            float(test.pvalue),
        )
    boot_low, boot_high = bootstrap_difference(
        control,
        treatment,
        rng=np.random.default_rng(seed),
        iterations=bootstrap_iterations,
    )
    return {
        "control": float(control.mean()),
        "treatment": float(treatment.mean()),
        "uplift": effect,
        "relative_uplift": float(effect / control.mean()) if control.mean() else 0.0,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p_value,
        "bootstrap_ci_low": boot_low,
        "bootstrap_ci_high": boot_high,
        "n_control": int(len(control)),
        "n_treatment": int(len(treatment)),
    }


def analyze_experiment(
    data: pd.DataFrame,
    *,
    seed: int = 42,
    bootstrap_iterations: int = 2_000,
) -> dict[str, dict[str, float | int]]:
    """Analyze primary, secondary, and guardrail metrics."""
    return {
        metric: analyze_metric(
            data,
            metric,
            seed=seed + index,
            bootstrap_iterations=bootstrap_iterations,
        )
        for index, metric in enumerate(METRICS)
    }


def segment_analysis(
    data: pd.DataFrame,
    dimensions: tuple[str, ...] = ("platform", "country", "acquisition_channel"),
) -> pd.DataFrame:
    """Return exploratory D7 effects and intervals for each segment."""
    rows = []
    seed = 100
    for dimension in dimensions:
        for segment, frame in data.groupby(dimension, observed=True):
            result = analyze_metric(
                frame,
                "retained_d7",
                seed=seed,
                bootstrap_iterations=300,
            )
            rows.append({"dimension": dimension, "segment": segment, **result})
            seed += 1
    return pd.DataFrame(rows)
