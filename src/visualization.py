"""Publication-ready figures for experiment, model, and rollout results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def save_experiment_figure(
    results: dict[str, dict[str, float | int]],
    output: Path,
) -> None:
    """Save control-versus-treatment retention rates."""
    labels = ["D7 retention", "D1 retention"]
    keys = ["retained_d7", "retained_d1"]
    control = [100 * float(results[key]["control"]) for key in keys]
    treatment = [100 * float(results[key]["treatment"]) for key in keys]

    figure, axis = plt.subplots(figsize=(7, 4))
    x = range(len(keys))
    axis.bar(
        [index - 0.18 for index in x],
        control,
        width=0.36,
        label="Control",
        color="#64748b",
    )
    axis.bar(
        [index + 0.18 for index in x],
        treatment,
        width=0.36,
        label="Treatment",
        color="#22c55e",
    )
    axis.set(
        xticks=list(x),
        xticklabels=labels,
        ylabel="Players retained (%)",
        title="Randomized experiment outcomes",
    )
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def save_model_figure(scored: pd.DataFrame, output: Path) -> None:
    """Save holdout ROC and precision-recall curves."""
    target = scored["churned_d7"]
    probability = scored["churn_probability"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(target, probability, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(target, probability, ax=axes[1])
    axes[0].set_title("D7 churn ROC curve")
    axes[1].set_title("D7 churn precision-recall curve")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def save_projection_figure(
    projection: dict[str, float | int],
    output: Path,
) -> None:
    """Save mean rollout effects with 95% simulation intervals."""
    retained_mean = float(projection["incremental_retained_players_mean"])
    retained_low = float(projection["incremental_retained_players_ci_low"])
    retained_high = float(projection["incremental_retained_players_ci_high"])
    revenue_mean = float(projection["incremental_revenue_mean"])
    revenue_low = float(projection["incremental_revenue_ci_low"])
    revenue_high = float(projection["incremental_revenue_ci_high"])

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].errorbar(
        [0],
        [retained_mean],
        yerr=[[retained_mean - retained_low], [retained_high - retained_mean]],
        fmt="o",
        capsize=6,
        color="#2563eb",
    )
    axes[0].axhline(0, color="#64748b", linestyle="--", linewidth=1)
    axes[0].set(
        xticks=[0],
        xticklabels=["Treatment − control"],
        ylabel="Incremental D7 retained players",
        title="Retention rollout projection",
    )

    axes[1].errorbar(
        [0],
        [revenue_mean],
        yerr=[[revenue_mean - revenue_low], [revenue_high - revenue_mean]],
        fmt="o",
        capsize=6,
        color="#16a34a",
    )
    axes[1].axhline(0, color="#64748b", linestyle="--", linewidth=1)
    axes[1].set(
        xticks=[0],
        xticklabels=["Treatment − control"],
        ylabel="Incremental D7 revenue (USD)",
        title="Revenue rollout projection",
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)
