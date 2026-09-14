"""Publication-ready figures for the experiment and churn model."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def save_experiment_figure(results: dict[str, dict[str, float]], output: Path) -> None:
    labels = ["D7 retention", "D1 retention"]
    keys = ["retained_d7", "retained_d1"]
    control = [100 * results[k]["control"] for k in keys]
    treatment = [100 * results[k]["treatment"] for k in keys]
    figure, axis = plt.subplots(figsize=(7, 4))
    x = range(len(keys))
    axis.bar([i - 0.18 for i in x], control, width=0.36, label="Control")
    axis.bar([i + 0.18 for i in x], treatment, width=0.36, label="Treatment")
    axis.set(xticks=list(x), xticklabels=labels, ylabel="Players retained (%)", title="Randomized experiment outcomes")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def save_model_figure(model, test_features: pd.DataFrame, target: pd.Series, output: Path) -> None:
    probability = model.predict_proba(test_features)[:, 1]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(target, probability, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(target, probability, ax=axes[1])
    axes[0].set_title("D7 churn ROC curve")
    axes[1].set_title("D7 churn precision-recall curve")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
