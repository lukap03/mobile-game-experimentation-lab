from src.data_generation import SimulationConfig, generate_telemetry
from src.modeling import (
    FEATURES,
    LEAKAGE_COLUMNS,
    build_pipeline,
    train_and_evaluate,
)


def test_features_exclude_outcomes():
    assert not LEAKAGE_COLUMNS.intersection(FEATURES)
    assert build_pipeline().steps[-1][0] == "classifier"


def test_model_returns_metrics_baseline_and_interpretation():
    data = generate_telemetry(SimulationConfig(10_000, 9))
    _, metrics, scored = train_and_evaluate(data, seed=9)
    expected = {
        "roc_auc",
        "pr_auc",
        "pr_auc_baseline",
        "brier_score",
        "precision",
        "recall",
        "f1",
        "confusion_matrix",
        "top_coefficients",
    }
    assert expected.issubset(metrics)
    bounded = (
        "roc_auc",
        "pr_auc",
        "pr_auc_baseline",
        "brier_score",
        "precision",
        "recall",
        "f1",
    )
    assert all(0 <= metrics[key] <= 1 for key in bounded)
    assert metrics["top_coefficients"]
    assert {"feature", "coefficient"} == set(metrics["top_coefficients"][0])
    assert len(scored) == 2_500
