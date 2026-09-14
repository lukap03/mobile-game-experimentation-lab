from src.data_generation import SimulationConfig, generate_telemetry
from src.modeling import FEATURES, LEAKAGE_COLUMNS, build_pipeline, train_and_evaluate


def test_features_exclude_outcomes():
    assert not LEAKAGE_COLUMNS.intersection(FEATURES)
    assert build_pipeline().steps[-1][0] == "classifier"


def test_model_returns_all_metrics():
    data = generate_telemetry(SimulationConfig(10_000, 9))
    _, metrics, scored = train_and_evaluate(data, seed=9)
    assert {"roc_auc", "pr_auc", "precision", "recall", "f1", "confusion_matrix"}.issubset(metrics)
    assert all(0 <= metrics[key] <= 1 for key in ("roc_auc", "pr_auc", "precision", "recall", "f1"))
    assert len(scored) == 2_500
