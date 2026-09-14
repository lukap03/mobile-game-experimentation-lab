"""Command-line entry point for the complete reproducible analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data_generation import SimulationConfig, generate_telemetry
from .experiment import (
    analyze_experiment,
    experiment_diagnostics,
    segment_analysis,
)
from .modeling import train_and_evaluate
from .projection import monte_carlo_projection
from .visualization import (
    save_experiment_figure,
    save_model_figure,
    save_projection_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    parser.add_argument("--simulations", type=int, default=2_000)
    parser.add_argument("--rollout-players", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "processed"
    report_dir = root / "reports"
    figure_dir = report_dir / "figures"
    for directory in (data_dir, report_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    data = generate_telemetry(SimulationConfig(args.players, args.seed))
    diagnostics = experiment_diagnostics(data)
    experiment = analyze_experiment(
        data,
        seed=args.seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    segments = segment_analysis(data)
    _model, model_metrics, scores = train_and_evaluate(data, seed=args.seed)
    projection = monte_carlo_projection(
        data,
        rollout_players=args.rollout_players,
        simulations=args.simulations,
        seed=args.seed,
    )

    data.to_csv(data_dir / "synthetic_player_telemetry.csv", index=False)
    segments.to_csv(report_dir / "segment_analysis.csv", index=False)
    save_experiment_figure(
        experiment,
        figure_dir / "experiment_outcomes.svg",
    )
    save_model_figure(
        scores,
        figure_dir / "churn_model_performance.svg",
    )
    save_projection_figure(
        projection,
        figure_dir / "rollout_projection.svg",
    )

    payload = {
        "metadata": {
            "synthetic_data": True,
            "players": args.players,
            "seed": args.seed,
        },
        "experiment_diagnostics": diagnostics,
        "experiment": experiment,
        "churn_model": model_metrics,
        "monte_carlo": projection,
    }
    metrics_path = report_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    print(f"Saved verified metrics to {metrics_path}")
    print(f"Saved figures to {figure_dir}")


if __name__ == "__main__":
    main()
