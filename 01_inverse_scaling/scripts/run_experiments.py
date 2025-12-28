#!/usr/bin/env python3
"""
Main entry point for running inverse scaling experiments.

Usage:
    python run_experiments.py --config config/experiment_config.yaml
    python run_experiments.py --quick  # Quick test run
    python run_experiments.py --generate-data  # Generate datasets only
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
import yaml
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataLoader, generate_full_dataset
from src.models import ModelRegistry, get_registry
from src.experiments import ExperimentRunner, ExperimentConfig, TrialStatus
from src.analysis import ResultsAnalyzer
from src.visualization import PublicationFigureGenerator, ScalingPlotter


def setup_logging(log_file: Path = None, level: str = "INFO"):
    """Configure logging."""
    handlers = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_data(config: dict, output_dir: Path):
    """Generate task datasets."""
    logging.info("Generating task datasets...")

    samples_per_task = config.get("evaluation", {}).get("num_samples", 500)
    seed = config.get("experiment", {}).get("seed", 42)

    generate_full_dataset(output_dir, samples_per_task, seed)
    logging.info(f"Datasets generated in {output_dir}")


def setup_models(config: dict, registry: ModelRegistry):
    """Register models from configuration."""
    models_config = config.get("models", {})

    for provider, model_list in models_config.items():
        if not isinstance(model_list, list):
            continue

        for model_info in model_list:
            if not model_info.get("enabled", True):
                continue

            try:
                registry.register(
                    name=model_info["name"],
                    provider=model_info.get("provider", provider),
                    estimated_params=model_info.get("estimated_params", "unknown"),
                    enabled=True,
                )
                logging.info(f"Registered model: {model_info['name']}")
            except Exception as e:
                logging.warning(f"Failed to register {model_info['name']}: {e}")


def run_full_experiment(
    config: dict,
    data_dir: Path,
    output_dir: Path,
    registry: ModelRegistry,
):
    """Run the full experiment suite."""
    logging.info("Starting full experiment...")

    # Setup runner
    runner = ExperimentRunner(data_dir, output_dir, registry)

    # Get enabled models and tasks
    enabled_models = registry.get_all_enabled()
    data_loader = DataLoader(data_dir)
    available_tasks = data_loader.get_available_tasks()

    logging.info(f"Models: {len(enabled_models)}")
    logging.info(f"Tasks: {len(available_tasks)}")

    # Create experiment config
    exp_config = ExperimentConfig(
        name=config.get("experiment", {}).get("name", "inverse_scaling_study"),
        models=enabled_models,
        tasks=available_tasks,
        samples_per_task=config.get("evaluation", {}).get("num_samples", 100),
        num_runs=config.get("evaluation", {}).get("num_runs", 1),
        temperature=config.get("evaluation", {}).get("temperature", 0.0),
        max_tokens=config.get("evaluation", {}).get("max_tokens", 256),
        seed=config.get("experiment", {}).get("seed", 42),
    )

    # Run experiment
    trial = runner.run_experiment(exp_config)
    logging.info(f"Experiment completed: {trial.trial_id}")
    logging.info(f"Status: {trial.status.value}")
    logging.info(f"Results: {len(trial.results)}")

    return trial


def analyze_results(trial, output_dir: Path):
    """Analyze experiment results."""
    logging.info("Analyzing results...")

    analyzer = ResultsAnalyzer()
    analysis = analyzer.analyze_trial(trial)

    # Save analysis
    analysis_path = output_dir / "analysis" / f"analysis_{trial.trial_id}.json"
    analyzer.export_results(analysis, analysis_path, "json")
    logging.info(f"Analysis saved to {analysis_path}")

    # Print summary
    overall = analysis.get("overall", {})
    print("\n" + "=" * 60)
    print("INVERSE SCALING ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total tasks evaluated: {overall.get('total_tasks', 0)}")
    print(f"Models evaluated: {overall.get('models_evaluated', 0)}")
    print(f"Inverse scaling detected: {overall.get('inverse_scaling_detected', 0)} tasks")
    print(f"Proportion: {overall.get('inverse_scaling_proportion', 0):.1%}")
    print()

    inverse_tasks = analysis.get("inverse_scaling_tasks", [])
    if inverse_tasks:
        print("Tasks exhibiting inverse scaling:")
        for task in inverse_tasks:
            print(f"  - {task}")
    else:
        print("No inverse scaling detected in any task.")

    print("=" * 60 + "\n")

    return analysis


def generate_figures(analysis: dict, trial_id: str, output_dir: Path):
    """Generate publication-ready figures."""
    logging.info("Generating figures...")

    figure_gen = PublicationFigureGenerator()
    figure_path = output_dir / "figures" / f"figure_{trial_id}"

    fig = figure_gen.create_main_figure(analysis, figure_path)
    saved = figure_gen.save_figure(fig, figure_path)

    logging.info(f"Figures saved: {[str(p) for p in saved]}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inverse scaling experiments"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "config" / "experiment_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test with minimal samples",
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate datasets only",
    )
    parser.add_argument(
        "--analyze-only",
        type=str,
        help="Analyze existing trial by ID",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    log_file = args.output_dir / "experiment.log"
    setup_logging(log_file, args.log_level)

    # Load configuration
    if args.config.exists():
        config = load_config(args.config)
    else:
        logging.warning(f"Config not found at {args.config}, using defaults")
        config = {}

    # Setup directories
    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate data only
    if args.generate_data:
        generate_data(config, data_dir)
        return

    # Initialize registry
    registry = get_registry()
    setup_models(config, registry)

    # Quick mode overrides
    if args.quick:
        config.setdefault("evaluation", {})["num_samples"] = 10
        config.setdefault("evaluation", {})["num_runs"] = 1

    # Analyze existing trial
    if args.analyze_only:
        runner = ExperimentRunner(data_dir, args.output_dir, registry)
        trial = runner.get_trial(args.analyze_only)
        if trial:
            analysis = analyze_results(trial, args.output_dir)
            generate_figures(analysis, trial.trial_id, args.output_dir)
        else:
            logging.error(f"Trial not found: {args.analyze_only}")
        return

    # Ensure data exists
    if not data_dir.exists() or not any(data_dir.glob("*.json")):
        logging.info("No data found, generating datasets...")
        generate_data(config, data_dir)

    # Run experiment
    trial = run_full_experiment(config, data_dir, args.output_dir, registry)

    if trial.status == TrialStatus.COMPLETED:
        # Analyze and generate figures
        analysis = analyze_results(trial, args.output_dir)
        generate_figures(analysis, trial.trial_id, args.output_dir)

        logging.info("Experiment pipeline completed successfully!")
    else:
        logging.error(f"Experiment failed: {trial.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
