#!/usr/bin/env python3
"""
Main experiment runner script.
"""

import argparse
import sys
import logging
from pathlib import Path
import yaml
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(log_file=None, level="INFO"):
    """Configure logging."""
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_experiment(config, output_dir):
    """Run the main experiment."""
    logging.info("Starting experiment...")
    
    results = {
        "experiment_name": config.get("experiment", {}).get("name", "unnamed"),
        "timestamp": datetime.now().isoformat(),
        "status": "completed",
        "results": {}
    }
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to {results_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "config" / "experiment_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "results",
        help="Output directory",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    
    args = parser.parse_args()
    
    log_file = args.output_dir / "experiment.log"
    setup_logging(log_file, args.log_level)
    
    if args.config.exists():
        config = load_config(args.config)
    else:
        logging.warning(f"Config not found: {args.config}")
        config = {}
    
    run_experiment(config, args.output_dir)
    logging.info("Experiment completed successfully!")


if __name__ == "__main__":
    main()
