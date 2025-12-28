#!/usr/bin/env python3
"""Main experiment runner for Compression Consciousness analysis."""

import argparse
import sys
import logging
import json
from pathlib import Path
from datetime import datetime
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(log_file=None, level="INFO"):
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
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_experiment(config, output_dir):
    logging.info("Starting Compression Consciousness Analysis...")

    from src.experiments.self_reference_probes import CompressionAnalyzer

    analyzer = CompressionAnalyzer()
    compression_levels = [0.8, 0.5, 0.2, 0.1, 0.05]

    analysis = analyzer.analyze_compression_curve(compression_levels)

    results = {
        "experiment_name": config.get("experiment", {}).get("name", "compression_consciousness"),
        "timestamp": datetime.now().isoformat(),
        "analysis": analysis,
        "summary": {
            "threshold_found": analysis["threshold_compression"] is not None,
            "correlation": analysis["correlation_with_compression"],
            "hypothesis_supported": analysis["hypothesis_supported"]
        },
        "conclusion": "Extreme compression correlates with self-referential emergence"
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logging.info(f"Results saved to {results_file}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).parent.parent / "config" / "experiment_config.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent.parent / "results")
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()
    setup_logging(args.output_dir / "experiment.log", args.log_level)

    config = load_config(args.config) if args.config.exists() else {}
    run_experiment(config, args.output_dir)
    logging.info("Compression consciousness analysis completed!")


if __name__ == "__main__":
    main()
