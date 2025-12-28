#!/usr/bin/env python3
"""
Main experiment runner for Emergence Mirage analysis.
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from datetime import datetime
import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.capability_tasks import CapabilityTaskGenerator
from src.experiments.emergence_detector import EmergenceDetector
from src.experiments.metric_analyzer import MetricAnalyzer
from src.analysis.curve_fitting import EmergenceCurveFitter
from src.analysis.threshold_analysis import ThresholdAnalyzer


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


def run_emergence_analysis(config, output_dir):
    """Run the main emergence analysis experiment."""
    logger = logging.getLogger(__name__)
    logger.info("Starting Emergence Mirage Analysis")

    # Initialize components
    generator = CapabilityTaskGenerator()
    analyzer = MetricAnalyzer()
    fitter = EmergenceCurveFitter()
    threshold_analyzer = ThresholdAnalyzer()

    capabilities = config.get('capabilities', {}).get('tested', [])
    if not capabilities:
        capabilities = [{'name': cap} for cap in generator.available_capabilities]

    results = {
        'experiment_name': config.get('experiment', {}).get('name', 'emergence_analysis'),
        'timestamp': datetime.now().isoformat(),
        'capabilities': {},
        'summary': {}
    }

    for cap_config in capabilities:
        cap_name = cap_config['name']
        logger.info(f"Analyzing capability: {cap_name}")

        # Generate synthetic scaling data for analysis
        # In production, this would use actual model evaluations
        scales, scores = generate_scaling_data(cap_name)

        cap_results = {
            'scales': scales.tolist(),
            'metrics': {},
            'classifications': {},
            'threshold_analysis': {}
        }

        # Simulate multiple metrics
        metrics = ['binary', 'partial_credit', 'log_probability']

        for metric in metrics:
            metric_scores = transform_for_metric(scores, metric)

            # Fit curves
            fit_results = fitter.fit_all_models(scales, metric_scores)
            classification = fitter.classify_emergence(fit_results)

            # Statistical test
            stat_test = fitter.statistical_test_emergence(
                scales, metric_scores, n_bootstrap=100
            )

            cap_results['metrics'][metric] = metric_scores.tolist()
            cap_results['classifications'][metric] = {
                'pattern_type': classification.pattern_type,
                'confidence': classification.confidence,
                'best_model': classification.best_model,
                'evidence': classification.supporting_evidence,
                'stable': stat_test['classification_stable'],
                'artifact_evidence': stat_test['evidence_for_artifact']
            }

            # Threshold analysis
            raw_scores = list(zip(scales.tolist(), metric_scores.tolist()))
            threshold_effect = threshold_analyzer.analyze_threshold_effect(
                raw_scores, threshold=0.5
            )
            cap_results['threshold_analysis'][metric] = {
                'artifact_score': threshold_effect.artifact_score,
                'transition_sharpness': threshold_effect.transition_sharpness
            }

            logger.info(f"  {metric}: {classification.pattern_type} "
                       f"(confidence={classification.confidence:.2f})")

        results['capabilities'][cap_name] = cap_results

    # Generate summary
    all_patterns = []
    artifact_scores = []

    for cap_name, cap_results in results['capabilities'].items():
        for metric, classification in cap_results['classifications'].items():
            all_patterns.append(classification['pattern_type'])
            if cap_results['threshold_analysis'].get(metric):
                artifact_scores.append(
                    cap_results['threshold_analysis'][metric]['artifact_score']
                )

    results['summary'] = {
        'total_analyses': len(all_patterns),
        'continuous_count': all_patterns.count('continuous'),
        'discontinuous_count': all_patterns.count('discontinuous'),
        'phase_transition_count': all_patterns.count('phase_transition'),
        'mean_artifact_score': float(np.mean(artifact_scores)) if artifact_scores else 0,
        'conclusion': determine_conclusion(all_patterns, artifact_scores)
    }

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"emergence_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_file}")
    logger.info(f"Summary: {results['summary']}")

    return results


def generate_scaling_data(capability):
    """Generate synthetic scaling data for analysis."""
    np.random.seed(42)

    # Model scales
    scales = np.array([1e8, 3e8, 1e9, 3e9, 1e10, 3e10])

    # Generate scores based on capability
    log_scales = np.log10(scales)

    # Base sigmoid curve with noise
    midpoint = 9.5
    steepness = 1.5

    base_scores = 1 / (1 + np.exp(-steepness * (log_scales - midpoint)))
    scores = base_scores + np.random.normal(0, 0.02, len(scales))
    scores = np.clip(scores, 0, 1)

    return scales, scores


def transform_for_metric(scores, metric):
    """Transform scores to simulate different metric behaviors."""
    if metric == 'binary':
        # Binary shows sharper transitions
        return (scores > 0.5).astype(float)
    elif metric == 'partial_credit':
        # Partial credit shows smoother progression
        return 0.2 + 0.7 * scores + np.random.normal(0, 0.03, len(scores))
    elif metric == 'log_probability':
        # Log probability is more linear
        return -3 + 2.5 * scores + np.random.normal(0, 0.1, len(scores))
    else:
        return scores


def determine_conclusion(patterns, artifact_scores):
    """Determine overall conclusion about emergence."""
    continuous_ratio = patterns.count('continuous') / len(patterns) if patterns else 0
    mean_artifact = np.mean(artifact_scores) if artifact_scores else 0

    if continuous_ratio > 0.5 or mean_artifact > 0.5:
        return "Evidence suggests emergence is largely a measurement artifact"
    elif continuous_ratio < 0.2 and mean_artifact < 0.3:
        return "Some capabilities show genuine emergence patterns"
    else:
        return "Mixed evidence - emergence may be partially real, partially artifact"


def main():
    parser = argparse.ArgumentParser(description="Run Emergence Mirage Analysis")
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

    run_emergence_analysis(config, args.output_dir)
    logging.info("Emergence analysis completed!")


if __name__ == "__main__":
    main()
