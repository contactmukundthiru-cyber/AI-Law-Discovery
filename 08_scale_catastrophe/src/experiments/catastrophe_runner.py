"""
Experiment runner for scale catastrophe analysis.
"""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
import logging

from ..data.scale_tasks import ScaleTask, ScaleTaskGenerator, CatastropheDetector

logger = logging.getLogger(__name__)


@dataclass
class CatastropheResult:
    """Result of catastrophe detection experiment."""
    task_type: str
    performance_by_scale: Dict[int, float]
    inverse_scaling_detected: bool
    u_shaped_detected: bool
    catastrophic_drops: List[Dict]
    severity: str


class CatastropheExperimentRunner:
    """
    Runs experiments to detect scale catastrophes.
    """

    def __init__(self, detector: CatastropheDetector):
        self.detector = detector
        self.generator = ScaleTaskGenerator()

    def run_scale_sweep(
        self,
        tasks: List[ScaleTask],
        model_scales: List[int],
        evaluator_fn: callable = None
    ) -> CatastropheResult:
        """Run tasks across model scales and detect catastrophes."""
        if evaluator_fn is None:
            evaluator_fn = self._synthetic_evaluator

        performance_by_scale = {}

        for scale in model_scales:
            correct = 0
            for task in tasks:
                result = evaluator_fn(task, scale)
                if result:
                    correct += 1

            performance_by_scale[scale] = correct / len(tasks)

        # Detect patterns
        inverse_result = self.detector.detect_inverse_scaling(performance_by_scale)
        u_result = self.detector.detect_u_shaped_curve(performance_by_scale)

        # Determine severity
        if inverse_result["catastrophic"]:
            severity = "critical"
        elif inverse_result["inverse_scaling"]:
            severity = "moderate"
        elif u_result["u_shaped"] and u_result["dip_depth"] > 0.1:
            severity = "mild"
        else:
            severity = "none"

        return CatastropheResult(
            task_type=tasks[0].task_type if tasks else "unknown",
            performance_by_scale=performance_by_scale,
            inverse_scaling_detected=inverse_result["inverse_scaling"],
            u_shaped_detected=u_result["u_shaped"],
            catastrophic_drops=inverse_result.get("performance_drops", []),
            severity=severity
        )

    def _synthetic_evaluator(self, task: ScaleTask, scale: int) -> bool:
        """Synthetic evaluator for testing without real models."""
        np.random.seed(hash(task.task_id) % 2**32)

        # Base probability depends on task difficulty
        base_prob = 1 - task.difficulty * 0.5

        # Scale effect depends on expected failure scale
        log_scale = np.log10(scale)

        if task.expected_failure_scale == "large":
            # Larger models fail more
            scale_effect = -0.1 * (log_scale - 8)
        elif task.expected_failure_scale == "small":
            # Smaller models fail more
            scale_effect = 0.1 * (log_scale - 8)
        else:
            scale_effect = 0

        prob = np.clip(base_prob + scale_effect, 0.1, 0.95)
        return np.random.random() < prob

    def run_full_analysis(
        self,
        model_scales: List[int] = None
    ) -> Dict[str, CatastropheResult]:
        """Run full catastrophe analysis across all task types."""
        if model_scales is None:
            model_scales = [10**8, 10**9, 10**10, 10**11, 10**12]

        all_tasks = self.generator.generate_all(n_per_type=50)
        results = {}

        for task_type, tasks in all_tasks.items():
            logger.info(f"Analyzing {task_type}...")
            result = self.run_scale_sweep(tasks, model_scales)
            results[task_type] = result

            logger.info(f"  Inverse scaling: {result.inverse_scaling_detected}")
            logger.info(f"  Severity: {result.severity}")

        return results
