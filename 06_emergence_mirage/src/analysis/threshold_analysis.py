"""
Threshold analysis for understanding metric-induced emergence.

Investigates how evaluation thresholds create artificial discontinuities.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class ThresholdEffect:
    """Analysis of how a threshold affects emergence appearance."""
    threshold_value: float
    apparent_emergence_scale: Optional[int]
    pre_threshold_mean: float
    post_threshold_mean: float
    transition_sharpness: float
    artifact_score: float  # 0 = real emergence, 1 = pure artifact


class ThresholdAnalyzer:
    """
    Analyzes how evaluation thresholds create emergence artifacts.

    The key insight is that continuous underlying capabilities can
    appear discontinuous when viewed through thresholded metrics.
    """

    def __init__(self):
        pass

    def analyze_threshold_effect(
        self,
        raw_scores: List[Tuple[int, float]],  # (param_count, continuous_score)
        threshold: float = 0.5
    ) -> ThresholdEffect:
        """
        Analyze how thresholding a continuous metric creates emergence appearance.

        Args:
            raw_scores: Continuous scores by model scale
            threshold: Threshold for binary conversion

        Returns:
            Analysis of threshold-induced emergence artifact
        """
        if not raw_scores:
            return ThresholdEffect(
                threshold_value=threshold,
                apparent_emergence_scale=None,
                pre_threshold_mean=0.0,
                post_threshold_mean=0.0,
                transition_sharpness=0.0,
                artifact_score=0.0
            )

        sorted_scores = sorted(raw_scores, key=lambda x: x[0])
        param_counts = [s[0] for s in sorted_scores]
        continuous = np.array([s[1] for s in sorted_scores])

        # Apply threshold to create binary metric
        binary = (continuous >= threshold).astype(float)

        # Find apparent emergence point in binary metric
        emergence_scale = self._find_binary_emergence(
            list(zip(param_counts, binary))
        )

        # Analyze the transition
        if emergence_scale:
            idx = param_counts.index(emergence_scale)
            pre_mean = np.mean(continuous[:idx]) if idx > 0 else 0
            post_mean = np.mean(continuous[idx:]) if idx < len(continuous) else 0
        else:
            pre_mean = np.mean(continuous[:len(continuous)//2])
            post_mean = np.mean(continuous[len(continuous)//2:])

        # Compute transition sharpness
        transition_sharpness = self._compute_transition_sharpness(
            continuous, threshold
        )

        # Compute artifact score
        # High if continuous metric shows gradual change but binary shows sudden
        continuous_smoothness = self._compute_smoothness(continuous)
        artifact_score = continuous_smoothness * (1 - transition_sharpness)

        return ThresholdEffect(
            threshold_value=threshold,
            apparent_emergence_scale=emergence_scale,
            pre_threshold_mean=pre_mean,
            post_threshold_mean=post_mean,
            transition_sharpness=transition_sharpness,
            artifact_score=artifact_score
        )

    def _find_binary_emergence(
        self,
        binary_scores: List[Tuple[int, float]]
    ) -> Optional[int]:
        """Find the scale where binary metric first consistently succeeds."""
        window_size = 1
        for i, (scale, score) in enumerate(binary_scores):
            if score >= 0.5:  # First success
                # Check if it stays successful
                remaining = [s for _, s in binary_scores[i:]]
                if np.mean(remaining) >= 0.7:
                    return scale
        return None

    def _compute_transition_sharpness(
        self,
        scores: np.ndarray,
        threshold: float
    ) -> float:
        """
        Compute how sharp the transition across threshold is.

        1.0 = instant jump, 0.0 = gradual transition
        """
        if len(scores) < 2:
            return 0.0

        # Find indices near threshold
        near_threshold = np.abs(scores - threshold) < 0.2
        transition_region = scores[near_threshold]

        if len(transition_region) < 2:
            return 1.0  # Very sharp transition

        # Sharpness is inverse of spread
        spread = np.std(transition_region)
        sharpness = 1.0 / (1.0 + spread * 10)

        return sharpness

    def _compute_smoothness(self, values: np.ndarray) -> float:
        """Compute smoothness of value progression."""
        if len(values) < 3:
            return 1.0

        diffs = np.diff(values)
        variance = np.var(diffs)
        return 1.0 / (1.0 + variance * 100)

    def optimal_threshold_search(
        self,
        raw_scores: List[Tuple[int, float]],
        n_thresholds: int = 20
    ) -> Dict[str, Any]:
        """
        Search for threshold that maximizes/minimizes emergence appearance.

        Returns analysis of how threshold choice affects emergence visibility.
        """
        thresholds = np.linspace(0.1, 0.9, n_thresholds)
        effects = []

        for thresh in thresholds:
            effect = self.analyze_threshold_effect(raw_scores, thresh)
            effects.append({
                'threshold': thresh,
                'emergence_scale': effect.apparent_emergence_scale,
                'artifact_score': effect.artifact_score,
                'transition_sharpness': effect.transition_sharpness
            })

        # Find threshold that makes emergence most apparent
        with_emergence = [e for e in effects if e['emergence_scale'] is not None]

        if with_emergence:
            most_emergence = max(with_emergence,
                                key=lambda x: x['transition_sharpness'])
            least_emergence = max(effects, key=lambda x: x['artifact_score'])
        else:
            most_emergence = None
            least_emergence = effects[0] if effects else None

        return {
            'threshold_effects': effects,
            'most_emergence_at': most_emergence,
            'least_emergence_at': least_emergence,
            'emergence_is_threshold_dependent': len(set(
                e['emergence_scale'] for e in effects
            )) > 1
        }

    def simulate_measurement_noise(
        self,
        true_scores: List[Tuple[int, float]],
        noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2]
    ) -> Dict[str, Any]:
        """
        Simulate how measurement noise affects emergence detection.

        Even with continuous underlying capability, noise + thresholding
        can create artificial emergence patterns.
        """
        results = {}

        for noise in noise_levels:
            noisy_scores = [
                (scale, score + np.random.normal(0, noise))
                for scale, score in true_scores
            ]

            # Analyze with default threshold
            effect = self.analyze_threshold_effect(noisy_scores, 0.5)

            results[f'noise_{noise}'] = {
                'emergence_detected': effect.apparent_emergence_scale is not None,
                'emergence_scale': effect.apparent_emergence_scale,
                'artifact_score': effect.artifact_score
            }

        return {
            'noise_analysis': results,
            'emergence_robust_to_noise': all(
                r['emergence_scale'] == results['noise_0.01']['emergence_scale']
                for r in results.values()
            )
        }
