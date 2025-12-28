"""
Metric analysis for emergence investigation.

Analyzes how different metrics affect the appearance of emergence.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats
from scipy.optimize import curve_fit
import logging

from .emergence_detector import EmergenceResult

logger = logging.getLogger(__name__)


@dataclass
class MetricComparison:
    """Comparison of emergence patterns across metrics."""
    capability: str
    metric_name: str
    scores_by_scale: List[Tuple[int, float]]  # (param_count, score)

    # Fitted curve parameters
    best_fit_model: str
    fit_params: Dict[str, float]
    r_squared: float

    # Emergence analysis
    apparent_emergence_point: Optional[int]  # Parameter count where emergence appears
    is_continuous: bool
    smoothness_score: float


class MetricAnalyzer:
    """
    Analyzes how different metrics reveal or obscure emergence patterns.
    """

    def __init__(self):
        self.curve_models = {
            'linear': self._linear,
            'logarithmic': self._logarithmic,
            'sigmoid': self._sigmoid,
            'step': self._step,
            'power_law': self._power_law
        }

    def analyze_metric_patterns(
        self,
        results: List[EmergenceResult],
        metrics: List[str] = None
    ) -> Dict[str, MetricComparison]:
        """
        Analyze emergence patterns for different metrics.

        Returns comparison of how each metric shows capability scaling.
        """
        if metrics is None:
            metrics = [
                'binary_accuracy',
                'partial_credit_score',
                'log_probability_score',
                'rank_score',
                'entropy_score'
            ]

        comparisons = {}

        for metric in metrics:
            scores = [
                (r.parameter_count, getattr(r, metric))
                for r in results
            ]
            scores.sort(key=lambda x: x[0])

            comparison = self._analyze_single_metric(
                results[0].capability,
                metric,
                scores
            )
            comparisons[metric] = comparison

        return comparisons

    def _analyze_single_metric(
        self,
        capability: str,
        metric_name: str,
        scores: List[Tuple[int, float]]
    ) -> MetricComparison:
        """Analyze emergence pattern for a single metric."""
        params = np.array([np.log10(s[0]) for s in scores])  # Log scale
        values = np.array([s[1] for s in scores])

        # Try fitting different curve models
        best_fit = None
        best_r2 = -np.inf
        best_params = {}

        for model_name, model_func in self.curve_models.items():
            try:
                popt, _ = self._fit_curve(model_func, params, values)
                predicted = model_func(params, *popt)
                r2 = self._r_squared(values, predicted)

                if r2 > best_r2:
                    best_r2 = r2
                    best_fit = model_name
                    best_params = self._params_to_dict(model_name, popt)
            except Exception as e:
                logger.debug(f"Failed to fit {model_name}: {e}")
                continue

        # Detect apparent emergence point
        emergence_point = self._detect_emergence_point(scores)

        # Compute smoothness score
        smoothness = self._compute_smoothness(values)

        # Determine if pattern is continuous
        is_continuous = best_fit in ['linear', 'logarithmic', 'power_law']

        return MetricComparison(
            capability=capability,
            metric_name=metric_name,
            scores_by_scale=scores,
            best_fit_model=best_fit or 'unknown',
            fit_params=best_params,
            r_squared=best_r2 if best_r2 > 0 else 0.0,
            apparent_emergence_point=emergence_point,
            is_continuous=is_continuous,
            smoothness_score=smoothness
        )

    def _fit_curve(
        self,
        func,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit a curve with appropriate bounds."""
        # Get initial guesses based on function type
        if func == self._linear:
            p0 = [0.1, 0.0]
            bounds = ([-10, -10], [10, 10])
        elif func == self._logarithmic:
            p0 = [0.1, 0.5]
            bounds = ([0, -10], [10, 10])
        elif func == self._sigmoid:
            p0 = [1.0, np.mean(x), 0.0, 1.0]
            bounds = ([0.01, min(x)-1, -1, 0], [100, max(x)+1, 2, 2])
        elif func == self._step:
            p0 = [np.mean(x), 0.1, 0.9]
            bounds = ([min(x)-1, 0, 0], [max(x)+1, 1, 1])
        elif func == self._power_law:
            p0 = [0.1, 1.0]
            bounds = ([0, 0], [10, 5])
        else:
            p0 = None
            bounds = (-np.inf, np.inf)

        return curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=5000)

    @staticmethod
    def _linear(x, a, b):
        return a * x + b

    @staticmethod
    def _logarithmic(x, a, b):
        return a * np.log(x + 1) + b

    @staticmethod
    def _sigmoid(x, k, x0, L, U):
        return L + (U - L) / (1 + np.exp(-k * (x - x0)))

    @staticmethod
    def _step(x, threshold, low, high):
        return np.where(x < threshold, low, high)

    @staticmethod
    def _power_law(x, a, b):
        return a * np.power(x + 1, b)

    def _r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R-squared value."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1 - (ss_res / ss_tot)

    def _params_to_dict(self, model_name: str, params: np.ndarray) -> Dict[str, float]:
        """Convert fitted parameters to dictionary."""
        param_names = {
            'linear': ['slope', 'intercept'],
            'logarithmic': ['scale', 'offset'],
            'sigmoid': ['steepness', 'midpoint', 'lower', 'upper'],
            'step': ['threshold', 'low', 'high'],
            'power_law': ['coefficient', 'exponent']
        }
        names = param_names.get(model_name, [f'p{i}' for i in range(len(params))])
        return dict(zip(names, params))

    def _detect_emergence_point(
        self,
        scores: List[Tuple[int, float]]
    ) -> Optional[int]:
        """
        Detect point where capability appears to emerge.

        Uses change-point detection to find sudden improvements.
        """
        if len(scores) < 3:
            return None

        values = np.array([s[1] for s in scores])
        param_counts = [s[0] for s in scores]

        # Compute first derivative (change rate)
        diffs = np.diff(values)

        # Find maximum improvement point
        if len(diffs) == 0:
            return None

        max_diff_idx = np.argmax(diffs)

        # Check if this is a significant jump (> 2x average change)
        avg_diff = np.mean(np.abs(diffs))
        if diffs[max_diff_idx] > 2 * avg_diff:
            return param_counts[max_diff_idx + 1]

        return None

    def _compute_smoothness(self, values: np.ndarray) -> float:
        """
        Compute smoothness score (1.0 = perfectly smooth, 0.0 = very jagged).

        Based on second derivative analysis.
        """
        if len(values) < 3:
            return 1.0

        # Compute second derivative
        second_deriv = np.diff(values, n=2)

        # Smoothness is inverse of second derivative variance
        variance = np.var(second_deriv)
        smoothness = 1.0 / (1.0 + variance * 100)

        return smoothness

    def compare_emergence_across_metrics(
        self,
        comparisons: Dict[str, MetricComparison]
    ) -> Dict[str, Any]:
        """
        Compare emergence patterns across different metrics.

        Determines if emergence is consistent or metric-dependent.
        """
        emergence_points = {
            name: c.apparent_emergence_point
            for name, c in comparisons.items()
            if c.apparent_emergence_point is not None
        }

        continuity = {
            name: c.is_continuous
            for name, c in comparisons.items()
        }

        # Check if emergence points agree
        if emergence_points:
            points = list(emergence_points.values())
            emergence_variance = np.var(np.log10(points)) if len(points) > 1 else 0
            emergence_consistent = emergence_variance < 0.5
        else:
            emergence_variance = 0
            emergence_consistent = True

        # Check if continuity assessments agree
        continuity_values = list(continuity.values())
        continuity_agreement = len(set(continuity_values)) == 1

        return {
            'emergence_points_by_metric': emergence_points,
            'continuity_by_metric': continuity,
            'emergence_consistent': emergence_consistent,
            'continuity_agreement': continuity_agreement,
            'emergence_is_metric_artifact': not emergence_consistent or not continuity_agreement,
            'smoothest_metric': max(comparisons.items(),
                                   key=lambda x: x[1].smoothness_score)[0],
            'best_fitting_metric': max(comparisons.items(),
                                      key=lambda x: x[1].r_squared)[0]
        }
