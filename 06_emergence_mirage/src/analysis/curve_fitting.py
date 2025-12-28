"""
Curve fitting for emergence analysis.

Fits various functional forms to capability scaling data to determine
whether emergence is continuous or discontinuous.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.optimize import curve_fit, minimize
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class CurveFitResult:
    """Result of fitting a curve to scaling data."""
    model_name: str
    parameters: Dict[str, float]
    r_squared: float
    aic: float  # Akaike Information Criterion
    bic: float  # Bayesian Information Criterion
    residuals: np.ndarray
    predictions: np.ndarray


@dataclass
class EmergenceClassification:
    """Classification of emergence pattern."""
    pattern_type: str  # 'continuous', 'discontinuous', 'phase_transition'
    confidence: float
    supporting_evidence: List[str]
    best_model: str


class EmergenceCurveFitter:
    """
    Fits scaling curves to determine emergence characteristics.

    Tests multiple hypotheses:
    1. Continuous improvement (linear, log, power law)
    2. Discontinuous emergence (step function, sigmoid with sharp transition)
    3. Phase transition (sigmoid with gradual transition)
    """

    def __init__(self):
        self.models = {
            'linear': (self._linear, ['slope', 'intercept'], 2),
            'logarithmic': (self._logarithmic, ['scale', 'offset'], 2),
            'power_law': (self._power_law, ['coefficient', 'exponent'], 2),
            'sigmoid': (self._sigmoid, ['k', 'x0', 'L', 'U'], 4),
            'step': (self._step_smooth, ['threshold', 'low', 'high', 'width'], 4),
            'double_sigmoid': (self._double_sigmoid,
                              ['k1', 'x01', 'k2', 'x02', 'L', 'M', 'U'], 7)
        }

    @staticmethod
    def _linear(x, slope, intercept):
        return slope * x + intercept

    @staticmethod
    def _logarithmic(x, scale, offset):
        return scale * np.log(x + 1) + offset

    @staticmethod
    def _power_law(x, coef, exp):
        return coef * np.power(x + 0.1, exp)

    @staticmethod
    def _sigmoid(x, k, x0, L, U):
        return L + (U - L) / (1 + np.exp(-k * (x - x0)))

    @staticmethod
    def _step_smooth(x, threshold, low, high, width):
        return low + (high - low) / (1 + np.exp(-(x - threshold) / max(width, 0.01)))

    @staticmethod
    def _double_sigmoid(x, k1, x01, k2, x02, L, M, U):
        s1 = 1 / (1 + np.exp(-k1 * (x - x01)))
        s2 = 1 / (1 + np.exp(-k2 * (x - x02)))
        return L + (M - L) * s1 + (U - M) * s2

    def fit_all_models(
        self,
        scales: np.ndarray,
        scores: np.ndarray
    ) -> Dict[str, CurveFitResult]:
        """Fit all models to the data and return results."""
        results = {}

        # Use log scale for parameters
        log_scales = np.log10(scales + 1)

        for name, (func, param_names, n_params) in self.models.items():
            try:
                result = self._fit_single_model(
                    func, param_names, n_params, log_scales, scores
                )
                results[name] = result
                results[name].model_name = name
            except Exception as e:
                logger.debug(f"Failed to fit {name}: {e}")

        return results

    def _fit_single_model(
        self,
        func: Callable,
        param_names: List[str],
        n_params: int,
        x: np.ndarray,
        y: np.ndarray
    ) -> CurveFitResult:
        """Fit a single model to the data."""
        # Get initial guess and bounds based on function
        p0, bounds = self._get_initial_params(func, x, y, n_params)

        # Fit curve
        popt, pcov = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=10000)

        # Compute predictions and residuals
        predictions = func(x, *popt)
        residuals = y - predictions

        # Compute fit statistics
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        n = len(y)
        k = n_params

        # AIC and BIC
        if ss_res > 0:
            log_likelihood = -n/2 * np.log(ss_res / n)
        else:
            log_likelihood = 0
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood

        return CurveFitResult(
            model_name="",
            parameters=dict(zip(param_names, popt)),
            r_squared=r_squared,
            aic=aic,
            bic=bic,
            residuals=residuals,
            predictions=predictions
        )

    def _get_initial_params(
        self,
        func: Callable,
        x: np.ndarray,
        y: np.ndarray,
        n_params: int
    ) -> Tuple[List[float], Tuple[List[float], List[float]]]:
        """Get initial parameters and bounds for fitting."""
        y_range = np.max(y) - np.min(y)
        y_mean = np.mean(y)
        x_range = np.max(x) - np.min(x)
        x_mean = np.mean(x)

        if n_params == 2:
            # Linear, logarithmic, power law
            p0 = [y_range / max(x_range, 1), y_mean]
            bounds = ([-100, -100], [100, 100])

        elif n_params == 4 and func == self._sigmoid:
            # Sigmoid
            p0 = [1.0, x_mean, np.min(y), np.max(y)]
            bounds = ([0.01, np.min(x)-1, -1, -1],
                     [100, np.max(x)+1, 2, 2])

        elif n_params == 4:
            # Step
            p0 = [x_mean, np.min(y), np.max(y), x_range/4]
            bounds = ([np.min(x)-1, -1, -1, 0.001],
                     [np.max(x)+1, 2, 2, x_range])

        elif n_params == 7:
            # Double sigmoid
            x_third = np.min(x) + x_range/3
            x_two_third = np.min(x) + 2*x_range/3
            p0 = [1.0, x_third, 1.0, x_two_third,
                  np.min(y), y_mean, np.max(y)]
            bounds = ([0.01, np.min(x)-1, 0.01, np.min(x)-1, -1, -1, -1],
                     [100, np.max(x)+1, 100, np.max(x)+1, 2, 2, 2])

        else:
            p0 = [0.5] * n_params
            bounds = ([-100] * n_params, [100] * n_params)

        return p0, bounds

    def classify_emergence(
        self,
        fit_results: Dict[str, CurveFitResult]
    ) -> EmergenceClassification:
        """
        Classify the emergence pattern based on model fits.

        Determines if emergence is real or a measurement artifact.
        """
        if not fit_results:
            return EmergenceClassification(
                pattern_type='unknown',
                confidence=0.0,
                supporting_evidence=[],
                best_model='none'
            )

        # Find best model by BIC (penalizes complexity)
        best_model = min(fit_results.items(), key=lambda x: x[1].bic)
        best_name, best_result = best_model

        evidence = []

        # Classify based on best-fitting model
        continuous_models = ['linear', 'logarithmic', 'power_law']
        discontinuous_models = ['step']
        transition_models = ['sigmoid', 'double_sigmoid']

        if best_name in continuous_models:
            pattern_type = 'continuous'
            evidence.append(f"Best fit is {best_name} (continuous model)")

            # Check if sigmoid/step also fit well
            for disc_model in discontinuous_models + transition_models:
                if disc_model in fit_results:
                    bic_diff = fit_results[disc_model].bic - best_result.bic
                    if bic_diff < 2:  # Models are comparable
                        evidence.append(
                            f"{disc_model} also fits well (BIC diff: {bic_diff:.2f})"
                        )

        elif best_name in discontinuous_models:
            pattern_type = 'discontinuous'
            evidence.append(f"Best fit is {best_name} (discontinuous model)")

            # Check transition sharpness
            if 'width' in best_result.parameters:
                width = best_result.parameters['width']
                if width < 0.1:
                    evidence.append(f"Sharp transition (width={width:.3f})")
                else:
                    evidence.append(f"Gradual transition (width={width:.3f})")
                    pattern_type = 'phase_transition'

        elif best_name in transition_models:
            # Check sigmoid steepness
            if 'k' in best_result.parameters:
                k = best_result.parameters['k']
                if k > 5:
                    pattern_type = 'discontinuous'
                    evidence.append(f"Sharp sigmoid (k={k:.2f})")
                else:
                    pattern_type = 'phase_transition'
                    evidence.append(f"Gradual sigmoid (k={k:.2f})")
            else:
                pattern_type = 'phase_transition'

        else:
            pattern_type = 'unknown'

        # Compute confidence based on R² and model comparison
        confidence = best_result.r_squared

        # Reduce confidence if multiple models fit similarly
        similar_fits = sum(
            1 for r in fit_results.values()
            if abs(r.bic - best_result.bic) < 5
        )
        if similar_fits > 2:
            confidence *= 0.7
            evidence.append(f"Multiple models fit similarly ({similar_fits})")

        return EmergenceClassification(
            pattern_type=pattern_type,
            confidence=confidence,
            supporting_evidence=evidence,
            best_model=best_name
        )

    def statistical_test_emergence(
        self,
        scales: np.ndarray,
        scores: np.ndarray,
        n_bootstrap: int = 1000
    ) -> Dict[str, Any]:
        """
        Statistical test for emergence vs. continuous improvement.

        Uses bootstrap to assess confidence in classification.
        """
        fit_results = self.fit_all_models(scales, scores)
        base_classification = self.classify_emergence(fit_results)

        # Bootstrap to assess stability
        bootstrap_patterns = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(scales), size=len(scales), replace=True)
            boot_scales = scales[indices]
            boot_scores = scores[indices]

            # Sort by scale
            sort_idx = np.argsort(boot_scales)
            boot_scales = boot_scales[sort_idx]
            boot_scores = boot_scores[sort_idx]

            try:
                boot_fits = self.fit_all_models(boot_scales, boot_scores)
                boot_class = self.classify_emergence(boot_fits)
                bootstrap_patterns.append(boot_class.pattern_type)
            except Exception:
                continue

        # Count pattern frequencies
        pattern_counts = {}
        for pattern in bootstrap_patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        total = len(bootstrap_patterns)
        pattern_probs = {
            k: v / total for k, v in pattern_counts.items()
        } if total > 0 else {}

        return {
            'classification': base_classification,
            'bootstrap_pattern_probabilities': pattern_probs,
            'classification_stable': (
                pattern_probs.get(base_classification.pattern_type, 0) > 0.7
            ),
            'evidence_for_artifact': (
                pattern_probs.get('continuous', 0) > 0.3 and
                base_classification.pattern_type != 'continuous'
            )
        }
