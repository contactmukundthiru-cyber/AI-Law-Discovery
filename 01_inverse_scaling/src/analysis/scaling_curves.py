"""
Scaling curve analysis for inverse scaling research.

Provides curve fitting to detect scaling patterns.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Callable
from enum import Enum
from scipy import optimize
from scipy import stats
import warnings


class CurveType(Enum):
    """Types of scaling curves."""
    LINEAR = "linear"
    LOG_LINEAR = "log_linear"
    POWER_LAW = "power_law"
    QUADRATIC = "quadratic"
    INVERSE_U = "inverse_u"
    SIGMOID = "sigmoid"
    PLATEAU = "plateau"


@dataclass
class FittedCurve:
    """Result of curve fitting."""
    curve_type: CurveType
    parameters: Dict[str, float]
    r_squared: float
    rmse: float
    aic: float
    bic: float
    predictions: np.ndarray
    residuals: np.ndarray
    confidence_bands: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "curve_type": self.curve_type.value,
            "parameters": self.parameters,
            "r_squared": self.r_squared,
            "rmse": self.rmse,
            "aic": self.aic,
            "bic": self.bic,
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Generate predictions for new x values."""
        x = np.asarray(x)
        return self._curve_function(x, **self.parameters)

    def _curve_function(self, x: np.ndarray, **params) -> np.ndarray:
        """Evaluate curve function."""
        if self.curve_type == CurveType.LINEAR:
            return params["a"] * x + params["b"]
        elif self.curve_type == CurveType.LOG_LINEAR:
            return params["a"] * np.log(x + 1) + params["b"]
        elif self.curve_type == CurveType.POWER_LAW:
            return params["a"] * np.power(x, params["b"]) + params["c"]
        elif self.curve_type == CurveType.QUADRATIC:
            return params["a"] * x**2 + params["b"] * x + params["c"]
        elif self.curve_type == CurveType.INVERSE_U:
            return params["a"] * x * np.exp(-params["b"] * x) + params["c"]
        elif self.curve_type == CurveType.SIGMOID:
            return params["a"] / (1 + np.exp(-params["b"] * (x - params["c"]))) + params["d"]
        elif self.curve_type == CurveType.PLATEAU:
            return params["a"] * (1 - np.exp(-params["b"] * x)) + params["c"]
        else:
            raise ValueError(f"Unknown curve type: {self.curve_type}")


class ScalingCurveAnalyzer:
    """
    Analyzer for fitting and comparing scaling curves.

    Used to detect inverse scaling patterns where performance
    decreases at larger scales.
    """

    def __init__(self):
        self.curve_functions = {
            CurveType.LINEAR: self._linear,
            CurveType.LOG_LINEAR: self._log_linear,
            CurveType.POWER_LAW: self._power_law,
            CurveType.QUADRATIC: self._quadratic,
            CurveType.INVERSE_U: self._inverse_u,
            CurveType.SIGMOID: self._sigmoid,
            CurveType.PLATEAU: self._plateau,
        }

    def fit_curve(
        self,
        x: np.ndarray,
        y: np.ndarray,
        curve_type: CurveType,
        weights: Optional[np.ndarray] = None,
    ) -> FittedCurve:
        """
        Fit a specific curve type to data.

        Args:
            x: Independent variable (model scale)
            y: Dependent variable (performance)
            curve_type: Type of curve to fit
            weights: Optional sample weights

        Returns:
            FittedCurve object
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(x)

        func = self.curve_functions[curve_type]
        initial_params = self._get_initial_params(x, y, curve_type)

        try:
            if weights is not None:
                # Weighted least squares
                def objective(params):
                    pred = func(x, *params)
                    return np.sum(weights * (y - pred) ** 2)
                result = optimize.minimize(objective, initial_params, method="Nelder-Mead")
                popt = result.x
            else:
                popt, _ = optimize.curve_fit(
                    func, x, y, p0=initial_params, maxfev=10000
                )
        except (RuntimeError, ValueError) as e:
            # Fallback to simple optimization
            def objective(params):
                pred = func(x, *params)
                return np.sum((y - pred) ** 2)
            result = optimize.minimize(objective, initial_params, method="Nelder-Mead")
            popt = result.x

        # Compute predictions and metrics
        predictions = func(x, *popt)
        residuals = y - predictions

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean(residuals ** 2))

        # Information criteria
        k = len(popt)  # Number of parameters
        aic = n * np.log(ss_res / n) + 2 * k
        bic = n * np.log(ss_res / n) + k * np.log(n)

        # Parameter names
        param_names = self._get_param_names(curve_type)
        parameters = dict(zip(param_names, popt))

        return FittedCurve(
            curve_type=curve_type,
            parameters=parameters,
            r_squared=r_squared,
            rmse=rmse,
            aic=aic,
            bic=bic,
            predictions=predictions,
            residuals=residuals,
        )

    def fit_all_curves(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Dict[CurveType, FittedCurve]:
        """
        Fit all curve types and return results.

        Args:
            x: Independent variable
            y: Dependent variable

        Returns:
            Dictionary of fitted curves
        """
        results = {}
        for curve_type in CurveType:
            try:
                results[curve_type] = self.fit_curve(x, y, curve_type)
            except Exception as e:
                warnings.warn(f"Failed to fit {curve_type}: {e}")
        return results

    def select_best_curve(
        self,
        x: np.ndarray,
        y: np.ndarray,
        criterion: str = "bic",
    ) -> FittedCurve:
        """
        Select the best fitting curve based on information criterion.

        Args:
            x: Independent variable
            y: Dependent variable
            criterion: 'aic' or 'bic'

        Returns:
            Best fitting curve
        """
        all_fits = self.fit_all_curves(x, y)

        if criterion == "aic":
            best_type = min(all_fits, key=lambda k: all_fits[k].aic)
        elif criterion == "bic":
            best_type = min(all_fits, key=lambda k: all_fits[k].bic)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        return all_fits[best_type]

    def detect_inverse_scaling(
        self,
        x: np.ndarray,
        y: np.ndarray,
        threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Detect inverse scaling pattern.

        Args:
            x: Model scales
            y: Performance metrics
            threshold: Minimum performance drop to consider significant

        Returns:
            Analysis results
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        # Sort by scale
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]

        # Fit curves
        linear_fit = self.fit_curve(x, y, CurveType.LINEAR)
        quadratic_fit = self.fit_curve(x, y, CurveType.QUADRATIC)
        inverse_u_fit = self.fit_curve(x, y, CurveType.INVERSE_U)

        # Check for negative linear trend
        linear_negative = linear_fit.parameters.get("a", 0) < 0

        # Check for inverted-U shape
        quad_a = quadratic_fit.parameters.get("a", 0)
        is_inverse_u = quad_a < 0

        # Find peak in inverse-U
        peak_x = None
        peak_y = None
        if is_inverse_u:
            # Peak at -b/(2a)
            a = quadratic_fit.parameters["a"]
            b = quadratic_fit.parameters["b"]
            if a != 0:
                peak_x = -b / (2 * a)
                peak_y = quadratic_fit.predict(np.array([peak_x]))[0]

        # Calculate performance drop at largest scale
        performance_drop = y_sorted[0] - y_sorted[-1]
        relative_drop = performance_drop / y_sorted[0] if y_sorted[0] != 0 else 0

        # Determine if inverse scaling is present
        inverse_scaling_detected = (
            (linear_negative and linear_fit.r_squared > 0.5) or
            (is_inverse_u and quadratic_fit.r_squared > 0.6 and
             peak_x is not None and x.min() < peak_x < x.max())
        )

        return {
            "inverse_scaling_detected": inverse_scaling_detected,
            "pattern_type": "inverse_u" if is_inverse_u else "linear_decrease" if linear_negative else "none",
            "linear_fit": {
                "slope": linear_fit.parameters.get("a", 0),
                "r_squared": linear_fit.r_squared,
                "is_negative": linear_negative,
            },
            "quadratic_fit": {
                "a": quadratic_fit.parameters.get("a", 0),
                "r_squared": quadratic_fit.r_squared,
                "is_inverse_u": is_inverse_u,
                "peak_x": peak_x,
                "peak_y": peak_y,
            },
            "performance_change": {
                "start": float(y_sorted[0]),
                "end": float(y_sorted[-1]),
                "absolute_drop": float(performance_drop),
                "relative_drop": float(relative_drop),
            },
            "best_fit": self.select_best_curve(x, y).curve_type.value,
        }

    # Curve functions
    @staticmethod
    def _linear(x, a, b):
        return a * x + b

    @staticmethod
    def _log_linear(x, a, b):
        return a * np.log(x + 1) + b

    @staticmethod
    def _power_law(x, a, b, c):
        return a * np.power(np.maximum(x, 1e-10), b) + c

    @staticmethod
    def _quadratic(x, a, b, c):
        return a * x**2 + b * x + c

    @staticmethod
    def _inverse_u(x, a, b, c):
        return a * x * np.exp(-b * np.maximum(x, 1e-10)) + c

    @staticmethod
    def _sigmoid(x, a, b, c, d):
        return a / (1 + np.exp(-b * (x - c))) + d

    @staticmethod
    def _plateau(x, a, b, c):
        return a * (1 - np.exp(-b * x)) + c

    def _get_initial_params(
        self,
        x: np.ndarray,
        y: np.ndarray,
        curve_type: CurveType,
    ) -> List[float]:
        """Get initial parameter estimates."""
        y_mean = np.mean(y)
        y_range = np.ptp(y) or 1
        x_mean = np.mean(x)
        x_range = np.ptp(x) or 1

        if curve_type == CurveType.LINEAR:
            slope = (y[-1] - y[0]) / x_range if x_range > 0 else 0
            return [slope, y_mean]
        elif curve_type == CurveType.LOG_LINEAR:
            return [y_range, y_mean]
        elif curve_type == CurveType.POWER_LAW:
            return [y_range, 0.5, y.min()]
        elif curve_type == CurveType.QUADRATIC:
            return [-y_range / (x_range ** 2), y_range / x_range, y_mean]
        elif curve_type == CurveType.INVERSE_U:
            return [y_range, 1 / x_mean, y.min()]
        elif curve_type == CurveType.SIGMOID:
            return [y_range, 1 / x_range, x_mean, y.min()]
        elif curve_type == CurveType.PLATEAU:
            return [y_range, 1 / x_mean, y.min()]
        else:
            return [1.0] * 3

    def _get_param_names(self, curve_type: CurveType) -> List[str]:
        """Get parameter names for curve type."""
        names = {
            CurveType.LINEAR: ["a", "b"],
            CurveType.LOG_LINEAR: ["a", "b"],
            CurveType.POWER_LAW: ["a", "b", "c"],
            CurveType.QUADRATIC: ["a", "b", "c"],
            CurveType.INVERSE_U: ["a", "b", "c"],
            CurveType.SIGMOID: ["a", "b", "c", "d"],
            CurveType.PLATEAU: ["a", "b", "c"],
        }
        return names.get(curve_type, ["p1", "p2", "p3"])
