"""Tests for analysis module."""

import pytest
import numpy as np

from src.analysis.statistics import (
    StatisticalAnalyzer,
    ConfidenceInterval,
    HypothesisTest,
    BootstrapEstimator,
)
from src.analysis.scaling_curves import (
    ScalingCurveAnalyzer,
    CurveType,
    FittedCurve,
)


class TestConfidenceInterval:
    """Tests for ConfidenceInterval dataclass."""

    def test_creation(self):
        ci = ConfidenceInterval(
            lower=0.4,
            upper=0.6,
            point_estimate=0.5,
            confidence_level=0.95,
            method="test",
        )
        assert ci.width == pytest.approx(0.2)
        assert ci.margin_of_error == pytest.approx(0.1)

    def test_contains(self):
        ci = ConfidenceInterval(0.4, 0.6, 0.5, 0.95, "test")
        assert ci.contains(0.5)
        assert ci.contains(0.4)
        assert not ci.contains(0.3)


class TestBootstrapEstimator:
    """Tests for bootstrap estimation."""

    def test_confidence_interval(self):
        data = np.random.normal(0.5, 0.1, size=100)
        estimator = BootstrapEstimator(n_bootstrap=1000, random_state=42)

        ci = estimator.confidence_interval(data)

        assert ci.lower < ci.point_estimate < ci.upper
        assert ci.confidence_level == 0.95
        assert 0.3 < ci.point_estimate < 0.7

    def test_permutation_test(self):
        # Two clearly different groups
        group1 = np.random.normal(0.3, 0.1, size=50)
        group2 = np.random.normal(0.7, 0.1, size=50)

        estimator = BootstrapEstimator(n_bootstrap=1000, random_state=42)
        result = estimator.permutation_test(group1, group2)

        assert result.p_value < 0.05
        assert result.significant

    def test_permutation_test_similar_groups(self):
        # Two similar groups
        data = np.random.normal(0.5, 0.1, size=100)
        group1 = data[:50]
        group2 = data[50:]

        estimator = BootstrapEstimator(n_bootstrap=1000, random_state=42)
        result = estimator.permutation_test(group1, group2)

        # Should not be significant
        assert result.p_value > 0.05 or not result.significant


class TestStatisticalAnalyzer:
    """Tests for statistical analyzer."""

    def test_confidence_interval_methods(self):
        data = np.random.normal(0.5, 0.1, size=100)
        analyzer = StatisticalAnalyzer(confidence_level=0.95)

        # Bootstrap
        ci_bootstrap = analyzer.compute_confidence_interval(data, method="bootstrap")
        assert ci_bootstrap.lower < ci_bootstrap.upper

        # T-distribution
        ci_t = analyzer.compute_confidence_interval(data, method="t")
        assert ci_t.lower < ci_t.upper

    def test_compare_groups(self):
        group1 = np.random.normal(0.3, 0.1, size=50)
        group2 = np.random.normal(0.7, 0.1, size=50)

        analyzer = StatisticalAnalyzer()

        # Mann-Whitney
        result_mw = analyzer.compare_groups(group1, group2, test="mann_whitney")
        assert result_mw.p_value < 0.05

        # T-test
        result_t = analyzer.compare_groups(group1, group2, test="t_test")
        assert result_t.p_value < 0.05

    def test_monotonicity(self):
        # Positive monotonic relationship
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

        analyzer = StatisticalAnalyzer()
        result = analyzer.test_monotonicity(x, y)

        assert result["spearman"]["correlation"] > 0.9
        assert not result["inverse_scaling_detected"]

    def test_inverse_scaling_detection(self):
        # Inverse relationship
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([0.9, 0.7, 0.5, 0.3, 0.1])

        analyzer = StatisticalAnalyzer()
        result = analyzer.test_monotonicity(x, y)

        assert result["spearman"]["correlation"] < -0.9
        assert result["inverse_scaling_detected"]

    def test_detect_peak(self):
        # Inverted U-shape
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([0.3, 0.6, 0.9, 0.6, 0.3])

        analyzer = StatisticalAnalyzer()
        peak = analyzer.detect_peak(x, y)

        assert peak is not None
        assert peak["peak_x"] == 3
        assert peak["peak_y"] == 0.9


class TestScalingCurveAnalyzer:
    """Tests for scaling curve analysis."""

    def test_fit_linear(self):
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=float)

        analyzer = ScalingCurveAnalyzer()
        result = analyzer.fit_curve(x, y, CurveType.LINEAR)

        assert result.r_squared > 0.99
        assert abs(result.parameters["a"] - 0.2) < 0.01

    def test_fit_quadratic(self):
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        # y = -0.1*x^2 + 0.8*x (inverted U)
        y = -0.1 * x**2 + 0.8 * x

        analyzer = ScalingCurveAnalyzer()
        result = analyzer.fit_curve(x, y, CurveType.QUADRATIC)

        assert result.r_squared > 0.99
        assert result.parameters["a"] < 0  # Negative quadratic term

    def test_detect_inverse_scaling(self):
        # Clear inverse scaling pattern
        x = np.array([1e9, 1e10, 5e10, 1e11, 5e11])
        y = np.array([0.9, 0.85, 0.75, 0.65, 0.50])

        analyzer = ScalingCurveAnalyzer()
        result = analyzer.detect_inverse_scaling(x, y)

        assert result["inverse_scaling_detected"]
        assert "negative" in result["pattern_type"] or result["linear_fit"]["is_negative"]

    def test_detect_positive_scaling(self):
        # Positive scaling
        x = np.array([1e9, 1e10, 5e10, 1e11, 5e11])
        y = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

        analyzer = ScalingCurveAnalyzer()
        result = analyzer.detect_inverse_scaling(x, y)

        assert not result["inverse_scaling_detected"]

    def test_select_best_curve(self):
        # Linear data
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=float)

        analyzer = ScalingCurveAnalyzer()
        best = analyzer.select_best_curve(x, y, criterion="bic")

        # Linear should fit best (or nearly as well as more complex)
        assert best.r_squared > 0.9

    def test_fit_all_curves(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
        y = np.array([0.2, 0.4, 0.55, 0.65, 0.72, 0.78, 0.82, 0.85], dtype=float)

        analyzer = ScalingCurveAnalyzer()
        results = analyzer.fit_all_curves(x, y)

        assert len(results) > 0
        assert all(isinstance(v, FittedCurve) for v in results.values())
