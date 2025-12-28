"""
Analysis module for inverse scaling research.

Provides statistical analysis, curve fitting, and hypothesis testing
for detecting inverse scaling patterns.
"""

from .statistics import (
    StatisticalAnalyzer,
    ConfidenceInterval,
    HypothesisTest,
    BootstrapEstimator,
)
from .scaling_curves import (
    ScalingCurveAnalyzer,
    CurveType,
    FittedCurve,
)
from .results_analyzer import ResultsAnalyzer

__all__ = [
    "StatisticalAnalyzer",
    "ConfidenceInterval",
    "HypothesisTest",
    "BootstrapEstimator",
    "ScalingCurveAnalyzer",
    "CurveType",
    "FittedCurve",
    "ResultsAnalyzer",
]
