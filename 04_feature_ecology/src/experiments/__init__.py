"""
Feature Ecology Experiments Module

Implements experiments to study ecological dynamics of features in neural networks.
"""

from .feature_tracking import FeatureTracker, FeaturePopulation
from .ecological_analysis import EcologicalAnalyzer, LotkaVolterraModel
from .diversity_metrics import DiversityCalculator, ExtinctionDetector

__all__ = [
    "FeatureTracker",
    "FeaturePopulation",
    "EcologicalAnalyzer",
    "LotkaVolterraModel",
    "DiversityCalculator",
    "ExtinctionDetector",
]
