"""
Training Scars Experiments Module

Implements experiments to detect and characterize permanent training scars
in neural networks.
"""

from .curriculum_experiments import CurriculumExperiment, DataOrderingExperiment
from .irreversibility_tests import IrreversibilityTester, ScarDetector
from .scar_analysis import ScarAnalyzer, ScarCharacterization

__all__ = [
    "CurriculumExperiment",
    "DataOrderingExperiment",
    "IrreversibilityTester",
    "ScarDetector",
    "ScarAnalyzer",
    "ScarCharacterization",
]
