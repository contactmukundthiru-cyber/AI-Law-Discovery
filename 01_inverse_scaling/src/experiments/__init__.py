"""
Experiments module for inverse scaling research.

Provides experiment runners, trial management, and result tracking.
"""

from .runner import ExperimentRunner, ExperimentConfig
from .trial import Trial, TrialResult, TrialStatus
from .scheduler import ExperimentScheduler

__all__ = [
    "ExperimentRunner",
    "ExperimentConfig",
    "Trial",
    "TrialResult",
    "TrialStatus",
    "ExperimentScheduler",
]
