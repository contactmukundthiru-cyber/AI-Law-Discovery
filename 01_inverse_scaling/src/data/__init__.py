"""
Data module for inverse scaling experiments.

This module provides task generators and data loaders for evaluating
inverse scaling phenomena across different task types.
"""

from .task_generators import (
    TaskGenerator,
    SimpleArithmeticGenerator,
    RuleFollowingGenerator,
    SimplePatternGenerator,
    LiteralInstructionGenerator,
    PersonaMaintenanceGenerator,
)
from .data_loader import DataLoader, TaskDataset
from .evaluators import (
    TaskEvaluator,
    ArithmeticEvaluator,
    ExactMatchEvaluator,
    FormatComplianceEvaluator,
)

__all__ = [
    "TaskGenerator",
    "SimpleArithmeticGenerator",
    "RuleFollowingGenerator",
    "SimplePatternGenerator",
    "LiteralInstructionGenerator",
    "PersonaMaintenanceGenerator",
    "DataLoader",
    "TaskDataset",
    "TaskEvaluator",
    "ArithmeticEvaluator",
    "ExactMatchEvaluator",
    "FormatComplianceEvaluator",
]
