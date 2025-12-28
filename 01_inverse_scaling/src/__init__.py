"""
Inverse Scaling Research Project
================================

This package investigates whether some capabilities in language models
peak at intermediate scale and then decline.

Modules:
    data: Task datasets and data loaders
    models: Model interface wrappers for multi-provider evaluation
    experiments: Experiment runners and trial management
    analysis: Statistical analysis and hypothesis testing
    visualization: Plotting utilities for publication-ready figures
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from . import data
from . import models
from . import experiments
from . import analysis
from . import visualization
