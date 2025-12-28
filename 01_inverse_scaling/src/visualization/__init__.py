"""
Visualization module for inverse scaling research.

Provides publication-ready plotting utilities for scaling curves,
comparison charts, and statistical visualizations.
"""

from .scaling_plots import ScalingPlotter
from .publication_figures import PublicationFigureGenerator
from .interactive_plots import InteractivePlotter

__all__ = [
    "ScalingPlotter",
    "PublicationFigureGenerator",
    "InteractivePlotter",
]
