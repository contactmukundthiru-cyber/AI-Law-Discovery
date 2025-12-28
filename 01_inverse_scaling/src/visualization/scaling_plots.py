"""
Scaling plots for inverse scaling research.

Provides matplotlib-based visualizations for scaling curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from ..analysis.scaling_curves import FittedCurve, CurveType
from ..analysis.statistics import ConfidenceInterval


class ScalingPlotter:
    """
    Plotter for scaling curve visualizations.
    """

    # Nature-style color palette
    COLORS = {
        "primary": "#2E86AB",
        "secondary": "#A23B72",
        "tertiary": "#F18F01",
        "quaternary": "#C73E1D",
        "positive": "#2ECC71",
        "negative": "#E74C3C",
        "neutral": "#7F8C8D",
    }

    # Publication-ready style settings
    STYLE = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }

    def __init__(self):
        plt.rcParams.update(self.STYLE)

    def plot_scaling_curve(
        self,
        x: np.ndarray,
        y: np.ndarray,
        model_names: Optional[List[str]] = None,
        fitted_curve: Optional[FittedCurve] = None,
        confidence_intervals: Optional[List[ConfidenceInterval]] = None,
        title: str = "Scaling Curve",
        xlabel: str = "Model Parameters",
        ylabel: str = "Accuracy",
        ax: Optional[Axes] = None,
        show_legend: bool = True,
        log_scale_x: bool = True,
    ) -> Tuple[Figure, Axes]:
        """
        Plot a scaling curve with optional fitted line and confidence intervals.

        Args:
            x: Model scales (x-axis)
            y: Performance metrics (y-axis)
            model_names: Names for each point
            fitted_curve: Optional fitted curve to overlay
            confidence_intervals: Optional CIs for each point
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            ax: Optional axes to plot on
            show_legend: Whether to show legend
            log_scale_x: Use log scale for x-axis

        Returns:
            Figure and Axes objects
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        x = np.asarray(x)
        y = np.asarray(y)

        # Sort by x for proper line plotting
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]

        # Plot confidence intervals if provided
        if confidence_intervals:
            ci_sorted = [confidence_intervals[i] for i in sort_idx]
            lower = np.array([ci.lower for ci in ci_sorted])
            upper = np.array([ci.upper for ci in ci_sorted])

            ax.fill_between(
                x_sorted, lower, upper,
                alpha=0.2, color=self.COLORS["primary"],
                label="95% CI"
            )

        # Plot data points
        ax.scatter(
            x_sorted, y_sorted,
            s=80, c=self.COLORS["primary"],
            edgecolors="white", linewidth=1.5,
            zorder=10, label="Observed"
        )

        # Add model name annotations
        if model_names:
            names_sorted = [model_names[i] for i in sort_idx]
            for xi, yi, name in zip(x_sorted, y_sorted, names_sorted):
                # Shorten model name for display
                short_name = name.split("/")[-1][:15]
                ax.annotate(
                    short_name,
                    (xi, yi),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=7,
                    alpha=0.7,
                )

        # Plot fitted curve if provided
        if fitted_curve:
            x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 100)
            if log_scale_x:
                x_smooth = np.logspace(
                    np.log10(x_sorted.min()),
                    np.log10(x_sorted.max()),
                    100
                )
            y_smooth = fitted_curve.predict(x_smooth)

            curve_color = (
                self.COLORS["negative"]
                if fitted_curve.curve_type in [CurveType.INVERSE_U, CurveType.QUADRATIC]
                and fitted_curve.parameters.get("a", 0) < 0
                else self.COLORS["tertiary"]
            )

            ax.plot(
                x_smooth, y_smooth,
                color=curve_color, linewidth=2,
                linestyle="--", alpha=0.8,
                label=f"Fitted ({fitted_curve.curve_type.value})"
            )

        # Set axis properties
        if log_scale_x:
            ax.set_xscale("log")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        if show_legend:
            ax.legend(loc="best", frameon=True, framealpha=0.9)

        ax.grid(True, alpha=0.3, linestyle="--")

        plt.tight_layout()
        return fig, ax

    def plot_task_comparison(
        self,
        task_results: Dict[str, Dict[str, float]],
        title: str = "Performance by Task and Scale",
        figsize: Tuple[int, int] = (12, 6),
    ) -> Tuple[Figure, Axes]:
        """
        Plot comparison across tasks showing inverse scaling patterns.

        Args:
            task_results: Dict mapping task names to {model: accuracy} dicts
            title: Plot title
            figsize: Figure size

        Returns:
            Figure and Axes objects
        """
        fig, ax = plt.subplots(figsize=figsize)

        tasks = list(task_results.keys())
        models = sorted(set(
            model for results in task_results.values()
            for model in results.keys()
        ))

        x = np.arange(len(tasks))
        width = 0.8 / len(models)

        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))

        for i, model in enumerate(models):
            accuracies = [
                task_results[task].get(model, 0)
                for task in tasks
            ]
            offset = (i - len(models) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset, accuracies, width,
                label=model.split("/")[-1][:20],
                color=colors[i], edgecolor="white", linewidth=0.5
            )

        ax.set_xlabel("Task")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45, ha="right")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return fig, ax

    def plot_inverse_scaling_summary(
        self,
        analyses: Dict[str, Any],
        figsize: Tuple[int, int] = (10, 8),
    ) -> Tuple[Figure, List[Axes]]:
        """
        Create summary visualization of inverse scaling findings.

        Args:
            analyses: Analysis results from ResultsAnalyzer
            figsize: Figure size

        Returns:
            Figure and list of Axes objects
        """
        fig = plt.figure(figsize=figsize)

        # Create grid
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        axes = []

        # 1. Pattern distribution (pie chart)
        ax1 = fig.add_subplot(gs[0, 0])
        pattern_dist = analyses.get("overall", {}).get("pattern_distribution", {})
        if pattern_dist:
            colors = [
                self.COLORS["negative"] if "inverse" in p or "negative" in p
                else self.COLORS["positive"] if "positive" in p
                else self.COLORS["neutral"]
                for p in pattern_dist.keys()
            ]
            ax1.pie(
                pattern_dist.values(),
                labels=[p.replace("_", " ").title() for p in pattern_dist.keys()],
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
            )
            ax1.set_title("Scaling Pattern Distribution")
        axes.append(ax1)

        # 2. Inverse scaling tasks bar
        ax2 = fig.add_subplot(gs[0, 1])
        inverse_tasks = analyses.get("inverse_scaling_tasks", [])
        all_tasks = list(analyses.get("tasks", {}).keys())

        task_colors = [
            self.COLORS["negative"] if t in inverse_tasks
            else self.COLORS["positive"]
            for t in all_tasks
        ]

        task_accs = []
        for t in all_tasks:
            task_data = analyses.get("tasks", {}).get(t, {})
            if hasattr(task_data, "accuracies") and task_data.accuracies:
                task_accs.append(np.mean(task_data.accuracies))
            elif isinstance(task_data, dict) and "accuracies" in task_data:
                task_accs.append(np.mean(task_data["accuracies"]))
            else:
                task_accs.append(0)

        bars = ax2.barh(all_tasks, task_accs, color=task_colors, edgecolor="white")
        ax2.set_xlabel("Mean Accuracy")
        ax2.set_title("Task Performance (Red = Inverse Scaling)")
        ax2.set_xlim(0, 1)
        axes.append(ax2)

        # 3. Summary statistics
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis("off")

        overall = analyses.get("overall", {})
        summary_text = f"""
        Inverse Scaling Analysis Summary
        ================================

        Total Tasks Evaluated: {overall.get('total_tasks', 'N/A')}
        Models Evaluated: {overall.get('models_evaluated', 'N/A')}
        Total Evaluations: {overall.get('total_evaluations', 'N/A')}

        Inverse Scaling Detected: {overall.get('inverse_scaling_detected', 0)} tasks
        Proportion with Inverse Scaling: {overall.get('inverse_scaling_proportion', 0):.1%}

        Tasks Exhibiting Inverse Scaling:
        {', '.join(inverse_tasks) if inverse_tasks else 'None detected'}
        """

        ax3.text(
            0.1, 0.9, summary_text,
            transform=ax3.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        axes.append(ax3)

        plt.tight_layout()
        return fig, axes

    def save_figure(
        self,
        fig: Figure,
        path: Path,
        formats: List[str] = ["png", "pdf", "svg"],
    ) -> List[Path]:
        """
        Save figure in multiple formats for publication.

        Args:
            fig: Figure to save
            path: Base path (without extension)
            formats: List of formats to save

        Returns:
            List of saved file paths
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for fmt in formats:
            save_path = path.with_suffix(f".{fmt}")
            fig.savefig(
                save_path,
                format=fmt,
                dpi=300 if fmt == "png" else None,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            saved_paths.append(save_path)

        return saved_paths
