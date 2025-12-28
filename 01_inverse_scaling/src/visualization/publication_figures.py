"""
Publication-ready figure generator for Nature-style submissions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json


class PublicationFigureGenerator:
    """
    Generator for publication-ready figures suitable for Nature submission.
    """

    # Nature figure specifications
    SINGLE_COLUMN_WIDTH = 3.5  # inches (89mm)
    DOUBLE_COLUMN_WIDTH = 7.2  # inches (183mm)
    MAX_HEIGHT = 9.0  # inches

    # Nature-compliant style
    NATURE_STYLE = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 0.5,
        "lines.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    }

    # Accessible color palette (colorblind-friendly)
    COLORS = [
        "#0072B2",  # Blue
        "#D55E00",  # Orange
        "#009E73",  # Green
        "#CC79A7",  # Pink
        "#F0E442",  # Yellow
        "#56B4E9",  # Light blue
        "#E69F00",  # Gold
    ]

    def __init__(self):
        self._apply_style()

    def _apply_style(self):
        """Apply Nature-compliant style."""
        plt.rcParams.update(self.NATURE_STYLE)

    def create_main_figure(
        self,
        analyses: Dict[str, Any],
        output_path: Optional[Path] = None,
    ) -> Figure:
        """
        Create main results figure for publication.

        This is Figure 1: Evidence for Inverse Scaling

        Args:
            analyses: Analysis results
            output_path: Optional path to save figure

        Returns:
            Matplotlib Figure
        """
        fig = plt.figure(figsize=(self.DOUBLE_COLUMN_WIDTH, 6))
        gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.4)

        # Panel a: Example scaling curves showing inverse scaling
        ax_a = fig.add_subplot(gs[0, 0])
        self._plot_panel_a(ax_a, analyses)

        # Panel b: Task comparison
        ax_b = fig.add_subplot(gs[0, 1])
        self._plot_panel_b(ax_b, analyses)

        # Panel c: Pattern distribution
        ax_c = fig.add_subplot(gs[0, 2])
        self._plot_panel_c(ax_c, analyses)

        # Panel d: Optimal scale analysis
        ax_d = fig.add_subplot(gs[1, 0])
        self._plot_panel_d(ax_d, analyses)

        # Panel e: Overthinking correlation
        ax_e = fig.add_subplot(gs[1, 1])
        self._plot_panel_e(ax_e, analyses)

        # Panel f: Statistical summary
        ax_f = fig.add_subplot(gs[1, 2])
        self._plot_panel_f(ax_f, analyses)

        # Add panel labels
        for ax, label in zip(
            [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f],
            ["a", "b", "c", "d", "e", "f"]
        ):
            ax.text(
                -0.15, 1.1, label,
                transform=ax.transAxes,
                fontsize=10, fontweight="bold",
                va="top", ha="right"
            )

        if output_path:
            self.save_figure(fig, output_path)

        return fig

    def _plot_panel_a(self, ax, analyses: Dict[str, Any]):
        """Panel a: Representative inverse scaling curves."""
        tasks = analyses.get("tasks", {})

        # Find tasks with inverse scaling
        inverse_tasks = [
            (name, data) for name, data in tasks.items()
            if hasattr(data, "inverse_scaling_detected") and data.inverse_scaling_detected
            or (isinstance(data, dict) and data.get("inverse_scaling_detected"))
        ]

        if not inverse_tasks:
            # Use any available data
            inverse_tasks = list(tasks.items())[:2]

        for i, (name, data) in enumerate(inverse_tasks[:3]):
            if hasattr(data, "model_scales"):
                x = np.array(data.model_scales)
                y = np.array(data.accuracies)
            elif isinstance(data, dict):
                x = np.array(data.get("model_scales", []))
                y = np.array(data.get("accuracies", []))
            else:
                continue

            if len(x) == 0:
                continue

            # Sort and plot
            sort_idx = np.argsort(x)
            ax.plot(
                x[sort_idx], y[sort_idx],
                "o-", color=self.COLORS[i],
                markersize=4, linewidth=1,
                label=name[:15]
            )

        ax.set_xscale("log")
        ax.set_xlabel("Parameters")
        ax.set_ylabel("Accuracy")
        ax.set_title("Inverse Scaling Examples")
        ax.legend(fontsize=6, loc="best")
        ax.set_ylim(0, 1)

    def _plot_panel_b(self, ax, analyses: Dict[str, Any]):
        """Panel b: Performance heatmap across tasks and scales."""
        tasks = analyses.get("tasks", {})

        if not tasks:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title("Task Performance")
            return

        # Create performance matrix
        task_names = list(tasks.keys())
        all_models = set()
        for data in tasks.values():
            if hasattr(data, "model_names"):
                all_models.update(data.model_names)
            elif isinstance(data, dict) and "model_names" in data:
                all_models.update(data["model_names"])

        model_names = sorted(all_models)

        if not model_names:
            ax.text(0.5, 0.5, "No model data", ha="center", va="center")
            return

        matrix = np.zeros((len(task_names), len(model_names)))

        for i, task in enumerate(task_names):
            data = tasks[task]
            if hasattr(data, "model_names"):
                names = data.model_names
                accs = data.accuracies
            elif isinstance(data, dict):
                names = data.get("model_names", [])
                accs = data.get("accuracies", [])
            else:
                continue

            for name, acc in zip(names, accs):
                if name in model_names:
                    j = model_names.index(name)
                    matrix[i, j] = acc

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([m.split("/")[-1][:8] for m in model_names], rotation=45, ha="right")
        ax.set_yticks(range(len(task_names)))
        ax.set_yticklabels([t[:12] for t in task_names])
        ax.set_title("Performance Matrix")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)

    def _plot_panel_c(self, ax, analyses: Dict[str, Any]):
        """Panel c: Distribution of scaling patterns."""
        overall = analyses.get("overall", {})
        pattern_dist = overall.get("pattern_distribution", {})

        if not pattern_dist:
            ax.text(0.5, 0.5, "No pattern data", ha="center", va="center")
            ax.set_title("Pattern Distribution")
            return

        patterns = list(pattern_dist.keys())
        counts = list(pattern_dist.values())

        colors = []
        for p in patterns:
            if "inverse" in p.lower() or "negative" in p.lower():
                colors.append(self.COLORS[1])  # Orange for inverse
            elif "positive" in p.lower():
                colors.append(self.COLORS[2])  # Green for positive
            else:
                colors.append(self.COLORS[5])  # Light blue for neutral

        bars = ax.bar(range(len(patterns)), counts, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(patterns)))
        ax.set_xticklabels([p.replace("_", "\n")[:10] for p in patterns], fontsize=6)
        ax.set_ylabel("Count")
        ax.set_title("Scaling Patterns")

    def _plot_panel_d(self, ax, analyses: Dict[str, Any]):
        """Panel d: Optimal scale distribution."""
        tasks = analyses.get("tasks", {})

        optimal_scales = []
        for name, data in tasks.items():
            if hasattr(data, "optimal_scale") and data.optimal_scale:
                optimal_scales.append(data.optimal_scale)
            elif isinstance(data, dict) and data.get("optimal_scale"):
                optimal_scales.append(data["optimal_scale"])

        if optimal_scales:
            ax.hist(
                np.log10(optimal_scales),
                bins=10,
                color=self.COLORS[0],
                edgecolor="black",
                linewidth=0.5,
            )
            ax.set_xlabel("log₁₀(Optimal Scale)")
            ax.set_ylabel("Count")
        else:
            ax.text(0.5, 0.5, "No optimal scales\ndetected", ha="center", va="center")

        ax.set_title("Optimal Scale Distribution")

    def _plot_panel_e(self, ax, analyses: Dict[str, Any]):
        """Panel e: Correlation between scale and performance drop."""
        tasks = analyses.get("tasks", {})

        x_data = []
        y_data = []

        for name, data in tasks.items():
            if hasattr(data, "curve_analysis"):
                curve = data.curve_analysis
            elif isinstance(data, dict) and "curve_analysis" in data:
                curve = data["curve_analysis"]
            else:
                continue

            perf_change = curve.get("performance_change", {})
            if perf_change:
                rel_drop = perf_change.get("relative_drop", 0)
                start = perf_change.get("start", 0)
                if start > 0:
                    x_data.append(start)
                    y_data.append(rel_drop)

        if x_data:
            ax.scatter(x_data, y_data, c=self.COLORS[0], s=20, alpha=0.7)
            ax.set_xlabel("Initial Accuracy")
            ax.set_ylabel("Performance Drop")
        else:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")

        ax.set_title("Performance vs Drop")

    def _plot_panel_f(self, ax, analyses: Dict[str, Any]):
        """Panel f: Statistical summary table."""
        ax.axis("off")

        overall = analyses.get("overall", {})

        summary = [
            ["Metric", "Value"],
            ["Tasks Evaluated", str(overall.get("total_tasks", "N/A"))],
            ["Models Evaluated", str(overall.get("models_evaluated", "N/A"))],
            ["Inverse Scaling Tasks", str(overall.get("inverse_scaling_detected", "N/A"))],
            ["Proportion Inverse", f"{overall.get('inverse_scaling_proportion', 0):.1%}"],
        ]

        table = ax.table(
            cellText=summary,
            loc="center",
            cellLoc="center",
            colWidths=[0.5, 0.4],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.5)

        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor("#E6E6E6")
            table[(0, i)].set_text_props(weight="bold")

        ax.set_title("Summary Statistics", pad=20)

    def create_supplementary_figure(
        self,
        analyses: Dict[str, Any],
        task_name: str,
        output_path: Optional[Path] = None,
    ) -> Figure:
        """
        Create supplementary figure for a specific task.

        Args:
            analyses: Analysis results
            task_name: Task to visualize
            output_path: Optional output path

        Returns:
            Matplotlib Figure
        """
        fig = plt.figure(figsize=(self.DOUBLE_COLUMN_WIDTH, 4))
        gs = GridSpec(1, 3, figure=fig, wspace=0.4)

        task_data = analyses.get("tasks", {}).get(task_name, {})

        # Panel a: Raw data with fitted curve
        ax_a = fig.add_subplot(gs[0, 0])
        self._plot_task_curve(ax_a, task_data)
        ax_a.set_title(f"{task_name}: Scaling Curve")

        # Panel b: Residuals
        ax_b = fig.add_subplot(gs[0, 1])
        self._plot_residuals(ax_b, task_data)
        ax_b.set_title("Fit Residuals")

        # Panel c: Statistical details
        ax_c = fig.add_subplot(gs[0, 2])
        self._plot_stats_summary(ax_c, task_data)
        ax_c.set_title("Statistical Analysis")

        # Panel labels
        for ax, label in zip([ax_a, ax_b, ax_c], ["a", "b", "c"]):
            ax.text(-0.15, 1.1, label, transform=ax.transAxes,
                   fontsize=10, fontweight="bold", va="top", ha="right")

        if output_path:
            self.save_figure(fig, output_path)

        return fig

    def _plot_task_curve(self, ax, task_data):
        """Plot task-specific scaling curve."""
        if hasattr(task_data, "model_scales"):
            x = np.array(task_data.model_scales)
            y = np.array(task_data.accuracies)
        elif isinstance(task_data, dict):
            x = np.array(task_data.get("model_scales", []))
            y = np.array(task_data.get("accuracies", []))
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        if len(x) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        sort_idx = np.argsort(x)
        ax.errorbar(
            x[sort_idx], y[sort_idx],
            fmt="o-", color=self.COLORS[0],
            markersize=5, capsize=3
        )
        ax.set_xscale("log")
        ax.set_xlabel("Parameters")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)

    def _plot_residuals(self, ax, task_data):
        """Plot residuals from curve fit."""
        curve_analysis = (
            task_data.curve_analysis if hasattr(task_data, "curve_analysis")
            else task_data.get("curve_analysis", {})
        )

        if not curve_analysis:
            ax.text(0.5, 0.5, "No fit data", ha="center", va="center")
            return

        # Placeholder - would need actual residuals
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")

    def _plot_stats_summary(self, ax, task_data):
        """Plot statistical summary."""
        ax.axis("off")

        stats = (
            task_data.statistical_tests if hasattr(task_data, "statistical_tests")
            else task_data.get("statistical_tests", {})
        )

        if not stats:
            ax.text(0.5, 0.5, "No stats", ha="center", va="center")
            return

        spearman = stats.get("spearman", {})
        text = f"""Statistical Summary

Spearman ρ: {spearman.get('correlation', 'N/A'):.3f}
p-value: {spearman.get('p_value', 'N/A'):.4f}
Significant: {spearman.get('significant', 'N/A')}
        """

        ax.text(0.1, 0.9, text, transform=ax.transAxes,
               fontsize=7, verticalalignment="top", fontfamily="monospace")

    def save_figure(
        self,
        fig: Figure,
        path: Path,
        formats: List[str] = ["pdf", "png", "eps"],
    ) -> List[Path]:
        """Save figure in publication formats."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        saved = []
        for fmt in formats:
            save_path = path.with_suffix(f".{fmt}")
            fig.savefig(
                save_path,
                format=fmt,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
                transparent=False,
            )
            saved.append(save_path)

        return saved
