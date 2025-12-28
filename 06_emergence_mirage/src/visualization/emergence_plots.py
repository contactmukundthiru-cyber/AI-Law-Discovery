"""
Visualization for emergence analysis.

Creates plots comparing emergence patterns across metrics and scales.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional
import logging

from ..experiments.emergence_detector import EmergenceResult
from ..experiments.metric_analyzer import MetricComparison
from ..analysis.curve_fitting import CurveFitResult

logger = logging.getLogger(__name__)


class EmergencePlotter:
    """Creates visualizations for emergence analysis."""

    def __init__(self, output_dir: str = "results/plots"):
        self.output_dir = output_dir

    def plot_metric_comparison(
        self,
        results: List[EmergenceResult],
        title: str = "Metric Comparison Across Scales"
    ) -> go.Figure:
        """
        Plot multiple metrics across model scales.

        Shows how different metrics reveal different emergence patterns.
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Binary Accuracy', 'Partial Credit', 'Log Probability',
                'Rank Score', 'Entropy Score', 'All Metrics'
            ]
        )

        # Sort by parameter count
        sorted_results = sorted(results, key=lambda r: r.parameter_count)
        scales = [r.parameter_count for r in sorted_results]
        log_scales = [np.log10(s) for s in scales]

        metrics = {
            'binary_accuracy': ('Binary Accuracy', 'blue'),
            'partial_credit_score': ('Partial Credit', 'green'),
            'log_probability_score': ('Log Probability', 'red'),
            'rank_score': ('Rank Score', 'purple'),
            'entropy_score': ('Entropy Score', 'orange')
        }

        positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]

        for (metric, (name, color)), (row, col) in zip(metrics.items(), positions):
            values = [getattr(r, metric) for r in sorted_results]

            fig.add_trace(
                go.Scatter(
                    x=log_scales,
                    y=values,
                    mode='lines+markers',
                    name=name,
                    line=dict(color=color),
                    showlegend=False
                ),
                row=row, col=col
            )

            # Add to combined plot
            fig.add_trace(
                go.Scatter(
                    x=log_scales,
                    y=values,
                    mode='lines+markers',
                    name=name,
                    line=dict(color=color)
                ),
                row=2, col=3
            )

        fig.update_layout(
            title=title,
            height=600,
            showlegend=True
        )

        # Update x-axis labels
        for i in range(1, 7):
            row = (i - 1) // 3 + 1
            col = (i - 1) % 3 + 1
            fig.update_xaxes(title_text="Log₁₀(Parameters)", row=row, col=col)

        return fig

    def plot_emergence_curve(
        self,
        comparison: MetricComparison,
        fit_results: Optional[Dict[str, CurveFitResult]] = None
    ) -> go.Figure:
        """
        Plot emergence curve with fitted models.
        """
        fig = go.Figure()

        # Extract data
        scales = [s[0] for s in comparison.scores_by_scale]
        scores = [s[1] for s in comparison.scores_by_scale]
        log_scales = [np.log10(s) for s in scales]

        # Plot actual data
        fig.add_trace(go.Scatter(
            x=log_scales,
            y=scores,
            mode='markers',
            name='Observed',
            marker=dict(size=10, color='black')
        ))

        # Plot fitted curves
        if fit_results:
            colors = ['blue', 'green', 'red', 'purple', 'orange']
            x_smooth = np.linspace(min(log_scales), max(log_scales), 100)

            for (name, result), color in zip(fit_results.items(), colors):
                fig.add_trace(go.Scatter(
                    x=x_smooth,
                    y=result.predictions if len(result.predictions) == 100
                      else np.interp(x_smooth, log_scales, result.predictions),
                    mode='lines',
                    name=f'{name} (R²={result.r_squared:.3f})',
                    line=dict(color=color, dash='dash')
                ))

        # Add emergence point if detected
        if comparison.apparent_emergence_point:
            emergence_log = np.log10(comparison.apparent_emergence_point)
            fig.add_vline(
                x=emergence_log,
                line_dash="dot",
                line_color="red",
                annotation_text="Apparent Emergence"
            )

        fig.update_layout(
            title=f"Emergence Curve: {comparison.metric_name}",
            xaxis_title="Log₁₀(Parameters)",
            yaxis_title="Score",
            showlegend=True
        )

        return fig

    def plot_threshold_analysis(
        self,
        threshold_effects: List[Dict[str, Any]],
        raw_scores: List[tuple]
    ) -> go.Figure:
        """
        Visualize how different thresholds affect emergence appearance.
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Raw Scores vs Thresholds', 'Emergence Points by Threshold']
        )

        # Plot raw scores
        scales = [s[0] for s in raw_scores]
        scores = [s[1] for s in raw_scores]
        log_scales = [np.log10(s) for s in scales]

        fig.add_trace(
            go.Scatter(
                x=log_scales,
                y=scores,
                mode='lines+markers',
                name='Raw Scores',
                line=dict(color='blue')
            ),
            row=1, col=1
        )

        # Add threshold lines
        for effect in threshold_effects[::3]:  # Every 3rd for clarity
            fig.add_hline(
                y=effect['threshold'],
                line_dash="dash",
                line_color="gray",
                opacity=0.5,
                row=1, col=1
            )

        # Plot emergence points by threshold
        thresholds = [e['threshold'] for e in threshold_effects]
        emergence_scales = [
            np.log10(e['emergence_scale']) if e['emergence_scale'] else None
            for e in threshold_effects
        ]

        valid_thresh = [t for t, e in zip(thresholds, emergence_scales) if e is not None]
        valid_emergence = [e for e in emergence_scales if e is not None]

        fig.add_trace(
            go.Scatter(
                x=valid_thresh,
                y=valid_emergence,
                mode='markers+lines',
                name='Emergence Point',
                marker=dict(size=8, color='red')
            ),
            row=1, col=2
        )

        fig.update_layout(
            title="Threshold Effect on Emergence Detection",
            height=400
        )
        fig.update_xaxes(title_text="Log₁₀(Parameters)", row=1, col=1)
        fig.update_xaxes(title_text="Threshold", row=1, col=2)
        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_yaxes(title_text="Log₁₀(Emergence Scale)", row=1, col=2)

        return fig

    def plot_emergence_classification(
        self,
        classification_results: Dict[str, Any]
    ) -> go.Figure:
        """
        Visualize emergence classification and bootstrap analysis.
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Pattern Classification', 'Bootstrap Distribution'],
            specs=[[{"type": "indicator"}, {"type": "pie"}]]
        )

        classification = classification_results['classification']

        # Add classification indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=classification.confidence * 100,
                title={'text': f"Pattern: {classification.pattern_type}"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "gray"},
                        {'range': [75, 100], 'color': "darkgray"}
                    ]
                }
            ),
            row=1, col=1
        )

        # Bootstrap distribution pie chart
        probs = classification_results.get('bootstrap_pattern_probabilities', {})
        if probs:
            fig.add_trace(
                go.Pie(
                    labels=list(probs.keys()),
                    values=list(probs.values()),
                    hole=0.4
                ),
                row=1, col=2
            )

        fig.update_layout(
            title="Emergence Classification Analysis",
            height=400
        )

        return fig

    def create_summary_figure(
        self,
        results: List[EmergenceResult],
        comparisons: Dict[str, MetricComparison],
        classification: Dict[str, Any]
    ) -> go.Figure:
        """
        Create comprehensive summary figure for publication.
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'A. Metric Comparison',
                'B. Best Fit Models',
                'C. Emergence Detection',
                'D. Artifact Analysis'
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )

        sorted_results = sorted(results, key=lambda r: r.parameter_count)
        log_scales = [np.log10(r.parameter_count) for r in sorted_results]

        # A. Metric comparison
        for metric, color in [
            ('binary_accuracy', 'blue'),
            ('partial_credit_score', 'green'),
            ('log_probability_score', 'red')
        ]:
            values = [getattr(r, metric) for r in sorted_results]
            fig.add_trace(
                go.Scatter(
                    x=log_scales, y=values,
                    mode='lines+markers',
                    name=metric.replace('_', ' ').title(),
                    line=dict(color=color)
                ),
                row=1, col=1
            )

        # B. Model fits
        for name, comp in list(comparisons.items())[:3]:
            scales = [np.log10(s[0]) for s in comp.scores_by_scale]
            scores = [s[1] for s in comp.scores_by_scale]
            fig.add_trace(
                go.Scatter(
                    x=scales, y=scores,
                    mode='markers',
                    name=name,
                    showlegend=False
                ),
                row=1, col=2
            )

        # C. Emergence points
        emergence_points = []
        for name, comp in comparisons.items():
            if comp.apparent_emergence_point:
                emergence_points.append({
                    'metric': name,
                    'point': np.log10(comp.apparent_emergence_point)
                })

        if emergence_points:
            fig.add_trace(
                go.Scatter(
                    x=[e['point'] for e in emergence_points],
                    y=[e['metric'] for e in emergence_points],
                    mode='markers',
                    marker=dict(size=15, symbol='diamond'),
                    name='Emergence Points'
                ),
                row=2, col=1
            )

        # D. Artifact score by metric
        artifact_scores = {
            name: comp.smoothness_score
            for name, comp in comparisons.items()
        }
        fig.add_trace(
            go.Bar(
                x=list(artifact_scores.keys()),
                y=list(artifact_scores.values()),
                name='Smoothness Score'
            ),
            row=2, col=2
        )

        fig.update_layout(
            title="Emergence Mirage Analysis Summary",
            height=800,
            showlegend=True
        )

        return fig

    def save_matplotlib_figure(
        self,
        results: List[EmergenceResult],
        filename: str = "emergence_analysis.pdf"
    ):
        """
        Create publication-quality matplotlib figure.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        sorted_results = sorted(results, key=lambda r: r.parameter_count)
        log_scales = [np.log10(r.parameter_count) for r in sorted_results]

        # A. Binary vs Continuous metrics
        ax = axes[0, 0]
        ax.plot(log_scales, [r.binary_accuracy for r in sorted_results],
               'o-', color='blue', label='Binary Accuracy')
        ax.plot(log_scales, [r.partial_credit_score for r in sorted_results],
               's-', color='green', label='Partial Credit')
        ax.set_xlabel('Log₁₀(Parameters)')
        ax.set_ylabel('Score')
        ax.set_title('A. Binary vs. Continuous Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # B. Probability-based metrics
        ax = axes[0, 1]
        ax.plot(log_scales, [r.log_probability_score for r in sorted_results],
               'o-', color='red', label='Log Probability')
        ax.plot(log_scales, [r.rank_score for r in sorted_results],
               's-', color='purple', label='Rank Score')
        ax.set_xlabel('Log₁₀(Parameters)')
        ax.set_ylabel('Score')
        ax.set_title('B. Probability-Based Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # C. Entropy analysis
        ax = axes[1, 0]
        ax.plot(log_scales, [r.entropy_score for r in sorted_results],
               'o-', color='orange')
        ax.set_xlabel('Log₁₀(Parameters)')
        ax.set_ylabel('Entropy Score')
        ax.set_title('C. Model Confidence (Inverse Entropy)')
        ax.grid(True, alpha=0.3)

        # D. All metrics normalized
        ax = axes[1, 1]
        for metric, color in [
            ('binary_accuracy', 'blue'),
            ('partial_credit_score', 'green'),
            ('log_probability_score', 'red')
        ]:
            values = [getattr(r, metric) for r in sorted_results]
            # Normalize to 0-1
            min_v, max_v = min(values), max(values)
            if max_v > min_v:
                normalized = [(v - min_v) / (max_v - min_v) for v in values]
            else:
                normalized = values
            ax.plot(log_scales, normalized, 'o-', color=color,
                   label=metric.replace('_', ' '))

        ax.set_xlabel('Log₁₀(Parameters)')
        ax.set_ylabel('Normalized Score')
        ax.set_title('D. Normalized Metric Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{filename}", dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved figure to {self.output_dir}/{filename}")
