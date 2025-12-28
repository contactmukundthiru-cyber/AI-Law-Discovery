"""
Interactive plots for the dashboard using Plotly.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional
import json


class InteractivePlotter:
    """
    Interactive plotting for web dashboard using Plotly.
    """

    COLORS = px.colors.qualitative.Set2

    def __init__(self):
        self.default_layout = {
            "template": "plotly_white",
            "font": {"family": "Arial, sans-serif", "size": 12},
            "margin": {"l": 60, "r": 40, "t": 60, "b": 60},
        }

    def scaling_curve_interactive(
        self,
        x: List[float],
        y: List[float],
        model_names: List[str],
        confidence_lower: Optional[List[float]] = None,
        confidence_upper: Optional[List[float]] = None,
        title: str = "Scaling Curve",
    ) -> str:
        """
        Create interactive scaling curve.

        Returns:
            JSON string for Plotly
        """
        fig = go.Figure()

        # Sort data
        sorted_indices = np.argsort(x)
        x_sorted = [x[i] for i in sorted_indices]
        y_sorted = [y[i] for i in sorted_indices]
        names_sorted = [model_names[i] for i in sorted_indices]

        # Add confidence band if provided
        if confidence_lower and confidence_upper:
            ci_lower = [confidence_lower[i] for i in sorted_indices]
            ci_upper = [confidence_upper[i] for i in sorted_indices]

            fig.add_trace(go.Scatter(
                x=x_sorted + x_sorted[::-1],
                y=ci_upper + ci_lower[::-1],
                fill="toself",
                fillcolor="rgba(68, 68, 68, 0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% CI",
                showlegend=True,
            ))

        # Add main line and points
        fig.add_trace(go.Scatter(
            x=x_sorted,
            y=y_sorted,
            mode="lines+markers",
            name="Performance",
            line=dict(color=self.COLORS[0], width=2),
            marker=dict(size=10, color=self.COLORS[0]),
            text=names_sorted,
            hovertemplate="<b>%{text}</b><br>Scale: %{x:.2e}<br>Accuracy: %{y:.3f}<extra></extra>",
        ))

        fig.update_layout(
            **self.default_layout,
            title=title,
            xaxis=dict(title="Model Parameters", type="log"),
            yaxis=dict(title="Accuracy", range=[0, 1]),
            hovermode="closest",
        )

        return fig.to_json()

    def task_comparison_interactive(
        self,
        task_results: Dict[str, Dict[str, float]],
        title: str = "Task Comparison",
    ) -> str:
        """
        Create interactive task comparison chart.

        Returns:
            JSON string for Plotly
        """
        tasks = list(task_results.keys())
        models = sorted(set(
            model for results in task_results.values()
            for model in results.keys()
        ))

        fig = go.Figure()

        for i, model in enumerate(models):
            accuracies = [
                task_results[task].get(model, 0)
                for task in tasks
            ]
            fig.add_trace(go.Bar(
                name=model.split("/")[-1][:20],
                x=tasks,
                y=accuracies,
                marker_color=self.COLORS[i % len(self.COLORS)],
                hovertemplate="<b>%{x}</b><br>Model: " + model + "<br>Accuracy: %{y:.3f}<extra></extra>",
            ))

        fig.update_layout(
            **self.default_layout,
            title=title,
            barmode="group",
            xaxis=dict(title="Task", tickangle=45),
            yaxis=dict(title="Accuracy", range=[0, 1]),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        return fig.to_json()

    def pattern_distribution_interactive(
        self,
        pattern_counts: Dict[str, int],
        title: str = "Scaling Pattern Distribution",
    ) -> str:
        """
        Create interactive pie chart for pattern distribution.

        Returns:
            JSON string for Plotly
        """
        labels = [p.replace("_", " ").title() for p in pattern_counts.keys()]
        values = list(pattern_counts.values())

        # Color based on pattern type
        colors = []
        for p in pattern_counts.keys():
            if "inverse" in p.lower() or "negative" in p.lower():
                colors.append("#E74C3C")  # Red
            elif "positive" in p.lower():
                colors.append("#2ECC71")  # Green
            else:
                colors.append("#3498DB")  # Blue

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            hole=0.3,
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>",
        )])

        fig.update_layout(
            **self.default_layout,
            title=title,
            showlegend=True,
        )

        return fig.to_json()

    def progress_gauge(
        self,
        progress: float,
        title: str = "Experiment Progress",
    ) -> str:
        """
        Create progress gauge.

        Returns:
            JSON string for Plotly
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=progress * 100,
            number={"suffix": "%"},
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": self.COLORS[0]},
                "steps": [
                    {"range": [0, 50], "color": "lightgray"},
                    {"range": [50, 80], "color": "gray"},
                    {"range": [80, 100], "color": "darkgray"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 100,
                },
            },
        ))

        fig.update_layout(
            **self.default_layout,
            height=250,
        )

        return fig.to_json()

    def real_time_metrics(
        self,
        timestamps: List[str],
        accuracies: List[float],
        latencies: List[float],
        title: str = "Real-time Metrics",
    ) -> str:
        """
        Create real-time metrics dashboard.

        Returns:
            JSON string for Plotly
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Accuracy Over Time", "Latency Over Time"),
            vertical_spacing=0.15,
        )

        # Accuracy trace
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=accuracies,
                mode="lines+markers",
                name="Accuracy",
                line=dict(color=self.COLORS[0]),
            ),
            row=1, col=1,
        )

        # Latency trace
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=latencies,
                mode="lines+markers",
                name="Latency (ms)",
                line=dict(color=self.COLORS[1]),
            ),
            row=2, col=1,
        )

        fig.update_layout(
            **self.default_layout,
            title=title,
            height=500,
            showlegend=False,
        )

        fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text="Latency (ms)", row=2, col=1)

        return fig.to_json()

    def model_comparison_heatmap(
        self,
        data: Dict[str, Dict[str, float]],
        title: str = "Model × Task Performance",
    ) -> str:
        """
        Create heatmap of model vs task performance.

        Returns:
            JSON string for Plotly
        """
        tasks = list(data.keys())
        models = sorted(set(
            model for task_data in data.values()
            for model in task_data.keys()
        ))

        # Build matrix
        z = []
        for task in tasks:
            row = [data[task].get(model, 0) for model in models]
            z.append(row)

        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=[m.split("/")[-1][:15] for m in models],
            y=tasks,
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            hovertemplate="Task: %{y}<br>Model: %{x}<br>Accuracy: %{z:.3f}<extra></extra>",
        ))

        fig.update_layout(
            **self.default_layout,
            title=title,
            xaxis=dict(tickangle=45),
        )

        return fig.to_json()

    def inverse_scaling_summary(
        self,
        analyses: Dict[str, Any],
    ) -> str:
        """
        Create comprehensive inverse scaling summary dashboard.

        Returns:
            JSON string for Plotly
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Tasks by Pattern",
                "Accuracy Distribution",
                "Inverse Scaling Evidence",
                "Model Scale Impact",
            ),
            specs=[
                [{"type": "pie"}, {"type": "histogram"}],
                [{"type": "bar"}, {"type": "scatter"}],
            ],
        )

        overall = analyses.get("overall", {})
        tasks = analyses.get("tasks", {})

        # Pattern distribution pie
        pattern_dist = overall.get("pattern_distribution", {"unknown": 1})
        fig.add_trace(
            go.Pie(
                labels=list(pattern_dist.keys()),
                values=list(pattern_dist.values()),
                hole=0.4,
            ),
            row=1, col=1,
        )

        # Accuracy histogram
        all_accs = []
        for task_data in tasks.values():
            if hasattr(task_data, "accuracies"):
                all_accs.extend(task_data.accuracies)
            elif isinstance(task_data, dict) and "accuracies" in task_data:
                all_accs.extend(task_data["accuracies"])

        if all_accs:
            fig.add_trace(
                go.Histogram(x=all_accs, nbinsx=20, marker_color=self.COLORS[0]),
                row=1, col=2,
            )

        # Inverse scaling tasks bar
        inverse_tasks = analyses.get("inverse_scaling_tasks", [])
        task_names = list(tasks.keys())
        is_inverse = [1 if t in inverse_tasks else 0 for t in task_names]

        fig.add_trace(
            go.Bar(
                x=task_names,
                y=is_inverse,
                marker_color=[self.COLORS[1] if i else self.COLORS[0] for i in is_inverse],
            ),
            row=2, col=1,
        )

        # Scale vs accuracy scatter
        scales = []
        accs = []
        for task_data in tasks.values():
            if hasattr(task_data, "model_scales") and hasattr(task_data, "accuracies"):
                scales.extend(task_data.model_scales)
                accs.extend(task_data.accuracies)
            elif isinstance(task_data, dict):
                scales.extend(task_data.get("model_scales", []))
                accs.extend(task_data.get("accuracies", []))

        if scales and accs:
            fig.add_trace(
                go.Scatter(
                    x=scales,
                    y=accs,
                    mode="markers",
                    marker=dict(color=self.COLORS[0], size=8, opacity=0.6),
                ),
                row=2, col=2,
            )

        fig.update_layout(
            **self.default_layout,
            height=700,
            showlegend=False,
            title="Inverse Scaling Analysis Summary",
        )

        fig.update_xaxes(type="log", row=2, col=2)
        fig.update_yaxes(range=[0, 1], row=2, col=2)

        return fig.to_json()
