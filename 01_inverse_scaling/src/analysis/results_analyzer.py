"""
Results analyzer for inverse scaling experiments.

Aggregates and analyzes experimental results to identify inverse scaling patterns.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

from ..experiments.trial import Trial, TrialResult
from .statistics import StatisticalAnalyzer, ConfidenceInterval
from .scaling_curves import ScalingCurveAnalyzer, CurveType


@dataclass
class ScalingAnalysis:
    """Analysis of scaling behavior for a task."""
    task_name: str
    model_scales: List[float]
    model_names: List[str]
    accuracies: List[float]
    confidence_intervals: List[ConfidenceInterval]
    scaling_pattern: str
    inverse_scaling_detected: bool
    optimal_scale: Optional[float]
    curve_analysis: Dict[str, Any]
    statistical_tests: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "model_scales": self.model_scales,
            "model_names": self.model_names,
            "accuracies": self.accuracies,
            "confidence_intervals": [ci.to_dict() for ci in self.confidence_intervals],
            "scaling_pattern": self.scaling_pattern,
            "inverse_scaling_detected": self.inverse_scaling_detected,
            "optimal_scale": self.optimal_scale,
            "curve_analysis": self.curve_analysis,
            "statistical_tests": self.statistical_tests,
        }


class ResultsAnalyzer:
    """
    Comprehensive analyzer for inverse scaling experiment results.
    """

    def __init__(self):
        self.stats = StatisticalAnalyzer()
        self.curves = ScalingCurveAnalyzer()

    def analyze_trial(self, trial: Trial) -> Dict[str, Any]:
        """
        Analyze a complete trial.

        Args:
            trial: Trial object with results

        Returns:
            Comprehensive analysis results
        """
        if not trial.results:
            return {"error": "No results to analyze"}

        # Convert to DataFrame for easier analysis
        df = self._results_to_dataframe(trial.results)

        # Group by task
        task_analyses = {}
        for task_name in df["task_type"].unique():
            task_df = df[df["task_type"] == task_name]
            task_analyses[task_name] = self._analyze_task(task_df)

        # Overall summary
        overall = self._compute_overall_summary(df, task_analyses)

        return {
            "trial_id": trial.trial_id,
            "name": trial.name,
            "tasks": task_analyses,
            "overall": overall,
            "inverse_scaling_tasks": [
                name for name, analysis in task_analyses.items()
                if analysis.inverse_scaling_detected
            ],
        }

    def _results_to_dataframe(self, results: List[TrialResult]) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        rows = []
        for result in results:
            rows.append({
                "model_name": result.model_name,
                "provider": result.model_provider,
                "estimated_params": result.estimated_params,
                "params_numeric": self._parse_params(result.estimated_params),
                "task_type": result.task_type,
                "subtask": result.subtask,
                "accuracy": result.accuracy,
                "mean_score": result.mean_score,
                "num_samples": result.num_samples,
                "correct": result.correct,
            })
        return pd.DataFrame(rows)

    def _parse_params(self, params_str: str) -> float:
        """Parse parameter string to numeric value."""
        if not params_str or params_str.lower() in ["unknown", "n/a"]:
            return np.nan

        params_str = params_str.upper().strip()
        multipliers = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}

        for suffix, mult in multipliers.items():
            if params_str.endswith(suffix):
                try:
                    return float(params_str[:-1]) * mult
                except ValueError:
                    return np.nan

        # Handle ranges like "200B+"
        params_str = params_str.rstrip("+")
        for suffix, mult in multipliers.items():
            if params_str.endswith(suffix):
                try:
                    return float(params_str[:-1]) * mult
                except ValueError:
                    return np.nan

        try:
            return float(params_str)
        except ValueError:
            return np.nan

    def _analyze_task(self, task_df: pd.DataFrame) -> ScalingAnalysis:
        """Analyze scaling for a single task."""
        # Sort by model scale
        task_df = task_df.sort_values("params_numeric")

        # Extract data
        scales = task_df["params_numeric"].values
        accuracies = task_df["accuracy"].values
        model_names = task_df["model_name"].tolist()

        # Filter out NaN scales
        valid_mask = ~np.isnan(scales)
        if not np.any(valid_mask):
            return ScalingAnalysis(
                task_name=task_df["task_type"].iloc[0],
                model_scales=[],
                model_names=model_names,
                accuracies=accuracies.tolist(),
                confidence_intervals=[],
                scaling_pattern="insufficient_data",
                inverse_scaling_detected=False,
                optimal_scale=None,
                curve_analysis={},
                statistical_tests={},
            )

        scales_valid = scales[valid_mask]
        accuracies_valid = accuracies[valid_mask]

        # Compute confidence intervals (using binomial for accuracy)
        confidence_intervals = []
        for acc, n in zip(task_df["accuracy"], task_df["num_samples"]):
            ci = self._binomial_ci(acc, n)
            confidence_intervals.append(ci)

        # Curve analysis
        curve_analysis = {}
        if len(scales_valid) >= 3:
            curve_analysis = self.curves.detect_inverse_scaling(
                scales_valid, accuracies_valid
            )

        # Statistical tests
        statistical_tests = {}
        if len(scales_valid) >= 3:
            statistical_tests = self.stats.test_monotonicity(
                scales_valid, accuracies_valid
            )

        # Detect optimal scale
        optimal_scale = None
        if curve_analysis.get("quadratic_fit", {}).get("peak_x"):
            optimal_scale = curve_analysis["quadratic_fit"]["peak_x"]

        # Determine scaling pattern
        scaling_pattern = self._determine_pattern(curve_analysis, statistical_tests)
        inverse_detected = curve_analysis.get("inverse_scaling_detected", False)

        return ScalingAnalysis(
            task_name=task_df["task_type"].iloc[0],
            model_scales=scales_valid.tolist(),
            model_names=model_names,
            accuracies=accuracies_valid.tolist(),
            confidence_intervals=confidence_intervals,
            scaling_pattern=scaling_pattern,
            inverse_scaling_detected=inverse_detected,
            optimal_scale=optimal_scale,
            curve_analysis=curve_analysis,
            statistical_tests=statistical_tests,
        )

    def _binomial_ci(
        self,
        proportion: float,
        n: int,
        confidence: float = 0.95,
    ) -> ConfidenceInterval:
        """Compute binomial confidence interval using Wilson score."""
        from scipy import stats

        z = stats.norm.ppf((1 + confidence) / 2)
        denominator = 1 + z**2 / n

        center = (proportion + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt(
            (proportion * (1 - proportion) + z**2 / (4 * n)) / n
        ) / denominator

        return ConfidenceInterval(
            lower=max(0, center - margin),
            upper=min(1, center + margin),
            point_estimate=proportion,
            confidence_level=confidence,
            method="wilson_score",
        )

    def _determine_pattern(
        self,
        curve_analysis: Dict[str, Any],
        statistical_tests: Dict[str, Any],
    ) -> str:
        """Determine the scaling pattern type."""
        if not curve_analysis:
            return "unknown"

        inverse_detected = curve_analysis.get("inverse_scaling_detected", False)
        pattern_type = curve_analysis.get("pattern_type", "none")
        best_fit = curve_analysis.get("best_fit", "linear")

        if inverse_detected:
            if pattern_type == "inverse_u":
                return "inverse_u_scaling"
            else:
                return "negative_scaling"
        elif statistical_tests.get("spearman", {}).get("correlation", 0) > 0.5:
            return "positive_scaling"
        elif abs(statistical_tests.get("spearman", {}).get("correlation", 0)) < 0.2:
            return "flat_scaling"
        else:
            return "weak_relationship"

    def _compute_overall_summary(
        self,
        df: pd.DataFrame,
        task_analyses: Dict[str, ScalingAnalysis],
    ) -> Dict[str, Any]:
        """Compute overall experiment summary."""
        total_tasks = len(task_analyses)
        inverse_scaling_count = sum(
            1 for a in task_analyses.values() if a.inverse_scaling_detected
        )

        # Pattern distribution
        patterns = [a.scaling_pattern for a in task_analyses.values()]
        pattern_counts = {}
        for p in patterns:
            pattern_counts[p] = pattern_counts.get(p, 0) + 1

        # Model performance summary
        model_summary = df.groupby("model_name").agg({
            "accuracy": ["mean", "std"],
            "params_numeric": "first",
        }).round(4)

        return {
            "total_tasks": total_tasks,
            "inverse_scaling_detected": inverse_scaling_count,
            "inverse_scaling_proportion": inverse_scaling_count / total_tasks if total_tasks > 0 else 0,
            "pattern_distribution": pattern_counts,
            "models_evaluated": df["model_name"].nunique(),
            "total_evaluations": len(df),
            "mean_accuracy_by_model": model_summary.to_dict(),
        }

    def compare_trials(
        self,
        trials: List[Trial],
    ) -> Dict[str, Any]:
        """
        Compare results across multiple trials.

        Args:
            trials: List of trials to compare

        Returns:
            Comparison analysis
        """
        all_analyses = []
        for trial in trials:
            analysis = self.analyze_trial(trial)
            analysis["trial_id"] = trial.trial_id
            all_analyses.append(analysis)

        # Aggregate inverse scaling findings
        inverse_tasks_all = set()
        for analysis in all_analyses:
            inverse_tasks_all.update(analysis.get("inverse_scaling_tasks", []))

        # Count consistent findings
        task_consistency = {}
        for task in inverse_tasks_all:
            count = sum(
                1 for a in all_analyses
                if task in a.get("inverse_scaling_tasks", [])
            )
            task_consistency[task] = count / len(trials)

        return {
            "num_trials": len(trials),
            "trial_ids": [t.trial_id for t in trials],
            "inverse_scaling_tasks_any": list(inverse_tasks_all),
            "task_consistency": task_consistency,
            "robust_inverse_scaling": [
                task for task, consistency in task_consistency.items()
                if consistency >= 0.5
            ],
            "individual_analyses": all_analyses,
        }

    def export_results(
        self,
        analysis: Dict[str, Any],
        output_path: Path,
        format: str = "json",
    ) -> None:
        """
        Export analysis results.

        Args:
            analysis: Analysis results dictionary
            output_path: Output file path
            format: 'json' or 'csv'
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            # Convert any non-serializable objects
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, ScalingAnalysis):
                    return obj.to_dict()
                elif hasattr(obj, "to_dict"):
                    return obj.to_dict()
                return obj

            def recursive_convert(d):
                if isinstance(d, dict):
                    return {k: recursive_convert(v) for k, v in d.items()}
                elif isinstance(d, list):
                    return [recursive_convert(v) for v in d]
                else:
                    return convert(d)

            with open(output_path, "w") as f:
                json.dump(recursive_convert(analysis), f, indent=2)

        elif format == "csv":
            # Flatten for CSV export
            rows = []
            for task_name, task_analysis in analysis.get("tasks", {}).items():
                if isinstance(task_analysis, ScalingAnalysis):
                    task_dict = task_analysis.to_dict()
                else:
                    task_dict = task_analysis

                for i, (scale, acc) in enumerate(zip(
                    task_dict.get("model_scales", []),
                    task_dict.get("accuracies", [])
                )):
                    rows.append({
                        "task": task_name,
                        "model_scale": scale,
                        "accuracy": acc,
                        "inverse_scaling": task_dict.get("inverse_scaling_detected", False),
                        "pattern": task_dict.get("scaling_pattern", "unknown"),
                    })

            pd.DataFrame(rows).to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")
