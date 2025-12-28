"""
Curriculum experiments for detecting training scars.

Tests how different data orderings during training create permanent scars.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingCheckpoint:
    """Checkpoint during training."""
    epoch: int
    step: int
    metrics: Dict[str, float]
    model_state: Optional[Dict] = None


@dataclass
class ScarEvidence:
    """Evidence of a training scar."""
    scar_type: str
    severity: float  # 0-1 scale
    affected_capabilities: List[str]
    training_phase: str
    reversibility_score: float
    evidence_metrics: Dict[str, Any]


class CurriculumExperiment:
    """
    Experiment comparing different curriculum orderings.

    Tests whether early data exposure creates permanent effects.
    """

    def __init__(
        self,
        model_class: type,
        model_config: Dict[str, Any],
        device: str = "auto",
    ):
        self.model_class = model_class
        self.model_config = model_config

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def run_curriculum_comparison(
        self,
        train_data: List[Any],
        curricula: Dict[str, List[int]],
        eval_fn: callable,
        epochs: int = 10,
        seed: int = 42,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare different curriculum orderings.

        Args:
            train_data: Training dataset
            curricula: Dict mapping curriculum name to data ordering indices
            eval_fn: Evaluation function
            epochs: Training epochs
            seed: Random seed

        Returns:
            Results for each curriculum
        """
        results = {}

        for curriculum_name, ordering in curricula.items():
            logger.info(f"Running curriculum: {curriculum_name}")

            # Set seed for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Create fresh model
            model = self.model_class(**self.model_config).to(self.device)

            # Reorder data
            ordered_data = [train_data[i] for i in ordering]

            # Train
            checkpoints = self._train_model(model, ordered_data, epochs)

            # Evaluate
            final_metrics = eval_fn(model)

            results[curriculum_name] = {
                "checkpoints": checkpoints,
                "final_metrics": final_metrics,
                "ordering": ordering[:100],  # Store first 100 indices
            }

        return results

    def _train_model(
        self,
        model: nn.Module,
        data: List[Any],
        epochs: int,
    ) -> List[TrainingCheckpoint]:
        """Train model and collect checkpoints."""
        checkpoints = []
        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0

            for i, batch in enumerate(data):
                optimizer.zero_grad()
                # Simplified training step
                loss = self._compute_loss(model, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            checkpoints.append(TrainingCheckpoint(
                epoch=epoch,
                step=epoch * len(data),
                metrics={"loss": epoch_loss / len(data)},
            ))

        return checkpoints

    def _compute_loss(self, model: nn.Module, batch: Any) -> torch.Tensor:
        """Compute loss for a batch (to be overridden)."""
        # Placeholder - actual implementation depends on task
        return torch.tensor(0.0, requires_grad=True)

    def detect_scars(
        self,
        results: Dict[str, Dict[str, Any]],
        baseline_curriculum: str,
    ) -> List[ScarEvidence]:
        """
        Detect scars by comparing curriculum results.

        Args:
            results: Results from run_curriculum_comparison
            baseline_curriculum: Name of baseline curriculum

        Returns:
            List of detected scars
        """
        scars = []
        baseline = results.get(baseline_curriculum, {})
        baseline_metrics = baseline.get("final_metrics", {})

        for curriculum_name, result in results.items():
            if curriculum_name == baseline_curriculum:
                continue

            metrics = result.get("final_metrics", {})

            # Compare metrics to detect differences
            for metric_name, baseline_value in baseline_metrics.items():
                if metric_name not in metrics:
                    continue

                curr_value = metrics[metric_name]
                diff = abs(curr_value - baseline_value)

                # Threshold for detecting a scar
                if diff > 0.1:  # 10% difference
                    severity = min(1.0, diff / baseline_value) if baseline_value != 0 else diff

                    scars.append(ScarEvidence(
                        scar_type=f"{curriculum_name}_effect",
                        severity=severity,
                        affected_capabilities=[metric_name],
                        training_phase="early" if "early" in curriculum_name.lower() else "late",
                        reversibility_score=0.0,  # To be determined by reversibility tests
                        evidence_metrics={
                            "baseline": baseline_value,
                            "observed": curr_value,
                            "difference": diff,
                        },
                    ))

        return scars


class DataOrderingExperiment:
    """
    Experiment testing the effects of data ordering.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate_orderings(
        self,
        data_size: int,
        num_orderings: int = 5,
    ) -> Dict[str, List[int]]:
        """
        Generate different data orderings for comparison.

        Returns:
            Dictionary mapping ordering name to index list
        """
        indices = list(range(data_size))

        orderings = {
            "sequential": indices.copy(),
            "reversed": indices[::-1],
            "random": self.rng.permutation(indices).tolist(),
        }

        # Easy-first ordering (assuming first half is "easier")
        half = data_size // 2
        orderings["easy_first"] = indices[:half] + indices[half:]
        orderings["hard_first"] = indices[half:] + indices[:half]

        # Interleaved ordering
        orderings["interleaved"] = []
        for i in range(half):
            orderings["interleaved"].append(indices[i])
            if i + half < data_size:
                orderings["interleaved"].append(indices[i + half])

        return orderings

    def analyze_ordering_effects(
        self,
        results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyze the effects of different orderings.

        Returns:
            Analysis summary
        """
        analysis = {
            "orderings_tested": list(results.keys()),
            "metric_comparisons": {},
            "ordering_rankings": {},
        }

        # Collect all metrics
        all_metrics = set()
        for result in results.values():
            all_metrics.update(result.get("final_metrics", {}).keys())

        # Compare orderings for each metric
        for metric in all_metrics:
            values = {}
            for ordering, result in results.items():
                if metric in result.get("final_metrics", {}):
                    values[ordering] = result["final_metrics"][metric]

            if values:
                sorted_orderings = sorted(values.items(), key=lambda x: x[1], reverse=True)
                analysis["metric_comparisons"][metric] = values
                analysis["ordering_rankings"][metric] = [o for o, v in sorted_orderings]

        return analysis
