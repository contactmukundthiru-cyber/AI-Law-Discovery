"""
Forgetting tracker for analyzing systematic forgetting during training.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ForgettingMetrics:
    """Metrics for tracking forgetting."""
    step: int
    forgotten_examples: List[int]
    retained_examples: List[int]
    forgetting_rate: float
    information_content: float
    compression_ratio: float


@dataclass
class ForgettingPattern:
    """Pattern of forgetting behavior."""
    pattern_type: str  # "random", "systematic", "structured"
    affected_features: List[str]
    forgetting_curve: List[float]
    recovery_potential: float


class ForgettingTracker:
    """
    Tracks what models forget during training.

    Monitors:
    - Per-example memorization and forgetting
    - Information content changes
    - Systematic forgetting patterns
    """

    def __init__(
        self,
        track_examples: bool = True,
        track_weights: bool = True,
        forgetting_threshold: float = 0.5,
    ):
        self.track_examples = track_examples
        self.track_weights = track_weights
        self.forgetting_threshold = forgetting_threshold

        self.example_history: Dict[int, List[bool]] = defaultdict(list)
        self.weight_history: List[Dict[str, np.ndarray]] = []
        self.metrics_history: List[ForgettingMetrics] = []

    def record_example_predictions(
        self,
        step: int,
        example_ids: List[int],
        correct: List[bool],
    ):
        """
        Record prediction correctness for examples.

        Tracks which examples were correctly predicted vs. forgotten.
        """
        for ex_id, is_correct in zip(example_ids, correct):
            self.example_history[ex_id].append(is_correct)

    def record_weights(
        self,
        step: int,
        model: torch.nn.Module,
        layer_names: Optional[List[str]] = None,
    ):
        """Record weight states for analysis."""
        weights = {}
        for name, param in model.named_parameters():
            if layer_names is None or any(ln in name for ln in layer_names):
                weights[name] = param.detach().cpu().numpy().copy()
        self.weight_history.append(weights)

    def compute_forgetting_events(
        self,
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        Compute forgetting events for each example.

        A forgetting event is when an example goes from correct to incorrect.

        Returns:
            Dict mapping example_id to list of (start_step, end_step) events
        """
        events = {}

        for ex_id, history in self.example_history.items():
            example_events = []
            in_forgetting = False
            start_step = 0

            for step, correct in enumerate(history):
                if not correct and not in_forgetting:
                    # Start of forgetting
                    in_forgetting = True
                    start_step = step
                elif correct and in_forgetting:
                    # End of forgetting (recovered)
                    example_events.append((start_step, step))
                    in_forgetting = False

            if in_forgetting:
                # Still forgotten at end
                example_events.append((start_step, len(history)))

            events[ex_id] = example_events

        return events

    def compute_forgetting_curve(self) -> np.ndarray:
        """
        Compute aggregate forgetting curve.

        Returns:
            Array of forgetting rates at each step
        """
        if not self.example_history:
            return np.array([])

        max_len = max(len(h) for h in self.example_history.values())
        forgetting_rates = []

        for step in range(max_len):
            total = 0
            forgotten = 0

            for history in self.example_history.values():
                if step < len(history):
                    total += 1
                    if not history[step]:
                        forgotten += 1

            rate = forgotten / total if total > 0 else 0
            forgetting_rates.append(rate)

        return np.array(forgetting_rates)

    def identify_forgetting_patterns(self) -> List[ForgettingPattern]:
        """
        Identify patterns in what gets forgotten.
        """
        patterns = []
        events = self.compute_forgetting_events()
        curve = self.compute_forgetting_curve()

        # Classify examples by forgetting behavior
        never_forgotten = []
        always_forgotten = []
        intermittent = []

        for ex_id, ex_events in events.items():
            history = self.example_history[ex_id]
            if not history:
                continue

            if len(ex_events) == 0:
                never_forgotten.append(ex_id)
            elif len(ex_events) == 1 and ex_events[0][1] == len(history):
                always_forgotten.append(ex_id)
            else:
                intermittent.append(ex_id)

        # Create patterns
        if len(always_forgotten) > len(self.example_history) * 0.1:
            patterns.append(ForgettingPattern(
                pattern_type="systematic",
                affected_features=["consistently_hard_examples"],
                forgetting_curve=curve.tolist(),
                recovery_potential=0.0,
            ))

        if intermittent:
            avg_events = np.mean([len(events[e]) for e in intermittent])
            patterns.append(ForgettingPattern(
                pattern_type="oscillating",
                affected_features=["unstable_examples"],
                forgetting_curve=curve.tolist(),
                recovery_potential=0.5,
            ))

        return patterns

    def compute_information_dynamics(self) -> Dict[str, Any]:
        """
        Compute information-theoretic measures of forgetting.
        """
        if not self.weight_history:
            return {}

        # Compute weight changes between steps
        weight_changes = []
        for i in range(1, len(self.weight_history)):
            prev = self.weight_history[i - 1]
            curr = self.weight_history[i]

            total_change = 0
            total_params = 0
            for name in prev:
                if name in curr:
                    change = np.abs(curr[name] - prev[name]).sum()
                    total_change += change
                    total_params += prev[name].size

            if total_params > 0:
                weight_changes.append(total_change / total_params)

        # Compute compression (weight magnitude decrease)
        initial_magnitude = sum(
            np.abs(w).sum() for w in self.weight_history[0].values()
        )
        final_magnitude = sum(
            np.abs(w).sum() for w in self.weight_history[-1].values()
        )

        compression = 1 - (final_magnitude / initial_magnitude) if initial_magnitude > 0 else 0

        return {
            "weight_change_trajectory": weight_changes,
            "total_compression": compression,
            "forgetting_curve": self.compute_forgetting_curve().tolist(),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of forgetting analysis."""
        events = self.compute_forgetting_events()
        curve = self.compute_forgetting_curve()
        patterns = self.identify_forgetting_patterns()

        total_examples = len(self.example_history)
        examples_with_forgetting = sum(1 for e in events.values() if e)

        return {
            "total_examples_tracked": total_examples,
            "examples_with_forgetting": examples_with_forgetting,
            "forgetting_proportion": examples_with_forgetting / total_examples if total_examples > 0 else 0,
            "mean_forgetting_rate": float(curve.mean()) if len(curve) > 0 else 0,
            "max_forgetting_rate": float(curve.max()) if len(curve) > 0 else 0,
            "patterns_detected": len(patterns),
            "pattern_types": [p.pattern_type for p in patterns],
        }
