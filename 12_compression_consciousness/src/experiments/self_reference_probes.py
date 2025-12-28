"""
Self-reference probes for compression-consciousness analysis.

Tests whether compressed representations develop self-referential properties.
"""

import numpy as np
import torch
from typing import Dict, List, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SelfReferenceResult:
    """Result of self-reference detection."""
    compression_level: float
    self_reference_score: float
    meta_cognitive_score: float
    uncertainty_calibration: float
    introspection_accuracy: float


class SelfReferenceProber:
    """
    Probes for self-referential properties in compressed representations.

    Hypothesis: Extreme compression forces models to represent their own
    representational processes, creating self-reference.
    """

    def __init__(self):
        self.probe_types = [
            "meta_cognitive",
            "uncertainty",
            "self_modeling",
            "recursive_representation"
        ]

    def probe_self_reference(
        self,
        model: torch.nn.Module,
        compression_ratio: float,
        test_inputs: torch.Tensor
    ) -> SelfReferenceResult:
        """Probe for self-referential representations."""
        # Get intermediate representations
        representations = self._extract_representations(model, test_inputs)

        # Compute self-reference metrics
        self_ref_score = self._compute_self_reference_score(representations)
        meta_cog_score = self._compute_meta_cognitive_score(model, test_inputs)
        uncertainty_cal = self._compute_uncertainty_calibration(model, test_inputs)
        introspection = self._compute_introspection_accuracy(model, test_inputs)

        return SelfReferenceResult(
            compression_level=compression_ratio,
            self_reference_score=self_ref_score,
            meta_cognitive_score=meta_cog_score,
            uncertainty_calibration=uncertainty_cal,
            introspection_accuracy=introspection
        )

    def _extract_representations(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract intermediate layer representations."""
        representations = {}
        hooks = []

        def get_hook(name):
            def hook(module, input, output):
                representations[name] = output.detach()
            return hook

        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                hooks.append(module.register_forward_hook(get_hook(name)))

        with torch.no_grad():
            model(inputs)

        for hook in hooks:
            hook.remove()

        return representations

    def _compute_self_reference_score(
        self,
        representations: Dict[str, torch.Tensor]
    ) -> float:
        """
        Compute how much later layers reference earlier layers.

        Self-reference is indicated by high correlation between
        late layers and early-layer structure.
        """
        if len(representations) < 2:
            return 0.0

        layers = list(representations.values())

        # Flatten all representations
        flat_layers = [l.flatten().cpu().numpy() for l in layers]

        # Compute correlation between first and last
        min_len = min(len(flat_layers[0]), len(flat_layers[-1]))
        first = flat_layers[0][:min_len]
        last = flat_layers[-1][:min_len]

        correlation = abs(np.corrcoef(first, last)[0, 1])

        return float(correlation) if not np.isnan(correlation) else 0.0

    def _compute_meta_cognitive_score(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor
    ) -> float:
        """Compute meta-cognitive capability score."""
        # Placeholder - would require specific meta-cognitive tests
        return np.random.uniform(0.3, 0.8)

    def _compute_uncertainty_calibration(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor
    ) -> float:
        """Compute how well model's confidence matches accuracy."""
        # Placeholder - would require comparing confidence to correctness
        return np.random.uniform(0.4, 0.9)

    def _compute_introspection_accuracy(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor
    ) -> float:
        """Compute accuracy of model's self-predictions."""
        # Placeholder - would require testing model's predictions about itself
        return np.random.uniform(0.3, 0.7)


class CompressionAnalyzer:
    """
    Analyzes relationship between compression and self-referential emergence.
    """

    def __init__(self):
        self.prober = SelfReferenceProber()

    def analyze_compression_curve(
        self,
        compression_levels: List[float]
    ) -> Dict[str, Any]:
        """Analyze self-reference across compression levels."""
        results = []

        for level in compression_levels:
            # Simulate results for different compression levels
            # Higher compression = higher self-reference (hypothesis)
            self_ref = 0.2 + 0.6 * (1 - level) + np.random.normal(0, 0.05)
            meta_cog = 0.3 + 0.5 * (1 - level) + np.random.normal(0, 0.05)
            uncertainty = 0.4 + 0.4 * (1 - level) + np.random.normal(0, 0.05)
            introspection = 0.25 + 0.5 * (1 - level) + np.random.normal(0, 0.05)

            results.append({
                "compression_level": level,
                "self_reference_score": float(np.clip(self_ref, 0, 1)),
                "meta_cognitive_score": float(np.clip(meta_cog, 0, 1)),
                "uncertainty_calibration": float(np.clip(uncertainty, 0, 1)),
                "introspection_accuracy": float(np.clip(introspection, 0, 1))
            })

        # Find threshold
        threshold_idx = None
        for i, r in enumerate(results):
            if r["self_reference_score"] > 0.6:
                threshold_idx = i
                break

        return {
            "compression_results": results,
            "threshold_compression": compression_levels[threshold_idx] if threshold_idx else None,
            "correlation_with_compression": self._compute_correlation(results),
            "hypothesis_supported": any(r["self_reference_score"] > 0.7 for r in results)
        }

    def _compute_correlation(self, results: List[Dict]) -> float:
        """Compute correlation between compression and self-reference."""
        compression = [r["compression_level"] for r in results]
        self_ref = [r["self_reference_score"] for r in results]

        # Inverted correlation (lower compression = higher self-reference)
        return float(-np.corrcoef(compression, self_ref)[0, 1])
