"""
Evolutionary pressure mapping.

Maps adversarial perturbations to evolutionary visual challenges.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import logging

from ..data.environmental_corruptions import CorruptionType, EnvironmentalCorruptions
from .adversarial_analysis import AdversarialStructure, CorruptionCorrelation

logger = logging.getLogger(__name__)


@dataclass
class EvolutionaryPressure:
    """Identified evolutionary pressure in adversarial subspace."""
    pressure_name: str
    strength: float  # How much of adversarial variance this explains
    related_corruptions: List[CorruptionType]
    biological_relevance: str
    confidence: float


@dataclass
class PressureMapping:
    """Complete mapping from adversarial space to evolutionary pressures."""
    pressures: List[EvolutionaryPressure]
    total_explained_variance: float
    unmapped_variance: float
    mapping_quality: float


class EvolutionaryPressureMapper:
    """
    Maps adversarial perturbations to evolutionary visual pressures.

    Hypothesis: Adversarial examples exploit the same vulnerabilities
    that biological evolution addressed over millions of years.
    """

    # Evolutionary pressure categories
    PRESSURE_CATEGORIES = {
        "predator_detection": {
            "corruptions": [CorruptionType.CAMOUFLAGE, CorruptionType.CLUTTER,
                          CorruptionType.PARTIAL],
            "biological_relevance": "Detection of camouflaged predators/prey"
        },
        "weather_adaptation": {
            "corruptions": [CorruptionType.FOG, CorruptionType.RAIN,
                          CorruptionType.SNOW],
            "biological_relevance": "Visual function in adverse weather"
        },
        "motion_tracking": {
            "corruptions": [CorruptionType.BLUR, CorruptionType.ZOOM,
                          CorruptionType.DEFOCUS],
            "biological_relevance": "Tracking moving objects/self-motion"
        },
        "lighting_adaptation": {
            "corruptions": [CorruptionType.BRIGHTNESS, CorruptionType.CONTRAST,
                          CorruptionType.SHADOWS],
            "biological_relevance": "Vision across lighting conditions"
        },
        "depth_perception": {
            "corruptions": [CorruptionType.PERSPECTIVE, CorruptionType.SCALE,
                          CorruptionType.DISTANCE],
            "biological_relevance": "3D spatial understanding"
        }
    }

    def __init__(self, corruption_generator: EnvironmentalCorruptions):
        self.corruptions = corruption_generator

    def map_adversarial_to_pressures(
        self,
        adversarial_structure: AdversarialStructure,
        corruption_correlations: Dict[CorruptionType, CorruptionCorrelation]
    ) -> PressureMapping:
        """
        Map adversarial perturbation structure to evolutionary pressures.
        """
        pressures = []
        total_explained = 0.0

        for pressure_name, pressure_info in self.PRESSURE_CATEGORIES.items():
            # Get correlations for this pressure's corruptions
            relevant_correlations = [
                corruption_correlations.get(ct)
                for ct in pressure_info["corruptions"]
                if ct in corruption_correlations
            ]

            if not relevant_correlations:
                continue

            # Compute aggregate strength
            similarities = [c.cosine_similarity for c in relevant_correlations if c]
            structural_sims = [c.structural_similarity for c in relevant_correlations if c]
            freq_overlaps = [c.frequency_overlap for c in relevant_correlations if c]

            if not similarities:
                continue

            strength = np.mean(similarities)
            structural_strength = np.mean(structural_sims)
            freq_strength = np.mean(freq_overlaps)

            # Combined confidence
            confidence = (strength + structural_strength + freq_strength) / 3

            pressures.append(EvolutionaryPressure(
                pressure_name=pressure_name,
                strength=float(strength),
                related_corruptions=pressure_info["corruptions"],
                biological_relevance=pressure_info["biological_relevance"],
                confidence=float(confidence)
            ))

            total_explained += confidence

        # Normalize and compute quality metrics
        if pressures:
            max_possible = len(self.PRESSURE_CATEGORIES)
            mapping_quality = min(total_explained / max_possible, 1.0)
        else:
            mapping_quality = 0.0

        return PressureMapping(
            pressures=sorted(pressures, key=lambda p: -p.strength),
            total_explained_variance=float(total_explained),
            unmapped_variance=float(1 - mapping_quality),
            mapping_quality=float(mapping_quality)
        )

    def compute_pressure_basis(
        self,
        images: np.ndarray,
        severity: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """
        Compute basis vectors for each evolutionary pressure.

        Returns directions in image space that correspond to each pressure.
        """
        pressure_bases = {}

        for pressure_name, pressure_info in self.PRESSURE_CATEGORIES.items():
            directions = []

            for corruption_type in pressure_info["corruptions"]:
                try:
                    for img in images:
                        result = self.corruptions.apply(img, corruption_type, severity)
                        direction = self.corruptions.get_corruption_vector(
                            img, result.corrupted_image
                        )
                        directions.append(direction.flatten())
                except Exception as e:
                    logger.debug(f"Could not compute {corruption_type}: {e}")

            if directions:
                # Average direction for this pressure
                directions = np.array(directions)
                mean_direction = directions.mean(axis=0)

                # Normalize
                norm = np.linalg.norm(mean_direction)
                if norm > 0:
                    mean_direction = mean_direction / norm

                pressure_bases[pressure_name] = mean_direction

        return pressure_bases

    def decompose_adversarial_perturbation(
        self,
        perturbation: np.ndarray,
        pressure_bases: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Decompose adversarial perturbation into evolutionary pressure components.

        This reveals which evolutionary challenges are "compressed" in the
        adversarial perturbation.
        """
        flat_pert = perturbation.flatten()
        pert_norm = np.linalg.norm(flat_pert)

        if pert_norm == 0:
            return {name: 0.0 for name in pressure_bases}

        normalized_pert = flat_pert / pert_norm

        components = {}
        for pressure_name, basis in pressure_bases.items():
            # Ensure same size
            min_len = min(len(normalized_pert), len(basis))
            projection = np.dot(normalized_pert[:min_len], basis[:min_len])
            components[pressure_name] = float(abs(projection))

        return components

    def analyze_robustness_correlation(
        self,
        standard_accuracy: Dict[CorruptionType, float],
        robust_accuracy: Dict[CorruptionType, float],
        adversarial_correlations: Dict[CorruptionType, CorruptionCorrelation]
    ) -> Dict[str, Any]:
        """
        Analyze whether adversarial training improves natural corruption robustness.

        Tests hypothesis that adversarial robustness = compressed evolutionary robustness.
        """
        results = {
            "per_corruption": {},
            "overall": {}
        }

        # Per-corruption analysis
        improvements = []
        correlations = []

        for ctype in standard_accuracy:
            if ctype in robust_accuracy and ctype in adversarial_correlations:
                improvement = robust_accuracy[ctype] - standard_accuracy[ctype]
                adv_correlation = adversarial_correlations[ctype].cosine_similarity

                improvements.append(improvement)
                correlations.append(adv_correlation)

                results["per_corruption"][ctype.value] = {
                    "standard_accuracy": standard_accuracy[ctype],
                    "robust_accuracy": robust_accuracy[ctype],
                    "improvement": improvement,
                    "adversarial_correlation": adv_correlation
                }

        # Overall correlation
        if improvements and correlations:
            correlation, p_value = stats.pearsonr(improvements, correlations)
            results["overall"] = {
                "improvement_correlation_r": float(correlation),
                "p_value": float(p_value),
                "mean_improvement": float(np.mean(improvements)),
                "supports_hypothesis": correlation > 0.3 and p_value < 0.05
            }

        return results

    def predict_corruption_robustness(
        self,
        adversarial_perturbations: np.ndarray,
        pressure_bases: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Predict which natural corruptions a model will be robust to
        based on adversarial perturbation structure.

        If adversarial perturbations encode evolutionary pressures,
        then perturbation structure should predict corruption robustness.
        """
        # Decompose each perturbation
        pressure_loadings = {name: [] for name in pressure_bases}

        for pert in adversarial_perturbations:
            components = self.decompose_adversarial_perturbation(pert, pressure_bases)
            for name, loading in components.items():
                pressure_loadings[name].append(loading)

        # Predict robustness based on loading magnitudes
        predictions = {}
        for pressure_name, loadings in pressure_loadings.items():
            mean_loading = np.mean(loadings)
            # Higher loading = more adversarial focus on this pressure
            # = potentially better robustness after adversarial training
            predicted_robustness = min(mean_loading * 2, 1.0)  # Scale to 0-1
            predictions[pressure_name] = float(predicted_robustness)

        return predictions
