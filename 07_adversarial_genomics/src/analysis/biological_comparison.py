"""
Biological comparison analysis.

Compares model representations to biological visual systems.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import logging

logger = logging.getLogger(__name__)


@dataclass
class V1FeatureMatch:
    """Match between model feature and V1 property."""
    feature_index: int
    v1_property: str
    similarity: float
    orientation: Optional[float] = None
    spatial_frequency: Optional[float] = None


@dataclass
class BiologicalAlignment:
    """Alignment between model and biological visual system."""
    brain_region: str
    alignment_score: float
    n_matched_features: int
    top_matches: List[V1FeatureMatch]


class BiologicalComparator:
    """
    Compares model features to biological visual system properties.

    Tests whether adversarial training induces biologically-similar features.
    """

    # V1 properties to compare
    V1_PROPERTIES = [
        "edge_detection",
        "orientation_selectivity",
        "spatial_frequency",
        "phase_selectivity",
        "contrast_normalization"
    ]

    def __init__(self):
        self.gabor_bank = None

    def create_gabor_filter_bank(
        self,
        size: int = 32,
        n_orientations: int = 8,
        n_frequencies: int = 4
    ) -> np.ndarray:
        """Create a bank of Gabor filters mimicking V1 simple cells."""
        filters = []

        for theta in np.linspace(0, np.pi, n_orientations, endpoint=False):
            for freq in np.logspace(-1, 0.5, n_frequencies):
                for phase in [0, np.pi/2]:
                    gabor = self._create_gabor(size, theta, freq, phase)
                    filters.append({
                        'filter': gabor,
                        'orientation': theta,
                        'frequency': freq,
                        'phase': phase
                    })

        self.gabor_bank = filters
        return np.array([f['filter'] for f in filters])

    def _create_gabor(
        self,
        size: int,
        theta: float,
        frequency: float,
        phase: float,
        sigma: float = None
    ) -> np.ndarray:
        """Create a single Gabor filter."""
        if sigma is None:
            sigma = size / 6

        x = np.linspace(-size/2, size/2, size)
        y = np.linspace(-size/2, size/2, size)
        X, Y = np.meshgrid(x, y)

        # Rotation
        x_theta = X * np.cos(theta) + Y * np.sin(theta)
        y_theta = -X * np.sin(theta) + Y * np.cos(theta)

        # Gabor function
        gaussian = np.exp(-(x_theta**2 + y_theta**2) / (2 * sigma**2))
        sinusoid = np.cos(2 * np.pi * frequency * x_theta + phase)

        gabor = gaussian * sinusoid

        # Normalize
        gabor = gabor / (np.linalg.norm(gabor) + 1e-8)

        return gabor

    def extract_first_layer_features(
        self,
        model_weights: np.ndarray
    ) -> np.ndarray:
        """Extract and normalize first layer convolutional features."""
        # Assume weights are in shape (out_channels, in_channels, H, W)
        if model_weights.ndim == 4:
            # Average over input channels
            features = model_weights.mean(axis=1)
        else:
            features = model_weights

        # Normalize each filter
        normalized = []
        for f in features:
            norm = np.linalg.norm(f)
            if norm > 0:
                normalized.append(f / norm)
            else:
                normalized.append(f)

        return np.array(normalized)

    def compare_to_gabors(
        self,
        model_features: np.ndarray
    ) -> List[V1FeatureMatch]:
        """Compare model features to Gabor filter bank."""
        if self.gabor_bank is None:
            self.create_gabor_filter_bank(model_features.shape[-1])

        matches = []

        for i, feature in enumerate(model_features):
            best_match = None
            best_similarity = 0.0
            best_gabor_info = None

            for gabor_info in self.gabor_bank:
                gabor = gabor_info['filter']

                # Resize if needed
                if gabor.shape != feature.shape:
                    from scipy.ndimage import zoom
                    scale = feature.shape[0] / gabor.shape[0]
                    gabor = zoom(gabor, scale)

                # Compute similarity
                similarity = abs(np.dot(feature.flatten(), gabor.flatten()))

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_gabor_info = gabor_info

            if best_gabor_info:
                matches.append(V1FeatureMatch(
                    feature_index=i,
                    v1_property="gabor_like",
                    similarity=float(best_similarity),
                    orientation=float(best_gabor_info['orientation']),
                    spatial_frequency=float(best_gabor_info['frequency'])
                ))

        return matches

    def compute_orientation_selectivity(
        self,
        model_features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute orientation selectivity index for model features.

        Orientation selectivity is a key property of V1 neurons.
        """
        selectivity_indices = []

        for feature in model_features:
            # Compute response to different orientations
            responses = []
            for theta in np.linspace(0, np.pi, 8, endpoint=False):
                # Create oriented grating
                size = feature.shape[-1]
                x = np.linspace(-1, 1, size)
                X, Y = np.meshgrid(x, x)
                grating = np.sin(2 * np.pi * (X * np.cos(theta) + Y * np.sin(theta)) * 2)

                # Compute response
                response = abs(np.sum(feature * grating))
                responses.append(response)

            responses = np.array(responses)

            # Orientation selectivity index: (max - mean) / (max + mean)
            max_resp = np.max(responses)
            mean_resp = np.mean(responses)
            osi = (max_resp - mean_resp) / (max_resp + mean_resp + 1e-8)

            selectivity_indices.append(osi)

        return {
            "mean_osi": float(np.mean(selectivity_indices)),
            "std_osi": float(np.std(selectivity_indices)),
            "highly_selective_ratio": float(np.mean(np.array(selectivity_indices) > 0.5)),
            "orientation_distribution": np.histogram(selectivity_indices, bins=10)[0].tolist()
        }

    def compare_standard_vs_robust(
        self,
        standard_features: np.ndarray,
        robust_features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compare features between standard and adversarially robust models.

        Tests hypothesis that robust models develop more biologically-similar features.
        """
        # Gabor similarity
        standard_matches = self.compare_to_gabors(standard_features)
        robust_matches = self.compare_to_gabors(robust_features)

        standard_gabor_sim = np.mean([m.similarity for m in standard_matches])
        robust_gabor_sim = np.mean([m.similarity for m in robust_matches])

        # Orientation selectivity
        standard_osi = self.compute_orientation_selectivity(standard_features)
        robust_osi = self.compute_orientation_selectivity(robust_features)

        # Statistical comparison
        standard_sims = [m.similarity for m in standard_matches]
        robust_sims = [m.similarity for m in robust_matches]

        t_stat, p_value = stats.ttest_ind(robust_sims, standard_sims)

        return {
            "gabor_similarity": {
                "standard": float(standard_gabor_sim),
                "robust": float(robust_gabor_sim),
                "improvement": float(robust_gabor_sim - standard_gabor_sim),
                "t_statistic": float(t_stat),
                "p_value": float(p_value)
            },
            "orientation_selectivity": {
                "standard_mean_osi": standard_osi["mean_osi"],
                "robust_mean_osi": robust_osi["mean_osi"],
                "improvement": float(robust_osi["mean_osi"] - standard_osi["mean_osi"])
            },
            "conclusion": self._generate_conclusion(
                robust_gabor_sim - standard_gabor_sim,
                p_value
            )
        }

    def _generate_conclusion(
        self,
        improvement: float,
        p_value: float
    ) -> str:
        """Generate conclusion about biological alignment."""
        if improvement > 0.1 and p_value < 0.05:
            return "Strong support: Adversarial training significantly increases V1-like features"
        elif improvement > 0.05 and p_value < 0.1:
            return "Moderate support: Adversarial training shows trend toward V1-like features"
        elif improvement > 0:
            return "Weak support: Small improvement in V1-like features, not statistically significant"
        else:
            return "No support: Adversarial training does not increase V1-like features"

    def full_biological_analysis(
        self,
        standard_weights: np.ndarray,
        robust_weights: np.ndarray
    ) -> BiologicalAlignment:
        """Run complete biological comparison analysis."""
        # Extract features
        standard_features = self.extract_first_layer_features(standard_weights)
        robust_features = self.extract_first_layer_features(robust_weights)

        # Compare
        comparison = self.compare_standard_vs_robust(standard_features, robust_features)

        # Get top matches
        robust_matches = self.compare_to_gabors(robust_features)
        top_matches = sorted(robust_matches, key=lambda m: -m.similarity)[:10]

        alignment_score = comparison["gabor_similarity"]["robust"]

        return BiologicalAlignment(
            brain_region="V1",
            alignment_score=float(alignment_score),
            n_matched_features=sum(1 for m in robust_matches if m.similarity > 0.5),
            top_matches=top_matches
        )
