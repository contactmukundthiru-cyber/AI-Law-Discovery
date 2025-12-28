"""
Adversarial structure analysis.

Analyzes the structure of adversarial perturbations to identify
patterns related to evolutionary visual challenges.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import logging

from ..models.robust_models import AdversarialAttacker, AdversarialExample
from ..data.environmental_corruptions import EnvironmentalCorruptions, CorruptionType

logger = logging.getLogger(__name__)


@dataclass
class AdversarialStructure:
    """Structure of adversarial perturbations."""
    principal_components: np.ndarray
    explained_variance: np.ndarray
    frequency_spectrum: np.ndarray
    spatial_pattern: str
    dominant_frequencies: List[float]


@dataclass
class CorruptionCorrelation:
    """Correlation between adversarial and environmental corruption."""
    corruption_type: CorruptionType
    cosine_similarity: float
    structural_similarity: float
    frequency_overlap: float


class AdversarialStructureAnalyzer:
    """
    Analyzes the structure of adversarial perturbations.

    Investigates whether adversarial perturbations share structure
    with natural environmental challenges.
    """

    def __init__(
        self,
        attacker: AdversarialAttacker,
        corruption_generator: EnvironmentalCorruptions
    ):
        self.attacker = attacker
        self.corruptions = corruption_generator

    def analyze_perturbation_structure(
        self,
        perturbation: torch.Tensor
    ) -> AdversarialStructure:
        """Analyze the structure of an adversarial perturbation."""
        pert_np = perturbation.cpu().numpy()

        # Flatten for PCA
        if pert_np.ndim > 2:
            flat = pert_np.reshape(pert_np.shape[0], -1) if pert_np.ndim == 4 else pert_np.flatten()
        else:
            flat = pert_np.flatten()

        # PCA analysis
        if flat.ndim == 1:
            flat = flat.reshape(1, -1)

        pca = PCA(n_components=min(10, flat.shape[1]))
        pca.fit(flat)

        # Frequency analysis
        spectrum = self._compute_frequency_spectrum(pert_np)
        dominant_freqs = self._find_dominant_frequencies(spectrum)

        # Classify spatial pattern
        pattern = self._classify_spatial_pattern(pert_np)

        return AdversarialStructure(
            principal_components=pca.components_,
            explained_variance=pca.explained_variance_ratio_,
            frequency_spectrum=spectrum,
            spatial_pattern=pattern,
            dominant_frequencies=dominant_freqs
        )

    def _compute_frequency_spectrum(self, perturbation: np.ndarray) -> np.ndarray:
        """Compute 2D FFT spectrum of perturbation."""
        if perturbation.ndim == 4:  # Batch
            perturbation = perturbation[0]
        if perturbation.ndim == 3:  # CHW -> HW (average channels)
            perturbation = perturbation.mean(axis=0)

        fft = np.fft.fft2(perturbation)
        spectrum = np.abs(np.fft.fftshift(fft))
        return spectrum

    def _find_dominant_frequencies(
        self,
        spectrum: np.ndarray,
        top_k: int = 5
    ) -> List[float]:
        """Find dominant frequencies in spectrum."""
        h, w = spectrum.shape
        center_h, center_w = h // 2, w // 2

        # Create frequency grid
        freq_h = np.fft.fftfreq(h)
        freq_w = np.fft.fftfreq(w)
        freq_magnitude = np.sqrt(freq_h[:, None]**2 + freq_w[None, :]**2)

        # Find peaks
        flat_spectrum = spectrum.flatten()
        flat_freq = np.fft.fftshift(freq_magnitude).flatten()

        top_indices = np.argsort(flat_spectrum)[-top_k:]
        dominant = [flat_freq[i] for i in top_indices]

        return sorted(dominant)

    def _classify_spatial_pattern(self, perturbation: np.ndarray) -> str:
        """Classify the spatial pattern of perturbation."""
        if perturbation.ndim == 4:
            perturbation = perturbation[0]
        if perturbation.ndim == 3:
            perturbation = perturbation.mean(axis=0)

        # Check for structured patterns
        h_gradient = np.abs(np.diff(perturbation, axis=1)).mean()
        v_gradient = np.abs(np.diff(perturbation, axis=0)).mean()
        total_var = np.var(perturbation)

        # Classify based on gradient patterns
        gradient_ratio = max(h_gradient, v_gradient) / (min(h_gradient, v_gradient) + 1e-8)

        if gradient_ratio > 3:
            return "directional"
        elif total_var < 0.01:
            return "uniform"
        elif self._check_periodicity(perturbation):
            return "periodic"
        else:
            return "random"

    def _check_periodicity(self, perturbation: np.ndarray) -> bool:
        """Check if perturbation has periodic structure."""
        spectrum = np.abs(np.fft.fft2(perturbation))
        # Periodic signals have concentrated peaks in spectrum
        peak_ratio = np.max(spectrum) / np.mean(spectrum)
        return peak_ratio > 10

    def correlate_with_corruptions(
        self,
        adversarial_example: AdversarialExample,
        image_np: np.ndarray
    ) -> Dict[CorruptionType, CorruptionCorrelation]:
        """
        Correlate adversarial perturbation with environmental corruptions.

        Tests hypothesis that adversarial directions encode natural challenges.
        """
        perturbation = adversarial_example.perturbation.cpu().numpy()
        if perturbation.ndim == 4:
            perturbation = perturbation[0]

        # Transpose from CHW to HWC if needed
        if perturbation.shape[0] in [1, 3] and perturbation.ndim == 3:
            perturbation = perturbation.transpose(1, 2, 0)

        correlations = {}

        for corruption_type in CorruptionType:
            try:
                result = self.corruptions.apply(image_np, corruption_type, severity=0.5)
                corruption_direction = self.corruptions.get_corruption_vector(
                    image_np, result.corrupted_image
                )

                # Compute correlations
                cosine_sim = self._cosine_similarity(
                    perturbation.flatten(),
                    corruption_direction.flatten()
                )
                structural_sim = self._structural_similarity(
                    perturbation, corruption_direction
                )
                freq_overlap = self._frequency_overlap(
                    perturbation, corruption_direction
                )

                correlations[corruption_type] = CorruptionCorrelation(
                    corruption_type=corruption_type,
                    cosine_similarity=cosine_sim,
                    structural_similarity=structural_sim,
                    frequency_overlap=freq_overlap
                )

            except Exception as e:
                logger.debug(f"Could not analyze {corruption_type}: {e}")

        return correlations

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _structural_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute structural similarity (simplified SSIM)."""
        # Ensure same shape
        if a.shape != b.shape:
            return 0.0

        # Mean and variance
        mean_a, mean_b = np.mean(a), np.mean(b)
        var_a, var_b = np.var(a), np.var(b)
        covar = np.mean((a - mean_a) * (b - mean_b))

        c1, c2 = 0.01**2, 0.03**2

        ssim = ((2*mean_a*mean_b + c1) * (2*covar + c2)) / \
               ((mean_a**2 + mean_b**2 + c1) * (var_a + var_b + c2))

        return float(ssim)

    def _frequency_overlap(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute overlap in frequency domain."""
        if a.ndim == 3:
            a = a.mean(axis=-1) if a.shape[-1] == 3 else a.mean(axis=0)
        if b.ndim == 3:
            b = b.mean(axis=-1) if b.shape[-1] == 3 else b.mean(axis=0)

        # Resize to same shape if needed
        if a.shape != b.shape:
            min_h = min(a.shape[0], b.shape[0])
            min_w = min(a.shape[1], b.shape[1])
            a = a[:min_h, :min_w]
            b = b[:min_h, :min_w]

        spectrum_a = np.abs(np.fft.fft2(a))
        spectrum_b = np.abs(np.fft.fft2(b))

        # Normalize spectra
        spectrum_a = spectrum_a / (np.sum(spectrum_a) + 1e-8)
        spectrum_b = spectrum_b / (np.sum(spectrum_b) + 1e-8)

        # Compute overlap (intersection over union in frequency space)
        intersection = np.minimum(spectrum_a, spectrum_b).sum()
        union = np.maximum(spectrum_a, spectrum_b).sum()

        return float(intersection / (union + 1e-8))

    def cluster_adversarial_directions(
        self,
        adversarial_examples: List[AdversarialExample],
        n_clusters: int = 5
    ) -> Dict[str, Any]:
        """
        Cluster adversarial directions to find common attack patterns.
        """
        # Extract perturbations
        perturbations = []
        for ex in adversarial_examples:
            pert = ex.perturbation.cpu().numpy().flatten()
            perturbations.append(pert)

        perturbation_matrix = np.array(perturbations)

        # PCA for dimensionality reduction
        pca = PCA(n_components=min(50, len(perturbations)))
        reduced = pca.fit_transform(perturbation_matrix)

        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(reduced)

        # Analyze each cluster
        cluster_analysis = {}
        for i in range(n_clusters):
            cluster_mask = clusters == i
            cluster_perts = perturbation_matrix[cluster_mask]

            # Mean perturbation for cluster
            mean_pert = cluster_perts.mean(axis=0)

            # Analyze structure
            cluster_analysis[f"cluster_{i}"] = {
                "size": int(cluster_mask.sum()),
                "mean_norm": float(np.linalg.norm(cluster_perts, axis=1).mean()),
                "variance": float(np.var(cluster_perts)),
                "center": kmeans.cluster_centers_[i].tolist()[:10]  # First 10 dims
            }

        return {
            "n_clusters": n_clusters,
            "cluster_labels": clusters.tolist(),
            "explained_variance_pca": pca.explained_variance_ratio_.tolist(),
            "cluster_analysis": cluster_analysis
        }
