"""
Pressure decomposition analysis.

Decomposes adversarial subspace into interpretable evolutionary pressure components.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from sklearn.decomposition import NMF, FastICA
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class PressureComponent:
    """A decomposed pressure component from adversarial space."""
    component_id: int
    direction: np.ndarray
    explained_variance: float
    interpretability_score: float
    matched_pressure: str
    confidence: float


class PressureDecomposer:
    """
    Decomposes adversarial perturbation space into evolutionary pressure components.
    """

    def __init__(self, n_components: int = 10):
        self.n_components = n_components

    def decompose_with_nmf(
        self,
        perturbation_matrix: np.ndarray
    ) -> List[PressureComponent]:
        """
        Use Non-negative Matrix Factorization to find pressure components.

        NMF is useful because it produces interpretable, additive components.
        """
        # Ensure non-negative
        pert_shifted = perturbation_matrix - perturbation_matrix.min()

        nmf = NMF(
            n_components=min(self.n_components, perturbation_matrix.shape[0]),
            init='nndsvda',
            random_state=42,
            max_iter=500
        )

        W = nmf.fit_transform(pert_shifted)
        H = nmf.components_

        components = []
        reconstruction = W @ H
        total_error = np.sum((pert_shifted - reconstruction) ** 2)
        total_variance = np.var(pert_shifted)

        for i in range(H.shape[0]):
            # Explained variance by this component
            partial_recon = np.outer(W[:, i], H[i])
            component_variance = np.var(partial_recon)
            explained = component_variance / (total_variance + 1e-8)

            # Interpretability: how sparse/localized is the component
            sparsity = 1 - (np.count_nonzero(H[i]) / len(H[i]))
            interpretability = sparsity

            components.append(PressureComponent(
                component_id=i,
                direction=H[i],
                explained_variance=float(explained),
                interpretability_score=float(interpretability),
                matched_pressure="unmatched",
                confidence=0.0
            ))

        return components

    def decompose_with_ica(
        self,
        perturbation_matrix: np.ndarray
    ) -> List[PressureComponent]:
        """
        Use Independent Component Analysis to find statistically independent pressures.

        ICA finds components that are maximally statistically independent,
        which may correspond to distinct evolutionary pressures.
        """
        n_comp = min(self.n_components, perturbation_matrix.shape[0],
                    perturbation_matrix.shape[1])

        ica = FastICA(n_components=n_comp, random_state=42, max_iter=500)

        try:
            S = ica.fit_transform(perturbation_matrix)
            A = ica.mixing_

            components = []
            for i in range(A.shape[1]):
                direction = A[:, i]

                # Explained variance (approximate for ICA)
                source_variance = np.var(S[:, i])
                total_variance = np.var(perturbation_matrix)
                explained = source_variance / (total_variance + 1e-8)

                # Non-Gaussianity as interpretability proxy
                _, p_value = stats.normaltest(S[:, i])
                non_gaussianity = 1 - p_value

                components.append(PressureComponent(
                    component_id=i,
                    direction=direction,
                    explained_variance=float(explained),
                    interpretability_score=float(non_gaussianity),
                    matched_pressure="unmatched",
                    confidence=0.0
                ))

            return components

        except Exception as e:
            logger.warning(f"ICA failed: {e}")
            return []

    def match_to_pressure_bases(
        self,
        components: List[PressureComponent],
        pressure_bases: Dict[str, np.ndarray]
    ) -> List[PressureComponent]:
        """
        Match decomposed components to known evolutionary pressure bases.
        """
        matched_components = []

        for comp in components:
            best_match = None
            best_similarity = 0.0

            for pressure_name, basis in pressure_bases.items():
                # Resize to match if needed
                min_len = min(len(comp.direction), len(basis))
                similarity = abs(np.dot(
                    comp.direction[:min_len] / (np.linalg.norm(comp.direction[:min_len]) + 1e-8),
                    basis[:min_len] / (np.linalg.norm(basis[:min_len]) + 1e-8)
                ))

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = pressure_name

            matched_components.append(PressureComponent(
                component_id=comp.component_id,
                direction=comp.direction,
                explained_variance=comp.explained_variance,
                interpretability_score=comp.interpretability_score,
                matched_pressure=best_match or "unmatched",
                confidence=float(best_similarity)
            ))

        return matched_components

    def analyze_pressure_coverage(
        self,
        matched_components: List[PressureComponent],
        threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Analyze how well the decomposition covers known evolutionary pressures.
        """
        matched_pressures = {}

        for comp in matched_components:
            if comp.confidence >= threshold:
                if comp.matched_pressure not in matched_pressures:
                    matched_pressures[comp.matched_pressure] = []
                matched_pressures[comp.matched_pressure].append({
                    "component_id": comp.component_id,
                    "confidence": comp.confidence,
                    "explained_variance": comp.explained_variance
                })

        total_explained = sum(
            comp.explained_variance
            for comp in matched_components
            if comp.confidence >= threshold
        )

        return {
            "matched_pressures": matched_pressures,
            "n_pressures_found": len(matched_pressures),
            "total_explained_variance": total_explained,
            "unmapped_components": [
                comp.component_id
                for comp in matched_components
                if comp.confidence < threshold
            ]
        }
