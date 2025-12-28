"""
Feature tracking for ecological analysis of neural networks.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeaturePopulation:
    """Represents a population of similar features."""
    population_id: int
    centroid: np.ndarray
    member_indices: List[int]
    population_size: int
    birth_step: int
    death_step: Optional[int] = None
    lineage: List[int] = field(default_factory=list)

    @property
    def is_alive(self) -> bool:
        return self.death_step is None

    @property
    def lifespan(self) -> Optional[int]:
        if self.death_step is None:
            return None
        return self.death_step - self.birth_step


@dataclass
class EcologicalSnapshot:
    """Snapshot of feature ecology at a training step."""
    step: int
    populations: List[FeaturePopulation]
    total_features: int
    diversity_index: float
    extinction_count: int
    birth_count: int


class FeatureTracker:
    """
    Tracks features through training to analyze ecological dynamics.

    Features are clustered into "populations" and tracked over time.
    """

    def __init__(
        self,
        n_clusters: int = 50,
        similarity_threshold: float = 0.8,
        min_population_size: int = 5,
    ):
        self.n_clusters = n_clusters
        self.similarity_threshold = similarity_threshold
        self.min_population_size = min_population_size

        self.snapshots: List[EcologicalSnapshot] = []
        self.all_populations: Dict[int, FeaturePopulation] = {}
        self._next_population_id = 0

    def extract_features(
        self,
        model: torch.nn.Module,
        layer_name: str,
    ) -> np.ndarray:
        """
        Extract feature vectors from a model layer.

        Returns:
            Feature matrix of shape (n_features, feature_dim)
        """
        for name, module in model.named_modules():
            if name == layer_name:
                if hasattr(module, 'weight'):
                    return module.weight.detach().cpu().numpy()

        raise ValueError(f"Layer {layer_name} not found")

    def cluster_features(
        self,
        features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster features into populations.

        Returns:
            (cluster_labels, cluster_centroids)
        """
        # Determine optimal number of clusters
        n_clusters = min(self.n_clusters, len(features) - 1)

        if n_clusters < 2:
            return np.zeros(len(features), dtype=int), features.mean(axis=0, keepdims=True)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        centroids = kmeans.cluster_centers_

        return labels, centroids

    def record_snapshot(
        self,
        step: int,
        features: np.ndarray,
    ) -> EcologicalSnapshot:
        """
        Record an ecological snapshot at a training step.
        """
        labels, centroids = self.cluster_features(features)

        # Create populations
        new_populations = []
        for cluster_id in range(len(centroids)):
            member_indices = np.where(labels == cluster_id)[0].tolist()
            if len(member_indices) >= self.min_population_size:
                pop = FeaturePopulation(
                    population_id=self._next_population_id,
                    centroid=centroids[cluster_id],
                    member_indices=member_indices,
                    population_size=len(member_indices),
                    birth_step=step,
                )
                new_populations.append(pop)
                self.all_populations[pop.population_id] = pop
                self._next_population_id += 1

        # Track births and deaths
        birth_count = len(new_populations)
        extinction_count = 0

        if self.snapshots:
            prev_snapshot = self.snapshots[-1]
            # Mark extinctions
            for prev_pop in prev_snapshot.populations:
                if prev_pop.is_alive:
                    # Check if population survived
                    survived = False
                    for new_pop in new_populations:
                        similarity = self._compute_similarity(
                            prev_pop.centroid, new_pop.centroid
                        )
                        if similarity > self.similarity_threshold:
                            survived = True
                            new_pop.lineage = prev_pop.lineage + [prev_pop.population_id]
                            break

                    if not survived:
                        prev_pop.death_step = step
                        extinction_count += 1

        # Compute diversity
        diversity = self._compute_diversity(new_populations)

        snapshot = EcologicalSnapshot(
            step=step,
            populations=new_populations,
            total_features=len(features),
            diversity_index=diversity,
            extinction_count=extinction_count,
            birth_count=birth_count,
        )

        self.snapshots.append(snapshot)
        return snapshot

    def _compute_similarity(
        self,
        centroid1: np.ndarray,
        centroid2: np.ndarray,
    ) -> float:
        """Compute cosine similarity between centroids."""
        norm1 = np.linalg.norm(centroid1)
        norm2 = np.linalg.norm(centroid2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(centroid1, centroid2) / (norm1 * norm2))

    def _compute_diversity(
        self,
        populations: List[FeaturePopulation],
    ) -> float:
        """Compute Shannon diversity index."""
        if not populations:
            return 0.0

        total = sum(p.population_size for p in populations)
        if total == 0:
            return 0.0

        diversity = 0.0
        for pop in populations:
            p = pop.population_size / total
            if p > 0:
                diversity -= p * np.log(p)

        return diversity

    def get_population_dynamics(self) -> Dict[str, Any]:
        """
        Analyze population dynamics over training.
        """
        if not self.snapshots:
            return {}

        steps = [s.step for s in self.snapshots]
        diversity = [s.diversity_index for s in self.snapshots]
        extinctions = [s.extinction_count for s in self.snapshots]
        births = [s.birth_count for s in self.snapshots]
        population_counts = [len(s.populations) for s in self.snapshots]

        # Compute lifespans
        lifespans = []
        for pop in self.all_populations.values():
            if pop.lifespan is not None:
                lifespans.append(pop.lifespan)

        return {
            "steps": steps,
            "diversity_trajectory": diversity,
            "extinction_trajectory": extinctions,
            "birth_trajectory": births,
            "population_count_trajectory": population_counts,
            "total_populations": len(self.all_populations),
            "total_extinctions": sum(extinctions),
            "mean_lifespan": np.mean(lifespans) if lifespans else None,
            "median_lifespan": np.median(lifespans) if lifespans else None,
        }
