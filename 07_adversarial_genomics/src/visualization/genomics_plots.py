"""
Visualization for adversarial genomics analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import logging

from ..experiments.evolutionary_mapping import PressureMapping, EvolutionaryPressure
from ..analysis.biological_comparison import BiologicalAlignment

logger = logging.getLogger(__name__)


class GenomicsPlotter:
    """Creates visualizations for adversarial genomics analysis."""

    def __init__(self, output_dir: str = "results/plots"):
        self.output_dir = output_dir

    def plot_pressure_mapping(
        self,
        mapping: PressureMapping,
        filename: str = "pressure_mapping.pdf"
    ):
        """Plot evolutionary pressure mapping results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Pressure strengths
        ax = axes[0]
        pressures = mapping.pressures[:10]
        names = [p.pressure_name for p in pressures]
        strengths = [p.strength for p in pressures]
        confidences = [p.confidence for p in pressures]

        x = np.arange(len(names))
        width = 0.35

        bars1 = ax.bar(x - width/2, strengths, width, label='Strength', color='steelblue')
        bars2 = ax.bar(x + width/2, confidences, width, label='Confidence', color='coral')

        ax.set_ylabel('Score')
        ax.set_title('Evolutionary Pressures in Adversarial Space')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right: Variance decomposition
        ax = axes[1]
        labels = ['Mapped', 'Unmapped']
        sizes = [mapping.total_explained_variance, mapping.unmapped_variance]
        colors = ['steelblue', 'lightgray']

        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Adversarial Variance Decomposition')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{filename}", dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved pressure mapping plot to {self.output_dir}/{filename}")

    def plot_biological_comparison(
        self,
        alignment: BiologicalAlignment,
        comparison: Dict[str, Any],
        filename: str = "biological_comparison.pdf"
    ):
        """Plot biological comparison results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Top left: Gabor similarity comparison
        ax = axes[0, 0]
        gabor_data = comparison.get('gabor_similarity', {})
        categories = ['Standard', 'Robust']
        values = [gabor_data.get('standard', 0), gabor_data.get('robust', 0)]
        colors = ['lightcoral', 'steelblue']

        bars = ax.bar(categories, values, color=colors)
        ax.set_ylabel('Gabor Similarity')
        ax.set_title('V1-like Feature Similarity')
        ax.set_ylim(0, 1)

        # Add significance indicator
        if gabor_data.get('p_value', 1) < 0.05:
            ax.annotate('*', xy=(0.5, max(values) + 0.05),
                       ha='center', fontsize=20)

        ax.grid(True, alpha=0.3)

        # Top right: Orientation selectivity
        ax = axes[0, 1]
        osi_data = comparison.get('orientation_selectivity', {})
        categories = ['Standard', 'Robust']
        values = [osi_data.get('standard_mean_osi', 0),
                 osi_data.get('robust_mean_osi', 0)]

        bars = ax.bar(categories, values, color=colors)
        ax.set_ylabel('Orientation Selectivity Index')
        ax.set_title('Orientation Selectivity')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # Bottom left: Top feature matches
        ax = axes[1, 0]
        if alignment.top_matches:
            orientations = [m.orientation or 0 for m in alignment.top_matches]
            similarities = [m.similarity for m in alignment.top_matches]

            ax.scatter(orientations, similarities, c='steelblue', s=100, alpha=0.7)
            ax.set_xlabel('Preferred Orientation (radians)')
            ax.set_ylabel('Gabor Similarity')
            ax.set_title('Top V1-like Features')
            ax.set_xlim(0, np.pi)
            ax.grid(True, alpha=0.3)

        # Bottom right: Summary text
        ax = axes[1, 1]
        ax.axis('off')

        summary_text = f"""
Biological Alignment Summary
============================

Brain Region: {alignment.brain_region}
Alignment Score: {alignment.alignment_score:.3f}
V1-matched Features: {alignment.n_matched_features}

Gabor Similarity Improvement: {gabor_data.get('improvement', 0):.3f}
p-value: {gabor_data.get('p_value', 1):.4f}

Conclusion:
{comparison.get('conclusion', 'N/A')}
        """

        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{filename}", dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved biological comparison to {self.output_dir}/{filename}")

    def plot_corruption_correlations(
        self,
        correlations: Dict[str, float],
        filename: str = "corruption_correlations.pdf"
    ):
        """Plot correlations between adversarial perturbations and corruptions."""
        fig, ax = plt.subplots(figsize=(12, 6))

        names = list(correlations.keys())
        values = list(correlations.values())

        # Sort by value
        sorted_idx = np.argsort(values)[::-1]
        names = [names[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]

        colors = ['steelblue' if v > 0.3 else 'lightcoral' for v in values]

        bars = ax.barh(names, values, color=colors)
        ax.set_xlabel('Cosine Similarity')
        ax.set_title('Adversarial-Corruption Correlation')
        ax.axvline(x=0.3, color='red', linestyle='--', label='Significance threshold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{filename}", dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved corruption correlations to {self.output_dir}/{filename}")

    def create_summary_figure(
        self,
        mapping: PressureMapping,
        alignment: BiologicalAlignment,
        correlations: Dict[str, float],
        filename: str = "adversarial_genomics_summary.pdf"
    ):
        """Create comprehensive summary figure."""
        fig = plt.figure(figsize=(16, 12))

        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # A: Pressure strengths
        ax = fig.add_subplot(gs[0, :2])
        pressures = mapping.pressures[:8]
        names = [p.pressure_name for p in pressures]
        strengths = [p.strength for p in pressures]

        ax.barh(names, strengths, color='steelblue')
        ax.set_xlabel('Strength')
        ax.set_title('A. Evolutionary Pressures in Adversarial Space')
        ax.grid(True, alpha=0.3, axis='x')

        # B: Variance pie
        ax = fig.add_subplot(gs[0, 2])
        sizes = [mapping.total_explained_variance, mapping.unmapped_variance]
        ax.pie(sizes, labels=['Mapped', 'Unmapped'],
              colors=['steelblue', 'lightgray'], autopct='%1.1f%%')
        ax.set_title('B. Variance Decomposition')

        # C: Corruption correlations
        ax = fig.add_subplot(gs[1, :])
        sorted_corr = sorted(correlations.items(), key=lambda x: -x[1])[:10]
        names = [c[0] for c in sorted_corr]
        values = [c[1] for c in sorted_corr]

        ax.bar(names, values, color='coral')
        ax.set_ylabel('Correlation')
        ax.set_title('C. Top Adversarial-Corruption Correlations')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        # D: Biological alignment
        ax = fig.add_subplot(gs[2, 0])
        ax.barh(['Alignment\nScore'], [alignment.alignment_score],
               color='forestgreen')
        ax.set_xlim(0, 1)
        ax.set_title('D. V1 Alignment')
        ax.grid(True, alpha=0.3, axis='x')

        # E: Top matches scatter
        ax = fig.add_subplot(gs[2, 1])
        if alignment.top_matches:
            orientations = [m.orientation or 0 for m in alignment.top_matches]
            similarities = [m.similarity for m in alignment.top_matches]
            ax.scatter(orientations, similarities, c='forestgreen', s=80, alpha=0.7)
        ax.set_xlabel('Orientation')
        ax.set_ylabel('Similarity')
        ax.set_title('E. V1 Feature Matches')
        ax.grid(True, alpha=0.3)

        # F: Summary text
        ax = fig.add_subplot(gs[2, 2])
        ax.axis('off')

        summary = f"""Key Findings:
• {len(mapping.pressures)} pressures identified
• {mapping.total_explained_variance:.1%} variance explained
• V1 alignment: {alignment.alignment_score:.2f}
• {alignment.n_matched_features} V1-like features"""

        ax.text(0.1, 0.8, summary, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='sans-serif',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.suptitle('Adversarial Genomics: Evolutionary Pressures in Adversarial Space',
                    fontsize=14, fontweight='bold')

        plt.savefig(f"{self.output_dir}/{filename}", dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved summary figure to {self.output_dir}/{filename}")
