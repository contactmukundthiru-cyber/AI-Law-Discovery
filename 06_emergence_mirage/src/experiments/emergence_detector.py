"""
Emergence detection experiments.

Tests whether capabilities emerge suddenly or continuously across model scales.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable
from scipy import stats
from scipy.optimize import curve_fit
import logging

from ..data.capability_tasks import TaskSample, CapabilityTaskGenerator
from ..models.model_interface import ModelInterface, ModelOutput

logger = logging.getLogger(__name__)


@dataclass
class EmergenceResult:
    """Results from emergence detection."""
    capability: str
    model_name: str
    parameter_count: int

    # Different metrics for same capability
    binary_accuracy: float
    partial_credit_score: float
    log_probability_score: float
    rank_score: float
    entropy_score: float

    # Per-sample results for detailed analysis
    sample_results: List[Dict[str, Any]]

    # Metadata
    n_samples: int
    mean_difficulty: float


class EmergenceDetector:
    """
    Detects whether capability emergence is real or a measurement artifact.

    Uses multiple evaluation metrics to check if emergence appears continuous
    with some metrics but discontinuous with others.
    """

    def __init__(
        self,
        task_generator: CapabilityTaskGenerator,
        n_samples_per_capability: int = 100
    ):
        self.task_generator = task_generator
        self.n_samples = n_samples_per_capability

    def evaluate_model(
        self,
        model: ModelInterface,
        capability: str,
        difficulty_range: Tuple[float, float] = (0.0, 1.0)
    ) -> EmergenceResult:
        """
        Evaluate a model on a capability with multiple metrics.

        Returns comprehensive results enabling emergence analysis.
        """
        # Generate tasks
        tasks = self.task_generator.generate_tasks(
            capability,
            self.n_samples,
            difficulty_range
        )

        logger.info(f"Evaluating {model.model_name} on {capability} "
                   f"({len(tasks)} samples)")

        sample_results = []

        for task in tasks:
            # Get model prediction with probability info
            output = model.generate(task.prompt, max_tokens=50)

            # Get probability of correct answer
            log_prob, token_probs = model.get_token_probabilities(
                task.prompt, task.correct_answer
            )

            # Compute various metrics
            result = self._compute_metrics(task, output, log_prob)
            sample_results.append(result)

        # Aggregate metrics
        binary_acc = np.mean([r['binary_correct'] for r in sample_results])
        partial_score = np.mean([r['partial_credit'] for r in sample_results])
        log_prob_score = np.mean([r['log_probability'] for r in sample_results])
        rank_score = np.mean([r['rank_score'] for r in sample_results])
        entropy_score = np.mean([r['entropy_score'] for r in sample_results])

        mean_difficulty = np.mean([t.difficulty for t in tasks])

        return EmergenceResult(
            capability=capability,
            model_name=model.model_name,
            parameter_count=model.parameter_count,
            binary_accuracy=binary_acc,
            partial_credit_score=partial_score,
            log_probability_score=log_prob_score,
            rank_score=rank_score,
            entropy_score=entropy_score,
            sample_results=sample_results,
            n_samples=len(tasks),
            mean_difficulty=mean_difficulty
        )

    def _compute_metrics(
        self,
        task: TaskSample,
        output: ModelOutput,
        target_log_prob: float
    ) -> Dict[str, Any]:
        """Compute multiple evaluation metrics for a single sample."""
        prediction = output.prediction.strip().lower()
        target = task.correct_answer.strip().lower()

        # 1. Binary (exact match)
        binary_correct = prediction == target or target in prediction

        # 2. Partial credit (character-level similarity)
        partial_credit = self._compute_partial_credit(prediction, target)

        # 3. Log probability of correct answer
        # Normalize by length
        log_probability = target_log_prob / max(len(target), 1)

        # 4. Rank of correct answer
        # Check if correct answer is in top-k predictions
        rank_score = self._compute_rank_score(output.top_k_predictions, target)

        # 5. Entropy-based score
        # Lower entropy = more confident prediction
        entropy_score = 1.0 / (1.0 + output.entropy)

        return {
            'task_id': task.task_id,
            'difficulty': task.difficulty,
            'prediction': prediction,
            'target': target,
            'binary_correct': float(binary_correct),
            'partial_credit': partial_credit,
            'log_probability': log_probability,
            'rank_score': rank_score,
            'entropy_score': entropy_score,
            'raw_entropy': output.entropy,
            'raw_log_prob': target_log_prob
        }

    def _compute_partial_credit(self, prediction: str, target: str) -> float:
        """Compute partial credit based on longest common subsequence."""
        if not prediction or not target:
            return 0.0

        # LCS-based similarity
        m, n = len(prediction), len(target)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if prediction[i-1] == target[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        lcs_len = dp[m][n]
        return (2 * lcs_len) / (m + n)

    def _compute_rank_score(
        self,
        top_k: List[Tuple[str, float]],
        target: str
    ) -> float:
        """Compute rank-based score for correct answer."""
        if not top_k:
            return 0.0

        for rank, (token, prob) in enumerate(top_k):
            if target.lower().startswith(token.strip().lower()):
                # Higher score for higher rank
                return 1.0 / (1 + rank)

        return 0.0

    def run_scale_sweep(
        self,
        models: List[ModelInterface],
        capability: str,
        difficulty_range: Tuple[float, float] = (0.0, 1.0)
    ) -> List[EmergenceResult]:
        """Run emergence detection across multiple model scales."""
        results = []

        for model in sorted(models, key=lambda m: m.parameter_count):
            result = self.evaluate_model(model, capability, difficulty_range)
            results.append(result)
            logger.info(f"  {model.model_name}: binary={result.binary_accuracy:.3f}, "
                       f"partial={result.partial_credit_score:.3f}, "
                       f"log_prob={result.log_probability_score:.3f}")

        return results


class SubThresholdProber:
    """
    Probes for sub-threshold capability existence.

    Attempts to detect capabilities that exist but are too weak
    to show up in standard evaluation.
    """

    def __init__(self, model: ModelInterface):
        self.model = model

    def probe_capability(
        self,
        task: TaskSample,
        n_samples: int = 100,
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        Probe for weak capability presence using sampling.

        Even if the model doesn't get the answer right deterministically,
        sampling may reveal that the correct answer has elevated probability.
        """
        correct_count = 0
        partial_scores = []
        correct_in_top_k = 0

        for _ in range(n_samples):
            output = self.model.generate(
                task.prompt,
                max_tokens=50,
                temperature=temperature
            )

            prediction = output.prediction.strip().lower()
            target = task.correct_answer.strip().lower()

            if prediction == target or target in prediction:
                correct_count += 1

            # Partial matching
            overlap = len(set(prediction.split()) & set(target.split()))
            if target.split():
                partial_scores.append(overlap / len(target.split()))
            else:
                partial_scores.append(0.0)

        return {
            'task_id': task.task_id,
            'sample_accuracy': correct_count / n_samples,
            'mean_partial_score': np.mean(partial_scores),
            'capability_detected': correct_count > 0,
            'capability_strength': correct_count / n_samples
        }

    def probability_distribution_analysis(
        self,
        prompt: str,
        correct_tokens: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze full probability distribution to detect sub-threshold capability.

        Even if the model doesn't select the correct answer, elevated
        probability of correct tokens indicates capability presence.
        """
        output = self.model.generate(prompt, max_tokens=1, temperature=0.0)

        # Check probability mass on correct tokens
        correct_prob_mass = 0.0
        for token, prob in output.top_k_predictions:
            if any(ct.lower() in token.lower() for ct in correct_tokens):
                correct_prob_mass += prob

        # Compare to baseline (random) probability
        baseline_prob = 1.0 / 50000  # Approximate vocab size

        return {
            'correct_probability_mass': correct_prob_mass,
            'probability_ratio': correct_prob_mass / baseline_prob,
            'above_chance': correct_prob_mass > baseline_prob,
            'capability_detected': correct_prob_mass > 0.01  # 1% threshold
        }
