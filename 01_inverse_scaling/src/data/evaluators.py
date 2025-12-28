"""
Task evaluators for inverse scaling experiments.

Provides evaluation logic for different task types to measure
model performance and detect inverse scaling patterns.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import difflib

from .task_generators import TaskInstance


@dataclass
class EvaluationResult:
    """Result of evaluating a single response."""
    task_instance: TaskInstance
    model_response: str
    is_correct: bool
    score: float  # 0.0 to 1.0
    metrics: Dict[str, Any]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_instance.task_type,
            "subtask": self.task_instance.subtask,
            "prompt": self.task_instance.prompt,
            "expected": self.task_instance.expected_answer,
            "response": self.model_response,
            "is_correct": self.is_correct,
            "score": self.score,
            "metrics": self.metrics,
            "error": self.error,
        }


class TaskEvaluator(ABC):
    """Abstract base class for task evaluators."""

    @abstractmethod
    def evaluate(
        self,
        instance: TaskInstance,
        response: str,
    ) -> EvaluationResult:
        """Evaluate a model response against the expected answer."""
        pass

    def evaluate_batch(
        self,
        instances: List[TaskInstance],
        responses: List[str],
    ) -> List[EvaluationResult]:
        """Evaluate a batch of responses."""
        return [
            self.evaluate(inst, resp)
            for inst, resp in zip(instances, responses)
        ]

    @staticmethod
    def normalize_response(response: str) -> str:
        """Normalize response for comparison."""
        # Strip whitespace and common prefixes
        response = response.strip()

        # Remove common response prefixes
        prefixes = [
            "The answer is",
            "Answer:",
            "Result:",
            "Output:",
            "Response:",
        ]
        for prefix in prefixes:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()

        # Remove trailing punctuation
        response = response.rstrip(".,!?")

        return response


class ArithmeticEvaluator(TaskEvaluator):
    """Evaluator for arithmetic tasks."""

    def evaluate(
        self,
        instance: TaskInstance,
        response: str,
    ) -> EvaluationResult:
        """Evaluate arithmetic response."""
        normalized = self.normalize_response(response)

        # Extract numbers from response
        numbers = re.findall(r'-?\d+\.?\d*', normalized)

        metrics = {
            "response_length": len(response),
            "numbers_found": len(numbers),
            "normalized_response": normalized,
        }

        # Check if any extracted number matches expected
        expected = instance.expected_answer.strip()
        is_correct = False
        score = 0.0

        if numbers:
            # Check if first number matches
            if numbers[0] == expected:
                is_correct = True
                score = 1.0
            # Check if any number matches (partial credit consideration)
            elif expected in numbers:
                is_correct = True
                score = 0.8  # Slight penalty for not being first

        # Overthinking detection
        metrics["overthinking_indicators"] = {
            "excessive_length": len(response) > 50,
            "contains_explanation": any(
                word in response.lower()
                for word in ["because", "therefore", "since", "so", "thus"]
            ),
            "multiple_numbers": len(numbers) > 1,
            "hedging_language": any(
                word in response.lower()
                for word in ["might", "could", "perhaps", "maybe", "probably"]
            ),
        }

        metrics["overthinking_score"] = sum(
            metrics["overthinking_indicators"].values()
        ) / len(metrics["overthinking_indicators"])

        return EvaluationResult(
            task_instance=instance,
            model_response=response,
            is_correct=is_correct,
            score=score,
            metrics=metrics,
        )


class ExactMatchEvaluator(TaskEvaluator):
    """Evaluator for exact match tasks."""

    def __init__(self, case_sensitive: bool = False, strip_whitespace: bool = True):
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace

    def evaluate(
        self,
        instance: TaskInstance,
        response: str,
    ) -> EvaluationResult:
        """Evaluate exact match response."""
        expected = instance.expected_answer
        actual = response

        if self.strip_whitespace:
            expected = expected.strip()
            actual = actual.strip()

        if not self.case_sensitive:
            expected = expected.lower()
            actual = actual.lower()

        # Handle multiple valid answers (separated by |)
        valid_answers = expected.split("|")
        actual_normalized = actual.lower() if not self.case_sensitive else actual

        is_correct = any(
            (va.lower() if not self.case_sensitive else va) == actual_normalized
            for va in valid_answers
        )

        # Calculate similarity for partial scoring
        best_similarity = max(
            difflib.SequenceMatcher(None, actual, va).ratio()
            for va in valid_answers
        )

        metrics = {
            "response_length": len(response),
            "expected_length": len(instance.expected_answer),
            "length_difference": abs(len(response) - len(instance.expected_answer)),
            "similarity": best_similarity,
            "exact_match": is_correct,
        }

        # Overthinking indicators
        metrics["extra_content"] = len(response) > len(instance.expected_answer) * 1.5
        metrics["added_explanation"] = any(
            marker in response.lower()
            for marker in [":", "=", "answer", "result", "note"]
        )

        score = 1.0 if is_correct else best_similarity * 0.5

        return EvaluationResult(
            task_instance=instance,
            model_response=response,
            is_correct=is_correct,
            score=score,
            metrics=metrics,
        )


class FormatComplianceEvaluator(TaskEvaluator):
    """Evaluator for format compliance tasks."""

    def evaluate(
        self,
        instance: TaskInstance,
        response: str,
    ) -> EvaluationResult:
        """Evaluate format compliance."""
        expected = instance.expected_answer.strip()
        actual = response.strip()

        metrics = {
            "response_length": len(response),
            "expected_length": len(expected),
        }

        # Check exact format match
        exact_match = actual == expected

        # Check structural compliance
        structural_match = self._check_structure(expected, actual)

        # Calculate scores
        if exact_match:
            score = 1.0
            is_correct = True
        elif structural_match:
            score = 0.7
            is_correct = False
            metrics["structural_match"] = True
        else:
            similarity = difflib.SequenceMatcher(None, actual, expected).ratio()
            score = similarity * 0.5
            is_correct = False

        metrics["format_violations"] = self._detect_violations(expected, actual)

        return EvaluationResult(
            task_instance=instance,
            model_response=response,
            is_correct=is_correct,
            score=score,
            metrics=metrics,
        )

    def _check_structure(self, expected: str, actual: str) -> bool:
        """Check if response has similar structure to expected."""
        # Extract structural elements
        expected_structure = re.findall(r'[\[\]{}():,\-=]', expected)
        actual_structure = re.findall(r'[\[\]{}():,\-=]', actual)
        return expected_structure == actual_structure

    def _detect_violations(self, expected: str, actual: str) -> List[str]:
        """Detect format violations."""
        violations = []

        if len(actual) > len(expected) * 2:
            violations.append("excessive_length")

        if "\n" in actual and "\n" not in expected:
            violations.append("unexpected_newlines")

        if re.search(r'^\s*(Here|The|This|Note|Answer)', actual, re.IGNORECASE):
            violations.append("added_preamble")

        if actual.count("{") != expected.count("{"):
            violations.append("brace_mismatch")

        if actual.count("[") != expected.count("["):
            violations.append("bracket_mismatch")

        return violations


class PersonaEvaluator(TaskEvaluator):
    """Evaluator for persona maintenance tasks."""

    def evaluate(
        self,
        instance: TaskInstance,
        response: str,
    ) -> EvaluationResult:
        """Evaluate persona maintenance."""
        expected = instance.expected_answer
        metadata = instance.metadata

        metrics = {
            "response_length": len(response),
            "word_count": len(response.split()),
        }

        is_correct = False
        score = 0.0

        # Check based on subtask type
        if instance.subtask == "single_trait_persona":
            trait = metadata.get("trait", "")
            markers = metadata.get("markers", [])

            if trait == "negative":
                marker_count = sum(1 for m in markers if m.lower() in response.lower())
                score = min(1.0, marker_count / 3)
                is_correct = score >= 0.5
                metrics["trait_markers_found"] = marker_count

            elif trait == "positive":
                marker_count = sum(1 for m in markers if m.lower() in response.lower())
                score = min(1.0, marker_count / 3)
                is_correct = score >= 0.5
                metrics["trait_markers_found"] = marker_count

            elif trait == "questions":
                is_question = "?" in response
                is_correct = is_question
                score = 1.0 if is_question else 0.0
                metrics["is_question"] = is_question

            elif trait == "brief":
                word_count = len(response.split())
                is_correct = word_count <= 5
                score = max(0.0, 1.0 - (word_count - 5) * 0.1) if word_count > 5 else 1.0
                metrics["exceeds_word_limit"] = word_count > 5

        elif instance.subtask == "restricted_vocabulary":
            allowed_words = metadata.get("allowed_words", [])
            response_words = set(re.findall(r'\b\w+\b', response.lower()))
            allowed_set = set(w.lower() for w in allowed_words)

            violations = response_words - allowed_set
            compliance_ratio = 1.0 - len(violations) / max(len(response_words), 1)

            is_correct = len(violations) == 0
            score = compliance_ratio
            metrics["vocabulary_violations"] = list(violations)
            metrics["compliance_ratio"] = compliance_ratio

        elif instance.subtask == "consistent_style":
            style = metadata.get("style", "")

            if style == "ALL CAPS":
                is_upper = response.upper() == response
                # Allow some flexibility for punctuation
                alpha_chars = [c for c in response if c.isalpha()]
                upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1)
                is_correct = upper_ratio >= 0.9
                score = upper_ratio
                metrics["uppercase_ratio"] = upper_ratio

            elif style == "all lowercase":
                alpha_chars = [c for c in response if c.isalpha()]
                lower_ratio = sum(1 for c in alpha_chars if c.islower()) / max(len(alpha_chars), 1)
                is_correct = lower_ratio >= 0.9
                score = lower_ratio
                metrics["lowercase_ratio"] = lower_ratio

            elif style == "Every Word Capitalized":
                words = response.split()
                capitalized = sum(1 for w in words if w and w[0].isupper())
                cap_ratio = capitalized / max(len(words), 1)
                is_correct = cap_ratio >= 0.8
                score = cap_ratio
                metrics["capitalization_ratio"] = cap_ratio

        return EvaluationResult(
            task_instance=instance,
            model_response=response,
            is_correct=is_correct,
            score=score,
            metrics=metrics,
        )


class UnifiedEvaluator:
    """
    Unified evaluator that routes to appropriate task-specific evaluators.
    """

    def __init__(self):
        self.evaluators = {
            "simple_arithmetic": ArithmeticEvaluator(),
            "rule_following": ExactMatchEvaluator(),
            "simple_patterns": ExactMatchEvaluator(),
            "literal_instructions": ExactMatchEvaluator(case_sensitive=False),
            "persona_maintenance": PersonaEvaluator(),
        }

        # Subtask-specific overrides
        self.subtask_evaluators = {
            "follow_format": FormatComplianceEvaluator(),
            "exact_format_output": FormatComplianceEvaluator(),
        }

    def evaluate(
        self,
        instance: TaskInstance,
        response: str,
    ) -> EvaluationResult:
        """Route to appropriate evaluator and evaluate."""
        # Check for subtask-specific evaluator first
        if instance.subtask in self.subtask_evaluators:
            evaluator = self.subtask_evaluators[instance.subtask]
        elif instance.task_type in self.evaluators:
            evaluator = self.evaluators[instance.task_type]
        else:
            # Default to exact match
            evaluator = ExactMatchEvaluator()

        return evaluator.evaluate(instance, response)

    def evaluate_batch(
        self,
        instances: List[TaskInstance],
        responses: List[str],
    ) -> List[EvaluationResult]:
        """Evaluate a batch of responses."""
        return [
            self.evaluate(inst, resp)
            for inst, resp in zip(instances, responses)
        ]

    def compute_aggregate_metrics(
        self,
        results: List[EvaluationResult],
    ) -> Dict[str, Any]:
        """Compute aggregate metrics from evaluation results."""
        if not results:
            return {}

        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        scores = [r.score for r in results]

        # Group by task type
        by_task = {}
        for r in results:
            task_type = r.task_instance.task_type
            if task_type not in by_task:
                by_task[task_type] = []
            by_task[task_type].append(r)

        # Group by subtask
        by_subtask = {}
        for r in results:
            subtask = r.task_instance.subtask
            if subtask not in by_subtask:
                by_subtask[subtask] = []
            by_subtask[subtask].append(r)

        # Compute overthinking metrics where available
        overthinking_scores = []
        for r in results:
            if "overthinking_score" in r.metrics:
                overthinking_scores.append(r.metrics["overthinking_score"])

        return {
            "total_samples": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0,
            "mean_score": sum(scores) / total if total > 0 else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "by_task_type": {
                task: {
                    "accuracy": sum(1 for r in rs if r.is_correct) / len(rs),
                    "mean_score": sum(r.score for r in rs) / len(rs),
                    "count": len(rs),
                }
                for task, rs in by_task.items()
            },
            "by_subtask": {
                subtask: {
                    "accuracy": sum(1 for r in rs if r.is_correct) / len(rs),
                    "mean_score": sum(r.score for r in rs) / len(rs),
                    "count": len(rs),
                }
                for subtask, rs in by_subtask.items()
            },
            "overthinking": {
                "mean": sum(overthinking_scores) / len(overthinking_scores) if overthinking_scores else None,
                "max": max(overthinking_scores) if overthinking_scores else None,
                "samples_measured": len(overthinking_scores),
            },
        }
