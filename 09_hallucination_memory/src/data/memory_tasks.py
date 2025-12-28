"""
Memory-based tasks for hallucination analysis.

Tests whether hallucinations are corrupted memories rather than fabrications.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import random


@dataclass
class MemoryTask:
    """Task for testing memory-based hallucination."""
    task_id: str
    query: str
    true_fact: str
    similar_facts: List[str]  # Potentially confusable facts
    expected_hallucination_type: str
    training_frequency: str  # "high", "medium", "low", "rare"


@dataclass
class HallucinationAnalysis:
    """Analysis of a hallucination event."""
    query: str
    model_response: str
    true_answer: str
    is_hallucination: bool
    hallucination_type: str  # "fabrication", "confusion", "partial", "plausible"
    source_similarity: float
    memory_trace_found: bool


class MemoryTaskGenerator:
    """
    Generates tasks to test memory corruption hypothesis.

    Hypothesis: Hallucinations are not random fabrications but
    corrupted or confused memories from training data.
    """

    def __init__(self):
        self.fact_database = self._build_fact_database()

    def _build_fact_database(self) -> Dict[str, List[Dict]]:
        """Build database of facts with confusable alternatives."""
        return {
            "historical_dates": [
                {"fact": "World War I ended in 1918", "confusable": ["1919", "1917", "1920"]},
                {"fact": "The Berlin Wall fell in 1989", "confusable": ["1991", "1987", "1990"]},
                {"fact": "Columbus reached America in 1492", "confusable": ["1493", "1491", "1494"]},
            ],
            "scientific_facts": [
                {"fact": "Water boils at 100°C at sea level", "confusable": ["99°C", "101°C", "98°C"]},
                {"fact": "Light speed is about 300,000 km/s", "confusable": ["200,000", "350,000", "280,000"]},
                {"fact": "Humans have 23 pairs of chromosomes", "confusable": ["24", "22", "46 total"]},
            ],
            "geographic_facts": [
                {"fact": "Mount Everest is 8,849 meters tall", "confusable": ["8,848", "8,850", "8,847"]},
                {"fact": "The Amazon is the second longest river", "confusable": ["longest", "third longest"]},
                {"fact": "Russia is the largest country by area", "confusable": ["China", "Canada", "USA"]},
            ],
            "person_facts": [
                {"fact": "Einstein published relativity in 1905", "confusable": ["1906", "1915", "1903"]},
                {"fact": "Shakespeare was born in 1564", "confusable": ["1565", "1563", "1560"]},
                {"fact": "Newton published Principia in 1687", "confusable": ["1686", "1688", "1690"]},
            ]
        }

    def generate_tasks(self, n_per_category: int = 20) -> List[MemoryTask]:
        """Generate memory confusion tasks."""
        tasks = []
        task_id = 0

        for category, facts in self.fact_database.items():
            for i in range(n_per_category):
                fact_data = random.choice(facts)

                # Create query about the fact
                task = MemoryTask(
                    task_id=f"memory_{task_id}",
                    query=self._fact_to_question(fact_data["fact"]),
                    true_fact=fact_data["fact"],
                    similar_facts=fact_data["confusable"],
                    expected_hallucination_type="confusion",
                    training_frequency=random.choice(["high", "medium", "low"])
                )
                tasks.append(task)
                task_id += 1

        return tasks

    def _fact_to_question(self, fact: str) -> str:
        """Convert a fact to a question."""
        templates = [
            "What is the answer to: {}?",
            "Can you tell me about: {}?",
            "What do you know about {}?",
        ]
        return random.choice(templates).format(fact.split(" is ")[0] if " is " in fact else fact[:30])


class HallucinationDetector:
    """
    Detects and classifies hallucinations as memory corruptions.
    """

    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold

    def analyze_response(
        self,
        task: MemoryTask,
        model_response: str
    ) -> HallucinationAnalysis:
        """Analyze a model response for hallucination patterns."""
        true_answer = task.true_fact

        # Check if response contains true answer
        is_correct = self._check_contains(model_response, true_answer)

        if is_correct:
            return HallucinationAnalysis(
                query=task.query,
                model_response=model_response,
                true_answer=true_answer,
                is_hallucination=False,
                hallucination_type="none",
                source_similarity=1.0,
                memory_trace_found=True
            )

        # Check if it's a confusable fact
        is_confusion = False
        source_sim = 0.0
        for confusable in task.similar_facts:
            if self._check_contains(model_response, confusable):
                is_confusion = True
                source_sim = 0.8  # High similarity indicates memory confusion
                break

        # Classify hallucination type
        if is_confusion:
            h_type = "confusion"  # Memory confusion
            memory_trace = True
        elif self._is_plausible(model_response, task):
            h_type = "plausible"  # Plausible but wrong
            source_sim = 0.5
            memory_trace = True
        else:
            h_type = "fabrication"  # True fabrication
            source_sim = 0.0
            memory_trace = False

        return HallucinationAnalysis(
            query=task.query,
            model_response=model_response,
            true_answer=true_answer,
            is_hallucination=True,
            hallucination_type=h_type,
            source_similarity=source_sim,
            memory_trace_found=memory_trace
        )

    def _check_contains(self, response: str, target: str) -> bool:
        """Check if response contains target (fuzzy)."""
        response_lower = response.lower()
        target_lower = target.lower()

        # Extract key parts
        key_parts = target_lower.split()[:3]
        matches = sum(1 for part in key_parts if part in response_lower)

        return matches >= len(key_parts) * 0.6

    def _is_plausible(self, response: str, task: MemoryTask) -> bool:
        """Check if response is plausible (wrong but sensible)."""
        # Simple heuristic: response has similar structure/length
        response_len = len(response.split())
        expected_len = len(task.true_fact.split())

        return 0.5 < response_len / expected_len < 2.0

    def compute_memory_corruption_ratio(
        self,
        analyses: List[HallucinationAnalysis]
    ) -> Dict[str, Any]:
        """Compute how many hallucinations are memory corruptions vs fabrications."""
        if not analyses:
            return {}

        hallucinations = [a for a in analyses if a.is_hallucination]

        if not hallucinations:
            return {"hallucination_rate": 0, "memory_corruption_ratio": 0}

        type_counts = {}
        for h in hallucinations:
            type_counts[h.hallucination_type] = type_counts.get(h.hallucination_type, 0) + 1

        memory_based = type_counts.get("confusion", 0) + type_counts.get("plausible", 0)
        fabrication = type_counts.get("fabrication", 0)

        return {
            "total_responses": len(analyses),
            "hallucination_rate": len(hallucinations) / len(analyses),
            "type_distribution": type_counts,
            "memory_corruption_ratio": memory_based / len(hallucinations) if hallucinations else 0,
            "fabrication_ratio": fabrication / len(hallucinations) if hallucinations else 0,
            "hypothesis_supported": memory_based > fabrication
        }
