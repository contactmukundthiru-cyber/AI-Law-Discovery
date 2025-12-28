"""
Capability task generators for emergence analysis.

Generates tasks for capabilities that have been claimed to show emergence.
"""

import random
import string
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Iterator
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class TaskSample:
    """A single task sample for evaluation."""
    task_id: str
    capability: str
    prompt: str
    correct_answer: str
    difficulty: float  # 0.0 to 1.0
    metadata: Dict[str, Any]


class BaseTaskGenerator(ABC):
    """Base class for capability task generators."""

    @abstractmethod
    def generate(self, n_samples: int, difficulty_range: Tuple[float, float] = (0.0, 1.0)) -> List[TaskSample]:
        """Generate task samples."""
        pass

    @property
    @abstractmethod
    def capability_name(self) -> str:
        """Name of the capability being tested."""
        pass


class ArithmeticTaskGenerator(BaseTaskGenerator):
    """
    Generate arithmetic tasks of varying difficulty.

    Arithmetic has been claimed to show emergence at large scales.
    """

    @property
    def capability_name(self) -> str:
        return "arithmetic"

    def generate(
        self,
        n_samples: int,
        difficulty_range: Tuple[float, float] = (0.0, 1.0)
    ) -> List[TaskSample]:
        samples = []

        for i in range(n_samples):
            difficulty = random.uniform(*difficulty_range)

            # Difficulty affects number of digits and operation complexity
            if difficulty < 0.2:
                # Single digit addition
                a, b = random.randint(1, 9), random.randint(1, 9)
                op = random.choice(['+', '-'])
            elif difficulty < 0.4:
                # Two digit operations
                a = random.randint(10, 99)
                b = random.randint(10, 99)
                op = random.choice(['+', '-', '*'])
            elif difficulty < 0.6:
                # Three digit operations
                a = random.randint(100, 999)
                b = random.randint(10, 99)
                op = random.choice(['+', '-', '*'])
            elif difficulty < 0.8:
                # Four digit operations
                a = random.randint(1000, 9999)
                b = random.randint(100, 999)
                op = random.choice(['+', '-', '*', '/'])
            else:
                # Multi-step operations
                a = random.randint(100, 999)
                b = random.randint(10, 99)
                c = random.randint(1, 9)

                if random.random() < 0.5:
                    prompt = f"What is {a} + {b} * {c}?"
                    answer = str(a + b * c)
                else:
                    prompt = f"What is ({a} + {b}) * {c}?"
                    answer = str((a + b) * c)

                samples.append(TaskSample(
                    task_id=f"arith_{i}",
                    capability="arithmetic",
                    prompt=prompt,
                    correct_answer=answer,
                    difficulty=difficulty,
                    metadata={"multi_step": True}
                ))
                continue

            # Simple two-operand case
            if op == '+':
                answer = a + b
            elif op == '-':
                answer = a - b
            elif op == '*':
                answer = a * b
            else:  # division
                answer = a  # Make it divide evenly
                a = a * b  # Now a / b = answer

            prompt = f"What is {a} {op} {b}?"

            samples.append(TaskSample(
                task_id=f"arith_{i}",
                capability="arithmetic",
                prompt=prompt,
                correct_answer=str(answer),
                difficulty=difficulty,
                metadata={"operation": op, "operands": [a, b]}
            ))

        return samples


class WordUnscramblingGenerator(BaseTaskGenerator):
    """
    Generate word unscrambling (anagram) tasks.

    This capability has been claimed to emerge at scale.
    """

    WORDS = {
        "easy": ["cat", "dog", "run", "sit", "hat", "map", "cup", "pen"],
        "medium": ["apple", "bread", "chair", "dance", "earth", "fresh", "grape"],
        "hard": ["computer", "elephant", "fountain", "grateful", "hospital"],
        "very_hard": ["algorithm", "complexity", "development", "environment"]
    }

    @property
    def capability_name(self) -> str:
        return "word_unscrambling"

    def _scramble_word(self, word: str) -> str:
        """Scramble a word ensuring it's different from original."""
        letters = list(word)
        for _ in range(10):  # Try up to 10 times
            random.shuffle(letters)
            scrambled = ''.join(letters)
            if scrambled != word:
                return scrambled
        return scrambled  # Return even if same (rare for long words)

    def generate(
        self,
        n_samples: int,
        difficulty_range: Tuple[float, float] = (0.0, 1.0)
    ) -> List[TaskSample]:
        samples = []

        for i in range(n_samples):
            difficulty = random.uniform(*difficulty_range)

            if difficulty < 0.25:
                word = random.choice(self.WORDS["easy"])
            elif difficulty < 0.5:
                word = random.choice(self.WORDS["medium"])
            elif difficulty < 0.75:
                word = random.choice(self.WORDS["hard"])
            else:
                word = random.choice(self.WORDS["very_hard"])

            scrambled = self._scramble_word(word)

            prompt = f"Unscramble this word: {scrambled}"

            samples.append(TaskSample(
                task_id=f"unscramble_{i}",
                capability="word_unscrambling",
                prompt=prompt,
                correct_answer=word,
                difficulty=difficulty,
                metadata={"original": word, "scrambled": scrambled}
            ))

        return samples


class ChainOfThoughtGenerator(BaseTaskGenerator):
    """
    Generate tasks requiring chain-of-thought reasoning.
    """

    @property
    def capability_name(self) -> str:
        return "chain_of_thought"

    def generate(
        self,
        n_samples: int,
        difficulty_range: Tuple[float, float] = (0.0, 1.0)
    ) -> List[TaskSample]:
        samples = []

        for i in range(n_samples):
            difficulty = random.uniform(*difficulty_range)

            if difficulty < 0.33:
                # Simple two-step reasoning
                steps = 2
            elif difficulty < 0.66:
                steps = 3
            else:
                steps = 4

            # Generate a multi-step word problem
            sample = self._generate_word_problem(i, steps, difficulty)
            samples.append(sample)

        return samples

    def _generate_word_problem(
        self,
        idx: int,
        steps: int,
        difficulty: float
    ) -> TaskSample:
        """Generate a multi-step word problem."""
        names = ["Alice", "Bob", "Carol", "David", "Eve"]
        items = ["apples", "books", "cookies", "dollars", "eggs"]

        name = random.choice(names)
        item = random.choice(items)

        initial = random.randint(5, 20)

        operations = []
        current = initial
        prompt_parts = [f"{name} has {initial} {item}."]

        for step in range(steps):
            op = random.choice(["add", "subtract", "multiply"])

            if op == "add":
                amount = random.randint(1, 10)
                current += amount
                prompt_parts.append(f"Then gets {amount} more.")
                operations.append(f"+{amount}")
            elif op == "subtract":
                amount = random.randint(1, min(5, current - 1))
                current -= amount
                prompt_parts.append(f"Then gives away {amount}.")
                operations.append(f"-{amount}")
            else:
                factor = random.choice([2, 3])
                current *= factor
                prompt_parts.append(f"Then the number {['doubles', 'triples'][factor-2]}.")
                operations.append(f"*{factor}")

        prompt = " ".join(prompt_parts) + f" How many {item} does {name} have now?"

        return TaskSample(
            task_id=f"cot_{idx}",
            capability="chain_of_thought",
            prompt=prompt,
            correct_answer=str(current),
            difficulty=difficulty,
            metadata={
                "steps": steps,
                "operations": operations,
                "initial": initial,
                "final": current
            }
        )


class TruthfulnessGenerator(BaseTaskGenerator):
    """
    Generate truthfulness evaluation tasks.

    Tests whether models prefer true statements over plausible falsehoods.
    """

    # Pairs of (true statement, plausible false statement)
    TRUTH_PAIRS = [
        ("The Earth orbits the Sun.", "The Sun orbits the Earth."),
        ("Water freezes at 0 degrees Celsius.", "Water freezes at 10 degrees Celsius."),
        ("Humans have 23 pairs of chromosomes.", "Humans have 24 pairs of chromosomes."),
        ("Light travels faster than sound.", "Sound travels faster than light."),
        ("The Great Wall of China is visible from space with aid.",
         "The Great Wall of China is easily visible from the Moon."),
        ("Goldfish have a memory span of months.", "Goldfish have a 3-second memory."),
        ("Bats are not blind.", "Bats are completely blind."),
        ("Lightning can strike the same place twice.",
         "Lightning never strikes the same place twice."),
    ]

    @property
    def capability_name(self) -> str:
        return "truthfulness"

    def generate(
        self,
        n_samples: int,
        difficulty_range: Tuple[float, float] = (0.0, 1.0)
    ) -> List[TaskSample]:
        samples = []

        for i in range(n_samples):
            difficulty = random.uniform(*difficulty_range)

            # Select a truth pair
            true_stmt, false_stmt = random.choice(self.TRUTH_PAIRS)

            # Randomly assign which is presented
            if random.random() < 0.5:
                statement = true_stmt
                is_true = True
            else:
                statement = false_stmt
                is_true = False

            prompt = f"Is the following statement true or false?\n\nStatement: {statement}\n\nAnswer:"

            samples.append(TaskSample(
                task_id=f"truth_{i}",
                capability="truthfulness",
                prompt=prompt,
                correct_answer="True" if is_true else "False",
                difficulty=difficulty,
                metadata={
                    "statement": statement,
                    "is_true": is_true,
                    "true_version": true_stmt,
                    "false_version": false_stmt
                }
            ))

        return samples


class CapabilityTaskGenerator:
    """
    Main interface for generating capability evaluation tasks.
    """

    def __init__(self):
        self.generators = {
            "arithmetic": ArithmeticTaskGenerator(),
            "word_unscrambling": WordUnscramblingGenerator(),
            "chain_of_thought": ChainOfThoughtGenerator(),
            "truthfulness": TruthfulnessGenerator(),
        }

    def generate_tasks(
        self,
        capability: str,
        n_samples: int,
        difficulty_range: Tuple[float, float] = (0.0, 1.0)
    ) -> List[TaskSample]:
        """Generate tasks for a specific capability."""
        if capability not in self.generators:
            raise ValueError(f"Unknown capability: {capability}. "
                           f"Available: {list(self.generators.keys())}")

        return self.generators[capability].generate(n_samples, difficulty_range)

    def generate_difficulty_sweep(
        self,
        capability: str,
        n_per_difficulty: int = 50,
        n_difficulty_levels: int = 10
    ) -> List[TaskSample]:
        """Generate tasks across full difficulty range."""
        all_samples = []

        for i in range(n_difficulty_levels):
            low = i / n_difficulty_levels
            high = (i + 1) / n_difficulty_levels
            samples = self.generate_tasks(
                capability, n_per_difficulty, (low, high)
            )
            all_samples.extend(samples)

        return all_samples

    @property
    def available_capabilities(self) -> List[str]:
        """List available capabilities."""
        return list(self.generators.keys())
