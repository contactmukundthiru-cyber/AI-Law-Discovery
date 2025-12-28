"""
Task generators for inverse scaling experiments.

Each generator creates tasks designed to potentially exhibit inverse scaling,
where larger models may actually perform worse than smaller ones.
"""

import random
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path


@dataclass
class TaskInstance:
    """A single task instance for evaluation."""
    task_type: str
    subtask: str
    prompt: str
    expected_answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    difficulty: str = "standard"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "subtask": self.subtask,
            "prompt": self.prompt,
            "expected_answer": self.expected_answer,
            "metadata": self.metadata,
            "difficulty": self.difficulty,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskInstance":
        return cls(**data)


class TaskGenerator(ABC):
    """Abstract base class for task generators."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

    @abstractmethod
    def generate(self, num_samples: int) -> List[TaskInstance]:
        """Generate task instances."""
        pass

    @property
    @abstractmethod
    def task_type(self) -> str:
        """Return the task type identifier."""
        pass

    def save(self, instances: List[TaskInstance], path: Path) -> None:
        """Save generated instances to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([inst.to_dict() for inst in instances], f, indent=2)

    def load(self, path: Path) -> List[TaskInstance]:
        """Load instances from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        return [TaskInstance.from_dict(d) for d in data]


class SimpleArithmeticGenerator(TaskGenerator):
    """
    Generate simple arithmetic problems.

    Hypothesis: Larger models may overthink these simple problems,
    adding unnecessary complexity or second-guessing obvious answers.
    """

    @property
    def task_type(self) -> str:
        return "simple_arithmetic"

    def generate(self, num_samples: int) -> List[TaskInstance]:
        instances = []
        samples_per_subtask = num_samples // 4

        # Single digit addition
        for _ in range(samples_per_subtask):
            a, b = self.rng.randint(1, 9), self.rng.randint(1, 9)
            instances.append(TaskInstance(
                task_type=self.task_type,
                subtask="single_digit_addition",
                prompt=f"What is {a} + {b}? Reply with only the number.",
                expected_answer=str(a + b),
                metadata={"operands": [a, b], "operation": "+"},
            ))

        # Single digit multiplication
        for _ in range(samples_per_subtask):
            a, b = self.rng.randint(1, 9), self.rng.randint(1, 9)
            instances.append(TaskInstance(
                task_type=self.task_type,
                subtask="single_digit_multiplication",
                prompt=f"What is {a} × {b}? Reply with only the number.",
                expected_answer=str(a * b),
                metadata={"operands": [a, b], "operation": "*"},
            ))

        # Simple subtraction (always positive result)
        for _ in range(samples_per_subtask):
            a = self.rng.randint(5, 20)
            b = self.rng.randint(1, a)
            instances.append(TaskInstance(
                task_type=self.task_type,
                subtask="simple_subtraction",
                prompt=f"What is {a} - {b}? Reply with only the number.",
                expected_answer=str(a - b),
                metadata={"operands": [a, b], "operation": "-"},
            ))

        # Basic division (clean results)
        for _ in range(samples_per_subtask):
            b = self.rng.randint(2, 9)
            result = self.rng.randint(1, 10)
            a = b * result
            instances.append(TaskInstance(
                task_type=self.task_type,
                subtask="basic_division",
                prompt=f"What is {a} ÷ {b}? Reply with only the number.",
                expected_answer=str(result),
                metadata={"operands": [a, b], "operation": "/"},
            ))

        self.rng.shuffle(instances)
        return instances


class RuleFollowingGenerator(TaskGenerator):
    """
    Generate tasks requiring strict rule following.

    Hypothesis: Larger models may be more prone to "helpful" elaboration
    that violates explicit simple rules.
    """

    @property
    def task_type(self) -> str:
        return "rule_following"

    def generate(self, num_samples: int) -> List[TaskInstance]:
        instances = []
        samples_per_subtask = num_samples // 4

        # Repeat exactly
        words = ["hello", "world", "test", "apple", "blue", "seven", "quick", "jump"]
        phrases = [
            "The cat sat.",
            "Red ball.",
            "Go now.",
            "Yes please.",
            "Stop here.",
            "Three words only.",
        ]

        for _ in range(samples_per_subtask):
            if self.rng.random() < 0.5:
                text = self.rng.choice(words)
            else:
                text = self.rng.choice(phrases)
            instances.append(TaskInstance(
                task_type=self.task_type,
                subtask="repeat_exactly",
                prompt=f"Repeat the following text exactly, with no additions: {text}",
                expected_answer=text,
                metadata={"original_text": text},
            ))

        # Count words
        for _ in range(samples_per_subtask):
            num_words = self.rng.randint(3, 12)
            sentence_words = self.rng.sample(
                ["the", "a", "is", "are", "was", "big", "small", "red", "blue",
                 "cat", "dog", "bird", "tree", "house", "car", "book", "pen",
                 "happy", "sad", "fast", "slow", "good", "bad", "new", "old"],
                num_words
            )
            sentence = " ".join(sentence_words)
            instances.append(TaskInstance(
                task_type=self.task_type,
                subtask="count_words",
                prompt=f"Count the number of words in this sentence and reply with only the number: {sentence}",
                expected_answer=str(num_words),
                metadata={"sentence": sentence, "word_count": num_words},
            ))

        # Reverse string
        for _ in range(samples_per_subtask):
            length = self.rng.randint(4, 8)
            original = "".join(self.rng.choices(string.ascii_lowercase, k=length))
            instances.append(TaskInstance(
                task_type=self.task_type,
                subtask="reverse_string",
                prompt=f"Reverse the following string and reply with only the reversed string: {original}",
                expected_answer=original[::-1],
                metadata={"original": original},
            ))

        # Follow format
        formats = [
            ("NAME: {name}, AGE: {age}", {"name": "John", "age": "25"}),
            ("ITEM: {item} | QTY: {qty}", {"item": "Apple", "qty": "3"}),
            ("[{code}] - {status}", {"code": "A1", "status": "OK"}),
            ("{{ID={id}}}", {"id": "12345"}),
        ]

        for _ in range(samples_per_subtask):
            template, values = self.rng.choice(formats)
            expected = template.format(**values)
            instances.append(TaskInstance(
                task_type=self.task_type,
                subtask="follow_format",
                prompt=f"Fill in this exact format with the given values. Output only the filled format, nothing else.\nFormat: {template}\nValues: {values}",
                expected_answer=expected,
                metadata={"template": template, "values": values},
            ))

        self.rng.shuffle(instances)
        return instances


class SimplePatternGenerator(TaskGenerator):
    """
    Generate simple pattern completion tasks.

    Hypothesis: Larger models may find more complex patterns where
    simple ones exist, leading to incorrect completions.
    """

    @property
    def task_type(self) -> str:
        return "simple_patterns"

    def generate(self, num_samples: int) -> List[TaskInstance]:
        instances = []
        samples_per_subtask = num_samples // 3

        # Numeric sequences (simple)
        for _ in range(samples_per_subtask):
            pattern_type = self.rng.choice(["increment", "constant", "double"])

            if pattern_type == "increment":
                start = self.rng.randint(1, 10)
                step = self.rng.randint(1, 3)
                sequence = [start + i * step for i in range(5)]
                next_val = start + 5 * step
            elif pattern_type == "constant":
                val = self.rng.randint(1, 20)
                sequence = [val] * 5
                next_val = val
            else:  # double
                start = self.rng.randint(1, 5)
                sequence = [start * (2 ** i) for i in range(5)]
                next_val = start * (2 ** 5)

            seq_str = ", ".join(map(str, sequence))
            instances.append(TaskInstance(
                task_type=self.task_type,
                subtask="numeric_sequences",
                prompt=f"What comes next in this sequence? Reply with only the number.\n{seq_str}, ?",
                expected_answer=str(next_val),
                metadata={"sequence": sequence, "pattern": pattern_type},
            ))

        # Alphabetic patterns
        for _ in range(samples_per_subtask):
            pattern_type = self.rng.choice(["consecutive", "skip_one", "repeat"])

            if pattern_type == "consecutive":
                start_idx = self.rng.randint(0, 20)
                letters = [chr(ord('A') + start_idx + i) for i in range(5)]
                next_letter = chr(ord('A') + start_idx + 5)
            elif pattern_type == "skip_one":
                start_idx = self.rng.randint(0, 15)
                letters = [chr(ord('A') + start_idx + i * 2) for i in range(5)]
                next_letter = chr(ord('A') + start_idx + 10)
            else:  # repeat
                letter = chr(ord('A') + self.rng.randint(0, 25))
                letters = [letter] * 5
                next_letter = letter

            seq_str = ", ".join(letters)
            instances.append(TaskInstance(
                task_type=self.task_type,
                subtask="alphabetic_patterns",
                prompt=f"What comes next in this sequence? Reply with only the letter.\n{seq_str}, ?",
                expected_answer=next_letter,
                metadata={"sequence": letters, "pattern": pattern_type},
            ))

        # Repetition patterns
        for _ in range(samples_per_subtask):
            base = self.rng.choice(["AB", "XY", "12", "##"])
            repetitions = self.rng.randint(3, 5)
            sequence = base * repetitions
            next_part = base

            instances.append(TaskInstance(
                task_type=self.task_type,
                subtask="repetition_patterns",
                prompt=f"This pattern repeats. What comes next? Reply with only the next segment.\nPattern: {sequence}...\nNext segment: ",
                expected_answer=next_part,
                metadata={"base": base, "shown_repetitions": repetitions},
            ))

        self.rng.shuffle(instances)
        return instances


class LiteralInstructionGenerator(TaskGenerator):
    """
    Generate tasks requiring literal interpretation.

    Hypothesis: Larger models may be more prone to "helpful" additions
    or interpretations that violate literal instructions.
    """

    @property
    def task_type(self) -> str:
        return "literal_instructions"

    def generate(self, num_samples: int) -> List[TaskInstance]:
        instances = []
        samples_per_subtask = num_samples // 4

        # Do not explain
        questions = [
            ("What is 2+2?", "4"),
            ("What color is the sky?", "blue"),
            ("How many legs does a dog have?", "4"),
            ("What is the capital of France?", "Paris"),
            ("What comes after Monday?", "Tuesday"),
            ("Is water wet?", "yes"),
            ("What is 10-3?", "7"),
            ("How many days in a week?", "7"),
        ]

        for _ in range(samples_per_subtask):
            q, a = self.rng.choice(questions)
            instances.append(TaskInstance(
                task_type=self.task_type,
                subtask="do_not_explain",
                prompt=f"{q}\n\nIMPORTANT: Reply with ONLY the answer. Do not explain, elaborate, or add anything else.",
                expected_answer=a,
                metadata={"question": q},
            ))

        # Respond with single word
        prompts = [
            ("Describe the weather today in one word.", ["sunny", "cloudy", "rainy", "clear", "warm", "cold"]),
            ("Name a color.", ["red", "blue", "green", "yellow", "orange", "purple", "black", "white"]),
            ("Name a fruit.", ["apple", "banana", "orange", "grape", "mango", "pear"]),
            ("Say yes or no: Is 5 greater than 3?", ["yes"]),
            ("Name a day of the week.", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
        ]

        for _ in range(samples_per_subtask):
            prompt, valid_answers = self.rng.choice(prompts)
            instances.append(TaskInstance(
                task_type=self.task_type,
                subtask="respond_with_single_word",
                prompt=f"{prompt}\n\nRespond with exactly ONE word. Nothing more.",
                expected_answer="|".join(valid_answers),  # Multiple valid answers
                metadata={"valid_answers": valid_answers},
            ))

        # Ignore context (distractor resistance)
        for _ in range(samples_per_subtask):
            target = self.rng.randint(1, 100)
            distractors = [
                f"Some people say the answer is {target + self.rng.randint(1, 10)}.",
                f"A common mistake is to answer {target - self.rng.randint(1, 10)}.",
                f"This reminds me of the number {self.rng.randint(1, 1000)}.",
            ]
            distractor = self.rng.choice(distractors)

            instances.append(TaskInstance(
                task_type=self.task_type,
                subtask="ignore_context",
                prompt=f"{distractor}\n\nIgnore everything above. Simply reply with the number: {target}",
                expected_answer=str(target),
                metadata={"target": target, "distractor": distractor},
            ))

        # Exact format output
        for _ in range(samples_per_subtask):
            format_type = self.rng.choice(["json", "csv", "brackets"])

            if format_type == "json":
                key = self.rng.choice(["name", "value", "result"])
                val = self.rng.choice(["test", "data", "output"])
                expected = f'{{{{"{ key}": "{val}"}}}}'
                instances.append(TaskInstance(
                    task_type=self.task_type,
                    subtask="exact_format_output",
                    prompt=f'Output exactly this JSON, nothing else: {{"{key}": "{val}"}}',
                    expected_answer=f'{{"{key}": "{val}"}}',
                    metadata={"format": "json"},
                ))
            elif format_type == "csv":
                vals = [str(self.rng.randint(1, 10)) for _ in range(3)]
                expected = ",".join(vals)
                instances.append(TaskInstance(
                    task_type=self.task_type,
                    subtask="exact_format_output",
                    prompt=f"Output these values as CSV (comma-separated, no spaces): {', '.join(vals)}",
                    expected_answer=expected,
                    metadata={"format": "csv", "values": vals},
                ))
            else:  # brackets
                content = self.rng.choice(["DONE", "OK", "YES", "NO"])
                expected = f"[{content}]"
                instances.append(TaskInstance(
                    task_type=self.task_type,
                    subtask="exact_format_output",
                    prompt=f"Output this exactly: [{content}]",
                    expected_answer=expected,
                    metadata={"format": "brackets", "content": content},
                ))

        self.rng.shuffle(instances)
        return instances


class PersonaMaintenanceGenerator(TaskGenerator):
    """
    Generate persona consistency tasks.

    Hypothesis: Larger models may have more difficulty maintaining
    simple, restricted personas due to their broader capabilities
    "leaking through."
    """

    @property
    def task_type(self) -> str:
        return "persona_maintenance"

    def generate(self, num_samples: int) -> List[TaskInstance]:
        instances = []
        samples_per_subtask = num_samples // 3

        # Single trait persona
        personas = [
            ("You are extremely pessimistic. Everything is bad.", "negative", ["bad", "terrible", "awful", "horrible", "worst", "never", "can't", "won't"]),
            ("You are extremely optimistic. Everything is great.", "positive", ["great", "wonderful", "amazing", "fantastic", "best", "always", "will", "can"]),
            ("You only speak in questions.", "questions", ["?"]),
            ("You are very brief. Never use more than 5 words.", "brief", None),  # Check word count
        ]

        questions = [
            "How is the weather today?",
            "What do you think about technology?",
            "Tell me about your day.",
            "What's your favorite food?",
            "How do you feel right now?",
        ]

        for _ in range(samples_per_subtask):
            persona_desc, trait, markers = self.rng.choice(personas)
            question = self.rng.choice(questions)

            instances.append(TaskInstance(
                task_type=self.task_type,
                subtask="single_trait_persona",
                prompt=f"PERSONA: {persona_desc}\n\nRespond to this IN CHARACTER: {question}",
                expected_answer=f"TRAIT:{trait}",  # Special marker for evaluation
                metadata={"persona": persona_desc, "trait": trait, "markers": markers},
            ))

        # Restricted vocabulary
        word_sets = [
            (["yes", "no", "maybe"], "You can ONLY use the words: yes, no, maybe"),
            (["good", "bad"], "You can ONLY respond with either 'good' or 'bad'"),
            (["one", "two", "three"], "You may only use the words: one, two, three"),
        ]

        for _ in range(samples_per_subtask):
            allowed_words, instruction = self.rng.choice(word_sets)
            question = self.rng.choice(questions)

            instances.append(TaskInstance(
                task_type=self.task_type,
                subtask="restricted_vocabulary",
                prompt=f"{instruction}\n\nQuestion: {question}",
                expected_answer=f"VOCAB:{','.join(allowed_words)}",
                metadata={"allowed_words": allowed_words},
            ))

        # Consistent style
        styles = [
            ("ALL CAPS", lambda s: s.upper()),
            ("all lowercase", lambda s: s.lower()),
            ("Every Word Capitalized", lambda s: s.title()),
        ]

        for _ in range(samples_per_subtask):
            style_name, _ = self.rng.choice(styles)
            question = self.rng.choice(questions)

            instances.append(TaskInstance(
                task_type=self.task_type,
                subtask="consistent_style",
                prompt=f"Respond to everything in {style_name}.\n\nQuestion: {question}",
                expected_answer=f"STYLE:{style_name}",
                metadata={"style": style_name},
            ))

        self.rng.shuffle(instances)
        return instances


def create_all_generators(seed: int = 42) -> Dict[str, TaskGenerator]:
    """Create all task generators with the same seed."""
    return {
        "simple_arithmetic": SimpleArithmeticGenerator(seed),
        "rule_following": RuleFollowingGenerator(seed),
        "simple_patterns": SimplePatternGenerator(seed),
        "literal_instructions": LiteralInstructionGenerator(seed),
        "persona_maintenance": PersonaMaintenanceGenerator(seed),
    }


def generate_full_dataset(
    output_dir: Path,
    samples_per_task: int = 500,
    seed: int = 42
) -> Dict[str, List[TaskInstance]]:
    """Generate the complete dataset for all tasks."""
    generators = create_all_generators(seed)
    dataset = {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for task_name, generator in generators.items():
        instances = generator.generate(samples_per_task)
        dataset[task_name] = instances
        generator.save(instances, output_dir / f"{task_name}.json")

    # Save combined dataset info
    info = {
        "tasks": list(dataset.keys()),
        "samples_per_task": samples_per_task,
        "total_samples": sum(len(v) for v in dataset.values()),
        "seed": seed,
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    return dataset
