"""Tests for task generators."""

import pytest
import tempfile
from pathlib import Path

from src.data.task_generators import (
    TaskInstance,
    SimpleArithmeticGenerator,
    RuleFollowingGenerator,
    SimplePatternGenerator,
    LiteralInstructionGenerator,
    PersonaMaintenanceGenerator,
    create_all_generators,
    generate_full_dataset,
)


class TestTaskInstance:
    """Tests for TaskInstance dataclass."""

    def test_creation(self):
        instance = TaskInstance(
            task_type="test",
            subtask="subtest",
            prompt="What is 1+1?",
            expected_answer="2",
        )
        assert instance.task_type == "test"
        assert instance.subtask == "subtest"
        assert instance.prompt == "What is 1+1?"
        assert instance.expected_answer == "2"

    def test_to_dict(self):
        instance = TaskInstance(
            task_type="test",
            subtask="subtest",
            prompt="Test prompt",
            expected_answer="answer",
            metadata={"key": "value"},
        )
        d = instance.to_dict()
        assert d["task_type"] == "test"
        assert d["metadata"]["key"] == "value"

    def test_from_dict(self):
        data = {
            "task_type": "test",
            "subtask": "sub",
            "prompt": "prompt",
            "expected_answer": "answer",
            "metadata": {},
            "difficulty": "easy",
        }
        instance = TaskInstance.from_dict(data)
        assert instance.task_type == "test"
        assert instance.difficulty == "easy"


class TestSimpleArithmeticGenerator:
    """Tests for arithmetic task generator."""

    def test_generate(self):
        generator = SimpleArithmeticGenerator(seed=42)
        instances = generator.generate(100)

        assert len(instances) == 100
        assert all(inst.task_type == "simple_arithmetic" for inst in instances)

    def test_deterministic(self):
        gen1 = SimpleArithmeticGenerator(seed=42)
        gen2 = SimpleArithmeticGenerator(seed=42)

        inst1 = gen1.generate(10)
        inst2 = gen2.generate(10)

        for i1, i2 in zip(inst1, inst2):
            assert i1.prompt == i2.prompt
            assert i1.expected_answer == i2.expected_answer

    def test_subtasks_covered(self):
        generator = SimpleArithmeticGenerator(seed=42)
        instances = generator.generate(100)

        subtasks = set(inst.subtask for inst in instances)
        expected = {
            "single_digit_addition",
            "single_digit_multiplication",
            "simple_subtraction",
            "basic_division",
        }
        assert subtasks == expected

    def test_correct_answers(self):
        generator = SimpleArithmeticGenerator(seed=42)
        instances = generator.generate(50)

        for inst in instances:
            if inst.subtask == "single_digit_addition":
                ops = inst.metadata["operands"]
                assert inst.expected_answer == str(ops[0] + ops[1])
            elif inst.subtask == "single_digit_multiplication":
                ops = inst.metadata["operands"]
                assert inst.expected_answer == str(ops[0] * ops[1])


class TestRuleFollowingGenerator:
    """Tests for rule following task generator."""

    def test_generate(self):
        generator = RuleFollowingGenerator(seed=42)
        instances = generator.generate(100)

        assert len(instances) == 100
        assert all(inst.task_type == "rule_following" for inst in instances)

    def test_reverse_string_correct(self):
        generator = RuleFollowingGenerator(seed=42)
        instances = generator.generate(100)

        reverse_instances = [i for i in instances if i.subtask == "reverse_string"]
        for inst in reverse_instances:
            original = inst.metadata["original"]
            assert inst.expected_answer == original[::-1]

    def test_count_words_correct(self):
        generator = RuleFollowingGenerator(seed=42)
        instances = generator.generate(100)

        count_instances = [i for i in instances if i.subtask == "count_words"]
        for inst in count_instances:
            assert inst.expected_answer == str(inst.metadata["word_count"])


class TestSimplePatternGenerator:
    """Tests for pattern task generator."""

    def test_generate(self):
        generator = SimplePatternGenerator(seed=42)
        instances = generator.generate(60)

        assert len(instances) == 60
        assert all(inst.task_type == "simple_patterns" for inst in instances)


class TestLiteralInstructionGenerator:
    """Tests for literal instruction task generator."""

    def test_generate(self):
        generator = LiteralInstructionGenerator(seed=42)
        instances = generator.generate(100)

        assert len(instances) == 100
        assert all(inst.task_type == "literal_instructions" for inst in instances)

    def test_do_not_explain(self):
        generator = LiteralInstructionGenerator(seed=42)
        instances = generator.generate(100)

        explain_instances = [i for i in instances if i.subtask == "do_not_explain"]
        assert len(explain_instances) > 0
        for inst in explain_instances:
            assert "ONLY" in inst.prompt or "only" in inst.prompt.lower()


class TestPersonaMaintenanceGenerator:
    """Tests for persona task generator."""

    def test_generate(self):
        generator = PersonaMaintenanceGenerator(seed=42)
        instances = generator.generate(60)

        assert len(instances) == 60
        assert all(inst.task_type == "persona_maintenance" for inst in instances)


class TestCreateAllGenerators:
    """Tests for generator factory."""

    def test_creates_all(self):
        generators = create_all_generators(seed=42)

        assert "simple_arithmetic" in generators
        assert "rule_following" in generators
        assert "simple_patterns" in generators
        assert "literal_instructions" in generators
        assert "persona_maintenance" in generators


class TestGenerateFullDataset:
    """Tests for full dataset generation."""

    def test_generate_to_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            dataset = generate_full_dataset(output_dir, samples_per_task=20, seed=42)

            # Check returned data
            assert len(dataset) == 5
            assert all(len(v) == 20 for v in dataset.values())

            # Check files created
            assert (output_dir / "dataset_info.json").exists()
            assert (output_dir / "simple_arithmetic.json").exists()
