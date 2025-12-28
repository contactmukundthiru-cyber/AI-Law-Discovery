"""
Physics task generators for probing physical understanding in LLMs.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
from pathlib import Path


@dataclass
class PhysicsInstance:
    """A single physics task instance."""
    task_type: str
    text: str
    label: Any  # Can be int (classification) or float (regression)
    physical_variables: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict:
        return {
            "task_type": self.task_type,
            "text": self.text,
            "label": self.label,
            "physical_variables": self.physical_variables,
            "metadata": self.metadata,
        }


class PhysicsTaskGenerator(ABC):
    """Abstract base class for physics task generators."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    @abstractmethod
    def generate(self, num_samples: int) -> List[PhysicsInstance]:
        """Generate physics task instances."""
        pass

    @property
    @abstractmethod
    def task_type(self) -> str:
        """Return task type identifier."""
        pass

    def save(self, instances: List[PhysicsInstance], path: Path) -> None:
        """Save instances to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([inst.to_dict() for inst in instances], f, indent=2)


class SpatialReasoningTask(PhysicsTaskGenerator):
    """
    Tasks probing spatial understanding.

    Tests whether models encode spatial relationships like:
    - Relative positions (left/right, above/below)
    - Distances (near/far)
    - Containment (inside/outside)
    """

    @property
    def task_type(self) -> str:
        return "spatial_reasoning"

    def generate(self, num_samples: int) -> List[PhysicsInstance]:
        instances = []
        samples_per_type = num_samples // 4

        # Relative position tasks
        objects = ["ball", "box", "cup", "book", "phone", "lamp", "chair", "table"]
        relations = [
            ("left of", "right of", 0, 1),
            ("right of", "left of", 1, 0),
            ("above", "below", 2, 3),
            ("below", "above", 3, 2),
        ]

        for _ in range(samples_per_type):
            obj1, obj2 = self.rng.sample(objects, 2)
            rel, inv_rel, label, inv_label = self.rng.choice(relations)

            text = f"The {obj1} is {rel} the {obj2}."
            instances.append(PhysicsInstance(
                task_type=self.task_type,
                text=text,
                label=label,
                physical_variables={"relation": rel, "obj1": obj1, "obj2": obj2},
                metadata={"subtask": "relative_position"},
            ))

        # Distance tasks
        for _ in range(samples_per_type):
            obj1, obj2 = self.rng.sample(objects, 2)
            distance = self.rng.choice(["very close to", "near", "far from", "very far from"])
            distance_value = {"very close to": 0, "near": 1, "far from": 2, "very far from": 3}[distance]

            text = f"The {obj1} is {distance} the {obj2}."
            instances.append(PhysicsInstance(
                task_type=self.task_type,
                text=text,
                label=distance_value,
                physical_variables={"distance": distance, "distance_value": distance_value},
                metadata={"subtask": "distance"},
            ))

        # Containment tasks
        containers = ["box", "room", "bag", "house", "car", "drawer"]
        for _ in range(samples_per_type):
            obj = self.rng.choice(objects)
            container = self.rng.choice(containers)
            inside = self.rng.choice([True, False])

            if inside:
                text = f"The {obj} is inside the {container}."
                label = 1
            else:
                text = f"The {obj} is outside the {container}."
                label = 0

            instances.append(PhysicsInstance(
                task_type=self.task_type,
                text=text,
                label=label,
                physical_variables={"inside": inside, "container": container},
                metadata={"subtask": "containment"},
            ))

        # Motion direction tasks
        directions = ["north", "south", "east", "west", "up", "down"]
        for _ in range(samples_per_type):
            obj = self.rng.choice(objects)
            direction = self.rng.choice(directions)
            dir_label = directions.index(direction)

            text = f"The {obj} is moving {direction}."
            instances.append(PhysicsInstance(
                task_type=self.task_type,
                text=text,
                label=dir_label,
                physical_variables={"direction": direction},
                metadata={"subtask": "motion_direction"},
            ))

        self.rng.shuffle(instances)
        return instances


class TemporalReasoningTask(PhysicsTaskGenerator):
    """
    Tasks probing temporal understanding.

    Tests whether models encode temporal relationships:
    - Ordering (before/after)
    - Duration (short/long)
    - Simultaneity
    """

    @property
    def task_type(self) -> str:
        return "temporal_reasoning"

    def generate(self, num_samples: int) -> List[PhysicsInstance]:
        instances = []
        samples_per_type = num_samples // 3

        events = [
            "the meeting started",
            "the rain began",
            "the bell rang",
            "the sun set",
            "the train arrived",
            "the movie ended",
            "the phone rang",
            "the alarm went off",
        ]

        # Temporal ordering
        for _ in range(samples_per_type):
            event1, event2 = self.rng.sample(events, 2)
            before = self.rng.choice([True, False])

            if before:
                text = f"First, {event1}. Then, {event2}."
                label = 0  # event1 before event2
            else:
                text = f"After {event2}, {event1}."
                label = 1  # event1 after event2

            instances.append(PhysicsInstance(
                task_type=self.task_type,
                text=text,
                label=label,
                physical_variables={"event1": event1, "event2": event2, "order": "before" if before else "after"},
                metadata={"subtask": "temporal_order"},
            ))

        # Duration
        durations = [
            ("a few seconds", 0),
            ("a minute", 1),
            ("an hour", 2),
            ("a day", 3),
            ("a week", 4),
            ("a month", 5),
        ]

        activities = [
            "the task took",
            "we waited for",
            "the process lasted",
            "the journey took",
        ]

        for _ in range(samples_per_type):
            activity = self.rng.choice(activities)
            duration, label = self.rng.choice(durations)

            text = f"{activity.capitalize()} {duration}."
            instances.append(PhysicsInstance(
                task_type=self.task_type,
                text=text,
                label=label,
                physical_variables={"duration": duration, "duration_rank": label},
                metadata={"subtask": "duration"},
            ))

        # Simultaneity
        for _ in range(samples_per_type):
            event1, event2 = self.rng.sample(events, 2)
            simultaneous = self.rng.choice([True, False])

            if simultaneous:
                text = f"While {event1}, {event2}."
                label = 1
            else:
                text = f"Long after {event1}, {event2}."
                label = 0

            instances.append(PhysicsInstance(
                task_type=self.task_type,
                text=text,
                label=label,
                physical_variables={"simultaneous": simultaneous},
                metadata={"subtask": "simultaneity"},
            ))

        self.rng.shuffle(instances)
        return instances


class CausalReasoningTask(PhysicsTaskGenerator):
    """
    Tasks probing causal understanding.

    Tests whether models encode causal relationships:
    - Cause and effect
    - Enablement/prevention
    - Counterfactuals
    """

    @property
    def task_type(self) -> str:
        return "causal_reasoning"

    def generate(self, num_samples: int) -> List[PhysicsInstance]:
        instances = []
        samples_per_type = num_samples // 3

        # Cause-effect pairs
        causal_pairs = [
            ("The glass fell", "it broke"),
            ("It rained heavily", "the streets flooded"),
            ("The fire spread", "the building burned down"),
            ("The temperature dropped", "the water froze"),
            ("The battery died", "the phone turned off"),
            ("The wind blew", "the leaves scattered"),
            ("The sun rose", "the room got brighter"),
            ("The ice melted", "a puddle formed"),
        ]

        # Causal direction
        for _ in range(samples_per_type):
            cause, effect = self.rng.choice(causal_pairs)
            forward = self.rng.choice([True, False])

            if forward:
                text = f"{cause}, so {effect}."
                label = 0  # cause -> effect
            else:
                text = f"{effect.capitalize()} because {cause.lower()}."
                label = 1  # effect <- cause

            instances.append(PhysicsInstance(
                task_type=self.task_type,
                text=text,
                label=label,
                physical_variables={"cause": cause, "effect": effect, "direction": "forward" if forward else "backward"},
                metadata={"subtask": "causal_direction"},
            ))

        # Enablement vs prevention
        for _ in range(samples_per_type):
            cause, effect = self.rng.choice(causal_pairs)
            enables = self.rng.choice([True, False])

            if enables:
                text = f"{cause}, which enabled {effect.replace('the ', 'the eventual ')}."
                label = 1
            else:
                text = f"Despite {cause.lower()}, {effect} did not occur."
                label = 0

            instances.append(PhysicsInstance(
                task_type=self.task_type,
                text=text,
                label=label,
                physical_variables={"enables": enables},
                metadata={"subtask": "enablement"},
            ))

        # Physical necessity
        for _ in range(samples_per_type):
            cause, effect = self.rng.choice(causal_pairs)
            necessary = self.rng.choice([True, False])

            if necessary:
                text = f"For {effect} to happen, {cause.lower()} was necessary."
                label = 1
            else:
                text = f"{effect.capitalize()} could happen even without {cause.lower()}."
                label = 0

            instances.append(PhysicsInstance(
                task_type=self.task_type,
                text=text,
                label=label,
                physical_variables={"necessary": necessary},
                metadata={"subtask": "necessity"},
            ))

        self.rng.shuffle(instances)
        return instances


class TrajectoryTask(PhysicsTaskGenerator):
    """
    Tasks probing understanding of physical trajectories.

    Tests whether models can predict object motion.
    """

    @property
    def task_type(self) -> str:
        return "trajectory"

    def generate(self, num_samples: int) -> List[PhysicsInstance]:
        instances = []

        objects = ["ball", "rock", "arrow", "car", "bird", "plane"]

        for _ in range(num_samples):
            obj = self.rng.choice(objects)

            # Generate trajectory description
            trajectory_type = self.rng.choice(["parabolic", "straight", "circular", "falling"])

            if trajectory_type == "parabolic":
                text = f"The {obj} was thrown upward at an angle. It rose, then fell."
                label = 0
            elif trajectory_type == "straight":
                text = f"The {obj} moved in a straight line at constant speed."
                label = 1
            elif trajectory_type == "circular":
                text = f"The {obj} moved in a circle around the center."
                label = 2
            else:  # falling
                text = f"The {obj} was dropped and fell straight down, accelerating."
                label = 3

            instances.append(PhysicsInstance(
                task_type=self.task_type,
                text=text,
                label=label,
                physical_variables={"trajectory_type": trajectory_type, "object": obj},
                metadata={"subtask": "trajectory_classification"},
            ))

        self.rng.shuffle(instances)
        return instances


def generate_all_physics_tasks(
    output_dir: Path,
    samples_per_task: int = 1000,
    seed: int = 42,
) -> Dict[str, List[PhysicsInstance]]:
    """Generate all physics task datasets."""
    generators = {
        "spatial": SpatialReasoningTask(seed),
        "temporal": TemporalReasoningTask(seed),
        "causal": CausalReasoningTask(seed),
        "trajectory": TrajectoryTask(seed),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = {}
    for name, generator in generators.items():
        instances = generator.generate(samples_per_task)
        dataset[name] = instances
        generator.save(instances, output_dir / f"{name}.json")

    return dataset
