"""
Data loader for inverse scaling experiments.

Provides unified interface for loading task datasets and managing
experiment data across different task types.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Union
from dataclasses import dataclass, field
import random
from concurrent.futures import ThreadPoolExecutor
import hashlib

from .task_generators import TaskInstance


@dataclass
class TaskDataset:
    """Container for a task dataset with metadata."""
    name: str
    instances: List[TaskInstance]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.instances)

    def __iter__(self) -> Iterator[TaskInstance]:
        return iter(self.instances)

    def __getitem__(self, idx: int) -> TaskInstance:
        return self.instances[idx]

    def sample(self, n: int, seed: Optional[int] = None) -> List[TaskInstance]:
        """Sample n instances from the dataset."""
        rng = random.Random(seed)
        return rng.sample(self.instances, min(n, len(self.instances)))

    def filter_by_subtask(self, subtask: str) -> "TaskDataset":
        """Filter dataset by subtask type."""
        filtered = [inst for inst in self.instances if inst.subtask == subtask]
        return TaskDataset(
            name=f"{self.name}_{subtask}",
            instances=filtered,
            metadata={**self.metadata, "filtered_subtask": subtask},
        )

    def get_subtasks(self) -> List[str]:
        """Get list of unique subtasks in this dataset."""
        return list(set(inst.subtask for inst in self.instances))

    @property
    def checksum(self) -> str:
        """Generate a checksum for dataset integrity verification."""
        content = json.dumps([inst.to_dict() for inst in self.instances], sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()


class DataLoader:
    """
    Unified data loader for inverse scaling experiments.

    Handles loading, caching, and management of task datasets.
    """

    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize data loader.

        Args:
            data_dir: Directory containing task data files
        """
        self.data_dir = Path(data_dir)
        self._cache: Dict[str, TaskDataset] = {}

    def load_task(self, task_name: str, use_cache: bool = True) -> TaskDataset:
        """
        Load a specific task dataset.

        Args:
            task_name: Name of the task to load
            use_cache: Whether to use cached data if available

        Returns:
            TaskDataset containing the loaded instances
        """
        if use_cache and task_name in self._cache:
            return self._cache[task_name]

        task_file = self.data_dir / f"{task_name}.json"

        if not task_file.exists():
            raise FileNotFoundError(f"Task file not found: {task_file}")

        with open(task_file, "r") as f:
            data = json.load(f)

        instances = [TaskInstance.from_dict(d) for d in data]
        dataset = TaskDataset(
            name=task_name,
            instances=instances,
            metadata={"source_file": str(task_file)},
        )

        if use_cache:
            self._cache[task_name] = dataset

        return dataset

    def load_all_tasks(self, use_cache: bool = True) -> Dict[str, TaskDataset]:
        """
        Load all available task datasets.

        Args:
            use_cache: Whether to use cached data

        Returns:
            Dictionary mapping task names to datasets
        """
        datasets = {}
        task_files = list(self.data_dir.glob("*.json"))

        for task_file in task_files:
            if task_file.name == "dataset_info.json":
                continue

            task_name = task_file.stem
            try:
                datasets[task_name] = self.load_task(task_name, use_cache)
            except Exception as e:
                print(f"Warning: Failed to load {task_name}: {e}")

        return datasets

    def get_available_tasks(self) -> List[str]:
        """Get list of available task names."""
        task_files = list(self.data_dir.glob("*.json"))
        return [f.stem for f in task_files if f.name != "dataset_info.json"]

    def get_dataset_info(self) -> Optional[Dict[str, Any]]:
        """Load dataset info if available."""
        info_file = self.data_dir / "dataset_info.json"
        if info_file.exists():
            with open(info_file, "r") as f:
                return json.load(f)
        return None

    def create_evaluation_batch(
        self,
        task_names: Optional[List[str]] = None,
        samples_per_task: int = 100,
        seed: int = 42,
    ) -> List[TaskInstance]:
        """
        Create a balanced evaluation batch across tasks.

        Args:
            task_names: Tasks to include (all if None)
            samples_per_task: Number of samples per task
            seed: Random seed for sampling

        Returns:
            List of TaskInstance objects for evaluation
        """
        if task_names is None:
            task_names = self.get_available_tasks()

        batch = []
        for task_name in task_names:
            dataset = self.load_task(task_name)
            samples = dataset.sample(samples_per_task, seed)
            batch.extend(samples)

        # Shuffle the combined batch
        rng = random.Random(seed)
        rng.shuffle(batch)

        return batch

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()

    def export_for_annotation(
        self,
        output_path: Path,
        task_names: Optional[List[str]] = None,
        format: str = "jsonl",
    ) -> None:
        """
        Export data for external annotation.

        Args:
            output_path: Path to write exported data
            task_names: Tasks to export (all if None)
            format: Export format ('jsonl' or 'csv')
        """
        if task_names is None:
            task_names = self.get_available_tasks()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_instances = []
        for task_name in task_names:
            dataset = self.load_task(task_name)
            all_instances.extend(dataset.instances)

        if format == "jsonl":
            with open(output_path, "w") as f:
                for inst in all_instances:
                    f.write(json.dumps(inst.to_dict()) + "\n")
        elif format == "csv":
            import csv
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["task_type", "subtask", "prompt", "expected_answer"])
                for inst in all_instances:
                    writer.writerow([
                        inst.task_type,
                        inst.subtask,
                        inst.prompt,
                        inst.expected_answer,
                    ])
        else:
            raise ValueError(f"Unsupported format: {format}")


class StreamingDataLoader:
    """
    Streaming data loader for large datasets.

    Loads data in chunks to manage memory for very large experiments.
    """

    def __init__(self, data_dir: Union[str, Path], chunk_size: int = 100):
        """
        Initialize streaming loader.

        Args:
            data_dir: Directory containing task data
            chunk_size: Number of instances per chunk
        """
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size

    def stream_task(self, task_name: str) -> Iterator[List[TaskInstance]]:
        """
        Stream a task dataset in chunks.

        Args:
            task_name: Name of task to stream

        Yields:
            Lists of TaskInstance objects
        """
        task_file = self.data_dir / f"{task_name}.json"

        with open(task_file, "r") as f:
            data = json.load(f)

        for i in range(0, len(data), self.chunk_size):
            chunk = data[i:i + self.chunk_size]
            yield [TaskInstance.from_dict(d) for d in chunk]

    def stream_all_tasks(self) -> Iterator[TaskInstance]:
        """
        Stream all tasks one instance at a time.

        Yields:
            Individual TaskInstance objects
        """
        for task_name in self.get_available_tasks():
            for chunk in self.stream_task(task_name):
                for instance in chunk:
                    yield instance

    def get_available_tasks(self) -> List[str]:
        """Get list of available task names."""
        task_files = list(self.data_dir.glob("*.json"))
        return [f.stem for f in task_files if f.name != "dataset_info.json"]
