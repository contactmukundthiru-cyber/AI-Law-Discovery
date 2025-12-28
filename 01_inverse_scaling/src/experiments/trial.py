"""
Trial management for inverse scaling experiments.

Defines trial structure, status tracking, and result containers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
import json
import uuid
from pathlib import Path


class TrialStatus(Enum):
    """Status of an experiment trial."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrialResult:
    """Result from a single model evaluation."""
    model_name: str
    model_provider: str
    estimated_params: str
    task_type: str
    subtask: str
    num_samples: int
    correct: int
    accuracy: float
    mean_score: float
    metrics: Dict[str, Any]
    responses: List[Dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.now)
    latency_stats: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_provider": self.model_provider,
            "estimated_params": self.estimated_params,
            "task_type": self.task_type,
            "subtask": self.subtask,
            "num_samples": self.num_samples,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "mean_score": self.mean_score,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "latency_stats": self.latency_stats,
            # Responses stored separately for space
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrialResult":
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        data["responses"] = data.get("responses", [])
        return cls(**data)


@dataclass
class Trial:
    """
    An experiment trial evaluating one or more models on tasks.
    """
    trial_id: str
    name: str
    description: str
    config: Dict[str, Any]
    status: TrialStatus = TrialStatus.PENDING
    results: List[TrialResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    progress: float = 0.0
    current_model: Optional[str] = None
    current_task: Optional[str] = None

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        config: Dict[str, Any],
    ) -> "Trial":
        """Create a new trial."""
        return cls(
            trial_id=str(uuid.uuid4())[:8],
            name=name,
            description=description,
            config=config,
        )

    def start(self) -> None:
        """Mark trial as started."""
        self.status = TrialStatus.RUNNING
        self.started_at = datetime.now()

    def complete(self) -> None:
        """Mark trial as completed."""
        self.status = TrialStatus.COMPLETED
        self.completed_at = datetime.now()
        self.progress = 1.0

    def fail(self, error: str) -> None:
        """Mark trial as failed."""
        self.status = TrialStatus.FAILED
        self.completed_at = datetime.now()
        self.error = error

    def cancel(self) -> None:
        """Cancel the trial."""
        self.status = TrialStatus.CANCELLED
        self.completed_at = datetime.now()

    def add_result(self, result: TrialResult) -> None:
        """Add a result to the trial."""
        self.results.append(result)

    def update_progress(
        self,
        progress: float,
        current_model: Optional[str] = None,
        current_task: Optional[str] = None,
    ) -> None:
        """Update trial progress."""
        self.progress = progress
        if current_model:
            self.current_model = current_model
        if current_task:
            self.current_task = current_task

    @property
    def duration(self) -> Optional[float]:
        """Get trial duration in seconds."""
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "status": self.status.value,
            "results": [r.to_dict() for r in self.results],
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "progress": self.progress,
            "current_model": self.current_model,
            "current_task": self.current_task,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trial":
        data = data.copy()
        data["status"] = TrialStatus(data["status"])
        data["results"] = [TrialResult.from_dict(r) for r in data.get("results", [])]
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("started_at"):
            data["started_at"] = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            data["completed_at"] = datetime.fromisoformat(data["completed_at"])
        return cls(**data)

    def save(self, output_dir: Path) -> Path:
        """Save trial to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save main trial data
        trial_file = output_dir / f"trial_{self.trial_id}.json"
        with open(trial_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        # Save detailed responses separately
        responses_dir = output_dir / f"trial_{self.trial_id}_responses"
        responses_dir.mkdir(exist_ok=True)

        for i, result in enumerate(self.results):
            response_file = responses_dir / f"result_{i}_{result.model_name}_{result.task_type}.json"
            with open(response_file, "w") as f:
                json.dump({
                    "summary": result.to_dict(),
                    "responses": result.responses,
                }, f, indent=2)

        return trial_file

    @classmethod
    def load(cls, trial_file: Path) -> "Trial":
        """Load trial from disk."""
        with open(trial_file, "r") as f:
            data = json.load(f)

        trial = cls.from_dict(data)

        # Try to load detailed responses
        responses_dir = trial_file.parent / f"trial_{trial.trial_id}_responses"
        if responses_dir.exists():
            for result in trial.results:
                pattern = f"result_*_{result.model_name}_{result.task_type}.json"
                matching = list(responses_dir.glob(pattern))
                if matching:
                    with open(matching[0], "r") as f:
                        resp_data = json.load(f)
                        result.responses = resp_data.get("responses", [])

        return trial


class TrialManager:
    """
    Manager for experiment trials.

    Handles trial persistence and retrieval.
    """

    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._trials: Dict[str, Trial] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing trials from storage."""
        for trial_file in self.storage_dir.glob("trial_*.json"):
            if "_responses" not in str(trial_file):
                try:
                    trial = Trial.load(trial_file)
                    self._trials[trial.trial_id] = trial
                except Exception as e:
                    print(f"Warning: Failed to load {trial_file}: {e}")

    def create_trial(
        self,
        name: str,
        description: str,
        config: Dict[str, Any],
    ) -> Trial:
        """Create and register a new trial."""
        trial = Trial.create(name, description, config)
        self._trials[trial.trial_id] = trial
        trial.save(self.storage_dir)
        return trial

    def get_trial(self, trial_id: str) -> Optional[Trial]:
        """Get a trial by ID."""
        return self._trials.get(trial_id)

    def list_trials(
        self,
        status: Optional[TrialStatus] = None,
    ) -> List[Trial]:
        """List all trials, optionally filtered by status."""
        trials = list(self._trials.values())
        if status:
            trials = [t for t in trials if t.status == status]
        return sorted(trials, key=lambda t: t.created_at, reverse=True)

    def update_trial(self, trial: Trial) -> None:
        """Update and persist a trial."""
        self._trials[trial.trial_id] = trial
        trial.save(self.storage_dir)

    def delete_trial(self, trial_id: str) -> bool:
        """Delete a trial."""
        if trial_id not in self._trials:
            return False

        trial = self._trials.pop(trial_id)

        # Remove files
        trial_file = self.storage_dir / f"trial_{trial_id}.json"
        if trial_file.exists():
            trial_file.unlink()

        responses_dir = self.storage_dir / f"trial_{trial_id}_responses"
        if responses_dir.exists():
            import shutil
            shutil.rmtree(responses_dir)

        return True

    def get_latest(self, n: int = 10) -> List[Trial]:
        """Get the n most recent trials."""
        all_trials = self.list_trials()
        return all_trials[:n]
