"""
Experiment runner for inverse scaling research.

Orchestrates model evaluation across multiple tasks and scales.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from ..data import DataLoader, TaskDataset, TaskInstance, UnifiedEvaluator, EvaluationResult
from ..models import ModelRegistry, ModelInterface, get_registry
from .trial import Trial, TrialResult, TrialStatus, TrialManager


logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    name: str
    models: List[str]  # Model keys (provider/name)
    tasks: List[str]  # Task names
    samples_per_task: int = 100
    num_runs: int = 1
    temperature: float = 0.0
    max_tokens: int = 256
    output_dir: str = "results"
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "models": self.models,
            "tasks": self.tasks,
            "samples_per_task": self.samples_per_task,
            "num_runs": self.num_runs,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "output_dir": self.output_dir,
            "seed": self.seed,
        }


class ExperimentRunner:
    """
    Main experiment runner for inverse scaling evaluation.

    Coordinates evaluation of multiple models across multiple tasks
    and aggregates results for analysis.
    """

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        registry: Optional[ModelRegistry] = None,
    ):
        """
        Initialize experiment runner.

        Args:
            data_dir: Directory containing task data
            output_dir: Directory for results
            registry: Model registry (uses global if not provided)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.registry = registry or get_registry()
        self.data_loader = DataLoader(data_dir)
        self.evaluator = UnifiedEvaluator()
        self.trial_manager = TrialManager(output_dir / "trials")

        self._progress_callbacks: List[Callable] = []
        self._cancel_flag = threading.Event()

    def add_progress_callback(self, callback: Callable) -> None:
        """Add a callback for progress updates."""
        self._progress_callbacks.append(callback)

    def _notify_progress(
        self,
        trial: Trial,
        progress: float,
        message: str = "",
    ) -> None:
        """Notify progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(trial, progress, message)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def run_experiment(
        self,
        config: ExperimentConfig,
        resume_trial_id: Optional[str] = None,
    ) -> Trial:
        """
        Run a full experiment.

        Args:
            config: Experiment configuration
            resume_trial_id: Optional trial ID to resume

        Returns:
            Completed Trial object
        """
        # Create or resume trial
        if resume_trial_id:
            trial = self.trial_manager.get_trial(resume_trial_id)
            if trial is None:
                raise ValueError(f"Trial not found: {resume_trial_id}")
            logger.info(f"Resuming trial {trial.trial_id}")
        else:
            trial = self.trial_manager.create_trial(
                name=config.name,
                description=f"Inverse scaling experiment: {config.name}",
                config=config.to_dict(),
            )
            logger.info(f"Created trial {trial.trial_id}")

        trial.start()
        self.trial_manager.update_trial(trial)

        try:
            # Calculate total work units
            total_work = len(config.models) * len(config.tasks) * config.num_runs
            completed_work = 0

            # Get completed model-task pairs
            completed_pairs = set()
            for result in trial.results:
                completed_pairs.add((result.model_name, result.task_type))

            # Run evaluations
            for run_idx in range(config.num_runs):
                for model_key in config.models:
                    if self._cancel_flag.is_set():
                        trial.cancel()
                        self.trial_manager.update_trial(trial)
                        return trial

                    for task_name in config.tasks:
                        # Skip if already completed
                        if (model_key, task_name) in completed_pairs:
                            completed_work += 1
                            continue

                        # Update progress
                        progress = completed_work / total_work
                        trial.update_progress(
                            progress=progress,
                            current_model=model_key,
                            current_task=task_name,
                        )
                        self._notify_progress(
                            trial,
                            progress,
                            f"Evaluating {model_key} on {task_name}",
                        )

                        # Run evaluation
                        try:
                            result = self._evaluate_model_task(
                                model_key=model_key,
                                task_name=task_name,
                                config=config,
                            )
                            trial.add_result(result)
                            self.trial_manager.update_trial(trial)

                        except Exception as e:
                            logger.error(f"Error evaluating {model_key} on {task_name}: {e}")
                            # Continue with other evaluations

                        completed_work += 1

            # Mark complete
            trial.complete()
            self.trial_manager.update_trial(trial)
            self._notify_progress(trial, 1.0, "Experiment completed")

            logger.info(f"Trial {trial.trial_id} completed with {len(trial.results)} results")

        except Exception as e:
            trial.fail(str(e))
            self.trial_manager.update_trial(trial)
            logger.error(f"Trial {trial.trial_id} failed: {e}")
            raise

        return trial

    def _evaluate_model_task(
        self,
        model_key: str,
        task_name: str,
        config: ExperimentConfig,
    ) -> TrialResult:
        """
        Evaluate a single model on a single task.

        Args:
            model_key: Model key (provider/name)
            task_name: Task name
            config: Experiment configuration

        Returns:
            TrialResult with evaluation metrics
        """
        logger.info(f"Evaluating {model_key} on {task_name}")

        # Get model
        provider, name = model_key.split("/")
        model = self.registry.get_model(
            name=name,
            provider=provider,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        # Get task data
        dataset = self.data_loader.load_task(task_name)
        samples = dataset.sample(config.samples_per_task, seed=config.seed)

        # Run evaluation
        responses = []
        eval_results = []
        latencies = []

        for instance in samples:
            # Generate response
            response = model.generate(instance.prompt)
            latencies.append(response.latency_ms)

            # Evaluate
            eval_result = self.evaluator.evaluate(instance, response.text)
            eval_results.append(eval_result)

            responses.append({
                "prompt": instance.prompt,
                "expected": instance.expected_answer,
                "response": response.text,
                "is_correct": eval_result.is_correct,
                "score": eval_result.score,
                "latency_ms": response.latency_ms,
                "metrics": eval_result.metrics,
            })

        # Compute aggregate metrics
        correct = sum(1 for r in eval_results if r.is_correct)
        accuracy = correct / len(eval_results) if eval_results else 0
        mean_score = sum(r.score for r in eval_results) / len(eval_results) if eval_results else 0

        aggregate_metrics = self.evaluator.compute_aggregate_metrics(eval_results)

        # Compute latency stats
        latency_stats = {
            "mean": sum(latencies) / len(latencies) if latencies else 0,
            "min": min(latencies) if latencies else 0,
            "max": max(latencies) if latencies else 0,
            "total": sum(latencies),
        }

        return TrialResult(
            model_name=model_key,
            model_provider=provider,
            estimated_params=model.config.estimated_params,
            task_type=task_name,
            subtask="all",
            num_samples=len(samples),
            correct=correct,
            accuracy=accuracy,
            mean_score=mean_score,
            metrics=aggregate_metrics,
            responses=responses,
            latency_stats=latency_stats,
        )

    def cancel(self) -> None:
        """Cancel any running experiment."""
        self._cancel_flag.set()

    def reset_cancel(self) -> None:
        """Reset cancel flag."""
        self._cancel_flag.clear()

    def get_trial(self, trial_id: str) -> Optional[Trial]:
        """Get a trial by ID."""
        return self.trial_manager.get_trial(trial_id)

    def list_trials(self, status: Optional[TrialStatus] = None) -> List[Trial]:
        """List all trials."""
        return self.trial_manager.list_trials(status)


class ParallelExperimentRunner(ExperimentRunner):
    """
    Experiment runner with parallel model evaluation.

    Evaluates multiple models concurrently for faster experiments.
    """

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        registry: Optional[ModelRegistry] = None,
        max_workers: int = 4,
    ):
        super().__init__(data_dir, output_dir, registry)
        self.max_workers = max_workers

    def _evaluate_model_task_parallel(
        self,
        model_key: str,
        task_name: str,
        config: ExperimentConfig,
    ) -> TrialResult:
        """Thread-safe evaluation wrapper."""
        return self._evaluate_model_task(model_key, task_name, config)

    def run_experiment(
        self,
        config: ExperimentConfig,
        resume_trial_id: Optional[str] = None,
    ) -> Trial:
        """Run experiment with parallel model evaluation."""
        # Create trial
        if resume_trial_id:
            trial = self.trial_manager.get_trial(resume_trial_id)
            if trial is None:
                raise ValueError(f"Trial not found: {resume_trial_id}")
        else:
            trial = self.trial_manager.create_trial(
                name=config.name,
                description=f"Parallel inverse scaling experiment: {config.name}",
                config=config.to_dict(),
            )

        trial.start()
        self.trial_manager.update_trial(trial)

        try:
            # Build work items
            work_items = []
            completed_pairs = {(r.model_name, r.task_type) for r in trial.results}

            for run_idx in range(config.num_runs):
                for model_key in config.models:
                    for task_name in config.tasks:
                        if (model_key, task_name) not in completed_pairs:
                            work_items.append((model_key, task_name))

            total_work = len(work_items)
            completed = 0

            # Run parallel evaluation
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._evaluate_model_task_parallel,
                        model_key,
                        task_name,
                        config,
                    ): (model_key, task_name)
                    for model_key, task_name in work_items
                }

                for future in as_completed(futures):
                    if self._cancel_flag.is_set():
                        executor.shutdown(wait=False)
                        trial.cancel()
                        break

                    model_key, task_name = futures[future]
                    try:
                        result = future.result()
                        trial.add_result(result)
                    except Exception as e:
                        logger.error(f"Error evaluating {model_key} on {task_name}: {e}")

                    completed += 1
                    progress = completed / total_work
                    trial.update_progress(progress)
                    self._notify_progress(trial, progress, f"Completed {completed}/{total_work}")

            if not self._cancel_flag.is_set():
                trial.complete()

            self.trial_manager.update_trial(trial)

        except Exception as e:
            trial.fail(str(e))
            self.trial_manager.update_trial(trial)
            raise

        return trial
