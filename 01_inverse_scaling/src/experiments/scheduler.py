"""
Experiment scheduler for inverse scaling research.

Manages queued experiments and scheduled runs.
"""

import threading
import queue
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from pathlib import Path

from .runner import ExperimentRunner, ExperimentConfig
from .trial import Trial, TrialStatus


logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Type of scheduled run."""
    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    RECURRING = "recurring"


@dataclass
class ScheduledExperiment:
    """A scheduled experiment."""
    schedule_id: str
    config: ExperimentConfig
    schedule_type: ScheduleType
    scheduled_time: datetime
    repeat_interval: Optional[timedelta] = None
    enabled: bool = True
    last_run: Optional[datetime] = None
    run_count: int = 0

    def should_run(self) -> bool:
        """Check if experiment should run now."""
        if not self.enabled:
            return False

        now = datetime.now()

        if self.schedule_type == ScheduleType.IMMEDIATE:
            return self.run_count == 0

        if self.schedule_type == ScheduleType.DELAYED:
            return now >= self.scheduled_time and self.run_count == 0

        if self.schedule_type == ScheduleType.RECURRING:
            if self.last_run is None:
                return now >= self.scheduled_time
            next_run = self.last_run + self.repeat_interval
            return now >= next_run

        return False


class ExperimentScheduler:
    """
    Scheduler for managing experiment execution.

    Features:
    - Queue-based experiment execution
    - Scheduled/delayed runs
    - Recurring experiments
    - Priority management
    """

    def __init__(
        self,
        runner: ExperimentRunner,
        max_concurrent: int = 1,
    ):
        """
        Initialize scheduler.

        Args:
            runner: Experiment runner instance
            max_concurrent: Maximum concurrent experiments
        """
        self.runner = runner
        self.max_concurrent = max_concurrent

        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._scheduled: Dict[str, ScheduledExperiment] = {}
        self._running: Dict[str, Trial] = {}
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._scheduler_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []

        self._schedule_id_counter = 0

    def start(self) -> None:
        """Start the scheduler."""
        self._stop_flag.clear()

        # Start worker thread
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        # Start scheduler thread
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()

        logger.info("Experiment scheduler started")

    def stop(self) -> None:
        """Stop the scheduler."""
        self._stop_flag.set()

        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)

        logger.info("Experiment scheduler stopped")

    def add_callback(self, callback: Callable) -> None:
        """Add callback for experiment events."""
        self._callbacks.append(callback)

    def _notify_callbacks(self, event: str, data: Any) -> None:
        """Notify callbacks of events."""
        for callback in self._callbacks:
            try:
                callback(event, data)
            except Exception as e:
                logger.warning(f"Callback error: {e}")

    def submit(
        self,
        config: ExperimentConfig,
        priority: int = 5,
    ) -> str:
        """
        Submit an experiment for immediate execution.

        Args:
            config: Experiment configuration
            priority: Priority (lower = higher priority)

        Returns:
            Schedule ID
        """
        schedule_id = self._generate_schedule_id()

        scheduled = ScheduledExperiment(
            schedule_id=schedule_id,
            config=config,
            schedule_type=ScheduleType.IMMEDIATE,
            scheduled_time=datetime.now(),
        )

        with self._lock:
            self._scheduled[schedule_id] = scheduled

        self._queue.put((priority, schedule_id))
        self._notify_callbacks("submitted", {"schedule_id": schedule_id, "config": config.to_dict()})

        logger.info(f"Submitted experiment {schedule_id}: {config.name}")
        return schedule_id

    def schedule(
        self,
        config: ExperimentConfig,
        run_at: datetime,
        priority: int = 5,
    ) -> str:
        """
        Schedule an experiment for future execution.

        Args:
            config: Experiment configuration
            run_at: When to run the experiment
            priority: Priority

        Returns:
            Schedule ID
        """
        schedule_id = self._generate_schedule_id()

        scheduled = ScheduledExperiment(
            schedule_id=schedule_id,
            config=config,
            schedule_type=ScheduleType.DELAYED,
            scheduled_time=run_at,
        )

        with self._lock:
            self._scheduled[schedule_id] = scheduled

        self._notify_callbacks("scheduled", {
            "schedule_id": schedule_id,
            "run_at": run_at.isoformat(),
        })

        logger.info(f"Scheduled experiment {schedule_id} for {run_at}")
        return schedule_id

    def schedule_recurring(
        self,
        config: ExperimentConfig,
        start_at: datetime,
        interval: timedelta,
        priority: int = 5,
    ) -> str:
        """
        Schedule a recurring experiment.

        Args:
            config: Experiment configuration
            start_at: When to start
            interval: Interval between runs
            priority: Priority

        Returns:
            Schedule ID
        """
        schedule_id = self._generate_schedule_id()

        scheduled = ScheduledExperiment(
            schedule_id=schedule_id,
            config=config,
            schedule_type=ScheduleType.RECURRING,
            scheduled_time=start_at,
            repeat_interval=interval,
        )

        with self._lock:
            self._scheduled[schedule_id] = scheduled

        logger.info(f"Scheduled recurring experiment {schedule_id}")
        return schedule_id

    def cancel(self, schedule_id: str) -> bool:
        """Cancel a scheduled experiment."""
        with self._lock:
            if schedule_id in self._scheduled:
                self._scheduled[schedule_id].enabled = False
                logger.info(f"Cancelled experiment {schedule_id}")
                return True
            return False

    def get_status(self, schedule_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a scheduled experiment."""
        with self._lock:
            if schedule_id not in self._scheduled:
                return None

            scheduled = self._scheduled[schedule_id]
            return {
                "schedule_id": schedule_id,
                "name": scheduled.config.name,
                "type": scheduled.schedule_type.value,
                "scheduled_time": scheduled.scheduled_time.isoformat(),
                "enabled": scheduled.enabled,
                "run_count": scheduled.run_count,
                "last_run": scheduled.last_run.isoformat() if scheduled.last_run else None,
            }

    def list_scheduled(self) -> List[Dict[str, Any]]:
        """List all scheduled experiments."""
        with self._lock:
            return [
                self.get_status(sid) for sid in self._scheduled
            ]

    def list_running(self) -> List[Dict[str, Any]]:
        """List currently running experiments."""
        with self._lock:
            return [
                {
                    "schedule_id": sid,
                    "trial_id": trial.trial_id,
                    "name": trial.name,
                    "progress": trial.progress,
                    "current_model": trial.current_model,
                    "current_task": trial.current_task,
                }
                for sid, trial in self._running.items()
            ]

    def _generate_schedule_id(self) -> str:
        """Generate unique schedule ID."""
        self._schedule_id_counter += 1
        return f"sched_{self._schedule_id_counter:04d}"

    def _worker_loop(self) -> None:
        """Worker thread main loop."""
        while not self._stop_flag.is_set():
            try:
                # Get next experiment (timeout to allow stop check)
                try:
                    priority, schedule_id = self._queue.get(timeout=1)
                except queue.Empty:
                    continue

                # Check if still valid
                with self._lock:
                    if schedule_id not in self._scheduled:
                        continue
                    scheduled = self._scheduled[schedule_id]
                    if not scheduled.enabled:
                        continue

                # Wait for capacity
                while len(self._running) >= self.max_concurrent:
                    if self._stop_flag.is_set():
                        return
                    time.sleep(0.5)

                # Run experiment
                self._run_experiment(scheduled)

            except Exception as e:
                logger.error(f"Worker error: {e}")

    def _scheduler_loop(self) -> None:
        """Scheduler thread for timed experiments."""
        while not self._stop_flag.is_set():
            try:
                now = datetime.now()

                with self._lock:
                    for schedule_id, scheduled in self._scheduled.items():
                        if scheduled.should_run():
                            # Queue for execution
                            self._queue.put((5, schedule_id))

                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Scheduler error: {e}")

    def _run_experiment(self, scheduled: ScheduledExperiment) -> None:
        """Execute a scheduled experiment."""
        schedule_id = scheduled.schedule_id
        config = scheduled.config

        try:
            logger.info(f"Starting experiment {schedule_id}: {config.name}")
            self._notify_callbacks("started", {"schedule_id": schedule_id})

            # Run the experiment
            trial = self.runner.run_experiment(config)

            with self._lock:
                self._running[schedule_id] = trial

            # Wait for completion
            while trial.status == TrialStatus.RUNNING:
                time.sleep(1)

            # Update scheduled info
            with self._lock:
                scheduled.last_run = datetime.now()
                scheduled.run_count += 1
                if schedule_id in self._running:
                    del self._running[schedule_id]

            self._notify_callbacks("completed", {
                "schedule_id": schedule_id,
                "trial_id": trial.trial_id,
                "status": trial.status.value,
            })

            logger.info(f"Completed experiment {schedule_id}")

        except Exception as e:
            logger.error(f"Experiment {schedule_id} failed: {e}")
            self._notify_callbacks("failed", {
                "schedule_id": schedule_id,
                "error": str(e),
            })

            with self._lock:
                if schedule_id in self._running:
                    del self._running[schedule_id]
