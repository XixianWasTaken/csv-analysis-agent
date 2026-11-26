"""One deadline and independent counters for all phases of an analysis task."""

import time
from dataclasses import dataclass

from csv_agent.config import RunLimits


@dataclass
class TaskBudget:
    limits: RunLimits
    started: float
    model_calls: int = 0
    executions: int = 0

    def remaining_seconds(self) -> float:
        seconds = self.started + self.limits.task_timeout_seconds - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("Analysis task timed out")
        return seconds

    @property
    def can_generate(self) -> bool:
        return self.model_calls < self.limits.model_call_budget

    @property
    def can_execute(self) -> bool:
        return self.executions < self.limits.max_executions

    def remaining(self) -> dict[str, int | float]:
        return {
            "model_calls": self.limits.model_call_budget - self.model_calls,
            "executions": self.limits.max_executions - self.executions,
            "seconds": self.remaining_seconds(),
        }
