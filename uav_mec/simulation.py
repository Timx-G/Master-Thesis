"""Canonical one-path episode execution over a fixed exogenous trace."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace

from .domain import EpisodeResult, ExperimentTrace, Schedule, SlotSnapshot
from .evaluation import EvaluationConfig, evaluate_schedule
from .model import NOMAChannel, ScenarioConfig

Scheduler = Callable[[SlotSnapshot], Schedule]


def run_episode(
    trace: ExperimentTrace,
    scheduler: Scheduler,
    scenario: ScenarioConfig,
    evaluation: EvaluationConfig,
) -> EpisodeResult:
    """Evaluate every unique arrival exactly once without queue retries."""
    channel = NOMAChannel(scenario.channel, scenario.uav.height)
    outcomes = []
    seen: set[int] = set()
    for snapshot in trace.slots:
        task_ids = [task.task_id for task in snapshot.tasks]
        ids = set(task_ids)
        if len(ids) != len(task_ids) or seen.intersection(ids):
            raise ValueError("trace repeats a task_id across or within slots")
        seen.update(ids)

        started = time.perf_counter()
        schedule = scheduler(snapshot)
        outcome = evaluate_schedule(snapshot, schedule, channel, evaluation)
        runtime_s = time.perf_counter() - started
        outcomes.append(replace(outcome, runtime_s=runtime_s))

    return EpisodeResult(trace.seed, tuple(outcomes))
