"""Canonical one-path episode execution over a fixed exogenous trace."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace

from .domain import (
    EpisodeResult,
    ExperimentTrace,
    Schedule,
    SlotOutcome,
    SlotSnapshot,
)
from .evaluation import EvaluationConfig, evaluate_schedule
from .model import NOMAChannel, ScenarioConfig

Scheduler = Callable[[SlotSnapshot], Schedule]
OutcomeMethod = Callable[[SlotSnapshot], SlotOutcome]


def _validate_outcome(
    snapshot: SlotSnapshot,
    outcome: SlotOutcome,
) -> None:
    if outcome.slot_index != snapshot.slot_index:
        raise ValueError("method returned an outcome for the wrong slot")
    expected_ids = tuple(sorted(task.task_id for task in snapshot.tasks))
    if outcome.arrived_task_ids != expected_ids:
        raise ValueError("method returned incorrect arrived_task_ids")

    decision_ids = tuple(decision.task_id for decision in outcome.decisions)
    if len(decision_ids) != len(set(decision_ids)) or set(decision_ids) != set(
        expected_ids
    ):
        raise ValueError("method must return exactly one decision per arrived task")
    completed_from_decisions = {
        decision.task_id for decision in outcome.decisions if decision.completed
    }
    missed_from_decisions = {
        decision.task_id for decision in outcome.decisions if not decision.completed
    }
    if (
        len(outcome.completed_task_ids) != len(set(outcome.completed_task_ids))
        or len(outcome.missed_task_ids) != len(set(outcome.missed_task_ids))
        or set(outcome.completed_task_ids) != completed_from_decisions
        or set(outcome.missed_task_ids) != missed_from_decisions
        or completed_from_decisions.intersection(missed_from_decisions)
    ):
        raise ValueError("outcome completion partition disagrees with decisions")


def run_outcome_episode(
    trace: ExperimentTrace,
    method: OutcomeMethod,
) -> EpisodeResult:
    """Run a complete-outcome method once per immutable trace slot."""
    outcomes = []
    seen: set[int] = set()
    for snapshot in trace.slots:
        task_ids = [task.task_id for task in snapshot.tasks]
        ids = set(task_ids)
        if len(ids) != len(task_ids) or seen.intersection(ids):
            raise ValueError("trace repeats a task_id across or within slots")
        seen.update(ids)

        started = time.perf_counter()
        outcome = method(snapshot)
        runtime_s = time.perf_counter() - started
        _validate_outcome(snapshot, outcome)
        outcomes.append(replace(outcome, runtime_s=runtime_s))
    return EpisodeResult(trace.seed, tuple(outcomes))


def run_episode(
    trace: ExperimentTrace,
    scheduler: Scheduler,
    scenario: ScenarioConfig,
    evaluation: EvaluationConfig,
) -> EpisodeResult:
    """Evaluate every unique arrival exactly once without queue retries."""
    channel = NOMAChannel(scenario.channel, scenario.uav.height)

    def method(snapshot: SlotSnapshot) -> SlotOutcome:
        schedule = scheduler(snapshot)
        return evaluate_schedule(snapshot, schedule, channel, evaluation)

    return run_outcome_episode(trace, method)
