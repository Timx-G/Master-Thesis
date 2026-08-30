"""Frozen experiment families, seed pools, and primary hypotheses."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .experiment import KNOWN_METHODS


@dataclass(frozen=True, slots=True)
class ExperimentFamily:
    name: str
    seed_start: int
    seed_count: int
    num_slots: int
    methods: tuple[str, ...]

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.seed_start, self.seed_start + self.seed_count))


@dataclass(frozen=True, slots=True)
class PrimaryHypothesis:
    hypothesis_id: str
    family: str
    treatment: str
    control: str
    mde_pp: float


@dataclass(frozen=True, slots=True)
class ExperimentRegistry:
    candidate_sample_sizes: tuple[int, ...]
    families: tuple[ExperimentFamily, ...]
    hypotheses: tuple[PrimaryHypothesis, ...]


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_registry(path: Path) -> ExperimentRegistry:
    """Load and validate the immutable v2 experiment registry."""
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    raw_candidates = raw.get("candidate_sample_sizes")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("candidate_sample_sizes must be a non-empty list")
    candidates = tuple(
        _positive_int(value, "candidate sample size")
        for value in raw_candidates
    )
    if tuple(sorted(set(candidates))) != candidates:
        raise ValueError("candidate_sample_sizes must be unique and increasing")

    raw_families = raw.get("families")
    if not isinstance(raw_families, list) or not raw_families:
        raise ValueError("families must be a non-empty table array")
    families: list[ExperimentFamily] = []
    for value in raw_families:
        if not isinstance(value, dict):
            raise ValueError("each family must be a table")
        name = value.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("family name must be a non-empty string")
        raw_methods = value.get("methods")
        if not isinstance(raw_methods, list) or not raw_methods:
            raise ValueError(f"family {name} methods must be non-empty")
        methods = tuple(raw_methods)
        if any(not isinstance(method, str) or not method for method in methods):
            raise ValueError(f"family {name} methods must be strings")
        if len(set(methods)) != len(methods):
            raise ValueError(f"family {name} methods must be unique")
        unknown = sorted(set(methods) - KNOWN_METHODS)
        if unknown:
            raise ValueError(f"unknown method {unknown[0]}")
        families.append(
            ExperimentFamily(
                name,
                _positive_int(value.get("seed_start"), f"{name}.seed_start"),
                _positive_int(value.get("seed_count"), f"{name}.seed_count"),
                _positive_int(value.get("num_slots"), f"{name}.num_slots"),
                methods,
            )
        )
    family_by_name = {family.name: family for family in families}
    if len(family_by_name) != len(families):
        raise ValueError("family names must be unique")
    seen_seed: dict[int, str] = {}
    for family in families:
        for seed in family.seeds:
            previous = seen_seed.setdefault(seed, family.name)
            if previous != family.name:
                raise ValueError(
                    f"seed pools overlap between {previous} and {family.name}"
                )

    raw_hypotheses = raw.get("hypotheses")
    if not isinstance(raw_hypotheses, list) or not raw_hypotheses:
        raise ValueError("hypotheses must be a non-empty table array")
    hypotheses: list[PrimaryHypothesis] = []
    for value in raw_hypotheses:
        if not isinstance(value, dict):
            raise ValueError("each hypothesis must be a table")
        hypothesis_id = value.get("hypothesis_id")
        family_name = value.get("family")
        treatment = value.get("treatment")
        control = value.get("control")
        if not isinstance(hypothesis_id, str) or not hypothesis_id:
            raise ValueError("hypothesis_id must be a non-empty string")
        if family_name not in family_by_name:
            raise ValueError(f"unknown hypothesis family {family_name}")
        family = family_by_name[family_name]
        for role, method in (("treatment", treatment), ("control", control)):
            if method not in family.methods:
                raise ValueError(
                    f"hypothesis {hypothesis_id} {role} is not in family {family.name}"
                )
        mde_pp = value.get("mde_pp")
        if not isinstance(mde_pp, (int, float)) or isinstance(mde_pp, bool):
            raise ValueError(f"hypothesis {hypothesis_id} mde_pp must be numeric")
        mde_pp = float(mde_pp)
        if not math.isfinite(mde_pp) or mde_pp <= 0.0:
            raise ValueError(f"hypothesis {hypothesis_id} mde_pp must be positive")
        hypotheses.append(
            PrimaryHypothesis(
                hypothesis_id,
                family.name,
                treatment,
                control,
                mde_pp,
            )
        )
    if len({value.hypothesis_id for value in hypotheses}) != len(hypotheses):
        raise ValueError("hypothesis IDs must be unique")
    return ExperimentRegistry(candidates, tuple(families), tuple(hypotheses))
