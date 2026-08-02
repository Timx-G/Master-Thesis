"""Authoritative thesis-v2 experiment registry.

The registry is deliberately independent of runner dispatch.  It describes
which methods, cases, seed candidates, and primary contrasts are admissible;
executors validate their resolved definitions against it before doing work.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from .experiment import method_spec_sha256, thesis_method_specs


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_CASE_ROLES = frozenset({"primary", "supporting", "correctness"})
_AUTHORITATIVE_PRIMARY_CONTRASTS = (
    ("H1", "full", "intra_global", "main_k3_n45"),
    (
        "H2",
        "full",
        "cross_bilateral",
        "overlap_high_coupling1_load12",
    ),
    ("H3", "fixed_radio_dd", "fixed_radio_edf", "dd_generated"),
    ("H4", "iea_dd", "exchange_edf", "joint_k3_n45"),
)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: object, field: str) -> int:
    if not _is_int(value) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if not _is_int(value) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a string sequence")
    result = tuple(value)
    if not result:
        raise ValueError(f"{field} must be non-empty")
    for item in result:
        _nonempty_string(item, field)
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must be unique")
    return result


def _seed_tuple(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be an integer sequence")
    result = tuple(value)
    if not result:
        raise ValueError(f"{field} must be non-empty")
    for seed in result:
        _nonnegative_int(seed, field)
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must be unique")
    return result


@dataclass(frozen=True, slots=True)
class RegisteredCase:
    case_id: str
    role: str
    fixed_sample_size: int | None

    def __post_init__(self) -> None:
        _nonempty_string(self.case_id, "case_id")
        if self.role not in _CASE_ROLES:
            raise ValueError("role must be primary, supporting, or correctness")
        if self.fixed_sample_size is not None:
            _positive_int(self.fixed_sample_size, "fixed_sample_size")


@dataclass(frozen=True, slots=True)
class ExperimentFamily:
    name: str
    config_path: Path
    seed_start: int
    seed_count: int
    num_slots: int
    methods: tuple[str, ...]
    sample_size_policy: str
    formal_order: int
    cases: tuple[RegisteredCase, ...]

    def __post_init__(self) -> None:
        _nonempty_string(self.name, "family.name")
        if not isinstance(self.config_path, Path):
            raise ValueError("family.config_path must be a Path")
        _nonnegative_int(self.seed_start, f"{self.name}.seed_start")
        _positive_int(self.seed_count, f"{self.name}.seed_count")
        _positive_int(self.num_slots, f"{self.name}.num_slots")
        methods = _string_tuple(self.methods, f"{self.name}.methods")
        if methods != self.methods:
            raise ValueError(f"{self.name}.methods must be stored as a tuple")
        _nonempty_string(
            self.sample_size_policy,
            f"{self.name}.sample_size_policy",
        )
        _positive_int(self.formal_order, f"{self.name}.formal_order")
        if not isinstance(self.cases, tuple) or not self.cases:
            raise ValueError(f"{self.name}.cases must be a non-empty tuple")
        case_ids = tuple(case.case_id for case in self.cases)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError(f"{self.name}.cases case_id values must be unique")

    @property
    def seeds(self) -> tuple[int, ...]:
        """Return this family's closed-open contiguous candidate range."""
        return tuple(range(self.seed_start, self.seed_start + self.seed_count))


@dataclass(frozen=True, slots=True)
class PrimaryHypothesis:
    hypothesis_id: str
    family: str
    case_id: str | None
    treatment: str
    control: str
    outcome: str
    estimand: str
    primary_test: str
    sensitivity_test: str
    multiplicity_family: str
    mde_pp: float

    def __post_init__(self) -> None:
        _nonempty_string(self.hypothesis_id, "hypothesis_id")
        _nonempty_string(self.family, f"{self.hypothesis_id}.family")
        if self.case_id is not None:
            _nonempty_string(self.case_id, f"{self.hypothesis_id}.case_id")
        _nonempty_string(self.treatment, f"{self.hypothesis_id}.treatment")
        _nonempty_string(self.control, f"{self.hypothesis_id}.control")
        if self.treatment == self.control:
            raise ValueError(f"{self.hypothesis_id}.treatment and control must differ")
        for field, value in (
            ("outcome", self.outcome),
            ("estimand", self.estimand),
            ("primary_test", self.primary_test),
            ("sensitivity_test", self.sensitivity_test),
            ("multiplicity_family", self.multiplicity_family),
        ):
            _nonempty_string(value, f"{self.hypothesis_id}.{field}")
        if (
            not isinstance(self.mde_pp, (int, float))
            or isinstance(self.mde_pp, bool)
            or not math.isfinite(float(self.mde_pp))
            or self.mde_pp <= 0.0
        ):
            raise ValueError(f"{self.hypothesis_id}.mde_pp must be positive")


@dataclass(frozen=True, slots=True)
class ExperimentRegistry:
    development_seeds: tuple[int, ...]
    pilot_seeds: tuple[int, ...]
    candidate_sample_sizes: tuple[int, ...]
    descriptive_externality_n: int
    method_factor_hashes: Mapping[str, str]
    families: tuple[ExperimentFamily, ...]
    hypotheses: tuple[PrimaryHypothesis, ...]

    def __post_init__(self) -> None:
        development = _seed_tuple(
            self.development_seeds,
            "development_seeds",
        )
        pilot = _seed_tuple(self.pilot_seeds, "pilot_seeds")
        if development != self.development_seeds:
            raise ValueError("development_seeds must be stored as a tuple")
        if pilot != self.pilot_seeds:
            raise ValueError("pilot_seeds must be stored as a tuple")
        if self.candidate_sample_sizes != (50, 100, 200):
            raise ValueError("candidate_sample_sizes must be exactly (50, 100, 200)")
        if self.descriptive_externality_n != 30:
            raise ValueError("descriptive_externality_n must be exactly 30")
        if not isinstance(self.method_factor_hashes, Mapping):
            raise ValueError("method_factor_hashes must be a mapping")
        factor_hashes = dict(self.method_factor_hashes)
        for method_id, factor_hash in factor_hashes.items():
            _nonempty_string(method_id, "method_factor_hashes.method_id")
            if (
                not isinstance(factor_hash, str)
                or len(factor_hash) != 64
                or any(character not in "0123456789abcdef" for character in factor_hash)
            ):
                raise ValueError(f"method_factor_hashes.{method_id} must be a SHA-256")
        object.__setattr__(
            self,
            "method_factor_hashes",
            MappingProxyType(dict(sorted(factor_hashes.items()))),
        )
        if not isinstance(self.families, tuple) or not self.families:
            raise ValueError("families must be a non-empty tuple")
        if not isinstance(self.hypotheses, tuple):
            raise ValueError("hypotheses must be a tuple")

        family_names = tuple(family.name for family in self.families)
        if len(set(family_names)) != len(family_names):
            raise ValueError("families.name values must be unique")
        formal_orders = tuple(family.formal_order for family in self.families)
        if len(set(formal_orders)) != len(formal_orders):
            raise ValueError("families.formal_order values must be unique")
        config_paths = tuple(family.config_path for family in self.families)
        if len(set(config_paths)) != len(config_paths):
            raise ValueError("families.config_path values must be unique")
        registered_method_ids = {
            method_id for family in self.families for method_id in family.methods
        }
        if set(factor_hashes) != registered_method_ids:
            raise ValueError(
                "method_factor_hashes must cover every registered method exactly"
            )
        authoritative_specs = thesis_method_specs()
        expected_hashes = {
            method_id: method_spec_sha256(authoritative_specs[method_id])
            for method_id in registered_method_ids
        }
        if factor_hashes != expected_hashes:
            raise ValueError("registered method factor hash mismatch")

        pools: list[tuple[str, set[int]]] = [
            ("development_seeds", set(development)),
            ("pilot_seeds", set(pilot)),
        ]
        pools.extend((family.name, set(family.seeds)) for family in self.families)
        for index, (left_name, left) in enumerate(pools):
            for right_name, right in pools[index + 1 :]:
                overlap = left.intersection(right)
                if overlap:
                    raise ValueError(
                        "seed pools overlap between "
                        f"{left_name} and {right_name} at {min(overlap)}"
                    )

        if len(self.hypotheses) != 4:
            raise ValueError("hypotheses must contain exactly four rows")
        hypothesis_ids = tuple(item.hypothesis_id for item in self.hypotheses)
        if len(set(hypothesis_ids)) != len(hypothesis_ids):
            raise ValueError("hypotheses.hypothesis_id values must be unique")
        primary_contrasts = tuple(
            (
                item.hypothesis_id,
                item.treatment,
                item.control,
                item.case_id or item.family,
            )
            for item in self.hypotheses
        )
        if primary_contrasts != _AUTHORITATIVE_PRIMARY_CONTRASTS:
            raise ValueError("primary contrast mapping/order mismatch")
        multiplicity = {item.multiplicity_family for item in self.hypotheses}
        if len(multiplicity) != 1:
            raise ValueError(
                "hypotheses.multiplicity_family must be shared by all four rows"
            )

        by_family = {family.name: family for family in self.families}
        for hypothesis in self.hypotheses:
            if hypothesis.family not in by_family:
                raise ValueError(f"{hypothesis.hypothesis_id}.family is not registered")
            family = by_family[hypothesis.family]
            if hypothesis.case_id is not None and hypothesis.case_id not in {
                case.case_id for case in family.cases
            }:
                raise ValueError(
                    f"{hypothesis.hypothesis_id}.case_id is not registered in "
                    f"{family.name}"
                )
            for field, method in (
                ("treatment", hypothesis.treatment),
                ("control", hypothesis.control),
            ):
                if method not in family.methods:
                    raise ValueError(
                        f"{hypothesis.hypothesis_id}.{field} is not registered "
                        f"in {family.name}.methods"
                    )


@dataclass(frozen=True, slots=True)
class ContrastSpec:
    """Compatibility view of a primary registry contrast."""

    hypothesis_id: str
    treatment_id: str
    control_id: str
    stratum: str

    def __post_init__(self) -> None:
        for field, value in (
            ("hypothesis_id", self.hypothesis_id),
            ("treatment_id", self.treatment_id),
            ("control_id", self.control_id),
            ("stratum", self.stratum),
        ):
            _nonempty_string(value, field)
        if self.treatment_id == self.control_id:
            raise ValueError("treatment_id and control_id must differ")


def _resolve_config_path(value: object, field: str) -> Path:
    raw_path = Path(_nonempty_string(value, field))
    candidate = raw_path if raw_path.is_absolute() else _REPOSITORY_ROOT / raw_path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(_REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError(f"{field} must resolve beneath the repository root") from exc
    if candidate.is_symlink() or not resolved.is_file():
        raise ValueError(f"{field} must resolve to a regular file")
    return resolved


def _table_array(value: object, field: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty table array")
    result: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{field}[{index}] must be a table")
        result.append(item)
    return tuple(result)


def _parse_case(value: Mapping[str, object], field: str) -> RegisteredCase:
    fixed = value.get("fixed_sample_size")
    if fixed is not None:
        fixed = _positive_int(fixed, f"{field}.fixed_sample_size")
    return RegisteredCase(
        case_id=_nonempty_string(value.get("case_id"), f"{field}.case_id"),
        role=_nonempty_string(value.get("role"), f"{field}.role"),
        fixed_sample_size=fixed,
    )


def _range_from_raw(
    raw: Mapping[str, object],
    prefix: str,
) -> tuple[int, ...]:
    start = _nonnegative_int(raw.get(f"{prefix}_start"), f"{prefix}_start")
    count = _positive_int(raw.get(f"{prefix}_count"), f"{prefix}_count")
    return tuple(range(start, start + count))


def load_registry(path: Path) -> ExperimentRegistry:
    """Parse and validate one immutable thesis-v2 TOML registry."""
    registry_path = Path(path)
    if registry_path.is_symlink() or not registry_path.is_file():
        raise ValueError("registry path must be a regular file")
    try:
        raw = tomllib.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError("registry TOML is invalid") from exc

    raw_candidates = raw.get("candidate_sample_sizes")
    if not isinstance(raw_candidates, list):
        raise ValueError("candidate_sample_sizes must be a list")
    candidates = tuple(
        _positive_int(item, "candidate_sample_sizes") for item in raw_candidates
    )

    method_factor_hashes: dict[str, str] = {}
    for index, value in enumerate(
        _table_array(raw.get("method_specs"), "method_specs")
    ):
        field = f"method_specs[{index}]"
        method_id = _nonempty_string(
            value.get("method_id"),
            f"{field}.method_id",
        )
        if method_id in method_factor_hashes:
            raise ValueError(f"duplicate method_specs method_id {method_id}")
        method_factor_hashes[method_id] = _nonempty_string(
            value.get("factor_hash"),
            f"{field}.factor_hash",
        )

    families: list[ExperimentFamily] = []
    for index, value in enumerate(_table_array(raw.get("families"), "families")):
        field = f"families[{index}]"
        name = _nonempty_string(value.get("name"), f"{field}.name")
        raw_cases = _table_array(value.get("cases"), f"{name}.cases")
        families.append(
            ExperimentFamily(
                name=name,
                config_path=_resolve_config_path(
                    value.get("config_path"),
                    f"{name}.config_path",
                ),
                seed_start=_nonnegative_int(
                    value.get("seed_start"),
                    f"{name}.seed_start",
                ),
                seed_count=_positive_int(
                    value.get("seed_count"),
                    f"{name}.seed_count",
                ),
                num_slots=_positive_int(
                    value.get("num_slots"),
                    f"{name}.num_slots",
                ),
                methods=_string_tuple(value.get("methods"), f"{name}.methods"),
                sample_size_policy=_nonempty_string(
                    value.get("sample_size_policy"),
                    f"{name}.sample_size_policy",
                ),
                formal_order=_positive_int(
                    value.get("formal_order"),
                    f"{name}.formal_order",
                ),
                cases=tuple(
                    _parse_case(case, f"{name}.cases[{case_index}]")
                    for case_index, case in enumerate(raw_cases)
                ),
            )
        )

    hypotheses: list[PrimaryHypothesis] = []
    for index, value in enumerate(_table_array(raw.get("hypotheses"), "hypotheses")):
        field = f"hypotheses[{index}]"
        hypothesis_id = _nonempty_string(
            value.get("hypothesis_id"),
            f"{field}.hypothesis_id",
        )
        raw_mde = value.get("mde_pp")
        if (
            not isinstance(raw_mde, (int, float))
            or isinstance(raw_mde, bool)
            or not math.isfinite(float(raw_mde))
            or float(raw_mde) <= 0.0
        ):
            raise ValueError(f"{hypothesis_id}.mde_pp must be positive")
        case_id = value.get("case_id")
        if case_id is not None:
            case_id = _nonempty_string(case_id, f"{hypothesis_id}.case_id")
        hypotheses.append(
            PrimaryHypothesis(
                hypothesis_id=hypothesis_id,
                family=_nonempty_string(value.get("family"), f"{hypothesis_id}.family"),
                case_id=case_id,
                treatment=_nonempty_string(
                    value.get("treatment"), f"{hypothesis_id}.treatment"
                ),
                control=_nonempty_string(
                    value.get("control"), f"{hypothesis_id}.control"
                ),
                outcome=_nonempty_string(
                    value.get("outcome"), f"{hypothesis_id}.outcome"
                ),
                estimand=_nonempty_string(
                    value.get("estimand"), f"{hypothesis_id}.estimand"
                ),
                primary_test=_nonempty_string(
                    value.get("primary_test"),
                    f"{hypothesis_id}.primary_test",
                ),
                sensitivity_test=_nonempty_string(
                    value.get("sensitivity_test"),
                    f"{hypothesis_id}.sensitivity_test",
                ),
                multiplicity_family=_nonempty_string(
                    value.get("multiplicity_family"),
                    f"{hypothesis_id}.multiplicity_family",
                ),
                mde_pp=float(raw_mde),
            )
        )

    return ExperimentRegistry(
        development_seeds=_range_from_raw(raw, "development_seed"),
        pilot_seeds=_range_from_raw(raw, "pilot_seed"),
        candidate_sample_sizes=candidates,
        descriptive_externality_n=_positive_int(
            raw.get("descriptive_externality_n"),
            "descriptive_externality_n",
        ),
        method_factor_hashes=method_factor_hashes,
        families=tuple(families),
        hypotheses=tuple(hypotheses),
    )


def registered_family(
    registry: ExperimentRegistry,
    name: str,
) -> ExperimentFamily:
    """Return the unique named family or raise ``KeyError(name)``."""
    for family in registry.families:
        if family.name == name:
            return family
    raise KeyError(name)


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        if name not in value:
            raise ValueError(f"definition.{name} is missing")
        return value[name]
    try:
        return getattr(value, name)
    except AttributeError as exc:
        raise ValueError(f"definition.{name} is missing") from exc


def _config_contract(
    family: ExperimentFamily,
) -> tuple[
    int,
    int,
    int,
    tuple[str, ...],
    str,
    tuple[RegisteredCase, ...],
]:
    try:
        raw = tomllib.loads(family.config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"{family.name}.config_path is invalid") from exc
    declared_family = _nonempty_string(
        raw.get("formal_family"),
        f"{family.name}.config.formal_family",
    )
    if declared_family != family.name:
        raise ValueError(f"{family.name}.config.formal_family mismatch")
    scenario = raw.get("scenario")
    if not isinstance(scenario, Mapping):
        raise ValueError(f"{family.name}.config.scenario is missing")
    metadata_required = raw.get("method_metadata_required", False)
    if not isinstance(metadata_required, bool):
        raise ValueError(
            f"{family.name}.config.method_metadata_required must be bool"
        )
    if metadata_required:
        specs = thesis_method_specs()
        expected_hashes = {
            method_id: method_spec_sha256(specs[method_id])
            for method_id in family.methods
        }
        declared_hashes = raw.get("method_factor_hashes")
        if declared_hashes != expected_hashes:
            raise ValueError(
                f"{family.name}.config.method_factor_hashes mismatch"
            )
        expected_sources = {
            method_id: specs[method_id].fixed_radio_source
            for method_id in family.methods
            if specs[method_id].fixed_radio_source is not None
        }
        declared_sources = raw.get("fixed_radio_sources")
        if (
            expected_sources
            and declared_sources is None
            or declared_sources is not None
            and declared_sources != expected_sources
        ):
            raise ValueError(
                f"{family.name}.config.fixed_radio_sources mismatch"
            )
    raw_cases = _table_array(raw.get("cases"), f"{family.name}.config.cases")
    return (
        _nonnegative_int(
            raw.get("seed_start"),
            f"{family.name}.config.seed_start",
        ),
        _positive_int(
            raw.get("seed_count"),
            f"{family.name}.config.seed_count",
        ),
        _positive_int(
            scenario.get("num_slots"),
            f"{family.name}.config.num_slots",
        ),
        _string_tuple(raw.get("methods"), f"{family.name}.config.methods"),
        _nonempty_string(
            raw.get("sample_size_policy"),
            f"{family.name}.config.sample_size_policy",
        ),
        tuple(
            _parse_case(case, f"{family.name}.config.cases[{index}]")
            for index, case in enumerate(raw_cases)
        ),
    )


def validate_registered_definition(
    registry: ExperimentRegistry,
    family_name: str,
    definition: object,
) -> None:
    """Reject the first registry, config, or resolved-definition drift."""
    family = registered_family(registry, family_name)
    (
        config_seed_start,
        config_seed_count,
        config_num_slots,
        config_methods,
        config_policy,
        config_cases,
    ) = _config_contract(family)
    comparisons = (
        ("seed_start", config_seed_start, family.seed_start),
        ("seed_count", config_seed_count, family.seed_count),
        ("num_slots", config_num_slots, family.num_slots),
        ("methods", config_methods, family.methods),
        ("sample_size_policy", config_policy, family.sample_size_policy),
        ("cases", config_cases, family.cases),
    )
    for field, actual, expected in comparisons:
        if actual != expected:
            raise ValueError(f"{family.name}.config.{field} mismatch")

    seeds = tuple(_field(definition, "seeds"))
    if seeds != family.seeds:
        raise ValueError(f"{family.name}.seeds mismatch")
    methods = tuple(_field(definition, "methods"))
    if methods != family.methods:
        raise ValueError(f"{family.name}.methods mismatch")
    scenario = _field(definition, "scenario")
    num_slots = _field(scenario, "num_slots")
    if num_slots != family.num_slots:
        raise ValueError(f"{family.name}.num_slots mismatch")
    sample_size_policy = _field(definition, "sample_size_policy")
    if sample_size_policy != family.sample_size_policy:
        raise ValueError(f"{family.name}.sample_size_policy mismatch")
    raw_cases = tuple(_field(definition, "cases"))
    cases = tuple(
        (
            _field(case, "case_id"),
            _field(case, "role"),
            _field(case, "fixed_sample_size"),
        )
        for case in raw_cases
    )
    expected_cases = tuple(
        (case.case_id, case.role, case.fixed_sample_size) for case in family.cases
    )
    if cases != expected_cases:
        raise ValueError(f"{family.name}.cases mismatch")

    try:
        definition_path = _field(definition, "config_path")
    except ValueError:
        definition_path = None
    if definition_path is not None:
        resolved = Path(definition_path).resolve()
        if resolved != family.config_path:
            raise ValueError(f"{family.name}.config_path mismatch")


def thesis_contrasts(
    registry: ExperimentRegistry | None = None,
) -> dict[str, ContrastSpec]:
    """Return the H1--H4 treatment/control mapping in registry order."""
    active = registry or load_registry(
        _REPOSITORY_ROOT / "configs/thesis/experiment_registry_v2.toml"
    )
    return {
        hypothesis.hypothesis_id: ContrastSpec(
            hypothesis_id=hypothesis.hypothesis_id,
            treatment_id=hypothesis.treatment,
            control_id=hypothesis.control,
            stratum=hypothesis.case_id or hypothesis.family,
        )
        for hypothesis in active.hypotheses
    }


__all__ = [
    "ContrastSpec",
    "ExperimentFamily",
    "ExperimentRegistry",
    "PrimaryHypothesis",
    "RegisteredCase",
    "load_registry",
    "registered_family",
    "thesis_contrasts",
    "validate_registered_definition",
]
