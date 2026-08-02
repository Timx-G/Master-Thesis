"""Configuration-driven trusted experiment runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path

import numpy as np
import scipy

from .artifacts import RunArtifactWriter
from .baselines_adapted import (
    GUO_DEFAULT_QOS_SPECTRAL_EFFICIENCY,
    guo_jagp_adapted,
)
from .baselines_v2 import (
    evaluate_fair_ofdma,
    greedy_schedule,
    ia_greedy_schedule,
    local_only_schedule,
)
from .closure import close_assignment, freeze_plan, rate_raw_assignment
from .domain import ChannelBasis, Schedule, SlotOutcome, SlotSnapshot
from .evaluation import (
    EvaluationConfig,
    build_outcome_from_realized,
    evaluate_frozen_plan,
)
from .externality import aggregate_victim_link_exposure
from .model import NOMAChannel, ScenarioConfig
from .resource_allocation import AllocationPolicy
from .scheduler import (
    Acceptance,
    SearchConfig,
    build_initial_schedule,
    network_objective,
    optimize_schedule,
    repair_sic,
)
from .simulation import run_outcome_episode
from .seed_access import (
    SeedAccessRecorder,
    development_permit,
    pilot_permit,
)
from .trace import generate_trace

_CANONICAL_PHYSICS = "canonical_noma_v2"
_CANONICAL_LATENCY = "canonical_task_latency_v2"
_POLICY_CLOSURE = "policy_closure"
_FROZEN_RATE_ADMISSION = "frozen_rate_conditional_admission"
_SHARED_INITIALIZERS = ("per_uav_ilp", "ia_greedy", "greedy")
_WIRELESS_KERNELS = frozenset(
    {
        "local_only",
        "fair_ofdma",
        "greedy",
        "ia_greedy",
        "per_uav_ilp",
        "exchange_stable",
        "iea_cap",
        "guo_jagp_adapted",
        "frozen_radio_snapshot",
        "exact_admission",
        "exact_schedule",
        "iea_cap_deadline_search",
    }
)
_INITIALIZER_IDS = frozenset(
    {"per_uav_ilp", "ia_greedy", "greedy", "guo_jagp_adapted"}
)
_NEIGHBORHOOD_SCOPES = frozenset(
    {"none", "intra_uav", "cross_uav", "global", "kernel_native"}
)
_ACCEPTANCE_RULES = frozenset(
    {"none", "bilateral", "global_lexicographic", "exact", "kernel_native"}
)
_TRAVERSAL_RULES = frozenset(
    {"none", "direct", "first_improvement", "exhaustive"}
)
_FIXED_RADIO_SOURCES = frozenset(
    {"dd_generated", "radio_neutral", "deadline_first_generated"}
)
_ADAPTATION_IDS = frozenset({"guo_jagp_to_canonical_noma_v2"})
_PHYSICS_MODELS = frozenset(
    {"local_compute_v2", "canonical_ofdma_v2", _CANONICAL_PHYSICS}
)
_LATENCY_MODELS = frozenset({_CANONICAL_LATENCY})
_CLOSURE_MODES = frozenset(
    {
        "local_execution",
        "direct_policy_allocation",
        _POLICY_CLOSURE,
        _FROZEN_RATE_ADMISSION,
        "frozen_rate_exact_admission",
    }
)
_SEARCH_KERNELS = frozenset(
    {"iea_cap", "exchange_stable", "iea_cap_deadline_search"}
)
_DIRECT_NOMA_KERNELS = frozenset(
    {"greedy", "ia_greedy", "per_uav_ilp", "guo_jagp_adapted"}
)
_DEDICATED_KERNELS = frozenset(
    {"frozen_radio_snapshot", "exact_admission", "exact_schedule"}
)


@dataclass(frozen=True, slots=True)
class MethodSpec:
    """Complete, immutable declaration of one auditable method."""

    method_id: str
    display_name: str
    wireless_kernel: str
    initializer_ids: tuple[str, ...]
    neighborhood_scope: str
    acceptance_rule: str
    traversal_rule: str
    allocation_policy: AllocationPolicy
    fixed_radio_source: str | None
    adaptation_id: str | None
    physics_model_id: str
    latency_model_id: str
    closure_mode: str
    realized_replay: bool

    def __post_init__(self) -> None:
        for field_name in (
            "method_id",
            "display_name",
            "wireless_kernel",
            "neighborhood_scope",
            "acceptance_rule",
            "traversal_rule",
            "physics_model_id",
            "latency_model_id",
            "closure_mode",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        for field_name, value, allowed in (
            ("wireless_kernel", self.wireless_kernel, _WIRELESS_KERNELS),
            (
                "neighborhood_scope",
                self.neighborhood_scope,
                _NEIGHBORHOOD_SCOPES,
            ),
            ("acceptance_rule", self.acceptance_rule, _ACCEPTANCE_RULES),
            ("traversal_rule", self.traversal_rule, _TRAVERSAL_RULES),
            ("physics_model_id", self.physics_model_id, _PHYSICS_MODELS),
            ("latency_model_id", self.latency_model_id, _LATENCY_MODELS),
            ("closure_mode", self.closure_mode, _CLOSURE_MODES),
        ):
            if value not in allowed:
                raise ValueError(f"unsupported {field_name} {value}")
        if not isinstance(self.initializer_ids, tuple):
            raise ValueError("initializer_ids must be stored as a tuple")
        for initializer_id in self.initializer_ids:
            if not isinstance(initializer_id, str) or not initializer_id.strip():
                raise ValueError("initializer_ids must contain non-empty strings")
        if len(set(self.initializer_ids)) != len(self.initializer_ids):
            raise ValueError("initializer_ids must be unique and ordered")
        if not set(self.initializer_ids) <= _INITIALIZER_IDS:
            raise ValueError("initializer_ids contain an unsupported initializer")
        if not isinstance(self.allocation_policy, AllocationPolicy):
            raise ValueError("allocation_policy must be an AllocationPolicy")
        for field_name in ("fixed_radio_source", "adaptation_id"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be None or non-empty")
        if (
            self.fixed_radio_source is not None
            and self.fixed_radio_source not in _FIXED_RADIO_SOURCES
        ):
            raise ValueError(
                f"unsupported fixed_radio_source {self.fixed_radio_source}"
            )
        if self.adaptation_id is not None and self.adaptation_id not in _ADAPTATION_IDS:
            raise ValueError(f"unsupported adaptation_id {self.adaptation_id}")
        if not isinstance(self.realized_replay, bool):
            raise ValueError("realized_replay must be bool")
        self._validate_compatible_factors()

    def _validate_compatible_factors(self) -> None:
        if self.wireless_kernel in _SEARCH_KERNELS:
            if not self.initializer_ids or self.traversal_rule != "first_improvement":
                raise ValueError("search kernels require initializers and first improvement")
            if self.closure_mode != _POLICY_CLOSURE:
                raise ValueError("search kernels require policy closure")
            if self.physics_model_id != _CANONICAL_PHYSICS or not self.realized_replay:
                raise ValueError("search kernels require canonical realized replay")
            if self.neighborhood_scope == "kernel_native":
                if (
                    self.wireless_kernel not in {"iea_cap", "exchange_stable"}
                    or self.acceptance_rule != "kernel_native"
                ):
                    raise ValueError("kernel_native factors require a native search kernel")
            elif (
                self.neighborhood_scope not in {"intra_uav", "cross_uav"}
                or self.acceptance_rule
                not in {"bilateral", "global_lexicographic"}
            ):
                raise ValueError("search scope and acceptance factors are incompatible")
        elif self.wireless_kernel in _DIRECT_NOMA_KERNELS:
            if (
                not self.initializer_ids
                or self.neighborhood_scope != "none"
                or self.acceptance_rule != "none"
                or self.traversal_rule != "direct"
                or self.closure_mode != _POLICY_CLOSURE
                or self.physics_model_id != _CANONICAL_PHYSICS
                or not self.realized_replay
            ):
                raise ValueError("direct NOMA kernel factors are incompatible")
        elif self.wireless_kernel == "frozen_radio_snapshot":
            if (
                self.initializer_ids
                or self.neighborhood_scope != "none"
                or self.acceptance_rule != "none"
                or self.traversal_rule != "none"
                or self.fixed_radio_source is None
                or self.closure_mode != _FROZEN_RATE_ADMISSION
                or self.physics_model_id != _CANONICAL_PHYSICS
                or not self.realized_replay
            ):
                raise ValueError("frozen-radio factors are incompatible")
        elif self.wireless_kernel == "exact_admission":
            if (
                self.initializer_ids
                or self.neighborhood_scope != "none"
                or self.acceptance_rule != "none"
                or self.traversal_rule != "exhaustive"
                or self.fixed_radio_source is not None
                or self.closure_mode != "frozen_rate_exact_admission"
                or self.physics_model_id != _CANONICAL_PHYSICS
                or not self.realized_replay
            ):
                raise ValueError("exact-admission factors are incompatible")
        elif self.wireless_kernel == "exact_schedule":
            if (
                self.initializer_ids
                or self.neighborhood_scope != "global"
                or self.acceptance_rule != "exact"
                or self.traversal_rule != "exhaustive"
                or self.fixed_radio_source is not None
                or self.closure_mode != _POLICY_CLOSURE
                or self.physics_model_id != _CANONICAL_PHYSICS
                or self.realized_replay
            ):
                raise ValueError("exact-schedule factors are incompatible")
        elif self.wireless_kernel == "local_only":
            if (
                self.initializer_ids
                or self.neighborhood_scope != "none"
                or self.acceptance_rule != "none"
                or self.traversal_rule != "direct"
                or self.closure_mode != "local_execution"
                or self.physics_model_id != "local_compute_v2"
                or not self.realized_replay
            ):
                raise ValueError("local-only factors are incompatible")
        elif self.wireless_kernel == "fair_ofdma" and (
            self.initializer_ids
            or self.neighborhood_scope != "none"
            or self.acceptance_rule != "none"
            or self.traversal_rule != "direct"
            or self.closure_mode != "direct_policy_allocation"
            or self.physics_model_id != "canonical_ofdma_v2"
            or not self.realized_replay
        ):
            raise ValueError("Fair-OFDMA factors are incompatible")

        if self.wireless_kernel != "frozen_radio_snapshot" and self.fixed_radio_source:
            raise ValueError("only frozen-radio methods may name a fixed source")
        if (self.wireless_kernel == "guo_jagp_adapted") != (
            self.adaptation_id is not None
        ):
            raise ValueError("adaptation_id must identify only the adapted kernel")


def _spec(
    method_id: str,
    display_name: str,
    wireless_kernel: str,
    initializer_ids: tuple[str, ...],
    neighborhood_scope: str,
    acceptance_rule: str,
    traversal_rule: str,
    allocation_policy: AllocationPolicy = AllocationPolicy.DEADLINE,
    fixed_radio_source: str | None = None,
    adaptation_id: str | None = None,
    physics_model_id: str = _CANONICAL_PHYSICS,
    latency_model_id: str = _CANONICAL_LATENCY,
    closure_mode: str = _POLICY_CLOSURE,
    realized_replay: bool = True,
) -> MethodSpec:
    return MethodSpec(
        method_id,
        display_name,
        wireless_kernel,
        initializer_ids,
        neighborhood_scope,
        acceptance_rule,
        traversal_rule,
        allocation_policy,
        fixed_radio_source,
        adaptation_id,
        physics_model_id,
        latency_model_id,
        closure_mode,
        realized_replay,
    )


def _fixed_radio_spec(
    method_id: str,
    display_name: str,
    source: str,
    policy: AllocationPolicy,
) -> MethodSpec:
    return _spec(
        method_id,
        display_name,
        "frozen_radio_snapshot",
        (),
        "none",
        "none",
        "none",
        policy,
        source,
        closure_mode=_FROZEN_RATE_ADMISSION,
    )


def thesis_method_specs() -> dict[str, MethodSpec]:
    """Return the single validated method-factor registry."""
    specs = (
        _spec(
            "local_only",
            "Local-only",
            "local_only",
            (),
            "none",
            "none",
            "direct",
            physics_model_id="local_compute_v2",
            closure_mode="local_execution",
        ),
        _spec(
            "fair_ofdma",
            "Fair-OFDMA",
            "fair_ofdma",
            (),
            "none",
            "none",
            "direct",
            physics_model_id="canonical_ofdma_v2",
            closure_mode="direct_policy_allocation",
        ),
        _spec("greedy", "Greedy", "greedy", ("greedy",), "none", "none", "direct"),
        _spec(
            "ia_greedy",
            "IA-Greedy",
            "ia_greedy",
            ("ia_greedy",),
            "none",
            "none",
            "direct",
        ),
        _spec(
            "per_uav_ilp",
            "per-UAV ILP",
            "per_uav_ilp",
            ("per_uav_ilp",),
            "none",
            "none",
            "direct",
        ),
        _spec(
            "exchange_stable",
            "Exchange-Stable",
            "exchange_stable",
            _SHARED_INITIALIZERS,
            "intra_uav",
            "bilateral",
            "first_improvement",
        ),
        _spec(
            "intra_global",
            "Intra-Global",
            "iea_cap",
            _SHARED_INITIALIZERS,
            "intra_uav",
            "global_lexicographic",
            "first_improvement",
        ),
        _spec(
            "cross_bilateral",
            "Cross-Bilateral",
            "iea_cap",
            _SHARED_INITIALIZERS,
            "cross_uav",
            "bilateral",
            "first_improvement",
        ),
        _spec(
            "single_start_first",
            "single-start first-improvement",
            "iea_cap",
            ("per_uav_ilp",),
            "cross_uav",
            "global_lexicographic",
            "first_improvement",
        ),
        _spec(
            "full",
            "full IEA-CAP",
            "iea_cap",
            _SHARED_INITIALIZERS,
            "cross_uav",
            "global_lexicographic",
            "first_improvement",
        ),
        _spec(
            "guo_jagp_adapted",
            "Guo-JAGP adapted",
            "guo_jagp_adapted",
            ("guo_jagp_adapted",),
            "none",
            "none",
            "direct",
            adaptation_id="guo_jagp_to_canonical_noma_v2",
        ),
        _fixed_radio_spec(
            "fixed_radio_dd",
            "DD-MCAA (static share)",
            "dd_generated",
            AllocationPolicy.DEADLINE,
        ),
        _fixed_radio_spec(
            "fixed_radio_edf",
            "Deadline-First-MinCPU (static share)",
            "dd_generated",
            AllocationPolicy.EDF_MIN_CPU,
        ),
        _fixed_radio_spec(
            "fixed_radio_dd_radio_neutral",
            "DD-MCAA (radio-neutral static share)",
            "radio_neutral",
            AllocationPolicy.DEADLINE,
        ),
        _fixed_radio_spec(
            "fixed_radio_edf_radio_neutral",
            "Deadline-First-MinCPU (radio-neutral static share)",
            "radio_neutral",
            AllocationPolicy.EDF_MIN_CPU,
        ),
        _fixed_radio_spec(
            "fixed_radio_dd_deadline_generated",
            "DD-MCAA (deadline-first-generated static share)",
            "deadline_first_generated",
            AllocationPolicy.DEADLINE,
        ),
        _fixed_radio_spec(
            "fixed_radio_edf_deadline_generated",
            "Deadline-First-MinCPU (deadline-first-generated static share)",
            "deadline_first_generated",
            AllocationPolicy.EDF_MIN_CPU,
        ),
        _spec(
            "iea_dd",
            "IEA-CAP + DD-MCAA static-share admission",
            "iea_cap",
            _SHARED_INITIALIZERS,
            "kernel_native",
            "kernel_native",
            "first_improvement",
            AllocationPolicy.DEADLINE,
        ),
        _spec(
            "iea_edf",
            "IEA-CAP + non-preemptive Deadline-First-MinCPU static-share admission",
            "iea_cap",
            _SHARED_INITIALIZERS,
            "kernel_native",
            "kernel_native",
            "first_improvement",
            AllocationPolicy.EDF_MIN_CPU,
        ),
        _spec(
            "exchange_dd",
            "Exchange-Stable + DD-MCAA static-share admission",
            "exchange_stable",
            _SHARED_INITIALIZERS,
            "kernel_native",
            "kernel_native",
            "first_improvement",
            AllocationPolicy.DEADLINE,
        ),
        _spec(
            "exchange_edf",
            "Exchange-Stable + non-preemptive Deadline-First-MinCPU static-share admission",
            "exchange_stable",
            _SHARED_INITIALIZERS,
            "kernel_native",
            "kernel_native",
            "first_improvement",
            AllocationPolicy.EDF_MIN_CPU,
        ),
        _spec(
            "exact_admission",
            "Exact Admission",
            "exact_admission",
            (),
            "none",
            "none",
            "exhaustive",
            closure_mode="frozen_rate_exact_admission",
        ),
        _spec(
            "exact",
            "Exact Schedule",
            "exact_schedule",
            (),
            "global",
            "exact",
            "exhaustive",
            realized_replay=False,
        ),
        # Compatibility controls retained for the original thesis suites.
        _spec(
            "ilp",
            "per-UAV ILP",
            "per_uav_ilp",
            ("per_uav_ilp",),
            "none",
            "none",
            "direct",
        ),
        _spec(
            "intra_bilateral",
            "Intra-Bilateral",
            "iea_cap",
            _SHARED_INITIALIZERS,
            "intra_uav",
            "bilateral",
            "first_improvement",
        ),
        _spec(
            "full_fixed_half",
            "full IEA-CAP + fixed-half equal allocation",
            "iea_cap_deadline_search",
            _SHARED_INITIALIZERS,
            "cross_uav",
            "global_lexicographic",
            "first_improvement",
            AllocationPolicy.FIXED_HALF_EQUAL,
        ),
        _spec(
            "full_crossing",
            "full IEA-CAP + crossing equal allocation",
            "iea_cap_deadline_search",
            _SHARED_INITIALIZERS,
            "cross_uav",
            "global_lexicographic",
            "first_improvement",
            AllocationPolicy.CROSSING_EQUAL,
        ),
    )
    result = {spec.method_id: spec for spec in specs}
    if len(result) != len(specs):
        raise RuntimeError("method registry contains duplicate method IDs")

    ignored = {"method_id", "display_name"}

    def differences(left: MethodSpec, right: MethodSpec) -> set[str]:
        return {
            field.name
            for field in MethodSpec.__dataclass_fields__.values()
            if field.name not in ignored
            and getattr(left, field.name) != getattr(right, field.name)
        }

    if differences(result["full"], result["intra_global"]) != {"neighborhood_scope"}:
        raise RuntimeError("H1 methods must isolate neighborhood_scope")
    if differences(result["full"], result["cross_bilateral"]) != {"acceptance_rule"}:
        raise RuntimeError("H2 methods must isolate acceptance_rule")
    joint_ids = ("iea_dd", "iea_edf", "exchange_dd", "exchange_edf")
    joint_differences = {
        field.name
        for field in MethodSpec.__dataclass_fields__.values()
        if field.name not in ignored
        and len({getattr(result[method_id], field.name) for method_id in joint_ids}) > 1
    }
    if joint_differences != {"wireless_kernel", "allocation_policy"}:
        raise RuntimeError("joint methods must isolate the exact two factors")
    return result


KNOWN_METHODS = frozenset(thesis_method_specs())


@dataclass(frozen=True, slots=True)
class ExperimentCase:
    """Registry-facing projection of one case declared by a config."""

    case_id: str
    role: str
    fixed_sample_size: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id must be a non-empty string")
        if self.role not in {"primary", "supporting", "correctness"}:
            raise ValueError("case role must be primary, supporting, or correctness")
        if self.fixed_sample_size is not None and (
            not isinstance(self.fixed_sample_size, int)
            or isinstance(self.fixed_sample_size, bool)
            or self.fixed_sample_size <= 0
        ):
            raise ValueError("fixed_sample_size must be a positive integer")


@dataclass(frozen=True, slots=True)
class ExperimentDefinition:
    run_id: str
    seeds: tuple[int, ...]
    methods: tuple[str, ...]
    scenario: ScenarioConfig
    evaluation: EvaluationConfig
    max_iterations: int
    csi_error_std: float = 0.0
    sample_size_policy: str | None = None
    cases: tuple[ExperimentCase, ...] = ()


def load_definition(path: Path) -> ExperimentDefinition:
    config_path = Path(path)
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    run_id = raw.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")

    has_explicit_seeds = "seeds" in raw
    has_seed_range = "seed_start" in raw or "seed_count" in raw
    if has_explicit_seeds == has_seed_range:
        raise ValueError("define seeds or seed_start/seed_count, but not both")
    if has_explicit_seeds:
        raw_seeds = raw["seeds"]
        if not isinstance(raw_seeds, list) or any(
            not isinstance(seed, int) or isinstance(seed, bool) for seed in raw_seeds
        ):
            raise ValueError("seeds must be an integer list")
        seeds = tuple(raw_seeds)
    else:
        seed_start = raw.get("seed_start")
        seed_count = raw.get("seed_count")
        if (
            not isinstance(seed_start, int)
            or isinstance(seed_start, bool)
            or not isinstance(seed_count, int)
            or isinstance(seed_count, bool)
            or seed_count <= 0
        ):
            raise ValueError("seed_start and seed_count must be valid integers")
        seeds = tuple(range(seed_start, seed_start + seed_count))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative")

    raw_methods = raw.get("methods")
    if not isinstance(raw_methods, list) or any(
        not isinstance(method, str) or not method for method in raw_methods
    ):
        raise ValueError("methods must be a string list")
    methods = tuple(raw_methods)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be non-empty and unique")
    unknown = sorted(set(methods) - KNOWN_METHODS)
    if unknown:
        raise ValueError(f"unknown method {unknown[0]}")
    specs = thesis_method_specs()
    metadata_required = raw.get("method_metadata_required", False)
    if not isinstance(metadata_required, bool):
        raise ValueError("method_metadata_required must be bool")
    expected_hashes = {
        method_id: method_spec_sha256(specs[method_id]) for method_id in methods
    }
    declared_hashes = raw.get("method_factor_hashes")
    if declared_hashes is None:
        if metadata_required:
            raise ValueError("method_factor_hashes table is required")
    else:
        if not isinstance(declared_hashes, dict):
            raise ValueError("method_factor_hashes must be a table")
        if declared_hashes != expected_hashes:
            raise ValueError("method_factor_hashes mismatch")
    expected_sources = {
        method_id: specs[method_id].fixed_radio_source
        for method_id in methods
        if specs[method_id].fixed_radio_source is not None
    }
    declared_sources = raw.get("fixed_radio_sources")
    if declared_sources is None:
        if metadata_required and expected_sources:
            raise ValueError("fixed_radio_sources table is required")
    else:
        if not isinstance(declared_sources, dict):
            raise ValueError("fixed_radio_sources must be a table")
        if declared_sources != expected_sources:
            raise ValueError("fixed_radio_sources mismatch")

    scenario_raw = raw.get("scenario")
    if not isinstance(scenario_raw, dict):
        raise ValueError("scenario section is required")
    scenario = ScenarioConfig(**scenario_raw)

    evaluation_raw = raw.get("evaluation")
    if not isinstance(evaluation_raw, dict):
        raise ValueError("evaluation section is required")
    try:
        allocation_policy = AllocationPolicy(evaluation_raw["allocation_policy"])
    except (KeyError, ValueError) as exc:
        raise ValueError("invalid evaluation allocation_policy") from exc
    evaluation = EvaluationConfig(
        n_subcarriers=scenario.n_subcarriers,
        bandwidth_hz=scenario.channel.bandwidth,
        transmit_power_w=scenario.channel.p_max,
        sic_residual=float(evaluation_raw["sic_residual"]),
        gamma_sic_min=float(evaluation_raw["gamma_sic_min"]),
        cpu_per_uav_hz=scenario.f_per_uav,
        allocation_policy=allocation_policy,
        cross_interference_scale=float(
            evaluation_raw.get("cross_interference_scale", 1.0)
        ),
    )

    search_raw = raw.get("search")
    if not isinstance(search_raw, dict):
        raise ValueError("search section is required")
    max_iterations = search_raw.get("max_iterations")
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
        raise ValueError("max_iterations must be an integer")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    trace_raw = raw.get("trace", {})
    if not isinstance(trace_raw, dict):
        raise ValueError("trace section must be a table")
    csi_error_std = float(trace_raw.get("csi_error_std", 0.0))
    if not np.isfinite(csi_error_std) or csi_error_std < 0.0:
        raise ValueError("csi_error_std must be finite and non-negative")

    sample_size_policy = raw.get("sample_size_policy")
    if sample_size_policy is not None and (
        not isinstance(sample_size_policy, str) or not sample_size_policy.strip()
    ):
        raise ValueError("sample_size_policy must be a non-empty string")
    raw_cases = raw.get("cases", [])
    if not isinstance(raw_cases, list):
        raise ValueError("cases must be a table array")
    cases: list[ExperimentCase] = []
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError(f"cases[{index}] must be a table")
        cases.append(
            ExperimentCase(
                case_id=raw_case.get("case_id"),
                role=raw_case.get("role"),
                fixed_sample_size=raw_case.get("fixed_sample_size"),
            )
        )
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("case_id values must be unique")
    return ExperimentDefinition(
        run_id=run_id,
        seeds=seeds,
        methods=methods,
        scenario=scenario,
        evaluation=evaluation,
        max_iterations=max_iterations,
        csi_error_std=csi_error_std,
        sample_size_policy=sample_size_policy,
        cases=tuple(cases),
    )


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _search_config_for_spec(
    spec: MethodSpec,
    *,
    max_iterations: int,
) -> SearchConfig:
    """Resolve explicit or kernel-native search factors without method-ID rules."""
    if not isinstance(spec, MethodSpec):
        raise TypeError("spec must be a MethodSpec")
    if spec.wireless_kernel not in _SEARCH_KERNELS:
        raise ValueError(f"{spec.method_id} is not a search method")
    if spec.neighborhood_scope == "kernel_native":
        native = {
            "iea_cap": (True, Acceptance.GLOBAL),
            "exchange_stable": (False, Acceptance.BILATERAL),
        }
        allow_cross_uav, acceptance = native[spec.wireless_kernel]
    else:
        allow_cross_uav = spec.neighborhood_scope == "cross_uav"
        acceptance = {
            "global_lexicographic": Acceptance.GLOBAL,
            "bilateral": Acceptance.BILATERAL,
        }[spec.acceptance_rule]
    return SearchConfig(allow_cross_uav, acceptance, max_iterations)


def _slot_method(
    name: str,
    scenario: ScenarioConfig,
    target_evaluation: EvaluationConfig,
    max_iterations: int,
    event_sink: list[dict[str, object]],
):
    specs = thesis_method_specs()
    if name not in specs:
        raise ValueError(f"unknown method {name}")
    spec = specs[name]
    if spec.wireless_kernel == "exact_schedule":
        raise ValueError("exact requires the dedicated exact benchmark runner")
    if spec.wireless_kernel in {
        "exact_admission",
        "frozen_radio_snapshot",
    }:
        raise ValueError(f"{name} requires the dedicated fixed-radio resource runner")
    channel = NOMAChannel(scenario.channel, scenario.uav.height)
    search_evaluation = (
        replace(
            target_evaluation,
            allocation_policy=AllocationPolicy.DEADLINE,
        )
        if spec.wireless_kernel == "iea_cap_deadline_search"
        else target_evaluation
    )

    def evaluate_noma(
        snapshot: SlotSnapshot,
        schedule: Schedule,
    ) -> SlotOutcome:
        closed = close_assignment(
            snapshot,
            rate_raw_assignment(
                snapshot,
                schedule,
                channel,
                target_evaluation,
                channel_basis=ChannelBasis.PLANNING,
            ),
            channel,
            target_evaluation,
        )
        realized = evaluate_frozen_plan(
            snapshot,
            freeze_plan(closed),
            channel,
            target_evaluation,
        )
        return build_outcome_from_realized(snapshot, realized)

    def method(snapshot: SlotSnapshot) -> SlotOutcome:
        planning = snapshot
        if spec.wireless_kernel == "local_only":
            return evaluate_noma(snapshot, local_only_schedule(planning))
        if spec.wireless_kernel == "fair_ofdma":
            return evaluate_fair_ofdma(
                snapshot,
                channel,
                target_evaluation,
            )
        if spec.wireless_kernel == "greedy":
            schedule = repair_sic(
                greedy_schedule(planning, scenario.n_subcarriers),
                planning,
                channel,
                search_evaluation,
            )
            return evaluate_noma(snapshot, schedule)
        if spec.wireless_kernel == "ia_greedy":
            schedule = ia_greedy_schedule(
                planning,
                channel,
                search_evaluation,
            )
            return evaluate_noma(snapshot, schedule)
        if spec.wireless_kernel == "guo_jagp_adapted":
            schedule = guo_jagp_adapted(
                planning,
                channel,
                search_evaluation,
                channel_basis=ChannelBasis.PLANNING,
                qos_spectral_efficiency=(GUO_DEFAULT_QOS_SPECTRAL_EFFICIENCY),
            )
            return evaluate_noma(snapshot, schedule)

        if spec.wireless_kernel == "per_uav_ilp":
            initialization = build_initial_schedule(
                planning,
                channel,
                search_evaluation,
            )
            return evaluate_noma(snapshot, initialization.schedule)

        def initialize(initializer_id: str) -> tuple[Schedule, str]:
            if initializer_id == "per_uav_ilp":
                initialization = build_initial_schedule(
                    planning,
                    channel,
                    search_evaluation,
                )
                return initialization.schedule, initialization.solver_status
            if initializer_id == "ia_greedy":
                return (
                    ia_greedy_schedule(
                        planning,
                        channel,
                        search_evaluation,
                    ),
                    "deterministic",
                )
            if initializer_id == "greedy":
                return (
                    repair_sic(
                        greedy_schedule(
                            planning,
                            scenario.n_subcarriers,
                        ),
                        planning,
                        channel,
                        search_evaluation,
                    ),
                    "deterministic",
                )
            raise ValueError(f"unknown initializer {initializer_id}")

        if not spec.initializer_ids:
            raise ValueError(f"{name} requires at least one initializer")
        if spec.traversal_rule != "first_improvement":
            raise ValueError(f"unsupported traversal_rule {spec.traversal_rule}")
        search = _search_config_for_spec(
            spec,
            max_iterations=max_iterations,
        )
        candidates = []
        for initializer_id in spec.initializer_ids:
            schedule, solver_status = initialize(initializer_id)
            result = optimize_schedule(
                schedule,
                planning,
                channel,
                search_evaluation,
                search,
            )
            if not result.converged:
                raise RuntimeError(
                    f"search hit max_iterations={max_iterations} "
                    f"at slot {snapshot.slot_index}"
                )
            candidates.append((initializer_id, solver_status, schedule, result))
        initializer_id, solver_status, initial_schedule, result = max(
            candidates,
            key=lambda candidate: network_objective(candidate[3].outcome),
        )
        event_sink.append(
            {
                "slot": snapshot.slot_index,
                "move_type": "initialization",
                "accepted": True,
                "reason": f"{initializer_id}:{solver_status}",
                "before": None,
                "after": None,
                "schedule_key": initial_schedule.stable_key(),
            }
        )
        event_sink.extend(
            {
                "slot": snapshot.slot_index,
                "move_type": event.move_type,
                "accepted": event.accepted,
                "reason": event.reason,
                "before": event.before_objective,
                "after": event.after_objective,
                "schedule_key": event.schedule_key,
            }
            for event in result.events
        )
        event_sink.append(
            {
                "slot": snapshot.slot_index,
                "move_type": "termination",
                "accepted": True,
                "reason": result.termination_reason,
                "before": None,
                "after": result.iterations,
                "schedule_key": result.schedule.stable_key(),
            }
        )
        return evaluate_noma(snapshot, result.schedule)

    return method


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def method_spec_to_dict(spec: MethodSpec) -> dict[str, object]:
    """Serialize every declared factor in a deterministic JSON shape."""
    if not isinstance(spec, MethodSpec):
        raise TypeError("spec must be a MethodSpec")
    value = _jsonable(asdict(spec))
    if not isinstance(value, dict):
        raise TypeError("serialized method spec must be a mapping")
    return value


def method_spec_sha256(spec: MethodSpec) -> str:
    """Hash the complete canonical serialization of one method spec."""
    payload = json.dumps(
        method_spec_to_dict(spec),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def definition_to_dict(
    definition: ExperimentDefinition,
) -> dict[str, object]:
    value = _jsonable(asdict(definition))
    if not isinstance(value, dict):
        raise TypeError("serialized experiment definition must be a mapping")
    return value


def run_definition(
    definition: ExperimentDefinition,
    config_path: Path,
    output_dir: Path,
    *,
    seed_access: SeedAccessRecorder,
) -> None:
    all_specs = thesis_method_specs()
    try:
        selected_specs = {name: all_specs[name] for name in definition.methods}
    except KeyError as exc:
        raise ValueError(f"unknown method {exc.args[0]}") from exc
    dedicated = tuple(
        spec.method_id
        for spec in selected_specs.values()
        if spec.wireless_kernel in _DEDICATED_KERNELS
    )
    if dedicated:
        raise ValueError(
            f"methods {dedicated} require a dedicated experiment runner"
        )
    if not isinstance(seed_access, SeedAccessRecorder):
        raise TypeError("seed_access must be a SeedAccessRecorder")
    seed_access.record(
        "run_definition",
        definition.seeds,
        {
            "run_id": definition.run_id,
            "config_path": str(Path(config_path)),
        },
    )
    writer = RunArtifactWriter(output_dir)
    task_rows: list[dict[str, object]] = []
    slot_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    method_summary = {name: [] for name in definition.methods}
    method_exposure_summary = {name: [] for name in definition.methods}

    for seed in definition.seeds:
        trace = generate_trace(
            definition.scenario,
            seed,
            definition.csi_error_std,
            seed_access=seed_access,
        )
        task_by_id = {
            task.task_id: task for snapshot in trace.slots for task in snapshot.tasks
        }
        for name in definition.methods:
            spec = selected_specs[name]
            target = replace(
                definition.evaluation,
                allocation_policy=spec.allocation_policy,
            )

            events: list[dict[str, object]] = []
            episode = run_outcome_episode(
                trace,
                _slot_method(
                    name,
                    definition.scenario,
                    target,
                    definition.max_iterations,
                    events,
                ),
            )
            method_summary[name].append(episode.on_time_rate)
            method_exposure_summary[name].append(
                math.fsum(
                    aggregate_victim_link_exposure(outcome.rates)
                    for outcome in episode.outcomes
                )
                / max(len(episode.outcomes), 1)
            )
            for outcome in episode.outcomes:
                aggregate_exposure_w = aggregate_victim_link_exposure(outcome.rates)
                plan_key_json = json.dumps(
                    _jsonable(outcome.plan_key),
                    allow_nan=False,
                    separators=(",", ":"),
                )
                outage_ids_json = json.dumps(
                    list(outcome.sic_outage_task_ids),
                    allow_nan=False,
                    separators=(",", ":"),
                )
                decision_by_id = {
                    decision.task_id: decision for decision in outcome.decisions
                }
                rate_by_id = {record.task_id: record for record in outcome.rates}
                for task_id in outcome.arrived_task_ids:
                    task = task_by_id[task_id]
                    decision = decision_by_id[task_id]
                    rate = rate_by_id.get(task_id)
                    task_rows.append(
                        {
                            "method": name,
                            "seed": seed,
                            "slot": outcome.slot_index,
                            "task_id": task_id,
                            "user_id": task.user_id,
                            "arrival_slot": task.arrival_slot,
                            "data_bits": task.data_bits,
                            "comp_cycles": task.comp_cycles,
                            "deadline_s": task.deadline_s,
                            "local_cpu_hz": task.local_cpu_hz,
                            "completed": decision.completed,
                            "delay_s": decision.delay_s,
                            "alpha": decision.alpha,
                            "cpu_hz": decision.cpu_hz,
                            "rate_bps": None if rate is None else rate.rate_bps,
                            "sinr": None if rate is None else rate.sinr,
                            "sic_feasible": (
                                None if rate is None else rate.sic_feasible
                            ),
                            "cross_interference_w": (
                                None if rate is None else rate.cross_interference_w
                            ),
                            "serving_uav": (None if rate is None else rate.serving_uav),
                            "subcarrier": (None if rate is None else rate.subcarrier),
                            "reason": decision.reason,
                            "plan_key": plan_key_json,
                            "sic_outage": (task_id in outcome.sic_outage_task_ids),
                        }
                    )
                slot_rows.append(
                    {
                        "method": name,
                        "seed": seed,
                        "slot": outcome.slot_index,
                        "arrivals": len(outcome.arrived_task_ids),
                        "completed": outcome.completed_count,
                        "slack_s": outcome.total_slack_s,
                        "cross_interference_w": (outcome.total_cross_interference_w),
                        "aggregate_victim_link_exposure_w": (aggregate_exposure_w),
                        "sic_violations": sum(
                            not record.sic_feasible for record in outcome.rates
                        ),
                        "plan_key": plan_key_json,
                        "sic_outage_task_ids": outage_ids_json,
                        "runtime_s": outcome.runtime_s,
                    }
                )
            event_rows.extend(
                {"method": name, "seed": seed, **event} for event in events
            )

    initializer_statuses = sorted(
        {
            str(row["reason"])
            for row in event_rows
            if row.get("move_type") == "initialization"
        }
    )
    termination_statuses = sorted(
        {
            str(row["reason"])
            for row in event_rows
            if row.get("move_type") == "termination"
        }
    )
    summary = {
        "run_id": definition.run_id,
        "seeds": list(definition.seeds),
        "initializer_status_counts": {
            status: sum(
                row.get("move_type") == "initialization" and row.get("reason") == status
                for row in event_rows
            )
            for status in initializer_statuses
        },
        "search_termination_counts": {
            status: sum(
                row.get("move_type") == "termination" and row.get("reason") == status
                for row in event_rows
            )
            for status in termination_statuses
        },
        "methods": {
            name: {
                "method_spec": method_spec_to_dict(selected_specs[name]),
                "method_spec_sha256": method_spec_sha256(selected_specs[name]),
                "on_time_rate_by_seed": values,
                "mean": float(np.mean(values)),
                "std": (float(np.std(values, ddof=1)) if len(values) > 1 else 0.0),
                "mean_aggregate_victim_link_exposure_w_by_seed": (
                    method_exposure_summary[name]
                ),
                "mean_aggregate_victim_link_exposure_w": float(
                    np.mean(method_exposure_summary[name])
                ),
                "std_aggregate_victim_link_exposure_w": (
                    float(np.std(method_exposure_summary[name], ddof=1))
                    if len(method_exposure_summary[name]) > 1
                    else 0.0
                ),
            }
            for name, values in method_summary.items()
        },
    }
    writer.write_json("summary.json", summary)
    writer.write_csv(
        "tasks.csv",
        task_rows,
        fieldnames=(
            "method",
            "seed",
            "slot",
            "task_id",
            "user_id",
            "arrival_slot",
            "data_bits",
            "comp_cycles",
            "deadline_s",
            "local_cpu_hz",
            "completed",
            "delay_s",
            "alpha",
            "cpu_hz",
            "rate_bps",
            "sinr",
            "sic_feasible",
            "cross_interference_w",
            "serving_uav",
            "subcarrier",
            "reason",
            "plan_key",
            "sic_outage",
        ),
    )
    writer.write_csv(
        "slots.csv",
        slot_rows,
        fieldnames=(
            "method",
            "seed",
            "slot",
            "arrivals",
            "completed",
            "slack_s",
            "cross_interference_w",
            "aggregate_victim_link_exposure_w",
            "sic_violations",
            "plan_key",
            "sic_outage_task_ids",
            "runtime_s",
        ),
    )
    writer.write_jsonl("events.jsonl", event_rows)
    config_path = Path(config_path)
    writer.finalize(
        {
            "run_id": definition.run_id,
            "git_sha": _git_sha(),
            "config_path": str(config_path),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "seeds": list(definition.seeds),
            "csi_error_std": definition.csi_error_std,
            "resolved_definition": definition_to_dict(definition),
            "method_specs": {
                name: {
                    "factor_hash": method_spec_sha256(selected_specs[name]),
                    "spec": method_spec_to_dict(selected_specs[name]),
                }
                for name in definition.methods
            },
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "solver": {
                "name": "scipy.optimize.milp/HiGHS",
                "package_version": scipy.__version__,
            },
        }
    )


def run_experiment(
    config_path: Path,
    output_dir: Path,
    *,
    seed_access: SeedAccessRecorder,
) -> None:
    run_definition(
        load_definition(config_path),
        config_path,
        output_dir,
        seed_access=seed_access,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a trusted UAV-MEC thesis experiment",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument(
        "--purpose",
        choices=("development", "pilot"),
        required=True,
        help="formal execution is available only through the guarded v2 runner",
    )
    args = parser.parse_args()
    permit = (
        development_permit(args.registry)
        if args.purpose == "development"
        else pilot_permit(args.registry)
    )
    run_experiment(
        args.config,
        args.output,
        seed_access=SeedAccessRecorder(args.ledger, permit),
    )


if __name__ == "__main__":
    main()
