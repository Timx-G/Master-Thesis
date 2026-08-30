"""Frozen mechanism, scalability, and robustness suite construction."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from .experiment import (
    ExperimentDefinition,
    load_definition,
    run_definition,
)
from .capacity_analysis import write_capacity_analysis
from .externality_analysis import write_externality_analysis
from .experiment_registry import load_registry


@dataclass(frozen=True, slots=True)
class SuiteCase:
    name: str
    definition: ExperimentDefinition


BASE_PATH = Path("configs/thesis/confirmatory_main.toml")
EXTERNALITY_BASE_PATH = Path("configs/thesis/externality_stress.toml")
CAPACITY_BASE_PATH = Path("configs/thesis/confirmatory_main_v2.toml")
JOINT_BASE_PATH = Path("configs/thesis/confirmatory_joint_v2.toml")
REGISTRY_PATH = Path("configs/thesis/experiment_registry_v2.toml")
SWEEP_METHODS = ("greedy", "exchange_stable", "full")
EXTERNALITY_METHODS = ("intra_global", "cross_bilateral", "full")


def _apply_registered_protocol(
    definition: ExperimentDefinition,
    family_name: str,
) -> ExperimentDefinition:
    registry = load_registry(REGISTRY_PATH)
    matches = tuple(family for family in registry.families if family.name == family_name)
    if len(matches) != 1:
        raise ValueError(f"experiment registry must define family {family_name!r} exactly once")
    family = matches[0]
    missing_methods = sorted(set(family.methods) - set(definition.methods))
    if missing_methods:
        raise ValueError(
            f"registered family {family_name} requires missing method {missing_methods[0]}"
        )
    return replace(
        definition,
        seeds=family.seeds,
        scenario=replace(definition.scenario, num_slots=family.num_slots),
    )


def _with_scenario(
    base: ExperimentDefinition,
    name: str,
    *,
    methods: tuple[str, ...] = SWEEP_METHODS,
    **changes: object,
) -> SuiteCase:
    scenario = replace(base.scenario, **changes)
    evaluation = replace(
        base.evaluation,
        n_subcarriers=scenario.n_subcarriers,
        bandwidth_hz=scenario.channel.bandwidth,
        transmit_power_w=scenario.channel.p_max,
        cpu_per_uav_hz=scenario.f_per_uav,
        result_ratio=scenario.task.result_ratio,
    )
    return SuiteCase(
        name,
        replace(
            base,
            run_id=name,
            methods=methods,
            scenario=scenario,
            evaluation=evaluation,
        ),
    )


def _mechanism_suite(
    base: ExperimentDefinition,
) -> tuple[SuiteCase, ...]:
    return tuple(
        SuiteCase(
            case_name,
            replace(
                base,
                run_id=case_name,
                methods=SWEEP_METHODS,
                evaluation=replace(
                    base.evaluation,
                    cross_interference_scale=scale,
                ),
            ),
        )
        for case_name, scale in (
            ("coupling0", 0.0),
            ("coupling05", 0.5),
            ("coupling1", 1.0),
        )
    )


def _scalability_suite(
    base: ExperimentDefinition,
) -> tuple[SuiteCase, ...]:
    users = tuple(
        _with_scenario(base, f"n{count}", num_users=count) for count in (10, 15, 20, 25, 30, 40, 50)
    )
    uavs = tuple(
        _with_scenario(
            base,
            f"k{count}",
            num_uavs=count,
            num_clusters=3 if count <= 3 else count,
        )
        for count in (1, 2, 3, 4, 5)
    )
    return users + uavs


def _robustness_suite(
    base: ExperimentDefinition,
) -> tuple[SuiteCase, ...]:
    cases: list[SuiteCase] = []
    for nf in (3.0, 7.0, 10.0):
        channel = replace(base.scenario.channel, noise_figure_db=nf)
        cases.append(
            _with_scenario(
                base,
                f"nf{int(nf)}",
                channel=channel,
            )
        )
    for label, error in (
        ("csi0", 0.0),
        ("csi05", 0.05),
        ("csi10", 0.10),
        ("csi20", 0.20),
    ):
        cases.append(
            SuiteCase(
                label,
                replace(
                    base,
                    run_id=label,
                    methods=SWEEP_METHODS,
                    csi_error_std=error,
                ),
            )
        )
    for label, residual in (
        ("eps0", 0.0),
        ("eps02", 0.02),
        ("eps05", 0.05),
        ("eps10", 0.10),
    ):
        cases.append(
            SuiteCase(
                label,
                replace(
                    base,
                    run_id=label,
                    methods=SWEEP_METHODS,
                    evaluation=replace(
                        base.evaluation,
                        sic_residual=residual,
                    ),
                ),
            )
        )
    for label, threshold in (
        ("sicm3", 10**-0.3),
        ("sic0", 1.0),
        ("sic3", 10**0.3),
        ("sic5", 10**0.5),
    ):
        cases.append(
            SuiteCase(
                label,
                replace(
                    base,
                    run_id=label,
                    methods=SWEEP_METHODS,
                    evaluation=replace(
                        base.evaluation,
                        gamma_sic_min=threshold,
                    ),
                ),
            )
        )
    for label, cpu_hz in (
        ("cpu2", 2e9),
        ("cpu4", 4e9),
        ("cpu8", 8e9),
    ):
        cases.append(_with_scenario(base, label, f_per_uav=cpu_hz))
    for label, deadline_s in (
        ("t06", 0.6),
        ("t08", 0.8),
        ("t10", 1.0),
    ):
        cases.append(
            _with_scenario(
                base,
                label,
                task=replace(
                    base.scenario.task,
                    t_max=deadline_s,
                ),
            )
        )
    cases.extend(
        (
            _with_scenario(
                base,
                "load_light",
                task=replace(
                    base.scenario.task,
                    data_min=0.2e6,
                    data_max=1.0e6,
                    comp_min=300.0,
                    comp_max=600.0,
                ),
            ),
            _with_scenario(base, "load_default"),
            _with_scenario(
                base,
                "load_heavy",
                task=replace(
                    base.scenario.task,
                    data_min=0.8e6,
                    data_max=3.0e6,
                    comp_min=800.0,
                    comp_max=1500.0,
                ),
            ),
        )
    )
    return tuple(cases)


def _externality_suite(
    base: ExperimentDefinition,
) -> tuple[SuiteCase, ...]:
    overlap_std = {"low": 150.0, "medium": 30.0, "high": 5.0}
    period_choices = {"08": (3,), "10": (2, 3, 3), "12": (2,)}
    cases: list[SuiteCase] = []
    for overlap in ("low", "high"):
        for coupling, scale in (("0", 0.0), ("1", 1.0)):
            for load in ("08", "12"):
                name = f"overlap_{overlap}_coupling{coupling}_load{load}"
                scenario = replace(
                    base.scenario,
                    cluster_std=overlap_std[overlap],
                    task=replace(
                        base.scenario.task,
                        period_choices=period_choices[load],
                    ),
                )
                cases.append(
                    SuiteCase(
                        name,
                        replace(
                            base,
                            run_id=name,
                            methods=EXTERNALITY_METHODS,
                            scenario=scenario,
                            evaluation=replace(
                                base.evaluation,
                                cross_interference_scale=scale,
                            ),
                        ),
                    )
                )
    center_name = "overlap_medium_coupling05_load10"
    cases.append(
        SuiteCase(
            center_name,
            replace(
                base,
                run_id=center_name,
                methods=EXTERNALITY_METHODS,
                scenario=replace(
                    base.scenario,
                    cluster_std=overlap_std["medium"],
                    task=replace(
                        base.scenario.task,
                        period_choices=period_choices["10"],
                    ),
                ),
                evaluation=replace(
                    base.evaluation,
                    cross_interference_scale=0.5,
                ),
            ),
        )
    )
    return tuple(cases)


def _capacity_suite(
    base: ExperimentDefinition,
) -> tuple[SuiteCase, ...]:
    load = tuple(
        _with_scenario(
            base,
            f"load_k3_n{users}",
            methods=EXTERNALITY_METHODS,
            num_uavs=3,
            num_users=users,
            num_clusters=3,
        )
        for users in (30, 45, 60)
    )
    target_arrival_probability = (
        base.scenario.cool_rate
        + (base.scenario.burst_rate - base.scenario.cool_rate) / base.scenario.num_clusters
    )
    equal_load_cases: list[SuiteCase] = []
    for uavs in (2, 3, 4, 5):
        clusters = max(3, uavs)
        cool_rate = (clusters * target_arrival_probability - base.scenario.burst_rate) / (
            clusters - 1
        )
        if not 0.0 <= cool_rate <= 1.0:
            raise ValueError("equal-load cool_rate falls outside [0, 1]")
        equal_load_cases.append(
            _with_scenario(
                base,
                f"scale_k{uavs}_n{15 * uavs}",
                methods=EXTERNALITY_METHODS,
                num_uavs=uavs,
                num_users=15 * uavs,
                num_clusters=clusters,
                cool_rate=cool_rate,
            )
        )
    equal_load = tuple(equal_load_cases)
    return load + equal_load


def build_suite(name: str) -> tuple[SuiteCase, ...]:
    if name == "externality":
        base_path = EXTERNALITY_BASE_PATH
    elif name == "capacity":
        base_path = CAPACITY_BASE_PATH
    elif name == "joint":
        base_path = JOINT_BASE_PATH
    else:
        base_path = BASE_PATH
    base = load_definition(base_path)
    if name == "mechanism":
        return _mechanism_suite(base)
    if name == "scalability":
        return _scalability_suite(_apply_registered_protocol(base, "scaling"))
    if name == "robustness":
        return _robustness_suite(_apply_registered_protocol(base, "robustness"))
    if name == "externality":
        return _externality_suite(_apply_registered_protocol(base, "externality"))
    if name == "capacity":
        return _capacity_suite(base)
    if name == "joint":
        return (
            SuiteCase(
                "joint_k3_n45",
                _apply_registered_protocol(base, "joint"),
            ),
        )
    raise ValueError(f"unknown suite {name}")


def _run_suite_case(
    arguments: tuple[SuiteCase, Path, Path],
) -> tuple[str, ExperimentDefinition, Path]:
    case, base_path, case_path = arguments
    run_definition(case.definition, base_path, case_path)
    return case.name, case.definition, case_path


def run_suite(
    name: str,
    output_dir: Path,
    *,
    workers: int = 1,
    seed_count: int | None = None,
    num_slots: int | None = None,
    development_seed_start: int | None = None,
) -> None:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if seed_count is not None and seed_count <= 0:
        raise ValueError("seed_count must be positive")
    if num_slots is not None and num_slots <= 0:
        raise ValueError("num_slots must be positive")
    if (seed_count is None) != (development_seed_start is None):
        raise ValueError(
            "seed_count and development_seed_start must be provided together"
        )
    if development_seed_start is not None:
        if (
            not isinstance(development_seed_start, int)
            or isinstance(development_seed_start, bool)
            or development_seed_start < 0
        ):
            raise ValueError("development_seed_start must be a nonnegative integer")
        development_seeds = set(
            range(development_seed_start, development_seed_start + seed_count)
        )
        formal_seeds = {
            seed
            for family in load_registry(REGISTRY_PATH).families
            for seed in family.seeds
        }
        if development_seeds & formal_seeds:
            raise ValueError("development seed pool overlaps a registered formal seed")
    output_dir.mkdir(parents=True, exist_ok=False)
    if name == "externality":
        base_path = EXTERNALITY_BASE_PATH
    elif name == "capacity":
        base_path = CAPACITY_BASE_PATH
    elif name == "joint":
        base_path = JOINT_BASE_PATH
    else:
        base_path = BASE_PATH
    cases = build_suite(name)
    if seed_count is not None or num_slots is not None:
        bounded: list[SuiteCase] = []
        for case in cases:
            definition = case.definition
            if seed_count is not None:
                if seed_count > len(definition.seeds):
                    raise ValueError("seed_count exceeds the frozen seed count")
                definition = replace(
                    definition,
                    seeds=tuple(
                        range(
                            development_seed_start,
                            development_seed_start + seed_count,
                        )
                    ),
                )
            if num_slots is not None:
                if num_slots > definition.scenario.num_slots:
                    raise ValueError("num_slots exceeds the frozen slot count")
                definition = replace(
                    definition,
                    scenario=replace(
                        definition.scenario,
                        num_slots=num_slots,
                    ),
                )
            bounded.append(SuiteCase(case.name, definition))
        cases = tuple(bounded)
    arguments = tuple((case, base_path, output_dir / case.name) for case in cases)
    if workers == 1:
        results = tuple(_run_suite_case(argument) for argument in arguments)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = tuple(
                executor.map(
                    _run_suite_case,
                    arguments,
                )
            )
    completed = {case_name: (definition, case_path) for case_name, definition, case_path in results}
    if name == "externality":
        write_externality_analysis(completed, output_dir / "analysis")
    elif name == "capacity":
        write_capacity_analysis(completed, output_dir / "analysis")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a frozen UAV-MEC thesis experiment suite",
    )
    parser.add_argument(
        "--suite",
        choices=(
            "mechanism",
            "scalability",
            "robustness",
            "externality",
            "capacity",
            "joint",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed-count", type=int)
    parser.add_argument("--num-slots", type=int)
    parser.add_argument("--development-seed-start", type=int)
    args = parser.parse_args()
    run_suite(
        args.suite,
        args.output,
        workers=args.workers,
        seed_count=args.seed_count,
        num_slots=args.num_slots,
        development_seed_start=args.development_seed_start,
    )


if __name__ == "__main__":
    main()
