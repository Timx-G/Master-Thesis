"""Frozen mechanism, scalability, and robustness suite construction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

from .experiment import (
    ExperimentDefinition,
    load_definition,
    run_definition,
)


@dataclass(frozen=True, slots=True)
class SuiteCase:
    name: str
    definition: ExperimentDefinition


BASE_PATH = Path("configs/thesis/confirmatory_main.toml")
SWEEP_METHODS = ("greedy", "exchange_stable", "full")


def _with_scenario(
    base: ExperimentDefinition,
    name: str,
    **changes: object,
) -> SuiteCase:
    scenario = replace(base.scenario, **changes)
    evaluation = replace(
        base.evaluation,
        n_subcarriers=scenario.n_subcarriers,
        bandwidth_hz=scenario.channel.bandwidth,
        transmit_power_w=scenario.channel.p_max,
        cpu_per_uav_hz=scenario.f_per_uav,
    )
    return SuiteCase(
        name,
        replace(
            base,
            run_id=name,
            methods=SWEEP_METHODS,
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
        _with_scenario(base, f"n{count}", num_users=count)
        for count in (10, 15, 20, 25, 30, 40, 50)
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
        cases.append(
            _with_scenario(base, label, f_per_uav=cpu_hz)
        )
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


def build_suite(name: str) -> tuple[SuiteCase, ...]:
    base = load_definition(BASE_PATH)
    if name == "mechanism":
        return _mechanism_suite(base)
    if name == "scalability":
        return _scalability_suite(base)
    if name == "robustness":
        return _robustness_suite(base)
    raise ValueError(f"unknown suite {name}")


def run_suite(name: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    for case in build_suite(name):
        run_definition(
            case.definition,
            BASE_PATH,
            output_dir / case.name,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a frozen UAV-MEC thesis experiment suite",
    )
    parser.add_argument(
        "--suite",
        choices=("mechanism", "scalability", "robustness"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_suite(args.suite, args.output)


if __name__ == "__main__":
    main()
