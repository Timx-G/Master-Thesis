"""Frozen mechanism, scalability, and robustness suite construction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import tomllib

from .experiment import (
    ExperimentDefinition,
    load_definition,
    run_definition,
    thesis_method_specs,
)
from .experiment_registry import load_registry, registered_family
from .seed_access import (
    SeedAccessRecorder,
    development_permit,
    pilot_permit,
)


@dataclass(frozen=True, slots=True)
class SuiteCase:
    name: str
    definition: ExperimentDefinition


BASE_PATH = Path("configs/thesis/confirmatory_main.toml")
_METHOD_SPECS = thesis_method_specs()
SWEEP_METHODS = tuple(
    _METHOD_SPECS[method_id].method_id for method_id in ("greedy", "exchange_stable", "full")
)


def _scenario_case(
    base: ExperimentDefinition,
    case_id: str,
    **changes: object,
) -> ExperimentDefinition:
    scenario = replace(base.scenario, **changes)
    return replace(
        base,
        run_id=f"{base.run_id}:{case_id}",
        scenario=scenario,
        evaluation=replace(
            base.evaluation,
            n_subcarriers=scenario.n_subcarriers,
            bandwidth_hz=scenario.channel.bandwidth,
            transmit_power_w=scenario.channel.p_max,
            cpu_per_uav_hz=scenario.f_per_uav,
        ),
    )


def build_registered_case_definitions(
    family_name: str,
    registry_path: Path = Path("configs/thesis/experiment_registry_v2.toml"),
) -> tuple[SuiteCase, ...]:
    """Resolve explicit v2 case parameters without parsing meaning from IDs."""

    registry = load_registry(registry_path)
    family = registered_family(registry, family_name)
    loaded = load_definition(family.config_path)
    base = replace(
        loaded,
        seeds=family.seeds,
        scenario=replace(
            loaded.scenario,
            num_slots=family.num_slots,
        ),
    )
    raw = tomllib.loads(family.config_path.read_text(encoding="utf-8"))
    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != len(family.cases):
        raise ValueError(f"{family_name} case table coverage mismatch")
    resolved: list[SuiteCase] = []
    for registered, parameters in zip(family.cases, raw_cases, strict=True):
        if not isinstance(parameters, dict) or parameters.get("case_id") != registered.case_id:
            raise ValueError(f"{family_name} case table order mismatch")
        if family_name == "externality":
            cluster_std = parameters.get("cluster_std")
            period_choices = parameters.get("period_choices")
            cross_scale = parameters.get("cross_interference_scale")
            if (
                not isinstance(cluster_std, (int, float))
                or not isinstance(period_choices, list)
                or not period_choices
                or not isinstance(cross_scale, (int, float))
            ):
                raise ValueError(f"{registered.case_id} lacks explicit externality parameters")
            definition = _scenario_case(
                base,
                registered.case_id,
                cluster_std=float(cluster_std),
                task=replace(
                    base.scenario.task,
                    period_choices=tuple(period_choices),
                ),
            )
            definition = replace(
                definition,
                evaluation=replace(
                    definition.evaluation,
                    cross_interference_scale=float(cross_scale),
                ),
            )
        elif family_name == "scaling":
            num_uavs = parameters.get("num_uavs")
            num_users = parameters.get("num_users")
            if type(num_uavs) is not int or type(num_users) is not int:
                raise ValueError(f"{registered.case_id} lacks explicit scaling parameters")
            definition = _scenario_case(
                base,
                registered.case_id,
                num_uavs=num_uavs,
                num_users=num_users,
                num_clusters=max(3, num_uavs),
            )
        elif family_name in {"main", "joint"}:
            definition = replace(base, run_id=f"{base.run_id}:{registered.case_id}")
        elif family_name == "robustness":
            factor = parameters.get("factor")
            definition = replace(base, run_id=f"{base.run_id}:{registered.case_id}")
            if factor == "csi_error_std":
                value = parameters.get("csi_error_std")
                if not isinstance(value, (int, float)):
                    raise ValueError(f"{registered.case_id} lacks csi_error_std")
                definition = replace(definition, csi_error_std=float(value))
            elif factor == "sic_residual":
                value = parameters.get("sic_residual")
                if not isinstance(value, (int, float)):
                    raise ValueError(f"{registered.case_id} lacks sic_residual")
                definition = replace(
                    definition,
                    evaluation=replace(definition.evaluation, sic_residual=float(value)),
                )
            elif factor == "sic_threshold_db":
                value = parameters.get("gamma_sic_min")
                if not isinstance(value, (int, float)):
                    raise ValueError(f"{registered.case_id} lacks gamma_sic_min")
                definition = replace(
                    definition,
                    evaluation=replace(definition.evaluation, gamma_sic_min=float(value)),
                )
            elif factor == "cpu_layout":
                values = parameters.get("cpu_per_uav_hz")
                if not isinstance(values, list) or len(values) != base.scenario.num_uavs:
                    raise ValueError(f"{registered.case_id} lacks a complete CPU layout")
                definition = replace(
                    definition,
                    evaluation=replace(
                        definition.evaluation,
                        cpu_per_uav_hz=tuple(float(value) for value in values),
                    ),
                )
            elif factor == "deadline_pressure":
                value = parameters.get("t_max")
                if not isinstance(value, (int, float)):
                    raise ValueError(f"{registered.case_id} lacks t_max")
                definition = _scenario_case(
                    base,
                    registered.case_id,
                    task=replace(base.scenario.task, t_min=None, t_max=float(value)),
                )
            elif factor == "arrival_mode":
                arrival_mode = parameters.get("arrival_mode")
                random_burst = parameters.get("random_burst")
                if arrival_mode == "periodic":
                    period_choices = parameters.get("period_choices")
                    jitter = parameters.get("period_jitter_slots")
                    if (
                        not isinstance(period_choices, list)
                        or not period_choices
                        or type(jitter) is not int
                        or random_burst is not False
                    ):
                        raise ValueError(
                            f"{registered.case_id} lacks periodic arrival parameters"
                        )
                    task = replace(
                        base.scenario.task,
                        arrival_mode="periodic",
                        period_choices=tuple(period_choices),
                        period_jitter_slots=jitter,
                    )
                    definition = _scenario_case(
                        base,
                        registered.case_id,
                        task=task,
                        random_burst=False,
                    )
                elif arrival_mode == "bernoulli" and random_burst is True:
                    numeric_names = (
                        "arrival_prob",
                        "burst_rate",
                        "cool_rate",
                        "burst_period_min",
                        "burst_period_max",
                    )
                    if any(not isinstance(parameters.get(name), (int, float)) for name in numeric_names):
                        raise ValueError(
                            f"{registered.case_id} lacks burst arrival parameters"
                        )
                    definition = _scenario_case(
                        base,
                        registered.case_id,
                        task=replace(
                            base.scenario.task,
                            arrival_mode="bernoulli",
                            arrival_prob=float(parameters["arrival_prob"]),
                        ),
                        random_burst=True,
                        burst_rate=float(parameters["burst_rate"]),
                        cool_rate=float(parameters["cool_rate"]),
                        burst_period_min=int(parameters["burst_period_min"]),
                        burst_period_max=int(parameters["burst_period_max"]),
                    )
                else:
                    raise ValueError(f"{registered.case_id} lacks a supported arrival mode")
            elif factor == "pathloss_model":
                model = parameters.get("pathloss_model")
                if not isinstance(model, str):
                    raise ValueError(f"{registered.case_id} lacks pathloss_model")
                definition = _scenario_case(
                    base,
                    registered.case_id,
                    channel=replace(base.scenario.channel, pathloss_model=model),
                )
            else:
                raise ValueError(f"{registered.case_id} has unknown robustness factor")
        else:
            raise ValueError(f"registered case resolver is unavailable for {family_name}")
        resolved.append(SuiteCase(registered.case_id, definition))
    return tuple(resolved)


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


def build_suite(name: str) -> tuple[SuiteCase, ...]:
    base = load_definition(BASE_PATH)
    if name == "mechanism":
        return _mechanism_suite(base)
    if name == "scalability":
        return _scalability_suite(base)
    if name == "robustness":
        return _robustness_suite(base)
    raise ValueError(f"unknown suite {name}")


def run_suite(
    name: str,
    output_dir: Path,
    *,
    seed_access: SeedAccessRecorder,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    for case in build_suite(name):
        run_definition(
            case.definition,
            BASE_PATH,
            output_dir / case.name,
            seed_access=seed_access,
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
    run_suite(
        args.suite,
        args.output,
        seed_access=SeedAccessRecorder(args.ledger, permit),
    )


if __name__ == "__main__":
    main()
