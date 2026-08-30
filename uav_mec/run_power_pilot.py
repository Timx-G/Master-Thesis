"""Validate pilot bundles and freeze the joint Holm sample-size decision."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from .analyze_results import verify_checksums
from .artifacts import RunArtifactWriter, sha256_file
from .experiment_registry import ExperimentRegistry, load_registry
from .power_analysis import select_sample_size


PILOT_SEEDS = tuple(range(1000, 1100))


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_pilot_differences(
    registry: ExperimentRegistry,
    bundles_by_family: Mapping[str, Path],
) -> tuple[dict[str, tuple[float, ...]], dict[str, str]]:
    """Load checksum-verified seed-paired differences in percentage points."""

    required_families = {
        hypothesis.family for hypothesis in registry.hypotheses
    }
    missing_families = sorted(required_families - set(bundles_by_family))
    if missing_families:
        raise ValueError(f"missing pilot family {missing_families[0]}")

    loaded: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
    input_hashes: dict[str, str] = {}
    for family in sorted(required_families):
        path = Path(bundles_by_family[family])
        verify_checksums(path)
        manifest = _read_json(path / "manifest.json")
        summary = _read_json(path / "summary.json")
        expected_seeds = list(PILOT_SEEDS)
        if manifest.get("seeds") != expected_seeds or summary.get("seeds") != expected_seeds:
            raise ValueError(
                f"pilot seeds for {family} must be exactly 1000--1099"
            )
        loaded[family] = (manifest, summary)
        input_hashes[family] = sha256_file(path / "checksums.sha256")

    differences: dict[str, tuple[float, ...]] = {}
    for hypothesis in registry.hypotheses:
        _manifest, summary = loaded[hypothesis.family]
        methods = summary.get("methods")
        if not isinstance(methods, dict):
            raise ValueError(
                f"pilot summary for {hypothesis.family} is missing methods"
            )
        values_by_role = []
        for method in (hypothesis.treatment, hypothesis.control):
            method_summary = methods.get(method)
            if not isinstance(method_summary, dict):
                raise ValueError(
                    f"pilot {hypothesis.family} is missing method {method}"
                )
            values = method_summary.get("on_time_rate_by_seed")
            if not isinstance(values, list) or len(values) != len(PILOT_SEEDS):
                raise ValueError(
                    f"pilot method {method} must have one value per pilot seed"
                )
            try:
                numeric = tuple(float(value) for value in values)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"pilot method {method} contains a non-numeric value"
                ) from exc
            values_by_role.append(numeric)
        treatment, control = values_by_role
        differences[hypothesis.hypothesis_id] = tuple(
            100.0 * (treated - baseline)
            for treated, baseline in zip(treatment, control, strict=True)
        )
    return differences, input_hashes


def write_power_pilot_analysis(
    registry_path: Path,
    bundles_by_family: Mapping[str, Path],
    output_dir: Path,
    *,
    simulations: int = 2000,
    analysis_seed: int = 0,
) -> None:
    """Write one immutable power-decision bundle from complete pilot inputs."""

    registry_file = Path(registry_path)
    registry = load_registry(registry_file)
    differences, input_hashes = load_pilot_differences(
        registry,
        bundles_by_family,
    )
    mdes = {
        hypothesis.hypothesis_id: hypothesis.mde_pp
        for hypothesis in registry.hypotheses
    }
    decision = select_sample_size(
        differences,
        mdes,
        candidates=registry.candidate_sample_sizes,
        simulations=simulations,
        target_power=0.8,
        alpha=0.05,
        seed=analysis_seed,
    )
    decision_payload = {
        **asdict(decision),
        "analysis_seed": analysis_seed,
        "alpha": 0.05,
        "candidate_sample_sizes": list(registry.candidate_sample_sizes),
        "pilot_seeds": list(PILOT_SEEDS),
        "mde_pp_by_hypothesis": mdes,
        "input_bundle_hashes": input_hashes,
    }

    writer = RunArtifactWriter(output_dir)
    writer.write_json(
        "pilot_differences.json",
        {
            hypothesis_id: list(values)
            for hypothesis_id, values in differences.items()
        },
    )
    writer.write_json("power_decision.json", decision_payload)
    writer.finalize(
        {
            "run_id": "power-pilot-v2",
            "registry_path": str(registry_file),
            "registry_sha256": sha256_file(registry_file),
            "input_bundle_hashes": input_hashes,
            "selected_n": decision.selected_n,
        }
    )


def _bundle_arguments(values: list[str]) -> dict[str, Path]:
    bundles: dict[str, Path] = {}
    for value in values:
        family, separator, path = value.partition("=")
        if not separator or not family or not path or family in bundles:
            raise ValueError("--bundle must be a unique FAMILY=PATH value")
        bundles[family] = Path(path)
    return bundles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the H1-H4 joint Holm power decision",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/thesis/experiment_registry_v2.toml"),
    )
    parser.add_argument("--bundle", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=2000)
    parser.add_argument("--analysis-seed", type=int, default=0)
    args = parser.parse_args()
    write_power_pilot_analysis(
        args.registry,
        _bundle_arguments(args.bundle),
        args.output,
        simulations=args.simulations,
        analysis_seed=args.analysis_seed,
    )


if __name__ == "__main__":
    main()
