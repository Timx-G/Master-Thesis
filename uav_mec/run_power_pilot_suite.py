"""Resume all H1-H4 pilot families and freeze their power decision."""

from __future__ import annotations

import argparse
from pathlib import Path

from .analyze_results import verify_checksums
from .run_power_pilot import write_power_pilot_analysis
from .run_shards import run_sharded_experiment


REGISTRY_PATH = Path("configs/thesis/experiment_registry_v2.toml")
PILOT_CONFIGS = (
    ("main", Path("configs/thesis/pilot_main_v2.toml")),
    ("externality", Path("configs/thesis/pilot_externality_v2.toml")),
    ("resource", Path("configs/thesis/pilot_resource_fixed_v2.toml")),
    ("joint", Path("configs/thesis/pilot_joint_v2.toml")),
)


def run_power_pilot_suite(
    output_dir: Path,
    *,
    shard_size: int = 1,
    simulations: int = 2000,
    analysis_seed: int = 0,
) -> Path:
    """Run or resume every pilot family, then write one immutable decision."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    bundles = {}
    for family, config in PILOT_CONFIGS:
        report = run_sharded_experiment(
            config,
            root / family,
            shard_size=shard_size,
        )
        bundles[family] = report.merged_path

    decision = root / "decision"
    if decision.exists():
        verify_checksums(decision)
        return decision
    write_power_pilot_analysis(
        REGISTRY_PATH,
        bundles,
        decision,
        simulations=simulations,
        analysis_seed=analysis_seed,
    )
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and resume all v2 power-pilot experiment families",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=2000)
    parser.add_argument("--analysis-seed", type=int, default=0)
    args = parser.parse_args()
    decision = run_power_pilot_suite(
        args.output,
        shard_size=args.shard_size,
        simulations=args.simulations,
        analysis_seed=args.analysis_seed,
    )
    print(f"power_decision={decision}")


if __name__ == "__main__":
    main()
