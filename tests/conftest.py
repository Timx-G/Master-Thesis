from pathlib import Path

import pytest

from uav_mec.seed_access import SeedAccessRecorder, development_permit


@pytest.fixture
def development_seed_access(tmp_path: Path) -> SeedAccessRecorder:
    return SeedAccessRecorder(
        ledger_path=tmp_path / "seed-access.jsonl",
        permit=development_permit(
            Path("configs/thesis/experiment_registry_v2.toml")
        ),
    )
