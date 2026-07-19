from pathlib import Path
import tomllib


THESIS_V2_SEMANTIC_CONFIGS = (
    Path("configs/thesis/confirmatory_main_v2.toml"),
    Path("configs/thesis/confirmatory_resource_fixed_v2.toml"),
    Path("configs/thesis/confirmatory_resource_strong.toml"),
    Path("configs/thesis/confirmatory_joint_v2.toml"),
    Path("configs/thesis/confirmatory_exact.toml"),
    Path("configs/thesis/pilot_main_v2.toml"),
    Path("configs/thesis/pilot_externality_v2.toml"),
    Path("configs/thesis/pilot_resource_fixed_v2.toml"),
    Path("configs/thesis/pilot_joint_v2.toml"),
    Path("configs/thesis/externality_stress.toml"),
    Path("configs/thesis/smoke.toml"),
    Path("configs/thesis/smoke_exact.toml"),
)


def test_project_metadata_matches_current_research_topic():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    assert project["requires-python"] == ">=3.11"
    assert project["description"] == (
        "Interference-aware pairing and deadline-driven resource allocation "
        "for full-reuse multi-UAV NOMA-MEC"
    )
    assert "pytest>=8.0,<9" in project["optional-dependencies"]["dev"]


def test_legacy_drl_dependencies_are_not_in_the_trusted_core():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    core = project["dependencies"]
    legacy = project["optional-dependencies"]["legacy-drl"]

    assert not any(
        name in dependency
        for name in ("torch", "gymnasium")
        for dependency in core
    )
    assert any(dependency.startswith("torch") for dependency in legacy)
    assert any(dependency.startswith("gymnasium") for dependency in legacy)


def test_trusted_core_ranges_remain_python311_compatible():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    core = data["project"]["dependencies"]

    assert "numpy>=1.26,<2.5" in core
    assert "matplotlib>=3.8,<3.11" in core
    assert "scipy>=1.11,<1.18" in core


def test_thesis_v2_configs_freeze_zero_rho_and_one_second_slots():
    for path in THESIS_V2_SEMANTIC_CONFIGS:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        scenario = raw["scenario"]
        task = scenario["task"]
        dt = scenario["uav"]["dt"]

        assert task["result_ratio"] == 0.0, path
        assert dt == 1.0, path
        for deadline_name in ("t_min", "t_max"):
            if deadline_name in task:
                assert 0.0 < task[deadline_name] <= dt, (
                    path,
                    deadline_name,
                )
        assert not any(
            "downlink" in key.lower()
            for section in raw.values()
            if isinstance(section, dict)
            for key in section
        ), path


def test_semantic_gate_is_the_documented_v2_pre_run_boundary():
    makefile = Path("Makefile").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    policy = Path("CLAUDE.md").read_text(encoding="utf-8")

    assert "semantic-gate:" in makefile
    phony_line = next(line for line in makefile.splitlines() if line.startswith(".PHONY:"))
    assert "semantic-gate" in phony_line.split()
    assert "canonical-v1" in readme
    assert "cannot validate thesis-v2" in readme
    assert readme.index("make semantic-gate") < readme.index(
        "## Full Confirmatory Reproduction"
    )
    assert "### Canonical thesis-v2 semantics" in policy
    assert "legacy-v1 only" in policy
