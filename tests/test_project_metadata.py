from pathlib import Path
import tomllib


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
