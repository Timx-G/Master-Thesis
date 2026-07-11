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
