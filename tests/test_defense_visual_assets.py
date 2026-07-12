import importlib.util
from pathlib import Path

import pytest


CONTENT_SOURCE = Path("deliverables/defense_2026/sources/defense_visual_content.py")


def load_visual_content():
    spec = importlib.util.spec_from_file_location(
        "defense_visual_content",
        CONTENT_SOURCE,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_algorithm_names_match_the_public_defense_contract():
    content = load_visual_content()

    assert content.ALGO_NAMES["full"]["acronym"] == "EAD-JSRA"
    assert content.ALGO_NAMES["innovation1"]["acronym"] == "IEA-CAP"
    assert content.ALGO_NAMES["innovation2"]["acronym"] == "DD-MCAA"
    assert content.METHOD_DISPLAY_NAMES["full"] == "EAD-JSRA（本文算法）"

    with pytest.raises(TypeError):
        content.ALGO_NAMES["full"]["acronym"] = "renamed"


def test_visual_asset_manifest_declares_the_six_approved_outputs():
    content = load_visual_content()

    expected_stems = {
        "01_background",
        "02_ead_jsra_framework",
        "03_iea_cap_flow",
        "04_dd_mcaa_flow",
        "05_progress",
        "table_algorithm_selection",
    }
    assert set(content.ASSET_SPECS) == expected_stems
    for stem in expected_stems:
        assert content.ASSET_SPECS[stem]["stem"] == stem
        assert content.ASSET_SPECS[stem]["title"]
