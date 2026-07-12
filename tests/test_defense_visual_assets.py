import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

CONTENT_SOURCE = Path("deliverables/defense_2026/sources/defense_visual_content.py")
GENERATOR_SOURCE = Path("deliverables/defense_2026/sources/generate_visual_assets.py")


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


def load_visual_renderer():
    spec = importlib.util.spec_from_file_location(
        "generate_visual_assets",
        GENERATOR_SOURCE,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    source_dir = str(GENERATOR_SOURCE.parent.resolve())
    sys.path.insert(0, source_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(source_dir)
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


def test_dd_mcaa_uses_a_chinese_title_and_typeset_formulae():
    content = load_visual_content()
    renderer = load_visual_renderer()

    assert "deadline" not in content.ASSET_SPECS["04_dd_mcaa_flow"]["title"].lower()
    formatted_steps = [renderer.format_formulas(step) for step in content.DD_MCAA_FLOW["steps"]]
    assert "$T_i = d_i - a_i$" in formatted_steps[0]
    assert r"$\alpha_i^{\min}$" in formatted_steps[1]
    assert r"$f_i^{\min}$" in formatted_steps[2]
    assert r"$(f_i^{\min},\ \mathrm{task\_id})$" in formatted_steps[4]


def test_generate_all_prunes_stale_assets(tmp_path, monkeypatch):
    renderer = load_visual_renderer()
    output_dir = tmp_path / "visuals"

    def fake_save(_fig, stem, target_dir):
        target_dir.mkdir(parents=True, exist_ok=True)
        png_path = target_dir / f"{stem}.png"
        pdf_path = target_dir / f"{stem}.pdf"
        png_path.write_bytes(b"png")
        pdf_path.write_bytes(b"pdf")
        return png_path, pdf_path

    monkeypatch.setattr(renderer, "save_asset", fake_save)
    monkeypatch.setattr(renderer, "RENDERERS", (("asset", lambda: None),))
    renderer.generate_all(output_dir)
    (output_dir / "stale.png").write_bytes(b"stale")

    renderer.generate_all(output_dir)

    assert not (output_dir / "stale.png").exists()


def test_generate_all_preserves_previous_release_when_rendering_fails(tmp_path, monkeypatch):
    renderer = load_visual_renderer()
    output_dir = tmp_path / "visuals"
    payload = {"value": b"old"}

    def fake_save(_fig, stem, target_dir):
        target_dir.mkdir(parents=True, exist_ok=True)
        png_path = target_dir / f"{stem}.png"
        pdf_path = target_dir / f"{stem}.pdf"
        png_path.write_bytes(payload["value"] + b"-png")
        pdf_path.write_bytes(payload["value"] + b"-pdf")
        return png_path, pdf_path

    def fail_render():
        raise RuntimeError("render failed")

    monkeypatch.setattr(renderer, "save_asset", fake_save)
    monkeypatch.setattr(renderer, "RENDERERS", (("asset", lambda: None),))
    renderer.generate_all(output_dir)
    previous_release = {path.name: path.read_bytes() for path in sorted(output_dir.iterdir())}

    payload["value"] = b"new"
    monkeypatch.setattr(
        renderer,
        "RENDERERS",
        (("asset", lambda: None), ("broken", fail_render)),
    )
    with pytest.raises(RuntimeError, match="render failed"):
        renderer.generate_all(output_dir)

    current_release = {path.name: path.read_bytes() for path in sorted(output_dir.iterdir())}
    assert current_release == previous_release


def test_generate_all_rolls_back_when_publication_is_interrupted(tmp_path, monkeypatch):
    renderer = load_visual_renderer()
    output_dir = tmp_path / "visuals"

    def fake_save(_fig, stem, target_dir):
        target_dir.mkdir(parents=True, exist_ok=True)
        png_path = target_dir / f"{stem}.png"
        pdf_path = target_dir / f"{stem}.pdf"
        png_path.write_bytes(b"png")
        pdf_path.write_bytes(b"pdf")
        return png_path, pdf_path

    monkeypatch.setattr(renderer, "save_asset", fake_save)
    monkeypatch.setattr(renderer, "RENDERERS", (("asset", lambda: None),))
    renderer.generate_all(output_dir)
    previous_release = {path.name: path.read_bytes() for path in sorted(output_dir.iterdir())}
    original_replace = Path.replace

    def interrupt_release_swap(path, target):
        if path.name == "release":
            raise KeyboardInterrupt
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", interrupt_release_swap)
    with pytest.raises(KeyboardInterrupt):
        renderer.generate_all(output_dir)

    assert output_dir.is_dir()
    current_release = {path.name: path.read_bytes() for path in sorted(output_dir.iterdir())}
    assert current_release == previous_release


def test_renderer_exports_all_assets(tmp_path):
    content = load_visual_content()
    stems = [spec["stem"] for spec in content.ASSET_SPECS.values()]

    subprocess.run(
        [sys.executable, str(GENERATOR_SOURCE), "--output", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    expected_outputs = [f"{stem}.{extension}" for stem in stems for extension in ("png", "pdf")]
    for stem in stems:
        png_path = tmp_path / f"{stem}.png"
        pdf_path = tmp_path / f"{stem}.pdf"
        assert png_path.is_file()
        assert pdf_path.is_file()

        with Image.open(png_path) as image:
            assert image.width >= 2400
            assert image.height >= 1350
            assert image.mode in {"RGB", "RGBA"}
            rgb = image.convert("RGB")
            colors = rgb.getcolors(maxcolors=rgb.width * rgb.height)
            assert colors is not None
            white_pixels = next(
                (count for count, color in colors if color == (255, 255, 255)),
                0,
            )
            non_white_ratio = 1 - white_pixels / (rgb.width * rgb.height)
            assert non_white_ratio > 0.02

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["dimensions"] == {
        "width_px": 2400,
        "height_px": 1350,
        "dpi": 150,
    }
    assert manifest["outputs"] == expected_outputs
    assert manifest["algorithm_acronyms"] == ["EAD-JSRA", "IEA-CAP", "DD-MCAA"]
    assert manifest["canonical_evidence_version"] == "canonical v1"
    assert set(manifest["sha256"]) == set(expected_outputs)
    assert all(len(digest) == 64 for digest in manifest["sha256"].values())
    for filename, digest in manifest["sha256"].items():
        assert hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest() == digest

    repeat_dir = tmp_path.with_name(f"{tmp_path.name}_repeat")
    subprocess.run(
        [sys.executable, str(GENERATOR_SOURCE), "--output", str(repeat_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    repeat_manifest = json.loads((repeat_dir / "manifest.json").read_text(encoding="utf-8"))
    assert repeat_manifest["sha256"] == manifest["sha256"]
