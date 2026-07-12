from pathlib import Path
import re


ACTIVE_PROSE = (
    Path("README.md"),
    Path("paper/main_v2.tex"),
    *sorted(Path("thesis/chapters").glob("*.tex")),
    Path("thesis/claims-matrix.md"),
    Path("defense/outline.md"),
    Path("defense/innovation-comparison.md"),
    Path("defense/risk-questions.md"),
)


def active_text() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in ACTIVE_PROSE
        if path.exists()
    )


def test_active_documents_remove_frozen_or_unsupported_claims():
    text = active_text()
    forbidden = (
        r"73\.4\\?%",
        r"\+11\.3",
        r"\+2\.6",
        r"94\.8\\?%",
        r"NP[- ]complete",
        r"NP 完全",
        r"the first to",
        r"首次.*提出",
    )
    for pattern in forbidden:
        assert not re.search(pattern, text, flags=re.IGNORECASE), pattern
    assert not re.search(
        r"TODO|TBD|FIXME|待补|占位",
        text,
        flags=re.IGNORECASE,
    )


def test_professional_thesis_has_two_and_only_two_named_innovations():
    comparison = Path("defense/innovation-comparison.md").read_text(
        encoding="utf-8"
    )
    assert comparison.count("## 创新点") == 2
    assert "干扰外部性感知的跨 UAV 联合关联与配对" in comparison
    assert "面向截止时间的卸载与 MEC 算力分配" in comparison


def test_thesis_has_eight_chapters_and_uses_generated_evidence():
    chapters = sorted(Path("thesis/chapters").glob("*.tex"))
    assert len(chapters) == 8
    text = active_text()
    for macro in (
        "\\FullOnTimePercent",
        "\\InnovationOneCrossGainPP",
        "\\InnovationTwoGainPP",
        "\\ExactOptimalPercent",
        "\\FullRuntimePNinetyFiveMs",
    ):
        assert macro in text
    main = Path("thesis/main.tex").read_text(encoding="utf-8")
    assert "\\input{../paper/generated_results.tex}" in main
    assert "\\input{chapters/01-introduction.tex}" in main
    assert "\\bibliography{../paper/refs_v2}" in main

    paper = Path("paper/main_v2.tex").read_text(encoding="utf-8")
    assert "\\input{generated_results.tex}" in paper
    assert "../figures/thesis/main_on_time.pdf" in paper
    assert "\\bibliography{refs_v2}" in paper


def test_reproduction_lock_is_exact_and_portable():
    lines = [
        line.strip()
        for line in Path("requirements-lock.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines
    assert all("==" in line for line in lines)
    assert all("file://" not in line and "-e " not in line for line in lines)
