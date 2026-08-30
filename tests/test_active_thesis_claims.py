from pathlib import Path
import re


ACTIVE_PROSE = (
    Path("README.md"),
    Path("paper/main_v2.tex"),
    Path("thesis/main.tex"),
    *sorted(Path("thesis/chapters").glob("*.tex")),
    Path("thesis/claims-matrix.md"),
    Path("defense/outline.md"),
    Path("defense/innovation-comparison.md"),
    Path("defense/risk-questions.md"),
)

APPROVED_THESIS_TITLE = (
    "面向时限任务的多无人机移动边缘计算干扰感知调度与卸载资源分配方法研究"
)
ALGORITHM_ACRONYMS = ("EAD-JSRA", "IEA-CAP", "DD-MCAA")
REPORT_SOURCES = tuple(
    Path("deliverables/defense_2026/sources") / name
    for name in (
        "开题报告_内容草稿.md",
        "开题报告_源稿.md",
        "中期报告_内容草稿.md",
        "中期报告_源稿.md",
    )
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
    assert "干扰外部性感知的跨无人机关联配对算法" in comparison
    assert "截止时间驱动的最小成本准入分配算法" in comparison
    for acronym in ("EAD-JSRA", "IEA-CAP", "DD-MCAA"):
        assert acronym in comparison
    assert not re.search(r"\bFull method\b", comparison, flags=re.IGNORECASE)


def test_approved_title_is_exact_in_active_thesis_and_report_sources():
    title_sources = (Path("README.md"), Path("thesis/main.tex"), *REPORT_SOURCES)
    for path in title_sources:
        assert APPROVED_THESIS_TITLE in path.read_text(encoding="utf-8"), path


def test_active_public_materials_name_the_algorithm_family_without_full_label():
    groups = {
        "README": (Path("README.md"),),
        "paper": (Path("paper/main_v2.tex"),),
        "thesis": (
            Path("thesis/main.tex"),
            *(
                Path("thesis/chapters") / name
                for name in (
                    "01-introduction.tex",
                    "04-interference-aware-pairing.tex",
                    "05-deadline-allocation.tex",
                    "06-implementation-experiments.tex",
                    "08-conclusion.tex",
                )
            ),
        ),
        "defense": (
            Path("defense/outline.md"),
            Path("defense/innovation-comparison.md"),
            Path("defense/risk-questions.md"),
        ),
        **{path.name: (path,) for path in REPORT_SOURCES},
    }
    for group, paths in groups.items():
        text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        for acronym in ALGORITHM_ACRONYMS:
            assert acronym in text, (group, acronym)
        assert not re.search(r"\bfull method\b", text, flags=re.IGNORECASE), group
        assert not re.search(r"(?<![\w-])Full(?![\w-])", text), group


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
