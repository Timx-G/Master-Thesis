from pathlib import Path
import re


def test_every_active_citation_has_a_bibliography_entry():
    tex_files = [
        Path("paper/main_v2.tex"),
        *sorted(Path("thesis/chapters").glob("*.tex")),
    ]
    cited = set()
    for path in tex_files:
        for group in re.findall(
            r"\\cite\{([^}]+)\}",
            path.read_text(encoding="utf-8"),
        ):
            cited.update(key.strip() for key in group.split(","))
    bibliography = Path("paper/refs_v2.bib").read_text(encoding="utf-8")
    defined = set(re.findall(r"@[A-Za-z]+\{([^,]+),", bibliography))
    assert cited
    assert not (cited - defined), (
        f"undefined citation keys: {sorted(cited - defined)}"
    )
