# USTC Thesis and Computer Networks Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a Chinese professional-master thesis that passes the 2026 USTC format and 30,000-Han-character gates, plus a Computer Networks manuscript and deterministic submission package whose claims, numbers, tables, and bilingual figures consume only canonical-v2 light evidence.

**Architecture:** Official template archives, user-controlled metadata, one audited bibliography, manuscript source, and generated evidence remain separate. Result-independent prose may be written after Plans 1 and 2 pass; result-bearing prose, abstracts, discussion, conclusions, and the final package remain blocked until Plan 4 publishes a checksummed artifacts/canonical/v2/light tree. The canonical publisher invokes the publication-asset generator before finalizing light, and both manuscripts then read only those finalized light assets.

**Tech Stack:** Python 3.11, pytest, TOML, CSV, JSON, SHA-256, Matplotlib, TikZ, BibTeX, USTC ustcthesis 5.0.0, Elsevier elsarticle, Tectonic, XeLaTeX, latexmk, deterministic ZIP archives.

---

## Preconditions and execution boundary

Run every command from:

    /home/link/my_workspace/.worktrees/master-thesis-ccfb

Use:

    PY=/home/link/miniconda/envs/llm-ops-torch26/bin/python
    RUFF=/home/link/.venvs/formatters/bin/ruff

Tasks 1 through 8 require Plans 1 and 2 to be green. Task 7 must land before Plan 4 publishes canonical-v2 because the publisher calls the bilingual asset generator. Tasks 9 through 13 require:

    artifacts/canonical/v2/MANIFEST.json
    artifacts/canonical/v2/SHA256SUMS
    artifacts/canonical/v2/light/acceptance.json
    artifacts/canonical/v2/light/claim_matrix.csv

No task in this plan opens a formal seed, changes a scientific result, uploads an external artifact, signs a declaration, or submits to a school or journal system. No manuscript source may contain a hand-entered formal performance value. Unknown personal, funding, authorship, achievement, and administrative values use the exact sentinel __REQUIRED__ until the responsible human supplies and approves them.

Every commit uses explicit path arguments. Before each commit run:

    git diff --check
    git diff --cached --name-status

The cached list must contain only paths named in that task.

## Locked external sources

The implementation must preserve these already audited USTC source bytes:

    /tmp/ustc-thesis-review.0MV1Hv/USTC-LaTeX-template-2026.zip
    SHA-256 a33fcad840f842710656551b7718b8b488074ce1c23da661e4aceddfe90a72ab

    /tmp/ustc-thesis-review.0MV1Hv/USTC-thesis-template-2026.docx
    SHA-256 5828193cde8adaada42edf5639f5273ed8e271e9198d6325f9d532bd99fc6e22

    /tmp/ustc-thesis-review.0MV1Hv/engineering-master-requirements.pdf
    SHA-256 9da74888a51e496c4d445d53d0dfb50746a49362c13ae7bbbbe898fa28a8a449

The USTC archive contains ustcthesis 5.0.0, class date 2026-07-01. The Word template is dated 2026-07-17 and therefore controls any conflicting administrative or layout requirement unless the school or college supplies a still newer written rule.

The current official Elsevier LaTeX instructions page is:

    https://www.elsevier.com/en-au/researcher/author/policies-and-guidelines/latex-instructions

Its journal-article template link resolves to:

    https://assets.ctfassets.net/o78em1y1w4i4/4MpsJHO0MOJ2xZuwGTAbOZ/7bc64af36477c5d6cfce335a1f872363/elsarticle.zip

The archive bytes audited on 2026-07-19 have:

    SHA-256 0b093093e84db49f99bcc9a7c3f69ed1fb61b0147c6d296427aff7963e7f50f6

The journal requirements source is:

    https://www.sciencedirect.com/journal/computer-networks/publish/guide-for-authors

### Task 1: Freeze official USTC and Elsevier template bytes

**Files:**
- Create: publication/template-locks/ustc-2026.json
- Create: publication/template-locks/elsevier-computer-networks-2026.json
- Create: publication/template-audits/ustc-word-vs-latex-2026.md
- Create: third_party/source_archives/ustcthesis-5.0.0.zip
- Create: third_party/source_archives/ustc-thesis-word-2026-07-17.docx
- Create: third_party/source_archives/engineering-master-requirements.pdf
- Create: third_party/source_archives/elsarticle-2026-07-19.zip
- Create: third_party/ustcthesis/ustcthesis.cls
- Create: third_party/ustcthesis/ustcthesis-numeric.bst
- Create: third_party/ustcthesis/LICENSE
- Create: third_party/elsevier/elsarticle.dtx
- Create: third_party/elsevier/elsarticle.ins
- Create: third_party/elsevier/README
- Create: third_party/elsevier/manifest.txt
- Create: third_party/elsevier/elsarticle.cls
- Create: third_party/elsevier/elsarticle-num.bst
- Create: thesis/figures/ustc-name.pdf
- Create: thesis/figures/ustc-logo.pdf
- Create: thesis/figures/ustc-title-page-heading.pdf
- Create: thesis/figures/ustc-name-stxingkai.pdf
- Create: uav_mec/extract_elsarticle_class.py
- Create: tests/test_publication_template_locks.py

- [ ] **Step 1: Write the failing archive and extracted-file lock tests**

Create tests/test_publication_template_locks.py with the complete checks below:

    import hashlib
    import json
    import re
    from pathlib import Path
    from zipfile import ZipFile


    ROOT = Path(__file__).parents[1]
    LOCKS = ROOT / "publication/template-locks"
    ARCHIVES = ROOT / "third_party/source_archives"

    EXPECTED_ARCHIVE_HASHES = {
        "ustcthesis-5.0.0.zip": (
            "a33fcad840f842710656551b7718b8b488074ce1c23da661e4aceddfe90a72ab"
        ),
        "ustc-thesis-word-2026-07-17.docx": (
            "5828193cde8adaada42edf5639f5273ed8e271e9198d6325f9d532bd99fc6e22"
        ),
        "engineering-master-requirements.pdf": (
            "9da74888a51e496c4d445d53d0dfb50746a49362c13ae7bbbbe898fa28a8a449"
        ),
        "elsarticle-2026-07-19.zip": (
            "0b093093e84db49f99bcc9a7c3f69ed1fb61b0147c6d296427aff7963e7f50f6"
        ),
    }

    USTC_EXTRACTED = {
        "ustcthesis/ustcthesis.cls": ROOT / "third_party/ustcthesis/ustcthesis.cls",
        "ustcthesis/ustcthesis-numeric.bst": (
            ROOT / "third_party/ustcthesis/ustcthesis-numeric.bst"
        ),
        "ustcthesis/LICENSE": ROOT / "third_party/ustcthesis/LICENSE",
        "ustcthesis/figures/ustc-name.pdf": ROOT / "thesis/figures/ustc-name.pdf",
        "ustcthesis/figures/ustc-logo.pdf": ROOT / "thesis/figures/ustc-logo.pdf",
        "ustcthesis/figures/ustc-title-page-heading.pdf": (
            ROOT / "thesis/figures/ustc-title-page-heading.pdf"
        ),
        "ustcthesis/figures/ustc-name-stxingkai.pdf": (
            ROOT / "thesis/figures/ustc-name-stxingkai.pdf"
        ),
    }


    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()


    def load_lock(name: str) -> dict:
        return json.loads((LOCKS / name).read_text(encoding="utf-8"))


    def test_template_archives_match_frozen_hashes():
        for name, expected in EXPECTED_ARCHIVE_HASHES.items():
            path = ARCHIVES / name
            assert path.is_file(), path
            assert sha256(path) == expected


    def test_ustc_extracted_files_are_byte_identical_to_locked_archive():
        archive = ARCHIVES / "ustcthesis-5.0.0.zip"
        with ZipFile(archive) as source:
            for member, destination in USTC_EXTRACTED.items():
                assert destination.read_bytes() == source.read(member), member
        cls = (ROOT / "third_party/ustcthesis/ustcthesis.cls").read_text(
            encoding="utf-8"
        )
        assert r"\newcommand\ustcthesisversion{5.0.0}" in cls
        assert r"\ProvidesExplClass{ustcthesis}{2026-07-01}" in cls


    def test_elsevier_source_files_are_byte_identical_to_locked_archive():
        lock = load_lock("elsevier-computer-networks-2026.json")
        archive = ARCHIVES / "elsarticle-2026-07-19.zip"
        with ZipFile(archive) as source:
            members = set(source.namelist())
            for item in lock["source_files"]:
                member = item["archive_member"]
                destination = ROOT / item["repository_path"]
                assert member in members
                assert destination.read_bytes() == source.read(member)


    def test_elsevier_class_is_a_locked_derivative_not_a_claimed_zip_member():
        lock = load_lock("elsevier-computer-networks-2026.json")
        archive = ARCHIVES / "elsarticle-2026-07-19.zip"
        with ZipFile(archive) as source:
            assert "elsarticle/elsarticle.cls" not in source.namelist()
            assert "elsarticle/LICENSE" not in source.namelist()
        generated = lock["generated_class"]
        assert generated["repository_path"] == "third_party/elsevier/elsarticle.cls"
        assert generated["source_dtx"] == "elsarticle/elsarticle.dtx"
        assert generated["installer"] == "elsarticle/elsarticle.ins"
        assert generated["official_command"] == [
            "latex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "elsarticle.ins",
        ]
        assert generated["executed_recipe"]["engine"] == "tectonic"
        assert generated["executed_recipe"]["helper"] == (
            "uav_mec/extract_elsarticle_class.py"
        )
        assert generated["sha256"] == (
            "01addb492c5d075b320636ac3a30d8a5762fa937220dea07a5e107042d25384d"
        )
        assert sha256(ROOT / generated["repository_path"]) == generated["sha256"]
        cls = (ROOT / "third_party/elsevier/elsarticle.cls").read_text(
            encoding="utf-8", errors="replace"
        )
        assert r"\ProvidesClass{\@shortjid}" in cls
        assert r"\def\RCSversion{3.4}" in cls
        assert r"\def\RCSdate{2024/04/04}" in cls


    def test_elsevier_license_is_audited_from_the_files_that_exist():
        dtx = (ROOT / "third_party/elsevier/elsarticle.dtx").read_text(
            encoding="utf-8"
        )
        readme = (ROOT / "third_party/elsevier/README").read_text(
            encoding="utf-8"
        )
        assert "LaTeX Project Public" in dtx
        assert "LATEX Project Public Licence" in readme
        assert not (ROOT / "third_party/elsevier/LICENSE").exists()


    def test_locks_name_official_sources_dates_and_hashes():
        ustc = load_lock("ustc-2026.json")
        elsevier = load_lock("elsevier-computer-networks-2026.json")
        assert ustc["schema_version"] == 1
        assert ustc["latex"]["version"] == "5.0.0"
        assert ustc["latex"]["release_date"] == "2026-07-02"
        assert ustc["word"]["release_date"] == "2026-07-17"
        assert ustc["latex"]["sha256"] == EXPECTED_ARCHIVE_HASHES[
            "ustcthesis-5.0.0.zip"
        ]
        assert elsevier["schema_version"] == 1
        assert elsevier["retrieved_on"] == "2026-07-19"
        assert elsevier["archive_sha256"] == EXPECTED_ARCHIVE_HASHES[
            "elsarticle-2026-07-19.zip"
        ]
        assert elsevier["journal"] == "Computer Networks"
        for value in (
            ustc["source_page_url"],
            elsevier["instructions_url"],
            elsevier["archive_url"],
            elsevier["journal_guide_url"],
        ):
            assert value.startswith("https://")


    def test_template_difference_audit_has_all_controlling_rows():
        audit = (
            ROOT / "publication/template-audits/ustc-word-vs-latex-2026.md"
        ).read_text(encoding="utf-8")
        required = (
            "题名与学籍信息",
            "实践导师",
            "原创性与使用授权声明",
            "生成式人工智能工具声明",
            "中英文摘要与关键词",
            "目录与图表清单",
            "符号与缩略语",
            "参考文献",
            "附录",
            "致谢",
            "在读期间科研成果",
            "双面打印与书脊",
            "签名与日期",
            "控制来源",
        )
        for item in required:
            assert item in audit

- [ ] **Step 2: Run RED**

Run:

    $PY -m pytest tests/test_publication_template_locks.py -q

Expected: failures report missing locks, archives, extracted class files, official figures, and audit.

- [ ] **Step 3: Copy the audited USTC source bytes and fetch the frozen Elsevier archive**

Run:

    mkdir -p build/template-cache
    cp /tmp/ustc-thesis-review.0MV1Hv/USTC-LaTeX-template-2026.zip \
      build/template-cache/ustcthesis-5.0.0.zip
    cp /tmp/ustc-thesis-review.0MV1Hv/USTC-thesis-template-2026.docx \
      build/template-cache/ustc-thesis-word-2026-07-17.docx
    cp /tmp/ustc-thesis-review.0MV1Hv/engineering-master-requirements.pdf \
      build/template-cache/engineering-master-requirements.pdf
    curl -fL \
      https://assets.ctfassets.net/o78em1y1w4i4/4MpsJHO0MOJ2xZuwGTAbOZ/7bc64af36477c5d6cfce335a1f872363/elsarticle.zip \
      -o build/template-cache/elsarticle-2026-07-19.zip
    sha256sum build/template-cache/ustcthesis-5.0.0.zip \
      build/template-cache/ustc-thesis-word-2026-07-17.docx \
      build/template-cache/engineering-master-requirements.pdf \
      build/template-cache/elsarticle-2026-07-19.zip

Expected: the four hashes equal EXPECTED_ARCHIVE_HASHES byte-for-byte. Stop if the Elsevier URL returns different bytes; update the design review before accepting a newer archive.

Use apply_patch for JSON and Markdown files. Copy the four verified archives to the exact third_party/source_archives names. Inspect the Elsevier ZIP member names first:

    unzip -l build/template-cache/elsarticle-2026-07-19.zip

The frozen archive does not contain elsarticle.cls or a standalone LICENSE file. Extract the exact existing members with:

    mkdir -p build/template-cache/elsevier-source
    unzip -q build/template-cache/elsarticle-2026-07-19.zip \
      elsarticle/elsarticle.dtx \
      elsarticle/elsarticle.ins \
      elsarticle/elsarticle-num.bst \
      elsarticle/README \
      elsarticle/manifest.txt \
      -d build/template-cache/elsevier-source

The README's normative derivation command is `latex elsarticle.ins`, but the audited environment has Tectonic only. Implement `uav_mec/extract_elsarticle_class.py` as a deterministic docstrip-equivalent adapter: verify the locked dtx/ins hashes; in a temporary directory change only the docstrip output name from `elsarticle.cls` to `elsarticle.aux`; run Tectonic's TeX pass with `--outfmt aux --pass tex`; require the generated output to contain exactly two output-name occurrences; normalize those header/footer names back to `elsarticle.cls`; verify class version/date and the expected final hash; atomically write only the requested output. The helper rejects any other textual difference, missing occurrence, engine failure, or hash mismatch. When a full TeX installation is available, the plan additionally runs the official `latex elsarticle.ins` command and compares its bytes. Absence of `latex` is not a research-delivery blocker because the fixed Tectonic adapter is the audited fallback; the final verification records which derivation path was used.

Run the executable path available in this environment:

    /home/link/.local/bin/tectonic --version
    $PY -m uav_mec.extract_elsarticle_class \
      --archive build/template-cache/elsarticle-2026-07-19.zip \
      --tectonic /home/link/.local/bin/tectonic \
      --output build/template-cache/elsevier-source/elsarticle/elsarticle.cls
    sha256sum build/template-cache/elsevier-source/elsarticle/elsarticle.cls

Expected: class SHA-256 is `01addb492c5d075b320636ac3a30d8a5762fa937220dea07a5e107042d25384d`. The test runs the helper twice into separate tmp_path outputs and requires byte equality. This is an explicitly audited Tectonic transport for the same docstrip selection, not a claim that cls existed in the ZIP.

When a locked TeX Live 2021-or-newer environment is later available for school submission, cross-check the README's official command:

    command -v latex
    (
      set -e
      cd build/template-cache/elsevier-source/elsarticle
      latex -interaction=nonstopmode -halt-on-error elsarticle.ins
      sha256sum elsarticle.cls elsarticle-num.bst elsarticle.dtx \
        elsarticle.ins README manifest.txt
    )

Expected hashes are:

    elsarticle.cls       01addb492c5d075b320636ac3a30d8a5762fa937220dea07a5e107042d25384d
    elsarticle-num.bst   0db2e53b2378cbe5815e436c1afe9dcac67123a32406fa8f11fad7978808e202
    elsarticle.dtx       0e8bdc4d34050ec6ccd4a54e43d15ed1f2ae8ea9c07c3efc688fad75a07ffac3
    elsarticle.ins       e171080918ed70d1676f44c9197373793f34ce69763415b2a8daa99a6ce7c879
    README               95ab70d6fd901aa8a15f6dbce919d08ef72ae172c939094914753a195043db4c
    manifest.txt         98107738354036eb489d5ec5438247cccc9145cd28135c31f439120f54566165

The Tectonic helper is the executable Task 1 path and must reproduce the locked class hash. The later official-command cross-check must produce the same hash; a mismatch stops school-submission preparation and records TeX/docstrip versions rather than changing the lock. The LPPL terms remain in elsarticle.dtx, README, and the generated class; do not invent a LICENSE archive member. Do not select the CRC-only ecrc package because the official page says to use it only after journal confirmation.

After all hashes pass, copy only the verified bytes with these exact commands:

    mkdir -p third_party/source_archives third_party/ustcthesis \
      third_party/elsevier thesis/figures build/template-cache/ustc-source
    cp build/template-cache/ustcthesis-5.0.0.zip \
      third_party/source_archives/ustcthesis-5.0.0.zip
    cp build/template-cache/ustc-thesis-word-2026-07-17.docx \
      third_party/source_archives/ustc-thesis-word-2026-07-17.docx
    cp build/template-cache/engineering-master-requirements.pdf \
      third_party/source_archives/engineering-master-requirements.pdf
    cp build/template-cache/elsarticle-2026-07-19.zip \
      third_party/source_archives/elsarticle-2026-07-19.zip
    unzip -q build/template-cache/ustcthesis-5.0.0.zip \
      ustcthesis/ustcthesis.cls \
      ustcthesis/ustcthesis-numeric.bst \
      ustcthesis/LICENSE \
      ustcthesis/figures/ustc-name.pdf \
      ustcthesis/figures/ustc-logo.pdf \
      ustcthesis/figures/ustc-title-page-heading.pdf \
      ustcthesis/figures/ustc-name-stxingkai.pdf \
      -d build/template-cache/ustc-source
    cp build/template-cache/ustc-source/ustcthesis/ustcthesis.cls \
      third_party/ustcthesis/ustcthesis.cls
    cp build/template-cache/ustc-source/ustcthesis/ustcthesis-numeric.bst \
      third_party/ustcthesis/ustcthesis-numeric.bst
    cp build/template-cache/ustc-source/ustcthesis/LICENSE \
      third_party/ustcthesis/LICENSE
    cp build/template-cache/ustc-source/ustcthesis/figures/ustc-name.pdf \
      thesis/figures/ustc-name.pdf
    cp build/template-cache/ustc-source/ustcthesis/figures/ustc-logo.pdf \
      thesis/figures/ustc-logo.pdf
    cp build/template-cache/ustc-source/ustcthesis/figures/ustc-title-page-heading.pdf \
      thesis/figures/ustc-title-page-heading.pdf
    cp build/template-cache/ustc-source/ustcthesis/figures/ustc-name-stxingkai.pdf \
      thesis/figures/ustc-name-stxingkai.pdf
    cp build/template-cache/elsevier-source/elsarticle/elsarticle.dtx \
      third_party/elsevier/elsarticle.dtx
    cp build/template-cache/elsevier-source/elsarticle/elsarticle.ins \
      third_party/elsevier/elsarticle.ins
    cp build/template-cache/elsevier-source/elsarticle/elsarticle-num.bst \
      third_party/elsevier/elsarticle-num.bst
    cp build/template-cache/elsevier-source/elsarticle/README \
      third_party/elsevier/README
    cp build/template-cache/elsevier-source/elsarticle/manifest.txt \
      third_party/elsevier/manifest.txt
    cp build/template-cache/elsevier-source/elsarticle/elsarticle.cls \
      third_party/elsevier/elsarticle.cls

- [ ] **Step 4: Write the two lock documents**

publication/template-locks/ustc-2026.json contains:

    {
      "schema_version": 1,
      "source_page_url": "https://gradschool.ustc.edu.cn/column/190",
      "latex": {
        "archive": "third_party/source_archives/ustcthesis-5.0.0.zip",
        "cached_source_name": "USTC-LaTeX-template-2026.zip",
        "version": "5.0.0",
        "class_date": "2026-07-01",
        "release_date": "2026-07-02",
        "sha256": "a33fcad840f842710656551b7718b8b488074ce1c23da661e4aceddfe90a72ab"
      },
      "word": {
        "archive": "third_party/source_archives/ustc-thesis-word-2026-07-17.docx",
        "release_date": "2026-07-17",
        "sha256": "5828193cde8adaada42edf5639f5273ed8e271e9198d6325f9d532bd99fc6e22"
      },
      "professional_requirements": {
        "archive": "third_party/source_archives/engineering-master-requirements.pdf",
        "sha256": "9da74888a51e496c4d445d53d0dfb50746a49362c13ae7bbbbe898fa28a8a449"
      },
      "precedence": "newer school or college written rule, then 2026-07-17 Word template, then ustcthesis 5.0.0"
    }

publication/template-locks/elsevier-computer-networks-2026.json contains the exact URLs above, archive hash, retrieval date, journal name, and a source_files array. Each row records archive_member, repository_path, and sha256 for the five existing members listed in Step 3. It also contains generated_class with repository_path, source_dtx, installer, `official_command`, an `executed_recipe` naming the locked Tectonic executable/version and helper/hash, expected sha256, and `"license_evidence": ["third_party/elsevier/elsarticle.dtx", "third_party/elsevier/README"]`. The lock must not call the generated class byte-identical to an archive member.

- [ ] **Step 5: Write the USTC Word-versus-LaTeX audit**

Create a row for every item asserted by the test. Each row records the Word rule, LaTeX support, chosen implementation, controlling source, and human action. The practice-supervisor row states that a professional degree may omit the field only after student/college confirmation. The AI row states that the generated class text includes an independent-completion declaration and therefore cannot be signed or marked approved by automation. The printing/signature rows remain human gates and are not claimed complete by a PDF build.

- [ ] **Step 6: Run GREEN**

Run:

    $PY -m pytest tests/test_publication_template_locks.py -q
    git diff --check

Expected: all template-lock tests pass and no whitespace errors remain.

- [ ] **Step 7: Commit the frozen template source**

Run:

    git add -- publication/template-locks/ustc-2026.json \
      publication/template-locks/elsevier-computer-networks-2026.json \
      publication/template-audits/ustc-word-vs-latex-2026.md \
      third_party/source_archives/ustcthesis-5.0.0.zip \
      third_party/source_archives/ustc-thesis-word-2026-07-17.docx \
      third_party/source_archives/engineering-master-requirements.pdf \
      third_party/source_archives/elsarticle-2026-07-19.zip \
      third_party/ustcthesis/ustcthesis.cls \
      third_party/ustcthesis/ustcthesis-numeric.bst \
      third_party/ustcthesis/LICENSE \
      third_party/elsevier/elsarticle.dtx \
      third_party/elsevier/elsarticle.ins \
      third_party/elsevier/README \
      third_party/elsevier/manifest.txt \
      third_party/elsevier/elsarticle.cls \
      third_party/elsevier/elsarticle-num.bst \
      thesis/figures/ustc-name.pdf \
      thesis/figures/ustc-logo.pdf \
      thesis/figures/ustc-title-page-heading.pdf \
      thesis/figures/ustc-name-stxingkai.pdf \
      uav_mec/extract_elsarticle_class.py \
      tests/test_publication_template_locks.py
    git diff --cached --name-status
    git commit -m "chore: freeze official publication templates"

Expected: exactly the named locks, audit, official source archives, extracted files, official template figures, and test are committed.

### Task 2: Add metadata sentinels and non-automatable human gates

**Files:**
- Create: publication/metadata.toml
- Create: publication/human-gates.toml
- Create: publication/human-gates.schema.json
- Create: uav_mec/publication_metadata.py
- Create: tests/test_publication_metadata.py
- Modify: .gitignore

- [ ] **Step 1: Write the failing metadata tests**

Create tests/test_publication_metadata.py:

    from pathlib import Path

    import pytest

    from uav_mec.publication_metadata import (
        AI_USE_TEXT,
        PublicationMetadata,
        REQUIRED_SENTINEL,
        load_human_gates,
        load_publication_metadata,
        render_metadata,
        unresolved_fields,
        validate_final_metadata,
        write_rendered_metadata,
    )


    ROOT = Path(__file__).parents[1]
    METADATA = ROOT / "publication/metadata.toml"
    GATES = ROOT / "publication/human-gates.toml"


    def synthetic_resolved_metadata_and_gates(
        target: str,
    ) -> tuple[PublicationMetadata, dict[str, str]]:
        if target not in {
            "journal_manuscript",
            "journal_package",
            "journal_submission",
        }:
            raise ValueError(f"unsupported synthetic target: {target}")
        data = load_publication_metadata(METADATA)
        data.journal.update(
            {
                "authors": ["synthetic-author"],
                "author_affiliation_ids": ["aff1"],
                "author_emails": ["synthetic-author@example.invalid"],
                "corresponding_author": "synthetic-author",
                "affiliations": ["synthetic-affiliation"],
                "credit": ["synthetic-author: methodology"],
                "funding": "synthetic-funding-statement",
                "competing_interests": "synthetic-interest-statement",
                "data_availability": "synthetic-local-package-statement",
                "author_biographies": ["synthetic-biography"],
                "author_photo_files": ["synthetic-photo.png"],
                "acknowledgments": "synthetic-acknowledgments",
                "graphical_abstract_included": False,
            }
        )
        gates = {key: "approved" for key in load_human_gates(GATES)}
        gates["graphical_abstract_decision_confirmed"] = "not_applicable"
        gates["external_archive_authorized"] = "not_applicable"
        return data, gates


    def test_metadata_schema_contains_every_required_user_field():
        data = load_publication_metadata(METADATA)
        required = {
            "ustc.student_name",
            "ustc.student_id",
            "ustc.supervisors",
            "ustc.practice_supervisor_status",
            "ustc.practice_supervisor",
            "ustc.department",
            "ustc.discipline",
            "ustc.research_direction",
            "ustc.submission_date",
            "ustc.clc",
            "ustc.udc",
            "ustc.achievements",
            "ustc.acknowledgments",
            "journal.authors",
            "journal.author_affiliation_ids",
            "journal.author_emails",
            "journal.corresponding_author",
            "journal.affiliations",
            "journal.credit",
            "journal.funding",
            "journal.competing_interests",
            "journal.data_availability",
            "journal.author_biographies",
            "journal.author_photo_files",
            "journal.acknowledgments",
            "disclosure.ai_tools",
            "disclosure.ai_use_ids",
        }
        assert required <= set(data.field_paths())


    def test_unknown_values_use_only_the_exact_sentinel():
        data = load_publication_metadata(METADATA)
        unresolved = unresolved_fields(data)
        assert unresolved
        assert set(unresolved.values()) == {REQUIRED_SENTINEL}
        serialized = METADATA.read_text(encoding="utf-8")
        assert "××" not in serialized
        assert "[待" not in serialized


    def test_draft_render_is_visible_and_final_render_refuses_sentinels(
        tmp_path: Path,
    ):
        data = load_publication_metadata(METADATA)
        gates = load_human_gates(GATES)
        draft = render_metadata(data, gates, language="zh", final=False)
        assert REQUIRED_SENTINEL in draft
        with pytest.raises(ValueError, match="unresolved publication metadata"):
            validate_final_metadata(data, gates)
        with pytest.raises(ValueError, match="unresolved publication metadata"):
            render_metadata(data, gates, language="en", final=True)


    def test_human_gates_are_explicit_and_cannot_be_self_approved():
        gates = load_human_gates(GATES)
        assert set(gates) == {
            "student_identity_confirmed",
            "degree_type_confirmed",
            "practice_supervisor_confirmed",
            "advisor_order_confirmed",
            "ai_disclosure_confirmed",
            "authorship_and_credit_confirmed",
            "funding_and_interests_confirmed",
            "achievement_list_confirmed",
            "thesis_journal_overlap_confirmed",
            "signed_declarations_supplied",
            "author_photos_supplied",
            "graphical_abstract_decision_confirmed",
            "competing_interests_form_supplied",
            "exclusive_submission_confirmed",
            "cover_letter_confirmed",
            "supplementary_material_confirmed",
            "scope_review_confirmed",
            "ai_core_content_independently_rechecked",
            "ai_core_derivations_reproduced",
            "advisor_final_review_confirmed",
            "paper_electronic_consistency_confirmed",
            "duplex_spine_confirmed",
            "right_hand_chapter_openings_confirmed",
            "external_archive_authorized",
            "school_submission_authorized",
            "journal_submission_authorized",
        }
        assert gates["graphical_abstract_decision_confirmed"] == "not_applicable"
        unresolved = {
            value
            for key, value in gates.items()
            if key != "graphical_abstract_decision_confirmed"
        }
        assert unresolved == {REQUIRED_SENTINEL}


    def test_practice_supervisor_none_is_an_explicit_confirmed_state(tmp_path: Path):
        source = METADATA.read_text(encoding="utf-8").replace(
            'practice_supervisor_status = "__REQUIRED__"',
            'practice_supervisor_status = "none"',
        ).replace(
            'practice_supervisor = "__REQUIRED__"',
            'practice_supervisor = "not_applicable"',
        )
        path = tmp_path / "metadata.toml"
        path.write_text(source, encoding="utf-8")
        data = load_publication_metadata(path)
        assert data.ustc["practice_supervisor_status"] == "none"
        assert data.ustc["practice_supervisor"] == "not_applicable"


    def test_journal_manuscript_allows_submission_only_biography_and_photo_sentinels():
        data, gates = synthetic_resolved_metadata_and_gates("journal_manuscript")
        data.journal["author_biographies"] = [REQUIRED_SENTINEL]
        data.journal["author_photo_files"] = [REQUIRED_SENTINEL]
        validate_final_metadata(data, gates, target="journal_manuscript")
        with pytest.raises(ValueError, match="unresolved publication metadata"):
            validate_final_metadata(data, gates, target="journal_package")


    def test_journal_package_requires_biographies_photos_but_not_upload_authority():
        data, gates = synthetic_resolved_metadata_and_gates("journal_package")
        gates["journal_submission_authorized"] = REQUIRED_SENTINEL
        validate_final_metadata(data, gates, target="journal_package")
        with pytest.raises(ValueError, match="journal_submission_authorized"):
            validate_final_metadata(data, gates, target="journal_submission")


    def test_journal_manuscript_still_rejects_a_required_core_field():
        data, gates = synthetic_resolved_metadata_and_gates("journal_manuscript")
        data.journal["funding"] = REQUIRED_SENTINEL
        with pytest.raises(ValueError, match="journal.funding"):
            validate_final_metadata(data, gates, target="journal_manuscript")


    def test_rendered_ustc_metadata_maps_every_official_setup_key(tmp_path: Path):
        data = load_publication_metadata(METADATA)
        paths = write_rendered_metadata(
            data,
            load_human_gates(GATES),
            tmp_path,
            final=False,
        )
        thesis = paths[0].read_text(encoding="utf-8")
        for key in (
            "title",
            "title*",
            "author",
            "student-id",
            "department",
            "discipline",
            "supervisor",
            "practice-supervisor",
            "research-direction",
            "date",
            "clc",
            "udc",
        ):
            assert f"  {key} =" in thesis


    def test_practice_supervisor_none_omits_key_and_not_applicable_text(tmp_path: Path):
        data = load_publication_metadata(METADATA)
        data.ustc["practice_supervisor_status"] = "none"
        data.ustc["practice_supervisor"] = "not_applicable"
        thesis_path, _journal_path, _mode_path = write_rendered_metadata(
            data,
            load_human_gates(GATES),
            tmp_path,
            final=False,
        )
        thesis = thesis_path.read_text(encoding="utf-8")
        assert "practice-supervisor =" not in thesis
        assert "not_applicable" not in thesis


    def test_ai_use_ids_are_semantically_equal_and_english_has_no_han(tmp_path: Path):
        data, gates = synthetic_resolved_metadata_and_gates("journal_manuscript")
        _thesis_path, journal_path, _mode_path = write_rendered_metadata(
            data,
            gates,
            tmp_path,
            final=True,
            target="journal_manuscript",
        )
        english = journal_path.read_text(encoding="utf-8")
        assert not re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", english)
        assert "was used for" in english
        for use_id in data.disclosure["ai_use_ids"]:
            assert AI_USE_TEXT[use_id]["en"] in english

- [ ] **Step 2: Run RED**

Run:

    $PY -m pytest tests/test_publication_metadata.py -q

Expected: import failure because the metadata module and schema files do not exist.

- [ ] **Step 3: Create the exact metadata source**

publication/metadata.toml uses these sections and keys:

    schema_version = 1

    [ustc]
    title_zh = "多无人机边缘计算的干扰外部性感知调度与时限卸载"
    title_en = "Externality-Aware Scheduling and Deadline-Constrained Offloading for Multi-UAV Edge Computing"
    student_name = "__REQUIRED__"
    student_id = "__REQUIRED__"
    supervisors = ["__REQUIRED__"]
    practice_supervisor_status = "__REQUIRED__"
    practice_supervisor = "__REQUIRED__"
    department = "__REQUIRED__"
    discipline = "__REQUIRED__"
    research_direction = "__REQUIRED__"
    submission_date = "__REQUIRED__"
    clc = "__REQUIRED__"
    udc = "__REQUIRED__"
    achievements = ["__REQUIRED__"]
    acknowledgments = "__REQUIRED__"

    [journal]
    title = "Externality-Aware Scheduling and Deadline-Constrained Offloading for Multi-UAV Edge Computing"
    authors = ["__REQUIRED__"]
    author_affiliation_ids = ["__REQUIRED__"]
    author_emails = ["__REQUIRED__"]
    corresponding_author = "__REQUIRED__"
    affiliations = ["__REQUIRED__"]
    credit = ["__REQUIRED__"]
    funding = "__REQUIRED__"
    competing_interests = "__REQUIRED__"
    data_availability = "__REQUIRED__"
    author_biographies = ["__REQUIRED__"]
    author_photo_files = ["__REQUIRED__"]
    acknowledgments = "__REQUIRED__"
    graphical_abstract_included = false

    [disclosure]
    ai_tools = ["Codex"]
    ai_use_ids = [
      "project_audit",
      "code_and_test_assistance",
      "literature_organization",
      "data_analysis_checking",
      "structural_revision",
      "language_revision"
    ]

Do not add a personal value from README, Git history, an old manuscript, or a defense file. The student supplies values directly.

publication/human-gates.toml has schema_version = 1 and a gates table containing every key asserted by the test. `graphical_abstract_decision_confirmed = "not_applicable"` is frozen because the journal does not require a graphical abstract and this plan implements no graphical-abstract branch; every other gate starts as __REQUIRED__. The three authorization gates remain independent: external archival authorization does not imply school or journal submission authorization.

publication/human-gates.schema.json defines __REQUIRED__, approved, rejected, and not_applicable. Only practice supervisor, external archive, and the frozen-disabled graphical abstract may use not_applicable. School submission additionally requires the student's independent recheck of AI-assisted core content, independent reproduction of core derivations, advisor final review, paper/electronic consistency, duplex/spine inspection, and right-hand chapter-opening inspection. Submission authorization must be approved only for the corresponding external upload and is not required for research delivery or local journal-package validation.

- [ ] **Step 4: Implement the typed metadata API**

Create uav_mec/publication_metadata.py with this complete first implementation:

    from __future__ import annotations

    import json
    import tomllib
    from collections.abc import Mapping
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Literal


    REQUIRED_SENTINEL = "__REQUIRED__"
    GATE_VALUES = frozenset(
        {REQUIRED_SENTINEL, "approved", "rejected", "not_applicable"}
    )
    AI_USE_TEXT = {
        "project_audit": {"zh": "项目审计", "en": "project auditing"},
        "code_and_test_assistance": {
            "zh": "代码与测试辅助",
            "en": "code and test assistance",
        },
        "literature_organization": {
            "zh": "文献组织",
            "en": "literature organization",
        },
        "data_analysis_checking": {
            "zh": "数据分析检查",
            "en": "data-analysis checking",
        },
        "structural_revision": {
            "zh": "结构重写",
            "en": "structural revision",
        },
        "language_revision": {"zh": "语言修改", "en": "language revision"},
    }
    USTC_REQUIRED_FIELDS = (
        "ustc.title_zh",
        "ustc.title_en",
        "ustc.student_name",
        "ustc.student_id",
        "ustc.supervisors",
        "ustc.practice_supervisor_status",
        "ustc.practice_supervisor",
        "ustc.department",
        "ustc.discipline",
        "ustc.research_direction",
        "ustc.submission_date",
        "ustc.clc",
        "ustc.udc",
        "ustc.achievements",
        "ustc.acknowledgments",
        "disclosure.ai_tools",
        "disclosure.ai_use_ids",
    )
    JOURNAL_MANUSCRIPT_REQUIRED_FIELDS = (
        "journal.title",
        "journal.authors",
        "journal.author_affiliation_ids",
        "journal.author_emails",
        "journal.corresponding_author",
        "journal.affiliations",
        "journal.credit",
        "journal.funding",
        "journal.competing_interests",
        "journal.data_availability",
        "journal.acknowledgments",
        "journal.graphical_abstract_included",
        "disclosure.ai_tools",
        "disclosure.ai_use_ids",
    )
    TARGET_REQUIRED_FIELDS = {
        "thesis_build": USTC_REQUIRED_FIELDS,
        "school_submission": USTC_REQUIRED_FIELDS,
        "journal_manuscript": JOURNAL_MANUSCRIPT_REQUIRED_FIELDS,
        "journal_package": JOURNAL_MANUSCRIPT_REQUIRED_FIELDS
        + ("journal.author_biographies", "journal.author_photo_files"),
        "journal_submission": JOURNAL_MANUSCRIPT_REQUIRED_FIELDS
        + ("journal.author_biographies", "journal.author_photo_files"),
    }
    TARGET_GATES = {
        "thesis_build": {
            "student_identity_confirmed",
            "degree_type_confirmed",
            "practice_supervisor_confirmed",
            "advisor_order_confirmed",
            "ai_disclosure_confirmed",
            "achievement_list_confirmed",
            "thesis_journal_overlap_confirmed",
            "scope_review_confirmed",
        },
        "school_submission": {
            "student_identity_confirmed",
            "degree_type_confirmed",
            "practice_supervisor_confirmed",
            "advisor_order_confirmed",
            "ai_disclosure_confirmed",
            "achievement_list_confirmed",
            "thesis_journal_overlap_confirmed",
            "signed_declarations_supplied",
            "ai_core_content_independently_rechecked",
            "ai_core_derivations_reproduced",
            "advisor_final_review_confirmed",
            "paper_electronic_consistency_confirmed",
            "duplex_spine_confirmed",
            "right_hand_chapter_openings_confirmed",
            "school_submission_authorized",
        },
        "journal_manuscript": {
            "ai_disclosure_confirmed",
            "authorship_and_credit_confirmed",
            "funding_and_interests_confirmed",
            "thesis_journal_overlap_confirmed",
            "graphical_abstract_decision_confirmed",
            "scope_review_confirmed",
        },
        "journal_package": {
            "ai_disclosure_confirmed",
            "authorship_and_credit_confirmed",
            "funding_and_interests_confirmed",
            "thesis_journal_overlap_confirmed",
            "graphical_abstract_decision_confirmed",
            "scope_review_confirmed",
            "author_photos_supplied",
            "competing_interests_form_supplied",
            "exclusive_submission_confirmed",
            "cover_letter_confirmed",
            "supplementary_material_confirmed",
        },
        "journal_submission": {
            "ai_disclosure_confirmed",
            "authorship_and_credit_confirmed",
            "funding_and_interests_confirmed",
            "thesis_journal_overlap_confirmed",
            "graphical_abstract_decision_confirmed",
            "scope_review_confirmed",
            "author_photos_supplied",
            "competing_interests_form_supplied",
            "exclusive_submission_confirmed",
            "cover_letter_confirmed",
            "supplementary_material_confirmed",
            "journal_submission_authorized",
        },
    }
    NOT_APPLICABLE_GATES = {
        "practice_supervisor_confirmed",
        "external_archive_authorized",
        "graphical_abstract_decision_confirmed",
    }


    @dataclass(frozen=True)
    class PublicationMetadata:
        schema_version: int
        ustc: dict[str, object]
        journal: dict[str, object]
        disclosure: dict[str, object]

        def field_paths(self) -> list[str]:
            paths: list[str] = []
            for section_name, section in (
                ("ustc", self.ustc),
                ("journal", self.journal),
                ("disclosure", self.disclosure),
            ):
                paths.extend(_leaf_paths(section_name, section))
            return paths


    def _leaf_paths(prefix: str, value: object) -> list[str]:
        if isinstance(value, Mapping):
            paths: list[str] = []
            for key in sorted(value):
                paths.extend(_leaf_paths(f"{prefix}.{key}", value[key]))
            return paths
        return [prefix]


    def _contains_required(value: object) -> bool:
        if value == REQUIRED_SENTINEL:
            return True
        if isinstance(value, Mapping):
            return any(_contains_required(item) for item in value.values())
        if isinstance(value, list):
            return any(_contains_required(item) for item in value)
        return False


    def _blank_paths(prefix: str, value: object) -> list[str]:
        if isinstance(value, Mapping):
            paths: list[str] = []
            for key in sorted(value):
                paths.extend(_blank_paths(f"{prefix}.{key}", value[key]))
            return paths
        if isinstance(value, list):
            if not value or any(isinstance(item, str) and not item.strip() for item in value):
                return [prefix]
            return []
        if isinstance(value, str) and not value.strip():
            return [prefix]
        return []


    def load_publication_metadata(path: Path) -> PublicationMetadata:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1:
            raise ValueError("publication metadata schema_version must be 1")
        sections: dict[str, dict[str, object]] = {}
        for name in ("ustc", "journal", "disclosure"):
            value = raw.get(name)
            if not isinstance(value, dict):
                raise ValueError(f"publication metadata section {name} must be a table")
            sections[name] = value
        return PublicationMetadata(
            schema_version=1,
            ustc=sections["ustc"],
            journal=sections["journal"],
            disclosure=sections["disclosure"],
        )


    def load_human_gates(path: Path) -> dict[str, str]:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1 or not isinstance(raw.get("gates"), dict):
            raise ValueError("human-gates schema_version and gates table are required")
        gates = raw["gates"]
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in gates.items()):
            raise ValueError("human gate names and values must be strings")
        invalid = {key: value for key, value in gates.items() if value not in GATE_VALUES}
        if invalid:
            raise ValueError(f"invalid human gate values: {invalid}")
        return dict(gates)


    def unresolved_fields(data: PublicationMetadata) -> dict[str, str]:
        sections = {
            "ustc": data.ustc,
            "journal": data.journal,
            "disclosure": data.disclosure,
        }
        return {
            path: REQUIRED_SENTINEL
            for section_name, section in sections.items()
            for path in _leaf_paths(section_name, section)
            if _contains_required(_resolve_path(sections, path))
        }


    def _resolve_path(sections: Mapping[str, object], path: str) -> object:
        value: object = sections
        for part in path.split("."):
            if not isinstance(value, Mapping):
                raise ValueError(f"metadata path is not a mapping: {path}")
            value = value[part]
        return value


    def localized_ai_uses(
        data: PublicationMetadata,
        language: Literal["zh", "en"],
    ) -> list[str]:
        use_ids = data.disclosure.get("ai_use_ids")
        if not isinstance(use_ids, list) or not use_ids:
            raise ValueError("disclosure.ai_use_ids must be a nonempty list")
        if len(use_ids) != len(set(use_ids)):
            raise ValueError("disclosure.ai_use_ids must be unique")
        try:
            return [AI_USE_TEXT[use_id][language] for use_id in use_ids]
        except KeyError as error:
            raise ValueError(f"unknown AI use semantic ID: {error.args[0]}") from error


    def validate_final_metadata(
        data: PublicationMetadata,
        gates: Mapping[str, str],
        *,
        target: Literal[
            "thesis_build",
            "school_submission",
            "journal_manuscript",
            "journal_package",
            "journal_submission",
        ] = "thesis_build",
    ) -> None:
        sections = {
            "ustc": data.ustc,
            "journal": data.journal,
            "disclosure": data.disclosure,
        }
        required_paths = TARGET_REQUIRED_FIELDS[target]
        unresolved = {
            path: REQUIRED_SENTINEL
            for path in required_paths
            if _contains_required(_resolve_path(sections, path))
        }
        if unresolved:
            raise ValueError(f"unresolved publication metadata: {sorted(unresolved)}")
        blank = [
            path
            for path in required_paths
            if _blank_paths(path, _resolve_path(sections, path))
        ]
        if blank:
            raise ValueError(f"blank publication metadata: {sorted(blank)}")

        if target.startswith("thesis") or target == "school_submission":
            status = data.ustc["practice_supervisor_status"]
            name = data.ustc["practice_supervisor"]
            if status == "none" and name != "not_applicable":
                raise ValueError("practice supervisor must be not_applicable when status is none")
            if status == "present" and name in {"not_applicable", REQUIRED_SENTINEL}:
                raise ValueError("practice supervisor name is required when status is present")
            if status not in {"none", "present"}:
                raise ValueError("practice supervisor status must be none or present")
            practice_gate = gates.get("practice_supervisor_confirmed")
            if status == "none" and practice_gate != "not_applicable":
                raise ValueError("practice supervisor absence needs a not_applicable gate")
            if status == "present" and practice_gate != "approved":
                raise ValueError("practice supervisor presence needs an approved gate")
        else:
            authors = data.journal["authors"]
            if not isinstance(authors, list) or any(
                not isinstance(author, str) for author in authors
            ):
                raise ValueError("journal.authors must be a list of strings")
            author_count = len(authors)
            parallel_author_fields = [
                "author_affiliation_ids",
                "author_emails",
                "credit",
            ]
            if target in {"journal_package", "journal_submission"}:
                parallel_author_fields.extend(
                    ["author_biographies", "author_photo_files"]
                )
            for field in parallel_author_fields:
                values = data.journal[field]
                if not isinstance(values, list) or len(values) != author_count:
                    raise ValueError(f"journal.{field} must have one item per author")
            if data.journal["corresponding_author"] not in data.journal["authors"]:
                raise ValueError("corresponding author must name one listed author")

        for key in sorted(TARGET_GATES[target]):
            value = gates.get(key, REQUIRED_SENTINEL)
            if value == REQUIRED_SENTINEL:
                raise ValueError(f"unresolved human gate: {key}")
            if value == "rejected":
                raise ValueError(f"human gate rejected: {key}")
            if value == "not_applicable" and key not in NOT_APPLICABLE_GATES:
                raise ValueError(f"human gate cannot be not_applicable: {key}")
            if value not in {"approved", "not_applicable"}:
                raise ValueError(f"human gate is not approved: {key}")

        ai_use_ids = data.disclosure["ai_use_ids"]
        if not isinstance(ai_use_ids, list) or not ai_use_ids:
            raise ValueError("disclosure.ai_use_ids must be a nonempty list")
        if len(ai_use_ids) != len(set(ai_use_ids)) or any(
            use_id not in AI_USE_TEXT for use_id in ai_use_ids
        ):
            raise ValueError("disclosure.ai_use_ids contains a duplicate or unknown ID")

        if target.startswith("journal"):
            graphical_gate = gates.get("graphical_abstract_decision_confirmed")
            if data.journal["graphical_abstract_included"] is not False:
                raise ValueError("graphical abstract branch is locked disabled")
            if graphical_gate != "not_applicable":
                raise ValueError("disabled graphical abstract needs a not_applicable gate")


    def render_metadata(
        data: PublicationMetadata,
        gates: Mapping[str, str],
        *,
        language: Literal["zh", "en"],
        final: bool,
        target: Literal[
            "thesis_build",
            "school_submission",
            "journal_manuscript",
            "journal_package",
            "journal_submission",
        ] | None = None,
    ) -> str:
        if final:
            selected_target = target or (
                "thesis_build" if language == "zh" else "journal_manuscript"
            )
            validate_final_metadata(data, gates, target=selected_target)
        section_name = "ustc" if language == "zh" else "journal"
        section = data.ustc if language == "zh" else data.journal
        rows = [f"language={language}"]
        for key in sorted(section):
            encoded = json.dumps(section[key], ensure_ascii=False, sort_keys=True)
            rows.append(f"{section_name}.{key}={encoded}")
        encoded_tools = json.dumps(
            data.disclosure["ai_tools"], ensure_ascii=False, sort_keys=True
        )
        encoded_ids = json.dumps(
            data.disclosure["ai_use_ids"], ensure_ascii=False, sort_keys=True
        )
        encoded_uses = json.dumps(
            localized_ai_uses(data, language), ensure_ascii=False, sort_keys=True
        )
        rows.append(f"disclosure.ai_tools={encoded_tools}")
        rows.append(f"disclosure.ai_use_ids={encoded_ids}")
        rows.append(f"disclosure.ai_use_text={encoded_uses}")
        return "\n".join(rows) + "\n"

The module never changes a human gate. The implementation reports a list field at its dotted parent path, so a sentinel inside authors or achievements cannot be hidden by an array index. Add unit cases for blank scalars, blank list elements, invalid gate values, present/none practice-supervisor mismatches, and each target value before leaving this task. The build targets explicitly do not require signed declarations, school/journal upload authorization, or author photos. Only school_submission and journal_submission require their respective administrative gates.

- [ ] **Step 5: Protect private human-supplied files**

Append these exact .gitignore rules:

    publication/private/
    publication/metadata.local.toml
    publication/human-gates.local.toml

The tracked sentinel files remain examples and schema sources. Final local builds may pass explicit private paths; no task copies private values back into tracked files.

- [ ] **Step 6: Run GREEN**

Run:

    $PY -m pytest tests/test_publication_metadata.py -q
    $RUFF check uav_mec/publication_metadata.py tests/test_publication_metadata.py

Expected: all metadata tests pass and lint is clean.

- [ ] **Step 7: Commit the metadata boundary**

Run:

    git add -- .gitignore publication/metadata.toml \
      publication/human-gates.toml \
      publication/human-gates.schema.json \
      uav_mec/publication_metadata.py \
      tests/test_publication_metadata.py
    git diff --cached --name-status
    git commit -m "feat: add publication metadata and human gates"

Expected: exactly the six named paths are committed; no private metadata is staged.

### Task 3: Establish one audited bibliography for both manuscripts

**Files:**
- Move: paper/refs_v2.bib to references/references.bib
- Create: references/metadata_audit.csv
- Create: uav_mec/publication_bibliography.py
- Modify: tests/test_latex_citations.py
- Create: tests/test_publication_bibliography.py
- Modify: thesis/main.tex
- Modify: paper/main_v2.tex

- [ ] **Step 1: Write the failing bibliography audit tests**

Create tests/test_publication_bibliography.py:

    import csv
    from pathlib import Path

    from uav_mec.publication_bibliography import (
        bibliography_keys,
        citation_keys,
        validate_audit,
    )


    ROOT = Path(__file__).parents[1]
    BIB = ROOT / "references/references.bib"
    AUDIT = ROOT / "references/metadata_audit.csv"


    def active_tex_files() -> list[Path]:
        return [
            ROOT / "thesis/main.tex",
            *sorted((ROOT / "thesis").glob("**/*.tex")),
            ROOT / "paper/main_v2.tex",
            *sorted((ROOT / "paper").glob("**/*.tex")),
        ]


    def test_both_manuscripts_use_the_single_bibliography():
        thesis = (ROOT / "thesis/main.tex").read_text(encoding="utf-8")
        paper = (ROOT / "paper/main_v2.tex").read_text(encoding="utf-8")
        assert "../references/references.bib" in thesis
        assert "../references/references" in paper
        assert "refs_v2" not in thesis + paper


    def test_every_citation_resolves_and_every_bib_key_is_unique():
        defined = bibliography_keys(BIB)
        cited = citation_keys(active_tex_files())
        assert len(defined) == len(set(defined))
        assert cited
        assert cited <= set(defined)


    def test_metadata_audit_has_one_verified_row_per_bib_key():
        with AUDIT.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert rows
        assert set(rows[0]) == {
            "key",
            "entry_type",
            "doi",
            "title_verified",
            "authors_verified",
            "year_verified",
            "venue_verified",
            "volume_pages_verified",
            "source_url",
            "checked_on",
            "status",
        }
        assert {row["key"] for row in rows} == set(bibliography_keys(BIB))
        validate_audit(BIB, AUDIT)


    def test_closest_work_citations_are_present():
        required = {
            "guo2019externality",
            "xu2020multiapnoma",
            "katwe2022multiuavnoma",
            "uwizeyimana2024association",
            "chen2026qos",
            "lyu2018deadline",
        }
        assert required <= set(bibliography_keys(BIB))

- [ ] **Step 2: Expand the active citation test before changing paths**

Replace tests/test_latex_citations.py with a recursive active-source scan. Exclude archive, build, third_party, and artifacts. Assert citation-to-bibliography and bibliography-to-citation agreement for active manuscripts; permit an uncited standards/reference entry only when metadata_audit.csv marks status = background_source. Reject inline thebibliography environments in active sources.

- [ ] **Step 3: Run RED**

Run:

    $PY -m pytest tests/test_latex_citations.py \
      tests/test_publication_bibliography.py -q

Expected: failures report the missing shared path, audit, API, and required closest-work keys.

- [ ] **Step 4: Move the dirty bibliography without losing bytes**

Before moving, record:

    sha256sum paper/refs_v2.bib
    git diff -- paper/refs_v2.bib

Then use:

    mkdir -p references
    git mv paper/refs_v2.bib references/references.bib

Re-run sha256sum and confirm the content hash is unchanged before adding audited entries. Preserve every user edit.

- [ ] **Step 5: Implement bibliography parsing and audit validation**

Create uav_mec/publication_bibliography.py with this complete implementation:

    from __future__ import annotations

    import csv
    import re
    from collections.abc import Iterable
    from datetime import date
    from pathlib import Path


    ENTRY_RE = re.compile(r"(?im)^\s*@([a-z]+)\s*\{\s*([^,\s]+)\s*,")
    CITE_RE = re.compile(
        r"\\(?:cite|citep|citet|citealp|citealt|citeauthor|citeyear|nocite)"
        r"\*?(?:\s*\[[^\]]*\]){0,2}\s*\{([^}]*)\}"
    )
    DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
    AUDIT_FIELDS = (
        "key",
        "entry_type",
        "doi",
        "title_verified",
        "authors_verified",
        "year_verified",
        "venue_verified",
        "volume_pages_verified",
        "source_url",
        "checked_on",
        "status",
    )
    STATUS_VALUES = {"verified", "background_source", "no_doi_expected"}
    VOLUME_NOT_APPLICABLE = {
        "book",
        "booklet",
        "manual",
        "misc",
        "techreport",
    }


    def _entries(path: Path) -> list[tuple[str, str]]:
        text = path.read_text(encoding="utf-8")
        return [
            (match.group(1).lower(), match.group(2))
            for match in ENTRY_RE.finditer(text)
        ]


    def bibliography_keys(path: Path) -> list[str]:
        return [key for _entry_type, key in _entries(path)]


    def citation_keys(paths: Iterable[Path]) -> set[str]:
        keys: set[str] = set()
        for path in dict.fromkeys(paths):
            text = path.read_text(encoding="utf-8")
            for match in CITE_RE.finditer(text):
                for key in match.group(1).split(","):
                    normalized = key.strip()
                    if normalized and normalized != "*":
                        keys.add(normalized)
        return keys


    def _normalized_doi(value: str) -> str:
        normalized = value.strip().lower()
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        return normalized


    def validate_audit(bib_path: Path, audit_path: Path) -> None:
        entries = _entries(bib_path)
        keys = [key for _entry_type, key in entries]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate bibliography keys")
        entry_types = {key: entry_type for entry_type, key in entries}

        with audit_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != AUDIT_FIELDS:
                raise ValueError("bibliography audit header mismatch")
            rows = list(reader)
        audit_keys = [row["key"] for row in rows]
        if len(audit_keys) != len(set(audit_keys)):
            raise ValueError("duplicate bibliography audit keys")
        if set(audit_keys) != set(keys):
            missing = sorted(set(keys) - set(audit_keys))
            unexpected = sorted(set(audit_keys) - set(keys))
            raise ValueError(f"bibliography audit coverage mismatch: {missing=}, {unexpected=}")

        seen_dois: set[str] = set()
        for row in rows:
            key = row["key"]
            entry_type = entry_types[key]
            if row["entry_type"].lower() != entry_type:
                raise ValueError(f"entry type mismatch for {key}")
            if row["status"] not in STATUS_VALUES:
                raise ValueError(f"invalid audit status for {key}")
            if not row["source_url"].startswith("https://"):
                raise ValueError(f"verified HTTPS source URL required for {key}")
            try:
                date.fromisoformat(row["checked_on"])
            except ValueError as error:
                raise ValueError(f"invalid checked_on date for {key}") from error

            for field in (
                "title_verified",
                "authors_verified",
                "year_verified",
                "venue_verified",
            ):
                if row[field] != "true":
                    raise ValueError(f"{field} must be true for {key}")
            volume_status = row["volume_pages_verified"]
            if volume_status == "not_applicable":
                if entry_type not in VOLUME_NOT_APPLICABLE:
                    raise ValueError(f"volume/pages cannot be not_applicable for {key}")
            elif volume_status != "true":
                raise ValueError(f"volume/pages audit must be true for {key}")

            doi_cell = row["doi"].strip()
            if doi_cell.lower().startswith("no-doi:"):
                reason = doi_cell.partition(":")[2].strip()
                if row["status"] != "no_doi_expected" or not reason:
                    raise ValueError(f"written no-DOI reason required for {key}")
                if entry_type not in {"article", "inproceedings"}:
                    raise ValueError(f"no-doi reason form is only for papers: {key}")
                continue
            if not doi_cell:
                if row["status"] != "no_doi_expected" or entry_type not in VOLUME_NOT_APPLICABLE:
                    raise ValueError(f"DOI or accepted no-DOI status required for {key}")
                continue
            doi = _normalized_doi(doi_cell)
            if not DOI_RE.fullmatch(doi):
                raise ValueError(f"malformed DOI for {key}")
            if doi in seen_dois:
                raise ValueError(f"duplicate DOI: {doi}")
            seen_dois.add(doi)

Add focused tests for duplicate BibTeX keys, duplicate DOI after normalization, an invalid calendar date, a blank source URL, a no-DOI paper without a reason, and an article whose volume/pages flag is not_applicable. Each test writes its own complete BibTeX and CSV fixture under tmp_path; it may not read a fixture promised by another task.

- [ ] **Step 6: Audit every entry and add the six frozen closest works**

Populate references/metadata_audit.csv from primary DOI/publisher pages. Add or correct the six keys asserted by the test. Do not relabel the Guo adaptation as an original implementation; its citation and adaptation contract remain distinct. Do not import a citation solely because an older manuscript used it. Remove duplicate metadata only after comparing all fields.

Update the two current drivers to the shared path. The later template tasks will replace their bibliography styles.

- [ ] **Step 7: Run GREEN**

Run:

    $PY -m pytest tests/test_latex_citations.py \
      tests/test_publication_bibliography.py -q
    $RUFF check uav_mec/publication_bibliography.py \
      tests/test_latex_citations.py tests/test_publication_bibliography.py

Expected: all citations resolve to the one bibliography and every bibliography entry has exactly one accepted audit row.

- [ ] **Step 8: Commit the shared bibliography**

Run:

    git add -- references/references.bib references/metadata_audit.csv \
      uav_mec/publication_bibliography.py \
      tests/test_latex_citations.py \
      tests/test_publication_bibliography.py \
      thesis/main.tex paper/main_v2.tex
    git diff --cached --name-status
    git commit -m "refactor: establish one audited bibliography"

Expected: the rename is visible, the old paper/refs_v2.bib path is absent, and only the named paths are committed.

### Task 4: Replace ctexbook with the official nine-chapter USTC driver

**Files:**
- Create: thesis/latexmkrc
- Create: thesis/ustcsetup.tex
- Create: thesis/frontmatter/abstract.tex
- Create: thesis/frontmatter/notation.tex
- Create: thesis/backmatter/acknowledgments.tex
- Create: thesis/backmatter/achievements.tex
- Create: thesis/chapters/07-results-mechanism.tex
- Move: thesis/chapters/07-deployment-limitations.tex to thesis/chapters/08-deployment-limitations.tex
- Move: thesis/chapters/08-conclusion.tex to thesis/chapters/09-conclusion.tex
- Modify: thesis/main.tex
- Modify: uav_mec/publication_metadata.py
- Create: tests/test_ustc_thesis_structure.py

- [ ] **Step 1: Write the failing USTC structure tests**

Create tests/test_ustc_thesis_structure.py:

    import re
    from pathlib import Path


    ROOT = Path(__file__).parents[1]
    MAIN = ROOT / "thesis/main.tex"

    CHAPTERS = (
        "01-introduction",
        "02-related-work",
        "03-system-model",
        "04-interference-aware-pairing",
        "05-deadline-allocation",
        "06-implementation-methodology",
        "07-results-mechanism",
        "08-deployment-limitations",
        "09-conclusion",
    )


    def compact(text: str) -> str:
        return re.sub(r"\s+", "", text)


    def test_ustc_driver_uses_the_exact_professional_master_options():
        text = compact(MAIN.read_text(encoding="utf-8"))
        assert (
            r"\documentclass[degree=master,degree-type=professional,"
            r"language=chinese]{ustcthesis}"
        ) in text
        assert "ctexbook" not in text


    def test_ustc_front_and_back_matter_order_is_complete():
        text = MAIN.read_text(encoding="utf-8")
        ordered = (
            r"\maketitle",
            r"\PublicationOriginalityDeclaration",
            r"\PublicationAIUsageDeclaration",
            r"\frontmatter",
            r"\include{frontmatter/abstract}",
            r"\tableofcontents",
            r"\listoffigures",
            r"\listoftables",
            r"\include{frontmatter/notation}",
            r"\mainmatter",
            r"\printbibliography",
            r"\appendix",
            r"\backmatter",
            r"\include{backmatter/acknowledgments}",
            r"\include{backmatter/achievements}",
        )
        positions = [text.index(token) for token in ordered]
        assert positions == sorted(positions)


    def test_ustc_driver_includes_exactly_nine_numbered_chapters():
        text = MAIN.read_text(encoding="utf-8")
        includes = re.findall(r"\\include\{chapters/([^}]+)\}", text)
        assert tuple(includes) == CHAPTERS
        for name in CHAPTERS:
            assert (ROOT / f"thesis/chapters/{name}.tex").is_file()


    def test_ustc_uses_shared_bibliography_and_official_numeric_style():
        setup = (ROOT / "thesis/ustcsetup.tex").read_text(encoding="utf-8")
        assert r"\bibliographystyle{ustcthesis-numeric}" in setup
        assert r"\addbibresource{../references/references.bib}" in setup
        assert "plainnat" not in setup + MAIN.read_text(encoding="utf-8")


    def test_ustc_setup_uses_runner_supplied_roots_not_a_hardcoded_build_path():
        setup = (ROOT / "thesis/ustcsetup.tex").read_text(encoding="utf-8")
        assert r"\PublicationMetadataRoot" in setup
        assert r"\PublicationCanonicalLightRoot" in setup
        assert "build/publication/metadata" not in setup


    def test_result_chapter_has_only_a_canonical_light_insertion_point():
        path = ROOT / "thesis/chapters/07-results-mechanism.tex"
        text = path.read_text(encoding="utf-8")
        assert r"\PublicationCanonicalInput{latex/thesis/results-body.tex}" in text
        assert "canonical/v1" not in text
        assert "../paper/generated_" not in text


    def test_front_and_back_matter_use_official_environments():
        abstract = (ROOT / "thesis/frontmatter/abstract.tex").read_text(
            encoding="utf-8"
        )
        notation = (ROOT / "thesis/frontmatter/notation.tex").read_text(
            encoding="utf-8"
        )
        acknowledgments = (
            ROOT / "thesis/backmatter/acknowledgments.tex"
        ).read_text(encoding="utf-8")
        achievements = (ROOT / "thesis/backmatter/achievements.tex").read_text(
            encoding="utf-8"
        )
        assert r"\begin{abstract}" in abstract
        assert r"\begin{abstract*}" in abstract
        assert r"\begin{notation}" in notation
        assert r"\begin{acknowledgments}" in acknowledgments
        assert r"\begin{achievements}" in achievements

- [ ] **Step 2: Run RED**

Run:

    $PY -m pytest tests/test_ustc_thesis_structure.py -q

Expected: failures identify ctexbook, missing official matter, eight chapters, old bibliography style, and absent source files.

- [ ] **Step 3: Extend metadata rendering for LaTeX build profiles**

Add argparse to the module imports and add these complete rendering helpers and CLI:

    def _string_list(value: object, field: str) -> list[str]:
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"{field} must be a list of strings")
        return value


    def _tex_escape(value: str) -> str:
        replacements = {
            "\\": r"\textbackslash{}",
            "{": r"\{",
            "}": r"\}",
            "$": r"\$",
            "&": r"\&",
            "#": r"\#",
            "_": r"\_",
            "%": r"\%",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        return "".join(replacements.get(character, character) for character in value)


    def _macro(name: str, value: str) -> str:
        return rf"\providecommand{{\{name}}}{{{_tex_escape(value)}}}"


    def _atomic_write(path: Path, text: str) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(text, encoding="utf-8", newline="\n")
        temporary.replace(path)


    def _journal_author_commands(data: PublicationMetadata) -> str:
        authors = _string_list(data.journal["authors"], "journal.authors")
        affiliation_ids = _string_list(
            data.journal["author_affiliation_ids"],
            "journal.author_affiliation_ids",
        )
        emails = _string_list(data.journal["author_emails"], "journal.author_emails")
        corresponding = data.journal["corresponding_author"]
        if not isinstance(corresponding, str):
            raise ValueError("journal.corresponding_author must be a string")
        lines: list[str] = []
        for author, ids, email in zip(authors, affiliation_ids, emails, strict=True):
            normalized_ids = ",".join(item.strip() for item in ids.split(",") if item.strip())
            if not normalized_ids:
                raise ValueError("every journal author needs at least one affiliation id")
            marker = r"\cormark[1]" if author == corresponding else ""
            lines.append(
                rf"\author[{_tex_escape(normalized_ids)}]"
                rf"{{{_tex_escape(author)}{marker}}}"
            )
            lines.append(rf"\ead{{{_tex_escape(email)}}}")
        affiliations = _string_list(
            data.journal["affiliations"], "journal.affiliations"
        )
        for index, affiliation in enumerate(affiliations, start=1):
            lines.append(rf"\address[aff{index}]{{{_tex_escape(affiliation)}}}")
        lines.append(r"\cortext[1]{Corresponding author}")
        return "\n".join(lines)


    def write_rendered_metadata(
        data: PublicationMetadata,
        gates: Mapping[str, str],
        output_dir: Path,
        *,
        final: bool,
        private_root: Path | None = None,
        target: Literal[
            "thesis_build",
            "school_submission",
            "journal_manuscript",
            "journal_package",
            "journal_submission",
        ] = "thesis_build",
    ) -> tuple[Path, Path, Path]:
        if final:
            validate_final_metadata(data, gates, target=target)
        output_dir.mkdir(parents=True, exist_ok=True)
        private_dir = private_root or Path("publication/private")
        originality = private_dir / "ustc-originality-authorization.pdf"
        ai_usage = private_dir / "ustc-ai-usage.pdf"
        if final and target == "school_submission":
            missing_private = [path for path in (originality, ai_usage) if not path.is_file()]
            if missing_private:
                raise ValueError(f"signed USTC declarations missing: {missing_private}")

        supervisors = _string_list(data.ustc["supervisors"], "ustc.supervisors")
        achievements = _string_list(data.ustc["achievements"], "ustc.achievements")
        zh_ai_uses = localized_ai_uses(data, "zh")
        en_ai_uses = localized_ai_uses(data, "en")
        ai_tools = _string_list(data.disclosure["ai_tools"], "disclosure.ai_tools")
        practice_status = str(data.ustc["practice_supervisor_status"])
        thesis_lines = [
            _macro("PublicationTitleZh", str(data.ustc["title_zh"])),
            _macro("PublicationTitleEn", str(data.ustc["title_en"])),
            _macro("PublicationStudentName", str(data.ustc["student_name"])),
            _macro("PublicationStudentId", str(data.ustc["student_id"])),
            _macro("PublicationSupervisors", ", ".join(supervisors)),
            _macro("PublicationDepartment", str(data.ustc["department"])),
            _macro("PublicationDiscipline", str(data.ustc["discipline"])),
            _macro("PublicationResearchDirection", str(data.ustc["research_direction"])),
            _macro("PublicationSubmissionDate", str(data.ustc["submission_date"])),
            _macro("PublicationCLC", str(data.ustc["clc"])),
            _macro("PublicationUDC", str(data.ustc["udc"])),
            _macro("PublicationAchievements", "; ".join(achievements)),
            _macro("PublicationAcknowledgments", str(data.ustc["acknowledgments"])),
        ]
        practice_setup: list[str] = []
        if practice_status != "none":
            thesis_lines.append(
                _macro(
                    "PublicationPracticeSupervisor",
                    str(data.ustc["practice_supervisor"]),
                )
            )
            practice_setup.append(
                "  practice-supervisor = {\\PublicationPracticeSupervisor},"
            )
        thesis_lines.extend(
            [
                r"\ustcsetup{",
                r"  title = {\PublicationTitleZh},",
                r"  title* = {\PublicationTitleEn},",
                r"  author = {\PublicationStudentName},",
                r"  student-id = {\PublicationStudentId},",
                r"  department = {\PublicationDepartment},",
                r"  discipline = {\PublicationDiscipline},",
                r"  supervisor = {\PublicationSupervisors},",
                *practice_setup,
                r"  research-direction = {\PublicationResearchDirection},",
                r"  date = {\PublicationSubmissionDate},",
                r"  clc = {\PublicationCLC},",
                r"  udc = {\PublicationUDC},",
                r"}",
            ]
        )
        purpose = "、".join(zh_ai_uses)
        if final and target == "school_submission":
            thesis_lines.extend(
                [
                    rf"\providecommand{{\PublicationOriginalityDeclaration}}"
                    rf"{{\declarationoforiginality[file={{{originality.as_posix()}}}]}}",
                    rf"\providecommand{{\PublicationAIUsageDeclaration}}"
                    rf"{{\declarationofaiusage[purpose={{{_tex_escape(purpose)}}},"
                    rf"file={{{ai_usage.as_posix()}}}]}}",
                ]
            )
        else:
            thesis_lines.extend(
                [
                    r"\providecommand{\PublicationOriginalityDeclaration}{\declarationoforiginality}",
                    rf"\providecommand{{\PublicationAIUsageDeclaration}}"
                    rf"{{\declarationofaiusage[purpose={{{_tex_escape(purpose)}}}]}}",
                ]
            )

        credit = _string_list(data.journal["credit"], "journal.credit")
        journal_lines = [
            _macro("PublicationJournalTitle", str(data.journal["title"])),
            _macro("PublicationJournalCRediT", "; ".join(credit)),
            _macro("PublicationJournalFunding", str(data.journal["funding"])),
            _macro(
                "PublicationJournalCompetingInterests",
                str(data.journal["competing_interests"]),
            ),
            _macro(
                "PublicationJournalDataAvailability",
                str(data.journal["data_availability"]),
            ),
            _macro(
                "PublicationJournalAcknowledgments",
                str(data.journal["acknowledgments"]),
            ),
            _macro(
                "PublicationAIStatement",
                f"{', '.join(ai_tools)} was used for "
                f"{', '.join(en_ai_uses)}; the authors reviewed the resulting material.",
            ),
            "\\providecommand{\\PublicationJournalAuthors}{%",
            _journal_author_commands(data),
            "}",
        ]
        mode_lines = [
            r"\newif\ifpublicationfinal",
            r"\publicationfinaltrue" if final else r"\publicationfinalfalse",
        ]
        thesis_path = output_dir / "thesis-metadata.tex"
        journal_path = output_dir / "journal-metadata.tex"
        mode_path = output_dir / "build-mode.tex"
        _atomic_write(thesis_path, "\n".join(thesis_lines) + "\n")
        _atomic_write(journal_path, "\n".join(journal_lines) + "\n")
        _atomic_write(mode_path, "\n".join(mode_lines) + "\n")
        return thesis_path, journal_path, mode_path


    def main(argv: list[str] | None = None) -> int:
        parser = argparse.ArgumentParser()
        parser.add_argument("--metadata", type=Path, required=True)
        parser.add_argument("--gates", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--mode", choices=("draft", "final"), required=True)
        parser.add_argument("--private-root", type=Path)
        parser.add_argument(
            "--target",
            choices=(
                "thesis_build",
                "school_submission",
                "journal_manuscript",
                "journal_package",
                "journal_submission",
            ),
            default="thesis_build",
        )
        args = parser.parse_args(argv)
        write_rendered_metadata(
            load_publication_metadata(args.metadata),
            load_human_gates(args.gates),
            args.output,
            final=args.mode == "final",
            private_root=args.private_root,
            target=args.target,
        )
        return 0


    if __name__ == "__main__":
        raise SystemExit(main())

The module import list must now include argparse. Draft build-mode.tex defines publicationfinalfalse. Final defines publicationfinaltrue and validates only its named target. The two signed declaration macros point only to the two private PDF paths encoded above, and school_submission verifies both files before writing output. thesis_build uses the official unsigned declaration pages, so an automated research-delivery PDF is not blocked by signatures or upload authorization. journal_manuscript excludes biography/photo fields; journal_package requires biographies, photos, COI form, cover-letter, exclusivity, and supplement gates but not upload authorization; journal_submission adds only journal_submission_authorized. The graphical-abstract branch is fixed false/not_applicable. AI uses are stable semantic IDs translated through AI_USE_TEXT, so Chinese and English declarations have identical use-ID sets and the English output contains no Han characters or false non-use statement. The module CLI is:

    $PY -m uav_mec.publication_metadata \
      --metadata publication/metadata.toml \
      --gates publication/human-gates.toml \
      --output build/publication/metadata \
      --mode draft \
      --target thesis_build

Add a focused test to tests/test_publication_metadata.py that calls write_rendered_metadata in draft mode, reads all three returned files, and asserts that each path is under tmp_path and contains no personal value absent from metadata.

- [ ] **Step 4: Write the official setup and driver**

thesis/ustcsetup.tex first requires the temporary driver to define `\PublicationMetadataRoot` and `\PublicationCanonicalLightRoot`; a missing definition raises PackageError. It then loads:

    \input{\PublicationMetadataRoot/build-mode.tex}
    \input{\PublicationMetadataRoot/thesis-metadata.tex}

It loads amsthm, graphicx, bicaption, booktabs, threeparttable, longtable, algorithm, algorithmic, siunitx, natbib with sort, and hyperref last. It declares proposition, theorem, lemma, and corollary environments by chapter, configures the official numeric bibliography, and defines:

    \newcommand{\CanonicalLightRoot}{\PublicationCanonicalLightRoot}
    \newcommand{\PublicationCanonicalInput}[1]{%
      \IfFileExists{\CanonicalLightRoot/#1}{%
        \input{\CanonicalLightRoot/#1}%
      }{%
        \ifpublicationfinal
          \PackageError{publication}{Missing canonical-v2 light asset: #1}%
            {Run canonical verification before the final build}%
        \else
          \fbox{\texttt{\_\_CANONICAL\_V2\_REQUIRED\_\_}}%
        \fi
      }%
    }

PublicationCanonicalInput inputs CanonicalLightRoot/#1 when the file exists. When absent in draft mode it renders the literal __CANONICAL_V2_REQUIRED__ in a framed box. When absent in final mode it raises PackageError and stops compilation.

thesis/main.tex follows the exact order asserted by the test. Put all USTC metadata in the generated thesis-metadata.tex; main.tex contains no author, student ID, advisor, college, discipline, funding, or achievement value.

The mapping is explicit: the driver class options freeze `degree=master` and `degree-type=professional`; generated thesis-metadata.tex maps title_zh to `title`, title_en to `title*`, student_name to `author`, student_id to `student-id`, department to `department`, discipline/legacy-major meaning to the current class key `discipline`, supervisors to `supervisor`, research_direction to `research-direction`, submission_date to `date`, and classification fields to `clc`/`udc`. A present practice supervisor maps to `practice-supervisor`. Status `none` omits that key and its macro completely, so `not_applicable` never reaches TeX. Tests inspect every mapping and both practice-supervisor branches.

Use the official archive latexmkrc as the base for thesis/latexmkrc. Retain XeLaTeX, xdvipdfmx, BibTeX, and clean-extension settings; do not add shell escape.

- [ ] **Step 5: Create the official matter without inventing human content**

frontmatter/abstract.tex defines both abstract environments and keyword settings. Before canonical-v2, its result clauses are injected with:

    \PublicationCanonicalInput{latex/thesis/abstract-results.tex}

The result-independent sentences state the problem, model boundary, method, and evidence protocol without a performance conclusion.

frontmatter/notation.tex defines symbols and abbreviations actually used by Chapters 3 through 7. Include K, N, S, h, estimated h, alpha, uplink rate, downlink rate, minimum CPU, Phi, cross-receiver exposure, IEA-CAP, DD-MCAA, and EAD-JSRA. Do not include a symbol that has no manuscript use.

backmatter/acknowledgments.tex contains only the environment and PublicationAcknowledgments macro from metadata. backmatter/achievements.tex renders only metadata-confirmed entries. In draft mode the sentinel remains visible; final mode is rejected earlier.

- [ ] **Step 6: Establish the nine chapter paths**

Use git mv for the two existing chapters. Rename the current Chapter 6 file:

    git mv thesis/chapters/06-implementation-experiments.tex \
      thesis/chapters/06-implementation-methodology.tex

Create 07-results-mechanism.tex with these exact section headings:

    \chapter{实验结果与机制分析}
    \section{主假设与全栈效应}
    \section{外部性机制证据}
    \section{资源准入与精确参照}
    \section{扩展性、鲁棒性与轨迹驱动结果}
    \section{运行时间与工程量级}
    \section{负面结果与主张门禁}
    \PublicationCanonicalInput{latex/thesis/results-body.tex}

The file contains no formal numeric literal and no fallback value.

- [ ] **Step 7: Run GREEN and metadata regression**

Run:

    $PY -m pytest tests/test_publication_metadata.py \
      tests/test_ustc_thesis_structure.py -q
    $RUFF check uav_mec/publication_metadata.py \
      tests/test_publication_metadata.py tests/test_ustc_thesis_structure.py

Expected: all structure and metadata tests pass. A final-mode build is still intentionally blocked by metadata and canonical sentinels.

- [ ] **Step 8: Commit the USTC structure**

Run:

    git add -- thesis/latexmkrc thesis/ustcsetup.tex thesis/main.tex \
      thesis/frontmatter/abstract.tex thesis/frontmatter/notation.tex \
      thesis/backmatter/acknowledgments.tex \
      thesis/backmatter/achievements.tex \
      thesis/chapters/06-implementation-methodology.tex \
      thesis/chapters/07-results-mechanism.tex \
      thesis/chapters/08-deployment-limitations.tex \
      thesis/chapters/09-conclusion.tex \
      uav_mec/publication_metadata.py \
      tests/test_publication_metadata.py \
      tests/test_ustc_thesis_structure.py
    git diff --cached --name-status
    git commit -m "feat: migrate thesis to official USTC structure"

Expected: the three chapter renames are shown correctly, the old Chapter 6 through 8 paths are absent, and no generated PDF or private metadata is staged.

### Task 5: Rewrite the result-independent USTC chapters and proof appendices

**Files:**
- Modify: thesis/chapters/01-introduction.tex
- Modify: thesis/chapters/02-related-work.tex
- Modify: thesis/chapters/03-system-model.tex
- Modify: thesis/chapters/04-interference-aware-pairing.tex
- Modify: thesis/chapters/05-deadline-allocation.tex
- Modify: thesis/chapters/06-implementation-methodology.tex
- Modify: thesis/chapters/08-deployment-limitations.tex
- Modify: thesis/chapters/09-conclusion.tex
- Create: thesis/appendices/01-proofs.tex
- Create: thesis/appendices/02-exact-oracles.tex
- Create: thesis/appendices/03-parameters-configurations.tex
- Create: thesis/appendices/04-statistical-protocol.tex
- Create: thesis/appendices/05-reproduction.tex
- Create: thesis/appendices/06-externality-certificate.tex
- Modify: thesis/main.tex
- Modify: tests/test_active_thesis_claims.py
- Create: tests/test_thesis_semantic_alignment.py

- [ ] **Step 1: Write the failing semantic-alignment tests**

Create tests/test_thesis_semantic_alignment.py:

    import re
    from pathlib import Path


    ROOT = Path(__file__).parents[1]
    CHAPTERS = ROOT / "thesis/chapters"


    def read(name: str) -> str:
        return (CHAPTERS / name).read_text(encoding="utf-8")


    def test_system_model_names_normative_raw_closed_and_channel_objects():
        text = read("03-system-model.tex")
        required = (
            "raw_radio_assignment",
            "rate_snapshot",
            "closed_executable_schedule",
            "h_{i,u}",
            r"\widehat{h}_{i,u}",
            "(user_id, job_index)",
            "静态并行虚拟 CPU 份额",
            r"R_i^{\mathrm{ul}}",
            r"R_i^{\mathrm{dl}}",
        )
        for token in required:
            assert token in text
        assert r"(1+\rho)" not in text


    def test_system_model_uses_the_normative_lexicographic_objective():
        text = read("03-system-model.tex")
        assert r"\Phi" in text
        assert r"-F_{\min}" in text
        assert r"-E_{\mathrm{cross}}" in text
        assert "总裕量" not in text


    def test_iea_chapter_covers_complete_neighborhood_and_certificates():
        text = read("04-interference-aware-pairing.tex")
        for token in (
            "Replace",
            "Insert",
            "Drop",
            "Position",
            "IntraRelocate",
            "CrossSwap",
            "CrossRelocate",
            "多起点",
            "最佳改进",
            "外部性证书",
            "依赖闭包",
            "no_improving_move",
            "iteration_cap",
        ):
            assert token in text
        assert "首次改进" not in text


    def test_dd_chapter_states_the_static_share_theorem_scope():
        text = read("05-deadline-allocation.tex")
        for token in (
            r"\alpha_i^{\min}",
            r"f_i^{\min}",
            "单位任务价值",
            "同基数最低 CPU",
            "Exact Admission Oracle",
            "静态并行虚拟 CPU 份额",
            "不适用于可抢占 EDF",
        ):
            assert token in text


    def test_methodology_uses_the_registered_v2_protocol():
        text = read("06-implementation-methodology.tex")
        for token in (
            "H1",
            "H2",
            "H3",
            "H4",
            "paired t-test",
            "Holm",
            "paired bootstrap",
            "requires_offload",
            "confirmatory_lock.json",
            "canonical-v2",
            "GeoLife",
        ):
            assert token in text
        forbidden = ("canonical v1", "50 种子", "所有基线一起 Holm")
        for token in forbidden:
            assert token not in text


    def test_related_work_cites_frozen_closest_work_and_adaptation():
        text = read("02-related-work.tex")
        for key in (
            "guo2019externality",
            "xu2020multiapnoma",
            "katwe2022multiuavnoma",
            "uwizeyimana2024association",
            "chen2026qos",
            "lyu2018deadline",
        ):
            assert key in text
        assert "guo_jagp_adapted" in text
        assert "原算法忠实复现" not in text


    def test_no_result_independent_chapter_reads_legacy_numbers():
        paths = tuple(
            path
            for path in sorted(CHAPTERS.glob("*.tex"))
            if path.name != "07-results-mechanism.tex"
        )
        text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        forbidden = (
            "canonical/v1",
            "../paper/generated_",
            "NP-hardness",
            "局部最优等于全局最优",
        )
        for token in forbidden:
            assert token not in text
        assert not re.search(r"\b(?:Full|Ours)\b", text)

- [ ] **Step 2: Update the active-claim test for the approved thesis**

tests/test_active_thesis_claims.py scans README.md, thesis, paper, publication, and references only. It does not scan archive, artifacts, build, third_party, results, defense, or deliverables. Replace the old title and eight-chapter assertion with:

    APPROVED_THESIS_TITLE = "多无人机边缘计算的干扰外部性感知调度与时限卸载"

Assert nine chapters, the three algorithm names, no scene-first novelty, no unsupported complexity claim, no canonical-v1 path, no paper/generated_ input, no ctexbook, and no IEEEtran. Do not require a positive result or a particular gate outcome.

- [ ] **Step 3: Run RED**

Run:

    $PY -m pytest tests/test_active_thesis_claims.py \
      tests/test_thesis_semantic_alignment.py -q

Expected: failures identify the old task identity, strong-only SIC formula, shared UL/DL expression, slack objective, first-improvement wording, v1 statistics, and incomplete closest-work positioning.

- [ ] **Step 4: Write Chapters 1 through 3 against frozen sources**

Chapter 1 has sections for background, engineering problem, research boundary, contribution boundary, evidence architecture, professional-master value, and chapter organization. It names the approved Chinese title and exactly two method contributions. EAD-JSRA is the integration, not a third independent innovation.

Chapter 2 has sections for UAV-MEC, full-reuse uplink NOMA, externality-aware association/grouping, deadline admission, six-paper closest-work matrix, Guo-JAGP adaptation contract, and the remaining gap. Every comparison row maps decision variables, objective, channel, power, mobility, compute model, retained behavior, adaptation, and omitted behavior. Use docs/audits/closest-work-model-matrix.csv from Plan 2 as the source.

Chapter 3 defines task identity, raw assignment, rate snapshot, closed executable schedule, true and estimated large-scale gain, receive-power role normalization, general ordered-layer SINR, every-layer SIC gate, separate uplink/downlink delay, static virtual CPU shares, delete-only closure, all-arrival and requires-offload denominators, cross-receiver victim-link exposure, and the normative lexicographic objective. Match the public functions and types delivered by Plans 1 and 2; do not preserve an old symbol merely for prose continuity.

- [ ] **Step 5: Write Chapters 4 through 6 against tested APIs**

Chapter 4 maps every neighborhood to the Plan 2 move generator, gives legal-state preservation, strict-improvement termination, the condition for a local-optimality statement, externality certificate conservation, full dependency closure, multi-start best improvement, iteration-cap limitation, and the implementation-specific complexity expression. Keep the full dependency-recalculation cost distinct from DD sorting cost.

Chapter 5 derives minimum offload ratio and minimum CPU with separate uplink/downlink transfer terms, proves monotonicity on the feasible interval, proves maximum cardinality and same-cardinality minimum CPU, states Exact Admission scope, explains policy-conditioned closure, and defines the EAD-JSRA feedback loop. It explicitly excludes preemptive EDF semantics.

Chapter 6 records the registry, disjoint seed pools, pilot versus formal distinction, power-lock invalidation, family order, common random numbers, H1 through H4 estimands, 2-by-2 joint decomposition, paired bootstrap interval, two-sided paired t-test, one four-test Holm family, rank-biserial sensitivity effect, fixed-radio H3, Exact nontriviality, trace lock, runtime/capacity budget, canonical-v2 full/light flow, and negative-result publication rule. It contains protocol values from locked configuration or generated macros, never copied result values.

- [ ] **Step 6: Write result-independent deployment and conclusion boundaries**

Chapter 8 covers centralized control flow, control-link assumptions, estimated-versus-realized execution, runtime counters, deployment limits, GeoLife interpretation, closest-work adaptation limitations, thesis-journal overlap, AI responsibility, and unoptimized variables. It does not describe an observed advantage.

Chapter 9 contains a result-independent recap of the problem and method boundaries plus future work. Its final evidence-dependent paragraphs are supplied only by:

    \PublicationCanonicalInput{latex/thesis/conclusion-results.tex}

- [ ] **Step 7: Write complete appendices**

The six appendix files contain, respectively:

1. Full legal-neighborhood, strict-improvement, finite-termination, dependency-closure, minimum-offload, minimum-CPU, and prefix-optimality proofs.
2. Separate Exact Admission and Exact Schedule definitions, certificates, timeout semantics, and three-layer gap ordering.
3. Parameter/source tables and exact registry/config identifiers, without result values.
4. H1 through H4 estimands, bootstrap, paired t-test, Holm, interaction, MDE-to-task conversion, and failure wording.
5. Environment, lock, shard, canonical verification, asset generation, and build commands.
6. A schema-level externality certificate case whose numerical row is inserted from canonical-v2 light.

Add all six includes after \appendix and before \backmatter in thesis/main.tex.

- [ ] **Step 8: Run GREEN**

Run:

    $PY -m pytest tests/test_active_thesis_claims.py \
      tests/test_thesis_semantic_alignment.py \
      tests/test_latex_citations.py \
      tests/test_publication_bibliography.py -q
    git diff --check

Expected: all result-independent semantic, claim-boundary, and citation checks pass. The 30,000-character final gate is intentionally deferred until the canonical result chapter and final abstract are written.

- [ ] **Step 9: Commit the result-independent thesis**

Run:

    git add -- thesis/main.tex \
      thesis/chapters/01-introduction.tex \
      thesis/chapters/02-related-work.tex \
      thesis/chapters/03-system-model.tex \
      thesis/chapters/04-interference-aware-pairing.tex \
      thesis/chapters/05-deadline-allocation.tex \
      thesis/chapters/06-implementation-methodology.tex \
      thesis/chapters/08-deployment-limitations.tex \
      thesis/chapters/09-conclusion.tex \
      thesis/appendices/01-proofs.tex \
      thesis/appendices/02-exact-oracles.tex \
      thesis/appendices/03-parameters-configurations.tex \
      thesis/appendices/04-statistical-protocol.tex \
      thesis/appendices/05-reproduction.tex \
      thesis/appendices/06-externality-certificate.tex \
      tests/test_active_thesis_claims.py \
      tests/test_thesis_semantic_alignment.py
    git diff --cached --name-status
    git commit -m "docs: align USTC thesis with normative v2 semantics"

Expected: only the named thesis sources and tests are committed; no result macro, figure, or generated PDF is staged.

### Task 6: Migrate the English manuscript to the Elsevier Computer Networks structure

**Files:**
- Modify: paper/main_v2.tex
- Create: paper/frontmatter/abstract.tex
- Create: paper/frontmatter/keywords.tex
- Create: paper/sections/01-introduction.tex
- Create: paper/sections/02-related-work.tex
- Create: paper/sections/03-system-model.tex
- Create: paper/sections/04-iea-cap.tex
- Create: paper/sections/05-dd-mcaa-ead-jsra.tex
- Create: paper/sections/06-experimental-methodology.tex
- Create: paper/sections/07-results-mechanism.tex
- Create: paper/sections/08-discussion-limitations.tex
- Create: paper/sections/09-conclusion.tex
- Create: paper/sections/appendices.tex
- Create: paper/declarations/acknowledgments.tex
- Create: paper/declarations/credit.tex
- Create: paper/declarations/funding.tex
- Create: paper/declarations/competing-interests.tex
- Create: paper/declarations/data-availability.tex
- Create: paper/declarations/ai-use.tex
- Create: paper/glossary.tex
- Create: publication/scope-review.md
- Create: tests/test_computer_networks_structure.py

- [ ] **Step 1: Write the failing journal-structure tests**

Create tests/test_computer_networks_structure.py:

    import re
    from pathlib import Path


    ROOT = Path(__file__).parents[1]
    MAIN = ROOT / "paper/main_v2.tex"

    SECTIONS = (
        "01-introduction",
        "02-related-work",
        "03-system-model",
        "04-iea-cap",
        "05-dd-mcaa-ead-jsra",
        "06-experimental-methodology",
        "07-results-mechanism",
        "08-discussion-limitations",
        "09-conclusion",
    )


    def test_active_paper_uses_elsarticle_and_not_ieee():
        text = MAIN.read_text(encoding="utf-8")
        assert r"\documentclass[preprint,12pt]{elsarticle}" in text
        assert "IEEEtran" not in text
        assert "IEEEkeywords" not in text
        assert r"\begin{frontmatter}" in text
        assert r"\end{frontmatter}" in text


    def test_paper_title_and_numbered_sections_match_the_approved_design():
        metadata = (ROOT / "publication/metadata.toml").read_text(
            encoding="utf-8"
        )
        assert (
            "Externality-Aware Scheduling and Deadline-Constrained Offloading "
            "for Multi-UAV Edge Computing"
        ) in metadata
        text = MAIN.read_text(encoding="utf-8")
        inputs = re.findall(r"\\input\{sections/([^}]+)\}", text)
        assert tuple(inputs[:9]) == SECTIONS
        for name in SECTIONS:
            assert (ROOT / f"paper/sections/{name}.tex").is_file()


    def test_journal_frontmatter_reads_generated_metadata_and_canonical_results():
        text = MAIN.read_text(encoding="utf-8")
        abstract = (ROOT / "paper/frontmatter/abstract.tex").read_text(
            encoding="utf-8"
        )
        assert r"\PublicationMetadataRoot/journal-metadata.tex" in text
        assert r"\PublicationCanonicalLightRoot" in text
        assert "build/publication/metadata" not in text
        assert r"\input{frontmatter/abstract}" in text
        assert r"\input{frontmatter/keywords}" in text
        assert (
            r"\PublicationCanonicalInput{latex/journal/abstract-results.tex}"
            in abstract
        )


    def test_journal_declarations_and_glossary_are_complete_and_ordered():
        text = MAIN.read_text(encoding="utf-8")
        ordered = (
            r"\input{glossary}",
            r"\input{declarations/acknowledgments}",
            r"\input{declarations/credit}",
            r"\input{declarations/funding}",
            r"\input{declarations/competing-interests}",
            r"\input{declarations/data-availability}",
            r"\input{declarations/ai-use}",
            r"\bibliographystyle{elsarticle-num}",
            r"\bibliography{../references/references}",
        )
        positions = [text.index(token) for token in ordered]
        assert positions == sorted(positions)


    def test_paper_results_read_only_canonical_v2_light():
        text = (
            ROOT / "paper/sections/07-results-mechanism.tex"
        ).read_text(encoding="utf-8")
        assert r"\PublicationCanonicalInput{latex/journal/results-body.tex}" in text
        assert "../figures/thesis" not in text
        assert "generated_results.tex" not in text
        assert "canonical/v1" not in text


    def test_network_scope_review_is_human_gated_and_forbids_acceptance_promises():
        review = (ROOT / "publication/scope-review.md").read_text(encoding="utf-8")
        for token in (
            "cross-layer network scheduling",
            "not a PHY-only paper",
            "guo_jagp_adapted",
            "mapping",
            "comparability",
            "scope_review_confirmed",
            "does not promise CCF-B acceptance",
        ):
            assert token in review

- [ ] **Step 2: Run RED**

Run:

    $PY -m pytest tests/test_computer_networks_structure.py -q

Expected: failures identify IEEEtran, the monolithic source, missing declarations, old bibliography style, and shared Chinese figure paths.

- [ ] **Step 3: Write the thin elsarticle driver**

paper/main_v2.tex uses the local third_party class through the build environment, not an unpinned TeX installation. Use preprint,12pt for the review manuscript. The temporary runner driver defines `\PublicationMetadataRoot` and `\PublicationCanonicalLightRoot`; paper/main_v2.tex raises PackageError if either is absent and never hardcodes build/publication/metadata. The driver:

1. loads the generated journal metadata and build mode;
2. defines CanonicalLightRoot and PublicationCanonicalInput with the same draft/final behavior as the thesis;
3. opens frontmatter;
4. renders title, authors, affiliations, corresponding author, abstract, and keywords;
5. inputs the nine numbered sections and appendices;
6. renders glossary and declarations in the tested order;
7. places the truthful AI declaration immediately before the references;
8. uses elsarticle-num and the shared bibliography.

Do not embed a biography, author photo, cover letter, or competing-interest Word form in the manuscript.

- [ ] **Step 4: Write the result-independent English sections**

Sections 1 through 6 mirror the same scientific facts as Chinese Chapters 1 through 6 without line-by-line translation. The Introduction centers cross-layer network scheduling and discrete network algorithms, not only NOMA decoding. Related Work uses the audited six-paper matrix and names guo_jagp_adapted accurately. System Model uses the same symbols and raw/closed semantics. Sections 4 and 5 state the same theorem scopes and proof labels. Section 6 states the pre-registered experiment and inference protocol without observed outcomes.

Section 7 contains the exact section headings Main Hypotheses and Full-Stack Effect, Externality Mechanism, Admission and Exact References, Scaling Robustness and Traces, Runtime and Engineering Magnitude, and Negative or Inconclusive Evidence. Its body is:

    \PublicationCanonicalInput{latex/journal/results-body.tex}

Section 8 provides scope, deployment assumptions, external validity, adapted-baseline limitations, overlap/copyright, and AI responsibility without asserting an outcome. Section 9 provides a result-independent method recap and:

    \PublicationCanonicalInput{latex/journal/conclusion-results.tex}

Appendices contain concise proof and protocol material selected from the thesis appendices; extensive tables move to supplementary material in Task 12.

- [ ] **Step 5: Write declarations that resolve only from controlled metadata**

Each declaration source calls a macro rendered from publication metadata. Funding and competing interests never default to none. Data Availability distinguishes a local evidence package from an externally archived public record and never claims a repository exists without external_archive_authorized = approved. The AI declaration names the actual tool and uses from metadata. Acknowledgments do not contain author or advisor names until supplied.

Create publication/scope-review.md as a separate human review record. It checks that the Computer Networks manuscript leads with cross-layer network scheduling and discrete network algorithms, is not a PHY-only/NOMA-only paper, maps every `guo_jagp_adapted` component to the cited Guo method, records adaptation/comparability limits, and does not promise CCF-B acceptance. The review remains unresolved until scope_review_confirmed is approved; research delivery and every journal target require that gate.

- [ ] **Step 6: Run GREEN and cross-manuscript citation checks**

Run:

    $PY -m pytest tests/test_computer_networks_structure.py \
      tests/test_latex_citations.py \
      tests/test_publication_bibliography.py \
      tests/test_active_thesis_claims.py -q
    git diff --check

Expected: the active English source uses only elsarticle, all nine sections exist, all declarations are ordered, and citations use the shared audited bibliography.

- [ ] **Step 7: Commit the Elsevier manuscript structure**

Run:

    git add -- paper/main_v2.tex \
      paper/frontmatter/abstract.tex paper/frontmatter/keywords.tex \
      paper/sections/01-introduction.tex \
      paper/sections/02-related-work.tex \
      paper/sections/03-system-model.tex \
      paper/sections/04-iea-cap.tex \
      paper/sections/05-dd-mcaa-ead-jsra.tex \
      paper/sections/06-experimental-methodology.tex \
      paper/sections/07-results-mechanism.tex \
      paper/sections/08-discussion-limitations.tex \
      paper/sections/09-conclusion.tex \
      paper/sections/appendices.tex \
      paper/declarations/acknowledgments.tex \
      paper/declarations/credit.tex \
      paper/declarations/funding.tex \
      paper/declarations/competing-interests.tex \
      paper/declarations/data-availability.tex \
      paper/declarations/ai-use.tex \
      paper/glossary.tex \
      publication/scope-review.md \
      tests/test_computer_networks_structure.py
    git diff --cached --name-status
    git commit -m "feat: migrate paper to Computer Networks structure"

Expected: exactly the active paper sources and structure test are committed; the historical paper remains untouched until Task 10.

### Task 7: Land the deterministic bilingual publication-asset generator before Plan 4

**Depends on:** Plans 1 and 2. This task is a hard prerequisite of Plan 4 and must be merged before any canonical-v2 publication command runs. It creates renderers and tests only; it does not open a formal seed or write a result into either manuscript.

**Files:**
- Create: publication/figure_manifest.toml
- Create: uav_mec/publication_assets.py
- Create: tests/test_publication_assets.py

- [ ] **Step 1: Write self-contained RED tests for the frozen generator interface**

Create tests/test_publication_assets.py. It imports exactly:

    from uav_mec.publication_assets import generate_bilingual_publication_assets

Define ordinary helpers in that file, not pytest fixtures promised elsewhere:

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()


    def write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )


    def write_synthetic_analysis(root: Path) -> Path:
        root.mkdir(parents=True)
        write_json(
            root / "acceptance.json",
            {
                "schema_version": 1,
                "all_integrity_gates_pass": True,
                "claim_gates": {
                    "H1": {"status": "meaningful_supported"},
                    "H2": {"status": "insufficient"},
                    "H3": {"status": "positive_but_small"},
                    "H4": {"status": "negative"},
                },
            },
        )
        write_json(
            root / "statistics.json",
            {
                "schema_version": 1,
                "joint": {
                    "labels": ["synthetic-a", "synthetic-b"],
                    "series": {"synthetic-method": [0.25, 0.50]},
                },
            },
        )
        write_json(
            root / "metrics.json",
            {
                "schema_version": 1,
                "publication": {
                    "externality": {"x": ["a", "b"], "y": [0.2, 0.4]},
                    "resource": {"x": ["a", "b"], "y": [0.3, 0.5]},
                    "exact": {"x": ["a", "b"], "y": [0.1, 0.2]},
                    "scaling": {"x": [1, 2], "y": [0.2, 0.3]},
                    "robustness": {"x": ["a", "b"], "y": [0.4, 0.3]},
                    "trace_driven": {"x": ["a", "b"], "y": [0.3, 0.4]},
                    "runtime": {"x": ["a", "b"], "y": [0.1, 0.2]},
                },
            },
        )
        rows = [
            {
                "claim_id": hypothesis,
                "category": "primary",
                "family": "synthetic",
                "case_id": "synthetic-case",
                "treatment": "synthetic-treatment",
                "control": "synthetic-control",
                "estimand": "synthetic-estimand",
                "mde_pp": "1.0",
                "estimate_pp": str(index),
                "ci_low_pp": str(index - 1),
                "ci_high_pp": str(index + 1),
                "p_t_raw": "0.5",
                "p_holm": "0.5",
                "p_wilcoxon_raw": "0.5",
                "rank_biserial": "0.0",
                "mean_added_tasks_per_episode": "0.1",
                "status": status,
                "wording_level": "synthetic-only",
                "evidence_paths": "[]",
            }
            for index, (hypothesis, status) in enumerate(
                (
                    ("H1", "meaningful_supported"),
                    ("H2", "insufficient"),
                    ("H3", "positive_but_small"),
                    ("H4", "negative"),
                ),
                start=1,
            )
        ]
        with (root / "statistical_claims.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return root

The numeric values above are explicitly synthetic test labels and are not manuscript results. Use those helpers in these exact tests:

    test_generator_has_the_frozen_two_path_mapping_signature
    test_generator_requires_an_initially_absent_staging_root
    test_generator_returns_every_regular_file_as_posix_path_and_sha256
    test_identical_analysis_bytes_produce_byte_identical_asset_trees
    test_each_bilingual_figure_pair_records_one_shared_data_sha256
    test_generated_latex_uses_gate_appropriate_wording_for_all_claim_statuses
    test_generator_rejects_failed_integrity_missing_input_nan_and_infinite_value
    test_generator_never_reads_canonical_v1_pilot_smoke_runs_or_formal_roots
    test_generator_writes_no_timestamp_hostname_absolute_path_or_random_identifier

For the source-boundary test, monkeypatch Path.open and Path.read_bytes with wrappers that record resolved read paths while delegating to the original methods. Assert every read is either beneath the supplied analysis_root or is publication/figure_manifest.toml. Do not monkeypatch an undefined fixture.

- [ ] **Step 2: Test the handoff contract without taking ownership of the publisher**

In tests/test_publication_assets.py, use `inspect.signature` to assert the public parameter names are exactly analysis_root and staging_root and the return annotation is Mapping[str, str]. Add a test that starts with an absent staging path, verifies all returned keys are safe POSIX-relative paths, and proves a second call with that existing path is rejected without changing any byte.

Do not edit uav_mec/publish_canonical.py, uav_mec/canonical_v2.py, tests/test_canonical_v2.py, or tests/test_canonical_publication.py in this task. Under the frozen master order, Plan 3 Task 10 does not exist yet: Plan 3 Tasks 1–9 run, this plan's Tasks 1–8 land, and only then Plan 3 Tasks 10–12 implement publisher integration. Plan 3 Task 10 owns the real-generator default, pre-manifest invocation-order test, full/light copy and provenance tests, missing/extra/digest-mismatch generator tests, and root checksum closure.

- [ ] **Step 3: Run RED**

Run:

    $PY -m pytest tests/test_publication_assets.py -q

Expected: import or assertion failures report the absent generator and manifest. No canonical output is created outside tmp_path.

- [ ] **Step 4: Freeze the figure and table manifest**

Create publication/figure_manifest.toml with schema_version = 1 and these exact IDs in order:

    system_model
    closure_pipeline
    iea_neighborhoods
    dd_admission
    main_hypotheses
    externality_mechanism
    resource_admission
    joint_two_by_two
    exact_three_layer_gap
    scaling
    robustness
    trace_driven
    runtime

Each `[[figures]]` table has exactly `id`, `kind`, `source_file`, `source_pointer`, `x_key`, `y_keys`, `zh_title`, `zh_x_label`, `zh_y_label`, `en_title`, `en_x_label`, `en_y_label`, and `claim_ids`. The first four use kind = "diagram" and a source_pointer into a static semantic specification in the same TOML file. The remaining nine use kind = "data", read only acceptance.json, statistics.json, metrics.json, or statistical_claims.csv, and identify their exact source keys. H1 through H4 are bound to main_hypotheses; every other claim_ids list names a supporting/correctness ID that Task 8 maps. Titles and axes describe the estimand and unit without stating an outcome.

For each data figure, generate one normalized UTF-8 CSV under tables and render both language variants from that exact CSV byte stream. Do not run two independent data queries. The figure manifest emitted into staging records `data_path`, `data_sha256`, `zh_path`, `zh_sha256`, `en_path`, and `en_sha256` for each row.

- [ ] **Step 5: Implement only the frozen pure function**

Create uav_mec/publication_assets.py with this public declaration exactly:

    def generate_bilingual_publication_assets(
        analysis_root: Path,
        staging_root: Path,
    ) -> Mapping[str, str]:

There is no rendered-assets CLI. The function follows this complete contract:

1. Resolve both paths, require analysis_root to be an existing nonsymlink directory, and require staging_root not to exist.
2. Read only the four verified analysis files named by Step 1 and publication/figure_manifest.toml. Reject a missing key, duplicate claim ID, schema mismatch, failed all_integrity_gates_pass, non-finite number, unsafe label, sentinel, symlink, or any path outside those roots before creating staging_root.
3. Create a sibling temporary directory, never the final canonical tree. Write only regular files under tables/, figures/, and latex/. Normalize all JSON with sorted keys and compact separators, all CSV with `lineterminator="\n"`, and all TeX/text with UTF-8 LF endings.
4. Render PDF with Matplotlib's noninteractive Agg backend, fixed dimensions, embedded fonts, fixed artist order, no transparency, and metadata `Creator="uav_mec.publication_assets"`, `CreationDate=None`, and `ModDate=None`. Set `svg.hashsalt` even though SVG is not emitted. Never call tight_layout; use explicit subplot margins.
5. For the four diagrams, render the same node/edge specification in both languages. For the nine data figures, parse the normalized CSV once and pass the same immutable data arrays to both renderers. Only titles, labels, legends, and captions differ.
6. Emit the exact paths `tables/{id}.csv` for data figures, `figures/zh/{id}.pdf`, `figures/en/{id}.pdf`, and `figures/figure-manifest.json`. The four diagrams do not get invented numeric CSV files.
7. Emit `latex/thesis/result-macros.tex`, `results-body.tex`, `abstract-results.tex`, `discussion-results.tex`, `conclusion-results.tex`, and `externality-certificate.tex`; emit `latex/journal/result-macros.tex`, `results-body.tex`, `abstract-results.tex`, `discussion-results.tex`, `conclusion-results.tex`, and `highlights.txt`. Every formal number is formatted once from the verified `statistical_claims.csv` row held by the resolver into a macro. Result sentences select wording from the five ClaimStatus values and never turn insufficient, small, uncertain, or negative evidence into a positive claim.
8. Emit `latex/publication-assets.json` with schema version, analysis input hashes, exact ordered output paths, figure data hashes, claim-status bindings, and direct source hashes for `publication/figure_manifest.toml`, `publication/claim_bindings.toml`, `uav_mec/publication_assets.py`, and `uav_mec/publication_claims.py` whenever the claim-binding pass is active. The publisher recomputes these hashes against the source snapshot and rejects drift. Do not include a clock time, machine name, process ID, absolute path, Git worktree path, or random value.
9. Compute every output hash after all writers close, compare the regular-file tree to the mapping keys, atomically rename the temporary directory to staging_root, and return an insertion-ordered dict keyed by sorted POSIX relative paths. On any exception, no staging_root remains.

Private helpers are typed, deterministic, and unit-tested. Use Decimal for publication formatting, `format(value, "f")` with manifest-controlled precision, and a single TeX escape function. No helper accepts a fallback numeric value.

- [ ] **Step 6: Record the exact Plan 3 Task 10 handoff**

Before committing, compare the implemented function to Plan 3's frozen `PublicationAssetGenerator` contract. Record in the Task 7 commit message body that the subsequent Plan 3 Task 10 must import this exact callable as the default of publish_canonical_v2, call it after analysis verification and before creating either root MANIFEST.json or SHA256SUMS, recompute its mapping, copy verified tables/, figures/, and latex/ into full/analysis, derive light only from full, write light/provenance.json, and then seal root closure.

This cross-plan ordering is the cycle break: Task 7 code lands before Plan 3 Task 10 and Plan 4; the later publisher invocation creates and hashes the actual result assets; Tasks 9 onward only read finalized light. No Plan 5 writing step regenerates an asset.

- [ ] **Step 7: Run GREEN and determinism twice**

Run:

    $PY -m pytest tests/test_publication_assets.py -q
    $RUFF check uav_mec/publication_assets.py \
      tests/test_publication_assets.py

Expected: all tests pass and the two synthetic output trees have identical path sets and hashes. Root closure is intentionally deferred to the named Plan 3 Task 10 integration tests.

- [ ] **Step 8: Commit the pre-Plan-4 generator**

Run:

    git add -- publication/figure_manifest.toml \
      uav_mec/publication_assets.py \
      tests/test_publication_assets.py
    git diff --cached --name-status
    git commit -m "feat: add deterministic bilingual publication assets" \
      -m "Handoff: Plan 3 Task 10 must invoke this generator before canonical manifest and checksum closure."

Expected: exactly the three named paths are committed. No publisher file, generated figure, table, canonical tree, or formal result is staged.

### Task 8: Bind claims and enforce canonical-v2 light as the only result-number source

**Depends on:** Task 7. This task also lands before Plan 4 so the canonical publisher hashes the resolved claim matrix and claim sentences on its only publication pass.

**Files:**
- Create: publication/claim_bindings.toml
- Create: uav_mec/publication_claims.py
- Modify: uav_mec/publication_assets.py
- Move: thesis/claims-matrix.md to archive/publication-v1/thesis-claims-matrix.md
- Move: paper/generated_results.tex to archive/publication-v1/paper-generated-results.tex
- Move: paper/generated_robustness.tex to archive/publication-v1/paper-generated-robustness.tex
- Move: paper/generated_statistics.tex to archive/publication-v1/paper-generated-statistics.tex
- Create: tests/test_publication_claims.py
- Create: tests/test_publication_numeric_provenance.py
- Modify: tests/test_publication_assets.py
- Modify: tests/test_active_thesis_claims.py

- [ ] **Step 1: Write RED tests for the exact publication claim schema**

Create tests/test_publication_claims.py with a local `write_synthetic_claim_inputs(root: Path) -> Path` helper. It writes complete acceptance.json, statistics.json, metrics.json, and the upstream nineteen-column `statistical_claims.csv` under tmp_path. Use synthetic labels and explicitly non-authoritative numeric values. Do not import a helper from Task 7 or depend on a formal artifact.

The generated publication matrix header is exactly:

    PUBLICATION_CLAIM_FIELDS = (
        "claim_id",
        "claim_text",
        "scope",
        "equation",
        "theorem",
        "test",
        "experiment_family",
        "artifact_path",
        "figure_or_table",
        "thesis_section",
        "paper_section",
        "gate_status",
    )

This is the approved twelve-column minimum schema; do not add a guessed thirteenth field. Add these exact tests:

    test_claim_bindings_cover_primary_theory_correctness_and_supporting_claims
    test_publication_claim_matrix_has_the_exact_twelve_column_schema
    test_every_binding_resolves_one_gate_and_one_canonical_evidence_path
    test_every_artifact_path_is_safe_posix_relative_and_exists_in_synthetic_root
    test_primary_claim_wording_covers_all_five_statuses_without_overclaiming
    test_theorem_and_correctness_claims_never_inherit_a_performance_gate
    test_thesis_and_paper_sections_are_nonempty_for_every_shared_claim
    test_claim_matrix_bytes_are_deterministic_for_identical_inputs
    test_publication_asset_manifest_hashes_figure_bindings_and_rendering_sources

For primary claims, parameterize the five statuses `meaningful_supported`, `positive_mde_observed_but_uncertain`, `positive_but_small`, `insufficient`, and `negative`. Assert the generated sentence uses the status-specific approved wording and never substitutes a stronger status. The test compares semantic wording keys, not a fabricated scientific value.

- [ ] **Step 2: Write RED tests for result-number provenance**

Create tests/test_publication_numeric_provenance.py. Define the active result-bearing source list explicitly:

    RESULT_SOURCES = (
        "thesis/frontmatter/abstract.tex",
        "thesis/chapters/07-results-mechanism.tex",
        "thesis/chapters/08-deployment-limitations.tex",
        "thesis/chapters/09-conclusion.tex",
        "paper/frontmatter/abstract.tex",
        "paper/sections/07-results-mechanism.tex",
        "paper/sections/08-discussion-limitations.tex",
        "paper/sections/09-conclusion.tex",
    )

Add exact tests:

    test_result_sources_use_only_publication_canonical_input_for_generated_text
    test_result_sources_have_no_literal_percent_pp_runtime_or_task_result
    test_result_sources_never_input_canonical_v1_pilot_runs_or_legacy_generated_tex
    test_every_result_macro_reference_resolves_in_canonical_light
    test_active_sources_do_not_read_archive_publication_v1
    test_canonical_light_is_the_only_result_number_source_for_both_languages

Implement the literal detector with this exact compiled expression:

    RESULT_LITERAL = re.compile(
        r"(?<![A-Za-z\\])[-+]?\d+(?:\.\d+)?\s*"
        r"(?:\\%|%|pp|percentage\s+points?|ms|milliseconds?|tasks?|倍|百分点)",
        re.IGNORECASE,
    )

Strip TeX comments before scanning. It is intentionally limited to result-bearing source files, so equation indices, protocol sample sizes, years, and theorem numbers elsewhere are not misclassified as observed results. Assert the only active artifact insertion form is `\PublicationCanonicalInput{latex/thesis/` or `\PublicationCanonicalInput{latex/journal/` as appropriate.

- [ ] **Step 3: Run RED**

Run:

    $PY -m pytest tests/test_publication_claims.py \
      tests/test_publication_numeric_provenance.py \
      tests/test_publication_assets.py \
      tests/test_active_thesis_claims.py -q

Expected: failures report the absent bindings/API, old generated files, old Markdown matrix, and unresolved canonical claim outputs.

- [ ] **Step 4: Freeze the result-independent claim bindings**

Create publication/claim_bindings.toml with schema_version = 1. Add one `[[claims]]` table for every ID below:

    MODEL-RAW-CLOSED
    THEORY-IEA-TERMINATION
    THEORY-DEPENDENCY-CLOSURE
    THEORY-DD-CARDINALITY
    CORRECT-INCREMENTAL-FULL
    CORRECT-DD-EXACT
    H1
    H2
    H3
    H4
    SUPPORT-EXTERNALITY
    SUPPORT-JOINT
    SUPPORT-EXACT-GAP
    SUPPORT-SCALING
    SUPPORT-ROBUSTNESS
    SUPPORT-TRACE
    SUPPORT-RUNTIME

Each table has exactly `claim_id`, `scope`, `equation`, `theorem`, `test`, `experiment_family`, `artifact_path`, `figure_or_table`, `thesis_section`, `paper_section`, `gate_source`, and `wording_key`. Inapplicable equation or theorem fields use the literal `not_applicable`; unfinished markers, empty strings, and angle-delimited dummy values are forbidden. `artifact_path` is a future canonical-root-relative path under full/analysis or light. `gate_source` names an exact JSON pointer in acceptance.json or an exact upstream claim ID. Theory rows use a correctness/integrity gate, never H1 through H4. Performance rows never use an integrity gate as evidence that an effect is positive.

Claim text is not stored as a manually resolved outcome. The wording_key selects one audited template, and publication_claims.py fills it from the canonical gate state. Formal numbers enter only through macros rendered from the upstream `statistical_claims.csv`.

- [ ] **Step 5: Implement claim resolution and exact path validation**

Create uav_mec/publication_claims.py with these complete public declarations:

    @dataclass(frozen=True, slots=True)
    class ClaimBinding:
        claim_id: str
        scope: str
        equation: str
        theorem: str
        test: str
        experiment_family: str
        artifact_path: str
        figure_or_table: str
        thesis_section: str
        paper_section: str
        gate_source: str
        wording_key: str


    def load_claim_bindings(path: Path) -> list[ClaimBinding]:


    def resolve_publication_claims(
        analysis_root: Path,
        bindings_path: Path,
    ) -> list[dict[str, str]]:


    def write_publication_claim_matrix(
        rows: Sequence[Mapping[str, str]],
        output_path: Path,
    ) -> str:


    def validate_publication_claim_matrix(
        matrix_path: Path,
        canonical_root: Path,
    ) -> None:

The declarations above are complete signatures, not stub bodies. Implement each with these exhaustive behaviors:

- `load_claim_bindings` uses tomllib, accepts schema version 1 only, enforces the exact binding fields, unique nonempty IDs, the exact seventeen-ID set above, safe POSIX artifact paths, known scopes, nonempty manuscript sections, and not_applicable rather than blanks.
- `resolve_publication_claims` verifies the four analysis inputs and reads the upstream statistical-claims table once. It resolves every gate_source exactly once, rejects contradictory acceptance/upstream statuses, selects the exact wording for its status, and returns rows in binding-file order with keys exactly equal to PUBLICATION_CLAIM_FIELDS.
- Wording tables have separate Chinese and English render functions but one semantic status key. meaningful_supported permits a bounded support statement; uncertain, small, insufficient, and negative each use explicit limited or negative wording. No branch uses “significant”, “improves”, “outperforms”, “validated”, “real-time”, or “optimal” unless the corresponding gate permits that exact scope.
- `write_publication_claim_matrix` writes UTF-8 CSV with the exact header, `newline=""`, `lineterminator="\n"`, and atomic replacement, then returns the SHA-256. It rejects extra/missing keys and duplicate IDs.
- `validate_publication_claim_matrix` rejects absolute paths, `..`, symlinks, missing evidence, files outside canonical_root, and evidence absent from root SHA256SUMS. It compares every gate status back to canonical acceptance/claim data and never trusts the CSV alone.

The asset manifest test changes one byte in each of `publication/figure_manifest.toml`, `publication/claim_bindings.toml`, `uav_mec/publication_assets.py`, and `uav_mec/publication_claims.py` within an isolated synthetic source snapshot and proves that the direct lineage map changes and that the publisher rejects an unbound source hash.

Use no current time, environment value, hostname, Git path, or fallback status.

- [ ] **Step 6: Integrate claim resolution into the single pre-hash asset pass**

Modify generate_bilingual_publication_assets so it calls resolve_publication_claims before creating staging_root, writes the sole approved twelve-column `claim_matrix.csv` at the staging root, and includes its digest in `latex/publication-assets.json` and the returned path/hash mapping. This file is the one shared thesis--journal claim matrix required by the approved specification; the nineteen-column analyzer detail remains separately named `statistical_claims.csv` and is not a second claim matrix. Result-macro and result-sentence renderers consume the resolved rows already held in memory; they do not reread a light or full canonical tree that does not yet exist.

Extend tests/test_publication_assets.py to assert the resolved matrix, both language result bodies, both abstracts, both conclusions, and highlights all carry the same semantic gate status for each claim. This is still before canonical publication; Plan 3 copies and hashes these bytes into full/light.

- [ ] **Step 7: Archive stale manually generated result sources**

After recording current hashes and diffs, run only these moves:

    mkdir -p archive/publication-v1
    git mv thesis/claims-matrix.md \
      archive/publication-v1/thesis-claims-matrix.md
    git mv paper/generated_results.tex \
      archive/publication-v1/paper-generated-results.tex
    git mv paper/generated_robustness.tex \
      archive/publication-v1/paper-generated-robustness.tex
    git mv paper/generated_statistics.tex \
      archive/publication-v1/paper-generated-statistics.tex

Do not edit the archived bytes. Active sources and tests may not read archive/publication-v1. The current PDF and historical pre-round manuscript remain untouched for Task 10.

- [ ] **Step 8: Run GREEN**

Run:

    $PY -m pytest tests/test_publication_claims.py \
      tests/test_publication_numeric_provenance.py \
      tests/test_publication_assets.py \
      tests/test_active_thesis_claims.py -q
    $RUFF check uav_mec/publication_claims.py \
      uav_mec/publication_assets.py \
      tests/test_publication_claims.py \
      tests/test_publication_numeric_provenance.py \
      tests/test_publication_assets.py

Expected: all claim bindings resolve, all evidence paths are safe in the synthetic root, stale result sources are inactive, and neither manuscript contains a hand-entered result literal.

- [ ] **Step 9: Commit claim provenance before Plan 4**

Run:

    git add -- publication/claim_bindings.toml \
      uav_mec/publication_claims.py \
      uav_mec/publication_assets.py \
      archive/publication-v1/thesis-claims-matrix.md \
      archive/publication-v1/paper-generated-results.tex \
      archive/publication-v1/paper-generated-robustness.tex \
      archive/publication-v1/paper-generated-statistics.tex \
      tests/test_publication_claims.py \
      tests/test_publication_numeric_provenance.py \
      tests/test_publication_assets.py \
      tests/test_active_thesis_claims.py
    git diff --cached --name-status
    git commit -m "feat: bind publication claims to canonical evidence"

Expected: exactly the named bindings, code, tests, and four renames are committed. No canonical artifact or manuscript result number is staged.

### Task 9: Consume finalized canonical-v2 light and pass manuscript content gates

**Depends on:** Plan 4 has published and verified artifacts/canonical/v2. Do not start this task from an analysis directory, formal run, or unsealed asset staging tree.

**Files:**
- Create: publication/section_budgets.toml
- Create: uav_mec/publication_quality.py
- Create: tests/test_publication_quality.py
- Modify: thesis/frontmatter/abstract.tex
- Modify: thesis/chapters/01-introduction.tex
- Modify: thesis/chapters/02-related-work.tex
- Modify: thesis/chapters/03-system-model.tex
- Modify: thesis/chapters/04-interference-aware-pairing.tex
- Modify: thesis/chapters/05-deadline-allocation.tex
- Modify: thesis/chapters/06-implementation-methodology.tex
- Modify: thesis/chapters/07-results-mechanism.tex
- Modify: thesis/chapters/08-deployment-limitations.tex
- Modify: thesis/chapters/09-conclusion.tex
- Modify: paper/frontmatter/abstract.tex
- Modify: paper/frontmatter/keywords.tex
- Modify: paper/sections/07-results-mechanism.tex
- Modify: paper/sections/08-discussion-limitations.tex
- Modify: paper/sections/09-conclusion.tex
- Modify: tests/test_publication_numeric_provenance.py

- [ ] **Step 1: Verify the canonical handoff before reading a result**

Run:

    test -f artifacts/canonical/v2/MANIFEST.json
    test -f artifacts/canonical/v2/SHA256SUMS
    $PY -m uav_mec.canonical_v2 verify \
      --root artifacts/canonical/v2
    (
      set -e
      cd artifacts/canonical/v2
      sha256sum --check SHA256SUMS
    )

Expected: the verifier and every checksum pass. Confirm that light contains `acceptance.json`, the sole twelve-column `claim_matrix.csv`, `statistical_claims.csv`, `figures/figure-manifest.json`, both language figure directories, both language LaTeX directories, and `provenance.json`. If any file is missing or changed, stop and repair/re-run Plan 4 publication; do not regenerate it in this plan.

- [ ] **Step 2: Freeze measurable source-quality budgets**

Create publication/section_budgets.toml with schema_version = 1 and these exact gates:

    thesis_total_min_han = 30000
    thesis_abstract_min_han = 500
    thesis_abstract_max_han = 800
    journal_abstract_max_words = 250
    journal_keywords_min = 1
    journal_keywords_max = 7
    journal_highlights_min = 3
    journal_highlights_max = 5
    journal_highlight_max_characters = 85

Add nine chapter rows with these writing targets, in order:

    thesis/chapters/01-introduction.tex                 4000
    thesis/chapters/02-related-work.tex                 4500
    thesis/chapters/03-system-model.tex                 5000
    thesis/chapters/04-interference-aware-pairing.tex   5000
    thesis/chapters/05-deadline-allocation.tex           5000
    thesis/chapters/06-implementation-methodology.tex    4000
    thesis/chapters/07-results-mechanism.tex             7000
    thesis/chapters/08-deployment-limitations.tex        3500
    thesis/chapters/09-conclusion.tex                    2000

The row values sum to a 40,000-Han planning target inside the approved 35,000--45,000 range; the hard school gate remains the expanded nine-chapter total of at least 30,000 Han characters. Do not pad with duplicated sentences, source listings, references, captions, metadata, or appendices. Count the canonical Chinese result snippets exactly once at their insertion sites.

- [ ] **Step 3: Write RED content-gate tests**

Create tests/test_publication_quality.py with local synthetic TeX/light helpers under tmp_path and active-repository tests. Add these exact cases:

    test_expand_canonical_inputs_rejects_escape_symlink_cycle_and_missing_file
    test_han_counter_strips_commands_comments_math_and_counts_visible_chinese
    test_english_word_counter_counts_visible_abstract_words
    test_active_thesis_has_exactly_nine_chapters_and_at_least_30000_han
    test_chinese_abstract_is_between_500_and_800_han_after_canonical_expansion
    test_journal_abstract_is_at_most_250_words_after_canonical_expansion
    test_journal_has_between_one_and_seven_keywords
    test_highlights_have_three_to_five_lines_each_at_most_85_characters
    test_final_sources_contain_no_required_or_canonical_missing_sentinel
    test_every_canonical_input_and_figure_exists_and_is_root_checksummed
    test_bilingual_figure_pairs_use_the_same_data_digest
    test_publication_claim_matrix_validates_against_canonical_root

The synthetic helper creates its own canonical root with root MANIFEST.json, root SHA256SUMS, light/provenance.json, a short TeX include graph, and safe dummy figure bytes. It rewrites SHA256SUMS after each negative setup. No test names or consumes an undefined fixture.

- [ ] **Step 4: Run RED before expanding prose**

Run:

    $PY -m pytest tests/test_publication_quality.py \
      tests/test_publication_numeric_provenance.py -q

Expected: at least the thesis character, abstract, canonical discussion-insertion, or final sentinel gates fail. Record the exact count report; do not weaken a threshold to make it pass.

- [ ] **Step 5: Implement the quality API and CLI**

Create uav_mec/publication_quality.py with these complete public declarations:

    @dataclass(frozen=True, slots=True)
    class PublicationQualityReport:
        thesis_han_characters: int
        thesis_abstract_han_characters: int
        journal_abstract_words: int
        journal_keyword_count: int
        journal_highlight_count: int
        checked_canonical_paths: Sequence[str]


    def expand_canonical_inputs(
        source_path: Path,
        canonical_light: Path,
    ) -> str:


    def visible_tex_text(source: str) -> str:


    def count_han_characters(source: str) -> int:


    def count_english_words(source: str) -> int:


    def validate_publication_sources(
        repository_root: Path,
        canonical_root: Path,
        budget_path: Path,
    ) -> PublicationQualityReport:

Import Sequence from collections.abc. The declarations are complete interfaces and must receive real bodies with these behaviors:

- Canonical expansion recognizes only `\PublicationCanonicalInput{relative/path}`. It resolves beneath canonical_root/light, rejects absolute paths, parent traversal, symlinks, cycles, missing paths, and a path absent from root SHA256SUMS, and preserves insertion order.
- visible_tex_text removes comments, math, labels, references, citations, and command names while retaining visible braced text. It rejects malformed braces instead of silently dropping text.
- count_han_characters counts Unicode U+3400 through U+4DBF and U+4E00 through U+9FFF after visible-text extraction. It does not count bibliography, appendix, metadata, source code, or captions toward the 30,000 gate.
- count_english_words counts alphabetic words with internal apostrophes/hyphens after visible-text extraction.
- validate_publication_sources first calls verify_canonical_v2, checks exactly the nine driver chapters, expands each once, enforces the total and abstract gates, parses keywords and highlights, rejects both sentinels, validates every canonical path against SHA256SUMS, compares bilingual data_sha256 values, and calls validate_publication_claim_matrix.
- The CLI is exactly `publication_quality --repository-root DIR --canonical-root DIR --budgets FILE`. It writes the report as canonical JSON to stdout and writes no file.

Add `main(argv: list[str] | None = None) -> int` and `if __name__ == "__main__"` entry code. Parser and validation errors exit nonzero before manuscript mutation.

- [ ] **Step 6: Complete the Chinese manuscript without hand-entering results**

Revise Chapters 1 through 6 and 8 through 9 to meet their planning targets with substantive argument, derivation, comparison, implementation rationale, threats to validity, and reproducibility detail. Chapter 7 keeps its result-independent headings and one canonical results-body insertion. Chapter 8 adds exactly:

    \PublicationCanonicalInput{latex/thesis/discussion-results.tex}

Chapter 9 retains exactly:

    \PublicationCanonicalInput{latex/thesis/conclusion-results.tex}

The Chinese abstract has result-independent problem/method/scope sentences plus exactly one `latex/thesis/abstract-results.tex` insertion. Edit it to the 500–800 Han gate after expansion. Do not copy a value or status from acceptance.json into tracked TeX. Cite every literature-derived statement. Do not claim a public repository, award, publication, funding source, supervisor, or deployment fact that controlled metadata/evidence does not establish.

- [ ] **Step 7: Complete the Computer Networks result boundary**

The paper abstract has result-independent context/method/scope text plus exactly one `latex/journal/abstract-results.tex` insertion and remains at most 250 words after expansion. The results section inputs only `latex/journal/results-body.tex`. Discussion adds exactly:

    \PublicationCanonicalInput{latex/journal/discussion-results.tex}

Conclusion keeps exactly:

    \PublicationCanonicalInput{latex/journal/conclusion-results.tex}

Use one to seven stable keywords. Highlights remain the finalized `artifacts/canonical/v2/light/latex/journal/highlights.txt` bytes and are not copied into a tracked result file. Result interpretations, negative findings, limitations, and scope wording come from the gated canonical snippets. Tracked English prose may connect sections and explain methods, but it may not restate a result using a literal number or stronger verb.

- [ ] **Step 8: Run GREEN and inspect counts**

Run:

    $PY -m pytest tests/test_publication_quality.py \
      tests/test_publication_numeric_provenance.py \
      tests/test_thesis_semantic_alignment.py \
      tests/test_computer_networks_structure.py \
      tests/test_latex_citations.py -q
    $PY -m uav_mec.publication_quality \
      --repository-root . \
      --canonical-root artifacts/canonical/v2 \
      --budgets publication/section_budgets.toml
    $RUFF check uav_mec/publication_quality.py \
      tests/test_publication_quality.py \
      tests/test_publication_numeric_provenance.py

Expected: all tests pass. The JSON report shows at least 30,000 thesis Han characters, a 500–800 Han Chinese abstract, no more than 250 journal-abstract words, accepted keyword/highlight counts, and a nonempty checksummed canonical-path list.

- [ ] **Step 9: Commit source adapters and substantive prose**

Run:

    git add -- publication/section_budgets.toml \
      uav_mec/publication_quality.py \
      tests/test_publication_quality.py \
      tests/test_publication_numeric_provenance.py \
      thesis/frontmatter/abstract.tex \
      thesis/chapters/01-introduction.tex \
      thesis/chapters/02-related-work.tex \
      thesis/chapters/03-system-model.tex \
      thesis/chapters/04-interference-aware-pairing.tex \
      thesis/chapters/05-deadline-allocation.tex \
      thesis/chapters/06-implementation-methodology.tex \
      thesis/chapters/07-results-mechanism.tex \
      thesis/chapters/08-deployment-limitations.tex \
      thesis/chapters/09-conclusion.tex \
      paper/frontmatter/abstract.tex \
      paper/frontmatter/keywords.tex \
      paper/sections/07-results-mechanism.tex \
      paper/sections/08-discussion-limitations.tex \
      paper/sections/09-conclusion.tex
    git diff --cached --name-status
    git commit -m "docs: complete canonical publication manuscripts"

Expected: exactly the named source, budget, quality module, and tests are committed. No file beneath artifacts/canonical/v2 is staged or changed.

### Task 10: Archive the historical manuscript and document the active publication workflow

**Files:**
- Move: paper/main_v2_pre_round1.tex to archive/publication-v1/main_v2-pre-round1.tex
- Move: paper/main_v2.pdf to archive/publication-v1/main_v2.pdf
- Create: publication/README.md
- Create: publication/overlap-ledger.md
- Modify: README.md
- Modify: uav_mec/code_CLAUDE.md
- Create: tests/test_publication_documentation.py

- [ ] **Step 1: Write RED documentation and archive-boundary tests**

Create tests/test_publication_documentation.py with these exact tests:

    test_active_readme_names_ustc_and_computer_networks_entry_points
    test_publication_readme_names_canonical_verify_quality_build_and_package_order
    test_code_readme_names_only_canonical_v2_light_for_result_writing
    test_overlap_ledger_maps_shared_material_and_distinct_contributions
    test_active_docs_do_not_claim_submission_public_archive_funding_or_achievement
    test_historical_manuscript_and_pdf_exist_only_under_archive_publication_v1
    test_active_latex_and_docs_do_not_read_archive_publication_v1

Use explicit path lists. Do not recursively treat this implementation plan or the historical archive as an active public claim. The false-claim detector looks for concrete assertions such as an assigned DOI, completed school submission, accepted journal article, named funding award, or public archive URL; it does not reject instructions that state those actions remain gated.

- [ ] **Step 2: Run RED**

Run:

    $PY -m pytest tests/test_publication_documentation.py -q

Expected: failures report historical files in paper/, absent active workflow documentation, and the missing overlap ledger.

- [ ] **Step 3: Preserve the historical bytes under an explicit archive**

Record and compare before moving:

    sha256sum paper/main_v2_pre_round1.tex paper/main_v2.pdf
    git diff -- paper/main_v2_pre_round1.tex paper/main_v2.pdf
    git mv paper/main_v2_pre_round1.tex \
      archive/publication-v1/main_v2-pre-round1.tex
    git mv paper/main_v2.pdf \
      archive/publication-v1/main_v2.pdf
    sha256sum archive/publication-v1/main_v2-pre-round1.tex \
      archive/publication-v1/main_v2.pdf

Expected: each destination hash equals its source hash. Do not edit or rebuild the archived PDF. Keep paper/algorithm.sty and paper/algorithmic.sty active until the final paper build proves whether they are required; do not silently substitute a system package.

- [ ] **Step 4: Write the active publication README**

publication/README.md documents, in order:

1. official USTC and Elsevier template locks and their verification test;
2. tracked sentinel metadata versus ignored private metadata;
3. the four validation targets and which human actions they gate;
4. the shared references/references.bib plus metadata_audit.csv;
5. frozen order `Plan 3 Tasks 1–9 -> Plan 5 Tasks 1–8 -> Plan 3 Tasks 10–12 -> Plan 4 -> Plan 5 Tasks 9–13`;
6. root-only canonical MANIFEST.json/SHA256SUMS and read-only light consumption;
7. claim bindings, resolved publication matrix, bilingual same-data figure manifest, and numeric-provenance check;
8. quick versus final builds and deterministic package audit;
9. external archive, signatures, author photos, school upload, and journal upload as separate human-controlled actions.

Use complete copy-pasteable commands with the exact PY and RUFF paths. Do not include an example person, grant, DOI, archive URL, acceptance state, or result value.

Update the root README publication section to link publication/README.md, thesis/main.tex, paper/main_v2.tex, artifacts/canonical/v2/light, and the final verification target. Update uav_mec/code_CLAUDE.md so code-facing artifact guidance names canonical-v2 full for audit and canonical-v2 light for manuscripts; keep canonical-v1 documented only as historical and never as an active default.

- [ ] **Step 5: Write the overlap ledger without making an outcome claim**

publication/overlap-ledger.md has these required columns:

    Material ID
    Shared scientific core
    Thesis location
    Journal location
    Thesis-only depth
    Journal-only focus
    Figure/table ownership
    Citation/copyright action
    Status source

Add rows for system model, IEA-CAP, DD-MCAA, EAD-JSRA, proof material, experiment protocol, each result family, figures/tables, appendices/supplement, and AI/tool disclosure. Status source points to a human overlap gate or canonical claim ID. The ledger states that the thesis provides full derivations, Chinese pedagogical context, engineering detail, and reproduction appendices; the journal provides a tighter cross-layer network contribution, selected proofs, focused evidence, and separate submission declarations. It does not assert publisher permission or school approval.

- [ ] **Step 6: Run GREEN**

Run:

    $PY -m pytest tests/test_publication_documentation.py \
      tests/test_publication_numeric_provenance.py \
      tests/test_latex_citations.py -q
    git diff --check

Expected: documentation, archive boundaries, active citations, and result-source boundaries pass.

- [ ] **Step 7: Commit the archival boundary and workflow docs**

Run:

    git add -- archive/publication-v1/main_v2-pre-round1.tex \
      archive/publication-v1/main_v2.pdf \
      publication/README.md \
      publication/overlap-ledger.md \
      README.md \
      uav_mec/code_CLAUDE.md \
      tests/test_publication_documentation.py
    git diff --cached --name-status
    git commit -m "docs: document active publication workflow"

Expected: exactly the two renames and five documentation/test paths are committed; no active PDF, canonical artifact, or private file is staged.

### Task 11: Add reproducible quick and final manuscript builds

**Files:**
- Create: publication/build-policy.toml
- Create: uav_mec/publication_build.py
- Create: tests/test_publication_build.py
- Modify: Makefile
- Modify: .gitignore
- Modify: thesis/latexmkrc

- [ ] **Step 1: Write RED build-policy and Makefile tests**

Create tests/test_publication_build.py with its own tmp_path repository helper containing tiny class, bibliography, metadata, gates, canonical root, thesis driver, and paper driver. Mock subprocess.run only in command-construction tests; integration tests use `shutil.which` and skip with a concrete missing-tool reason.

Add these exact tests:

    test_makefile_exposes_publication_metadata_quick_final_quality_and_package_targets
    test_makefile_defaults_to_canonical_v2_light_and_never_v1
    test_quick_build_uses_tectonic_draft_metadata_and_isolated_output
    test_thesis_research_final_uses_tectonic_and_thesis_build_target
    test_paper_research_final_uses_tectonic_and_journal_manuscript_target
    test_school_submission_mode_is_thesis_only_and_uses_latexmk_xelatex_bibtex
    test_engine_policy_rejects_an_invalid_target_mode_pair
    test_generated_driver_points_to_exact_output_metadata_and_canonical_roots
    test_quick_and_research_final_compile_the_generated_driver_not_source_main
    test_build_environment_prepends_locked_class_and_bst_paths
    test_build_environment_sets_source_date_epoch_locale_and_utc
    test_research_final_rejects_missing_tectonic_metadata_canonical_or_checksum_failure
    test_log_scanner_rejects_errors_undefined_references_and_large_overfull_boxes
    test_log_scanner_accepts_a_clean_complete_log
    test_build_result_records_commands_tool_versions_inputs_outputs_and_hashes
    test_build_never_writes_beside_thesis_or_paper_sources

The synthetic metadata helper fills all required fields with labels beginning `synthetic-`, sets only the target's human gates, and creates no real identity or result. The fake canonical root has root MANIFEST.json/SHA256SUMS and checksummed light files. No test depends on a private user file.

- [ ] **Step 2: Run RED**

Run:

    $PY -m pytest tests/test_publication_build.py -q

Expected: failures report the absent build module/policy and missing Makefile targets.

- [ ] **Step 3: Freeze the build policy**

Create publication/build-policy.toml with schema_version = 1 and exact sections for `quick`, `research_final`, `school_submission`, and `log_scan`:

- quick executable `tectonic`, offline/cached operation, keep logs, no shell escape, and draft metadata;
- research_final executable `/home/link/.local/bin/tectonic`, cached operation, no shell escape, and target-specific final metadata; it is the reproducible research-delivery engine for both manuscripts and cannot be deferred as an environment blocker;
- school_submission executables `latexmk`, `xelatex`, and `bibtex`, minimum TeX Live year 2021, no shell escape, thesis-only target, and school_submission metadata/gates; this separate mode exists for the school's required XeLaTeX/print workflow;
- deterministic environment `SOURCE_DATE_EPOCH=0`, `FORCE_SOURCE_DATE=1`, `TZ=UTC`, `LC_ALL=C.UTF-8`;
- fatal log patterns `LaTeX Error`, `Undefined control sequence`, `Citation`, `undefined`, `There were undefined references`, `Label(s) may have changed`, `Emergency stop`, `Fatal error`, and `multiply defined` combined with precise context so ordinary prose containing a word is not rejected;
- maximum overfull horizontal box width 2.0 pt;
- allowed output roots `build/publication/quick`, `build/publication/final`, and `build/publication/school-submission`, with the third accepted only for thesis/school_submission.

Record tool names, not a guessed local TeX path or version. The runtime preflight captures actual `--version` output into the build report.

- [ ] **Step 4: Implement the typed build runner**

Create uav_mec/publication_build.py with these complete public declarations:

    class BuildMode(StrEnum):
        QUICK = "quick"
        RESEARCH_FINAL = "research_final"
        SCHOOL_SUBMISSION = "school_submission"


    class BuildTarget(StrEnum):
        THESIS = "thesis"
        PAPER = "paper"


    @dataclass(frozen=True, slots=True)
    class BuildRequest:
        repository_root: Path
        canonical_root: Path
        metadata_path: Path
        gates_path: Path
        policy_path: Path
        output_dir: Path
        target: BuildTarget
        mode: BuildMode


    @dataclass(frozen=True, slots=True)
    class BuildResult:
        target: str
        mode: str
        pdf_path: Path
        log_path: Path
        report_path: Path
        pdf_sha256: str


    def scan_latex_log(log_path: Path, *, max_overfull_pt: float) -> None:


    def run_publication_build(request: BuildRequest) -> BuildResult:


    def main(argv: list[str] | None = None) -> int:

Implement all bodies with these exact behaviors:

1. Resolve paths; require repository, canonical, metadata, gates, policy, source driver, locked class, locked bibliography style, and shared bibliography to be regular nonsymlink files. Require output_dir beneath the policy's mode root and refuse output inside thesis/, paper/, artifacts/, third_party/, publication/private/, or archive/.
2. For every mode, call verify_canonical_v2 and publication quality before TeX. Generate metadata only into `output_dir/metadata`. Quick uses final=False. research_final uses final=True with thesis_build or journal_manuscript. school_submission uses final=True with school_submission and rejects target paper.
3. Prepend absolute locked paths to TEXINPUTS and BSTINPUTS while preserving the platform's empty trailing element for default TeX search. Set the deterministic environment from policy. Do not inherit TEXMFOUTPUT, shell-escape flags, or a caller-supplied output directory variable.
4. Before TeX, write `output_dir/driver.tex` atomically. It defines TeX-safe absolute `\PublicationMetadataRoot` as `output_dir/metadata` and `\PublicationCanonicalLightRoot` as `canonical_root/light`, then inputs the absolute active thesis/main.tex or paper/main_v2.tex. Both source drivers require those macros and contain no build-directory fallback. Quick and research_final run `/home/link/.local/bin/tectonic output_dir/driver.tex` with cached resources, keep-logs, and output_dir/output. school_submission runs `latexmk -r thesis/latexmkrc -xelatex -bibtex output_dir/driver.tex` with explicit aux/output directories. Tests inspect the real quick/research_final argv, driver bytes, metadata paths, and outputs under tmp_path.
5. Capture stdout/stderr to the target log, require exit zero and a nonempty regular PDF, then scan the complete log. Reject any unresolved citation/reference, rerun request, fatal error, or overfull hbox above 2.0 pt.
6. Write build-report.json atomically with schema version, target/mode, exact argv arrays, sanitized environment keys, tool version strings, template-lock hashes, metadata/gate input hashes, canonical root MANIFEST/SHA256SUMS hashes, source closure hash, PDF/log relative paths, and their hashes. Use canonical JSON and no wall-clock timestamp, absolute worktree path, hostname, PID, or user name.
7. A failure may leave diagnostic files under the requested build directory but never a success report. It never changes source, canonical, third_party, archive, or private bytes.

The CLI is exactly:

    publication_build --repository-root DIR \
      --canonical-root DIR \
      --metadata FILE \
      --gates FILE \
      --policy FILE \
      --output-dir DIR \
      --target thesis|paper \
      --mode quick|research_final|school_submission

It rejects unknown options and prints the deterministic report path and PDF hash on success.

- [ ] **Step 5: Add Makefile entry points**

Define these variables without overriding caller values:

    PY ?= /home/link/miniconda/envs/llm-ops-torch26/bin/python
    RUFF ?= /home/link/.venvs/formatters/bin/ruff
    CANONICAL_ROOT ?= artifacts/canonical/v2
    CANONICAL_LIGHT ?= artifacts/canonical/v2/light
    PUBLICATION_METADATA ?= publication/metadata.toml
    PUBLICATION_GATES ?= publication/human-gates.toml
    PUBLICATION_BUILD_POLICY ?= publication/build-policy.toml

Add exact phony targets:

    publication-template-locks
    publication-metadata-draft
    publication-canonical-verify
    publication-quality
    thesis-quick
    paper-quick
    publication-quick
    thesis-final
    paper-final
    publication-final
    thesis-school-submission
    publication-check
    submission-package
    submission-audit

publication-quick depends on both quick builds. thesis-final and paper-final use research_final/Tectonic; publication-final depends on canonical verification, quality, and both reproducible research-final builds. thesis-school-submission is separate, thesis-only, and invokes school_submission/XeLaTeX after its human gates pass. publication-check runs all publication Python tests, citation tests, template locks, numeric provenance, quality, and both quick builds. submission targets are wired in Task 12; until that commit, declare their names with an explicit failing message that the packager task has not landed, then replace those recipes in Task 12. Do not call a recursive clean target.

Update thesis/latexmkrc from the official frozen base only for school_submission mode: force XeLaTeX, BibTeX, runner-supplied output/aux directories, halt-on-error, and no shell escape. Research delivery does not depend on latexmk. Do not add a dependency on latexmk's current directory outside thesis/.

- [ ] **Step 6: Ignore only reproducible build outputs**

Append exact rules to .gitignore:

    build/publication/quick/
    build/publication/final/
    build/publication/school-submission/
    build/publication/package-staging/
    dist/submission/

Do not ignore publication metadata schemas, template locks, source manuscripts, package manifest policy, or audit reports intended to be reviewed.

- [ ] **Step 7: Run GREEN and available integration builds**

Run:

    $PY -m pytest tests/test_publication_build.py \
      tests/test_publication_metadata.py \
      tests/test_publication_quality.py -q
    $RUFF check uav_mec/publication_build.py \
      tests/test_publication_build.py
    make publication-template-locks
    make publication-canonical-verify
    make publication-quality
    make publication-quick

Expected: unit tests and Tectonic quick builds pass. If Tectonic's cache lacks a locked dependency, provision and lock that cache in the build policy rather than leaving research delivery blocked.

After the user supplies target-resolved local metadata/gates, run the reproducible final builds with the already available Tectonic engine:

    make publication-final \
      PUBLICATION_METADATA=publication/metadata.local.toml \
      PUBLICATION_GATES=publication/human-gates.local.toml

Expected: both research-final PDFs, logs, generated drivers, metadata roots, and deterministic build reports exist under build/publication/final and all log gates pass. Absence of latexmk/xelatex/bibtex does not block research_delivery or journal_package.

Only when preparing the school-upload PDF, preflight and run:

    command -v latexmk
    command -v xelatex
    command -v bibtex
    make thesis-school-submission \
      PUBLICATION_METADATA=publication/metadata.local.toml \
      PUBLICATION_GATES=publication/human-gates.local.toml

Expected: the XeLaTeX build runs only after all school_submission human gates pass. A missing school TeX Live toolchain blocks that external-submission mode only.

- [ ] **Step 8: Commit the build system**

Run:

    git add -- publication/build-policy.toml \
      uav_mec/publication_build.py \
      tests/test_publication_build.py \
      Makefile \
      .gitignore \
      thesis/latexmkrc
    git diff --cached --name-status
    git commit -m "build: add reproducible publication builds"

Expected: exactly the six named paths are committed. No PDF, log, build report, private metadata, or canonical file is staged.

### Task 12: Build and audit a deterministic flat Computer Networks submission package

**Depends on:** A passing paper-final build with journal_manuscript metadata. Actual journal submission additionally requires every journal_submission gate and private file named below.

**Files:**
- Create: publication/submission-policy.toml
- Create: paper/submission/cover-letter.tex
- Create: paper/submission/overlap-disclosure.tex
- Create: paper/submission/supplementary-material.tex
- Create: paper/submission/author-biographies.tex
- Create: paper/submission/README.md
- Create: uav_mec/submission_package.py
- Create: tests/test_submission_package.py
- Modify: Makefile

- [ ] **Step 1: Write self-contained deterministic-package RED tests**

Create tests/test_submission_package.py. Its `write_submission_fixture(root: Path) -> dict[str, Path]` helper creates a complete synthetic repository subtree, root-only canonical MANIFEST.json/SHA256SUMS, light provenance and assets, resolved journal metadata, approved journal_submission gates, private dummy photo bytes, a structurally valid dummy DOCX ZIP for the competing-interest form, a clean paper build report, paper PDF bytes, locked class/style, shared bibliography, active TeX sources, and submission templates. Every synthetic human value begins `synthetic-`; no repository private file is read.

Add these exact tests:

    test_submission_requires_journal_submission_metadata_and_every_admin_gate
    test_submission_requires_competing_interest_docx_photos_and_clean_final_build
    test_submission_rejects_stale_build_report_or_canonical_checksum
    test_flattened_manuscript_has_no_parent_path_subdirectory_or_canonical_dependency
    test_package_has_required_manuscript_source_pdf_bib_class_style_figures_and_statements
    test_package_highlights_have_three_to_five_lines_with_85_character_limit
    test_package_biographies_are_at_most_100_words_per_author
    test_package_contains_no_required_sentinel_absolute_path_symlink_or_special_file
    test_package_manifest_hashes_every_payload_entry_exactly_once
    test_zip_members_are_sorted_flat_and_have_fixed_timestamp_permissions
    test_two_packages_from_identical_inputs_are_byte_identical
    test_submission_audit_recomputes_zip_entry_and_outer_hashes
    test_packager_never_uploads_sends_signs_or_changes_human_gates

For the no-external-action test, monkeypatch socket creation and subprocess calls to fail. The packager is a local file transformation and needs neither. Check the gates file hash before and after.

- [ ] **Step 2: Run RED**

Run:

    $PY -m pytest tests/test_submission_package.py -q

Expected: import failure or missing policy/template/package behavior. Nothing is written outside tmp_path.

- [ ] **Step 3: Freeze package contents and private-input policy**

Create publication/submission-policy.toml with schema_version = 1, journal = "Computer Networks", template_lock = `publication/template-locks/elsevier-computer-networks-2026.json`, and these exact payload classes:

    manuscript.pdf
    manuscript.tex
    elsarticle.cls
    elsarticle-num.bst
    references.bib
    highlights.txt
    cover-letter.pdf
    cover-letter.tex
    overlap-disclosure.pdf
    overlap-disclosure.tex
    supplementary-material.pdf
    supplementary-material.tex
    credit-statement.txt
    funding-statement.txt
    competing-interests.docx
    data-availability.txt
    ai-use.txt
    author-biographies.tex
    one author-photo file per author, assigned zero-padded sequential names beginning with author-photo-01 and retaining each safe original extension
    English figures named figure-{ordered two-digit index}-{figure-id}.pdf
    PACKAGE-MANIFEST.json

All ZIP members are at the archive root; no member contains `/`. Set fixed ZIP timestamp `1980-01-01 00:00:00`, Unix mode 0644, UTF-8 flag, DEFLATE compression level 9, sorted member order, and no archive comment or extra field. PACKAGE-MANIFEST.json has schema version, journal, canonical MANIFEST/SHA256SUMS hashes, final paper build-report hash, and a sorted payload mapping of member name to SHA-256; it excludes itself to avoid a self-hash cycle. The outer audit separately hashes the complete ZIP and every member including PACKAGE-MANIFEST.json.

Private inputs are exactly:

    publication/private/competing-interests.docx
    publication/private/author-photos/ followed by each basename listed in resolved metadata

The tracked policy contains no person or funding value. The packager verifies DOCX ZIP structure and photo regular-file/type/extension constraints but never interprets a signature or marks a gate approved.

- [ ] **Step 4: Write truthful result-independent submission templates**

paper/submission/cover-letter.tex uses generated title/author metadata and controlled macros for contribution, scope, overlap, data availability, competing interests, funding, AI use, exclusive-consideration confirmation, and corresponding-author contact. It does not name an editor, assign a manuscript number, claim acceptance, claim public archiving, or contain a result literal. The final PDF is blocked until cover_letter_confirmed and exclusive_submission_confirmed are approved.

paper/submission/overlap-disclosure.tex renders publication/overlap-ledger.md into a concise disclosure of shared scientific core and distinct depth/focus. It does not assert permission. paper/submission/supplementary-material.tex includes only the policy-listed proof/protocol tables and canonical supporting assets; supplementary_material_confirmed controls the selection. paper/submission/author-biographies.tex is a metadata-driven shell that emits one biography and one photo reference per author. Tracked templates contain no personal values.

paper/submission/README.md maps every payload to its source, human owner, automated validation, and journal upload category. It states that package creation is not journal submission.

- [ ] **Step 5: Implement flattening, deterministic ZIP, and audit**

Create uav_mec/submission_package.py with these complete public declarations:

    @dataclass(frozen=True, slots=True)
    class SubmissionPackageResult:
        zip_path: Path
        audit_path: Path
        zip_sha256: str
        member_hashes: Mapping[str, str]


    def build_computer_networks_submission(
        repository_root: Path,
        canonical_root: Path,
        metadata_path: Path,
        gates_path: Path,
        policy_path: Path,
        paper_build_report: Path,
        private_root: Path,
        output_dir: Path,
    ) -> SubmissionPackageResult:


    def audit_submission_package(
        zip_path: Path,
        audit_path: Path,
        policy_path: Path,
    ) -> None:


    def main(argv: list[str] | None = None) -> int:

Implement every body with this exhaustive algorithm:

1. Resolve and validate all roots/files before creating output. Call verify_canonical_v2, validate_publication_sources, validate_publication_claim_matrix, validate_final_metadata with target journal_submission, and the final-build report validator. Recompute every referenced hash.
2. Require the four extra journal submission gates `competing_interests_form_supplied`, `exclusive_submission_confirmed`, `cover_letter_confirmed`, and `supplementary_material_confirmed` to be approved along with authors/photos/upload authorization. Never write a gate.
3. Create a temporary directory under build/publication/package-staging. Recursively expand ordinary `\input`/`\include` directives from paper/main_v2.tex with cycle, escape, symlink, missing-file, and duplicate-driver guards. Expand PublicationCanonicalInput only through canonical light. Rewrite the bibliography to `references`, class/style to the flat locked names, and every English canonical figure reference to its policy-assigned flat name. The resulting manuscript.tex has no dependency containing `/`, `..`, `artifacts`, `build`, `archive`, or an absolute path.
4. Copy source bytes for class, style, bibliography, English figures, private DOCX, and photos. Generate declaration text only from resolved metadata/gates. Generate biography TeX with at most 100 visible words per author. Compile cover letter, overlap disclosure, and supplementary material under the final deterministic TeX environment; scan their logs with scan_latex_log.
5. Validate the full staging allowlist, regular files, flat safe names, no symlink/special file, sentinels, result provenance, abstract/keyword/highlight limits, and bibliography resolution. No unlisted byte enters the ZIP.
6. Write canonical PACKAGE-MANIFEST.json. Build ZIP entries in sorted order with a fresh ZipInfo for each member, fixed date_time `(1980, 1, 1, 0, 0, 0)`, create_system 3, external_attr `0o100644 << 16`, empty extra/comment, and `writestr` using ZIP_DEFLATED and compresslevel 9. Never use ZipFile.write because it imports filesystem times.
7. Write submission-audit.json beside the ZIP using atomic replacement. It contains policy and input hashes, sorted member sizes/hashes, package-manifest hash, and outer ZIP hash; it contains no timestamp, absolute path, or private-file content. Run audit_submission_package before returning.
8. On failure, do not publish the output ZIP/audit. On success, atomically rename into initially absent exact output filenames and refuse an existing ZIP or audit rather than overwriting it. Never upload, email, sign, delete an existing user package, or modify canonical/private/source files.

The CLI subcommands are exactly:

    submission_package build --repository-root DIR --canonical-root DIR \
      --metadata FILE --gates FILE --policy FILE \
      --paper-build-report FILE --private-root DIR --output-dir DIR

    submission_package audit --zip FILE --audit FILE --policy FILE

Unknown options are rejected before output creation.

- [ ] **Step 6: Replace the Makefile submission stubs**

Define:

    PUBLICATION_PRIVATE_ROOT ?= publication/private
    PAPER_BUILD_REPORT ?= build/publication/final/paper/build-report.json
    SUBMISSION_OUTPUT ?= dist/submission

submission-package depends on paper-final, publication-quality, and publication-canonical-verify, then calls the build subcommand with journal local metadata/gates. submission-audit depends on the ZIP/audit and calls the audit subcommand. Neither target uploads anything.

- [ ] **Step 7: Run GREEN and build twice**

Run:

    $PY -m pytest tests/test_submission_package.py \
      tests/test_publication_build.py \
      tests/test_publication_quality.py -q
    $RUFF check uav_mec/submission_package.py \
      tests/test_submission_package.py

After local journal submission metadata, approved gates, private DOCX/photos, and final TeX tools are present, run:

    make submission-package \
      SUBMISSION_OUTPUT=build/publication/package-staging/package-a \
      PUBLICATION_METADATA=publication/metadata.local.toml \
      PUBLICATION_GATES=publication/human-gates.local.toml
    make submission-package \
      SUBMISSION_OUTPUT=build/publication/package-staging/package-b \
      PUBLICATION_METADATA=publication/metadata.local.toml \
      PUBLICATION_GATES=publication/human-gates.local.toml
    cmp build/publication/package-staging/package-a/computer-networks-submission.zip \
      build/publication/package-staging/package-b/computer-networks-submission.zip
    make submission-package \
      PUBLICATION_METADATA=publication/metadata.local.toml \
      PUBLICATION_GATES=publication/human-gates.local.toml
    make submission-audit \
      PUBLICATION_METADATA=publication/metadata.local.toml \
      PUBLICATION_GATES=publication/human-gates.local.toml

Expected: both ZIP byte streams are identical and audit passes. If human/private/toolchain inputs are absent, report their exact names; do not generate dummy submission content outside tests.

- [ ] **Step 8: Commit source templates and packager**

Run:

    git add -- publication/submission-policy.toml \
      paper/submission/cover-letter.tex \
      paper/submission/overlap-disclosure.tex \
      paper/submission/supplementary-material.tex \
      paper/submission/author-biographies.tex \
      paper/submission/README.md \
      uav_mec/submission_package.py \
      tests/test_submission_package.py \
      Makefile
    git diff --cached --name-status
    git commit -m "feat: add deterministic journal submission package"

Expected: exactly the nine named source/policy/code/test paths are committed. No ZIP, PDF, audit output, private DOCX/photo, local metadata, or canonical artifact is staged.

### Task 13: Produce a deterministic release audit and perform final verification

**Files:**
- Create: publication/release-policy.toml
- Create: publication/release-checklist.md
- Create: uav_mec/publication_release.py
- Create: tests/test_publication_release.py
- Modify: publication/README.md
- Modify: Makefile
- Modify: .gitignore

- [ ] **Step 1: Write RED release-mode tests**

Create tests/test_publication_release.py with a local synthetic release tree containing valid template locks, canonical root closure, quality report inputs, thesis/paper build reports, optional journal package/audit, metadata, and gates. Add these exact tests:

    test_research_delivery_requires_templates_canonical_quality_and_both_final_builds
    test_research_delivery_reports_signatures_photos_and_uploads_as_pending_not_passed
    test_school_submission_requires_signed_declarations_and_school_authorization
    test_journal_submission_requires_deterministic_package_and_all_journal_gates
    test_release_audit_rejects_stale_source_canonical_build_or_package_hash
    test_release_audit_contains_no_timestamp_absolute_path_hostname_or_personal_value
    test_identical_release_inputs_produce_byte_identical_audit_json
    test_release_audit_never_changes_gates_uploads_or_signs

Each test writes every input beneath tmp_path and computes its own hashes. The synthetic private fields use synthetic labels. No test reads publication/private or artifacts/formal.

- [ ] **Step 2: Run RED**

Run:

    $PY -m pytest tests/test_publication_release.py -q

Expected: import failure because release policy and audit module do not exist.

- [ ] **Step 3: Freeze the three release modes**

Create publication/release-policy.toml with schema_version = 1 and modes:

- `research_delivery`: requires official template locks, one audited bibliography, canonical-v2 root verification, claim/number provenance, manuscript quality, clean thesis-final and paper-final build reports. It records signed declarations, school/journal upload authorization, author photos, competing-interest form, and external archival authorization as human status; it does not require or approve them.
- `school_submission`: adds target school_submission metadata validation, signed declaration PDFs, and school_submission_authorized = approved. It does not imply journal upload.
- `journal_submission`: adds target journal_submission validation, clean deterministic submission ZIP/audit, private DOCX/photos, exclusive-submission/cover-letter/supplement gates, and journal_submission_authorized = approved. It does not imply school upload or public archive authorization.

Every mode requires zero source/canonical drift. External archival authorization is reported independently in all modes and cannot be inferred from package creation.

publication/release-checklist.md lists each automated command, expected evidence file, responsible human gate, and failure owner. Its checkboxes remain human-maintained; release tooling reads gates, not checkbox marks. It contains no claimed completed submission.

- [ ] **Step 4: Implement deterministic release aggregation**

Create uav_mec/publication_release.py with these complete declarations:

    class ReleaseMode(StrEnum):
        RESEARCH_DELIVERY = "research_delivery"
        SCHOOL_SUBMISSION = "school_submission"
        JOURNAL_SUBMISSION = "journal_submission"


    @dataclass(frozen=True, slots=True)
    class ReleaseAuditResult:
        mode: str
        output_path: Path
        output_sha256: str
        passed_checks: Sequence[str]
        pending_human_actions: Sequence[str]


    def audit_publication_release(
        repository_root: Path,
        canonical_root: Path,
        metadata_path: Path,
        gates_path: Path,
        release_policy_path: Path,
        thesis_build_report: Path,
        paper_build_report: Path,
        output_path: Path,
        *,
        mode: ReleaseMode,
        submission_zip: Path | None = None,
        submission_audit: Path | None = None,
    ) -> ReleaseAuditResult:


    def main(argv: list[str] | None = None) -> int:

Import Sequence and implement all bodies. The auditor:

1. Recomputes template lock/archive/source hashes, bibliography audit, active citations, canonical verification/root checksums, claim matrix evidence, numeric provenance, content quality, and both build-report input/output hashes.
2. Calls validate_final_metadata only for the requested mode's target. research_delivery validates thesis_build and journal_manuscript separately; it does not call a submission target.
3. For school_submission, verifies the two private signed PDFs without reading signature meaning. For journal_submission, calls audit_submission_package and binds the package outer hash to the release audit.
4. Writes canonical JSON with schema version, mode, sorted passed check IDs, sorted pending human action IDs, input hash map, canonical/build/package hashes, and no timestamp, absolute path, hostname, user, PID, personal metadata value, or formal result value.
5. Fails a required automated or mode-specific human gate. In research_delivery only, non-required administrative gates are listed as pending; they are never labeled passed. No mode mutates a gate or performs an external action.
6. Requires output_path to be initially absent beneath build/publication/release, writes through atomic rename, recomputes its hash, and returns it.

The CLI is exactly:

    publication_release audit --repository-root DIR --canonical-root DIR \
      --metadata FILE --gates FILE --policy FILE \
      --thesis-build-report FILE --paper-build-report FILE \
      --mode research_delivery|school_submission|journal_submission \
      --output FILE \
      [--submission-zip FILE --submission-audit FILE]

The bracketed options are accepted only together and are required only for journal_submission. Unknown or mode-inapplicable options fail before output creation.

- [ ] **Step 5: Wire the release audit and ignore its reproducible output**

Append to .gitignore:

    build/publication/release/

Add Makefile variables:

    THESIS_BUILD_REPORT ?= build/publication/final/thesis/build-report.json
    PAPER_BUILD_REPORT ?= build/publication/final/paper/build-report.json
    RELEASE_MODE ?= research_delivery
    RELEASE_AUDIT ?= build/publication/release/release-audit.json

Add `publication-release-audit`. It depends on publication-final and invokes the audit command. When RELEASE_MODE is journal_submission, require the submission ZIP/audit variables and submission-audit target. It never invokes an upload.

Update publication/README.md to name all three modes, their distinct gates, the release audit path, and the rule that a passing research delivery audit is not a school or journal submission.

- [ ] **Step 6: Run GREEN, then commit the release tooling**

Run:

    $PY -m pytest tests/test_publication_release.py \
      tests/test_submission_package.py \
      tests/test_publication_build.py \
      tests/test_publication_quality.py -q
    $RUFF check uav_mec/publication_release.py \
      tests/test_publication_release.py
    git diff --check

Expected: all release-mode, package, build, and quality tests pass.

Commit:

    git add -- publication/release-policy.toml \
      publication/release-checklist.md \
      uav_mec/publication_release.py \
      tests/test_publication_release.py \
      publication/README.md \
      Makefile \
      .gitignore
    git diff --cached --name-status
    git commit -m "chore: add publication release audit"

Expected: exactly the seven named paths are committed. No release audit, build output, package, private file, or canonical artifact is staged.

- [ ] **Step 7: Run the full automated verification after the final commit**

Run from the worktree root:

    git status --short
    git diff --check
    $PY -m pytest tests/test_publication_template_locks.py \
      tests/test_publication_metadata.py \
      tests/test_publication_bibliography.py \
      tests/test_latex_citations.py \
      tests/test_ustc_thesis_structure.py \
      tests/test_thesis_semantic_alignment.py \
      tests/test_active_thesis_claims.py \
      tests/test_computer_networks_structure.py \
      tests/test_publication_assets.py \
      tests/test_publication_claims.py \
      tests/test_publication_numeric_provenance.py \
      tests/test_publication_quality.py \
      tests/test_publication_documentation.py \
      tests/test_publication_build.py \
      tests/test_submission_package.py \
      tests/test_publication_release.py -q
    $RUFF check uav_mec/publication_metadata.py \
      uav_mec/publication_bibliography.py \
      uav_mec/publication_assets.py \
      uav_mec/publication_claims.py \
      uav_mec/publication_quality.py \
      uav_mec/publication_build.py \
      uav_mec/submission_package.py \
      uav_mec/publication_release.py
    $PY -m uav_mec.canonical_v2 verify \
      --root artifacts/canonical/v2
    (
      set -e
      cd artifacts/canonical/v2
      sha256sum --check SHA256SUMS
    )

Expected: every test/lint/checksum gate passes and the subshell leaves the caller at the worktree root. git status shows no unintended source or canonical modification.

- [ ] **Step 8: Run final builds, package audit, and release audit at the authorized level**

For research delivery, run with resolved local metadata/gates:

    make publication-final \
      PUBLICATION_METADATA=publication/metadata.local.toml \
      PUBLICATION_GATES=publication/human-gates.local.toml
    make publication-release-audit \
      RELEASE_MODE=research_delivery \
      PUBLICATION_METADATA=publication/metadata.local.toml \
      PUBLICATION_GATES=publication/human-gates.local.toml

For an authorized journal package, additionally run:

    make submission-package \
      PUBLICATION_METADATA=publication/metadata.local.toml \
      PUBLICATION_GATES=publication/human-gates.local.toml
    make submission-audit \
      PUBLICATION_METADATA=publication/metadata.local.toml \
      PUBLICATION_GATES=publication/human-gates.local.toml
    make publication-release-audit \
      RELEASE_MODE=journal_submission \
      RELEASE_AUDIT=build/publication/release/journal-submission-audit.json \
      PUBLICATION_METADATA=publication/metadata.local.toml \
      PUBLICATION_GATES=publication/human-gates.local.toml

Expected: research delivery may list administrative actions as pending. Journal submission mode passes only with every required human gate and deterministic package evidence. Neither command uploads.

- [ ] **Step 9: Final handoff and stop boundary**

Report:

- template archive and derived-class hashes;
- canonical root MANIFEST/SHA256SUMS hashes;
- thesis Han count and both abstract/keyword/highlight gate results;
- thesis/paper PDF and build-report paths/hashes;
- claim matrix, bilingual figure manifest, and same-data verification result;
- submission ZIP/audit paths/hashes when journal mode is authorized;
- release-audit path/hash and its mode;
- each remaining human gate by exact name.

Do not report a school upload, journal upload, signed declaration, public archive, funding fact, authorship approval, achievement, or acceptance unless its responsible human gate and external system evidence actually exist. Stop after local handoff; external submission is outside this plan.
