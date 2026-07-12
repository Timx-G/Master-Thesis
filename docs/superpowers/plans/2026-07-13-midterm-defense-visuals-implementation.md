# Midterm Defense Visuals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate five polished Chinese defense figures and one algorithm-selection table, establish the EAD-JSRA/IEA-CAP/DD-MCAA naming contract, and rebuild the existing 17-slide midterm defense deck around those assets.

**Architecture:** A focused Python renderer will own defense-only visual content and export deterministic 16:9 PNG/PDF assets plus a JSON manifest. The existing PptxGenJS builder will embed those tracked assets and retain editable slide notes, while automated tests validate names, dimensions, evidence text, page order, and stale-content removal. Canonical method keys remain unchanged; only presentation labels change.

**Tech Stack:** Python 3.11, Matplotlib, Pillow, JSON, PptxGenJS, pytest, LibreOffice Writer/PDF QA where available.

---

### Task 1: Establish The Algorithm-Naming And Visual-Asset Contract

**Files:**
- Create: `tests/test_defense_visual_assets.py`
- Create: `deliverables/defense_2026/sources/defense_visual_content.py`
- Modify: `tests/test_active_thesis_claims.py`

- [ ] **Step 1: Write failing naming and manifest tests**

Create tests that import `defense_visual_content.py` through `importlib.util` and assert:

```python
assert content.ALGO_NAMES["full"]["acronym"] == "EAD-JSRA"
assert content.ALGO_NAMES["innovation1"]["acronym"] == "IEA-CAP"
assert content.ALGO_NAMES["innovation2"]["acronym"] == "DD-MCAA"
assert set(content.ASSET_SPECS) == {
    "01_background",
    "02_ead_jsra_framework",
    "03_iea_cap_flow",
    "04_dd_mcaa_flow",
    "05_progress",
    "table_algorithm_selection",
}
```

Extend the active-claim test to require the three acronyms in `defense/innovation-comparison.md` and prohibit `Full method` as a public method name.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
.venv-lock/bin/python -m pytest tests/test_defense_visual_assets.py tests/test_active_thesis_claims.py -q
```

Expected: FAIL because `defense_visual_content.py` and the naming contract do not exist.

- [ ] **Step 3: Implement the content contract**

Create `defense_visual_content.py` with immutable dictionaries for the three algorithm names, the six output filenames/titles, canonical evidence strings, progress work packages, and algorithm-selection rows. Include the mapping `full -> EAD-JSRA（本文算法）`; do not rename canonical keys.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the same pytest command. Expected: PASS.

- [ ] **Step 5: Commit the contract**

```bash
git add tests/test_defense_visual_assets.py tests/test_active_thesis_claims.py \
  deliverables/defense_2026/sources/defense_visual_content.py
git commit -m "test: define defense visual naming contract"
```

---

### Task 2: Build The Deterministic Vector Visual Renderer

**Files:**
- Create: `deliverables/defense_2026/sources/generate_visual_assets.py`
- Modify: `tests/test_defense_visual_assets.py`
- Create outputs: `deliverables/defense_2026/visuals/*.png`
- Create outputs: `deliverables/defense_2026/visuals/*.pdf`
- Create output: `deliverables/defense_2026/visuals/manifest.json`

- [ ] **Step 1: Add failing renderer-output tests**

Add a subprocess test that runs the generator into `tmp_path`, then checks every declared PNG and PDF exists. With Pillow, assert every PNG is at least 2400×1350, uses RGB/RGBA, and has a non-white pixel ratio above 2%. Parse `manifest.json` and assert the output list, dimensions, algorithm acronyms, and canonical evidence version.

- [ ] **Step 2: Run the renderer test and verify RED**

Run:

```bash
.venv-lock/bin/python -m pytest tests/test_defense_visual_assets.py::test_renderer_exports_all_assets -v
```

Expected: FAIL because `generate_visual_assets.py` is absent.

- [ ] **Step 3: Implement shared drawing primitives**

Implement a 16×9-inch Matplotlib canvas at 150 dpi with Noto Sans CJK SC, stable colors, rounded panels, arrows, decision diamonds, status markers, formula text, and a footer boundary note. Centralize `save_asset(fig, stem, output_dir)` so the same figure writes PNG and vector PDF.

- [ ] **Step 4: Implement the six render functions**

Implement:

```python
render_background()
render_ead_jsra_framework()
render_iea_cap_flow()
render_dd_mcaa_flow()
render_algorithm_selection_table()
render_progress()
```

Use only strings from `defense_visual_content.py`. Ensure the overall framework places DD-MCAA inside candidate evaluation, IEA-CAP ends at a neighborhood-local optimum, and the progress figure marks only school-template/defense work as pending.

- [ ] **Step 5: Generate tracked assets and verify GREEN**

Run:

```bash
.venv-lock/bin/python deliverables/defense_2026/sources/generate_visual_assets.py \
  --output deliverables/defense_2026/visuals
.venv-lock/bin/python -m pytest tests/test_defense_visual_assets.py -q
```

Expected: six PNGs, six PDFs, one manifest, all tests PASS.

- [ ] **Step 6: Commit the renderer and outputs**

```bash
git add deliverables/defense_2026/sources/generate_visual_assets.py \
  deliverables/defense_2026/visuals tests/test_defense_visual_assets.py
git commit -m "feat: generate defense algorithm visuals"
```

---

### Task 3: Apply The Public Algorithm Names Consistently

**Files:**
- Modify: `paper/main_v2.tex`
- Modify: `thesis/chapters/01-introduction.tex`
- Modify: `thesis/chapters/04-interference-aware-pairing.tex`
- Modify: `thesis/chapters/05-deadline-allocation.tex`
- Modify: `thesis/chapters/06-implementation-experiments.tex`
- Modify: `thesis/chapters/08-conclusion.tex`
- Modify: `defense/innovation-comparison.md`
- Modify: `defense/outline.md`
- Modify: `deliverables/defense_2026/sources/中期报告_内容草稿.md`
- Modify: `deliverables/defense_2026/sources/中期报告_源稿.md`
- Modify: `uav_mec/generate_figures_v2.py`
- Modify: `tests/test_figure_generation_v2.py`

- [ ] **Step 1: Add failing public-name tests**

Require active paper/thesis/defense prose to contain `EAD-JSRA`, `IEA-CAP`, and `DD-MCAA`. Generate figures into `tmp_path` and assert the main plot contains the EAD-JSRA display label in its serialized metadata or accompanying manifest; prohibit presentation labels `Full` and `Full method` while allowing the internal JSON key `full`.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
.venv-lock/bin/python -m pytest tests/test_active_thesis_claims.py \
  tests/test_figure_generation_v2.py -q
```

Expected: FAIL on missing acronyms and legacy display labels.

- [ ] **Step 3: Update public prose and plot display mapping**

Introduce each full name once, then use its acronym consistently. In `generate_figures_v2.py`, centralize display labels so canonical key `full` renders as `EAD-JSRA`; keep all input parsing and statistics keyed by `full`.

- [ ] **Step 4: Regenerate paper figures and reports**

Run:

```bash
make assets PYTHON=.venv-lock/bin/python
node deliverables/defense_2026/sources/build_reports.js
```

- [ ] **Step 5: Verify public-name tests and document builds**

Run:

```bash
.venv-lock/bin/python -m pytest tests/test_active_thesis_claims.py \
  tests/test_figure_generation_v2.py tests/test_latex_citations.py -q
make paper thesis PYTHON=.venv-lock/bin/python
```

Expected: PASS and both PDFs compile.

- [ ] **Step 6: Commit naming integration**

```bash
git add paper thesis defense deliverables/defense_2026/sources \
  figures/thesis uav_mec/generate_figures_v2.py tests
git commit -m "docs: name the EAD-JSRA algorithm family"
```

---

### Task 4: Rebuild The 17-Slide Midterm Defense Deck

**Files:**
- Modify: `deliverables/defense_2026/sources/build_slides.js`
- Modify: `deliverables/defense_2026/sources/check_slides.js`
- Modify: `deliverables/defense_2026/sources/中期答辩_逐页大纲.md`
- Modify output: `deliverables/defense_2026/outputs/中期答辩_关铭锡.pptx`

- [ ] **Step 1: Make the slide checker fail on the old order**

Extend `check_slides.js` to assert 17 pages and require ordered slide-title fragments for background, EAD-JSRA framework, progress, IEA-CAP, strict 2×2, DD-MCAA, main result, gates, reproducibility, risk, and summary. Require all three acronyms and reject `Full method`, stale test counts, placeholder tokens, and claims of global optimality.

- [ ] **Step 2: Run the checker and verify RED**

Run:

```bash
node deliverables/defense_2026/sources/check_slides.js
```

Expected: FAIL because the existing deck lacks the new asset order and acronyms.

- [ ] **Step 3: Add an asset-slide renderer and rebuild specs**

Add a `visualAsset` slide kind that places the 16:9 PNG without distortion, adds page/section metadata without covering the figure, and attaches at least three Chinese speaker-note sentences. Rebuild the midterm specs in the approved 17-page order. Remove the redundant metric-card page and merge its evidence into the main-result and gate slides.

- [ ] **Step 4: Rebuild the deck and verify GREEN**

Run:

```bash
node deliverables/defense_2026/sources/build_slides.js
node deliverables/defense_2026/sources/check_slides.js
unzip -t deliverables/defense_2026/outputs/中期答辩_关铭锡.pptx
```

Expected: 17 slides, 17 note pages, no stale content, valid OOXML.

- [ ] **Step 5: Update the page-by-page outline**

Rewrite `中期答辩_逐页大纲.md` to match the actual 17-page order, the three algorithm names, exact evidence, and an 11-minute speaking allocation.

- [ ] **Step 6: Commit the deck**

```bash
git add deliverables/defense_2026/sources deliverables/defense_2026/outputs/中期答辩_关铭锡.pptx
git commit -m "docs: rebuild midterm deck around algorithm visuals"
```

---

### Task 5: Perform Visual QA And Final Verification

**Files:**
- Modify: `deliverables/defense_2026/QA_REPORT.md`
- Modify: `deliverables/defense_2026/README.md`

- [ ] **Step 1: Create contact sheets for all six assets**

Use Pillow to combine the generated PNGs into two labeled contact sheets under ignored `deliverables/defense_2026/previews/rendered_visuals/`. Inspect original-resolution assets for overflow, unreadable text, ambiguous arrows, inconsistent spacing, and low contrast.

- [ ] **Step 2: Fix user-visible defects once and regenerate**

Apply only user-visible fixes, regenerate all assets and the deck, and recheck the affected original-resolution images. Do not iterate on sub-pixel cosmetic differences.

- [ ] **Step 3: Update QA documentation**

Record asset dimensions, file list, naming contract, slide count, notes count, content checks, visual inspection results, and the known local Impress-rendering limitation. Document that PNG/PDF assets were visually inspected independently even if the system cannot render PPTX.

- [ ] **Step 4: Run the full verification gate**

Run:

```bash
.venv-lock/bin/python -m pytest -q
make verify PYTHON=.venv-lock/bin/python
node deliverables/defense_2026/sources/build_reports.js
node deliverables/defense_2026/sources/build_slides.js
node deliverables/defense_2026/sources/check_slides.js
git diff --check
```

Expected: all Python tests pass, paper/thesis compile, reports and 17-slide deck regenerate, slide checker passes, and no whitespace errors remain.

- [ ] **Step 5: Commit the verified package**

```bash
git add deliverables/defense_2026 README.md paper thesis figures tests uav_mec
git commit -m "docs: finalize visual midterm defense package"
```
