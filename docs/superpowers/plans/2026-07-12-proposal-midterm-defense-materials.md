# Proposal And Midterm Defense Materials Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce editable, defense-ready proposal and midterm DOCX/PPTX materials whose experimental sections are wired to explicit placeholders for the new canonical experiment data.

**Architecture:** A single structured content layer defines student metadata, report sections, slide narratives, verified current progress, and data placeholders. JavaScript generators create Markdown, DOCX, and PPTX outputs from that layer; a placeholder manifest makes later canonical-data replacement deterministic. Validation combines Office file checks, extracted-text checks, PDF rendering, and independent visual review.

**Tech Stack:** Node.js, `docx`, PptxGenJS, react-icons/sharp where available, JSON, Markdown, LibreOffice, Poppler, the local DOCX/PPTX skill scripts, and standard shell validation tools.

---

## File Map

- `deliverables/defense_2026/README.md`: delivery index, evidence boundary, rebuild commands, and placeholder replacement workflow.
- `deliverables/defense_2026/data_placeholders.json`: machine-readable new-experiment placeholder registry.
- `deliverables/defense_2026/sources/materials_content.js`: shared metadata, verified progress facts, report section content, references, and slide narratives.
- `deliverables/defense_2026/sources/build_reports.js`: Markdown and DOCX generator.
- `deliverables/defense_2026/sources/build_slides.js`: proposal and midterm PPTX generator.
- `deliverables/defense_2026/sources/开题报告_源稿.md`: generated editable proposal source.
- `deliverables/defense_2026/sources/中期报告_源稿.md`: generated editable midterm source.
- `deliverables/defense_2026/outputs/开题报告_关铭锡.docx`: proposal report.
- `deliverables/defense_2026/outputs/开题答辩_关铭锡.pptx`: proposal defense deck.
- `deliverables/defense_2026/outputs/中期报告_关铭锡.docx`: midterm report.
- `deliverables/defense_2026/outputs/中期答辩_关铭锡.pptx`: midterm defense deck.
- `deliverables/defense_2026/previews/`: rendered PDFs and slide/page images used for QA.

---

### Task 1: Create The Evidence And Placeholder Contract

**Files:**
- Create: `deliverables/defense_2026/README.md`
- Create: `deliverables/defense_2026/data_placeholders.json`

- [ ] **Step 1: Create the delivery directories**

Run:

```bash
mkdir -p deliverables/defense_2026/{sources,outputs,previews}
```

Expected: the three directories exist and contain no generated artifacts yet.

- [ ] **Step 2: Define the placeholder registry**

Create JSON entries with this exact schema:

```json
{
  "schema_version": 1,
  "evidence_policy": "Only new canonical experiment bundles may replace these values.",
  "placeholders": [
    {
      "key": "MAIN_ONTIME_OURS_PCT",
      "label_zh": "本文方法按时完成率",
      "unit": "%",
      "source": "canonical/main/summary.json",
      "status": "awaiting_new_canonical_experiment",
      "used_in": ["开题报告", "开题答辩", "中期报告", "中期答辩"]
    }
  ]
}
```

Include keys for `MAIN_ONTIME_OURS_PCT`, `MAIN_ONTIME_GREEDY_PCT`, `MAIN_ONTIME_OFDMA_PCT`, `MAIN_CI95_PP`, `MAIN_RUNTIME_S`, `ABLATION_GLOBAL_ON_PCT`, `ABLATION_CROSSUAV_ON_PCT`, `SIC_SENSITIVITY_SERIES`, `UAV_SCALING_SERIES`, `USER_SCALING_SERIES`, `CSI_ROBUSTNESS_SERIES`, `EXACT_GAP_MEAN_PCT`, `EXACT_GAP_MAX_PCT`, `CANONICAL_COMMIT`, `CANONICAL_RUN_DATE`, and `CANONICAL_MANIFEST_SHA256`.

- [ ] **Step 3: Write the delivery README**

Document the four outputs, the verified-progress boundary (`121 passed` at `fd0cb7c`), the prohibition on legacy results, the placeholder syntax `{{KEY}}`, and these rebuild commands:

```bash
node deliverables/defense_2026/sources/build_reports.js
node deliverables/defense_2026/sources/build_slides.js
```

- [ ] **Step 4: Validate the JSON and placeholder uniqueness**

Run:

```bash
python -m json.tool deliverables/defense_2026/data_placeholders.json >/dev/null
python -c "import json; p=json.load(open('deliverables/defense_2026/data_placeholders.json')); k=[x['key'] for x in p['placeholders']]; assert len(k)==len(set(k)) and len(k)>=15"
```

Expected: both commands exit 0.

---

### Task 2: Author The Shared Academic Content

**Files:**
- Create: `deliverables/defense_2026/sources/materials_content.js`

- [ ] **Step 1: Define metadata and truth labels**

Use this metadata:

```javascript
const metadata = {
  student: "关铭锡",
  advisor: "曲桦",
  university: "【学校名称】",
  college: "【学院名称】",
  major: "【专业名称】",
  studentId: "【学号】",
  date: "【提交日期】",
  title: "面向全频复用多无人机NOMA-MEC网络的干扰感知用户配对与截止时间资源分配研究",
};
```

Define status labels exactly as `已完成并验证`, `进行中`, `设计完成待实现`, and `待新canonical实验`.

- [ ] **Step 2: Define the proposal report content**

Provide full Chinese prose for: abstract, background and significance, domestic/international research status, audited research gap, system assumptions, mathematical problem, research objectives, innovation 1, innovation 2, joint workflow, experiment matrix, statistics, schedule, risks, expected outputs, and references. Do not assert a performance advantage; experimental result paragraphs must refer to named `{{KEY}}` placeholders.

- [ ] **Step 3: Define the midterm report content**

Provide full Chinese prose for: original route, evidence-driven pivot, ten credibility problems, current trusted architecture, completed Task 1-5 outputs, physics corrections, analytical resource derivation, maximum-cardinality proof, `121 passed` engineering evidence, innovation progress, new-experiment placeholders, remaining Tasks 6-25, risks, recovery actions, and graduation schedule.

- [ ] **Step 4: Define references and defense language**

Use references already present in `paper/main_v2.tex`, `docs/参考文献2.0.md`, and the approved design spec. Add a short `speakerNotes` field per slide that distinguishes verified facts from planned work.

- [ ] **Step 5: Scan the content for disallowed legacy claims**

Run:

```bash
rg -n "73\.4|82\.5|68\.4|DC-PPO.*%|旧结果.*证明" deliverables/defense_2026/sources/materials_content.js
```

Expected: no matches that present legacy percentages as evidence for the new method.

---

### Task 3: Generate The Two Reports

**Files:**
- Create: `deliverables/defense_2026/sources/build_reports.js`
- Generate: `deliverables/defense_2026/sources/开题报告_源稿.md`
- Generate: `deliverables/defense_2026/sources/中期报告_源稿.md`
- Generate: `deliverables/defense_2026/outputs/开题报告_关铭锡.docx`
- Generate: `deliverables/defense_2026/outputs/中期报告_关铭锡.docx`

- [ ] **Step 1: Implement shared DOCX styles**

Use A4 portrait dimensions, 25.4 mm margins, page-number footer, black 16 pt Cambria/compatible title, 12 pt body, 1.5 line spacing, numbered headings with outline levels, full-width DXA tables with explicit cell widths, and a cover page with the five editable metadata placeholders.

- [ ] **Step 2: Implement structured blocks**

Support paragraphs, numbered/bulleted lists, equations as centered Cambria text, data-placeholder callouts with light gray shading, tables, page breaks, references, and a generated table of contents field.

- [ ] **Step 3: Generate Markdown and DOCX from the same content**

Run:

```bash
node deliverables/defense_2026/sources/build_reports.js
```

Expected: both Markdown files and both DOCX files exist and are non-empty.

- [ ] **Step 4: Validate and extract report text**

Run:

```bash
python /home/link/my_workspace/.claude/skills/docx/scripts/office/validate.py deliverables/defense_2026/outputs/开题报告_关铭锡.docx
python /home/link/my_workspace/.claude/skills/docx/scripts/office/validate.py deliverables/defense_2026/outputs/中期报告_关铭锡.docx
extract-text deliverables/defense_2026/outputs/开题报告_关铭锡.docx > /tmp/proposal-report.txt
extract-text deliverables/defense_2026/outputs/中期报告_关铭锡.docx > /tmp/midterm-report.txt
```

Expected: validation succeeds; extracted text contains the title, both innovation headings, and data-placeholder notice.

---

### Task 4: Generate The Proposal Defense Deck

**Files:**
- Create: `deliverables/defense_2026/sources/build_slides.js`
- Generate: `deliverables/defense_2026/outputs/开题答辩_关铭锡.pptx`

- [ ] **Step 1: Implement the shared slide system**

Use 16:9, Arial/Calibri/Cambria, white content slides, dark teal title/conclusion slides, dark red only for constraints and risks, 0.5-inch margins, no decorative edge bars, and reusable helpers for title, footer, rounded cards, status chips, native tables, process arrows, timelines, and placeholder charts.

- [ ] **Step 2: Build the 14-slide proposal narrative**

Create: cover; application scenario; core engineering conflict; literature gap; research question and assumptions; mathematical model; overall workflow; innovation 1; innovation 2; joint algorithm; experiment matrix; canonical-data placeholder dashboard; schedule/risk; conclusion. Every slide must contain at least one visual structure.

- [ ] **Step 3: Add data placeholders without fabricated marks**

Placeholder charts must show labeled empty axes or neutral frames containing `待新版本 canonical 实验填充` and the exact `{{KEY}}`; do not draw arbitrary bars, lines, confidence intervals, or percentages.

- [ ] **Step 4: Add speaker notes**

Each slide receives 2-4 concise Chinese speaker-note sentences, including explicit wording for planned work and unfilled data.

---

### Task 5: Generate The Midterm Defense Deck

**Files:**
- Modify: `deliverables/defense_2026/sources/build_slides.js`
- Generate: `deliverables/defense_2026/outputs/中期答辩_关铭锡.pptx`

- [ ] **Step 1: Build the 16-slide midterm narrative**

Create: cover; thesis objective; route evolution; why trajectory/DRL was pivoted; credibility audit; trusted data flow; current progress; physics-layer correction; innovation 2 derivation; maximum-cardinality proof; innovation 1 implementation roadmap; test/reproducibility evidence; main/ablation data placeholders; robustness/optimality placeholders; remaining schedule and risk closure; conclusion.

- [ ] **Step 2: Present progress accurately**

Show Tasks 1-5 as completed as of `fd0cb7c`, Tasks 6-23 as engineering/research work remaining, Task 24 as experiments, and Task 25 as final thesis/defense packaging. Label `121 passed` as software verification, not algorithm performance.

- [ ] **Step 3: Generate both PPTX files**

Run:

```bash
node deliverables/defense_2026/sources/build_slides.js
```

Expected: both PPTX files exist and each has the specified slide count.

---

### Task 6: Perform Content And Visual QA

**Files:**
- Generate: `deliverables/defense_2026/previews/*.pdf`
- Generate: `deliverables/defense_2026/previews/proposal-slide-*.jpg`
- Generate: `deliverables/defense_2026/previews/midterm-slide-*.jpg`
- Create: `deliverables/defense_2026/QA_REPORT.md`

- [ ] **Step 1: Extract and scan PPT content**

Run:

```bash
extract-text deliverables/defense_2026/outputs/开题答辩_关铭锡.pptx > /tmp/proposal-slides.txt
extract-text deliverables/defense_2026/outputs/中期答辩_关铭锡.pptx > /tmp/midterm-slides.txt
rg -ni "lorem|ipsum|T[O]DO|\[insert|xxx" /tmp/proposal-slides.txt /tmp/midterm-slides.txt
```

Expected: no accidental template placeholders; deliberate Chinese metadata/data placeholders remain.

- [ ] **Step 2: Convert all Office outputs to PDF**

Use the local `scripts/office/soffice.py` helpers for DOCX and PPTX. Move the resulting PDFs into `deliverables/defense_2026/previews/` with stable English names.

- [ ] **Step 3: Render all PPT pages to JPEG**

Run `pdftoppm -jpeg -r 150` for both deck PDFs with `proposal-slide` and `midterm-slide` prefixes.

- [ ] **Step 4: Dispatch independent visual review**

Ask a fresh subagent to inspect every slide for overflow, overlap, poor contrast, cramped gaps, alignment errors, and unintended placeholder artifacts. Record findings by slide number in `QA_REPORT.md`.

- [ ] **Step 5: Apply one visible-defect correction pass**

Fix all user-visible issues, regenerate affected files, and rerender both complete decks. Stop after the second inspection unless a new overflow, overlap, missing-content, or corruption defect appears.

- [ ] **Step 6: Verify final files and placeholder coverage**

Run:

```bash
find deliverables/defense_2026/outputs -maxdepth 1 -type f -size +10k -print
rg -o "\{\{[A-Z0-9_]+\}\}" deliverables/defense_2026/sources/*.md | sort -u
git diff --check
```

Expected: four non-empty Office outputs, all registered experiment placeholders represented, and a clean diff check.

---

### Task 7: Final Delivery Review

**Files:**
- Modify: `deliverables/defense_2026/README.md`
- Modify: `deliverables/defense_2026/QA_REPORT.md`

- [ ] **Step 1: Cross-check evidence labels**

Confirm every performance slide/table is either a registered placeholder or explicitly labeled historical context; confirm no old percentage is presented as support for the new method.

- [ ] **Step 2: Cross-check title and innovation consistency**

Confirm the same thesis title and the same two innovation statements appear in all four outputs.

- [ ] **Step 3: Document replacement instructions**

Explain how Task 24 canonical artifacts map to placeholder keys and which generator commands rebuild the final defense materials.

- [ ] **Step 4: Commit the deliverables**

```bash
git add deliverables/defense_2026 docs/superpowers/plans/2026-07-12-proposal-midterm-defense-materials.md
git commit -m "docs: add proposal and midterm defense materials"
```
