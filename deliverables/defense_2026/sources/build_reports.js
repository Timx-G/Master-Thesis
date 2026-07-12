#!/usr/bin/env node

"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const {
  AlignmentType,
  BorderStyle,
  Document,
  Footer,
  Header,
  HeadingLevel,
  LevelFormat,
  LineRuleType,
  PageBreak,
  PageNumber,
  Packer,
  Paragraph,
  SectionType,
  ShadingType,
  Table,
  TableCell,
  TableLayoutType,
  TableRow,
  TextRun,
  VerticalAlign,
  WidthType,
} = require("docx");

const A4_WIDTH_DXA = 11906;
const A4_HEIGHT_DXA = 16838;
const PAGE_MARGIN_DXA = 1440;
const CONTENT_WIDTH_DXA = A4_WIDTH_DXA - 2 * PAGE_MARGIN_DXA;
const BODY_FONT = {
  ascii: "Noto Serif CJK SC",
  hAnsi: "Noto Serif CJK SC",
  eastAsia: "Noto Serif CJK SC",
};
const HEADING_FONT = {
  ascii: "Noto Sans CJK SC",
  hAnsi: "Noto Sans CJK SC",
  eastAsia: "Noto Sans CJK SC",
};
const CODE_FONT = {
  ascii: "Consolas",
  hAnsi: "Consolas",
  eastAsia: "Noto Sans CJK SC",
};
const COLORS = {
  teal: "0F5B5D",
  tealDark: "173F42",
  tealLight: "DDEDEE",
  gray: "F2F4F5",
  grayBorder: "AAB4B7",
  text: "202426",
  muted: "5A6668",
  risk: "8F2633",
  placeholder: "FFF2CC",
  white: "FFFFFF",
};

const BASE_DIR = path.resolve(__dirname, "..");
const OUTPUT_DIR = path.join(BASE_DIR, "outputs");

const REPORTS = [
  {
    kind: "proposal",
    draft: path.join(__dirname, "开题报告_内容草稿.md"),
    source: path.join(__dirname, "开题报告_源稿.md"),
    output: path.join(OUTPUT_DIR, "开题报告_关铭锡.docx"),
    reportName: "开题报告",
    headerText: "专业硕士学位论文开题报告",
  },
  {
    kind: "midterm",
    draft: path.join(__dirname, "中期报告_内容草稿.md"),
    source: path.join(__dirname, "中期报告_源稿.md"),
    output: path.join(OUTPUT_DIR, "中期报告_关铭锡.docx"),
    reportName: "中期报告",
    headerText: "专业硕士学位论文中期报告",
  },
];

const page = {
  size: { width: A4_WIDTH_DXA, height: A4_HEIGHT_DXA },
  margin: {
    top: PAGE_MARGIN_DXA,
    right: PAGE_MARGIN_DXA,
    bottom: PAGE_MARGIN_DXA,
    left: PAGE_MARGIN_DXA,
    header: 720,
    footer: 720,
    gutter: 0,
  },
};

function splitTableRow(line) {
  const trimmed = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  return trimmed.split("|").map((cell) => cell.trim());
}

function isTableSeparator(line) {
  const cells = splitTableRow(line);
  return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function extractCover(markdown, report) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const firstH1 = lines.findIndex((line) => /^#\s+/.test(line));
  const firstH2 = lines.findIndex((line, index) => index > firstH1 && /^##\s+/.test(line));
  const tableStart = lines.findIndex(
    (line, index) =>
      index > firstH2 && /^\s*\|/.test(line) && isTableSeparator(lines[index + 1] || ""),
  );

  if (firstH1 < 0 || firstH2 < 0 || tableStart < 0) {
    throw new Error(`${report.reportName}草稿缺少标准封面标题或元数据表`);
  }

  let tableEnd = tableStart + 2;
  while (tableEnd < lines.length && /^\s*\|/.test(lines[tableEnd])) {
    tableEnd += 1;
  }

  const metadata = {};
  for (let index = tableStart + 2; index < tableEnd; index += 1) {
    const cells = splitTableRow(lines[index]);
    if (cells.length >= 2) metadata[cells[0]] = cells.slice(1).join(" | ");
  }

  const h1Text = lines[firstH1].replace(/^#\s+/, "").trim();
  const h2Text = lines[firstH2].replace(/^##\s+/, "").trim();
  const h1IsReportLabel = /报告/.test(h1Text);
  const h2IsReportLabel = /报告/.test(h2Text);
  const coverHeading = h1IsReportLabel && !h2IsReportLabel ? h1Text : h2Text;
  const thesisTitle = h1IsReportLabel && !h2IsReportLabel ? h2Text : h1Text;
  const bodyMarkdown = lines.slice(tableEnd).join("\n").replace(/^\s+/, "");

  return { coverHeading, thesisTitle, metadata, bodyMarkdown };
}

function readBracedGroup(text, start) {
  if (text[start] !== "{") return null;
  let depth = 0;
  for (let index = start; index < text.length; index += 1) {
    if (text[index] === "{") depth += 1;
    if (text[index] === "}") depth -= 1;
    if (depth === 0) {
      return { value: text.slice(start + 1, index), end: index + 1 };
    }
  }
  return null;
}

function replaceLatexCommand(text, command, argumentCount, formatter) {
  let output = "";
  let cursor = 0;
  const marker = `\\${command}`;
  while (cursor < text.length) {
    const found = text.indexOf(marker, cursor);
    if (found < 0) {
      output += text.slice(cursor);
      break;
    }
    output += text.slice(cursor, found);
    let position = found + marker.length;
    while (/\s/.test(text[position] || "")) position += 1;
    const groups = [];
    let valid = true;
    for (let count = 0; count < argumentCount; count += 1) {
      const group = readBracedGroup(text, position);
      if (!group) {
        valid = false;
        break;
      }
      groups.push(group.value);
      position = group.end;
      while (/\s/.test(text[position] || "")) position += 1;
    }
    if (!valid) {
      output += marker;
      cursor = found + marker.length;
      continue;
    }
    output += formatter(groups);
    cursor = position;
  }
  return output;
}

function latexToReadableText(value) {
  let text = value;
  for (let pass = 0; pass < 6 && /\\frac\s*\{/.test(text); pass += 1) {
    text = replaceLatexCommand(text, "frac", 2, ([numerator, denominator]) =>
      `(${latexToReadableText(numerator)})/(${latexToReadableText(denominator)})`,
    );
  }
  text = replaceLatexCommand(text, "sqrt", 1, ([radicand]) => `√(${latexToReadableText(radicand)})`);
  for (const command of ["mathrm", "mathcal", "mathbf", "boldsymbol", "text"]) {
    text = replaceLatexCommand(text, command, 1, ([content]) => latexToReadableText(content));
  }
  text = text
    .replace(/\\bar\s*\{?([A-Za-z])\}?/g, "$1̄")
    .replace(/\\alpha/g, "α")
    .replace(/\\rho/g, "ρ")
    .replace(/\\gamma/g, "γ")
    .replace(/\\sigma/g, "σ")
    .replace(/\\tau/g, "τ")
    .replace(/\\eta/g, "η")
    .replace(/\\varepsilon/g, "ε")
    .replace(/\\pi/g, "π")
    .replace(/\\sum/g, "∑")
    .replace(/\\le/g, "≤")
    .replace(/\\ge/g, "≥")
    .replace(/\\neq/g, "≠")
    .replace(/\\in/g, "∈")
    .replace(/\\times/g, "×")
    .replace(/\\cdot/g, "·")
    .replace(/\\ldots/g, "…")
    .replace(/\\cdots/g, "⋯")
    .replace(/\\(?:left|right)/g, "")
    .replace(/\\(?:qquad|quad|;|,|!)/g, " ")
    .replace(/\\max/g, "max")
    .replace(/\\min/g, "min")
    .replace(/\\log/g, "log")
    .replace(/\\([{}|])/g, "$1")
    .replace(/\\\(|\\\)/g, "")
    .replace(/\\(?:mathrm|mathcal|mathbf|boldsymbol)\s*/g, "")
    .replace(/\\([A-Za-z]+)/g, "$1")
    .replace(/\^\{([^{}]+)\}/g, "^($1)")
    .replace(/_\{([^{}]+)\}/g, "_($1)")
    .replace(/\\\\/g, " ")
    .replace(/\s{2,}/g, " ");
  return text;
}

function parseInline(text, base = {}) {
  const tokenPattern = /(\{\{[A-Z0-9_]+\}\}|【[^】]+】|\*\*[^*]+\*\*|`[^`]+`|\*[^*\n]+\*|\[[^\]]+\]\([^)]+\))/g;
  const runs = [];
  let cursor = 0;
  let match;

  const append = (value, extra = {}, normalizeMath = true) => {
    if (!value) return;
    runs.push(
      new TextRun({
        text: normalizeMath ? latexToReadableText(value) : value,
        font: BODY_FONT,
        color: COLORS.text,
        ...base,
        ...extra,
      }),
    );
  };

  while ((match = tokenPattern.exec(text)) !== null) {
    append(text.slice(cursor, match.index));
    const token = match[0];
    if (/^\{\{/.test(token) || /^【/.test(token)) {
      append(token, {
        bold: true,
        color: COLORS.risk,
        shading: { fill: COLORS.placeholder, type: ShadingType.CLEAR },
      });
    } else if (token.startsWith("**")) {
      append(token.slice(2, -2), { bold: true });
    } else if (token.startsWith("`")) {
      append(
        token.slice(1, -1),
        {
          font: CODE_FONT,
          color: COLORS.tealDark,
          shading: { fill: "E8EEEE", type: ShadingType.CLEAR },
        },
        false,
      );
    } else if (token.startsWith("*")) {
      append(token.slice(1, -1), { italics: true });
    } else {
      const linkMatch = token.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      append(linkMatch ? `${linkMatch[1]}（${linkMatch[2]}）` : token, {
        color: COLORS.teal,
        underline: { color: COLORS.teal },
      });
    }
    cursor = match.index + token.length;
  }
  append(text.slice(cursor));
  return runs.length ? runs : [new TextRun({ text: "", font: BODY_FONT })];
}

function isDisplayMathStart(line) {
  const trimmed = line.trim();
  return trimmed === "\\[" || trimmed === "$$";
}

function parseMarkdown(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const blocks = [];
  let index = 0;
  let listInstance = 0;

  const isSpecial = (line, nextLine = "") =>
    /^#{1,6}\s+/.test(line) ||
    /^\s*```/.test(line) ||
    isDisplayMathStart(line) ||
    /^\s*>/.test(line) ||
    /^\s*([-+*]|\d+[.)])\s+/.test(line) ||
    /^\s*---\s*$/.test(line) ||
    (/^\s*\|/.test(line) && isTableSeparator(nextLine));

  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      blocks.push({ type: "heading", level: heading[1].length, text: heading[2].trim() });
      index += 1;
      continue;
    }

    if (/^\s*---\s*$/.test(line)) {
      blocks.push({ type: "pageBreak" });
      index += 1;
      continue;
    }

    if (/^\s*```/.test(line)) {
      const language = line.trim().slice(3).trim();
      const codeLines = [];
      index += 1;
      while (index < lines.length && !/^\s*```/.test(lines[index])) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index >= lines.length) throw new Error("Markdown 代码块未闭合");
      index += 1;
      blocks.push({ type: "code", language, lines: codeLines });
      continue;
    }

    if (isDisplayMathStart(line)) {
      const closing = line.trim() === "\\[" ? "\\]" : "$$";
      const equationLines = [];
      index += 1;
      while (index < lines.length && lines[index].trim() !== closing) {
        equationLines.push(lines[index].trim());
        index += 1;
      }
      if (index >= lines.length) throw new Error("Markdown 公式块未闭合");
      index += 1;
      blocks.push({ type: "equation", text: equationLines.join(" ") });
      continue;
    }

    if (/^\s*>/.test(line)) {
      const quoteLines = [];
      while (index < lines.length && /^\s*>/.test(lines[index])) {
        quoteLines.push(lines[index].replace(/^\s*>\s?/, "").trim());
        index += 1;
      }
      blocks.push({ type: "callout", text: quoteLines.join(" ") });
      continue;
    }

    if (/^\s*\|/.test(line) && isTableSeparator(lines[index + 1] || "")) {
      const header = splitTableRow(line);
      index += 2;
      const rows = [];
      while (index < lines.length && /^\s*\|/.test(lines[index])) {
        rows.push(splitTableRow(lines[index]));
        index += 1;
      }
      blocks.push({ type: "table", header, rows });
      continue;
    }

    const listMatch = line.match(/^(\s*)([-+*]|\d+[.)])\s+(.+)$/);
    if (listMatch) {
      const ordered = /^\d/.test(listMatch[2]);
      const items = [];
      listInstance += 1;
      while (index < lines.length) {
        const item = lines[index].match(/^(\s*)([-+*]|\d+[.)])\s+(.+)$/);
        if (!item || /^\d/.test(item[2]) !== ordered) break;
        items.push({
          level: Math.min(2, Math.floor(item[1].replace(/\t/g, "    ").length / 2)),
          text: item[3],
        });
        index += 1;
      }
      blocks.push({ type: "list", ordered, items, instance: listInstance });
      continue;
    }

    const paragraphLines = [line.trim()];
    index += 1;
    while (
      index < lines.length &&
      lines[index].trim() &&
      !isSpecial(lines[index], lines[index + 1] || "")
    ) {
      paragraphLines.push(lines[index].trim());
      index += 1;
    }
    blocks.push({ type: "paragraph", text: paragraphLines.join(" ") });
  }

  return blocks;
}

function tableColumnWidths(columnCount) {
  if (columnCount <= 0) throw new Error("表格至少需要一列");
  if (columnCount === 2) return [2250, CONTENT_WIDTH_DXA - 2250];
  const base = Math.floor(CONTENT_WIDTH_DXA / columnCount);
  const widths = Array(columnCount).fill(base);
  widths[widths.length - 1] += CONTENT_WIDTH_DXA - base * columnCount;
  return widths;
}

function makeTableCell(text, width, header = false) {
  const border = { style: BorderStyle.SINGLE, size: 4, color: COLORS.grayBorder };
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    shading: { fill: header ? COLORS.tealLight : COLORS.white, type: ShadingType.CLEAR },
    borders: { top: border, bottom: border, left: border, right: border },
    margins: { top: 70, bottom: 70, left: 100, right: 100 },
    children: [
      new Paragraph({
        spacing: { before: 0, after: 0, line: 240, lineRule: LineRuleType.AUTO },
        children: parseInline(text, { font: HEADING_FONT, size: 18, bold: header }),
      }),
    ],
  });
}

function makeTable(block) {
  const columnCount = block.header.length;
  const widths = tableColumnWidths(columnCount);
  const normalize = (row) => Array.from({ length: columnCount }, (_, index) => row[index] || "");
  const rows = [
    new TableRow({
      cantSplit: true,
      tableHeader: true,
      children: normalize(block.header).map((cell, index) => makeTableCell(cell, widths[index], true)),
    }),
    ...block.rows.map(
      (row) =>
        new TableRow({
          cantSplit: true,
          children: normalize(row).map((cell, index) => makeTableCell(cell, widths[index], false)),
        }),
    ),
  ];

  return new Table({
    width: { size: CONTENT_WIDTH_DXA, type: WidthType.DXA },
    columnWidths: widths,
    layout: TableLayoutType.FIXED,
    rows,
  });
}

function headingLevel(markdownLevel) {
  if (markdownLevel <= 2) return HeadingLevel.HEADING_1;
  if (markdownLevel === 3) return HeadingLevel.HEADING_2;
  return HeadingLevel.HEADING_3;
}

function blocksToDocx(blocks, layout) {
  const children = [];
  for (const block of blocks) {
    if (block.type === "heading") {
      children.push(
        new Paragraph({
          heading: headingLevel(block.level),
          pageBreakBefore: /^参考文献\s*$/.test(block.text),
          keepNext: true,
          children: parseInline(block.text),
        }),
      );
    } else if (block.type === "pageBreak") {
      children.push(new Paragraph({ children: [new PageBreak()] }));
    } else if (block.type === "table") {
      children.push(makeTable(block));
      children.push(new Paragraph({ spacing: { after: 80 }, children: [] }));
    } else if (block.type === "callout") {
      children.push(
        new Paragraph({
          border: { left: { style: BorderStyle.SINGLE, size: 18, color: COLORS.teal, space: 8 } },
          shading: { fill: COLORS.gray, type: ShadingType.CLEAR },
          indent: { left: 260, right: 180 },
          spacing: {
            before: layout.blockSpace,
            after: layout.blockSpace,
            line: layout.bodyLine,
            lineRule: LineRuleType.AUTO,
          },
          children: parseInline(block.text),
        }),
      );
    } else if (block.type === "equation") {
      children.push(
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: {
            before: layout.blockSpace,
            after: layout.blockSpace,
            line: layout.bodyLine,
            lineRule: LineRuleType.AUTO,
          },
          children: [
            new TextRun({
              text: latexToReadableText(block.text),
              font: BODY_FONT,
              size: layout.bodySize,
              italics: true,
            }),
          ],
        }),
      );
    } else if (block.type === "code") {
      for (const [lineIndex, line] of block.lines.entries()) {
        children.push(
          new Paragraph({
            shading: { fill: "EEF2F2", type: ShadingType.CLEAR },
            border: { left: { style: BorderStyle.SINGLE, size: 8, color: COLORS.grayBorder, space: 4 } },
            indent: { left: 160, right: 160 },
            spacing: {
              before: lineIndex === 0 ? 100 : 0,
              after: lineIndex === block.lines.length - 1 ? 100 : 0,
              line: 240,
            },
            children: [new TextRun({ text: line || " ", font: CODE_FONT, size: 18, color: COLORS.tealDark })],
          }),
        );
      }
    } else if (block.type === "list") {
      for (const item of block.items) {
        children.push(
          new Paragraph({
            numbering: {
              reference: block.ordered ? "ordered-list" : "bullet-list",
              level: item.level,
              instance: block.instance,
            },
            spacing: {
              before: 20,
              after: 20,
              line: layout.bodyLine,
              lineRule: LineRuleType.AUTO,
            },
            children: parseInline(item.text),
          }),
        );
      }
    } else if (block.type === "paragraph") {
      const isReference = /^\[\d+\]/.test(block.text);
      const hasPlaceholder = /\{\{[A-Z0-9_]+\}\}/.test(block.text);
      children.push(
        new Paragraph({
          alignment: isReference ? AlignmentType.LEFT : AlignmentType.JUSTIFIED,
          indent: isReference ? { left: 420, hanging: 420 } : { firstLine: 480 },
          shading: hasPlaceholder ? { fill: COLORS.gray, type: ShadingType.CLEAR } : undefined,
          border: hasPlaceholder
            ? { left: { style: BorderStyle.SINGLE, size: 10, color: COLORS.risk, space: 5 } }
            : undefined,
          spacing: {
            before: 0,
            after: layout.paragraphAfter,
            line: layout.bodyLine,
            lineRule: LineRuleType.AUTO,
          },
          children: parseInline(block.text),
        }),
      );
    }
  }
  return children;
}

function metadataValue(metadata, keys, fallback) {
  for (const key of keys) {
    if (metadata[key]) return metadata[key];
  }
  return fallback;
}

function makeCoverTable(metadata) {
  const rows = [
    ["学校", metadataValue(metadata, ["学校"], "【学校名称】")],
    ["学院", metadataValue(metadata, ["学院"], "【学院名称】")],
    ["专业领域", metadataValue(metadata, ["专业领域", "专业"], "【专业名称】")],
    ["学生姓名", metadataValue(metadata, ["学生姓名", "姓名"], "关铭锡")],
    ["学号", metadataValue(metadata, ["学号"], "【学号】")],
    ["指导教师", metadataValue(metadata, ["指导教师", "导师"], "曲桦")],
    ["报告日期", metadataValue(metadata, ["报告日期", "提交日期"], "【提交日期】")],
  ];
  return makeTable({ header: ["项目", "内容"], rows });
}

function makeCover(cover, report) {
  const school = metadataValue(cover.metadata, ["学校"], "【学校名称】");
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 420, after: 700 },
      children: parseInline(school, { font: HEADING_FONT, size: 32, bold: true }),
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 420 },
      children: [
        new TextRun({ text: "专业硕士学位论文", font: HEADING_FONT, size: 38, bold: true, color: "000000" }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 800 },
      children: [
        new TextRun({ text: report.reportName, font: HEADING_FONT, size: 44, bold: true, color: "000000" }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 700, line: 420, lineRule: LineRuleType.AUTO },
      children: parseInline(cover.thesisTitle, {
        font: HEADING_FONT,
        size: 32,
        bold: true,
        color: "000000",
      }),
    }),
    makeCoverTable(cover.metadata),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 300 },
      children: [
        new TextRun({
          text: "封面信息为可编辑字段，请按培养单位模板核对后填写",
          font: HEADING_FONT,
          size: 18,
          color: COLORS.muted,
        }),
      ],
    }),
  ];
}

function makeHeader(text) {
  return new Header({
    children: [
      new Paragraph({
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: COLORS.teal, space: 2 } },
        alignment: AlignmentType.CENTER,
        spacing: { after: 60 },
        children: [new TextRun({ text, font: HEADING_FONT, size: 18, color: COLORS.muted })],
      }),
    ],
  });
}

function makeFooter() {
  return new Footer({
    children: [
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [
          new TextRun({ text: "第 ", font: BODY_FONT, size: 18, color: COLORS.muted }),
          new TextRun({ children: [PageNumber.CURRENT], font: BODY_FONT, size: 18, color: COLORS.muted }),
          new TextRun({ text: " 页", font: BODY_FONT, size: 18, color: COLORS.muted }),
        ],
      }),
    ],
  });
}

function documentStyles(layout) {
  return {
    default: {
      document: {
        run: { font: BODY_FONT, size: layout.bodySize, color: COLORS.text },
        paragraph: {
          spacing: {
            line: layout.bodyLine,
            lineRule: LineRuleType.AUTO,
            after: layout.paragraphAfter,
          },
        },
      },
    },
    paragraphStyles: [
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: {
          font: HEADING_FONT,
          size: layout.compact ? 30 : 32,
          bold: true,
          color: "000000",
        },
        paragraph: {
          spacing: {
            before: layout.compact ? 220 : 300,
            after: layout.compact ? 120 : 180,
          },
          outlineLevel: 0,
          keepNext: true,
        },
      },
      {
        id: "Heading2",
        name: "Heading 2",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: {
          font: HEADING_FONT,
          size: layout.compact ? 26 : 28,
          bold: true,
          color: "000000",
        },
        paragraph: {
          spacing: {
            before: layout.compact ? 170 : 240,
            after: layout.compact ? 90 : 140,
          },
          outlineLevel: 1,
          keepNext: true,
        },
      },
      {
        id: "Heading3",
        name: "Heading 3",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: {
          font: HEADING_FONT,
          size: layout.compact ? 22 : 24,
          bold: true,
          color: "000000",
        },
        paragraph: {
          spacing: {
            before: layout.compact ? 140 : 200,
            after: layout.compact ? 70 : 100,
          },
          outlineLevel: 2,
          keepNext: true,
        },
      },
      {
        id: "TOCTitle",
        name: "TOC Title",
        basedOn: "Normal",
        next: "Normal",
        run: { font: HEADING_FONT, size: 32, bold: true, color: "000000" },
        paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 100, after: 240 } },
      },
    ],
  };
}

function numberingConfig() {
  const orderedLevels = [];
  const bulletLevels = [];
  for (let level = 0; level < 3; level += 1) {
    const left = 720 + level * 420;
    orderedLevels.push({
      level,
      format: LevelFormat.DECIMAL,
      text: `%${level + 1}.`,
      alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left, hanging: 360 } } },
    });
    bulletLevels.push({
      level,
      format: LevelFormat.BULLET,
      text: level === 0 ? "•" : level === 1 ? "◦" : "▪",
      alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left, hanging: 360 } } },
    });
  }
  return {
    config: [
      { reference: "ordered-list", levels: orderedLevels },
      { reference: "bullet-list", levels: bulletLevels },
    ],
  };
}

function makeStaticToc(blocks) {
  const entries = blocks.filter((block) => block.type === "heading" && block.level <= 2);
  if (!entries.length) throw new Error("正文缺少可用于生成目录的章节标题");
  return entries.map(
    (entry) =>
      new Paragraph({
        indent: { left: 240, right: 240 },
        border: {
          bottom: { style: BorderStyle.DOTTED, size: 2, color: "CBD3D5", space: 2 },
        },
        spacing: { before: 20, after: 50, line: 260, lineRule: LineRuleType.AUTO },
        children: parseInline(entry.text, {
          font: HEADING_FONT,
          size: 20,
          color: COLORS.tealDark,
        }),
      }),
  );
}

function buildDocument(markdown, report) {
  const cover = extractCover(markdown, report);
  const bodyBlocks = parseMarkdown(cover.bodyMarkdown);
  const layout = report.kind === "midterm"
    ? { bodySize: 20, bodyLine: 280, paragraphAfter: 50, blockSpace: 60, compact: true }
    : { bodySize: 21, bodyLine: 300, paragraphAfter: 80, blockSpace: 90, compact: false };
  const bodyChildren = [
    new Paragraph({
      style: "TOCTitle",
      children: [new TextRun({ text: "目录", font: HEADING_FONT, bold: true })],
    }),
    ...makeStaticToc(bodyBlocks),
    new Paragraph({ children: [new PageBreak()] }),
    ...blocksToDocx(bodyBlocks, layout),
  ];

  const document = new Document({
    title: `${cover.thesisTitle}——${report.reportName}`,
    subject: "专业硕士学位论文答辩材料",
    creator: "关铭锡",
    lastModifiedBy: "Codex",
    description: "由同目录 Markdown 草稿可复现生成，实验性能字段保留 canonical 数据占位符。",
    keywords: "多无人机,NOMA,MEC,截止时间,资源分配,专业硕士",
    features: { updateFields: true },
    styles: documentStyles(layout),
    numbering: numberingConfig(),
    sections: [
      {
        properties: { page, type: SectionType.NEXT_PAGE },
        children: makeCover(cover, report),
      },
      {
        properties: { page: { ...page, pageNumbers: { start: 1 } }, type: SectionType.NEXT_PAGE },
        headers: { default: makeHeader(report.headerText) },
        footers: { default: makeFooter() },
        children: bodyChildren,
      },
    ],
  });

  return { document, cover, blockCount: bodyBlocks.length };
}

async function packReport(markdown, report) {
  const built = buildDocument(markdown, report);
  const buffer = await Packer.toBuffer(built.document);
  if (!Buffer.isBuffer(buffer) || buffer.length < 10_000) {
    throw new Error(`${report.reportName} DOCX 打包结果异常（${buffer.length} bytes）`);
  }
  return { ...built, buffer };
}

function writeAtomic(target, content) {
  fs.mkdirSync(path.dirname(target), { recursive: true });
  const temporary = `${target}.tmp-${process.pid}`;
  fs.writeFileSync(temporary, content);
  fs.renameSync(temporary, target);
}

async function runSelfTest() {
  const sample = `# 【学校名称】专业硕士学位论文开题报告

## 面向全频复用多无人机 NOMA-MEC 网络的研究

| 项目 | 内容 |
| --- | --- |
| 学校 | 【学校名称】 |
| 学院 | 【学院名称】 |
| 专业领域 | 【专业名称】 |
| 学生姓名 | 关铭锡 |
| 学号 | 【学号】 |
| 指导教师 | 曲桦 |
| 报告日期 | 【提交日期】 |

> **实验数据声明：** 待新版本 canonical 实验填充 {{SELF_TEST_METRIC}}。

## 摘要

这是用于验证生成器的临时内存样例，不会写入正式源稿。

## 1 研究内容

### 1.1 结构化块

1. 有序列表项。
2. 包含 \`code\` 和 **加粗文字**。

- 无序列表项。
- 第二个列表项。

\\[
\\max_{x} \\sum_i z_i
\\]

| 指标 | 状态 | 数值 |
| --- | --- | --- |
| 主实验 | 待填 | {{SELF_TEST_METRIC}} |

\`\`\`text
Schedule -> Rate -> Allocation
\`\`\`

## 参考文献

[1] 临时验证条目，仅用于生成器自检。
`;
  const report = REPORTS[0];
  const packed = await packReport(sample, report);
  const output =
    process.env.REPORT_SELFTEST_OUTPUT ||
    path.join(os.tmpdir(), "defense-report-generator-selftest.docx");
  writeAtomic(output, packed.buffer);
  console.log(`[self-test] ${output} (${packed.buffer.length} bytes, ${packed.blockCount} blocks)`);
}

async function generateReports() {
  const missing = REPORTS.filter((report) => !fs.existsSync(report.draft));
  if (missing.length) {
    console.log(
      `[pending] 报告生成器已就绪；尚缺：${missing.map((report) => path.basename(report.draft)).join("、")}`,
    );
    console.log("[pending] 未生成或覆盖任何正式源稿/DOCX。两份草稿齐备后重新运行本命令即可。");
    return;
  }

  const drafts = REPORTS.map((report) => ({
    report,
    markdown: fs.readFileSync(report.draft, "utf8"),
  }));
  for (const { report, markdown } of drafts) {
    if (markdown.trim().length < 2_000) {
      throw new Error(`${path.basename(report.draft)} 内容过短，拒绝生成正式报告`);
    }
  }

  const packed = await Promise.all(
    drafts.map(({ report, markdown }) => packReport(markdown, report)),
  );
  for (let index = 0; index < REPORTS.length; index += 1) {
    const { report, markdown } = drafts[index];
    writeAtomic(report.source, markdown);
    writeAtomic(report.output, packed[index].buffer);
    console.log(
      `[generated] ${path.basename(report.source)} + ${path.basename(report.output)} ` +
        `(${packed[index].buffer.length} bytes, ${packed[index].blockCount} blocks)`,
    );
  }
}

async function main() {
  if (process.argv.includes("--self-test")) {
    await runSelfTest();
    return;
  }
  await generateReports();
}

main().catch((error) => {
  console.error(`[error] ${error.stack || error.message}`);
  process.exitCode = 1;
});
