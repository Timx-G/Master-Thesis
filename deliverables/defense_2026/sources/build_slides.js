const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const JSZip = require("jszip");
const xmljs = require("xml-js");
const SHAPE = new pptxgen().ShapeType;

const W = 13.333;
const H = 7.5;
const FONT = "Noto Sans CJK SC";
const FORMULA_FONT = "Cambria";
const C = {
  teal: "0F5B5D",
  tealDark: "173F42",
  tealPale: "E4F0F0",
  brick: "8F2633",
  brickPale: "F5E7E9",
  green: "2F7D58",
  greenPale: "E7F2EC",
  ink: "202426",
  muted: "5A6668",
  line: "AAB4B7",
  pale: "F2F4F5",
  white: "FFFFFF",
  yellow: "FFF2CC",
};

const BASE = path.resolve(__dirname, "..");
const OUTPUT = path.join(BASE, "outputs");
const ROOT = path.resolve(BASE, "../..");
const FIGURES = path.join(ROOT, "figures", "thesis");
fs.mkdirSync(OUTPUT, { recursive: true });

function addText(slide, text, x, y, w, h, options = {}) {
  slide.addText(text, {
    x, y, w, h,
    fontFace: FONT,
    fontSize: 15,
    color: C.ink,
    margin: 0,
    valign: "mid",
    breakLine: false,
    fit: "shrink",
    ...options,
  });
}

function addBullets(slide, items, x, y, w, h, options = {}) {
  const runs = [];
  items.forEach((item, index) => {
    runs.push({
      text: item,
      options: {
        bullet: { indent: 15 },
        breakLine: index < items.length - 1,
      },
    });
  });
  addText(slide, runs, x, y, w, h, {
    fontSize: options.fontSize || 14,
    breakLine: true,
    paraSpaceAfterPt: options.paraSpaceAfterPt || 6,
    valign: options.valign || "top",
    ...options,
  });
}

function addTitle(slide, title, index, total, section) {
  addText(slide, title, 0.55, 0.34, 10.9, 0.58, { fontSize: 28, bold: true, color: C.ink });
  addText(slide, section, 11.52, 0.4, 1.25, 0.28, {
    fontSize: 10, bold: true, color: C.teal, align: "right",
  });
  addText(slide, `${index}/${total}`, 12.08, 7.08, 0.72, 0.2, {
    fontSize: 9, color: C.muted, align: "right",
  });
}

function addCard(slide, x, y, w, h, options = {}) {
  slide.addShape(SHAPE.roundRect, {
    x, y, w, h,
    rectRadius: 0.04,
    fill: { color: options.fill || C.white, transparency: options.transparency || 0 },
    line: { color: options.line || C.line, width: options.lineWidth || 0.8, dash: options.dash },
    shadow: options.shadow === false ? undefined : { type: "outer", color: "000000", opacity: 0.08, blur: 1, angle: 45, distance: 1 },
  });
}

function addStatus(slide, text, x, y, w, color = C.teal) {
  slide.addShape(SHAPE.roundRect, {
    x, y, w, h: 0.34,
    fill: { color, transparency: 88 },
    line: { color, width: 0.8 },
    rectRadius: 0.04,
  });
  addText(slide, text, x + 0.08, y + 0.03, w - 0.16, 0.27, {
    fontSize: 10, bold: true, color, align: "center",
  });
}

function addArrow(slide, x1, y1, x2, y2, color = C.teal, width = 1.5) {
  slide.addShape(SHAPE.line, {
    x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: { color, width, beginArrowType: "none", endArrowType: "triangle" },
  });
}

function addNetwork(slide, x, y, w, h, dark = false) {
  const ink = dark ? C.white : C.tealDark;
  const userY = y + h - 0.58;
  const uavs = [x + w * 0.28, x + w * 0.72];
  uavs.forEach((ux, index) => {
    slide.addShape(SHAPE.hexagon, {
      x: ux - 0.28, y: y + 0.25, w: 0.56, h: 0.48,
      fill: { color: index === 0 ? C.teal : C.green }, line: { color: C.white, width: 1 },
    });
    addText(slide, `UAV ${index + 1}\nMEC`, ux - 0.38, y + 0.77, 0.76, 0.44, {
      fontSize: 11, bold: true, color: ink, align: "center",
    });
  });
  for (let i = 0; i < 6; i += 1) {
    const ux = x + 0.35 + i * (w - 0.7) / 5;
    slide.addShape(SHAPE.ellipse, {
      x: ux - 0.12, y: userY, w: 0.24, h: 0.24,
      fill: { color: i % 2 ? C.yellow : C.brickPale }, line: { color: i % 2 ? C.teal : C.brick, width: 0.8 },
    });
    addArrow(slide, ux, userY - 0.02, uavs[i < 3 ? 0 : 1], y + 1.18, i % 2 ? C.teal : C.brick, 0.9);
  }
  slide.addShape(SHAPE.line, {
    x: uavs[0], y: y + 0.48, w: uavs[1] - uavs[0], h: 0,
    line: { color: C.brick, width: 2, dash: "dash", beginArrowType: "triangle", endArrowType: "triangle" },
  });
  addText(slide, "同频外部性", x + w * 0.39, y + 0.05, w * 0.22, 0.28, {
    fontSize: 11, bold: true, color: dark ? C.white : C.brick, align: "center",
  });
}

function drawTitleSlide(slide, spec) {
  slide.background = { color: C.tealDark };
  addNetwork(slide, 0.55, 3.55, 12.2, 3.15, true);
  addText(slide, spec.title, 0.72, 0.72, 11.85, 1.38, {
    fontSize: 25, bold: true, color: C.white, align: "left", valign: "bottom",
  });
  addText(slide, spec.subtitle, 0.75, 2.28, 6.8, 0.48, { fontSize: 17, color: "DDEDEE" });
  addText(slide, "关铭锡  |  指导教师：曲桦", 0.75, 2.86, 5.4, 0.34, { fontSize: 13, color: C.white, bold: true });
  addText(slide, "【学校名称】 · 【学院名称】 · 【提交日期】", 7.35, 2.9, 5.1, 0.28, {
    fontSize: 11, color: "DDEDEE", align: "right",
  });
  addStatus(slide, spec.badge, 9.22, 6.78, 3.24, C.brick);
}

function drawTriangle(slide) {
  const pts = [
    { x: 6.18, y: 1.45, label: "同频干扰", color: C.brick },
    { x: 3.18, y: 5.25, label: "SIC 解码", color: C.teal },
    { x: 9.18, y: 5.25, label: "MEC CPU", color: C.green },
  ];
  addArrow(slide, 6.22, 1.98, 3.8, 5.06, C.line, 2);
  addArrow(slide, 3.86, 5.42, 8.92, 5.42, C.line, 2);
  addArrow(slide, 8.98, 5.05, 6.5, 1.98, C.line, 2);
  pts.forEach((p) => {
    slide.addShape(SHAPE.ellipse, { x: p.x - 0.72, y: p.y - 0.36, w: 1.44, h: 0.72, fill: { color: p.color }, line: { color: C.white, width: 1 } });
    addText(slide, p.label, p.x - 0.62, p.y - 0.15, 1.24, 0.3, { fontSize: 13, bold: true, color: C.white, align: "center" });
  });
  slide.addShape(SHAPE.ellipse, { x: 5.3, y: 3.24, w: 1.82, h: 1.0, fill: { color: C.yellow }, line: { color: C.tealDark, width: 1.5 } });
  addText(slide, "唯一任务\nOnTimeRate", 5.48, 3.42, 1.46, 0.64, { fontSize: 15, bold: true, color: C.tealDark, align: "center" });
}

function drawFourLayers(slide, labels) {
  labels.forEach((label, i) => {
    const y = 1.36 + i * 1.15;
    addCard(slide, 1.0 + i * 0.42, y, 7.35 - i * 0.84, 0.82, { fill: i % 2 ? C.tealPale : C.pale, line: i === 3 ? C.brick : C.line, shadow: false });
    addText(slide, `${i + 1}`, 1.18 + i * 0.42, y + 0.18, 0.34, 0.32, { fontSize: 15, bold: true, color: i === 3 ? C.brick : C.teal });
    addText(slide, label, 1.62 + i * 0.42, y + 0.14, 5.9 - i * 0.84, 0.42, { fontSize: 16, bold: true });
  });
  addCard(slide, 9.0, 1.58, 3.0, 4.4, { fill: C.white, line: C.teal });
  addText(slide, "研究目标", 9.35, 1.9, 2.3, 0.38, { fontSize: 20, bold: true, color: C.teal, align: "center" });
  addText(slide, "把服务归属、无线可行性、CPU 预算和绝对 deadline 放进同一条可审计数据流", 9.45, 2.62, 2.1, 2.3, { fontSize: 17, bold: true, color: C.tealDark, align: "center", valign: "mid" });
}

function drawTimeline(slide, items, y = 3.1) {
  const start = 0.9;
  const gap = 11.55 / Math.max(items.length - 1, 1);
  slide.addShape(SHAPE.line, { x: start, y, w: 11.55, h: 0, line: { color: C.line, width: 2 } });
  items.forEach((item, i) => {
    const x = start + i * gap;
    const color = item.color || (i === items.length - 1 ? C.teal : C.line);
    slide.addShape(SHAPE.ellipse, { x: x - 0.18, y: y - 0.18, w: 0.36, h: 0.36, fill: { color }, line: { color: C.white, width: 1 } });
    addText(slide, item.title, x - 0.72, y - 1.08, 1.44, 0.62, { fontSize: 13, bold: true, color, align: "center", valign: "bottom" });
    addText(slide, item.detail, x - 0.78, y + 0.35, 1.56, 1.02, { fontSize: 10.5, color: C.muted, align: "center", valign: "top" });
  });
}

function drawPipeline(slide, steps, y = 2.65) {
  const x0 = 0.65;
  const gap = 0.16;
  const w = (12.05 - gap * (steps.length - 1)) / steps.length;
  steps.forEach((step, i) => {
    const x = x0 + i * (w + gap);
    addCard(slide, x, y, w, 1.32, { fill: i % 2 ? C.tealPale : C.white, line: i === 3 ? C.brick : C.teal, shadow: false });
    addText(slide, `${i + 1}`, x + 0.08, y + 0.1, 0.26, 0.28, { fontSize: 11, bold: true, color: i === 3 ? C.brick : C.teal });
    addText(slide, step, x + 0.12, y + 0.45, w - 0.24, 0.56, { fontSize: 12.5, bold: true, align: "center" });
    if (i < steps.length - 1) addArrow(slide, x + w, y + 0.66, x + w + gap - 0.02, y + 0.66, C.line, 1);
  });
}

function drawSchedule(slide) {
  [0, 1].forEach((side) => {
    const x = side ? 7.2 : 1.0;
    addText(slide, side ? "候选动作后" : "当前调度", x, 1.34, 4.8, 0.34, { fontSize: 16, bold: true, color: side ? C.green : C.teal, align: "center" });
    for (let r = 0; r < 2; r += 1) {
      for (let c = 0; c < 3; c += 1) {
        const fill = (r + c + side) % 2 ? C.tealPale : C.pale;
        addCard(slide, x + c * 1.56, 1.86 + r * 1.0, 1.34, 0.72, { fill, line: C.line, shadow: false });
        addText(slide, `UAV${r + 1} / S${c + 1}\n${side && c === 1 ? "CrossUAV" : "strong · weak"}`, x + c * 1.56 + 0.08, 1.98 + r * 1.0, 1.18, 0.44, { fontSize: 10.5, align: "center" });
      }
    }
  });
  addArrow(slide, 5.84, 2.52, 6.84, 2.52, C.brick, 2.4);
  const checks = ["唯一性", "容量", "SIC", "CPU", "字典序"];
  checks.forEach((label, i) => {
    const x = 2.05 + i * 1.88;
    addCard(slide, x, 4.45, 1.42, 0.62, { fill: i === 2 ? C.brickPale : C.white, line: i === 2 ? C.brick : C.teal, shadow: false });
    addText(slide, label, x + 0.08, 4.62, 1.26, 0.25, { fontSize: 12, bold: true, align: "center", color: i === 2 ? C.brick : C.tealDark });
    if (i < checks.length - 1) addArrow(slide, x + 1.42, 4.76, x + 1.82, 4.76, C.line, 1);
  });
}

function drawResource(slide) {
  addCard(slide, 0.78, 1.28, 6.0, 4.92, { fill: C.white, line: C.teal });
  addText(slide, "相对预算与两步解析需求", 1.08, 1.58, 5.4, 0.36, { fontSize: 18, bold: true, color: C.teal });
  addText(slide, "Tᵢ = dᵃᵇˢᵢ - aᵢ", 1.1, 2.15, 5.3, 0.5, { fontFace: FORMULA_FONT, fontSize: 22, bold: true, color: C.tealDark, align: "center" });
  addText(slide, "αᵐⁱⁿᵢ = max(0, 1 - Tᵢ fˡᵒᶜᵃˡᵢ / Cᵢ)", 1.08, 3.02, 5.4, 0.55, { fontFace: FORMULA_FONT, fontSize: 19, color: C.ink, align: "center" });
  addText(slide, "fᵐⁱⁿᵢ = αᵐⁱⁿᵢ Cᵢ / [Tᵢ - αᵐⁱⁿᵢ(1+ρ)Dᵢ/Rᵢ]", 1.02, 4.06, 5.52, 0.62, { fontFace: FORMULA_FONT, fontSize: 17, color: C.ink, align: "center" });
  addStatus(slide, "aᵢ + τᵢ ≤ dᵃᵇˢᵢ", 2.4, 5.38, 2.8, C.brick);
  addCard(slide, 7.18, 1.28, 5.35, 4.92, { fill: C.greenPale, line: C.green });
  addText(slide, "单位收益 · 最小 CPU 成本", 7.55, 1.58, 4.6, 0.36, { fontSize: 18, bold: true, color: C.green });
  const widths = [0.7, 1.0, 1.32, 1.62];
  let x = 7.62;
  widths.forEach((bw, i) => {
    slide.addShape(SHAPE.rect, { x, y: 3.0, w: bw, h: 1.0, fill: { color: i < 3 ? C.green : C.pale }, line: { color: C.white, width: 1 } });
    addText(slide, `c${i + 1}`, x, 3.3, bw, 0.28, { fontSize: 11, bold: true, color: i < 3 ? C.white : C.muted, align: "center" });
    x += bw + 0.08;
  });
  addText(slide, "Fₖ 预算", 7.62, 4.28, 4.3, 0.3, { fontSize: 12, color: C.muted, align: "center" });
  addText(slide, "按 (fᵐⁱⁿ, task_id) 排序\n选择预算可容纳的最长前缀", 7.76, 4.82, 4.1, 0.76, { fontSize: 15, bold: true, color: C.green, align: "center" });
}

function drawMatrix(slide, rowLabels, colLabels, x, y, w, h, values = null) {
  const cellW = w / 2;
  const cellH = h / 2;
  addText(slide, colLabels[0], x, y - 0.42, cellW, 0.28, { fontSize: 11, bold: true, color: C.teal, align: "center" });
  addText(slide, colLabels[1], x + cellW, y - 0.42, cellW, 0.28, { fontSize: 11, bold: true, color: C.teal, align: "center" });
  rowLabels.forEach((label, r) => addText(slide, label, x - 0.85, y + r * cellH + cellH / 2 - 0.15, 0.72, 0.3, { fontSize: 10.5, bold: true, color: C.muted, align: "right" }));
  for (let r = 0; r < 2; r += 1) {
    for (let c = 0; c < 2; c += 1) {
      const value = values ? values[r][c] : null;
      slide.addShape(SHAPE.rect, { x: x + c * cellW, y: y + r * cellH, w: cellW - 0.04, h: cellH - 0.04, fill: { color: value && value.highlight ? C.greenPale : C.white }, line: { color: value && value.highlight ? C.green : C.line, width: 1 } });
      addText(slide, value ? value.text : "未提供", x + c * cellW + 0.08, y + r * cellH + 0.28, cellW - 0.2, cellH - 0.5, { fontSize: value ? 17 : 10.5, bold: Boolean(value), color: value && value.highlight ? C.green : C.tealDark, align: "center" });
    }
  }
}

function drawAudit(slide, items) {
  items.forEach((item, i) => {
    const c = i % 2;
    const r = Math.floor(i / 2);
    const x = 0.72 + c * 6.2;
    const y = 1.25 + r * 0.99;
    addCard(slide, x, y, 5.72, 0.76, { fill: r % 2 ? C.white : C.pale, line: item.status === "已修复" ? C.green : C.line, shadow: false });
    addText(slide, `${i + 1}`, x + 0.14, y + 0.2, 0.32, 0.3, { fontSize: 12, bold: true, color: item.status === "已修复" ? C.green : C.brick });
    addText(slide, item.text, x + 0.54, y + 0.12, 3.95, 0.48, { fontSize: 11.5, bold: true });
    addStatus(slide, item.status, x + 4.6, y + 0.2, 0.9, item.status === "已修复" ? C.green : C.brick);
  });
}

function drawProgress(slide) {
  const tasks = ["Tasks 1-5\n可信基础", "Tasks 6-11\n调度闭环", "Tasks 12-16\n基线与统计", "Tasks 17-23\n证据工程", "Task 24\n正式实验"];
  tasks.forEach((task, i) => {
    const x = 0.78 + i * 1.76;
    const y = 5.45 - i * 0.72;
    addCard(slide, x, y, 1.5, 0.72, { fill: C.greenPale, line: C.green, shadow: false });
    addText(slide, task, x + 0.08, y + 0.12, 1.34, 0.45, { fontSize: 11.5, bold: true, color: C.green, align: "center" });
  });
  addCard(slide, 9.55, 1.52, 2.85, 3.96, { fill: C.white, line: C.teal });
  addText(slide, "257 passed", 9.82, 2.1, 2.3, 0.72, { fontSize: 27, bold: true, color: C.teal, align: "center" });
  addText(slide, "Tasks 1-24 完成", 9.75, 3.08, 2.45, 0.32, { fontSize: 12, bold: true, color: C.muted, align: "center" });
  addText(slide, "7 / 7\n正式实验门禁", 10.05, 3.76, 1.84, 1.05, { fontSize: 17, bold: true, color: C.green, align: "center" });
}

function drawPhysics(slide) {
  addCard(slide, 0.78, 1.36, 5.75, 4.95, { fill: C.white, line: C.teal });
  addText(slide, "PSD → 带宽 → 噪声", 1.08, 1.7, 5.15, 0.38, { fontSize: 18, bold: true, color: C.teal, align: "center" });
  const nodes = ["-174 dBm/Hz", "B' = 5/3 MHz", "-104.7815 dBm", "3.3254×10⁻¹⁴ W"];
  nodes.forEach((n, i) => {
    addCard(slide, 1.08, 2.42 + i * 0.82, 5.15, 0.56, { fill: i === 3 ? C.greenPale : C.pale, line: i === 3 ? C.green : C.line, shadow: false });
    addText(slide, n, 1.25, 2.55 + i * 0.82, 4.8, 0.27, { fontFace: i ? FORMULA_FONT : FONT, fontSize: 14, bold: i === 3, color: i === 3 ? C.green : C.ink, align: "center" });
    if (i < 3) addArrow(slide, 3.65, 2.98 + i * 0.82, 3.65, 3.2 + i * 0.82, C.teal, 1);
  });
  addCard(slide, 6.95, 1.36, 5.58, 4.95, { fill: C.tealPale, line: C.teal });
  addText(slide, "Urban 空地路径损耗", 7.28, 1.7, 4.92, 0.38, { fontSize: 18, bold: true, color: C.teal, align: "center" });
  slide.addShape(SHAPE.line, { x: 7.55, y: 5.55, w: 3.95, h: 0, line: { color: C.line, width: 2 } });
  slide.addShape(SHAPE.hexagon, { x: 10.5, y: 2.35, w: 0.62, h: 0.5, fill: { color: C.teal }, line: { color: C.white } });
  slide.addShape(SHAPE.ellipse, { x: 7.85, y: 5.15, w: 0.28, h: 0.28, fill: { color: C.brick }, line: { color: C.white } });
  addArrow(slide, 7.99, 5.12, 10.78, 2.9, C.brick, 1.5);
  addText(slide, "水平距离 r", 8.7, 5.68, 1.5, 0.28, { fontSize: 11, color: C.muted, align: "center" });
  addText(slide, "高度 H", 11.0, 3.75, 0.72, 0.3, { fontSize: 11, color: C.muted, align: "center" });
  addText(slide, "L̄ = LFSPL + PLoS·ηLoS + (1-PLoS)·ηNLoS", 7.45, 4.35, 4.55, 0.44, { fontFace: FORMULA_FONT, fontSize: 13.5, color: C.tealDark, align: "center" });
}

function drawProof(slide) {
  addText(slide, "c(1) ≤ c(2) ≤ … ≤ c(n)", 0.9, 1.38, 5.55, 0.48, { fontFace: FORMULA_FONT, fontSize: 24, bold: true, color: C.teal, align: "center" });
  const heights = [0.55, 0.86, 1.18, 1.55, 1.92];
  heights.forEach((hh, i) => {
    slide.addShape(SHAPE.rect, { x: 1.2 + i * 0.9, y: 4.45 - hh, w: 0.58, h: hh, fill: { color: i < 3 ? C.green : C.pale }, line: { color: C.white } });
    addText(slide, `c${i + 1}`, 1.16 + i * 0.9, 4.58, 0.66, 0.26, { fontSize: 10, color: C.muted, align: "center" });
  });
  slide.addShape(SHAPE.line, { x: 0.98, y: 3.18, w: 4.95, h: 0, line: { color: C.brick, width: 2, dash: "dash" } });
  addText(slide, "CPU 预算 Fₖ", 4.15, 2.82, 1.65, 0.28, { fontSize: 11, bold: true, color: C.brick, align: "right" });
  addCard(slide, 7.05, 1.32, 5.35, 4.95, { fill: C.white, line: C.teal });
  addText(slide, "交换论证", 7.4, 1.72, 4.65, 0.4, { fontSize: 20, bold: true, color: C.teal, align: "center" });
  addText(slide, "较大成本已选\n+\n较小成本未选", 7.48, 2.48, 1.8, 1.46, { fontSize: 16, bold: true, color: C.brick, align: "center" });
  addArrow(slide, 9.34, 3.2, 10.18, 3.2, C.teal, 2);
  addText(slide, "交换后\n预算不增", 10.32, 2.72, 1.55, 0.9, { fontSize: 17, bold: true, color: C.green, align: "center" });
  addStatus(slide, "解析证明", 7.56, 5.28, 1.28, C.green);
  addStatus(slide, "实现", 9.05, 5.28, 1.1, C.green);
  addStatus(slide, "暴力枚举", 10.36, 5.28, 1.44, C.green);
}

function drawTests(slide) {
  const categories = [
    ["领域/物理", 51, C.teal], ["调度/评价", 62, C.brick], ["实验/统计", 58, C.green], ["产物/文档", 86, C.tealDark],
  ];
  let x = 0.88;
  categories.forEach(([label, n, color]) => {
    const w = 0.03 * n + 0.65;
    slide.addShape(SHAPE.rect, { x, y: 2.1, w, h: 1.15, fill: { color }, line: { color: C.white, width: 1 } });
    addText(slide, `${label}\n${n}`, x + 0.05, 2.34, w - 0.1, 0.65, { fontSize: 12, bold: true, color: C.white, align: "center" });
    x += w + 0.08;
  });
  drawPipeline(slide, ["源码", "257 tests", "canonical v1", "论文与答辩"], 4.45);
  addStatus(slide, "干净 Python 3.11 环境：257 passed", 4.15, 6.16, 5.05, C.green);
}

function addContainedImage(slide, imagePath, x, y, w, h) {
  const data = fs.readFileSync(imagePath);
  const width = data.readUInt32BE(16);
  const height = data.readUInt32BE(20);
  const scale = Math.min(w / width, h / height);
  const fittedW = width * scale;
  const fittedH = height * scale;
  slide.addImage({ path: imagePath, x: x + (w - fittedW) / 2, y: y + (h - fittedH) / 2, w: fittedW, h: fittedH });
}

function drawEvidenceCards(slide, metrics) {
  metrics.forEach((metric, i) => {
    const c = i % 2;
    const r = Math.floor(i / 2);
    const x = 0.78 + c * 6.15;
    const y = 1.32 + r * 2.55;
    addCard(slide, x, y, 5.7, 2.08, { fill: metric.fill || C.white, line: metric.color || C.teal, shadow: false });
    addText(slide, metric.label, x + 0.25, y + 0.22, 5.2, 0.32, { fontSize: 13, bold: true, color: metric.color || C.teal });
    addText(slide, metric.value, x + 0.25, y + 0.67, 5.2, 0.62, { fontSize: 27, bold: true, color: C.ink, align: "center" });
    addText(slide, metric.detail, x + 0.25, y + 1.43, 5.2, 0.36, { fontSize: 11.5, color: C.muted, align: "center" });
  });
}

function drawFigureEvidence(slide, spec) {
  addCard(slide, 0.72, 1.22, 4.15, 5.55, { fill: C.white, line: spec.color || C.teal, shadow: false });
  addText(slide, spec.kicker, 1.02, 1.55, 3.55, 0.35, { fontSize: 13, bold: true, color: spec.color || C.teal, align: "center" });
  addText(slide, spec.value, 1.02, 2.08, 3.55, 0.72, { fontSize: 27, bold: true, color: C.ink, align: "center" });
  addText(slide, spec.interval, 1.02, 2.9, 3.55, 0.42, { fontSize: 12, bold: true, color: C.muted, align: "center" });
  addBullets(slide, spec.bullets, 1.05, 3.62, 3.5, 2.35, { fontSize: 12.5 });
  addStatus(slide, spec.status, 1.35, 6.15, 2.9, spec.color || C.green);
  addContainedImage(slide, path.join(FIGURES, spec.figure), 5.15, 1.23, 7.45, 5.55);
}

function drawGates(slide, gates) {
  gates.forEach((gate, i) => {
    const c = i % 2;
    const r = Math.floor(i / 2);
    const x = 0.78 + c * 6.15;
    const y = 1.25 + r * 1.3;
    addCard(slide, x, y, 5.7, 0.92, { fill: C.greenPale, line: C.green, shadow: false });
    addText(slide, "PASS", x + 0.18, y + 0.24, 0.8, 0.3, { fontSize: 12, bold: true, color: C.green, align: "center" });
    addText(slide, gate, x + 1.15, y + 0.16, 4.25, 0.48, { fontSize: 12.5, bold: true, color: C.tealDark });
  });
}

function drawRiskPlan(slide, phases) {
  drawTimeline(slide, phases, 2.85);
  const risks = [
    ["创新贡献不足", "回到设计或收缩声明"],
    ["P95 超时", "剪枝与预算门禁"],
    ["CSI 退化", "预注册鲁棒性"],
    ["证据链断裂", "canonical 自动制图"],
  ];
  risks.forEach((risk, i) => {
    const x = 0.78 + i * 3.06;
    addCard(slide, x, 5.1, 2.72, 0.98, { fill: i % 2 ? C.white : C.brickPale, line: C.brick, shadow: false });
    addText(slide, risk[0], x + 0.12, 5.25, 2.48, 0.28, { fontSize: 12, bold: true, color: C.brick, align: "center" });
    addText(slide, risk[1], x + 0.12, 5.59, 2.48, 0.26, { fontSize: 10.5, color: C.muted, align: "center" });
  });
}

function drawSummary(slide, left, right, footer) {
  slide.background = { color: C.tealDark };
  addText(slide, "两个创新点，一条可复现证据链", 0.75, 0.58, 11.8, 0.62, { fontSize: 27, bold: true, color: C.white, align: "center" });
  addCard(slide, 0.9, 1.62, 5.45, 3.48, { fill: C.white, line: C.teal, shadow: false });
  addCard(slide, 6.98, 1.62, 5.45, 3.48, { fill: C.white, line: C.green, shadow: false });
  addText(slide, "创新点一", 1.22, 1.95, 4.82, 0.42, { fontSize: 21, bold: true, color: C.teal, align: "center" });
  addText(slide, left, 1.3, 2.62, 4.65, 1.72, { fontSize: 17, bold: true, color: C.tealDark, align: "center" });
  addText(slide, "创新点二", 7.3, 1.95, 4.82, 0.42, { fontSize: 21, bold: true, color: C.green, align: "center" });
  addText(slide, right, 7.38, 2.62, 4.65, 1.72, { fontSize: 17, bold: true, color: C.tealDark, align: "center" });
  addArrow(slide, 6.34, 3.32, 6.94, 3.32, C.white, 2);
  const chain = ["模型", "算法", "测试", "canonical", "论文数字"];
  chain.forEach((label, i) => {
    const x = 1.35 + i * 2.25;
    addStatus(slide, label, x, 5.72, 1.45, i < 3 ? C.teal : C.brick);
    if (i < chain.length - 1) addArrow(slide, x + 1.48, 5.89, x + 2.12, 5.89, C.white, 1.2);
  });
  addText(slide, footer, 2.0, 6.52, 9.35, 0.36, { fontSize: 14, color: C.white, bold: true, align: "center" });
}

function drawGenericColumns(slide, spec) {
  if (spec.visual === "network") {
    addCard(slide, 0.72, 1.22, 5.05, 5.45, { fill: C.white, line: C.line });
    addBullets(slide, spec.bullets, 1.02, 1.55, 4.45, 4.75, { fontSize: spec.fontSize || 14 });
    addNetwork(slide, 6.15, 1.55, 6.5, 4.95, false);
  } else if (spec.visual === "triangle") drawTriangle(slide);
  else if (spec.visual === "layers") drawFourLayers(slide, spec.layers);
  else if (spec.visual === "schedule") drawSchedule(slide);
  else if (spec.visual === "resource") drawResource(slide);
  else if (spec.visual === "physics") drawPhysics(slide);
  else if (spec.visual === "proof") drawProof(slide);
  else if (spec.visual === "progress") drawProgress(slide);
  else if (spec.visual === "tests") drawTests(slide);
}

function warnIfSlideElementsOutOfBounds(slide, index) {
  const issues = [];
  for (const obj of slide._slideObjects || []) {
    const o = obj.options || obj;
    if (![o.x, o.y, o.w, o.h].every((v) => typeof v === "number")) continue;
    if (o.x < -0.01 || o.y < -0.01 || o.x + o.w > W + 0.01 || o.y + o.h > H + 0.01) {
      issues.push(`${obj._type || "object"}@${o.x},${o.y},${o.w},${o.h}`);
    }
  }
  if (issues.length) throw new Error(`slide ${index}: out of bounds: ${issues.join("; ")}`);
}

function warnIfSlideHasOverlaps(slide, index) {
  const texts = (slide._slideObjects || [])
    .filter((obj) => obj._type === "text" &&
      ((typeof obj.text === "string" && obj.text.trim()) || (Array.isArray(obj.text) && obj.text.length)))
    .map((obj) => ({ ...(obj.options || obj), text: obj.text }))
    .filter((o) => [o.x, o.y, o.w, o.h].every((v) => typeof v === "number"));
  for (let i = 0; i < texts.length; i += 1) {
    for (let j = i + 1; j < texts.length; j += 1) {
      const a = texts[i];
      const b = texts[j];
      const overlapW = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
      const overlapH = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
      if (overlapW > 0.08 && overlapH > 0.08) {
        throw new Error(`slide ${index}: text boxes overlap (${i}, ${j}) ` +
          `${JSON.stringify([a.x, a.y, a.w, a.h, a.text || ""])} vs ` +
          `${JSON.stringify([b.x, b.y, b.w, b.h, b.text || ""])}`);
      }
    }
  }
}

function addNotes(slide, notes) {
  slide.addNotes(notes.map((note, i) => `${i + 1}. ${note}`).join("\n"));
}

function makeDeck(meta, specs) {
  const pptx = new pptxgen();
  pptx.layout = "LAYOUT_WIDE";
  pptx.author = "关铭锡";
  pptx.subject = "专业硕士学位论文答辩材料";
  pptx.title = meta.title;
  pptx.company = "【学校名称】";
  pptx.lang = "zh-CN";
  pptx.theme = {
    headFontFace: FONT,
    bodyFontFace: FONT,
    lang: "zh-CN",
  };

  specs.forEach((spec, idx) => {
    const slide = pptx.addSlide();
    slide.background = { color: C.white };
    if (spec.kind === "title") drawTitleSlide(slide, spec);
    else if (spec.kind === "timeline") {
      addTitle(slide, spec.title, idx + 1, specs.length, meta.section);
      drawTimeline(slide, spec.items, 3.1);
      addText(slide, spec.caption, 1.2, 5.72, 10.9, 0.56, { fontSize: 14, bold: true, color: C.tealDark, align: "center" });
    } else if (spec.kind === "pipeline") {
      addTitle(slide, spec.title, idx + 1, specs.length, meta.section);
      drawPipeline(slide, spec.steps);
      addBullets(slide, spec.bullets, 1.0, 4.55, 11.25, 1.35, { fontSize: 13.5 });
    } else if (spec.kind === "schedule") {
      addTitle(slide, spec.title, idx + 1, specs.length, meta.section);
      drawSchedule(slide);
      addText(slide, spec.caption, 1.25, 5.55, 10.85, 0.56, { fontSize: 13, bold: true, color: C.tealDark, align: "center" });
    } else if (spec.kind === "resource") {
      addTitle(slide, spec.title, idx + 1, specs.length, meta.section);
      drawResource(slide);
    } else if (spec.kind === "audit") {
      addTitle(slide, spec.title, idx + 1, specs.length, meta.section);
      drawAudit(slide, spec.items);
    } else if (spec.kind === "risk") {
      addTitle(slide, spec.title, idx + 1, specs.length, meta.section);
      drawRiskPlan(slide, spec.phases);
    } else if (spec.kind === "summary") {
      drawSummary(slide, spec.left, spec.right, spec.footer);
    } else if (spec.kind === "matrix") {
      addTitle(slide, spec.title, idx + 1, specs.length, meta.section);
      addBullets(slide, spec.bullets, 0.82, 1.32, 4.2, 4.95, { fontSize: 13.5 });
      drawMatrix(slide, spec.rows, spec.cols, 6.1, 2.0, 5.75, 3.75, spec.values);
    } else if (spec.kind === "evidence") {
      addTitle(slide, spec.title, idx + 1, specs.length, meta.section);
      drawEvidenceCards(slide, spec.metrics);
    } else if (spec.kind === "figure") {
      addTitle(slide, spec.title, idx + 1, specs.length, meta.section);
      drawFigureEvidence(slide, spec);
    } else if (spec.kind === "gates") {
      addTitle(slide, spec.title, idx + 1, specs.length, meta.section);
      drawGates(slide, spec.gates);
    } else {
      addTitle(slide, spec.title, idx + 1, specs.length, meta.section);
      drawGenericColumns(slide, spec);
    }
    addNotes(slide, spec.notes);
    warnIfSlideElementsOutOfBounds(slide, idx + 1);
    warnIfSlideHasOverlaps(slide, idx + 1);
  });
  return pptx;
}

async function repairNotesMasterOrder(filePath) {
  const zip = await JSZip.loadAsync(fs.readFileSync(filePath));
  const xmlPath = "ppt/presentation.xml";
  const xml = await zip.file(xmlPath).async("string");
  const document = xmljs.xml2js(xml, { compact: false, alwaysChildren: true });
  const root = document.elements.find((element) => element.name === "p:presentation");
  if (!root || !root.elements) throw new Error(`${filePath}: presentation root missing`);

  const notesIndex = root.elements.findIndex((element) => element.name === "p:notesMasterIdLst");
  const slidesIndex = root.elements.findIndex((element) => element.name === "p:sldIdLst");
  if (notesIndex < 0 || slidesIndex < 0) throw new Error(`${filePath}: notes or slide list missing`);
  if (notesIndex > slidesIndex) {
    const [notesMaster] = root.elements.splice(notesIndex, 1);
    const targetIndex = root.elements.findIndex((element) => element.name === "p:sldIdLst");
    root.elements.splice(targetIndex, 0, notesMaster);
    zip.file(xmlPath, xmljs.js2xml(document, { compact: false, spaces: 0 }));
    fs.writeFileSync(filePath, await zip.generateAsync({
      type: "nodebuffer",
      compression: "DEFLATE",
      compressionOptions: { level: 6 },
    }));
  }
}

const midterm = [
  { kind: "title", title: "面向全频复用多无人机 NOMA-MEC 网络的干扰感知用户配对与截止时间资源分配研究", subtitle: "专业学位硕士研究生中期答辩", badge: "canonical v1 · 7/7 门禁通过", notes: ["本次汇报使用正式确认性实验数据。", "论文形成两个可独立消融、可组合运行的创新点。", "历史轨迹和 DRL 结果不进入当前性能证据。"] },
  { kind: "generic", title: "研究目标：最大化独立任务的按时完成数", visual: "triangle", bullets: ["决策：服务 UAV、子载波、NOMA 角色、卸载比例、CPU", "约束：唯一服务、容量、接收功率次序、硬 SIC、CPU 预算", "任务判定：aᵢ + τᵢ ≤ dᵃᵇˢᵢ", "主指标：按时完成唯一任务数 / 到达唯一任务数"], notes: ["论文最终目标是任务级按时完成，而不是瞬时吞吐率。", "服务无人机变化会同时影响信道、干扰和服务器负载。", "失败任务只计一次，避免统计分母漂移。"] },
  { kind: "timeline", title: "研究路线：从可信审计到正式证据", items: [{ title: "问题收缩", detail: "固定悬停\n退出轨迹/DRL", color: C.line }, { title: "可信模型", detail: "唯一任务\nSchedule", color: C.teal }, { title: "双创新", detail: "外部性调度\n+ deadline资源", color: C.teal }, { title: "证据工程", detail: "配对统计\ncanonical", color: C.green }, { title: "正式实验", detail: "50 unseen seeds\n7/7 gates", color: C.green }, { title: "毕业交付", detail: "论文/报告\n答辩材料", color: C.brick }], caption: "Tasks 1-24 已完成；当前任务是论文和答辩材料定稿", notes: ["路线调整保留多无人机 NOMA-MEC 工程场景。", "旧结果只解释研究演化，不支持新方法。", "正式实验已完成，现在不是等待数据阶段。"] },
  { kind: "generic", title: "当前完成度：核心算法和正式实验均已完成", visual: "progress", bullets: ["可信任务、物理层与统一评价器", "ILP 初始化、SIC 修复与三类邻域", "解析资源需求、最大基数分配与证明", "基线、消融、配对统计与 canonical v1", "论文与答辩交付正在定稿"], notes: ["Tasks 1-24 已完成并形成提交记录。", "257 项测试是软件正确性证据。", "7/7 门禁是正式实验结论证据。"] },
  { kind: "pipeline", title: "可信架构：所有方法经过同一条评价链", steps: ["不可变 trace", "显式 Schedule", "真实服务信道", "全频干扰/SIC", "最小资源", "完成集合", "配对统计", "canonical v1"], bullets: ["同一外生轨迹保证方法间按种子配对比较", "服务归属、物理层、CPU 和任务完成状态均从最终 Schedule 派生", "manifest 与 SHA256 把代码、配置、结果和论文数字连接起来"], notes: ["核心原则是消除互相矛盾的状态副本。", "主方法和所有基线不能绕过统一评价器。", "论文图表只由 canonical 资产生成。"] },
  { kind: "schedule", title: "创新点一：干扰外部性感知的跨 UAV 联合关联与配对", caption: "Replace / Position / CrossUAV · 每次候选全局重算信道、干扰、SIC 与 CPU", notes: ["CrossUAV 动作同步改变真实服务归属。", "全局准则评价动作对全部接收端和服务器的外部性。", "有限可行集上只接受字典序严格改善，因此搜索有限终止。"] },
  { kind: "figure", title: "创新点一结果：两个组件均获得独立正贡献", kicker: "CrossUAV 邻域贡献", value: "+0.0487 pp", interval: "95% CI [0.0138, 0.0841]", bullets: ["Holm p = 0.0201", "秩二列效应量 0.521", "全局准则另贡献 +0.1179 pp", "高完成率区域增益小但稳定为正"], status: "严格 2×2 消融支持", color: C.teal, figure: "pairing_ablation.png", notes: ["CrossUAV 与全局接受准则分别消融。", "创新点一的绝对增益不大，因此强调机制和约束一致性。", "结论限定在本文正式实验配置。"] },
  { kind: "resource", title: "创新点二：deadline 驱动的卸载与 MEC 算力分配", notes: ["截止时间先转化为最小卸载比例和最小 CPU 需求。", "单位完成收益下，最小 CPU 成本最长前缀实现最大完成基数。", "解析证明、实现和小规模枚举共同验证。"] },
  { kind: "figure", title: "创新点二结果：直接优化 deadline 带来主要增益", kicker: "相对 crossing / equal", value: "+8.6898 pp", interval: "95% CI [8.3736, 9.0106]", bullets: ["deadline-driven：98.11%", "crossing/equal：89.42%", "fixed-half/equal：76.13%", "Holm p < 1.6×10⁻¹⁴"], status: "主要性能来源", color: C.green, figure: "resource_ablation.png", notes: ["三种资源策略共享相同无线调度。", "创新点二效果幅度大且跨种子方向一致。", "该模块承担论文主要性能贡献。"] },
  { kind: "evidence", title: "正式主实验：完整方法达到 98.11% 按时完成率", metrics: [
    { label: "完整方法", value: "98.11%", detail: "95% CI [97.94%, 98.26%]", color: C.green, fill: C.greenPale },
    { label: "相对 ExchangeStable", value: "+0.1907 pp", detail: "95% CI [0.1304, 0.2567]", color: C.teal, fill: C.tealPale },
    { label: "最终约束", value: "0 违规", detail: "SIC 与 CPU 预算均无违规", color: C.brick, fill: C.brickPale },
    { label: "N=20 调度运行时间", value: "P95 40.93 ms", detail: "低于 1 s 预注册门槛", color: C.tealDark, fill: C.pale },
  ], notes: ["完整方法在 50 个未见随机种子上评估。", "强基线差值使用配对 bootstrap 与 Wilcoxon-Holm。", "运行时间是调度算法墙钟时间。"] },
  { kind: "figure", title: "主方法与七类基线：完成率和强基线差值同时报告", kicker: "预注册强基线", value: "+0.1907 pp", interval: "Holm p = 6.71×10⁻⁶", bullets: ["Greedy：+3.24 pp", "Per-UAV ILP：+1.30 pp", "IA-Greedy：+0.49 pp", "Fair OFDMA：+0.58 pp"], status: "配对统计支持", color: C.teal, figure: "main_on_time.png", notes: ["主结果图展示所有方法的按时完成率。", "强基线 ExchangeStable 是最关键主比较。", "同时报告绝对完成率和配对差值，避免只看柱高。"] },
  { kind: "matrix", title: "创新点一实验设计：邻域范围 × 接受准则", bullets: ["Intra/CrossUAV 控制邻域范围", "Bilateral/Global 控制接受准则", "四格共享初始化、轨迹和评价器", "CrossUAV 贡献：full − intra_global", "Global 贡献：full − cross_bilateral"], rows: ["Intra", "CrossUAV"], cols: ["Bilateral", "Global"], values: [[{ text: "97.9146%" }, { text: "98.0567%" }], [{ text: "97.9874%" }, { text: "98.1054%", highlight: true }]], notes: ["严格四格避免初始化和物理语义漂移。", "两个组件的置信区间均为正。", "不把模块联合增益冒充单个组件贡献。"] },
  { kind: "figure", title: "机制与边界：CrossUAV 只在存在跨接收端耦合时有意义", kicker: "机制验证", value: "外部性可追踪", interval: "负控制 + 干扰耦合实验", bullets: ["K=1：无跨 UAV 邻域来源", "零耦合：外部性效应应收缩", "真实案例记录动作前后状态", "不把相关性当作机制证明"], status: "机制实验已完成", color: C.brick, figure: "mechanism_interference.png", notes: ["机制实验验证 CrossUAV 改善来自跨接收端干扰耦合。", "单 UAV 和零干扰用于负控制。", "详细事件日志支持答辩追问。"] },
  { kind: "gates", title: "正式实验验收：7 / 7 门禁全部通过", gates: ["创新点一 CrossUAV 置信区间严格为正", "创新点一全局接受准则置信区间严格为正", "创新点二 deadline 贡献置信区间严格为正", "完整方法优于预注册强基线", "最终调度无 SIC / CPU 预算违规", "小规模完整枚举执行完毕且完成数差距为 0", "N=20 运行时间 P95 低于 1 s"], notes: ["门禁在确认性实验之前预先声明。", "所有门禁结果由 acceptance.json 统一给出。", "通过门禁不代表超出本文假设的部署保证。"] },
  { kind: "generic", title: "工程验证与可复现交付", visual: "tests", bullets: ["Python 3.11 干净环境全量测试通过", "canonical v1 保存 manifest、配置和 SHA256", "论文数字由生成宏和图表资产统一注入", "中期报告与 PPT 已切换为正式数据"], notes: ["测试保护任务语义、物理层、算法、统计和产物。", "canonical 原始运行数据不手工修改。", "论文和答辩材料禁止重新录入旧数字。"] },
  { kind: "risk", title: "从中期到毕业：剩余工作集中在论文定稿与答辩", phases: [{ title: "核心算法", detail: "已完成", color: C.green }, { title: "正式实验", detail: "已完成", color: C.green }, { title: "论文主体", detail: "八章初稿", color: C.teal }, { title: "学校模板", detail: "待适配", color: C.brick }, { title: "答辩材料", detail: "本轮定稿", color: C.teal }, { title: "毕业答辩", detail: "风险问答", color: C.brick }], notes: ["剩余风险不是核心实验缺失。", "创新点一应准确表述为小幅但稳定的独立贡献。", "最终提交前需补学校封面、授权页和格式要求。"] },
  { kind: "summary", left: "已实现并独立验证\n跨 UAV 外部性调度\n+0.05 pp / +0.12 pp", right: "已实现并独立验证\ndeadline 最小资源分配\n+8.69 pp", footer: "98.11% 按时完成 · 0 约束违规 · 257 tests · 7/7 gates", notes: ["两个创新点边界清晰且均有正式数据支持。", "创新点二承担主要性能贡献。", "当前项目已进入专业硕士论文和毕业答辩定稿阶段。"] },
];

async function main() {
  const midtermDeck = makeDeck({ title: "中期答辩", section: "中期答辩" }, midterm);
  const midtermPath = path.join(OUTPUT, "中期答辩_关铭锡.pptx");
  await midtermDeck.writeFile({ fileName: midtermPath });
  await repairNotesMasterOrder(midtermPath);
  console.log(`[generated] midterm=${midterm.length} slides`);
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});
