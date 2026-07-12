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

function addPlaceholder(slide, title, keys, x, y, w, h) {
  slide.addShape(SHAPE.rect, {
    x, y, w, h,
    fill: { color: C.white, transparency: 100 },
    line: { color: C.line, width: 1, dash: "dash" },
  });
  addText(slide, title, x + 0.16, y + 0.12, w - 0.32, 0.28, { fontSize: 12, bold: true, color: C.tealDark });
  addText(slide, "待新版本 canonical 实验填充", x + 0.16, y + 0.48, w - 0.32, 0.34, {
    fontSize: 12, bold: true, color: C.brick, align: "center",
  });
  addText(slide, keys.map((key) => `{{${key}}}`).join("\n"), x + 0.16, y + 0.92, w - 0.32, h - 1.05, {
    fontSize: 8.5, color: C.muted, valign: "top", breakLine: true,
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

function drawMatrix(slide, rowLabels, colLabels, x, y, w, h) {
  const cellW = w / 2;
  const cellH = h / 2;
  addText(slide, colLabels[0], x, y - 0.42, cellW, 0.28, { fontSize: 11, bold: true, color: C.teal, align: "center" });
  addText(slide, colLabels[1], x + cellW, y - 0.42, cellW, 0.28, { fontSize: 11, bold: true, color: C.teal, align: "center" });
  rowLabels.forEach((label, r) => addText(slide, label, x - 0.85, y + r * cellH + cellH / 2 - 0.15, 0.72, 0.3, { fontSize: 10.5, bold: true, color: C.muted, align: "right" }));
  for (let r = 0; r < 2; r += 1) {
    for (let c = 0; c < 2; c += 1) {
      slide.addShape(SHAPE.rect, { x: x + c * cellW, y: y + r * cellH, w: cellW - 0.04, h: cellH - 0.04, fill: { color: C.white, transparency: 100 }, line: { color: C.line, width: 1, dash: "dash" } });
      addText(slide, "待新版本 canonical\n实验填充", x + c * cellW + 0.08, y + r * cellH + 0.28, cellW - 0.2, cellH - 0.5, { fontSize: 10.5, color: C.brick, align: "center" });
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
  const tasks = ["Task 1\n环境基线", "Task 2\n领域记录", "Task 3\n物理层", "Task 4\n解析需求", "Task 5\n最大基数"];
  tasks.forEach((task, i) => {
    const x = 0.78 + i * 1.76;
    const y = 5.45 - i * 0.72;
    addCard(slide, x, y, 1.5, 0.72, { fill: C.greenPale, line: C.green, shadow: false });
    addText(slide, task, x + 0.08, y + 0.12, 1.34, 0.45, { fontSize: 11.5, bold: true, color: C.green, align: "center" });
  });
  addCard(slide, 9.55, 1.52, 2.85, 3.96, { fill: C.white, line: C.teal });
  addText(slide, "121 passed", 9.82, 2.1, 2.3, 0.72, { fontSize: 27, bold: true, color: C.teal, align: "center" });
  addText(slide, "截止 fd0cb7c", 9.9, 3.08, 2.14, 0.32, { fontSize: 12, bold: true, color: C.muted, align: "center" });
  addText(slide, "软件测试证据\n≠\n算法性能", 10.05, 3.76, 1.84, 1.05, { fontSize: 15, bold: true, color: C.brick, align: "center" });
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
    ["元数据", 1, C.line], ["领域记录", 25, C.teal], ["物理层", 33, C.brick], ["资源需求", 26, C.green], ["分配器", 36, C.tealDark],
  ];
  let x = 0.88;
  categories.forEach(([label, n, color]) => {
    const w = 0.03 * n + 0.65;
    slide.addShape(SHAPE.rect, { x, y: 2.1, w, h: 1.15, fill: { color }, line: { color: C.white, width: 1 } });
    addText(slide, `${label}\n${n}`, x + 0.05, 2.34, w - 0.1, 0.65, { fontSize: 12, bold: true, color: C.white, align: "center" });
    x += w + 0.08;
  });
  drawPipeline(slide, ["源码", "121 tests", "canonical", "统计图表"], 4.45);
  addStatus(slide, "当前停在软件验证节点，性能待实验", 4.35, 6.16, 4.65, C.brick);
}

function drawDataBoard(slide, groups) {
  const cols = groups.length === 4 ? 2 : groups.length;
  groups.forEach((group, i) => {
    const c = i % cols;
    const r = Math.floor(i / cols);
    const w = cols === 2 ? 5.72 : 3.83;
    const h = cols === 2 ? 2.32 : 4.86;
    const x = 0.7 + c * (w + 0.38);
    const y = 1.25 + r * (h + 0.32);
    addPlaceholder(slide, group.title, group.keys, x, y, w, h);
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
    } else if (spec.kind === "data") {
      addTitle(slide, spec.title, idx + 1, specs.length, meta.section);
      drawDataBoard(slide, spec.groups);
    } else if (spec.kind === "risk") {
      addTitle(slide, spec.title, idx + 1, specs.length, meta.section);
      drawRiskPlan(slide, spec.phases);
    } else if (spec.kind === "summary") {
      drawSummary(slide, spec.left, spec.right, spec.footer);
    } else if (spec.kind === "matrix") {
      addTitle(slide, spec.title, idx + 1, specs.length, meta.section);
      addBullets(slide, spec.bullets, 0.82, 1.32, 4.2, 4.95, { fontSize: 13.5 });
      drawMatrix(slide, spec.rows, spec.cols, 6.1, 2.0, 5.75, 3.75);
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

const proposal = [
  { kind: "title", title: "面向全频复用多无人机 NOMA-MEC 网络的干扰感知用户配对与截止时间资源分配研究", subtitle: "专业学位硕士研究生开题答辩", badge: "新版本性能数据待 canonical 实验", notes: ["本课题研究固定悬停多无人机 NOMA-MEC 网络中的联合调度与资源分配。", "开题阶段重点说明问题价值、两个创新点和可验证的研究路线。", "所有性能区域均保留新版本实验占位，不使用旧结果。"] },
  { kind: "generic", title: "应用场景：空中边缘计算的时限任务", visual: "network", bullets: ["应急巡检任务具有唯一身份和绝对截止时刻", "多 UAV 提供覆盖与 MEC 算力", "全频复用提高利用率，也产生跨 UAV 同频干扰", "最终目标是按时完成任务，而不是仅提高瞬时速率"], notes: ["应用对象是具有硬截止时间的独立计算任务。", "无线调度和边缘算力必须共同决定任务是否按时完成。", "该工程目标符合专业硕士的问题导向。"] },
  { kind: "generic", title: "核心矛盾：全频复用带来的四层耦合", visual: "layers", layers: ["服务归属改变目标信道", "同频复用改变跨 UAV 干扰", "NOMA 角色改变 SIC 可行性", "MEC CPU 与 deadline 决定实际完成"], bullets: ["一个跨 UAV 动作同时影响多个接收端", "局部速率增益不保证网络完成数改善", "需要统一 Schedule 和统一评价器"], notes: ["四层耦合说明先关联后配对的局部流程可能失真。", "调度动作必须触发全局物理量和 CPU 结果重算。", "因此论文以任务完成数作为统一评价出口。"] },
  { kind: "timeline", title: "研究现状与缺口：已有组件尚未形成闭环", items: [{ title: "UAV-MEC", detail: "覆盖与算力\n常优化轨迹", color: C.line }, { title: "NOMA", detail: "用户聚类\n功率与SIC", color: C.teal }, { title: "联合优化", detail: "多变量耦合\n常固定关联", color: C.brick }, { title: "本课题", detail: "外部性调度\n+ deadline资源", color: C.green }], caption: "缺口：真实服务归属、跨 UAV 外部性、SIC 与绝对 deadline 尚未形成一条可审计闭环", notes: ["已有研究分别覆盖 UAV-MEC、NOMA 和联合资源优化。", "主要缺口不是组件缺失，而是调度语义和完成指标没有统一。", "本课题据此形成两个相互独立又能串联的创新模块。"] },
  { kind: "generic", title: "研究问题与边界：做什么、不做什么", visual: "triangle", bullets: ["输入：任务、位置、全局 CSI、系统预算", "决策：服务 UAV、子载波、角色、卸载比例、CPU", "主目标：最大化绝对 deadline 前完成的唯一任务数", "边界：固定悬停、等功率、集中式 CSI；不研究轨迹和功率控制"], notes: ["主动收缩边界有利于聚焦主导矛盾。", "失败任务只记录一次，不允许跨时隙重复统计。", "所有范围外假设将在论文局限性中明确。"] },
  { kind: "generic", title: "系统模型：同频干扰、SIC 与绝对 deadline", visual: "physics", bullets: ["Icross 按目标接收 UAV 信道重算", "Rᵢ = Bₛ log₂(1+γᵢ)，强用户满足 SIC 门限", "Tᵢ = dᵃᵇˢᵢ - aᵢ > 0", "aᵢ + τᵢ ≤ dᵃᵇˢᵢ ⇔ τᵢ ≤ Tᵢ"], notes: ["噪声由功率谱密度、噪声系数和实际带宽一次计算。", "NOMA 强弱角色由实际接收功率确定。", "相对预算和绝对截止被明确连接，避免公式口径混用。"] },
  { kind: "pipeline", title: "总体技术路线：一条可信数据流", steps: ["固定轨迹", "候选池", "Schedule", "邻域动作", "物理重算", "资源选择", "统一评估", "canonical"], bullets: ["创新点一覆盖 Schedule、邻域和全局物理重算", "创新点二把速率转化为最小 CPU 成本与最大完成集合", "所有表格和图只读取带提交、配置和校验和的 canonical 产物"], notes: ["同一外生轨迹确保算法间配对比较公平。", "两个创新通过显式接口连接而不是混成黑箱。", "canonical 产物负责把代码、统计和论文数字闭环。"] },
  { kind: "schedule", title: "创新点一：干扰外部性感知的跨 UAV 联合关联与配对", caption: "有限可行集 + 字典序严格改善：完成数 ↑ → slack ↑ → 跨 UAV 干扰 ↓ → 稳定编码", notes: ["CrossUAV 动作会同步改变真实服务归属。", "每个候选都重算服务信道、同频干扰、SIC 和完成集合。", "严格二乘二消融将独立验证邻域和接受准则，目前不宣称增益。"] },
  { kind: "resource", title: "创新点二：deadline 驱动的卸载与 MEC 算力分配", notes: ["解析公式直接从截止时间约束得到最低资源需求。", "单位收益条件下，最小 CPU 成本前缀实现最大完成任务数。", "该模块可固定 Schedule 独立消融，也可作为创新点一的下游评价器。"] },
  { kind: "pipeline", title: "联合算法：两个创新点协同但可区分", steps: ["ILP 初始化", "邻域枚举", "速率/SIC", "最小资源", "完成集合", "字典序接受"], bullets: ["创新一输出调度与速率；创新二输出完成集合与 CPU 分配", "模块增量分解键：{{MODULE_BASELINE_PCT}} · {{MODULE_ASSOC_ONLY_PCT}} · {{MODULE_RESOURCE_ONLY_PCT}} · {{MODULE_JOINT_PCT}}", "每次候选记录前后状态与拒绝原因"], notes: ["联合算法通过明确输入输出保持可解释性。", "模块分解实验能区分两个创新的独立贡献。", "结构化事件日志用于机制分析和工程审计。"] },
  { kind: "matrix", title: "实验设计：独立消融、机制验证与配对统计", bullets: ["2×2 四格与交互项完整注册", "创新二三策略共享相同 Schedule", "确认性 seeds 1000-1049", "配对 bootstrap、Wilcoxon、Holm 与效应量", "K=1 与零干扰作为负控制"], rows: ["Intra", "CrossUAV"], cols: ["Bilateral", "Global"], notes: ["两个创新点采用不同的受控消融。", "所有方法共享外生轨迹、评价器和统计流程。", "小规模最优性只有覆盖完整可行域时才称为 exact optimum。"] },
  { kind: "data", title: "数据占位仪表板：只注册，不预填", groups: [
    { title: "主实验与统计", keys: ["MAIN_ONTIME_OURS_PCT", "MAIN_ONTIME_GREEDY_PCT", "MAIN_ONTIME_OFDMA_PCT", "MAIN_CI95_PP", "MAIN_PAIRED_DIFF_CI95_PP", "MAIN_RUNTIME_S"] },
    { title: "严格 2×2", keys: ["ABLATION_NONE_PCT", "ABLATION_GLOBAL_ONLY_PCT", "ABLATION_CROSSUAV_ONLY_PCT", "ABLATION_BOTH_PCT", "ABLATION_INTERACTION_PP", "ABLATION_GLOBAL_ON_PCT", "ABLATION_CROSSUAV_ON_PCT"] },
    { title: "资源与模块分解", keys: ["RESOURCE_DEADLINE_AWARE_PCT", "RESOURCE_EQUAL_CPU_PCT", "RESOURCE_RATE_FIRST_PCT", "MODULE_BASELINE_PCT", "MODULE_ASSOC_ONLY_PCT", "MODULE_RESOURCE_ONLY_PCT", "MODULE_JOINT_PCT"] },
    { title: "机制、最优性与来源", keys: ["SIC_SENSITIVITY_SERIES", "UAV_SCALING_SERIES", "USER_SCALING_SERIES", "CSI_ROBUSTNESS_SERIES", "EXACT_GAP_MEAN_PCT", "EXACT_GAP_MAX_PCT", "CANONICAL_COMMIT", "CANONICAL_RUN_DATE", "CANONICAL_MANIFEST_SHA256"] },
  ], notes: ["本页不绘制柱高、折线点或置信区间。", "二十九个键分别对应预注册 canonical 来源。", "若确认性结果不支持预期贡献，将调整算法或收缩声明。"] },
  { kind: "risk", title: "实施计划与风险闭环", phases: [{ title: "可信基础", detail: "对象/物理/公式", color: C.green }, { title: "统一仿真", detail: "Schedule与评估", color: C.teal }, { title: "创新一", detail: "邻域与接受", color: C.teal }, { title: "证据工程", detail: "统计与产物", color: C.line }, { title: "确认实验", detail: "未见seeds", color: C.brick }, { title: "毕业交付", detail: "论文与答辩", color: C.brick }], notes: ["计划按可检查产物和质量门禁推进。", "核心风险均配置触发条件和处置路径。", "最终门禁是干净环境重建与论文数字可追溯。"] },
  { kind: "summary", left: "显式 Schedule + CrossUAV 邻域\n全局重算干扰、SIC 与完成集合", right: "相对预算 Tᵢ + 最小资源需求\n最大基数 MEC CPU 分配", footer: "旧结果不作为新方案证据 · 性能结论等待 canonical 实验", notes: ["论文围绕两个鲜明且可独立验证的创新展开。", "两项创新共享统一模型和证据链。", "开题阶段不对尚未产生的性能结果作任何强声明。"] },
];

const midterm = [
  { kind: "title", title: "面向全频复用多无人机 NOMA-MEC 网络的干扰感知用户配对与截止时间资源分配研究", subtitle: "专业学位硕士研究生中期答辩", badge: "证据截止 fd0cb7c · 性能待 canonical 实验", notes: ["中期汇报重点说明路线调整和当前可信成果。", "创新点一设计完成待实现，创新点二核心已完成并验证。", "历史路线只解释研究演化，不支持新方法性能结论。"] },
  { kind: "generic", title: "工程目标：最大化独立任务的按时完成数", visual: "triangle", bullets: ["决策：服务 UAV、子载波、NOMA 角色、卸载比例、CPU", "约束：唯一服务、容量、接收功率次序、SIC、CPU 预算", "Tᵢ = dᵃᵇˢᵢ - aᵢ；aᵢ + τᵢ ≤ dᵃᵇˢᵢ", "主指标：按时完成的唯一任务数 / 到达的唯一任务总数"], notes: ["论文不再把吞吐率作为最终目标。", "服务无人机变化会同时影响信道、干扰和服务器负载。", "绝对 deadline 判定直接对应专业应用中的时限要求。"] },
  { kind: "timeline", title: "研究路线演化：保留场景，重构问题", items: [{ title: "单 UAV", detail: "轨迹/DRL\n探索", color: C.line }, { title: "多 UAV", detail: "Grouped NOMA\n扩展", color: C.line }, { title: "可信审计", detail: "发现数据流\n系统问题", color: C.brick }, { title: "双创新", detail: "调度外部性\n+ deadline资源", color: C.green }], caption: "前期资产继续复用；旧性能数字不进入新论文证据", notes: ["路线调整保留了场景模型和工程资产。", "可信性审计揭示了旧证据链的关键问题。", "当前主线聚焦更直接影响任务完成的两个矛盾。"] },
  { kind: "generic", title: "科学纠偏：复杂算法必须解决主导矛盾", visual: "layers", layers: ["高 LoS 覆盖", "水平距离敏感性受限", "传输项可能小于计算项", "轨迹优化边际作用可能受限"], bullets: ["旧现象只支持重新建模，不证明新算法", "退出：轨迹、DRL、联合功率控制", "保留：多 UAV、Grouped NOMA、MEC、全频复用"], notes: ["前期现象提示轨迹可能不是当前参数下的主导变量。", "因此先修复任务语义、物理层和评价链路。", "新方法把复杂度用于跨 UAV 外部性和硬截止资源竞争。"] },
  { kind: "audit", title: "可信性审计：先修证据链，再谈性能", items: [
    { text: "CrossUAV 后服务归属未贯穿", status: "待实现" }, { text: "最终调度缺少统一 SIC 门禁", status: "待实现" },
    { text: "失败任务可能跨时隙重复计数", status: "待实现" }, { text: "噪声可能被重复按带宽缩放", status: "已修复" },
    { text: "2×2 消融初始化不一致", status: "待实现" }, { text: "NP 规约未落到物理问题", status: "收缩声明" },
    { text: "受限搜索冒充 true optimum", status: "收缩声明" }, { text: "多个 runner 状态推进漂移", status: "待实现" },
    { text: "旧结果缺少 manifest", status: "待实现" }, { text: "核心不变量缺少测试", status: "已修复" },
  ], notes: ["这些问题会改变速率、可行性或统计分母。", "中期只把已经修复的项目标为完成。", "其余问题已拆入后续任务并配置失败即停止门禁。"] },
  { kind: "pipeline", title: "可信架构：唯一数据流", steps: ["Task", "Schedule", "信道/干扰", "SIC", "资源需求", "Allocation", "SlotResult", "canonical"], bullets: ["Schedule 是服务归属唯一权威来源", "同一评估器服务主方法和所有基线", "任务身份、拒绝原因和证据哈希全程保留"], notes: ["可信架构的核心是消除多份相互矛盾的状态。", "物理层和 MEC 负载都从最终 Schedule 派生。", "统一结果对象为统计和论文数字提供稳定接口。"] },
  { kind: "generic", title: "当前完成度：Tasks 1-5 已完成并验证", visual: "progress", bullets: ["Task 1：Python/pytest 元数据", "Task 2：不可变领域记录", "Task 3：噪声与 Urban 路损", "Task 4：最小卸载与 CPU 需求", "Task 5：最大基数分配器"], notes: ["当前五项工作覆盖数据模型、物理层和资源分配核心。", "一百二十一项测试是软件验证，不是算法性能。", "创新点一仍需统一评估器、初始化和邻域搜索。"] },
  { kind: "generic", title: "物理层修复：带宽、噪声和场景语义一致", visual: "physics", bullets: ["Pn(dBm) = -174 + NF + 10 log₁₀(B')", "5 MHz/3、NF=7 dB：-104.7815 dBm", "Urban：a=9.61，b=0.16，ηLoS=1 dB，ηNLoS=20 dB", "公式数值是单测基准，不是性能结果"], notes: ["旧实现混合了噪声值和带宽缩放语义。", "新实现从功率谱密度出发一次计算噪声。", "Urban 场景参数和非法输入检查均已写入测试。"] },
  { kind: "resource", title: "创新点二：把无线速率转化为最小资源成本", notes: ["相对预算由绝对截止和到达时刻唯一确定。", "解析需求把连续卸载变量转化为可加 CPU 成本。", "该核心已完成并验证，但性能贡献仍待统一实验。"] },
  { kind: "generic", title: "硬 CPU 预算下的最大按时完成任务数", visual: "proof", bullets: ["按 (CPU 成本, task_id) 稳定排序", "选择满足总预算的最长前缀", "最小 m+1 成本和超预算 ⇒ 任意 m+1 集合不可行", "小规模完成基数与完整子集枚举一致"], notes: ["每个完成任务对主目标贡献相同。", "最小成本前缀通过交换论证实现最大基数。", "稳定平局规则保证同一输入得到相同代表解。"] },
  { kind: "schedule", title: "创新点一：设计完成，进入实现阶段", caption: "当前状态：设计完成待实现 · 依赖 Schedule 校验、统一评价器、ILP 初始化和三类邻域", notes: ["创新点一的算法规格和邻域语义已经冻结。", "跨 UAV 动作会触发网络级物理和资源重算。", "生产实现尚未完成，因此本页不展示任何性能数字。"] },
  { kind: "generic", title: "中期工程证据：测试通过，但性能尚未生成", visual: "tests", bullets: ["fd0cb7c：121 passed", "覆盖有限性、不可变性、噪声带宽、解析最小性、ULP 边界和稳定前缀", "软件测试证据 ≠ 按时完成率、显著性或运行时间"], notes: ["现有测试说明 Tasks 1-5 满足当前自动化规格。", "资源分配结果已与小规模暴力枚举交叉验证。", "后续仍需配对实验和统计形成性能证据。"] },
  { kind: "data", title: "性能证据位置一：主比较与双创新消融", groups: [
    { title: "主实验与统计", keys: ["MAIN_ONTIME_OURS_PCT", "MAIN_ONTIME_GREEDY_PCT", "MAIN_ONTIME_OFDMA_PCT", "MAIN_CI95_PP", "MAIN_PAIRED_DIFF_CI95_PP", "MAIN_RUNTIME_S"] },
    { title: "严格 2×2", keys: ["ABLATION_NONE_PCT", "ABLATION_GLOBAL_ONLY_PCT", "ABLATION_CROSSUAV_ONLY_PCT", "ABLATION_BOTH_PCT", "ABLATION_INTERACTION_PP", "ABLATION_GLOBAL_ON_PCT", "ABLATION_CROSSUAV_ON_PCT"] },
    { title: "创新点二三策略", keys: ["RESOURCE_DEADLINE_AWARE_PCT", "RESOURCE_EQUAL_CPU_PCT", "RESOURCE_RATE_FIRST_PCT"] },
    { title: "双模块增量", keys: ["MODULE_BASELINE_PCT", "MODULE_ASSOC_ONLY_PCT", "MODULE_RESOURCE_ONLY_PCT", "MODULE_JOINT_PCT"] },
  ], notes: ["主实验和消融页面结构已经冻结。", "四格矩阵、三策略和模块分解都从同一 canonical bundle 回填。", "当前不放入任何旧数字或视觉化假数据。"] },
  { kind: "data", title: "性能证据位置二：鲁棒性、扩展性与来源", groups: [
    { title: "敏感性与扩展性", keys: ["SIC_SENSITIVITY_SERIES", "UAV_SCALING_SERIES", "USER_SCALING_SERIES"] },
    { title: "鲁棒性与完整最优性", keys: ["CSI_ROBUSTNESS_SERIES", "EXACT_GAP_MEAN_PCT", "EXACT_GAP_MAX_PCT"] },
    { title: "canonical provenance", keys: ["CANONICAL_COMMIT", "CANONICAL_RUN_DATE", "CANONICAL_MANIFEST_SHA256"] },
  ], notes: ["这一页用于限定方法的适用范围和失败条件。", "小规模差距只有覆盖完整可行域时使用 exact 表述。", "每个图都必须回溯到提交、日期和 manifest 校验和。"] },
  { kind: "risk", title: "从中期到毕业：五道门禁，不跳步跑实验", phases: [{ title: "可信模型", detail: "Tasks 6-8", color: C.teal }, { title: "算法闭环", detail: "Tasks 9-13", color: C.teal }, { title: "证据工程", detail: "Tasks 14-23", color: C.line }, { title: "正式实验", detail: "Task 24", color: C.brick }, { title: "毕业交付", detail: "Task 25", color: C.brick }], notes: ["后续工作按依赖和质量门禁推进。", "Schedule 和统一评估未通过前不会启动正式实验。", "确认性结果不支持创新时将回到设计或调整声明。"] },
  { kind: "summary", left: "设计完成待实现\n干扰外部性感知的跨 UAV 联合关联与配对", right: "核心已完成并验证\ndeadline 解析需求与最大基数 CPU 分配", footer: "121 passed 仅证明软件规格 · 性能结论等待新版本 canonical 实验", notes: ["中期阶段已经形成两个边界清晰的创新点。", "创新一和创新二的完成状态必须分别表述。", "下一阶段重点是完成调度闭环并形成可复现性能证据。"] },
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
