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
  templateBlue: "1557AE",
  templateBright: "0070C0",
  templatePale: "E9EBF5",
  templatePaleStrong: "CFD5EA",
  templateIce: "DAEDEF",
  templateRed: "FF0000",
};

const BASE = path.resolve(__dirname, "..");
const OUTPUT = path.join(BASE, "outputs");
const ROOT = path.resolve(BASE, "../..");
const FIGURES = path.join(ROOT, "figures", "thesis");
const VISUALS = path.join(BASE, "visuals");
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

function drawApplicationBackground(slide, spec, index, total) {
  const imagePath = path.join(VISUALS, spec.image);
  slide.background = { color: C.white };
  drawTemplateFrame(slide, "选题背景", index, total);

  addText(slide, spec.title, 0.66, 0.88, 5.38, 0.86, {
    fontSize: 27,
    bold: true,
    color: C.templateBlue,
    valign: "top",
  });
  addText(slide, spec.lead, 0.68, 1.86, 5.28, 0.72, {
    fontSize: 14.5,
    color: C.ink,
    valign: "top",
  });

  spec.contexts.forEach((context, itemIndex) => {
    const y = 2.86 + itemIndex * 0.98;
    slide.addShape(SHAPE.ellipse, {
      x: 0.72,
      y: y + 0.03,
      w: 0.42,
      h: 0.42,
      fill: { color: itemIndex === 1 ? C.templateBright : C.templateBlue },
      line: { color: C.white, width: 0.8 },
    });
    addText(slide, `${itemIndex + 1}`, 0.72, y + 0.1, 0.42, 0.22, {
      fontSize: 11,
      bold: true,
      color: C.white,
      align: "center",
    });
    addText(slide, context.title, 1.34, y, 1.34, 0.32, {
      fontSize: 16,
      bold: true,
      color: C.templateBlue,
    });
    addText(slide, context.detail, 2.72, y - 0.01, 3.18, 0.52, {
      fontSize: 12.5,
      color: C.muted,
      valign: "top",
    });
  });

  slide.addShape(SHAPE.rect, {
    x: 0.68,
    y: 6.12,
    w: 5.28,
    h: 0.58,
    fill: { color: C.templateIce },
    line: { color: C.templatePaleStrong, width: 0.8 },
  });
  addText(slide, spec.conclusion, 0.88, 6.24, 4.88, 0.3, {
    fontSize: 14,
    bold: true,
    color: C.templateBlue,
    align: "center",
  });

  slide.addShape(SHAPE.rect, {
    x: 6.3,
    y: 1.08,
    w: 6.45,
    h: 5.62,
    fill: { color: C.templatePale },
    line: { color: C.templatePaleStrong, width: 0.9 },
  });
  slide.addImage({
    path: imagePath,
    x: 6.3,
    y: 1.08,
    w: 6.45,
    h: 3.63,
    altText: "多无人机为应急通信、智能巡检和临时热点提供空中通信与边缘计算服务",
  });
  addText(slide, "多无人机 MEC", 6.72, 4.98, 2.2, 0.38, {
    fontSize: 18,
    bold: true,
    color: C.templateBlue,
  });
  addText(slide, "以机动平台补充固定基础设施，同时提供通信覆盖与近端计算能力。", 6.72, 5.44, 5.55, 0.55, {
    fontSize: 13,
    color: C.ink,
  });
  ["空中接入", "近端算力", "协同覆盖"].forEach((label, itemIndex) => {
    const x = 6.72 + itemIndex * 1.82;
    slide.addShape(SHAPE.rect, {
      x,
      y: 6.13,
      w: 1.52,
      h: 0.42,
      fill: { color: itemIndex === 1 ? C.templateBright : C.templateBlue },
      line: { color: C.white, width: 0.8 },
    });
    addText(slide, label, x, 6.22, 1.52, 0.22, {
      fontSize: 11.5,
      bold: true,
      color: C.white,
      align: "center",
    });
  });
}

function drawBackgroundInfographic(slide, spec, index, total, section) {
  slide.background = { color: C.white };
  drawTemplateFrame(slide, "选题背景", index, total);
  addText(slide, spec.title, 0.66, 0.8, 11.75, 0.62, {
    fontSize: 27,
    bold: true,
    color: C.templateBlue,
  });

  const panels = [
    {
      x: 0.58,
      title: "应用需求",
      fill: C.templatePale,
      header: C.templateBlue,
      items: ["应急巡检", "临时热点", "独立计算任务", "绝对截止时刻"],
      accentIndex: 3,
    },
    {
      x: 4.58,
      title: "系统耦合",
      fill: C.templateIce,
      header: C.templateBright,
      items: ["服务归属", "跨 UAV 同频干扰", "NOMA / SIC", "MEC CPU 动态变化"],
      accentIndex: 1,
    },
    {
      x: 8.58,
      title: "研究目标",
      fill: C.templatePaleStrong,
      header: C.templateBlue,
      items: ["联合决定关联与配对", "联合决定卸载与算力", "统一复核任务完成状态", "最大化按时完成任务数"],
      accentIndex: 3,
    },
  ];

  panels.forEach((panel) => {
    slide.addShape(SHAPE.rect, {
      x: panel.x,
      y: 1.58,
      w: 3.42,
      h: 4.92,
      fill: { color: panel.fill },
      line: { color: panel.header, width: 1.2 },
    });
    slide.addShape(SHAPE.rect, {
      x: panel.x,
      y: 1.58,
      w: 3.42,
      h: 0.66,
      fill: { color: panel.header },
      line: { color: panel.header, width: 0.8 },
    });
    addText(slide, panel.title, panel.x + 0.22, 1.73, 2.98, 0.32, {
      fontSize: 19,
      bold: true,
      color: C.white,
    });
    panel.items.forEach((item, itemIndex) => {
      const y = 2.67 + itemIndex * 0.78;
      slide.addShape(SHAPE.ellipse, {
        x: panel.x + 0.28,
        y: y + 0.07,
        w: 0.17,
        h: 0.17,
        fill: { color: itemIndex === panel.accentIndex ? C.templateRed : panel.header },
        line: { color: itemIndex === panel.accentIndex ? C.templateRed : panel.header },
      });
      addText(slide, item, panel.x + 0.62, y, 2.48, 0.34, {
        fontSize: 14,
        bold: itemIndex === panel.accentIndex,
        color: itemIndex === panel.accentIndex ? C.templateRed : C.ink,
      });
    });
  });

  addArrow(slide, 4.05, 4.0, 4.5, 4.0, C.templateBright, 2.2);
  addArrow(slide, 8.05, 4.0, 8.5, 4.0, C.templateBright, 2.2);

  [5.02, 6.22].forEach((x, uavIndex) => {
    slide.addShape(SHAPE.hexagon, {
      x,
      y: 5.43,
      w: 0.52,
      h: 0.4,
      fill: { color: uavIndex ? C.templateBright : C.templateBlue },
      line: { color: C.white, width: 0.8 },
    });
  });
  slide.addShape(SHAPE.line, {
    x: 5.51,
    y: 5.63,
    w: 0.72,
    h: 0,
    line: { color: C.templateRed, width: 1.8, dash: "dash" },
  });
  addText(slide, "同频外部性", 5.18, 5.94, 1.65, 0.26, {
    fontSize: 10.5,
    bold: true,
    color: C.templateRed,
    align: "center",
  });

  addText(slide, "适用边界：固定悬停、集中式控制、时隙级调度；不优化轨迹与发射功率。", 0.7, 6.72, 10.95, 0.3, {
    fontSize: 11.5,
    color: C.muted,
    align: "center",
  });
}

function drawTemplateFrame(slide, section, index, total) {
  addText(slide, section, 1.032, 0.018, 1.555, 0.63, {
    fontSize: 24,
    bold: true,
    color: C.templateBlue,
    fill: { color: C.white },
  });
  slide.addShape(SHAPE.rect, {
    x: 2.62,
    y: 0.25,
    w: 8.76,
    h: 0.22,
    fill: { color: C.templateBright },
    line: { color: C.templateBright, width: 0.5 },
  });
  slide.addShape(SHAPE.rect, {
    x: 0,
    y: 7.22,
    w: W,
    h: 0.28,
    fill: { color: C.templateBright },
    line: { color: C.templateBright, width: 0.5 },
  });
  addText(slide, `${index}/${total}`, 12.08, 6.84, 0.72, 0.22, {
    fontSize: 9.5,
    color: C.templateBlue,
    align: "right",
  });
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
  addText(slide, "270 passed", 9.82, 2.1, 2.3, 0.72, { fontSize: 27, bold: true, color: C.teal, align: "center" });
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
  drawPipeline(slide, ["源码", "270 tests", "canonical v1", "论文与答辩"], 4.45);
  addStatus(slide, "干净 Python 3.11 环境：270 passed", 4.15, 6.16, 5.05, C.green);
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

function drawNextResearchPlan(slide, spec, index, total) {
  slide.background = { color: C.white };
  drawTemplateFrame(slide, "后续计划", index, total);
  addText(slide, spec.title, 0.66, 0.8, 11.75, 0.62, {
    fontSize: 27, bold: true, color: C.templateBlue,
  });

  const stages = [
    {
      period: "2026.07—09",
      title: "扩展实验与强基线验证",
      tasks: "基础场景已跑通；补充等负载规模、强干扰、异构算力及 CSI 误差，并加入 EDF、Exact-Oracle 及文献适配基线。",
      output: "机制边界、最优差距与文献对比证据",
      color: C.templateBlue,
      fill: C.templatePale,
    },
    {
      period: "2026.10—12",
      title: "论文撰写与成果投稿",
      tasks: "完善理论推导、算法复杂度和图表分析；完成英文学术论文修改与投稿。",
      output: "学位论文初稿与投稿稿件",
      color: C.templateBright,
      fill: C.templateIce,
    },
    {
      period: "2027.01—02",
      title: "学位论文修改定稿",
      tasks: "根据导师意见系统修改；完成格式审查、重复率检测与预评审材料。",
      output: "学位论文送审稿",
      color: C.templateBlue,
      fill: C.templatePaleStrong,
    },
    {
      period: "2027.03—05",
      title: "预答辩与毕业答辩",
      tasks: "完成预答辩整改、答辩材料定稿和风险问题演练；提交最终归档材料。",
      output: "通过专业硕士学位答辩",
      color: C.templateRed,
      fill: C.white,
    },
  ];

  stages.forEach((stage, stageIndex) => {
    const x = 0.62 + stageIndex * 3.12;
    slide.addShape(SHAPE.rect, {
      x, y: 1.74, w: 2.78, h: 4.45,
      fill: { color: stage.fill }, line: { color: stage.color, width: 1.0 },
    });
    slide.addShape(SHAPE.ellipse, {
      x: x + 0.97, y: 1.48, w: 0.84, h: 0.84,
      fill: { color: stage.color }, line: { color: C.white, width: 1.2 },
    });
    addText(slide, String(stageIndex + 1), x + 0.97, 1.69, 0.84, 0.3, {
      fontSize: 18, bold: true, color: C.white, align: "center",
    });
    addText(slide, stage.period, x + 0.18, 2.43, 2.42, 0.32, {
      fontSize: 13, bold: true, color: stage.color, align: "center",
    });
    addText(slide, stage.title, x + 0.18, 2.91, 2.42, 0.56, {
      fontSize: 16, bold: true, color: C.templateBlue, align: "center", valign: "mid",
    });
    addText(slide, "主要任务", x + 0.22, 3.72, 2.34, 0.25, {
      fontSize: 11, bold: true, color: C.muted,
    });
    addText(slide, stage.tasks, x + 0.22, 4.06, 2.34, 0.86, {
      fontSize: 11.2, color: C.ink, valign: "top",
    });
    addText(slide, "阶段交付", x + 0.22, 5.18, 2.34, 0.25, {
      fontSize: 11, bold: true, color: C.muted,
    });
    addText(slide, stage.output, x + 0.22, 5.48, 2.34, 0.46, {
      fontSize: 11.2, bold: true, color: stage.color, align: "center", valign: "mid",
    });
    if (stageIndex < stages.length - 1) {
      addArrow(slide, x + 2.82, 1.9, x + 3.08, 1.9, C.templateBright, 1.2);
    }
  });

  addText(slide, "总体目标：完成可送审的学位论文，形成可投稿的学术成果，并通过专业硕士学位答辩。", 1.0, 6.64, 11.0, 0.3, {
    fontSize: 13, bold: true, color: C.templateRed, align: "center",
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

function drawRelatedResearch(slide, spec, index, total) {
  slide.background = { color: C.white };
  drawTemplateFrame(slide, "研究方案", index, total);
  addText(slide, spec.title, 0.66, 0.8, 11.75, 0.62, {
    fontSize: 27, bold: true, color: C.templateBlue,
  });

  const panels = [
    {
      x: 0.55,
      title: "多 UAV 协同卸载",
      papers: [
        ["Ju et al. · JNCA 2024", "MADDPG 联合卸载与轨迹，优化能耗和时延"],
        ["Zhao et al. · Computer Networks 2024", "TD3 联合协同计算、资源与充电调度"],
      ],
      gap: "重点在轨迹、能耗与长期效用，未显式处理全频复用 NOMA 的跨 UAV 外部性。",
      fill: C.templatePale,
    },
    {
      x: 4.55,
      title: "NOMA 空中 MEC",
      papers: [
        ["Wu et al. · ESWA 2025", "MATD3 联合 NOMA 卸载、三维轨迹与功率"],
        ["Zhang et al. · JSA 2026", "同频多 UAV：关联、轨迹与功率联合优化"],
      ],
      gap: "已考虑 NOMA 与同频多 UAV，但主要目标仍是系统时延或吞吐量。",
      fill: C.templateIce,
    },
    {
      x: 8.55,
      title: "时限与 QoS 卸载",
      papers: [
        ["Chen et al. · DCN 2024", "峰值 AoI 边界与 DDQN，优化违约概率和能耗"],
        ["Islam & Ghose · JSA 2025", "DELTA：子截止时间与异构 MEC 资源调度"],
      ],
      gap: "能够表达时限或统计边界，但尚未与多 UAV 全频 NOMA 调度形成统一完成基数目标。",
      fill: C.templatePaleStrong,
    },
  ];

  panels.forEach((panel, panelIndex) => {
    slide.addShape(SHAPE.rect, {
      x: panel.x, y: 1.55, w: 3.43, h: 4.9,
      fill: { color: panel.fill },
      line: { color: C.templateBlue, width: 1.0 },
    });
    slide.addShape(SHAPE.rect, {
      x: panel.x, y: 1.55, w: 3.43, h: 0.62,
      fill: { color: panelIndex === 1 ? C.templateBright : C.templateBlue },
      line: { color: panelIndex === 1 ? C.templateBright : C.templateBlue, width: 0.6 },
    });
    addText(slide, panel.title, panel.x + 0.18, 1.69, 3.07, 0.3, {
      fontSize: 18, bold: true, color: C.white, align: "center",
    });
    panel.papers.forEach((paper, paperIndex) => {
      const y = 3.4 + paperIndex * 0.82;
      addText(slide, paper[0], panel.x + 0.25, y, 2.93, 0.28, {
        fontSize: 11.5, bold: true, color: C.templateBlue,
      });
      addText(slide, paper[1], panel.x + 0.25, y + 0.31, 2.93, 0.38, {
        fontSize: 10.5, color: C.ink, valign: "top",
      });
    });
    addText(slide, "与本文差异", panel.x + 0.25, 5.08, 1.2, 0.26, {
      fontSize: 11.5, bold: true, color: C.templateRed,
    });
    addText(slide, panel.gap, panel.x + 0.25, 5.43, 2.93, 0.68, {
      fontSize: 11.5, color: C.ink, valign: "top",
    });
  });

  // Three compact, editable research-route illustrations.
  [1.12, 2.08].forEach((x, uavIndex) => {
    slide.addShape(SHAPE.hexagon, {
      x, y: 2.48, w: 0.48, h: 0.38,
      fill: { color: uavIndex ? C.templateBright : C.templateBlue },
      line: { color: C.white, width: 0.8 },
    });
  });
  [0.92, 1.58, 2.26, 2.92].forEach((x, userIndex) => {
    slide.addShape(SHAPE.ellipse, {
      x, y: 3.04, w: 0.16, h: 0.16,
      fill: { color: userIndex % 2 ? C.templateBright : C.templateBlue },
      line: { color: C.white, width: 0.5 },
    });
    const targetX = userIndex < 2 ? 1.36 : 2.32;
    slide.addShape(SHAPE.line, {
      x: x + 0.08, y: 3.02, w: targetX - x - 0.08, h: -0.18,
      line: { color: C.templateBlue, width: 0.8 },
    });
  });

  addText(slide, "任务", 4.92, 2.55, 0.62, 0.28, {
    fontSize: 11, bold: true, color: C.templateBlue, align: "center",
  });
  addArrow(slide, 5.56, 2.69, 6.12, 2.69, C.templateBright, 1.8);
  addText(slide, "卸载", 6.12, 2.55, 0.62, 0.28, {
    fontSize: 11, bold: true, color: C.templateBlue, align: "center",
  });
  addArrow(slide, 6.76, 2.69, 7.32, 2.69, C.templateBright, 1.8);
  slide.addShape(SHAPE.rect, {
    x: 7.34, y: 2.4, w: 0.34, h: 0.58,
    fill: { color: C.templateBlue }, line: { color: C.white, width: 0.7 },
  });
  addText(slide, "CPU", 7.18, 3.08, 0.66, 0.22, {
    fontSize: 10.5, bold: true, color: C.templateBlue, align: "center",
  });

  slide.addShape(SHAPE.line, {
    x: 9.15, y: 2.76, w: 2.5, h: 0,
    line: { color: C.templateBlue, width: 2 },
  });
  [9.28, 10.32, 11.42].forEach((x, itemIndex) => {
    slide.addShape(SHAPE.ellipse, {
      x, y: 2.63, w: 0.26, h: 0.26,
      fill: { color: itemIndex === 2 ? C.templateRed : C.templateBright },
      line: { color: C.white, width: 0.6 },
    });
  });
  addText(slide, "到达", 9.08, 3.02, 0.66, 0.22, { fontSize: 10, color: C.muted, align: "center" });
  addText(slide, "卸载", 10.12, 3.02, 0.66, 0.22, { fontSize: 10, color: C.muted, align: "center" });
  addText(slide, "时限／AoI", 11.06, 3.02, 0.92, 0.22, { fontSize: 10, bold: true, color: C.templateRed, align: "center" });

  slide.addShape(SHAPE.rect, {
    x: 0.72, y: 6.62, w: 11.9, h: 0.42,
    fill: { color: C.templateIce },
    line: { color: C.templatePaleStrong, width: 0.7 },
  });
  addText(slide, spec.conclusion, 0.96, 6.7, 10.72, 0.24, {
    fontSize: 13, bold: true, color: C.templateBlue, align: "center",
  });
}

function drawResearchFramework(slide, spec, index, total) {
  slide.background = { color: C.white };
  drawTemplateFrame(slide, "研究方案", index, total);
  addText(slide, spec.title, 0.66, 0.8, 11.75, 0.62, {
    fontSize: 27, bold: true, color: C.templateBlue,
  });

  slide.addShape(SHAPE.rect, {
    x: 0.55, y: 1.55, w: 2.15, h: 4.88,
    fill: { color: C.templatePale }, line: { color: C.templateBlue, width: 1.0 },
  });
  addText(slide, "系统输入", 0.78, 1.78, 1.69, 0.36, {
    fontSize: 19, bold: true, color: C.templateBlue, align: "center",
  });
  ["任务到达与截止时刻", "信道增益与干扰", "UAV／子载波资源", "本地与 MEC 算力"].forEach((item, itemIndex) => {
    const y = 2.44 + itemIndex * 0.82;
    slide.addShape(SHAPE.rect, {
      x: 0.78, y, w: 1.69, h: 0.5,
      fill: { color: itemIndex % 2 ? C.templateIce : C.white },
      line: { color: C.templatePaleStrong, width: 0.7 },
    });
    addText(slide, item, 0.88, y + 0.1, 1.49, 0.28, {
      fontSize: 11.5, bold: true, color: C.ink, align: "center",
    });
  });

  const modules = [
    {
      y: 1.55,
      title: "创新点一 · IEA-CAP",
      subtitle: "干扰外部性感知的跨无人机关联配对",
      steps: ["显式调度", "三类邻域", "候选生成", "接受／拒绝"],
    },
    {
      y: 4.1,
      title: "候选全网评价 · 内嵌 DD-MCAA",
      subtitle: "评价结果反馈 IEA-CAP 接受准则",
      steps: ["信道／干扰／SIC", "αₘᵢₙ／fₘᵢₙ", "最长前缀", "完成状态", "字典序目标"],
    },
  ];
  modules.forEach((module, moduleIndex) => {
    slide.addShape(SHAPE.rect, {
      x: 3.12, y: module.y, w: 5.38, h: 2.18,
      fill: { color: moduleIndex ? C.templateIce : C.templatePaleStrong },
      line: { color: moduleIndex ? C.templateBright : C.templateBlue, width: 1.1 },
    });
    addText(slide, module.title, 3.38, module.y + 0.2, 2.48, 0.34, {
      fontSize: 17.5, bold: true, color: C.templateBlue,
    });
    addText(slide, module.subtitle, 5.88, module.y + 0.22, 2.34, 0.3, {
      fontSize: 11.5, bold: true, color: C.muted, align: "right",
    });
    const gap = 0.09;
    const boxW = (4.88 - gap * (module.steps.length - 1)) / module.steps.length;
    module.steps.forEach((step, stepIndex) => {
      const x = 3.37 + stepIndex * (boxW + gap);
      slide.addShape(SHAPE.rect, {
        x, y: module.y + 1.07, w: boxW, h: 0.62,
        fill: { color: stepIndex === module.steps.length - 1 ? C.templateBright : C.white },
        line: { color: C.templateBlue, width: 0.7 },
      });
      addText(slide, step, x + 0.04, module.y + 1.24, boxW - 0.08, 0.26, {
        fontSize: 11, bold: true,
        color: stepIndex === module.steps.length - 1 ? C.white : C.templateBlue,
        align: "center",
      });
      if (stepIndex < module.steps.length - 1) {
        addArrow(slide, x + boxW, module.y + 1.38, x + boxW + gap - 0.01, module.y + 1.38, C.templateBlue, 0.9);
      }
    });
  });
  addArrow(slide, 2.72, 3.98, 3.05, 3.98, C.templateBright, 2.2);
  addArrow(slide, 5.81, 3.78, 5.81, 4.04, C.templateBright, 1.7);
  addArrow(slide, 8.18, 4.04, 8.18, 3.78, C.templateRed, 1.7);
  addText(slide, "完成集合与目标反馈", 6.68, 3.78, 1.34, 0.24, {
    fontSize: 9.5, bold: true, color: C.templateRed, align: "center",
  });

  slide.addShape(SHAPE.rect, {
    x: 8.93, y: 2.12, w: 3.85, h: 3.78,
    fill: { color: C.templatePale }, line: { color: C.templateBlue, width: 1.1 },
  });
  addText(slide, "联合输出", 9.2, 2.38, 3.31, 0.34, {
    fontSize: 19, bold: true, color: C.templateBlue, align: "center",
  });
  addText(slide, "EAD-JSRA", 9.52, 2.98, 2.68, 0.48, {
    fontSize: 24, bold: true, color: C.templateBright, align: "center",
  });
  ["服务归属与 NOMA 角色", "卸载比例与 MEC CPU", "唯一任务完成集合", "约束与事件日志"].forEach((item, itemIndex) => {
    addText(slide, item, 9.38, 3.72 + itemIndex * 0.45, 2.95, 0.27, {
      fontSize: 12.5, bold: itemIndex === 2, color: itemIndex === 2 ? C.templateRed : C.ink, align: "center",
    });
  });
  addArrow(slide, 8.54, 3.98, 8.88, 3.98, C.templateBright, 2.2);

  addText(slide, spec.footer, 0.8, 6.67, 10.85, 0.3, {
    fontSize: 12.5, bold: true, color: C.templateBlue, align: "center",
  });
}

function drawResearchContents(slide, spec, index, total) {
  slide.background = { color: C.white };
  drawTemplateFrame(slide, "研究方案", index, total);
  addText(slide, spec.title, 0.66, 0.8, 11.75, 0.62, {
    fontSize: 27, bold: true, color: C.templateBlue,
  });

  const columns = [
    { x: 0.55, title: "研究内容一 · IEA-CAP", fill: C.templatePale, header: C.templateBlue },
    { x: 6.82, title: "研究内容二 · DD-MCAA", fill: C.templateIce, header: C.templateBright },
  ];
  columns.forEach((column) => {
    slide.addShape(SHAPE.rect, {
      x: column.x, y: 1.55, w: 5.96, h: 4.72,
      fill: { color: column.fill }, line: { color: column.header, width: 1.1 },
    });
    slide.addShape(SHAPE.rect, {
      x: column.x, y: 1.55, w: 5.96, h: 0.62,
      fill: { color: column.header }, line: { color: column.header, width: 0.6 },
    });
    addText(slide, column.title, column.x + 0.2, 1.69, 5.56, 0.3, {
      fontSize: 18, bold: true, color: C.white, align: "center",
    });
  });

  // Editable schedule illustration for IEA-CAP.
  addText(slide, "候选调度", 0.92, 2.43, 1.28, 0.28, {
    fontSize: 12, bold: true, color: C.templateBlue, align: "center",
  });
  addText(slide, "全网重评", 4.92, 2.43, 1.1, 0.28, {
    fontSize: 12, bold: true, color: C.templateRed, align: "center",
  });
  for (let r = 0; r < 2; r += 1) {
    for (let c = 0; c < 3; c += 1) {
      const x = 0.92 + c * 0.86;
      const y = 2.83 + r * 0.52;
      slide.addShape(SHAPE.rect, {
        x, y, w: 0.72, h: 0.4,
        fill: { color: (r + c) % 2 ? C.white : C.templatePaleStrong },
        line: { color: C.templateBlue, width: 0.65 },
      });
      addText(slide, `U${r + 1}S${c + 1}`, x, y + 0.09, 0.72, 0.2, {
        fontSize: 9.5, bold: true, color: C.templateBlue, align: "center",
      });
    }
  }
  addArrow(slide, 3.54, 3.35, 4.47, 3.35, C.templateBright, 2.0);
  ["唯一性", "干扰", "SIC", "CPU"].forEach((label, itemIndex) => {
    const x = 4.64 + (itemIndex % 2) * 0.77;
    const y = 2.83 + Math.floor(itemIndex / 2) * 0.52;
    slide.addShape(SHAPE.rect, {
      x, y, w: 0.66, h: 0.4,
      fill: { color: itemIndex === 1 ? C.templateRed : C.templateBlue },
      line: { color: C.white, width: 0.6 },
    });
    addText(slide, label, x, y + 0.09, 0.66, 0.2, {
      fontSize: 9.5, bold: true, color: C.white, align: "center",
    });
  });

  const leftItems = [
    "对象：服务归属、子载波与 NOMA 角色",
    "机制：Replace／Position／CrossUAV",
    "判据：完成数→截止裕量→全网干扰",
    "边界：给定邻域局部最优、有限终止",
  ];
  leftItems.forEach((item, itemIndex) => {
    addText(slide, item, 0.92, 4.02 + itemIndex * 0.47, 5.1, 0.29, {
      fontSize: 12.5, bold: itemIndex === 2, color: itemIndex === 2 ? C.templateRed : C.ink,
    });
  });

  // Editable deadline and CPU-admission illustration for DD-MCAA.
  slide.addShape(SHAPE.line, {
    x: 7.28, y: 3.12, w: 2.48, h: 0,
    line: { color: C.templateBlue, width: 2 },
  });
  [7.42, 8.38, 9.48].forEach((x, itemIndex) => {
    slide.addShape(SHAPE.ellipse, {
      x, y: 2.99, w: 0.26, h: 0.26,
      fill: { color: itemIndex === 2 ? C.templateRed : C.templateBright },
      line: { color: C.white, width: 0.6 },
    });
  });
  addText(slide, "到达", 7.18, 3.38, 0.72, 0.22, { fontSize: 10, color: C.muted, align: "center" });
  addText(slide, "卸载", 8.15, 3.38, 0.72, 0.22, { fontSize: 10, color: C.muted, align: "center" });
  addText(slide, "截止", 9.25, 3.38, 0.72, 0.22, { fontSize: 10, bold: true, color: C.templateRed, align: "center" });
  const cpuWidths = [0.48, 0.68, 0.92];
  let cpuX = 10.28;
  cpuWidths.forEach((width, itemIndex) => {
    slide.addShape(SHAPE.rect, {
      x: cpuX, y: 2.8, w: width, h: 0.64,
      fill: { color: itemIndex < 2 ? C.templateBlue : C.templatePaleStrong },
      line: { color: C.white, width: 0.7 },
    });
    addText(slide, `f${itemIndex + 1}`, cpuX, 2.99, width, 0.2, {
      fontSize: 9.5, bold: true, color: itemIndex < 2 ? C.white : C.templateBlue, align: "center",
    });
    cpuX += width + 0.08;
  });
  addText(slide, "MEC CPU 准入", 10.28, 3.5, 2.15, 0.24, {
    fontSize: 11.5, bold: true, color: C.templateBlue, align: "center",
  });

  const rightItems = [
    "对象：卸载比例与每 UAV MEC CPU",
    "推导：最小卸载比例 αₘᵢₙ 与最小 CPU fₘᵢₙ",
    "准入：按 (fₘᵢₙ, task_id) 排序取最长前缀",
    "边界：每 UAV CPU 预算内、等价值任务最大完成基数",
  ];
  rightItems.forEach((item, itemIndex) => {
    addText(slide, item, 7.18, 4.02 + itemIndex * 0.47, 5.18, 0.29, {
      fontSize: 12.5, bold: itemIndex === 2, color: itemIndex === 2 ? C.templateRed : C.ink,
    });
  });

  addArrow(slide, 3.48, 6.29, 3.48, 6.53, C.templateBright, 1.5);
  addArrow(slide, 9.8, 6.29, 9.8, 6.53, C.templateBright, 1.5);
  slide.addShape(SHAPE.rect, {
    x: 2.18, y: 6.56, w: 8.98, h: 0.42,
    fill: { color: C.templateBlue }, line: { color: C.templateBlue, width: 0.6 },
  });
  addText(slide, spec.footer, 2.42, 6.64, 8.5, 0.24, {
    fontSize: 13, bold: true, color: C.white, align: "center",
  });
}

function drawProgressData(slide, spec, index, total) {
  slide.background = { color: C.white };
  drawTemplateFrame(slide, "研究进展", index, total);
  addText(slide, spec.title, 0.66, 0.8, 11.75, 0.62, {
    fontSize: 27, bold: true, color: C.templateBlue,
  });

  slide.addShape(SHAPE.rect, {
    x: 0.55, y: 1.55, w: 4.78, h: 4.96,
    fill: { color: C.templatePale }, line: { color: C.templateBlue, width: 1.0 },
  });
  addText(slide, "数据生成与证据流", 0.84, 1.8, 4.2, 0.34, {
    fontSize: 19, bold: true, color: C.templateBlue, align: "center",
  });
  const stages = [
    ["冻结配置", "任务、信道、UAV 与资源参数"],
    ["50 个确认性种子", "1000—1049；每个种子独立生成"],
    ["不可变外生轨迹", "同一种子的任务 ID 与信道完全一致"],
    ["统一评价器", "所有方法共用真实服务信道、SIC 与时延口径"],
    ["canonical v1", "manifest、SHA-256、统计量与验收门禁"],
  ];
  stages.forEach((stage, stageIndex) => {
    const y = 2.34 + stageIndex * 0.76;
    slide.addShape(SHAPE.rect, {
      x: 0.88, y, w: 4.12, h: 0.54,
      fill: { color: stageIndex === stages.length - 1 ? C.templateBlue : C.white },
      line: { color: C.templateBlue, width: 0.7 },
    });
    addText(slide, stage[0], 1.04, y + 0.08, 1.35, 0.24, {
      fontSize: 11.5, bold: true,
      color: stageIndex === stages.length - 1 ? C.white : C.templateBlue,
    });
    addText(slide, stage[1], 2.43, y + 0.07, 2.38, 0.32, {
      fontSize: 10.3,
      color: stageIndex === stages.length - 1 ? C.white : C.ink,
      align: "right",
    });
    if (stageIndex < stages.length - 1) {
      addArrow(slide, 2.94, y + 0.55, 2.94, y + 0.72, C.templateBright, 1.1);
    }
  });

  const facts = [
    ["主实验规模", "N=20 · K=2 · S=3\n100 时隙／种子"],
    ["无线与算力", "5 MHz · 2 GHz 载频\n4 GHz CPU／UAV"],
    ["任务负载", "0.4—2.0 Mbit\n500—1000 cycles/bit"],
    ["时限与到达", "相对时限 0.8 s\n绝对截止＝到达＋0.8 s"],
  ];
  facts.forEach((fact, factIndex) => {
    const col = factIndex % 2;
    const row = Math.floor(factIndex / 2);
    const x = 5.72 + col * 3.55;
    const y = 1.55 + row * 1.42;
    slide.addShape(SHAPE.rect, {
      x, y, w: 3.18, h: 1.16,
      fill: { color: factIndex % 2 ? C.templateIce : C.templatePaleStrong },
      line: { color: factIndex % 2 ? C.templateBright : C.templateBlue, width: 0.9 },
    });
    addText(slide, fact[0], x + 0.18, y + 0.16, 1.16, 0.28, {
      fontSize: 12, bold: true, color: C.templateBlue,
    });
    addText(slide, fact[1], x + 1.36, y + 0.13, 1.62, 0.66, {
      fontSize: 12.5, bold: true, color: C.ink, align: "center",
    });
  });

  slide.addShape(SHAPE.rect, {
    x: 5.72, y: 4.5, w: 6.73, h: 2.01,
    fill: { color: C.white }, line: { color: C.templateBlue, width: 1.0 },
  });
  addText(slide, "可信度控制", 5.98, 4.73, 1.5, 0.32, {
    fontSize: 17, bold: true, color: C.templateBlue,
  });
  const controls = [
    "冻结 config SHA-256 与代码 git SHA",
    "同种子按方法配对，禁止重新抽样和确认后调参",
    "任务、时隙、SIC、CPU 与状态语义逐项校验",
    "鲁棒性、负控制与小规模完整枚举共同验证",
  ];
  controls.forEach((control, controlIndex) => {
    const col = controlIndex % 2;
    const row = Math.floor(controlIndex / 2);
    const x = 5.98 + col * 3.18;
    const y = 5.2 + row * 0.58;
    slide.addShape(SHAPE.ellipse, {
      x, y: y + 0.07, w: 0.16, h: 0.16,
      fill: { color: controlIndex === 3 ? C.templateRed : C.templateBright },
      line: { color: C.white, width: 0.5 },
    });
    addText(slide, control, x + 0.28, y, 2.72, 0.36, {
      fontSize: 10.8, bold: controlIndex === 3, color: C.ink,
    });
  });

  addText(slide, spec.footer, 0.82, 6.69, 10.84, 0.27, {
    fontSize: 12.5, bold: true, color: C.templateRed, align: "center",
  });
}

function drawProgressEffects(slide, spec, index, total) {
  slide.background = { color: C.white };
  drawTemplateFrame(slide, "研究进展", index, total);
  addText(slide, spec.title, 0.66, 0.8, 11.75, 0.62, {
    fontSize: 27, bold: true, color: C.templateBlue,
  });

  slide.addShape(SHAPE.rect, {
    x: 0.55, y: 1.5, w: 7.05, h: 4.98,
    fill: { color: C.white }, line: { color: C.templatePaleStrong, width: 0.8 },
  });
  addContainedImage(slide, path.join(FIGURES, spec.figure), 0.72, 1.66, 6.71, 4.65);

  const metrics = [
    ["完整方法", "98.11%", "95% CI [97.94%, 98.26%]"],
    ["deadline 资源准入", "+8.6898 pp", "95% CI [8.3736, 9.0106]"],
    ["相对 ExchangeStable", "+0.1907 pp", "95% CI [0.1304, 0.2567]"],
    ["CrossUAV 邻域", "+0.0487 pp", "95% CI [0.0138, 0.0841]"],
  ];
  metrics.forEach((metric, metricIndex) => {
    const col = metricIndex % 2;
    const row = Math.floor(metricIndex / 2);
    const x = 7.92 + col * 2.43;
    const y = 1.5 + row * 1.62;
    slide.addShape(SHAPE.rect, {
      x, y, w: 2.18, h: 1.36,
      fill: { color: metricIndex === 1 ? C.templateIce : C.templatePale },
      line: { color: metricIndex === 1 ? C.templateBright : C.templateBlue, width: 0.9 },
    });
    addText(slide, metric[0], x + 0.12, y + 0.12, 1.94, 0.26, {
      fontSize: 10.8, bold: true, color: C.templateBlue, align: "center",
    });
    addText(slide, metric[1], x + 0.12, y + 0.48, 1.94, 0.4, {
      fontSize: metricIndex === 1 ? 20 : 22, bold: true,
      color: metricIndex === 1 ? C.templateRed : C.templateBright, align: "center",
    });
    addText(slide, metric[2], x + 0.12, y + 0.98, 1.94, 0.22, {
      fontSize: 9.4, color: C.muted, align: "center",
    });
  });

  slide.addShape(SHAPE.rect, {
    x: 7.92, y: 4.8, w: 4.61, h: 1.0,
    fill: { color: C.templatePaleStrong }, line: { color: C.templateBlue, width: 0.9 },
  });
  addText(slide, "全局接受准则贡献", 8.15, 4.96, 1.83, 0.26, {
    fontSize: 11.5, bold: true, color: C.templateBlue,
  });
  addText(slide, "+0.1179 pp", 10.02, 4.88, 2.17, 0.4, {
    fontSize: 21, bold: true, color: C.templateBright, align: "center",
  });
  addText(slide, "95% CI [0.0697, 0.1645]", 8.15, 5.39, 4.04, 0.22, {
    fontSize: 9.5, color: C.muted, align: "center",
  });
  addText(slide, "0 约束违规  ·  N=20 调度运行时间 P95 40.93 ms", 7.92, 6.09, 4.61, 0.3, {
    fontSize: 11.5, bold: true, color: C.templateBlue, align: "center",
  });
  addText(slide, spec.footer, 0.8, 6.69, 10.85, 0.27, {
    fontSize: 12.5, bold: true, color: C.templateRed, align: "center",
  });
}

function drawProgressAchievements(slide, spec, index, total) {
  slide.background = { color: C.white };
  drawTemplateFrame(slide, "研究进展", index, total);
  addText(slide, spec.title, 0.66, 0.8, 11.75, 0.62, {
    fontSize: 27, bold: true, color: C.templateBlue,
  });

  const panels = [
    {
      x: 0.55, title: "算法成果", fill: C.templatePale,
      items: ["IEA-CAP 跨 UAV 关联配对", "DD-MCAA 截止时间资源准入", "EAD-JSRA 联合调度与分配"],
    },
    {
      x: 4.55, title: "理论与实验成果", fill: C.templateIce,
      items: ["给定邻域有限终止与局部最优边界", "每 UAV 等价值任务最大完成基数", "7／7 门禁通过；小规模枚举差距为 0"],
    },
    {
      x: 8.55, title: "工程与论文成果", fill: C.templatePaleStrong,
      items: ["统一仿真平台与 270 项自动化测试", "canonical v1、manifest 与 SHA-256", "学位论文主体及英文学术论文稿已形成"],
    },
  ];
  panels.forEach((panel, panelIndex) => {
    const header = panelIndex === 1 ? C.templateBright : C.templateBlue;
    slide.addShape(SHAPE.rect, {
      x: panel.x, y: 1.55, w: 3.43, h: 4.9,
      fill: { color: panel.fill }, line: { color: header, width: 1.0 },
    });
    slide.addShape(SHAPE.rect, {
      x: panel.x, y: 1.55, w: 3.43, h: 0.62,
      fill: { color: header }, line: { color: header, width: 0.6 },
    });
    addText(slide, panel.title, panel.x + 0.2, 1.69, 3.03, 0.3, {
      fontSize: 18, bold: true, color: C.white, align: "center",
    });
    panel.items.forEach((item, itemIndex) => {
      const y = 3.72 + itemIndex * 0.72;
      slide.addShape(SHAPE.ellipse, {
        x: panel.x + 0.28, y: y + 0.08, w: 0.17, h: 0.17,
        fill: { color: itemIndex === 2 ? C.templateRed : header },
        line: { color: C.white, width: 0.5 },
      });
      addText(slide, item, panel.x + 0.6, y, 2.53, 0.44, {
        fontSize: 12.3, bold: itemIndex === 2, color: C.ink,
      });
    });
  });

  // Editable icons for algorithm, proof/evidence, and reproducible delivery.
  [1.04, 2.1].forEach((x, itemIndex) => {
    slide.addShape(SHAPE.ellipse, {
      x, y: 2.57, w: 0.72, h: 0.72,
      fill: { color: itemIndex ? C.templateBright : C.templateBlue },
      line: { color: C.white, width: 0.8 },
    });
    addText(slide, itemIndex ? "DD" : "IEA", x, 2.79, 0.72, 0.22, {
      fontSize: 11, bold: true, color: C.white, align: "center",
    });
  });
  addArrow(slide, 1.76, 2.93, 2.08, 2.93, C.templateRed, 1.6);
  addText(slide, "EAD-JSRA", 1.08, 3.32, 1.7, 0.26, {
    fontSize: 12, bold: true, color: C.templateBlue, align: "center",
  });

  ["终止", "最优边界", "门禁"].forEach((label, itemIndex) => {
    const x = 4.88 + itemIndex * 0.96;
    slide.addShape(SHAPE.rect, {
      x, y: 2.56, w: 0.78, h: 0.64,
      fill: { color: itemIndex === 2 ? C.templateRed : C.templateBlue },
      line: { color: C.white, width: 0.7 },
    });
    addText(slide, label, x, 2.77, 0.78, 0.22, {
      fontSize: 10, bold: true, color: C.white, align: "center",
    });
  });
  [0, 1, 2].forEach((itemIndex) => {
    const x = 9.18 + itemIndex * 0.72;
    slide.addShape(SHAPE.rect, {
      x, y: 2.48 + itemIndex * 0.12, w: 1.42, h: 0.64,
      fill: { color: itemIndex === 2 ? C.templateBlue : C.white, transparency: itemIndex === 2 ? 0 : 8 },
      line: { color: C.templateBlue, width: 0.8 },
    });
  });
  addText(slide, "代码 · 数据 · 论文", 9.3, 3.34, 2.3, 0.28, {
    fontSize: 11.5, bold: true, color: C.templateBlue, align: "center",
  });

  addText(slide, spec.footer, 0.82, 6.69, 10.83, 0.27, {
    fontSize: 12.5, bold: true, color: C.templateRed, align: "center",
  });
}

function drawEvidenceBasis(slide, spec, index, total) {
  slide.background = { color: C.white };
  drawTemplateFrame(slide, "研究进展", index, total);
  addText(slide, spec.title, 0.66, 0.8, 11.75, 0.62, {
    fontSize: 27, bold: true, color: C.templateBlue,
  });

  const panels = [
    { x: 0.55, w: 4.08, title: "关键参数设置", fill: C.templatePale, header: C.templateBlue },
    { x: 4.84, w: 3.9, title: "仿真实施流程", fill: C.templateIce, header: C.templateBright },
    { x: 8.95, w: 3.83, title: "可信性控制", fill: C.templatePaleStrong, header: C.templateBlue },
  ];
  panels.forEach((panel) => {
    slide.addShape(SHAPE.rect, {
      x: panel.x, y: 1.55, w: panel.w, h: 4.95,
      fill: { color: panel.fill }, line: { color: panel.header, width: 1.0 },
    });
    slide.addShape(SHAPE.rect, {
      x: panel.x, y: 1.55, w: panel.w, h: 0.62,
      fill: { color: panel.header }, line: { color: panel.header, width: 0.6 },
    });
    addText(slide, panel.title, panel.x + 0.18, 1.69, panel.w - 0.36, 0.3, {
      fontSize: 17.5, bold: true, color: C.white, align: "center",
    });
  });

  const parameterRows = [
    ["信道模型", "Urban LoS概率与附加损耗采用Al-Hourani 2014；UAV场景边界参考3GPP TR 36.777"],
    ["物理层", "2 GHz，5 MHz，-174 dBm/Hz，NF=7 dB，终端功率 0.2 W"],
    ["系统规模", "800 m×800 m，H=100 m，K=2，N=20，S=3；每块最多2用户"],
    ["任务与算力", "0.4—2.0 Mbit，500—1000 cycles/bit，本地1 GHz，MEC 4 GHz/UAV"],
    ["实验规模", "相对时限0.8 s；50种子×100时隙；确认前冻结配置与分析规则"],
  ];
  parameterRows.forEach((row, rowIndex) => {
    const y = 2.34 + rowIndex * 0.77;
    addText(slide, row[0], 0.82, y, 0.88, 0.28, {
      fontSize: 11, bold: true, color: rowIndex === 4 ? C.templateRed : C.templateBlue,
    });
    addText(slide, row[1], 1.72, y - 0.01, 2.6, 0.49, {
      fontSize: 10.2, bold: rowIndex === 4, color: C.ink, valign: "top",
    });
  });

  const simulationSteps = [
    ["1", "冻结配置与种子", "读取 TOML；种子1000—1049"],
    ["2", "生成外生轨迹", "簇状位置、2 m/s随机游走、突发到达、任务负载"],
    ["3", "计算任务与信道", "生成唯一 task_id、真实信道增益和绝对截止时刻"],
    ["4", "同轨迹算法比较", "全部方法共享任务/信道；统一重算干扰、SIC、CPU和时延"],
    ["5", "统计与发布", "配对bootstrap、Wilcoxon-Holm；校验后发布canonical v1"],
  ];
  simulationSteps.forEach((step, stepIndex) => {
    const y = 2.31 + stepIndex * 0.78;
    slide.addShape(SHAPE.ellipse, {
      x: 5.12, y: y + 0.04, w: 0.34, h: 0.34,
      fill: { color: stepIndex === 4 ? C.templateRed : C.templateBright },
      line: { color: C.white, width: 0.6 },
    });
    addText(slide, step[0], 5.12, y + 0.11, 0.34, 0.18, {
      fontSize: 10, bold: true, color: C.white, align: "center",
    });
    addText(slide, step[1], 5.62, y, 1.28, 0.25, {
      fontSize: 11, bold: true, color: C.templateBlue,
    });
    addText(slide, step[2], 6.9, y - 0.01, 1.52, 0.43, {
      fontSize: 9.8, color: C.ink, align: "right", valign: "top",
    });
    if (stepIndex < simulationSteps.length - 1) {
      addArrow(slide, 5.29, y + 0.4, 5.29, y + 0.73, C.templateBright, 1.0);
    }
  });

  const realityBlocks = [
    ["模型依据", "空地传播采用 Al-Hourani Urban 模型，场景参数参考 3GPP TR 36.777。", C.templateBlue],
    ["公平比较", "所有算法共享相同任务、位置、移动轨迹和信道 realization。", C.templateBright],
    ["统计检验", "采用同种子配对、bootstrap 置信区间和 Wilcoxon-Holm 检验。", C.templateBlue],
    ["稳健性检验", "覆盖 N、K、噪声、CSI、SIC、CPU、时限及系统负载变化。", C.templateRed],
  ];
  realityBlocks.forEach((block, blockIndex) => {
    const y = 2.34 + blockIndex * 0.94;
    addText(slide, block[0], 9.25, y, 1.12, 0.28, {
      fontSize: 11.5, bold: true, color: block[2],
    });
    addText(slide, block[1], 10.36, y - 0.01, 2.08, 0.62, {
      fontSize: 10.3, bold: blockIndex === 3, color: C.ink, valign: "top",
    });
  });

  addText(slide, spec.footer, 0.82, 6.69, 10.84, 0.27, {
    fontSize: 12.5, bold: true, color: C.templateRed, align: "center",
  });
}

function drawInnovationOneEvidence(slide, spec, index, total) {
  slide.background = { color: C.white };
  drawTemplateFrame(slide, "研究进展", index, total);
  addText(slide, spec.title, 0.66, 0.8, 11.75, 0.62, {
    fontSize: 27, bold: true, color: C.templateBlue,
  });
  slide.addShape(SHAPE.rect, {
    x: 0.55, y: 1.5, w: 7.12, h: 4.98,
    fill: { color: C.white }, line: { color: C.templatePaleStrong, width: 0.8 },
  });
  addContainedImage(slide, path.join(FIGURES, spec.figure), 0.72, 1.66, 6.78, 4.62);

  const cards = [
    ["CrossUAV 邻域独立贡献", "+0.0487 pp", "95% CI [0.0138, 0.0841]  ·  Holm p=0.0201"],
    ["全局接受准则独立贡献", "+0.1179 pp", "95% CI [0.0697, 0.1645]  ·  Holm p=0.0004"],
  ];
  cards.forEach((card, cardIndex) => {
    const y = 1.5 + cardIndex * 1.5;
    slide.addShape(SHAPE.rect, {
      x: 7.96, y, w: 4.58, h: 1.26,
      fill: { color: cardIndex ? C.templateIce : C.templatePale },
      line: { color: cardIndex ? C.templateBright : C.templateBlue, width: 0.9 },
    });
    addText(slide, card[0], 8.18, y + 0.14, 2.25, 0.28, {
      fontSize: 12, bold: true, color: C.templateBlue,
    });
    addText(slide, card[1], 10.46, y + 0.12, 1.78, 0.42, {
      fontSize: 21, bold: true, color: C.templateBright, align: "center",
    });
    addText(slide, card[2], 8.18, y + 0.74, 4.05, 0.26, {
      fontSize: 9.8, color: C.muted, align: "center",
    });
  });

  slide.addShape(SHAPE.rect, {
    x: 7.96, y: 4.56, w: 4.58, h: 1.9,
    fill: { color: C.templatePaleStrong }, line: { color: C.templateBlue, width: 0.9 },
  });
  addText(slide, "统计结果", 8.2, 4.78, 4.1, 0.3, {
    fontSize: 16, bold: true, color: C.templateBlue, align: "center",
  });
  const evidence = [
    "严格 2×2 消融，分别控制邻域范围与接受准则",
    "50 个确认性种子采用同种子配对比较",
    "两项增益的 95% 置信区间均严格大于 0",
    "最终调度满足 SIC 与 MEC CPU 预算约束",
  ];
  evidence.forEach((item, itemIndex) => {
    const y = 5.25 + itemIndex * 0.29;
    slide.addShape(SHAPE.ellipse, {
      x: 8.18, y: y + 0.07, w: 0.13, h: 0.13,
      fill: { color: itemIndex === 3 ? C.templateRed : C.templateBright },
      line: { color: C.white, width: 0.4 },
    });
    addText(slide, item, 8.43, y, 3.8, 0.23, {
      fontSize: 10.2, bold: itemIndex === 3, color: C.ink,
    });
  });
  addText(slide, spec.footer, 0.8, 6.69, 10.86, 0.27, {
    fontSize: 12.5, bold: true, color: C.templateRed, align: "center",
  });
}

function drawInnovationTwoEvidence(slide, spec, index, total) {
  slide.background = { color: C.white };
  drawTemplateFrame(slide, "研究进展", index, total);
  addText(slide, spec.title, 0.66, 0.8, 11.75, 0.62, {
    fontSize: 27, bold: true, color: C.templateBlue,
  });
  slide.addShape(SHAPE.rect, {
    x: 0.55, y: 1.5, w: 7.12, h: 4.98,
    fill: { color: C.white }, line: { color: C.templatePaleStrong, width: 0.8 },
  });
  addContainedImage(slide, path.join(FIGURES, spec.figure), 0.72, 1.66, 6.78, 4.62);

  const rates = [
    ["DD-MCAA", "98.11%", C.templateBlue],
    ["Crossing／Equal", "89.42%", C.templateBright],
    ["Fixed 0.5／Equal", "76.13%", C.templateBlue],
  ];
  rates.forEach((rate, rateIndex) => {
    const y = 1.5 + rateIndex * 0.82;
    slide.addShape(SHAPE.rect, {
      x: 7.96, y, w: 4.58, h: 0.66,
      fill: { color: rateIndex === 0 ? C.templatePaleStrong : C.white },
      line: { color: rate[2], width: 0.8 },
    });
    addText(slide, rate[0], 8.18, y + 0.15, 2.4, 0.26, {
      fontSize: 12.5, bold: true, color: C.templateBlue,
    });
    addText(slide, rate[1], 10.62, y + 0.11, 1.58, 0.34, {
      fontSize: 18, bold: true, color: rateIndex === 0 ? C.templateRed : C.templateBright, align: "center",
    });
  });
  slide.addShape(SHAPE.rect, {
    x: 7.96, y: 4.08, w: 4.58, h: 1.08,
    fill: { color: C.templateIce }, line: { color: C.templateBright, width: 0.9 },
  });
  addText(slide, "相对 Crossing／Equal", 8.18, 4.25, 1.9, 0.28, {
    fontSize: 11.5, bold: true, color: C.templateBlue,
  });
  addText(slide, "+8.6898 pp", 10.1, 4.17, 2.08, 0.42, {
    fontSize: 21, bold: true, color: C.templateRed, align: "center",
  });
  addText(slide, "95% CI [8.3736, 9.0106] · Holm p<1.6×10⁻¹⁴ · rank-biserial=1.0", 8.18, 4.71, 4.02, 0.25, {
    fontSize: 9.4, color: C.muted, align: "center",
  });

  slide.addShape(SHAPE.rect, {
    x: 7.96, y: 5.4, w: 4.58, h: 1.06,
    fill: { color: C.templatePale }, line: { color: C.templateBlue, width: 0.9 },
  });
  addText(slide, "完整方法综合结果", 8.18, 5.54, 1.9, 0.28, {
    fontSize: 12, bold: true, color: C.templateBlue,
  });
  addText(slide, "按时完成率 98.11% · 相对强基线 +0.1907 pp\n0 约束违规 · N=20 调度时间 P95 40.93 ms", 8.18, 5.86, 4.02, 0.45, {
    fontSize: 10.2, bold: true, color: C.ink, align: "center",
  });
  addText(slide, spec.footer, 0.8, 6.69, 10.86, 0.27, {
    fontSize: 12.5, bold: true, color: C.templateRed, align: "center",
  });
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
    else if (spec.kind === "applicationBackground") {
      drawApplicationBackground(slide, spec, idx + 1, specs.length);
    } else if (spec.kind === "backgroundInfographic") {
      drawBackgroundInfographic(slide, spec, idx + 1, specs.length, meta.section);
    } else if (spec.kind === "relatedResearch") {
      drawRelatedResearch(slide, spec, idx + 1, specs.length);
    } else if (spec.kind === "researchFramework") {
      drawResearchFramework(slide, spec, idx + 1, specs.length);
    } else if (spec.kind === "researchContents") {
      drawResearchContents(slide, spec, idx + 1, specs.length);
    } else if (spec.kind === "progressData") {
      drawProgressData(slide, spec, idx + 1, specs.length);
    } else if (spec.kind === "progressEffects") {
      drawProgressEffects(slide, spec, idx + 1, specs.length);
    } else if (spec.kind === "progressAchievements") {
      drawProgressAchievements(slide, spec, idx + 1, specs.length);
    } else if (spec.kind === "evidenceBasis") {
      drawEvidenceBasis(slide, spec, idx + 1, specs.length);
    } else if (spec.kind === "innovationOneEvidence") {
      drawInnovationOneEvidence(slide, spec, idx + 1, specs.length);
    } else if (spec.kind === "innovationTwoEvidence") {
      drawInnovationTwoEvidence(slide, spec, idx + 1, specs.length);
    } else if (spec.kind === "nextResearchPlan") {
      drawNextResearchPlan(slide, spec, idx + 1, specs.length);
    }
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

async function enforceManualPlayback(filePath) {
  const zip = await JSZip.loadAsync(fs.readFileSync(filePath));
  const slideNames = Object.keys(zip.files).filter((name) => /^ppt\/slides\/slide\d+\.xml$/.test(name));
  for (const slideName of slideNames) {
    let xml = await zip.file(slideName).async("string");
    xml = xml
      .replace(/<p:transition\b[^>]*\/>/g, "")
      .replace(/<p:transition\b[^>]*>[\s\S]*?<\/p:transition>/g, "")
      .replace(/<p:timing\b[^>]*>[\s\S]*?<\/p:timing>/g, "");
    zip.file(slideName, xml);
  }

  const propsPath = "ppt/presProps.xml";
  let props = await zip.file(propsPath).async("string");
  props = props
    .replace(/<p:showPr\b[^>]*\/>/g, "")
    .replace(/<p:showPr\b[^>]*>[\s\S]*?<\/p:showPr>/g, "");
  if (/<p:presentationPr\b[^>]*\/>/.test(props)) {
    props = props.replace(
      /<p:presentationPr\b([^>]*)\/>/,
      '<p:presentationPr$1><p:showPr useTimings="0"/></p:presentationPr>',
    );
  } else {
    props = props.replace(
      "</p:presentationPr>",
      '<p:showPr useTimings="0"/></p:presentationPr>',
    );
  }
  zip.file(propsPath, props);
  fs.writeFileSync(filePath, await zip.generateAsync({
    type: "nodebuffer",
    compression: "DEFLATE",
    compressionOptions: { level: 6 },
  }));
}

const midterm = [
  { kind: "title", title: "面向时限任务的多无人机移动边缘计算干扰感知调度与卸载资源分配方法研究", subtitle: "专业学位硕士研究生中期答辩", badge: "canonical v1 · 7/7 门禁通过", notes: ["本次汇报围绕多无人机移动边缘计算中的时限任务处理问题展开。", "研究重点是跨无人机干扰外部性与有限边缘算力竞争。", "汇报内容依次介绍选题背景、研究方法、阶段性进展和后续计划。"] },
  { kind: "applicationBackground", image: "00_application_context.png", title: "时限任务需要可快速部署的空中边缘计算", lead: "固定基础设施难以覆盖突发、偏远和临时场景；多无人机 MEC 可同步提供空中接入与近端算力。", contexts: [{ title: "应急通信", detail: "地面设施受损时，快速恢复覆盖与现场计算能力。" }, { title: "智能巡检", detail: "图像与感知任务需在规定时限内完成分析并返回结果。" }, { title: "临时热点", detail: "突发用户密度使无线资源与边缘算力同时承压。" }], conclusion: "应用诉求：快速部署、广域覆盖、低时延任务处理", notes: ["应急通信、智能巡检和临时热点具有共同特征，即基础设施条件不稳定且任务具有明确时限。", "无人机能够快速形成空中接入点，搭载 MEC 服务器后还可在网络边缘提供近端计算能力。", "多无人机协同能够扩展覆盖范围，但无线接入与边缘算力必须同时配置。", "因此，本研究首先关注如何在突发和资源受限场景中保障更多任务按时完成。"] },
  { kind: "backgroundInfographic", image: "01_background.png", title: "选题背景：多无人机全频复用下的任务时限矛盾", notes: ["多无人机全频复用能够提高频谱利用率，但不同接收端之间形成持续的同频干扰耦合。", "NOMA 用户配对进一步引入接收功率顺序和硬 SIC 约束，用户改变服务无人机时必须重新评价全网信道。", "无线速率决定卸载时延，有限 MEC CPU 又决定任务能否在截止时刻前完成。", "研究目标因此不是单纯提高瞬时速率，而是在统一约束下最大化按时完成的唯一任务数。"] },
  { kind: "relatedResearch", title: "相关研究：2024—2026 年关键工作与本文定位", conclusion: "近两年进展：多 UAV、NOMA 与 deadline 分别深化；本文聚焦三者交叉处的可审计联合优化。", notes: ["2024 年 Ju 等在 JNCA 采用 MADDPG 联合优化多 UAV 卸载与轨迹，Zhao 等在 Computer Networks 进一步考虑协同计算、资源与充电调度。", "2025 年 Wu 等在 ESWA 研究多 UAV MEC-NOMA 的卸载、轨迹与功率，2026 年 Zhang 等在 JSA 建模同频多 UAV NOMA 并联合优化关联、轨迹和功率。", "时限方向上，Chen 等在 2024 年以峰值 AoI 边界约束卸载，Islam 和 Ghose 在 2025 年提出 DELTA 处理子截止时间与异构 MEC 资源。", "这些工作分别强化多 UAV 协同、NOMA 资源管理或时限约束，但尚未共同解决跨 UAV 干扰外部性与绝对截止时刻完成基数。"] },
  { kind: "researchFramework", title: "研究框架：调度外部性与截止时间资源准入协同优化", footer: "统一评价口径：所有候选调度均重算真实服务信道、干扰、SIC、MEC 负载与任务完成状态。", notes: ["研究框架从任务、信道、无线资源和计算资源四类输入开始。", "IEA-CAP 通过显式调度和三类邻域生成候选调度，但并不直接接受候选。", "每个候选都进入内嵌 DD-MCAA 的全网评价，由最小资源准入得到完成集合与字典序目标，再反馈 IEA-CAP 决定接受或拒绝。", "闭环最终形成 EAD-JSRA，统一输出服务归属、卸载比例、算力分配和唯一任务完成集合。"] },
  { kind: "researchContents", title: "研究内容：两个可独立验证的算法创新点", footer: "EAD-JSRA：IEA-CAP 调度结构 + DD-MCAA 资源准入 + 统一时延复核", notes: ["研究内容一面向服务归属、子载波和 NOMA 角色，设计 IEA-CAP 跨无人机关联配对方法。", "该方法使用 Replace、Position 和 CrossUAV 三类邻域，并按照完成数、截止裕量和全网干扰进行字典序接受。", "研究内容二推导最小卸载比例和最小 MEC CPU，按最小成本稳定排序后选择最长可行前缀。", "两个模块分别具有给定邻域有限终止，以及每 UAV CPU 预算内等价值任务最大完成基数的清晰理论边界。"] },
  { kind: "evidenceBasis", title: "实验设置与仿真流程", footer: "50 个确认性种子 · 100 时隙/种子 · 全部算法共享相同任务与信道轨迹", notes: ["本页汇报实验参数、仿真实施步骤和可信性控制。Urban 空地 LoS 概率和附加损耗采用 Al-Hourani 模型，并参考 3GPP TR 36.777；热噪声、载频、带宽和终端功率均使用明确物理单位。", "区域、用户数、无人机数、子载波数、任务范围和 0.8 秒相对时限属于受控仿真设置，用于形成无线与计算资源同时竞争的非平凡场景，并非某一城市的外场实测标定。", "仿真从冻结 TOML 和确认性种子开始，生成簇状用户、随机游走、突发到达、任务负载和真实信道；所有方法共享同一轨迹，并由统一评价器重算干扰、SIC、CPU 和截止状态。", "因此，当前结果用于验证统一模型下的算法相对性能；后续仍需通过外场数据或半实物平台进一步检验部署有效性。"] },
  { kind: "innovationOneEvidence", title: "IEA-CAP 消融实验结果", figure: "pairing_ablation.png", footer: "实验结论：CrossUAV 邻域与全网接受准则均产生统计显著的独立增益。", notes: ["左图是创新点一的严格二乘二消融，分别切换是否允许 CrossUAV 邻域，以及使用双边还是全网接受准则，其余初始化、资源分配和评价器保持一致。", "在全局接受准则下加入 CrossUAV 邻域，按时完成率独立提高 0.0487 个百分点，95% 置信区间为 0.0138 到 0.0841。", "在允许 CrossUAV 邻域时由双边准则切换到全网准则，独立提高 0.1179 个百分点，95% 置信区间为 0.0697 到 0.1645。", "两项结果均采用 50 个确认性种子的同种子配对比较；K=1 和耦合扫描仅用于界定适用边界，不作单调因果外推。"] },
  { kind: "innovationTwoEvidence", title: "DD-MCAA 资源分配受控消融实验", figure: "resource_ablation.png", footer: "实验定位：在相同无线调度下隔离 deadline 驱动准入的独立贡献；不作为 SOTA 对比。", notes: ["本页是创新点二的受控消融，不是 SOTA 比较。三种策略共享完全相同的无线调度，仅替换卸载比例与 CPU 分配规则。", "DD-MCAA 的按时完成率为 98.11%，Crossing 加平均 CPU 为 89.42%，固定一半卸载加平均 CPU 为 76.13%。", "相对 Crossing 策略，DD-MCAA 提高 8.6898 个百分点，95% 置信区间为 8.3736 到 9.0106，50 个确认性种子的差值方向完全一致。", "后续将加入 EDF、Min-Slack、Exact-Oracle 和近年文献适配算法，分别补充经典调度、最优差距和文献先进方法比较。"] },
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
  { kind: "nextResearchPlan", title: "下一步研究计划", notes: ["目前已跑通 K=2、N=20 的基础实验场景，但网络规模与竞争压力仍不足。下一阶段将补充等负载规模扩展、高并发强干扰、异构算力和 CSI 误差实验，并加入 EDF、Min-Slack、Exact-Oracle 和近年文献适配算法，分别验证机制边界、最优差距及相对文献先进方法的性能。", "在实验闭环基础上完善学位论文理论推导、复杂度分析和结果讨论，同时完成英文学术论文修改与投稿，以同行评审检验研究质量。", "2027 年初根据导师与评审意见完成学位论文系统修改、格式审查和送审材料准备。", "最后完成预答辩整改、答辩材料定稿和风险问题演练，确保研究内容、实验结论和创新点表述保持一致。"] },
  { kind: "summary", left: "已实现并独立验证\n跨 UAV 外部性调度\n+0.05 pp / +0.12 pp", right: "已实现并独立验证\ndeadline 最小资源分配\n+8.69 pp", footer: "98.11% 按时完成 · 0 约束违规 · 270 tests · 7/7 gates", notes: ["两个创新点边界清晰且均有正式数据支持。", "创新点二承担主要性能贡献。", "当前项目已进入专业硕士论文和毕业答辩定稿阶段。"] },
];

async function main() {
  const midtermDeck = makeDeck({ title: "中期答辩", section: "中期答辩" }, midterm);
  const midtermPath = path.join(OUTPUT, "中期答辩_关铭锡.pptx");
  await midtermDeck.writeFile({ fileName: midtermPath });
  await repairNotesMasterOrder(midtermPath);
  await enforceManualPlayback(midtermPath);
  console.log(`[generated] midterm=${midterm.length} slides`);
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});
