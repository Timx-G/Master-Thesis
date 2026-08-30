const fs = require("fs");
const path = require("path");
const JSZip = require("jszip");

const BASE = path.resolve(__dirname, "..");
function decodeXml(text) {
  return text
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'");
}

function xmlText(xml) {
  return [...xml.matchAll(/<a:t>([\s\S]*?)<\/a:t>/g)].map((match) => decodeXml(match[1])).join("\n");
}

async function checkDeck(fileName, expectedSlides) {
  const filePath = path.join(BASE, "outputs", fileName);
  if (!fs.existsSync(filePath)) throw new Error(`${fileName}: output missing`);

  const zip = await JSZip.loadAsync(fs.readFileSync(filePath));
  const presentationXml = await zip.file("ppt/presentation.xml").async("string");
  const notesMasterPos = presentationXml.indexOf("<p:notesMasterIdLst>");
  const slideListPos = presentationXml.indexOf("<p:sldIdLst>");
  if (notesMasterPos < 0 || slideListPos < 0 || notesMasterPos > slideListPos) {
    throw new Error(`${fileName}: notesMasterIdLst must precede sldIdLst`);
  }
  const slideNames = Object.keys(zip.files)
    .filter((name) => /^ppt\/slides\/slide\d+\.xml$/.test(name))
    .sort((a, b) => Number(a.match(/\d+/)[0]) - Number(b.match(/\d+/)[0]));
  const noteNames = Object.keys(zip.files)
    .filter((name) => /^ppt\/notesSlides\/notesSlide\d+\.xml$/.test(name))
    .sort((a, b) => Number(a.match(/\d+/)[0]) - Number(b.match(/\d+/)[0]));

  if (slideNames.length !== expectedSlides) {
    throw new Error(`${fileName}: expected ${expectedSlides} slides, found ${slideNames.length}`);
  }
  if (noteNames.length !== expectedSlides) {
    throw new Error(`${fileName}: expected notes on every slide, found ${noteNames.length}`);
  }

  const slideXmls = await Promise.all(slideNames.map((name) => zip.file(name).async("string")));
  const slideTexts = slideXmls.map(xmlText);
  const slideText = slideTexts.join("\n");
  const noteTexts = (await Promise.all(noteNames.map((name) => zip.file(name).async("string")))).map(xmlText);

  if (/\{\{[A-Z0-9_]+\}\}|待新版本|121 passed|fd0cb7c|设计完成待实现|性能尚未生成/.test(slideText)) {
    throw new Error(`${fileName}: stale placeholder or pre-experiment statement found`);
  }
  if (/73\.4%|82\.5%|68\.4%/.test(slideText)) throw new Error(`${fileName}: legacy performance number found`);
  if (!slideText.includes("创新点一") || !slideText.includes("创新点二")) {
    throw new Error(`${fileName}: innovation labels missing`);
  }
  const approvedTitle = "面向时限任务的多无人机移动边缘计算干扰感知调度与卸载资源分配方法研究";
  if (!slideTexts[0].includes(approvedTitle)) {
    throw new Error(`${fileName}: approved thesis title missing from cover`);
  }
  for (const phrase of ["应急通信", "智能巡检", "临时热点", "快速部署、广域覆盖、低时延任务处理"]) {
    if (!slideTexts[1].includes(phrase)) {
      throw new Error(`${fileName}: slide 2 missing background phrase ${phrase}`);
    }
  }
  if (!slideTexts[2].includes("多无人机全频复用下的任务时限矛盾")) {
    throw new Error(`${fileName}: slide 3 background title missing`);
  }
  for (const [slideIndex, xml] of [1, 2, 3, 4, 5, 6, 7, 8].map((index) => [index, slideNames[index]])) {
    const slideXml = await zip.file(xml).async("string");
    for (const color of ["1557AE", "0070C0"]) {
      if (!slideXml.includes(`val="${color}"`)) {
        throw new Error(`${fileName}: slide ${slideIndex + 1} missing reference-template color ${color}`);
      }
    }
  }
  const slide3Xml = await zip.file(slideNames[2]).async("string");
  for (const color of ["E9EBF5", "CFD5EA", "DAEDEF"]) {
    if (!slide3Xml.includes(`val="${color}"`)) {
      throw new Error(`${fileName}: slide 3 missing reference-template pale color ${color}`);
    }
  }
  if (/val="(?:0F5B5D|2F7D58|8F2633)"/.test(slide3Xml)) {
    throw new Error(`${fileName}: slide 3 retains colors from the previous theme`);
  }
  const researchSchemeText = slideTexts.slice(3, 6).join("\n");
  for (const phrase of ["相关研究", "研究框架", "研究内容", "IEA-CAP", "DD-MCAA", "EAD-JSRA"]) {
    if (!researchSchemeText.includes(phrase)) {
      throw new Error(`${fileName}: research-scheme slides missing phrase ${phrase}`);
    }
  }
  for (const phrase of ["2024—2026 年关键工作", "Ju et al. · JNCA 2024", "Wu et al. · ESWA 2025", "Zhang et al. · JSA 2026", "Islam & Ghose · JSA 2025", "内嵌 DD-MCAA", "完成集合与目标反馈", "每 UAV CPU 预算内"]) {
    if (!researchSchemeText.includes(phrase)) {
      throw new Error(`${fileName}: research-scheme evidence or boundary missing phrase ${phrase}`);
    }
  }
  if (/\b(?:2016|2017|2018|2019|2022)\b/.test(slideTexts[3])) {
    throw new Error(`${fileName}: related-research slide contains literature outside the 2024-2026 window`);
  }
  for (let index = 3; index <= 5; index += 1) {
    const slideXml = await zip.file(slideNames[index]).async("string");
    if (/val="(?:0F5B5D|2F7D58|8F2633)"/.test(slideXml)) {
      throw new Error(`${fileName}: slide ${index + 1} retains colors from the previous theme`);
    }
  }
  const progressText = slideTexts.slice(6, 9).join("\n");
  for (const phrase of ["实验设置与仿真流程", "Al-Hourani 2014", "3GPP TR 36.777", "全部算法共享相同任务与信道轨迹", "IEA-CAP 消融实验结果", "+0.0487 pp", "+0.1179 pp", "DD-MCAA 资源分配对比结果", "98.11%", "89.42%", "76.13%", "+8.6898 pp", "完整方法综合结果"]) {
    if (!progressText.includes(phrase)) {
      throw new Error(`${fileName}: research-progress slides missing phrase ${phrase}`);
    }
  }
  const nextPlanText = slideTexts[20];
  for (const phrase of ["下一步研究计划", "外部有效性补强", "复杂移动性", "公开轨迹或半实物验证", "论文撰写与成果投稿", "学位论文修改定稿", "预答辩与毕业答辩", "通过专业硕士学位答辩"]) {
    if (!nextPlanText.includes(phrase)) {
      throw new Error(`${fileName}: next-research-plan slide missing phrase ${phrase}`);
    }
  }
  for (let index = 6; index <= 8; index += 1) {
    if (/val="(?:0F5B5D|2F7D58|8F2633)"/.test(slideXmls[index])) {
      throw new Error(`${fileName}: slide ${index + 1} retains colors from the previous theme`);
    }
  }
  const playbackXmlNames = Object.keys(zip.files).filter((name) => /^ppt\/.*\.xml$/.test(name));
  const playbackXml = (await Promise.all(playbackXmlNames.map((name) => zip.file(name).async("string")))).join("\n");
  if (/<p:(?:transition|timing|anim|animClr|animEffect|animMotion|animRot|animScale|cmd|seq|par|cTn|tnLst|bldLst)\b|<a:hlinkClick\b|\baction="|\badvTm="|\badvClick="/.test(playbackXml)) {
    throw new Error(`${fileName}: animation, transition, or automatic-advance timing found`);
  }
  const presProps = await zip.file("ppt/presProps.xml").async("string");
  if (!/<p:showPr\b[^>]*useTimings="0"/.test(presProps)) {
    throw new Error(`${fileName}: manual playback setting useTimings=0 missing`);
  }
  if (/问题收缩|退出轨迹\/DRL|Tasks 1-24/.test(`${slideTexts[1]}\n${slideTexts[2]}`)) {
    throw new Error(`${fileName}: first two content slides contain implementation-history wording`);
  }
  for (const evidence of ["98.11%", "+8.6898 pp", "270 passed", "7 / 7", "P95 40.93 ms"]) {
    if (!slideText.includes(evidence)) throw new Error(`${fileName}: missing formal evidence ${evidence}`);
  }
  noteTexts.forEach((text, index) => {
    const sentenceCount = (text.match(/[。！？]/g) || []).length;
    if (sentenceCount < 3) throw new Error(`${fileName}: slide ${index + 1} has fewer than 3 note sentences`);
  });

  console.log(`[ok] ${fileName}: ${slideNames.length} slides, ${noteNames.length} note pages, formal evidence present`);
}

async function main() {
  await checkDeck("中期答辩_关铭锡.pptx", 22);
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
