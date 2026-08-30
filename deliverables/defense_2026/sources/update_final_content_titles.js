const fs = require("fs");
const path = require("path");
const JSZip = require("jszip");

const deckPath = path.resolve(__dirname, "../outputs/关铭锡-3124358145-中期答辩.pptx");
const bodySlides = [4, 5, 7, 8, 9, 11, 12, 13, 15];

const textReplacements = new Map([
  [13, [
    ["DD-MCAA 资源分配对比结果", "DD-MCAA 资源分配受控消融实验"],
    [
      "实验结论：在相同无线调度下，deadline 驱动的最小资源准入显著提高任务按时完成率。",
      "实验定位：在相同无线调度下隔离 deadline 驱动准入的独立贡献；不作为 SOTA 对比。",
    ],
  ]],
  [15, [
    ["扩展实验与机制验证", "扩展实验与强基线验证"],
    [
      "目前已跑通 K=2、N=20 基础场景；补充等负载规模、高并发强干扰、异构算力与 CSI 误差实验。",
      "基础场景已跑通；补充等负载规模、强干扰、异构算力及 CSI 误差，并加入 EDF、Exact-Oracle 及文献适配基线。",
    ],
    ["可扩展性、稳健性与机制边界证据", "机制边界、最优差距与文献对比证据"],
  ]],
]);

function escapeXml(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function updateSectionTitle(xml, slideNumber) {
  let found = false;
  const updated = xml.replace(/<p:sp>[\s\S]*?<\/p:sp>/g, (shape) => {
    if (!/<p:cNvPr id="2" name="Text 0"\/>/.test(shape)) return shape;
    found = true;
    let next = shape.replace(
      /<a:xfrm>[\s\S]*?<\/a:xfrm>/,
      '<a:xfrm><a:off x="943728" y="16070"/><a:ext cx="1422184" cy="576248"/></a:xfrm>',
    );
    if (/<a:noFill\/>/.test(next)) {
      next = next.replace(
        /<a:noFill\/>/,
        '<a:solidFill><a:schemeClr val="bg1"/></a:solidFill>',
      );
    }
    next = next.replace(/sz="1800"/g, 'sz="2400"');
    return next;
  });
  if (!found) throw new Error(`Slide ${slideNumber} section title shape not found`);
  return updated;
}

function replaceSlideText(xml, slideNumber, replacements) {
  let updated = xml;
  for (const [before, after] of replacements) {
    const escapedBefore = escapeXml(before);
    const escapedAfter = escapeXml(after);
    if (updated.includes(escapedAfter)) continue;
    if (!updated.includes(escapedBefore)) {
      throw new Error(`Slide ${slideNumber} text not found: ${before}`);
    }
    updated = updated.replace(escapedBefore, escapedAfter);
  }
  return updated;
}

async function main() {
  const zip = await JSZip.loadAsync(fs.readFileSync(deckPath));
  for (const slideNumber of bodySlides) {
    const slidePath = `ppt/slides/slide${slideNumber}.xml`;
    let xml = await zip.file(slidePath).async("string");
    xml = updateSectionTitle(xml, slideNumber);
    if (textReplacements.has(slideNumber)) {
      xml = replaceSlideText(xml, slideNumber, textReplacements.get(slideNumber));
    }
    zip.file(slidePath, xml);
  }

  const output = await zip.generateAsync({
    type: "nodebuffer",
    compression: "DEFLATE",
    compressionOptions: { level: 6 },
  });
  fs.writeFileSync(`${deckPath}.tmp`, output);
  fs.renameSync(`${deckPath}.tmp`, deckPath);
  console.log(`[updated] ${path.basename(deckPath)}: content titles and experiment positioning`);
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});
