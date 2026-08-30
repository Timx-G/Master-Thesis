const fs = require("fs");
const path = require("path");
const JSZip = require("jszip");

const deckPath = path.resolve(__dirname, "../outputs/关铭锡-3124358145-中期答辩.pptx");

const replacements = [
  [["外部有效性补强", "实验场景扩展与机制验证"], "扩展实验与机制验证"],
  [
    ["在现有稳健性实验基础上，扩展复杂移动性、时变 CSI 及公开轨迹或半实物验证。"],
    "目前已跑通 K=2、N=20 基础场景；补充等负载规模、高并发强干扰、异构算力与 CSI 误差实验。",
  ],
  [["扩展实验与外部有效性分析", "验证可扩展性、稳健性及外部性机制有效边界"], "可扩展性、稳健性与机制边界证据"],
];

function escapeXml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

async function main() {
  const zip = await JSZip.loadAsync(fs.readFileSync(deckPath));
  const slidePath = "ppt/slides/slide15.xml";
  let xml = await zip.file(slidePath).async("string");

  for (const [beforeOptions, after] of replacements) {
    const options = Array.isArray(beforeOptions) ? beforeOptions : [beforeOptions];
    const escapedAfter = escapeXml(after);
    if (xml.includes(escapedAfter)) continue;

    const matchedBefore = options.find((before) => xml.includes(escapeXml(before)));
    if (!matchedBefore) {
      throw new Error(`Slide 15 text not found: ${options.join(" | ")}`);
    }
    xml = xml.replace(escapeXml(matchedBefore), escapedAfter);
  }

  zip.file(slidePath, xml);
  const output = await zip.generateAsync({
    type: "nodebuffer",
    compression: "DEFLATE",
    compressionOptions: { level: 6 },
  });
  fs.writeFileSync(`${deckPath}.tmp`, output);
  fs.renameSync(`${deckPath}.tmp`, deckPath);
  console.log(`[updated] ${path.basename(deckPath)}: slide 15 research plan`);
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});
