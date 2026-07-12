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

  const slideText = (await Promise.all(slideNames.map((name) => zip.file(name).async("string"))))
    .map(xmlText)
    .join("\n");
  const noteTexts = (await Promise.all(noteNames.map((name) => zip.file(name).async("string")))).map(xmlText);

  if (/\{\{[A-Z0-9_]+\}\}|待新版本|121 passed|fd0cb7c|设计完成待实现|性能尚未生成/.test(slideText)) {
    throw new Error(`${fileName}: stale placeholder or pre-experiment statement found`);
  }
  if (/73\.4%|82\.5%|68\.4%/.test(slideText)) throw new Error(`${fileName}: legacy performance number found`);
  if (!slideText.includes("创新点一") || !slideText.includes("创新点二")) {
    throw new Error(`${fileName}: innovation labels missing`);
  }
  for (const evidence of ["98.11%", "+8.6898 pp", "257 passed", "7 / 7", "P95 40.93 ms"]) {
    if (!slideText.includes(evidence)) throw new Error(`${fileName}: missing formal evidence ${evidence}`);
  }
  noteTexts.forEach((text, index) => {
    const sentenceCount = (text.match(/[。！？]/g) || []).length;
    if (sentenceCount < 3) throw new Error(`${fileName}: slide ${index + 1} has fewer than 3 note sentences`);
  });

  console.log(`[ok] ${fileName}: ${slideNames.length} slides, ${noteNames.length} note pages, formal evidence present`);
}

async function main() {
  await checkDeck("中期答辩_关铭锡.pptx", 17);
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
