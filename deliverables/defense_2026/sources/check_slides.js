const fs = require("fs");
const path = require("path");
const JSZip = require("jszip");

const BASE = path.resolve(__dirname, "..");
const manifest = JSON.parse(fs.readFileSync(path.join(BASE, "data_placeholders.json"), "utf8"));
const registeredKeys = manifest.placeholders.map((item) => item.key).sort();

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

async function checkDeck(fileName, expectedSlides, { requiresSoftwareEvidence = false } = {}) {
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

  for (const key of registeredKeys) {
    if (!slideText.includes(`{{${key}}}`)) throw new Error(`${fileName}: missing placeholder {{${key}}}`);
  }
  if (!slideText.includes("待新版本 canonical 实验填充")) {
    throw new Error(`${fileName}: missing canonical placeholder notice`);
  }
  if (/73\.4%|82\.5%|68\.4%/.test(slideText)) throw new Error(`${fileName}: legacy performance number found`);
  if (!slideText.includes("创新点一") || !slideText.includes("创新点二")) {
    throw new Error(`${fileName}: innovation labels missing`);
  }
  if (requiresSoftwareEvidence && !slideText.includes("121 passed")) {
    throw new Error(`${fileName}: software verification evidence missing`);
  }
  noteTexts.forEach((text, index) => {
    const sentenceCount = (text.match(/[。！？]/g) || []).length;
    if (sentenceCount < 3) throw new Error(`${fileName}: slide ${index + 1} has fewer than 3 note sentences`);
  });

  console.log(`[ok] ${fileName}: ${slideNames.length} slides, ${noteNames.length} note pages, 29/29 keys`);
}

async function main() {
  await checkDeck("中期答辩_关铭锡.pptx", 16, { requiresSoftwareEvidence: true });
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
