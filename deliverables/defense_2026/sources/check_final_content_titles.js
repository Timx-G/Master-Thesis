const assert = require("assert");
const fs = require("fs");
const path = require("path");
const JSZip = require("jszip");

const deckPath = path.resolve(__dirname, "../outputs/关铭锡-3124358145-中期答辩.pptx");
const bodySlides = [4, 5, 7, 8, 9, 11, 12, 13, 15];

function sectionTitleShape(xml) {
  const shape = xml.match(/<p:sp>[\s\S]*?<p:cNvPr id="2" name="Text 0"\/>[\s\S]*?<\/p:sp>/);
  assert.ok(shape, "section title shape is missing");
  return shape[0];
}

async function main() {
  const zip = await JSZip.loadAsync(fs.readFileSync(deckPath));

  for (const slideNumber of bodySlides) {
    const xml = await zip.file(`ppt/slides/slide${slideNumber}.xml`).async("string");
    const shape = sectionTitleShape(xml);
    assert.match(shape, /<a:off x="943728" y="16070"\/>/, `slide ${slideNumber} section title position`);
    assert.match(shape, /<a:ext cx="1422184" cy="576248"\/>/, `slide ${slideNumber} section title size`);
    assert.match(shape, /sz="2400" b="1"/, `slide ${slideNumber} section title typography`);
    assert.match(shape, /<a:solidFill><a:schemeClr val="bg1"\/><\/a:solidFill>/, `slide ${slideNumber} section title background`);
  }

  const slide13 = await zip.file("ppt/slides/slide13.xml").async("string");
  assert.match(slide13, /DD-MCAA 资源分配受控消融实验/);
  assert.match(slide13, /不作为 SOTA 对比/);

  const slide15 = await zip.file("ppt/slides/slide15.xml").async("string");
  assert.match(slide15, /扩展实验与强基线验证/);
  assert.match(slide15, /EDF、Exact-Oracle 及文献适配基线/);
  assert.match(slide15, /最优差距与文献对比证据/);

  const notes13 = await zip.file("ppt/notesSlides/notesSlide13.xml").async("string");
  const notes15 = await zip.file("ppt/notesSlides/notesSlide15.xml").async("string");
  assert.match(notes13, /受控消融，不是 SOTA 比较/);
  assert.match(notes15, /Exact-Oracle 和近年文献适配算法/);

  console.log("[ok] final content titles and experiment positioning are consistent");
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
