const assert = require("assert");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const JSZip = require("jszip");

const outputDir = path.resolve(__dirname, "../outputs");
const finalPath = path.join(outputDir, "关铭锡-3124358145-中期答辩.pptx");
const templatePath = path.join(outputDir, "牛有锋--中期答辩.pptx");
const bodySlides = [4, 5, 7, 8, 9, 11, 12, 13, 15];

function sha256(buffer) {
  return crypto.createHash("sha256").update(buffer).digest("hex");
}

async function main() {
  const [finalZip, templateZip] = await Promise.all([
    JSZip.loadAsync(fs.readFileSync(finalPath)),
    JSZip.loadAsync(fs.readFileSync(templatePath)),
  ]);

  for (const slideNumber of bodySlides) {
    const rels = await finalZip.file(`ppt/slides/_rels/slide${slideNumber}.xml.rels`).async("string");
    assert.match(rels, /Target="\.\.\/slideLayouts\/slideLayout3\.xml"/);
  }

  const layout = await finalZip.file("ppt/slideLayouts/slideLayout3.xml").async("string");
  assert.match(layout, /<p:cSld name="标题幻灯片">/, "body layout must come from the approved template");
  assert.match(layout, /x="9194801" y="264123"/, "template title extension is missing");
  assert.match(layout, /name="组合 15"/, "template title emblem is missing");

  const rels = await finalZip.file("ppt/slideLayouts/_rels/slideLayout3.xml.rels").async("string");
  const imageTarget = rels.match(/Type="[^"]+\/image" Target="\.\.\/media\/([^"]+)"/);
  assert.ok(imageTarget, "body layout image relationship is missing");

  const [actualImage, templateImage] = await Promise.all([
    finalZip.file(`ppt/media/${imageTarget[1]}`).async("nodebuffer"),
    templateZip.file("ppt/media/image1.png").async("nodebuffer"),
  ]);
  assert.strictEqual(sha256(actualImage), sha256(templateImage), "template title image does not match");

  console.log("[ok] final body slides use the approved template title layout");
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
