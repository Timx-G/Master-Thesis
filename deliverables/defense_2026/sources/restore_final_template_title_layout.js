const fs = require("fs");
const path = require("path");
const JSZip = require("jszip");

const outputDir = path.resolve(__dirname, "../outputs");
const finalPath = path.join(outputDir, "关铭锡-3124358145-中期答辩.pptx");
const templatePath = path.join(outputDir, "牛有锋--中期答辩.pptx");

async function main() {
  const [finalZip, templateZip] = await Promise.all([
    JSZip.loadAsync(fs.readFileSync(finalPath)),
    JSZip.loadAsync(fs.readFileSync(templatePath)),
  ]);

  const [layoutXml, layoutRels, titleImage] = await Promise.all([
    templateZip.file("ppt/slideLayouts/slideLayout1.xml").async("string"),
    templateZip.file("ppt/slideLayouts/_rels/slideLayout1.xml.rels").async("string"),
    templateZip.file("ppt/media/image1.png").async("nodebuffer"),
  ]);

  const imageName = "template_body_title.png";
  const updatedRels = layoutRels.replace("../media/image1.png", `../media/${imageName}`);
  if (updatedRels === layoutRels) {
    throw new Error("Template body layout image relationship was not found");
  }

  finalZip.file("ppt/slideLayouts/slideLayout3.xml", layoutXml);
  finalZip.file("ppt/slideLayouts/_rels/slideLayout3.xml.rels", updatedRels);
  finalZip.file(`ppt/media/${imageName}`, titleImage);

  const output = await finalZip.generateAsync({
    type: "nodebuffer",
    compression: "DEFLATE",
    compressionOptions: { level: 6 },
  });
  fs.writeFileSync(`${finalPath}.tmp`, output);
  fs.renameSync(`${finalPath}.tmp`, finalPath);
  console.log(`[updated] ${path.basename(finalPath)}: restored approved body title layout`);
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});
