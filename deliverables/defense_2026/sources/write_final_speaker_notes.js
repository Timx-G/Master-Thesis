const fs = require("fs");
const path = require("path");
const JSZip = require("jszip");

const deckPath = path.resolve(__dirname, "../outputs/关铭锡-3124358145-中期答辩.pptx");

const notes = [
  "各位老师好，我是关铭锡。今天汇报的题目是《面向时限任务的多无人机移动边缘计算干扰感知调度与卸载资源分配方法研究》。研究重点是跨无人机干扰与有限边缘算力下的时限任务联合优化。",
  "本次汇报包括选题背景、研究方案、研究进展和后续计划四部分。",
  "首先汇报选题背景。",
  "应急通信、智能巡检和临时热点等场景往往存在覆盖不足和近端算力缺失。多无人机搭载 MEC 后可以同时提供空中接入与边缘计算。但任务通常具有明确截止时刻，因此系统目标不仅是提高吞吐量，更要保障更多任务按时完成。",
  "在全频复用下，服务归属变化会改变多个接收端的同频干扰，NOMA 配对还受 SIC 约束；无线速率影响卸载时延，MEC 算力影响执行时延。本文联合优化关联、配对、卸载比例与算力，以最大化按时完成任务数。研究边界是固定悬停、集中式控制和时隙级调度。",
  "下面汇报研究方案。",
  "近两年研究主要覆盖多无人机协同卸载、NOMA 空中 MEC 和时限约束卸载。现有工作重点优化轨迹、功率、能耗或平均时延，但对全频复用下的跨接收端干扰外部性，以及绝对截止时刻下的任务完成数量，仍缺少统一处理。本文由此确定研究切入点。",
  "总体框架如图所示。IEA-CAP 生成关联与配对候选；每个候选进入全网评价，并内嵌 DD-MCAA 计算最小卸载比例和最小 CPU。评价器统一重算干扰、SIC、服务器负载和完成状态，再把结果反馈给 IEA-CAP。两部分组合为 EAD-JSRA，输出调度与资源分配结果。",
  "研究包含两个创新点。第一，IEA-CAP 设计 Replace、Position 和 CrossUAV 三类邻域，并按完成数、截止裕量和全网干扰接受严格改善。第二，DD-MCAA 推导最小卸载比例与最小 MEC CPU，再按 CPU 需求排序，在每架无人机预算内选择最长可行前缀，从而最大化等价值任务完成数。",
  "下面汇报目前已经完成的研究工作和实验结果。",
  "实验采用两架无人机、二十个用户和三个子载波，任务时限为零点八秒。传播模型采用 Al-Hourani Urban，并参考 3GPP 场景。正式实验为五十个种子、每个种子一百时隙。所有算法共享任务、轨迹和信道样本，由统一评价器复核约束与时延。统计采用配对 bootstrap 和 Wilcoxon-Holm 检验，并完成规模、CSI、算力、时限和负载等稳健性分析。当前结论用于说明同一模型下的相对性能。",
  "这里是创新点一的二乘二消融。CrossUAV 邻域独立提高零点零四八七个百分点，全网接受准则独立提高零点一一七九个百分点。两项百分之九十五置信区间均严格大于零。虽然绝对增益较小，但方向稳定，且最终调度满足 SIC 和 MEC 算力约束。这说明 IEA-CAP 能够处理跨无人机外部性，并保持调度与物理评价一致。",
  "本页是创新点二的受控消融，不是 SOTA 比较。三种策略共享相同无线调度，仅替换卸载比例和 CPU 分配。DD-MCAA 完成率为百分之九十八点一一，相对 Crossing 加平均 CPU 提高八点六八九八个百分点，说明截止时间驱动准入是主要性能来源。后续将加入 EDF、Exact-Oracle 和近年文献适配算法，补充经典调度、最优差距及文献方法比较。",
  "最后汇报下一步研究计划。",
  "下一阶段首先补强实验。目前已跑通两架无人机、二十个用户的基础场景，但网络规模与竞争压力仍不足。后续将补充等负载规模扩展、高并发强干扰、异构算力和 CSI 误差实验，并加入 EDF、Min-Slack、Exact-Oracle 和近年文献适配算法，用于验证机制边界、最优差距以及相对文献先进方法的性能。随后完善理论与复杂度分析，推进论文投稿、学位论文送审和毕业答辩。",
  "以上是我的中期汇报。恳请各位老师批评指正，谢谢各位老师。",
];

function escapeXml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function notesBody(text) {
  const paragraphs = text.split(/\n+/).map((paragraph) => (
    `<a:p><a:r><a:rPr lang="zh-CN" dirty="0"/><a:t xml:space="preserve">${escapeXml(paragraph)}</a:t></a:r><a:endParaRPr lang="zh-CN" dirty="0"/></a:p>`
  )).join("");
  return `<p:txBody><a:bodyPr/><a:lstStyle/>${paragraphs}</p:txBody>`;
}

function replaceBodyPlaceholder(xml, text) {
  let replaced = false;
  const updated = xml.replace(/<p:sp>[\s\S]*?<\/p:sp>/g, (shapeXml) => {
    if (!/<p:ph\b[^>]*type="body"/.test(shapeXml)) return shapeXml;
    if (!/<p:txBody>[\s\S]*?<\/p:txBody>/.test(shapeXml)) {
      throw new Error("Notes body placeholder is missing p:txBody");
    }
    replaced = true;
    return shapeXml.replace(/<p:txBody>[\s\S]*?<\/p:txBody>/, notesBody(text));
  });
  if (!replaced) throw new Error("Notes body placeholder not found");
  return updated;
}

function stripPlaybackTiming(xml) {
  return xml
    .replace(/<mc:AlternateContent\b[\s\S]*?<\/mc:AlternateContent>/g, (block) => (
      /<p:transition\b/.test(block) ? "" : block
    ))
    .replace(/<p:transition\b[^>]*\/>/g, "")
    .replace(/<p:transition\b[^>]*>[\s\S]*?<\/p:transition>/g, "")
    .replace(/<p:timing\b[^>]*>[\s\S]*?<\/p:timing>/g, "");
}

function disableAutomaticPlayback(xml) {
  let updated = xml
    .replace(/<p:showPr\b[^>]*\/>/g, "")
    .replace(/<p:showPr\b[^>]*>[\s\S]*?<\/p:showPr>/g, "");
  updated = updated.replace(
    "</p:presentationPr>",
    '<p:showPr useTimings="0"/></p:presentationPr>',
  );
  return updated;
}

async function main() {
  const zip = await JSZip.loadAsync(fs.readFileSync(deckPath));
  const noteNames = Object.keys(zip.files)
    .filter((name) => /^ppt\/notesSlides\/notesSlide\d+\.xml$/.test(name))
    .sort((a, b) => Number(a.match(/\d+/)[0]) - Number(b.match(/\d+/)[0]));

  if (noteNames.length !== notes.length) {
    throw new Error(`Expected ${notes.length} note pages, found ${noteNames.length}`);
  }

  for (let index = 0; index < noteNames.length; index += 1) {
    const name = noteNames[index];
    const xml = await zip.file(name).async("string");
    zip.file(name, replaceBodyPlaceholder(xml, notes[index]));
  }

  const pptXmlNames = Object.keys(zip.files).filter((name) => /^ppt\/.*\.xml$/.test(name));
  for (const name of pptXmlNames) {
    const xml = await zip.file(name).async("string");
    zip.file(name, stripPlaybackTiming(xml));
  }
  const presPropsName = "ppt/presProps.xml";
  const presProps = await zip.file(presPropsName).async("string");
  zip.file(presPropsName, disableAutomaticPlayback(presProps));

  const output = await zip.generateAsync({
    type: "nodebuffer",
    compression: "DEFLATE",
    compressionOptions: { level: 6 },
  });
  fs.writeFileSync(`${deckPath}.tmp`, output);
  fs.renameSync(`${deckPath}.tmp`, deckPath);
  console.log(`[updated] ${path.basename(deckPath)}: ${notes.length} speaker notes`);
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});
