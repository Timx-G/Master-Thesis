# 2026 年开题报告与中期答辩材料

本目录用于生成和交付关铭锡的开题报告、中期报告及中期答辩演示文稿。全部待验证的性能数据均由统一占位符管理，只有新版本 canonical 实验产物可以替换占位符。

## 三份正式交付物

| 材料 | 目标文件 |
| --- | --- |
| 开题报告 | `outputs/开题报告_关铭锡.docx` |
| 中期报告 | `outputs/中期报告_关铭锡.docx` |
| 中期答辩 PPT | `outputs/中期答辩_关铭锡.pptx` |

`sources/` 保存可编辑源稿和生成器，`previews/` 保存用于内容与视觉检查的 PDF 和逐页预览图。Office 成品由源文件生成，不应直接手工固化尚未验证的性能数字。

## 证据边界

- 截至提交 `fd0cb7c` 的 `121 passed` 仅代表软件自动化测试通过，是工程验证状态，不代表算法性能、优越性或统计显著性。
- 旧 `results/`、历史 DRL 实验以及可信性审计已否定的数据不得用于替换占位符，也不得作为新方案优于基线的证据。
- 最终性能结论必须来自新 canonical 实验包，并同时具备可追溯的 manifest、代码提交、运行日期和摘要校验信息。
- 新实验尚未完成时，报告和幻灯片必须保留占位框及“待新版本 canonical 实验填充”说明，不得绘制虚构的柱高、折线点或误差线。

## 数据占位符

占位符使用 `{{KEY}}` 语法，例如 `{{MAIN_ONTIME_OURS_PCT}}`。完整契约见 [`data_placeholders.json`](data_placeholders.json)，其中记录了每个键的含义、单位、canonical 来源、使用位置、格式和状态。

当前契约在原有 16 个基础键上扩展了 13 个细粒度键，共 29 个键。生成报告、幻灯片和 canonical 数据映射时必须覆盖全部 29 个键，不得遗漏，也不得用语义相近的键相互替代：

- `MAIN_CI95_PP` 是本文方法自身按时完成率的 95% 置信区间半宽；`MAIN_PAIRED_DIFF_CI95_PP` 是本文方法与预注册主比较基线之配对差值的 95% 置信区间，两者的统计对象和输出格式不同。
- `ABLATION_GLOBAL_ON_PCT` 与 `ABLATION_CROSSUAV_ON_PCT` 是保留兼容性的汇总键；严格 2x2 结果必须使用 `ABLATION_NONE_PCT`、`ABLATION_GLOBAL_ONLY_PCT`、`ABLATION_CROSSUAV_ONLY_PCT`、`ABLATION_BOTH_PCT` 和 `ABLATION_INTERACTION_PP`，不得用两个汇总键代替四格与交互项。
- `RESOURCE_*` 三键比较创新点二的资源分配策略；`MODULE_*` 四键分解关联/配对模块与资源分配模块的独立及联合增量，不得把策略消融当作模块贡献分解。

替换流程如下：

1. 完成新 canonical 实验并验证其 manifest、提交哈希、运行日期和产物校验和。
2. 按 `data_placeholders.json` 中的 `source` 将 canonical 数据映射到对应键；不得从旧结果目录或人工估计中取值。
3. 在共享内容源中替换相应 `{{KEY}}`，保留精度、单位、置信区间和序列结构的一致性。
4. 重新生成三份材料并执行占位符扫描、Office 校验、文本提取和逐页视觉检查。

## 重新构建

在仓库根目录执行：

```bash
node deliverables/defense_2026/sources/build_reports.js
node deliverables/defense_2026/sources/build_slides.js
```

在 canonical 实验完成前，上述命令生成的实验结果区域应继续显示占位状态。
