# 2026 专业硕士答辩材料

本目录可复现生成关铭锡的开题报告、中期报告和中期答辩演示文稿。当前版本已切换到正式 `canonical v1` 证据，不再包含实验占位页。

## 正式交付物

| 材料 | 文件 |
|---|---|
| 开题报告 | `outputs/开题报告_关铭锡.docx` |
| 中期报告 | `outputs/中期报告_关铭锡.docx` |
| 中期答辩 PPT | `outputs/中期答辩_关铭锡.pptx` |

用户已明确：开题阶段只需要报告，不生成开题答辩 PPT。

## 证据口径

- 唯一正式来源：`../../artifacts/canonical/v1/`。
- 正式确认性实验使用 50 个未见随机种子 1000--1049。
- 完整方法按时完成率为 98.11%，95% CI 为 [97.94%, 98.26%]。
- 创新点一 CrossUAV 邻域与全局接受准则分别贡献 +0.0487 pp 和 +0.1179 pp。
- 创新点二相对 crossing/equal 贡献 +8.6898 pp。
- 正式实验门禁 7/7 通过，SIC 与 CPU 预算违规均为 0。
- Python 3.11 干净环境全量验证结果为 257 passed。

旧 `results/`、历史 DRL 实验和旧答辩材料中的数字不得作为本文两项创新的性能证据。`data_placeholders.json` 仅保留为早期材料的数据契约档案，正式输出不再读取其中的占位状态。

## 重新生成

在本目录运行：

```bash
node sources/build_reports.js
node sources/build_slides.js
node sources/check_slides.js
```

生成器将覆盖三份正式输出。PPT 图表直接读取 `../../figures/thesis/` 中由 canonical v1 生成的论文资产。

## 质量检查

1. `check_slides.js` 检查页数、逐页讲稿、正式证据和旧占位内容。
2. LibreOffice 将 DOCX/PPTX 转为 PDF，验证 Office 兼容性。
3. `render_previews.py` 生成逐页 PNG 和联系表，用于检查溢出、遮挡和图表清晰度。
4. 最终材料仍需填写学校、学院、专业、学号和提交日期，并按学校模板调整封面与签字页。
