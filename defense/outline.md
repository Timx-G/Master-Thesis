# 20-Slide Professional-Master Defense Outline

1. **研究题目与工程目标**：全频复用多 UAV NOMA-MEC；目标是单位时隙按时完成任务数。
2. **应用场景**：灾害、临时热点、基础设施薄弱区域；明确集中式固定悬停边界。
3. **核心矛盾**：跨 UAV 干扰外部性与有限 MEC CPU 同时约束截止时间。
4. **研究问题**：如何在一秒时隙内生成结构、SIC、CPU 全可行调度。
5. **系统模型**：唯一任务、显式 Schedule、真实服务 UAV、全频复用 SINR。
6. **可信度修复**：不可变同种子轨迹、任务只计一次、硬约束与 checksum 数据链。
7. **创新点一痛点**：双边匹配为何遗漏第三方干扰。
8. **创新点一算法**：ILP 初始化、SIC 修复、Replace/Position/CrossUAV。
9. **创新点一接受准则**：全网词典序 first-improvement 与有限终止。
10. **创新点一证据**：`pairing_ablation.pdf`，`\InnovationOneCrossGainPP`、`\InnovationOneAcceptanceGainPP` 及 CI。
11. **创新点二推导**：`alpha_min`、正传输余量和 `f_min`。
12. **创新点二证明**：按最小 CPU 排序的最大基数交换论证。
13. **创新点二证据**：`resource_ablation.pdf`，`\InnovationTwoGainPP` 及 CI。
14. **系统实现**：配置、trace、scheduler、evaluator、writer、analyzer、publisher。
15. **实验协议**：50 未见种子、强基线、公平 OFDMA、2×2/资源消融、paired statistics。
16. **主结果**：`main_on_time.pdf`，`\FullOnTimePercent` 与 `\StrongBaselineGainPP`。
17. **机制与扩展性**：`mechanism_interference.pdf`、`scalability_users.pdf`、`scalability_uavs.pdf`。
18. **可信度结果**：`\ExactOptimalPercent`、`\SICViolationCount`、`\FullRuntimePNinetyFiveMs` 与鲁棒性表。
19. **部署与限制**：集中式 CSI/信令；K=1 非优势；无轨迹、功率、真实数据联合验证。
20. **结论**：只回到两项创新；创新一小而稳定，创新二工程增益明显。
