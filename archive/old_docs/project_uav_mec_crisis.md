---
name: UAV-MEC论文危机诊断与重构方向
description: DC-PPO算法贡献为零（Fixed-λ全胜），环境太简单（无多目标trade-off），五位博士专家诊断+重构方案
type: project
originSessionId: 05a1bf03-608a-4c23-8dc5-5ea954d04538
---
# 2026-04-11 论文危机诊断

## 致命发现
1. **Fixed-λ=5 PPO全胜DC-PPO 5/5 seeds，平均+3.6pp** — 自适应λ机制是负优化
2. 用户能耗trade-off: 1.8%（平凡）
3. 飞行能耗trade-off: 1.0%（平凡）
4. Fairness spread: 8.5pp（不足20pp门槛）
5. 手工策略（α=0.3, center）= 87.1% >> DRL训练agent 45.5%

## 根因
**200m×200m + 100m高度 = 问题太简单。** UAV在任何位置都能覆盖所有用户，能耗被悬停功率主导，无真实multi-objective trade-off。

## 五位博士专家收敛方向（Path A）
1. 区域扩大到1000×1000m + 3-4用户集群
2. 加任务排队（multi-slot queuing）
3. UAV高度可变（50-150m）
4. Multi-UAV（2-3架）
5. 电池预算约束（B=5000J/episode）
6. 异构deadline（0.3-2.0s）
7. Warm-start λ（从经验最优值开始）
8. BCD upper bound + CPO baseline
9. 20 seeds + last-checkpoint评估
10. 预计4周工作量

## 已修复的环境改进
- Reward: throughput-based（策略相关）✅
- MEC分配: 加权 ✅
- SIC: ε=0.02 ✅
- GAE overflow: clip ✅
- OFDMA rates: 已实现 ✅
