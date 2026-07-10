---
name: 单UAV DRL论文教训
description: 为什么单UAV场景DRL无法超越BCD/手工策略的完整教训，避免重复踩坑
type: feedback
originSessionId: 05a1bf03-608a-4c23-8dc5-5ea954d04538
---
# 单UAV场景的教训（2026-04-11）

## 核心教训
单UAV + 小区域(200-500m) + 全向天线 → UAV位置不影响结果 → DRL学的"往哪飞"贡献≈0pp → DRL无法超越简单optimizer/手工策略。

## 具体踩坑记录
1. DC-PPO自适应λ: Fixed-λ=5全胜5/5 seeds (-3.6pp)
2. 用户能耗约束: 变化1.8%（平凡）
3. UAV飞行能耗约束: 变化1.0%（悬停功率主导）
4. Fairness约束: spread 8.5pp（不够）
5. 动态burst: BCD per-user optimizer仍能应对
6. 42维PPO: 150K样本 vs 需要>5M，维度诅咒
7. BC+PPO fine-tune: catastrophic forgetting
8. Hybrid DRL(5维)+Optimizer: 接近但不超过BCD

## Why
- 500m区域+100m高度: 所有用户3D距离112-316m，路径损耗差仅7dB
- Optimizer通过per-user α/power调整补偿7dB差异
- UAV飞不飞，结果差0pp → DRL的trajectory component无价值

## How to apply
- DRL只在"决策不可被凸优化/搜索替代"时才有价值
- 需要：多agent协调 / 不可分解的时序耦合 / 组合爆炸
- 单UAV+小区域 = 凸优化的天下，不要用DRL
- Multi-UAV协调是genuine NP-hard → DRL有不可替代价值
