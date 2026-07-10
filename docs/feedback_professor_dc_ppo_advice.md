---
name: 教授对DC-PPO论文的三条改进建议
description: 丰富baseline(加CPO/PPO-Lag)、微观预测器改为Safety Layer投影、NOMA+轨迹联合增益消融实验
type: feedback
originSessionId: 05a1bf03-608a-4c23-8dc5-5ea954d04538
---
教授给出三条actionable advice，必须在DC-PPO论文中执行：

1. **丰富Baseline**：必须加CPO或PPO-Lagrangian作为对比，证明DC-PPO（双层）优于单层PPO-Lag
2. **微观Constraint Predictor改为Safety Layer投影**：min_a ||a - a_RL||² s.t. C_predict(s,a) ≤ ε，有成熟数学支撑，不破坏PPO收敛性
3. **NOMA+轨迹联合增益消融**：必须做"UAV悬停+NOMA" vs "OMA+UAV动态轨迹" vs "NOMA+UAV动态轨迹"三组对比，展示乘数效应

**Why:** 这三条直接对应审稿人必问的问题，提前做好=减少revision轮次
**How to apply:** 实现DC-PPO时直接按此设计，不做事后补丁
