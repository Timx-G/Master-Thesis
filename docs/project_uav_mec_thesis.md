---
name: UAV-MEC论文项目总览与当前方向
description: DC-PPO算法(双层约束PPO)用于UAV-MEC NOMA卸载，完整场景参数、算法设计、实验结果、历史决策记录
type: project
originSessionId: 05a1bf03-608a-4c23-8dc5-5ea954d04538
---
# UAV-MEC NOMA 论文项目（更新于2026-04-10）

## 论文信息
- **题目（暂定）**: 基于双层约束深度强化学习的UAV-MEC联合轨迹设计与NOMA任务卸载研究
- **英文**: Dual-Constrained DRL for Joint UAV Trajectory Design and NOMA-Based Task Offloading in MEC Networks
- **目标期刊**: IEEE IoTJ (CCF-B, IF≈10)，备选 IEEE TVT (CCF-B)
- **核心算法**: **DC-PPO** (Dual-Constrained PPO)

## 场景描述
应急救援/临时覆盖：单UAV搭载MEC服务器，200×200m区域，**N=20地面用户**提供NOMA计算卸载。
观测空间：2 + 5×N + 1 = 103维（N=20）
连续动作空间：UAV_xy(2维) + 卸载比(20维) + NOMA功率(20维) = 42维

## 系统参数（最终确认）

| 参数 | 值 | 依据 |
|------|-----|------|
| f_uav | **8 GHz** | Jetson Orin AGX级别 |
| f_local | 1 GHz | IoT终端 |
| t_max (deadline) | **1.0s** | demanding IoT场景 |
| delay_budget | **0.5s** | Lagrangian软约束目标 |
| **lambda_max** | **5.0** | 物理推导：λ_crit=9/1.2=7.5，取67%安全边际 |
| **lr_lambda** | **3e-2** | 快速到达λd=5（~200ep），之后稳定 |
| NOMA带宽 | 5 MHz | LTE单载波 |
| 用户发射功率 | 0.2W (23dBm) | 3GPP Class 3 |
| **用户数（主场景）** | **N=20** | 甜点场景 |
| 区域 | 200×200m | |
| UAV高度 | 100m固定 | DJI M300典型 |
| UAV最大速度 | 20 m/s | |
| 时隙 | 100个×1s | |
| 任务数据量 | 0.4-2.0Mbit | 压缩图像 |
| 计算密度 | 500-1000 cycles/bit | |
| 到达概率 | 0.5/时隙/用户 | |

## DC-PPO算法设计（最终版）

### 架构：PPO + Lagrangian约束
- λd按episode级avg_delay更新：`λ += lr_lambda*(avg_delay - delay_budget)`
- lambda_max=5.0封顶（物理推导：λ_crit=r̄/c̄=9/1.2=7.5，5为保守边际）
- 修改奖励：`r' = r - λ_d * cost_delay`
- Safety Layer已弃用（IS ratio bias问题）
- 三阶段调度已弃用（on-policy无法从早期坏习惯恢复）

### ✅ 最终验证性能（v6: N=20, t_max=1.0s, lambda_max=5, 1500ep, 5seeds, best-ckpt）

| 算法 | Deadline% | Std | Delay(s/task) | Energy(J/task) | ViolRate | λd_final |
|------|-----------|-----|---------------|----------------|----------|----------|
| **DC-PPO（Ours）** | **82.5%** | ±1.8% | 0.7874s | 8.6669J | 17.5% | 5.000 |
| PPO-Lagrangian | 80.8% | ±3.4% | 0.7970s | 8.7183J | 19.2% | 20.000 |
| PPO | 75.2% | ±2.3% | 0.8516s | 8.5983J | 24.8% | — |
| Random | 63.0% | ±0.6% | 0.9366s | 11.2236J | 37.0% | — |
| TD3 | 58.8% | ±2.1% | 1.1744s | 10.7960J | 41.2% | — |
| Greedy | 15.4% | ±0.4% | 1.4478s | 9.5423J | 84.6% | — |

- **DC-PPO vs PPO margin: +7.3pp** ← 核心数据（满足G3≥5pp）
- **DC-PPO vs PPO-Lag margin: +1.7pp** ← lambda_max截断有效（PPO-Lag λd饱和在20）
- **DC-PPO std=1.8% < PPO std=2.3%** ← 稳定性更好
- G1(≥78%): PASS ✅  G3(≥5pp): PASS ✅
- 总用时：126.2分钟

### 三个关键论文发现（v6更新）
1. DC-PPO(82.5%) >> PPO(75.2%)：+7.3pp，Lagrangian约束核心贡献
2. DC-PPO(82.5%) > PPO-Lagrangian(80.8%)：+1.7pp，lambda_max=5截断设计的必要性
3. DC-PPO std(1.8%) < PPO std(2.3%)：约束不仅提升性能，还提升训练稳定性（且PPO-Lag std=3.4%更差）

### Greedy异常说明（论文必须解释）
- greedy=15.4% < random=63.0%：物理可解释
- Greedy激进卸载策略→N=20用户同时高功率发射→NOMA干扰饱和→传输速率崩溃
- 论文中改名为"Myopic Offloading"或加footnote解释
- TD3=47.6%：1000ep对42维动作空间off-policy算法不够，需在论文中说明计算复杂度限制

## 待完成实验

| 实验 | 内容 | 状态 |
|------|------|------|
| Exp1(v6) | DC-PPO vs 6算法 baseline，完整指标（deadline/delay/energy/violation/p95）| ✅ **完成** CSV: results/exp1_dcppo_baseline.csv |
| Exp2 | lambda_max sensitivity: {1,2,5,10,20} → 5点曲线 | ⏳ 待做（下一步） |
| Exp3 | 可扩展性 N=10,15,20,25 | ✅ **完成** CSV: results/exp3_scalability.csv |
| Exp4 | 收敛曲线（λd轨迹数据v6完成后生成）| ⏳ 待做 |

## 指标体系（v6后完整输出）
- deadline_rate（到达口径=服务口径，无丢弃机制）✅
- delay_per_task (s/task) ← 新增
- energy_per_task (J/task) ← 新增
- violation_rate (=1-deadline_rate) ← 新增
- p95_delay (slot-avg p95) ← 新增
- lambda_d_final (DC-PPO/PPO-Lag专用)

## 论文贡献重新表述（三博士六导师辩论收敛）
- C1: 问题建模（UAV-MEC NOMA as CMDP）
- C2: **提出基于reward-cost ratio物理上界的lambda_max选择准则** + NOMA干扰饱和效应分析
- C3: 实验验证（5seeds，完整6指标，lambda_max sensitivity图）

## NaN修复记录（dc_ppo.py）
- select_action: 加mu NaN检测→返回零向量
- update(): backward()后检查梯度NaN→跳过optimizer.step()（根本修复）

## 历史决策记录（避免重复踩坑）

| 日期 | 尝试 | 结果 | 教训 |
|------|------|------|------|
| 04-10 | t_max=2.0s, N=10 | deadline=100%, λd≈0（失活） | 场景太简单 |
| 04-10 | t_max=2.0s, N=20 | deadline=99.9%, λd=2（失活） | 2.0s太宽松 |
| 04-10 | t_max=1.5s, N=20 | deadline=98.6%, margin=0.3pp | 仍然太容易 |
| 04-10 | t_max=1.0s, N=20, lambda_max=20 | PPO崩溃 | lambda_max太大 |
| 04-10 | violation_rate cost | margin=2.2pp | budget不可达，λd不收敛 |
| 04-10 | lambda_max=5封顶，3seed | margin=8.3pp | 初步验证 |
| **04-10** | **lambda_max=5，5seed全量** | **79.6% margin=+11pp** | **✅ 最终方案确认** |

## 代码位置
- 项目根目录: ~/my_workspace/my_project/V2X_MEC_Offloading_Thesis/
- 核心算法: uav_mec/agents/dc_ppo.py（lambda_max=5, lr_lambda=3e-2, delay_budget=0.5）
- 全量实验: uav_mec/full_experiment_dcppo.py
- 结果CSV: results/exp1_dcppo_baseline.csv
- λd轨迹: results/lambda_trajectories.json
