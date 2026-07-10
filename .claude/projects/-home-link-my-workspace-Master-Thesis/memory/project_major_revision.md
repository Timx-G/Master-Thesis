---
name: IEEE IoTJ Major Revision Status
description: Status and scope of the 2026-04-13 FULL major revision for the multi-UAV NOMA-MEC paper
type: project
---

Paper: "Interference-Aware User Pairing and Task Offloading for Grouped NOMA Multi-UAV MEC Networks"
Target: IEEE Internet of Things Journal (CCF-B)
Status: **FULL MAJOR REVISION COMPLETED 2026-04-13** (5-reviewer, all critical issues addressed)

**Why:** 5 reviewers all returned Major Revision with critical issues in theory, experiments, statistics, writing, and positioning.

**Critical changes made (2026-04-13 session):**

### Experiments (code changes):
- Added `exchange_stable_cross_uav_pairing()` to `baselines_pairing.py` — ExStable+CrossUAV ablation
- Added `strategy_exchange_stable_cross_uav()` to `run_main_experiment.py`
- Updated methods dict to include ExStable+CrossUAV
- Fixed rank-biserial formula: r = 1 - 2*min(W+, W-)/(n*(n+1)/2)
- Updated statistical output to show exact W, Holm multipliers, corrected p-values
- Re-ran full 20-seed experiment: ExStable+CrossUAV=69.6% (vs ExStable=69.2%, +0.44pp, p=0.0001, r=0.46)

### Paper changes (main_v2.tex):
1. Abstract: removed "best matching-based baseline (exchange-stable)" claim; added ExStable+CrossUAV key result
2. Introduction C3: removed specific experimental numbers; added ExStable+CrossUAV ablation
3. Introduction comparison: softened "unexplored" → "underexplored joint combination"
4. Table 1 footnotes: clarified xu2025dynamic is single-BS (no X-UAV Interf); expanded li2023joint footnote
5. System model: added SIC feasibility constraint eq (eq:sic_feasibility); stated Urban Macro (UMa) params
6. System model: added noise figure (10dB NF); effective noise = -107 dBm/subcarrier
7. System model: added Jensen inequality bias quantification (~≤0.5pp)
8. Algorithm: renamed Stage 1b→Stage 2 (swap), Stage 2→Stage 3 (KKT); algorithm now has 3 stages
9. Algorithm 1 caption: removed experiment variant definition; expanded pseudocode for swap ops
10. Complexity: clarified O(N³) per iteration (N² candidates × N recomputation); justified O(N²) for fixed K,S
11. NP-hardness: added "Proof strategy" paragraph explaining why specific instance → general NP-hardness; renamed Steps 3/4 to Completeness/Soundness
12. Notation table: added f_{alloc,i}
13. Results: added ExStable+CrossUAV row to Table 2 (Tab:main_results) with footnote
14. Results: fixed statistical reporting with exact W values (W=210 for Ours vs Greedy), Holm multipliers, corrected p-values
15. Results: clarified Cohen's d_paired vs d_unpaired (reviewer was using d_unpaired≈0.38; paper uses d_paired=1.01)
16. Results: added "Critical ablation --- ExStable+CrossUAV" analysis paragraph
17. Results: strengthened "Why Greedy < Random" with physical explanation of cross-UAV interference concentration
18. Results: added K=1 spatial coverage analysis (21.7dB path-loss penalty for corner users)
19. Results: clarified Jain Fairness Index input vector (per-slot deadline rates q_t)
20. Results: added S-scaling monotonic decrease physical explanation (bandwidth per sub + utilization)
21. Results: condensed DRL section (PPO/MAPPO) from ~30 lines to ~8 lines
22. Results: removed EE table entirely (acknowledged to track deadline_rate only)
23. Related Work: softened "Previously Unexplored" → enumerated 4 formal distinctions from multi-cell NOMA
24. Related Work: added concrete technical comparison with li2023joint (orthogonal bands = OFDMA(OrthoBank))
25. Related Work: added ExStable+CrossUAV as evidence that bilateral stability insufficient
26. Conclusion: updated to include ExStable+CrossUAV result as key finding #1
27. Appendix: condensed from ~90 lines to ~25 lines; moved detailed proof to "online supplementary"
28. Paper: added "pp" definition on first use (i.e., absolute difference in %)
29. Institutional: replaced "XXX University" with USTC affiliation
30. Baselines list: updated count from 10 → 13; added ExStable+CrossUAV; expanded OFDMA to two items
31. Trimmed verbose sections: Remarks 1a/1b condensed; Remark 3 (Lemma scope) condensed; Di et al. distinction condensed; DRL section condensed; li2023joint/multi-cell NOMA formal distinction condensed

**Paper metrics after revision:**
- Lines: 1176 (original: 1222, after early edits: 1310, now reduced)
- PDF size: 461KB (was ~485KB)
- Figures: 8, Tables: 10
- New experiments saved to: results/phase2_main_experiment.json

**Key finding from new experiment:**
ExStable+CrossUAV = 69.6% (+0.44pp over ExStable=69.2%), proving that bilateral stability criterion is the bottleneck, not cross-UAV topology. Ours=73.2% is +3.6pp beyond ExStable+CrossUAV (r=1.00, all 20 seeds). This definitively resolves R3-C3.

**How to apply:** When continuing the revision, focus on any remaining issues:
- Page length still may exceed 14 pages (further cuts needed if editor requests)
- \! negative spacing not yet removed (minor cosmetic issue)
- Some sections still verbose but content is correct

## Round 2 Revision (2026-04-13, same session)

**CR1-CR5 Critical fixes:**
- CR1: Holm 乘数逻辑修正：Greedy(最显著) ×13，BG+Swap(最不显著) ×1；代码 tie-handling 修复
- CR2: SIC 硬过滤器：γ_SIC_min=0dB 加入 ILP+BG 初始化（<0.1% pairs 过滤）
- CR3: NP-hardness Soundness: 改为 2-subcarrier gadget (coverage + weight subcarrier，link users 共享元素)
- CR4: 3篇真实文献加入：Lai2022(TMC), Zhang2021(TCOMM), Ebrahim2023(Access)，作者已填入
- CR5: MAPPO 图已存在，加入"确认收敛非失败"说明

**MJ1-MJ7 修复:**
- MJ1: 复杂度统一为 O(N²)，实测 O(N^1.7)，删除矛盾陈述
- MJ2: ExStable+CrossUAV 跨UAV swap接受率 = 0/180 = 0%（实验验证），双边条件结构性阻止
- MJ3: Jensen 偏差公式修正（Rician Var=(1+2κ)/(1+κ)²，κ=1时std≈0.87非0.5）
- MJ4: K=1 完整链条：路损→速率(1.07Mbps)→延迟(0.072s<0.8s)→高性能来自算力集中
- MJ6: Finding3 vs ILP 矛盾解决（ILP 在限制迭代时仍有统计价值）
- MJ7: γ_SIC_min=0dB 添加信息论依据和 Saito2013/Ding2019 引用

**论文当前状态:** 1222 lines, 48 refs, PDF=470KB。所有 46 个引用已解析，编译通过。

## Round 3 Revision (2026-04-13, same session)

**FATAL issues fixed:**
- FATAL-1: Holm stats — Added Tab:holm with correct complete table. W=153 is from SEPARATE test (ExStable+CrossUAV vs ExStable), NOT part of 13-comparison Holm family. BG+Swap updated: W=192, p=2.41e-4, Holm×1. ExStable+CrossUAV ablation text fixed to say "standalone test, W=153, p=1.47e-4" without applying wrong Holm multiplier.
- FATAL-2: K=1 calc errors — Fixed: t_local_full=0.75s (not 0.075s), α*=0.229 (not 0.032), τ=0.578s (not 0.072s). K=1 CAN meet deadline via 22.9% offloading; high performance is compute-concentration not coverage.
- FATAL-3: NP-hardness S=2n → S=2|T|. Added that S=2|T|≤2n³ is polynomial.
- FATAL-4: 18→14 pages. Key cuts: Related Work condensed 4 subsections→3 paragraphs, Conclusion 5 findings→3, removed overhead table, compressed block-fading/Jensen/SIC/task sections, moved Appendix to brief outline.

**MJ issues fixed:**
- MJ1: Complexity unified as O(N²) overall, O(N^1.3) empirical
- MJ2: Proposition 3 added (bilateral instability of cross-UAV moves — structural proof)  
- MJ3: K=1 narrative fixed (compute-concentration, not coverage violation)
- MJ4: yang2022cost distinction clarified (user-BS allocation vs SIC pairing)
- MJ5: Jensen bias uses only analytic bound (CSI proxy note added as disclaimer)
- MJ6: ILP vs Finding3 addressed in condensed conclusion

**Priority 3 fixed:**
- Algorithm goto → WHILE + improved flag
- MAPPO: clarified "doesn't converge (oscillates 40-65%), 46.6% is final checkpoint"
- Conclusion C1 (NP-hardness finding included)
- 73.2%→73.3% updated throughout (minor precision change from full-precision JSON)
- 0.44pp unified (was 0.4/0.44pp inconsistency)

**Paper final state:** 1045 lines, 14 pages, 447KB, 8 figs, 10 tables, 48 refs, all citations resolve.
**Code state:** baselines_pairing.py (SIC 0dB filter in ILP only), pairing.py (no filter in BG), run_main_experiment.py (4-decimal precision, correct Holm).

## Round 4 Revision (2026-04-13, same session)

**FATAL-NEW-1: W=153 p值 (R3)** — 审稿人p≈0.038是错误的（漏掉3个零差分）。正确：n_eff=17（3零排除），W=153=W_max(n_eff=17)，p_exact=7.63e-6（远比报告更显著！）。论文已更新：添加n_eff=17说明和精确p值。

**FATAL-NEW-2: NP-hardness Soundness (R1)** — 彻底重写：从3DM gadget改为MWSP减归（Garey-Johnson SP7）。方向：MWSP ≤_p NOMA-Pairing。每个MWSP集S_j映射到4用户NOMA配置；信道增益构造实现指定权重；用户互斥≡集合不相交。无soundness缺口。

**MAJOR-NEW-3: Proposition 3 (R1/R2)** — 降级为"Structural insight (empirical observation)"，承认形式化证明需要额外参数假设，报告0/180实验验证。

**MAJOR-NEW-4: 复杂度1.7→1.3 (R2)** — 两处O(N^1.7)改为O(N^1.3)（正确计算：log(59/7)/log(50/10)≈1.30）。

**M1: yang2022cost区分 (R4)** — 更新为5点列表，分别说明yang2022cost(user-BS分配≠SIC pairing)、ni2021ris(随机几何≠槽级调度)、liu2023hybrid(功率/模式选择≠配对)、全重用vs部分重用、非光滑目标。

**Priority 3** — +0.44pp全文统一（结论第914行已修），Unicode ×→$\times$，Conclusion加入C1 NP-hardness发现。

**Paper final state (Round 4):** ~1049 lines, 14 pages, 448KB, 48 refs (42 cited), clean compile.

## Round 5 Revision (2026-04-13, same session)

**FATAL-A: SP7引用错误** — SP7=SET SPLITTING (错误)。改为GT20 (MAXIMUM INDEPENDENT SET, Garey-Johnson)。论文C1和Appendix均已更新。

**FATAL-B: NP-hardness Soundness** — 彻底重写：使用MWIS归约+唯一dummy用户方案。每个NOMA配置c_j有UNIQUE dummy用户d_j(h=10^{-3})和共享元素用户(h=10^{-6})。d_j在且仅在c_j中出现→只有c_j能实现w=1→soundness无缝。共享元素用户编码图边，用户互斥≡图独立集。用户-节点互斥证明正确。

**FATAL-C: W=153与r=0.46矛盾** — 修复为r=1.00。审稿人用n=20计算(错误)。正确：n_eff=17(3零差排除)，W=153=W_max(n_eff=17)，T=min(153,0)=0，r=1-2*0/153=1.00。所有文中r=0.46已改为r=1.00，添加T=0解释。

**Blocker B1: ILP主导→Swap主导** — 复杂度O(N²)来自ILP枚举，但运行时被Swap主导(11-13ms vs ILP<1ms)。两处修正。

**Blocker B2: 5 seeds→20 seeds, 0/180→0/660** — 实验重跑：20 seeds×100 slots确认0/660接受率。

**Blocker B3: "rarely"→"never"** — 改为"never sufficiently offsets (0/660)"。

**Minor: Abstract≤150词** — 从284词压缩到114词。

**Paper final state (Round 5):** 1046 lines, 14 pages, 450KB, clean compile. MWIS proof correct. All stats consistent.

## Round 6 Revision (2026-04-13, same session)

**六审结果:** R1+R3 和 R2+R4+R5 一致 Minor Revision（六轮来首次一致！）

**M1: MWIS→IS (GT20)** — GT20=无权重IS（非MWIS）。归约用w=1（IS，非加权）。全文MWIS→IS，包括Abstract、C1、Appendix、Conclusion。N2顺便清除了"MWSP"笔误。

**M2: NOMA 2+2用户约束** — Appendix明确说明每个配置c_j的4个用户：UAV1={d_j,p_j^{(1)}}（unique dummy h=10^{-3} + unique padding h=10^{-6}）；UAV2按度数分情况：deg=0用两个padding，deg=1用edge-user+padding，deg≥2用前两个edge-users，deg>2需辅助子载波（standard gadget，不影响IS论证）。

**N2: MWSP→IS** — 结论第913行 proved via IS reduction。
**N3: 0/660定义** — 明确"660 individual swap candidate evaluations"。
**N4: 11.1pp舍入** — 添加"using full-precision seed means; rounded 73.3%-62.1%=11.2pp"。
**N5: liu2023hybrid公式** — 添加"their Problem~1 and Algorithm~1"。

**Paper final state (Round 6):** 1049 lines, 14 pages, 451KB, CLEAN compile. MWIS/MWSP=0。

## Round 7 Revision (2026-04-13, same session)

**七审结果:** Minor Revision（审稿人明确表示完成后直接推荐Accept，无需第八轮！）

**Fix①: deg>2 aux subcarrier gadget (Major):**
- 旧文本: "standard gadget, does not affect the k-IS argument" (空洞断言)
- 新文本: 完整aux gadget构造：
  * S = k + |E_ovf| 子载波（overflow边各一个aux子载波）
  * 每条overflow边(j,k): c_j^{(jk)}={d_j^{(jk)},p_j^{(jk)},e_{jk}^(1),e_{jk}^(2)}, c_k^{(jk)}={d_k^{(jk)},p_k^{(jk)},e_{jk}^(1),e_{jk}^(2)}
  * 共享e_{jk}^(1/2)阻止c_j^{aux}和c_k^{aux}同时被选
  * Bundle requirement: 选顶点v_j意味着选其所有aux config → 如果v_j和v_k同时被选，需要同时选c_j^{aux}和c_k^{aux}但它们冲突→矛盾
  * S = O(k+|E|) 多项式，soundness正确

**Fix②: +11.1pp → +11.2pp (Significant):**
- 原因: 73.3% - 62.1% = 11.2pp（显示值算术）
- 实际gap = 11.1485pp → 四舍五入为11.1pp（全精度计算）
- 选择Option B: 主结果段落改为+11.2pp，删除括号说明
- 保持表格gap列为+11.1（由全精度数据计算）

**Optional fixes:**
- Abstract: "proved via IS reduction (Problem GT20, [garey1979computers])"
- liu2023hybrid: "their problem formulation and proposed algorithm"

**Paper final state (Round 7):** 1049 lines, 14 pages, 451KB, clean compile. All 8 checks pass.
**STATUS: READY FOR SUBMISSION**

## Final Consistency Fix (2026-04-13, end of session)

**+11.1pp vs +11.2pp 全文统一 (回滚至11.1pp):**
- 理由：实验真实计算值 11.1485pp → 四舍五入11.1pp（正确）；11.2pp是73.3-62.1显示精度假象
- Abstract第28行、主结果第600行、Cohen'd第663行 → 全部改回+11.1pp
- 全文18处11.1pp，0处11.2pp，完全一致

**Final paper state (投稿就绪):**
- 14 pages, 450KB, 1049 lines, CLEAN compile
- 11.1pp unified (18 occurrences, 0 for 11.2pp)  
- No 3DM in main text, no Master thesis in thanks
- IS GT20 reduction consistent throughout
- Abstract clean (no raw W/r/p stats)
- SINR formulas correct (verified against code)
- Channel model correct (η dB→linear conversion in code)
