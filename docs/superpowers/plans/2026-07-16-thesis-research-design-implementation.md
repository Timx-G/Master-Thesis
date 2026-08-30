# 论文研究设计实施计划

> **执行要求：** 按任务顺序实施；每项行为变更先写失败测试，再写最小实现，并在任务结束时运行所列回归测试。当前工作区包含尚未提交的相关改动，实施期间只做逐文件差异审计，不自动提交或覆盖既有修改。

**目标：** 将 `2026-07-15-thesis-research-design.md` 中的周期任务、IEA-CAP、DD-MCAA、联合实验和可审计证据协议完整落实到现有 Python 仿真平台。

**架构：** 先固定任务、CPU 与目标函数语义，再扩展邻域和搜索控制；随后让资源响应模块真正嵌入每次候选评价，最后补齐 Exact、正式实验注册、统计门禁、断点续跑和 canonical v2。所有方法共享 `ExperimentTrace`，所有结果经统一 `SlotOutcome` 和同一分析链路输出。

**技术栈：** Python 3.11、NumPy、SciPy、pytest、TOML、CSV/JSONL、配对 bootstrap/Wilcoxon/Holm。

---

### Task 1：周期任务配置与稳定作业身份

**Files:**
- Modify: `uav_mec/model.py`
- Modify: `uav_mec/domain.py`
- Modify: `uav_mec/trace.py`
- Modify: `tests/test_trace.py`
- Modify: `tests/test_domain.py`

- [ ] **Step 1：写周期轨迹失败测试**

新增测试，要求 `arrival_mode="periodic"` 时每个用户只在 `phase + q*period` 到达，同一用户的 `data_bits`、`cycles_per_bit` 和 `deadline_s` 跨作业保持不变，`job_index` 从 0 严格递增，作业身份不再依赖 `slot*num_users+user`。

```python
def test_periodic_trace_freezes_user_profiles_and_job_identity():
    cfg = ScenarioConfig(
        num_users=6,
        num_slots=9,
        task=TaskConfig(
            arrival_mode="periodic",
            period_choices=(2, 2, 3),
            period_jitter_slots=0,
            t_min=0.6,
            t_max=1.0,
        ),
    )
    trace = generate_trace(cfg, 1000)
    by_user = {
        user: [task for slot in trace.slots for task in slot.tasks if task.user_id == user]
        for user in range(cfg.num_users)
    }
    for tasks in by_user.values():
        assert tasks
        assert [task.job_index for task in tasks] == list(range(len(tasks)))
        assert len({task.data_bits for task in tasks}) == 1
        assert len({task.comp_cycles / task.data_bits for task in tasks}) == 1
        assert len({task.deadline_s for task in tasks}) == 1
        gaps = [right.arrival_slot - left.arrival_slot for left, right in zip(tasks, tasks[1:])]
        assert len(set(gaps)) <= 1
    assert len({(task.user_id, task.job_index) for slot in trace.slots for task in slot.tasks}) == sum(
        len(slot.tasks) for slot in trace.slots
    )
```

- [ ] **Step 2：运行测试并确认因缺少配置字段失败**

Run: `/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_trace.py::test_periodic_trace_freezes_user_profiles_and_job_identity -q`

Expected: `TaskConfig.__init__()` 报告未知的 `arrival_mode`。

- [ ] **Step 3：实现周期配置和每用户冻结 profile**

在 `TaskConfig` 增加并校验：

```python
arrival_mode: str = "bernoulli"
period_choices: tuple[int, ...] = (2, 2, 3)
period_jitter_slots: int = 0
```

在 `Task` 增加 `job_index: int = 0` 并校验非负整数。`generate_trace` 为周期模式使用独立 profile RNG，预生成每用户 `(period, phase, data_bits, density, deadline)`；发布时刻只允许严格递增，抖动为 0 或 1 个时隙且不得造成重复/倒序。伯努利模式保留原随机流和旧 `task_id`，周期模式的整数 `task_id` 使用 `(user_id << 32) | job_index`。

- [ ] **Step 4：补充无效配置和抖动测试**

覆盖空周期集合、非正周期、非法模式、负抖动、抖动大于 1，以及同种子完全复现。

- [ ] **Step 5：运行任务语义回归**

Run: `/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_domain.py tests/test_trace.py -q`

Expected: all passed。

### Task 2：异质 CPU 预算和规范化全网目标

**Files:**
- Modify: `uav_mec/evaluation.py`
- Modify: `uav_mec/domain.py`
- Modify: `uav_mec/scheduler.py`
- Modify: `tests/test_evaluation.py`
- Modify: `tests/test_search.py`

- [ ] **Step 1：写目标函数失败测试**

```python
def test_network_objective_prefers_lower_minimum_mec_cpu_after_cardinality():
    high = outcome(completed=2, total_mec_cpu_hz=2_000_001.0, interference=1e-9)
    low = outcome(completed=2, total_mec_cpu_hz=2_000_000.0, interference=2e-9)
    assert network_objective(low) > network_objective(high)
```

另写测试要求 `EvaluationConfig.cpu_per_uav_hz=(2e9, 4e9)` 时分别按对应预算分配，长度与 `num_uavs` 不一致时报错。

- [ ] **Step 2：运行测试并确认旧 slack 目标失败**

Run: `/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_search.py tests/test_evaluation.py -q`

Expected: 新测试失败，旧测试继续通过。

- [ ] **Step 3：实现 CPU 成本字段与预算解析**

`SlotOutcome` 增加 `total_mec_cpu_hz`，由所有 `completed and alpha>0` 决策的 `cpu_hz` 稳定求和。`EvaluationConfig` 接受标量或 tuple，并提供：

```python
def cpu_budget_for(self, uav: int, num_uavs: int) -> float:
    ...
```

标量扩展为同质预算；tuple 必须长度等于 `num_uavs`。

- [ ] **Step 4：替换目标并冻结量化**

```python
def network_objective(outcome: SlotOutcome) -> tuple[int, int, int]:
    return (
        outcome.completed_count,
        -round(outcome.total_mec_cpu_hz),
        -round(outcome.total_cross_interference_w / 1e-18),
    )
```

所有事件和 Exact 使用该目标；不再用 `total_slack_s` 接受候选。

- [ ] **Step 5：运行目标与资源回归**

Run: `/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_evaluation.py tests/test_search.py tests/test_exact.py tests/test_resource_allocator.py -q`

Expected: all passed。

### Task 3：完整 Replace/Position/CrossUAV 邻域

**Files:**
- Modify: `uav_mec/scheduler.py`
- Modify: `tests/test_neighborhoods.py`

- [ ] **Step 1：写空位置动作失败测试**

测试邻域包含：未调度任务插入非满块、已调度任务删除、同 UAV 重定位、跨 UAV 重定位、同 UAV 交换和跨 UAV 交换；每个结果经 `normalize_roles` 后合法且唯一。

- [ ] **Step 2：运行并确认 Insert/Drop/Relocate 缺失**

Run: `/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_neighborhoods.py -q`

Expected: 新动作测试失败。

- [ ] **Step 3：实现完整动作集合**

移除 Replace 的最近 UAV 限制。用统一 block occupancy 枚举：

```python
MoveType = Literal[
    "replace", "insert", "drop", "position_swap", "intra_relocate",
    "cross_swap", "cross_relocate",
]
```

每个动作后规范化 NOMA 角色，按 `(move_type, stable_key)` 排序去重。

- [ ] **Step 4：运行邻域和调度结构回归**

Run: `/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_neighborhoods.py tests/test_schedule.py -q`

Expected: all passed。

### Task 4：最佳改进、多起点和公平结构消融

**Files:**
- Modify: `uav_mec/scheduler.py`
- Modify: `uav_mec/experiment.py`
- Modify: `tests/test_search.py`
- Modify: `tests/test_experiment_cli.py`

- [ ] **Step 1：写最佳改进和多起点失败测试**

构造一个首个改善候选次于后续候选的实例，要求 `improvement="best"` 选择本轮最大目标；多起点结果必须等于三个终止解中的最大 `(Phi, initializer_id, stable_key)`。

- [ ] **Step 2：运行并确认 SearchConfig 不支持新策略**

Run: `/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_search.py -q`

- [ ] **Step 3：实现 SearchStrategy 和 optimize_multistart**

`SearchConfig` 增加 `improvement`、`initializer_id`；global 最佳改进按量化 `Phi` 排序，bilateral 只能按参与块效用排序。新增 `optimize_multistart(initializers, ...)`，完整方法默认 ILP、IA-Greedy、fair-load greedy 三起点。

- [ ] **Step 4：锁定 H1/H2 控制变量**

`full`、`intra_global`、`cross_bilateral` 共享三起点、最佳改进、候选穷尽、tie-break 和停止条件；前者只分别关闭 CrossUAV 或替换为 bilateral 排序/接受。`full_single_first` 作为独立方法保留。

- [ ] **Step 5：运行搜索与 CLI 回归**

Run: `/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_search.py tests/test_experiment_cli.py tests/test_mechanism_controls.py -q`

### Task 5：外部性证书与增量精确评价

**Files:**
- Modify: `uav_mec/evaluation.py`
- Modify: `uav_mec/scheduler.py`
- Modify: `uav_mec/experiment.py`
- Modify: `tests/test_search.py`
- Create: `tests/test_incremental_evaluation.py`

- [ ] **Step 1：写第三方资源变化和全量等价失败测试**

测试证书同时记录参与者/第三方的完成、CPU、rate、SINR、SIC 差异及逐接收端子载波 `delta_interference`。随机合法动作的增量结果必须与完整评价的 decisions、rates、Phi 完全一致。

- [ ] **Step 2：实现 `EvaluationCache` 和依赖闭包**

缓存 `(uav,subcarrier)` 的接收记录。动作涉及不同子载波时更新最多 `2K` 个同频接收块；收集速率或归属改变的任务，并对包含这些任务的所有 MEC 服务器重新执行当前资源策略。

- [ ] **Step 3：候选接受前做一次完整复核**

增量评价只用于候选筛选；选出的最佳候选必须完整重算且 `Phi`、约束、证书一致，否则抛出硬错误。

- [ ] **Step 4：运行 10,000 动作等价测试和搜索回归**

Run: `/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_incremental_evaluation.py tests/test_search.py -q`

### Task 6：资源响应嵌入无线搜索和 2×2 联合实验

**Files:**
- Modify: `uav_mec/experiment.py`
- Modify: `uav_mec/run_thesis_suite.py`
- Modify: `tests/test_resource_policies.py`
- Modify: `tests/test_thesis_suites.py`

- [ ] **Step 1：写嵌入策略失败测试**

对 `full_edf_min_cpu` 构造 DD 与 EDF 对候选排序不同的实例，断言搜索 events 的 `after` 来自 EDF，而不是最终阶段才替换资源策略。

- [ ] **Step 2：删除固定 DEADLINE 搜索策略**

将 `_slot_method` 中的：

```python
search_evaluation = replace(target_evaluation, allocation_policy=AllocationPolicy.DEADLINE)
```

替换为候选评价和最终评价均使用 `target_evaluation`。四个联合方法共享轨迹、初始化器和候选预算。

- [ ] **Step 3：增加冻结 joint suite**

方法固定为 `exchange_edf`、`exchange_dd`、`iea_edf`、`iea_dd`，输出种子级两主效应和交互项。

- [ ] **Step 4：运行资源策略与 suite 回归**

Run: `/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_resource_policies.py tests/test_thesis_suites.py -q`

### Task 7：完整字典序 Exact 证据

**Files:**
- Modify: `uav_mec/exact.py`
- Modify: `uav_mec/run_exact_benchmark.py`
- Modify: `tests/test_exact.py`
- Modify: `tests/test_exact_benchmark.py`

- [ ] **Step 1：写三层 gap 失败测试**

要求 exact 输出量化 `Phi`，benchmark 输出 `completion_gap`、条件 `cpu_gap_hz`、条件 `interference_gap_w`、`cardinality_optimal` 和 `lexicographic_optimal`。

- [ ] **Step 2：实现完整目标比较和非平凡性统计**

只有三层量化目标相等才计入 `lexicographic_optimal`；另统计 `exact_completed < arrived_offload_tasks` 的时隙比例。

- [ ] **Step 3：运行 Exact 回归**

Run: `/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_exact.py tests/test_exact_benchmark.py -q`

### Task 8：正式实验注册与 Holm 功效冻结

**Files:**
- Create: `uav_mec/experiment_registry.py`
- Create: `uav_mec/power_analysis.py`
- Create: `configs/thesis/experiment_registry_v2.toml`
- Create: `tests/test_experiment_registry.py`
- Create: `tests/test_power_analysis.py`

- [ ] **Step 1：写种子池互斥和完整 Holm 功效测试**

注册表必须拒绝重叠种子池、未知方法、缺失 MDE 或主假设比较。功效模块使用 pilot 配对差 bootstrap 模拟 H1–H4 完整 Holm 决策并只从 `{50,100,200}` 选择样本量。

- [ ] **Step 2：实现只读冻结注册表**

注册 Resource/Externality/Main/Robustness/Exact/Joint/Scaling 的种子范围、时隙、主比较、MDE 和配置哈希。正式运行不得接受 CLI 覆盖 seed/slot。

- [ ] **Step 3：运行注册和统计回归**

Run: `/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_experiment_registry.py tests/test_power_analysis.py tests/test_statistics.py -q`

### Task 9：九点外部性、现实轨迹和运行成本审计

**Files:**
- Modify: `uav_mec/run_thesis_suite.py`
- Modify: `uav_mec/externality_analysis.py`
- Create: `uav_mec/geolife.py`
- Create: `uav_mec/runtime_audit.py`
- Modify: `tests/test_thesis_suites.py`
- Create: `tests/test_geolife.py`
- Create: `tests/test_runtime_audit.py`

- [ ] **Step 1：写冻结 9 点设计测试**

外部性 suite 只能包含 `2^3` 端点和中心点；overlap 必须按活跃用户两个最佳 UAV 大尺度增益 3 dB 规则计算并资格校验。

- [ ] **Step 2：实现 GeoLife ENU 管线**

验证压缩包 SHA-256，只读带方式标签的步行/骑行轨迹，WGS-84 投影到局部 ENU，按时隙重采样，只允许平移、旋转和裁剪，不允许尺度缩放；输出源参与者 ID 供聚类 bootstrap。

- [ ] **Step 3：实现运行成本审计**

K3N45/K5N75 各 `5 seeds × 20 slots`，输出 P50/P95、候选数、峰值内存、日志字节和总 CPU 时间预测。

- [ ] **Step 4：运行 suite/轨迹/审计回归**

Run: `/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_thesis_suites.py tests/test_externality_analysis.py tests/test_geolife.py tests/test_runtime_audit.py -q`

### Task 10：可恢复分片、分析门禁和 canonical v2

**Files:**
- Modify: `uav_mec/artifacts.py`
- Modify: `uav_mec/run_thesis_suite.py`
- Modify: `uav_mec/analyze_results.py`
- Modify: `uav_mec/publish_canonical.py`
- Modify: `tests/test_artifacts.py`
- Modify: `tests/test_result_analysis.py`
- Modify: `tests/test_canonical_publication.py`

- [ ] **Step 1：写中断恢复和不可变分片测试**

每个 `(family,case,seed-shard)` 原子写出数据、配置和哈希；重复运行跳过哈希一致的完成分片，拒绝覆盖哈希不一致分片。

- [ ] **Step 2：实现 H1–H4 与正确性硬门禁**

分析输出配对 bootstrap CI、Wilcoxon、Holm、效应量、Exact 三层 gap、约束违规、增量等价和运行时门禁。任何硬门禁失败都禁止发布 v2。

- [ ] **Step 3：扩展 canonical 发布族**

发布 Main、Externality、Resource、Exact、Joint、Scaling、Robustness、Trace-driven，并生成 `claim_matrix.csv`、`checksums.sha256` 和源码快照哈希。

- [ ] **Step 4：运行证据链回归**

Run: `/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_artifacts.py tests/test_result_analysis.py tests/test_canonical_publication.py -q`

### Task 11：全量验证和首批 smoke

**Files:**
- Modify: `configs/thesis/smoke.toml`
- Modify: `Makefile`
- Modify: `README.md`

- [ ] **Step 1：运行格式和全量测试**

Run: `/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest -q`

Expected: 0 failures。

- [ ] **Step 2：运行周期主场景单种子 smoke**

Run: `/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.experiment --config configs/thesis/smoke.toml --output artifacts/smoke/research-design-v2`

Expected: 完整生成 manifest、tasks、slots、events、summary、checksums，零约束违规。

- [ ] **Step 3：运行 Exact 和 joint smoke**

使用开发种子各运行 1–2 个时隙，验证三层 gap、嵌入资源策略和 2×2 交互输出。

- [ ] **Step 4：审计差异和产物**

确认未修改无关文件；记录测试数量、smoke 路径、运行时间和仍未启动的正式实验。正式种子只能在功效配置冻结后打开。
