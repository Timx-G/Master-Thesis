"""Canonical Chinese content for the 2026 midterm-defense visual assets."""

from types import MappingProxyType


def _freeze(value):
    """Recursively expose visual content through read-only containers."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


ALGO_NAMES = _freeze(
    {
        "full": {
            "chinese": "外部性感知与截止时间驱动的联合调度资源分配算法",
            "english": (
                "Externality-Aware and Deadline-Driven Joint Scheduling " "and Resource Allocation"
            ),
            "acronym": "EAD-JSRA",
            "display": "EAD-JSRA（本文算法）",
        },
        "innovation1": {
            "chinese": "干扰外部性感知的跨无人机关联配对算法",
            "english": ("Interference-Externality-Aware Cross-UAV Association " "and Pairing"),
            "acronym": "IEA-CAP",
            "display": "IEA-CAP",
        },
        "innovation2": {
            "chinese": "截止时间驱动的最小成本准入分配算法",
            "english": ("Deadline-Driven Minimum-Cost Admission and Allocation"),
            "acronym": "DD-MCAA",
            "display": "DD-MCAA",
        },
    }
)

METHOD_DISPLAY_NAMES = _freeze(
    {
        "full": "EAD-JSRA（本文算法）",
    }
)

ASSET_SPECS = _freeze(
    {
        "01_background": {
            "stem": "01_background",
            "kind": "figure",
            "title": "选题背景：全频复用多无人机 NOMA-MEC 的任务时限矛盾",
        },
        "02_ead_jsra_framework": {
            "stem": "02_ead_jsra_framework",
            "kind": "figure",
            "title": "总体研究框架：EAD-JSRA 的双创新候选评价闭环",
        },
        "03_iea_cap_flow": {
            "stem": "03_iea_cap_flow",
            "kind": "figure",
            "title": "IEA-CAP：干扰外部性感知的确定性首次改进搜索",
        },
        "04_dd_mcaa_flow": {
            "stem": "04_dd_mcaa_flow",
            "kind": "figure",
            "title": "DD-MCAA：deadline 解析最小需求与最大基数准入",
        },
        "05_progress": {
            "stem": "05_progress",
            "kind": "figure",
            "title": "研究进度：核心研究与正式实验完成，进入论文及毕业交付阶段",
        },
        "table_algorithm_selection": {
            "stem": "table_algorithm_selection",
            "kind": "table",
            "title": "算法选择依据：为什么采用可行初始化、确定性搜索与解析分配",
        },
    }
)

BACKGROUND_CONTENT = _freeze(
    {
        "columns": (
            {
                "heading": "应用需求",
                "items": (
                    "应急巡检",
                    "临时热点",
                    "独立计算任务",
                    "绝对截止时刻",
                ),
            },
            {
                "heading": "系统耦合",
                "items": (
                    "服务归属",
                    "跨 UAV 同频干扰",
                    "NOMA/SIC",
                    "MEC CPU 同时变化",
                ),
            },
            {
                "heading": "研究目标",
                "items": (
                    "联合决定关联、配对、卸载和算力",
                    "最大化按时完成任务数",
                ),
            },
        ),
        "boundary": ("适用边界：固定悬停、集中式控制、时隙级调度；" "不优化轨迹与功率。"),
    }
)

EAD_JSRA_FRAMEWORK = _freeze(
    {
        "steps": (
            "唯一任务与信道快照",
            "每 UAV MILP 可行初始化与 SIC 修复",
            "IEA-CAP 候选调度",
            "统一信道、干扰与 SIC 评价",
            "DD-MCAA 资源准入",
            "完成集合与全网目标",
            "接受或检查下一候选",
        ),
        "loop": "接受候选后重新扫描；否则检查下一候选。",
        "boundary": ("终止仅表示给定邻域下的局部最优，不代表一般规模全局最优。"),
        "evidence_chain": (
            "统一轨迹",
            "配对统计",
            "7/7 门禁",
            "canonical v1",
            "论文图表",
        ),
    }
)

IEA_CAP_FLOW = _freeze(
    {
        "steps": (
            "每 UAV MILP 从正收益单、双用户候选中产生可行初始化",
            "确定性 SIC 修复删除违规弱用户并重新规范角色",
            "生成 Replace、Position、CrossUAV 三类邻域",
            "按实际服务 UAV 重算信道、跨 UAV 干扰、SIC、CPU 和完成集合",
            ("用全网词典序目标 Φ=（完成任务数，总截止裕量，" "负跨 UAV 干扰）判断严格改善"),
            "接受首个改善并重新扫描",
            "无改善时输出给定邻域下的局部最优",
        ),
        "acceptance": "全网首次改进接受准则",
        "boundary": ("全网接受准则不等于一般规模全局最优；" "输出仅保证给定邻域下的局部最优。"),
    }
)

DD_MCAA_FLOW = _freeze(
    {
        "steps": (
            "计算剩余预算 T_i=d_i-a_i",
            "推导最小卸载比例 alpha_i^min",
            "推导最小 MEC CPU f_i^min",
            "筛除分母非正、速率非法或需求超预算的任务",
            "按 (f_i^min, task_id) 稳定排序",
            "选择 CPU 预算可容纳的最长前缀",
            "用统一时延函数复核完成集合",
        ),
        "theorem_boundary": "定理边界：任务价值相同、单 UAV 硬 CPU 预算。",
        "excluded_scope": ("不推广到有权任务、跨 UAV CPU 迁移或长期队列。"),
    }
)

CANONICAL_EVIDENCE = _freeze(
    {
        "version": "canonical v1",
        "on_time_rate": ("EAD-JSRA 按时完成率 98.11%，95% CI [97.94%, 98.26%]"),
        "cross_uav_gain": "IEA-CAP 中 CrossUAV 的独立贡献约 +0.05 个百分点",
        "global_acceptance_gain": ("IEA-CAP 中全网接受准则的独立贡献约 +0.12 个百分点"),
        "resource_gain": ("DD-MCAA 相对 crossing/equal 的贡献为 +8.69 个百分点"),
        "constraint_violations": "最终调度 SIC 与 CPU 预算违规均为 0",
        "runtime_p95": "N=20 调度运行时间 P95 为 40.93 ms",
        "enumeration": ("小规模枚举差距为 0，仅作为已枚举实例的经验参考"),
        "automated_tests": "257 项自动化测试",
        "unseen_seeds": "50 个未见种子",
        "formal_gates": "7/7 正式实验门禁",
        "thesis_draft": "八章论文内容稿",
        "midterm_deck": "17 页中期 PPT",
    }
)

PROGRESS_WORK_PACKAGES = _freeze(
    (
        {
            "name": "可信模型",
            "status": "已完成",
            "detail": "统一物理、时延与可行性评价口径",
        },
        {
            "name": "IEA-CAP",
            "status": "已完成",
            "detail": "跨 UAV 邻域与全网接受准则",
        },
        {
            "name": "DD-MCAA",
            "status": "已完成",
            "detail": "解析最小需求与最大基数准入",
        },
        {
            "name": "实验与证据工程",
            "status": "已完成",
            "detail": "正式实验门禁、canonical v1 与论文图表",
        },
        {
            "name": "论文定稿",
            "status": "进行中",
            "detail": "学校模板适配、文字校核与定稿",
        },
        {
            "name": "毕业答辩",
            "status": "进行中",
            "detail": "答辩材料整合与演练",
        },
    )
)

PROGRESS_EVIDENCE = (
    CANONICAL_EVIDENCE["automated_tests"],
    CANONICAL_EVIDENCE["unseen_seeds"],
    CANONICAL_EVIDENCE["formal_gates"],
    CANONICAL_EVIDENCE["version"],
    CANONICAL_EVIDENCE["thesis_draft"],
    CANONICAL_EVIDENCE["midterm_deck"],
)

ALGORITHM_SELECTION_COLUMNS = ("候选方案", "优势", "局限", "本文定位")

ALGORITHM_SELECTION_ROWS = _freeze(
    (
        {
            "candidate": "完整联合枚举",
            "strength": "可直接比较已枚举实例的联合方案",
            "limitation": "组合规模随任务、信道和配对数快速增长",
            "role": "仅用于小规模经验参考",
        },
        {
            "candidate": "深度强化学习",
            "strength": "适合长期状态演化下的策略学习",
            "limitation": "当前任务强调硬约束、确定性与可复现证据",
            "role": "退出核心路线，保留为方法讨论与对比",
        },
        {
            "candidate": "双边交换匹配",
            "strength": "局部协调清晰，具有可解释的交换机制",
            "limitation": "双边准则无法完整评价第三方干扰外部性",
            "role": "作为强基线和接受准则对照",
        },
        {
            "candidate": "全网联合 MILP",
            "strength": "可显式表达离散决策与硬约束",
            "limitation": "一般规模联合求解的组合复杂度高",
            "role": "MILP 仅用于每 UAV 可行初始化",
        },
        {
            "candidate": "EAD-JSRA",
            "strength": "兼顾硬约束、可解释性与在线运行时间",
            "limitation": "只保证给定邻域下的局部最优",
            "role": "本文总体联合调度资源分配算法",
        },
    )
)
