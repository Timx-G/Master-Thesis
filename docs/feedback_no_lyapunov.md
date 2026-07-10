---
name: 禁用Lyapunov优化
description: 用户明确拒绝Lyapunov方法，认为过时（2024-2025顶刊仅5-8%），任何方案不得使用
type: feedback
---

绝不使用Lyapunov优化方法。

**Why:** 用户调研发现2024-2025年顶刊论文中Lyapunov仅占5-8%，认为审稿人会质疑方法新颖性，降低接受率。用户原话："还在用Lyapunov? 你刚才得出的结论让你吃了？"

**How to apply:** 长期约束处理一律使用Lagrangian对偶方法（参考Tessler et al. 2019 ICLR），或其他2024-2025最新方法。对在线优化问题也不建议Lyapunov。
