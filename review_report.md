## REVIEWER 2 REPORT

### Summary
The paper investigates joint user pairing and computation offloading in a multi-UAV MEC network employing grouped NOMA with shared spectrum. The authors formulate the problem as a combinatorial optimization task, incorporating cross-UAV interference which is often ignored in orthogonal multi-UAV MEC studies. To solve this NP-hard problem, the authors propose a two-stage algorithm comprising a per-UAV ILP initialization and a cross-UAV swap refinement, followed by a conditionally optimal offloading ratio computation. Extensive simulations demonstrate that the proposed method outperforms several baselines, including intra-UAV swap, matching-based approaches, and multi-agent reinforcement learning. While the algorithmic design is practical and the results are promising, there are several theoretical and analytical claims that require rigorous revision, particularly regarding the NP-hardness reduction, the optimality of the offloading ratio, and the approximation guarantees.

### Strengths
1. **Explicit Modeling of Cross-UAV Interference:** The paper correctly identifies and models the coupling effect of cross-UAV interference in full-frequency-reuse NOMA, moving beyond the simplified orthogonal spectrum assumption.
2. **Effective Cross-UAV Swap Mechanism:** The extension of swap matching to the cross-UAV domain is a practical and computationally efficient approach to handling inter-cell externalities, as evidenced by its fast empirical convergence.
3. **Comprehensive Baseline Comparisons:** The evaluation against diverse baselines—ranging from theoretical matching (Exchange-Stable) to MAPPO—provides a thorough contextualization of the proposed algorithm's performance.
4. **Reproducibility Details:** The explicit discussion of computation overhead, algorithmic complexity, and hardware specifications is highly commendable.

### Major Concerns (Must Fix)

1. **Proposition 1 (NP-hardness proof rigor):**
   *Gap:* The proof sketch and Appendix A prove NP-hardness by restricting the problem to a specific instance ($f_{\text{UAV}} = f_{\text{threshold}}$). While showing a special case is NP-hard implies the general problem is NP-hard, the formal decision version of the general joint problem is never explicitly stated. Furthermore, the reduction from 3-Dimensional Matching builds a synthetic channel environment ($h=10^{-6}$ vs $h=10^{-3}$) to artificially force $w(c) \in \{0, 1\}$. 
   *Fix Needed:* Explicitly define the decision version of the problem before Proposition 1. Ensure Appendix A maps the 3DM sets to the pairing sets cleanly, showing that the artificial channel gains do not violate the geometric distance models established in Section III.A.

2. **Lemma 1 and "Optimal" Offloading Ratio Claim:**
   *Gap:* Section IV.C (Lemma 1) claims a "Closed-Form Optimal Offloading Ratio". However, Remark 3 admits this ratio is only conditionally optimal given a fixed $n_{\text{offload}}$, artificially maintained by clipping $\alpha \geq 0.01$. The paper notes a true iterative best-response solver yields an additional 1.3 percentage point improvement. Calling the clipped closed-form solution "optimal" for the joint problem is misleading.
   *Fix Needed:* Revise the terminology. Rename Lemma 1 to "Conditionally Optimal Offloading Ratio". Explicitly state that it assumes a fixed active set (enforced by the numerical clip) and is an approximation of the globally optimal offloading assignment.

3. **Proposition 2 (Convergence Bound):**
   *Gap:* Proposition 2 proves the swap terminates in at most $|\mathcal{A}|$ iterations because the objective is an integer bounded by $|\mathcal{A}|$. This is a trivial upper bound. Furthermore, the paper uses a "first-improvement" strategy rather than a "best-improvement" strategy.
   *Fix Needed:* Briefly justify the choice of "first-improvement" over "best-improvement" regarding empirical speed vs. convergence steps. Acknowledge that while the outer loop is bounded by $|\mathcal{A}|$, the number of individual swap evaluations is $O(|\mathcal{A}| \cdot N_{\text{served}}^2)$.

4. **ILP Initialization vs. Greedy:**
   *Gap:* The proposed algorithm uses a per-UAV ILP for initialization. However, Table III shows BG+Swap achieves 72.5% while ILP+Swap achieves 73.2%—a marginal 0.7pp difference. Solving an NP-hard ILP is structurally less scalable than a greedy approach.
   *Fix Needed:* Discuss whether the 0.7pp benefit justifies the exponential worst-case complexity of the ILP initialization. Consider recommending BenefitGreedy+Swap for larger $N$, or clearly delimit the $N$ for which the ILP remains tractable.

5. **Upper Bound and Generalization:**
   *Gap:* Section III.D presents a "relaxed UB" of 93.2%, acknowledging it is weak. To provide a tight optimality gap, the paper relies on an exhaustive search for $N=8$, claiming 94.8% of the true optimum. Combinatorial gaps often scale non-linearly.
   *Fix Needed:* Explicitly caveat that the 94.8% optimality ratio observed for $N=8$ may not generalize to $N=20$ or $N=30$. The empirical monotonic scaling is good but does not guarantee the same absolute proximity to the global optimum as the state space explodes.

6. **Approximation Ratio Claim:**
   *Gap:* In Remark 2, the paper claims the empirical 94.8% performance is "well above any constant-factor approximation guarantee" of $\frac{2}{5}$ for the greedy heuristic.
   *Fix Needed:* Remove or heavily qualify this sentence. Comparing an empirical average-case result on a specific distribution to a theoretical worst-case bound ($\frac{2}{5}$) is mathematically invalid.

### Minor Concerns
1. **SIC Decodability Filter:** Relying purely on the deadline metric to "implicitly disfavor" pairs with similar channel gains (Section III.B) works empirically, but adding an explicit minimum SINR constraint would be more theoretically robust.
2. **Equation 7 (Objective Function):** The denominator $|\mathcal{A}|$ includes all active users. It would be helpful to clarify immediately below the equation that this includes both MEC-served and local-only users.
3. **DRL Baseline:** The MAPPO baseline uses a continuous action space. Briefly explain why a discrete formulation (e.g., Pointer Networks) was not used, given continuous mappings to combinatorial spaces are inherently limited.
4. **Result Return Delay:** Acknowledge that using uplink rate $R_i$ as a proxy for downlink delay might drift if downlink transmit power differs significantly from IoT device power, even under TDD reciprocity.

### Recommendation
**Major Revision.** The paper addresses a highly relevant problem with a practical algorithm that demonstrates strong empirical results. However, the theoretical framing—specifically the "optimal" offloading claims, the NP-hardness proof completeness, and the invalid comparison of average-case empirical results to worst-case theoretical bounds—needs rigorous tightening before it meets the standards of an IEEE Transactions journal.
