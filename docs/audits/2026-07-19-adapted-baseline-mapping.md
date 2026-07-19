# Guo-JAGP adapted-baseline mapping audit

## Identity and claim boundary

- Source: F. Guo, H. Lu, D. Zhu, and Z. Gu, “Joint User Association,
  Grouping and Power Allocation in Uplink NOMA Systems with QoS
  Constraints,” *IEEE ICC*, 2019,
  [DOI 10.1109/ICC.2019.8761779](https://doi.org/10.1109/ICC.2019.8761779).
- Implementation identifier: `guo_jagp_adapted`.
- Audit basis: the primary paper, including Eq. 5, Eqs. 9--10, Eq. 11,
  and Algorithm 1. The paper definitions were checked on 2026-07-20.
- Claim status: this is a closest-work adaptation, not a reproduction.
  The source paper's results, Nash-equilibrium claim, Pareto claim, and
  performance values are not attributed to this implementation.

The retained potential is

\[
P_{\mathrm{Guo}}(s)=\sum_g\sum_{n\in g}p_n(s),
\]

where each `p_n` is the QoS-required power from the source recursion. It is
not the thesis-v2 completion, allocated-CPU, deadline-slack, victim-link
exposure, or closure objective. An infeasible user power makes the group and
schedule potential infinite.

## Component-by-component mapping

| Component | Original-paper semantics | Available source/equation | Adaptation in `guo_jagp_adapted` | Shared evaluator boundary | Residual comparability limitation |
|---|---|---|---|---|---|
| decision variables | Every user chooses a combined BS/channel group; QoS-minimal transmit powers follow from the chosen association and grouping. | Problem (3), simplified problem (6), and Algorithm 1. | The raw decision is a task-to-`(serving_uav, subcarrier)` placement. Required powers are derived only to rank placements and enforce the ceiling; they are not emitted as canonical transmit powers. | The returned raw schedule alone crosses into policy closure and frozen realized replay. | UAV service, task admission, and the later fixed-power evaluation are not source decision variables. |
| objective/potential | The system objective is minimum total user transmit power. A player's `C_n = p_n + E_n` equals its own power plus the within-group power externality. Unilateral `C_n` differences equal total-power differences. | Eqs. 9--10 define `C_n` and its group-power contribution; Eq. 11 gives the exact-potential difference. | Candidate responses minimize total QoS-required power for the fixed selected tasks; moves require a strict decrease. | Thesis completion, CPU, deadline slack, cross-UAV exposure, and closure never enter this potential; they are evaluated only afterward. | Source optimality, NE, and Pareto statements do not transfer to the capacity-limited task subset or to thesis-v2 performance. |
| association | A group index jointly identifies the serving BS and its orthogonal channel. | System model and the definition of `pi_n` preceding Eq. 7. | A serving UAV substitutes for a BS, and a subcarrier identifies the group within that UAV. All UAV/subcarrier placements are enumerated. | Canonical evaluation retains the selected serving UAV unless closure rejects the task. | Aerial receivers, MEC service, and full-reuse UAV coupling differ from the terrestrial multi-BS model. |
| grouping | Users sharing a channel form one NOMA group; the formulation does not impose this implementation's pair-only capacity. | System model, constraints (3d)--(3e), and Algorithm 1. | Each `(uav, subcarrier)` block has a two-user block cap. At most `2*K*S` feasible offload-required tasks are frozen before the game so dropping a user cannot lower the potential. | Closure may subsequently remove tasks but does not rerun the adapted game. | Capacity selection, admission, deterministic tie breaks, and pair-only groups are implementation choices. |
| rate/SIC | The source uses a Shannon spectral-efficiency constraint and perfect within-group SIC. It detects from best to worst channel and computes powers from worst to best. | Eqs. 1--2 and Theorem 1; Eq. 5 is the recursion. | Entries are ordered by `(-selected_gain, task_id)`. `guo_required_powers` applies Eq. 5 with the frozen scalar target `0.5` bit/s/Hz. | Canonical NOMA later applies the configured residual SIC, hard SIC threshold, bandwidth, and full receiver-side checks. | The source permits heterogeneous targets and its SIC/power semantics are not numerically identical to the canonical evaluator. |
| power | Continuous user powers are minimized subject to each user's QoS target and `p_n <= p_max`. | Eq. 5 and constraints (3b)--(3c)/(6b)--(6c). | QoS-required powers define feasibility and potential; any required power above `cfg.transmit_power_w` yields infinite potential. | Canonical replay uses its configured uniform transmit power rather than the adaptation's derived per-user powers. | The adapted schedule is comparable as an association/grouping policy, not as a reproduction of source power allocation or source energy results. |
| channel | The source assumes known CSI; neighboring BSs use orthogonal spectrum or enhanced inter-cell interference coordination. Its simulation uses terrestrial large-scale loss plus Rayleigh fading. | Section II system model and Section V simulation setup. | The selected planning-basis UAV air-to-ground power gain replaces `|h_n|^2`; receiver noise uses the canonical subcarrier bandwidth. | Cross-UAV co-channel exposure is intentionally absent from the Guo potential and enters only through the shared canonical evaluator. | Full-reuse aerial interference and estimated-versus-realized channels are outside the source model. |
| CPU | No MEC computation or CPU allocator appears in the source problem. | Not present in the source system model or JAGP formulation. | No CPU term or CPU decision is introduced into the adapted potential. | The configured canonical CPU policy is applied during closure after the raw schedule is returned. | CPU-driven rejection and completion cannot be interpreted as behavior of the Guo algorithm. |
| task/deadline | The source serves communication users with spectral-efficiency QoS; it has no computation task, arrival process, or hard MEC deadline. | QoS constraint in Eqs. 2--3; no task/deadline model. | Only tasks that require offloading and have feasible solo required power can enter the frozen selected set. Task size and deadline do not rank Guo moves. | Canonical task latency, CPU admission, deadlines, and completion are evaluated after the adapted search. | Traffic admission and deadline performance are thesis-v2 semantics, not source-paper semantics. |
| mobility | The source simulation randomly places users around fixed terrestrial BSs; mobility and UAV trajectory are not game decisions. | Section V simulation setup. | The game runs independently on one immutable planning snapshot and never changes positions. | Trace generation supplies exogenous per-slot geometry; realized replay does not trigger regrouping. | No claim is made that source convergence covers time-varying UAV or user mobility. |
| stopping rule | Algorithm 1 updates users sequentially and stops when the association/grouping vector no longer changes; the paper invokes finite improvement of its exact-potential game. | Eq. 11, Corollary 1, and Algorithm 1 steps 10--20. | Users are visited in frozen `(minimum feasible solo power, task_id)` order. A move is accepted only for strict total-potential decrease; equal-potential deterministic ties do not move. A visited-state guard audits strict descent. | Closure and realized replay occur only after termination and cannot feed back into the game. | Termination certifies only no improving unilateral placement for the frozen adapted state space, not source-code equivalence, source NE, or a thesis-v2 local optimum. |

## Executable audit anchors

- `tests/test_baselines_adapted.py::test_guo_power_recursion_matches_hand_calculation`
  freezes the strongest-to-weakest result at `(0.5, 1.0)` for gains `(4, 1)`,
  unit noise, and unit spectral-efficiency targets.
- `tests/test_baselines_adapted.py::test_guo_potential_is_required_power_only_and_enforces_ceiling`
  proves that CPU, residual SIC, hard canonical SIC threshold, and cross-UAV
  scale do not enter `P_Guo`, while the power ceiling does.
- `tests/test_baselines_adapted.py::test_guo_adaptation_is_deterministic_and_returns_a_raw_schedule`
  checks deterministic capacity-safe raw grouping.
- `tests/test_thesis_suites.py::test_guo_adaptation_uses_the_common_closure_and_replay_path`
  checks the common evaluator boundary without relabeling its result as a
  Guo-paper outcome.

## Unresolved reproduction evidence

No author-code release is included in this repository, and no author-code
behavior was available from the publisher record or author-uploaded paper
inspected for this audit. Initialization details that are not completely
determined by the pseudocode, software tie behavior, and numerical tolerances
therefore remain unverifiable. They are implemented as declared deterministic
adaptation choices above, never as inferred original behavior.
