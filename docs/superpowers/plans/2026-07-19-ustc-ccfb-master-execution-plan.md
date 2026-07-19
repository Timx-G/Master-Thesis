# USTC–CCF-B Dual-Track Master Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the approved USTC professional-master/CCF-B design into a safely isolated, tested, reproducible research implementation, a confirmatory evidence bundle, and two submission-quality manuscripts.

**Architecture:** Work proceeds through five independently reviewable plans. The original dirty checkout is used only to freeze the incomplete exploratory pilot and create an explicit research checkpoint; all implementation then moves to a sibling worktree. Model/algorithm semantics and evidence infrastructure must pass their readiness gates before confirmatory seeds are opened, and result-dependent manuscript text must be generated only from canonical-v2.

**Tech Stack:** Python 3.11, NumPy, SciPy, pytest, TOML, CSV/JSONL, Git worktrees, SHA-256 manifests, Tectonic/XeLaTeX/latexmk, USTC `ustcthesis`, Elsevier `elsarticle`.

---

## Approved source of truth

The normative specification is:

```text
docs/superpowers/specs/2026-07-19-ustc-ccfb-dual-track-completion-design.md
commit adda1539d480044df728366a878dcdc0e069e5c4
```

When an older plan, `CLAUDE.md`, generated result macro, canonical-v1 file, or pilot artifact conflicts with the approved specification, the approved specification wins. No performance claim may be preserved merely because it appears in an older manuscript.

## Plan suite and dependency order

| Order | Plan | Exit artifact |
|---:|---|---|
| 1 | `2026-07-19-01-safety-and-model-semantics-implementation.md` | Isolated worktree, clean baseline, normative raw/closed schedule and delay semantics |
| 2 | `2026-07-19-02-algorithms-oracles-and-baselines-implementation.md` | Auditable IEA/DD/Exact implementation and closest-work baseline |
| 3 | `2026-07-19-03-evidence-infrastructure-implementation.md` | Registry, locks, statistics, shards, gates and canonical-v2 publisher |
| 4 | `2026-07-19-04-confirmatory-execution-implementation.md` | Frozen pilot decision, all formal evidence families and canonical-v2 |
| 5 | `2026-07-19-05-thesis-and-journal-publication-implementation.md` | USTC thesis, Computer Networks manuscript and verified submission package |

The frozen interleaving is `Plan 1 -> Plan 2 -> Plan 3 Tasks 1--9 -> Plan 5 Tasks 1--8 -> Plan 3 Tasks 10--12 -> Plan 4 -> Plan 5 Tasks 9--13`. Plan 5's first six tasks are result-independent, while Tasks 7--8 land the pure bilingual generator and claim binding required by Plan 3's publisher. Plan 5's result-bearing prose, abstracts, discussion, conclusion, final builds, package, and release audit remain blocked until Plan 4 publishes canonical-v2.

## Global repository boundaries

The original checkout is:

```text
/home/link/my_workspace/Master-Thesis
```

The approved implementation worktree is:

```text
/home/link/my_workspace/.worktrees/master-thesis-ccfb
```

The implementation branch is:

```text
feature/ustc-ccfb-v2
```

The original checkout remains the owner of excluded user materials and the incomplete exploratory pilot. Never run `git add .`, `git add -A`, `git stash`, `git clean`, `git reset`, or switch the original checkout's branch. Every commit uses explicit path arguments and is inspected with `git diff --cached --name-status` before committing.

## Global test and commit protocol

Use this interpreter for all Python checks:

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python
```

Every behavior change follows RED–GREEN–REFACTOR:

1. add one focused test;
2. run that exact test and observe the expected semantic failure;
3. implement the minimum behavior;
4. rerun the focused test and its local regression group;
5. inspect the diff and commit only the named files.

Configuration-only, prose-only and generated-file steps do not invent artificial unit tests; they use schema, placeholder, checksum or document-build gates specified in the corresponding plan.

No task may report success from an earlier test run. Completion claims require a fresh command, exit code zero and the observed pass/build count.

### Task 1: Execute the single authoritative safety checkpoint

**Files:**
- Follow: `2026-07-19-01-safety-and-model-semantics-implementation.md`, Task 1, Steps 1--6
- Create there: `docs/audits/2026-07-19-partial-pilot-provenance.md`
- Create there: `docs/audits/2026-07-19-research-checkpoint-allowlist.txt`
- Modify there: `.gitignore`

- [ ] **Step 1: Follow Plan 1's safety task verbatim**

Plan 1 Task 1 is the only normative definition of the partial-pilot audit, nine exact generated-output ignore rules, allowlist validator, safety commit, checkpoint test, checkpoint commit, and prohibited paths. Do not copy an older three-rule ignore list or create a second checkpoint record from this master plan.

- [ ] **Step 2: Verify the safety outputs without staging anything else**

Run from `/home/link/my_workspace/Master-Thesis` after Plan 1 Task 1 Step 6:

```bash
test -f docs/audits/2026-07-19-partial-pilot-provenance.md
test -f docs/audits/2026-07-19-research-checkpoint-allowlist.txt
rg -n '^classification: exploratory_incomplete$' \
  docs/audits/2026-07-19-partial-pilot-provenance.md
git check-ignore artifacts/preregistration/v2/seed-access.jsonl
git check-ignore artifacts/runs/v2-confirmatory/formal_run_state.json
git check-ignore artifacts/canonical/v2/MANIFEST.json
git diff --cached --exit-code
```

Expected: both audit files exist, the incomplete-pilot classification is exact, all three representative v2 paths are ignored, and the index is empty. The old partial pilot remains untouched, unresumed, unpooled, and unusable for power or formal evidence.

### Task 2: Enter and verify the Plan 1 isolated worktree

**Files:**
- Follow: `2026-07-19-01-safety-and-model-semantics-implementation.md`, Task 1, Step 7
- Verify tag: `thesis-v2-research-checkpoint-20260719`
- Verify branch: `feature/ustc-ccfb-v2`
- Verify worktree: `/home/link/my_workspace/.worktrees/master-thesis-ccfb`

- [ ] **Step 1: Execute Plan 1's tag/worktree/baseline step exactly once**

Plan 1 owns the annotated checkpoint tag, exact SHA comparison, worktree creation, full pytest baseline, canonical-v1 checksum verifier, direct non-regenerating Tectonic builds, and the two minimal smoke diagnostics. If the tag, branch, or worktree already exists, inspect whether it resolves to the recorded checkpoint SHA; reuse it only when all three identities agree, otherwise stop and report the collision.

- [ ] **Step 2: Verify the handoff boundary**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
test "$(git branch --show-current)" = "feature/ustc-ccfb-v2"
checkpoint_sha=$(git rev-parse thesis-v2-research-checkpoint-20260719^{commit})
test "$(git rev-parse HEAD)" = "$checkpoint_sha"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

Expected: branch, annotated tag, HEAD, and clean worktree all identify the same selective checkpoint. All excluded user materials remain in the original checkout and outside the implementation branch.

### Task 3: Execute the implementation plans with review gates

**Files:**
- Use: the five plan files listed above

- [ ] **Step 1: Execute Plan 1 task-by-task**

Use `subagent-driven-development` with a fresh implementer per task. Require RED evidence, GREEN evidence, a specification review and a code-quality review before each task is marked complete.

- [ ] **Step 2: Execute Plan 2 only after model semantics are green**

The Plan 1 full regression and semantics contract must pass. Do not preserve an algorithm API that conflicts with raw/closed schedule separation merely for backward compatibility.

- [ ] **Step 3: Execute Plan 3 Tasks 1--9**

Freeze the registry, seed guard, inference, Resource snapshots, Exact evidence, generic shards, runtime/trace/provenance locks, formal dispatcher, analyzer and claim gates. Do not implement the canonical publisher yet because its default bilingual generator is owned by Plan 5 Task 7.

- [ ] **Step 4: Execute Plan 5 Tasks 1--8 only**

Freeze the official templates, establish metadata and bibliography gates, migrate the two drivers, draft Introduction, Related Work, System Model, Algorithms, Proofs and Experimental Methodology, and land the deterministic bilingual asset generator required by the canonical publisher. Do not write final numerical claims.

- [ ] **Step 5: Finish Plan 3 Tasks 10--12 and publish a readiness report**

Implement and test the canonical publisher with the Plan 5 generator, add guarded Make targets, and run the evidence implementation gate. The readiness report must prove that formal seed candidates remain unseen, the final runner consumes `confirmatory_lock.json`, and canonical-v2/full will be self-contained.

- [ ] **Step 6: Execute Plan 4 only after the readiness report passes**

Opening any formal seed before the lock is a hard protocol failure. A source/config/analysis hash change after power locking invalidates the lock and returns execution to the final pilot task.

- [ ] **Step 7: Finish Plan 5 from canonical-v2**

Abstracts, results, discussion and conclusions must use generated macros and claim gates. Negative H1–H4 outcomes remain in both evidence and prose.

### Task 4: Final integrated verification

**Files:**
- Verify: repository, canonical-v2, thesis, journal package

- [ ] **Step 1: Run the complete software verification**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest -q
```

Expected: zero failures and no warnings promoted by strict pytest configuration.

- [ ] **Step 2: Verify canonical closure**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.canonical_v2 verify \
  --root artifacts/canonical/v2
(
  cd artifacts/canonical/v2
  sha256sum --check SHA256SUMS
)
```

Expected: every file is checksummed, every light artifact resolves to full, every claim evidence path exists, and all integrity/correctness gates pass.

- [ ] **Step 3: Build both documents from a clean environment**

```bash
make publication-final \
  PUBLICATION_METADATA=publication/metadata.local.toml \
  PUBLICATION_GATES=publication/human-gates.local.toml
make publication-release-audit \
  RELEASE_MODE=research_delivery \
  PUBLICATION_METADATA=publication/metadata.local.toml \
  PUBLICATION_GATES=publication/human-gates.local.toml
```

Expected: both final PDFs and deterministic build reports exist; placeholder, citation, figure-language, result-source, format and research-delivery release checks pass. A journal submission package/audit is an additional authorized mode, not a prerequisite for the research-delivery gate.

- [ ] **Step 4: Separate research delivery from human/admin completion**

Report the verified research deliverables, then list signatures, advisor review, plagiarism checks, school submission, external archival authorization and journal upload as explicitly pending human/administrative actions. Never claim a degree or journal submission has occurred when it has not.
