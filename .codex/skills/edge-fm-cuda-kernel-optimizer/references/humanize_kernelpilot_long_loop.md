# Edge-FM Humanize + KernelPilot Long Loop

Use this reference when a kernel optimization needs more than a short NCU loop:
multiple candidate versions, independent review, source provenance, profile
digests, and explicit failure memory.

## When To Escalate

Use the long loop for high-value Edge-FM paths such as QKV/OProj, W4A16/W8A8
linear, prefill attention/KV, fused gate-up, `lm_head_top1`,
DeepGEMM/TensorRT bridge experiments, and other work where a single edit is
unlikely to settle the bottleneck.

Do not start the long loop when the hotspot is unknown. First use nsys,
benchmark reports, or operator/layer tests from `edge_fm_workflow.md` to lock
down:

- target kernel or operator role
- target GPU and shape matrix
- correctness reference
- baseline latency and acceptance gate
- benchmark command and profile command

## Vendored Resources

The runtime and knowledge pack are vendored under this skill:

- Humanize runtime:
  `vendor/humanize/{scripts,hooks,prompt-template,templates,config,agents}`
- KernelPilot knowledge:
  `vendor/kernel-pilot/{knowledge,references,scripts}`
- Internal references:
  `vendor/humanize/skills/humanize-kernel-agent-loop/SKILL.md`,
  `vendor/kernel-pilot/skills/kernel-knowledge/SKILL.md`,
  `vendor/kernel-pilot/skills/profile-evidence/SKILL.md`

Install the Codex Stop hook only when the user wants to run RLCR:

```bash
bash .codex/skills/edge-fm-cuda-kernel-optimizer/scripts/install_humanize_hooks.sh --dry-run
bash .codex/skills/edge-fm-cuda-kernel-optimizer/scripts/install_humanize_hooks.sh
```

The installer only points `$CODEX_HOME/hooks.json` at the vendored Humanize
runtime. It does not start a loop.

## Standalone Repo Contract

Prefer a clean standalone repo under:

```text
deliverables/kernel_opt/<task_name>/
```

Create the standalone repo before RLCR starts and keep the source framework
checkout read-only unless the user explicitly asks for an in-place patch.

Required layout:

```text
.gitignore
.humanize/kernel-agent/refined-plan.md
README.md
src/<task_name>/
tests/
benchmarks/
artifacts/attempt-ledger.md
artifacts/optimization-ledger.md
artifacts/source-idea-ledger.md
artifacts/lineage.jsonl
artifacts/profile-digests/README.md
```

`.humanize/` remains local loop state and must not be committed.

## Planning Rules

The refined plan must include acceptance criteria for:

- baseline correctness and benchmark parity
- representative shape/dtype coverage
- one baseline Nsight Compute digest after baseline benchmark succeeds
- benchmark output with per-shape latency, geomean, best/worst cases, and environment metadata
- attempt ledger for every candidate, including correctness failures and regressions
- optimization ledger only for correct candidates with measured improvement
- lineage entries with parent version, motivation, source keys, result, and selected/rejected status
- source idea ledger with repo/path/symbol or PR, hypothesis, result, and a do-not-reread key
- explicit rollback/fallback rules before any change is migrated back into `src/`

If baseline code seeds a candidate, record source path, commit, license/notice,
copied files, and first delta before mutating it. If the user asked for
from-scratch work, use baseline code only for correctness, benchmark, and
profile comparison.

## Knowledge Pass

Keep the first pass scoped:

1. Read `vendor/kernel-pilot/knowledge/README.md`.
2. Read `vendor/kernel-pilot/knowledge/routing/topics/index.md`.
3. Read `vendor/kernel-pilot/knowledge/references/index.md`.
4. Pick one relevant topic page, usually `matmul-gemm.md`, `attention.md`,
   `kv-cache.md`, `activation-fusion.md`, `normalization.md`, or
   `quantization-fp8.md`.
5. Pick the closest framework/source page: `cutlass`, `deepgemm`, `triton`,
   `tensorrt-llm`, `flashinfer`, `sglang`, `vllm`, or source-only guides.
6. For PR-driven repos, pair the PR guide with the source guide. For source-only
   repos, inspect source guides and current source paths directly.

After two consecutive weak rounds with less than 1% geomean improvement over
the prior best, expand research before editing again: read at least 50 new
code-first sources before prose sources and log each one with a do-not-reread
key in `artifacts/source-idea-ledger.md`.

## Profile Evidence

Write a Profile Evidence Digest when:

- baseline benchmark passed and no baseline digest exists
- a correct candidate is within +/-2% of baseline
- a correct candidate regresses on any configured case
- two consecutive iterations improve less than 1%
- a candidate is much faster than expected
- a reviewer asks for profile evidence

Each digest must include the `.ncu-rep`, CSV export, bottleneck class,
supporting metrics, ranked hypotheses, and one concrete next edit. Do not run
full NCU for every tiny edit; use it for baseline, regressions, plateaus,
surprising wins, and profile-driven changes.

## Starting RLCR

Start only from a clean standalone git repo with a refined plan:

```bash
EDGE_FM_REPO_ROOT="${EDGE_FM_REPO_ROOT:-/path/to/edge-fm-x}"
"$EDGE_FM_REPO_ROOT/.codex/skills/edge-fm-cuda-kernel-optimizer/vendor/humanize/scripts/setup-rlcr-loop.sh" \
  .humanize/kernel-agent/refined-plan.md --yolo
```

After setup:

1. Read `.humanize/rlcr/<timestamp>/round-0-prompt.md`.
2. Execute the current round.
3. Commit candidate changes and artifacts inside the standalone repo.
4. Write the required round summary.
5. Stop normally so the Humanize Codex Stop hook can review.

If the Stop hook blocks, follow the generated next-round prompt exactly.

## Migration Back To Edge-FM

Do not migrate exploratory code into `src/` until the standalone candidate is:

- correct against the reference
- faster by the agreed gate
- explained by benchmark and profile evidence
- recorded in ledgers and lineage

After migration, rebuild the relevant build directory, install bindings if
needed, run the closest operator/layer pytest first, then run the target
engine/benchmark case. If the in-repo path loses the standalone speedup, inspect
shape, launch config, compile options, operator routing, CUDA graph behavior,
and tensor layout before claiming a regression or win.
