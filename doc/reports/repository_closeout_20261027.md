# Repository Closeout

Date: 2026-10-27, Asia/Shanghai.

## Scope

- Base: the existing remote `codex/vlaforge-paper-artifact` at `aa5558f`.
- Runtime change: `a61917d`, native TensorRT contexts/plugins, declared host
  input preparation, explicit region-chain/state interfaces, generic ONNX
  lowerings and focused regression coverage.
- Delivery: the read-only task board, Orin reports, three-platform summary,
  small audit receipts and the self-contained paper package.
- Packaging correction: include the CUDA plugin source and the pinned native
  tokenizer Rust crate in wheels. A wheel-content regression failed before
  the correction and passed afterwards.

The older main working directory was behind the remote by three commits.
385 pending files matched the retained historical snapshot `191692d`; the
remaining difference was an obsolete codegen golden digest. These legacy
files must not replace the reorganized adapters already committed upstream.
Preserve them in a named local Git stash before fast-forwarding the main
checkout. Do not force-push or rewrite the remote branch.

## Verification

| Check | Result |
|---|---|
| Python suite, excluding opt-in CUDA AOTI and real-model markers | 3060 passed, 57 skipped, 17 deselected; 6 subtests passed |
| Wheel regression suite after packaging correction | 4 passed |
| Fresh Release CPU configure/build and CTest | 14 passed |
| Task-board Node tests | 16 passed, including no local-artifacts dependency |
| Browser workflows and desktop/tablet/390px/320px views | 11 passed |
| Standalone paper verifier | 272 files, 282 relative links passed |
| Source and delivery secret-pattern scan | No credential/key candidates found |

The skipped Python checks require optional packages, SDK compilation or
opt-in model assets. Local Cargo is unavailable, so the optional Rust bridge
was not rebuilt in this closeout. Orin TensorRT/CUDA plugins, Horizon runtime
and real-checkpoint campaigns were not rerun; historical platform evidence
and acceptance boundaries remain authoritative. CPU/unit results do not
substitute for board measurements.

## Retention And Portability

The paper folder is at
[`paper_handoff/edgefm_three_platform_20261026`](../../paper_handoff/edgefm_three_platform_20261026/README.md).
Its manifest, bytes and original experimental numbers are unchanged. ZIP
duplicates are ignored; the unpacked folder is versioned. `.gitattributes`
disables newline conversion for manifest-covered files and audit receipts.

[`doc/evidence`](../evidence/README.md) contains 45 small historical receipts,
about 0.6 MiB. Its index also points to 13 byte-exact figures already in the
paper folder. Figure matching requires exact path identity and original and
packaged hashes. No missing image is replaced by another experiment.

The board now uses declared published-figure metadata instead of hardcoded
deleted SmolVLA paths. It retains explicit HTTP410 notices for retired
originals; unknown/unexpected missing evidence still fails checks. The
project badge follows the actual project status. Fresh browser tests can
start a local server without an old machine-specific launch receipt.

Models, engines/HBMs, compiled caches, raw vehicle cases, unrelated BEV Git
bundles and temporary validation output remain outside the committed tree.
The broader cleanup and the loss of direct replay capability are documented
in [the cleanup report](local_bulk_cleanup_20261026.md). Core model/run counts,
MSE/cosine gates, negative results and the J6M stage-only boundary are unchanged.
