# Final H20 Task Validation

Production HEAD remains `592fc320d57398571dcb4ab686d5fb55fe9bac86` in the
existing dirty worktree. All 305 implementation files still match the frozen
CogACT source provenance, SHA256
`7f22cb27d56f0f25a67077f4b90615e6ac59cea5542d795a6adb6eab04ccdc82`.
Report/diagnostic helpers are separate experiment artifacts. No completed GPU
campaign was rerun for these CPU checks and the user PDF is unchanged.

## Executed Checks

| Check | Actual Result | Scope |
|---|---|---|
| Final focused regression | 160 passed, zero skipped or failed; 4.62 s | Generated C++ shared-library ABI and ownership, complete mixed typed outputs, three recorded-input adapters, timing/fidelity and numerical-context contracts, actual-pilot corruption checks |
| Full CPU regression | 2259 passed, 74 skipped, zero failed; 78.17 s | Entire default `vlaforge` pytest suite with CUDA visibility disabled |
| Collection integrity regression | 4 passed; 0.12 s | Missing model, wrong model identity, wrong actual GPU platform and missing formal evidence all reject without publishing a table |
| Actual four-model collection qualification | 40960 measured calls, 92160 complete output tensors; CDF inspected | Recompute tables and distributions from four previously audited complete campaigns; this is tool qualification, not final five-model acceptance |
| Actual incomplete five-model collection | Rejected before output creation | CogACT final evidence was absent at this qualification point; no partial five-model table was emitted |
| Existing published evidence refresh | 2529 file identities verified unchanged | Both five-model tensor tables, six operator families and both Qwen official multimodal baselines; no GPU execution |
| Final five-model collection | 51200 measured calls, 149760 complete output tensors; CDF inspected | All 50 original-input workers, independent remote/local audits and complete typed quality metrics |
| Final published evidence refresh | 2935 file identities verified unchanged | Includes all five original-input campaigns as well as the earlier tensor/operator/Qwen formal results; no GPU rerun |
| Qwen native closeout regression | 2275 passed, 62 skipped, zero failed; 79.17 s | Full default suite with CUDA hidden and the unavailable external `third_party/openpi` fixture file excluded; four Qwen native audits, CDFs and 80 lifecycle assertions completed separately on H20 |
| CogACT same-boundary closeout regression | 2276 passed, 62 skipped, zero failed | Full default suite after adding the boundary-comparison audit; 10 formal workers, complete output/RNG audits and identical-boundary comparison completed separately on H20 |

The focused tests compile and load a real generated CPU C++ Session library.
They check the actual C ABI layout, preserve an I64 value above `2**53`, retain
owned F64 output storage across repeated calls, and reject incorrect input ABI,
schema/library identity and foreign-thread use. The output/timing/replay audit
tests use copied real H20 pilot evidence and rebind the altered report hashes
before checking that semantic corruption is still rejected.

The 74 full-suite skips remain skips. They cover opt-in CUDA/LibTorch SDK runs,
unavailable optional local dependencies or checkpoints and unrelated real-model
integrations. Actual H20 model execution is established by the separate formal
campaigns and their audits, not by these skipped tests. The full-suite warnings
include deliberate malformed-archive and framework deprecation cases; no test
failed. The 160 focused cases overlap the full suite, so the counts must not be
added as distinct production tests.

## Artifacts

All paths in this table are relative to the absolute directory
`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/`.

| Artifact | SHA256 |
|---|---|
| `final-host-focused-20260910-001.xml` | `c82f51e8a043cde6d7cb559a8505f86d499b305dcd5b50b6b54ba313ef156c2d` |
| `final-full-cpu-20260910-001.xml` | `f8fcc8798aed8b314302ad5b199b8923bb145a8e5f283b82258b889c5278276a` |
| `final-collection-integrity-20260910-001.xml` | `eb1bb6818d5a1a7e279648547848d580604c03966ee6527eca16bffdfc17c0ae` |
| `test_host_collection_integrity.py` | `a510869195d20d9e02e9c6037d39c003971b616495770d0064a8345548f836f2` |
| `collect_host_pipeline_formal.py` | `e6fef511b7b9dd6328d9759051a4f4b63aff55b2bafe9b98bcbab3ea495da64a` |
| `raw-four-collection-qualification-001/report.json` | `24a9acd4cb3e7c00212b9eb421f83447495e2dbb16bb97f9598b13b40dff65b8` |
| `published-evidence-integrity-001.json` | `481c215bf9f85552e3dca23aca75521cae1d7a336e226e78f75609755633cb8c` |
| `published-evidence-integrity-002.json` | `900f2aad51a81810c3e2344e7485ca6c979d2f0f13e68e33560067f54b288cf0` |
| `raw-five-summary-001/report.json` | `077362fc3ec81e74c8440374c677f525dd1ae6271b5d196e753849b7d2b9856c` |
| `qwen-native-lifecycle-probe-20260910-001/lifecycle-audit.json` | `6297810f95794daba4306435ccd66682e6bfa6a78b581677b3e914e76c73afd5` |
| `qwen-native-lifecycle-closeout-20260910-002.xml` | `dfb0c5f4f235f70c791c1cc4a1666289f2c95986a1f7e3680a6b9b6cdf17e66f` |
| `cogact-timing-boundary-001/full-cpu-regression.xml` | `ffa79a298882026fb32ceefa9532321e5370b17a8fac76d59909412fe19a2e33` |
| `cogact-timing-boundary-001/timing-comparison-audit.json` | `b5516f6c4697c36bb75a35ba0b3b57a0f4f7aa144ce332e11fb5d2dd2ac91f8e` |

The final CPU commands ran from
`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/vlaforge/` using
`/home/zhangzimo/.venvs/edgefm-vla-report-py313-20260906/bin/python`,
`CUDA_VISIBLE_DEVICES=` and `PYTHONPATH=python:tools`.
The full invocation was `python -m pytest -q` with the named JUnit destination;
the focused invocation selected the native Session, recorded host adapters,
host-timing, multi-output and numerical-context test files plus
`test_host_pipeline_evidence.py`. The separate collection test ran from the
current worktree root. JUnit files retain the actual timestamps and host identity.

The final five-model collection now includes complete CogACT formal data,
remote/local full-output audits and inspected CDFs. Its create-only input
index is `raw-five-collection-index-001.json`, SHA256
`cf95dfbd42a6ac0df9b02487e4caf573a62fe5eeed902ec99729598da77a8ebb`.
The earlier missing-model rejection and four-model qualification are historical
test evidence, not the final acceptance data. Paper-level claim limits are
reviewed separately in `h20_paper_scope_review_20260910.md`; the final delivery
entrypoint is `h20_vla_experiment_delivery_20260910.md`.
