# Original-Input Host Pipeline

The shared host pipeline now measures original decoded RGB, native state, text
and saved action noise in CPU RAM through complete normalized/native CPU action
outputs. It includes official resizing, normalization, tokenization, H2D, vision,
the full action computation, native scaling and output D2H. File/video decoding,
initialization, robot transport, quality checking and logging are outside the
timed interval. Two adjacent segments are recorded: preprocessing including H2D,
and inference including GPU observation processing, postprocessing and D2H.
They sum exactly to each full call. Separate component attribution inside each
segment is not claimed.

`validation/host_pipeline.py` has no model or backend dispatch. A pipeline supplies
`prepare(sample)` and `infer(prepared)`; all returned ports must already be
complete contiguous CPU Torch tensors with exactly the declared dtype and shape.
The shared typed Session output comparator validates full output bytes and
per-output metrics. `adapters/openpi/openpi_host_pipeline.py` supplies official PyTorch
and generated Session implementations for both pi0 and pi0.5. GPU observation
preprocessing occurs once per timed path. Qualification separately checks the
original-input processor against every frozen model-tensor input.

`validation/native_session.py` uses the generated Session's existing C ABI,
checks library/bundle/schema identity, validates exact input metadata and keeps
borrowed storage alive. Every output owns CPU storage. Its replay diagnostics
read the existing compiled loop counters; no new C++ runtime API or model branch
was added. The Python host is explicit and does not receive a no-Python
deployment certificate. The older standalone Session results remain separate.

## Audited Formal Results

pi0 and pi0.5 each completed five independent official PyTorch workers and five
generated Session workers, with 128 warmups and 1024 measured calls per worker.
Both controllers are terminal with success. Remote independent audits and local
rechecks of every complete output passed. Each model has 10240 measured latencies
and 23040 complete output tensors including warmups. The two formal CDF/tail PNGs
were inspected; raw latency CSVs, exact CDF CSVs and PDFs are retained. No outlier
was removed and no pilot sample enters these results.

| Model | Configuration | Mean ms | p99 ms | Fresh Chunks/s | Preprocess + H2D Mean ms | Inference + Postprocess + D2H Mean ms |
|---|---|---:|---:|---:|---:|---:|
| pi0 | Official PyTorch | 221.305113 | 350.640813 | 4.518648 | 23.844723 | 197.460390 |
| pi0 | Session required | 108.152217 | 113.976963 | 9.246227 | 23.093390 | 85.058827 |
| pi0.5 | Official PyTorch | 256.840307 | 274.030360 | 3.893470 | 23.621839 | 233.218468 |
| pi0.5 | Session required | 116.853524 | 122.893219 | 8.557722 | 23.231911 | 93.621612 |

Normalized F32 and native F64 action chunks match both frozen references byte
for byte in every call. All MSE and maximum absolute errors are zero; the minimum
computed cosine is 0.9999999999999998, within floating-point rounding of one.
Every native worker starts with zero counters and records 1152 replays of the
captured ten-step loop, with zero ordinary fallbacks. Each model cycles the same
16 recorded observations from one ALOHA episode; this is not robot success-rate
or broad task-quality evidence. GPU ownership was exclusive per monitored worker,
but shared CPU/NAS isolation was not established. The official distributions have
visible process groups and long tails; the raw samples and maxima remain intact.

The measured boundary is decoded RGB/state/text/saved-noise in CPU RAM through
complete CPU actions, as specified above. It includes actual preprocessing and
postprocessing. File/video decoding, initialization and robot transport remain
outside the interval. The two timing segments are measured on each call and sum
exactly; individual components inside a segment were not timed separately.
Both implementations use an explicit Python host. These results do not replace
the earlier standalone no-Python Session certificates or provide board evidence.

Local complete formal archives are under
`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/raw-host-formal/`.
The `pi0/` and `pi05/` directories contain `audit-formal.json`,
`local-formal-audit.json`, all worker/ownership records, raw complete output bytes
and `figures/`. The remote protocol/source/library identities below are unchanged.
Controller PIDs 85392 and 80976 are finished, not running jobs.

| Model | Remote Formal Audit SHA256 | Local Full-Data Audit SHA256 | Figure Report SHA256 |
|---|---|---|---|
| pi0 | `71d71a55c86afe877c1f10d16793c2478104666fa01b86107688c03de4ef24bc` | `8680b9292bab29a6e5e6cb8a8141a8bdd4a0e4e09932c8fc7c3618e5464b12be` | `7a4a28e38a1d98c12a5cc254717e88fa28ff4e664e97e53e11351407866a4245` |
| pi0.5 | `332a17c079e74123545eb5e9779e93f3980ee4179ed08989b8078348e7f49854` | `68c0327cff24dd8ed5bc1ff6a64ec3d8683bfd31de723341153e90ed5793b686` | `fc59163863cfbbb670d9c1f93f8fe363e7d359013498ae9014de0ea6fe0f46cd` |

## Verified Pilots

Both models completed one independent official worker and one independent native
worker, each with 16 warmups and 32 measured calls over all 16 recorded ALOHA
observations. Remote independent audit and local complete-data checks passed:
192 calls and 384 complete normalized/native output tensors in total, all byte
exact against both frozen references. Every normalized F32 and native F64 value
is retained. Native loops report 48 replays of 10 captured steps and zero
ordinary fallbacks. These are pilots, not formal latency results.

| Model | Pilot Controller | Official Worker | Native Worker | Formal Controller |
|---|---:|---:|---:|---:|
| pi0.5 | 78638 | 78662 | 79365 | 80976 |
| pi0 | 81706 | 81753 | 82693 | 85392 |

Remote host: `zzm-h20-x8-2`.
pi0.5 GPU1 UUID: `GPU-29fb9c6e-2a11-e303-5f26-ad079fa86fa5`.
pi0 GPU2 UUID: `GPU-3596876b-e679-6289-f784-c8d98294ed23`.
Both formal controllers were launched after successful independent pilot audits
and have now completed all ten workers, complete audits and CDFs described above.
Each formal configuration requires five independent processes, 128 warmups and
1024 measured calls per process. Configuration order rotates by repeat.

Remote root:
`/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/raw-input-20260910-001/`.
Model subdirectories `pi05/` and `pi0/` contain frozen protocols, controller PIDs,
worker environment manifests, complete binary outputs, per-call samples,
before/after replay counters, process maps, stdout/stderr and registered owner
monitor records. No job should be restarted on the basis of an old status file.

Local pilot root:
`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/raw-host-generic-pilots/`.
Each model directory contains `audit-pilot.json` and `local-pilot-audit.json`.

| Model | Protocol SHA256 | Remote Pilot Audit SHA256 | Local Audit SHA256 |
|---|---|---|---|
| pi0.5 | `90ea1acb991d59b526ff038c77686be65bf8521a66d10d3bcd782fd935eeead4` | `93452bf9085a908fec4a6bbb1ba7d3a046b2a312c50841bd844848f410261a47` | `1ee85b2be00f60d3fa1cf3e8865d0b0da3050242cf704d7a62cbda791345dab8` |
| pi0 | `f290597b9edbe436248b38932038a9433531f2164054e9e5f619eec241eda5bd` | `83d047f1709d87872edd7c2ce1bcf5c94e14c0efa12eb6019f79d5a83b749cee` | `45be91509dd913cec34273dbed4d96edc33ee8f9e836bcd3bed596aa73a27e85` |

The adapter reuses the exact captured checkpoints, processor assets and typed
references from the completed host-model-tensor campaigns. No reference was
silently replaced by the new pipeline's own outputs. The new 302-file snapshot
retains source commit `592fc320d57398571dcb4ab686d5fb55fe9bac86`, with an explicit
dirty patch and hashes for untracked implementation files. Source provenance
SHA256: `3d1c2a1522362250a30a2cae027bb6aa8a2616618d9e73cd10ed52d1135b5d29`.
It is isolated from the completed 298-file host-tensor snapshot.

The shared libraries preserve the verified generated C++ sources and native
payloads. They are separate PIC builds using the original CMake targets and
dependencies; the original bundles were not changed. pi0.5 library SHA256:
`cbd6158bc6900a18222913f14457b778fae6e24ab2374b18203ff04e2995625c`.
pi0 library SHA256:
`277bc73fe96c3c1ba6782cee9252319d12aeeb3382a3e5690e0ec132e0b97b74`.
An earlier CMake imported-target scope failure remains in the isolated
`raw-input-20260909-001/pi05-shared-001/` build and is not counted as passed.
Separate standalone helpers were built, but these new pilot executions use the
Python host; standalone helper execution is not claimed.

## Tests and Remaining Work

Focused Python/actual generated C++ ABI, output storage, timing and OpenPI
regressions: 71 passed, 3 optional OpenPI dependency tests skipped. Complete CPU
regression: 2219 passed, 74 skipped, zero failures. The actual C++ ABI size/offset
probe includes mixed F64/I64 outputs and replay metadata. Four additional checks
against copied real H20 pilot evidence pass; they reject a changed output even
after rebinding report hashes, inconsistent timing, and deleted replay counters.
No GPU work was rerun by those corruption checks.

Focused XML SHA256:
`6a943e8551bd00ffc9501d0113288df773876aeb8405a0aae1f328b2e61567c9`.
Full CPU XML SHA256:
`db7ba1e9032f98cb3a1303ac1bbab2484927ea9dd31fb3ae4fb63bf1c2234427`.
Both are under the local recovery artifact root above.

Historical next-step snapshot: the isolated source adds the SmolVLA raw-input adapter and reuses
the shared timing/native Session contracts. Its 69 focused tests and complete
CPU suite (2235 passed, 74 skipped, zero failures) pass. The 16 new raw records
were decoded on H20 from the pinned dataset and checked against the original
selection; uint8 pixel roundtrips are exact. They are not benchmark results yet.
Full CPU XML SHA256:
`5c216a87d3f2e5d5cd2f5ea7db8b04ce1df265fa257a98ae173a39c519e7a038`.
Focused XML SHA256:
`d19c424745acffb0b27e754042f8fae7288a9fd7ac46fdb31f233385eb603dff`.

Final update: original-input SmolVLA, RDT and CogACT formal pipelines are now
complete, with the same benchmark contract and independent remote/local audits.
The final five-model original-input comparison and CDF are linked from
`h20_vla_experiment_delivery_20260910.md`. The earlier Qwen native export
blocker was superseded by the fixed-profile native result in
`qwen35_native_formal_20260910.md`; the rejected cache-copy routes remain
historical evidence. Orin/J6M/BPU execution remains deferred. The host
pipeline's Python and decoded-input qualifications remain unchanged.
