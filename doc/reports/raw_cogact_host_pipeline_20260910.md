# Recorded CogACT Host Pipeline

The original-input official and generated-Session formal campaigns are complete:
10/10 workers, independent remote audit, local complete-output rechecking and
inspected CDF PNG/PDF. Controller 143713 on `zzm-h20-x8-2` and its final worker
166189 ended successfully. No campaign restart is needed.

## Accepted Formal Evidence

Each configuration uses five independent processes, 128 warmups and 1024 measured
calls per process. All 10240 measured calls are retained. The complete output
audit includes warmups: 57600 five-port typed tensors are byte exact against the
frozen references. All floating action outputs have MSE and max absolute error
zero; minimum cosine is 0.9999999999999998. Integer RNG receipts are compared
exactly without converting them to floating point or assigning a cosine score.
Each required worker starts with zero replay counters and finishes with 1152
ten-step replays and zero ordinary fallback.

| Configuration | Mean ms | p99 ms | Std ms | Fresh Chunks/s |
|---|---:|---:|---:|---:|
| Official PyTorch components | 120.286270 | 127.398996 | 2.700216 | 8.313501 |
| Session required | 79.971651 | 82.199409 | 0.982177 | 12.504431 |

Mean preprocessing/H2D is 3.713346/3.531708 ms; mean inference/postprocessing/D2H
is 116.572923/76.439942 ms for official/required. Initialization is excluded and
reported separately: 198.506367/223.112195 s. Sampled peak worker memory is
30458/29550 MiB. These measurements retain the model and host-boundary
qualifications below.

Full local formal archive:
`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/raw-host-formal/cogact/`.

| Formal Evidence | SHA256 |
|---|---|
| `audit-formal.json` | `c9d8e4c5aa4450ec0f2df43e5b76b7466a4a2fcbc4950e156c4d895e903a133f` |
| `local-formal-audit.json` | `b1023beb23daf722832da99868c251ec19e5cb5fccae85907c14b08457b9a07a` |
| `figures/report.json` | `ca2f96066ab7c2e3068d93c5cbf28c6f0f691aafcf0eba6135559167da861732` |

The unified five-model result is in
[the H20 delivery report](h20_vla_experiment_delivery_20260910.md).
Pilot measurements below are retained as qualification evidence only.

## Accepted Pilot Evidence

Each of the two independent workers ran 16 warmups and 32 measured calls over
16 recorded observations. All 96 calls and 480 complete typed output tensors
match both frozen references byte for byte. The five outputs are raw F32 actions,
normalized F32 actions, native F64 actions, the U8 final CUDA RNG state and the
I64 actual draw count. Each action tensor has shape `[16,7]`. Native execution
records 48 ten-step replays from initially zero counters and zero ordinary
fallback. No output is narrowed to F32 for auditing.

| Pilot Only | Worker PID | NVML PID | Mean ms |
|---|---:|---:|---:|
| Official PyTorch components | 134983 | 3348645 | 115.607823 |
| Session required | 137970 | 3391940 | 80.231734 |

Pilot controller 134652 ended successfully. The full original-input formal
protocol requires five fresh processes per configuration, each with 128 warmups
and 1024 measured calls. The detached formal controller 143713 was launched after
the successful audited pilot; the accepted terminal results are reported above.

The shared timer starts with decoded RGB, instruction and the saved RNG tape
in CPU RAM. It includes the qualified tokenizer/image transforms, per-call H2D,
generation/action inference, native output conversion and all output D2H.
Initialization, image-file decoding, file IO, validation/logging and robot
transport are excluded. This Python host does not receive a no-Python certificate.

## Input and Model Scope

The input set is 16 spaced frames from one recorded Fractal episode: frames
0, 8, 16, ..., 112 and 114. The original manifest, decoded RGB hashes,
instructions, reference NPZ files and seven preprocessed tensor ports are
checked before measuring either implementation. The saved generator state
reconstructs all eleven actual draws and all twelve generator states exactly.
This is not a sixteen-episode or robot task-success evaluation.

The official branch uses the original generation and DDIM engines and original
NumPy action postprocessing formula, with qualified shared preprocessing. It
counts actual `randn`/`randn_like` calls without replacing their values. The
Session branch consumes the archived external RNG tape. This is an explicitly
qualified composition of official components, not the untouched high-level
`predict_action` wrapper or autonomous C++ RNG generation.

The checkpoint contains 7,630,224,071 actual parameter elements, including the
LLM and vision towers. It must not be labeled as a 3B complete model. Public
OpenVLA dependency configuration is used; equivalence to the original gated
Meta configuration remains unverified. Native action values retain the original
dataset normalization scale; physical robot calibration is not established.

The original bundle is unchanged. Model differences remain in
`adapters/cogact_host_pipeline.py`; the timer, complete-output contract, audit
driver and `NativeTensorSession` C ABI are shared with the other four VLAs.
The native image processor uses original timm registry metadata and the original
dual-image transform without constructing unrelated dummy model weights.

## Identity and Locations

Remote root:
`/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/raw-cogact-20260910-001/`.
Campaign `campaign-001/` is bound to H20-2 GPU2 UUID
`GPU-3596876b-e679-6289-f784-c8d98294ed23`.
The 305-file implementation snapshot and source provenance are frozen.

| Evidence | SHA256 |
|---|---|
| Campaign protocol | `08a5e9a3519a14e73c3695941bd637eac4d30a4d982650334b41a1d9d47559a6` |
| Source provenance | `7f22cb27d56f0f25a67077f4b90615e6ac59cea5542d795a6adb6eab04ccdc82` |
| Shared Session library | `ef3eca721dd0cdfcddefd5c23794c6b146363d54ed89fd756da5a65ad6b11f41` |
| Remote pilot audit | `4d6af65a5a4c100f9c64895a3b6d56e16078de284d54d2f630c14dd70c9e2c87` |
| Local complete-data pilot audit | `b74a41dab35e3080c275135bcb3e748b118bb97ca004f358f67d54f845dcbbe4` |
| Recorded observation manifest | `85251486df36a62651e2f12ab916318c3560daf63b40de419b3329250b45eae1` |
| Original 16-sample reference series | `5e6c39e9b9d43a8ec9663592424ab0e74ab142e2812d6970eb5ab85704a5b8db` |
| Complete checkpoint | `1d35ec754c5c1ed7ab9ac22c9aba478e7269bab7db5b40df2703b2dc1e1009c0` |

The complete local pilot archive is
`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/raw-host-generic-pilots/cogact/`.
It includes raw samples, all five output byte streams, worker reports, ownership
monitors, remote audit and `local-pilot-audit.json`.
The verified reference archive is in the adjacent `host-cogact-formal/prepared/`.
The original 16-sample source is NAS `cogact/series-003/`; the older 115-frame
`series-005` report is separate and does not enlarge this campaign's coverage.

## Validation

Focused adapter tests: 90 passed. Full CPU regression after these production
changes: 2259 passed, 74 skipped, zero failures. It includes input integrity,
random-call observation and owner-thread rejection tests. The actual H20 CPU
processor check also matched all 48 token/DINO/SigLIP tensors over 16 raw images
without loading model weights or allocating CUDA tensors.

Under the current worktree's `artifacts/recovery-audit-20260909/`:

| Artifact | SHA256 |
|---|---|
| `cogact-host-focused-tests-002.xml` | `89f8176aefbd7912367e8037181295a300b4174871d8eec3efc2298b0b7c6902` |
| `cogact-host-full-cpu-001.xml` | `1dad1145afa0d2c7749c68a9954e7119bc353700ca7c9aa71367371292e60ad0` |
| `cogact-cpu-processor-001.json` | `054d3db7c9991b84c7a9e0ea16e5899b2e273c601ea5c04a4ac19a050d81e651` |

All ten terminal workers, independent remote audit, local complete-output audit,
recomputed latency/throughput and inspected CDF figures are now present. The final
cross-model CPU checks are in `h20_final_validation_20260910.md`. Original Meta
configuration equivalence, autonomous C++ RNG and board execution remain outside
this accepted public-dependency, external-RNG H20 profile.
