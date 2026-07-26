# VLAForge Claim–Evidence Map

> Status: paper audit checklist
>
> Scope: Host-CUDA paper claim on RTX 3060 / CUDA 12.8. Orin, a real vehicle,
> sensor middleware, and OpenVLA real L4 are not completion gates.

This file maps each paper-facing statement to committed machine-readable
evidence and its exact boundary. A claim must be narrowed if its required row
does not pass; fixture evidence can never upgrade a real-model claim.

## Evidence levels

| Level | Meaning | Minimum evidence |
|---|---|---|
| L0 | source/paper contract mapping | pinned upstream revision and inspected entry points |
| L1 | deterministic executable fixture | verified Semantic IR/Plan parity; explicitly labelled fixture |
| L2 | real frontend | real checkpoint/source tensors; eager/export parity |
| L3 | real artifact | compiled real checkpoint artifact parity |
| L4 | real generated deployment | no-Python generated C++ Session parity |

## Core claims

| ID | Paper claim | Required evidence | Current evidence | Status | Boundary |
|---|---|---|---|---|---|
| C1 | VLAForge represents a caller-driven stateful model invocation with a 15-op VLA-specific core | frozen schema/opcode audit, verifier tests, architecture scanner | `vlaforge_heldout_v01/heldout_audit.json`, `vlaforge_architecture_v01/architecture_surface.json` | passed | not a tensor algebra, scheduler, or middleware IR |
| C2 | Authoritative state and derived cache have different lifetime/failure contracts | state/cache memory records, commit/abort tests, real stateful model | SmolVLA and MindDrive real L4, `paper_ablations.json` | passed | derived cache may be dropped; authoritative state may not |
| C3 | `InputRevision` safely controls exact cross-Run reuse | same/new/missing/reset traces and cache-only performance control | 40-task exact-reuse ablation | passed | caller must assign truthful revisions; missing is safe-by-default |
| C4 | State and named outputs commit transactionally | validation failure, version sequence, prior-output preservation, retry | SmolVLA, DiffusionDrive, and 16-state/10-output MindDrive real L4 failure/retry | passed | vehicle safety and fallback policy remain external |
| C5 | Generated C++ adds low orchestration overhead over identical direct artifacts | five workloads, five independent processes, direct-vs-generated exact output | 150-task CUDA matrix | passed | approximately 0.5% on two models and one RTX 3060 only |
| C6 | Generated deployment is no-Python and contract verified | clean wheel, non-Git cwd, invalid Python env, `ldd`, negative schema/hash/target cases | reproducibility report and deployment-boundary ablation | passed | does not imply a Python-free compile toolchain |
| C7 | A frozen core covers robot and driving VLA paradigms | model matrix, real robot/driving artifacts, held-out real model with core delta zero | SmolVLA L4, OpenVLA L3, MindDrive complete driving-VLA L4, DiffusionDrive planner L4 | passed | MindDrive is full six-camera-to-trajectory; DiffusionDrive is not a language VLA |
| C8 | Memory is statically bounded and stable | Plan certificate, memory-class split, 10k soak | static-arena ablation, SmolVLA/DiffusionDrive 10k soaks, and MindDrive 1k 16-state soak | passed | packing savings are small; claim boundedness, not compression |

## Performance evidence

| Evidence | Protocol | Result used in paper | Source |
|---|---|---|---|
| full-compute matrix | 2 models × 5 workloads × 3 paths × 5 fresh processes; 5 warmups; 30 samples | 30 cells, 150 tasks, 4,500 samples, all parity passed | `doc/reports/vlaforge_cuda_matrix_v01/cuda_paper_matrix.json` |
| SmolVLA overhead | direct AOTI control vs generated Session | 0.43–0.63%, mean 0.508% | same |
| DiffusionDrive overhead | direct AOTI control vs generated Session | 0.15–0.71%, mean 0.509% | same |
| exact reuse | 4 modes × 2 models × 5 processes; 100 samples | DiffusionDrive same revision 3.064 ms vs full 16.403 ms, 5.353x | `doc/reports/vlaforge_ablations_v01/paper_ablations.json` |
| MindDrive generated L4 | 4 revision modes × 5 fresh processes; 1 warmup + 10 samples; 2,000 cluster bootstraps | full/same/new/missing warm means 1270.38/260.01/1279.27/1281.75 ms; same vs new 4.92x | `doc/reports/vlaforge_minddrive_v01/minddrive_l4_benchmark.json` |
| MindDrive stateful soak | 1 warmup + 1,000 same-revision Runs | 1,000 hits, 16,000 state commits, CUDA drift 0, RSS drift +60 KiB | `doc/reports/vlaforge_minddrive_v01/minddrive_l4_soak.json` |
| static memory | unpacked logical-lifetime control vs verified packed plan | small byte savings; 10k Runs with zero CUDA drift | same |
| deployment boundary | direct AOTI, generated C++, clean installed wheel | direct/generated exact; no `libpython`; all negative cases rejected | same |

## Correctness and failure evidence

| Property | SmolVLA | DiffusionDrive | MindDrive | Held-out AutoVLA |
|---|---|---|---|---|
| real eager/frontend | L2 | L2 | complete L2 | L2 partition, exact |
| compiled artifact | L3 | L3 | complete held-out L3 | L3 candidate not promoted |
| generated C++ | L4 | L4 | L4 | not required |
| same/new/missing revision | passed | passed | passed on real generated L4; full/same/new/missing each have 5 fresh-process traces | same/new passed |
| episode reset | passed | passed | passed for 16 states | stateless partition |
| authoritative state commit/abort | queue/cursor versions passed | stateless, N/A | 128 commits + one abort passed | stateless partition |
| transactional named outputs | action output | six planning outputs | ten trajectory/detection/motion outputs | trajectory + action tokens passed |
| no-Python Session | passed | passed | typed/generic passed | not required at held-out L2 |

## Generalization evidence

| Paradigm | Representative | Outputs/state | Highest honest evidence | Core delta |
|---|---|---|---:|---:|
| discrete token policy | RT-1-like | token + detokenized action; no state | L1 fixture | 0 |
| action chunk | ACT-like | chunk; Adapter queue/cursor | L1 fixture | 0 |
| optional-modality diffusion | Octo-like | action chunk; condition cache | L1 fixture | 0 |
| autoregressive robot VLM | OpenVLA | action tokens/action; loop KV SSA | real L3 | 0 |
| flow/chunk VLA | SmolVLA | continuous chunk; queue/cursor | real L4 | 0 |
| multi-embodiment DiT | GR00T-like | chunk + aux; bounded DiT | L1 fixture | 0 |
| driving trajectory | trajectory fixture | one trajectory; no state | L1 fixture | 0 |
| driving autoregressive | AutoVLA | trajectory + tokens | real L2 partition | 0 |
| driving diffusion | DiffusionDrive | candidates/scores/trajectory/aux | real L4 | 0 |
| stateful multimodal driving VLA | MindDrive 0.5B | trajectory/path/commands/detection/motion + 16 states | real L4 | 0 |
| external-feature hybrid | DriveVLM-Dual-like | trajectory/prediction/map/VQA | L1 fixture + C++ plugin fixture | 0 |

## Artifact/reproduction gates

| Gate | Required result | Current source |
|---|---|---|
| installed wheel only | compile bundle and generated runner without repo imports | `vlaforge_reproducibility_v01` |
| invalid Python environment | runner succeeds with invalid `PYTHONHOME/PYTHONPATH` | same |
| no Python link | `ldd` has no `libpython` | same |
| schema/ABI/hash/target mismatch | every negative case rejected | same |
| Python suite | all non-opt-in tests pass | 238 passed, 10 explicit real-environment opt-in skipped |
| CPU CTest | 7/7 | passed |
| CUDA CTest | 8/8 | passed |
| live CUDA AOTI | pass | 1/1 passed |
| clean installed wheel/no-Python | current wheel, non-Git cwd, invalid Python environment | passed |
| clean worktree/report provenance | `source_dirty=false` for formal runs | matrix, ablation, AutoVLA L2, release and reproducibility reports pass |

## Prohibited claim inflation

The paper and abstract must not say or imply:

- VLAForge designed or optimized the model CUDA kernels;
- cache-hit or action-queue fast-path latency is full-compute model latency;
- fixture support is real checkpoint support;
- OpenVLA has real L4 unless a generated real-checkpoint Session actually
  succeeds;
- one RTX 3060 proves cross-GPU or embedded performance;
- static arena packing delivers meaningful memory compression in the current
  two real L4 workloads;
- VLAForge owns sensor synchronization, ROS/Cyber, periodic scheduling,
  dropped-frame policy, control publication, or vehicle safety;
- Orin, a real vehicle, or a second machine is required for the current
  Host-CUDA paper completion.

## Final completion rule

The current paper is submission-ready when:

1. the held-out real model reaches at least honest L2, or a formal
   12-GiB/31-GiB resource blocker narrows C7;
2. the paper draft and figures reference only committed machine-readable data;
3. Model Cards and the real-evidence index match final levels;
4. the reproducibility manifest includes matrix, ablation, held-out, paper, and
   external archive identities;
5. the final Python/C++/CUDA gate passes.

Orin, OpenVLA real L4, real-vehicle/sensor integration, cross-GPU performance,
and second-machine reproduction remain optional evidence after these five
conditions pass.

Current status: all five conditions pass. The machine-readable final decision
is `doc/reports/vlaforge_paper_completion_v01/paper_completion.json`.
