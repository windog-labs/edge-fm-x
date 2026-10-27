# CogACT Parallel Host Preparation

2026-10-22. Host source/export work only. No Orin access, target build,
other-platform performance run, GPU use, Torch installation, shared source
edit, source-artifact modification or checkpoint substitution was performed.
The parent campaign retains ownership of Orin RDT and shared native F64/I64
host-buffer work. Neither activity establishes CogACT target acceptance.

## Source Identity

The actual checkpoint is `CogACT/CogACT-Base` revision
`6550bf0992f162fc5d74f14ffee30771a9433363`, 30,521,280,578 bytes, at:

```text
zzm-4090-x8-1:/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/assets/CogACT/6550bf09/checkpoints/CogACT-Base.pt
```

A fresh streaming SHA256 on that host reproduced
`1d35ec754c5c1ed7ab9ac22c9aba478e7269bab7db5b40df2703b2dc1e1009c0`.
The original repo's `cogact/hf-model-info.json`, checkpoint audit and complete
reference report bind that file to all **7,630,224,071 installed elements**:
action 88,986,887; projector 71,385,600; vision 730,911,680; language
6,738,939,904. This is the public-dependency candidate, with upstream source
`microsoft/CogACT@b174a1b86deedfab4d198d935207e7bb0527994e` and declared
`arnoldland/openvla@5603207085d55148682e2a35b868ad77d7b42ece`.

The dependency is public `openvla/openvla-7b` revision
`47a0ec7fc4ec123775a391911046cf33cf9ed83f`. SentencePiece byte identity with
the public Meta LFS hash does not establish gated original Meta configuration
equivalence. No nominal 3B identity, gated-checkpoint identity, random fallback,
smaller checkpoint or pruned checkpoint is claimed.

The local original source/evidence root is:

```text
/home/zhangzimo/Repos/private/edge-fm-x/artifacts/edgefm-vla-goal/20260906-044509/cogact/
```

Its `partition-capture-candidate-008/remote/{report.json,module.json,seed-42,seed-43}`
contains complete real inputs, traces and outputs. These are distinct from the
115-frame `cogact-data-002` handoff and the later sixteen-frame H20 raw profile.
The new component check uses one observation and two seeds, not 115 fresh
observations. No CogACT ONNX files were found under the historical NAS CogACT
root before this work; actual TorchScript and strict-export graphs are present.

## Complete Path And ABI

The original observation is released Fractal episode 0/frame 0, decoded RGB
`uint8[256,320,3]`, dataset revision
`91bf7d7f7ce50770a1ba5c6db14b8d1c0815122e`, with instruction
`pick rxbar chocolate from bottom drawer and place on counter`.

The pinned upstream image transform provides both `dino` and `siglip`
`float32[1,3,224,224]`. The actual fast Llama tokenizer/prompt builder supplies
`int64[1,31]`, including cognition suffix `[29871,2]`, batch one, no padding.
`prepare_cogact_observation` is the existing processor API; native online
decoded-observation processing has not been implemented by this preparation.

The unchanged four-region path is:

1. `vf_cached_cogact_prefix_17`: both second-last DINO/SigLIP features, fused
   projector, token embeddings, all 32 original Llama decoder layers and final
   norm; emits `float32[1,1,4096]` cognition. Its name does not authorize cached
   online encoder features. Source storage is FP32, VLM arithmetic follows the
   captured BF16 autocast profile. The already-declared unused vocabulary-head
   calculation is absent; its 131,334,144 elements remain in the full source
   checkpoint audit. Export residency is separate from checkpoint identity.
2. `cogact_initialize`: consumes real initial noise `float32[1,16,7]`, all
   twelve `uint8[12,16]` CUDA state snapshots and `uint8[16]` initial receipt;
   emits CFG sample `float32[2,16,7]`, `int64[1]` index, `uint8[16]` receipt
   and `bool[1]` validity.
3. `cogact_step`, executed **ten times**: full FP32 DiT-B with CFG batch two,
   scale 1.5 and DDIM original timesteps `90,80,70,60,50,40,30,20,10,0`.
   Every call retains the complete `float32[10,2,16,7]` step-noise tape and
   byte-state table. It returns all four carry tensors. Eta zero does not
   remove any source draw or noise input.
4. `cogact_finish`: retains full `float32[16,7]` raw and normalized actions,
   clip/gripper conversion, FP32 scale preparation, FP64 checkpoint
   `fractal20220817_data` q01/q99/mask conversion, and full `float64[16,7]`
   native actions. It also emits exact `uint8[16]` final RNG receipt,
   `int64[1]` draw count 11, and `bool[1]` transaction acceptance. The five
   public output ports are committed only after acceptance.

There is no input-ID narrowing, action-chunk shortening, solver-step reduction,
output omission, checkpoint precision rewrite or new parameter pruning. The
existing exporter performs its declared constant-folding pass, including casts
already present in the traced mixed-precision graph; exported tensor residency
must not be read as the original FP32 checkpoint's parameter/storage count.

## RNG Boundary

The qualified tape has one real CUDA `randn[1,16,7]` draw and ten real
`randn_like[2,16,7]` draws, including the eta-zero calls. It carries before,
after-initial and all ten after-step states. This host work reuses these exact
bytes and makes no new random draw. The action checks verify all receipt and
draw-count bytes without cosine comparisons on integer/byte outputs.

Historical H20 autonomous `CudaRngProvider` uses LibTorch CUDA generators/ATen
and is a separate boundary. It is not supplied by ONNX export or by native
host-buffer F64/I64 copying. This preparation supplies neither autonomous
TRT-only RNG nor a native CogACT postprocess provider.

## Actual Exports And Checks

All new local artifacts are under `artifacts/orin/20261022-07/`.
The three small archives copied from the NAS are unchanged and independently
hashed before/after use. They come from `torchscript-candidate-002/artifacts/`,
reused without recompilation by the historical candidate 004.

| Region | Archive SHA256 | ONNX SHA256 |
|---|---|---|
| Initialize | `68d9e80c8b79c1b2ffad8b3e900223b191c2cf6b7bc22ef9e79c43ef0e89e1ee` | `13cca7f15ac7c9f3f51fc49ddccdb9f3f7091a023b5d3f54650b4f51fc39f1ee` |
| Full action step | `1c987894e539b01a42a24bb9714f337912b802fa5d98aac90126530715799c4b` | `57f4ac00edcc98f0358dc6367a83b806fb4a86177fa77b893c57673ad4d2f623` |
| Finish, including FP64 | `03035688f85b61c7f0f18c104826c075508622fbe4fe952ed17dd37f42a66520` | `b6fbb8c69581971ab48ae68ef7c2ba5ab2e52f0b7fcab0afcfba475d03fe8c89` |

`prepare_cogact_inputs.py` hashes the frozen source receipts and every real
source payload before reading the trusted trace archive. It prepares all four
regions' CPU inputs without casting them. Component action validation uses
the recorded official/partition cognition, so it excludes online prefix
coverage explicitly.

`run_export_request.py` is model-neutral and invokes the existing
`vlaforge/tools/export_torchscript_region_onnx.py`, SHA256
`83f019970a4bd0af5bacd21d2dc70b58b66d487646387645dd65eed3bbf56908`.
It pins archive/input/tool bytes, checks available memory, runs sequentially
with hidden GPUs and bounded threads/virtual memory, and retains logs and
failure reports. Source files and shared exporter/runtime code are unmodified.
Device-constant relocations are explicit and recorded, not dtype conversions.

All three opset-18 ONNX checkers pass. `validate_action_chain.py` checks every
port on all ten carried steps for seeds 42/43, ORT versus CPU TorchScript and
both versus frozen source. All **270 comparisons pass** the unchanged floating
gates `MSE < 1e-5`, `cosine > 0.9999`, with exact shape/dtype/bytes for every
nonfloating port. This is tolerance parity, not byte equality of actions.
Worst complete-output source MSE: raw `5.1436534728377716e-8`, normalized
`4.1686066849090167e-8`, FP64 native `4.45845175622723e-9`.
All ten invalid-receipt/short-loop/invalid-validity/padding/nonfinite
transactions are rejected. CPU RNG state is unchanged. The nine CPU guard
tests reject changed pins, F64 downcasts, dropped RNG ports, shortened chunks,
large-integer byte mismatches, nonfinite values and relaxed floating gates.

## Combined Prefix Export

Fresh NAS SHA256 matches archive
`03e85fbcd194d20dcd37a9f1e2032ba59a625981e83dd402718923d569f5f0e9`,
29,580,184,174 bytes. The parent explicitly authorized bounded CPU work on
`zzm-4090-x8-1`; all GPUs remain hidden. The reusable environment is
`envs/cogact-cu128-py311` (Torch 2.10.0+cu128), reading existing RDT task-local
`host-deps` for ONNX 1.21.0/ORT 1.24.4. No package was installed.

The host's `/tmp` has only about 2.1 GiB free. The isolated output directory is
therefore `/dev/shm/edgefm-cogact-20261022-07/`, on a 479 GiB tmpfs. The export
requires 220 GiB available before launch, one CPU and a 192 GiB address-space
ceiling. PID `2252449` started `2026-10-22 13:47:08 UTC`; its exact argv,
`/proc` start ticks and boot identity are preserved in the owner receipt.

The prefix export exited successfully. It retains all three input ports and
the complete FP32 cognition output: 7,610 nodes, 939 initializers, 1,695,357
protobuf bytes and **752 external files / 16,071,568,704 external bytes**.
Graph SHA256 is
`01fbce7e30a02cef277c70b49e65c303ec8626314bac344a403fa9f12ce669d4`.
Every external file has its own hash in the export receipt. All files are
copied to local `prefix-native/`; these are source-semantic mixed-precision
graphs, not accepted TRT engines.

The existing generic exporter records FP32 opmath for the source's 51 BF16
GELUs and 32 BF16 SiLUs, with BF16 result boundaries. It explicitly relocates
one device constant in `submod_1`. This is declared source composite lowering,
not a full-model FP32 replacement.

Basic ONNX checker success does not imply runtime/type support. The first
actual ORT probe rejects native BF16 `Conv` `/submod_1/Conv_1` at opset 18:
`Type 'tensor(bfloat16)' ... of operator (Conv) ... is invalid`.
The unmodified failure is `receipts/prefix-cpu-onnx-001/report.json`.

## Portable CPU Reference And Limits

The existing generic `lower_onnx_bfloat16_carrier.py`, SHA256
`e45ca0f0bd6e71067ad3eb1287406dfe96ad5ea2dc571dc33482de9b75c3a84e`,
then emitted a standard portable CPU reference. Its declared representation
uses FLOAT storage for BF16 values, preserving **1,938 explicit BF16 rounding
boundaries**, with 594 BF16 constant tensors promoted to their exactly
representable FLOAT values. Integer ports are unchanged. No plugin, target
compute policy or source equivalence is certified by this transformation.
The original native BF16 graph and all failures are retained separately.

The reference contains 11,486 nodes, 2,110,436 protobuf bytes and one
29,467,326,976-byte external weight file. Graph SHA256:
`e126706db3f5cf23b830e1fc5782b5aa01ce2be3d9c44c88d086b90b5bd734ca`.
External data SHA256:
`c5b46ae88307e9396f8e15eb807042fd8b77bd74dc40831b63e02c59eb66812b`.
The transformation receipt remains `lowered-not-runtime-validated`; subsequent
execution evidence is separate. Files are under local `prefix-carrier/` and
remote `/dev/shm/edgefm-cogact-20261022-07/prefix-carrier-build-001/`.

One untimed real prefix forward executes with ORT CPU and graph optimizations
disabled to retain the explicit precision boundaries. It emits finite full
`float32[1,1,4096]` cognition. Both this output and the original CPU TorchScript
prefix output are saved without truncation, with independent process identity
and output hashes. The source checkpoint was never loaded as a smaller model.

The computed portable cognition then feeds the unchanged ten-step action
chain for both actual seed tapes. All ten carries and all five final output
ports are checked. Worst final-output metrics against frozen source are:

| Complete Output | Worst MSE | Minimum Cosine | Status |
|---|---:|---:|---|
| Raw FP32 `[16,7]` | `5.575131229275925e-6` | `0.9999904493681134` | Pass |
| Normalized FP32 `[16,7]` | `5.562244557482766e-6` | `0.9999903805913873` | Pass |
| Native FP64 `[16,7]` | `6.832823774802038e-7` | `0.9999977716924475` | Pass |
| Final U8 RNG receipt | Exact full bytes | Not applicable | Pass |
| I64 draw count | Exact value 11 | Not applicable | Pass |

These terminal results do not conceal the intermediate precision discrepancy.
Original CPU TorchScript cognition versus frozen H20 cognition has MSE
`5.0120844474426816e-5`, cosine `0.9999696797435503`. The portable reference has
MSE `6.085812950533793e-5`, cosine `0.9999628651997583`. Both intermediate MSE
checks fail the same `<1e-5` diagnostic threshold. Accordingly, both computed
prefix/action receipts retain aggregate **failed** status: 272 comparisons,
two failed cognition comparisons, 270 passed comparisons, and ten rejected
invalid transactions each. No threshold was relaxed or failed run replaced.

The portable prefix also differs from the original CPU TorchScript prefix:
MSE `6.034559051995587e-5`, cosine `0.9999634962374894`. Its same MSE diagnostic
fails independently of the CPU-versus-H20 discrepancy. Explicit BF16 rounding
boundaries therefore do not establish source equivalence. The three-way
comparison is retained in `receipts/preparation-summary.json`.

The joined calculation is an offline host source check: one computed prefix
observation and two full action tapes. It is not online raw E2E, a native
Session, sixteen-frame coverage, autonomous RNG, Orin support or target
performance. The additional FLOAT carrier is a CPU reference for precision
diagnosis, not an approved full-model FP32 target fallback. F64/I64 host
materialization does not implement native CogACT postprocessing or authorize
TensorRT DOUBLE arithmetic.

## Reproduction And Handoff

`receipts/asset-audit.json` binds source reports, every graph and every external
weight file. `receipts/preparation-summary.json` preserves the separate action,
prefix and terminal results. `manifest.json` hashes all local artifact files
and this report. Guard-test evidence is `receipts/export-guards.xml` (9 passed).
Real input/output hashes, no-new-RNG evidence and all CPU resource/owner records
remain under `receipts/`. Runtime providers/exporter sources were not edited.
Full ONNX type checking passes the three action/control graphs and portable
prefix reference, and rejects the native BF16 prefix. All 753 external tensor
files match their export hashes. All export/probe/transfer processes have
exited; `receipts/processes-final-001.json` records zero task processes. Remote
tmpfs files remain available for reproduction; all graph/weight payloads also
have durable local copies in the assigned artifact directory.

The light next command reproduces the full portable join and its two preserved
intermediate MSE failures using the already-computed actual prefix output:

```sh
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 -B artifacts/orin/20261022-07/validate_action_chain.py \
  --cognition artifacts/orin/20261022-07/receipts/prefix-cpu-carrier-001/cognition.npy \
  --receipt artifacts/orin/20261022-07/receipts/full-portable-prefix-action-002.json
```

For a fresh portable prefix forward on the high-memory host, use the existing
CogACT environment and RDT `host-deps` shown above, GPUs hidden, then run:

```sh
python -B /dev/shm/edgefm-cogact-20261022-07/probe_prefix.py \
  --request /dev/shm/edgefm-cogact-20261022-07/prefix-carrier-request.json \
  --mode onnx \
  --output-root /dev/shm/edgefm-cogact-20261022-07/prefix-cpu-carrier-002
```

`probe_prefix.py` itself applies one-CPU affinity and a 160 GiB address-space
ceiling. Its native request instead reproduces the BF16 Conv type failure.
Next model work is explicit prefix precision diagnosis, wider source checks,
and declared native raw preprocessing/postprocess/RNG partitions before any
Orin qualification. The complete original checkpoint and reusable neural
exports are now located and bound; target acceptance remains unperformed.
