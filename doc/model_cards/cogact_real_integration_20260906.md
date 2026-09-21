# CogACT Real Integration

## Identity

This is the official-code **public-dependency candidate**, not a claim that the
gated original Meta configuration was obtained. Complete CogACT-Base weights
are fixed at HF revision `6550bf0992f162fc5d74f14ffee30771a9433363`, SHA-256
`1d35ec754c5c1ed7ab9ac22c9aba478e7269bab7db5b40df2703b2dc1e1009c0`.
The source is `microsoft/CogACT@b174a1b86deedfab4d198d935207e7bb0527994e`,
with declared `arnoldland/openvla@5603207085d55148682e2a35b868ad77d7b42ece`.
All four installed parameter groups were checked against every checkpoint tensor;
the upstream random action-head fallback is rejected.

The existing public official OpenVLA tokenizer/config candidate is pinned to
`47a0ec7fc4ec123775a391911046cf33cf9ed83f`. Its SentencePiece bytes match the
original Meta public LFS SHA. Full unknown original configuration equivalence
is not established. The isolated max-position 2048/4096 counterfactual passed
for the current actual positions 0 through 286, not for arbitrary profiles.

## Parameters And Compute

| Component | Installed Parameters |
| --- | ---: |
| Action DiT-B | 88,986,887 |
| Projector | 71,385,600 |
| DINO + SigLIP vision | 730,911,680 |
| LLM, including vocabulary head | 6,738,939,904 |
| Complete stack | **7,630,224,071** |

The full checkpoint file is 30,521,280,578 bytes. Parameters are installed FP32;
the official VLM executes with BF16 autocast, and the action head runs FP32.
This complete stack must not be relabeled as a 3B model or as only the DiT size.

The online partition requests the same second-last DINO/SigLIP features, then
calls the same native Llama decoder. CogACT consumes decoder hidden states, not
vocabulary logits. Its online prefix therefore omits the unused vocabulary-head
calculation (131,334,144 parameters in that head). The full reference model is
still loaded and its complete parameter audit includes this head. Serialized
artifact residency/parameter pruning must be measured separately; skipping
compute is not evidence of a smaller source checkpoint.

The fixed prefix is batch one, unpadded and has no prior tokens. It calls the
original decoder layers and final norm with the original positions, no past
state and `attention_mask=None`, preserving the eager implicit-causal SDPA
route. Native SDPA and absence of a static past cache are required. The unused
temporary KV values in the upstream full call are not cross-invocation context.
Padding, other attention implementations and cached decoding are not supported
by this bounded profile.

## Input And Scheduler

One actual Fractal/Google Robot observation: released episode 0/frame 0, with
instruction `pick rxbar chocolate from bottom drawer and place on counter`.
Dataset revision is `91bf7d7f7ce50770a1ba5c6db14b8d1c0815122e`; the RGB is decoded
from its released MP4, not original sensor bytes. Checkpoint normalization uses
`fractal20220817_data` statistics. Two preselected seeds are 42 and 43.

The complete action chunk is 16x7, with CFG batch 2, CFG scale 1.5 and ten DDIM
steps over original timesteps 90 through 0. Eta is zero, but the official CUDA
generator still consumes one initial `randn` and ten `randn_like` draws. An
explicit external producer performs those real draws. The Tensor IR consumes
their tape and carries the step index, sample, byte-state receipt and validity.
It atomically returns raw FP32, normalized FP32, native FP64 actions, final RNG
receipt and draw count 11. Autonomous C++ PRNG generation is not implemented.

## Evidence And Limits

Local evidence root:
`artifacts/edgefm-vla-goal/20260906-044509/cogact/`.
Remote base on H20-2:
`/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/cogact/`.

- `reference-candidate-001`, `reference-candidate-maxpos4096-002`: complete
  official reference and one-field counterfactual, exact outputs/intermediates/RNG.
- `partition-candidate-001`: official versus four-region Invocation, all ten
  steps, CFG tensors, complete actions and final real global RNG match exactly.
- `partition-capture-candidate-003` onward: actual invalid-receipt, padded-input
  and non-finite-noise transactions are rejected.
- `004`: three strict zero-error region captures, with prefix export failure.
- `set-capture-diagnostic-001`: minimal CPU set/tuple compiler counterfactual.
- `005`/`006`: prefix strict export succeeds after the tuple adaptation, but its
  execution fails a Float/BFloat16 assertion on unused logits. The diagnostic
  `.pt2e` is deliberately not a passed production capture.
- `prefix-export-diagnostic-001`: same saved real export/input/context, both
  outer-autocast cases fail at the same assertion. FX records the exact FP32
  input/weight, BF16 linear output and inconsistent FP32 exported metadata.
- `007`: omitting unused logits clears the dtype assertion, but strict export
  materializes a causal mask and changes cognition values. This run fails the
  unchanged zero-error gate.
- `prefix-export-diagnostic-002`: mask-only substitution changes output stride
  and fails a downstream view. `003` additionally copies equal values into the
  expected layout and restores complete cognition bitwise equality. These are
  diagnostic graphs, never production artifacts.
- `partition-capture-candidate-008`: production Adapter native layer/norm path;
  both seeds pass every step, CFG tensor, full raw/normalized/native action,
  producer RNG state and all five Invocation outputs. All four strict captures
  have max-abs 0 and pass the new recursive effect audit, including 97 named
  nested prefix GraphModules. Reloading the four saved graphs and executing the
  full ten-step Invocation again gives all five outputs bitwise equal, with no
  hidden global RNG consumption. No export graph is manually rewritten.

## Ordinary TorchScript Deployment

`torchscript-candidate-002` exported each of the four unchanged real 008 graphs
using the public `export_torchscript_region` device-preserving ATen profile.
Strict trace, archive save/load and complete per-region output bytes passed.
One Python Interpreter then ran seeds 42, 43, 42 with fresh revisions for every
input. All ten steps' four carry tensors, all five outputs and global RNG checks
passed with MSE/max-abs 0. The JIT optimizer is explicitly disabled.

`torchscript-candidate-004` reuses those exact four archives without recompiling.
The source checkpoint is unchanged. Its prefix archive is 29,580,184,174 bytes,
SHA-256 `03e85fbcd194d20dcd37a9f1e2032ba59a625981e83dd402718923d569f5f0e9`.
The public IR/Plan/bundle path produces a standalone C++ Session. The same
Session executes three fresh calls, seeds 42, 43, 42. Full raw FP32, normalized
FP32 and native FP64 16x7 actions, the 16-byte RNG receipt and int64 draw count
11 all match the candidate reference byte-for-byte. `ldd` and live process maps
contain no `libpython` or `libtorch_python`; Python paths are disabled for the
child. This is ordinary execution, not graph replay or a latency benchmark.

Failed 002/003 deployment attempts remain separate: 002 advertised abbreviated
dtype capability names against long capture spellings; 003 reached memory
planning and rejected the old long IR names. The public explicit dtype helper
now supplies canonical names for new Adapter declarations. For immutable 008
evidence, 004 records 110 per-TensorType alias conversions across IR/contracts,
keeps the original and canonical IR, hashes and I/O digests, and generates a new
certificate. No old certificate is inherited, no arbitrary metadata string is
rewritten, and no model tensor is cast or copied by this metadata conversion.

The current gate is **candidate reference + capture + TorchScript compilation
+ ordinary no-Python Session**, for one observation and two seeds. The Python
context schema 2 is observed/checked offline; this does not establish C++
numerical-provider enforcement or retroactively upgrade older records. External
RNG tapes, original Meta configuration equivalence, larger observation studies,
autonomous C++ PRNG, replay, benchmark tables/CDF and Orin/BPU validation remain
independent requirements. This is not full paper-fidelity acceptance.
