# CogACT Real Reference Contract

This adapter is model-specific integration code. No runtime, IR, Plan or public
compiler code branches on the CogACT model name. The current bounded milestone
is a real-weight official-code reference candidate, four-region partition,
strict saved-graph execution and ordinary no-Python TorchScript deployment for
one observation and two seeds. It is not a performance result or paper fidelity
acceptance.

## Immutable Identity

- Official CogACT source: `microsoft/CogACT`, commit
  `b174a1b86deedfab4d198d935207e7bb0527994e` (MIT).
- Official declared OpenVLA dependency: `arnoldland/openvla`, commit
  `5603207085d55148682e2a35b868ad77d7b42ece`.
- Declared data-loader dependency: `moojink/dlimp_openvla`, commit
  `040105d256bd28866cc6620621a3d5f7b6b91b46`. The actual dependency is imported;
  training dataset imports are not replaced by stubs.
- Public `CogACT/CogACT-Base`, revision
  `6550bf0992f162fc5d74f14ffee30771a9433363`, full checkpoint 30,521,280,578 bytes,
  SHA-256 `1d35ec754c5c1ed7ab9ac22c9aba478e7269bab7db5b40df2703b2dc1e1009c0`.
- The actual config selects `prism-dinosiglip-224px+7b`, DiT-B, past window 0,
  future window 15, 7 action coordinates. Checkpoint state contains 7,630,224,071
  tensor elements, all FP32. Do not label this released asset as a 3B model.

The strict loader verifies full checkpoint bytes plus configuration/statistics,
requires all four state groups, calls upstream `load_vla` and checks every
installed state tensor against the checkpoint. Missing action, vision, projector
or LLM weights are fatal. Upstream's optional randomly initialized action-head
fallback is forbidden. Actual model parameter counts are reported separately
from checkpoint tensor-element counts.

## Public Dependency Candidate

Direct access to the original `meta-llama/Llama-2-7b-hf` configuration was denied
by the service for the available local authentication. No gate was bypassed and
no new terms were accepted. An existing official public
`openvla/openvla-7b@47a0ec7fc4ec123775a391911046cf33cf9ed83f` dependency is used as
an explicitly labelled candidate, never as evidence of original Meta access.

Its `tokenizer.model` SHA-256 exactly matches the original Meta repository's
public LFS metadata: `9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347`.
All public tokenizer files are independently bound to official pinned metadata;
BOS/EOS/PAD IDs, vocabulary length and cognition-suffix IDs are checked. The
public text configuration is materialized exactly as the public OpenVLA code
does: `transformers==4.40.1 LlamaConfig(**text_config)`. Resolved fields and hashes
are saved. Its `max_position_embeddings` is 2048. The original full Meta config
is unavailable, so full field equivalence is **not verified**. Actual rotary
positions are observed, but an in-range observation alone cannot establish all
configuration equivalence.

Both official timm pretrained dependencies are supplied as pinned local files,
using only the factory's `pretrained_cfg_overlay` file-loading option. Default
architecture and preprocessing config remain upstream-controlled:

- `timm/vit_large_patch14_reg4_dinov2.lvd142m@f3c408e77602bb412aa65fb03dfa0d5f95cb3832`
- `timm/ViT-SO400M-14-SigLIP@9179d15177ece40964c50492136eda2f3e0c9f61`

The model then strictly installs its complete fine-tuned vision state. No other
model's cached vision weights are silently substituted.

## Official Action Semantics

The full official `predict_action` is called unchanged. It constructs its own
prompt, tokenization, two-vision preprocessing and VLM cognition token, then
performs a complete fresh action chunk:

- 100-step trained squared-cosine diffusion schedule, respaced official DDIM N10.
- CFG 1.5 uses paired conditional/unconditional batches and the learned
  unconditional embedding. Initial noise is duplicated, not independently drawn.
- The released DDIM implementation calls `randn_like` at every step even when
  `eta=0`. Preserve these draws and the resulting RNG transition.
- Save initial noise, all 10 per-step noise draws, scheduler arrays and timestep
  map, every CFG input/output and step result, and CPU/CUDA/NumPy/Python RNG state.
- The raw diffusion output is distinct from the official normalized result:
  upstream clips to `[-1, 1]` and binarizes coordinate 6 at 0.5, then applies the
  checkpoint's selected `q01`, `q99` and mask to obtain native actions.
- Save all 16x7 raw, normalized and native values. Do not validate only the first
  action or relabel postprocessed normalized values as raw scheduler output.

An uninstrumented official call and an instrumented call begin from identical
saved RNG state. Their full outputs and RNG transitions must agree exactly.
This checks observational transparency, not a compiled model's fidelity.

## Real Input And Evidence

`prepare_real_cogact_inputs.py` extracts the explicitly selected episode/frame
from the pinned public IPEC LeRobot conversion of Fractal/RT-1, checks source
file hashes, robot type, frame index and video/parquet timestamp agreement.
The image is a real Google Robot observation, not simulation or an illustrative
repository image. Its boundary is RGB decoded from the released dataset MP4,
not original sensor or JPEG bytes. The instruction is read from the actual task
index. Use `fractal20220817_data` checkpoint statistics, not another embodiment.

The initial campaign uses episode 0/frame 0 and seeds 42 and 43, selected before
model execution. It is not a robot task-success or dataset-wide quality study.
Artifacts live under `artifacts/edgefm-vla-goal/20260906-044509/cogact/` and remote
assets remain on the shared NAS. Candidate reports retain
`strict_original_meta_config=false`, `paper_fidelity_gate=not_verified` and
`no_python_deployment=false`. Orin/BPU validation remains deferred.

## Verified Bounded Results

`reference-candidate-001` completed on H20 GPU 0 with every installed state tensor
equal to the pinned checkpoint and a measured parameter count of 7,630,224,071.
Both full 16x7 actions passed finite and bitwise observational checks. Actual
rotary positions were 0 through 286. All parameters were FP32, the VLM used BF16
autocast, and the action head remained FP32. A separate import audit observed
`float32_matmul_precision=high` immediately after importing Torch and after each
later dependency import; this is an environment fact, not evidence of a CogACT
constructor setter.

`reference-candidate-maxpos4096-002` explicitly changed only
`max_position_embeddings` from 2048 to 4096. Its complete raw, normalized and
native actions, all CFG/step tensors, 11 noise draws and every saved RNG state
were bitwise identical for both seeds. `maxpos-full-comparison.json` records
MSE=0 and max-abs=0, plus the raw computed cosine values. The latter can exceed
1 by float64 rounding and are not used as a strict equality gate.

The frozen Transformers 4.40.1 standard `LlamaRotaryEmbedding.forward` computes
from `inv_freq` and actual `position_ids`; its backward-compatibility cosine/sine
buffers are not used by this forward. CPU FP32/BF16 component checks agree with
the full-model diagnostic. Changing the field doubles these unused cache
allocations, so memory accounting must still retain the actual configuration.
This result does not establish equivalence for unknown original fields,
different rotary scaling variants, other inputs, or the unavailable Meta config.

## Explicit RNG Partition

`cogact_partitioned.py` uses the existing `InvocationBuilder` without modifying
the common runtime or dispatching on a model name. Four Tensor regions implement
cached vision/language cognition, initialization, a bounded ten-step DDIM loop,
and full-chunk postprocessing/validation. The loop carries the complete 2x16x7
CFG sample, CUDA int64 step index, byte state receipt and validity predicate.
The real DiT retains both conditional and unconditional branches. The scheduler
uses the checkpoint's exact coefficient arrays, original timesteps 90 through 0,
CFG scale 1.5, eta 0, and original floating-point operation ordering.

The batch-one, unpadded all-multimodal prefix is an explicit specialization of
the upstream mixed-batch orchestration. The output transaction additionally
rejects padded inputs or an incorrect cognition-token suffix. It does not claim
that arbitrary padding, text-only inputs or other batch profiles are supported.

The producer performs the real initial `randn` and all ten upstream
`randn_like` calls on CUDA, including the eta-zero calls. Input ports contain the
initial draw, ten step draws, 12 before/after CUDA-state snapshots and the initial
state receipt. The IR consumes each tape entry and returns the final receipt
and draw count 11 atomically with raw, normalized and native 16x7 actions.
This is **not autonomous C++ PRNG generation**. A receipt checks tape ordering,
not that a caller-provided noise tensor was mathematically generated from its
claimed state. The verified producer and immutable input-pack hashes provide
that provenance. Producer-side global state has already advanced before an IR
transaction starts; an output rejection does not undo those external draws.

`partition-candidate-001` verified seeds 42 and 43 on real H20 weights/input:
all ten intermediate samples and CFG inputs/outputs, complete normalized/native
actions and all five Invocation outputs were bitwise equal to the official
candidate. The producer's final CUDA/CPU/Python/NumPy RNG snapshots exactly
matched the official execution. The partition performed no hidden RNG draws.
These are partition checks, not captured-artifact or no-Python deployment proof.

## Strict Capture Profile

`partition-capture-candidate-008` also closes strict export and reloaded-graph
execution for both seeds, with all four capture maximum errors zero and every
full Invocation output bitwise equal. The new public recursive effect auditor
is frozen at SHA-256
`deba514f7aad96e63d05a605daa492de13d2e11273946516444d505464eba77e`.
All 97 named nested prefix GraphModules pass; eval dropout and zero-dropout
SDPA are deterministic, and the partition contains no hidden RNG consumption.

The Adapter preserves upstream second-last DINO/SigLIP feature selection using
equivalent tuple indices, avoiding the Torch 2.10 captured-set copying failure.
Only decoder hidden states are consumed, so online prefix execution omits the
unused vocabulary-head calculation while retaining the complete reference
checkpoint/parameter audit. For fixed batch-one unpadded prefill with no prior
tokens, it invokes the same native decoder layers and final norm with the same
positions and `attention_mask=None`. This preserves the upstream eager
implicit-causal SDPA route; native SDPA and no static past state are required.
Temporary unused KV values are not persistent cross-call state. Arbitrary
padding, cached decoding and other attention backends remain unsupported.

Attempts 002 through 007 remain failed/incomplete evidence. In particular, 007
shows that exporting the full native decoder orchestration materializes a
causal mask and changes cognition values. Diagnostic mask replacement alone
then fails due to output stride; mask replacement plus a value-preserving copy
restores bitwise equality. Neither diagnostic graph is the production artifact.
The successful 008 Adapter exports normally without hand-editing FX graphs or
relaxing numerical assertions. Its context schema 2 is observed/checked only in
Python; runtime policy enforcement and other frozen context versions remain
separate evidence domains.

## Generic RNG Follow-Up

The autonomous path uses the shared `vlaforge::CudaRngProvider` rather than a
CogACT-named operator or a hidden global-generator setter. The provider owns a
LibTorch CUDA generator, exposes complete CPU byte state, supports reset,
snapshot and restore, and rejects short, wrong-dtype or wrong-shape state
without mutating the existing generator. The adapter's fixed draw schedule is
expressed as typed input ports and a typed Session state contract.

Matching upstream Torch required reproducing the actual dispatch-dependent
counter reservations and conversions, not merely using an algorithm named
Philox. The C++ provider smoke test covers seeds `42`, `43`, `0` and
`2**64-1`, three consecutive requests per seed, every intermediate state,
restore/interleaving and default-generator isolation. The packaged runner then
uses the provider for the real 1 initial plus 10 per-step draws and produces
all five complete outputs byte-exact against the frozen reference.

The formal cross-implementation comparison starts native timing before C++
RNG preparation and official timing before restoring the saved CUDA generator
state. Both timed regions include the required random work, model/DDIM
execution and a final CUDA synchronization. Static input H2D, input binding,
preprocessing and complete output conversion stay outside both regions. This
is the only supported same-boundary comparison; older native measurements that
excluded RNG preparation are retained as historical, non-comparable evidence.

## Ordinary Native Deployment

`build_real_cogact_torchscript.py` invokes the shared device-preserving
`export_torchscript_region` helper on all four immutable 008 exported programs.
The 002 campaign passes strict trace and complete archive save/load checks, then
full N10 execution for seeds 42, 43, 42 with all step/carry/output bytes exact.
Each input revision changes on every call, so repeated observation bytes do not
silently turn these calls into a queued action or cache-only benchmark.

The successful 004 campaign verifies and reuses these same archives, without
recompiling or changing their devices/precision. It calls the public compiler
and bundle builder with ordinary loop execution, generates the Session C ABI,
and validates three complete calls in one standalone C++ process. All five
outputs match the candidate reference byte-for-byte, including FP64 native
postprocessing and the final RNG receipt/count. Both linked dependencies and
live process maps are checked for Python libraries; none are loaded.

Historical IR parsing is unchanged. The new generic
`frontend.tensor_types.canonical_tensor_dtype` and `tensor_type_from_torch`
perform explicit, lazy-imported framework-to-IR type conversion. Unknown,
complex, quantized and float8 representations are rejected rather than cast.
For old 008 evidence, the deployment tool converts only actual TensorType
objects, leaves arbitrary metadata strings unchanged, and saves a complete
110-entry conversion ledger, original/canonical IR hashes and both I/O digests.
The public compiler generates a new certificate; old certificates are never
transferred to changed IR. Failed capability/memory dtype attempts 002/003 are
preserved, not relabelled as successful deployments.

Remote artifacts and bundle remain on the NAS under
`cogact/torchscript-candidate-004/session/bundle/`. Local evidence includes all
small outputs, commands, compiler/runtime metadata, stdout/exit codes, `ldd`,
process maps and source/artifact hashes. Large `.pt` files remain on the NAS.
The worker monitor establishes PID identity by a register/reset/re-register
handshake before any Torch tensor or Session exists; the reset applies only to
that child process. It refuses another target-GPU compute owner and only
terminates its own worker process group on collision. One-second monitoring
cannot prove the absence of a shorter transient external process.

No runtime numerical provider is claimed by this legacy TorchScript profile.
The strict output evidence is bounded to these actual calls. Autonomous PRNG
and the fixed-profile H20 formal comparison are now verified. Larger
observation coverage, the original gated Meta configuration, replay and board
execution remain outside this profile.
