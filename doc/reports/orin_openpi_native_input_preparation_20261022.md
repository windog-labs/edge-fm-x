# OpenPI Native Decoded Input Preparation

Date: 2026-10-22. Status: **concrete native CPU implementation delivered;
strict frozen-input exactness remains incomplete at F64 gripper state**.
All new implementation, parameter data, tests, outputs, and negative evidence
are under `artifacts/orin/20261022-11/`. Shared production files, the sequence
lowerer/tests, and all `20261022-09` sequence evidence were not changed.

The compiled C++ runner prepares all ten input tensors online from original
decoded images, native state, text, and saved noise. It contains no model-name
dispatch, neural computation, cached image/text encoder features, or random
draws. Its native dependencies are SentencePiece, the existing standalone
Pillow provider, json-c, OpenSSL, and standard CPU libraries. It does not link
Python, Torch, or CUDA.

## Real Frozen Validation

The [final comparison](../../artifacts/orin/20261022-11/validation-002/validation.json)
covers **all values, shapes, and dtypes of all 10 tensors on 16 observations
per model**: 320 tensor instances in total. There are 16 distinct recorded
frames from one episode, evaluated with two different processor profiles.
No board-handoff tensors were substituted.

| Prepared Input | pi0 Exact | pi0.5 Exact | Preserved Type/Shape |
| --- | ---: | ---: | --- |
| Three images | 48/48 | 48/48 | F32 `[1,3,224,224]` |
| Three image masks | 48/48 | 48/48 | BOOL `[1]` |
| Language tokens | 16/16 | 16/16 | I64 `[1,48]` / `[1,200]` |
| Language masks | 16/16 | 16/16 | BOOL `[1,48]` / `[1,200]` |
| Saved noise | 16/16 | 16/16 | F32 `[1,50,32]`, exact original bytes |
| State | **3/16** | **3/16** | F64 `[1,32]`, no downcast |

All 320 dtype/shape checks pass. **294/320 tensor instances are byte exact;
only 3/16 complete observations per model pass the strict exactness gate.**
The native implementation is therefore not promoted as a fully exact source
processor or target-qualified deployment.

The first failed prepared boundary is state on observation index 3, dataset
frame 59, coordinate 6. pi0 produces `-0.5839158176682175` versus frozen
`-0.5839158176682178`; pi0.5 produces `-0.43966693685010316` versus frozen
`-0.4396669368501034`. Across the full corpus, each model has 16 differing
scalar values in 13 observations, confined to gripper coordinates 6 and 13.
Worst absolute differences are `6.106226635438361e-16` for pi0 and
`4.440892098500626e-16` for pi0.5. Worst state MSEs are
`2.1281525885478956e-32` and `1.232595164407831e-32`, respectively.

The original `MSE < 1e-5` and `cosine > 0.9999` checks are also recorded and
pass, but do not replace or relax the required byte-exact gate. Every actual
tensor and first mismatch is retained in `validation-002/`.

## Source And Data Binding

The task read and bound the requested source-runtime contract, its OpenPI
preparation report, and the source-contract inventory. The upstream revision
is `215abfb217dbac7d5f1273282331b9b1866c0479`. All ten supplied upstream
hashes were verified, as were additional processor source files, normalization
assets against their conversion records, and the actual tokenizer model.

The authoritative raw index is:

```text
/home/zhangzimo/Repos/private/edge-fm-x/artifacts/edgefm-vla-goal/20260906-044509/openpi-inventory/openpi-series-001/inputs/index.json
SHA256 01a53810e3d2b78114f635c9b96fe8df582b502ce57ebd91176f7ba6b564e757
```

It matches the input-index hash in both original
`raw-host-formal/pi0/protocol.json` and `raw-host-formal/pi05/protocol.json`,
including pi0.5's historical `h100-series-inputs` origin. Each selected raw
manifest and NPZ array was hash/shape/dtype verified. The raw pack preserves
all four decoded cameras as CHW U8, the native F32 state, original prompt,
and original saved F32 noise; it performs no image preprocessing offline.

Expected values come exclusively from the original
`host-pi0-formal/prepared/data/` and `host-pi05-formal/prepared/data/` roots.
Every positional reference was checked against `frozen-files.json` and also
against the corresponding original named input in its frozen protocol.
The inventory supplies ABI metadata, not replacement board-handoff payloads.
All original source/raw/reference files were hash-rechecked after validation.

## Native Primitives And Profiles

[prepare_observation.cpp](../../artifacts/orin/20261022-11/prepare_observation.cpp)
implements a generic profile interpreter with explicit tensor declarations,
SHA256-verified raw buffers/assets, and the following CPU primitives:

- Ordered F64 scalar programs: loads, parameter vectors/scalars, addition,
  subtraction, multiplication, division, square, clamp, and an explicitly
  named `system-libm-f64` asin provider. Contraction and fast math are disabled.
- Native Pillow bilinear resize to the aspect-fit dimensions, then black
  centering/padding, CHW publication, and declared F32 normalization. Optional
  missing wrists produce black images with false masks; the base is required.
- Declared ASCII source prompt cleaning and text templates, with optional
  F64 state discretization using declared increasing bins and no clipping.
- Pinned native SentencePiece encoding, declared BOS insertion, separate
  text segments, I64 publication, truncation, padding, and boolean masks.
- F64 zero padding and exact byte-copy/reshape of saved noise.

Model-specific semantics live in
[pi0-profile.json](../../artifacts/orin/20261022-11/raw-pack-002/pi0-profile.json)
and [pi05-profile.json](../../artifacts/orin/20261022-11/raw-pack-002/pi05-profile.json).
The shared C++ logic does not examine either model name.

The state declarations reproduce the actual order: F32 native state is
multiplied by the integer joint-flip mask into F64, coordinates 6/13 undergo
the original gripper geometry with a clamped asin argument, then F64 z-score
or quantile normalization is applied. Epsilon remains `1e-6`. pi0.5 digitizes
the **14 normalized, unpadded F64 values** with
`digitize(state, linspace(-1,1,257)[:-1]) - 1`, without clipping underflow bins.
Only afterward is state padded to 32 dimensions.

The original prompt is `pen_uncap`, read online from each raw observation.
pi0 cleans it and encodes the newline separately, with maximum length 48.
pi0.5 declares `Task: {cleaned}, State: {discrete state};\nAction: ` with
maximum length 200. Tokenization is the actual SentencePiece algorithm,
not a handwritten approximation or a lookup of frozen tokens.

Tokenizer SHA256 is
`8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`.
SentencePiece v0.2.0 source was built task-locally at upstream commit
`17d7580d6407802f85855d2cc9190634e2c95624`, with archive SHA256
`7710982d3b438e790646f8be623f3bbe6af1e2f2619129d1f946d4070d26bc34`.
No package installation or global environment modification was performed.
The upstream provider is [SentencePiece](https://github.com/google/sentencepiece/tree/v0.2.0).

## Residual And Negative Evidence

The [source CPU oracle](../../artifacts/orin/20261022-11/source-oracle-002.json)
executes the pinned source state functions/normalization methods and actual
PaligemmaTokenizer class with existing NumPy 1.26.4 and SentencePiece 0.2.0,
without importing/constructing the neural model. Its state bytes agree with
the native implementation on all 16 observations per model, but also match
the original frozen state on only 3/16. Its source Pillow-resized U8 pixels
match all 48 reference images per model, and its text matches all frozen IDs
and masks.

Thus the F64 discrepancy is present when rerunning the actual source CPU
functions on this host, not introduced by F32 normalization or an omitted
source transform. A scalar/vector transcendental-provider difference is a
possible explanation, but the frozen runs do not retain intermediate state
values or the original CPU ufunc/libm dispatch identity, so the first
internal arithmetic divergence is not proven. The explicit residual is the
F64 gripper preparation boundary, not neural-time computation.

The current CPU exposes AVX2 and no AVX512. The NumPy v1.26.4 SVML submodule
was identified at commit `1b21e453f6b1ba6a6aca392b1d810d9d41576123` and one
AVX512 asin source file was retained for investigation; it is **not compiled
or used**. No higher-precision approximation, observation-specific correction,
or dtype demotion was inserted to force the frozen bytes to match.

The original source supports Unicode prompts. This task's profile explicitly
bounds prompt cleaning to ASCII, which covers all 16 original raw prompts;
non-ASCII input is rejected before publication and covered by a test. It is
an additional concrete unsupported boundary for future inputs.

The first retained attempt (`attempt-001/`, `build-001/`, `raw-pack-001/`,
`validation-001/`) followed the separate shared/JAX resize helper and failed
image exactness. Tracing the actual `ResizeImages` import established that
it calls `openpi_client.image_tools`, which uses Pillow instead. The final
implementation reuses the existing C++ Pillow provider and preserves the
failed first attempt without relabeling it as a pass.

## Tests And Reuse

**17 focused native tests pass**, recorded in
[native-tests-001.xml](../../artifacts/orin/20261022-11/native-tests-001.xml).
They cover both port profiles, F64/I64 preservation, online optional camera
masks, changed saved-noise copying, tampered data, incompatible dtype,
unexpected/missing cameras, prompt cleaning, Unicode rejection, 14-value
unclipped discretization, truncation, no overwrites, unsupported math-provider
rejection, and contained nonsymlink raw paths. The original lowerer/native
buffer suite was not modified or rerun by this task.

Reusable build and invocation, from the worktree root:

```sh
cmake -S artifacts/orin/20261022-11 -B artifacts/orin/20261022-11/build-new -DCMAKE_BUILD_TYPE=Release
cmake --build artifacts/orin/20261022-11/build-new --target prepare_observation -j 2
artifacts/orin/20261022-11/build-new/prepare_observation \
  artifacts/orin/20261022-11/raw-pack-002/pi05-profile.json \
  artifacts/orin/20261022-11/raw-pack-002/sample-00/raw.json \
  artifacts/orin/20261022-11/prepared-new
```

For a fresh full comparison, replace the final output with a new validation
directory and run:

```sh
env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  python artifacts/orin/20261022-11/validate_preparation.py \
  --binary artifacts/orin/20261022-11/build-new/prepare_observation \
  --pack artifacts/orin/20261022-11/raw-pack-002 \
  --output artifacts/orin/20261022-11/validation-new
```

The comparison returns exit code **2** while the exact F64 residual remains;
all individual outputs and mismatches are retained. The CLI file transport is
only the correctness harness; no file-I/O or CPU-model latency benchmark is
claimed. Parent-owned F64 neural partitioning, precision, target integration,
and formal campaign tables remain separate work.

## Hashes

```text
prepare_observation.cpp
b2f0ffafac488d90ec0afccb055212bf66f017ee386bd677bb3b562b0d5a9c20
build-002/prepare_observation
8f2a31fa3981a62b4d1bb0bcd47f4743da059eae5e0c820fbc6aeb818f2b6c4d
pi0 profile
d2b15737bbe9fedea9602502bcda743b06ed6a57ea4d3c101bca3ebc7b618e2d
pi0.5 profile
185e0d923c2ce804df4b84335f2ef32971b8c1c7fabd3d00e6c3eb9d097b97f9
reused standalone Pillow provider
4b23587cd23a9dd3f37b6b3299ba0d9c0b417f7643ee88112c1fdb4ca48d05f9
```

[delivery-001.json](../../artifacts/orin/20261022-11/delivery-001.json) pins the
implementation, parameter profiles, test and validation reports, source/oracle
evidence, dependencies, this report, and prior sequence-delivery protection
checks. No Orin access, GPU/model inference, installation, other-platform
performance experiment, or production-file modification was performed.
