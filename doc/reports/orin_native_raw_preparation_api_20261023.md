# Native Raw Preparation API

Date: 2026-10-23. A working C++17 static API is published under
`artifacts/orin/20261023-01/`. Assets and decoded observations are loaded
outside the preprocessing boundary; every `Prepare` performs the actual
image/state/text/token/noise transformations entirely in memory.

**32/32 observation cases and all 320 tensor instances match fresh executions
of the existing root11 CLI exactly**, including dtype, shape, bytes, and typed
alignment. **23 native finite/ownership/lifetime checks and seven audit tests
pass.** A syscall trace covers every one of the 32 actual `Prepare` calls and
finds **zero file or descriptor I/O calls** inside those boundaries.

This is API/CLI input-preparation evidence, not a target action or performance
result. No shared production file, root11 source/profile/evidence, Orin state,
or GPU/model runtime was modified or executed.

## Files And Evidence

- [Public API](../../artifacts/orin/20261023-01/native_raw_preparation.h) and
  [implementation](../../artifacts/orin/20261023-01/native_raw_preparation.cpp).
- [Runner ABI adapter](../../artifacts/orin/20261023-01/raw_runner_preparation_adapter.h).
- [Static library](../../artifacts/orin/20261023-01/build-001/libnative_raw_preparation.a)
  and [CMake target](../../artifacts/orin/20261023-01/CMakeLists.txt).
- [Standalone public-header consumer](../../artifacts/orin/20261023-01/raw_preparation_example.cpp),
  [separately linked executable](../../artifacts/orin/20261023-01/build-001/raw_preparation_example),
  [two successful direct-link consumer runs](../../artifacts/orin/20261023-01/external-consumer-001.json),
  and [two successful parent-CMake consumer runs](../../artifacts/orin/20261023-01/cmake-consumer-001.json).
- [Complete validation receipt](../../artifacts/orin/20261023-01/validation-001/validation.json),
  [all 320 comparisons](../../artifacts/orin/20261023-01/validation-001/api-results/comparison.json),
  and all actual tensor bytes under `validation-001/api-results/`.
- [Native failure checks](../../artifacts/orin/20261023-01/validation-001/failure-tests.json),
  [seven audit tests](../../artifacts/orin/20261023-01/contracts-tests-001.xml),
  and [filesystem audit](../../artifacts/orin/20261023-01/validation-001/prepare-IO-audit.json).
- [Pinned core extraction](../../artifacts/orin/20261023-01/build-001/generated/root11-extraction.json)
  and [delivery manifest](../../artifacts/orin/20261023-01/delivery-001.json).

## API Contract

Namespace: `vlaforge::native_input`.

| Method | Boundary | Behavior |
| --- | --- | --- |
| `Processor::LoadAssets(profile, expected_sha256, &processor)` | Before timing | Verify/parse the profile and tokenizer; initialize SentencePiece once |
| `processor->LoadDecodedObservation(raw_manifest, expected_sha256, &observation)` | Before timing | Load/hash decoded fixture buffers; no preprocessing |
| `processor->SnapshotDecoded(tensor_views, texts, &observation)` | Before timing | Copy caller-owned decoded RAM into an immutable owned snapshot |
| `processor->Prepare(observation, &chunk)` | Inside preprocessing timing | Execute all declared operations and materialize fresh typed tensor buffers |
| `DescribePreparedInputs(chunk, model_input_order, &bindings)` | Metadata/H2D setup | Check exact ABI and return CPU binding descriptors plus owning leases |

The production decoded-RAM path uses `SnapshotDecoded`, so file manifests are
not required for live input delivery. For the qualified profiles, inputs are
CHW U8 cameras, F32 `state_native[14]`, F32 saved `noise_saved[50,32]`, and the
original instruction string. Optional wrist images and image masks retain
root11 behavior. No compressed image decoding is introduced.

`LoadAssets` and both decoded-input entrypoints must run on the processor's
owner thread/process. `Prepare` rejects a different owner, an observation
captured by another processor, an empty observation, or a nonempty output.
Every successful call creates a distinct chunk. It does not reuse prepared
inputs, encoder features, token results, or noise from a previous chunk.

`PreparedChunk` is move-only. `TensorLease` is an owning immutable lease over
the typed output storage; its data and dimensions remain valid after the
originating chunk or processor is destroyed. Immutable leases can be read on
another thread. Caller RAM can change or be released after `SnapshotDecoded`
returns without altering the snapshot.

The runner must retain `InputBindings::keepalive` through H2D and Session
completion, especially with asynchronous transfers. CPU binding pointers are
input-only even though the existing C ABI uses `void*`. The API does not
allocate pinned host memory or device memory, launch a backend, or synchronize
a GPU. Those operations remain the runner's responsibility.

Outputs use the existing typed `HostTensorBuffer` storage, including actual
`vector<double>` and `vector<int64_t>` ownership for F64/I64. There is no
reinterpretation through an F32 carrier. The adapter returns existing
`VLAForgeDType` values and CPU `VLAForgeBoundTensor` descriptors, with stable
dimensions and explicit typed alignment. Mismatched model dtype/shape is an
error, not an implicit conversion.

## Reused Core And Timing Boundary

The build reads the immutable root11 processor source only if its complete
SHA256 is:

```text
b2f0ffafac488d90ec0afccb055212bf66f017ee386bd677bb3b562b0d5a9c20
```

It mechanically extracts the original helpers and complete operation loop
into a private namespace. Numeric, image, prompt, discretization,
SentencePiece, padding, and saved-noise-copy statements are unchanged. The
CLI entrypoint and file-publication helper are omitted. The exact extraction
is version-locked and refuses a changed source; it is not a general C++ parser.

The API reuses the existing pinned SentencePiece static provider and native
Pillow shared provider. It adds no model-name dispatch. The existing generic
profile declarations determine all operations. Only parsed declarations and
tokenizer assets persist between calls; actual transformed buffers and text
are built again inside `Prepare`.

`Prepare` includes fresh raw-buffer working copies, online Pillow resize/pad
and F32 normalization, source-ordered F64 state math, prompt construction and
tokenization, masks, exact saved-noise copying, finite checks, and publication
into owned aligned buffers. No files, subprocesses, temporary files, asset
reloads, integrity hashes, or reference reads occur in that method.

The API requires `FE_TONEAREST` and rejects a changed rounding context rather
than silently rewriting the caller's numerical environment. Compilation uses
`-fno-fast-math -ffp-contract=off`. Valid finite named-profile inputs preserve
the root11 arithmetic exactly; new guards reject invalid/nonfinite inputs or
misuse before publishing a partial result.

## Integration Example

The compiled [consumer example](../../artifacts/orin/20261023-01/raw_preparation_example.cpp)
uses only public API headers. Parent integration can follow this structure;
no production runner was edited:

```cpp
using namespace vlaforge::native_input;

// Owner thread, before timing. Read hashes and input order from pinned assets.
std::unique_ptr<Processor> processor;
check(Processor::LoadAssets(profile_path, profile_sha256, &processor));
DecodedObservation observation;
check(processor->SnapshotDecoded(decoded_views, instruction_texts, &observation));

PreparedChunk chunk;
const auto start = Clock::now();
const auto status = processor->Prepare(observation, &chunk);
const auto prepared_at = Clock::now();
check(status);

InputBindings inputs;
check(DescribePreparedInputs(chunk, model_input_order, &inputs));
// Copy inputs.cpu_tensors to existing device buffers without changing dtype.
// Bind the runner's device views, execute/wait, and retain inputs.keepalive
// until transfers and Session execution are complete.
```

`check` above accepts `native_input::Status` and should fail the observation
on a non-OK result. Its error string owns its memory. Metadata lookup/order
comes from the model contract outside timing. CPU-view descriptors must not
be mislabeled as CUDA pointers.

The prepared `state` remains F64 `[1,32]`, available for native finishing.
The adapter explicitly rejects asking for that value as F32, or for I64 tokens
as I32. Parent-owned original-Cast routing and F64 neural-time partitioning
remain separate operations; this API does not insert `_to_copy`, modify the
neural IR, alter a scheduler, or narrow model ports.

## Build And Link

From the worktree root, with existing root11/provider artifacts available:

```sh
cmake -S artifacts/orin/20261023-01 -B artifacts/orin/20261023-01/build-new -DCMAKE_BUILD_TYPE=Release
cmake --build artifacts/orin/20261023-01/build-new -j 2
```

For a parent CMake consumer, use C++17 and link the target:

```cmake
add_subdirectory(/absolute/path/to/artifacts/orin/20261023-01 native-input-build)
set_property(TARGET raw_runner PROPERTY CXX_STANDARD 17)
target_link_libraries(raw_runner PRIVATE native_raw_preparation)
```

The target supplies public API/repository include paths and its static-link
dependencies: SentencePiece, Pillow, OpenSSL Crypto, json-c, and Threads.
The public headers expose no SentencePiece or JSON implementation types.

The separately compiled public consumer was also linked successfully with:

```sh
repo=/home/zhangzimo/.codex/worktrees/f801/edge-fm-x
api="$repo/artifacts/orin/20261023-01"
pillow="$repo/artifacts/orin/20261021-07/host-image-build-002"
g++ -std=c++17 -O2 -fno-fast-math -ffp-contract=off \
  -I "$api" -I "$repo/vlaforge/include" "$api/raw_preparation_example.cpp" \
  "$api/build-001/libnative_raw_preparation.a" \
  "$repo/artifacts/orin/20261022-11/build-002/sentencepiece/src/libsentencepiece.a" \
  "$pillow/libvlaforge_pillow_image_backend.so" \
  -ljson-c -lcrypto -pthread -Wl,-rpath,"$pillow" \
  -o "$api/build-001/raw_preparation_example"
```

No new dependency installation, global configuration, Orin connection,
Torch/CUDA linkage, or other-platform model benchmark was performed.

## Verification And Preserved Gates

The validation script runs 32 fresh existing-CLI invocations using the current
root11 qualified operational profiles, loads their actual reference bytes and
all decoded observations before any `Prepare` marker, then compares every
API value, dtype, shape, byte count, alignment, and runner binding descriptor.
All 320 comparisons are byte exact. All API output bytes are retained under
the new validation directory.

The syscall audit follows the process and threads. It observes exactly 32
complete `Prepare` intervals and zero file/descriptor calls within them.
Reference comparison and evidence publication are after the end marker. No
latencies or model performance results are reported. Audit tests inject file
reads, file opens, and runtime logging inside fake intervals to prove that
the audit rejects those cases and incomplete coverage.

The 23 native checks cover pinned-asset/raw hash failures, empty/foreign raw
ownership, foreign-thread and post-fork calls, null outputs, RNE mismatch,
live-output/lease overwrite rejection, F64/I64 alignment, ABI demotion/narrowing
rejection, nonaliasing fresh chunks, owned RAM snapshots, changed saved-noise
copying, NaN/Inf raw noise, nonfinite computation without partial publication,
invalid asset parameters, nonfinite file fixtures, and lease lifetime across
chunk/processor destruction. The standalone public consumer succeeds for both
profiles after destroying the processor and chunk while retaining its leases.

For a fresh validation run, use a new output directory:

```sh
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 \
  python artifacts/orin/20261023-01/validate_native_api.py \
  --build artifacts/orin/20261023-01/build-new \
  --output artifacts/orin/20261023-01/validation-new
```

This is equivalence to the **explicit named native profile**, pinned at
`62f5664e6b63d4cc5a57adbca044e1ab26bbeaa5f2ca9e131288edda8a8311a8`.
The separate literal CPU image-rounding comparison remains failed at 224/320,
and the historical full-F64-input gate remains failed at 3/16 per model.
Native-finishing/action residuals are not reevaluated. Root11 evidence and
profiles remain unchanged. The protected artifact hash inventory is about
1.87 MB; neural weights or tens-of-GB artifact inventories were not rehashed.

Same-Orin, same-precision conventional TensorRT versus optimized, with complete
normalized/native action gates, remains parent-owned and unverified here.

## Identities

```text
native_raw_preparation.h
2c622fbbecba69d5e6907b9174fb6c32484caebad19b3609dbad60c010e41b4b
native_raw_preparation.cpp
bd26ae39a50d7b520015569635b2bba8d65659567eadfc7dcdeceea0094f7077
build-001/libnative_raw_preparation.a
91eadda59138810949b9e8330189eb7d452e3ba4f6b5f246c6d9aa5ea393fda5
unchanged root11 operation-loop snapshot
a6841907940077415529745f9dce2f7e71c0423d6e711e7851f307d8d39fa809
```
