# Orin Native Raw Preparation API Build

## Result

CPU-only native aarch64 build and validation passed. The unchanged root01 API
matches both a freshly rebuilt, unchanged root11 native CLI and the frozen host
`system-libm-f64` named-profile outputs on **32 observations / 320 full tensors**.
Every dtype, shape, byte, and typed alignment passed, including all F64 state
and I64 token values. No ARM libm residual was observed in these fixtures.
This is not a general cross-platform libm guarantee.

The historical original full-F64 gate remains failed, with only 3/16 exact
observations per model. The old literal official CPU pixel gate also remains
failed. Their receipts and native finishing residuals were not changed or
requalified. The new comparison is explicitly to the existing named profile,
not a relabeling of literal Torch CPU pixel division or the old frozen states.

No CUDA context, GPU execution, neural inference, model build, or performance
experiment was performed. Same-Orin same-precision TensorRT comparisons and
full normalized/native action gates remain parent-owned and unverified here.

## Accepted Artifacts

Local root: `artifacts/orin/20261023-07/`.

Exclusive target root: `/data/datalog/edgefm-orin/20261023-07/raw-api-native-001/`.

- `package-004.tar.gz`: relocatable frozen source/assets/reference package.
- Target `relocated-003/package/`: accepted package extraction.
- Target `result-003/build/`: working aarch64 libraries, CLI, API validator,
  invariant tests, and public-header consumer.
- `result-003/validation.json`: complete accepted CPU checks.
- `result-003/host-named-profile-comparison.json`: all 320 host/native records.
- `result-003/link-manifest-native-002.json`: verified explicit native link
  command and two direct-linked public-consumer runs. Its library list
  supersedes the initial generic `link-manifest.json` list by adding `atomic`.
- `result-003-handoff.json`: hashes for all 1,045 target result files, compiler
  flags, and memory admission observations.
- `result-003-evidence.tar.gz`: local copy of receipts, libraries, executables,
  compiler flags and cache. Actual prepared tensor dumps remain on target;
  each is sealed in the handoff and compared bytewise before publication.

| Artifact | SHA256 |
| --- | --- |
| Source/assets archive | `e2313a622d6579b6b966952966880df2664ba958030bcf6d0d86111eb8e3c7f8` |
| Package manifest | `5998a5742e976ba01430194dbb8973dc5626ffad7827ed529d334f52b707e3af` |
| Native API static library | `a580fa9f4b27cf183f6f5b648f928cd1354c4a996f1f2449841c4501a925553d` |
| Native Pillow provider | `d915444e09bdfef2292b56684d02a7d7c1942e0e6511fd4fecf8ed4041d272cb` |
| Native SentencePiece static library | `ef48843187fab6e6450f71e4bf5ad7a5a93a934c2183dca5f23d709f4a1d3d3a` |
| Target handoff | `ace96252145c2f5f2059b51fed68588bf20ba221cb7cc87bde88bda573388a14` |
| Evidence archive | `447b92b61430f53578bd6a39b05a723d2d6cd7dc051e95c2b6f73e333d3b169f` |

## Source And Environment

The package has 841 pinned files, 232,662,427 bytes. It includes original
raw-host decoded observations, saved noise, prompt/text declarations, the
SentencePiece model, profiles, and comparison-only host outputs. It contains
no neural weights or cached encoder features. Reference tensors are loaded
outside Prepare and never passed to it.

`frozen/` preserves repository-relative source paths. Root01 API implementation,
public headers, adapter, extractor, validators and root11 CLI are unchanged.
The original extractor runs against the staged original source, not a rewritten
or host-absolute dependency. Original source SHA256 is
`b2f0ffafac488d90ec0afccb055212bf66f017ee386bd677bb3b562b0d5a9c20`;
generated core SHA256 remains
`966dfac33140af857a19fd15c52d2c44bed213046ad54c683f9a84f72f5e5ddc`.
The new top-level CMake wrapper only supplies native provider builds and paths.
There is no model-name dispatch or arithmetic correction.

Providers were rebuilt from already available source: Pillow 10.4.0 standalone
C imaging core plus the existing C++ bridge; SentencePiece v0.2.0, source
`17d7580d6407802f85855d2cc9190634e2c95624`. Pillow archive SHA256 is
`166c1cd4d24309b30d61f79f4a9114b7b2313d7450912277855ff5dfd7cd4a06`;
tokenizer model SHA256 is
`8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`.
Existing Python headers are compile-time-only. Native executables and provider
do not link Python, Torch, TensorRT, or CUDA. No installation was needed.

Target: aarch64, kernel `5.10.120-mp-tegra`, GCC 9.4.0, glibc 2.31, json-c
0.13.1, OpenSSL 1.1.1f, CMake 4.3.1. The unchanged upstream CMake source needs
the wrapper's `CMAKE_POLICY_VERSION_MINIMUM=3.5` on this CMake version.
`environment.json` pins every loaded CPU library, including libm, libc and
libstdc++. Numeric core uses `-fno-fast-math -ffp-contract=off` and requires RNE;
F64 is not demoted. OMP/OpenBLAS thread limits are 1; the actual Prepare audit
observed one task. Pillow retains its existing serialized memory-arena access.

## Checks

| Check | Result |
| --- | --- |
| Fresh native CLI / API, all 32 cases | 320/320 full tensors exact |
| Frozen host named profile / native API | 320/320 full tensors exact |
| All full F64 states / host named profile | 32/32 byte-exact |
| Native finite/ownership/alignment/ABI failure checks | 23 passed |
| Public-header consumers, CMake and direct static link | Both profiles passed in each |
| Complete Prepare syscall boundaries | 32 |
| File/descriptor I/O inside Prepare | 0 |
| GPU device opens in traced API process | 0 |
| Tracer clean/open/read/write/thread-read/incomplete probes | 6 passed |

`strace` is absent on the board. The task-local Linux ptrace auditor launches
only its own child, follows clone/fork events, and observes syscall entries at
the unchanged validator markers. It never attaches to existing processes or
signals them. Any syscall other than a narrow memory/time/synchronization
allowlist fails inside Prepare. Actual Prepare entries were only `getpid`
(32) and `brk` (9). Worker-thread reads, runtime logging, opens, and incomplete
boundaries were separately shown to fail the auditor.

Assets and decoded snapshots load outside the timer. Prepare reruns image
resize/normalization, state conversion/normalization, prompt construction,
SentencePiece tokenization, masks, padding and exact saved-noise copy. It
publishes fresh owning typed storage without temporary files or new RNG draws.
The invariant suite checks NaN/Inf rejection, failure atomicity, thread/PID and
raw-owner rejection, correct RNE handling, live-output protection, and leases
surviving processor/chunk destruction. F64 state remains available for native
finishing, and the adapter rejects F64/I64 narrowing.

Compilation used at most two jobs. Every new compiler task waits below 6 GiB
available, leaving the parent's 2 GiB reserve plus headroom. The lowest recorded
accepted compile admission was 35,076,200 KiB (about 33.45 GiB); no memory pause
was required. Accepted build scratch is inside `result-003/compiler-tmp/`,
which was empty after completion. No global settings or services were changed.

## Reuse

Existing ready-to-link build:

```sh
BASE=/data/datalog/edgefm-orin/20261023-07/raw-api-native-001
P=$BASE/relocated-003/package
B=$BASE/result-003/build
```

Parent CMake integration uses the new package root, not frozen root01's original
host-path CMake file:

```cmake
add_subdirectory(/data/datalog/edgefm-orin/20261023-07/raw-api-native-001/relocated-003/package raw-api-build)
set_property(TARGET raw_observation_runner PROPERTY CXX_STANDARD 17)
target_link_libraries(raw_observation_runner PRIVATE native_raw_preparation)
```

The exact tested direct command is in `link-manifest-native-002.json`. Its
essential link interface is:

```sh
g++ -std=c++17 -O2 \
  -I"$P/frozen/artifacts/orin/20261023-01" -I"$P/frozen/vlaforge/include" \
  parent_driver.cpp "$B/libnative_raw_preparation.a" \
  "$B/sentencepiece/src/libsentencepiece.a" -L"$B" \
  -lvlaforge_pillow_image_backend -ljson-c -lcrypto -pthread -latomic -lm \
  -Wl,-rpath,"$B" -o parent_driver
```

Load assets and snapshot decoded RAM on the owner thread outside timing.
Inside preprocessing timing call only `Processor::Prepare` into an empty
`PreparedChunk`. Afterward bind exact contract dtype/shape/order with
`DescribePreparedInputs`. Retain `InputBindings::keepalive` through async H2D
and Session completion. The consumer can include only the public API/adapter
and runtime ABI headers; SentencePiece/json-c internals stay private.

For a new full CPU-only build/validation under the allowed target root:

```sh
env PYTHONDONTWRITEBYTECODE=1 python3 "$P/run_native.py" --output "$BASE/result-new"
```

The script creates scoped compiler scratch, configures native CMake, builds
with `--parallel 2` and the memory launcher, rebuilds/runs all 32 fresh CLI
cases, audits API calls and checks the complete tensor/ownership suite. It
requires a new output directory and never overwrites earlier receipts.

## Retained Attempts And Limits

Incomplete local `package-001/` stopped on an internal SentencePiece test
symlink. Later packages materialize only repository-contained symlinks and
record them. `package-002` / target `result-001` retained the CMake 4 policy
failure. `package-003` / `result-002` built successfully and passed all 320
native tensors and 23 checks, then stopped on a harness receipt-name collision.
Its compiler used default transient `/tmp` scratch; persistent artifacts stayed
in scope. Final `package-004` / `result-003` fixes the receipt label and explicitly
scopes compiler scratch. The first target extraction preserved future host
timestamps against a 2023 board clock; subsequent fresh extractions used
`tar -m --keep-old-files`. No clock correction or old-file overwrite occurred.

Root01/root11, old receipts, shared production code, original raw-host exports,
and the parent's board jobs were not edited. No target model fidelity, action
gate, runtime-loop integration, latency, power, or throughput claim follows
from this CPU preparation result.
