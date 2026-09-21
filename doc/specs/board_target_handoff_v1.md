# Board Data Handoff and Target Driver v1

This is the D1/D2 preparation interface, not a claim that a board backend exists.
It never promotes workstation inference, cross-compilation, engine inspection or
offline data validation to Orin/BPU execution. The new public modules contain no
model-name dispatch. New models supply an existing Module, a benchmark protocol
v1/v2 or data-only protocol, and provenance. New boards supply an explicit target
descriptor and driver.

## Data Contract

`vlaforge/tools/board_handoff.py prepare` consumes an existing frozen Session
protocol, its exact Module, and `vlaforge.board_source_selection/1`. It reuses
`validation/session_benchmark.py` for complete Tensor sizes, endian storage,
floating/integer roles, multi-output ABI order and metrics. FP64 is explicit;
integer references never pass through floating-point conversion.

For a verified input/reference series that has not completed the formal timing
schedule, use `vlaforge.board_data_protocol/1`. It has exactly five fields:

| Field | Contract |
| --- | --- |
| `schema` | `vlaforge.board_data_protocol/1` |
| `boundary` | Explicit, nonempty input/output boundary description |
| `checkpoint_sha256` | Lowercase 64-digit SHA256 of the selected source checkpoint |
| `outputs` | All Module ports in ABI order, using the existing `name`/`role`/optional `active_dimensions` declarations |
| `samples` | Nonempty ordered list; each item contains exactly `sample_id`, named `inputs`, and named `outputs` |

Sample identities are unique nonempty strings or nonnegative integers, never
booleans. Every output carries both `direct` and `eager` NPY paths. Input paths
refer to complete raw tensor bytes. One output is `primary-action`; other ports
are `float` or `exact`. Integer/RNG receipts retain their exact dtype and bytes.
All actual source samples can be delivered without subsampling to a timing
schedule. This schema rejects benchmark fields such as `warmup`, `measured`,
`processes`, `quality_gate`, `bundles`, and board-acceptance flags. It cannot be
passed to the Session benchmark runner as a completed or runnable benchmark.

Both source protocol forms share the same portable pack, validator and target
dispatcher. Source CUDA tensor metadata describes the reference, not the future
board device placement. This version still requires an existing static contiguous
CUDA output contract; it does not add a BPU provider or claim arbitrary input
backends. Data integrity and source reference comparisons are separate from
fresh target numerical and performance acceptance.

The create-only destination contains:

- `source/{protocol,module,selection}.json`: original source records, including
  absolute historical paths as provenance only.
- `samples/NNNN/input-NNN.bin`: every complete input, preserving saved noise.
- `samples/NNNN/output-NNN-{eager,direct}.{npy,bin}`: every complete output,
  original NPY and strictly typed little-endian raw storage. No active-dimension
  selection removes stored values.
- `provenance/NNNN/*`: explicitly selected files, checked against their original
  SHA256 before and after copying.
- `targets/{orin-cuda,horizon-bpu}.json`: pending descriptor templates, not drivers.
- `manifest.json`: published only after the entire copied pack validates. It
  lists each file's relative path, exact bytes and SHA256, source identity and
  limitations. A failed preparation can leave an incomplete directory but cannot
  publish this completion marker; retry into a new directory.

Validation is independent of the original absolute paths. It rejects unknown
manifest fields/schema, path traversal, symlink files/directories/manifest,
duplicate paths, undeclared files, truncation/hash mismatch, incomplete samples,
input/output reordering, nonfinite or invalid boolean storage, NPY/raw dtype or
full-shape mismatch, and any offline `board_executed`, `board_verified`,
`physical_units_verified` or `robot_calibration_verified` value other than the
JSON boolean `false`. Pin the manifest SHA256 independently when transferring;
hashes provide integrity relative to that pin, not a digital signature or proof
that arbitrary user-supplied provenance is authoritative.

The pack includes neither checkpoint weights nor x86 executables/engines. A
historical CUDA `bundles` path in `source/protocol.json` is **not** a board build
input. The target driver must produce a new target-native bundle/protocol with
new hashes and current numerical policy. Source policy is not silently upgraded.

## Target Descriptor

`vlaforge.board_target/1` has exactly these fields:

| Field | Contract |
| --- | --- |
| `family` | `orin-cuda` or `horizon-bpu` |
| `architecture` | `aarch64`; CPU ISA alone does not prove board identity |
| `accelerator` | `cuda` or `bpu`, matching the family |
| `provider_contract` | `cuda-region-session` or `bpu-region-session` |
| `sku` | Exact device SKU, initially `null` |
| `sdk_versions` | Explicit nonempty version mapping, initially `null` |
| `memory_bytes` | Positive physical accelerator/shared-memory capacity, initially `null` |
| `provider` | `{contract,path,sha256}` for the installed provider, initially `null` |
| `driver` | `{contract,path,sha256}`, contract `vlaforge.board_target_driver/1`, initially `null` |
| `device_access` | Explicit device nodes, initially `null` |
| `execution_partition` | Stage-to-device mapping including every CPU fallback, initially `null` |
| `clock_power_thermal_policy` | Explicit measurement policy, initially `null` |

Unknown SDK/SKU/provider information stays `null` and blocks all four stages.
Boolean-as-integer capacity, empty mappings, unknown fields, wrong family/ISA or
provider/driver contract are rejected. Complete descriptors alone do not prove
driver functionality. An explicitly installed SHA-bound driver is required.
Keep a filled descriptor **outside** the immutable input pack; do not rewrite
pending templates and their manifest to pretend they were captured earlier.

## Driver Boundary

The dispatch wrappers invoke an installed executable, without a shell:

```text
DRIVER preflight|build|run|collect --pack ABS_PACK --target-descriptor ABS_JSON --output ABS_NEW_DIR
```

`preflight` and `run` require AArch64 on the calling host and read/write access
to declared device nodes before dispatch. The driver must additionally verify
real board identity, SDK and compiler/runtime ABI against the descriptor. A
cross-build may run on x86, but that is only build evidence. Installed provider
and driver file SHA256 are verified before launch. Dispatch records argv, PID,
host ISA, start/end time, return code and a complete combined driver log.

The common dispatcher does **not** implement a CUDA/BPU driver. Its result is
`pending`, `failed`, or `driver-returned`, never board acceptance. Even a zero
driver exit is unadjudicated; the dispatch report keeps both board flags false.
This v1 interface is not an automatic board-result attestation system.

Required responsibilities of an actual driver:

1. Preflight: exact SKU, board/accelerator identity, OS/kernel, memory, SDK/OE/
   compiler/runtime versions, device access, current owners, capacity and thermal
   state. Refuse conflicts and terminate only its own process if ownership
   changes. BPU host/accelerator PID namespaces must be evidenced, not guessed.
2. Build: local-source snapshot hashes including dirty/untracked implementation,
   dependency lock, target compiler argv/version, checkpoint identity, module
   and each new artifact hash, native ELF ISA/loader dependencies, provider and
   exact stage partition, SDK legality, typed I/O and numerical policy. Never
   reuse x86 CUDA code or label a cross-build as board execution.
3. Run: no-Python model process, actual provider mappings/loader and process
   identity evidence, complete input/reference hashes, every generation step and
   transaction, all complete output bytes, explicit dtype/layout/device transfer,
   same-precision strict and paper gates separately. Preserve scheduler, CFG,
   timestep, seed and authoritative saved noise. Native scale is not robot
   calibration. Declared CPU fallback remains visible per stage and in total.
   Do not infer inverse normalization from recovered input statistics, action
   dimension count or the word "finish". The Smol16 source exports normalized
   active-six actions only; its corrected v2 handoff and historical v1 erratum
   are recorded in `doc/reports/board_handoff_preparation_20260907.md`. A new
   denormalization output Region needs separate capture/native output evidence.
4. Collect: raw per-call samples, balanced 16-sample ordering, 128 warmups and at
   least 1024 measured calls in five processes for each applicable policy, CDF
   and throughput, complete MSE/max-abs/cosine and bitwise counts, missing/failed
   calls, temperature/frequency/power time series and steady-state policy. CUDA
   completion and BPU completion need provider-specific timing. An unsupported
   replay policy is pending/unsupported, not a fake successful ordinary run.

## Existing Tools to Reuse

- Orin's current substrate is `scripts/orin/{build_backend,cross_compile_bundle,
  run_bundle_on_orin}.sh` plus TensorRT RegionExecutable. The old scripts are
  TensorRT-10/SM87-specific. Their JetPack image default is a historical build
  configuration, **not** a verified SDK on the user's forthcoming board.
- `docker/orin/README.md` documents linear typed Tensor I/O and no silent dtype,
  shape, layout, device or alignment conversion. It is not a real-model engine.
- `tools/benchmark_session.py` and `tools/session_benchmark_runner.cpp.in` are
  unchanged. Once an Orin CUDA driver supplies an actual target build, a fresh
  board protocol may reuse their typed output and full-call verification.
  Current CUDA resident-tensor timing is not automatically valid for a BPU.
- `tools/report_deployment_metrics.py` already accepts paired complete actions
  and raw latency CSV. Its metric output does not attest the source hardware;
  pair it with actual driver evidence and independently checked identities.

There is currently no implemented BPU Session provider or validated board driver
in this handoff. No SDK is downloaded, no engine fabricated and no board run is
claimed. Exact Orin SKU/JetPack/CUDA/TensorRT and BPU SKU/OE/compiler/runtime,
memory budget, missing provider coverage, transfer behavior, thermal protocol and
real device access must be resolved before actual D1/D2 acceptance.
