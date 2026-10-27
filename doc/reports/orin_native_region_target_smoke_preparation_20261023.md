# Native Region Target Smoke Preparation, 2026-10-23

Status: **prepared, not target-run**. New files are confined to `artifacts/orin/20261023-02/` and this report. No SSH, Orin access, GPU execution, neural inference, performance run, installation, shared-production edit, or change to prior06/10/14/15/16 evidence was performed. Parent retains all target scheduling and ModelIR/Plan/TRT integration ownership. This smoke does not parse or execute the pi0 engine implicated in the parent's parser failure.

## Frozen Contracts

Preparation read root16 requests and all provenance input pins, checked source/semantic/physical-plan identities against its delivery, recomputed the **full public ModuleIO** digest using the existing serializer, and cross-checked physical artifact IDs. It did not substitute a region-only fixture schema. Requests, provenance, full modules, physical plans, and full I/O schemas are frozen under the new root.

| Dataset | Full ModuleIO SHA-256 |
| --- | --- |
| pi0 | `8751fa709441f09883da726d2cc2d6b52f7a5894bf5876ee36b6803e2dff9575` |
| pi05 | `2cca61bd2132c82d053daba6ea951fa0e1f2d0125dd66db8bbb13a3fece2bd74` |

| Profile | Physical Region ID / Name | Exact Signature | Borrowed CUDA Workspace |
| --- | --- | --- | --- |
| pi0-r0 | 0 / checked_input_openpi_step_0 | F64[1,32] -> F32[1,32], RNE | 384 B, alignment 256 |
| pi0-r1 | 1 / checked_input_vf_cached_openpi_prefix_59_6 | I64[1,48] -> I32[1,48], checked | 448 B, alignment 256 |
| pi0-r3 | 3 / openpi_native_output | native signature below | 0 B |
| pi05-r0 | 0 / checked_input_vf_cached_openpi_prefix_59_6 | I64[1,200] -> I32[1,200], checked | 1056 B, alignment 256 |
| pi05-r2 | 2 / openpi_native_output | native signature below | 0 B |

All public ports are contiguous CUDA tensors on `cuda:0`, ABI2, target `sm_87`. Native inputs are **complete** state F64[1,32], normalized actions F32[1,50,32], incoming accepted Tensor BOOL[1]; outputs are **native F64[1,50,14], computed accepted Tensor BOOL[1]**, in that order. No scalar outputs, no normalized output from this provider. The normalized public output remains the source loop's direct branch. Frozen `source-control.json` files retain native invoke -> `vla.validate` -> atomic output group/transaction commit. Ten solver iterations remain unchanged.

Native variant: `shared-plugin/numeric-cuda-transport-native-accepted/1`.
Cast variants: `shared-plugin/checked-cast-cuda-sync-f64-f32-rne/1` and `shared-plugin/checked-cast-cuda-sync-i64-i32-checked/1`.

## Implementation

[profiles.json](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/orin/20261023-02/profiles.json) and each `profiles/<profile_id>/build-profile.json` are generated data, not model-name branches. The numeric descriptor, hex coefficient headers, numeric core, and checked-cast14 sources are frozen byte-for-byte. The new copy of the native15 wrapper has exactly one metadata-only change: create requires the compiled physical `region_id`. All compiled configuration is hidden; each production DSO exports only `vlaforge_region_executable_value_api`.

[build.py](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/orin/20261023-02/build.py) compiles five real provider DSOs, CPU-only linker-guard DSOs, a frozen **real external ABI2 loader**, CPU metadata harness, and one GPU harness from [smoke.cpp](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/orin/20261023-02/smoke.cpp). Build output must be a new task-root directory. Logs, argv, return codes, build-time source hashes, DSO hashes, and `CppArtifactRegionDefinition` registrations are emitted. Build requires `CUDA_VISIBLE_DEVICES=` and never runs its outputs. Native provider CPU arithmetic is strict FP: `-fno-fast-math -ffp-contract=off -frounding-math`; FE_TONEAREST is checked, not set by the provider. CUDA sources use sm_87 without fast-math/FTZ flags. No cross-architecture binary portability is claimed.

The native wrapper is the same necessary **non-neural** CPU finishing path as native15. Actual core/source acceptance is checked before pending state; `accepted` derives from canonical incoming true AND complete successful finite numeric finishing, not a compatibility placeholder. Run publishes neither output. Synchronize rechecks borrowed metadata, acceptance, FENV, and actual finishing; native H2D completes before computed true is even enqueued. Cast run similarly leaves public output unchanged until synchronize.

Before any D2H read, native establishes correct-device `cudaDeviceSynchronize`, covering nonblocking shared/replay producers; it repeats that barrier at acceptance recheck. No default-stream ordering assumption is used. Native per-call traffic is D2H 6657 + 1 B, H2D 5600 + 1 B, two device-wide barriers and four private-stream completion waits. Its pinned CPU staging is preallocated at load: 12264 B. Include all transfers, synchronization, and CPU finishing in E2E. All neural work stays TensorRT. No zero-copy, FP32-neural fallback, context sidecar, or capture capability is claimed. Native belongs **after** the bounded solver loop, outside captured replay.

Known invalid, numeric, metadata, or FENV rejection before publication leaves both public outputs unchanged. A CUDA fault during the multi-buffer commit is **not hardware-atomic**; status is poisoned/unproven and native may already have changed. The harness pins DSOs, executable instances, descriptors, CUDA allocations/workspaces, streams, events, and pinned host buffers to process exit, even on success. Any backend/CUDA fault or unproven assertion uses `_Exit(86)` without teardown, unloading, or device changes. Later non-backend errors never clear sticky taint. Successful completion checks provider current-device preservation and explicitly restores the caller's original device before process exit. No actual multi-device or hardware-fault behavior has been verified here.

## Evidence

Final accepted host build: [build-host-final-004/receipt.json](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/orin/20261023-02/build-host-final-004/receipt.json), x86_64 GCC 11.4, CUDA 12.8.61, 17 successful compile/link commands. Additional CUDA 11.8.89 compile-only build is retained under `build-host-cuda118-003/`. **CUDA 11.4 is not installed locally; native CUDA11.4/aarch64 build and all GPU branches remain pending.** The native build script uses C++17/CUDA runtime APIs present in the intended older SDK, permits its SDK root, and avoids Python APIs newer than Python3.8 on its build path; this is preparation, not an 11.4 compilation claim.

[tests-host-final-004/receipt.json](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/orin/20261023-02/tests-host-final-004/receipt.json): **14 CPU tests, 0 failures/errors**. The real loader's CPU-metadata/core path passed **91837 checks**; a separate fail-closed CUDA linker fault path passed **59 checks** across all five providers. The former requires zero CUDA calls, and the latter blocks, counts, and never forwards its five deliberate failing calls. Test-only numeric entry symbols exist only in guard DSOs, not production DSOs. CPU mode explicitly refuses GPU mode.

All **128 complete paired source chunks / 89600 F64 native values** are bit exact, MSE=0, maxabs=0, with all 14 dimensions and 50 steps compared. [fixtures.json](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/orin/20261023-02/fixtures.json) preserves both models x board_handoff/raw_host x 16 observations x direct/eager, using each row's original state, normalized actions, and native reference. These are not 128 independent observations; pi05 board/raw inputs must not be mixed. Full per-case CPU results are `tests-host-final-004/cpu-source128.jsonl`. No target fidelity pass is inferred.

The compiled but **unrun** GPU harness checks a pending delayed nonblocking producer before calling ABI2 `run`, requires producer completion afterward, checks unchanged public buffers before synchronize, compares all source128 chunks, and validates computed accepted=true. It tests full normalized-tail NaN/Inf, used state Inf, false/noncanonical acceptance, invalid dtype, changed dimensions, changed FENV, and delayed false/noncanonical acceptance before commit. Cast tests include RNE midpoint/subnormal/signed-zero/finite-upper-bound success and nonfinite/F32-overflow/I32-overflow rejection, with poisoned lifecycle and no publication. CUDA failure takes the retention path, not a rollback claim.

Registration tests reject wrong full schema, physical ID, native/cast signature, DSO hash, and unknown profile group. Production export-table checks reject leaked configuration symbols. Original and frozen source/reference hashes were rechecked. `tests-host-002/driver-failure.json` preserves a receipt-writer relative-path error after 13 successful tests; its outputs were retained, and the final resolved-path driver/test run above replaces it.

Key SHA-256 pins:

| File | SHA-256 |
| --- | --- |
| profiles.json | `b032222aea9e81667dd4a5013a068c1e5b3f3d70b6c69a4cb50b13b7533cb83a` |
| fixtures.json | `1ad577bd4fae2f45774f5f4c239a160d305f6cb57df38095869bd4324da7eb68` |
| smoke.cpp | `d585e6ecc706c1d3d93e5d2eaf71f4430af95f4090ea5a06961b937812415028` |
| build.py | `c3409814792b3ee510d1834d0bd360b783056c65a7d35482d38dc7ae242e366f` |
| parent_registration.py | `16fbd65368380ed2d3b09e77edeb9fe75ebc1ba4779544978af3d24d8ae8222e` |
| final CPU test receipt | `918e8a61f8442528e3d3fdfb167f193d597964b84e5e1d83e9740aa52517e06b` |
| final host build receipt | `abfbc39be386585e374c9c88b58ee5bb9f25f5e054b55f2d5477ec84e452f63c` |
| final host GPU harness | `ca298e2e840308ff4432f782854310692393b896e0f43dba9ea34745c6d4a16d` |
| final host pi0-r0 provider | `02aa132a52b173cdef9753def902cd69786f4da79cec98dd3134641989d7777b` |
| final host pi0-r1 provider | `c74a27da0152df9ba4e30466dd2a4a7556aa4fa2f44f4d8aa78efa0bec6d916c` |
| final host pi0-r3 provider | `9f3e4134d1fc4268ec27330ca5938926e1b88343672ad395ff07cf48071a9ab6` |
| final host pi05-r0 provider | `48654dc627ddd4d3e162a1b3cd0a37c9b801c0bdc451ea187e175cafa3e4bf7e` |
| final host pi05-r2 provider | `0309e7f26169b87a6b5ef462455df976864286a902012417f68dfa603a59a5a9` |

## Parent Commands

Stage this task root, including `frozen/`, `profiles/`, `fixtures/`, and JSON manifests, preserving relative paths. Do **not** run `prepare.py` on target: it is the host-only source-lineage freeze step. Do **not** deploy the x86 DSOs/harness as Orin binaries. Existing CUDA headers/compiler, C++17 compiler, Python3, and OpenSSL development headers/libcrypto are prerequisites; missing dependencies are blockers, not authorization to install.

After parent schedules native compilation, from its staged task root:

```sh
TASK=/absolute/staged/artifacts/orin/20261023-02
CUDA_ROOT=/usr/local/cuda-11.4 sh "$TASK/build_native.sh" "$TASK/build-native-cuda114-001"
```

This emits real native DSOs and updated hashes/registrations; it does not run a GPU. Preserve a failed build directory and use a new output name for retries. After separate parent GPU authorization:

```sh
sh "$TASK/run_parent_smoke.sh" --parent-scheduled "$TASK/build-native-cuda114-001" > "$TASK/target-smoke-001.jsonl" 2> "$TASK/target-smoke-001.stderr"
```

Exit 0 requires all five profiles and source128 comparisons. Exit 86 means tainted/unproven, potentially partial commit, process-lifetime retention. Preserve stdout/stderr and native build receipt; never convert it into an accepted run by dropping failed cases. All target results are currently absent.

On the parent compiler host, [parent_registration.py](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/orin/20261023-02/parent_registration.py) exposes `registrations(build_root, dataset, semantic_path, physical_plan_path)` returning a real `Mapping[str, CppArtifactRegionDefinition]`, suitable for the existing `generate_cpp_session(..., artifact_regions=...)` argument. It checks the full ModuleIO, exact current physical IDs/signatures, and actual DSO hashes. Select the explicit profile group; no model-name code branch is used. Merge these bindings with parent's TRT/other bindings, keep artifact paths relative to the build/bundle root, provision the declared CUDA workspaces, and retain original validate/atomic-commit structure. Do not overwrite a parent binding silently. Final host examples are `parent-registration-pi0-final-004.json` and `parent-registration-pi05-final-004.json`; target rebuild registrations must replace their x86 hashes.

The delivery manifest records source/binary/test hashes and **prepared-not-target-run** status. Parent must still review the adapter, compile with native11.4, execute the smoke, and integrate/verify the full model Plan and E2E. Export/host compilation is not target support or campaign completion.
