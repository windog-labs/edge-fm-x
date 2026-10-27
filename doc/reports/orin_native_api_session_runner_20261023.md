# Native API Session Runner

Status: **host software-tested, source/link integration ready; real Session and target execution pending**. Writes are confined to `artifacts/orin/20261023-09/` and this report. Root01/02/05, shared production, workers07/08, board files and existing evidence were not edited. No SSH, Orin/GPU execution, neural inference, install or model performance experiment was performed by this worker.

## Ready Delivery

Use [driver-source-package-002.tar.gz](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/orin/20261023-09/driver-source-package-002.tar.gz), 72603 bytes, SHA256 `a03f0043fc162ef515a362894c7e096aee499908ce26304ea4c7671adf1adc32`. Top directory is `driver/`. It includes only source, headers, generated public-port data and manifests; no x86 executables/libraries, raw-asset duplication, neural weights or synthetic Session providers. Package001 remains immutable as the earlier strict-only development checkpoint and must not be used for the new human-policy pilot.

The authoritative target sources are `generated-target-policy-004/pi0-profile/` and `generated-target-policy-004/pi05-profile/`. [generate_runner.py](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/orin/20261023-09/generate_runner.py) uses the real public ModuleIO serializer/digest, explicit prepared-name maps, complete output roles, static tensor dtype/shape/layout/device/alignment and original sixteen raw-observation pins. Model names do not dispatch behavior. F64 state and I64 tokens remain exact typed storage, not F32/I32 carriers. Public input/output IDs and names drive binding/readback rather than positional or semantic-role guesses.

The new template is derived from the existing frozen `raw_observation_runner.cpp.in`. [native_api_session_runner.cpp.in](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/orin/20261023-09/native_api_session_runner.cpp.in) consumes the actual root01 `Processor`, `SnapshotDecoded`, `Prepare`, `PreparedChunk`, `TensorLease` and `InputBindings` API. It calls the generated native C++ Session through the stable ABI2 C facade, exactly as existing raw runners do. One actual DSO/bundle and arbitrary caller-selected mode run per isolated worker process. The driver implements no neural operation, solver approximation, alternate checkpoint, model-name branch or FP32-neural fallback. All ten solver steps belong to the real generated Session's `Run`; the driver does not substitute a shorter host loop.

## Request Clock

Before timing, load/verify profile, tokenizer and decoded raw assets; snapshot decoded RAM through `SnapshotDecoded`; load all complete reference bytes; initialize the Session/bundle/engines, transport streams and stable device/output storage. Only decoded raw arrays/text are cached, never preprocessed features or inputs.

Each request's single `steady_clock` interval includes fresh request/binding allocation, **fresh Prepare** (all image resize/normalization, prompt/tokenization, F64 state processing/padding, masks and exact saved-noise copy), `DescribePreparedInputs`, complete H2D, binding with a fresh monotonic revision, the entire Session run, correct-device completion, every typed output read and complete D2H to CPU. No RNG draw replaces saved noise. Legacy completion uses explicit device-wide synchronization, not default-stream ordering; private nonblocking transfer streams also complete explicitly. Full normalized F32 `[1,50,32]` and native F64 `[1,50,14]` chunks are transferred, never selected actions/dimensions.

Timing fields are differences from timestamps in that **same** request and their sum equals its full interval. There is no stage-average reconstruction. Hashing, quality metrics, reference comparisons, files/logging, process maps and actual replay getter queries are outside the interval. Warmup/measure counts are explicit; measured requests must cover at least sixteen and cycle all sixteen observations.

Prepared leases, CPU/device descriptors and buffers survive through proven complete CPU outputs. Per-request ownership is released only afterward. Backend/CUDA/status/metadata failures with unproven lifetime exit 86 without unwinding/freeing borrowed storage or unloading DSOs. Backend resources remain resident to process exit even on success. Native commits are not falsely described as hardware-atomic. Successful completion restores the original caller device; the new target branch has not been executed here.

## Launch-Fixed Quality

Before any real pilot, parent explicitly selects one immutable policy. Default **`strict-bitexact`** is for software/regression: every output must be finite and byte-exact. Explicit **`--quality-policy frozen-normalized-v1`** applies the fixed human criteria to the **complete normalized F32 chunk**: **MSE < 1e-5 AND cosine > 0.9999**. All outputs must be finite. Native F64 errors are supplemental and its human gate is **finite only**. `bitwise_exact` remains independently reported, including true flags when the user-mode outputs are exact. No threshold override or per-step/slice acceptance is available. Cosine is 1 when both norms are zero, 0 when only one is zero.

The policy is serialized/hash-pinned before requests in `quality-policy.json`, repeated in metadata, every request and final status, and included in frozen-reference schema `/2`. A different policy against that reference rejects. Policy/version alignment occurs now, **before** actual pilot launch, not after a failure. No old SmolVLA/RDT gate or failed result is changed or relabeled. This is a new prelaunch OpenPI driver contract, not a claim that the full formal campaign protocol has been frozen or passed.

Every output logs full count, bytes/hash, finiteness, bitwise equality, MSE, cosine, RMSE, mean absolute and max-absolute error outside timing. Rows separately log the complete normalized gate, native-finite flag, all-output finiteness, all-output bitwise equality, policy hash and selected-policy acceptance. Known completed human-policy quality failures collect **all raw observations**, retain false verdicts/full bytes, write `quality-policy-rejected`, and exit 12. Strict regression fails fast. Neither failed quality run is promoted to a formal timing/CDF pass; CUDA/backend failure still exits immediately.

[audit_complete_outputs.py](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/orin/20261023-09/audit_complete_outputs.py) independently checks pinned complete bytes/public ModuleIO/policy and computes full errors plus all 50 step and every dimension diagnostics. These slices are diagnostic only, never substitute gates. [freeze_reference.py](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/orin/20261023-09/freeze_reference.py) accepts only successful complete baseline capture, all sixteen measured observations and repeat consistency. It pins CUDA UUID, ModuleIO, processor/raw identities and the launch policy. Another device, CPU/mock origin, changed policy or incomplete/failed capture rejects. Same-Orin baseline agreement is not original-source fidelity proof.

## Worker Interfaces

[integration-audit-quality-002.json](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/orin/20261023-09/integration-audit-quality-002.json) checks Locke/root08's actual `driver-handoff-001.json`, latest pinned requests and **entire** ModuleIO, not digest alone. Root08 explicitly owns Session generation and supplies ABI2 `vlaforge_model_session_api`, `vlaforge_model_session_create_from_bundle`, and actual replay-task expectations. Its native `primary-action` role is serialization in the legacy resident runner, **not** root09's MSE selector; root09 gates the explicit normalized public-output ID. Ordinary compilation with no replay has no fabricated counters. Shared/replay variants query actual `vlaforge_model_session_get_replay_info` and validate task/state/captured steps/invocation counters from caller data, never mode-label assumptions or ten-times step counts.

Full schema digests remain pi0 `8751fa709441f09883da726d2cc2d6b52f7a5894bf5876ee36b6803e2dff9575` and pi05 `2cca61bd2132c82d053daba6ea951fa0e1f2d0125dd66db8bbb13a3fece2bd74`. Root08's current real Session DSO/bundle paths remain null pending complete actual engine artifacts and inspection. Root09 does not fabricate them or attempt TensorRT inference.

Erdos/root07's tested [link-manifest-native-002.json](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/orin/20261023-07/result-003/link-manifest-native-002.json) is frozen unchanged, SHA256 `35f403d1654acc8008c8db0aa38b5d47b8208f3007f76e0d4d1afde8a8def073`. [build_from_native_manifest.py](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/orin/20261023-09/build_from_native_manifest.py) and CMake use its existing native API/SentencePiece/Pillow paths, owner memory guard, `json-c/crypto/pthread/atomic/m` dependencies and explicit RPATH. No x86 library copying or native provider rebuild is needed. Compiler scratch is task-local and build mode hides GPUs. Root07's separately reported 320/320 named-profile tensor equality does not erase historical original full-F64/literal-pixel failures or establish neural action gates.

## Host Evidence

`software-tests-quality-005/software-test-receipt.json` records actual unchanged root01 preprocessing on both sets of sixteen raw observations with **explicit NONMODEL byte-copy C++ Session subclasses**, not neural stand-ins presented as model runs. The CPU contracts deliberately reverse input/output declaration order to test named maps and roles. Three arbitrary modes complete **102 strict compared requests / 204 complete chunks**, all byte-exact with MSE/RMSE/meanabs/maxabs 0 and cosine 1. Strace observes **102 complete request intervals and zero file/descriptor I/O inside them**. Complete CPU outputs, actual getter counters, timestamp accounting, warmup and sixteen-sample cycling are checked.

Additional software tests prove exact outputs remain flagged exact in human mode; finite native differences are supplemental; strict still rejects those bytes; normalized failures reject and preserve all sixteen observations; and a reference-policy switch rejects. Full offline step/dimension audits are retained. Ten explicit Session failure probes cover failed run, wrong output dtype/rank/extent and wrong replay counters across both profiles. These tests have no CUDA/TensorRT/Torch/Python dependency in their runner processes and are not model performance experiments.

`contract-tests-quality-003/receipt.json`: **19 tests, zero failures/errors**. Separate C++ quality tests pass **17 checks**, including both-zero/one-zero cosine, signed-zero byte distinction, MSE-only/cosine-only failure, full-tail detection, NaN rejection, exact strict threshold boundaries and native finite-only human acceptance. Earlier failed sparse-ID fixture construction and split audit-marker parsing attempts remain recorded, not removed. Contract tests also cover bad schema/name map/F64 role, changed DSO/reference device, incomplete capture, incorrect measure count, output overwrite, wrong mode and file-I/O/incomplete trace rejection.

Both real target-profile branches compile/link on the host against existing CUDA12.8 with GPUs hidden, under `build-target-policy-004/`; **neither executable was run**. This is not native CUDA11.4/aarch64 compilation or target-runtime evidence. Root07 native libraries are only referenced for parent builds.

| File | SHA256 |
| --- | --- |
| Runner template | `1812e97ab9be58b4ee69dd38edc74eb5332456a8b1d33059c60c98a4fb86ab2e` |
| Generator | `0c78f7c98277066e4ef3e6e47d0c244464b9283fc0a7eb78748324868faf94a3` |
| Quality core | `e5f3b93ad09f2bdd2c58b3dcda3d2a869b151de81480c2ccd01a09cbe8854b74` |
| Software receipt | `f1be3940f24469413402737fe45965235cb0231395cda25900dee89b3b00415e` |
| Contract receipt | `c7404e0b40cf42f09e7c4826040338a54f54acf2d14b93a67d7d17bc6e1f018c` |
| Integration audit | `4a31fcfe4d104de78e97649ebbf79939b702a15ae17c0d683ab6e32e27d85f50` |

## Exact Next Step

[PARENT_COMMANDS.md](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/orin/20261023-09/PARENT_COMMANDS.md) contains the complete native build/CMake arguments, mode manifest schema, explicit parent scheduling, baseline/reference freeze and offline audit commands. From the staged package, this is **CPU compilation only**:

```sh
R=/absolute/staged/20261023-09/driver
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python3 "$R/build_from_native_manifest.py" \
  --manifest "$R/frozen/link-manifest-native-002.json" \
  --manifest-sha 35f403d1654acc8008c8db0aa38b5d47b8208f3007f76e0d4d1afde8a8def073 \
  --generated "$R/generated-target-policy-004/pi0-profile" \
  --output "$R/native-build-001/pi0-profile" --cuda-root /usr/local/cuda-11.4
```

Parent separately supplies real root08 Session DSO/bundle identities and actual mode/replay manifests, freezes the launch quality policy, and schedules GPU use only after its current engine job exits. Use `--quality-policy frozen-normalized-v1` consistently for the new human-policy baseline and candidate. Driver source is ready; real model/Orin execution, complete-source fidelity and formal E2E/CDF acceptance remain parent-owned and pending.
