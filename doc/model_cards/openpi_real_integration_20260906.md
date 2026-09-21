# OpenPI Shared Real-Model Integration

## Evidence Boundary

This is the source integration record for the pi0 and pi0.5 shared Adapter.
Both official checkpoints completed download, independent remote verification,
strict CPU conversion, and actual recorded-input 10-step CPU reference/partition
parity. Pi0 additionally completed real export persistence and independent
serialized-only full-action replay, establishing CPU L2 evidence. Neither model
has L3/L4 compiled or no-Python deployment evidence here.
The existing pi0 L1 fixture is unchanged.

The implemented boundary is official prepared tensors to the complete normalized
action chunk. The Python reference keeps the official raw input transforms and
native-action-scale output transforms. Their no-Python equivalents, pi0.5 real
checkpoint capture, compiled artifacts, generated Session parity, and performance
measurements remain pending. Static schema tests and export of mask/timestep utility functions
are not real-model capture evidence.

## Pinned Source

- Repository: <https://github.com/Physical-Intelligence/openpi>
- Revision: `215abfb217dbac7d5f1273282331b9b1866c0479`
- Local checkout: `/home/zhangzimo/Repos/private/edge-fm-x/third_party/openpi`
- Source checkout was obtained with Git, without large checkpoints or submodule
  datasets. It is ignored by the parent repository and must be synchronized
  separately when preparing a remote environment.
- Source gate checks the exact HEAD and rejects tracked modifications. Runtime
  imports reject a foreign already-imported OpenPI package and verify the hashes
  of the required transformers replacement files.
- Pinned upstream dependencies include Python >= 3.11, torch 2.7.1,
  transformers 4.53.2, JAX 0.5.3, flax 0.10.2, and orbax-checkpoint 0.11.13.
  The source gate enforces transformers and its patch; record actual torch/JAX
  versions in every real audit. A different compiler environment needs its own
  parity evidence, not an assumed compatibility claim.

Use an isolated environment with `UV_LINK_MODE=copy` before installing and
applying upstream transformers patches. The upstream README warns that patching
hardlinked package files can affect the shared uv cache and unrelated projects.
Do not apply these patches to the SmolVLA or user environments.

## Frozen Weight Inventory

The official GCS JSON API was read on 2026-09-06. The initial inventory gate
downloaded metadata only. The subsequently authorized pi0 preparation and pi05
download runs are recorded below.

| Checkpoint | Objects | Total Bytes | Metadata SHA256 |
|---|---:|---:|---|
| pi0_base | 33 | 12014440489 | `b74f88c47a0d2830b9ad2d813e046fd46c54dd3324b3a20be947f9896df686c3` |
| pi05_base | 29 | 12441749581 | `093ae411a8d9c883ba3778c9b657cde352210da647c400caef4f004a784f363a` |

Metadata files, including object names, sizes, generations, MD5, and CRC32C:

```text
/home/zhangzimo/Repos/private/edge-fm-x/artifacts/edgefm-vla-goal/20260906-044509/openpi-inventory/pi0_base_gcs_metadata.json
/home/zhangzimo/Repos/private/edge-fm-x/artifacts/edgefm-vla-goal/20260906-044509/openpi-inventory/pi05_base_gcs_metadata.json
```

Original API URLs:

```text
https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/pi0_base/&fields=items(name,size,generation,md5Hash,crc32c),nextPageToken
https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/pi05_base/&fields=items(name,size,generation,md5Hash,crc32c),nextPageToken
```

Neither response contains a next-page token. The combined source download is
24,456,190,070 bytes (about 24.46 GB decimal), plus tokenizer/environment packages.
Reserve additional disk for both converted safetensors and exports; the converted
pi0 file is 7,002,873,776 bytes. Peak CPU RAM during conversion was not measured.

Six pi05 data objects are official GCS composite objects with CRC32C but no MD5.
The expanded frozen inventory adds `componentCount`:

```text
pi05_base_gcs_metadata_composite.json
SHA256: 71b250f943af3fc7e751a453845ae1d203413d9dea1be4d53a3600aed7ab245d
https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/pi05_base/&fields=items(name,size,generation,componentCount,md5Hash,crc32c),nextPageToken
```

Every previously captured field, including generation, size, MD5 when present,
and CRC32C, matches the original inventory exactly. Only `componentCount` was
added; the object set and pagination are unchanged. The download gate requires
a positive integer `componentCount` and absent `md5Hash` for the composite path,
and validates its CRC32C using `google-crc32c`. Ordinary objects require MD5 and
also verify CRC32C when present. Every network response must identify the exact
pinned generation. All objects additionally receive a locally computed SHA256;
that local hash is not represented as a provider-supplied cryptographic digest.
Reports explicitly record `checksum_type`, `md5_verified`, and
`crc32c_verified`. Composite objects must never be reported as MD5-verified.
This follows the [official GCS composite-object integrity contract](https://docs.cloud.google.com/storage/docs/composite-objects).

Generation-pinned, checksum-gated download for an isolated destination:

```bash
PYTHONPATH=<repo>/vlaforge/python <openpi-python> \
  -m vlaforge.adapters.openpi.openpi_download \
  --inventory-path <run>/openpi-inventory/pi0_base_gcs_metadata.json \
  --destination <weights-root>/pi0_base --checkpoint-name pi0_base \
  --endpoint storage-download.googleapis.com --log-file <run>/pi0-download.jsonl
```

If the bucket changes during download, the frozen inventory gate fails. Fetch
the recorded generation explicitly or record a new inventory as a new dataset
version; do not silently update hashes to accept changed checkpoint content.
HTTP HEAD requests for the guessed `pi0_base_pytorch`, `pi05_base_pytorch`,
`pi0_pytorch_base`, and `pi05_pytorch_base` safetensors paths returned 404. The
verified download path is the original JAX checkpoint plus audited conversion,
not an invented ready-made PyTorch URL.

## CPU Preparation And Offline Assets

An independent CPU-only environment was built at:

```text
/home/zhangzimo/.venvs/edgefm-openpi-cpu-py311-20260906
```

It uses CPython 3.11.16, torch 2.7.1+cpu, torchvision 0.22.1+cpu, JAX/JAXlib
0.5.3, and transformers 4.53.2 with the pinned official replacement files.
No shared transformers installation was modified. Core dependency versions follow
the upstream lock; transitive versions are frozen from the actual isolated
installation, not represented as a byte-identical upstream training environment.
Official pi0/pi05 config import and the complete upstream conversion module import
passed. The probe reported `cuda_available=false` and only `TFRT_CPU_0`.

Evidence lives under the inventory directory:

- `openpi-cpu-requirements.txt`: selected upstream inference/conversion versions.
- `cpu-torch-pip.log`, `openpi-dependencies-pip.log`: actual installation logs.
- `openpi-cpu-freeze.txt`, `openpi-cpu-pip-check.txt`: full installed lock and a
  passing dependency consistency check.
- `openpi-cpu-environment.json`, `openpi-cpu-converter-import.txt`: source/config,
  required patch hashes, CPU-only probes, and converter dependency import.
- `openpi-cpu-wheelhouse.sha256`, `openpi-cpu-wheelhouse-pip.log`: all locked
  binary wheels obtained for offline installation. Local bundle is 435 MiB at
  `/home/zhangzimo/.cache/edgefm-vla-goal/openpi/cpu-wheelhouse`; check target
  CPython 3.11 and glibc compatibility before installing these x86_64 wheels.
- `openpi-cpu-tests.xml`: focused tests; the real action-model test remains
  explicitly skipped until a converted checkpoint and saved inputs exist.

The pi0 source download writes only to
`/home/zhangzimo/.cache/edgefm-vla-goal/openpi/pi0_base`, outside artifacts.
Initial original-domain proxy TLS failures and slow direct-route attempts are
retained as failed/interrupted logs. The working official alternate GCS endpoint
run was PID `1085542`, logged as `pi0-download-gcs-alternate.jsonl`; it resumed
owned partial bytes and reverified every completed file. It exited successfully
at 2026-09-06 06:31:26 UTC after all 33 objects passed size, MD5, and local SHA256
checks. `pi0-download-verified.json` archives its complete report. An independent
remote check produced the byte-identical
`h100-pi0-independent-verification.json` before conversion. These original pi0
reports predate the composite-checksum extension and do not claim an additional
CRC32C verification.

The authorized pi05 run started at 2026-09-06 06:47:57 UTC, PID `1181245`, using
the expanded inventory above and destination
`/home/zhangzimo/.cache/edgefm-vla-goal/openpi/pi05_base`. Its progress log is
`pi05-download-gcs-alternate.jsonl`. That run failed after eight proxy TLS EOF
errors at 07:14:29 UTC, preserving its owned partial at 591,396,864 bytes. The
original failure is not a completed checkpoint. A 1 MiB pinned-generation range
probe verified the official JSON media route, then the same source downloader
resumed at that exact offset with `--endpoint storage.googleapis.com --api json`.
The resumed run started at 07:23:12 UTC, PID `1199463`, with log
`pi05-download-gcs-json-media.jsonl`. It exited 0 at 07:34:50 UTC with all 29
objects and 12,441,749,581 bytes verified. All six composite objects passed
CRC32C with `md5_verified=false`; the other 23 passed their ordinary MD5/CRC32C
gates. Every object has a local SHA256. The complete report is archived as
`pi05-download-verified.json`. Remote transfer and independent remote verification
subsequently completed. `pi05-local-remote-verification-equivalence.txt` confirms
that the local and H100 verification reports contain identical JSON data.
The strict CPU conversion below also completed; this is not a JAX numerical
equivalence claim.
JSON media and XML routes share all ownership, generation, size, and checksum
gates. The transport suite passed 23 tests, including ordinary/composite partial
resume on both routes. An active PID or partial-file progress is not completeness evidence.
Require the final `complete` event and `pi05_base/vlaforge_download.json` before
transfer acceptance or conversion.

The separate official PaliGemma tokenizer is now downloaded and checked:

```text
Source: gs://big_vision/paligemma_tokenizer.model
Generation: 1711547605575873
Bytes: 4264023
MD5 (base64): FCCtyYVnIKVZ6KhyhLGV4g==
SHA256: 8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6
OPENPI_DATA_HOME=/home/zhangzimo/.cache/edgefm-vla-goal/openpi/processor-cache
```

Original metadata API:
`https://storage.googleapis.com/storage/v1/b/big_vision/o/paligemma_tokenizer.model?fields=name,size,generation,md5Hash,crc32c`

`paligemma_tokenizer_gcs_metadata.json`, `paligemma-tokenizer-download.json`, and
`paligemma-tokenizer-offline-probe.json` bind its source, local bytes, and actual
official tokenization results. The tokenizer probe checks both formats and that
pi05 token IDs change with state. This is processor evidence, not action-model
inference. Download/reuse through the source-controlled module:

```bash
PYTHONPATH=<repo>/vlaforge/python <openpi-python> \
  -m vlaforge.adapters.openpi.openpi_assets \
  --inventory-path <run>/openpi-inventory/paligemma_tokenizer_gcs_metadata.json \
  --cache-root <processor-cache> --endpoint storage-download.googleapis.com
```

Set `OPENPI_DATA_HOME=<processor-cache>` for every real OpenPI process. The
Adapter checks the fixed tokenizer's size, MD5, and SHA256 before construction;
missing/changed bytes fail offline. Normalization factories are pointed only at
the verified local conversion assets, preventing the upstream profile from
silently fetching its original normalization URI. No global cache or proxy
configuration was modified.

Do not run full checkpoint conversion on the 31 GiB local machine while the
SmolVLA compiler is active. The conversion gate is CPU-only but may hold multiple
copies of the real model. Use the isolated H100/H20 CPU environment with fresh
memory preflight, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, and
`CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu`.

H100 preparation uses the isolated root
`/mnt/data/LingXiTeam/workspace/zzm/repos/edgefm-vla-goal-20260906/`:

```text
envs/openpi-cpu-py311/                 CPU-only venv
envs/openpi-python311-runtime/        standalone CPython runtime
wheelhouse-openpi-cpu/                SHA256-checked offline wheels
source-openpi-cpu/                    rsynced local VLAForge Python source
assets/openpi/upstream/openpi/        exact official Git source
assets/openpi/processor-cache/        fixed tokenizer asset
assets/openpi/weights/pi0_base/        all 33 source objects independently verified
assets/openpi/weights/pi0_base_strict_pytorch/  strict converted real weights and assets
runs/openpi-cpu-20260906/             remote logs and inputs
```

The hardware preflight found approximately 1.8 TiB available RAM, 50 TiB free
space on the target mount, and glibc 2.35. Every synchronization first ran a
dry run and omitted `--delete`. Weight transfer excludes owned partial files and
uses `--ignore-existing`; a later hash gate must reject any mismatched existing
remote file instead of overwriting it.

H100 installed only the checked offline wheelhouse with `--no-index`. Its full
`pip freeze --all` matches the local lock exactly. `h100-openpi-cpu-environment.json`
records a passing official source/patch/tokenizer/config/converter probe with no
CUDA device. An initial Git ownership failure from preserving the local UID with
`rsync -a` is retained in `h100-openpi-cpu-tests-initial-ownership-failure.log`.
Only this new upstream checkout was resynchronized with `--chown=0:0`, matching
the remote login UID; no global `safe.directory` exception or weaker source gate
was introduced. The post-fix focused suite is archived separately.

Reproduce the environment probe from local source on either host:

```bash
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENPI_DATA_HOME=<processor-cache> PYTHONPATH=<repo>/vlaforge/python \
  <openpi-python> -m vlaforge.adapters.openpi.openpi_probe \
  --source-root <official-source> --require-cpu
```

The H100 root also contains `envs/openpi-cu128-py311/`, separately built from
the checked CPU wheelhouse plus the root task's existing `wheelhouse-py311/`.
`openpi-cu128-requirements.txt` preserves the CPU freeze except for torch
2.10.0+cu128 and torchvision 0.25.0+cu128. The actual freeze additionally records
their CUDA bindings, NVIDIA libraries, and Triton 3.6.0. `pip check`, official
source/patch/tokenizer/config/converter import, and 40 focused CPU-side tests
passed; the real-model test is still skipped. The probe used
`CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu` and performed no GPU work.

Evidence files are `h100-openpi-cu128-environment.json`,
`h100-openpi-cu128-freeze.txt`, `h100-openpi-cu128-pip-check.txt`, and
`h100-openpi-cu128-cpu-tests.stdout.log`. This environment readiness does not
validate either torch 2.7 CPU versus torch 2.10 CUDA model outputs or the compiled
deployment path. Those are separate pending real-model comparisons.

## Strict Conversion

Use the new executable module, not the unchecked upstream CLI directly:

```bash
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH=<repo>/vlaforge/python \
  <openpi-python> -m vlaforge.adapters.openpi.openpi_checkpoint \
  --source-root <repo>/third_party/openpi \
  --checkpoint-dir <weights-root>/pi0_base \
  --inventory-path <run>/openpi-inventory/pi0_base_gcs_metadata.json \
  --checkpoint-name pi0_base --config-name pi0_aloha \
  --output-dir <weights-root>/pi0_base_strict_pytorch
```

For pi0.5, replace `pi0_base` with `pi05_base`, select `--config-name pi05_aloha`,
and use a separate output directory. These are base-model ALOHA profiles with the
official 50-step horizon, 32-dimensional padded action space, and `trossen`
normalization assets. They are not renamed task-finetuned checkpoints. A DROID
or LIBERO profile must explicitly change the config/assets and measured horizon.

The wrapper reuses the pinned upstream tensor-layout conversion functions. It
adds the following mandatory gates:

1. Verify every object from the frozen GCS inventory by size and its explicit
   provider checksum type before Orbax reads any checkpoint. Ordinary objects
   require MD5; composite objects require the metadata-gated CRC32C path described
   above. Record each locally computed SHA256 and the verified checksum flags.
2. Instantiate the real official architecture, convert every projection and
   prefix/expert tensor, and reject missing executable parameters, unexpected
   parameters, mismatched shapes, or conflicting tied aliases.
3. Resolve only genuine parameter aliases. The official
   `paligemma_with_expert.gemma_expert.lm_head.weight` is never called by the
   action path, which uses `gemma_expert.model` and `action_out_proj` instead.
   If missing from the JAX conversion, explicitly zero this unused tensor and
   record its element count. No executed parameter can use this exception or a
   random initializer. This storage is not counted as evidence of trained action
   computation, and residency/parameter reports must preserve the distinction.
4. Copy `<checkpoint>/assets` and hash all copied normalization files. The
   pinned upstream script instead uses `<checkpoint>.parent/assets` and can
   silently miss the actual assets.
5. Write `vlaforge_conversion.json` only after weights and assets exist. It binds
   source, input inventory, actual local checkpoint hashes, model config,
   strict-loading result, converted file SHA256, and parameter count.

An existing conversion output directory is refused; a failed output is not a
resume/completion marker. Real JAX-vs-PyTorch numerical equivalence still requires
the actual weights and saved observations. A complete conversion report proves
conversion integrity, not inference correctness.

The current conversion follows the upstream default: restore source values as
float32 and store the converted model as bfloat16. This storage cast is explicitly
recorded as `source_restore_precision`, `converted_storage_precision`, and
`casting_policy`; `jax_pytorch_numerical_equivalence` remains `not-assessed` until
the independent comparison runs. Do not mix that cross-framework/cast difference
with same-converted-weight prefix/step/backend optimization error. The same-
precision optimized path and later low-bit deployment must keep separate
acceptance tables.

### Completed Pi0 CPU Conversion

The H100 CPU-only conversion, PID `195079`, exited 0. It used torch 2.7.1+cpu,
JAX CPU, `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, and
`nice -n 10`. Fresh preflight found approximately 1.7 TiB available host memory;
the cgroup v1 limit was 1,473,173,782,528 bytes with 296,362,721,280 bytes in use.
No GPU was used.

```text
Output: <H100-root>/assets/openpi/weights/pi0_base_strict_pytorch/model.safetensors
Bytes: 7002873776
SHA256: 07e8a2ef8438e3cf839bc0992b2f2e07a95dac6049b156696bd47bbe79cb1518
Model parameters: 3501372176
Explicitly zeroed unused expert-head elements: 263323648
Strict missing required parameters: []
Unexpected parameters: []
```

The strict loader resolved the genuinely tied PaliGemma language head from its
input embedding. The zeroed expert head is excluded from trained executable
parameter claims. The report records the upstream float32-to-bfloat16 storage
cast and leaves JAX/PyTorch numerical equivalence `not-assessed`.

Evidence: `pi0-strict-conversion-report.json`,
`h100-pi0-strict-conversion.stdout.log`,
`h100-pi0-strict-conversion.stderr.log`, and
`h100-pi0-strict-conversion-process.txt`. An initial command failed before model
execution because `/usr/bin/time` is unavailable on the target; that failure is
retained separately as `h100-pi0-strict-conversion-initial-launch.stderr.log`.
The successful command did not use that utility. Conversion integrity is now
verified; full action inference, actual prefix/step parity, capture, and no-Python
deployment were not verified by that conversion step. The subsequent actual
CPU action audit is recorded separately below.

The subsequent actual CPU Policy-load probe, PID `222062`, also exited 0. It
strictly loaded the converted model, verified normalization assets and the
offline tokenizer, and built the official processing Policy. The report
`h100-pi0-policy-load.json` records 88.10 seconds, Linux maximum RSS 15,077,656 KiB,
4,335,264 float32 elements and 3,497,036,912 bfloat16 elements, all on CPU. The
official model configuration is horizon 50 and padded action dimension 32.
`h100-pi0-policy-load.stderr.log` is empty. This is real-checkpoint Policy-load
evidence only: no action forward call, capture, or deployment occurred.

```bash
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENPI_DATA_HOME=<processor-cache> PYTHONPATH=<repo>/vlaforge/python \
  <openpi-python> -m vlaforge.adapters.openpi.openpi_load_probe \
  --source-root <official-source> \
  --checkpoint-dir <weights-root>/pi0_base_strict_pytorch \
  --checkpoint-sha256 07e8a2ef8438e3cf839bc0992b2f2e07a95dac6049b156696bd47bbe79cb1518 \
  --config-name pi0_aloha
```

## Shared Adapter Contract

Implementation files:

```text
vlaforge/python/vlaforge/adapters/openpi/openpi_checkpoint.py
vlaforge/python/vlaforge/adapters/openpi/openpi_download.py
vlaforge/python/vlaforge/adapters/openpi/openpi_assets.py
vlaforge/python/vlaforge/adapters/openpi/openpi_probe.py
vlaforge/python/vlaforge/adapters/openpi/openpi_load_probe.py
vlaforge/python/vlaforge/adapters/openpi/openpi_inputs.py
vlaforge/python/vlaforge/adapters/openpi/openpi_reference.py
vlaforge/python/vlaforge/adapters/openpi/openpi_capture.py
vlaforge/python/vlaforge/adapters/openpi/openpi_frontend.py
vlaforge/tests/models/test_openpi_frontend.py
vlaforge/tests/models/test_openpi_download.py
vlaforge/tests/models/test_openpi_assets.py
vlaforge/tests/models/test_openpi_inputs.py
```

`OpenPIConfig` pins source directory, converted checkpoint SHA256, named official
robot/profile config, device, and positive integer `num_steps`.
`load_openpi` requires the strict conversion report, rechecks weights/assets,
loads safetensors with a strict all-parameter gate, and uses the official
normalization/processor composition. Missing or changed required data fails
before model construction. It does not accept the unrelated LeRobot checkpoint
schema as though it were OpenPI.

`prepare_openpi_inputs` preserves official model preprocessing and pi0.5 state
tokenization. The supplied `noise` must be the saved float32 tensor, not a seed.
`build_openpi_frontend` builds the following pure tensor stages and checks the
complete N-step result against official `sample_actions` before returning:

```text
prepared images + image masks + tokens + token mask [+ pi05 state dependency]
  -> prefix padding + flattened K/V tensors
saved noise + initial scalar time=1
  -> N iterations carrying (sample, time)
  -> complete normalized action chunk + finite-output validation
```

The prefix is computed once per fresh conditioning value. Its cache includes
all image/mask/token dependencies and also the state when state is tokenized.
pi0 still supplies continuous proprioception to each step. pi0.5 injects timestep
through the official AdaRMS path. Each step preserves the official Euler formula,
float32 dt/time accumulation, immutable prefix K/V, and full padded chunk. No
core opcode, C++ model-name switch, or second runtime is introduced.

The input boundary already contains model-transformed tensors. Consumers that
reuse explicit input revisions remain responsible for honest identity stamps;
the raw-I/O preparation path must rerun when observation or proprioception
changes. Reusing pi0.5 tokens while changing state is not valid raw preprocessing.

`capture_openpi_frontend` returns each real `capture_region` outcome. Export
failure is preserved as unsupported/failed; no region is replaced by eager code.
The source's config mutations are performed during loading, outside the pure
prefix and step functions. Actual ALOHA preprocessing produces float64 state;
the Adapter preserves it using the existing IR `f64` type. The official
`embed_suffix` performs its own float32 cast before the state projection. No
early state cast or normalization reordering was introduced to satisfy capture.

## Real ALOHA Input And CPU Action Audit

The pinned OpenPI ALOHA README identifies the official
[ALOHA pen-uncap dataset](https://huggingface.co/datasets/physical-intelligence/aloha_pen_uncap_diverse/tree/e82d8b40b8ac66c0b40273dd80a077dfc40b732e).
Only episode 0 and its small `info.json`/`tasks.jsonl` files were acquired:

```text
Dataset revision: e82d8b40b8ac66c0b40273dd80a077dfc40b732e
Episode bytes: 576308575
Episode LFS SHA256: f0cc336aadee7ee21225581998f80253b597721a3adbfcff8b1e4e6f25691bc2
Raw frame: episode 0, frame 0, timestamp 0
Raw input NPZ bytes: 3694714
Raw input NPZ SHA256: 06fd84bd276d21b5db8a2972dc9f5c67ab1d97d118ce25e259b1b879ba0850e2
Saved noise SHA256: f690c039a5d6e35e8cd5cb6f0bec9d5ea31b79294fc429872311c283679b7f52
```

The source-controlled acquisition tool checks the fixed episode LFS SHA256 and
size, small-file Git blob identities, and all local SHA256 digests. The full
original episode remains outside artifacts in the isolated `aloha-dataset`
cache. Packing uses PyArrow 20.0.0 from an isolated `input-pack-deps` target,
without changing either model environment; its installation log and actual
combined dependency freeze are archived.

The frame pack retains all four recorded 480x640 RGB images, original 14-element
`observation.state`, the dataset task string, and saved float32 Gaussian action
noise of shape [50,32]. Only noise is generated, using an explicitly recorded
NumPy PCG64 seed; the image/state data are not synthetic. Both encoded image-byte
hashes and decoded array hashes are recorded. State order is unchanged, matching
the pinned `LeRobotAlohaDataConfig` input pipeline; metadata joint-name labels
are not used to invent a reorder. Restoration calls the official
`transforms.unflatten_dict` on slash-separated NPZ keys, rejects ambiguous paths,
verifies all shapes/dtypes/hashes, and never enables pickle.

Local evidence includes `aloha-download-report.json`,
`aloha-episode0-frame0/manifest.json`, `aloha-episode0-frame0/observation.npz`,
`openpi-input-pack-deps-pip.log`, and `openpi-input-pack-environment-freeze.txt`.
The raw pack is synchronized under `<H100-root>/assets/openpi/inputs/`.

The actual H100 CPU preparation produced three model image inputs [1,3,224,224],
tokens [1,48], float64 state [1,32], and float32 noise [1,50,32]. Its complete
report has `status=prepared`, 84.72 seconds, and 15,096,292 KiB maximum RSS.
The SSH transport closed afterward with exit 255; that transport warning is
retained in `h100-pi0-recorded-input-prepare.stderr.log`, and no process remained.
This is not represented as a clean SSH exit. The remotely completed report and
prepared arrays were retrieved independently.

The subsequent 10-step reference/partition process, PID `245992`, exited 0 with
empty stderr. It used the same strict PyTorch weights, raw frame, saved noise,
official Euler schedule, and frozen `atol=rtol=0`. Every context tensor remained
unchanged across the step audit. Both complete action spaces passed:

| Space | Shape | Compared Values | Cosine | MSE | Max Abs | Exact Values |
|---|---|---:|---:|---:|---:|---|
| Normalized | [1,50,32] | 1600 | 1 | 0 | 0 | true |
| Native ALOHA action scale | [50,14] | 700 | 1 | 0 | 0 | true |

This is one actual recorded sample, not an aggregate robot/task-quality study.
The report explicitly records `physical_units_verified=false` and
`robot_calibration_verified=false`: canonical model postprocessing is not
physical calibration or robot execution evidence. The compatibility array names
`physical_reference`/`physical_partitioned` denote this native scale only.

Full raw output arrays and reports are returned under
`h100-pi0-recorded-input-partition-001/`; `actions.npz` SHA256 is
`7f6f2c42028261c5fbd426d03c9c6b9267a8aec6a08dc808785229b3a3cabdbb`.
The audit took 96.01 seconds including model loading, with 13.06 seconds covering
both reference and partition evaluations, and maximum RSS 15,091,120 KiB. These
are audit wall times, not a warmed end-to-end throughput or latency benchmark.
This establishes same-converted-weight CPU parity, not JAX conversion parity,
GPU parity, low-bit acceptance, or no-Python deployment.

Reproduce with `vlaforge.adapters.openpi.openpi_reference`, the same arguments as the
load probe, plus `--input-manifest <pack>/manifest.json --output-dir <new-run>`
and `--num-steps 10 --partition`. `--prepare-only` performs no action forward
calls. `--capture` additionally captures each region, saves and reloads its
`torch.export` archive, checks region outputs, and runs the complete serialized
Invocation IR using only those reloaded regions in the Python interpreter.
The first audit in `pi0-recorded-input-capture-001` exported and persisted all
four actual regions, then passed their individual reload checks. It subsequently
failed because the audit helper passed raw tensors to an Interpreter requiring
`TensorView`. The helper was corrected against the existing typed binding
contract, with a red/green serialization regression. That original failed report
is preserved; later independent runs provide separate acceptance evidence.

## Pi0 Serialized Replay And Numerical Policy

Independent replay loads only hash-verified region archives, serialized Invocation
IR, prepared input bytes, and the recorded full action reference. It does not
construct OpenPI or reload a Policy. The four archives in the first capture are:

| Region | Bytes | SHA256 |
|---|---:|---|
| prefix | 7024848417 | `5a03be393d38fd47c487d74dc048d733f8cae0f10437b2944e071dfcc05b8ab2` |
| time | 13533 | `b12288b18be07c5e2a8e620f9511825b3d2c7ecb59a073e5376ecb6d5578740b` |
| step | 7031171379 | `e0216824f28f67dca0ce84141c36778b1c19c49ee7cb26469886fbef4deb028e` |
| finite | 15086 | `0ce29af1b63879e47a3286427df5dd6c91964972ebdb57fb864390684d1a807e` |

The Invocation IR SHA256 is
`da1bafacda3e11abb250ca824e0868d8254f24546d89e3e822c20a45f4b7c282`.
Using these same archives, inputs, reference and zero tolerances exposed a
process-state dependency in the official constructor:
`pi0_pytorch.py` sets `torch.set_float32_matmul_precision("high")`.

| Independent Replay | Matmul Policy | MSE | Max Abs | Mismatched Values | Outcome |
|---|---|---:|---:|---:|---|
| `pi0-recorded-input-replay-002` | fresh-process `highest` | 0.000024238842256324 | 0.03133583068847656 | 1600/1600 | failed |
| `pi0-recorded-input-replay-003-high` | explicit official `high` | 0 | 0 | 0/1600 | passed |

The failed run's cosine was 0.9999148142211369, demonstrating why cosine alone
does not prove equivalence. The passing run has cosine 1 and exact full normalized
output equality. Its returned actions SHA256 is
`c05453990c8827b8367a43f5aa56818321f854e56142d10551084969b12bece6`.
This is CPU Python IR replay, not C++ deployment or a warmed latency result.

The reusable `vlaforge.numerical_context` module now defines a strict versioned
20-field policy covering matmul/TF32, reduced-precision reduction, CPU/CUDA
autocast, SDPA enable flags and math-reduction policy, and deterministic/cuDNN
flags. `snapshot`, strict JSON serialization, `require_current`, and the explicit
`offline_restore` guard are model-independent. Unknown, missing, duplicate,
inconsistent or unavailable required fields fail rather than receiving defaults.
The guard restores previous state after success, body failure or partial setup
failure, and rejects nested/concurrent guards. It is explicitly not concurrency
safe against unrelated execution: backend controls are process-global and
autocast state is current-thread state. Hardware, library versions and thread
counts remain separate reproducibility evidence.

`CaptureEvidence` records the observed policy and verifies capture and parity
checks leave it unchanged. This is Python observation/enforcement only; no
ArtifactContract or C++ policy enforcement is claimed. OpenPI now uses this
shared module instead of local single-field restore logic. Legacy capture reports
without a full policy require an explicit complete context file plus
`--allow-legacy-context`, with that provenance recorded. No original report is
rewritten to invent historically unobserved fields.

A new real pi0 capture in `pi0-recorded-input-capture-004-numerical-context`,
PID 264479, completed with exit 0. It directly records the complete context after
official Policy loading and on every captured region. All four region
capture/save/reload checks and the full 10-step serialized-IR replay in that
process passed exactly. The independent fresh-process `replay-005-numerical-context`
failed (MSE 0.000035084726330389136, max-abs 0.05700492858886719) and is preserved.
Thus in-process capture success does not establish this new guard's independent
replay correctness.

Diagnostic 006 loaded the same archives after the official single `high` setter;
its first inference, inference after a full-policy restore, and post-restore
inference all matched exactly. Diagnostic 007 applied the full restore before
loading and failed with MSE 0.000024238842256324 and max-abs 0.03133583068847656,
despite all recorded getters matching. Both use two intra-op threads, 96 inter-op
threads and MKLDNN enabled. These observations implicate initialization timing
or setter side effects, not changed graph text (prefix/step graph text is identical
between captures 001 and 004). The exact backend-level cause remains under
investigation; small CPU GEMM probes did not reproduce it.

The restore implementation now writes only differing policy fields and uses
the matmul precision setter once to establish its TF32 alias. This also avoids
needlessly rewriting an already-matching caller policy. Fresh real replay 008
failed with MSE 0.000024238842256324 and max-abs 0.03133583068847656;
the complete Python restore is not accepted for this CPU artifact. The new public replayer writes its final report only after the
guard has verified and restored the caller's original policy.

Diagnostics 009 (snapshot before setting high) and 012 (the first ten getters
after setting high) have exact first and subsequent outputs. Diagnostics 010
(direct differential `_apply`), 011 (post-high complete snapshot), 013 (second
ten getters), 014 (five SDPA getters), and 016 (only
`fp16_bf16_reduction_math_sdp_allowed`) reproduce a first-call discrepancy.
015 (only `flash_sdp_enabled`) is exact. In 011/013/014/016, subsequent outputs
are exact. This is a reproducible correlation, not proof that the getter writes
policy: the pinned PyTorch 2.7.1 C++ getter and its Python binding simply return
the field. Source audit: [Module.cpp](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/csrc/Module.cpp#L793).

Diagnostic 017 additionally hashes every exported parameter/buffer and every
explicit input. All model state is unchanged before/after, every input is
unchanged, and the first discrepancy is already in the prefix output with
identical input hashes/strides. Later step inputs consequently differ. This
narrows investigation to first-prefix execution or backend initialization; it
does not justify dropping the first output, relaxing tolerances, or declaring
the complete CPU context validated. Full logs and region boundaries are in
`h100-pi0-numerical-context-diagnostic-017-region-boundaries/`.

## Pi0.5 Strict Conversion And Actual Actions

H100 CPU strict conversion completed with PID 259262 and exit 0. The resulting
`pi05_base_strict_pytorch/model.safetensors` is 7,233,650,408 bytes, SHA256
`7ed2fb2f91b084efc387022383035f6908aa620f6bfa0da52204a4e52e7851b6`.
All 3,616,757,520 model parameter elements are accounted for, with no missing
required or unexpected keys. The explicitly unused expert vocabulary head
(263,323,648 elements) is zero-initialized, not substituted for an executed
trained weight. The official conversion restores float32 source tensors and
stores BF16 weights; JAX-versus-PyTorch numerical equivalence is not assessed.
Full conversion evidence is `pi05-strict-conversion-report.json`.

The same recorded ALOHA frame and saved noise then ran actual `pi05_aloha`
reference and shared prefix/step partition, PID 261557, exit 0. Both the full
normalized [1,50,32] output and native ALOHA [50,14] output have MSE 0, max-abs 0,
zero mismatches and exact-value equality under atol=rtol=0. The report and complete
outputs are in `h100-pi05-recorded-input-partition-001/`; actions SHA256 is
`d082da940c133b010bfffd7b4f2f7b95dc8c00715449a14c8cdbfdb9bf652f47`.
The source report records official matmul policy `high`. This validates the
pi0.5 state-tokenization and AdaRMS step path with real weights, not a fixture.
This CPU run does not establish pi0.5 export, CUDA, C++, calibration or robot execution.

## Actual H100 CUDA Evidence

After a fresh resource check, the parent allocated physical H100 GPU 2, UUID
`GPU-fd97302e-15f8-ed02-5d53-3c8c14d53536`. Each process sees only that UUID as
`cuda:0`; CPU threads remain limited to two, JAX stays on CPU, and the separate
OpenPI CUDA environment retains official transformers 4.53.2 patches. These
runs use torch 2.10.0+cu128, distinct from the torch 2.7.1 CPU conversion/reference
environment. Cross-version or cross-device equality is not claimed.

Both official ten-step references completed with the frozen real ALOHA frame,
saved noise, and strictly converted complete weights:

| Run | PID / exit | Complete action NPZ SHA256 | Scope |
| --- | --- | --- | --- |
| `h100-pi0-cuda-reference-001` | 277209 / 0 | `8ace904ad003f02eba4bfd34f68c20ba801ac3acfb3222707b0e6e5bd4b4d223` | Official CUDA reference |
| `h100-pi05-cuda-reference-001` | 278644 / 0 | `da5def9e633f28fe7246e5f4e30c2ad176c3812c32812ad13874e193eed274db` | Official CUDA reference |

These single cold audits are not throughput/latency benchmarks. Native ALOHA
postprocessing is retained; physical units and robot calibration remain unverified.

Pi0 CUDA partition/capture 002 failed before export with all 1,600 normalized
values differing, max-abs 0.03191521763801575. Independent diagnostics 003/004
established the cause: preprocessing produced image shape [1,3,224,224] with
channels-last stride [150528,1,672,3], but the Adapter materialized NCHW stride
[150528,50176,224,1] before convolution. Replacing only the diagnostic image
inputs reproduces MSE 0.000029782201148331787 and the identical max-abs error.
Original versus contiguous K/V cache and in-place versus out-of-place Euler
time updates are exact; repeated official actions are exact.

The shared Adapter now records each official image memory format and restores
it explicitly inside the prefix graph while keeping a contiguous external
tensor ABI. Both contiguous and channels-last formats are supported; unknown
strided layouts fail closed. There is no model-name branch, no changed
checkpoint, and no relaxed tolerance. Focused layout/device tests passed
38 tests with the separate real-model opt-in test explicitly skipped.

Pi0 CUDA capture 005 (PID 281538, exit 0) passed official-versus-partition parity
for all 1,600 normalized and 700 native-action values, four real region captures,
individual archive reload parity, and a full ten-step Invocation IR replay.
All MSE/max-abs/mismatch counts are zero. Returned evidence is in
`h100-pi0-cuda-capture-005-image-layout/`; the approximately 14 GB of actual
`.pt2` archives remain on H100 under the corresponding isolated run directory.
The IR SHA256 is `f36d0ac11942f3e6470cb0e0ceab75de1615577e8ad3fbcf4bb1cab28c00c1f3`.

Independent pi0 CUDA saved-only replay 006 also passed the first complete output
with no OpenPI construction or discarded warmup. It restores the complete
shared numerical context, verifies archive/input/reference hashes, and compares
all 1,600 values exactly. Its final report verifies restoration of the caller's
original `highest` policy. Actions SHA256 is
`b4412fb8f7aa0238c0b2bf7a6fe821fee821c0e0c5a6f0b85b0b65e3103359c8`.
Evidence: `h100-pi0-cuda-saved-only-006/`. This closes actual pi0 CUDA L2, not
AOTI, no-Python C++ Session, continuous benchmark, or board execution.

Pi0.5 CUDA partition/capture 002 (PID 284257, exit 0) also passed all normalized
and native values exactly, all four real captures and per-region archive reloads,
and complete ten-step IR replay. IR SHA256 is
`06cf286c6d965c02e2c60be32a4f425da69b3ab21e628ba15a89e995ff81fa83`.
The independent saved-only replay 003 (PID 285934, exit 0) passed all 1,600 values
exactly with the complete shared context and caller-policy restoration verified.
Actions SHA256 is `120aa16afef09f905f1675fbb2a2621e1295ab224293b2df5d7ee90ab909e89d`.
Evidence: `h100-pi05-cuda-capture-002-image-layout/` and
`h100-pi05-cuda-saved-only-003/`. Both variants now have actual CUDA L2 evidence.

The shared `openpi_aoti` orchestration calls the existing generic
`vlaforge compile-artifact` CLI inside the recorded numerical context. It checks
capture/IR/input/archive hashes, actual target, exact configuration scalar types,
backend pass records, package audit and final package digest. The initial helper
run 001 failed before compilation due to a Path/string error; its stderr is
retained. Corrected pi0 eager-numerics run 002 (PID 288715, exit 0) produced time,
finite and real step packages (424,630, 431,500 and 638,389,802 bytes respectively).
This is compilation only, not numerical acceptance. Audit 003 caught a missing
explicit `torch._inductor.codecache` import before package execution. Audit 004
then caught a singleton output tuple binding error. Both failed helper reports
are retained. The singleton adapter binding is fixed and covered by a unit test.
Partial artifact runs explicitly report a hybrid export/AOTI pipeline, never a
complete artifact or native C++ deployment.

Corrected pi0 hybrid audit 005 completed all ten real steps with an exported
prefix and AOTI time/finite/step. Time/finite outputs were exact, but same-input
step action outputs were not. Final normalized action cosine was
0.9999976904489936, MSE 6.091567772522104e-7, and max-abs 0.003613114356994629.
All 1,600 values mismatched. This is a strict numerical failure, not lossless
validation. Prefix compile 006 (PID 292853, exit 0) produced a 5,853,482,991-byte
package with SHA256 `ce2858c752ceaa84f6fa5afc1f9ada3367d48cc16b49e15cac1633c46af1efef`.
Full four-artifact audit 007 also completed all steps but failed: cosine
0.9999378376551323, MSE 1.7464798115476964e-5, max-abs 0.027894020080566406,
1,600 mismatches. Its same-process saved-export baseline remained exact.
All actual per-invocation outputs and final actions are retained. This proves
execution of all compiled Regions under Python IR orchestration, not native
Session acceptance. pi0.5 eager-numerics compile 001 (PID 297792, exit 0)
completed time/finite/step/prefix. Full artifact audit 002 (PID 300766)
completed all ten steps, with normalized cosine 0.9999953343472654,
MSE 1.0663439368697745e-6, max-abs 0.005490541458129883, and all 1,600 values
mismatching. The same-process saved-export baseline was exact. Its 36 prefix
K/V outputs also differed. This is another strict failure; full actions and
every actual Region output have been returned to local evidence storage.

The unchanged generic `aten-preserving` backend was then applied to pi0 in
compile 009 (PID 302984). All four packages compiled. Audit 010 demonstrated
bitwise exact values for all 36 real prefix K/V outputs, but stopped at the
time initializer: the AOTI proxy rejected `aten.ones.default([])` with
`Expected SymIntList or IntList but got None`. There is no full action result
for this attempt. The original export is only 12,393 bytes, with the exact
scalar-one/timestep semantics preserved. The generic proxy serialization
diagnosis is separate from model adapters; no model-specific time replacement
or tolerance relaxation has been made.

`openpi_session` now binds all actual packages through the existing generic
`build_artifact_compile_bundle`, using an explicit hash-recorded runner
template. It preserves the official float64 state input and independently
compares native complete actions to both the same AOTI Python result and the
official model result. It explicitly records that native numerical policy
enforcement is not verified. pi0.5 native attempt 003 reached CMake but failed
to find OpenSSL; no executable was produced. Existing `/opt/conda` OpenSSL
headers/library were located, and attempt 004 rebuilt with only the
process-local `OPENSSL_ROOT_DIR=/opt/conda`. It produced a verified bundle and
actually executed the C++ Session three times (parent PID 308742, native PID
309489, native exit 0). `ldd` contains no libpython and PYTHONHOME/PYTHONPATH
were set to nonexistent paths during native execution. All three complete
1,600-element outputs share SHA256
`9e8c3a6c461a0a4231492d0b6b678cf717a298e1e736b18f838694191742186d`.
However, strict fidelity failed against both references:

| Native pi0.5 comparison | Cosine | MSE | Max-abs |
| --- | --- | --- | --- |
| C++ vs same AOTI Python | 0.999998931741552 | 2.4303444139178836e-7 | 0.0028092265129089355 |
| C++ vs official model | 0.9999954709620339 | 1.0333326646675647e-6 | 0.004783749580383301 |

This is true no-Python execution evidence with failed numerical acceptance,
not completed L4 validation. Local `native-build-evidence/` holds generated
C++ and compiler metadata, explicitly not a runnable bundle. The full remote
bundle stays in `pi05-aoti-eager-004-native-openssl/bundle/`; every artifact
remains SHA-bound. Three repeats of one frame are not a latency CDF.

CPU diagnostic 020 traced the complete actual prefix. Parameters, persistent
and nonpersistent buffers, lifted tensors, and explicit inputs were unchanged.
The first differing node was `wrap_with_set_grad_enabled`: the upstream Gemma
RoPE subgraph returned differing bfloat16 cosine values of shape [1,816,256],
while sine was exact. All prior vision/prefix nodes matched. This does not prove
the pure SDPA getter changes policy. The first complete action differed while
the second and third matched. The real CPU subgraph pack 021 now contains the
original 37,356-byte HOP export, 844,499-byte independent inputs/reference, and
both actual cold and second references. BF16 inverse frequencies [128], I64
position IDs [1,816], output BF16 [1,816,256], strides and storage offsets are
recorded without conversion. Independent eager and saved-reload execution match
the second reference, whose cosine SHA also matches the complete-model exact
round from diagnostic 020. Cold and second references differ and are retained.

Fresh-worker small-pack diagnostic 022 reproduces a first cosine mismatch under
default policy, high-plus-SDPA-getter, and the complete shared context guard;
later calls are exact. Single `set_float32_matmul_precision("high")` is exact
on its first call. Nested node trace 023 proves identical matmul, transpose and
concatenation bytes before `aten.cos.default`, where the first difference
appears. Sine remains exact. Reducing OMP/MKL from two threads to one in 024
made all three small-pack calls exact under the complete guard. This identifies
a CPU thread/cold-start sensitivity, not a proven write effect of the pure
getter. Independent whole-model saved-only one-thread verification 025 still
failed against the original two-thread reference: cosine 0.9999979076042858,
MSE 5.413334314092898e-7, max-abs 0.0024886131286621094. Thus the small-pack
one-thread success does not close complete-model context fidelity; it also
changes a recorded execution environment condition. Both results remain
separate, and the original two-thread CPU failure remains unresolved.

The separately extracted actual H100 CUDA RoPE pack 008 preserves original
CUDA tensors and nested subgraph. First/second actual references, independent
eager, and export save/reload all match exactly. Its EP is 49,294 bytes
(SHA256 `609cb1d7369f7a1dbfa2922896a72d7fb78941ed18b840351e86acf1d1fa85a0`),
with an 844,563-byte inputs/reference pack. This is workload extraction and
correctness evidence, not a measured RoPE speedup or full-model integration.
The distinct `vlaforge.actual_subgraph_examples/1` report schema explicitly
identifies the higher-order subgraph; existing single-ATen benchmark admission
must not bypass its provenance and cold-reference checks.

Before the capture run, the actual isolated OpenPI environment passed 64 focused
tests with only the opt-in whole-model capture test skipped. That skip is not
the standalone real checkpoint audit above. In the minimal validation environment,
eight input-format negative tests passed and the genuine official-dependency
tree-restoration test explicitly skipped because transformers is unavailable.
The actual official tree restoration passed in the isolated OpenPI environment;
dependency skips are not presented as successful upstream execution.

## Public Dual-Stage AOTI Closure

Pi0 public compile 011 and full audit 012 use a new frozen local-source snapshot
`source-openpi-aoti-public-1215` on H100. The public pre-AOT and post-grad empty
list handling is present in both source-bound ledgers; no isolated diagnostic
time package or previous-profile artifact is substituted. Compiler source SHA:
`74279641cc1b499ba656513efc1736217883588248f6a037252277b3f2fcdc5f`
(`aoti_export.py`); CLI SHA:
`961e939b2581844a4c2475b1b7cdb23368b00b841bdd555679018bb1f0878a45`.

All four real captured Regions were newly compiled with `aten-preserving`.
The complete actual Invocation IR then ran 13 calls: prefix, timestep, ten
denoising steps and finite output. Every Region output compared against the
same-input saved EP is exact. All 1,600 normalized action values are exact
against the official same-precision reference: cosine 1, MSE/max-abs zero.
The separately executed saved EP baseline is also exact. Audit PID 331000,
H100 GPU2 UUID as above, Torch 2.10.0+cu128; 102.17 seconds is diagnostic wall
time including verification and loading, not an inference latency sample.

The full report and all outputs are local in
`h100-pi0-aoti-aten-012-public-full-audit/` under the evidence root. Report SHA256
`e84ee78ecdd9ef6ff2fb0f52b0d5a7d96168f96fa919c9be4e6e6aece68415b5`;
30,264,056-byte complete actions/Region NPZ SHA256
`9112c81f9ee66738fceec194e44f839c2360d95fa3c0aa78a538d6413bddc0d5`.
Compiler manifests/logs are local in `h100-pi0-aoti-aten-011-public-dual-stage/`;
the actual packages remain at the corresponding hash-bound H100 run paths.
This closes actual single-record L3 Python AOTI fidelity for pi0, not native
Session fidelity, multi-record statistics, latency, JAX conversion parity, or
board deployment. Old eager-numerics and failed proxy runs remain unchanged.

Pi05 public compile 011 and full audit 012 subsequently completed under the same
frozen source/profile. All 13 calls and every same-input Region output are exact;
all 1,600 normalized action values match the official same-precision reference
bitwise, with MSE/max-abs zero and cosine 1. The independently executed saved EP
baseline is exact. Audit PID 335551; 120.62 seconds is diagnostic wall time, not
an inference latency sample. Local report:
`h100-pi05-aoti-aten-012-public-full-audit/report.json`, SHA256
`08fe288976e40c3a4f7a0f5aec03ff9db83b4e25b43993b9ecdd78245b5774a3`;
35,867,688-byte complete output NPZ SHA256
`e7751d323541b00cc7ba49c56dd8789dbd7c99ce42dbcd76ae06c177a05f9f7f`.
This also closes single-record L3 Python AOTI fidelity for pi05, not native
Session fidelity or the new v2 numerical-context chain. Earlier eager-numerics
failures and their original reports remain unchanged.

## Native Runtime Boundary Diagnostic

The pi05 native trace attempts 005 and 007 failed during C++ compilation, before
any GPU execution. In Torch 2.10, both `allowFP16ReductionCuBLAS()` and
`allowBF16ReductionCuBLAS()` return typed reduction option enums, not the bools
returned by the older CPU Torch headers. The diagnostic now records both enums
explicitly instead of incorrectly labeling them bools. The original build
failures remain in the corresponding `h100-pi05-aoti-eager-*-native-trace/` folders.
This offline linker wrapper observes the existing runtime and never installs a
numerical policy or replaces the public backend.

Independent Python worker 006 loaded the exact four packages from native bundle
004, retaining all complete per-Region input/output bytes and layout metadata.
Its explicit multi-factor candidate used default `highest`/TF32-false policy,
single-threaded package loading, a dedicated stream, contiguous output copies,
and disabled inference mode within each package call. Both complete 1,600-value
outputs were bitwise equal to the native 004 result (MSE/max-abs zero). They
still failed the official same-precision reference, with MSE
`1.0333326646675647e-6` and max-abs `0.004783749580383301`.
Evidence: `h100-pi05-runtime-probe-006-cpp-default/`, PID 317384, H100 GPU UUID
`GPU-fd97302e-15f8-ed02-5d53-3c8c14d53536`, Torch 2.10.0+cu128, two CPU threads
and 96 inter-op threads. This identifies a reproducible runtime-boundary
candidate, not by itself a unique causal factor, latency result, or completed
L4 fidelity gate.

Independent workers 008 and 009 repeat that comparison with the same new probe
source and all factors fixed except default versus captured numerical policy.
008 again matches native 004 bitwise. 009 matches the original same-artifact
Python audit 002 bitwise across all 1,600 action values. Its observed setter
sequence is `torch.set_float32_matmul_precision("high")` on entry followed by
`"highest"` on exit. The complete recorded 20-field snapshots differ only in
matmul precision and its TF32 alias. The first unequal boundary is the first
`openpi_step` action output, MSE `4.642827754777144e-8`, max-abs
`0.001058340072631836`; prefix outputs, timestep and all first-step inputs are
byte-identical. This is a causal single-factor result for this saved workload,
not evidence that every model/runtime discrepancy has the same cause.

Native trace 010 (parent PID 326225, native PID 326979) completed two actions,
both retaining native 004's full output SHA256
`9e8c3a6c461a0a4231492d0b6b678cf717a298e1e736b18f838694191742186d`.
The trace therefore did not change this workload's result. Its C++ getters show
highest/TF32-false, two CPU threads, 96 inter-op threads, and FP16/BF16 reduction
option enum zero. Against Python 008, all 940 complete input/output boundaries
over 26 calls are byte-identical; 880 boundaries have different underlying
addresses modulo 256, without output divergence in this comparison.
Evidence: `pi05-runtime-probe-008-vs-native010-boundaries-v2.json` and
`pi05-runtime-probe-008-vs-009-boundaries.json`. These diagnostic runs do not
close the separate AOTI-versus-official fidelity failure.

The original 010 report's descriptive Region ID map accidentally used source
declaration order. The comparison gate rejected that mismatch. The v2 companion
comparison derives IDs from the checksum-bound compiled `semantic_ir.json`,
matching C++ codegen's enumeration, and records both the incorrect descriptive
map and the verified runtime map explicitly. Original reports are unchanged;
future trace reporting uses the compiled map directly.

Torch 2.10's reduction enums additionally distinguish split-K permissions. Its
legacy Python bool getter projects only the reduced-precision component, while
`*_reduced_precision_reduction_split_k` exposes the second component. The historical
20-field Python schema does not encode this additional domain and must not be
advertised as a complete restoration contract for arbitrary Torch 2.10 policies.
Actual native trace enums are retained, and the current same-workload output
evidence remains separate from future provider/schema extension work.
The isolated CPU-only real Torch 2.10 counterexample
`splitk-v1-context-001.json` confirms the gap: caller reduction pairs
`(false, false)` become `(false, true)` after a v1 guard that temporarily enabled
reduced precision. The 20-field snapshot still reports equality with the caller.
Both FP16 and BF16 exhibit the loss; the diagnostic explicitly restores the
original complete pairs in `finally`. No GPU model was executed. This is a
confirmed v1 restoration defect, not a passing policy test.

The replacement `vlaforge.python_numerical_context/2` explicitly serializes all
22 policy fields, full Python package version, and the supported reduction API
domain. Torch 2.10.0 requires both split-K booleans and permits only complete
pairs `(true,true)`, `(false,true)`, `(false,false)`; Torch 2.7.1 has a distinct
bool-only API with required null split-K fields. Unknown releases/APIs fail
closed. Old v1 records round-trip unchanged as partial observations and are
rejected for restoration; no absent values are invented. Restore now uses
complete two-argument setters and restores the caller after body or setup errors.
The guard remains process-global and not concurrency-safe; autocast checks
refer to the actual calling thread, not all threads in a process.

Final module SHA256:
`c1880c077c21b77d2b8f7fd447132d031aafad2991a93f06735090dcc899741c`.
Actual isolated Torch 2.10 CPU counterexample rerun
`splitk-v2-context-003-final.json`, PID 373363, SHA256
`3ba817b9f0a7e185b9c8f03385225a47d654f64ca87142f3241fdd126936b38d`,
verifies complete caller `(false,false)` pairs after normal and exceptional
bodies. It also verifies unchanged legacy round-trip and refusal without state
mutation. Actual Torch 2.10 tests passed 45; the complete local Torch 2.7.1
frontend/OpenPI suite passed 183 with 11 explicit skips (ten 2.10-only tests
executed remotely, one opt-in whole-model test). The C++ provider handoff and
actual installed-header hashes are in `numerical-context-v2-provider-handoff.md`.

All eight H100 GPUs were subsequently occupied by unrelated processes; raw
UUID/owner evidence is retained in `h100-v2-capture-resource-blocked-*.csv`.
At that checkpoint no v2 real-model capture/compilation had run. The later H20
pi0 v2 L2/L3 run is recorded below. Both models' previous exact H100 L2/L3
outputs remain bound to their historical v1 source and records.

Independent LibTorch 2.10 C++ CPU provider verification is retained in
`h100-libtorch-numerical-cpu-001/`: native PID 416234 passed 591 checks over all
22 mathematical fields, four validation boundaries and nine actual FP16/BF16
reduction combinations. It exercises independent Session-owned lease sets,
process conflicts, release, actual worker-thread autocast drift, invalid typed
requirements and no provider setter side effects. `ldd.log` and `native.maps`
show a shared provider and no mapped libpython. The associated Python v2,
projection and codegen suite passed 68 tests. These are native CPU provider
contract tests, not full VLA Session/output fidelity or CUDA execution evidence.

A second frozen-source run, `source-libtorch-numerical-cpu-1402`, compiled two
actual generated CPU TorchScript fixture Sessions: one with the complete
observed v2 policy and a legacy control without requirements. Twelve independent
native processes (six modes per build) passed: normal/cache reuse, overlapping
Session lifetime, initialization drift, run-entry drift, calling-thread autocast
drift, and drift between pending output and commit. The bound Session rejects
drift, keeps its previous committed output and remains failed after the caller
restores its policy. The legacy control retains unbound behavior. All processes
have maps/ldd evidence with no libpython. The test uses a two-element AddOne
fixture, not random VLA weights or model fidelity evidence. Ten static tests
plus this native integration test passed in 80.55 seconds including two builds.

Complete logs, generated source, tiny TS artifacts and copied native binaries
are local in `h100-libtorch-generated-cpu-002/`; dependencies and RPATH still
refer to the isolated H100 environment, so this is not a portable runtime bundle.
The returned provider source (213 files), native binaries and all fixture
artifacts were independently SHA-verified locally. Summary:
`libtorch-numerical-cpu-evidence-verified.json`, SHA256
`c8ab98199f3407c17876bc68efcb7389bdab607e1be7c3865c3f4325e0344f3f`.
Provider source SHA256
`a72b956895505c029b6d3e22b9c061242662c87d56aa0ed3b205dda74a2c22b5`;
actual provider library SHA256
`1c34a927ec2c25bfb389d7c09b6c92990ce9e2d7f5b5427b2eb4e4e05275ab5f`.

Final smoke source SHA256
`7256013b2c04d53eb1796b7d2ad4c52c82e6600092a298bcaea0a8281069c24d`
adds an unsupported-release compile branch, keeping tests buildable with older
Torch backends. Actual Torch 2.7.1+cpu PID 437654 rejected a complete valid 2.10
request by release without changing existing reduction/matmul state; actual
Torch 2.10.0+cu128 PID 439160 reran all 591 checks. Both native processes have no
libpython mapping. Reports: `h100-libtorch-numerical-cpu-003-torch271/` and
`h100-libtorch-numerical-cpu-004-torch210/`. This is fail-closed compatibility,
not claimed numerical-provider support for older Torch releases.

The OpenPI artifact verifier now requires the generic compiler's separate
`backend_program_audit` pre-AOT ledger, matching the active pass/source records
and requiring a rewrite list. The existing post-grad and package ledgers remain
independent. Missing or tampered pre-AOT records are rejected, including old
aten-preserving manifests without the new ledger. Historical packages remain
bound to their original frozen source; they are not relabeled as rebuilt.
The focused AOTI/native-rendering suite passed 24 tests, including new ledger
negative cases (`openpi-pre-aot-native-tests.xml`).

## Pure Tensor Native-Scale Output

`adapters/openpi/openpi_output.py` lowers the pinned official output processor sequence
by processor type, with explicit extension registration and refusal of unknown
processors. There is no model-name branch. Current processors are Unnormalize,
AbsoluteActions, and AlohaOutputs. The implementation keeps statistics dtype,
quantile arithmetic order, NumPy integer-mask promotion, and the original
in-place addition's destination dtype. It does not merge the operations into a
different affine expression.

Actual CPU audit PID1393684 uses already hash-verified H100 pi0, H100 pi0.5 and
new H20 pi0 actions and state. All three official 700-value native-scale outputs
are exact for both eager tensors and torch.export save/reload, with zero MSE and
max-abs. This constructs only official processors, not a model, and does not
run a GPU. Report `openpi-actual-tensor-output-cpu-002/report.json` SHA256:
`a9c7f99d476813564f8481dfaec0883a63bcba9b51c086227063b7bf3eb1b5d4`.
The lowerer SHA256 is
`d3508c3e9b601f1d67110bfa7cc7cb614a6014b4d7bc229194af61166b035dc6`;
the exact lowerer and probe are frozen under `tensor-output-source-002/`.
Thirteen separate tensor fixtures pass, including dtype, masks and save/reload.

This is not yet CUDA compilation or no-Python native-scale deployment. The new
finish Region and complete updated bundle remain a separate gate after the
current normalized-action bundle is validated. Native ALOHA scale is still not
verified robot calibration or physical execution.

## Verification And Next Gate

The explicit native worker bootstrap has now passed independent real CPU tests.
An actual `std::thread::id` reuse bug was reproduced on the first provider
version and fixed by the root-owned retained thread-token implementation.
Fourteen fresh native processes validate the complete policy, medium precision,
all nine reduction pairs, lease/thread restrictions and real setter-failure
rollback/poison using an external LD_PRELOAD interposer. Seven further generated
Session fixture processes exercise the public Python initializer renderer and
prove that omitting explicit bootstrap is rejected without hidden Session
setters. These are native CPU interface checks, not pretrained-model evidence.
The independently checked archive, source/binary hashes and retained original
failure are documented in `openpi-inventory/libtorch-bootstrap-README.md`;
aggregate SHA256 `0db263b2c4d4bca92caf0e43215929744b0374d45136dacb41095b644f822733`.

The local CUDA OpenPI environment is isolated at
`/home/zhangzimo/.venvs/edgefm-openpi-cu128-py311-20260906`, installed from the
existing pinned wheelhouses with Torch 2.10.0+cu128 and the official Transformers
4.53.2 patches. A CPU-only source/dependency/tokenizer probe passes. Both models'
official native output processors were independently reconstructed without
weights and replayed on their saved real outputs: 700/700 native ALOHA values
match exactly for each model, MSE/max-abs zero. This does not verify robot
calibration or new local CUDA model outputs. Report
`local-openpi-native-processor-001.json`, SHA256
`387643b43dfd06221ea16d58d08a910ff4337e7065f5ee3fed5c7a4e3acacf63`.

New Adapter-only phased persistence uses the existing public capture API:
`openpi_reference --partition --capture-save-only` writes real region archives,
original examples and expected outputs without reloading a second weight set.
`openpi_phased` runs only after the first Linux process identity has exited,
checks tensor dtype/device/shape/stride/offset and all hashes, then independently
validates every region and complete normalized/native action outputs. A distinct
hash-bound joined report is admitted to AOTI only after that second worker passes;
the original incomplete report is never upgraded in place. Complete worker CUDA
peaks are recorded including capture, unlike the older reference-only peak fields.
The first 46 CPU regression tests pass; the expanded actual Torch2.10 CPU suite
passes 186 tests with three explicit optional real-model/native-build skips.
Actual local RTX v2 model execution remains unrun. Independently, H20 pi0 has
now completed a new real v2 L2 run as described below; fixture tests are not its
source of evidence.

All 33 H20 raw pi0 objects passed independent generation/size/MD5/CRC32C and
local-SHA verification. CPU-only strict conversion under the separate Torch2.10
environment produces the identical complete converted SHA256 as the historical
H100 Torch2.7.1 CPU conversion. H20 reference/capture PID3139186 and independent
saved-only PID3140829 pass all four Regions and all 1600 normalized plus 700
native-scale action values with zero MSE/max-abs. The v2 joined report SHA256 is
`c4395da439f188fe12396d2c2e150c4126a5b14b97f65d729586e1f3d7670bf7`.
See `openpi-inventory/h20-openpi-v2-README.md` for immutable source reports,
hardware UUID, input provenance and the exact scope. The independent new AOTI
audit PID3148762 has now passed all 13 actual calls and the complete 1600-value
output, exact with zero MSE/max-abs. Its report SHA256 is
`711cb47158ce5c328dbc3d40c9a4bc3435d27db51238a36ae020f2d1763380d5`.
All four public aten-preserving packages record actual complete v2 contexts
before and after compilation. Compilation took 779.79 seconds and the tool
audit 216.30 seconds; neither duration is an inference latency measurement.

The first native attempt was rejected before construction because the Adapter
omitted the numerical artifact's explicit /4 schema. After that local wiring
fix and 23 focused CPU tests, independent native-002 built C++ successfully but
its process PID3154555 exited 127 before main: the shared numerical provider DSO
was not delivered into the bundle, and RUNPATH pointed at the removed temporary
build directory. Both attempts remain failed, not native numerical passes.
Following the public delivery/RPATH fixes, independent native004 passes three
complete 1600-value normalized outputs byte-for-byte against official and same
AOTI references. Actual native PID3164543 loads the delivered numerical provider
from bundle/lib; explicit worker initialization and 24-field Session validation
pass, with no Python library in the recorded maps. The independently checked
summary SHA256 is `2cc0a18d6a03099cd84289cdbf63197d288b7eae281ac61d9accc385023011d2`.
This is three repeated calls on one recorded frame with ordinary loop execution,
not a fresh-input campaign, latency measurement, CUDA Graph or board evidence.
This model's independent Python native-scale postprocessing is not counted as
no-Python native-scale deployment; the pure tensor finish still needs complete
native-bundle integration and execution.

The pure Tensor processor now independently passes actual H20 eager, strict
capture/save-reload and public AOTI for all 700 native action values. Initial
CUDA scalar FP64 division had 20 one-ULP differences despite cosine 1; that
failed zero-tolerance result was retained, and the identical divisor is now
an explicit FP64 Tensor buffer. A checked stage also returns a finite-value
predicate over both normalized and native actions. Complete byte verification
summary SHA256 is `df129394d199125c70e3d072c34f52978d850510f4591ec85286e19de1a293f1`.

The derived new Invocation IR retains normalized output as an auxiliary port
and publishes native actions in the same transaction; no original captured
Region or cache/loop computation is rewritten. Real-artifact CPU preparation
passes new I/O digest, all five numerical bindings, IR/Plan and generic typed
runner checks. The attempted complete new run was stopped when an external
GPU7 owner appeared, before full IR/native execution. Therefore the Tensor
processor success is not yet a no-Python native-scale full-model pass.
Exact failure records, successful processor evidence, source freezes and the
next owner-gated command are in `openpi-inventory/h20-openpi-v2-README.md`.

The independent saved-only worker peaks at 14.37GB reserved CUDA memory,
although reference/capture peaks at 7.33GB. Both Region archives redundantly
retain the complete model state. Read-only serialized graph inspection finds
1.163GB unreferenced prefix storage and 6.375GB unreferenced step storage.
Consequently, process separation alone does not demonstrate that the full
chain fits a 12GB RTX. No generic state-pruning optimization has been applied
to these already frozen artifacts.

CPU tests cover strict file/weight gates, tied aliases, unused-weight accounting,
invalid profiles, both source-shaped typed Interfaces, exact context dependency
sets, variadic loop/memory planning, Plan serialization, and export of the actual
official mask/timestep utility functions. These tests do not execute a random
miniature VLA and do not imply pretrained pi0/pi0.5 support.

The composite-checksum extension passed 50 focused CPU tests with one explicit
real-model skip (`openpi-composite-tests.xml`). Negative tests cover bad CRC32C,
size, missing/wrong response generation, missing/invalid component count, and
mixed MD5/composite metadata. Ordinary and composite owned-partial resume paths
are both tested. These transport byte fixtures are not model inference evidence.

The opt-in real test uses `VLAFORGE_OPENPI_REAL_CONFIG=<run.json>` with:

```json
{
  "adapter": {
    "source_root": "<repo>/third_party/openpi",
    "checkpoint_dir": "<weights-root>/pi0_base_strict_pytorch",
    "config_name": "pi0_aloha",
    "checkpoint_sha256": "<converted SHA256>",
    "device": "cuda:0",
    "num_steps": 10
  },
  "input_manifest": "<saved official raw observation/noise pack>/manifest.json",
  "tolerances": {"absolute_tolerance": 0.0, "relative_tolerance": 0.0}
}
```

The NPZ contains slash-flattened raw observation keys expected by the selected
official robot profile, a scalar string prompt, and `noise`; loading disallows pickle. Freeze
input/noise hashes and numerical tolerances before running real tests. The zero
tolerances above are a strict starting contract, not an observed numerical result.

```bash
VLAFORGE_OPENPI_REAL_CONFIG=<run.json> <openpi-python> -m pytest \
  vlaforge/tests/models/test_openpi_frontend.py \
  -k real_openpi_saved_input_full_chunk_and_capture -q
```

The numerical-context, frontend and OpenPI layout/device test run passed 132 tests
with one explicit opt-in real-model skip (`openpi-cuda-layout-regression.xml`).
Thirteen additional AOTI contract unit tests passed (`openpi-aoti-contract-tests-v2.log`);
they test metadata tampering and metrics, not actual AOTI package execution. The
standalone real checkpoint results above are recorded separately from this suite.
The expanded operator/native rendering/AOTI contract suite passed 23 tests
(`openpi-operator-aoti-session-tests-v2.xml`), including original tensor offsets,
mutation/RNG rejection, float64 state preservation, and strict template tokens.
The latest complete OpenPI model-contract plus numerical-context suite passed
119 tests with one explicit real-model opt-in skip (`openpi-full-focused-1118.xml`).
The actual standalone weights/capture/native runs are not that skipped test.

Earlier CUDA reference reports retained a stale `cpu` word in their human
`evidence_level` string. Their explicit device, GPU UUID, CUDA allocation,
package environment and tensors are authoritative. The reporting code now
derives that string from the actual device. Historical reports are preserved
byte-for-byte rather than rewritten after their hashes were bound.

Both models' actual CUDA capture/save/reload and independent saved-only replay
using the recorded 20-field Python numerical context have passed as above.
Both models' historical public dual-stage aten-preserving AOTI complete-action
audits have also passed. New pi0 v2 H20 capture/reload and full AOTI action audit
are now complete. Its native004 bundle also passes complete normalized actions
and actual delivered-provider checks as above. Native-scale Tensor finish
integration and pi05 v2 capture remain next gates. The distinct CPU cold-start restoration diagnostic
remains unresolved.

On the local machine, the isolated pi05 converted checkpoint is now complete:
`/home/zhangzimo/.cache/edgefm-vla-goal/openpi/pi05_base_strict_pytorch_chunked_001`.
The bounded four-range transfer finished PID1408329 with all 7,233,650,408 bytes
SHA256 `7ed2fb2f91b084efc387022383035f6908aa620f6bfa0da52204a4e52e7851b6`.
Separate CPU verification PID1412297 checked that full hash again, all nine
normalization assets and the strict historical conversion record; safetensors
inspection finds 812 BF16 tensors / 3,616,757,520 elements. Its report SHA256 is
`a525f16715d9394d44728e2d90a4ae21be41808edd74e6c165f0dfea064f34c5`.
This is actual local asset integrity, not local pi05 model/GPU execution. The
historical CPU conversion is not relabeled as a newly performed conversion.

H20-2 has glibc 2.35 and sufficient
isolated NAS space. Its independent OpenPI environment is now installed and
passes `pip check`, source/patch/tokenizer gates and CPU imports. All 75 extra
hash-pinned wheels (257,187,777 bytes) were verified; the full package freeze is
identical to the isolated H100 CUDA environment. This is environment preparation,
not H20 GPU/model execution evidence.

H20 GCS JSON and alternate-XML streams were too slow (approximately 50-65 KB/s)
and were stopped explicitly, retaining their owned partials and failure logs.
The H100 ProxyJump probe timed out; the existing mariodeng route succeeded.
Four-way rsync showed approximately 0.45 MB/s aggregate, not the predicted
2 MB/s. The initial controller's one-hour per-file timeout was unsuitable for
this measured rate. Its owned controller/children were stopped at 25 minutes,
exit 143, preserving 747,212,800 partial bytes. Controller 002 resumes the same
owned staging files using rsync delta transfer and four workers; it has no
total-duration timeout, a 600-second no-I/O timeout, and explicit child cleanup
on termination. Its historical active transfer state was not verification or
conversion evidence. Local PID1253642 eventually completed; state/logs:
`h20-pi0-rsync-4way-resume-002/`.
Four completed objects totaling 6,594,393,489 bytes were independently verified
on H20 by PID 3108373 against fixed generation metadata, size, official MD5 and
CRC32C, and the local verified SHA256. This is only a subset: the separate report
`h20-pi0-subset-check-001/report.json` explicitly keeps
`checkpoint_complete_verified=false`; SHA256
`4197c921e9e03a28845dd494096e1805b50ab6fb8d88f4ea8d237788aec03620`.
The active transfer controller's own state is unchanged by this read-only audit.
Remote staging: `assets/openpi/weights/pi0_base-rsync-4way` under the H20 Goal
root. Run the independent generation/size/provider-checksum/local-SHA verifier
on all 33 objects before conversion. That full independent verification and
strict CPU conversion have now completed as recorded above; the earlier subset
report and controller transfer states remain immutable historical evidence.
No second H20 NAS raw copy was created.
Keep normalized and native-action-scale outputs distinct from verified robot
physical units. This subtask supplies no throughput or latency claims.

The later `native-output-session-002` retry began only after all H20 GPUs were
observed idle. It again ended on an external-owner interruption: own container
PID3203316/NVML749656 was terminated (-15) when foreign PID749888 appeared,
before complete model computation. Controller SHA256
`4d02756f526a81ba6f69395bfd83851db96b365830020dac801d6a88fd9104e0` is retained
under `openpi-inventory/h20-openpi-native-output-session-002/`. Neither this
retry nor CPU preparation closes the new dual-output native acceptance gate.
Pi05 H20 asset transfer began separately with local controller PID1437803,
four capped rsync streams and full local/range re-verification; see
`openpi-inventory/h20-pi05-local-push-001/report.json`. It is still a transfer,
not a verified remote checkpoint or a pi05 v2 model execution.

## Pi05 RTX V2 Pruned Capture, September 7

The new local run `openpi-inventory/local-pi05-v2-pruned-001/` completed all six
owner-gated processes (controller PID1451461): actual official pi05_aloha
reference, ten-step partition, four persisted Regions, each original/pruned
example replay in an independent worker, then a fresh complete pruned IR worker
PID1458469. The physical device was RTX3060 UUID
`GPU-ce878329-6b58-5666-292d-94185f5e5585`, mapped only to logical cuda:0.
Its reference is newly computed on this RTX, not inherited H100/H20 output.
The checkpoint remains SHA256 `7ed2fb2f91b084efc387022383035f6908aa620f6bfa0da52204a4e52e7851b6`;
the actual ALOHA episode0/frame0 manifest is
`45200efb18c9c877058a15121e474f077f72d163e32a88148dd7d7c52e4d25d9`.
Original raw observations, official preprocessing and unchanged saved noise were
used. All stages observed the full 22-field Python numerical-context v2 in the
isolated Torch2.10.0+cu128 / patched official transformers4.53.2 environment.

Complete normalized FP32 `[1,50,32]` and official CPU-native ALOHA FP64 `[50,14]`
outputs are bitwise identical to this new official reference: 1600 and 700
values respectively, MSE/max-abs 0. Native cosine is 0.9999999999999999 from
floating-point metric arithmetic; complete output bytes, not cosine, establish
this run's exactness. No robot calibration, board run or no-Python deployment is
claimed by this evidence. Canonical postprocessing remains Python at this gate.

Actual allocator peaks (bytes) are:

| Stage | Allocated | Reserved |
| --- | ---: | ---: |
| Official reference + capture/persist | 7,683,556,352 | 7,809,794,048 |
| Prefix original/prune/reload worker | 7,654,700,544 | 7,782,531,072 |
| Full pruned IR worker | 7,090,468,352 | 7,235,174,400 |

The prefix worker released its original and in-memory pruned EP owners before
serialized pruned reload; live allocated storage was then 29,222,400 bytes.
This is measured staged ownership, not evidence that two full model copies fit
in 12GB. The new shared `unused_state` schema2 pass removes unused lifted state,
preserves recursive computation/ports, and seals retained tensor identities.

Independent CPU audit `local-pi05-v2-pruned-independent-002.json`, SHA256
`0c1a10574529db5e569e367745263b616f29974f147fbb049bd0bec7421e81ea`, rechecks the
entire checkpoint, actual local upstream source, original dataset files, all
frozen source files, monitor owners, every serialized complete Region example
and the complete action byte streams. Its new create-only derived capture is
`local-pi05-v2-pruned-001/derived-capture-001/capture.json`, SHA256
`830ef59473767cc48c45c4709383c2a9c962d98403f5714ba3fefb60c40bd062`.
The original persisted report remains unchanged and incomplete on its own.
Derived graph identity explicitly hashes the sealed recursive graph map and
signature; it is not mislabeled as the original frontend graph digest.

CPU test evidence `openpi-pruned-cpu-004.xml` contains 47 passing focused tests.
These are synthetic contract tests, distinct from the real RTX run. Earlier
002 join tests failed because recursive graph SHA maps were initially treated
as strings; 003 fixed that contract. The independent checker also retains its
001 failure on a zero-dimensional Tensor byte view; 002 reshapes before the
byte view and performs the complete independent audit. Neither checker failure
changes the original model outputs. The earlier CPU prepare worker succeeded,
but its controller's wrong schema-string postcheck remains a failure, with a
separate correctly parsed independent preparation report.

Next, `local-pi05-v2-pruned-aoti-001/` controller PID1462377 is running the
separate-owner EP trace, four public aten-preserving compilations, and a fresh
pure-AOTI whole IR audit. This in-flight stage is not a completed AOTI or C++
acceptance result. It never loads all EP and all AOTI weights simultaneously.

The full current OpenPI CPU suite subsequently passed 200 tests with one explicit
real-model opt-in skip (`openpi-all-cpu-20260907-002.xml`). Its preceding 001 run
retains one failure because the new generic runner bootstrap placeholder was
not yet in the owned output renderer's strict token set. The local renderer now
explicitly leaves that marker empty for its separately generated initializer;
no public template/runtime or frozen experiment was modified.

H20 dual-output session003 again stopped on a new foreign GPU7 owner, before
model computation (own PID3228465/NVML1024118, foreign PID1031837). Its complete
control/log archive is `h20-openpi-native-output-session-003/`; the 003 attempt
does not close pi0's full native-action C++ gate. No immediate retry is queued.

## September 7: pi05 RTX AOTI and Normalized Native Closure

The previously in-flight `local-pi05-v2-pruned-aoti-001/` completed successfully.
Its fresh EP-only trace and separate AOTI-only process compare all 13 actual
invocation calls, including complete inputs and outputs. All normalized 1600
FP32 values and the separately CPU-postprocessed 700 FP64 values are bitwise
equal to the new same-RTX official reference. The four public aten-preserving
packages bind pre-AOT, post-grad and package-pass audits with actual v2 numerical
observations before and after compilation. Audit SHA256 is
`300ddb2431c67ef01aafd8a6e2df647c0100d98ae843721aadc55fcca7e3499a`.
Independent CPU re-verification of 142 frozen source files, artifacts and all
13 call archives passed in `local-pi05-v2-pruned-aoti-independent-001.json`,
SHA256 `a5d79fd55b17f330f53573afd2dc3b8a5b1add9ece04fc246f1585f07865d1a1`.
The AOTI process sampled 7022 MiB in NVML. Its 133848576-byte PyTorch allocator
peak excludes externally managed AOTI constants and is not full-model memory.

`local-pi05-native-public-001/` preserves the separate CPU build-only report as
`built-unexecuted`, then prepares the public typed native runner. Actual native
PID1514714 completed one warmup and three diagnostic calls on the RTX3060;
`local-pi05-native-diagnostic-001/report.json`, SHA256
`2c344ca5618181bed12f674317044994a540144cab2069110a2643a70e2f9ca0`,
verifies all 1600 normalized values bitwise against both official and same-AOTI
references on every call. MSE/max-abs are zero. Actual process maps identify the
numerical provider DSO and no libpython. The frozen public runner performs the
explicit worker initialization before tensor/Session creation, with full v2
require-current bindings. Sampled process peak is 7020 MiB. Python search paths
are invalid and `LD_LIBRARY_PATH` is empty for this native process. This is four
diagnostic calls of one recorded frame, not a formal throughput/CDF experiment.

These normalized C++ outputs do not include inverse normalization. The new
`local-pi05-native-output-cpu-001/report.json`, SHA256
`84deb13fe11073a7180128c760d93655afe927104f3c40a225c7f3f40d09786c`,
independently passes the actual official processor chain, pure-Tensor eager,
strict export and reload for all 700 FP64 values, with a checked finite predicate.
This CPU stage constructs no model and is not native C++ output validation.
The reusable `openpi_output_capture` Adapter tool dispatches by processor type;
pi05's selected normalization assets and official configuration determine its
semantics. CUDA processor capture and the joined two-output C++ bundle remain
pending. No robot calibration or physical-units verification is claimed.

The create-only `local-pi05-typed-native-join-001.json` binds the unchanged
build-only report to the independent typed native diagnostic, prepared files,
full raw output and actual process/provider evidence. SHA256 is
`520c9652844be59cd244e2dc508be4d80b9c27c9616ef79dcb5a65d664fc0ead`.
It supplies a distinct validated source for the output extension, without
rewriting either historical report. Focused synthetic tests are separate:
30 passing output capture/lowering tests and 40 passing join/output-IR tests.

After the H20 became idle, pi0 session004 reached the real AOTI loader but
failed before complete direct execution: the upstream legacy package loader
hardcodes Linux `/tmp/XXXXXX`, and the overlay was full. A separate controller
issue configured a NAS TMPDIR that did not exist, but merely creating it cannot
redirect this C++ loader. This is verified against the pinned
[PyTorch v2.10 package loader](https://raw.githubusercontent.com/pytorch/pytorch/v2.10.0/torch/csrc/inductor/aoti_package/model_package_loader.cpp).
The complete failure is retained; no shared `/tmp` or foreign files were
removed. A future attempt needs an explicitly verified extracted-package path
or another available execution resource, not an assumed TMPDIR fix.
