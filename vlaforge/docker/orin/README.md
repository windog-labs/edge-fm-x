# VLAForge Jetson Orin TensorRT backend

This package is the prebuilt JetPack/ARM64 deployment substrate for VLAForge.
It contains:

- the installed VLAForge runtime and TensorRT RegionExecutable SDK;
- an AArch64 on-device backend smoke executable;
- a validator/launcher for a real generated model bundle;
- driverless Docker compile evidence and its build report.

It does not contain a real model TensorRT engine. TensorRT engines are tied to
the TensorRT/CUDA/SM target and must be produced for the selected model and the
JetPack line used on the Orin.

## Build on the development host

The host needs Docker and registered `arm64` binfmt:

```bash
vlaforge/scripts/orin/build_backend.sh \
  --output "$HOME/VLAForge/orin-backend"
```

The build uses `nvcr.io/nvidia/l4t-jetpack:r36.4.0` and verifies:

1. VLAForge runtime and TensorRT backend ARM64 compilation;
2. core runtime CTest in the driverless container;
3. installed SDK discovery and consumer linkage;
4. backend-aware generated TensorRT Session compilation;
5. AArch64 ELF and no-Python dynamic dependency boundaries.

The L4T Docker image has no real Orin GPU/driver stack. It cannot establish
engine deserialization, `enqueueV3`, numerical parity, latency, memory, power,
or thermal evidence.

For a Compile Bundle that already contains real TensorRT 10/SM87 engines and
was generated with `backend="tensorrt"`, cross-compile and re-verify its runner:

```bash
vlaforge/scripts/orin/cross_compile_bundle.sh \
  --bundle /path/to/input-bundle \
  --output "$HOME/VLAForge/model-bundle-orin"
```

The input bundle is never modified. The copied output receives an AArch64
runner, updated manifest hashes/toolchain provenance, and
`metadata/orin_cross_compile.json`. Engine execution remains an on-device gate.

## First run on a Jetson Orin

Copy the extracted `delivery/` directory to an Orin running JetPack r36.4,
then execute:

```bash
cd delivery
bin/run_backend_smoke_on_orin.sh
```

The smoke builds a one-layer identity TensorRT engine on that device, loads it
through the VLAForge RegionExecutable value ABI, binds caller-owned CUDA
buffers, executes `enqueueV3`, synchronizes, and checks the output. It also
runs with invalid `PYTHONHOME`/`PYTHONPATH`.

Expected final line:

```text
TensorRT Region on-device smoke passed
```

## Run a real generated model bundle

A real bundle must contain target-specific SM87 TensorRT engines, verified
artifact hashes, `bundle.json`, and an AArch64 generated runner:

```text
model-bundle/
├── artifacts/*.engine
├── bin/vlaforge_generated_runner
├── bundle.json
├── generated/
└── metadata/
```

Validate without invoking the model:

```bash
delivery/bin/run_bundle_on_orin.sh --dry-run /path/to/model-bundle
```

Then run it:

```bash
delivery/bin/run_bundle_on_orin.sh /path/to/model-bundle [runner arguments...]
```

The Session is passive: the bottom-software process pushes prepared
Tensor/Scalar inputs and calls `Run`. VLAForge does not collect or synchronize
sensors, schedule periodic ticks, publish middleware messages, or perform
vehicle safety control.

## Supported backend contract

- TensorRT 10 serialized engines through `enqueueV3`;
- CUDA device selection and exact SM target check (`sm_87` for Orin);
- TensorRT major-version variant check;
- named engine inputs/outputs bound in exported engine I/O order;
- F32/F16/BF16/I32/I64/BOOL/U8 linear tensors;
- static or bounded dynamic input shapes;
- device tensors plus explicit TensorRT host-location tensors;
- caller-owned, borrowed-until-Run-returns memory;
- no silent dtype, shape, layout, device, or alignment conversion.

Scalar/POD model inputs must be tensorized by the Adapter or an external
preprocessing Region before entering a TensorRT engine. Non-linear TensorRT I/O
formats are rejected; an explicit preprocessing Region should perform any
packing or layout conversion.
