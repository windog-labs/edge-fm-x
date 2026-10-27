# VLAForge Invocation Compiler v0.2

VLAForge compiles an externally invoked, stateful VLA program into a
schema-checked, no-Python C++ Session. TensorRegion implementations are
provided by verified external artifacts through the RegionExecutable ABI;
there is no legacy engine/model/operator hierarchy in this repository.

Canonical design:

- [Invocation IR v0.2](../doc/vlaforge_invocation_ir_v0_2.md)
- [Development plan](../doc/vlaforge_development_plan.md)
- [Paper design](../doc/vlaforge_paper_design.md)
- [Model evidence matrix](../doc/vlaforge_model_adaptation_matrix.md)
- [Model cards](../doc/model_cards/README.md)
- [Generic Python Invocation Interface](spec/python-invocation.md)
- [Device-Preserving ATen Regions](spec/torchscript-aten.md)
- [A100/H20 high-memory handoff](../doc/vlaforge_high_memory_handoff.md)
- [J6P VLM: two complete HBMs and cache ABI](../doc/j6p_vlm_deployment_and_kv_cache.md)

## Scope

The host pushes static Tensor/Scalar inputs and calls `Session::Run()`.
VLAForge does not acquire or synchronize sensors, maintain rates/deadlines,
drop frames, interact with ROS/Cyber, or publish actions.

Invocation IR explicitly models:

- input identity through optional `InputRevision`;
- authoritative versioned state;
- pure TensorRegion calls;
- structured branches and bounded loops;
- exact derived cache contracts;
- atomic named output groups.

## Development setup

```bash
cd vlaforge
python -m pip install -e '.[test]'
python -m pytest -q
```

Build the C/C++ runtime and ABI tests:

```bash
cmake -S . -B build-v02 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON
cmake --build build-v02
ctest --test-dir build-v02 --output-on-failure
```

The clean generated-C++ tests intentionally run with invalid
`PYTHONHOME/PYTHONPATH` and check `ldd` for Python dependencies.

### Native TensorRT

Enable `VLAFORGE_BUILD_TENSORRT_BACKEND=ON` to load target-built engines without
a Python model runtime. Engines requiring installed NVIDIA standard plugins
also need `VLAFORGE_TENSORRT_STANDARD_PLUGINS=ON`, `NvInferPlugin.h` and
`libnvinfer_plugin` from the same target SDK. Plugin registration occurs before
engine deserialization; this option defaults to OFF and does not provide
arbitrary custom operators or certify a model's numerical outputs.

The host-side `tools/fuse_onnx_layernorm_trt_plugin.py` conservatively rewrites
typed, static last-axis FLOAT LayerNorm dataflow to the installed `LayerNorm`
plugin. Its CLI requires `--input`, new `--output` and new `--report` paths;
`--standard-reference` emits a standard ONNX semantic shadow instead. Weights,
epsilon and unmatched computation are preserved. Inspect target plugin/version,
actual FP32 plugin IO and complete-model output fidelity before adoption.
Neither a successful rewrite nor a successful engine build is a model pass.

`VLAFORGE_TENSORRT_BFLOAT16_PLUGIN=ON` optionally builds the TensorRT 8
`VLAForgeBFloat16Round/1` and `VLAForgeBFloat16MatMul/1` plugins and registers
them before engine loading. The first rounds FLOAT carriers to BF16 values;
the second uses cuBLAS BF16 operands/FLOAT accumulation and returns BF16-rounded
results in FLOAT storage. Neither adds native BF16 TensorRT tensor bindings.
MatMul accepts static positive rank-2..8 inputs with ONNX batch broadcasting.
It owns a cuBLAS handle per clone/context, uses declared TensorRT workspace,
and creates no provider-owned CUDA allocations or handles inside enqueue.
Library-internal resource behavior is a separate measurement obligation.
Set `CMAKE_CUDA_COMPILER` and `CMAKE_CUDA_ARCHITECTURES` explicitly for the target.
The standalone DSO also supports `trtexec --plugins=<library>` during engine
construction. Keep that exact DSO with engines that depend on it.

`tools/lower_onnx_bfloat16_carrier.py` promotes a flat functional BF16 graph
to FLOAT carriers with explicit input/arithmetic rounding. `--reference` emits
portable BF16 Cast boundaries instead of plugins. Integer types are unchanged;
unsupported casts/control flow fail rather than silently changing semantics.
The opt-in `--native-bf16-matmul` rewrites only proven all-BF16 source MatMul
nodes; it does not reinterpret FLOAT operations, Conv or Gemm. Static shape
tensor values are propagated for shape proof, without guessing dynamic inputs.
The portable shadow helper expands reviewed MatMul plugins as FLOAT
accumulation between BF16 input/output rounding boundaries. Provider tests and
successful graph rewriting are not complete-model accuracy or speedup evidence.
`tools/prepare_trt_precision_profile.py --dense-output-type fp16` explicitly
requests FP16 dense outputs when needed by target implementations, but adds
FP16 result rounding before FLOAT consumers. The default remains FP32 outputs.
Neither carrier lowering nor a precision profile certifies compute precision,
input/intermediate ranges or complete-model numerical equivalence.

`tools/export_torchscript_region_onnx.py` retains the native FLOAT internal
arithmetic of 16-bit GELU/SiLU/var/var_mean/std and casts their results back to the original
dtype. This avoids accidental extra rounding in legacy ONNX decompositions;
it is not whole-model FP32 execution. Symbolic overrides are scoped/restored,
and existing custom policies are rejected rather than overwritten. Export
reports distinguish checker success, static callback counts and runtime parity.

LayerNorm rewriting accepts both proven primitive dataflow and standard ONNX
`LayerNormalization`, without dropping observable statistics. A known rounding
plugin may be replaced by Identity only in a shape-proof shadow; the actual
numeric graph keeps its rounding. External-weight rewrites require sibling
output graphs and preserve/hash their payloads. Producer partitioning supports
external metadata without loading all weights into a protobuf; its CLI copies
payloads instead of using hard links rejected by modern ONNX checkers.

`project_region_inputs` adapts plain pure, uncached Region calls to explicit
retained artifact-input indices, including zero inputs. It preserves every
result, external port and loop/state consumer; the caller must supply the
pruned artifact's semantic evidence. It neither guesses unused arguments nor
certifies that dtype specialization preserves a model.

Precision profiles accept exact `--fp32-dense-node` exceptions for independently
observed range requirements; other dense nodes keep their declared FP16 policy.
`tools/lower_onnx_float_predicates.py --operator IsInf` can retain native IsNaN
support and bypass only proven FLOAT-to-DOUBLE widening used for classification.
Actual full-range target conformance is mandatory: layer FP32 constraints alone
did not preserve IsInf bounds in one mixed-precision TRT 8.5 test. An isolated,
independently verified FP32 check is not whole-model FP32 fallback.

### Native Input Preparation

`tools/materialize_native_observations.py` generates a declared C++ host input
pipeline. Supported operations include byte scaling, state standardization,
explicit scatter/divide and optional BF16-RNE in FLOAT storage, HF tokenization
with BOOL or INT32 masks, and ordered image views to normalized CHW FLOAT.
Every invocation prepares fresh inputs. Channel order, resize/padding, numeric
rounding and truncation are explicit contracts, not inferred model defaults.

The optional `VLAFORGE_BUILD_PILLOW_IMAGE=ON` backend delegates image resampling
to a pinned, unmodified Pillow C imaging core. Supply absolute paths through
`VLAFORGE_PILLOW_CORE_LIBRARY` (PIC static library),
`VLAFORGE_PILLOW_CORE_INCLUDE` (matching `Imaging.h`), and
`VLAFORGE_PILLOW_PYTHON_INCLUDE` (matching platform headers, including
`Python.h` and `pyconfig.h`). These headers are compile-time only; the backend
does not link or start a Python interpreter. Keep the core source/archive hash,
build flags and license with deployment evidence. Source qualification must
compare every prepared tensor against the model's frozen processor outputs;
the module's smoke tests do not certify an arbitrary processor configuration.

## CLI

The deterministic interface examples are Python modules:

- `examples/iterative_frontend.py` demonstrates continuous and autoregressive
  InvocationBuilder programs;
- `examples/j6p_vlm_kv_cache.py` shows the J6P VLM interface for two complete HBMs;
- `examples/external_bev_plugin/` contains the C++ plugin fixture.

The CLI operates on generated program or bundle paths produced by the build
tools; the old `examples/smolvla/program.vla` path is no longer part of the
repository.

## Real-model evidence

Opt-in frontend/eager gates remain separate from the offline suite:

```text
tools/run_real_smolvla.py
tools/run_real_openvla.py
tools/audit_real_smolvla_frontend.py
tools/audit_real_openvla_frontend.py
tools/compile_real_aoti_exports.py
tools/audit_cuda_aoti_region.py
```

These tools require an explicitly supplied local checkpoint/revision and never
download weights as part of the default test suite. Only evidence produced
through the current Invocation IR, Plan, and passive Session ABI belongs in a
release or paper claim.

## Extension rules

Prefer Adapter/template composition, then typed Region/backend plugins, then
static I/O/validator/cache/variant extensions. A new core opcode is allowed
only when a new cross-artifact control or state semantic cannot be expressed
otherwise, and requires verifier, reference, Plan, codegen/runtime,
serialization, and tests.
