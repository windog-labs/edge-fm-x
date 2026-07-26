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

## CLI

```bash
vlaforge verify examples/smolvla/program.vla
vlaforge run examples/smolvla/program.vla \
  --adapter smolvla-fixture \
  --trace /tmp/vlaforge-smolvla-trace.json
vlaforge compile examples/smolvla/program.vla \
  --output /tmp/vlaforge-bundle
```

The examples in the repository are deterministic fixtures. A fixture reaching
generated C++ is `fixture-L4`; it is not evidence that a real checkpoint has
reached L4.

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
