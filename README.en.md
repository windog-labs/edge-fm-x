# VLAForge

English | [中文](README.md)

VLAForge is a stateful invocation compiler for robot and autonomous-driving
VLA deployment. It compiles an Adapter-declared model invocation into a
schema-verified, no-Python C++ Session. The host prepares inputs and calls
`Session::Run()`.

## Scope

VLAForge models only deployment semantics:

- Tensor/Scalar inputs with optional `InputRevision`;
- authoritative persistent state separated from invalidatable derived cache;
- TensorRegions, structured branches, and bounded loops;
- exact reuse contracts;
- atomically committed named output groups;
- a stable C ABI, model-specific typed C++ wrappers, and a Region plugin ABI.

It does not acquire or synchronize sensors, maintain periods, deadlines, or
frame rates, integrate ROS/Cyber, publish vehicle commands, or implement a
vehicle safety layer.

## Repository layout

```text
vlaforge/
  python/vlaforge/   Semantic IR, compiler, Plan, codegen, deployment
  runtime/           no-Python C/C++ runtime
  backends/          AOTInductor/TorchScript Region backends
  include/           stable C/C++ ABI
  examples/          deterministic architecture fixtures
  tests/             Python, C++, CUDA, and real-model gates
  tools/             capture, artifact, benchmark, and paper audits
doc/
  model_cards/       model adaptation cards
  reports/           reproducible experiments and claim-evidence
```

The retired EdgeFM engine, LLM/VLM operators, Qwen examples,
TensorRT/CUTLASS submodules, and custom CUDA kernels are not part of this
branch or the VLAForge build graph.

## Python development

```bash
python -m pip install -e 'vlaforge[test]'
python -m pytest -q vlaforge/tests
```

## C/C++ runtime

Build the CPU runtime and ABI tests:

```bash
cmake --preset host-cpu
cmake --build --preset host-cpu --parallel
ctest --preset host-cpu
```

Build the CUDA AOTI backend with the active PyTorch installation:

```bash
export VLAFORGE_TORCH_CMAKE_PREFIX_PATH="$(
  python -c 'import torch; print(torch.utils.cmake_prefix_path)'
)"
cmake --preset host-cuda
cmake --build --preset host-cuda --parallel
ctest --preset host-cuda
```

## Evidence boundary

Real SmolVLA and DiffusionDrive have generated no-Python C++ Host-CUDA
Sessions; OpenVLA has real Host-CUDA L3 evidence. Paper-grade statistics,
ablations, Model Cards, reproducibility manifests, and release gates are under
[doc/reports](doc/reports/README.md).

Performance claims are limited to RTX 3060 (`sm_86`) / CUDA 12.8. Orin,
cross-GPU evaluation, and second-machine reproduction are optional extensions.

Canonical documents:

- [Invocation IR v0.2](doc/vlaforge_invocation_ir_v0_2.md)
- [Development plan](doc/vlaforge_development_plan.md)
- [Paper design](doc/vlaforge_paper_design.md)
- [Model adaptation matrix](doc/vlaforge_model_adaptation_matrix.md)
- [Model adaptation cards](doc/model_cards/README.md)
