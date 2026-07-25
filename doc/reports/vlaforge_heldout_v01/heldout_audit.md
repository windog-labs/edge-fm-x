# VLAForge frozen-core held-out audit

Status: **passed**. Frozen core match: **True**.

This report combines pinned source audit (L0) with deterministic executable fixtures (L1). It is not checkpoint, artifact, or real generated-C++ evidence.

| Model | Source | Fixture | Template | Runs | Core op delta | Semantic/Plan parity |
|---|---|---|---|---:|---:|---|
| Octo | L0-pinned-source-audit | L1-deterministic-executable-fixture | DiffusionPolicy | 3 | 0 | outputs/state/trace exact |
| GR00T N1.7 | L0-pinned-source-audit | L1-deterministic-executable-fixture | MultiEmbodimentDiT | 2 | 0 | outputs/state/trace exact |
| AutoVLA | L0-pinned-source-audit | L1-deterministic-executable-fixture | AutoregressiveTrajectory | 3 | 0 | outputs/state/trace exact |

## Frozen core

- Freeze revision: `766e27bdfb34c3311dda7d444862b7f95d05c7b8`
- Current revision: `ec9a56c3a5e00c75a803757b96a78bdc33eb5aee`
- Combined fingerprint: `cc2d1b63e2d6cbcd65935b37d69b5f18fae4d2d177c7026a69c6e78f5c80ae6d`
- Core working-tree changes: none

## Per-model evidence

### Octo

- Upstream revision: `241fb3514b7c40957a86d869fecb7c7fc353f540`
- Adapter: `build_octo_like_fixture` (181 LOC)
- Generic opcodes: `vla.for, vla.input.read, vla.invoke, vla.output.create, vla.output.group, vla.return, vla.txn.begin, vla.txn.commit, vla.validate, vla.yield`
- Control: 1 bounded for, 0 structured if
- Exact cache events: 1 hit / 2 miss
- Static arena: 256 bytes (saved 320 bytes)
- Output digest: `dd0d9e73115fdbd31cff398778dc78ed26c38caece70b881f2c20a5e833d9468`
- Unsupported: real checkpoint L2, artifact L3, real C++ L4

### GR00T N1.7

- Upstream revision: `9c7e746b2cd37a810070a98ef41d290a07e806c2`
- Adapter: `build_groot_n1_like_fixture` (186 LOC)
- Generic opcodes: `vla.for, vla.input.read, vla.invoke, vla.output.create, vla.output.group, vla.return, vla.txn.begin, vla.txn.commit, vla.validate, vla.yield`
- Control: 1 bounded for, 0 structured if
- Exact cache events: 1 hit / 1 miss
- Static arena: 320 bytes (saved 320 bytes)
- Output digest: `238f4d917b8b903d223f0c604fc144b74b1baa2b61bb4b3d445da800dadc868d`
- Unsupported: real checkpoint L2, artifact L3, real C++ L4

### AutoVLA

- Upstream revision: `ba34eed74ce6729e7986592d0e66cbaca397b4fa`
- Adapter: `build_driving_ar_fixture` (240 LOC)
- Generic opcodes: `vla.for, vla.if, vla.input.read, vla.invoke, vla.output.create, vla.output.group, vla.return, vla.txn.begin, vla.txn.commit, vla.validate, vla.yield`
- Control: 1 bounded for, 1 structured if
- Exact cache events: 1 hit / 2 miss
- Static arena: 256 bytes (saved 384 bytes)
- Output digest: `d035e3ef59acac2488bb09d40b40a12b683dc79749f86de3fe5db0d04d12da71`
- Unsupported: real checkpoint L2, artifact L3, real C++ L4
