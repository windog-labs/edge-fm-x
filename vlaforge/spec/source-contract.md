# VLAForge Restricted Source Contract v0.2

VLAForge captures model computation, not arbitrary Python applications.

## Adapter declaration

An Adapter declares:

- static Tensor/Scalar InputPorts and named OutputPorts;
- required/optional/default values and bounded profiles;
- authoritative state and episode reset behavior;
- pure TensorRegion boundaries and artifact variants;
- bounded loops and structured branches;
- validators and transactional output groups.

The declaration generates Semantic IR ports, stable IDs, I/O schema digest,
generic C ABI, and model-specific typed C++ wrapper.

The Adapter does not read sensors, synchronize timestamps, assemble a physical
schedule, publish commands, or own middleware messages. Bottom software
converts its objects into TensorView/ScalarValue before binding.

## Captureable TensorRegion

A captureable callable must:

- have declared Tensor/Scalar input/output types;
- be deterministic for the same explicit values;
- expose persistent state and RNG as explicit values;
- use only invocation-local hidden workspace;
- perform no file/network/middleware/external I/O;
- satisfy static or bounded shape profiles.

Unsupported capture returns a versioned diagnostic; it never silently falls
back to eager Python in a no-Python bundle.

## Evidence levels

- L0: pinned source/paper contract mapping.
- L1: deterministic executable fixture.
- L2: real frontend capture and eager parity.
- L3: real compiled artifact parity.
- L4: generated no-Python C++ Session parity.

Fixture-L4 is labelled separately and never counts as real-checkpoint L4.
Each Model Adaptation Card records upstream revision, checkpoint/license,
I/O/state/cache partition, Region split, dynamic profile, Adapter LOC, new core
op count, unsupported items, evidence paths, memory, and performance.
