# Checked Output Stages

`vlaforge.adapters.shared.output_stage.attach_checked_output_stage` composes a new
pure Tensor Region after an existing complete output. It is model-neutral:
model names, robot dimensions, statistics and scheduling rules are not selected
by this helper or by the runtime. An Adapter supplies the implementation and
its explicit tensor contract. Existing Python -> IR -> Plan -> C++ lowering
handles invocation, device storage and atomic publication.

## Explicit Sources

Every new Region argument has an `OutputStageInput` binding:

- `source-output`: the exact existing complete output port.
- `acceptance`: the existing predicate operand, not a new constant `true`.
- `module-input`: an explicitly named, dominating input read on the same device,
  for processors that also need observed state or another declared input.

The current implementation accepts one invocation and one original output,
one complete source-output binding, exactly one acceptance binding, and the
standard predicate validator / transaction suffix. Unknown validators,
ambiguous reads, shape/type mismatches, SSA collisions and existing output or
Region names are rejected. It does not guess how to rewrite arbitrary control
flow or existing multi-output programs.

## Preserve Acceptance

The Adapter's new Region returns `(complete_output, accepted)`. Its acceptance
must include the incoming predicate by conjunction. A typical tensor wrapper is:

```python
def forward(self, source, incoming_accepted):
    output = self.processor(source)
    accepted = incoming_accepted & torch.isfinite(output).all().reshape(1)
    return output, accepted
```

An opaque Region declaration alone does not prove this equation. Capture/effect
audit, false-incoming-predicate tests, nonfinite tests and complete model output
validation remain required. In particular, a finite cropped output must not
override rejection caused by an invalid discarded source coordinate.

The original output remains available as an auxiliary, and the new output joins
its existing group. Both outputs commit together under the new predicate.
Previous model Regions, loop and state operations, input bindings and scheduler
carry are retained. This is not a new runtime opcode or a model-specific C++
postprocessing implementation.

## New Deployment Identity

Composition changes the public output schema and semantic IR. Compile a new
Plan/certificate, bind contracts to the new IO digest and canonical Region IDs,
and independently validate every complete output. Reusing unchanged weight
archives is allowed only with their exact source/compile identities; reusing
an old deployment certificate or relabeling a build-only report is not.

Reference spaces must be explicit: normalized model output, unified action
storage, official robot-scaled output and physically calibrated robot units are
different claims. Timing that excludes a newly added processor cannot be copied
into the new complete-output latency table.

## Current Evidence

RDT uses `(unified, accepted)` with the unchanged official arm ordering and BF16
gripper scales. Its new H20 C++ Session validates 32 complete dual-output calls
against both official and same-artifact references, with no Python runtime.
Smol's mean/std inverse transform and OpenPI's state-dependent transforms use
the same explicit-source interface; their individual processor tests do not
automatically certify a newly composed full Session.

See `doc/edgefm_vla_goal_status.md` for the current per-model gates. Existing
OpenPI two-input snapshots do not inherit the later acceptance-preserving
three-input interface; they remain useful evidence for the finite inputs they
actually executed, not for the new rejection semantics.
