# VLAForge v0.1 VLA Profile

The IR serves VLA inference and control deployment. It is not a general
reactive-programming, workflow, or distributed-scheduling IR.

## Required business patterns

Every public construct must support at least one of these patterns:

1. **Observation snapshot** — image, language, and proprioception sampled at a
   known epoch with a maximum age.
2. **Persistent policy state** — action queue, RNG state, history, or a cache
   that the real source retains across policy invocations.
3. **Pure model regions** — vision/language prefix, autoregressive decode,
   action expert, solver step, or action decode.
4. **Bounded action generation** — a finite autoregressive or flow/diffusion
   loop with explicit carried tensors.
5. **Action visibility** — state and an action become externally visible only
   after validation and commit.

## Core v0.1 operations

- `vla.sample_input`
- `vla.txn.begin`
- `vla.state.read`
- `vla.snapshot.value`
- `vla.invoke`
- `vla.if`
- `vla.for`
- `vla.yield`
- `vla.state.stage_write`
- `vla.validate`
- `vla.action.create`
- `vla.txn.commit` / `vla.txn.abort`
- `vla.action.publish`
- `vla.reset`
- `vla.return`

`vla.if` is needed for action-queue refill versus reuse. `vla.for` covers both
bounded denoising and bounded action-token generation.

## Compatibility-only operations

`vla.while`, `vla.async`, and `vla.await` have minimal reference semantics
because the original goal requested them. They are not active paper
contributions and must not grow into general scheduling infrastructure unless
a real SmolVLA, OpenVLA, or π0 path cannot be represented by the core profile.

## State admission rule

A model adapter may declare a persistent `StateSlot` only when the source
retains the value across policy invocations.

- SmolVLA `select_action`: action queue is persistent.
- SmolVLA prefix KV: local to one action-chunk inference; keep it as SSA.
- SmolVLA solver sample: loop-carried SSA, not persistent state.
- OpenVLA `predict_action`: no cross-tick policy state in the reference path.
- Transformer KV inside one `generate()` call: region/loop-local unless the
  deployment source explicitly retains it across calls.

Inventing previous-action, prefix-cache, or history state merely to exercise an
IR feature is prohibited.

## New-operation admission rule

A new operation requires:

1. a source location in a supported real model;
2. an explanation of why composition of existing core operations is
   insufficient;
3. verifier semantics;
4. positive and negative tests;
5. evidence that the operation is not model-named.

Metadata and TensorRegion names may describe model methods. Core opcodes may
not contain model names.
