# VLAForge IR Foundation

This directory contains the executable Python reference semantics for the
stateful/temporal VLA IR described in
[`../doc/vlaforge_development_plan.md`](../doc/vlaforge_development_plan.md).
It is intentionally isolated from EdgeFM's engine/model/operator hierarchy.

The first milestone implements:

- versioned persistent state and logical epochs;
- SSA-style tensor regions and structured control flow;
- transaction-scoped state writes and action commit;
- type, state-version, freshness, effect and commit verification;
- a deterministic reference interpreter and trace format;
- liveness, state dependency and bounded physical-slot analyses.

The Python IR is the normative v0 semantics. It is not the final deployment
runtime and does not attempt to replace a tensor compiler.

The intentionally small public abstraction boundary is documented in
[`spec/vla-profile.md`](spec/vla-profile.md). Prefix KV or solver tensors are
not promoted to persistent state unless the real source retains them across
policy invocations.

## Development setup

```bash
cd vlaforge
python -m pip install -e '.[test]'
pytest
```

The default test suite is offline and never downloads model weights. Tests
marked `real_model` are separate evidence gates and must be executed explicitly
before G2 can be claimed.

Real adapters deliberately keep framework internals inside pure TensorRegions:

- SmolVLA exposes prefix preparation and the bounded flow solver while keeping
  its cross-tick action queue explicit.
- OpenVLA exposes deterministic action-token generation and detokenization; it
  has no persistent state in the reference `predict_action` path.

The runners are:

```text
tools/run_real_smolvla.py
tools/run_real_openvla.py
```

Model dependencies and weights are not package dependencies. Each gate runs in
an explicitly pinned external environment.
