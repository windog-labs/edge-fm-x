# Invocation IR v0.2 Necessity Ablations

- Gate passed: `true`
- Scope: adversarial deployment-contract tests; not vehicle safety proof

| Removed contract | Adversarial case | Result |
|---|---|---|
| missing InputRevision is assigned a fresh identity per Bind/Run | same borrowed tensor is bound twice without a revision | fault detected |
| bundle/session I/O schema digest match | caller initializes a session with a stale schema | fault detected |
| state staging and named outputs share one validated transaction | output validation fails after state staging | fault detected |
| Session returns only committed output groups | an Adapter inserts an action.publish opcode | fault detected |

These tests justify input identity, schema binding, and atomic state/output semantics. They do not claim sensor synchronization, runtime scheduling, or vehicle safety.
