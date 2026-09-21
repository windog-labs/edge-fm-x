# H100 OpenPI Replay Inventory

2026-09-09 03:40 UTC. Shared H100 run roots contain real pi0 and pi0.5
reference/capture/replay evidence. This inventory is not a five-model
same-platform matrix and not board acceptance.

## H100 Artifacts

Remote roots use the shared `openpi-cuda-20260906` run under the H100
`edgefm-vla-goal-20260906` tree.

- `pi0-recorded-reference-001`, `pi0-recorded-capture-005-image-layout`,
  `pi0-saved-only-replay-006`.
- `pi05-recorded-reference-001`, `pi05-recorded-capture-002-image-layout`,
  `pi05-saved-only-replay-003`.
- AOTI/public dual-output and runtime probe directories for both models.

`pi05-saved-only-replay-003/report.json` reports a real saved-capture replay
with exact full normalized action fidelity on H100 and
`no_python_deployment="not-run"`. This is Python Invocation IR replay, not a
formal no-Python Session or latency CDF.

## Remaining For G3

- The same five-model matrix still lacks H100 formal replay/CDF for
  CogACT/RDT and current H100 no-Python replay for pi0/pi0.5.
- pi0.5 on H20 remains unverified; H20 only has the other models' replay data.
- Existing local RTX pi0.5 formal numbers do not fill H100/H20 cells.
