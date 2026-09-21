# pi0.5 Reference Version Discrepancy

2026-09-09 05:40 UTC. H100 frame-0 pi0.5 input arrays are byte-identical to
local pi0.5 series frame 0 (image/token/state/noise hashes match), and both
report `pi05_aloha`, `num_steps=10`, dataset revision
`e82d8b40b8ac66c0b40273dd80a077dfc40b732e`.

Despite that, the H100 normalized official reference hashes to
`9fe3c67f...`, while no local 16-frame normalized reference matches. Therefore
the local 16-frame pack and H100 single-frame pack cannot be merged into one
same-platform formal matrix without resolving the reference/checkpoint
generation discrepancy.

## Evidence

- H100 frame0 image/state/noise SHA matches local frame0.
- H100 normalized reference raw SHA `9fe3c67fa042...`.
- Local 16-frame normalized reference hashes are all different.
- H100 native no-Python output is byte-exact to the H100 reference.

## Consequence

H100 pi0.5 native single-frame pass stands. Formal 16-frame H100 replay must
use H100-generated references for all frames, not the local pack, until the
generation source difference is identified.

## Input Compatibility

Local `local-pi05-series-001/reference/{0..15}/prepared.npz` files contain the
same keys/shapes/dtypes as H100 `prepared_inputs.npz`
(`image_0/1/2`, masks, `language_tokens/mask`, `state`, `noise`). So the issue
is not a missing or incompatible input schema. The correct recovery route is
to run the H100 reference adapter against these 16 raw observations/noise
packs, not to feed prepared arrays into a different reference code path.
