# ExternalFeature Hybrid fixture

| Field | Value |
|---|---|
| Evidence | source-faithful executable fixture L1 + no-Python generated C++ fixture |
| Real-model claim | none; this is not DriveVLM-Dual checkpoint evidence |
| Core op delta | 0 |
| Adapter | `build_hybrid_external_feature_fixture()` |
| Plugin implementation | `vlaforge/examples/external_bev_plugin/plugin.cpp` |
| Plugin ABI | `vlaforge.region_executable/2`, `shared-plugin/1` |

## Deployment contract

The bottom-software caller prepares and push-binds a `[4,4]` external BEV
tensor, optional bounded `[6,3]` agent features, optional scalar `valid_count`,
and a `[3]` route command. VLAForge does not pull sensors, synchronize frames,
parse middleware objects, or publish a trajectory.

The first shared-library Region converts BEV features to four tokens. The
second consumes those tokens plus agents/count/route and transactionally
returns three named outputs: `[6,2]` trajectory, `[6,2]` agent prediction, and
an `i64` VQA token. There is no authoritative persistent state. The BEV token
is a derived exact cache whose identity follows `InputRevision`, episode,
model/artifact identity, and the state snapshot.

## Dynamic plugin evidence

The generated Session verifies artifact size/SHA256 and the model I/O schema
before `dlopen`, then validates callable ABI, target, backend variant, Tensor
and Scalar descriptors. The integration test covers:

- required/optional/default bindings and borrowed-until-Run-returns inputs;
- typed C++ and generic C ABI output equivalence;
- repeated revision hit and new revision invalidation;
- backend failure preserving the previous committed output, followed by retry;
- `ResetEpisode`;
- invalid schema, target, backend variant, ABI, missing entrypoint, and
  tampered shared library;
- execution with invalid `PYTHONHOME/PYTHONPATH` and no `libpython` link.

The plugin has 280 lines and its model-specific runner has 213 lines. The
generic loader/runtime is shared by all models. Unsupported items are arbitrary
host objects, unbounded inputs, runtime addition of unknown ports, sensor
callbacks, middleware I/O, and vehicle safety logic.

## Evidence paths

- `vlaforge/tests/deployment/test_external_region_plugin.py`
- `vlaforge/tests/cpp/external_region_plugin_smoke.cpp`
- `vlaforge/examples/external_bev_plugin/README.md`

No latency, power, real checkpoint, Orin, or closed-loop claim is attached to
this fixture.
