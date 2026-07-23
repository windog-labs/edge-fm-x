# VLAForge Textual IR v0.1

The v0.1 parser-round-trippable form is:

```text
!vlaforge.ir 0.1
<canonical structural payload>
```

The payload is deterministic UTF-8 JSON with sorted keys. It serializes the
complete executable IR: types, clocks, state declarations, tensor-region
signatures, policies, SSA values, operations, attributes, and nested regions.
It is not a deployment manifest.

The explicit magic and schema header prevent a bundle manifest or future
scheduled-plan payload from being parsed as semantic IR.

Canonical serialization requirements:

- all tuple/list order with semantic meaning is preserved;
- mapping keys are sorted;
- enum values use their stable string spelling;
- every IR type carries a `kind`;
- operation results include full types;
- locations are optional and do not affect semantics;
- runtime callables and model weights are never serialized.

Example excerpt:

```text
!vlaforge.ir 0.1
{
  "module": "flow_policy_fixture",
  "schema": "0.1",
  "states": [
    {
      "name": "rng",
      "version_clock": "control",
      "retention": 3,
      "payload": {"kind": "scalar", "name": "i64"}
    }
  ],
  "policies": [...]
}
```

`print_module(parse_module(text))` is byte-stable for canonical input.

An MLIR surface syntax is planned after the Python semantics, verifier rules,
and model coverage stabilize. The canonical payload remains useful as a
versioned interchange and golden-test format.

