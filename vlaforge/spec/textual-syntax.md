# VLAForge Textual Invocation IR v0.2

The parser-round-trippable interchange form is:

```text
!vlaforge.ir 0.2
<canonical structural payload>
```

The payload is deterministic UTF-8 JSON with sorted keys. It serializes the
complete executable Semantic IR:

- statically typed input ports with stable IDs, required/optional/default
  rules, device/layout/alignment, and bounded-profile metadata;
- named output ports, stable IDs, and transactional output groups;
- authoritative persistent state and episode-reset policy;
- typed pure TensorRegions and artifact metadata;
- passive invocations, SSA values, operations, attributes, and nested blocks.

It contains no physical clock, tick, deadline, middleware endpoint, runtime
callable, or model weight.

Canonical serialization requirements:

- declaration and semantically ordered list order is preserved;
- mapping keys are sorted;
- enum values use stable string spellings;
- every IR type carries a `kind`;
- operation results include full types;
- locations are optional and do not affect semantics;
- input/output IDs are contiguous declaration IDs and form part of the schema;
- callables and artifact bytes are referenced by immutable deployment
  contracts, never embedded into Semantic IR.

Example excerpt:

```text
!vlaforge.ir 0.2
{
  "inputs": [
    {
      "alignment": 64,
      "device": "cuda",
      "id": 0,
      "name": "camera_history",
      "payload": {
        "dtype": "f16",
        "kind": "tensor",
        "layout": "nchw",
        "shape": [4, 3, 224, 224]
      },
      "required": true
    }
  ],
  "invocations": [{"name": "run", "body": {"arguments": [], "operations": []}}],
  "module": "robot_policy",
  "outputs": [
    {
      "group": "policy",
      "id": 0,
      "name": "action_chunk",
      "payload": {
        "dtype": "f32",
        "kind": "tensor",
        "layout": "contiguous",
        "shape": [8, 7]
      }
    }
  ],
  "regions": [],
  "schema": "0.2",
  "states": []
}
```

`print_module(parse_module(text))` is byte-stable for canonical input. The
separate `vlaforge.io_schema/2` digest is embedded into the bundle, generic C
ABI, and model-specific C++ wrapper so a model upgrade cannot silently rebind
an old host integration.

