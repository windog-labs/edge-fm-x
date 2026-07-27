# External BEV Region plugin

This customer-style shared library implements the stable
`VLAForgeRegionExecutableValueApi`. It receives only statically contracted
Tensor/Scalar values:

- Region 0 packs a `[4,4]` external BEV tensor into four BEV tokens.
- Region 1 consumes BEV tokens, bounded agent features, scalar `valid_count`,
  and a route tensor, then returns trajectory, prediction, and scalar VQA
  outputs.

It does not pull sensors, synchronize timestamps, parse ROS/Cyber/protobuf, or
publish vehicle commands. The bottom-software caller owns those responsibilities
and binds already prepared values to the generated Session.

Build it with the model bundle's exact I/O schema digest:

```bash
cmake -S . -B build \
  -DVLAFORGE_INCLUDE_DIR=/path/to/vlaforge/include \
  -DVLAFORGE_EXPECTED_SCHEMA_DIGEST=<64-hex-digest>
cmake --build build --parallel
```

The bundle records and verifies the `.so` size/SHA256, target, backend variant,
callable ABI, and model I/O schema before `dlopen`. Plugin value descriptors
are borrowed only through `synchronize()`.
