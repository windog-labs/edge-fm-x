# Native Host Wide Types

Checked 2026-10-22 13:57 UTC. **Host software preparation only; no new Orin
model execution, F64 TensorRT arithmetic support or deployment acceptance.**
The active RDT collector continues to use its original frozen source/binaries.
No running target files, engines, precision profiles or inputs were replaced.

## Common Input Gap

The frozen OpenPI contracts include F64 state and original I64 IDs; native
OpenPI/CogACT outputs also include F64 values. The C ABI already declares
these types, but the native observation materializer and its host buffer
previously accepted only F32/I32/U8/BOOL. Silently narrowing source arrays would
change the contract rather than add support.

The shared native host tools now preserve F64/I64 storage and copy bytes under
the existing explicit `copy` declaration. Allocation checks use the actual
element width before allocating. FLOAT and DOUBLE buffers share nonfinite
validation, BOOL rejects bytes above one, and U8/integers retain their own
semantics. The raw-observation runner performs the corresponding typed checks,
including F64, using only the old buffer field/data interface so frozen input
package headers remain compatible with newly prepared runners.
No model-name dispatch or automatic conversion was added. Existing F32
normalization, image preparation, tokenization and numerical policies remain
unchanged.

## Verification

`artifacts/orin/20261022-08/native-host-types-regression-002.xml` records
**121 passing host tests** across native materialization, raw timing,
complete multi-output checks and Session runner code generation. New tests
compile/run the generated C++ profile and check:

- Byte-exact DOUBLE/INT64 copy, including signed zero, subnormal/large DOUBLE
  values and integers beyond FLOAT/DOUBLE exact-integer ranges.
- Fresh input changes propagate on a second call, with no prepared-input cache.
- F64-to-F32 and I64-to-I32 copy contracts are rejected rather than narrowed.
- DOUBLE NaN/infinities, FLOAT NaN, malformed BOOL and byte-count overflow.
- The actual materializer CLI preserves every source byte in the generated raw
  payloads. These inputs are unit fixtures, not new model observations/results.
- The real runner's finite-check function compiles and executes against the
  original buffer interface, without requiring a new method on frozen headers.

The first 120-test receipt and `source/` snapshot are retained. A further
compatibility test exposed the intermediate runner's dependency on the new
buffer method; `legacy-buffer-compatibility-red-002.xml` records that failure.
The earlier red-001 also contains a missing test-fixture include, corrected
before red-002. The final runner keeps the old field/data interface and passes
the new check. No failed receipt was overwritten or attributed to a target run.

Reproduction:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=vlaforge/python python -m pytest \
  vlaforge/tests/unit/test_materialize_native_observations.py \
  vlaforge/tests/unit/test_raw_observation_timing.py \
  vlaforge/tests/unit/test_session_benchmark_multi_output.py \
  vlaforge/tests/codegen/test_session_runner.py -q
```

Final validated snapshots are in `artifacts/orin/20261022-08/source-002/`.

| Source | SHA256 |
|---|---|
| `declared_host_pipeline.h` | `6a60557b7cf01dd6e086896f06098afa53d926fce4a52d49dfff4e54774da8bd` |
| `materialize_native_observations.py` | `5521cc86da53b5c374083b26962679269ee1817acf578c283e1b76999828e45e` |
| `raw_observation_runner.cpp.in` | `1ec32428fa39be682deac45bc56a87e355c8bdee58d7e1d630e5efbc14c7ae35` |
| `test_materialize_native_observations.py` | `d48ff264336b30b4490280e472b66ffb7c54d9307a8647e2a664d50a1071ab4a` |

## Remaining Model Gates

Typed transport is not complete online OpenPI preprocessing, F64 arithmetic,
native action postprocessing, RNG receipt generation or a supported TensorRT
binding. Those require source-derived declarations/partitions and independent
complete-model checks on Orin. Any later explicit narrowing still requires its
own range/semantic validation; this change performs none. Formal model count
stays 1/5, and the full five-model Goal remains incomplete.
