# Complete Multi-Output Session Benchmark

`vlaforge.session_latency_protocol/1` retains its single floating output,
`tensor-contract.json`, CSV and raw-file conventions. Existing evidence is not
upgraded. Explicit `vlaforge.session_latency_protocol/2` adds a complete output
contract; unknown protocol/sample/declaration/reference fields and duplicate
JSON keys are rejected.

The v2 `outputs` list must name every Module output exactly once in ABI order.
Exactly one entry has role `primary-action`; the other entries have role
`float` or `exact`. Only the primary entry may declare `active_dimensions`, a
nonempty unique last-axis selection. Tensor dtype, shape and device come from
the Module, not reference sizes or names. All ports require the same explicit
CUDA device and supported static positive dimensions.

```json
{
  "outputs": [
    {"name": "action", "role": "primary-action", "active_dimensions": [0, 2]},
    {"name": "native_action", "role": "float"},
    {"name": "receipt", "role": "exact"}
  ],
  "samples": [
    {
      "sample_id": "observation-0",
      "inputs": {"noise": "/absolute/path/noise.bin"},
      "outputs": {
        "action": {"direct": "/absolute/path/direct.npy", "eager": "/absolute/path/reference.npy"},
        "native_action": {"direct": "/absolute/path/native-direct.npy", "eager": "/absolute/path/native-reference.npy"},
        "receipt": {"direct": "/absolute/path/receipt-direct.npy", "eager": "/absolute/path/receipt-reference.npy"}
      }
    }
  ]
}
```

This excerpt is not a full protocol. Existing fixed timing, five-process formal
requirements, paired provenance, policy selection, GPU ownership and quality
declarations still apply. Pilot remains 16 warmup plus 32 measured calls and is
not a formal CDF campaign. Dataset coverage is independent of call counts.

## Types and Fidelity

- Floating outputs support f16, bf16, f32 and f64. Every output is independently
  finite-checked and compared across its complete shape. BF16 NumPy references
  require the existing exact FP32 storage roundtrip; other dtypes require exact
  NumPy dtype equality, with explicit little-endian archive storage.
- Exact outputs support u8, i32, i64, u64 and bool. Complete bytes must equal
  both the direct artifact and eager references. Integers never pass through
  floating conversion or cosine calculations, including values above 2**53.
- Same-artifact full-byte equality is mandatory for every returned tensor.
  `eager_validation=bitwise` additionally gates every floating output. The
  `quality_gate=passed` numeric gate uses only the primary action's declared
  active dimensions. Auxiliary floats, integer receipts and inactive padding
  never dilute its error. Per-output float metrics remain separate.
- References for all outputs are required. An external RNG receipt may be a
  reference output, but validating it does not establish autonomous C++ PRNG
  generation or the identity of a missing official model dependency.

## Native Boundary and Archives

The primary output can occupy any ABI output index. The same Session run and
CUDA completion interval is timed; output reads, D2H, validation, metrics and
logging stay outside single-call latency and inside the separate steady-loop
wall boundary. Secondary outputs are read before the next call/binding update.
Every return is checked for dtype, complete shape, byte size, contiguous layout,
nonnull storage and the expected CUDA device. Integer values are byte-checked
without numeric conversion. Any missing/invalid output fails the process and
retains incomplete evidence.

The primary retains `outputs.<dtype>` and its legacy CSV metrics. Each other
output has `output-<ABI-index>.<dtype>`. Every file contains the complete bytes
for every warmup and measured call, in call order. `multi-output-fidelity.json`
contains per-output/per-call evidence, independent Python metrics and bindings
to execution, samples, protocol and tensor contract. `report.json` binds all raw
output hashes plus that fidelity report; aggregation rereads every complete
raw tensor and verifies those bindings. It does not accept a passed process
label alone. `validated_outputs` retains its historical call-count meaning;
v2 adds `validated_output_tensors` explicitly.

Reference bytes are cached only within one verification call and one output
port. Every distinct sample still loads both reference files, every actual
warmup/measured output is checked, and the full fidelity JSON and hashes must
match. The cache is discarded on return; another verification rereads changed
references. Frozen-file checks remain in the prepare/run/report protocol.
This reduces repeated small reads on network storage without changing native
timing, raw output coverage or numerical gates.

Allocator observation v2 and its independent raw/report/execution bindings are
unchanged and remain optional. Additional byte validation is not evidence of
zero allocation, leak freedom, sensor-to-action latency or board execution.
