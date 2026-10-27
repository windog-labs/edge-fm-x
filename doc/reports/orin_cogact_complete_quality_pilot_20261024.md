# CogACT Orin Complete Quality Pilot

Checkpoint:2026-10-23 17:32UTC /2026-10-24 01:32Beijing.
**D1-D2 complete-model pilot verified; formal E2E/CDF still pending.**

## Identity And Computation

This is the frozen official-code public-dependency candidate with
**7,630,224,071parameters**, not a measured nominal3B/original-Meta-equivalent
configuration. Checkpoint SHA256
`1d35ec754c5c1ed7ab9ac22c9aba478e7269bab7db5b40df2703b2dc1e1009c0`.
Source revision6550bf0992f162fc5d74f14ffee30771a9433363and original public
configuration limitations remain unchanged.

Actual target:Jetson AGX Orin serial1421524312594,CUDA11.4,TensorRT8.5.2.2,
L4T35.4.1,existing MAXN. No Torch/torchvision model is installed or run by this
deployment. Full native C++ preparation -> transfer ->12TRT engines plus four
typed native providers -> all ten original denoising iterations -> complete CPU
output tuple executes on the actual Orin. Prefix is fresh on every request;
no precomputed features, model truncation or output-only replay is substituted.

The seven public inputs retain I64tokens,F32images/noise,U8twelve-state RNG tape.
Five complete outputs retain rawF32[16,7],normalizedF32[16,7],nativeF64[16,7],
U8receipt[16] and I64draw count[1]. Every observed draw count is11and every final
receipt matches the final original tape row. This is explicit external noise/
state tape, not an autonomous RNG implementation or independent robot trajectory.

The declared TRT policy enables FP16dense with explicit FP16result rounding,
preserves96source-FLOAT dense nodes in the prefix and77source-FLOAT dense nodes
in the neural step, and disables TF32. The added result rounding is not asserted
BF16/source equivalent. Initial unsupported FP16compute/FP32output format,
original failed build, diagnostic micrographs and native integration failures
are preserved. Generic control optional-sidecar and checked-variant namespace
repairs retain algorithms, typed ports, private stream ownership and strict
public validators. No quantization search or whole-model FP32fallback is used.

## Same-Orin Quality

| Evidence | Result | Boundary |
|---|---|---|
| Ordinary full execution |16real observations,80complete finite tensors | Baseline recorded on actual Orin, not a HOST reference |
| Arena full execution |16matching observations,80complete tensors | All graph/weight/precision/noise/resource contracts shared |
| Independent complete comparison |80/80byte-exact | Includes all five outputs, not only first step/action |
| Normalized primary worst MSE / minimum cosine |0 /1 | Fixed MSE<1e-5,cosine>0.9999passes |
| Native supplemental worst MSE / minimum cosine |0 /1 | F64representation retained |
| Opaque U8/I64 | Exact complete bytes, draw count11 | No float metric or cast substituted |

The same-platform primary gate compares ordinary TRT and Arena TRT with the
same numerical deployment profile. It is not a mathematical equivalence or
robot-success claim. Byte-exact paired results are reported separately from
the permitted tolerance. The16observations are from real recorded data; cycling
them in later benchmarks must not be described as independent trajectories.

Two Session source sets have identical semantic Module, precision and all16
resource contracts. Only the actual Plan's Arena layout changes:
127,529,728 ->44,083,456bytes. These are compiled layouts, not measured model
physical peak. Both share the same16regular hard-linked resource files,
16,446,921,220bytes, all completely SHA/inode/size/mtime verified. Both complete
Session libraries were actually ARM64 compiled;101HOST tests/two syntax checks
alone were not treated as target execution.

## Timing And Resource Limits

No timing from these pilots is admitted to the paper main table or formal CDF.
There is no128warmup/1024measured/five-process cohort yet. Baseline ran with CPU
services active; Arena ran after authorized temporary stopping of only the two
Java services. Their timing ratio is not a controlled optimization speedup.

Earlier Arena attempts were refused before any model call:22,849,736KiBand,
after stopping the two Java services,24,493,148KiBwere below the fixed
25,165,824KiBstartup line. The original manager restored both services. When
actual system availability changed, one fresh-path retry under the same authority
observed26,080,124KiB, passed unchanged startup admission and completed the pilot.
Minimum sampled MemAvailable was10,889,012KiBabove the4GiBsafety floor. Neither
admission nor numerical thresholds were lowered. Permission gaps267mean GPU
exclusivity is not proved by visible-owner scans alone. System memory is unified;
CPU/GPU counters are not summed into a model-memory figure.

The successful75.808764064swindow ended before its effective/original cutoff.
Both Java processes were restored through the original Supervisor, with unchanged
manager birth/argv,JAR,cwd,configuration and other managed programs. This is
process restoration, not application health. No additional daemon was stopped;
the earlier request for additional-service interruption is no longer needed.

## H20 Supplement

Existing archived H20 outputs are used only as a cross-device supplement. All112
input tensors across the same16frames match exactly. No new H20experiment or
cross-platform latency/speedup denominator is introduced.

| Output | Worst MSE | Minimum Cosine | Worst Max-Abs | Diagnostic Threshold Pass |
|---|---:|---:|---:|---:|
| Raw |8.0160295e-4|0.996831707|0.275961282|13/16|
| Normalized |1.2113142e-4|0.999516312|0.089644849|13/16|
| Native |1.9716224e-5|0.999894109|0.042180864|15/16|

The original series report directly anchors each native.npychecksum. Archived
raw/normalized arrays are fully hashed now but not individually original-report
anchored; that provenance limitation is retained. Differences in CUDA/BF16 and
declared FP16result rounding are not hidden. Negative supplemental rows do not
become primary passes, and they do not replace the same-Orin comparison.

## Evidence And Next Gate

- [Offline five-output formal auditor qualification](../../artifacts/orin/20261023-103/qualification-001.json):47local tests, including all27legacy regressions and20typed tests. Integer/opaque outputs are exact bytes, never float-cast or scored by cosine. The unchanged original auditor and whole-source inverse are pinned. The simulated57,600-tensor cohort is explicitly NONMODEL, not actual formal outputs.
- [Read-only restored Java process observation](../../artifacts/orin/20261023-103/readonly-services-001.json):both programs Running under original Supervisor at17:58UTC; no application-health or memory-admission claim.
- [Complete pair audit](../../artifacts/orin/20261023-100/pair-audit-001.json).
- [Pilot/resource/original-service restoration audit](../../artifacts/orin/20261023-100/pilot-delivery-audit-001.json).
- [Baseline complete-output audit](../../artifacts/orin/20261023-97/baseline-audit-001.json).
- [Actual engine/provider query audit](../../artifacts/orin/20261023-92/query-audit-001.json).
- [Matched-input H20 supplement](../../artifacts/orin/20261023-99/h20-supplement-002.json).
- [Whole-source/Plan binding](../../artifacts/orin/20261023-90/actual-001/candidate/assessment.json).
- [Actual two-mode ARM64 build](../../artifacts/orin/20261023-96/target-native-001/receipt.json).

Follow-up:the [independent ordinary repeat](orin_cogact_ordinary_repeat_20261024.md)
now passes all80complete outputs byte-exact and verifies the original-manager
restoration. This closes the extra repeat gate; its zero-warmup timing is still
not a controlled formal comparison. All16shared resources are rehashed again.

Later [source qualification](orin_cogact_formal_source_readiness_20261024.md)
also checks the actual typed-pilot/collector interface and24GiB/4GiBresource
wiring. It adds no inference or formal samples and does not activate the template.

Remaining gates:separate formal service authority, final window/resource policy
review and fresh preflight, then the unchanged two modes × five independent
processes ×128warmup+1024measured,batch1protocol. Complete finite/full-output
audits for every formal call, resource/restoration closure and final CDFs remain
unexecuted. Formal model coverage stays4/5and shared D4/optional Qwen remain
separately incomplete. No PDF is modified.
