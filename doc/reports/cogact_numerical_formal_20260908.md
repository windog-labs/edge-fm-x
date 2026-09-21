# CogACT Numerical-Provider Formal Result

2026-09-08 15:05 UTC. The fresh numerical-bound candidate now has an
independently audited formal CUDA latency table and raw CDF. This does not
complete the available-hardware Goal or final paper acceptance.

## Scope

The candidate is the four-Region CogACT Base public-dependency implementation
whose fresh TorchScript compile records and explicit `PROVIDER_REQUIRED`
bindings were completed earlier on 2026-09-08. All five ABI outputs are
enforced: raw F32 `[16,7]`, normalized F32 `[16,7]`, native F64 `[16,7]`,
RNG U8 `[16]` and draw-count I64 `[1]`. Integer outputs remain exact bytes.

Measurements ran on `zzm-h20-x8-2` GPU1
`GPU-29fb9c6e-2a11-e303-5f26-ad079fa86fa5`, driver 535.161.08, at the same
resident CUDA model-tensor boundary as the pilot: 16 real spaced Fractal
frames, online DINO+SigLIP and text prefix, N10 CFG/DDIM, native F64 output
conversion and Session completion. 128 warmups and 1024 measured calls ran in
each of five independent processes per policy.

The original formal controller recorded 15 workers and report generation as
passed. Its final audit used an older script whose JSON-list/tuple comparison
failed; that failed receipt is preserved. An audit-only recovery reran only the
fixed `audit_formal_bound_002.py`, checked no model/report/output file changed,
and did not restart inference.

## Result

All 17280 complete calls and 86400 complete output tensors are byte-exact
against both the archived official-code reference and the fresh direct
artifact. Floating MSE/max-abs are zero and minimum computed cosine is
0.9999999999999998. Every required worker reports N10 capture, 1152 replays
and zero ordinary calls.

| Policy | Mean ms | p50 ms | p95 ms | p99 ms | Max ms | Std ms | Chunks/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 84.869524 | 83.889342 | 88.991887 | 90.371135 | 103.007530 | 2.170365 | 11.782793 |
| batch-only | 86.191139 | 87.395971 | 90.143048 | 92.157494 | 114.754667 | 2.848236 | 11.602121 |
| required | 76.230755 | 75.961983 | 77.803763 | 78.860851 | 83.432806 | 0.883742 | 13.118065 |

Required mean is 10.178882% lower than same-source off. Whole-worker sampled
peak device memory is 29433 MiB for off/batch-only and 29573 MiB for required
(+140 MiB), so this remains execution-policy evidence, not memory savings or an
Agent-kernel gain. The old run-004 unbound formal numbers are not inherited by
this provider-bound candidate.

## Evidence

- Original formal pipeline SHA: `aaeb361df416257147009f92c82610ef140b3381d8795e6bd50ac9d98bea63bc`.
- Audit recovery receipt SHA: `95d44cd91d869d2a50e4eec7339df6a3aa9352b17fc45d4d2ec175bbdad4bc77`.
- Independent formal audit SHA:
  `92df2490f9383ad6ba75bafb3eaf8bb4bd568168116f8abaeb0d438632eb7014`.
- Figures: `formal-figures-001/latency-cdf-tail.png` and `.pdf`; PNG is
  nonblank by pixel check, human visual inspection not repeated in this run.
- Raw samples and complete outputs: local
  `formal-retrieval-001/evidence/prepared/runs/`, table and CDF CSVs beside it.

## Remaining Limitations

Original Meta configuration remains unavailable; the installed public
dependency candidate has about 7.63B elements and is not relabeled 3B.
Autonomous C++ RNG is still absent: the external producer supplies the
11-draw tape. Sixteen spaced frames come from one real Fractal episode.
No vendor baseline, board run, physical calibration or low-bit acceptance is
inherited. Board output cells remain unfilled.
