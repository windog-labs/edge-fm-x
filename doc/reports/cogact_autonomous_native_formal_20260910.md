# CogACT Autonomous Native Formal Result

The autonomous C++ RNG path for the real public-dependency CogACT candidate is
complete on H20. A private `CudaRngProvider` owns a LibTorch CUDA generator,
generates the official 1 initial and 10 per-step draws, records all 12 state
snapshots, and never reads the process default generator or a pre-recorded
runtime RNG tape. The generated Session receives the real draws and state
receipts through its typed input contract.

> Historical timing note: the comparison in this report is superseded by
> [the boundary-aligned CogACT comparison](cogact_timing_boundary_formal_20260910.md).
> The original native timer excluded C++ RNG preparation while the official
> timer included its sampler draws, so the `1.401x` figure below must not be
> used as a same-boundary speedup claim. The output and autonomous-RNG evidence
> in this report remains valid.

Each formal configuration used five independent processes with 128 warmup and
1024 measured calls per process. All complete action and RNG outputs were
byte-exact against the frozen implementation reference. The independent
official and native campaigns used the same three input tensors, seed schedule
`42,43,42`, FP32/BF16 model precision, and a model-and-sampler timing boundary.

## Formal Comparison

| Configuration | Mean ms | p50 ms | p95 ms | p99 ms | Std ms | Calls/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Official PyTorch model + DDIM | 123.419 | 123.795 | 129.996 | 135.362 | 5.182 | 8.103 |
| Autonomous native Session | 88.096 | 89.469 | 90.781 | 94.766 | 2.822 | 11.351 |

The native path reduces mean model latency by 28.62% and increases model-only
throughput by 1.401x. The official campaign includes the same model and DDIM
work inside its timed region. Both exclude input H2D, RNG reset and complete
output conversion; native additionally excludes C++ RNG generation as recorded
by the runner protocol.

These numbers are historical only. The later boundary-aligned campaign includes
C++ RNG preparation in the native timed region and reports
`1.456x` on the all-sample mean, with `1.393x` as the median paired-GPU mean
ratio. See the newer report before citing any speedup.

The native CDF is `formal-autonomous-014/latency-cdf.{csv,png,pdf}`. The
official CDF is `official-model-formal-002/latency-cdf.{csv,png,pdf}`.

## Autonomous RNG Validation

The C++ provider smoke test validates:

- initial state and exact draw sequence for seeds `42`, `43`, `0` and
  `2**64-1`;
- three consecutive requests per seed, including all intermediate and final
  state receipts;
- byte equality with the frozen Python CUDA-generator tapes;
- reset, snapshot, restore and independent-provider interleaving;
- rejection of short, wrong-dtype and wrong-shape state without mutating the
  existing provider;
- no mutation of the process default generator.

The packaged runner was then executed for three consecutive real requests with
the private provider. All five outputs matched the frozen reference byte for
byte, and `/proc/self/maps` contained neither `libpython` nor
`libtorch_python`.

## Evidence

Native bundle:

`zzm-h20-x8-1:/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/cogact/torchscript-candidate-autonomous-014/session/bundle/`

Native formal campaign:

`zzm-h20-x8-1:/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/cogact/formal-autonomous-014/`

Official model campaign:

`zzm-h20-x8-2:/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/cogact/official-model-formal-002/`

| Artifact | SHA256 |
| --- | --- |
| Native bundle manifest | `6466309e2e7d9f4d601ab1ad5426a239de81fde198eb4ba55cd0efa917fae1e4` |
| Native runner source | `8e68c0e453bb3dbe460a3a0597d407379a4696b5d1a6050d24c668af073dbfb9` |
| Native runner binary | `49f1a2d0329935411d8bf0e56d93b67d5c6c49d025e73d43a7c3639ad629ce07` |
| Private RNG smoke report | `646fc355555789eb6c9f0a93e10f882f075b5f120e57144fe31b0d2fa2f9fbd5` |
| Native campaign report | `de16a6cbb7dae155c367c00dfa969a13f65c335ca86710730d72577dcf9ee5bd` |
| Native independent audit | `9ec10b66d5554135504bf0e0c96684cb7b0917cf429c72be2d2699b5d04e2a28` |
| Native CDF CSV | `591865b92bc23d7953d57659b41a2357286e7f02f050f2c1acf2a4915fb3da51` |
| Native CDF PNG | `1cb4509e952c01b9e6f633ffe49e0c8437d4dd57fa0b21ccdcb2959f7a8d379c` |
| Native CDF PDF | `38d691fe321afd8889e84af1a2969b09abf4e62fa20c21048740797420681dc4` |
| Official campaign report | `feeefe5e154902cfd163a1cd25aebb39f5fdd9ac7c90863748cedd9ffeb9c9de` |
| Official independent audit | `10e05a8e9eed47b82a5c30a0fd63c88450cb193a59392d97b8d086d24cf124f7` |
| Official CDF CSV | `c1c4ac64c84430c5b09702d8edcb8d36b6c0489a3458cdb02c2323006e1dab12` |
| Official CDF PNG | `a304618d2823a2995b34db849aa99e7cea715673c0463fcb48cab4bf35585a59` |
| Official CDF PDF | `bb4d1bf28302fd661a8d902219cca7a0ca52686fb222890042d042b076f35414` |

The local audit archive is
`artifacts/recovery-audit-20260909/cogact-autonomous-stage-002/`. Full CPU
regression passed `2260`, skipped `74`, with zero failures; the JUnit evidence
is `tests/full-cpu-regression.xml`, SHA256
`7a236c61605dbe1f6bc02e86fa3746c922dcb2c664c065ee47a35fdc8b3b517e`.

## Scope And Qualification

The checkpoint contains 7,630,224,071 parameter elements and therefore must not
be described as a complete 3B model. The candidate uses the public OpenVLA
dependency configuration. The prior tokenizer audit establishes SentencePiece
byte identity with the public Meta LFS hash, but equivalence of the complete
gated Meta configuration remains unverified. The formal input profile is one
frozen observation repeated with seeds `42,43,42`; broader observation and
robot-task coverage is unchanged.

Failed candidates 012 and 013, the earlier candidate-011 external-tape
campaign, and all previous failure reports are retained. Candidate 012 failed
compilation and candidate 013 was stopped during NAS artifact adoption; neither
replaces candidate 014 evidence.
