# Half Native Whole-Loop Replay Formal

2026-09-09 02:20 UTC. Baseline, FP16 and BF16 each completed the formal
required-policy replay campaign on RTX 3060: five independent processes, 128
warmups and 1024 measured calls. Every required process reports
`REPLAY_FINAL,11,1,10,1152,0` and all outputs are byte-exact.

## Timing

| Lane | Mean ms | p50 ms | p95 ms | p99 ms | Max ms | Std ms | Chunks/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 77.170051 | 77.213260 | 77.645345 | 77.925430 | 78.328394 | 0.298163 | 12.958395 |
| FP16 | 77.217964 | 77.195550 | 77.661356 | 77.896281 | 78.902417 | 0.208335 | 12.950354 |
| BF16 | 77.271447 | 77.273812 | 77.649636 | 77.849121 | 78.507106 | 0.189118 | 12.941391 |

Required replay mean reduction relative to each lane's formal off mean:
17.33% baseline, 18.01% FP16 and 17.61% BF16. Half precision does not change
the replay benefit materially.

## Evidence

- Reports and raw samples:
  `half-native-benchmark-001/required-prepared-{baseline,fp16,bf16}/`.
- Required CDF CSV per lane in the same directories.
- Formal report SHAs: baseline `5d9b175bccc621e1258df42381767812a0ea4dcd3ad6064b1e8ef7090489851e`,
  FP16 `67bc77ae72d309777b19b06af3a20a959701985308d00d6bc9efd23cf7da3546`,
  BF16 `cc7d6ae413588c9730c2672e178658e31a9146eacf8ca394388fbd3c1c8a472e`.
- Combined off/required CDF PNG SHA:
  `0be6d91c2352ae775a4c6308f03b5012e0e829d6d43c0ef8802706cfd31f5d25`.
- Replay is complete-loop graph execution, not an Agent-selected kernel or a
  memory-saving claim.
