# CogACT CUDA Formal Run

本报告保留 H20-2 CUDA 系列的原始结果，不能代替 Orin/BPU 或 vendor 对照。

2026-09-08 验收口径勘误：目录和标题中的 Formal 是历史标签。该系列每进程115次调用，
包括首个 timed run，未按公共 benchmark 协议剔除128次预热并采集每进程1024次稳态样本。
因此输出保真证据成立，但公共协议正式稳态 CDF 仍未验收；不删除原时延样本或尾部。

## Result

| Item | Value |
| --- | --- |
| Checkpoint | `CogACT-Base.pt`, 7,630,224,071 installed tensor elements |
| Observation source | Locked Fractal/Google Robot episode 0, 115 chronological decoded RGB frames |
| Reference gate | 115/115 frames; ten DDIM steps, CFG tensors, 11 RNG draws, and full action exact |
| Native run | 10 independent H20-2 processes, 115 frames each, 1150 calls |
| Output gate | raw / normalized / native all 1150/1150 byte-exact; MSE and max-abs 0 |
| Cosine minimum | raw `0.999999826`, normalized `0.999999757`, native `0.9999999999999998` |
| Latency | mean `139.889050`, p50 `134.535095`, p95 `140.828603`, p99 `156.612784`, max `745.119087` ms |
| Throughput | `7.148522` fresh chunks/s |

## Evidence

- Reference/partition report: `artifacts/edgefm-vla-goal/20260906-044509/cogact/series-005/remote/report.json`, SHA256 `0aac2ca6791b3606c9d49624b7e3cf133b029946f5060520bd4f380bdace06bd`.
- Full-output independent audit: `.../series-005-native-formal-010/remote/independent-audit-002.json`, SHA256 `9c6946c96a5e3574a047fa8026dbb9ede33f4cdaf028cdf388ff210e278c22e9`.
- Raw latency samples: `.../latency-samples.json`; CSV: `.../latency.csv`; CDF: `.../latency-cdf.png`.
- Bundle input manifest: `.../series-005-native-inputs/manifest.json`, SHA256 `004ac37527db78c08ac93c77951795231c696f58ad81efdc7955eda402a737f5`.

The timing boundary is resident model tensors with explicit H2D input upload and D2H output copy. Session/bundle creation occurs before the timed loop and is excluded; the first timed run can include first-execution overhead. The bundle uses an external producer's explicit RNG tape, so autonomous C++ RNG is not demonstrated. The run covers one real episode's changing frames, not 115 independent episodes, and remains a candidate public-dependency configuration because the original Meta config was unavailable.
