# VLAForge Paper Benchmark

> **Archived v0.1 benchmark.** The measurements below use the removed
> tick/epoch runner and cannot support Invocation IR v0.2 performance claims.

- Revision: `d818bf6d611e8ad257a7ff434934c8e14e8f868f`
- Gate passed: `true`
- Exact state/action/evidence: `true`

| Model | Workload | Mode | Measurement | n | p50 us | p95 us | p99 us | RSS MiB | VRAM MiB |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| smolvla | nominal | off | measured | 30 | 209853.430 | 210700.518 | 210800.841 | 2542.0 | 1964.0 |
| smolvla | nominal | cache | measured | 30 | 120003.836 | 210690.183 | 210974.167 | 2541.5 | 1964.0 |
| smolvla | nominal | licm | measured | 30 | 47848.750 | 48179.178 | 49770.267 | 2541.8 | 1964.0 |
| smolvla | nominal | combined | measured | 30 | 39291.883 | 48177.042 | 49942.349 | 2532.5 | 1964.0 |
| smolvla | repeat | off | measured | 30 | 210281.257 | 211049.969 | 211690.770 | 2542.3 | 1964.0 |
| smolvla | repeat | cache | measured | 30 | 29844.610 | 30118.646 | 31044.204 | 2541.6 | 1964.0 |
| smolvla | repeat | licm | measured | 30 | 47892.548 | 50532.584 | 51015.547 | 2541.9 | 1964.0 |
| smolvla | repeat | combined | measured | 30 | 29916.577 | 31551.266 | 31635.919 | 2532.3 | 1964.0 |
| smolvla | all-miss | off | measured | 30 | 210347.943 | 211395.610 | 211862.276 | 2542.4 | 1964.0 |
| smolvla | all-miss | cache | measured | 30 | 211296.807 | 211986.379 | 212034.302 | 2532.4 | 1964.0 |
| smolvla | all-miss | licm | measured | 30 | 47946.841 | 50731.366 | 50915.309 | 2532.3 | 1964.0 |
| smolvla | all-miss | combined | measured | 30 | 47886.488 | 50658.149 | 50705.265 | 2541.6 | 1964.0 |
| smolvla | stale | off | measured | 30 | 211278.900 | 211775.792 | 212032.739 | 2532.3 | 1964.0 |
| smolvla | stale | cache | measured | 30 | 211304.276 | 211748.745 | 212020.705 | 2532.4 | 1964.0 |
| smolvla | stale | licm | measured | 30 | 48089.904 | 50633.750 | 50908.944 | 2532.4 | 1964.0 |
| smolvla | stale | combined | measured | 30 | 47937.198 | 50682.564 | 50924.505 | 2532.5 | 1964.0 |
| openvla | nominal | off | measured | 30 | 33158838.391 | 33373212.048 | 34313151.901 | 16231.2 | 0.0 |
| openvla | nominal | cache | measured | 30 | 18145733.312 | 34328115.372 | 34793571.893 | 16967.4 | 0.0 |
| openvla | nominal | combined | reused from cache | 30 | 18145733.312 | 34328115.372 | 34793571.893 | 16967.4 | 0.0 |
| openvla | repeat | off | reused from nominal/off | 30 | 33158838.391 | 33373212.048 | 34313151.901 | 16231.2 | 0.0 |
| openvla | repeat | cache | measured | 30 | 2916611.463 | 2943439.041 | 2983405.740 | 16068.3 | 0.0 |
| openvla | repeat | combined | reused from cache | 30 | 2916611.463 | 2943439.041 | 2983405.740 | 16068.3 | 0.0 |
| openvla | all-miss | off | reused from nominal/off | 30 | 33158838.391 | 33373212.048 | 34313151.901 | 16231.2 | 0.0 |
| openvla | all-miss | cache | measured | 30 | 33324024.859 | 33559222.313 | 33653104.610 | 17013.9 | 0.0 |
| openvla | all-miss | combined | reused from cache | 30 | 33324024.859 | 33559222.313 | 33653104.610 | 17013.9 | 0.0 |
| openvla | stale | off | measured | 30 | 33064058.530 | 33363521.582 | 33418765.736 | 17072.9 | 0.0 |
| openvla | stale | cache | measured | 30 | 33117132.780 | 33910310.149 | 34165194.719 | 16993.2 | 0.0 |
| openvla | stale | combined | reused from cache | 30 | 33117132.780 | 33910310.149 | 34165194.719 | 16993.2 | 0.0 |

Bootstrap 95% confidence intervals, exact commands, environment, hashes, compiler arena, declared backend tensors, process RSS, and whole-process VRAM are retained in `paper_benchmark.json`.
