# VLAForge Invocation v0.2 Host Reference Benchmark

- Revision: `working-tree`
- Gate passed: `true`
- Exact Semantic/Plan outputs: `true`
- Scope: Python fixture reference executors only; not real model, generated C++, CUDA, or Orin performance

| Model | Workload | Mode | Measurement | n | p50 us | p95 us | p99 us |
|---|---|---|---|---:|---:|---:|---:|
| openvla_fixture | repeat | semantic | measured | 30 | 116.201 | 124.900 | 192.422 |
| openvla_fixture | repeat | plan | measured | 30 | 116.297 | 125.928 | 165.629 |
| openvla_fixture | new-revision | semantic | measured | 30 | 125.011 | 132.953 | 179.884 |
| openvla_fixture | new-revision | plan | measured | 30 | 129.703 | 135.731 | 183.172 |
| smolvla_fixture | repeat | semantic | measured | 30 | 155.992 | 334.436 | 335.195 |
| smolvla_fixture | repeat | plan | measured | 30 | 158.372 | 343.582 | 354.316 |
| smolvla_fixture | new-revision | semantic | measured | 30 | 158.367 | 341.140 | 564.031 |
| smolvla_fixture | new-revision | plan | measured | 30 | 159.974 | 338.870 | 343.623 |
| driving_diffusion_fixture | repeat | semantic | measured | 30 | 489.717 | 552.038 | 560.661 |
| driving_diffusion_fixture | repeat | plan | measured | 30 | 486.267 | 552.093 | 696.981 |
| driving_diffusion_fixture | new-revision | semantic | measured | 30 | 522.553 | 583.985 | 614.848 |
| driving_diffusion_fixture | new-revision | plan | measured | 30 | 517.776 | 576.084 | 616.104 |
