# VLAForge paper-grade Host-CUDA matrix

Status: **passed**.

Each cell uses five independent processes and five deterministic content profiles per model. Confidence intervals use process-cluster bootstrap resampling.

| Model | Workload | Path | Mean ms [95% CI] | p50 | p90 | p99 |
|---|---|---|---:|---:|---:|---:|
| diffusiondrive | baseline | direct_artifact | 16.663 [16.636, 16.686] | 16.656 | 17.241 | 17.286 |
| diffusiondrive | baseline | eager | 19.451 [19.413, 19.488] | 19.456 | 19.759 | 19.863 |
| diffusiondrive | baseline | generated_session | 16.688 [16.609, 16.774] | 16.711 | 17.288 | 17.349 |
| diffusiondrive | context_plus_0p01 | direct_artifact | 16.652 [16.574, 16.730] | 16.641 | 17.224 | 17.299 |
| diffusiondrive | context_plus_0p01 | eager | 19.516 [19.470, 19.564] | 19.499 | 19.768 | 20.247 |
| diffusiondrive | context_plus_0p01 | generated_session | 16.713 [16.641, 16.781] | 16.693 | 17.300 | 17.772 |
| diffusiondrive | noise_plus_1pct | direct_artifact | 16.633 [16.568, 16.692] | 16.646 | 17.220 | 17.293 |
| diffusiondrive | noise_plus_1pct | eager | 19.462 [19.418, 19.506] | 19.465 | 19.717 | 20.082 |
| diffusiondrive | noise_plus_1pct | generated_session | 16.743 [16.687, 16.798] | 16.718 | 17.289 | 17.375 |
| diffusiondrive | observation_minus_1pct | direct_artifact | 16.626 [16.559, 16.702] | 16.617 | 17.191 | 17.307 |
| diffusiondrive | observation_minus_1pct | eager | 19.460 [19.423, 19.518] | 19.441 | 19.709 | 19.941 |
| diffusiondrive | observation_minus_1pct | generated_session | 16.735 [16.651, 16.816] | 16.715 | 17.325 | 17.374 |
| diffusiondrive | observation_plus_1pct | direct_artifact | 16.678 [16.649, 16.713] | 16.649 | 17.226 | 17.526 |
| diffusiondrive | observation_plus_1pct | eager | 19.468 [19.435, 19.501] | 19.456 | 19.737 | 19.923 |
| diffusiondrive | observation_plus_1pct | generated_session | 16.797 [16.730, 16.835] | 16.818 | 17.294 | 17.379 |
| smolvla | baseline | direct_artifact | 45.466 [45.388, 45.517] | 45.174 | 46.255 | 48.116 |
| smolvla | baseline | eager | 113.198 [111.958, 114.503] | 113.064 | 115.454 | 116.585 |
| smolvla | baseline | generated_session | 45.752 [45.615, 45.884] | 45.359 | 46.489 | 48.634 |
| smolvla | context_plus_0p01 | direct_artifact | 45.519 [45.454, 45.563] | 45.177 | 46.292 | 48.165 |
| smolvla | context_plus_0p01 | eager | 112.707 [111.855, 114.001] | 112.271 | 115.157 | 115.624 |
| smolvla | context_plus_0p01 | generated_session | 45.737 [45.715, 45.760] | 45.443 | 46.527 | 48.567 |
| smolvla | noise_plus_1pct | direct_artifact | 45.500 [45.486, 45.516] | 45.176 | 46.342 | 47.820 |
| smolvla | noise_plus_1pct | eager | 112.470 [111.293, 113.593] | 112.358 | 114.459 | 115.273 |
| smolvla | noise_plus_1pct | generated_session | 45.717 [45.640, 45.781] | 45.367 | 46.729 | 48.405 |
| smolvla | observation_minus_1pct | direct_artifact | 45.519 [45.433, 45.608] | 45.182 | 46.296 | 48.135 |
| smolvla | observation_minus_1pct | eager | 113.875 [111.582, 115.942] | 113.471 | 116.998 | 117.683 |
| smolvla | observation_minus_1pct | generated_session | 45.715 [45.618, 45.812] | 45.433 | 46.645 | 48.269 |
| smolvla | observation_plus_1pct | direct_artifact | 45.517 [45.395, 45.687] | 45.235 | 46.295 | 48.026 |
| smolvla | observation_plus_1pct | eager | 111.450 [110.428, 112.545] | 111.212 | 113.249 | 114.403 |
| smolvla | observation_plus_1pct | generated_session | 45.757 [45.720, 45.793] | 45.290 | 47.584 | 48.270 |

## Measurement boundary

- Fresh-process initialization and first Run are reported separately from steady-state latency.
- OS page cache is not forcibly dropped; initialization is a fresh-process measurement, not a physical cold-storage claim.
- Generated cells use full recomputation. DiffusionDrive cache hits and SmolVLA queue-consumption fast paths are excluded.
- AOTI/cuDNN/CUTLASS/Triton model kernels are upstream work. VLAForge claims only measured generated-Session orchestration.
- This is RTX 3060 Host-CUDA evidence, not Orin evidence.
