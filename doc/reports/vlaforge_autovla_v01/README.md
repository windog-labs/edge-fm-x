# AutoVLA held-out real-model evidence

This directory records the held-out driving autoregressive model selected after
the VLAForge 15-op core was frozen.

## Passing evidence

`autovla_frontend_l2.json` is
`L2-partitioned-real-checkpoint-frontend` evidence produced from clean revision
`f750e544f04c76e45fecb6ab06d98ca9e89c3c62`.

- released 16,292,664,780-byte checkpoint and source/Qwen/codebook hashes pass;
- real final Qwen MLP, final norm/action projection, and codebook rollout are
  split into three TensorRegions;
- eager, strict export, Semantic IR and Plan trajectory/action-token outputs
  are exact;
- Semantic/Plan traces are exact;
- revisions `[100,100,101]` produce one exact-cache hit and two misses;
- transactional named outputs are `trajectory [10,3]` and
  `action_tokens [10]`;
- frozen core op delta is zero;
- peak CUDA allocated is 533,944,320 bytes and peak Host RSS is
  1,473,228,800 bytes.

This is a real-weight post-attention decoder partition, not full camera-to-
trajectory AutoVLA and not generated C++ evidence.

## Non-promoted L3 candidate

`autovla_artifact_l3_candidate.json` records both successful compilation and a
failed promotion decision. The predefined conservative AOTI profile produced
three `sm_86` packages. Final action tokens are exact, trajectory max absolute
error is `1.91e-6`, and repeatability is bit-exact. Decoder-hidden and logits
NRMSE (`6.65e-3` and `4.54e-3`) exceed the predeclared `1e-3` Region threshold.
The tolerance was not changed after observing the result, so the honest level
remains L2.

External byte-for-byte reproduction roots:

- `/tmp/vlaforge-autovla-checkpoint-a7d7ba3`;
- `/tmp/vlaforge-autovla-l2-f750e54`;
- optional L3 candidate:
  `/tmp/vlaforge-autovla-l3-conservative-f750e54`.
