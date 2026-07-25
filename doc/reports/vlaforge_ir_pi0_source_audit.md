# VLAForge IR: π0 / π0.5 held-out source audit

> **Historical source audit.** Use the current pinned contract and evidence
> status in `doc/model_cards/pi0.md` for v0.2.

Date: 2026-07-23

Evidence level: source audit only; no checkpoint execution is claimed.

## Provenance

- LeRobot repository:
  `/home/zhangzimo/Repos/public/lerobot-v0.4.4`
- Revision: `8fff0fde`
- π0 source: `src/lerobot/policies/pi0/modeling_pi0.py`
- π0.5 source: `src/lerobot/policies/pi05/modeling_pi05.py`

## Source behavior

Both policies have the same VLA control skeleton:

1. `reset()` creates an action queue.
2. `select_action()` refills the queue with `predict_action_chunk()` only when
   it is empty, then pops one control action.
3. `sample_actions()` creates or accepts explicit noise.
4. Prefix embeddings and KV are computed once inside that chunk inference.
5. `x_t` is carried through a bounded `num_inference_steps` loop; the default is
   10.
6. The generated chunk is trimmed to the real action dimension.

Relevant source locations:

| Behavior | π0 | π0.5 |
| --- | --- | --- |
| bounded `sample_actions` | line 803 | line 780 |
| action queue reset | line 1130 | line 1108 |
| refill/pop `select_action` | line 1229 | line 1203 |
| chunk boundary | line 1246 | line 1220 |

## IR mapping

| Source value/control | VLAForge representation |
| --- | --- |
| action queue | persistent `StateSlot` |
| queue empty/refill | `vla.if` |
| explicit noise or RNG token | input / persistent RNG state |
| prefix KV | local TensorRegion output |
| solver `x_t` | `vla.for` carried SSA value |
| 10 flow steps | bounded `vla.for` |
| final action chunk | validate, transaction commit, publish |

No new core operation is required beyond the SmolVLA path. In particular,
prefix KV and solver `x_t` must not be promoted to persistent state because the
source does not retain them across policy invocations.

## RTC boundary

π0 and π0.5 optionally call an RTC processor with `prev_chunk_left_over`,
`inference_delay`, and `execution_horizon`. The base held-out contract keeps RTC
disabled. A future RTC adapter should first model these as explicit VLA inputs
or true source-retained state; it does not justify adding generic async/future
operations to the public IR.

## Held-out conclusion

The compiler/runtime core remains frozen for π0/π0.5 source mapping. A real
checkpoint run is still required before this can count as model-execution
evidence.
