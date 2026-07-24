# VLAForge IR Foundation

This directory contains the executable Python reference semantics for the
stateful/temporal VLA IR described in
[`../doc/vlaforge_development_plan.md`](../doc/vlaforge_development_plan.md).
It is intentionally isolated from EdgeFM's engine/model/operator hierarchy.

The first milestone implements:

- versioned persistent state and logical epochs;
- SSA-style tensor regions and structured control flow;
- transaction-scoped state writes and action commit;
- type, state-version, freshness, effect and commit verification;
- a deterministic reference interpreter and trace format;
- liveness, state dependency and bounded physical-slot analyses.

The Python IR is the normative v0 semantics. It is not the final deployment
runtime and does not attempt to replace a tensor compiler.

The intentionally small public abstraction boundary is documented in
[`spec/vla-profile.md`](spec/vla-profile.md). Prefix KV or solver tensors are
not promoted to persistent state unless the real source retains them across
policy invocations.

## Development setup

```bash
cd vlaforge
python -m pip install -e '.[test]'
python -m pytest -q
python examples/smolvla/build_fixture.py
vlaforge verify examples/smolvla/program.vla
vlaforge run examples/smolvla/program.vla \
  --adapter smolvla-fixture \
  --trace /tmp/vlaforge-smolvla-trace.json
```

The default test suite is offline and never downloads model weights. Tests
marked `real_model` are separate evidence gates and must be executed explicitly
before G2 can be claimed.

Real adapters deliberately keep framework internals inside pure TensorRegions:

- SmolVLA exposes prefix preparation and the bounded flow solver while keeping
  its cross-tick action queue explicit.
- OpenVLA exposes deterministic action-token generation and detokenization; it
  has no persistent state in the reference `predict_action` path.

The runners are:

```text
tools/run_real_smolvla.py
tools/run_real_openvla.py
```

Model dependencies and weights are not package dependencies. Each gate runs in
an explicitly pinned external environment.

Their complete command lines are discoverable through:

```bash
python tools/run_real_smolvla.py --help
python tools/run_real_openvla.py --help
```

Both runners require an explicit checkpoint path, write a normalized trace and
schema-versioned JSON report, and exit nonzero when the eager-versus-IR
contract fails.

## Reproduce the local real-model gates

The following commands are the exact pinned workspace gates used for the
evidence reports. They do not download weights or mutate shared Python
packages.

SmolVLA:

```bash
cd /home/zhangzimo/Repos/private/edge-fm-x
export VLAFORGE_SMOLVLA_POLICY_PATH="$PWD/examples/smolvla/SmolVLA-Base"
export VLAFORGE_SMOLVLA_VLM_PATH="$PWD/examples/smolvla/SmolVLM2-500M-Video-Instruct"
export VLAFORGE_MODEL_DEVICE=cuda:0
export VLAFORGE_LEROBOT_REVISION=8fff0fde
PYTHONPATH="$PWD/vlaforge/python:/home/zhangzimo/Repos/public/lerobot-v0.4.4/src" \
  /home/zhangzimo/miniconda3/envs/horizon_quant/bin/python \
  -m pytest -q vlaforge/tests/models/test_real_smolvla.py -m real_model
```

OpenVLA:

```bash
cd /home/zhangzimo/Repos/private/edge-fm-x
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLAFORGE_OPENVLA_CHECKPOINT=/home/zhangzimo/.cache/vlaforge/openvla-7b
export VLAFORGE_OPENVLA_REVISION=47a0ec7fc4ec123775a391911046cf33cf9ed83f
export VLAFORGE_MODEL_DEVICE=cuda:0
export VLAFORGE_OPENVLA_UNNORM_KEY=bridge_orig
export VLAFORGE_OPENVLA_LOAD_IN_4BIT=1
PYTHONPATH="$PWD/vlaforge/python" \
  /home/zhangzimo/.venvs/vlaforge-openvla/bin/python \
  -m pytest -q vlaforge/tests/models/test_real_openvla.py -m real_model
```

## Generated C++ and whole-program optimization gates

The no-Python generated-C++ contracts and exact real-model commands are
recorded in:

- [`../doc/reports/vlaforge_cpp_smolvla_real.md`](../doc/reports/vlaforge_cpp_smolvla_real.md)
- [`../doc/reports/vlaforge_cpp_openvla_real.md`](../doc/reports/vlaforge_cpp_openvla_real.md)
- [`../doc/reports/vlaforge_whole_program_optimizations.md`](../doc/reports/vlaforge_whole_program_optimizations.md)

`generate_real_smolvla_cpp.py` and `generate_real_openvla_cpp.py` produce the
normal audited runner by default. Pass `--optimization-benchmark` only for the
instrumented cache/LICM measurement build; this keeps deployment-source golden
digests stable.

The combined benchmark/audit entry point is:

```bash
PYTHONPATH=vlaforge/python \
python vlaforge/tools/benchmark_whole_program_optimizations.py --help
```

It runs generated C++ with an invalid Python environment, measures tick p99
and peak RSS, verifies exact action/evidence/non-Region traces, measures
compiler-pass cost and static-arena peak, and exits nonzero on a Gate G4
regression.
