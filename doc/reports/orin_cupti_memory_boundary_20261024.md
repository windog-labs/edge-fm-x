# Orin CUPTI Memory Observability Boundary

Follow-up: the [separate public TensorRT SDK diagnostic](orin_sdk_layer_reservations_20261024.md)
now actually runs IProfiler and context-requirement queries without requesting
either denied CUPTI API. This does not remove the CUPTI access failure, provide
physical peaks or qualify a complete-model result; the boundaries below remain.

Checked: 2026-10-23 23:53 UTC. **Two real profiler API paths are refused by
current permissions. No new model memory, allocation peak, latency or VLA
acceptance result is produced.** These are diagnostic-access failures, not
unsupported VLA operators or model-quality failures. Ordinary TensorRT
inference is not rejected by this result; CogACT still has its separate
unchanged24GiBcapacity prerequisite.

## Actual Native Attempts

No Nsight Systems CLI is found in the checked installed paths. Existing CUDA
11.4CUPTI library/headers are present, with header/runtime CUPTI API14matching.
No package, SDK, driver permission, power mode or original application is
changed. The code uses the installed typed SDK, not guessed memory-record
layouts or model-specific allocation rules.

| Declared Path | Native Compilation | Actual Access Result | Work Reached |
|---|---|---|---|
| Memory2 + Driver/Runtime Activity observer and3.5MiBallocation control | Both commands exit0 | `cuptiActivityRegisterCallbacks` returns35; constructor exits112 | Version query succeeds; declared allocation control main never executes |
| Separate Callback-only subscriber, no Activity API or allocation requested | Compilation exits0 | `cuptiSubscribe` returns35; program exits122 | Header/runtime versions both14; driver/runtime domain enablement is not attempted |

The second path is a distinct API-capability check, not a hidden retry of the
first probe or permission escalation. Its unattempted domain/unsubscribe values
remain JSONnull, not successful zero status. Neither path loads a model engine.
Any CUPTI-internal initialization activity is outside the declared allocation
control; absent user traces do not prove that the SDK touches no GPU resource.

Installed `cupti_result.h`defines35as `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`.
[NVIDIA's explanation](https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-cupti)
classifies that error as profiling-access restriction. This report records
the precise calls observed here, not a claim that every CUPTI feature or all
CUDA tracing on all devices is unsupported. The read-only `sudo -n -l`query
returns `a password is required`; no elevated diagnostic is executed and no
credentials or driver policy changes are attempted.

## No Missing-Data Substitution

The first observer has no valid activity records and never reaches its final
mark. Its declared control-result and legacy runtime-marker files are absent.
The second probe reports no enabled callback domains and no allocation request.
These observations are **missing coverage, not zero model allocation or peak**.
The boundary receipt stores model memory/peak as null, with qualificationfalse.
Any empty, failed, dropped-record or parse-error trace is rejected by the host
qualification check. No numerical tolerance or memory admission is lowered.

The original eight-GiBsmall-diagnostic admission/two-GiBruntime floor remain;
they do not substitute for CogACT24/4GiBgates. All stages run serially with
owned process/cgroup cleanup, explicit timeouts and frozen source/SDK files.
Both original Java programs retain PID942765/942805, births/argv/cwd and
Supervisor4237parent. No stop/start/restore RPC or global profiling setter
is used. The requests remain native CUDA controls, not Torch model execution.

## Closure

Both owned units are failed/MainPID0; their post hooks observe empty cgroups.
Independent read-only closure rehashes52actual target paths, including the
source/compiled programs, SDK headers/library and captures, and checks allnine
tracked controller/compiler/control/post identities terminal. Compiler child
trees are bounded by the same cgroups, not claimed as individually inventoried
PIDs. Nine source/actual/empty-trace/cleanup checks pass; this verifies the
negative boundary, **not** a passed profiler or model experiment.

Post-control available memory23,791,688KiBstill falls below CogACT25,165,824KiB.
No model job remains running. Prior formal four-model data, the111matched-power
supplement and112table export are unchanged; required model acceptance stays4/5.

## Evidence And Next Gate

- [Permission boundary and source/process closure](../../artifacts/orin/20261023-113/boundary-closure-001.json).
- [Activity API refusal](../../artifacts/orin/20261023-113/control-001/observer-status.json).
- [Callback API refusal and unattempted domains](../../artifacts/orin/20261023-113/callback-001/result.json).
- [Activity request/source pins](../../artifacts/orin/20261023-113/control-plan-001.json).
- [Callback-only request/source pins](../../artifacts/orin/20261023-113/callback-plan-001.json).
- [Nine boundary checks](../../artifacts/orin/20261023-113/boundary-tests-001.xml).
- [Installed error definition](../../artifacts/orin/20261023-113/sdk-headers/cupti_result.h).

Do not rerun either one-shot unit or run a full VLA under this unqualified
observer. CUPTI needs a newly authorized available profiling route before new
captures. Other public TensorRT diagnostic interfaces can be evaluated under
their own declared scope without CUPTI; they are not assumed to expose a full
physical peak. Static context reservations, requested bytes, sampled RSS and
layer timings must not be relabeled as interchangeable quantities or E2E.

Read-only archived rechecks:

```sh
env PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=artifacts/orin/20261023-113:vlaforge/python python3 -B -m pytest -q \
  -p no:cacheprovider artifacts/orin/20261023-113/test_cupti_boundary.py
```
