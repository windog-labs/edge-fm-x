"""Model-neutral typed resident-tensor diagnostic workers from the common template.

This renders a runnable host, not a benchmark result or deployment certificate.
The worker owns its input buffers and reads every declared output. Model authors
do not supply C++ allocation, binding, publication or tensor comparison code.
"""

import json
import re

from vlaforge.validation.session_benchmark import (
    MULTI_SCHEMA,
    benchmark_output_contract,
    input_specs,
    replay_checks,
)


def render_resident_tensor_runner(
    module, template, *, outputs, samples, owner_handshake=True,
    numerical_bindings=(), acknowledge_exclusive_process=False,
    acknowledge_calling_thread=False, replay_policy="off", host_io=False,
):
    """Render all typed outputs, optionally bootstrapping an exclusive worker.

    The default does not set numerical state. When bindings are supplied, every
    Region must have a compatible provider-required policy; explicit ownership
    acknowledgements are required. Initialization follows the optional CUDA
    identity handshake/reset and precedes all input and Session allocation.
    Optional replay telemetry uses the same per-invocation contract as the
    formal benchmark, including application warmups. No allocator queries run.
    With host_io enabled, inputs are recopied from host storage for every call;
    timing includes binding and complete output copies, excluding validation IO.
    """
    if type(samples) is not int or not 1 <= samples <= 100000:
        raise ValueError("runner sample count must be an explicit bounded positive integer")
    if type(owner_handshake) is not bool:
        raise ValueError("runner owner handshake must be boolean")
    if type(host_io) is not bool:
        raise ValueError("runner host IO must be boolean")
    contract = benchmark_output_contract(module, {"schema": MULTI_SCHEMA, "outputs": list(outputs)})
    declarations, ordinal, count = input_specs(module, contract=contract)
    additional = []
    for item in contract["outputs"]:
        if item["role"] == "primary-action":
            continue
        additional.append("{" + f'{item["index"]}u, VLAFORGE_DTYPE_{item["dtype"].upper()}, '
            + "{" + ",".join(map(str, item["shape"])) + "}, "
            + f'{item["size_bytes"]}u, {item["count"]}u, '
            + str(item["role"] != "exact").lower() + ", "
            + ", ".join(json.dumps(item[key]) for key in ("raw_file", "direct_file", "eager_file")) + "}")
    if replay_policy not in ("off", "batch-only", "required"):
        raise ValueError("runner replay policy must be off, batch-only, or required")
    replay_rows = []
    if replay_policy != "off":
        from vlaforge.compiler import compile_module
        selected = compile_module(module, loop_execution=replay_policy,
                                  default_device=module.inputs[0].device,
                                  state_device=module.inputs[0].device)
        replay_rows = [
            {"task_id": task.id, "policy": task.attributes["replay"],
             "steps": len(range(task.attributes["lower"], task.attributes["upper"],
                                task.attributes["step"]))}
            for task in selected.plan.tasks
            if task.opcode == "vla.for" and task.attributes.get("replay", "off") != "off"
        ]
    replay_audit, replay_failure = replay_checks(replay_rows, per_call=True)
    values = {"INPUTS": declarations, "ORDINAL": str(ordinal), "SAMPLES": str(samples),
        "OUTPUT_COUNT": str(count), "OUTPUT_BYTES": str(contract["size_bytes"]),
        "OUTPUT_DTYPE": "VLAFORGE_DTYPE_" + contract["dtype"].upper(),
        "OUTPUT_SHAPE": ",".join(map(str, contract["shape"])),
        "ACTIVE_INDICES": ",".join(map(str, contract["active_indices"])),
        "OUTPUT_FILE": contract["raw_file"], "PRIMARY_OUTPUT_INDEX": str(contract["index"]),
        "MULTI_OUTPUT_DEFINE": "#define VLAFORGE_BENCHMARK_MULTI_OUTPUT 1",
        "ADDITIONAL_OUTPUT_SPECS": ",\n".join(additional), "OWNER_HANDSHAKE": str(owner_handshake).lower(),
        "HOST_IO": str(host_io).lower()}
    for key in ("ALLOCATOR_INCLUDE", "ALLOCATOR_BEFORE_SESSION", "ALLOCATOR_AFTER_LOAD",
                "ALLOCATOR_AFTER_WARMUP", "ALLOCATOR_AFTER_MEASURED", "ALLOCATOR_AFTER_DESTROY",
                "SESSION_LIFECYCLE_BEGIN", "SESSION_LIFECYCLE_END"):
        values[key] = ""
    values["REPLAY_AUDIT"] = replay_audit
    values["REPLAY_FAILURE"] = replay_failure
    if set(re.findall(r"@([A-Z_]+)@", template)) != set(values) | {"NUMERICAL_WORKER_BOOTSTRAP"}:
        raise ValueError("resident tensor runner template contract changed")
    bindings = tuple(numerical_bindings)
    initializer, call = "", ""
    if bindings:
        from vlaforge.codegen.numerical import generate_libtorch_worker_initializer
        from vlaforge.deployment.numerical import (
            PROVIDER_REQUIRED,
            RegionNumericalBinding,
        )

        if (len(bindings) != len(module.regions)
                or {item.region_name for item in bindings if isinstance(item, RegionNumericalBinding)}
                    != {region.name for region in module.regions}
                or any(not isinstance(item, RegionNumericalBinding)
                       or item.runtime_enforcement != PROVIDER_REQUIRED for item in bindings)):
            raise ValueError("runner numerical bindings must cover every Region with provider enforcement")
        initializer = generate_libtorch_worker_initializer(bindings,
            acknowledge_exclusive_process=acknowledge_exclusive_process,
            acknowledge_calling_thread=acknowledge_calling_thread)
        digest = bindings[0].requirement.policy.digest()
        call = f"""  {{
    const auto numerical_status = vlaforge_initialize_numerical_worker();
    if (numerical_status.code != VLAFORGE_STATUS_OK) {{
      std::fprintf(stderr, "numerical worker bootstrap failed: %u\\n", static_cast<unsigned>(numerical_status.code));
      return 21;
    }}
    std::fprintf(stderr, "NUMERICAL_WORKER_BOOTSTRAP_OK,{digest}\\n");
  }}"""
    for key, value in values.items():
        template = template.replace("@" + key + "@", value)
    marker = "// @NUMERICAL_WORKER_BOOTSTRAP@"
    if template.count(marker) != 1:
        raise ValueError("runner template needs one post-handshake numerical initialization marker")
    if bindings:
        return initializer + template.replace(marker, call), contract
    return template.replace("@NUMERICAL_WORKER_BOOTSTRAP@", ""), contract
