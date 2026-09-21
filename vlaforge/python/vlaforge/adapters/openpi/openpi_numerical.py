"""Bind actual OpenPI compile observations to the generic LibTorch provider.

No setters or model-specific numerical defaults live here. The public renderer
creates a separately acknowledged, explicit native worker initialization call.
"""

from __future__ import annotations

from vlaforge.deployment.libtorch_numerical import policy_from_context
from vlaforge.deployment.numerical import (
    PROVIDER_REQUIRED,
    NumericalCompileRecord,
    NumericalRequirement,
    RegionNumericalBinding,
    canonical_json,
)
from vlaforge.numerical_context import NumericalContext


def v2_from_legacy(legacy):
    """Project a legacy /1 context into the current enforceable /2 domain.

    This never silently upgrades in place. It keeps the legacy flag values and
    adds the currently observed Torch/version/reduction fields so a native
    provider can enforce the same matmul/cudnn policy.
    """
    import torch

    from vlaforge.numerical_context import snapshot

    current = snapshot()
    data = current.to_dict()
    for key, value in legacy.to_dict().items():
        if key != "schema":
            data[key] = value
    return NumericalContext.from_dict(data)


def binding_from_observed_compile(name, context, region, manifest, entry, build):
    """Reject legacy observations; project both newly observed complete v2 flags."""
    before = NumericalContext.from_dict(entry["observed_numerical_context_before"])
    after = NumericalContext.from_dict(entry["observed_numerical_context_after"])
    if before != context or after != context:
        raise ValueError(
            "reference and both actual compile boundary contexts must match"
        )
    if (
        entry.get("region") != name
        or entry.get("status") != "compiled-unvalidated"
        or entry.get("export") != region.get("archive")
        or manifest.get("exported_program", {}).get("sha256")
        != region["archive"]["sha256"]
        or manifest.get("torch_version") != context.torch_version
        or build.get("numerical_context") != context.to_dict()
        or manifest.get("target") != build.get("target")
        or manifest.get("inductor_profile") != build.get("profile")
        or canonical_json(manifest.get("inductor_configs"))
        != canonical_json(build.get("configs"))
    ):
        raise ValueError(
            "numeric compile binding source/profile/package provenance mismatch"
        )
    policy = policy_from_context(context)
    requested = policy_from_context(before)
    observed = policy_from_context(after)
    artifact = manifest["artifact"]
    record = NumericalCompileRecord(
        backend="aoti",
        target=manifest["target"],
        compiler_version=manifest["torch_version"],
        exported_program_sha256=region["archive"]["sha256"],
        graph_sha256=region["capture_evidence"]["graph_digest"],
        artifact_sha256=artifact["sha256"],
        artifact_size_bytes=artifact["size_bytes"],
        reference_policy=policy,
        requested_compile_policy=requested,
        observed_compile_policy=observed,
        configuration_json=canonical_json(
            {
                "profile": build["profile"],
                "inductor_configs": build["configs"],
                "compiler_source_files": build["source_files"],
                "context_implementation": build["context_implementation"],
                "complete_python_context_before": before.to_dict(),
                "complete_python_context_after": after.to_dict(),
                "observation_scope": "immediately before and after public compiler CLI; full output verification remains a separate gate",
                "backend_program_audit": manifest["backend_program_audit"],
                "backend_graph_passes": manifest["backend_graph_passes"],
                "backend_graph_rewrites": manifest["backend_graph_rewrites"],
                "backend_package_audit": manifest["backend_package_audit"],
            }
        ),
    )
    binding = RegionNumericalBinding(
        name,
        NumericalRequirement(
            policy, "same-precision", policy.digest(), record.digest()
        ),
        record,
        PROVIDER_REQUIRED,
    )
    binding.require_runtime_deployable()
    return binding


def binding_from_legacy_compile(name, context, region, manifest, entry, build):
    """Bind pre-observed-context compile reports that kept one complete context.

    H100 artifacts from the earlier public compile path stored the full
    numerical context at build-report level and verified caller restoration,
    but not per-Region before/after fields. The same compile manifest, target,
    profile and configuration are still bound and required.
    """
    problems = []
    if entry.get("region") != name or entry.get("status") != "compiled-unvalidated":
        problems.append("entry-region-or-status")
        raise ValueError("legacy numeric binding entry does not match selected Region")
    if entry.get("export") != region.get("archive"):
        problems.append("entry-export")
        raise ValueError("legacy compile export digest differs from selected Region")
    if manifest.get("exported_program", {}).get("sha256") != region["archive"]["sha256"]:
        problems.append("exported-program-sha")
    if context.torch_version is not None:
        if manifest.get("torch_version") != context.torch_version:
            problems.append("torch-version")
    elif manifest.get("torch_version") != build.get("torch") or manifest.get(
        "cuda_version"
    ) != build.get("cuda"):
        problems.append("torch-cuda-version")
    if manifest.get("target") != build.get("target"):
        problems.append("target")
    if manifest.get("inductor_profile") != build.get("profile"):
        problems.append("profile")
    if canonical_json(manifest.get("inductor_configs")) != canonical_json(build.get("configs")):
        problems.append("configs")
    if build.get("numerical_context") != context.to_dict():
        problems.append("numerical-context")
    if build.get("caller_policy_restoration_verified") is not True:
        problems.append("caller-restoration")
    if problems:
        raise ValueError(
            "legacy numeric compile binding mismatch: " + ",".join(problems)
        )
    policy = policy_from_context(v2_from_legacy(context))
    artifact = manifest["artifact"]
    record = NumericalCompileRecord(
        backend="aoti",
        target=manifest["target"],
        compiler_version=manifest["torch_version"],
        exported_program_sha256=region["archive"]["sha256"],
        graph_sha256=region["capture_evidence"]["graph_digest"],
        artifact_sha256=artifact["sha256"],
        artifact_size_bytes=artifact["size_bytes"],
        reference_policy=policy,
        requested_compile_policy=policy,
        observed_compile_policy=policy,
        configuration_json=canonical_json(
            {
                "profile": build["profile"],
                "inductor_configs": build["configs"],
                "compiler_source_files": build["source_files"],
                "context_implementation": build["context_implementation"],
                "complete_python_context": context.to_dict(),
                "enforceable_python_context": v2_from_legacy(context).to_dict(),
                "observation_scope": (
                    "legacy build-report-level complete context with caller "
                    "restoration verified; no per-Region before/after fields"
                ),
                "backend_program_audit": manifest["backend_program_audit"],
                "backend_graph_passes": manifest["backend_graph_passes"],
                "backend_graph_rewrites": manifest["backend_graph_rewrites"],
                "backend_package_audit": manifest["backend_package_audit"],
            }
        ),
    )
    binding = RegionNumericalBinding(
        name,
        NumericalRequirement(
            policy, "same-precision", policy.digest(), record.digest()
        ),
        record,
        PROVIDER_REQUIRED,
    )
    binding.require_runtime_deployable()
    return binding


def render_explicit_numerical_worker(
    runner, bindings, *, acknowledge_exclusive_process, acknowledge_calling_thread
):
    """Wrap the generated runner's entrypoint with one explicit public bootstrap."""
    from vlaforge.codegen.numerical import generate_libtorch_worker_initializer

    if runner.count("int main(int argc, char** argv)") != 1:
        raise ValueError("explicit worker wrapper requires the checked main signature")
    initializer = generate_libtorch_worker_initializer(
        tuple(bindings),
        acknowledge_exclusive_process=acknowledge_exclusive_process,
        acknowledge_calling_thread=acknowledge_calling_thread,
    )
    return (
        "#include <cstdio>\n"
        + initializer
        + "\n#define main vlaforge_explicit_worker_body\n"
        + runner
        + "\n#undef main\n"
        + """
int main(int argc, char** argv) {
  const auto initialized = vlaforge_initialize_numerical_worker();
  if (initialized.code != VLAFORGE_STATUS_OK) {
    std::fprintf(stderr, "explicit numerical worker initialization failed: %u\\n",
        static_cast<unsigned>(initialized.code));
    return 60;
  }
  return vlaforge_explicit_worker_body(argc, argv);
}
"""
    )
