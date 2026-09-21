"""Typed Session checks and a separate, explicitly requested worker bootstrap."""

from __future__ import annotations


def _literal(value: str) -> str:
    # Octal bytes avoid C++ universal-character restrictions and hex greediness.
    return '"' + "".join(f"\\{byte:03o}" for byte in value.encode("utf-8")) + '"'


def _requirement_table(requirement, suffix):
    entries = []
    for name, value in requirement.policy.values:
        integer, real, text, size = "0", "0.0", "nullptr", 0
        if type(value) is bool:
            kind, integer = "BOOL", str(int(value))
        elif type(value) is int:
            if not -(2**63) <= value < 2**63:
                raise ValueError(
                    "numerical integer is outside the native int64 transport"
                )
            kind = "I64"
            integer = (
                "(-9223372036854775807LL - 1LL)" if value == -(2**63) else f"{value}LL"
            )
        elif type(value) is float:
            kind, real = "F64", value.hex()
        else:
            kind, text, size = "STRING", _literal(value), len(value.encode("utf-8"))
        entries.append(
            f"  {{{_literal(name)}, {len(name.encode('utf-8'))}u, VLAFORGE_NUMERICAL_{kind}, "
            f"{integer}, {real}, {text}, {size}u}},"
        )
    namespace = requirement.policy.namespace
    return f"""constexpr VLAForgeNumericalEntry kNumericalEntries{suffix}[] = {{
{chr(10).join(entries)}
}};
constexpr VLAForgeNumericalRequirementView kNumericalRequirement{suffix}{{
    sizeof(VLAForgeNumericalRequirementView), VLAFORGE_NUMERICAL_ABI_VERSION,
    {_literal(namespace)}, {len(namespace.encode("utf-8"))}u,
    "{requirement.policy.digest()}", 64u, "{requirement.digest()}", 64u,
    {1 if requirement.execution_lane == "same-precision" else 2}u,
    kNumericalEntries{suffix}, {len(entries)}u}};"""


def generate_libtorch_worker_initializer(
    bindings, *, acknowledge_exclusive_process: bool, acknowledge_calling_thread: bool
) -> str:
    """Render bootstrap source to prepend to a deployment runner, not a Session.

    The runner explicitly calls ``vlaforge_initialize_numerical_worker()`` before
    constructing Sessions or starting model/device work and checks its status.
    Both acknowledgements declare caller ownership; they cannot establish it.
    Only already verified LibTorch bindings with one common policy are accepted.
    """
    from vlaforge.deployment.libtorch_numerical import require_libtorch_policy
    from vlaforge.deployment.numerical import (
        NumericalContractError,
        RegionNumericalBinding,
    )

    if (
        acknowledge_exclusive_process is not True
        or acknowledge_calling_thread is not True
    ):
        raise NumericalContractError("explicit process/thread ownership is required")
    bindings = tuple(bindings)
    if not bindings:
        raise NumericalContractError(
            "worker initialization requires numerical bindings"
        )
    for binding in bindings:
        if not isinstance(binding, RegionNumericalBinding):
            raise NumericalContractError(
                "worker initialization requires typed bindings"
            )
        binding.require_runtime_deployable()
        if binding.compile_record.backend not in ("aoti", "torchscript"):
            raise NumericalContractError(
                "worker initialization requires LibTorch bindings"
            )
        require_libtorch_policy(binding.requirement.policy)
    requirement = bindings[0].requirement
    if any(item.requirement.policy != requirement.policy for item in bindings[1:]):
        raise NumericalContractError(
            "conflicting numerical policies require separate workers"
        )
    return f"""#include "vlaforge/backends/libtorch_numerical.h"
namespace {{
{_requirement_table(requirement, "Worker")}
}}
extern "C" VLAForgeStatus vlaforge_initialize_numerical_worker() {{
  const VLAForgeLibTorchNumericalWorkerOptions options{{
      sizeof(VLAForgeLibTorchNumericalWorkerOptions),
      VLAFORGE_LIBTORCH_NUMERICAL_WORKER_ABI_VERSION,
      VLAFORGE_LIBTORCH_NUMERICAL_EXCLUSIVE_PROCESS | VLAFORGE_LIBTORCH_NUMERICAL_CALLING_THREAD}};
  return vlaforge_libtorch_numerical_initialize_worker(&kNumericalRequirementWorker, &options);
}}
"""


def validate_bindings(certificate, artifacts) -> None:
    from collections.abc import Mapping

    expected = tuple(
        artifacts[name].numerical_binding
        for name in sorted(artifacts)
        if artifacts[name].numerical_binding is not None
    )
    if isinstance(certificate, Mapping):
        if "numerical_bindings" in certificate:
            raise ValueError(
                "numerical runtime enforcement is unimplemented for an unverified certificate mapping"
            )
        actual = ()
    else:
        actual = getattr(certificate, "numerical_bindings", ())
    for binding in actual:
        binding.require_runtime_deployable()
    if tuple(item.digest() for item in actual) != tuple(
        item.digest() for item in expected
    ):
        raise ValueError("verified certificate/artifact numerical bindings differ")


class NumericalCodegen:
    def __init__(self, module, artifacts):
        self.backends = {
            index: artifacts[region.name].backend
            for index, region in enumerate(module.regions)
            if region.name in artifacts
        }
        self.bindings = {
            index: artifacts[region.name].numerical_binding
            for index, region in enumerate(module.regions)
            if region.name in artifacts
            and artifacts[region.name].numerical_binding is not None
        }
        self.enabled = bool(self.bindings)

    def tables(self):
        return "\n".join(
            _requirement_table(binding.requirement, index)
            for index, binding in self.bindings.items()
        )

    def declarations(self):
        return (
            """
  vlaforge::runtime::Status DrainNumerical() noexcept;
  vlaforge::runtime::Status CheckNumerical(VLAForgeNumericalBoundary boundary) noexcept;
  vlaforge::runtime::Status FailNumerical(vlaforge::runtime::Status status) noexcept;
"""
            if self.enabled
            else ""
        )

    def fields(self, count):
        return (
            f"""
  vlaforge::runtime::NumericalLeaseSet numerical_leases_;
  std::array<bool, {count}> numerical_region_loaded_{{}};
  bool numerical_drain_failed_ = false;
"""
            if self.enabled
            else ""
        )

    def preflight(self, index):
        if index not in self.bindings:
            return ""
        backend = self.backends[index]
        if backend in ("aoti", "torchscript"):
            setup = """    const auto* numerical_provider = vlaforge_libtorch_numerical_provider_api();
    c_status = vlaforge_numerical_provider_api_validate(numerical_provider);"""
        elif backend == "shared_plugin":
            setup = f"""    const VLAForgeNumericalProviderApi* numerical_provider = nullptr;
    c_status = vlaforge_external_region_plugin_numerical_provider(
        region_plugins_[{index}u], &numerical_provider);"""
        else:
            raise ValueError(f"numerical provider unavailable for backend {backend}")
        return f"""{setup}
    if (c_status.code != VLAFORGE_STATUS_OK) {{
      DestroyRegions();
      return vlaforge::runtime::Status::Error(
          vlaforge::runtime::StatusCode::kFailedPrecondition, {index}u,
          "required numerical provider sidecar is absent or invalid");
    }}
    auto numerical_status = numerical_leases_.Add(
        {index}u, numerical_provider, &kNumericalRequirement{index});
    if (!numerical_status.ok()) {{ DestroyRegions(); return numerical_status; }}
"""

    def bind(self, index):
        if index not in self.bindings:
            return ""
        return f"""    auto numerical_status = numerical_leases_.Bind(
        {index}u, region_executables_[{index}u]);
    if (!numerical_status.ok()) {{ DestroyRegion({index}u); return numerical_status; }}
"""

    def acquire(self):
        return (
            """  auto numerical_status = numerical_leases_.AcquireAll();
  if (!numerical_status.ok()) { DestroyRegions(); return numerical_status; }
  numerical_status = numerical_leases_.Validate(VLAFORGE_NUMERICAL_BEFORE_LOAD);
  if (!numerical_status.ok()) { DestroyRegions(); return numerical_status; }
"""
            if self.enabled
            else ""
        )

    def after_load(self):
        return (
            """  auto checked = CheckNumerical(VLAFORGE_NUMERICAL_AFTER_LOAD);
  if (!checked.ok()) { DestroyRegions(); return checked; }
"""
            if self.enabled
            else ""
        )

    def run_entry(self):
        return (
            """  auto numerical_status = CheckNumerical(VLAFORGE_NUMERICAL_RUN_ENTRY);
  if (!numerical_status.ok()) { return numerical_status; }
"""
            if self.enabled
            else ""
        )

    def before_commit(self):
        return (
            [
                "status = CheckNumerical(VLAFORGE_NUMERICAL_BEFORE_COMMIT);",
                "if (!status.ok()) { return status; }",
            ]
            if self.enabled
            else []
        )

    def destruction_prefix(self):
        return (
            """  (void)DrainNumerical();
  if (numerical_drain_failed_) {
    // Pending work may retain all storage/provider pointers. Exit the worker.
    arena_.Abandon();
    state_arena_.Abandon();
    for (auto& arena : output_slots_a_) { if (arena) { arena->Abandon(); } }
    for (auto& arena : output_slots_b_) { if (arena) { arena->Abandon(); } }
    region_executables_.fill(nullptr);
    region_plugins_.fill(nullptr);
    execution_contexts_.fill(nullptr);
    numerical_leases_.Abandon();
    return;
  }
"""
            if self.enabled
            else ""
        )

    def methods(self, cache_reset, destroy_replays):
        if not self.enabled:
            return ""
        return f"""
vlaforge::runtime::Status ModelSession::DrainNumerical() noexcept {{
  auto result = vlaforge::runtime::Status::Ok();
  for (std::size_t i = 0; i < region_executables_.size(); ++i) {{
    if (region_executables_[i] != nullptr && numerical_region_loaded_[i]) {{
      if (region_apis_[i]->synchronize(region_executables_[i]).code != VLAFORGE_STATUS_OK) {{
        numerical_drain_failed_ = true;
        result = vlaforge::runtime::Status::Error(
            vlaforge::runtime::StatusCode::kFailedPrecondition, i,
            "numerical boundary drain failed; exit worker process");
      }}
    }}
  }}
  for (auto* context : execution_contexts_) {{
    if (context != nullptr && vlaforge_execution_context_synchronize(context).code != VLAFORGE_STATUS_OK) {{
      numerical_drain_failed_ = true;
      result = vlaforge::runtime::Status::Error(
          vlaforge::runtime::StatusCode::kFailedPrecondition, 0u,
          "numerical context drain failed; exit worker process");
    }}
  }}
  return result;
}}

vlaforge::runtime::Status ModelSession::FailNumerical(
    vlaforge::runtime::Status status) noexcept {{
  const auto drained = DrainNumerical();
  if (!drained.ok()) {{ status = drained; }}
{cache_reset}
  if (!numerical_drain_failed_) {{ {destroy_replays} }}
  initialization_status_ = status;
  return Fail(status);
}}

vlaforge::runtime::Status ModelSession::CheckNumerical(
    VLAForgeNumericalBoundary boundary) noexcept {{
  if (boundary != VLAFORGE_NUMERICAL_RUN_ENTRY) {{
    auto status = DrainNumerical();
    if (!status.ok()) {{ return FailNumerical(status); }}
  }}
  const auto status = numerical_leases_.Validate(boundary);
  return status.ok() ? status : FailNumerical(status);
}}
"""
