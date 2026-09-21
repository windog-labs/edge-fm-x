"""Backend-owned execution capabilities, separate from model effect audits."""

from collections.abc import Iterable

from vlaforge.deployment.contract import BackendCapability


def torchscript_backend_capability(
    target: str, supported_dtypes: Iterable[str], *, shared_context: bool = False
) -> BackendCapability:
    """Static tensor-only ATen profile with optional CUDA shared context.

    Internal CPU scalar constants retain their placement. JIT optimization is
    disabled with a thread-local guard, not a process-global numerical setter.
    The default profile uses synchronous calls. The shared-context variant
    advertises a stream/graph provider, not proof that any artifact captures
    safely; effect, schedule and runtime capture gates still apply. Neither
    variant claims ownership of internal ATen workspace or numerical policy.
    """
    if target != "cpu" and not target.startswith("sm_"):
        raise ValueError("TorchScript target requires cpu or a CUDA SM")
    if shared_context and target == "cpu":
        raise ValueError("TorchScript shared context requires CUDA")
    return BackendCapability(
        backend="torchscript", target=target,
        supported_dtypes=tuple(supported_dtypes),
        supports_dynamic_shapes=False,
        supports_device_resident_io=target.startswith("sm_"),
        requires_synchronize=True,
        supports_external_cuda_graph=shared_context,
        supports_execution_context=shared_context,
    )


def aoti_backend_capability(
    target: str, supported_dtypes: Iterable[str], *, dynamic_shapes: bool = False
) -> BackendCapability:
    """Describe the AOTI provider, not a proof that an artifact captures safely.

    External graph support uses the explicit shared stream and LibTorch pool
    provider. Individual artifacts still require effect/memory/schedule checks
    and successful runtime warmup/capture. CPU has no graph provider.
    """

    cuda = target.startswith("sm_")
    return BackendCapability(
        backend="aoti",
        target=target,
        supported_dtypes=tuple(supported_dtypes),
        supports_dynamic_shapes=dynamic_shapes,
        supports_device_resident_io=cuda,
        requires_synchronize=True,
        supports_external_cuda_graph=cuda,
        supports_execution_context=cuda,
    )
