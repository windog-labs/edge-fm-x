"""Strict PyTorch numerical policy for offline Python capture and replay.

Backend flags are process-global; autocast settings are current-thread state.
The restore guard is NOT safe alongside unrelated model execution or policy
changes. It protects only other guards in this module, not external callers.
This module neither specifies hardware/version reproducibility nor enforces C++.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
import json
from threading import Lock


LEGACY_SCHEMA = "vlaforge.python_numerical_context/1"
SCHEMA = "vlaforge.python_numerical_context/2"
_RESTORE_LOCK = Lock()
_DTYPES = frozenset(("float16", "bfloat16", "float32", "float64"))
_VERSION_FIELDS = frozenset(
    (
        "schema",
        "torch_version",
        "reduction_api",
        "cuda_matmul_allow_fp16_reduced_precision_reduction_split_k",
        "cuda_matmul_allow_bf16_reduced_precision_reduction_split_k",
    )
)
_BOOLEAN_REDUCTION_API = "torch-2.7.1/bool-reduction"
_SPLIT_K_REDUCTION_API = "torch-2.10.0/reduction-and-split-k"


class NumericalContextError(ValueError):
    """Incomplete, unsupported, conflicting, or changed numerical policy."""


def _reduction_api(version):
    from packaging.version import InvalidVersion, Version

    try:
        parsed = Version(version)
    except (InvalidVersion, TypeError) as error:
        raise NumericalContextError("invalid PyTorch policy version") from error
    if (
        parsed.epoch
        or parsed.is_prerelease
        or parsed.is_devrelease
        or parsed.is_postrelease
    ):
        raise NumericalContextError("unsupported PyTorch policy version")
    if parsed.release == (2, 7, 1):
        return _BOOLEAN_REDUCTION_API
    if parsed.release == (2, 10, 0):
        return _SPLIT_K_REDUCTION_API
    raise NumericalContextError(f"unsupported PyTorch policy version: {version}")


@dataclass(frozen=True, slots=True)
class NumericalContext:
    float32_matmul_precision: str
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool
    cuda_matmul_allow_fp16_reduced_precision_reduction: bool
    cuda_matmul_allow_bf16_reduced_precision_reduction: bool
    autocast_cpu_enabled: bool
    autocast_cpu_dtype: str
    autocast_cuda_enabled: bool
    autocast_cuda_dtype: str
    autocast_cache_enabled: bool
    sdpa_flash_enabled: bool
    sdpa_mem_efficient_enabled: bool
    sdpa_math_enabled: bool
    sdpa_cudnn_enabled: bool
    sdpa_math_allow_fp16_bf16_reduction: bool
    deterministic_algorithms_enabled: bool
    deterministic_algorithms_warn_only: bool
    cudnn_enabled: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    schema: str = LEGACY_SCHEMA
    torch_version: str | None = None
    reduction_api: str | None = None
    cuda_matmul_allow_fp16_reduced_precision_reduction_split_k: bool | None = None
    cuda_matmul_allow_bf16_reduced_precision_reduction_split_k: bool | None = None

    def __post_init__(self):
        for field in fields(self):
            if field.name in _VERSION_FIELDS:
                continue
            value = getattr(self, field.name)
            if field.name == "float32_matmul_precision":
                if value not in ("highest", "high", "medium"):
                    raise NumericalContextError("invalid float32_matmul_precision")
            elif field.name.endswith("_dtype"):
                if not isinstance(value, str) or value not in _DTYPES:
                    raise NumericalContextError(f"unsupported {field.name}: {value!r}")
            elif type(value) is not bool:
                raise NumericalContextError(f"{field.name} must be a boolean")
        if self.cuda_matmul_allow_tf32 != (self.float32_matmul_precision != "highest"):
            raise NumericalContextError("inconsistent matmul precision and TF32 policy")
        if self.schema == LEGACY_SCHEMA:
            if any(
                getattr(self, name) is not None for name in _VERSION_FIELDS - {"schema"}
            ):
                raise NumericalContextError(
                    "legacy context cannot claim uncaptured version domains"
                )
            return
        if self.schema != SCHEMA:
            raise NumericalContextError("unsupported numerical context schema")
        if not isinstance(
            self.torch_version, str
        ) or self.reduction_api != _reduction_api(self.torch_version):
            raise NumericalContextError("numerical context version/API domain mismatch")
        for dtype in ("fp16", "bf16"):
            prefix = f"cuda_matmul_allow_{dtype}_reduced_precision_reduction"
            split_k = getattr(self, prefix + "_split_k")
            if self.reduction_api == _BOOLEAN_REDUCTION_API:
                if split_k is not None:
                    raise NumericalContextError(
                        "bool-only API cannot claim captured split-K state"
                    )
            elif type(split_k) is not bool:
                raise NumericalContextError(
                    "split-K policy must be an explicit boolean"
                )
            elif getattr(self, prefix) and not split_k:
                raise NumericalContextError(
                    "unsupported reduced-precision/split-K pair"
                )

    @property
    def partial(self) -> bool:
        return self.schema == LEGACY_SCHEMA

    def _require_enforceable(self):
        if self.partial:
            raise NumericalContextError(
                "legacy /1 context is a partial observation, not an enforceable /2 policy; recapture required"
            )

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        if self.partial:
            value = {
                name: item
                for name, item in value.items()
                if name not in _VERSION_FIELDS
            }
            return {"schema": LEGACY_SCHEMA, **value}
        return value

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise NumericalContextError("numerical context must be an object")
        schema = value.get("schema")
        if schema not in (LEGACY_SCHEMA, SCHEMA):
            raise NumericalContextError("unsupported numerical context schema")
        required = {field.name for field in fields(cls)}
        if schema == LEGACY_SCHEMA:
            required -= _VERSION_FIELDS - {"schema"}
        if set(value) != required:
            raise NumericalContextError(
                f"numerical context fields differ: missing={sorted(required - set(value))}, "
                f"unknown={sorted(set(value) - required)}"
            )
        return cls(**value)

    @classmethod
    def from_json(cls, text):
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise NumericalContextError(
                        f"duplicate numerical context key: {key}"
                    )
                result[key] = value
            return result

        return cls.from_dict(json.loads(text, object_pairs_hook=unique_object))

    def require_current(self) -> None:
        self._require_enforceable()
        actual = snapshot()
        differences = {
            field.name: {
                "required": getattr(self, field.name),
                "observed": getattr(actual, field.name),
            }
            for field in fields(self)
            if getattr(self, field.name) != getattr(actual, field.name)
        }
        if differences:
            raise NumericalContextError(f"numerical context mismatch: {differences}")


def snapshot() -> NumericalContext:
    """Read every required flag; unavailable APIs fail instead of inventing defaults."""
    import torch

    cuda = torch.backends.cuda
    try:
        reduction_api = _reduction_api(torch.__version__)
        split_k = {}
        for dtype in ("fp16", "bf16"):
            name = f"allow_{dtype}_reduced_precision_reduction_split_k"
            split_k["cuda_matmul_" + name] = (
                getattr(cuda.matmul, name)
                if reduction_api == _SPLIT_K_REDUCTION_API
                else None
            )
        return NumericalContext(
            schema=SCHEMA,
            torch_version=str(torch.__version__),
            reduction_api=reduction_api,
            **split_k,
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            cuda_matmul_allow_tf32=cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            cuda_matmul_allow_fp16_reduced_precision_reduction=cuda.matmul.allow_fp16_reduced_precision_reduction,
            cuda_matmul_allow_bf16_reduced_precision_reduction=cuda.matmul.allow_bf16_reduced_precision_reduction,
            autocast_cpu_enabled=torch.is_autocast_enabled("cpu"),
            autocast_cpu_dtype=str(torch.get_autocast_dtype("cpu")).removeprefix(
                "torch."
            ),
            autocast_cuda_enabled=torch.is_autocast_enabled("cuda"),
            autocast_cuda_dtype=str(torch.get_autocast_dtype("cuda")).removeprefix(
                "torch."
            ),
            autocast_cache_enabled=torch.is_autocast_cache_enabled(),
            sdpa_flash_enabled=cuda.flash_sdp_enabled(),
            sdpa_mem_efficient_enabled=cuda.mem_efficient_sdp_enabled(),
            sdpa_math_enabled=cuda.math_sdp_enabled(),
            sdpa_cudnn_enabled=cuda.cudnn_sdp_enabled(),
            sdpa_math_allow_fp16_bf16_reduction=cuda.fp16_bf16_reduction_math_sdp_allowed(),
            deterministic_algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
            deterministic_algorithms_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            cudnn_enabled=torch.backends.cudnn.enabled,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
        )
    except (AttributeError, TypeError, RuntimeError) as exc:
        raise NumericalContextError(
            f"required numerical policy API unavailable: {exc}"
        ) from exc


def _apply(context: NumericalContext) -> None:
    import torch

    context._require_enforceable()
    cuda = torch.backends.cuda
    current = snapshot()
    if (current.torch_version, current.reduction_api) != (
        context.torch_version,
        context.reduction_api,
    ):
        raise NumericalContextError(
            "cannot restore a different PyTorch version/API domain"
        )
    # Matmul precision establishes its TF32 alias too, without a second write.
    if current.float32_matmul_precision != context.float32_matmul_precision:
        torch.set_float32_matmul_precision(context.float32_matmul_precision)
    if current.cudnn_allow_tf32 != context.cudnn_allow_tf32:
        torch.backends.cudnn.allow_tf32 = context.cudnn_allow_tf32
    for dtype in ("fp16", "bf16"):
        field = f"cuda_matmul_allow_{dtype}_reduced_precision_reduction"
        split_field = field + "_split_k"
        required = getattr(context, field)
        observed = getattr(current, field)
        if context.reduction_api == _SPLIT_K_REDUCTION_API:
            required = (required, getattr(context, split_field))
            observed = (observed, getattr(current, split_field))
        if observed != required:
            setattr(
                cuda.matmul,
                f"allow_{dtype}_reduced_precision_reduction",
                required,
            )
    for device in ("cpu", "cuda"):
        dtype_field = f"autocast_{device}_dtype"
        enabled_field = f"autocast_{device}_enabled"
        if getattr(current, dtype_field) != getattr(context, dtype_field):
            torch.set_autocast_dtype(
                device, getattr(torch, getattr(context, dtype_field))
            )
        if getattr(current, enabled_field) != getattr(context, enabled_field):
            torch.set_autocast_enabled(device, getattr(context, enabled_field))
    if current.autocast_cache_enabled != context.autocast_cache_enabled:
        torch.set_autocast_cache_enabled(context.autocast_cache_enabled)
    for field, setter in (
        ("sdpa_flash_enabled", cuda.enable_flash_sdp),
        ("sdpa_mem_efficient_enabled", cuda.enable_mem_efficient_sdp),
        ("sdpa_math_enabled", cuda.enable_math_sdp),
        ("sdpa_cudnn_enabled", cuda.enable_cudnn_sdp),
        (
            "sdpa_math_allow_fp16_bf16_reduction",
            cuda.allow_fp16_bf16_reduction_math_sdp,
        ),
    ):
        if getattr(current, field) != getattr(context, field):
            setter(getattr(context, field))
    if (
        current.deterministic_algorithms_enabled
        != context.deterministic_algorithms_enabled
        or current.deterministic_algorithms_warn_only
        != context.deterministic_algorithms_warn_only
    ):
        torch.use_deterministic_algorithms(
            context.deterministic_algorithms_enabled,
            warn_only=context.deterministic_algorithms_warn_only,
        )
    for field in ("enabled", "benchmark", "deterministic"):
        if getattr(current, "cudnn_" + field) != getattr(context, "cudnn_" + field):
            setattr(torch.backends.cudnn, field, getattr(context, "cudnn_" + field))
    context.require_current()


@contextmanager
def offline_restore(context: NumericalContext, *, acknowledge_process_global: bool):
    """Temporarily restore all fields in an isolated offline Python worker only.

    Callers must exclude unrelated execution and external policy mutations.
    Concurrent or nested guards are rejected; this is NOT general thread safety.
    Always restores the previous policy, including when setup or the body fails.
    """
    if acknowledge_process_global is not True:
        raise NumericalContextError(
            "offline restoration requires process-global acknowledgement"
        )
    if not isinstance(context, NumericalContext):
        raise NumericalContextError(
            "offline restoration requires a validated NumericalContext"
        )
    context._require_enforceable()
    if not _RESTORE_LOCK.acquire(blocking=False):
        raise NumericalContextError(
            "concurrent or nested numerical context restoration is unsafe"
        )
    previous = None
    try:
        previous = snapshot()
        _apply(context)
        yield context
        context.require_current()
    finally:
        try:
            if previous is not None:
                _apply(previous)
        finally:
            _RESTORE_LOCK.release()
