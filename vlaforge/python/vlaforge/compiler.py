"""VLA-specific whole-program compiler for passive Invocation IR v0.2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from vlaforge.ir.program import Module
from vlaforge.ir.serializer import io_schema_digest, module_digest
from vlaforge.plan import (
    ArtifactVariant,
    PlanModule,
    StorageOverride,
    lower_to_plan,
    physicalize_plan,
)
from vlaforge.transforms import (
    analyze_structured_loop_invariance,
    canonicalize,
    configure_exact_cache,
)


COMPILATION_CERTIFICATE_SCHEMA = "vlaforge.compilation_certificate/2"
EXACT_CACHE_IDENTITIES = ("episode", "model", "artifact")


class CompilerProfile(str, Enum):
    """The only optimization choices exposed by the deployment compiler."""

    OFF = "off"
    VERIFIED = "verified"
    FORCE_ON = "force-on"

    @property
    def test_only(self) -> bool:
        return self is CompilerProfile.FORCE_ON

    @classmethod
    def parse(cls, value: "CompilerProfile | str") -> "CompilerProfile":
        if isinstance(value, cls):
            return value
        aliases = {
            "conservative": cls.OFF,
            "auto": cls.VERIFIED,
            "force_on": cls.FORCE_ON,
        }
        spelling = str(value)
        return aliases[spelling] if spelling in aliases else cls(spelling)


@dataclass(frozen=True, slots=True)
class PassCertificate:
    name: str
    enabled: bool
    applied: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "applied": self.applied,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PassCertificate":
        return cls(
            name=str(data["name"]),
            enabled=bool(data["enabled"]),
            applied=bool(data["applied"]),
            reason=str(data["reason"]),
        )


@dataclass(frozen=True, slots=True)
class ExactCacheCertificate:
    task_id: int
    region: str
    requested: bool
    enabled: bool
    reason: str
    input_ids: tuple[int, ...] = ()
    state_ids: tuple[int, ...] = ()
    identity_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.task_id < 0 or not self.region or not self.reason:
            raise ValueError("cache certificate requires task, region, and reason")
        if self.enabled:
            if not self.requested:
                raise ValueError("enabled cache must be explicitly requested")
            if self.identity_fields != EXACT_CACHE_IDENTITIES:
                raise ValueError("exact cache identity fields are incomplete")
        if len(self.input_ids) != len(set(self.input_ids)):
            raise ValueError("cache input ids must be unique")
        if len(self.state_ids) != len(set(self.state_ids)):
            raise ValueError("cache state ids must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "region": self.region,
            "requested": self.requested,
            "enabled": self.enabled,
            "reason": self.reason,
            "input_ids": list(self.input_ids),
            "state_ids": list(self.state_ids),
            "identity_fields": list(self.identity_fields),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExactCacheCertificate":
        return cls(
            task_id=int(data["task_id"]),
            region=str(data["region"]),
            requested=bool(data["requested"]),
            enabled=bool(data["enabled"]),
            reason=str(data["reason"]),
            input_ids=tuple(int(item) for item in data.get("input_ids", ())),
            state_ids=tuple(int(item) for item in data.get("state_ids", ())),
            identity_fields=tuple(
                str(item) for item in data.get("identity_fields", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class LoopInvariantCertificate:
    region: str
    loop: str
    disposition: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "region": self.region,
            "loop": self.loop,
            "disposition": self.disposition,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any]
    ) -> "LoopInvariantCertificate":
        return cls(
            region=str(data["region"]),
            loop=str(data["loop"]),
            disposition=str(data["disposition"]),
            reason=str(data["reason"]),
        )


@dataclass(frozen=True, slots=True)
class ArenaCertificate:
    enabled: bool
    baseline_bytes: int
    compiled_bytes: int
    baseline_allocations: int
    compiled_allocations: int

    @property
    def saved_bytes(self) -> int:
        return self.baseline_bytes - self.compiled_bytes

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "baseline_bytes": self.baseline_bytes,
            "compiled_bytes": self.compiled_bytes,
            "saved_bytes": self.saved_bytes,
            "baseline_allocations": self.baseline_allocations,
            "compiled_allocations": self.compiled_allocations,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArenaCertificate":
        result = cls(
            enabled=bool(data["enabled"]),
            baseline_bytes=int(data["baseline_bytes"]),
            compiled_bytes=int(data["compiled_bytes"]),
            baseline_allocations=int(data["baseline_allocations"]),
            compiled_allocations=int(data["compiled_allocations"]),
        )
        if int(data.get("saved_bytes", result.saved_bytes)) != result.saved_bytes:
            raise ValueError("arena saved_bytes does not match layout sizes")
        return result


@dataclass(frozen=True, slots=True)
class CompilationCertificate:
    profile: CompilerProfile
    test_only: bool
    input_semantic_digest: str
    compiled_semantic_digest: str
    io_schema_digest: str
    plan_digest: str
    passes: tuple[PassCertificate, ...]
    caches: tuple[ExactCacheCertificate, ...]
    loops: tuple[LoopInvariantCertificate, ...]
    arena: ArenaCertificate
    schema: str = COMPILATION_CERTIFICATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != COMPILATION_CERTIFICATE_SCHEMA:
            raise ValueError(
                f"unsupported compilation certificate schema: {self.schema}"
            )
        for value in (
            self.input_semantic_digest,
            self.compiled_semantic_digest,
            self.io_schema_digest,
            self.plan_digest,
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError("certificate digests must be SHA-256")
        if self.test_only != self.profile.test_only:
            raise ValueError("certificate test_only flag disagrees with profile")
        task_ids = tuple(item.task_id for item in self.caches)
        if task_ids != tuple(sorted(set(task_ids))):
            raise ValueError("cache certificates must have sorted unique task ids")

    def cache_for_task(self, task_id: int) -> ExactCacheCertificate | None:
        return next(
            (item for item in self.caches if item.task_id == task_id),
            None,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "profile": self.profile.value,
            "test_only": self.test_only,
            "input_semantic_digest": self.input_semantic_digest,
            "compiled_semantic_digest": self.compiled_semantic_digest,
            "io_schema_digest": self.io_schema_digest,
            "plan_digest": self.plan_digest,
            "passes": [item.to_dict() for item in self.passes],
            "caches": [item.to_dict() for item in self.caches],
            "loops": [item.to_dict() for item in self.loops],
            "arena": self.arena.to_dict(),
        }

    def canonical_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
        )

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CompilationCertificate":
        return cls(
            schema=str(data["schema"]),
            profile=CompilerProfile.parse(str(data["profile"])),
            test_only=bool(data["test_only"]),
            input_semantic_digest=str(data["input_semantic_digest"]),
            compiled_semantic_digest=str(data["compiled_semantic_digest"]),
            io_schema_digest=str(data["io_schema_digest"]),
            plan_digest=str(data["plan_digest"]),
            passes=tuple(
                PassCertificate.from_dict(item)
                for item in data.get("passes", ())
            ),
            caches=tuple(
                ExactCacheCertificate.from_dict(item)
                for item in data.get("caches", ())
            ),
            loops=tuple(
                LoopInvariantCertificate.from_dict(item)
                for item in data.get("loops", ())
            ),
            arena=ArenaCertificate.from_dict(data["arena"]),
        )


@dataclass(frozen=True, slots=True)
class CompilationResult:
    module: Module
    plan: PlanModule
    baseline_plan: PlanModule
    certificate: CompilationCertificate


def compile_module(
    module: Module,
    *,
    profile: CompilerProfile | str = CompilerProfile.VERIFIED,
    artifact_variants: Mapping[str, ArtifactVariant] | None = None,
    default_device: str = "cpu",
    state_device: str = "cpu",
    allow_test_profile: bool = False,
) -> CompilationResult:
    """Compile one caller-driven VLA invocation with auditable contracts."""

    selected = CompilerProfile.parse(profile)
    if selected.test_only and not allow_test_profile:
        raise ValueError(
            "force-on is a test-only profile; pass allow_test_profile=True"
        )
    enabled = selected is not CompilerProfile.OFF
    input_digest = module_digest(module)
    requested = {
        region.name: bool(region.metadata.get("memoize", False))
        for region in module.regions
    }
    canonical = canonicalize(module)
    compiled_module = configure_exact_cache(canonical, enabled=enabled)
    loop_analysis = analyze_structured_loop_invariance(compiled_module)
    lowered = lower_to_plan(
        compiled_module,
        artifact_variants=artifact_variants,
    )
    state_overrides = {
        state.name: StorageOverride(device=state_device)
        for state in lowered.states
    }
    baseline = physicalize_plan(
        lowered,
        state_overrides=state_overrides,
        default_device=default_device,
        reuse_temporaries=False,
    )
    compiled = physicalize_plan(
        lowered,
        state_overrides=state_overrides,
        default_device=default_device,
        reuse_temporaries=enabled,
    )

    caches = []
    for task in compiled.tasks:
        if task.opcode != "vla.invoke":
            continue
        region = str(task.attributes["region"])
        cache_requested = requested[region]
        if not cache_requested:
            continue
        cache_enabled = bool(
            compiled_module.region(region).metadata.get("memoize", False)
        )
        region_metadata = compiled_module.region(region).metadata
        caches.append(
            ExactCacheCertificate(
                task_id=task.id,
                region=region,
                requested=True,
                enabled=cache_enabled,
                reason=(
                    "exact key uses input revisions, committed state versions, "
                    "and episode/model/artifact identity"
                    if cache_enabled
                    else "compiler profile disables exact cache reuse"
                ),
                input_ids=(
                    tuple(
                        compiled_module.input(name).input_id
                        for name in region_metadata.get(
                            "cache_input_ports",
                            tuple(
                                port.name
                                for port in compiled_module.inputs
                            ),
                        )
                    )
                    if cache_enabled
                    else ()
                ),
                state_ids=(
                    tuple(
                        next(
                            index
                            for index, state in enumerate(
                                compiled_module.states
                            )
                            if state.name == name
                        )
                        for name in region_metadata.get(
                            "cache_state_slots",
                            tuple(
                                state.name
                                for state in compiled_module.states
                            ),
                        )
                    )
                    if cache_enabled
                    else ()
                ),
                identity_fields=(
                    EXACT_CACHE_IDENTITIES if cache_enabled else ()
                ),
            )
        )

    assert baseline.arena is not None
    assert compiled.arena is not None
    arena = ArenaCertificate(
        enabled=enabled,
        baseline_bytes=baseline.arena.size_bytes,
        compiled_bytes=compiled.arena.size_bytes,
        baseline_allocations=len(baseline.arena.physical_buffers),
        compiled_allocations=len(compiled.arena.physical_buffers),
    )
    loops = tuple(
        LoopInvariantCertificate(
            item.region,
            item.loop,
            item.disposition,
            item.reason,
        )
        for item in loop_analysis.decisions
    )
    certificate = CompilationCertificate(
        profile=selected,
        test_only=selected.test_only,
        input_semantic_digest=input_digest,
        compiled_semantic_digest=module_digest(compiled_module),
        io_schema_digest=io_schema_digest(compiled_module),
        plan_digest=compiled.digest(),
        passes=(
            PassCertificate(
                "exact_cache_contract",
                enabled,
                any(item.enabled for item in caches),
                (
                    "only explicit exact memoize regions are enabled"
                    if enabled
                    else "disabled by off profile"
                ),
            ),
            PassCertificate(
                "structured_loop_invariance",
                enabled,
                any(
                    item.disposition == "prehoisted"
                    for item in loops
                ),
                "bounded-for preheader and loop-carried decisions recorded",
            ),
            PassCertificate(
                "static_arena_reuse",
                enabled,
                arena.saved_bytes > 0,
                (
                    "liveness-based temporary packing"
                    if enabled
                    else "disabled by off profile"
                ),
            ),
        ),
        caches=tuple(sorted(caches, key=lambda item: item.task_id)),
        loops=loops,
        arena=arena,
    )
    return CompilationResult(
        module=compiled_module,
        plan=compiled,
        baseline_plan=baseline,
        certificate=certificate,
    )
