"""Small production compiler profile and legality-certificate contract.

The profile is deliberately VLA-specific.  It selects only the temporal passes
implemented by VLAForge and records every decision consumed by generated C++.
It is not a general-purpose pass manager.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from vlaforge.ir.program import Module
from vlaforge.ir.serializer import module_digest
from vlaforge.plan import (
    ArtifactVariant,
    PlanModule,
    lower_to_plan,
    physicalize_plan,
)
from vlaforge.transforms import (
    synthesize_epoch_memoization,
    temporal_loop_invariant_code_motion,
)


COMPILATION_CERTIFICATE_SCHEMA = "vlaforge.compilation_certificate/1"


class CompilerProfile(str, Enum):
    """Supported production optimization policies."""

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
class CertifiedDependency:
    kind: str
    value: str
    subject: str
    subject_id: int
    max_age_ns: int | None = None
    max_versions: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"epoch", "state_version"}:
            raise ValueError(f"unsupported certified dependency: {self.kind}")
        if not self.value or not self.subject or self.subject_id < 0:
            raise ValueError("certified dependency requires value and subject")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "value": self.value,
            "subject": self.subject,
            "subject_id": self.subject_id,
            "max_age_ns": self.max_age_ns,
            "max_versions": self.max_versions,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CertifiedDependency":
        return cls(
            kind=str(data["kind"]),
            value=str(data["value"]),
            subject=str(data["subject"]),
            subject_id=int(data["subject_id"]),
            max_age_ns=(
                None
                if data.get("max_age_ns") is None
                else int(data["max_age_ns"])
            ),
            max_versions=(
                None
                if data.get("max_versions") is None
                else int(data["max_versions"])
            ),
        )


@dataclass(frozen=True, slots=True)
class CacheLegalityCertificate:
    task_id: int
    region: str
    legal: bool
    enabled: bool
    reason: str
    dependencies: tuple[CertifiedDependency, ...] = ()

    def __post_init__(self) -> None:
        if self.task_id < 0 or not self.region or not self.reason:
            raise ValueError("cache certificate requires task, region, and reason")
        if self.enabled and (not self.legal or not self.dependencies):
            raise ValueError("enabled cache requires a complete legal signature")

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "region": self.region,
            "legal": self.legal,
            "enabled": self.enabled,
            "reason": self.reason,
            "dependencies": [item.to_dict() for item in self.dependencies],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CacheLegalityCertificate":
        return cls(
            task_id=int(data["task_id"]),
            region=str(data["region"]),
            legal=bool(data["legal"]),
            enabled=bool(data["enabled"]),
            reason=str(data["reason"]),
            dependencies=tuple(
                CertifiedDependency.from_dict(item)
                for item in data.get("dependencies", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class LICMLegalityCertificate:
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
    def from_dict(cls, data: Mapping[str, Any]) -> "LICMLegalityCertificate":
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
    plan_digest: str
    passes: tuple[PassCertificate, ...]
    caches: tuple[CacheLegalityCertificate, ...]
    licm: tuple[LICMLegalityCertificate, ...]
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
            self.plan_digest,
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError("certificate digests must be SHA-256")
        if self.test_only != self.profile.test_only:
            raise ValueError("certificate test_only flag disagrees with profile")
        task_ids = [item.task_id for item in self.caches]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("duplicate cache task certificate")
        if tuple(sorted(task_ids)) != tuple(task_ids):
            raise ValueError("cache certificates must be sorted by task id")

    def cache_for_task(self, task_id: int) -> CacheLegalityCertificate | None:
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
            "plan_digest": self.plan_digest,
            "passes": [item.to_dict() for item in self.passes],
            "caches": [item.to_dict() for item in self.caches],
            "licm": [item.to_dict() for item in self.licm],
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
            plan_digest=str(data["plan_digest"]),
            passes=tuple(
                PassCertificate.from_dict(item)
                for item in data.get("passes", ())
            ),
            caches=tuple(
                CacheLegalityCertificate.from_dict(item)
                for item in data.get("caches", ())
            ),
            licm=tuple(
                LICMLegalityCertificate.from_dict(item)
                for item in data.get("licm", ())
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
    allow_test_profile: bool = False,
) -> CompilationResult:
    """Compile a Semantic IR module under an explicit optimization profile."""

    selected = CompilerProfile.parse(profile)
    if selected.test_only and not allow_test_profile:
        raise ValueError(
            "force-on is a test-only profile; pass allow_test_profile=True"
        )

    input_digest = module_digest(module)
    enabled = selected is not CompilerProfile.OFF
    licm_decisions = ()
    if enabled:
        memoized = synthesize_epoch_memoization(module)
        licm_result = temporal_loop_invariant_code_motion(memoized)
        compiled_module = licm_result.module
        licm_decisions = licm_result.decisions
    else:
        compiled_module = module

    lowered = lower_to_plan(
        compiled_module,
        artifact_variants=artifact_variants,
    )
    baseline = physicalize_plan(lowered, reuse_temporaries=False)
    compiled = (
        physicalize_plan(lowered, reuse_temporaries=True)
        if enabled
        else baseline
    )

    input_ids = {
        stream.name: index for index, stream in enumerate(compiled_module.inputs)
    }
    state_ids = {
        state.name: index for index, state in enumerate(compiled_module.states)
    }
    caches = []
    region_metadata = {
        region.name: region.metadata for region in compiled_module.regions
    }
    for task in compiled.tasks:
        if task.opcode != "vla.invoke":
            continue
        region = str(task.attributes["region"])
        if not bool(region_metadata[region].get("memoize", False)):
            continue
        if not enabled:
            caches.append(
                CacheLegalityCertificate(
                    task.id,
                    region,
                    legal=False,
                    enabled=False,
                    reason="compiler profile disables temporal memoization",
                )
            )
            continue
        dependencies = []
        for item in task.attributes.get("memoize_dependencies", ()):
            kind = str(item["kind"])
            subject = str(item["subject"])
            subject_id = (
                input_ids[subject]
                if kind == "epoch"
                else state_ids[subject]
            )
            dependencies.append(
                CertifiedDependency(
                    kind=kind,
                    value=str(item["value"]),
                    subject=subject,
                    subject_id=subject_id,
                    max_age_ns=(
                        None
                        if item.get("max_age_ns") is None
                        else int(item["max_age_ns"])
                    ),
                    max_versions=(
                        None
                        if item.get("max_versions") is None
                        else int(item["max_versions"])
                    ),
                )
            )
        caches.append(
            CacheLegalityCertificate(
                task.id,
                region,
                legal=bool(dependencies),
                enabled=bool(dependencies),
                reason=str(
                    task.attributes.get(
                        "memoize_semantics",
                        "missing dependency certificate",
                    )
                ),
                dependencies=tuple(dependencies),
            )
        )

    licm_certificates = tuple(
        LICMLegalityCertificate(
            item.region,
            item.loop,
            item.disposition,
            item.reason,
        )
        for item in licm_decisions
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
    passes = (
        PassCertificate(
            "epoch_memoization",
            enabled,
            any(item.enabled for item in caches),
            (
                "enabled only for invokes with complete epoch/state signatures"
                if enabled
                else "disabled by conservative profile"
            ),
        ),
        PassCertificate(
            "temporal_licm",
            enabled,
            any(
                item.disposition in {"moved", "prehoisted"}
                for item in licm_certificates
            ),
            (
                "legality decisions serialized per candidate"
                if enabled
                else "disabled by conservative profile"
            ),
        ),
        PassCertificate(
            "static_arena_reuse",
            enabled,
            arena.saved_bytes > 0,
            (
                "exact liveness interference packing"
                if enabled
                else "disabled by conservative profile"
            ),
        ),
    )
    certificate = CompilationCertificate(
        profile=selected,
        test_only=selected.test_only,
        input_semantic_digest=input_digest,
        compiled_semantic_digest=module_digest(compiled_module),
        plan_digest=compiled.digest(),
        passes=passes,
        caches=tuple(sorted(caches, key=lambda item: item.task_id)),
        licm=licm_certificates,
        arena=arena,
    )
    return CompilationResult(
        module=compiled_module,
        plan=compiled,
        baseline_plan=baseline,
        certificate=certificate,
    )
