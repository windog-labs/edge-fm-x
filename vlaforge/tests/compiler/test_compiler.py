from __future__ import annotations

from dataclasses import replace

import pytest

from vlaforge.adapters import build_openvla_fixture, build_smolvla_fixture
from vlaforge.compiler import (
    CompilationCertificate,
    CompilerProfile,
    compile_module,
)
from vlaforge.plan import BufferClass
from vlaforge.transforms import MemoizationSynthesisError


@pytest.mark.parametrize(
    ("spelling", "expected"),
    (
        ("off", CompilerProfile.OFF),
        ("conservative", CompilerProfile.OFF),
        ("verified", CompilerProfile.VERIFIED),
        ("auto", CompilerProfile.VERIFIED),
    ),
)
def test_compiler_profiles_and_aliases(spelling, expected) -> None:
    result = compile_module(
        build_openvla_fixture().module,
        profile=spelling,
    )
    assert result.certificate.profile is expected
    assert result.certificate.test_only is False


def test_force_on_requires_explicit_test_authority() -> None:
    module = build_openvla_fixture().module
    with pytest.raises(ValueError, match="test-only"):
        compile_module(module, profile="force-on")
    result = compile_module(
        module,
        profile="force-on",
        allow_test_profile=True,
    )
    assert result.certificate.profile is CompilerProfile.FORCE_ON
    assert result.certificate.test_only


def test_verified_certificate_round_trip_and_digest_binding() -> None:
    result = compile_module(
        build_openvla_fixture().module,
        profile="verified",
    )
    certificate = result.certificate
    assert certificate.plan_digest == result.plan.digest()
    assert certificate.compiled_semantic_digest == result.plan.semantic_digest
    assert certificate.caches[0].legal
    assert certificate.caches[0].enabled
    assert [item.kind for item in certificate.caches[0].dependencies] == [
        "epoch",
        "epoch",
    ]
    decoded = CompilationCertificate.from_dict(certificate.to_dict())
    assert decoded == certificate
    assert decoded.digest() == certificate.digest()


def test_off_profile_is_conservative_and_has_no_cache_storage() -> None:
    result = compile_module(
        build_openvla_fixture().module,
        profile="off",
    )
    assert not any(item.enabled for item in result.certificate.caches)
    assert all(
        buffer.buffer_class is not BufferClass.TEMPORAL_CACHE
        for buffer in result.plan.buffers
    )
    assert result.plan.digest() == result.baseline_plan.digest()


def test_verified_cache_output_cannot_alias_across_ticks() -> None:
    result = compile_module(
        build_openvla_fixture().module,
        profile="verified",
    )
    cache_buffers = [
        buffer
        for buffer in result.plan.buffers
        if buffer.buffer_class is BufferClass.TEMPORAL_CACHE
    ]
    assert cache_buffers
    cache_physical = {
        physical.id
        for physical in result.plan.arena.physical_buffers
        if any(
            logical in {item.id for item in cache_buffers}
            for logical in physical.logical_buffers
        )
    }
    for physical in result.plan.arena.physical_buffers:
        if physical.id in cache_physical:
            assert len(physical.logical_buffers) == 1


def test_missing_unversioned_dependency_fails_verified_compile() -> None:
    fixture = build_smolvla_fixture()
    regions = tuple(
        replace(region, metadata={"memoize": True})
        if region.name == "queue_zero"
        else region
        for region in fixture.module.regions
    )
    with pytest.raises(
        MemoizationSynthesisError,
        match="memoize.missing_epoch_or_state",
    ):
        compile_module(
            replace(fixture.module, regions=regions),
            profile="verified",
        )


def test_compilation_is_deterministic_for_two_vla_structures() -> None:
    for fixture in (build_smolvla_fixture(), build_openvla_fixture()):
        first = compile_module(fixture.module, profile="verified")
        second = compile_module(fixture.module, profile="verified")
        assert first.plan.canonical_json() == second.plan.canonical_json()
        assert (
            first.certificate.canonical_json()
            == second.certificate.canonical_json()
        )
