from dataclasses import replace

import pytest

from vlaforge.adapters import (
    DRIVING_FIXTURES,
    ROBOT_MATRIX_FIXTURES,
    build_openvla_fixture,
    build_pi0_fixture,
    build_smolvla_fixture,
)
from vlaforge.compiler import (
    EXACT_CACHE_IDENTITIES,
    CompilationCertificate,
    CompilerProfile,
    compile_module,
)
from vlaforge.ir.serializer import io_schema_digest
from vlaforge.plan import BufferClass
from vlaforge.transforms import ExactCacheContractError


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
    result = compile_module(build_openvla_fixture().module, profile=spelling)
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


def test_verified_certificate_binds_io_plan_and_exact_cache_key() -> None:
    result = compile_module(build_openvla_fixture().module)
    certificate = result.certificate
    assert certificate.plan_digest == result.plan.digest()
    assert certificate.compiled_semantic_digest == result.plan.semantic_digest
    assert certificate.io_schema_digest == result.plan.io_schema_digest
    assert certificate.io_schema_digest == io_schema_digest(result.module)
    cache = certificate.caches[0]
    assert cache.enabled
    assert cache.input_ids == (0, 1)
    assert cache.state_ids == ()
    assert cache.identity_fields == EXACT_CACHE_IDENTITIES
    assert all(
        "epoch" not in item.name and "clock" not in item.name
        for item in certificate.passes
    )
    decoded = CompilationCertificate.from_dict(certificate.to_dict())
    assert decoded == certificate
    assert decoded.digest() == certificate.digest()


def test_off_profile_has_no_cross_run_derived_cache() -> None:
    result = compile_module(build_openvla_fixture().module, profile="off")
    assert not any(item.enabled for item in result.certificate.caches)
    assert all(
        buffer.buffer_class is not BufferClass.DERIVED_CACHE
        for buffer in result.plan.buffers
    )
    assert result.plan.digest() == result.baseline_plan.digest()


def test_exact_cache_storage_never_aliases_temporary_buffers() -> None:
    result = compile_module(build_openvla_fixture().module)
    cache_ids = {
        buffer.id
        for buffer in result.plan.buffers
        if buffer.buffer_class is BufferClass.DERIVED_CACHE
    }
    assert cache_ids
    assert result.plan.arena is not None
    for physical in result.plan.arena.physical_buffers:
        if cache_ids.intersection(physical.logical_buffers):
            assert len(physical.logical_buffers) == 1


def test_guarded_approximate_reuse_cannot_masquerade_as_exact() -> None:
    fixture = build_smolvla_fixture()
    regions = tuple(
        replace(
            region,
            metadata={"memoize": True, "reuse_kind": "guarded_approximate"},
        )
        if region.name == "encode_observation"
        else region
        for region in fixture.module.regions
    )
    with pytest.raises(ExactCacheContractError, match="rejects"):
        compile_module(replace(fixture.module, regions=regions))


def test_compilation_is_deterministic_across_robot_and_driving_fixtures() -> None:
    fixtures = (
        build_smolvla_fixture(),
        build_openvla_fixture(),
        build_pi0_fixture(),
        *(builder() for builder in ROBOT_MATRIX_FIXTURES),
        *(builder() for builder in DRIVING_FIXTURES),
    )
    for fixture in fixtures:
        first = compile_module(fixture.module)
        second = compile_module(fixture.module)
        assert first.plan.canonical_json() == second.plan.canonical_json()
        assert (
            first.certificate.canonical_json()
            == second.certificate.canonical_json()
        )


def test_new_architectures_need_no_model_specific_compiler_pass() -> None:
    expected = (
        "exact_cache_contract",
        "structured_loop_invariance",
        "static_arena_reuse",
    )
    for builder in (*ROBOT_MATRIX_FIXTURES, *DRIVING_FIXTURES):
        result = compile_module(builder().module)
        assert tuple(item.name for item in result.certificate.passes) == expected
