from dataclasses import replace

import pytest

from vlaforge.adapters import (
    build_driving_diffusion_fixture,
    build_openvla_fixture,
    build_smolvla_fixture,
)
from vlaforge.interpreter import Interpreter
from vlaforge.ir.attrs import Effect
from vlaforge.transforms import (
    ExactCacheContractError,
    analyze_structured_loop_invariance,
    canonicalize,
    configure_exact_cache,
)


def _outputs(fixture, module):
    runtime = Interpreter(
        module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    return tuple(
        tuple(
            (output.output, output.value)
            for output in runtime.run(
                inputs=item.inputs
            ).committed_outputs.outputs
        )
        for item in fixture.runs
    )


def test_canonicalize_preserves_io_ids_and_reference_outputs() -> None:
    fixture = build_smolvla_fixture()
    transformed = canonicalize(fixture.module)
    assert tuple(port.input_id for port in transformed.inputs) == (0,)
    assert tuple(port.output_id for port in transformed.outputs) == (0,)
    assert _outputs(fixture, transformed) == _outputs(fixture, fixture.module)


def test_exact_cache_configuration_preserves_outputs() -> None:
    fixture = build_driving_diffusion_fixture()
    configured = configure_exact_cache(fixture.module, enabled=True)
    disabled = configure_exact_cache(fixture.module, enabled=False)
    assert any(region.metadata.get("memoize") for region in configured.regions)
    assert not any(region.metadata.get("memoize") for region in disabled.regions)
    assert _outputs(fixture, configured) == _outputs(fixture, disabled)


def test_exact_cache_rejects_non_pure_region() -> None:
    fixture = build_smolvla_fixture()
    regions = tuple(
        replace(region, effects=(Effect.RANDOM,))
        if region.name == "encode_observation"
        else region
        for region in fixture.module.regions
    )
    with pytest.raises(ExactCacheContractError, match="pure"):
        configure_exact_cache(
            replace(fixture.module, regions=regions),
            enabled=True,
        )


def test_exact_cache_rejects_guarded_approximate_contract() -> None:
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
        configure_exact_cache(
            replace(fixture.module, regions=regions),
            enabled=True,
        )


def test_loop_analysis_distinguishes_preheader_and_loop_carried() -> None:
    fixture = build_openvla_fixture()
    analysis = analyze_structured_loop_invariance(fixture.module)
    dispositions = {
        (item.region, item.disposition) for item in analysis.decisions
    }
    assert ("encode_context", "prehoisted") in dispositions
    assert ("next_action_token", "loop_carried") in dispositions


def test_transform_results_are_deterministic() -> None:
    fixture = build_openvla_fixture()
    first = canonicalize(configure_exact_cache(fixture.module, enabled=True))
    second = canonicalize(configure_exact_cache(fixture.module, enabled=True))
    assert first == second
    assert (
        analyze_structured_loop_invariance(first)
        == analyze_structured_loop_invariance(second)
    )
