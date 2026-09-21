"""Native runner rendering contracts, not actual model/runtime validation."""

import ast
import inspect
from types import SimpleNamespace as Spec

import pytest

from vlaforge.adapters.openpi.openpi_session import render_runner


TEMPLATE = "@INPUTS@ @ORDINAL@ @SAMPLES@ @OUTPUT_COUNT@ @REPLAY_AUDIT@ @REPLAY_FAILURE@"


def module(dtype="f64", device="cuda:0"):
    return Spec(
        inputs=[
            Spec(name="state", device=device, payload=Spec(shape=(1, 32), dtype=dtype))
        ],
        outputs=[Spec(device=device, payload=Spec(shape=(1, 50, 32), dtype="f32"))],
    )


def test_official_float64_state_is_not_downcast_in_native_runner():
    rendered = render_runner(module(), TEMPLATE, device="cuda:0")
    assert '"state", VLAFORGE_DTYPE_F64, {1,32}, 256u' in rendered
    assert "1600" in rendered
    assert "@" not in rendered


def test_unknown_template_token_fails_closed():
    with pytest.raises(ValueError, match="substitution"):
        render_runner(module(), TEMPLATE + " @UNKNOWN@", device="cuda:0")


def test_missing_template_token_fails_closed():
    with pytest.raises(ValueError, match="substitution"):
        render_runner(module(), TEMPLATE.replace("@INPUTS@", ""), device="cuda:0")


def test_cpu_input_is_not_silently_moved_to_cuda():
    with pytest.raises(ValueError, match="CUDA"):
        render_runner(module(device="cpu"), TEMPLATE, device="cuda:0")


def test_unknown_dtype_is_rejected():
    with pytest.raises(ValueError, match="CUDA"):
        render_runner(module(dtype="invented"), TEMPLATE, device="cuda:0")


def test_native_contract_call_selects_explicit_numerical_schema():
    """Static wiring regression; real native construction is a separate run."""
    from vlaforge.adapters.openpi import openpi_session
    from vlaforge.deployment.contract import ARTIFACT_SCHEMA, NUMERICAL_ARTIFACT_SCHEMA

    calls = [
        node for node in ast.walk(ast.parse(inspect.getsource(openpi_session.main)))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RegionArtifactContract"
    ]
    assert len(calls) == 1
    schema = next(item.value for item in calls[0].keywords if item.arg == "schema")
    expression = compile(ast.Expression(schema), "<schema-wiring>", "eval")
    namespace = {
        "ARTIFACT_SCHEMA": ARTIFACT_SCHEMA,
        "NUMERICAL_ARTIFACT_SCHEMA": NUMERICAL_ARTIFACT_SCHEMA,
    }
    assert eval(expression, namespace, {"binding": None}) == ARTIFACT_SCHEMA
    assert eval(expression, namespace, {"binding": object()}) == NUMERICAL_ARTIFACT_SCHEMA


def test_numerical_session_evidence_captures_actual_maps_after_each_run():
    rendered = render_runner(
        module(), TEMPLATE, device="cuda:0", native_policy_evidence=True
    )
    assert '"/proc/self/maps"' in rendered
    assert 'current.find("libpython")' in rendered
    assert '"/run-" + std::to_string(run) + ".maps"' in rendered
