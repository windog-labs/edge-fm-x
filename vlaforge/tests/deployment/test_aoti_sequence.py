from __future__ import annotations

from dataclasses import replace

import pytest

from vlaforge.deployment import (
    AotiSequenceArtifact,
    AotiSequenceManifest,
    AotiSequenceNode,
    AotiSequenceValue,
)


def _manifest() -> AotiSequenceManifest:
    return AotiSequenceManifest(
        region_name="composed_region",
        target="sm_86",
        device="cuda:0",
        values=(
            AotiSequenceValue(0, "input", "input", 0, "f32", (4, 4)),
            AotiSequenceValue(
                1, "hidden", "temporary", None, "f32", (4, 4)
            ),
            AotiSequenceValue(2, "output", "output", 0, "f32", (4, 4)),
        ),
        artifacts=(
            AotiSequenceArtifact(
                0, "first", "physical/first.so", "1" * 64, 101
            ),
            AotiSequenceArtifact(
                1, "second", "physical/second.so", "2" * 64, 202
            ),
        ),
        nodes=(
            AotiSequenceNode(0, (0,), (1,)),
            AotiSequenceNode(1, (1,), (2,)),
        ),
    )


def test_aoti_sequence_roundtrips_canonical_runtime_text() -> None:
    manifest = _manifest()
    restored = AotiSequenceManifest.from_text(manifest.canonical_text())

    assert restored.region_name == manifest.region_name
    assert restored.target == manifest.target
    assert restored.device == manifest.device
    assert restored.input_count == 1
    assert restored.output_count == 1
    assert [item.to_dict() for item in restored.values] == [
        {
            **item.to_dict(),
            "name": f"value_{item.value_id}",
        }
        for item in manifest.values
    ]
    assert [item.path for item in restored.artifacts] == [
        item.path for item in manifest.artifacts
    ]
    assert restored.nodes == manifest.nodes
    assert restored.canonical_text() == manifest.canonical_text()


def test_aoti_sequence_rejects_undefined_dataflow_input() -> None:
    manifest = _manifest()
    with pytest.raises(ValueError, match="reads undefined"):
        replace(
            manifest,
            nodes=(
                AotiSequenceNode(0, (1,), (2,)),
                AotiSequenceNode(1, (0,), (1,)),
            ),
        )


def test_aoti_sequence_rejects_redefined_value() -> None:
    manifest = _manifest()
    with pytest.raises(ValueError, match="redefines"):
        replace(
            manifest,
            nodes=(
                AotiSequenceNode(0, (0,), (1,)),
                AotiSequenceNode(1, (1,), (1,)),
            ),
        )


def test_aoti_sequence_rejects_non_dense_external_bindings() -> None:
    manifest = _manifest()
    values = list(manifest.values)
    values[0] = replace(values[0], binding=2)
    with pytest.raises(ValueError, match="input bindings"):
        replace(manifest, values=tuple(values))


def test_aoti_sequence_rejects_artifact_path_escape() -> None:
    with pytest.raises(ValueError, match="normalized and relative"):
        AotiSequenceArtifact(0, "bad", "../bad.so", "3" * 64, 1)


def test_aoti_sequence_parser_rejects_trailing_tokens() -> None:
    with pytest.raises(ValueError, match="trailing tokens"):
        AotiSequenceManifest.from_text(_manifest().canonical_text() + "bad\n")
