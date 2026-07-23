from dataclasses import replace

import pytest

from vlaforge.adapters import build_openvla_fixture, build_smolvla_fixture
from vlaforge.ir.parser import ParseError, parse_module
from vlaforge.ir.printer import print_module
from vlaforge.ir.serializer import canonical_json, module_digest
from vlaforge.ir.types import (
    ActionType,
    EpochType,
    FutureType,
    PendingType,
    ScalarType,
    SnapshotType,
    TensorType,
    type_from_dict,
)


@pytest.mark.parametrize(
    "ir_type",
    [
        ScalarType("i64"),
        TensorType((1, None, 7), "f16", "channels_last"),
        EpochType("control"),
        SnapshotType("cache", TensorType((2,), "f32")),
        PendingType("cache", TensorType((2,), "f32")),
        ActionType(TensorType((16, 7), "f32")),
        FutureType(ScalarType("bool")),
    ],
)
def test_type_round_trip(ir_type):
    assert type_from_dict(ir_type.to_dict()) == ir_type


@pytest.mark.parametrize(
    "factory", [build_smolvla_fixture, build_openvla_fixture]
)
def test_textual_ir_round_trip_is_byte_stable(factory):
    module = factory().module
    text = print_module(module)
    rebuilt = parse_module(text)
    assert rebuilt == module
    assert print_module(rebuilt) == text
    assert canonical_json(rebuilt) == canonical_json(module)
    assert module_digest(rebuilt) == module_digest(module)


def test_parser_rejects_missing_header():
    with pytest.raises(ParseError, match="expected header"):
        parse_module("{}\n")


def test_parser_rejects_header_payload_version_mismatch():
    module = build_smolvla_fixture().module
    text = print_module(module).replace("!vlaforge.ir 0.1", "!vlaforge.ir 9.9", 1)
    with pytest.raises(ValueError, match="unsupported"):
        parse_module(text)


def test_duplicate_declaration_names_are_rejected():
    module = build_smolvla_fixture().module
    with pytest.raises(ValueError, match="duplicate clock"):
        replace(module, clocks=module.clocks + (module.clocks[0],))

