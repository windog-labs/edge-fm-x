"""Deterministic static C++ generation for verified physical Plans."""

from vlaforge.codegen.cpp import (
    CodegenUnsupportedError,
    generate_compiled_cpp_session,
    generate_cpp_session,
)
from vlaforge.codegen.fixtures import (
    openvla_fixture_regions,
    openvla_fixture_runner_source,
    openvla_fixture_validators,
)
from vlaforge.codegen.model import (
    CppRegionDefinition,
    CppValidatorDefinition,
    GeneratedSources,
)
from vlaforge.codegen.real_aoti import (
    AotiTensorSpec,
    SmolVLAAotiSpec,
    generate_real_smolvla_aoti_runner,
    smolvla_spec_from_exported_programs,
)
from vlaforge.codegen.real_torchscript import (
    OpenVLATorchScriptSpec,
    generate_real_openvla_torchscript_runner,
    openvla_spec_from_capture_reports,
)

__all__ = [
    "CodegenUnsupportedError",
    "AotiTensorSpec",
    "CppRegionDefinition",
    "CppValidatorDefinition",
    "GeneratedSources",
    "OpenVLATorchScriptSpec",
    "SmolVLAAotiSpec",
    "generate_cpp_session",
    "generate_compiled_cpp_session",
    "generate_real_smolvla_aoti_runner",
    "generate_real_openvla_torchscript_runner",
    "openvla_fixture_regions",
    "openvla_fixture_runner_source",
    "openvla_fixture_validators",
    "smolvla_spec_from_exported_programs",
    "openvla_spec_from_capture_reports",
]
