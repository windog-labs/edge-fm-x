"""Deterministic static C++ generation for verified physical Plans."""

from vlaforge.codegen.cpp import (
    CodegenUnsupportedError,
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

__all__ = [
    "CodegenUnsupportedError",
    "CppRegionDefinition",
    "CppValidatorDefinition",
    "GeneratedSources",
    "generate_cpp_session",
    "openvla_fixture_regions",
    "openvla_fixture_runner_source",
    "openvla_fixture_validators",
]
