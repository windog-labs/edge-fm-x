"""Deterministic no-Python C++ generation for Invocation IR v0.2."""

from vlaforge.codegen.cpp import (
    CodegenUnsupportedError,
    generate_compiled_cpp_session,
    generate_cpp_session,
)
from vlaforge.codegen.fixtures import (
    driving_diffusion_regions,
    driving_diffusion_runner_source,
    driving_diffusion_validators,
    hybrid_external_feature_regions,
    hybrid_external_feature_runner_source,
    hybrid_external_feature_validators,
    openvla_fixture_regions,
    openvla_fixture_runner_source,
    openvla_fixture_validators,
    smolvla_fixture_regions,
    smolvla_fixture_runner_source,
    smolvla_fixture_validators,
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
    "generate_compiled_cpp_session",
    "generate_cpp_session",
    "driving_diffusion_regions",
    "driving_diffusion_runner_source",
    "driving_diffusion_validators",
    "hybrid_external_feature_regions",
    "hybrid_external_feature_runner_source",
    "hybrid_external_feature_validators",
    "openvla_fixture_regions",
    "openvla_fixture_runner_source",
    "openvla_fixture_validators",
    "smolvla_fixture_regions",
    "smolvla_fixture_runner_source",
    "smolvla_fixture_validators",
]
