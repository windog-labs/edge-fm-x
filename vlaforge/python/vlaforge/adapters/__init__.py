"""Model adapters are isolated from the model-independent core IR."""

from vlaforge.adapters.common import AdapterFixture, FixtureTick
from vlaforge.adapters.openvla import build_openvla_fixture
from vlaforge.adapters.openvla_frontend import (
    OpenVLAFrontendConfig,
    audit_real_openvla_frontend,
)
from vlaforge.adapters.openvla_real import (
    RealOpenVLAConfig,
    RealOpenVLAEvidence,
    build_real_openvla_action_program,
    run_real_openvla,
)
from vlaforge.adapters.smolvla import build_smolvla_fixture
from vlaforge.adapters.smolvla_frontend import audit_real_smolvla_frontend
from vlaforge.adapters.smolvla_real import (
    RealSmolVLAConfig,
    RealSmolVLAEvidence,
    build_real_smolvla_action_program,
    run_real_smolvla,
)

__all__ = [
    "AdapterFixture",
    "FixtureTick",
    "RealOpenVLAConfig",
    "RealOpenVLAEvidence",
    "OpenVLAFrontendConfig",
    "RealSmolVLAConfig",
    "RealSmolVLAEvidence",
    "build_real_openvla_action_program",
    "audit_real_openvla_frontend",
    "build_real_smolvla_action_program",
    "build_openvla_fixture",
    "build_smolvla_fixture",
    "audit_real_smolvla_frontend",
    "run_real_openvla",
    "run_real_smolvla",
]
