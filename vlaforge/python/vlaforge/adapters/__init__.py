"""Model adapters are isolated from the model-independent core IR."""

from vlaforge.adapters.common import AdapterFixture, FixtureTick
from vlaforge.adapters.openvla import build_openvla_fixture
from vlaforge.adapters.smolvla import build_smolvla_fixture

__all__ = [
    "AdapterFixture",
    "FixtureTick",
    "build_openvla_fixture",
    "build_smolvla_fixture",
]

