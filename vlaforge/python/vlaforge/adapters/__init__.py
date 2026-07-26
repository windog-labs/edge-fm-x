"""Model adapters isolated from the model-independent Invocation IR."""

from vlaforge.adapters.common import AdapterFixture, FixtureRun
from vlaforge.adapters.autovla_real import build_real_autovla_program
from vlaforge.adapters.driving import (
    DRIVING_FIXTURES,
    build_driving_ar_fixture,
    build_driving_diffusion_fixture,
    build_driving_trajectory_fixture,
    build_hybrid_external_feature_fixture,
)
from vlaforge.adapters.diffusiondrive_real import (
    build_real_diffusiondrive_program,
)
from vlaforge.adapters.openvla import build_openvla_fixture
from vlaforge.adapters.model_contracts import (
    MODEL_CONTRACTS,
    UpstreamModelContract,
    model_contract,
)
from vlaforge.adapters.minddrive_real import build_real_minddrive_program
from vlaforge.adapters.openvla_real import (
    build_real_openvla_action_program,
)
from vlaforge.adapters.pi0 import build_pi0_fixture
from vlaforge.adapters.robot_matrix import (
    ROBOT_MATRIX_FIXTURES,
    build_act_like_fixture,
    build_groot_n1_like_fixture,
    build_octo_like_fixture,
    build_rt1_like_fixture,
)
from vlaforge.adapters.smolvla import build_smolvla_fixture
from vlaforge.adapters.smolvla_real import (
    build_real_smolvla_action_program,
)
from vlaforge.adapters.transactional_fallback import (
    build_transactional_fallback_fixture,
)

__all__ = [
    "DRIVING_FIXTURES",
    "ROBOT_MATRIX_FIXTURES",
    "AdapterFixture",
    "FixtureRun",
    "MODEL_CONTRACTS",
    "UpstreamModelContract",
    "build_act_like_fixture",
    "build_driving_ar_fixture",
    "build_driving_diffusion_fixture",
    "build_driving_trajectory_fixture",
    "build_real_autovla_program",
    "build_real_diffusiondrive_program",
    "build_groot_n1_like_fixture",
    "build_hybrid_external_feature_fixture",
    "build_real_minddrive_program",
    "build_octo_like_fixture",
    "build_openvla_fixture",
    "build_pi0_fixture",
    "build_real_openvla_action_program",
    "build_real_smolvla_action_program",
    "build_rt1_like_fixture",
    "build_smolvla_fixture",
    "build_transactional_fallback_fixture",
    "model_contract",
]
