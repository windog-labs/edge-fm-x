"""Pinned upstream contracts used by the adaptation evidence matrix.

The registry is descriptive L0 evidence. A record does not imply that its
checkpoint has been captured or compiled; executable evidence is tracked
separately by each Model Adaptation Card.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UpstreamModelContract:
    name: str
    family: str
    repository: str
    revision: str
    license: str
    checkpoint: str
    source_entries: tuple[str, ...]
    fixture_factory: str | None
    current_evidence: str
    unsupported: tuple[str, ...] = ()


MODEL_CONTRACTS = (
    UpstreamModelContract(
        "RT-1",
        "robot/discrete-action",
        "https://github.com/google-research/robotics_transformer",
        "4569641b8111f3f402c32d8e24becd2a6e952ecc",
        "Apache-2.0",
        "not pinned; fixture only",
        ("robotics_transformer/model.py",),
        "build_rt1_like_fixture",
        "L0+L1",
        ("real checkpoint capture", "compiled artifact", "C++ parity"),
    ),
    UpstreamModelContract(
        "ACT",
        "robot/action-chunk",
        "https://github.com/tonyzhaozh/act",
        "742c753c0d4a5d87076c8f69e5628c79a8cc5488",
        "MIT",
        "not pinned; fixture only",
        ("policy.py", "detr/models/detr_vae.py"),
        "build_act_like_fixture",
        "L0+L1",
        ("real checkpoint capture", "compiled artifact", "C++ parity"),
    ),
    UpstreamModelContract(
        "Octo",
        "robot/generalist-diffusion",
        "https://github.com/octo-models/octo",
        "241fb3514b7c40957a86d869fecb7c7fc353f540",
        "MIT",
        "rail-berkeley/octo-base-1.5 (hash not downloaded)",
        (
            "octo/model/octo_model.py:OctoModel.sample_actions",
            "octo/model/components/action_heads.py:DiffusionActionHead",
        ),
        "build_octo_like_fixture",
        "L0+L1",
        ("JAX frontend capture", "checkpoint parity", "C++ parity"),
    ),
    UpstreamModelContract(
        "OpenVLA",
        "robot/autoregressive-vla",
        "https://github.com/openvla/openvla",
        "c8f03f48af692657d3060c19588038c7220e9af9",
        "MIT",
        "openvla/openvla-7b (runtime-configured; hash not bundled)",
        (
            "prismatic/extern/hf/modeling_prismatic.py",
            "experiments/robot/openvla_utils.py",
        ),
        "build_openvla_fixture",
        "L0+L1+L2+fixture-L4",
        ("real artifact parity", "real no-Python C++ parity"),
    ),
    UpstreamModelContract(
        "PI0",
        "robot/flow-matching-vla",
        "https://github.com/huggingface/lerobot",
        "0d383d09f2051444de211739196a28cc94736861",
        "Apache-2.0",
        "not pinned; structure fixture only",
        (
            "src/lerobot/policies/pi0/modeling_pi0.py",
            "src/lerobot/policies/pi0/configuration_pi0.py",
        ),
        "build_pi0_fixture",
        "L0+L1",
        ("real checkpoint capture", "compiled artifact", "C++ parity"),
    ),
    UpstreamModelContract(
        "SmolVLA",
        "robot/flow-matching-vla",
        "https://github.com/huggingface/lerobot",
        "0d383d09f2051444de211739196a28cc94736861",
        "Apache-2.0",
        "lerobot/smolvla_base (runtime-configured; hash not bundled)",
        (
            "src/lerobot/policies/smolvla/modeling_smolvla.py",
            "src/lerobot/policies/smolvla/configuration_smolvla.py",
        ),
        "build_smolvla_fixture",
        "L0+L1+L2+fixture-L4",
        ("real artifact parity", "real no-Python C++ parity"),
    ),
    UpstreamModelContract(
        "GR00T N1.7",
        "robot/multi-embodiment-dit",
        "https://github.com/NVIDIA/Isaac-GR00T",
        "9c7e746b2cd37a810070a98ef41d290a07e806c2",
        "Apache-2.0 code; NVIDIA Open Model License weights",
        "NVIDIA/GR00T-N1.7-3B (hash not downloaded)",
        (
            "gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
            "gr00t/model/modules/dit.py",
            "gr00t/policy/gr00t_policy.py",
        ),
        "build_groot_n1_like_fixture",
        "L0+L1",
        ("real checkpoint capture", "TensorRT artifact", "C++ parity"),
    ),
    UpstreamModelContract(
        "DiffusionDrive",
        "driving/truncated-diffusion",
        "https://github.com/hustvl/DiffusionDrive",
        "9b52ed0ec06b073d82d6f392ab084c7b301c8681",
        "MIT",
        "release checkpoint not downloaded",
        (
            "navsim/agents/diffusiondrive/transfuser_agent.py",
            "navsim/agents/diffusiondrive/transfuser_model_v2.py:"
            "TrajectoryHead.forward_test",
        ),
        "build_driving_diffusion_fixture",
        "L0+L1+fixture-L4",
        (
            "real checkpoint capture",
            "real artifact parity",
            "real no-Python C++ parity",
        ),
    ),
    UpstreamModelContract(
        "AutoVLA",
        "driving/autoregressive-vla",
        "https://github.com/ucla-mobility/AutoVLA",
        "ba34eed74ce6729e7986592d0e66cbaca397b4fa",
        "UCLA Academic Software License",
        "release checkpoint not downloaded",
        (
            "models/autovla.py:AutoVLA.predict",
            "models/action_tokenizer.py:"
            "ActionTokenizer.decode_token_ids_to_trajectory",
        ),
        "build_driving_ar_fixture",
        "L0+L1",
        ("real checkpoint capture", "compiled artifact", "C++ parity"),
    ),
    UpstreamModelContract(
        "ReCogDrive",
        "driving/vlm-plus-diffusion",
        "https://github.com/xiaomi-research/recogdrive",
        "d54404796de7a44ca418b96057e3f8c3de3e8c0d",
        "Apache-2.0",
        "release checkpoint not downloaded",
        (
            "navsim/agents/recogdrive/recogdrive_agent.py",
            "navsim/agents/recogdrive/recogdrive_diffusion_planner.py",
            "navsim/agents/recogdrive/recogdrive_dit.py",
        ),
        "build_hybrid_external_feature_fixture",
        "L0+L1-structural",
        ("real checkpoint capture", "cross-artifact parity", "C++ parity"),
    ),
    UpstreamModelContract(
        "UniDriveVLA",
        "driving/multi-expert-vla",
        "https://github.com/xiaomi-research/unidrivevla",
        "a93c175af893b35dc16618e659eca4d18bb1ec86",
        "repository license needs legal verification",
        "release checkpoint not downloaded",
        (
            "Bench2Drive/projects/mmdet3d_plugin/models/detectors/"
            "unidrivevla.py",
            "Bench2Drive/projects/mmdet3d_plugin/models/vla/"
            "unidrivevla_vlm_qwenvl3.py",
            "Bench2Drive/projects/mmdet3d_plugin/models/vla/"
            "unified_perception_decoder.py",
        ),
        "build_hybrid_external_feature_fixture",
        "L0+L1-structural",
        ("license confirmation", "real capture", "artifact/C++ parity"),
    ),
    UpstreamModelContract(
        "OpenDriveVLA",
        "driving/multi-task-vla",
        "https://github.com/DriveVLA/OpenDriveVLA",
        "10e8095bc618d508cb70cca37b6956ac4db6e9f3",
        "Apache-2.0",
        "OpenDriveVLA/OpenDriveVLA-0.5B (gated; hash not downloaded)",
        ("opendrivevla/modeling_opendrivevla.py",),
        "build_hybrid_external_feature_fixture",
        "L0+L1-structural",
        ("gated checkpoint access", "real capture", "artifact/C++ parity"),
    ),
)


def model_contract(name: str) -> UpstreamModelContract:
    for contract in MODEL_CONTRACTS:
        if contract.name == name:
            return contract
    raise KeyError(name)
