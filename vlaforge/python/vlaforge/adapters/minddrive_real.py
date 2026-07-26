"""MindDrive 0.5B stateful-invocation deployment contract.

The adapter keeps the VLAForge core model-independent.  Six-camera
preprocessing and episode identity are external contracts; the Semantic IR
contains only tensors, exact input identity, authoritative temporal state,
pure TensorRegions, and one transactional group of named driving outputs.

Model loading and torch.export capture stay lazy so the base ``vlaforge``
package does not depend on MMCV, flash-attn, or the upstream repository.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vlaforge.frontend.builder import ModuleBuilder
from vlaforge.ir import ops
from vlaforge.ir.program import (
    Block,
    InputPort,
    Invocation,
    OutputPort,
    StateSlot,
    TensorRegion,
    Value,
)
from vlaforge.ir.types import PendingOutputType, ScalarType, TensorType


MINDDRIVE_UPSTREAM_REVISION = (
    "1a4085dab1c20895a0c8d2b67b4f8e65712fa8de"
)
MINDDRIVE_CHECKPOINT_REVISION = (
    "5cf1eafc7f6d1028006f2d97d083d8e9aa4c0b12"
)
MINDDRIVE_CHECKPOINT_SHA256 = (
    "39c86eddeaf57b15b9aeb54beb9f698539f6ee1e83529724b1bfc8c5e11b4ba0"
)
MINDDRIVE_CHECKPOINT_SIZE = 6_593_355_869
MINDDRIVE_VLM_SHA256 = (
    "6fc9882475867279ee66e505ded47b5d722fc09b0d34bc7684a26080d662825f"
)
MINDDRIVE_VLM_SIZE = 1_892_090_688

# These thresholds are locked before evaluating the second archived frame.
# Frame 00400 is the backend-calibration sample and frame 00401 is the
# held-out validation sample.  The comparison is deliberately separate from
# eager/export parity: torch.export must remain exact to its SDPA eager module.
MINDDRIVE_VISION_BACKEND_MAX_ABS = 0.5
MINDDRIVE_VISION_BACKEND_NRMSE = 1.0e-3

MINDDRIVE_INPUT_TYPES = (
    ("camera_images", TensorType((1, 6, 3, 640, 640), "f32")),
    ("decision_input_ids", TensorType((53,), "i64")),
    ("planning_input_ids", TensorType((71,), "i64")),
    ("ego_route_command", TensorType((1, 1, 1, 6), "f32")),
    ("trajectory_noise", TensorType((1, 1, 32), "f32")),
    ("path_noise", TensorType((1, 1, 32), "f32")),
    ("can_bus", TensorType((1, 18), "f32")),
    ("lidar2img", TensorType((1, 6, 4, 4), "f32")),
    ("camera_intrinsics", TensorType((1, 6, 4, 4), "f32")),
    ("timestamp", TensorType((1,), "f64")),
    ("ego_pose", TensorType((1, 4, 4), "f32")),
    ("ego_pose_inverse", TensorType((1, 4, 4), "f32")),
    ("route_command_index", TensorType((1,), "f32")),
)

MINDDRIVE_IMAGE_FEATURES = TensorType((1, 6, 1024, 40, 40), "f32")
MINDDRIVE_VISION_TOKENS = TensorType((1, 529, 896), "f32")
MINDDRIVE_DECISION_PROMPT_LENGTH = 53
MINDDRIVE_ACTION_PROMPT_LENGTH = 71
MINDDRIVE_IMAGE_TOKEN_INDEX = 27
MINDDRIVE_DECISION_SEQUENCE_LENGTH = 581
MINDDRIVE_ACTION_SEQUENCE_LENGTH = 599
MINDDRIVE_ACTION_HIDDEN_POSITIONS = (596, 598)
MINDDRIVE_DECISION_LOGITS = TensorType((1, 7), "f32")
MINDDRIVE_ACTION_HIDDEN = TensorType((2, 896), "f32")
MINDDRIVE_DECISION_DCE_MAX_ABS = 2.0e-5
MINDDRIVE_DECISION_DCE_NRMSE = 1.0e-6
# Locked from frame 00400 before executing the exported Region on frame 00401.
# The small non-zero allowance covers only the source wrapper's equivalent
# GRU/output re-association; strict-export eager parity remains exact.
MINDDRIVE_TRAJECTORY_DECODER_MAX_ABS = 3.0e-6
MINDDRIVE_TRAJECTORY_DECODER_NRMSE = 1.0e-6

MINDDRIVE_OUTPUT_TYPES = (
    ("trajectory", TensorType((6, 2), "f32")),
    ("path_trajectory", TensorType((20, 2), "f32")),
    ("detection_scores", TensorType((300,), "f32")),
    ("detection_labels", TensorType((300,), "i64")),
    ("motion_trajectories", TensorType((300, 1, 12), "f32")),
    ("detection_boxes", TensorType((300, 9), "f32")),
    ("speed_command", ScalarType("i64")),
    ("path_command", ScalarType("i64")),
)

MINDDRIVE_STATE_TYPES = (
    (
        "detection_memory_embedding",
        TensorType((1, 600, 256), "f32"),
    ),
    (
        "detection_memory_reference_point",
        TensorType((1, 600, 3), "f32"),
    ),
    (
        "detection_memory_timestamp",
        TensorType((1, 600, 1), "f64"),
    ),
    (
        "detection_memory_egopose",
        TensorType((1, 600, 4, 4), "f32"),
    ),
    (
        "detection_memory_velocity",
        TensorType((1, 600, 2), "f32"),
    ),
    ("detection_sample_time", TensorType((1,), "f64")),
    (
        "detection_memory_canbus",
        TensorType((1, 2, 19), "f32"),
    ),
    (
        "detection_memory_canbus_length",
        TensorType((1,), "i64"),
    ),
    (
        "detection_memory_scene_query",
        TensorType((1, 256, 256), "f32"),
    ),
    (
        "detection_scene_memory_timestamp",
        TensorType((1, 256, 1), "f64"),
    ),
    ("map_memory_embedding", TensorType((1, 600, 256), "f32")),
    (
        "map_memory_reference_point",
        TensorType((1, 600, 11, 3), "f32"),
    ),
    ("map_memory_timestamp", TensorType((1, 600, 1), "f64")),
    (
        "map_memory_egopose",
        TensorType((1, 600, 4, 4), "f32"),
    ),
    ("map_sample_time", TensorType((1,), "f64")),
    ("map_memory_mask", TensorType((1, 600, 1), "f64")),
)

_INITIALIZED = ScalarType("bool")
_COMMON_PLANNER_INPUTS = (
    ("image_features", MINDDRIVE_IMAGE_FEATURES),
    *MINDDRIVE_INPUT_TYPES[1:],
)
_BRANCH_OUTPUT_TYPES = (
    *(payload for _, payload in MINDDRIVE_OUTPUT_TYPES),
    *(payload for _, payload in MINDDRIVE_STATE_TYPES),
    _INITIALIZED,
)
_BRANCH_VALUE_NAMES = (
    *(f"{name}_next" for name, _ in MINDDRIVE_OUTPUT_TYPES),
    *(f"{name}_next" for name, _ in MINDDRIVE_STATE_TYPES),
    "state_initialized_next",
)


@dataclass(frozen=True, slots=True)
class MindDriveVisionBackendEvidence:
    """FlashAttention-to-SDPA equivalence for one real camera frame."""

    maximum_absolute_error: float
    root_mean_square_error: float
    normalized_root_mean_square_error: float
    reference_absolute_maximum: float
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "vlaforge.minddrive_vision_backend_equivalence/1",
            "backend_substitution": "flash-attn-2-varlen-to-aten-sdpa",
            "maximum_absolute_error": self.maximum_absolute_error,
            "root_mean_square_error": self.root_mean_square_error,
            "normalized_root_mean_square_error": (
                self.normalized_root_mean_square_error
            ),
            "reference_absolute_maximum": self.reference_absolute_maximum,
            "thresholds": {
                "maximum_absolute_error": (
                    MINDDRIVE_VISION_BACKEND_MAX_ABS
                ),
                "normalized_root_mean_square_error": (
                    MINDDRIVE_VISION_BACKEND_NRMSE
                ),
            },
            "passed": self.passed,
        }


def build_real_minddrive_program(*, device: str = "cuda:0") -> Any:
    """Build the complete passive MindDrive invocation Semantic IR.

    ``state_initialized`` chooses the source-faithful first-frame Region after
    construction or ``ResetEpisode``.  Every later caller-driven ``Run`` uses
    the stateful Region.  Both branches return the same fixed tensor ABI, all
    authoritative state is staged, and state plus the complete named output
    group becomes visible only after validation succeeds.
    """

    builder = ModuleBuilder("minddrive_0_5b_real")
    for name, payload in MINDDRIVE_INPUT_TYPES:
        builder.add_input(
            InputPort(name, payload, device=device, alignment=64)
        )
    for name, payload in MINDDRIVE_OUTPUT_TYPES:
        builder.add_output(
            OutputPort(
                name,
                payload,
                group="driving",
                device=device,
                alignment=64,
            )
        )
    builder.add_state(StateSlot("state_initialized", _INITIALIZED))
    for name, payload in MINDDRIVE_STATE_TYPES:
        builder.add_state(StateSlot(name, payload))

    builder.add_region(
        TensorRegion(
            "vision_encoder",
            (
                Value(
                    "camera_images",
                    dict(MINDDRIVE_INPUT_TYPES)["camera_images"],
                ),
            ),
            (MINDDRIVE_IMAGE_FEATURES,),
            metadata={
                "memoize": True,
                "cache_input_ports": ["camera_images"],
                "cache_state_slots": [],
                "loop_invariant": True,
                "derived_cache": "exact_image_features",
            },
        )
    )
    builder.add_region(
        TensorRegion(
            "decision_expert",
            (
                Value(
                    "decision_input_ids",
                    dict(MINDDRIVE_INPUT_TYPES)["decision_input_ids"],
                ),
                Value("vision_tokens", MINDDRIVE_VISION_TOKENS),
            ),
            (MINDDRIVE_DECISION_LOGITS,),
            metadata={
                "source_component": "Qwen2-0.5B decision_expert LoRA",
                "bounded_prefill": MINDDRIVE_DECISION_SEQUENCE_LENGTH,
                "compiler_transform": "exact_vocabulary_projection_dce",
            },
        )
    )
    builder.add_region(
        TensorRegion(
            "action_expert",
            (
                Value(
                    "planning_input_ids",
                    dict(MINDDRIVE_INPUT_TYPES)["planning_input_ids"],
                ),
                Value("vision_tokens", MINDDRIVE_VISION_TOKENS),
            ),
            (MINDDRIVE_ACTION_HIDDEN,),
            metadata={
                "source_component": "Qwen2-0.5B action_expert LoRA",
                "bounded_prefill": MINDDRIVE_ACTION_SEQUENCE_LENGTH,
                "selected_hidden_positions": list(
                    MINDDRIVE_ACTION_HIDDEN_POSITIONS
                ),
            },
        )
    )
    builder.add_region(
        TensorRegion(
            "trajectory_decoder",
            (
                Value("action_hidden", MINDDRIVE_ACTION_HIDDEN),
                Value("decision_logits", MINDDRIVE_DECISION_LOGITS),
                Value(
                    "ego_route_command",
                    dict(MINDDRIVE_INPUT_TYPES)["ego_route_command"],
                ),
                Value(
                    "trajectory_noise",
                    dict(MINDDRIVE_INPUT_TYPES)["trajectory_noise"],
                ),
                Value(
                    "path_noise",
                    dict(MINDDRIVE_INPUT_TYPES)["path_noise"],
                ),
            ),
            (
                dict(MINDDRIVE_OUTPUT_TYPES)["trajectory"],
                dict(MINDDRIVE_OUTPUT_TYPES)["path_trajectory"],
                dict(MINDDRIVE_OUTPUT_TYPES)["speed_command"],
                dict(MINDDRIVE_OUTPUT_TYPES)["path_command"],
            ),
            metadata={
                "source_component": (
                    "probabilistic GRU trajectory and path heads"
                ),
                "rng_semantics": "explicit_tensor_inputs",
                "trajectory_steps": 6,
                "path_steps": 20,
            },
        )
    )
    common_arguments = tuple(
        Value(name, payload) for name, payload in _COMMON_PLANNER_INPUTS
    )
    builder.add_region(
        TensorRegion(
            "first_frame_planner",
            common_arguments,
            _BRANCH_OUTPUT_TYPES,
            metadata={
                "state_semantics": "reset_memory_then_run",
                "source_model": "xiaomi-mlab/MindDrive",
            },
        )
    )
    builder.add_region(
        TensorRegion(
            "stateful_planner",
            (
                *common_arguments,
                *(
                    Value(name, payload)
                    for name, payload in MINDDRIVE_STATE_TYPES
                ),
            ),
            _BRANCH_OUTPUT_TYPES,
            metadata={
                "state_semantics": "read_latest_then_run",
                "source_model": "xiaomi-mlab/MindDrive",
            },
        )
    )

    input_operations = []
    for name, payload in MINDDRIVE_INPUT_TYPES:
        input_operations.append(
            ops.input_read(
                f"{name}_value",
                f"{name}_revision",
                payload,
                name,
            )
        )

    common_operands = (
        "image_features_value",
        *(f"{name}_value" for name, _ in MINDDRIVE_INPUT_TYPES[1:]),
    )
    stateful_operands = (
        *common_operands,
        *(f"{name}_value" for name, _ in MINDDRIVE_STATE_TYPES),
    )
    branch_results = tuple(
        Value(name, payload)
        for name, payload in zip(
            _BRANCH_VALUE_NAMES,
            _BRANCH_OUTPUT_TYPES,
            strict=True,
        )
    )
    stateful_branch = Block.of(
        (
            ops.invoke(
                _BRANCH_VALUE_NAMES,
                _BRANCH_OUTPUT_TYPES,
                "stateful_planner",
                stateful_operands,
            ),
            ops.yield_values(*_BRANCH_VALUE_NAMES),
        )
    )
    first_frame_branch = Block.of(
        (
            ops.invoke(
                _BRANCH_VALUE_NAMES,
                _BRANCH_OUTPUT_TYPES,
                "first_frame_planner",
                common_operands,
            ),
            ops.yield_values(*_BRANCH_VALUE_NAMES),
        )
    )

    operations = [
        *input_operations,
        ops.transaction_begin("txn"),
        ops.invoke(
            ("image_features_value",),
            (MINDDRIVE_IMAGE_FEATURES,),
            "vision_encoder",
            ("camera_images_value",),
        ),
        ops.state_read_latest(
            "state_initialized_snapshot",
            _INITIALIZED,
            "state_initialized",
            "txn",
        ),
        ops.snapshot_value(
            "state_initialized_value",
            _INITIALIZED,
            "state_initialized_snapshot",
        ),
    ]
    for name, payload in MINDDRIVE_STATE_TYPES:
        operations.extend(
            (
                ops.state_read_latest(
                    f"{name}_snapshot",
                    payload,
                    name,
                    "txn",
                ),
                ops.snapshot_value(
                    f"{name}_value",
                    payload,
                    f"{name}_snapshot",
                ),
            )
        )
    operations.append(
        ops.if_op(
            branch_results,
            "state_initialized_value",
            stateful_branch,
            first_frame_branch,
        )
    )
    operations.append(
        ops.stage_write(
            "state_initialized_pending",
            _INITIALIZED,
            "state_initialized",
            "txn",
            "state_initialized_next",
        )
    )
    for name, payload in MINDDRIVE_STATE_TYPES:
        operations.append(
            ops.stage_write(
                f"{name}_pending",
                payload,
                name,
                "txn",
                f"{name}_next",
            )
        )
    operations.append(
        ops.validate(
            "driving_outputs_valid",
            "trajectory_next",
            "minddrive_output_contract",
        )
    )

    pending_types = tuple(
        PendingOutputType(name, payload)
        for name, payload in MINDDRIVE_OUTPUT_TYPES
    )
    for name, payload in MINDDRIVE_OUTPUT_TYPES:
        operations.append(
            ops.output_create(
                f"{name}_pending_output",
                f"{name}_next",
                payload,
                name,
            )
        )
    operations.extend(
        (
            ops.output_group(
                "driving_pending_outputs",
                "driving",
                tuple(
                    (f"{name}_pending_output", pending)
                    for (name, _), pending in zip(
                        MINDDRIVE_OUTPUT_TYPES,
                        pending_types,
                        strict=True,
                    )
                ),
            ),
            ops.transaction_commit(
                "driving_committed_outputs",
                pending_types,
                "driving",
                "txn",
                "driving_pending_outputs",
                "driving_outputs_valid",
            ),
            ops.return_values("driving_committed_outputs"),
        )
    )
    builder.add_invocation(
        Invocation(
            "run",
            Block.of(operations),
            metadata={
                "adapter_template": "MultiTaskDriving",
                "source_model": "xiaomi-mlab/MindDrive",
                "source_revision": MINDDRIVE_UPSTREAM_REVISION,
                "checkpoint_revision": MINDDRIVE_CHECKPOINT_REVISION,
                "core_op_delta": 0,
                "scheduling": "external_caller_driven",
                "episode_identity": "external_reset_episode",
                "authoritative_state_count": len(MINDDRIVE_STATE_TYPES),
                "derived_cache": "vision_encoder",
            },
        )
    )
    return builder.build()


def make_minddrive_torch_initial_state(
    torch: Any, *, device: str = "cuda:0"
) -> dict[str, object]:
    """Materialize fixed-shape empty authoritative state for a Session."""

    dtypes = {
        "bool": torch.bool,
        "f32": torch.float32,
        "f64": torch.float64,
        "i64": torch.int64,
    }
    state: dict[str, object] = {"state_initialized": False}
    for name, payload in MINDDRIVE_STATE_TYPES:
        state[name] = torch.zeros(
            payload.shape,
            dtype=dtypes[payload.dtype],
            device=device,
        )
    return state


def load_real_minddrive_model(
    source_root: str | Path,
    release_root: str | Path,
    *,
    device: str = "cuda:0",
    verify_hashes: bool = True,
) -> Any:
    """Load the pinned upstream model with a strict inference-state projection.

    The checkpoint also contains training-only value-network tensors and
    non-persistent rotary buffers.  Those keys are audited and excluded; every
    key in the inference model must still be present, and no other unexpected
    key is accepted.
    """

    import torch

    source = Path(source_root).resolve()
    release = Path(release_root).resolve()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != MINDDRIVE_UPSTREAM_REVISION:
        raise ValueError(
            f"MindDrive revision mismatch: {revision} != "
            f"{MINDDRIVE_UPSTREAM_REVISION}"
        )

    checkpoint = release / "minddrive_rltrain.pth"
    vlm_root = release / "llava-qwen2-0.5b"
    vlm_weights = vlm_root / "model.safetensors"
    _verify_release_file(
        checkpoint,
        expected_size=MINDDRIVE_CHECKPOINT_SIZE,
        expected_sha256=MINDDRIVE_CHECKPOINT_SHA256,
        verify_hash=verify_hashes,
    )
    _verify_release_file(
        vlm_weights,
        expected_size=MINDDRIVE_VLM_SIZE,
        expected_sha256=MINDDRIVE_VLM_SHA256,
        verify_hash=verify_hashes,
    )

    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from mmcv import Config
    from mmcv.models import build_model

    config_path = (
        source
        / "adzoo"
        / "minddrive"
        / "configs"
        / "minddrive_qwen2_05B_infer.py"
    )
    config = Config.fromfile(str(config_path))
    config.model.tokenizer = str(vlm_root)
    config.model.lm_head = str(vlm_root)
    model = build_model(
        config.model,
        train_cfg=config.get("train_cfg"),
        test_cfg=config.get("test_cfg"),
    )
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    state = payload["state_dict"]
    model_keys = set(model.state_dict())
    checkpoint_keys = set(state)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    disallowed = [
        key for key in unexpected if not _is_allowed_training_extra(key)
    ]
    if missing or disallowed:
        raise ValueError(
            "MindDrive checkpoint/model mismatch: "
            f"missing={len(missing)}, disallowed_unexpected={len(disallowed)}"
        )
    model.load_state_dict(
        {key: value for key, value in state.items() if key in model_keys},
        strict=True,
    )
    del state
    del payload
    return model.eval().to(device)


def make_exportable_minddrive_vision_encoder(model: Any) -> Any:
    """Create the real EVA vision Region with an exportable attention backend.

    The upstream evaluation GridMask is an identity but evaluates Python RNG
    before checking ``training``.  It is therefore omitted statically.  EVA's
    raw FlashAttention PyCapsule has no dispatcher schema and cannot be
    represented by torch.export, so each inner attention call is replaced by
    ATen scaled-dot-product attention with the same FP16 Q/K/V, scale,
    no-dropout, and non-causal contract.  All checkpoint parameters remain the
    original strict-loaded tensors.
    """

    import copy

    import torch
    import torch.nn.functional as functional

    class ExportableSDPA(torch.nn.Module):
        def __init__(self, *, scale: float | None) -> None:
            super().__init__()
            self.scale = scale

        def forward(
            self,
            query: Any,
            key_value: Any,
            causal: bool = False,
            key_padding_mask: Any = None,
        ) -> tuple[Any, None]:
            if key_padding_mask is not None:
                raise ValueError(
                    "MindDrive EVA deployment profile has no padding mask"
                )
            query_fp16 = query.to(torch.float16).permute(0, 2, 1, 3)
            key_fp16 = key_value[:, :, 0].to(torch.float16).permute(
                0, 2, 1, 3
            )
            value_fp16 = key_value[:, :, 1].to(torch.float16).permute(
                0, 2, 1, 3
            )
            output = functional.scaled_dot_product_attention(
                query_fp16,
                key_fp16,
                value_fp16,
                dropout_p=0.0,
                is_causal=causal,
                scale=self.scale,
            )
            return output.permute(0, 2, 1, 3).to(torch.float32), None

    # Upstream registers the same rotary-embedding module both on the backbone
    # and under many blocks.  PyTorch 2.4 strict export cannot restore FQNs for
    # that repeated alias and fails in its parameter/buffer mapper.  Give every
    # block its own identical constant buffers, then remove the unused parent
    # aliases.  This changes registration identity only, never tensor values or
    # computation.
    rotary_aliases_canonicalized = 0
    for block in model.img_backbone.blocks:
        rope = getattr(block.attn, "rope", None)
        if rope is not None:
            block.attn.rope = copy.deepcopy(rope)
            rotary_aliases_canonicalized += 1
    model.img_backbone.rope_win = None
    model.img_backbone.rope_glb = None

    replaced = 0
    for module in model.img_backbone.modules():
        inner = getattr(module, "inner_attn", None)
        if inner is None or not hasattr(module, "flash_attn"):
            continue
        module.inner_attn = ExportableSDPA(
            scale=getattr(inner, "softmax_scale", None)
        )
        replaced += 1
    if replaced == 0:
        raise ValueError("MindDrive EVA contains no FlashAttention modules")

    class ExportableEVABackbone(torch.nn.Module):
        """EVA forward without PyTorch 2.4's duplicate ``.data`` weight use."""

        def __init__(self, backbone: Any) -> None:
            super().__init__()
            if backbone.patch_embed.proj.weight.dtype != torch.float32:
                raise ValueError(
                    "pinned MindDrive FP32 EVA profile changed dtype"
                )
            self.patch_embed = backbone.patch_embed
            self.blocks = backbone.blocks
            self.pos_embed = backbone.pos_embed
            self.pretrain_use_cls_token = bool(
                backbone.pretrain_use_cls_token
            )
            positions = int(backbone.pos_embed.shape[1])
            if self.pretrain_use_cls_token:
                positions -= 1
            self.pretrained_position_size = int(positions**0.5)
            if self.pretrained_position_size**2 != positions:
                raise ValueError(
                    "MindDrive EVA absolute position grid is not square"
                )

        def forward(self, images: Any) -> Any:
            # The strict-loaded deployment profile is FP32.  Upstream reads
            # ``patch_embed.proj.weight.data.dtype`` before invoking the same
            # convolution.  PyTorch 2.4 consequently registers that parameter
            # twice in the traced GraphModule and underflows its FQN mapper.
            # A static dtype is numerically identical and removes the duplicate
            # parameter use without a torch.export monkeypatch.
            features = self.patch_embed(images.to(torch.float32))
            absolute_position = self.pos_embed
            if self.pretrain_use_cls_token:
                absolute_position = absolute_position[:, 1:]
            if (
                self.pretrained_position_size != features.shape[1]
                or self.pretrained_position_size != features.shape[2]
            ):
                absolute_position = functional.interpolate(
                    absolute_position.reshape(
                        1,
                        self.pretrained_position_size,
                        self.pretrained_position_size,
                        -1,
                    )
                    .permute(0, 3, 1, 2)
                    .float(),
                    size=(features.shape[1], features.shape[2]),
                    mode="bicubic",
                    align_corners=False,
                ).to(absolute_position.dtype)
                absolute_position = absolute_position.permute(0, 2, 3, 1)
            else:
                absolute_position = absolute_position.reshape(
                    1,
                    features.shape[1],
                    features.shape[2],
                    -1,
                )
            features = features + absolute_position.to(features.dtype)
            for block in self.blocks:
                features = block(features)
            return features.permute(0, 3, 1, 2)

    class VisionEncoder(torch.nn.Module):
        def __init__(self, detector: Any) -> None:
            super().__init__()
            self.backbone = ExportableEVABackbone(detector.img_backbone)
            self.neck = (
                detector.img_neck
                if getattr(detector, "with_img_neck", False)
                else None
            )
            self.position_level = int(detector.position_level)

        def forward(self, camera_images: Any) -> Any:
            batch = camera_images.shape[0]
            flattened = camera_images.reshape(
                -1,
                camera_images.shape[-3],
                camera_images.shape[-2],
                camera_images.shape[-1],
            )
            features = self.backbone(flattened)
            if self.neck is not None:
                features = self.neck([features])
                if isinstance(features, dict):
                    features = list(features.values())
                selected = features[self.position_level]
            else:
                if self.position_level != 0:
                    raise ValueError(
                        "MindDrive no-neck profile changed position level"
                    )
                selected = features
            return selected.reshape(
                batch,
                -1,
                selected.shape[1],
                selected.shape[2],
                selected.shape[3],
            )

    encoder = VisionEncoder(model).eval()
    encoder.replaced_flash_attention_modules = replaced
    encoder.canonicalized_rotary_aliases = rotary_aliases_canonicalized
    return encoder


def compare_minddrive_vision_backends(
    reference: Any,
    candidate: Any,
) -> MindDriveVisionBackendEvidence:
    """Apply the locked FlashAttention-to-SDPA numerical contract."""

    import torch

    reference_tensor = reference.detach().to(torch.float64)
    candidate_tensor = candidate.detach().to(torch.float64)
    if reference_tensor.shape != candidate_tensor.shape:
        raise ValueError(
            "MindDrive vision backend shape mismatch: "
            f"{tuple(reference_tensor.shape)} != "
            f"{tuple(candidate_tensor.shape)}"
        )
    difference = candidate_tensor - reference_tensor
    maximum_absolute_error = float(difference.abs().max().item())
    root_mean_square_error = float(
        torch.sqrt(torch.mean(difference.square())).item()
    )
    reference_absolute_maximum = float(reference_tensor.abs().max().item())
    denominator = max(reference_absolute_maximum, 1.0e-12)
    normalized_root_mean_square_error = (
        root_mean_square_error / denominator
    )
    passed = (
        maximum_absolute_error <= MINDDRIVE_VISION_BACKEND_MAX_ABS
        and normalized_root_mean_square_error
        <= MINDDRIVE_VISION_BACKEND_NRMSE
    )
    return MindDriveVisionBackendEvidence(
        maximum_absolute_error=maximum_absolute_error,
        root_mean_square_error=root_mean_square_error,
        normalized_root_mean_square_error=(
            normalized_root_mean_square_error
        ),
        reference_absolute_maximum=reference_absolute_maximum,
        passed=passed,
    )


def make_minddrive_decision_expert(model: Any) -> Any:
    """Build the real Qwen2 decision-expert prefill Region.

    The upstream method projects the penultimate hidden state to the complete
    151k-token vocabulary and immediately gathers seven meta-action rows.
    Keeping only those exact rows is a semantics-preserving compiler DCE, not
    a reduced or synthetic model.
    """

    import torch
    import torch.nn.functional as functional

    model.lm_head.set_adapter("decision_expert")
    implementation = (
        model.lm_head.inference_action_distribution.__self__
    )
    _make_minddrive_qwen_rotary_exportable(implementation.model)
    meta_action_ids = tuple(
        int(item) for item in implementation.config.meta_action_token_idx[:7]
    )

    class DecisionExpert(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = implementation.model
            selected_rows = implementation.lm_head.weight[
                list(meta_action_ids)
            ].detach().clone()
            self.register_buffer("meta_action_rows", selected_rows)

        def forward(
            self, input_ids: Any, vision_tokens: Any
        ) -> Any:
            prefix_ids = input_ids[:MINDDRIVE_IMAGE_TOKEN_INDEX]
            suffix_ids = input_ids[MINDDRIVE_IMAGE_TOKEN_INDEX + 1 :]
            prefix = self.language_model.embed_tokens(prefix_ids)
            suffix = self.language_model.embed_tokens(suffix_ids)
            inputs_embeds = torch.cat(
                (prefix, vision_tokens[0], suffix), dim=0
            ).unsqueeze(0)
            hidden = self.language_model(
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
            )[0]
            selected_logits = functional.linear(
                hidden[:, -2, :],
                self.meta_action_rows,
            )
            return functional.log_softmax(selected_logits, dim=-1)

    return DecisionExpert().eval()


def make_minddrive_action_expert(model: Any) -> Any:
    """Build the real Qwen2 action-expert prefill Region."""

    import torch

    model.lm_head.set_adapter("action_expert")
    implementation = model.lm_head.inference_waypoints.__self__
    _make_minddrive_qwen_rotary_exportable(implementation.model)

    class ActionExpert(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = implementation.model

        def forward(
            self, input_ids: Any, vision_tokens: Any
        ) -> Any:
            prefix_ids = input_ids[:MINDDRIVE_IMAGE_TOKEN_INDEX]
            suffix_ids = input_ids[MINDDRIVE_IMAGE_TOKEN_INDEX + 1 :]
            prefix = self.language_model.embed_tokens(prefix_ids)
            suffix = self.language_model.embed_tokens(suffix_ids)
            inputs_embeds = torch.cat(
                (prefix, vision_tokens[0], suffix), dim=0
            ).unsqueeze(0)
            hidden = self.language_model(
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
            )[0]
            return torch.stack(
                (
                    hidden[0, MINDDRIVE_ACTION_HIDDEN_POSITIONS[0], :],
                    hidden[0, MINDDRIVE_ACTION_HIDDEN_POSITIONS[1], :],
                ),
                dim=0,
            )

    return ActionExpert().eval()


def make_minddrive_trajectory_decoder(model: Any) -> Any:
    """Build the real probabilistic GRU trajectory/path decode Region.

    Upstream draws two implicit Gaussian tensors.  They are explicit Region
    inputs here so retry, parity, and transactional failure semantics are
    deterministic and auditable.
    """

    import torch

    present_distribution = model.present_distribution
    path_present_distribution = model.pw_present_distribution
    predict_model = model.predict_model
    path_predict_model = model.pw_predict_model
    trajectory_head = model.ego_fut_decoder
    path_head = model.pw_ego_fut_decoder

    class ExportablePredictModel(torch.nn.Module):
        """Source-equivalent GRU predictor without eager flat-weight mutation.

        ``nn.GRU.forward`` refreshes a cached flattened-weight view before
        invoking ATen.  That eager implementation detail reads storage data
        pointers and is therefore illegal under FakeTensor strict export.
        Calling the same ATen GRU primitive with the named parameters keeps
        the numerical operation intact and removes only that hidden mutation.
        """

        def __init__(self, upstream: Any) -> None:
            super().__init__()
            self.gru = upstream.gru
            self.linear1 = upstream.linear1
            self.linear2 = upstream.linear2
            self.linear3 = upstream.linear3

        def forward(self, value: Any, hidden: Any) -> Any:
            flat_weights = []
            for layer in range(self.gru.num_layers):
                flat_weights.extend(
                    (
                        getattr(self.gru, f"weight_ih_l{layer}"),
                        getattr(self.gru, f"weight_hh_l{layer}"),
                        getattr(self.gru, f"bias_ih_l{layer}"),
                        getattr(self.gru, f"bias_hh_l{layer}"),
                    )
                )
            value, _ = torch._VF.gru(
                value,
                hidden,
                flat_weights,
                self.gru.bias,
                self.gru.num_layers,
                self.gru.dropout,
                False,
                self.gru.bidirectional,
                self.gru.batch_first,
            )
            value = torch.relu(self.linear1(value))
            value = torch.relu(self.linear2(value))
            return self.linear3(value)

    class TrajectoryDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.present_distribution = present_distribution
            self.path_present_distribution = path_present_distribution
            self.predict_model = ExportablePredictModel(predict_model)
            self.path_predict_model = ExportablePredictModel(
                path_predict_model
            )
            self.trajectory_head = trajectory_head
            self.path_head = path_head
            self.register_buffer(
                "path_index_to_value",
                torch.tensor(
                    (2, 4, 1, 0, 3, 5),
                    dtype=torch.int64,
                    device=next(trajectory_head.parameters()).device,
                ),
            )

        @staticmethod
        def _sample(
            distribution: Any,
            current_state: Any,
            noise: Any,
        ) -> Any:
            mean, log_sigma = distribution(current_state)
            sampled = mean + torch.exp(log_sigma) * noise
            return sampled.permute(0, 2, 1).expand(
                current_state.shape[0],
                sampled.shape[-1],
                current_state.shape[1],
            )

        @staticmethod
        def _future_states(
            predictor: Any,
            sample: Any,
            hidden: Any,
            current_state: Any,
            steps: int,
        ) -> Any:
            future_input = sample.unsqueeze(0).expand(
                steps, -1, -1, -1
            )
            future_input = future_input.reshape(steps, -1, 32)
            gru_hidden = hidden.permute(1, 0, 2).reshape(4, -1, 224)
            future = predictor(future_input, gru_hidden.contiguous())
            future = future.reshape(steps, 1, -1, future.shape[2])
            current = current_state.unsqueeze(0).repeat(
                steps, 1, 1, 1
            )
            return torch.cat((current, future), dim=-1)

        @staticmethod
        def _decode_steps(
            head: Any,
            states: Any,
            *,
            modes: int,
            steps: int,
        ) -> Any:
            step_outputs = []
            for index in range(steps):
                decoded = head(states[index]).reshape(1, modes, 2)
                step_outputs.append(decoded)
            return torch.stack(step_outputs, dim=2)

        def forward(
            self,
            action_hidden: Any,
            decision_logits: Any,
            route_command: Any,
            trajectory_noise: Any,
            path_noise: Any,
        ) -> tuple[Any, Any, Any, Any]:
            ego_feature = action_hidden.reshape(1, 2, 896)
            trajectory_hidden = ego_feature[:, 0].unsqueeze(1)
            path_hidden = ego_feature[:, 1].unsqueeze(1)
            trajectory_sample = self._sample(
                self.present_distribution,
                trajectory_hidden,
                trajectory_noise,
            )
            path_sample = self._sample(
                self.path_present_distribution,
                path_hidden,
                path_noise,
            )
            trajectory_states = self._future_states(
                self.predict_model,
                trajectory_sample,
                trajectory_hidden,
                trajectory_hidden,
                6,
            )
            path_states = self._future_states(
                self.path_predict_model,
                path_sample,
                path_hidden,
                path_hidden,
                20,
            )
            trajectory_modes = self._decode_steps(
                self.trajectory_head,
                trajectory_states[:, :, 0, :].unsqueeze(1),
                modes=7,
                steps=6,
            )
            path_modes = self._decode_steps(
                self.path_head,
                path_states[:, :, 0, :].unsqueeze(1),
                modes=6,
                steps=20,
            )
            speed_index = torch.argmax(decision_logits[0], dim=-1)
            raw_path_index = torch.argmax(
                route_command[0, 0, 0], dim=-1
            )
            path_value = torch.gather(
                self.path_index_to_value,
                0,
                raw_path_index.reshape(1),
            )[0]
            trajectory = torch.gather(
                trajectory_modes[0],
                0,
                speed_index.reshape(1, 1, 1).expand(1, 6, 2),
            )[0].cumsum(dim=-2)
            path = torch.gather(
                path_modes[0],
                0,
                path_value.reshape(1, 1, 1).expand(1, 20, 2),
            )[0].cumsum(dim=-2)
            return trajectory, path, speed_index, path_value

    return TrajectoryDecoder().eval()


def _make_minddrive_qwen_rotary_exportable(language_model: Any) -> None:
    """Remove an inference-only autocast context unsupported by torch 2.4.

    Transformers forces Qwen RoPE arithmetic to FP32 under a disabled autocast
    context.  This profile is already FP32, so the context has no numerical
    effect but becomes an unverifiable ``_enter_autocast`` node in PyTorch
    2.4.  The replacement uses the same pinned inverse frequencies and scaling
    directly in ATen.
    """

    import torch

    upstream = language_model.rotary_emb
    if "dynamic" in str(upstream.rope_type):
        raise ValueError("MindDrive deployment profile requires static RoPE")

    class ExportableRotary(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "inv_freq",
                upstream.inv_freq.detach().clone(),
            )
            scaling = upstream.attention_scaling
            if torch.is_tensor(scaling):
                self.register_buffer(
                    "attention_scaling",
                    scaling.detach().clone(),
                )
            else:
                self.attention_scaling = float(scaling)

        def forward(self, hidden: Any, position_ids: Any) -> tuple[Any, Any]:
            inverse = self.inv_freq[None, :, None].float().expand(
                position_ids.shape[0], -1, 1
            )
            positions = position_ids[:, None, :].float()
            frequencies = (inverse @ positions).transpose(1, 2)
            embedding = torch.cat((frequencies, frequencies), dim=-1)
            cosine = embedding.cos() * self.attention_scaling
            sine = embedding.sin() * self.attention_scaling
            return cosine.to(hidden.dtype), sine.to(hidden.dtype)

    language_model.rotary_emb = ExportableRotary()


def _is_allowed_training_extra(key: str) -> bool:
    return (
        key.startswith("value_net.")
        or key.startswith("value_net_pro.")
        or (
            key.startswith("lm_head.")
            and key.endswith("rotary_emb.inv_freq")
        )
    )


def _verify_release_file(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    verify_hash: bool,
) -> None:
    if path.stat().st_size != expected_size:
        raise ValueError(f"MindDrive release size mismatch: {path}")
    if not verify_hash:
        return
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"MindDrive release SHA256 mismatch: {path}")
