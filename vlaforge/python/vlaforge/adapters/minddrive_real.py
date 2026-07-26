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
    ("can_bus", TensorType((1, 18), "f32")),
    ("lidar2img", TensorType((1, 6, 4, 4), "f32")),
    ("camera_intrinsics", TensorType((1, 6, 4, 4), "f32")),
    ("timestamp", TensorType((1,), "f64")),
    ("ego_pose", TensorType((1, 4, 4), "f32")),
    ("ego_pose_inverse", TensorType((1, 4, 4), "f32")),
    ("route_command_index", TensorType((1,), "f32")),
)

MINDDRIVE_IMAGE_FEATURES = TensorType((1, 6, 1024, 40, 40), "f32")

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
