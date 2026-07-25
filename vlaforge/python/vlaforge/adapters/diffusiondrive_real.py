"""Real-checkpoint DiffusionDrive deployment adapter.

The model dependencies remain lazy so importing :mod:`vlaforge.adapters` does
not require NAVSIM, timm, diffusers, or PyTorch.  Sensor construction and time
synchronization are deliberately outside this adapter: the generated Session
accepts already-prepared camera, LiDAR-BEV, ego-status, and explicit noise
tensors.
"""

from __future__ import annotations

import hashlib
import sys
import types
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
    TensorRegion,
    Value,
)
from vlaforge.ir.types import PendingOutputType, ScalarType, TensorType


DIFFUSIONDRIVE_UPSTREAM_REVISION = (
    "9b52ed0ec06b073d82d6f392ab084c7b301c8681"
)
DIFFUSIONDRIVE_HF_REVISION = (
    "8e3cc29cfdb5aa1a4c0818012f9a250d5153bc71"
)
DIFFUSIONDRIVE_CHECKPOINT_SHA256 = (
    "008ffc39cc6c57ff9007025217e601f408818afa036c0bae4e543907993a005b"
)
DIFFUSIONDRIVE_CHECKPOINT_SIZE = 729_518_199

_BATCH_SIZE = 1
_NUM_MODES = 20
_NUM_POSES = 8
_TRAJECTORY_DIMS = 3
_DIFFUSION_DIMS = 2
_MODEL_DIMS = 256
_NUM_AGENTS = 30
_AGENT_DIMS = 5
_BEV_HEIGHT = 64
_BEV_WIDTH = 64
_SEMANTIC_HEIGHT = 128
_SEMANTIC_WIDTH = 256
_SEMANTIC_CLASSES = 7
_PLANNER_STATE_SIZE = (
    _NUM_MODES * _NUM_POSES * _DIFFUSION_DIMS
    + _NUM_MODES * _NUM_POSES * _TRAJECTORY_DIMS
    + _NUM_MODES
)


@dataclass(frozen=True, slots=True)
class RealDiffusionDriveConfig:
    source_root: Path
    checkpoint: Path
    device: str = "cuda:0"
    upstream_revision: str = DIFFUSIONDRIVE_UPSTREAM_REVISION
    checkpoint_revision: str = DIFFUSIONDRIVE_HF_REVISION


@dataclass(frozen=True, slots=True)
class RealDiffusionDriveRegions:
    model: Any
    condition_encoder: Any
    initialize_planner_state: Any
    make_denoise_timestep: Any
    denoise_planner_step: Any
    decode_planner_outputs: Any
    example_inputs: dict[str, Any]
    checkpoint_sha256: str
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def run_real_diffusiondrive_chain(
    regions: RealDiffusionDriveRegions,
) -> dict[str, Any]:
    """Execute the explicit Region partition with the fixed example inputs."""

    import torch

    inputs = regions.example_inputs
    with torch.no_grad():
        condition = regions.condition_encoder(
            inputs["camera_feature"],
            inputs["lidar_feature"],
            inputs["status_feature"],
        )
        planner_state = regions.initialize_planner_state(inputs["noise"])
        for step in range(2):
            timestep = regions.make_denoise_timestep(
                torch.tensor(step, dtype=torch.int64)
            )
            planner_state = regions.denoise_planner_step(
                planner_state,
                timestep,
                condition[0],
                condition[1],
                condition[2],
                condition[3],
            )
        candidates, scores, trajectory = (
            regions.decode_planner_outputs(planner_state)
        )
    return {
        "candidate_trajectories": candidates,
        "candidate_scores": scores,
        "trajectory": trajectory,
        "bev_semantic_map": condition[4],
        "agent_states": condition[5],
        "agent_labels": condition[6],
        "planner_state": planner_state,
    }


def run_upstream_diffusiondrive_with_explicit_noise(
    regions: RealDiffusionDriveRegions,
) -> dict[str, Any]:
    """Run pinned upstream ``forward`` with explicit noise and full outputs.

    Upstream returns only the selected trajectory from its planner head.  A
    read-only forward hook captures the final candidate and score tensors that
    are already computed by the unmodified upstream decoder so the deployment
    partition can prove parity for every transactional planning output.
    """

    import torch

    inputs = regions.example_inputs
    noise = inputs["noise"]
    original_randn = torch.randn
    planner_outputs: dict[str, Any] = {}

    def capture_planner_outputs(
        _module: Any,
        _arguments: tuple[Any, ...],
        output: tuple[list[Any], list[Any]],
    ) -> None:
        pose_list, score_list = output
        planner_outputs["candidate_trajectories"] = pose_list[-1]
        planner_outputs["candidate_scores"] = score_list[-1]

    hook = regions.model._trajectory_head.diff_decoder.register_forward_hook(
        capture_planner_outputs
    )

    def explicit_randn(*shape: object, **kwargs: object) -> Any:
        requested = (
            tuple(shape[0])
            if len(shape) == 1 and hasattr(shape[0], "__iter__")
            else tuple(int(item) for item in shape)
        )
        if requested == tuple(noise.shape):
            device = kwargs.get("device", noise.device)
            dtype = kwargs.get("dtype", noise.dtype)
            return noise.to(device=device, dtype=dtype).clone()
        return original_randn(*shape, **kwargs)

    torch.randn = explicit_randn
    try:
        with torch.inference_mode():
            output = regions.model(
                {
                    "camera_feature": inputs["camera_feature"],
                    "lidar_feature": inputs["lidar_feature"],
                    "status_feature": inputs["status_feature"],
                }
            )
        if set(planner_outputs) != {
            "candidate_trajectories",
            "candidate_scores",
        }:
            raise RuntimeError(
                "upstream DiffusionDrive planner hook captured no outputs"
            )
        return {**output, **planner_outputs}
    finally:
        hook.remove()
        torch.randn = original_randn


def capture_real_diffusiondrive_regions(
    regions: RealDiffusionDriveRegions,
    output_dir: str | Path,
    *,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-5,
) -> tuple[Any, ...]:
    """Export and persist the five real-checkpoint TensorRegions."""

    import torch
    from vlaforge.frontend import capture_region, save_exported_region

    module = build_real_diffusiondrive_program()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inputs = regions.example_inputs
    with torch.no_grad():
        condition = regions.condition_encoder(
            inputs["camera_feature"],
            inputs["lidar_feature"],
            inputs["status_feature"],
        )
        initial = regions.initialize_planner_state(inputs["noise"])
        first_timestep = regions.make_denoise_timestep(
            torch.tensor(0, dtype=torch.int64)
        )
        first_step = regions.denoise_planner_step(
            initial,
            first_timestep,
            condition[0],
            condition[1],
            condition[2],
            condition[3],
        )
        second_timestep = regions.make_denoise_timestep(
            torch.tensor(1, dtype=torch.int64)
        )
        final_step = regions.denoise_planner_step(
            first_step,
            second_timestep,
            condition[0],
            condition[1],
            condition[2],
            condition[3],
        )

    declarations = (
        (
            "condition_encoder",
            regions.condition_encoder,
            (
                inputs["camera_feature"],
                inputs["lidar_feature"],
                inputs["status_feature"],
            ),
        ),
        (
            "initialize_planner_state",
            regions.initialize_planner_state,
            (inputs["noise"],),
        ),
        (
            "make_denoise_timestep",
            regions.make_denoise_timestep,
            (torch.tensor(0, dtype=torch.int64),),
        ),
        (
            "denoise_planner_step",
            regions.denoise_planner_step,
            (
                first_step,
                second_timestep,
                condition[0],
                condition[1],
                condition[2],
                condition[3],
            ),
        ),
        (
            "decode_planner_outputs",
            regions.decode_planner_outputs,
            (final_step,),
        ),
    )
    captures = []
    for name, implementation, arguments in declarations:
        capture = capture_region(
            module.region(name),
            implementation,
            arguments,
            strict=True,
            explicit_rng=name == "initialize_planner_state",
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        capture.require_supported()
        save_exported_region(
            capture,
            program_path=output / f"{name}.pt2e",
            evidence_path=output / f"{name}.capture.json",
        )
        captures.append(capture)
    return tuple(captures)


def build_real_diffusiondrive_program(
    *,
    device: str = "cuda:0",
    denoise_steps: int = 2,
) -> Any:
    """Build the source-faithful real DiffusionDrive invocation program."""

    if denoise_steps != 2:
        raise ValueError("pinned DiffusionDrive uses exactly two denoise steps")

    camera = TensorType((_BATCH_SIZE, 3, 256, 1024), "f32")
    lidar = TensorType((_BATCH_SIZE, 1, 256, 256), "f32")
    status = TensorType((_BATCH_SIZE, 8), "f32")
    noise = TensorType(
        (_BATCH_SIZE, _NUM_MODES, _NUM_POSES, _DIFFUSION_DIMS),
        "f32",
    )
    ego_query = TensorType((_BATCH_SIZE, 1, _MODEL_DIMS), "f32")
    agents_query = TensorType(
        (_BATCH_SIZE, _NUM_AGENTS, _MODEL_DIMS), "f32"
    )
    cross_bev = TensorType(
        (_BATCH_SIZE, _MODEL_DIMS, _BEV_HEIGHT, _BEV_WIDTH), "f32"
    )
    status_encoding = TensorType(
        (_BATCH_SIZE, 1, _MODEL_DIMS), "f32"
    )
    bev_semantic = TensorType(
        (
            _BATCH_SIZE,
            _SEMANTIC_CLASSES,
            _SEMANTIC_HEIGHT,
            _SEMANTIC_WIDTH,
        ),
        "f32",
    )
    agent_states = TensorType(
        (_BATCH_SIZE, _NUM_AGENTS, _AGENT_DIMS), "f32"
    )
    agent_labels = TensorType((_BATCH_SIZE, _NUM_AGENTS), "f32")
    planner_state = TensorType(
        (_BATCH_SIZE, _PLANNER_STATE_SIZE), "f32"
    )
    timestep = TensorType((_BATCH_SIZE,), "i64")
    candidates = TensorType(
        (
            _BATCH_SIZE,
            _NUM_MODES,
            _NUM_POSES,
            _TRAJECTORY_DIMS,
        ),
        "f32",
    )
    scores = TensorType((_BATCH_SIZE, _NUM_MODES), "f32")
    trajectory = TensorType(
        (_BATCH_SIZE, _NUM_POSES, _TRAJECTORY_DIMS), "f32"
    )
    index = ScalarType("index")

    builder = ModuleBuilder("diffusiondrive_real_cuda_l4")
    for name, payload in (
        ("camera_feature", camera),
        ("lidar_feature", lidar),
        ("status_feature", status),
        ("noise", noise),
    ):
        builder.add_input(
            InputPort(name, payload, device=device, alignment=64)
        )
    for name, payload in (
        ("candidate_trajectories", candidates),
        ("candidate_scores", scores),
        ("trajectory", trajectory),
        ("bev_semantic_map", bev_semantic),
        ("agent_states", agent_states),
        ("agent_labels", agent_labels),
    ):
        builder.add_output(
            OutputPort(
                name,
                payload,
                group="planning",
                device=device,
                alignment=64,
            )
        )

    builder.add_region(
        TensorRegion(
            "condition_encoder",
            (
                Value("camera", camera),
                Value("lidar", lidar),
                Value("status", status),
            ),
            (
                ego_query,
                agents_query,
                cross_bev,
                status_encoding,
                bev_semantic,
                agent_states,
                agent_labels,
            ),
            metadata={
                "memoize": True,
                "cache_input_ports": [
                    "camera_feature",
                    "lidar_feature",
                    "status_feature",
                ],
                "cache_state_slots": [],
                "loop_invariant": True,
                "template": "DiffusionPlanner",
            },
        )
    )
    builder.add_region(
        TensorRegion(
            "initialize_planner_state",
            (Value("noise", noise),),
            (planner_state,),
        )
    )
    builder.add_region(
        TensorRegion(
            "make_denoise_timestep",
            (Value("step", index),),
            (timestep,),
        )
    )
    builder.add_region(
        TensorRegion(
            "denoise_planner_step",
            (
                Value("state", planner_state),
                Value("timestep", timestep),
                Value("ego_query", ego_query),
                Value("agents_query", agents_query),
                Value("cross_bev", cross_bev),
                Value("status_encoding", status_encoding),
            ),
            (planner_state,),
        )
    )
    builder.add_region(
        TensorRegion(
            "decode_planner_outputs",
            (Value("state", planner_state),),
            (candidates, scores, trajectory),
        )
    )

    loop = Block.of(
        (
            ops.invoke(
                ("denoise_timestep",),
                (timestep,),
                "make_denoise_timestep",
                ("denoise_index",),
            ),
            ops.invoke(
                ("planner_state_next",),
                (planner_state,),
                "denoise_planner_step",
                (
                    "planner_state_iter",
                    "denoise_timestep",
                    "ego_query",
                    "agents_query",
                    "cross_bev",
                    "status_encoding",
                ),
            ),
            ops.yield_values("planner_state_next"),
        )
    )

    pending = tuple(
        PendingOutputType(name, payload)
        for name, payload in (
            ("candidate_trajectories", candidates),
            ("candidate_scores", scores),
            ("trajectory", trajectory),
            ("bev_semantic_map", bev_semantic),
            ("agent_states", agent_states),
            ("agent_labels", agent_labels),
        )
    )
    body = Block.of(
        (
            ops.input_read(
                "camera_value", "camera_revision", camera, "camera_feature"
            ),
            ops.input_read(
                "lidar_value", "lidar_revision", lidar, "lidar_feature"
            ),
            ops.input_read(
                "status_value", "status_revision", status, "status_feature"
            ),
            ops.input_read(
                "noise_value", "noise_revision", noise, "noise"
            ),
            ops.transaction_begin("txn"),
            ops.invoke(
                (
                    "ego_query",
                    "agents_query",
                    "cross_bev",
                    "status_encoding",
                    "bev_semantic_map",
                    "agent_states",
                    "agent_labels",
                ),
                (
                    ego_query,
                    agents_query,
                    cross_bev,
                    status_encoding,
                    bev_semantic,
                    agent_states,
                    agent_labels,
                ),
                "condition_encoder",
                ("camera_value", "lidar_value", "status_value"),
            ),
            ops.invoke(
                ("planner_state_initial",),
                (planner_state,),
                "initialize_planner_state",
                ("noise_value",),
            ),
            ops.for_loop(
                Value("planner_state_final", planner_state),
                "planner_state_initial",
                Value("denoise_index", index),
                Value("planner_state_iter", planner_state),
                loop,
                lower=0,
                upper=denoise_steps,
            ),
            ops.validate(
                "planner_valid",
                "planner_state_final",
                "finite_planner_state",
            ),
            ops.invoke(
                (
                    "candidate_trajectories",
                    "candidate_scores",
                    "trajectory",
                ),
                (candidates, scores, trajectory),
                "decode_planner_outputs",
                ("planner_state_final",),
            ),
            ops.output_create(
                "pending_candidates",
                "candidate_trajectories",
                candidates,
                "candidate_trajectories",
            ),
            ops.output_create(
                "pending_scores",
                "candidate_scores",
                scores,
                "candidate_scores",
            ),
            ops.output_create(
                "pending_trajectory",
                "trajectory",
                trajectory,
                "trajectory",
            ),
            ops.output_create(
                "pending_bev",
                "bev_semantic_map",
                bev_semantic,
                "bev_semantic_map",
            ),
            ops.output_create(
                "pending_agent_states",
                "agent_states",
                agent_states,
                "agent_states",
            ),
            ops.output_create(
                "pending_agent_labels",
                "agent_labels",
                agent_labels,
                "agent_labels",
            ),
            ops.output_group(
                "pending_outputs",
                "planning",
                (
                    ("pending_candidates", pending[0]),
                    ("pending_scores", pending[1]),
                    ("pending_trajectory", pending[2]),
                    ("pending_bev", pending[3]),
                    ("pending_agent_states", pending[4]),
                    ("pending_agent_labels", pending[5]),
                ),
            ),
            ops.transaction_commit(
                "committed_outputs",
                pending,
                "planning",
                "txn",
                "pending_outputs",
                "planner_valid",
            ),
            ops.return_values("committed_outputs"),
        )
    )
    builder.add_invocation(
        Invocation(
            "plan",
            body,
            metadata={
                "adapter_template": "DiffusionPlanner",
                "source_model": "hustvl/DiffusionDrive",
                "source_revision": DIFFUSIONDRIVE_UPSTREAM_REVISION,
                "denoise_steps": denoise_steps,
                "candidate_count": _NUM_MODES,
                "core_op_delta": 0,
                "rng_contract": "explicit_noise_input",
            },
        )
    )
    return builder.build()


def load_real_diffusiondrive_regions(
    config: RealDiffusionDriveConfig,
) -> RealDiffusionDriveRegions:
    """Load the pinned upstream modules and checkpoint for capture.

    NAVSIM dataset classes are not needed for model execution.  Small,
    value-equivalent enum/config stubs isolate the model from the nuPlan
    ingestion stack, while the executed backbone, transformer, diffusion
    decoder, and checkpoint tensors remain the pinned upstream implementation.
    """

    import numpy as np
    import torch

    if config.upstream_revision != DIFFUSIONDRIVE_UPSTREAM_REVISION:
        raise ValueError(
            "DiffusionDrive upstream revision does not match the pinned "
            f"revision {DIFFUSIONDRIVE_UPSTREAM_REVISION}"
        )
    if config.checkpoint_revision != DIFFUSIONDRIVE_HF_REVISION:
        raise ValueError(
            "DiffusionDrive checkpoint revision does not match the pinned "
            f"revision {DIFFUSIONDRIVE_HF_REVISION}"
        )
    source_root = config.source_root.resolve()
    checkpoint_path = config.checkpoint.resolve()
    model_source = (
        source_root
        / "navsim"
        / "agents"
        / "diffusiondrive"
        / "transfuser_model_v2.py"
    )
    if not model_source.is_file():
        raise FileNotFoundError(model_source)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if checkpoint_path.stat().st_size != DIFFUSIONDRIVE_CHECKPOINT_SIZE:
        raise ValueError("DiffusionDrive checkpoint size mismatch")
    checkpoint_sha256 = _sha256(checkpoint_path)
    if checkpoint_sha256 != DIFFUSIONDRIVE_CHECKPOINT_SHA256:
        raise ValueError("DiffusionDrive checkpoint SHA256 mismatch")
    if not torch.cuda.is_available() and config.device.startswith("cuda"):
        raise RuntimeError("DiffusionDrive CUDA capture requires a CUDA device")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    raw_state = checkpoint.get("state_dict")
    if not isinstance(raw_state, dict):
        raise ValueError("DiffusionDrive checkpoint has no state_dict")
    state = {_model_key(str(key)): value for key, value in raw_state.items()}
    anchor_key = next(
        (
            key
            for key in state
            if key.endswith("_trajectory_head.plan_anchor")
        ),
        None,
    )
    if anchor_key is None:
        raise ValueError("DiffusionDrive checkpoint has no trajectory anchor")
    anchor = state[anchor_key].detach().cpu().numpy()
    if anchor.shape != (_NUM_MODES, _NUM_POSES, _DIFFUSION_DIMS):
        raise ValueError(
            f"unexpected DiffusionDrive anchor shape: {anchor.shape}"
        )

    _install_upstream_stubs()
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    import timm

    original_create_model = timm.create_model
    original_numpy_load = np.load

    def offline_create_model(
        name: str, *args: object, **kwargs: object
    ) -> Any:
        kwargs["pretrained"] = False
        kwargs.pop("pretrained_cfg_overlay", None)
        return original_create_model(name, *args, **kwargs)

    def checkpoint_anchor_load(
        path: object, *args: object, **kwargs: object
    ) -> Any:
        if str(path) == "__vlaforge_checkpoint_plan_anchor__":
            return anchor.copy()
        return original_numpy_load(path, *args, **kwargs)

    timm.create_model = offline_create_model
    np.load = checkpoint_anchor_load
    try:
        from navsim.agents.diffusiondrive.transfuser_model_v2 import (
            V2TransfuserModel,
        )

        upstream_config = _upstream_config()
        model = V2TransfuserModel(upstream_config)
    finally:
        timm.create_model = original_create_model
        np.load = original_numpy_load

    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_keys = tuple(sorted(missing))
    unexpected_keys = tuple(sorted(unexpected))
    if missing_keys or unexpected_keys:
        raise ValueError(
            "DiffusionDrive checkpoint/model mismatch: "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )
    model = model.eval().to(config.device)

    implementations = _build_region_implementations(
        model, device=config.device
    )
    generator = torch.Generator(device=config.device)
    generator.manual_seed(20260725)
    example_inputs = {
        "camera_feature": torch.randn(
            (_BATCH_SIZE, 3, 256, 1024),
            generator=generator,
            device=config.device,
            dtype=torch.float32,
        ),
        "lidar_feature": torch.randn(
            (_BATCH_SIZE, 1, 256, 256),
            generator=generator,
            device=config.device,
            dtype=torch.float32,
        ),
        "status_feature": torch.randn(
            (_BATCH_SIZE, 8),
            generator=generator,
            device=config.device,
            dtype=torch.float32,
        ),
        "noise": torch.randn(
            (
                _BATCH_SIZE,
                _NUM_MODES,
                _NUM_POSES,
                _DIFFUSION_DIMS,
            ),
            generator=generator,
            device=config.device,
            dtype=torch.float32,
        ),
    }
    return RealDiffusionDriveRegions(
        model=model,
        condition_encoder=implementations[0],
        initialize_planner_state=implementations[1],
        make_denoise_timestep=implementations[2],
        denoise_planner_step=implementations[3],
        decode_planner_outputs=implementations[4],
        example_inputs=example_inputs,
        checkpoint_sha256=checkpoint_sha256,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
    )


def _build_region_implementations(
    model: Any, *, device: str
) -> tuple[Any, Any, Any, Any, Any]:
    import torch
    import torch.nn.functional as functional

    trajectory_head = model._trajectory_head
    trajectory_head.diffusion_scheduler.set_timesteps(1000, device=device)
    scheduler_alphas = (
        trajectory_head.diffusion_scheduler.alphas_cumprod.detach()
        .clone()
        .to(device=device)
    )
    scheduler_final_alpha = (
        trajectory_head.diffusion_scheduler.final_alpha_cumprod.detach()
        .clone()
        .to(device=device)
    )

    class ConditionEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = model._backbone
            self.bev_downscale = model._bev_downscale
            self.status_encoding = model._status_encoding
            self.keyval_embedding = model._keyval_embedding
            self.query_embedding = model._query_embedding
            self.tf_decoder = model._tf_decoder
            self.bev_semantic_head = model._bev_semantic_head
            self.agent_head = model._agent_head
            self.bev_proj = model.bev_proj

        def forward(
            self, camera: Any, lidar: Any, status: Any
        ) -> tuple[Any, ...]:
            batch_size = status.shape[0]
            bev_upscale, bev_feature, _ = self.backbone(camera, lidar)
            spatial_shape = bev_upscale.shape[2:]
            concat_shape = bev_feature.shape[2:]
            bev_tokens = self.bev_downscale(bev_feature).flatten(-2, -1)
            bev_tokens = bev_tokens.permute(0, 2, 1)
            status_token = self.status_encoding(status)
            keyval = torch.concatenate(
                [bev_tokens, status_token[:, None]], dim=1
            )
            keyval = keyval + self.keyval_embedding.weight[None, ...]
            concat_bev = (
                keyval[:, :-1]
                .permute(0, 2, 1)
                .contiguous()
                .view(batch_size, -1, concat_shape[0], concat_shape[1])
            )
            concat_bev = functional.interpolate(
                concat_bev,
                size=spatial_shape,
                mode="bilinear",
                align_corners=False,
            )
            cross_bev = torch.cat([concat_bev, bev_upscale], dim=1)
            cross_bev = self.bev_proj(
                cross_bev.flatten(-2, -1).permute(0, 2, 1)
            )
            cross_bev = (
                cross_bev.permute(0, 2, 1)
                .contiguous()
                .view(
                    batch_size,
                    -1,
                    spatial_shape[0],
                    spatial_shape[1],
                )
            )
            query = self.query_embedding.weight[None, ...].repeat(
                batch_size, 1, 1
            )
            query_out = self.tf_decoder(query, keyval)
            ego_query, agents_query = query_out.split((1, _NUM_AGENTS), dim=1)
            bev_semantic = self.bev_semantic_head(bev_upscale)
            agents = self.agent_head(agents_query)
            return (
                ego_query,
                agents_query,
                cross_bev,
                status_token[:, None],
                bev_semantic,
                agents["agent_states"],
                agents["agent_labels"],
            )

    class InitializePlannerState(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "plan_anchor",
                trajectory_head.plan_anchor.detach().clone(),
                persistent=True,
            )
            self.register_buffer(
                "alpha_at_truncation",
                scheduler_alphas[8].clone().reshape(1, 1, 1, 1),
                persistent=True,
            )

        def forward(self, noise: Any) -> Any:
            anchor = self.plan_anchor.unsqueeze(0).repeat(
                noise.shape[0], 1, 1, 1
            )
            normalized = _norm_odometry(anchor)
            alpha = self.alpha_at_truncation.to(
                dtype=normalized.dtype
            )
            image = (
                alpha.sqrt() * normalized
                + (1 - alpha).sqrt() * noise
            )
            zeros_candidates = torch.zeros(
                (
                    noise.shape[0],
                    _NUM_MODES,
                    _NUM_POSES,
                    _TRAJECTORY_DIMS,
                ),
                device=noise.device,
                dtype=noise.dtype,
            )
            zeros_scores = torch.zeros(
                (noise.shape[0], _NUM_MODES),
                device=noise.device,
                dtype=noise.dtype,
            )
            return torch.cat(
                (
                    image.flatten(1),
                    zeros_candidates.flatten(1),
                    zeros_scores,
                ),
                dim=1,
            )

    class MakeDenoiseTimestep(torch.nn.Module):
        def forward(self, step: Any) -> Any:
            return torch.where(
                step.to(device=device).reshape(1) == 0,
                torch.full(
                    (1,), 10, dtype=torch.int64, device=device
                ),
                torch.zeros((1,), dtype=torch.int64, device=device),
            )

    class DenoisePlannerStep(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.plan_anchor_encoder = trajectory_head.plan_anchor_encoder
            self.time_mlp = trajectory_head.time_mlp
            self.diff_decoder = trajectory_head.diff_decoder
            self.register_buffer(
                "alpha_at_ten",
                scheduler_alphas[10].clone(),
                persistent=True,
            )
            self.register_buffer(
                "alpha_at_nine",
                scheduler_alphas[9].clone(),
                persistent=True,
            )
            self.register_buffer(
                "alpha_at_zero",
                scheduler_alphas[0].clone(),
                persistent=True,
            )
            self.register_buffer(
                "final_alpha_cumprod",
                scheduler_final_alpha.clone(),
                persistent=True,
            )

        def forward(
            self,
            packed: Any,
            timestep: Any,
            ego_query: Any,
            agents_query: Any,
            cross_bev: Any,
            status_encoding: Any,
        ) -> Any:
            image_size = _NUM_MODES * _NUM_POSES * _DIFFUSION_DIMS
            image = packed[:, :image_size].reshape(
                -1, _NUM_MODES, _NUM_POSES, _DIFFUSION_DIMS
            )
            bounded = torch.clamp(image, min=-1, max=1)
            noisy_points = _denorm_odometry(bounded)
            half_hidden = 32
            dimension = torch.arange(
                half_hidden,
                dtype=torch.float32,
                device=noisy_points.device,
            )
            dimension = 10000 ** (
                2 * (dimension.div(2, rounding_mode="floor")) / half_hidden
            )
            x_position = noisy_points[..., 0] * (2 * torch.pi)
            y_position = noisy_points[..., 1] * (2 * torch.pi)
            x_position = x_position[..., None] / dimension
            y_position = y_position[..., None] / dimension
            x_position = torch.stack(
                (
                    x_position[..., 0::2].sin(),
                    x_position[..., 1::2].cos(),
                ),
                dim=-1,
            ).flatten(-2)
            y_position = torch.stack(
                (
                    y_position[..., 0::2].sin(),
                    y_position[..., 1::2].cos(),
                ),
                dim=-1,
            ).flatten(-2)
            position = torch.cat(
                (y_position, x_position), dim=-1
            ).flatten(-2)
            trajectory_features = self.plan_anchor_encoder(position).view(
                image.shape[0], _NUM_MODES, -1
            )
            expanded_timestep = timestep.expand(image.shape[0])
            time_embedding = self.time_mlp(expanded_timestep).view(
                image.shape[0], 1, -1
            )
            pose_list, score_list = self.diff_decoder(
                trajectory_features,
                noisy_points,
                cross_bev,
                cross_bev.shape[2:],
                agents_query,
                ego_query,
                time_embedding,
                status_encoding,
                None,
            )
            candidates = pose_list[-1]
            scores = score_list[-1]
            normalized_prediction = _norm_odometry(
                candidates[..., :2]
            )
            timestep_value = timestep[0]
            first_step = timestep_value == 10
            alpha = torch.where(
                first_step, self.alpha_at_ten, self.alpha_at_zero
            )
            previous_alpha = torch.where(
                first_step,
                self.alpha_at_nine,
                self.final_alpha_cumprod,
            )
            beta = 1 - alpha
            predicted_original = normalized_prediction.clamp(-1, 1)
            predicted_epsilon = (
                image - alpha.sqrt() * normalized_prediction
            ) / beta.sqrt()
            direction = (1 - previous_alpha).sqrt() * predicted_epsilon
            next_image = (
                previous_alpha.sqrt() * predicted_original + direction
            )
            return torch.cat(
                (
                    next_image.flatten(1),
                    candidates.flatten(1),
                    scores,
                ),
                dim=1,
            )

    class DecodePlannerOutputs(torch.nn.Module):
        def forward(self, packed: Any) -> tuple[Any, Any, Any]:
            image_size = _NUM_MODES * _NUM_POSES * _DIFFUSION_DIMS
            candidate_size = (
                _NUM_MODES * _NUM_POSES * _TRAJECTORY_DIMS
            )
            candidates = packed[
                :, image_size : image_size + candidate_size
            ].reshape(
                -1,
                _NUM_MODES,
                _NUM_POSES,
                _TRAJECTORY_DIMS,
            )
            scores = packed[:, image_size + candidate_size :]
            mode = scores.argmax(dim=-1)
            gather_index = mode[:, None, None, None].expand(
                -1, 1, _NUM_POSES, _TRAJECTORY_DIMS
            )
            trajectory = torch.gather(
                candidates, 1, gather_index
            ).squeeze(1)
            return candidates, scores, trajectory

    return (
        ConditionEncoder().eval(),
        InitializePlannerState().eval(),
        MakeDenoiseTimestep().eval(),
        DenoisePlannerStep().eval(),
        DecodePlannerOutputs().eval(),
    )


def _install_upstream_stubs() -> None:
    config_module_name = (
        "navsim.agents.diffusiondrive.transfuser_config"
    )
    feature_module_name = (
        "navsim.agents.diffusiondrive.transfuser_features"
    )
    enum_module_name = "navsim.common.enums"

    config_module = types.ModuleType(config_module_name)
    config_module.TransfuserConfig = type("TransfuserConfig", (), {})
    sys.modules[config_module_name] = config_module

    class BoundingBox2DIndex:
        X = 0
        Y = 1
        HEADING = 2
        LENGTH = 3
        WIDTH = 4
        POINT = slice(0, 2)

        @classmethod
        def size(cls) -> int:
            return 5

    feature_module = types.ModuleType(feature_module_name)
    feature_module.BoundingBox2DIndex = BoundingBox2DIndex
    sys.modules[feature_module_name] = feature_module

    class StateSE2Index:
        X = 0
        Y = 1
        HEADING = 2

    enum_module = types.ModuleType(enum_module_name)
    enum_module.StateSE2Index = StateSE2Index
    sys.modules[enum_module_name] = enum_module


def _upstream_config() -> Any:
    sampling = types.SimpleNamespace(num_poses=_NUM_POSES)
    return types.SimpleNamespace(
        trajectory_sampling=sampling,
        image_architecture="resnet34",
        lidar_architecture="resnet34",
        bkb_path="",
        plan_anchor_path="__vlaforge_checkpoint_plan_anchor__",
        latent=False,
        max_height_lidar=100.0,
        pixels_per_meter=4.0,
        hist_max_per_pixel=5,
        lidar_min_x=-32,
        lidar_max_x=32,
        lidar_min_y=-32,
        lidar_max_y=32,
        lidar_split_height=0.2,
        use_ground_plane=False,
        lidar_seq_len=1,
        camera_width=1024,
        camera_height=256,
        lidar_resolution_width=256,
        lidar_resolution_height=256,
        img_vert_anchors=8,
        img_horz_anchors=32,
        lidar_vert_anchors=8,
        lidar_horz_anchors=8,
        block_exp=4,
        n_layer=2,
        n_head=4,
        n_scale=4,
        embd_pdrop=0.1,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        gpt_linear_layer_init_mean=0.0,
        gpt_linear_layer_init_std=0.02,
        gpt_layer_norm_init_weight=1.0,
        perspective_downsample_factor=1,
        transformer_decoder_join=True,
        detect_boxes=True,
        use_bev_semantic=True,
        use_semantic=False,
        use_depth=False,
        add_features=True,
        tf_d_model=256,
        tf_d_ffn=1024,
        tf_num_layers=3,
        tf_num_head=8,
        tf_dropout=0.0,
        num_bounding_boxes=30,
        trajectory_weight=12.0,
        trajectory_cls_weight=10.0,
        trajectory_reg_weight=8.0,
        diff_loss_weight=20.0,
        agent_class_weight=10.0,
        agent_box_weight=1.0,
        bev_semantic_weight=14.0,
        use_ema=False,
        num_bev_classes=7,
        bev_features_channels=64,
        bev_down_sample_factor=4,
        bev_upsample_factor=2,
    )


def _model_key(key: str) -> str:
    for prefix in (
        "agent._transfuser_model.",
        "_transfuser_model.",
        "agent.",
    ):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _norm_odometry(value: Any) -> Any:
    x_value = 2 * (value[..., 0:1] + 1.2) / 56.9 - 1
    y_value = 2 * (value[..., 1:2] + 20) / 46 - 1
    heading = 2 * (value[..., 2:3] + 2) / 3.9 - 1
    import torch

    return torch.cat((x_value, y_value, heading), dim=-1)


def _denorm_odometry(value: Any) -> Any:
    x_value = (value[..., 0:1] + 1) / 2 * 56.9 - 1.2
    y_value = (value[..., 1:2] + 1) / 2 * 46 - 20
    heading = (value[..., 2:3] + 1) / 2 * 3.9 - 2
    import torch

    return torch.cat((x_value, y_value, heading), dim=-1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
