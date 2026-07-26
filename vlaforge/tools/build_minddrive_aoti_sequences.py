#!/usr/bin/env python3
"""Build verified physical AOTI sequences for MindDrive vision and map."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vlaforge.deployment import (
    AotiSequenceArtifact,
    AotiSequenceManifest,
    AotiSequenceNode,
    AotiSequenceValue,
)


_DTYPE_NAMES = {
    "torch.bool": "bool",
    "torch.float16": "f16",
    "torch.float32": "f32",
    "torch.float64": "f64",
    "torch.int32": "i32",
    "torch.int64": "i64",
    "torch.uint8": "u8",
}
_MAP_STATE_NAMES = (
    "map_memory_embedding",
    "map_memory_reference_point",
    "map_memory_timestamp",
    "map_memory_egopose",
    "map_sample_time",
    "map_memory_mask",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PhysicalArtifact:
    name: str
    source: Path
    sha256: str
    size_bytes: int
    inputs: tuple[dict[str, Any], ...]
    outputs: tuple[dict[str, Any], ...]

    @property
    def sequence_path(self) -> str:
        return f"physical/{self.source.name}"

    @property
    def bundle_path(self) -> str:
        return f"artifacts/{self.sequence_path}"


class SequenceBuilder:
    def __init__(self) -> None:
        self.values: list[AotiSequenceValue] = []
        self.artifacts: list[AotiSequenceArtifact] = []
        self.nodes: list[AotiSequenceNode] = []
        self.sources: dict[str, str] = {}
        self._artifact_ids: dict[str, int] = {}

    def value(
        self,
        name: str,
        role: str,
        spec: dict[str, Any],
        binding: int | None = None,
    ) -> int:
        value_id = len(self.values)
        self.values.append(
            AotiSequenceValue(
                value_id,
                name,
                role,
                binding,
                _dtype(spec),
                _shape(spec),
            )
        )
        return value_id

    def artifact(self, physical: PhysicalArtifact) -> int:
        if physical.name in self._artifact_ids:
            return self._artifact_ids[physical.name]
        artifact_id = len(self.artifacts)
        self.artifacts.append(
            AotiSequenceArtifact(
                artifact_id,
                physical.name,
                physical.sequence_path,
                physical.sha256,
                physical.size_bytes,
            )
        )
        self.sources[physical.bundle_path] = str(physical.source)
        self._artifact_ids[physical.name] = artifact_id
        return artifact_id

    def node(
        self,
        physical: PhysicalArtifact,
        inputs: tuple[int, ...],
        outputs: tuple[int, ...],
    ) -> None:
        if len(inputs) != len(physical.inputs):
            raise ValueError(
                f"{physical.name}: physical input arity changed"
            )
        if len(outputs) != len(physical.outputs):
            raise ValueError(
                f"{physical.name}: physical output arity changed"
            )
        for value_id, spec in zip(inputs, physical.inputs, strict=True):
            _require_value_matches(
                self.values[value_id], spec, physical.name, "input"
            )
        for value_id, spec in zip(outputs, physical.outputs, strict=True):
            _require_value_matches(
                self.values[value_id], spec, physical.name, "output"
            )
        self.nodes.append(
            AotiSequenceNode(self.artifact(physical), inputs, outputs)
        )

    def manifest(self, region_name: str) -> AotiSequenceManifest:
        return AotiSequenceManifest(
            region_name=region_name,
            target="sm_86",
            device="cuda:0",
            values=tuple(self.values),
            artifacts=tuple(self.artifacts),
            nodes=tuple(self.nodes),
        )


def _dtype(spec: dict[str, Any]) -> str:
    try:
        return _DTYPE_NAMES[str(spec["dtype"])]
    except KeyError as error:
        raise ValueError(
            f"unsupported MindDrive physical dtype: {spec.get('dtype')}"
        ) from error


def _shape(spec: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(item) for item in spec["shape"])


def _require_value_matches(
    value: AotiSequenceValue,
    spec: dict[str, Any],
    artifact: str,
    role: str,
) -> None:
    if value.dtype != _dtype(spec) or value.shape != _shape(spec):
        raise ValueError(
            f"{artifact}: {role} value {value.name} metadata changed"
        )


def _physical_from_manifest(
    aggregate: dict[str, Any],
) -> dict[str, PhysicalArtifact]:
    physical = {}
    for item in aggregate["regions"]:
        artifact = item["artifact"]
        source = Path(artifact["path"]).resolve()
        expected_hash = str(artifact["sha256"])
        expected_size = int(artifact["size_bytes"])
        if (
            not source.is_file()
            or source.stat().st_size != expected_size
            or _sha256(source) != expected_hash
        ):
            raise ValueError(
                f"MindDrive physical artifact failed verification: {source}"
            )
        abi = item["physical_tensor_abi"]
        physical[str(item["name"])] = PhysicalArtifact(
            name=str(item["name"]),
            source=source,
            sha256=expected_hash,
            size_bytes=expected_size,
            inputs=tuple(abi["inputs"]),
            outputs=tuple(abi["compiled_outputs"]),
        )
    return physical


def _sdpa_from_report(
    report: dict[str, Any],
) -> dict[str, PhysicalArtifact]:
    physical = {}
    for item in report["profiles"]:
        artifact = item["artifact"]
        source = Path(artifact["path"]).resolve()
        expected_hash = str(artifact["sha256"])
        expected_size = int(artifact["size_bytes"])
        if (
            not source.is_file()
            or source.stat().st_size != expected_size
            or _sha256(source) != expected_hash
        ):
            raise ValueError(
                f"MindDrive SDPA artifact failed verification: {source}"
            )
        name = f"sdpa_{item['profile']}"
        abi = item["physical_tensor_abi"]
        physical[name] = PhysicalArtifact(
            name=name,
            source=source,
            sha256=expected_hash,
            size_bytes=expected_size,
            inputs=tuple(abi["inputs"]),
            outputs=(abi["output"],),
        )
    if set(physical) != {"sdpa_window", "sdpa_global"}:
        raise ValueError("MindDrive SDPA profile coverage changed")
    return physical


def _build_vision(
    physical: dict[str, PhysicalArtifact],
) -> tuple[AotiSequenceManifest, dict[str, str]]:
    builder = SequenceBuilder()
    stem = physical["vision_stem"]
    camera = builder.value("camera_images", "input", stem.inputs[0], 0)
    features = builder.value(
        "features_stem", "temporary", stem.outputs[0]
    )
    builder.node(stem, (camera,), (features,))

    for index in range(24):
        pre = physical[f"vision_block_{index:02d}_pre"]
        post = physical[f"vision_block_{index:02d}_post"]
        shortcut = builder.value(
            f"block_{index:02d}_shortcut",
            "temporary",
            pre.outputs[0],
        )
        query = builder.value(
            f"block_{index:02d}_query",
            "temporary",
            pre.outputs[1],
        )
        key_value = builder.value(
            f"block_{index:02d}_key_value",
            "temporary",
            pre.outputs[2],
        )
        builder.node(pre, (features,), (shortcut, query, key_value))
        profile = (
            "global"
            if _shape(pre.outputs[1])[1] == 1600
            else "window"
        )
        sdpa = physical[f"sdpa_{profile}"]
        attention = builder.value(
            f"block_{index:02d}_attention",
            "temporary",
            sdpa.outputs[0],
        )
        builder.node(sdpa, (query, key_value), (attention,))
        features = builder.value(
            f"features_block_{index:02d}",
            "temporary",
            post.outputs[0],
        )
        builder.node(post, (shortcut, attention), (features,))

    finish = physical["vision_finish"]
    image_features = builder.value(
        "image_features", "output", finish.outputs[0], 0
    )
    builder.node(finish, (features,), (image_features,))
    return builder.manifest("vision_encoder"), builder.sources


def _build_map(
    physical: dict[str, PhysicalArtifact],
) -> tuple[AotiSequenceManifest, dict[str, str]]:
    builder = SequenceBuilder()
    front = physical["map_front"]
    layers = tuple(
        physical[f"map_decoder_layer_{index:02d}"]
        for index in range(6)
    )
    finish = physical["map_finish"]
    external_specs = (
        front.inputs[0],
        layers[0].inputs[3],
        front.inputs[1],
        finish.inputs[8],
        front.inputs[2],
        *front.inputs[3:],
    )
    external_names = (
        "image_features",
        "position_embedding",
        "timestamp",
        "ego_pose",
        "ego_pose_inverse",
        *_MAP_STATE_NAMES,
    )
    external = tuple(
        builder.value(name, "input", spec, index)
        for index, (name, spec) in enumerate(
            zip(external_names, external_specs, strict=True)
        )
    )
    front_outputs = tuple(
        builder.value(
            f"front_{index:02d}",
            "temporary",
            spec,
        )
        for index, spec in enumerate(front.outputs)
    )
    builder.node(
        front,
        (
            external[0],
            external[2],
            external[4],
            *external[5:],
        ),
        front_outputs,
    )
    query = front_outputs[0]
    decoded = []
    for index, layer in enumerate(layers):
        next_query = builder.value(
            f"decoded_query_{index:02d}",
            "temporary",
            layer.outputs[0],
        )
        builder.node(
            layer,
            (
                query,
                front_outputs[1],
                front_outputs[2],
                external[1],
                front_outputs[3],
                front_outputs[4],
                front_outputs[5],
            ),
            (next_query,),
        )
        query = next_query
        decoded.append(query)
    output_names = (
        "map_classes",
        "map_coordinates",
        "map_queries",
        "map_tokens",
        *(f"{name}_next" for name in _MAP_STATE_NAMES),
    )
    outputs = tuple(
        builder.value(name, "output", spec, index)
        for index, (name, spec) in enumerate(
            zip(output_names, finish.outputs, strict=True)
        )
    )
    builder.node(
        finish,
        (
            *decoded,
            front_outputs[6],
            external[2],
            external[3],
            front_outputs[7],
            front_outputs[4],
            front_outputs[8],
            front_outputs[9],
            front_outputs[10],
            front_outputs[11],
        ),
        outputs,
    )
    return builder.manifest("map_encoder"), builder.sources


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--sdpa-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    aggregate = json.loads(
        args.artifact_manifest.resolve().read_text(encoding="utf-8")
    )
    sdpa_report = json.loads(
        args.sdpa_report.resolve().read_text(encoding="utf-8")
    )
    if (
        aggregate.get("schema")
        != "vlaforge.minddrive_aoti_artifact_manifest/1"
        or not aggregate.get("passed")
        or int(aggregate.get("expected_region_count", -1)) != 64
    ):
        raise ValueError("MindDrive aggregate artifact manifest is not frozen")
    if (
        sdpa_report.get("schema")
        != "vlaforge.minddrive_sdpa_aoti24/1"
        or not sdpa_report.get("passed")
    ):
        raise ValueError("MindDrive SDPA report is not passed")

    physical = _physical_from_manifest(aggregate)
    physical.update(_sdpa_from_report(sdpa_report))
    vision, vision_sources = _build_vision(physical)
    map_sequence, map_sources = _build_map(physical)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_records = []
    for name, manifest in (
        ("vision_encoder", vision),
        ("map_encoder", map_sequence),
    ):
        path = output_root / f"{name}.vfseq"
        path.write_text(manifest.canonical_text(), encoding="utf-8")
        restored = AotiSequenceManifest.from_text(
            path.read_text(encoding="utf-8")
        )
        if restored.canonical_text() != manifest.canonical_text():
            raise ValueError(f"{name}: sequence roundtrip changed")
        manifest_records.append(
            {
                "region": name,
                "path": str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "value_count": len(manifest.values),
                "artifact_count": len(manifest.artifacts),
                "node_count": len(manifest.nodes),
                "input_count": manifest.input_count,
                "output_count": manifest.output_count,
            }
        )
    sources = {**vision_sources, **map_sources}
    report = {
        "schema": "vlaforge.minddrive_aoti_sequences/1",
        "passed": True,
        "target": "sm_86",
        "device": "cuda:0",
        "manifests": manifest_records,
        "auxiliary_files": [
            {
                "bundle_path": bundle_path,
                "source": source,
                "sha256": _sha256(Path(source)),
                "size_bytes": Path(source).stat().st_size,
            }
            for bundle_path, source in sorted(sources.items())
        ],
        "logical_region_count": 2,
        "physical_artifact_count": len(sources),
        "core_op_delta": 0,
    }
    report_path = output_root / "sequence_manifest.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
