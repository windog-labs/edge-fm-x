#!/usr/bin/env python3
"""Audit held-out VLA adapters against a frozen VLAForge core.

This tool deliberately separates two evidence levels:

* L0: pinned upstream source contracts read directly from Git objects.
* L1: deterministic executable fixtures run through Semantic IR and the
  verified, physicalized Plan.

It does not download checkpoints and must not be used to claim real-model
frontend, artifact, or generated-C++ parity.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _SOURCE_ROOT.parent
sys.path.insert(0, str(_SOURCE_ROOT / "python"))

from vlaforge.adapters import (  # noqa: E402
    AdapterFixture,
    build_driving_ar_fixture,
    build_groot_n1_like_fixture,
    build_octo_like_fixture,
    model_contract,
)
from vlaforge.analysis import verify  # noqa: E402
from vlaforge.analysis.verifier import ALLOWED_OPS  # noqa: E402
from vlaforge.compiler import CompilerProfile, compile_module  # noqa: E402
from vlaforge.interpreter import Interpreter  # noqa: E402
from vlaforge.interpreter.trace import normalize_value  # noqa: E402
from vlaforge.ir.program import Block, Operation  # noqa: E402
from vlaforge.plan import PlanExecutor  # noqa: E402


REPORT_SCHEMA = "vlaforge.frozen_core_heldouts/1"
FROZEN_CORE_PATHS = (
    "vlaforge/python/vlaforge/ir",
    "vlaforge/python/vlaforge/compiler.py",
    "vlaforge/python/vlaforge/plan",
    "vlaforge/python/vlaforge/codegen",
    "vlaforge/python/vlaforge/deployment",
    "vlaforge/runtime",
    "vlaforge/include/vlaforge",
)

_MODEL_SPECS: Mapping[str, Mapping[str, Any]] = {
    "Octo": {
        "factory": build_octo_like_fixture,
        "root_argument": "octo_root",
        "source_patterns": {
            "octo/model/octo_model.py": (
                "def sample_actions(",
            ),
            "octo/model/components/action_heads.py": (
                "class DiffusionActionHead",
                "diffusion_steps: int = 20",
                "jax.lax.scan(",
            ),
        },
    },
    "GR00T N1.7": {
        "factory": build_groot_n1_like_fixture,
        "root_argument": "groot_root",
        "source_patterns": {
            "gr00t/model/gr00t_n1d7/gr00t_n1d7.py": (
                "self.action_head = Gr00tN1d7ActionHead(config)",
                "for t in range(self.num_inference_timesteps):",
                "embodiment_id = action_input.embodiment_id",
                "self.action_encoder(actions, timesteps_tensor, embodiment_id)",
            ),
            "gr00t/model/modules/dit.py": (
                "self.timestep_encoder = TimestepEncoder(",
                "temb = self.timestep_encoder(timestep)",
            ),
            "gr00t/policy/gr00t_policy.py": (
                "class Gr00tPolicy(",
            ),
        },
    },
    "AutoVLA": {
        "factory": build_driving_ar_fixture,
        "root_argument": "autovla_root",
        "source_patterns": {
            "models/autovla.py": (
                "def predict(self, input_features):",
                "outputs = self.vlm.generate(",
                "actions_tokens = actions_tokens[:self.trajectory_sampling.num_poses]",
                "decode_token_ids_to_trajectory(actions_tokens)",
            ),
            "models/action_tokenizer.py": (
                "def decode_token_ids_to_trajectory(",
                "trajectory = torch.cat(",
            ),
        },
    },
}


def _run_git(
    repository: Path,
    arguments: Iterable[str],
    *,
    text: bool = True,
) -> str | bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=text,
    )
    return completed.stdout.strip() if text else completed.stdout


def _git_revision(repository: Path) -> str:
    return str(_run_git(repository, ("rev-parse", "HEAD")))


def _git_blob(repository: Path, revision: str, path: str) -> bytes:
    return bytes(
        _run_git(
            repository,
            ("show", f"{revision}:{path}"),
            text=False,
        )
    )


def _literal_matches(
    source: str, patterns: Iterable[str]
) -> tuple[dict[str, int], tuple[str, ...]]:
    lines = source.splitlines()
    matches: dict[str, int] = {}
    missing: list[str] = []
    for pattern in patterns:
        line = next(
            (
                index
                for index, value in enumerate(lines, start=1)
                if pattern in value
            ),
            None,
        )
        if line is None:
            missing.append(pattern)
        else:
            matches[pattern] = line
    return matches, tuple(missing)


def _source_audit(
    name: str,
    repository: Path,
    source_patterns: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    contract = model_contract(name)
    revision = _git_revision(repository)
    if revision != contract.revision:
        raise ValueError(
            f"{name}: expected upstream {contract.revision}, got {revision}"
        )
    files = []
    for relative_path, patterns in source_patterns.items():
        payload = _git_blob(repository, revision, relative_path)
        source = payload.decode("utf-8")
        matches, missing = _literal_matches(source, patterns)
        if missing:
            raise ValueError(
                f"{name}: source contract mismatch in {relative_path}: "
                f"{missing}"
            )
        blob_id = str(
            _run_git(
                repository,
                ("rev-parse", f"{revision}:{relative_path}"),
            )
        )
        files.append(
            {
                "relative_path": relative_path,
                "git_blob": blob_id,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "matched_patterns": [
                    {"literal": pattern, "line": matches[pattern]}
                    for pattern in patterns
                ],
            }
        )
    return {
        "repository": contract.repository,
        "revision": revision,
        "license": contract.license,
        "checkpoint": contract.checkpoint,
        "source_entries": list(contract.source_entries),
        "files": files,
        "passed": True,
        "evidence_level": "L0-pinned-source-audit",
    }


def _fingerprint_paths(
    repository: Path,
    revision: str,
) -> dict[str, Any]:
    objects = {
        path: str(
            _run_git(repository, ("rev-parse", f"{revision}:{path}"))
        )
        for path in FROZEN_CORE_PATHS
    }
    canonical = json.dumps(
        objects,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "revision": str(
            _run_git(repository, ("rev-parse", revision))
        ),
        "objects": objects,
        "combined_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _working_tree_status(
    repository: Path,
    paths: Iterable[str],
) -> list[str]:
    output = str(
        _run_git(
            repository,
            (
                "status",
                "--short",
                "--untracked-files=all",
                "--",
                *paths,
            ),
        )
    )
    return output.splitlines() if output else []


def audit_frozen_core(
    repository: Path,
    freeze_revision: str,
) -> dict[str, Any]:
    frozen = _fingerprint_paths(repository, freeze_revision)
    current = _fingerprint_paths(repository, "HEAD")
    worktree_status = _working_tree_status(repository, FROZEN_CORE_PATHS)
    matches = (
        frozen["objects"] == current["objects"] and not worktree_status
    )
    if not matches:
        raise ValueError(
            "frozen VLAForge core changed after freeze revision: "
            f"{worktree_status or 'committed core object mismatch'}"
        )
    return {
        "paths": list(FROZEN_CORE_PATHS),
        "frozen": frozen,
        "current": current,
        "worktree_status": worktree_status,
        "matches": True,
    }


def _walk_operations(block: Block) -> Iterable[Operation]:
    for operation in block.operations:
        yield operation
        for nested in operation.regions:
            yield from _walk_operations(nested)


def _port_record(port: Any) -> dict[str, Any]:
    record = {
        "id": getattr(port, "input_id", getattr(port, "output_id", None)),
        "name": port.name,
        "type": port.payload.to_dict(),
        "device": port.device,
        "alignment": port.alignment,
    }
    if hasattr(port, "required"):
        record.update(
            {
                "required": port.required,
                "default": normalize_value(port.default),
                "extension": port.extension,
                "value_range": normalize_value(port.value_range),
                "valid_for": port.valid_for,
            }
        )
    else:
        record["group"] = port.group
    return record


def _state_record(state: Any) -> dict[str, Any]:
    return {
        "name": state.name,
        "type": state.payload.to_dict(),
        "retention": state.retention,
        "reset_on_episode": state.reset_on_episode,
        "ownership": state.ownership.value,
    }


def _output_digest(values: list[Any]) -> str:
    payload = json.dumps(
        normalize_value(values),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_fixture(
    name: str,
    factory: Callable[[], AdapterFixture],
) -> dict[str, Any]:
    fixture = factory()
    diagnostics = verify(fixture.module, raise_on_error=False)
    if diagnostics:
        raise ValueError(f"{name}: Semantic IR diagnostics: {diagnostics}")
    invocation = fixture.module.invocations[0]
    if invocation.metadata.get("core_op_delta") != 0:
        raise ValueError(f"{name}: fixture must declare core_op_delta=0")

    operations = tuple(_walk_operations(invocation.body))
    opcodes = tuple(sorted({operation.opcode for operation in operations}))
    unknown = tuple(sorted(set(opcodes) - ALLOWED_OPS))
    if unknown:
        raise ValueError(f"{name}: unknown core operations: {unknown}")

    compilation = compile_module(
        fixture.module,
        profile=CompilerProfile.VERIFIED,
    )
    semantic = Interpreter(
        fixture.module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    planned = PlanExecutor(
        compilation.plan,
        compilation.module,
        regions=fixture.regions,
        validators=fixture.validators,
        initial_state=fixture.initial_state,
    )
    outputs: list[Any] = []
    for index, run in enumerate(fixture.runs):
        expected = semantic.run(inputs=run.inputs)
        actual = planned.run(inputs=run.inputs)
        if expected.committed_outputs != actual.committed_outputs:
            raise ValueError(f"{name}: output mismatch at Run {index}")
        if expected.state != actual.state:
            raise ValueError(f"{name}: state mismatch at Run {index}")
        outputs.append(expected.committed_outputs)
    trace_equal = semantic.trace.to_data() == planned.trace.to_data()
    if not trace_equal:
        raise ValueError(f"{name}: Semantic/Plan trace mismatch")

    cache_events = [
        event
        for event in semantic.trace.events
        if event.kind == "cache"
    ]
    source_lines, source_line = inspect.getsourcelines(factory)
    source_file = Path(inspect.getsourcefile(factory) or "").resolve()
    return {
        "module": fixture.module.name,
        "adapter_factory": factory.__name__,
        "adapter_source": source_file.relative_to(_REPOSITORY_ROOT).as_posix(),
        "adapter_start_line": source_line,
        "adapter_loc": len(source_lines),
        "adapter_template": invocation.metadata["adapter_template"],
        "source_contract": invocation.metadata["source_contract"],
        "evidence_kind": fixture.evidence_kind,
        "evidence_level": "L1-deterministic-executable-fixture",
        "core_op_delta": 0,
        "opcodes": list(opcodes),
        "unknown_opcodes": [],
        "inputs": [_port_record(port) for port in fixture.module.inputs],
        "outputs": [_port_record(port) for port in fixture.module.outputs],
        "states": [_state_record(state) for state in fixture.module.states],
        "regions": [
            {
                "name": region.name,
                "inputs": [
                    {"name": value.name, "type": value.type.to_dict()}
                    for value in region.inputs
                ],
                "outputs": [value.to_dict() for value in region.outputs],
                "memoize": bool(region.metadata.get("memoize", False)),
                "metadata": normalize_value(dict(region.metadata)),
            }
            for region in fixture.module.regions
        ],
        "bounded_for_count": sum(
            operation.opcode == "vla.for" for operation in operations
        ),
        "structured_if_count": sum(
            operation.opcode == "vla.if" for operation in operations
        ),
        "run_count": len(fixture.runs),
        "semantic_plan_output_state_parity": True,
        "semantic_plan_trace_parity": True,
        "output_sha256": _output_digest(outputs),
        "cache_events": {
            "total": len(cache_events),
            "hits": sum(bool(event.data["hit"]) for event in cache_events),
            "misses": sum(
                not bool(event.data["hit"]) for event in cache_events
            ),
        },
        "compilation_certificate": compilation.certificate.to_dict(),
        "unsupported_evidence": [
            "real checkpoint frontend parity",
            "real compiled artifact parity",
            "real generated no-Python C++ Session parity",
        ],
        "passed": True,
    }


def _repository_state(repository: Path) -> dict[str, Any]:
    revision = str(_run_git(repository, ("rev-parse", "HEAD")))
    tracked = str(
        _run_git(
            repository,
            ("status", "--short", "--untracked-files=no"),
        )
    )
    return {
        "revision": revision,
        "source_dirty": bool(tracked),
        "tracked_status": tracked.splitlines() if tracked else [],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# VLAForge frozen-core held-out audit",
        "",
        (
            f"Status: **{report['status']}**. Frozen core match: "
            f"**{report['frozen_core']['matches']}**."
        ),
        "",
        "This report combines pinned source audit (L0) with deterministic "
        "executable fixtures (L1). It is not checkpoint, artifact, or real "
        "generated-C++ evidence.",
        "",
        "| Model | Source | Fixture | Template | Runs | Core op delta | "
        "Semantic/Plan parity |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for model in report["models"]:
        fixture = model["fixture"]
        lines.append(
            f"| {model['name']} | {model['source']['evidence_level']} | "
            f"{fixture['evidence_level']} | {fixture['adapter_template']} | "
            f"{fixture['run_count']} | {fixture['core_op_delta']} | "
            "outputs/state/trace exact |"
        )
    lines.extend(
        [
            "",
            "## Frozen core",
            "",
            f"- Freeze revision: `{report['frozen_core']['frozen']['revision']}`",
            f"- Current revision: `{report['frozen_core']['current']['revision']}`",
            (
                "- Combined fingerprint: "
                f"`{report['frozen_core']['frozen']['combined_sha256']}`"
            ),
            "- Core working-tree changes: none",
            "",
            "## Per-model evidence",
            "",
        ]
    )
    for model in report["models"]:
        fixture = model["fixture"]
        certificate = fixture["compilation_certificate"]
        lines.extend(
            [
                f"### {model['name']}",
                "",
                f"- Upstream revision: `{model['source']['revision']}`",
                (
                    f"- Adapter: `{fixture['adapter_factory']}` "
                    f"({fixture['adapter_loc']} LOC)"
                ),
                f"- Generic opcodes: `{', '.join(fixture['opcodes'])}`",
                (
                    "- Control: "
                    f"{fixture['bounded_for_count']} bounded for, "
                    f"{fixture['structured_if_count']} structured if"
                ),
                (
                    "- Exact cache events: "
                    f"{fixture['cache_events']['hits']} hit / "
                    f"{fixture['cache_events']['misses']} miss"
                ),
                (
                    "- Static arena: "
                    f"{certificate['arena']['compiled_bytes']} bytes "
                    f"(saved {certificate['arena']['saved_bytes']} bytes)"
                ),
                f"- Output digest: `{fixture['output_sha256']}`",
                "- Unsupported: real checkpoint L2, artifact L3, real C++ L4",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-revision", required=True)
    parser.add_argument("--octo-root", type=Path, required=True)
    parser.add_argument("--groot-root", type=Path, required=True)
    parser.add_argument("--autovla-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    models = []
    for name, spec in _MODEL_SPECS.items():
        repository = getattr(args, spec["root_argument"]).resolve()
        source = _source_audit(
            name,
            repository,
            spec["source_patterns"],
        )
        fixture = audit_fixture(name, spec["factory"])
        models.append(
            {
                "name": name,
                "source": source,
                "fixture": fixture,
                "evidence_boundary": (
                    "pinned real source L0 + deterministic executable L1; "
                    "no checkpoint/artifact/generated-C++ claim"
                ),
            }
        )

    report = {
        "schema": REPORT_SCHEMA,
        "status": "passed",
        "passed": True,
        "repository": _repository_state(_REPOSITORY_ROOT),
        "frozen_core": audit_frozen_core(
            _REPOSITORY_ROOT,
            args.freeze_revision,
        ),
        "models": models,
        "summary": {
            "heldout_models": len(models),
            "robot_models": 2,
            "driving_models": 1,
            "all_source_contracts_passed": True,
            "all_fixtures_passed": True,
            "all_semantic_plan_parity": True,
            "core_op_delta": 0,
            "frozen_core_unchanged": True,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
