#!/usr/bin/env python3
"""Run a declarative fixed-shape Horizon model chain through file bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def remote_join(root: str, relative: str) -> str:
    return str(Path(root) / relative)


def run(command: list[str], *, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=text,
    )


def materialize_constant(
    path: Path,
    *,
    dtype: str,
    shape: list[int],
    values: list[Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values, dtype=np.dtype(dtype)).reshape(shape)
    array.tofile(path)


def initialize_variables(
    root: Path,
    declarations: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    variables: dict[str, dict[str, Any]] = {}
    for name, declaration in declarations.items():
        kind = declaration.get("kind")
        if kind == "file":
            path = Path(declaration["path"]).resolve(strict=True)
        elif kind == "constant":
            path = root / "constants" / f"{name}.bin"
            materialize_constant(
                path,
                dtype=declaration["dtype"],
                shape=declaration["shape"],
                values=declaration["values"],
            )
        else:
            raise ValueError(f"unsupported variable kind for {name}: {kind}")
        variables[name] = {
            "kind": kind,
            "path": str(path),
            "sha256": digest(path),
            "size_bytes": path.stat().st_size,
        }
    return variables


def parse_outputs(directory: Path, expected: int) -> list[Path]:
    pattern = re.compile(r"model_infer_output_(\d+)_")
    indexed: list[tuple[int, Path]] = []
    for path in directory.glob("*.bin"):
        match = pattern.search(path.name)
        if match:
            indexed.append((int(match.group(1)), path))
    indexed.sort()
    if len(indexed) != expected:
        raise ValueError(
            f"expected {expected} output dumps in {directory}, found {len(indexed)}"
        )
    indices = [index for index, _ in indexed]
    if indices != list(range(expected)):
        raise ValueError(f"output dump indices are not contiguous: {indices}")
    return [path for _, path in indexed]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve(strict=True)
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "vlaforge.horizon_file_chain/1":
        raise ValueError("unexpected Horizon chain manifest schema")
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"chain output already exists: {output}")
    output.mkdir(parents=True)

    board = manifest["board"]
    host = board["host"]
    remote_root = board["remote_root"]
    executable = board["hrt_model_exec"]
    variables = initialize_variables(output, manifest["variables"])
    report: dict[str, Any] = {
        "schema": "vlaforge.horizon_file_chain_run/1",
        "status": "started",
        "manifest": str(manifest_path),
        "manifest_sha256": digest(manifest_path),
        "board": board,
        "variables": variables,
        "stages": [],
        "started_ns": time.time_ns(),
    }
    write_json(output / "chain-report.json", report)

    try:
        run(["ssh", host, f"mkdir -p {shlex.quote(remote_root)}"])
        for stage in manifest["stages"]:
            stage_name = stage["name"]
            hbm = Path(stage["hbm"]).resolve(strict=True)
            input_names = list(stage["inputs"])
            output_names = list(stage["outputs"])
            repeat = int(stage.get("repeat", 1))
            update = dict(stage.get("update", {}))
            missing = set(input_names) - set(variables)
            if missing:
                raise ValueError(
                    f"{stage_name} inputs are undefined: {sorted(missing)}"
                )
            remote_hbm = remote_join(remote_root, f"models/{hbm.name}")
            remote_hbm_parent = str(Path(remote_hbm).parent)
            run(["ssh", host, f"mkdir -p {shlex.quote(remote_hbm_parent)}"])
            run(["scp", "-q", str(hbm), f"{host}:{remote_hbm}"])

            for iteration in range(repeat):
                iteration_name = (
                    stage_name if repeat == 1 else f"{stage_name}-{iteration:02d}"
                )
                run_root = output / "runs" / iteration_name
                input_root = run_root / "inputs"
                input_root.mkdir(parents=True, exist_ok=True)
                remote_input_root = remote_join(
                    remote_root, f"runs/{iteration_name}/inputs"
                )
                remote_dump_root = remote_join(
                    remote_root, f"runs/{iteration_name}/dumps"
                )
                run(["ssh", host, f"mkdir -p {shlex.quote(remote_input_root)}"])
                run(["ssh", host, f"mkdir -p {shlex.quote(remote_dump_root)}"])

                remote_inputs: list[str] = []
                local_inputs: dict[str, Any] = {}
                for index, name in enumerate(input_names):
                    source = Path(variables[name]["path"])
                    local_path = input_root / f"{index:02d}-{name}.bin"
                    run(["cp", str(source), str(local_path)])
                    remote_path = remote_join(
                        remote_root,
                        f"runs/{iteration_name}/inputs/{index:02d}-{name}.bin",
                    )
                    run(["scp", "-q", str(local_path), f"{host}:{remote_path}"])
                    remote_inputs.append(remote_path)
                    local_inputs[name] = {
                        "path": str(local_path),
                        "sha256": digest(local_path),
                        "size_bytes": local_path.stat().st_size,
                    }

                command = [
                    executable,
                    "infer",
                    "--model_file",
                    remote_hbm,
                    "--input_file",
                    ",".join(remote_inputs),
                    "--frame_count",
                    "1",
                    "--enable_dump",
                    "true",
                    "--dump_path",
                    remote_dump_root,
                    "--dump_format",
                    "bin",
                    "--remove_padding_process",
                    "true",
                ]
                command_text = (
                    f"cd {shlex.quote(remote_root)} && {shlex.join(command)}"
                )
                started_ns = time.time_ns()
                result = run(["ssh", host, command_text])
                elapsed_ns = time.time_ns() - started_ns
                (run_root / "stdout.log").write_text(result.stdout)
                (run_root / "stderr.log").write_text(result.stderr)

                local_dump_root = run_root / Path(remote_dump_root).name
                run(
                    [
                        "scp",
                        "-q",
                        "-r",
                        f"{host}:{remote_dump_root}",
                        str(run_root),
                    ]
                )
                output_paths = parse_outputs(
                    local_dump_root, expected=len(output_names)
                )
                outputs: dict[str, Any] = {}
                for name, path in zip(output_names, output_paths, strict=True):
                    stable = run_root / f"{name}.bin"
                    run(["cp", str(path), str(stable)])
                    variables[name] = {
                        "kind": "generated",
                        "path": str(stable),
                        "sha256": digest(stable),
                        "size_bytes": stable.stat().st_size,
                        "source_dump": str(path),
                    }
                    outputs[name] = variables[name]
                for source_name, target_name in update.items():
                    if source_name not in variables:
                        raise ValueError(
                            f"{stage_name} update source is undefined: {source_name}"
                        )
                    if target_name not in variables:
                        raise ValueError(
                            f"{stage_name} update target is undefined: {target_name}"
                        )
                    variables[target_name] = dict(variables[source_name])
                    variables[target_name]["updated_from"] = source_name
                report["stages"].append(
                    {
                        "name": stage_name,
                        "iteration": iteration,
                        "hbm": str(hbm),
                        "hbm_sha256": digest(hbm),
                        "command": command_text,
                        "inputs": local_inputs,
                        "outputs": outputs,
                        "updates": update,
                        "elapsed_ns": elapsed_ns,
                    }
                )
                report["variables"] = variables
                write_json(output / "chain-report.json", report)
        report.update(
            {
                "status": "passed",
                "elapsed_ns": time.time_ns() - report["started_ns"],
            }
        )
    except Exception as error:
        report.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "message": str(error),
                "elapsed_ns": time.time_ns() - report["started_ns"],
            }
        )
        write_json(output / "chain-report.json", report)
        raise
    write_json(output / "chain-report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
