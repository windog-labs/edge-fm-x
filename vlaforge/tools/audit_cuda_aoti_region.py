#!/usr/bin/env python3
"""Compile and audit one real CUDA AOTInductor RegionExecutable artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import torch

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_INSTALLED_PACKAGE_MODE = (
    os.environ.get("VLAFORGE_AUDIT_INSTALLED_PACKAGE") == "1"
)
if not _INSTALLED_PACKAGE_MODE:
    sys.path.insert(0, str(_SOURCE_ROOT / "python"))

import vlaforge  # noqa: E402
from vlaforge.codegen import CppValidatorDefinition  # noqa: E402
from vlaforge.deployment import (  # noqa: E402
    ArtifactIdentity,
    ArtifactKind,
    ArtifactResidency,
    BackendCapability,
    EffectAudit,
    RegionArtifactContract,
    ValueContract,
    WorkspaceContract,
    build_artifact_compile_bundle,
)
from vlaforge.frontend.builder import ModuleBuilder  # noqa: E402
from vlaforge.ir import ops  # noqa: E402
from vlaforge.ir.program import (  # noqa: E402
    Block,
    InputPort,
    Invocation,
    OutputPort,
    TensorRegion,
    Value,
)
from vlaforge.ir.types import (  # noqa: E402
    PendingOutputType,
    TensorType,
)
from vlaforge.ir.serializer import io_schema_digest  # noqa: E402


class AuditTensorRegion(torch.nn.Module):
    """Small real tensor program used to exercise the production AOTI ABI."""

    def forward(
        self, values: torch.Tensor, gain: torch.Tensor
    ) -> torch.Tensor:
        return (torch.sin(values) + values.square()) * gain


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        **kwargs,
    )


def _runtime_target(device: str) -> tuple[str, str | None]:
    if device == "cpu":
        return "cpu", None
    major, minor = torch.cuda.get_device_capability(torch.device(device))
    return f"sm_{major}{minor}", f"{major}.{minor}"


def _parse_output(text: str) -> list[float]:
    line = next(
        (item for item in text.splitlines() if item.startswith("OUTPUT,")),
        None,
    )
    if line is None:
        raise RuntimeError(f"C++ runner did not emit OUTPUT: {text}")
    return [float(item) for item in line.split(",")[1:]]


def _semantic_module():
    matrix = TensorType((4, 4), "f32")
    gain = TensorType((), "f32")
    pending = PendingOutputType("result", matrix)
    builder = ModuleBuilder("aoti_generated_session_audit")
    builder.add_input(
        InputPort("values", matrix, device="cuda:0", alignment=4)
    )
    builder.add_input(
        InputPort("gain", gain, device="cuda:0", alignment=4)
    )
    builder.add_output(
        OutputPort(
            "result", matrix, group="audit", device="cpu", alignment=4
        )
    )
    builder.add_region(
        TensorRegion(
            "audit_tensor_region",
            (Value("values_arg", matrix), Value("gain_arg", gain)),
            (matrix,),
        )
    )
    builder.add_invocation(
        Invocation(
            "act",
            Block.of(
                (
                    ops.input_read(
                        "values_value", "values_revision", matrix, "values"
                    ),
                    ops.input_read(
                        "gain_value", "gain_revision", gain, "gain"
                    ),
                    ops.transaction_begin("txn"),
                    ops.invoke(
                        ("result_value",),
                        (matrix,),
                        "audit_tensor_region",
                        ("values_value", "gain_value"),
                    ),
                    ops.validate(
                        "result_valid", "result_value", "finite_result"
                    ),
                    ops.output_create(
                        "pending_result", "result_value", matrix, "result"
                    ),
                    ops.output_group(
                        "pending_outputs",
                        "audit",
                        (("pending_result", pending),),
                    ),
                    ops.transaction_commit(
                        "committed_outputs",
                        (pending,),
                        "audit",
                        "txn",
                        "pending_outputs",
                        "result_valid",
                    ),
                    ops.return_values("committed_outputs"),
                )
            ),
        )
    )
    return builder.build()


def _generated_runner_source() -> str:
    return r"""
#include "session_generated.h"

#include <ATen/ATen.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>

namespace {

bool Check(VLAForgeStatus status, const char* operation) {
  if (status.code == VLAFORGE_STATUS_OK) {
    return true;
  }
  std::fprintf(stderr, "%s failed: %.*s\n", operation,
               static_cast<int>(status.message_size), status.message);
  return false;
}

VLAForgeBoundTensor Bound(at::Tensor& tensor,
                         const std::int64_t* dimensions,
                         std::uint32_t rank,
                         VLAForgeDeviceKind kind) {
  return VLAForgeBoundTensor{
      sizeof(VLAForgeBoundTensor),
      {tensor.data_ptr(), static_cast<std::uint64_t>(tensor.nbytes()),
       dimensions, rank, VLAFORGE_DTYPE_F32,
       {kind, tensor.is_cuda() ? tensor.get_device() : 0}},
      VLAFORGE_LAYOUT_CONTIGUOUS,
      4u};
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::fprintf(stderr, "usage: %s BUNDLE_ROOT [negative-mode]\n", argv[0]);
    return 2;
  }
  const std::string mode = argc == 3 ? argv[2] : "normal";
  at::Tensor values =
      at::arange(16, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA))
          .reshape({4, 4})
          .div(8.0);
  at::Tensor gain =
      at::full({}, 0.75,
               at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  constexpr std::array<std::int64_t, 2> kShape = {4, 4};
  auto values_bound =
      Bound(values, kShape.data(), 2u, VLAFORGE_DEVICE_CUDA);
  auto gain_bound = Bound(gain, nullptr, 0u, VLAFORGE_DEVICE_CUDA);
  constexpr std::array<std::int64_t, 2> kWrongShape = {2, 8};
  if (mode == "wrong-shape") {
    values_bound.tensor.dimensions = kWrongShape.data();
  } else if (mode == "wrong-dtype") {
    values_bound.tensor.dtype = VLAFORGE_DTYPE_F64;
  } else if (mode == "wrong-device") {
    values_bound.tensor.device = {VLAFORGE_DEVICE_CPU, 0};
  } else if (mode == "wrong-layout") {
    values_bound.layout = VLAFORGE_LAYOUT_NHWC;
  } else if (mode != "normal" &&
             mode != "schema-mismatch" &&
             mode != "abi-mismatch") {
    std::fprintf(stderr, "unknown negative mode\n");
    return 2;
  }
  const VLAForgeInputStamp stamp{
      sizeof(VLAForgeInputStamp), 1u, 0u, {}, 7u, 0u};

  VLAForgeSession* session = nullptr;
  const std::string bundle_root(argv[1]);
  if (!Check(vlaforge_model_session_create_from_bundle(
                 bundle_root.data(), bundle_root.size(), &session),
             "create from bundle")) {
    return 3;
  }
  const auto* api = vlaforge_model_session_api();
  VLAForgeSessionApi invalid_api = *api;
  if (mode == "abi-mismatch") {
    invalid_api.abi_version += 1u;
    api = &invalid_api;
  }
  std::array<char, VLAFORGE_SCHEMA_DIGEST_HEX_SIZE> wrong_digest{};
  wrong_digest.fill('0');
  const char* expected_digest =
      mode == "schema-mismatch"
          ? wrong_digest.data()
          : vlaforge_generated::kSchemaDigest;
  const bool passed =
      Check(vlaforge_session_api_validate(
                api, expected_digest,
                VLAFORGE_SCHEMA_DIGEST_HEX_SIZE),
            "validate session API") &&
      Check(api->bind_tensor(session, 0u, &values_bound, &stamp),
            "bind values") &&
      Check(api->bind_tensor(session, 1u, &gain_bound, &stamp),
            "bind gain") &&
      Check(api->run(session), "run");
  VLAForgeBoundTensor result{};
  const bool read =
      passed && Check(api->read_output_tensor(session, 0u, &result),
                      "read result");
  if (read) {
    const auto* output = static_cast<const float*>(result.tensor.data);
    std::printf("OUTPUT");
    for (std::size_t index = 0; index < 16u; ++index) {
      std::printf(",%.9g", static_cast<double>(output[index]));
    }
    std::printf("\n");
  }
  api->destroy(session);
  return read ? 0 : 4;
}
"""


def _audit_generated_session(
    root: Path,
    package_path: Path,
    package_sha256: str,
    graph_sha256: str,
    expected: list[float],
    residency: ArtifactResidency,
    runtime_root: Path,
    source_revision: str,
    target: str,
    torch_cuda_arch_list: str,
) -> dict[str, object]:
    module = _semantic_module()
    artifact_relative = "artifacts/audit_tensor_region.pt2"
    bundle_root = root / f"compile-bundle-{residency.value}"
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    matrix = TensorType((4, 4), "f32")
    gain = TensorType((), "f32")
    artifact = RegionArtifactContract(
        region_id=0,
        region_name="audit_tensor_region",
        inputs=(
            ValueContract.from_ir("values_arg", matrix, device="cuda:0"),
            ValueContract.from_ir("gain_arg", gain, device="cuda:0"),
        ),
        outputs=(
            ValueContract.from_ir("output_0", matrix, device="cpu"),
        ),
        io_schema_digest=io_schema_digest(module),
        identity=ArtifactIdentity(
            model_name="AuditTensorRegion",
            upstream_revision="local:audit-v1",
            checkpoint_identity="none:synthetic-deterministic",
            graph_sha256=graph_sha256,
        ),
        artifact_kind=ArtifactKind.AOTI_PACKAGE,
        artifact_path=artifact_relative,
        artifact_sha256=package_sha256,
        artifact_size_bytes=package_path.stat().st_size,
        workspace=WorkspaceContract(),
        capability=BackendCapability(
            backend="aoti",
            target=target,
            supported_dtypes=("f32",),
            supports_dynamic_shapes=False,
            supports_device_resident_io=True,
            requires_synchronize=True,
        ),
        effect_audit=EffectAudit(),
        backend_variant=f"torch-{torch.__version__}",
        residency=residency,
    )
    manifest = build_artifact_compile_bundle(
        module,
        bundle_root,
        region_artifacts={"audit_tensor_region": artifact},
        artifact_sources={"audit_tensor_region": package_path},
        validators={
            "finite_result": CppValidatorDefinition(
                "finite_result",
                """if (data == nullptr || size_bytes != 16u * sizeof(float)) {
  return false;
}
const auto* values = static_cast<const float*>(data);
for (std::size_t index = 0; index < 16u; ++index) {
  if (!std::isfinite(values[index])) {
    return false;
  }
}
return true;""",
            )
        },
        runner_source=_generated_runner_source(),
        runtime_root=runtime_root,
        cmake_prefix_path=torch.utils.cmake_prefix_path,
        backend_versions={
            "aoti": f"torch-{torch.__version__}",
            "cuda": str(torch.version.cuda),
        },
        profile="off",
        source_revision=source_revision,
        source_dirty=False,
        environment={"TORCH_CUDA_ARCH_LIST": torch_cuda_arch_list},
    )
    runner = bundle_root / "bin" / "vlaforge_generated_runner"
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONHOME": "/definitely/not/a/python/home",
            "PYTHONPATH": "/definitely/not/a/python/path",
        }
    )
    completed = _run([str(runner), str(bundle_root)], env=environment)
    actual = _parse_output(completed.stdout)
    max_abs_error = max(
        (
            abs(left - right)
            for left, right in zip(actual, expected, strict=True)
        ),
        default=0.0,
    )
    if max_abs_error > 1e-6:
        raise RuntimeError(
            f"generated Session numeric mismatch: {max_abs_error}"
        )
    linked = _run(["ldd", str(runner)]).stdout
    if "libpython" in linked.lower():
        raise RuntimeError("generated Session unexpectedly links Python")
    negative_cases: dict[str, str] = {}
    expected_failures = {
        "wrong-shape": "contract mismatch",
        "wrong-dtype": "contract mismatch",
        "wrong-device": "contract mismatch",
        "wrong-layout": "contract mismatch",
        "schema-mismatch": "session schema digest mismatch",
        "abi-mismatch": "unsupported session ABI",
    }
    for mode, expected_message in expected_failures.items():
        failed = subprocess.run(
            [str(runner), str(bundle_root), mode],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if failed.returncode == 0 or expected_message not in failed.stderr:
            raise RuntimeError(
                f"generated Session did not reject {mode}: "
                f"{failed.stderr}"
            )
        negative_cases[mode] = "rejected"

    bundled_artifact = bundle_root / artifact_relative
    original_payload = bundled_artifact.read_bytes()
    missing_path = bundled_artifact.with_suffix(".missing")
    bundled_artifact.rename(missing_path)
    missing = subprocess.run(
        [str(runner), str(bundle_root)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    missing_path.rename(bundled_artifact)
    if missing.returncode == 0 or "does not exist" not in missing.stderr:
        raise RuntimeError("generated Session did not reject missing artifact")
    negative_cases["missing-artifact"] = "rejected"

    corrupted = bytearray(original_payload)
    corrupted[0] ^= 0xFF
    bundled_artifact.write_bytes(corrupted)
    corrupt = subprocess.run(
        [str(runner), str(bundle_root)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    bundled_artifact.write_bytes(original_payload)
    if corrupt.returncode == 0 or "SHA-256 mismatch" not in corrupt.stderr:
        raise RuntimeError("generated Session did not reject corrupt artifact")
    negative_cases["corrupt-artifact"] = "rejected"
    manifest.verify_files(bundle_root)
    return {
        "status": "passed",
        "artifact_residency": residency.value,
        "bundle_manifest": str(bundle_root / "bundle.json"),
        "bundle_digest": manifest.digest(),
        "runner": str(runner),
        "max_abs_error": max_abs_error,
        "output_elements": len(actual),
        "python_linked": False,
        "invalid_python_environment_run": True,
        "bundle_verified": True,
        "schema_validated": True,
        "abi_validated": True,
        "io_schema_digest": manifest.io_schema_digest,
        "artifact_target": artifact.capability.target,
        "artifact_sha256": artifact.artifact_sha256,
        "backend_variant": artifact.backend_variant,
        "source_revision": manifest.reproducibility.source_revision,
        "negative_cases": negative_cases,
    }


def _audit_direct_backend(
    *,
    root: Path,
    source_root: Path,
    build_dir: Path,
    package_path: Path,
    device: str,
    expected_target: str,
    expected: list[float],
) -> dict[str, object]:
    configure = _run(
        [
            "cmake",
            "-S",
            str(source_root),
            "-B",
            str(build_dir),
            "-DVLAFORGE_BUILD_AOTI_BACKEND=ON",
            "-DBUILD_TESTING=ON",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
        ]
    )
    build = _run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "vlaforge_aoti_region_smoke",
            "--parallel",
            "4",
        ]
    )
    runner = build_dir / "tests" / "cpp" / "vlaforge_aoti_region_smoke"
    clean_environment = dict(os.environ)
    clean_environment.update(
        {
            "PYTHONHOME": "/definitely/not/a/python/home",
            "PYTHONPATH": "/definitely/not/a/python/path",
        }
    )
    run_start = time.perf_counter()
    completed = _run(
        [str(runner), str(package_path), device, expected_target, "normal"],
        env=clean_environment,
    )
    run_seconds = time.perf_counter() - run_start
    actual = _parse_output(completed.stdout)
    if len(actual) != len(expected):
        raise RuntimeError(
            f"output length mismatch: {len(actual)} != {len(expected)}"
        )
    max_abs_error = max(
        (
            abs(left - right)
            for left, right in zip(actual, expected, strict=True)
        ),
        default=0.0,
    )
    if max_abs_error > 1e-6:
        raise RuntimeError(f"AOTI numeric mismatch: {max_abs_error}")

    linked = _run(["ldd", str(runner)]).stdout
    linked_libraries = [
        line.strip().split()[0].lower()
        for line in linked.splitlines()
        if line.strip()
    ]
    if any(name.startswith("libpython") for name in linked_libraries):
        raise RuntimeError("C++ AOTI runner unexpectedly links Python")
    negative_cases = {}
    wrong_target = subprocess.run(
        [str(runner), str(package_path), device, "sm_00", "normal"],
        check=False,
        capture_output=True,
        text=True,
        env=clean_environment,
    )
    if (
        wrong_target.returncode == 0
        or "target mismatch" not in wrong_target.stderr
    ):
        raise RuntimeError("AOTI backend did not reject wrong target")
    negative_cases["wrong-target"] = "rejected"

    invalid_package = root / "artifacts" / "invalid.pt2"
    invalid_package.write_bytes(b"not an AOTI package")
    load_failure = subprocess.run(
        [str(runner), str(invalid_package), device, expected_target, "normal"],
        check=False,
        capture_output=True,
        text=True,
        env=clean_environment,
    )
    if load_failure.returncode == 0 or "load failed" not in load_failure.stderr:
        raise RuntimeError("AOTI backend did not report load failure")
    negative_cases["load-failure"] = "rejected"

    for mode, expected_message in (
        ("missing-input-binding", "run failed"),
        ("wrong-output-shape", "output metadata mismatch"),
    ):
        failed = subprocess.run(
            [
                str(runner),
                str(package_path),
                device,
                expected_target,
                mode,
            ],
            check=False,
            capture_output=True,
            text=True,
            env=clean_environment,
        )
        if failed.returncode == 0 or expected_message not in failed.stderr:
            raise RuntimeError(
                f"AOTI backend did not reject {mode}: {failed.stderr}"
            )
        negative_cases[mode] = "rejected"
    return {
        "status": "passed",
        "run_seconds": run_seconds,
        "max_abs_error": max_abs_error,
        "output_elements": len(actual),
        "cpp_runner": str(runner),
        "python_linked": False,
        "invalid_python_environment_run": True,
        "negative_cases": negative_cases,
        "configure_tail": configure.stdout.splitlines()[-5:],
        "build_tail": build.stdout.splitlines()[-5:],
    }


def _audit(
    root: Path,
    device: str = "cuda",
    *,
    runtime_root: Path = _SOURCE_ROOT,
    source_revision: str = "local:audit-v1",
) -> dict[str, object]:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    expected_target, torch_cuda_arch_list = _runtime_target(device)
    if torch_cuda_arch_list is not None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = torch_cuda_arch_list

    source_root = runtime_root
    package_path = root / "artifacts" / "audit_tensor_region.pt2"
    package_path.parent.mkdir(parents=True, exist_ok=True)
    build_dir = root / "build"

    model = AuditTensorRegion().eval().to(device)
    values = (
        torch.arange(16, dtype=torch.float32, device=device)
        .reshape(4, 4)
        .div(8.0)
    )
    gain = torch.tensor(0.75, dtype=torch.float32, device=device)
    expected = model(values, gain).detach().cpu().flatten().tolist()

    compile_start = time.perf_counter()
    exported = torch.export.export(model, (values, gain), strict=True)
    graph_sha256 = hashlib.sha256(
        exported.graph_module.code.encode("utf-8")
    ).hexdigest()
    actual_package = Path(
        torch._inductor.aoti_compile_and_package(
            exported, package_path=str(package_path)
        )
    )
    compile_seconds = time.perf_counter() - compile_start
    if actual_package != package_path or not package_path.is_file():
        raise RuntimeError(
            f"AOTI package path mismatch: {actual_package} != {package_path}"
        )

    direct_backend = (
        _audit_direct_backend(
            root=root,
            source_root=source_root,
            build_dir=build_dir,
            package_path=package_path,
            device=device,
            expected_target=expected_target,
            expected=expected,
        )
        if (source_root / "tests/cpp/aoti_region_smoke.cpp").is_file()
        else {
            "status": "not_run",
            "reason": (
                "installed runtime distribution intentionally excludes "
                "test-only C++ sources; generated Session exercises the "
                "production backend"
            ),
            "negative_cases": {},
        }
    )
    package_sha256 = _sha256(package_path)
    generated_session = _audit_generated_session(
        root,
        package_path,
        package_sha256,
        graph_sha256,
        expected,
        ArtifactResidency.SESSION,
        runtime_root,
        source_revision,
        expected_target,
        torch_cuda_arch_list or "",
    )
    invocation_resident_generated_session = _audit_generated_session(
        root,
        package_path,
        package_sha256,
        graph_sha256,
        expected,
        ArtifactResidency.INVOCATION,
        runtime_root,
        source_revision,
        expected_target,
        torch_cuda_arch_list or "",
    )

    return {
        "schema": "vlaforge.aoti_audit/1",
        "status": "passed",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "target": expected_target,
        "torch_cuda_arch_list": torch_cuda_arch_list,
        "device": (
            torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
        ),
        "package_import": str(Path(vlaforge.__file__).resolve()),
        "installed_package_mode": _INSTALLED_PACKAGE_MODE,
        "runtime_root": str(runtime_root.resolve()),
        "package": {
            "path": str(package_path),
            "sha256": package_sha256,
            "size_bytes": package_path.stat().st_size,
        },
        "compile_seconds": compile_seconds,
        "direct_backend_smoke": direct_backend,
        "run_seconds": direct_backend.get("run_seconds"),
        "max_abs_error": direct_backend.get("max_abs_error"),
        "output_elements": direct_backend.get("output_elements"),
        "cpp_runner": direct_backend.get("cpp_runner"),
        "python_linked": direct_backend.get("python_linked"),
        "invalid_python_environment_run": direct_backend.get(
            "invalid_python_environment_run"
        ),
        "backend_negative_cases": direct_backend["negative_cases"],
        "generated_session": generated_session,
        "invocation_resident_generated_session": (
            invocation_resident_generated_session
        ),
        "configure_tail": direct_backend.get("configure_tail", []),
        "build_tail": direct_backend.get("build_tail", []),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work-dir",
        help="Persistent audit directory; a temporary directory is used by default.",
    )
    parser.add_argument("--report", help="Optional JSON report path.")
    parser.add_argument(
        "--device", choices=("cuda", "cpu"), default="cuda"
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=_SOURCE_ROOT,
        help="C++ runtime source root; defaults to the source checkout.",
    )
    parser.add_argument(
        "--source-revision",
        default="local:audit-v1",
        help="Immutable source revision recorded in generated bundles.",
    )
    args = parser.parse_args(argv)

    context = (
        nullcontext(Path(args.work_dir).resolve())
        if args.work_dir
        else tempfile.TemporaryDirectory(prefix="vlaforge-aoti-audit-")
    )
    with context as selected:
        root = (
            selected
            if isinstance(selected, Path)
            else Path(selected).resolve()
        )
        root.mkdir(parents=True, exist_ok=True)
        report = _audit(
            root,
            args.device,
            runtime_root=args.runtime_root.resolve(),
            source_revision=args.source_revision,
        )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
