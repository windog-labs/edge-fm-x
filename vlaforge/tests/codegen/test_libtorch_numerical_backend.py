"""Transport/code generation fixtures, not model or native numerical evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from vlaforge.codegen import CppArtifactRegionDefinition, generate_compiled_cpp_session
from vlaforge.compiler import NUMERICAL_COMPILATION_CERTIFICATE_SCHEMA, compile_module
from vlaforge.deployment.libtorch_numerical import NAMESPACE, REDUCTION_API
from vlaforge.deployment.numerical import (
    PROVIDER_REQUIRED,
    NumericalCompileRecord,
    NumericalContractError,
    NumericalPolicy,
    NumericalRequirement,
    RegionNumericalBinding,
)
from vlaforge.frontend import InvocationBuilder, tensor_region
from vlaforge.ir.program import InputPort, OutputPort, Value
from vlaforge.ir.types import TensorType


def _policy():
    boolean_names = (
        "autocast_cache_enabled",
        "autocast_cpu_enabled",
        "autocast_cuda_enabled",
        "cuda_matmul_allow_bf16_reduced_precision_reduction",
        "cuda_matmul_allow_bf16_reduced_precision_reduction_split_k",
        "cuda_matmul_allow_fp16_reduced_precision_reduction",
        "cuda_matmul_allow_fp16_reduced_precision_reduction_split_k",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "cudnn_enabled",
        "deterministic_algorithms_enabled",
        "deterministic_algorithms_warn_only",
        "sdpa_cudnn_enabled",
        "sdpa_flash_enabled",
        "sdpa_math_allow_fp16_bf16_reduction",
        "sdpa_math_enabled",
        "sdpa_mem_efficient_enabled",
    )
    return NumericalPolicy(
        NAMESPACE,
        tuple(
            sorted(
                {
                    **dict.fromkeys(boolean_names, False),
                    "autocast_cpu_dtype": "bfloat16",
                    "autocast_cuda_dtype": "float16",
                    "float32_matmul_precision": "highest",
                    "torch_release": "2.10.0",
                    "reduction_api": REDUCTION_API,
                }.items()
            )
        ),
    )


def _sources(
    backends=("aoti", "torchscript"),
    *,
    policy=None,
    legacy=False,
    artifact=None,
    runner=None,
    bootstrap=False,
):
    vector = TensorType((2,), "f32")

    @tensor_region("a_prefix", inputs=(Value("x", vector),), outputs=(vector,))
    def prefix(x):
        return x

    @tensor_region("b_step", inputs=(Value("x", vector),), outputs=(vector,))
    def step(x):
        return x

    builder = InvocationBuilder(
        "libtorch_policy_codegen_fixture",
        inputs=(
            InputPort("x", vector),
            InputPort("accepted", TensorType((1,), "bool")),
        ),
        outputs=(OutputPort("result", vector),),
    )
    (context,) = builder.call(prefix, builder.input("x"), cache=True)
    (result,) = builder.call(step, context)
    program = builder.finish({"result": result}, accepted=builder.input("accepted"))
    compiled = compile_module(program.module)
    policy = _policy() if policy is None else policy
    definitions = {}
    bindings = []
    digest = (
        "3" * 64
        if artifact is None
        else hashlib.sha256(artifact.read_bytes()).hexdigest()
    )
    size = 1 if artifact is None else artifact.stat().st_size
    for region, backend in zip(compiled.module.regions, backends, strict=True):
        record = NumericalCompileRecord(
            backend,
            "cpu",
            "synthetic-codegen-fixture/1",
            "1" * 64,
            "2" * 64,
            digest,
            size,
            policy,
            policy,
            policy,
            '{"fixture":true}',
        )
        binding = RegionNumericalBinding(
            region.name,
            NumericalRequirement(
                policy, "same-precision", policy.digest(), record.digest()
            ),
            record,
            PROVIDER_REQUIRED,
        )
        definitions[region.name] = CppArtifactRegionDefinition(
            region.name,
            backend,
            "fixture.pt2" if backend == "aoti" else "fixture.pt",
            digest,
            size,
            compiled.plan.io_schema_digest,
            "cpu",
            "cpu",
            "aoti/1" if backend == "aoti" else "torchscript-aten/1",
            numerical_binding=None if legacy else binding,
        )
        if not legacy:
            bindings.append(binding)
    if bindings:
        compiled = replace(
            compiled,
            certificate=replace(
                compiled.certificate,
                schema=NUMERICAL_COMPILATION_CERTIFICATE_SCHEMA,
                numerical_bindings=tuple(bindings),
            ),
        )
    if bootstrap:
        from vlaforge.codegen.numerical import generate_libtorch_worker_initializer

        initializer = generate_libtorch_worker_initializer(
            bindings,
            acknowledge_exclusive_process=True,
            acknowledge_calling_thread=True,
        )
        runner = "#define VLAFORGE_TEST_WORKER_BOOTSTRAP 1\n" + initializer + runner
    return generate_compiled_cpp_session(
        compiled,
        artifact_regions=definitions,
        validators=program.cpp_validators(),
        runner_source=runner,
    ).as_dict()


def _decode_literals(source):
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), source)


@pytest.mark.parametrize(
    "backends",
    [("aoti", "aoti"), ("torchscript", "torchscript"), ("aoti", "torchscript")],
)
def test_all_24_typed_fields_reach_each_builtin_backend(backends):
    code = _decode_literals(_sources(backends)["session_generated.cpp"])
    for name, value in _policy().values:
        kind = "BOOL" if type(value) is bool else "STRING"
        expected = f'{{"{name}", {len(name)}u, VLAFORGE_NUMERICAL_{kind}, '
        assert code.count(expected) == 2
    assert code.count(f'"{NAMESPACE}", {len(NAMESPACE)}u') == 2
    assert code.count("vlaforge_libtorch_numerical_provider_api()") == 2
    assert "vlaforge_external_region_plugin_numerical_provider" not in code
    assert "setFloat32MatmulPrecision" not in code
    assert "setAllowTF32" not in code
    assert "set_autocast" not in code


def test_mixed_libtorch_backend_lifetimes_share_provider_and_validate_boundaries():
    code = _sources()["session_generated.cpp"]
    initialize = code.split("ModelSession::InitializeRegions(", 1)[1]
    assert initialize.index("numerical_leases_.Add") < initialize.index("AcquireAll()")
    assert initialize.index("VLAFORGE_NUMERICAL_BEFORE_LOAD") < initialize.index(
        "LoadRegion(0u)"
    )
    assert initialize.index("LoadRegion(1u)") < initialize.index(
        "VLAFORGE_NUMERICAL_AFTER_LOAD"
    )
    run = code.split("ModelSession::Run() noexcept", 1)[1]
    assert run.index("VLAFORGE_NUMERICAL_RUN_ENTRY") < run.index("PrepareInputs()")
    assert run.index("VLAFORGE_NUMERICAL_BEFORE_COMMIT") < run.index(
        "state_store_.Commit"
    )
    destroy = code.split("void ModelSession::DestroyRegions() noexcept", 1)[1].split(
        "\n}", 1
    )[0]
    assert destroy.index("DrainNumerical()") < destroy.index("DestroyRegion(")
    assert destroy.index("DestroyRegion(") < destroy.index("numerical_leases_.Clear()")
    assert "numerical_leases_.Abandon()" in destroy


@pytest.mark.parametrize(
    "name,value",
    [
        ("cuda_matmul_allow_fp16_reduced_precision_reduction_split_k", None),
        ("cuda_matmul_allow_bf16_reduced_precision_reduction_split_k", 0),
        ("torch_release", "2.10.0+cu128"),
        ("reduction_api", "torch-2.7.1/bool-reduction"),
        ("float32_matmul_precision", "high"),
    ],
)
def test_codegen_cannot_upgrade_missing_split_k_or_accept_inconsistent_policy(
    name, value
):
    policy = _policy()
    fields = dict(policy.values)
    if value is None:
        del fields[name]
    else:
        fields[name] = value
    with pytest.raises(NumericalContractError):
        _sources(
            policy=NumericalPolicy(policy.namespace, tuple(sorted(fields.items())))
        )


def test_legacy_artifacts_remain_unbound_without_invented_observation():
    sources = _sources(legacy=True)
    code = sources["session_generated.cpp"]
    assert "vlaforge_libtorch_numerical_provider_api()" not in code
    assert "kNumericalRequirement" not in code
    assert "numerical_leases_" not in code


_NATIVE_RUNNER = r"""
#include "session_generated.h"
#include <ATen/Context.h>
#include <ATen/autocast_mode.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>

using vlaforge_generated::ModelSession;
unsigned commits = 0, aborts = 0;
bool inject = false;
void Trace(void*, const vlaforge::runtime::TraceEvent* event) {
  using Kind = vlaforge::runtime::TraceKind;
  if (event->kind == Kind::kOutputGroupCommit) ++commits;
  if (event->kind == Kind::kTransactionAbort) ++aborts;
  if (event->kind == Kind::kOutputGroupPending && inject) {
    at::globalContext().setBenchmarkCuDNN(!at::globalContext().benchmarkCuDNN());
    inject = false;
  }
}
bool Bind(ModelSession& session, unsigned revision, float* data) {
  static const std::int64_t shape[]{2}, flag_shape[]{1};
  static std::uint8_t accepted = 1;
  const VLAForgeBoundTensor input{sizeof(input),
      {data, 8, shape, 1, VLAFORGE_DTYPE_F32, {VLAFORGE_DEVICE_CPU, 0}}, VLAFORGE_LAYOUT_CONTIGUOUS, 4};
  const VLAForgeBoundTensor flag{sizeof(flag),
      {&accepted, 1, flag_shape, 1, VLAFORGE_DTYPE_BOOL, {VLAFORGE_DEVICE_CPU, 0}}, VLAFORGE_LAYOUT_CONTIGUOUS, 1};
  VLAForgeInputStamp stamp{};
  stamp.struct_size = sizeof(stamp);
  stamp.has_revision = 1;
  stamp.revision = revision;
  return session.BindTensor(0, input, &stamp).ok() && session.BindTensor(1, flag, &stamp).ok();
}
bool Output(ModelSession& session, float expected) {
  VLAForgeBoundTensor output{};
  if (!session.ReadOutputTensor(0, &output).ok()) return false;
  const auto* values = static_cast<const float*>(output.tensor.data);
  return values[0] == expected && values[1] == expected + 1;
}
int main(int argc, char** argv) {
  if (argc != 5) return 1;
  const std::string mode = argv[2];
#ifdef VLAFORGE_TEST_WORKER_BOOTSTRAP
  if (mode != "missing-bootstrap" &&
      vlaforge_initialize_numerical_worker().code != VLAFORGE_STATUS_OK) return 18;
#endif
  const bool legacy = std::string(argv[3]) == "legacy";
  const bool original_benchmark = at::globalContext().benchmarkCuDNN();
  if (mode == "initdrift") at::globalContext().setBenchmarkCuDNN(!original_benchmark);
  auto first = std::make_unique<ModelSession>(argv[1]);
  if (mode == "missing-bootstrap") {
    if (first->initialization_status().ok() ||
        at::globalContext().float32MatmulPrecision() != at::Float32MatmulPrecision::HIGHEST) return 19;
  } else if (mode == "initdrift") {
    const bool ok = first->initialization_status().ok();
    const bool no_setter = at::globalContext().benchmarkCuDNN() != original_benchmark;
    at::globalContext().setBenchmarkCuDNN(original_benchmark);
    if (ok != legacy || !no_setter) return 2;
  } else {
    if (!first->initialization_status().ok()) {
      std::cerr << first->initialization_status().message << '\n'; return 3;
    }
    first->SetTraceSink({nullptr, Trace});
    float data[]{1, 2};
    if (!Bind(*first, 1, data) || !first->Run().ok() || !Output(*first, 3) || commits != 1) return 4;
    if (mode == "multisession") {
      auto second = std::make_unique<ModelSession>(argv[1]);
      if (!second->initialization_status().ok() || !Bind(*second, 1, data) || !second->Run().ok() || !Output(*second, 3)) return 5;
      first.reset();
      data[0] = 5; data[1] = 6;
      if (!Bind(*second, 2, data) || !second->Run().ok() || !Output(*second, 7)) return 6;
      second.reset();
      auto reacquired = std::make_unique<ModelSession>(argv[1]);
      if (!reacquired->initialization_status().ok()) return 7;
    } else if (mode == "normal") {
      if (!Bind(*first, 1, data) || !first->Run().ok() || !Output(*first, 3) || commits != 2) return 8;
    } else {
      data[0] = 11; data[1] = 12;
      if (!Bind(*first, 2, data)) return 9;
      bool failed = false, retained_drift = false;
      if (mode == "threaddrift") {
        const bool main_autocast = at::autocast::is_autocast_enabled(at::kCPU);
        std::thread worker([&] {
          const bool own = at::autocast::is_autocast_enabled(at::kCPU);
          at::autocast::set_autocast_enabled(at::kCPU, !main_autocast);
          failed = !first->Run().ok();
          retained_drift = at::autocast::is_autocast_enabled(at::kCPU) != main_autocast;
          at::autocast::set_autocast_enabled(at::kCPU, own);
        });
        worker.join();
        if (at::autocast::is_autocast_enabled(at::kCPU) != main_autocast) return 10;
      } else {
        if (mode == "entrydrift") at::globalContext().setBenchmarkCuDNN(!original_benchmark);
        else if (mode == "exitdrift") inject = true;
        else return 11;
        failed = !first->Run().ok();
        retained_drift = at::globalContext().benchmarkCuDNN() != original_benchmark;
      }
      if (!retained_drift) return 12;
      at::globalContext().setBenchmarkCuDNN(original_benchmark);
      if (legacy) {
        if (failed || !Output(*first, 13) || commits != 2 || aborts != 0) return 13;
      } else {
        if (!failed || !Output(*first, 3) || commits != 1 || aborts != (mode == "exitdrift" ? 1u : 0u)) return 14;
        (void)Bind(*first, 3, data);
        if (first->Run().ok() || !Output(*first, 3)) return 15;
      }
    }
  }
  std::ifstream input("/proc/self/maps");
  const std::string maps((std::istreambuf_iterator<char>(input)), {});
  if (maps.empty() || maps.find("libpython") != std::string::npos) return 16;
  std::ofstream output(argv[4]); output << maps;
  if (!output.good()) return 17;
  std::cout << "{\"status\":\"passed\",\"evidence_level\":\"generated_cpu_fixture_session\",\"pid\":"
            << getpid() << ",\"mode\":\"" << mode << "\",\"model_output_validation\":false,\"libpython_mapped\":false}\n";
  return 0;
}
"""


@pytest.mark.skipif(
    os.environ.get("VLAFORGE_RUN_LIBTORCH_NUMERICAL_CPU") != "1",
    reason="opt-in real LibTorch CPU generated-Session contract build",
)
@pytest.mark.parametrize("bootstrap", (False, True))
def test_actual_native_libtorch_generated_sessions_and_legacy_control(
    tmp_path, bootstrap
):
    torch = pytest.importorskip("torch")
    from vlaforge.deployment.libtorch_numerical import policy_from_context
    from vlaforge.numerical_context import offline_restore, snapshot

    assert torch.__version__.split("+", 1)[0] == "2.10.0"
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    observed = snapshot()
    if bootstrap:
        desired = replace(
            observed,
            float32_matmul_precision="medium",
            cuda_matmul_allow_tf32=True,
            cuda_matmul_allow_fp16_reduced_precision_reduction=False,
            cuda_matmul_allow_fp16_reduced_precision_reduction_split_k=False,
            cuda_matmul_allow_bf16_reduced_precision_reduction=False,
            cuda_matmul_allow_bf16_reduced_precision_reduction_split_k=False,
        )
        with offline_restore(desired, acknowledge_process_global=True):
            observed = snapshot()
    policy = policy_from_context(observed)
    runtime = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    }

    def run(command, folder, *, native=False):
        child_env = (
            {**env, "PYTHONHOME": "/nonexistent", "PYTHONPATH": "/nonexistent"}
            if native
            else env
        )
        result = subprocess.run(
            command, text=True, capture_output=True, env=child_env, check=False
        )
        with (folder / "commands.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "argv": command,
                        "exit_code": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                )
                + "\n"
            )
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    class AddOne(torch.nn.Module):
        def forward(self, value):
            return value + 1

    for legacy in (False,) if bootstrap else (False, True):
        folder = tmp_path / ("legacy" if legacy else "bound")
        bundle = folder / "bundle"
        bundle.mkdir(parents=True)
        artifact = bundle / "fixture.pt"
        with offline_restore(observed, acknowledge_process_global=True):
            torch.jit.trace(AddOne(), (torch.tensor([1.0, 2.0]),)).save(str(artifact))
        (folder / "observed-python-context.json").write_text(
            json.dumps(observed.to_dict(), indent=2)
        )
        generated = folder / "generated"
        generated.mkdir()
        for name, source in _sources(
            ("torchscript", "torchscript"),
            policy=policy,
            legacy=legacy,
            artifact=artifact,
            runner=_NATIVE_RUNNER,
            bootstrap=bootstrap,
        ).items():
            (generated / name).write_text(source)
        build = folder / "build"
        command = [
            "cmake",
            "-S",
            str(generated),
            "-B",
            str(build),
            "-DBUILD_TESTING=OFF",
            f"-DVLAFORGE_RUNTIME_ROOT={runtime}",
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
        ]
        if os.environ.get("OPENSSL_ROOT_DIR"):
            command.append("-DOPENSSL_ROOT_DIR=" + os.environ["OPENSSL_ROOT_DIR"])
        run(command, folder)
        run(["cmake", "--build", str(build), "-j2"], folder)
        binary = build / "vlaforge_generated_runner"
        linked = run(["ldd", str(binary)], folder).stdout
        assert "libpython" not in linked and "not found" not in linked
        assert "libvlaforge_libtorch_numerical_backend.so" in linked
        modes = (
            "normal",
            "multisession",
            "initdrift",
            "entrydrift",
            "threaddrift",
            "exitdrift",
        ) + (("missing-bootstrap",) if bootstrap else ())
        for mode in modes:
            result = run(
                [
                    str(binary),
                    str(bundle),
                    mode,
                    "legacy" if legacy else "bound",
                    str(folder / (mode + ".maps")),
                ],
                folder,
                native=True,
            )
            assert '"status":"passed"' in result.stdout
        manifest = {
            "evidence_level": "generated_cpu_fixture_session",
            "model_output_validation": False,
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "policy": policy.to_dict(),
            "legacy": legacy,
            "explicit_worker_bootstrap": bootstrap,
            "torch_package": torch.__version__,
        }
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    assert not torch.cuda.is_initialized()
