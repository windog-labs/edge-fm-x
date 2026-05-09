import json
import os
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.edge_fm_build_paths import prepend_built_python_paths

prepend_built_python_paths(PROJECT_ROOT)

import edge_fm
import torch
from scripts.operator_table.utils import resolve_operator_table_path, resolve_target_hw_profile
from tests._support.temp_paths import make_temp_dir

QWEN_1P5B_MODEL_PATH = (
    PROJECT_ROOT / "examples" / "qwen2.5-1.5b-instruct" / "qwen2.5-1.5b-instruct"
)
QWEN_0P5B_MODEL_PATH = (
    PROJECT_ROOT / "examples" / "qwen2.5-0.5b-instruct" / "qwen2.5-0.5b-instruct"
)
QWEN_3B_MODEL_PATH = (
    PROJECT_ROOT / "examples" / "qwen2.5-3b-instruct" / "qwen2.5-3b-instruct"
)
OPERATOR_IMPL_TABLE_PATH = resolve_operator_table_path(
    model_path=QWEN_1P5B_MODEL_PATH,
    model_name="Qwen2.5",
)
DEFAULT_DEVICE_ID = int(os.environ.get("EDGE_FM_TEST_DEVICE_ID", "0"))
CUDA_HW_PROFILE = resolve_target_hw_profile()
DEFAULT_PREFILL_LENGTHS = [512, 1024, 2048]
DEFAULT_DECODE_LENGTHS = [32, 64]


def torch_device(device_id: int = DEFAULT_DEVICE_ID) -> str:
    return f"cuda:{device_id}"


def ensure_cuda(device_id: int = DEFAULT_DEVICE_ID) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for operator tests")
    if device_id >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested cuda:{device_id}, but only {torch.cuda.device_count()} device(s) are visible"
        )
    torch.cuda.set_device(device_id)


def _edge_fm_dtype(torch_dtype: torch.dtype) -> edge_fm.DType:
    if torch_dtype == torch.bfloat16:
        return edge_fm.DType.BFloat16
    if torch_dtype == torch.float16:
        return edge_fm.DType.Float16
    if torch_dtype == torch.float32:
        return edge_fm.DType.Float32
    if torch_dtype == torch.int32:
        return edge_fm.DType.Int32
    if torch_dtype == torch.int64:
        return edge_fm.DType.Int64
    if torch_dtype == torch.int8:
        return edge_fm.DType.Int8
    if torch_dtype == torch.uint8:
        return edge_fm.DType.UInt8
    raise TypeError(f"Unsupported torch dtype for edge_fm.Tensor view: {torch_dtype}")


def _edge_fm_device(torch_tensor: torch.Tensor) -> tuple[edge_fm.Device, int]:
    if torch_tensor.device.type == "cuda":
        return edge_fm.Device.GPU, torch_tensor.device.index or 0
    if torch_tensor.device.type == "cpu":
        return edge_fm.Device.CPU, 0
    raise TypeError(f"Unsupported torch device for edge_fm.Tensor view: {torch_tensor.device}")


def tensor_to_edge_fm_tensor(torch_tensor: torch.Tensor) -> edge_fm.Tensor:
    if not torch_tensor.is_contiguous():
        raise ValueError("tensor_to_edge_fm_tensor expects a contiguous torch.Tensor")
    device, device_id = _edge_fm_device(torch_tensor)
    return edge_fm.Tensor(
        torch_tensor.data_ptr(),
        list(torch_tensor.shape),
        _edge_fm_dtype(torch_tensor.dtype),
        device,
        device_id,
        False,
    )


def edge_fm_tensor_to_torch(tensor: edge_fm.Tensor) -> torch.Tensor:
    return torch.from_dlpack(tensor.to_dlpack())


def write_json_file(prefix: str, name: str, payload: dict) -> Path:
    temp_dir = make_temp_dir(prefix)
    path = temp_dir / name
    path.write_text(json.dumps(payload))
    return path


def load_operator_impl_table() -> dict:
    return json.loads(OPERATOR_IMPL_TABLE_PATH.read_text())


def write_operator_impl_table(records: list[dict]) -> Path:
    return write_json_file(
        "efm_operator_table_",
        "operator_impl_table.json",
        {
            "schema": "edgefm_operator_impl_table_v1",
            "records": records,
        },
    )


def make_engine_config(
    model_path: Path = QWEN_1P5B_MODEL_PATH,
    *,
    device_id: int = DEFAULT_DEVICE_ID,
    operator_impl_table_path: Path | None = None,
    model_name: str = "Qwen2.5",
    hw_profile: str | None = None,
    sampling: dict | None = None,
) -> Path:
    config = {
        "model_name": model_name,
        "runtime": {
            "device": "cuda",
            "device_id": device_id,
            "hw_profile": hw_profile or CUDA_HW_PROFILE,
        },
        "prefill_model_path": str(model_path),
    }
    if operator_impl_table_path is not None:
        config["operator_impl_table_path"] = str(operator_impl_table_path)
    if sampling is not None:
        config["sampling"] = sampling
    return write_json_file("efm_engine_config_", "engine_config.json", config)


def bench_cuda_ms(fn, *, warmup: int = 30, iters: int = 200) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    measurements = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        measurements.append(start.elapsed_time(end))
    return measurements


def median_cuda_ms(fn, *, warmup: int = 30, iters: int = 200) -> float:
    return statistics.median(bench_cuda_ms(fn, warmup=warmup, iters=iters))


def reset_weight_loader() -> None:
    loader = edge_fm.WeightLoader.instance()
    loader.clear_stage(edge_fm.ModelStage.Prefill)
    loader.clear_stage(edge_fm.ModelStage.Decode)


def dtype_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.bfloat16:
        return 1e-2, 1e-2
    if dtype == torch.float16:
        return 1e-3, 1e-3
    return 1e-4, 1e-4
