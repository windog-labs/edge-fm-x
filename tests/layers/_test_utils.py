import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.edge_fm_build_paths import prepend_built_python_paths
from scripts.operator_table.utils import resolve_operator_table_path, resolve_target_hw_profile

prepend_built_python_paths(PROJECT_ROOT)

DEFAULT_DEVICE_ID = int(os.environ.get("EDGE_FM_TEST_DEVICE_ID", os.environ.get("EDGE_FM_DEVICE_ID", "0")))
CUDA_HW_PROFILE = resolve_target_hw_profile()
QWEN_0P5B_MODEL_PATH = (
    PROJECT_ROOT / "examples" / "qwen2.5-0.5b-instruct" / "qwen2.5-0.5b-instruct"
)
OPERATOR_IMPL_TABLE_PATH = resolve_operator_table_path(
    model_path=QWEN_0P5B_MODEL_PATH,
    model_name="Qwen2.5",
)


def make_layer_engine_config(model_path: str | Path, *, with_operator_table: bool = True) -> dict:
    config = {
        "model_name": "Qwen2.5",
        "runtime": {
            "device": "cuda",
            "device_id": DEFAULT_DEVICE_ID,
            "hw_profile": CUDA_HW_PROFILE,
        },
        "prefill_model_path": str(model_path),
    }
    if with_operator_table:
        config["operator_impl_table_path"] = str(OPERATOR_IMPL_TABLE_PATH)
    return config
