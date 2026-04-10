from __future__ import annotations

import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_CONFIG_DIR = REPO_ROOT / "examples" / "config"
SHARED_OPERATOR_TABLE_PATH = EXAMPLES_CONFIG_DIR / "operator_impl_table.json"
LLM_OPERATOR_TABLE_PATH = EXAMPLES_CONFIG_DIR / "operator_impl_table_llm.json"
VLM_OPERATOR_TABLE_PATH = EXAMPLES_CONFIG_DIR / "operator_impl_table_vlm.json"


def load_model_config(model_path: Path) -> dict:
    config_path = model_path / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text())


def resolve_model_family(
    *,
    model_path: Path | None = None,
    model_name: str | None = None,
    config: dict | None = None,
) -> str:
    normalized_name = (model_name or "").strip().lower()
    if "qwen2.5-vl" in normalized_name or "qwen2_5_vl" in normalized_name or normalized_name == "vlm":
        return "vlm"
    if normalized_name:
        return "llm"

    if model_path is not None:
        path_name = model_path.name.lower()
        if "qwen2.5-vl" in path_name or "-vl-" in path_name:
            return "vlm"

    cfg = config
    if cfg is None and model_path is not None:
        cfg = load_model_config(model_path)
    cfg = cfg or {}
    if isinstance(cfg.get("text_config"), dict) or isinstance(cfg.get("vision_config"), dict):
        return "vlm"

    return "llm"


def resolve_engine_model_name(
    model_path: Path,
    *,
    explicit_model_name: str | None = None,
    config: dict | None = None,
) -> str:
    if explicit_model_name:
        return explicit_model_name
    family = resolve_model_family(model_path=model_path, config=config)
    return "Qwen2.5-VL" if family == "vlm" else "Qwen2.5"


def resolve_operator_model_name(
    *,
    model_path: Path | None = None,
    model_name: str | None = None,
    config: dict | None = None,
) -> str:
    family = resolve_model_family(model_path=model_path, model_name=model_name, config=config)
    return "qwen2_5_vl" if family == "vlm" else "qwen2_5"


def default_operator_table_path_for_family(family: str) -> Path:
    if family == "vlm":
        return VLM_OPERATOR_TABLE_PATH if VLM_OPERATOR_TABLE_PATH.exists() else SHARED_OPERATOR_TABLE_PATH
    return LLM_OPERATOR_TABLE_PATH if LLM_OPERATOR_TABLE_PATH.exists() else SHARED_OPERATOR_TABLE_PATH


def resolve_operator_table_path(
    operator_table_path: Path | None = None,
    *,
    model_path: Path | None = None,
    model_name: str | None = None,
    config: dict | None = None,
) -> Path:
    if operator_table_path is not None:
        return operator_table_path.resolve()

    family = resolve_model_family(model_path=model_path, model_name=model_name, config=config)
    family_env_key = "EDGE_FM_OPERATOR_IMPL_TABLE_VLM" if family == "vlm" else "EDGE_FM_OPERATOR_IMPL_TABLE_LLM"
    family_env_value = os.environ.get(family_env_key, "").strip()
    if family_env_value:
        return Path(family_env_value).expanduser().resolve()

    generic_env_value = os.environ.get("EDGE_FM_OPERATOR_IMPL_TABLE", "").strip()
    if generic_env_value:
        return Path(generic_env_value).expanduser().resolve()

    return default_operator_table_path_for_family(family).resolve()
