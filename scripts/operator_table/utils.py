from __future__ import annotations

import copy
import json
import os
import platform as py_platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from scripts.edge_fm_build_paths import DEFAULT_PLATFORM, SUPPORTED_PLATFORMS, resolve_platform


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_CONFIG_DIR = REPO_ROOT / "examples" / "config"
BASE_CONFIG_DIR = EXAMPLES_CONFIG_DIR / "base"
PLATFORM_CONFIG_DIR = EXAMPLES_CONFIG_DIR / "platform"
PLATFORM_HW_PROFILE_MAP = {
    "3060": "cuda_sm86",
    "a800": "cuda_sm80",
    "orin": "cuda_sm87",
    "j6m": "horizon",
}


def normalize_platform_name(platform_name: str | None) -> str:
    normalized = (platform_name or "").strip().lower()
    if normalized not in SUPPORTED_PLATFORMS:
        return DEFAULT_PLATFORM
    return normalized


def resolve_platform_name(platform_name: str | None = None) -> str:
    return normalize_platform_name(resolve_platform(REPO_ROOT, explicit_platform=platform_name))


def platform_hw_profile(platform_name: str | None = None) -> str:
    return PLATFORM_HW_PROFILE_MAP[resolve_platform_name(platform_name)]


def platform_config_path(platform_name: str | None = None) -> Path:
    return PLATFORM_CONFIG_DIR / resolve_platform_name(platform_name)


def base_engine_default_path() -> Path:
    return BASE_CONFIG_DIR / "engine_default.json"


def platform_engine_default_path(platform_name: str | None = None) -> Path:
    return platform_config_path(platform_name) / "engine_default.json"


def base_operator_table_path(family: str | None = None) -> Path:
    if family == "llm":
        return BASE_CONFIG_DIR / "operator_impl_table_llm.json"
    if family == "vlm":
        return BASE_CONFIG_DIR / "operator_impl_table_vlm.json"
    return BASE_CONFIG_DIR / "operator_impl_table.json"


def platform_operator_table_path(platform_name: str | None = None, family: str | None = None) -> Path:
    config_dir = platform_config_path(platform_name)
    if family == "llm":
        return config_dir / "operator_impl_table_llm.json"
    if family == "vlm":
        return config_dir / "operator_impl_table_vlm.json"
    return config_dir / "operator_impl_table.json"


def all_supported_platforms() -> tuple[str, ...]:
    return tuple(SUPPORTED_PLATFORMS)


def all_operator_table_paths() -> list[Path]:
    paths = [
        base_operator_table_path("llm"),
        base_operator_table_path("vlm"),
    ]
    for platform_name in SUPPORTED_PLATFORMS:
        paths.extend(
            [
                platform_operator_table_path(platform_name, "llm"),
                platform_operator_table_path(platform_name, "vlm"),
                platform_operator_table_path(platform_name, None),
            ]
        )
    return [path for path in paths if path.exists()]


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


def default_operator_table_path_for_family(family: str, platform_name: str | None = None) -> Path:
    platform_path = platform_operator_table_path(platform_name, family)
    if platform_path.exists():
        return platform_path

    if family == "vlm":
        return base_operator_table_path("vlm")
    return base_operator_table_path("llm")


def resolve_operator_table_path(
    operator_table_path: Path | None = None,
    *,
    model_path: Path | None = None,
    model_name: str | None = None,
    config: dict | None = None,
    platform_name: str | None = None,
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

    return default_operator_table_path_for_family(family, platform_name).resolve()


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_text(cmd: list[str], *, cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def _detect_cuda_toolchain() -> dict:
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")).expanduser()
    nvcc_path = cuda_home / "bin" / "nvcc"
    if not nvcc_path.exists():
        resolved = _run_text(["bash", "-lc", "command -v nvcc"])
        if resolved:
            nvcc_path = Path(resolved)
            cuda_home = nvcc_path.parents[1]

    nvcc_raw = _run_text([str(nvcc_path), "--version"]) if nvcc_path.exists() else ""
    cuda_release = ""
    cuda_version = ""
    for line in nvcc_raw.splitlines():
        if "release " in line:
            cuda_release = line.split("release ", 1)[1].split(",", 1)[0].strip()
        if "V" in line and "release " in line:
            cuda_version = line.rsplit("V", 1)[-1].strip()

    return {
        "cuda_home": str(cuda_home) if cuda_home.exists() else str(cuda_home),
        "nvcc_path": str(nvcc_path) if nvcc_path.exists() else "",
        "nvcc_raw": nvcc_raw,
        "cuda_release": cuda_release,
        "cuda_version": cuda_version,
    }


def _git_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    git_commit = _run_text(["git", "rev-parse", "HEAD"], cwd=path)
    git_describe = _run_text(["git", "describe", "--always", "--dirty"], cwd=path)
    payload: dict[str, str] = {}
    if git_commit:
        payload["git_commit"] = git_commit
    if git_describe:
        payload["git_describe"] = git_describe
    return payload


def _detect_dependencies(cuda_release: str) -> dict:
    deps = {
        "cublasLt": {
            "bundled_with_cuda_release": cuda_release,
        }
    }

    flashinfer = _git_metadata(REPO_ROOT / "third_party" / "flashinfer")
    if flashinfer:
        deps["flashinfer"] = flashinfer

    cutlass = _git_metadata(REPO_ROOT / "third_party" / "cutlass")
    if cutlass:
        deps["cutlass"] = cutlass

    tensorrt_edgellm = _git_metadata(REPO_ROOT / "third_party" / "TensorRT-Edge-LLM")
    if tensorrt_edgellm:
        deps["tensorrt_edgellm"] = tensorrt_edgellm

    return deps


def _sanitize_dependencies_metadata(table_metadata: dict) -> None:
    dependencies = table_metadata.get("dependencies")
    if not isinstance(dependencies, dict):
        return

    for dep_name in ("flashinfer", "cutlass", "tensorrt_edgellm"):
        dep_metadata = dependencies.get(dep_name)
        if isinstance(dep_metadata, dict):
            dep_metadata.pop("path", None)

    dependencies.pop("tensorrt_package_dir", None)


def _sanitize_generator_metadata(table_metadata: dict) -> None:
    generator = table_metadata.get("generator")
    if isinstance(generator, dict):
        generator.pop("repo_root", None)

    table_metadata.pop("source_operator_table_path", None)


def _deep_merge(base: dict, extra: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def build_operator_impl_table_payload(
    records: list[dict],
    *,
    base_table: dict | None = None,
    source_table_path: Path | None = None,
    generator: str | None = None,
    extra_metadata: dict | None = None,
) -> dict:
    base_table = copy.deepcopy(base_table or {})
    payload = {
        "schema": base_table.get("schema", "edgefm_operator_impl_table_v1"),
        "records": copy.deepcopy(records),
    }

    cuda = _detect_cuda_toolchain()
    table_metadata = {
        "generated_at_utc": _utc_now_str(),
        "generator": {
            "script": str(Path(generator).resolve()) if generator else "",
        },
        "toolchain": {
            "python_version": py_platform.python_version(),
            "cuda": cuda,
        },
        "dependencies": _detect_dependencies(cuda.get("cuda_release", "")),
        "compatibility_hints": {
            "cuda_release_major_minor_should_match": True,
            "notes": (
                "Shape-tuned cublasLt / FlashInfer records should be treated as "
                "toolchain-sensitive. Rebuild and re-tune when CUDA major.minor changes."
            ),
        },
    }

    existing_metadata = base_table.get("table_metadata", {})
    if isinstance(existing_metadata, dict):
        table_metadata = _deep_merge(existing_metadata, table_metadata)
    if extra_metadata:
        table_metadata = _deep_merge(table_metadata, extra_metadata)
    _sanitize_generator_metadata(table_metadata)
    _sanitize_dependencies_metadata(table_metadata)

    payload["table_metadata"] = table_metadata
    return payload
