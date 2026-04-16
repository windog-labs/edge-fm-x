#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.operator_table.utils import (
    BASE_CONFIG_DIR,
    PLATFORM_HW_PROFILE_MAP,
    all_supported_platforms,
    base_engine_default_path,
    base_operator_table_path,
    build_operator_impl_table_payload,
    platform_config_path,
    platform_uses_materialized_operator_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize platform-specific config trees from base config files.")
    parser.add_argument(
        "--platform",
        action="append",
        choices=all_supported_platforms(),
        help="Platform(s) to materialize. Defaults to all supported platforms.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def unique_records(records: list[dict]) -> list[dict]:
    unique: list[dict] = []
    seen: set[str] = set()
    for record in records:
        key = json.dumps(record, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def build_engine_default(platform_name: str) -> dict:
    payload = copy.deepcopy(load_json(base_engine_default_path()))
    runtime = payload.setdefault("runtime", {})
    runtime["hw_profile"] = PLATFORM_HW_PROFILE_MAP[platform_name]
    if platform_name == "j6m":
        runtime["device"] = "horizon"
        runtime["use_cuda_graph"] = False
    else:
        runtime["device"] = "cuda"
    return payload


def materialize_platform(platform_name: str) -> None:
    platform_dir = platform_config_path(platform_name)
    platform_dir.mkdir(parents=True, exist_ok=True)

    engine_default = build_engine_default(platform_name)
    dump_json(platform_dir / "engine_default.json", engine_default)

    if not platform_uses_materialized_operator_tables(platform_name):
        for stale_name in (
            "operator_impl_table.json",
            "operator_impl_table_llm.json",
            "operator_impl_table_vlm.json",
        ):
            stale_path = platform_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()
        return

    base_llm = load_json(base_operator_table_path("llm"))
    base_vlm = load_json(base_operator_table_path("vlm"))
    base_merged = load_json(base_operator_table_path()) if base_operator_table_path().exists() else {
        "schema": "edgefm_operator_impl_table_v1",
        "records": [],
    }

    llm_payload = build_operator_impl_table_payload(
        list(base_llm.get("records", [])),
        base_table=base_llm,
        source_table_path=platform_dir / "operator_impl_table_llm.json",
        generator=__file__,
        extra_metadata={"platform": {"name": platform_name, "hw_profile": PLATFORM_HW_PROFILE_MAP[platform_name]}},
    )
    dump_json(platform_dir / "operator_impl_table_llm.json", llm_payload)

    vlm_payload = build_operator_impl_table_payload(
        list(base_vlm.get("records", [])),
        base_table=base_vlm,
        source_table_path=platform_dir / "operator_impl_table_vlm.json",
        generator=__file__,
        extra_metadata={"platform": {"name": platform_name, "hw_profile": PLATFORM_HW_PROFILE_MAP[platform_name]}},
    )
    dump_json(platform_dir / "operator_impl_table_vlm.json", vlm_payload)

    merged_records = unique_records(
        list(base_merged.get("records", []))
        + list(base_llm.get("records", []))
        + list(base_vlm.get("records", []))
    )
    merged_payload = build_operator_impl_table_payload(
        merged_records,
        base_table=base_merged,
        source_table_path=platform_dir / "operator_impl_table.json",
        generator=__file__,
        extra_metadata={"platform": {"name": platform_name, "hw_profile": PLATFORM_HW_PROFILE_MAP[platform_name]}},
    )
    dump_json(platform_dir / "operator_impl_table.json", merged_payload)


def main() -> None:
    args = parse_args()
    platforms = args.platform or list(all_supported_platforms())
    for platform_name in platforms:
        materialize_platform(platform_name)
        print(platform_config_path(platform_name))


if __name__ == "__main__":
    main()
