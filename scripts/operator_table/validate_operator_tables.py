#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.operator_table.utils import (
    all_supported_platforms,
    base_engine_default_path,
    base_operator_table_path,
    platform_config_path,
    platform_uses_materialized_operator_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate base/platform operator table assets.")
    parser.add_argument(
        "--platform",
        action="append",
        choices=all_supported_platforms(),
        help="Platform(s) to validate. Defaults to all supported platforms.",
    )
    return parser.parse_args()


def validate_json_object(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def validate_operator_table(path: Path) -> None:
    payload = validate_json_object(path)
    records = payload.get("records")
    if not isinstance(records, list):
        raise TypeError(f"{path} must contain a records array")


def validate_engine_default(path: Path) -> None:
    payload = validate_json_object(path)
    if not isinstance(payload.get("runtime"), dict):
        raise TypeError(f"{path} must contain runtime settings")


def main() -> None:
    args = parse_args()
    platforms = args.platform or list(all_supported_platforms())

    validate_engine_default(base_engine_default_path())
    validate_operator_table(base_operator_table_path("llm"))
    validate_operator_table(base_operator_table_path("vlm"))

    for platform_name in platforms:
        platform_dir = platform_config_path(platform_name)
        validate_engine_default(platform_dir / "engine_default.json")
        if platform_uses_materialized_operator_tables(platform_name):
            validate_operator_table(platform_dir / "operator_impl_table_llm.json")
            validate_operator_table(platform_dir / "operator_impl_table_vlm.json")
            validate_operator_table(platform_dir / "operator_impl_table.json")
        print(platform_dir)


if __name__ == "__main__":
    main()
