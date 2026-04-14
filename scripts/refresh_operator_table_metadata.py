#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from operator_table_utils import (
    LLM_OPERATOR_TABLE_PATH,
    SHARED_OPERATOR_TABLE_PATH,
    VLM_OPERATOR_TABLE_PATH,
    build_operator_impl_table_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh table_metadata for operator_impl_table JSON files in place."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Operator table JSON paths to refresh. Defaults to shared + llm + vlm tables.",
    )
    return parser.parse_args()


def candidate_paths(args: argparse.Namespace) -> list[Path]:
    if args.paths:
        return [Path(item).expanduser().resolve() for item in args.paths]

    ordered = [SHARED_OPERATOR_TABLE_PATH, LLM_OPERATOR_TABLE_PATH, VLM_OPERATOR_TABLE_PATH]
    return [path.resolve() for path in ordered if path.exists()]


def refresh_table(path: Path) -> None:
    base_table = json.loads(path.read_text())
    records = list(base_table.get("records", []))
    payload = build_operator_impl_table_payload(
        records,
        base_table=base_table,
        source_table_path=path,
        generator=__file__,
    )
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    for path in candidate_paths(parse_args()):
        refresh_table(path)
        print(path)


if __name__ == "__main__":
    main()
