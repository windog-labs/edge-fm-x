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
    all_operator_table_paths,
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

    return [path.resolve() for path in all_operator_table_paths()]


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
