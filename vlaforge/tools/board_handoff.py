"""Prepare and verify portable board input data; never infer board acceptance."""

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from vlaforge.validation.board_handoff import prepare, stage, validate, write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare")
    for name in ("protocol", "module", "provenance", "output", "report"):
        create.add_argument("--" + name, required=True)
    check = commands.add_parser("validate")
    check.add_argument("--pack", required=True)
    check.add_argument("--report", required=True)
    dispatch = commands.add_parser("stage")
    dispatch.add_argument("phase", choices=("preflight", "build", "run", "collect"))
    for name in ("pack", "target-descriptor", "output"):
        dispatch.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.protocol, args.module, args.provenance, args.output)
    elif args.command == "validate":
        result = validate(args.pack)
    else:
        result = stage(args.pack, args.target_descriptor, args.phase, args.output)
    if args.command != "stage":
        write_new(args.report, result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] in ("passed", "driver-returned") else 2


if __name__ == "__main__":
    raise SystemExit(main())
