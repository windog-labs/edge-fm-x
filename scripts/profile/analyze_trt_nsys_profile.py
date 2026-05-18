#!/usr/bin/env python3
"""Aggregate TRT-Edge-LLM and EdgeFM kernels from an Nsight Systems profile."""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate TRT-Edge-LLM/EdgeFM nsys kernels by demangled name")
    parser.add_argument("--input", required=True, help=".nsys-rep or exported .sqlite")
    parser.add_argument("--output", default="", help="Optional markdown output path")
    parser.add_argument("--json-output", default="", help="Optional structured JSON output path")
    parser.add_argument("--force-export", action="store_true", default=False)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--min-share-pct", type=float, default=0.5)
    return parser.parse_args()


def ensure_sqlite(input_path: Path, force_export: bool) -> Path:
    if input_path.suffix == ".sqlite":
        return input_path
    if input_path.suffix != ".nsys-rep":
        raise ValueError(f"unsupported input suffix: {input_path}")
    sqlite_path = input_path.with_suffix(".sqlite")
    if sqlite_path.exists() and not force_export:
        return sqlite_path
    cmd = [
        "nsys",
        "export",
        "-q",
        "true",
        "-t",
        "sqlite",
        "--lazy=false",
        "-f",
        "true",
        "-o",
        str(sqlite_path),
        str(input_path),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "nsys export failed\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return sqlite_path


def fetch_kernel_rows(sqlite_path: Path) -> list[dict[str, Any]]:
    con = sqlite3.connect(sqlite_path)
    con.row_factory = sqlite3.Row
    try:
        runtime_rows = con.execute(
            """
            SELECT r.start, r.end, r.globalTid, r.correlationId
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            WHERE r.correlationId IS NOT NULL
            """
        ).fetchall()
        runtime_by_correlation = {int(row["correlationId"]): dict(row) for row in runtime_rows}
        ranges_by_tid: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in con.execute(
            """
            SELECT
                n.start,
                n.end,
                n.globalTid,
                COALESCE(n.text, s.value, '<unknown>') AS name
            FROM NVTX_EVENTS n
            LEFT JOIN StringIds s ON n.textId = s.id
            WHERE n.end IS NOT NULL
            """
        ):
            ranges_by_tid[int(row["globalTid"])].append(dict(row))
        for ranges in ranges_by_tid.values():
            ranges.sort(key=lambda item: (item["start"], item["end"]))

        rows = con.execute(
            """
            SELECT
                k.start AS start_ns,
                k.end AS end_ns,
                k.correlationId AS correlation_id,
                k.registersPerThread AS registers_per_thread,
                k.gridX AS grid_x,
                k.gridY AS grid_y,
                k.gridZ AS grid_z,
                k.blockX AS block_x,
                k.blockY AS block_y,
                k.blockZ AS block_z,
                k.staticSharedMemory AS static_smem,
                k.dynamicSharedMemory AS dynamic_smem,
                COALESCE(d.value, s.value, m.value, '<unknown>') AS kernel_name
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            LEFT JOIN StringIds d ON k.demangledName = d.id
            LEFT JOIN StringIds s ON k.shortName = s.id
            LEFT JOIN StringIds m ON k.mangledName = m.id
            WHERE k.end > k.start
            ORDER BY k.start
            """
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            nvtx_context = find_nvtx_context(
                runtime_by_correlation.get(int(item["correlation_id"] or -1)),
                ranges_by_tid,
            )
            item["nvtx_node"] = nvtx_context["node"]
            item["nvtx_stage"] = nvtx_context["stage"]
            result.append(item)
        return result
    finally:
        con.close()


def find_nvtx_context(
    runtime_row: dict[str, Any] | None,
    ranges_by_tid: dict[int, list[dict[str, Any]]],
) -> dict[str, str]:
    if not runtime_row:
        return {"node": "<unmapped>", "stage": "<unmapped>"}
    runtime_start = int(runtime_row["start"])
    runtime_end = int(runtime_row["end"])
    tid = int(runtime_row["globalTid"])
    best: dict[str, Any] | None = None
    stage = "<unmapped>"
    for item in ranges_by_tid.get(tid, []):
        if int(item["start"]) > runtime_start:
            break
        if int(item["end"]) >= runtime_end:
            name = str(item["name"])
            if name in {
                "EDGEFM_PREFILL_MODEL",
                "EDGEFM_DECODE_MODEL",
                "EDGEFM_PREFILL",
                "EDGEFM_DECODE",
                "EDGEFM_GENERATE",
            }:
                if stage == "<unmapped>" or stage == "EDGEFM_GENERATE":
                    stage = name
            if best is None or int(item["end"]) - int(item["start"]) < int(best["end"]) - int(best["start"]):
                best = item
    return {"node": str(best["name"]) if best is not None else "<unmapped>", "stage": stage}


def classify_kernel(name: str) -> str:
    lower = name.lower()
    if "gemm" in lower or "matmul" in lower or "xmma" in lower or "cutlass" in lower:
        return "gemm"
    if "rmsnorm" in lower or "layernorm" in lower or "norm" in lower:
        return "norm"
    if "fmha" in lower or "attention" in lower or "flash" in lower:
        return "attention"
    if "silu" in lower or "swiglu" in lower or "gelu" in lower:
        return "activation"
    if "rope" in lower or "rotary" in lower:
        return "rope"
    if "copy" in lower or "memcpy" in lower or "memset" in lower:
        return "memory"
    if "sampling" in lower or "topk" in lower:
        return "sampling"
    if "elementwise" in lower or "unary" in lower or "cast" in lower:
        return "elementwise"
    return "other"


def classify_node_role(node_name: str, kernel_name: str) -> str:
    node = node_name.lower()
    kernel = kernel_name.lower()
    if "layer_" in node and "gateup" in node:
        return "mlp_gate_up"
    if "layer_" in node and "downproj" in node:
        return "mlp_down"
    if "layer_" in node and "qkvlinear" in node:
        return "attn_qkv"
    if "layer_" in node and "oproj" in node:
        return "attn_o_proj"
    if "gateup_matmul_edgefm_layout" in node:
        return "mlp_gate_up"
    if "__myl_fccast_myl0_8" in node or "__myl_fccast_myl2_0" in node:
        return "mlp_down"
    if "__myl_fcaddcast_myl0_7" in node:
        return "attn_qkv"
    if "__myl_fccast_myl0_6" in node:
        return "attn_o_proj"
    if "__myl_slicsiluslicmul" in node or "__myl_castcastmultanhmuladdmulmulcast" in node:
        return "activation_swiglu"
    if "__myl_cast" in node or "__myl_castresh" in node:
        return "bridge_cast"
    if "edgefm_prefill_model" in node:
        if "singleprefill" in kernel or "fmha" in kernel or "attention" in kernel:
            return "prefill_attention"
        if "rmsnorm" in kernel or "layernorm" in kernel or "norm" in kernel:
            return "norm"
        if "act_and_mul" in kernel or "silu" in kernel or "swiglu" in kernel:
            return "activation_swiglu"
    if "lmhead" in node or "lm_head" in node:
        return "lm_head"
    if "embedding" in node:
        return "embedding"
    if "prefill_sampler" in node or "decode_sampler" in node:
        return "sampling"
    if "mlp/up_proj" in node or "mlp/gate_proj" in node:
        return "mlp_gate_up"
    if "mlp/down_proj" in node:
        return "mlp_down"
    if "self_attn/o_proj" in node:
        return "attn_o_proj"
    if "attentionplugin" in node:
        return "attention_plugin"
    if "__myl_fccast" in node and "gemm" in kernel:
        return "attn_qkv"
    if "lm_head" in node:
        return "lm_head"
    if "silumul" in node:
        return "activation_swiglu"
    if "sqrtdiv" in node or "mulmean" in node:
        return "norm"
    if "topk" in kernel or "sampling" in kernel:
        return "sampling"
    return "other"


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_ns = sum(int(row["end_ns"]) - int(row["start_ns"]) for row in rows)
    by_name: dict[str, dict[str, Any]] = {}
    by_category: dict[str, dict[str, Any]] = defaultdict(lambda: {"total_ns": 0, "launches": 0})
    by_role: dict[str, dict[str, Any]] = defaultdict(lambda: {"total_ns": 0, "launches": 0})
    by_stage_role: dict[str, dict[str, Any]] = defaultdict(lambda: {"total_ns": 0, "launches": 0})

    for row in rows:
        name = " ".join(str(row["kernel_name"]).split())
        node_name = " ".join(str(row.get("nvtx_node", "<unmapped>")).split())
        stage_name = " ".join(str(row.get("nvtx_stage", "<unmapped>")).split())
        duration_ns = int(row["end_ns"]) - int(row["start_ns"])
        role = classify_node_role(node_name, name)
        stage_role = f"{stage_name}:{role}"
        item = by_name.setdefault(
            name,
            {
                "kernel_name": name,
                "category": classify_kernel(name),
                "total_ns": 0,
                "launches": 0,
                "max_ns": 0,
                "sample_grid": [row["grid_x"], row["grid_y"], row["grid_z"]],
                "sample_block": [row["block_x"], row["block_y"], row["block_z"]],
                "registers_per_thread": row["registers_per_thread"],
                "static_smem": row["static_smem"],
                "dynamic_smem": row["dynamic_smem"],
                "sample_nvtx_node": node_name,
                "sample_nvtx_stage": stage_name,
            },
        )
        item["total_ns"] += duration_ns
        item["launches"] += 1
        item["max_ns"] = max(int(item["max_ns"]), duration_ns)
        category_item = by_category[item["category"]]
        category_item["total_ns"] += duration_ns
        category_item["launches"] += 1
        role_item = by_role[role]
        role_item["total_ns"] += duration_ns
        role_item["launches"] += 1
        stage_role_item = by_stage_role[stage_role]
        stage_role_item["total_ns"] += duration_ns
        stage_role_item["launches"] += 1

    kernel_table = sorted(by_name.values(), key=lambda item: item["total_ns"], reverse=True)
    category_table = [
        {"category": category, **values}
        for category, values in sorted(by_category.items(), key=lambda item: item[1]["total_ns"], reverse=True)
    ]
    role_table = [
        {"role": role, **values}
        for role, values in sorted(by_role.items(), key=lambda item: item[1]["total_ns"], reverse=True)
    ]
    stage_role_table = [
        {"stage_role": stage_role, **values}
        for stage_role, values in sorted(by_stage_role.items(), key=lambda item: item[1]["total_ns"], reverse=True)
    ]
    for item in kernel_table + category_table + role_table + stage_role_table:
        item["total_ms"] = item["total_ns"] / 1e6
        item["share_pct"] = (item["total_ns"] * 100.0 / total_ns) if total_ns else 0.0
        item["avg_us"] = (item["total_ns"] / item["launches"] / 1e3) if item["launches"] else 0.0
        if "max_ns" in item:
            item["max_us"] = item["max_ns"] / 1e3

    return {
        "total_kernel_ms": total_ns / 1e6,
        "total_launches": len(rows),
        "unique_kernel_names": len(kernel_table),
        "category_table": category_table,
        "role_table": role_table,
        "stage_role_table": stage_role_table,
        "kernel_table": kernel_table,
    }


def md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(input_path: Path, sqlite_path: Path, summary: dict[str, Any], *, top: int, min_share_pct: float) -> str:
    lines = [
        "# Nsys Kernel Summary",
        "",
        f"- input: `{input_path}`",
        f"- sqlite: `{sqlite_path}`",
        f"- total kernel time: `{summary['total_kernel_ms']:.3f} ms`",
        f"- total launches: `{summary['total_launches']}`",
        f"- unique kernel names: `{summary['unique_kernel_names']}`",
        "",
        "## Categories",
        "",
        "| Category | Total | Share | Launches | Avg |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["category_table"]:
        lines.append(
            f"| `{md_escape(item['category'])}` | `{item['total_ms']:.3f} ms` | "
            f"`{item['share_pct']:.2f}%` | `{item['launches']}` | `{item['avg_us']:.3f} us` |"
        )

    lines.extend(
        [
            "",
            "## NVTX Roles",
            "",
            "| Role | Total | Share | Launches | Avg |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in summary["role_table"]:
        lines.append(
            f"| `{md_escape(item['role'])}` | `{item['total_ms']:.3f} ms` | "
            f"`{item['share_pct']:.2f}%` | `{item['launches']}` | `{item['avg_us']:.3f} us` |"
        )

    lines.extend(
        [
            "",
            "## NVTX Stage Roles",
            "",
            "| Stage Role | Total | Share | Launches | Avg |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in summary["stage_role_table"]:
        lines.append(
            f"| `{md_escape(item['stage_role'])}` | `{item['total_ms']:.3f} ms` | "
            f"`{item['share_pct']:.2f}%` | `{item['launches']}` | `{item['avg_us']:.3f} us` |"
        )

    lines.extend(
        [
            "",
            "## Kernels",
            "",
            "| Kernel | Category | Total | Share | Launches | Avg | Max | Sample launch | Sample NVTX stage | Sample NVTX node |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    rendered = 0
    for item in summary["kernel_table"]:
        if item["share_pct"] < min_share_pct:
            continue
        if top and rendered >= top:
            break
        sample = (
            f"grid={item['sample_grid']} block={item['sample_block']} "
            f"regs={item['registers_per_thread']} smem={item['static_smem']}+{item['dynamic_smem']}"
        )
        lines.append(
            f"| `{md_escape(item['kernel_name'])}` | `{item['category']}` | "
            f"`{item['total_ms']:.3f} ms` | `{item['share_pct']:.2f}%` | "
            f"`{item['launches']}` | `{item['avg_us']:.3f} us` | "
            f"`{item['max_us']:.3f} us` | `{md_escape(sample)}` | "
            f"`{md_escape(item.get('sample_nvtx_stage', ''))}` | "
            f"`{md_escape(item.get('sample_nvtx_node', ''))}` |"
        )
        rendered += 1
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    sqlite_path = ensure_sqlite(input_path, args.force_export)
    summary = aggregate(fetch_kernel_rows(sqlite_path))
    markdown = render_markdown(
        input_path,
        sqlite_path,
        summary,
        top=args.top,
        min_share_pct=args.min_share_pct,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown)
    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2) + "\n")
    sys.stdout.write(markdown)


if __name__ == "__main__":
    main()
