#!/usr/bin/env python3
"""Compact Edge-FM Nsight Systems triage."""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


MIN_RENDER_SHARE_PCT = 1.0
STAGE_RANGE_NAMES = {
    "EDGEFM_PREFILL": "prefill",
    "EDGEFM_GENERATION": "decode",
    "EDGEFM_GENERATE": "all",
}
IGNORE_LAYER_RANGE_NAMES = frozenset(STAGE_RANGE_NAMES)
KERNEL_CATEGORY_HINTS = {
    "attention": ("flash", "attention", "fmha", "paged", "mla"),
    "linear": ("gemm", "gemv", "cublas", "cutlass", "xmma", "mma", "matmul"),
    "norm": ("rmsnorm", "layernorm", "norm"),
    "activation": ("silu", "swiglu", "gelu", "relu", "sigmoid"),
    "memory": ("memcpy", "memset"),
}
KNOWN_PATHS = {
    ("prefill", "attention"): {
        "known_path": "FlashInfer prefill attention path",
        "inspect": "src/operators/attention_op.cu, src/layers/attention.cu",
        "next_action": "Run scripts/tune/tune_qwen_attention_prefill.py for the matching head dims before designing a new prefill attention kernel.",
    },
    ("decode", "attention"): {
        "known_path": "FlashInfer decode tuned attention path",
        "inspect": "src/operators/attention_op.cu, src/layers/attention.cu",
        "next_action": "Check the decode operator_impl_table record first, then rerun scripts/tune/tune_qwen_attention_decode.py on the exact hotspot shape.",
    },
    ("all", "attention"): {
        "known_path": "FlashInfer attention path",
        "inspect": "src/operators/attention_op.cu, src/layers/attention.cu",
        "next_action": "Split the trace into prefill/decode or capture separate cases before deciding the attention hotspot is novel.",
    },
    ("all", "linear:fused_qkv"): {
        "known_path": "QKV linear operator-table path",
        "inspect": "src/layers/linear.cu, src/operators/linear_impl.cu",
        "next_action": "Retune the matching fused_qkv linear path with scripts/tune/tune_qwen_cublaslt.py before changing runtime structure.",
    },
    ("all", "linear:attention_output"): {
        "known_path": "Attention output linear operator-table path",
        "inspect": "src/layers/linear.cu, src/operators/linear_impl.cu",
        "next_action": "Check the attention_output record and rerun scripts/tune/tune_qwen_cublaslt.py for the hotspot batch/shape.",
    },
    ("all", "linear:mlp_down"): {
        "known_path": "MLP down-projection operator-table path",
        "inspect": "src/layers/linear.cu, src/operators/linear_impl.cu",
        "next_action": "Treat this as a cublasLt or CUTLASS selection problem first; retune with scripts/tune/tune_qwen_cublaslt.py.",
    },
    ("all", "linear"): {
        "known_path": "Generic linear operator-table path",
        "inspect": "src/layers/linear.cu, src/operators/linear_impl.cu",
        "next_action": "Validate hw_profile and selected linear impl first, then retune with scripts/tune/tune_qwen_cublaslt.py.",
    },
    ("decode", "fused_gate_up"): {
        "known_path": "Decode fused gate-up or SwiGLU fast path",
        "inspect": "src/operators/fused_gate_up_activation_op.cu, src/layers/activation.cu",
        "next_action": "Check whether the decode fused gate-up path fired, then rerun scripts/tune/tune_qwen_decode_swiglu.py.",
    },
    ("all", "norm"): {
        "known_path": "Existing FlashInfer norm path",
        "inspect": "src/operators/norm_op.cu, src/layers/layernorm.cu",
        "next_action": "Confirm the run already selected flashinfer_norm before proposing a new norm fusion.",
    },
    ("all", "embedding"): {
        "known_path": "Embedding or lm_head path",
        "inspect": "src/layers/embed_head.cu, src/layers/linear.cu",
        "next_action": "Check whether the hotspot is true model work or just a short-context artifact before adding a new embedding fast path.",
    },
}


@dataclass(frozen=True)
class NvtxRange:
    start: int
    end: int
    name: str
    global_tid: int

    @property
    def duration(self) -> int:
        return max(0, self.end - self.start)


@dataclass(frozen=True)
class RuntimeEvent:
    start: int
    end: int
    global_tid: int
    correlation_id: int
    name: str


@dataclass(frozen=True)
class KernelEvent:
    start: int
    end: int
    global_pid: int
    stream_id: int
    correlation_id: Optional[int]
    name: str

    @property
    def duration(self) -> int:
        return max(0, self.end - self.start)


@dataclass(frozen=True)
class MappingHint:
    stage: str
    layer_name: str


@dataclass
class KernelAggregate:
    stage: str
    layer_name: str
    kernel_name: str
    total_ns: int = 0
    launches: int = 0


class RangeIndex:
    def __init__(self, ranges: Iterable[NvtxRange]):
        self.ranges = sorted(ranges, key=lambda item: (item.start, item.end))
        self.starts = [item.start for item in self.ranges]

    def find_innermost(self, ts: int) -> Optional[NvtxRange]:
        idx = bisect_right(self.starts, ts) - 1
        best: Optional[NvtxRange] = None
        while idx >= 0:
            item = self.ranges[idx]
            if item.start > ts:
                idx -= 1
                continue
            if item.end >= ts:
                if best is None or item.duration < best.duration:
                    best = item
            idx -= 1
        return best


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compact Edge-FM Nsight Systems triage. "
            "Prints kernel, known-path, and action tables from a single trace "
            "or a graph-off mapping + graph-on formal pair."
        )
    )
    parser.add_argument("--input", default=None, help="Single .nsys-rep, .sqlite, or directory.")
    parser.add_argument("--mapping-input", default=None, help="Graph-off mapping trace.")
    parser.add_argument("--formal-input", default=None, help="Graph-on formal trace.")
    parser.add_argument("--output", default="", help="Optional markdown output path.")
    parser.add_argument("--export-dir", default="", help="Optional directory for generated sqlite files.")
    parser.add_argument("--force-export", action="store_true", default=False)
    parser.add_argument("--min-share-pct", type=float, default=MIN_RENDER_SHARE_PCT)
    parser.add_argument("--kernel-table-limit", type=int, default=0)
    parser.add_argument("--known-table-limit", type=int, default=12)
    parser.add_argument("--action-table-limit", type=int, default=12)
    parser.add_argument("--json", action="store_true", help="Print structured JSON instead of markdown.")
    return parser


def resolve_args(argv: list[str]) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    single = bool(args.input)
    dual = bool(args.mapping_input or args.formal_input)
    if single == dual:
        parser.error("Use either --input or the pair --mapping-input/--formal-input.")
    if dual and (not args.mapping_input or not args.formal_input):
        parser.error("Two-trace mode requires both --mapping-input and --formal-input.")
    return args


def normalize_kernel_name(name: str) -> str:
    return " ".join((name or "").split())


def format_ms(ns: int | float) -> str:
    return f"{float(ns) / 1e6:.3f} ms"


def pct(value: int | float, total: int | float) -> float:
    if total <= 0:
        return 0.0
    return float(value) * 100.0 / float(total)


def escape_md_cell(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ")


def stage_label(stage: str) -> str:
    return {"prefill": "Prefill", "decode": "Decode", "all": "All"}.get(stage, stage)


def discover_trace_target(raw_path: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Trace path not found: {path}")

    candidates = sorted(path.glob("*.sqlite")) + sorted(path.glob("*.nsys-rep"))
    if not candidates:
        raise FileNotFoundError(f"No .sqlite or .nsys-rep found under {path}")
    return candidates[0]


def ensure_sqlite(path: Path, *, export_dir: Optional[Path], force_export: bool) -> Path:
    if path.suffix == ".sqlite":
        return path
    if path.suffix != ".nsys-rep":
        raise ValueError(f"Unsupported trace input: {path}")

    target_dir = export_dir if export_dir is not None else path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = target_dir / f"{path.stem}.sqlite"
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
        str(path),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Failed to export sqlite with nsys export.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return sqlite_path


def open_sqlite(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    return con


def require_tables(con: sqlite3.Connection, tables: Iterable[str]) -> None:
    existing = {
        row[0]
        for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    missing = [name for name in tables if name not in existing]
    if missing:
        raise RuntimeError(
            "SQLite export is missing required tables: "
            + ", ".join(missing)
            + ". Re-export with `nsys export -t sqlite --lazy=false`."
        )


def load_nvtx_ranges(con: sqlite3.Connection) -> list[NvtxRange]:
    rows = con.execute(
        """
        SELECT
            n.start AS start,
            n.end AS end,
            n.globalTid AS global_tid,
            COALESCE(NULLIF(n.text, ''), s.value, '') AS name
        FROM NVTX_EVENTS AS n
        LEFT JOIN StringIds AS s
            ON s.id = n.textId
        WHERE n.end IS NOT NULL
        """
    ).fetchall()
    out: list[NvtxRange] = []
    for row in rows:
        name = (row["name"] or "").strip()
        if not name:
            continue
        out.append(
            NvtxRange(
                start=int(row["start"]),
                end=int(row["end"]),
                name=name,
                global_tid=int(row["global_tid"] or 0),
            )
        )
    return out


def load_runtime_events(con: sqlite3.Connection) -> dict[int, RuntimeEvent]:
    rows = con.execute(
        """
        SELECT
            r.start AS start,
            r.end AS end,
            r.globalTid AS global_tid,
            r.correlationId AS correlation_id,
            COALESCE(s.value, '') AS name
        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
        LEFT JOIN StringIds AS s
            ON s.id = r.nameId
        WHERE r.correlationId IS NOT NULL
        """
    ).fetchall()
    chosen: dict[int, RuntimeEvent] = {}
    for row in rows:
        correlation_id = int(row["correlation_id"])
        event = RuntimeEvent(
            start=int(row["start"]),
            end=int(row["end"]),
            global_tid=int(row["global_tid"] or 0),
            correlation_id=correlation_id,
            name=(row["name"] or "").strip(),
        )
        previous = chosen.get(correlation_id)
        if previous is None:
            chosen[correlation_id] = event
            continue
        prev_launch = "launch" in previous.name.lower()
        this_launch = "launch" in event.name.lower()
        if this_launch and not prev_launch:
            chosen[correlation_id] = event
    return chosen


def load_kernel_events(con: sqlite3.Connection) -> list[KernelEvent]:
    rows = con.execute(
        """
        SELECT
            k.start AS start,
            k.end AS end,
            k.globalPid AS global_pid,
            k.streamId AS stream_id,
            k.correlationId AS correlation_id,
            COALESCE(sd.value, ss.value, '') AS name
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        LEFT JOIN StringIds AS sd
            ON sd.id = k.demangledName
        LEFT JOIN StringIds AS ss
            ON ss.id = k.shortName
        """
    ).fetchall()
    out: list[KernelEvent] = []
    for row in rows:
        name = normalize_kernel_name(row["name"] or "")
        if not name:
            continue
        out.append(
            KernelEvent(
                start=int(row["start"]),
                end=int(row["end"]),
                global_pid=int(row["global_pid"] or 0),
                stream_id=int(row["stream_id"] or 0),
                correlation_id=int(row["correlation_id"]) if row["correlation_id"] is not None else None,
                name=name,
            )
        )
    return out


def build_range_indexes(ranges: Iterable[NvtxRange]) -> dict[int, RangeIndex]:
    grouped: dict[int, list[NvtxRange]] = defaultdict(list)
    for item in ranges:
        grouped[item.global_tid].append(item)
    return {tid: RangeIndex(items) for tid, items in grouped.items()}


def find_stage_range(
    indexes: dict[int, RangeIndex],
    global_index: RangeIndex,
    global_tid: Optional[int],
    ts: int,
) -> Optional[NvtxRange]:
    if global_tid is not None and global_tid in indexes:
        found = indexes[global_tid].find_innermost(ts)
        if found is not None:
            return found
    return global_index.find_innermost(ts)


def infer_family(stage: str, layer_name: str, kernel_name: str) -> tuple[str, str]:
    lname = layer_name.lower()
    kname = kernel_name.lower()

    if ".attn.qkv_fused" in lname or "qkv_fused" in lname:
        return "linear:fused_qkv", "fused_qkv"
    if ".attn.o_proj" in lname:
        return "linear:attention_output", "attention_output"
    if ".mlp.down_proj" in lname:
        return "linear:mlp_down", "mlp_down"
    if ".mlp.gate_up_fused" in lname or "gate_up_fused" in lname:
        return "fused_gate_up", "fused_gate_up"
    if lname.endswith(".attn") or ".attn" in lname:
        return "attention", "attention"
    if "layernorm" in lname or lname.endswith("norm") or ".input_layernorm" in lname:
        return "norm", "norm"
    if "embed" in lname or "lm_head" in lname:
        return "embedding", "embedding"

    for family, hints in KERNEL_CATEGORY_HINTS.items():
        if any(hint in kname for hint in hints):
            if family == "linear":
                return "linear", "linear"
            return family, family
    return "other", "other"


def build_mapping_hints(
    kernels: Iterable[KernelEvent],
    runtimes: dict[int, RuntimeEvent],
    stage_indexes: dict[int, RangeIndex],
    stage_global: RangeIndex,
    layer_indexes: dict[int, RangeIndex],
) -> dict[str, MappingHint]:
    stage_votes: dict[str, Counter] = defaultdict(Counter)
    layer_votes: dict[str, Counter] = defaultdict(Counter)
    for kernel in kernels:
        runtime = runtimes.get(kernel.correlation_id or -1)
        ts = runtime.start if runtime is not None else kernel.start
        gtid = runtime.global_tid if runtime is not None else None
        stage_range = find_stage_range(stage_indexes, stage_global, gtid, ts)
        stage = STAGE_RANGE_NAMES.get(stage_range.name, "all") if stage_range is not None else "all"
        layer_name = "unknown"
        if gtid is not None and gtid in layer_indexes:
            layer_range = layer_indexes[gtid].find_innermost(ts)
            if layer_range is not None:
                layer_name = layer_range.name
        stage_votes[kernel.name][stage] += kernel.duration
        layer_votes[kernel.name][layer_name] += kernel.duration

    hints: dict[str, MappingHint] = {}
    for kernel_name, counts in stage_votes.items():
        stage = counts.most_common(1)[0][0] if counts else "all"
        layer_name = "unknown"
        if layer_votes[kernel_name]:
            layer_name = layer_votes[kernel_name].most_common(1)[0][0]
        hints[kernel_name] = MappingHint(stage=stage, layer_name=layer_name)
    return hints


def aggregate_kernels(
    kernels: Iterable[KernelEvent],
    runtimes: dict[int, RuntimeEvent],
    stage_indexes: dict[int, RangeIndex],
    stage_global: RangeIndex,
    layer_indexes: dict[int, RangeIndex],
    mapping_hints: Optional[dict[str, MappingHint]] = None,
) -> dict[str, list[KernelAggregate]]:
    buckets: dict[tuple[str, str, str], KernelAggregate] = {}
    totals: Counter = Counter()

    for kernel in kernels:
        runtime = runtimes.get(kernel.correlation_id or -1)
        ts = runtime.start if runtime is not None else kernel.start
        gtid = runtime.global_tid if runtime is not None else None

        stage_range = find_stage_range(stage_indexes, stage_global, gtid, ts)
        stage = STAGE_RANGE_NAMES.get(stage_range.name, "all") if stage_range is not None else "all"
        layer_name = "unknown"
        if gtid is not None and gtid in layer_indexes:
            layer_range = layer_indexes[gtid].find_innermost(ts)
            if layer_range is not None:
                layer_name = layer_range.name

        hint = None if mapping_hints is None else mapping_hints.get(kernel.name)
        if hint is not None:
            if stage == "all" and hint.stage != "all":
                stage = hint.stage
            if layer_name == "unknown" and hint.layer_name != "unknown":
                layer_name = hint.layer_name

        key = (stage, layer_name, kernel.name)
        agg = buckets.get(key)
        if agg is None:
            agg = KernelAggregate(stage=stage, layer_name=layer_name, kernel_name=kernel.name)
            buckets[key] = agg
        agg.total_ns += kernel.duration
        agg.launches += 1
        totals[stage] += kernel.duration

    rendered: dict[str, list[KernelAggregate]] = defaultdict(list)
    for agg in buckets.values():
        rendered[agg.stage].append(agg)
    for stage, rows in rendered.items():
        rows.sort(key=lambda item: item.total_ns, reverse=True)
    rendered["__totals__"] = [KernelAggregate(stage=stage, layer_name="", kernel_name="", total_ns=total) for stage, total in totals.items()]
    return rendered


def resolve_known_path(stage: str, family: str) -> Optional[dict[str, str]]:
    return KNOWN_PATHS.get((stage, family)) or KNOWN_PATHS.get(("all", family))


def render_kernel_rows(
    per_stage: dict[str, list[KernelAggregate]],
    *,
    min_share_pct: float,
    table_limit: int,
) -> tuple[list[dict[str, object]], Counter]:
    totals = Counter({row.stage: row.total_ns for row in per_stage.get("__totals__", [])})
    rendered: list[dict[str, object]] = []
    stage_order = {"prefill": 0, "decode": 1, "all": 2}
    for stage in sorted(
        [key for key in per_stage if not key.startswith("__")],
        key=lambda item: (stage_order.get(item, 99), item),
    ):
        visible = 0
        for agg in per_stage[stage]:
            share_pct = pct(agg.total_ns, totals[stage])
            if share_pct < min_share_pct:
                continue
            if table_limit and visible >= table_limit:
                break
            family, layer_role = infer_family(stage, agg.layer_name, agg.kernel_name)
            rendered.append(
                {
                    "stage": stage,
                    "layer_name": agg.layer_name,
                    "kernel_name": agg.kernel_name,
                    "total_ns": agg.total_ns,
                    "share_pct": share_pct,
                    "launches": agg.launches,
                    "family": family,
                    "layer_role": layer_role,
                }
            )
            visible += 1
    return rendered, totals


def build_known_path_rows(kernel_rows: list[dict[str, object]], *, limit: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in kernel_rows:
        family = str(item["family"])
        if family == "other":
            continue
        stage = str(item["stage"])
        key = (stage, family, str(item["layer_name"]))
        if key in seen:
            continue
        path = resolve_known_path(stage, family)
        if path is None:
            continue
        rows.append(
            {
                "stage": stage,
                "layer_name": item["layer_name"],
                "family": family,
                "share_pct": item["share_pct"],
                "kernel_name": item["kernel_name"],
                "known_path": path["known_path"],
                "inspect": path["inspect"],
            }
        )
        seen.add(key)
        if limit and len(rows) >= limit:
            break
    return rows


def build_action_rows(kernel_rows: list[dict[str, object]], *, limit: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    priority = 1
    for item in kernel_rows:
        family = str(item["family"])
        if family == "other":
            continue
        stage = str(item["stage"])
        path = resolve_known_path(stage, family)
        if path is None:
            continue
        key = (stage, family)
        if key in seen:
            continue
        rows.append(
            {
                "priority": priority,
                "stage": stage,
                "signal": f"{item['layer_name']} -> {item['kernel_name']}",
                "share_pct": item["share_pct"],
                "inspect": path["inspect"],
                "next_action": path["next_action"],
            }
        )
        priority += 1
        seen.add(key)
        if limit and len(rows) >= limit:
            break
    return rows


def render_markdown(
    *,
    mode: str,
    trace_paths: dict[str, str],
    sqlite_paths: dict[str, str],
    kernel_rows: list[dict[str, object]],
    known_rows: list[dict[str, object]],
    action_rows: list[dict[str, object]],
    totals: Counter,
) -> str:
    lines = [
        "# Edge-FM Nsight Systems Triage",
        "",
        f"- Mode: `{mode}`",
    ]
    for label, path in trace_paths.items():
        lines.append(f"- {label}: `{path}`")
    for label, path in sqlite_paths.items():
        lines.append(f"- {label} sqlite: `{path}`")

    lines.extend(
        [
            "",
            "## Kernel Table",
            "",
            "| Stage | Layer | Kernel | GPU time | Share | Launches | Family |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    if not kernel_rows:
        lines.append("| - | - | No kernels found in the trace. | - | - | - | - |")
    else:
        for row in kernel_rows:
            lines.append(
                "| {stage} | {layer} | {kernel} | {gpu_time} | {share:.1f}% | {launches} | {family} |".format(
                    stage=escape_md_cell(stage_label(str(row["stage"]))),
                    layer=escape_md_cell(str(row["layer_name"])),
                    kernel=escape_md_cell(str(row["kernel_name"])),
                    gpu_time=format_ms(row["total_ns"]),
                    share=float(row["share_pct"]),
                    launches=int(row["launches"]),
                    family=escape_md_cell(str(row["family"])),
                )
            )

    lines.extend(
        [
            "",
            "## Known-Path Table",
            "",
            "| Stage | Layer | Signal | Share | Existing path | Inspect first |",
            "| --- | --- | --- | ---: | --- | --- |",
        ]
    )
    if not known_rows:
        lines.append("| - | - | No known-path match above threshold. | - | - | - |")
    else:
        for row in known_rows:
            lines.append(
                "| {stage} | {layer} | {signal} | {share:.1f}% | {path} | {inspect} |".format(
                    stage=escape_md_cell(stage_label(str(row["stage"]))),
                    layer=escape_md_cell(str(row["layer_name"])),
                    signal=escape_md_cell(str(row["kernel_name"])),
                    share=float(row["share_pct"]),
                    path=escape_md_cell(str(row["known_path"])),
                    inspect=escape_md_cell(str(row["inspect"])),
                )
            )

    lines.extend(
        [
            "",
            "## Action Table",
            "",
            "| Priority | Stage | Signal | Share | Inspect first | Next action |",
            "| ---: | --- | --- | ---: | --- | --- |",
        ]
    )
    if not action_rows:
        lines.append("| 1 | - | No action rows above threshold. | - | - | Capture a stage-focused trace with the existing profile helpers first. |")
    else:
        for row in action_rows:
            lines.append(
                "| {priority} | {stage} | {signal} | {share:.1f}% | {inspect} | {action} |".format(
                    priority=int(row["priority"]),
                    stage=escape_md_cell(stage_label(str(row["stage"]))),
                    signal=escape_md_cell(str(row["signal"])),
                    share=float(row["share_pct"]),
                    inspect=escape_md_cell(str(row["inspect"])),
                    action=escape_md_cell(str(row["next_action"])),
                )
            )

    conclusion = "No GPU kernels were found."
    if kernel_rows:
        top = max(kernel_rows, key=lambda item: float(item["total_ns"]))
        stage_total = totals[str(top["stage"])]
        conclusion = (
            f"{stage_label(str(top['stage']))} is dominated by "
            f"`{top['layer_name']}` -> `{top['kernel_name']}` "
            f"at {pct(top['total_ns'], stage_total):.1f}% of stage GPU time."
        )
    lines.extend(["", f"Conclusion: {conclusion}"])
    return "\n".join(lines) + "\n"


def triage_trace(
    trace_path: Path,
    *,
    export_dir: Optional[Path],
    force_export: bool,
) -> tuple[Path, list[KernelEvent], dict[int, RuntimeEvent], dict[int, RangeIndex], RangeIndex, dict[int, RangeIndex]]:
    sqlite_path = ensure_sqlite(trace_path, export_dir=export_dir, force_export=force_export)
    con = open_sqlite(sqlite_path)
    try:
        require_tables(con, ["NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_RUNTIME", "StringIds"])
        nvtx = load_nvtx_ranges(con)
        runtimes = load_runtime_events(con)
        kernels = load_kernel_events(con)
    finally:
        con.close()

    stage_ranges = [item for item in nvtx if item.name in STAGE_RANGE_NAMES]
    layer_ranges = [
        item
        for item in nvtx
        if item.name not in IGNORE_LAYER_RANGE_NAMES and item.duration > 0
    ]
    if not stage_ranges:
        stage_ranges = [NvtxRange(start=0, end=2**63 - 1, name="EDGEFM_GENERATE", global_tid=0)]
    stage_indexes = build_range_indexes(stage_ranges)
    stage_global = RangeIndex(stage_ranges)
    layer_indexes = build_range_indexes(layer_ranges)
    return sqlite_path, kernels, runtimes, stage_indexes, stage_global, layer_indexes


def main(argv: list[str]) -> int:
    args = resolve_args(argv)
    export_dir = Path(args.export_dir).expanduser().resolve() if args.export_dir else None

    if args.input:
        trace = discover_trace_target(args.input)
        sqlite_path, kernels, runtimes, stage_indexes, stage_global, layer_indexes = triage_trace(
            trace,
            export_dir=export_dir,
            force_export=args.force_export,
        )
        mapping_hints = build_mapping_hints(kernels, runtimes, stage_indexes, stage_global, layer_indexes)
        per_stage = aggregate_kernels(
            kernels,
            runtimes,
            stage_indexes,
            stage_global,
            layer_indexes,
            mapping_hints=mapping_hints,
        )
        mode = "single-trace"
        trace_paths = {"input": str(trace)}
        sqlite_paths = {"input": str(sqlite_path)}
    else:
        mapping_trace = discover_trace_target(args.mapping_input)
        formal_trace = discover_trace_target(args.formal_input)
        mapping_sqlite, mapping_kernels, mapping_runtimes, mapping_stage_indexes, mapping_stage_global, mapping_layer_indexes = triage_trace(
            mapping_trace,
            export_dir=export_dir,
            force_export=args.force_export,
        )
        formal_sqlite, formal_kernels, formal_runtimes, formal_stage_indexes, formal_stage_global, formal_layer_indexes = triage_trace(
            formal_trace,
            export_dir=export_dir,
            force_export=args.force_export,
        )
        mapping_hints = build_mapping_hints(
            mapping_kernels,
            mapping_runtimes,
            mapping_stage_indexes,
            mapping_stage_global,
            mapping_layer_indexes,
        )
        per_stage = aggregate_kernels(
            formal_kernels,
            formal_runtimes,
            formal_stage_indexes,
            formal_stage_global,
            formal_layer_indexes,
            mapping_hints=mapping_hints,
        )
        mode = "mapping-formal"
        trace_paths = {"mapping": str(mapping_trace), "formal": str(formal_trace)}
        sqlite_paths = {"mapping": str(mapping_sqlite), "formal": str(formal_sqlite)}

    kernel_rows, totals = render_kernel_rows(
        per_stage,
        min_share_pct=args.min_share_pct,
        table_limit=args.kernel_table_limit,
    )
    known_rows = build_known_path_rows(kernel_rows, limit=args.known_table_limit)
    action_rows = build_action_rows(kernel_rows, limit=args.action_table_limit)

    payload = {
        "mode": mode,
        "trace_paths": trace_paths,
        "sqlite_paths": sqlite_paths,
        "kernel_rows": kernel_rows,
        "known_path_rows": known_rows,
        "action_rows": action_rows,
        "stage_totals_ns": dict(totals),
    }

    output_text = json.dumps(payload, indent=2) if args.json else render_markdown(
        mode=mode,
        trace_paths=trace_paths,
        sqlite_paths=sqlite_paths,
        kernel_rows=kernel_rows,
        known_rows=known_rows,
        action_rows=action_rows,
        totals=totals,
    )

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text)
    else:
        sys.stdout.write(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
