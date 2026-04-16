#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for script_path in [SCRIPT_DIR, SCRIPTS_ROOT]:
    if str(script_path) not in sys.path:
        sys.path.insert(0, str(script_path))
from edge_fm_build_paths import prepend_built_python_paths

prepend_built_python_paths(REPO_ROOT)

from operator_table.utils import (
    BASE_CONFIG_DIR,
    build_operator_impl_table_payload,
    resolve_operator_model_name,
    resolve_target_hw_profile,
)
from temp_paths import make_temp_dir
import tune_qwen_attention_decode as tune_attention_decode
import tune_qwen_attention_prefill as tune_attention_prefill
import tune_qwen_cublaslt as tune_cublaslt


LLM_TABLE_PATH = BASE_CONFIG_DIR / "operator_impl_table_llm.json"
VLM_TABLE_PATH = BASE_CONFIG_DIR / "operator_impl_table_vlm.json"

MODEL_SPECS = {
    "llm": OrderedDict(
        [
            ("0.5b", REPO_ROOT / "examples" / "qwen2.5-0.5b-instruct" / "qwen2.5-0.5b-instruct"),
            ("1.5b", REPO_ROOT / "examples" / "qwen2.5-1.5b-instruct" / "qwen2.5-1.5b-instruct"),
            ("3b", REPO_ROOT / "examples" / "qwen2.5-3b-instruct" / "qwen2.5-3b-instruct"),
        ]
    ),
    "vlm": OrderedDict(
        [
            ("0.5b", REPO_ROOT / "examples" / "qwen2.5-vl-0.5b" / "qwen2.5-vl-0.5b"),
            ("3b", REPO_ROOT / "examples" / "qwen2.5-vl-3b-instruct" / "qwen2.5-vl-3b-instruct"),
            ("7b", REPO_ROOT / "examples" / "qwen2.5-vl-7b-instruct" / "qwen2.5-vl-7b-instruct"),
        ]
    ),
}

LINEAR_DECODE_KINDS = ["fused_qkv", "attention_output", "mlp_down", "fused_gate_up", "lm_head"]
LINEAR_PREFILL_KINDS = ["fused_qkv", "attention_output", "mlp_down", "fused_gate_up"]


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def table_state(base_table: dict, records: list[dict]) -> dict:
    state = copy.deepcopy(base_table)
    state["records"] = copy.deepcopy(records)
    return state


def write_table(path: Path, *, base_table: dict, records: list[dict], extra_metadata: dict) -> None:
    payload = build_operator_impl_table_payload(
        records,
        base_table=base_table,
        source_table_path=path,
        generator=__file__,
        extra_metadata=extra_metadata,
    )
    path.write_text(json.dumps(payload, indent=2) + "\n")


def result_path(prefix: str, name: str) -> Path:
    out_dir = make_temp_dir(prefix)
    return out_dir / name


def unique_json_objects(items: list[dict]) -> list[dict]:
    unique: list[dict] = []
    seen = set()
    for item in items:
        key = json.dumps(item, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def find_existing_attention_impl_params(
    records: list[dict],
    *,
    model_name: str,
    hw_profile: str,
    stage: str,
    shape_sig: str,
) -> dict | None:
    for record in records:
        if (
            record.get("model_name") == model_name
            and record.get("hw_profile") == hw_profile
            and record.get("op_kind") == "attention"
            and record.get("stage") == stage
            and record.get("shape_sig") == shape_sig
        ):
            return copy.deepcopy(record.get("impl_params") or {})
    return None


def remove_generic_decode_attention_fallback(
    records: list[dict],
    *,
    model_name: str,
    hw_profile: str,
) -> list[dict]:
    cleaned = []
    for record in records:
        if (
            record.get("model_name") == model_name
            and record.get("hw_profile") == hw_profile
            and record.get("op_kind") == "attention"
            and record.get("stage") == "decode"
            and record.get("shape_sig", "") == ""
            and record.get("impl_id") == "flashinfer_attention_decode_sm80_tuned"
        ):
            continue
        cleaned.append(record)
    return cleaned


def decode_candidate_grid(existing: dict | None) -> list[dict]:
    candidates: list[dict] = [{}]
    if existing is not None:
        candidates.append(existing)

    candidate_families = [
        {
            "min_chunk_size": 64,
            "chunk_alignment": 64,
            "chunk_candidates": [64, 128, 256, 512],
            "no_split_values": [192, 256, 384],
        },
        {
            "min_chunk_size": 128,
            "chunk_alignment": 128,
            "chunk_candidates": [128, 256, 512, 1024],
            "no_split_values": [256, 384, 512],
        },
    ]
    for short_seq_bdz in [3, 4]:
        for long_seq_bdz in [3, 4]:
            for long_seq_threshold in [1024, 1536]:
                for family in candidate_families:
                    for no_split_kv_threshold in family["no_split_values"]:
                        candidates.append(
                            {
                                "short_seq_bdz": short_seq_bdz,
                                "long_seq_bdz": long_seq_bdz,
                                "long_seq_threshold": long_seq_threshold,
                                "no_split_kv_threshold": no_split_kv_threshold,
                                "min_chunk_size": family["min_chunk_size"],
                                "chunk_alignment": family["chunk_alignment"],
                                "chunk_candidates": family["chunk_candidates"],
                            }
                        )
    return unique_json_objects(candidates)


def attention_prefill_candidate_grid(existing: dict | None) -> list[dict]:
    candidates: list[dict] = [{}]
    if existing is not None:
        candidates.append(existing)
    for cta_tile_q in [16, 64, 128]:
        candidates.append({"prefill_cta_tile_q": cta_tile_q})
    return unique_json_objects(candidates)


def tune_attention_prefill_for_model(
    *,
    model_path: Path,
    device_id: int,
    hw_profile: str,
    base_table: dict,
    source_table_path: Path,
    records: list[dict],
    warmup: int,
    iters: int,
    seq_lens: list[int],
) -> tuple[list[dict], dict]:
    dims = tune_attention_prefill.load_model_attention_dims(model_path)
    operator_model_name = resolve_operator_model_name(model_path=model_path)
    shape_sig = tune_attention_prefill.attention_shape_sig(dims)
    existing = find_existing_attention_impl_params(
        records, model_name=operator_model_name, hw_profile=hw_profile, stage="prefill", shape_sig=shape_sig
    )

    candidates = []
    for impl_params in attention_prefill_candidate_grid(existing):
        report = tune_attention_prefill.benchmark_candidate(
            model_path=model_path,
            operator_model_name=operator_model_name,
            dims=dims,
            base_records=records,
            impl_params=impl_params,
            seq_lens=seq_lens,
            device_id=device_id,
            warmup=warmup,
            iters=iters,
            hw_profile=hw_profile,
        )
        candidates.append(report)

    best = min(candidates, key=lambda item: item["total_median_ms"])
    new_records = tune_attention_prefill.build_tuned_records(
        records,
        operator_model_name=operator_model_name,
        hw_profile=hw_profile,
        dims=dims,
        impl_params=best["impl_params"],
    )
    return new_records, {
        "op_kind": "attention",
        "stage": "prefill",
        "shape_sig": shape_sig,
        "model_path": str(model_path),
        "existing_impl_params": existing,
        "best": best,
        "candidates": candidates,
    }


def tune_attention_decode_for_model(
    *,
    model_path: Path,
    device_id: int,
    hw_profile: str,
    base_table: dict,
    source_table_path: Path,
    records: list[dict],
    warmup: int,
    iters: int,
    kv_lens: list[int],
) -> tuple[list[dict], dict]:
    dims = tune_attention_decode.load_model_attention_dims(model_path)
    operator_model_name = resolve_operator_model_name(model_path=model_path)
    shape_sig = tune_attention_decode.attention_shape_sig(dims)
    existing = find_existing_attention_impl_params(
        records, model_name=operator_model_name, hw_profile=hw_profile, stage="decode", shape_sig=shape_sig
    )

    candidates = []
    for impl_params in decode_candidate_grid(existing):
        report = tune_attention_decode.benchmark_candidate(
            model_path=model_path,
            operator_model_name=operator_model_name,
            dims=dims,
            base_records=records,
            impl_params=impl_params,
            kv_lens=kv_lens,
            device_id=device_id,
            warmup=warmup,
            iters=iters,
            hw_profile=hw_profile,
        )
        candidates.append(report)

    best = min(candidates, key=lambda item: item["total_median_ms"])
    new_records = tune_attention_decode.build_tuned_records(
        records,
        operator_model_name=operator_model_name,
        hw_profile=hw_profile,
        dims=dims,
        impl_params=best["impl_params"],
    )
    new_records = remove_generic_decode_attention_fallback(
        new_records,
        model_name=operator_model_name,
        hw_profile=hw_profile,
    )
    return new_records, {
        "op_kind": "attention",
        "stage": "decode",
        "shape_sig": shape_sig,
        "model_path": str(model_path),
        "existing_impl_params": existing,
        "best": best,
        "candidates": candidates,
    }


def current_linear_record(
    records: list[dict],
    *,
    operator_model_name: str,
    hw_profile: str,
    kind: str,
    stage: str,
    shape_sig: str,
) -> dict | None:
    for record in records:
        if (
            record.get("model_name") == operator_model_name
            and record.get("hw_profile") == hw_profile
            and record.get("op_kind") == "linear"
            and record.get("layer_role") == tune_cublaslt.LAYER_ROLE_BY_KIND[kind]
            and record.get("stage") == stage
            and record.get("shape_sig") == shape_sig
        ):
            return copy.deepcopy(record)
    return None


def tune_linear_shape(
    *,
    model_path: Path,
    device_id: int,
    hw_profile: str,
    base_table: dict,
    source_table_path: Path,
    records: list[dict],
    kind: str,
    stage: str,
    m: int,
    warmup: int,
    iters: int,
) -> tuple[list[dict], dict]:
    dims = tune_cublaslt.load_model_dims(model_path)
    operator_model_name = resolve_operator_model_name(model_path=model_path)
    shape_sig = tune_cublaslt.shape_sig_for(kind, m=m, dims=dims)
    existing = current_linear_record(
        records,
        operator_model_name=operator_model_name,
        hw_profile=hw_profile,
        kind=kind,
        stage=stage,
        shape_sig=shape_sig,
    )

    baseline = tune_cublaslt.benchmark_candidate(
        model_path=model_path,
        dims=dims,
        base_records=records,
        operator_model_name=operator_model_name,
        hw_profile=hw_profile,
        kind=kind,
        stage=stage,
        m=m,
        impl_params=None,
        device_id=device_id,
        warmup=warmup,
        iters=iters,
    )
    heuristic_candidate_count = int(baseline["debug"].get("heuristic_candidate_count") or 0)

    candidates = [baseline]
    for algo_index in range(heuristic_candidate_count):
        candidates.append(
            tune_cublaslt.benchmark_candidate(
                model_path=model_path,
                dims=dims,
                base_records=records,
                operator_model_name=operator_model_name,
                hw_profile=hw_profile,
                kind=kind,
                stage=stage,
                m=m,
                impl_params={"algo_index": algo_index},
                device_id=device_id,
                warmup=warmup,
                iters=iters,
            )
        )

    best = min(candidates, key=lambda item: item["median_ms"])
    new_records = tune_cublaslt.build_tuned_records(
        records,
        operator_model_name=operator_model_name,
        hw_profile=hw_profile,
        kind=kind,
        stage=stage,
        m=m,
        dims=dims,
        impl_params=best["impl_params"],
    )
    return new_records, {
        "op_kind": "linear",
        "layer_kind": kind,
        "stage": stage,
        "m": m,
        "shape_sig": shape_sig,
        "model_path": str(model_path),
        "existing_record": existing,
        "best": best,
        "candidates": candidates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retune Qwen operator_impl tables for the current CUDA platform")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--families", default="llm,vlm")
    parser.add_argument("--llm-models", default="0.5b,1.5b,3b")
    parser.add_argument("--vlm-models", default="0.5b,3b,7b")
    parser.add_argument("--prefill-list", default="512,1024,2048")
    parser.add_argument("--kv-lens", default="512,1024,2048")
    parser.add_argument("--attention-warmup", type=int, default=20)
    parser.add_argument("--attention-iters", type=int, default=100)
    parser.add_argument("--linear-warmup", type=int, default=10)
    parser.add_argument("--linear-iters", type=int, default=60)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--hw-profile", default="", help="Target runtime hw_profile, defaults to current platform")
    parser.add_argument("--skip-attention-prefill", action="store_true")
    parser.add_argument("--skip-attention-decode", action="store_true")
    parser.add_argument("--skip-linear", action="store_true")
    parser.add_argument("--skip-linear-decode", action="store_true")
    parser.add_argument("--skip-linear-prefill", action="store_true")
    return parser.parse_args()


def run_family(
    *,
    family: str,
    model_keys: list[str],
    device_id: int,
    hw_profile: str,
    prefill_list: list[int],
    kv_lens: list[int],
    attention_warmup: int,
    attention_iters: int,
    linear_warmup: int,
    linear_iters: int,
    output_dir: Path | None,
    skip_attention_prefill: bool,
    skip_attention_decode: bool,
    skip_linear: bool,
    skip_linear_decode: bool,
    skip_linear_prefill: bool,
) -> dict:
    source_table_path = LLM_TABLE_PATH if family == "llm" else VLM_TABLE_PATH
    table_path = source_table_path if output_dir is None else output_dir / source_table_path.name
    base_table = load_json(source_table_path)
    records = copy.deepcopy(base_table["records"])
    family_report = {
        "family": family,
        "table_path": str(table_path),
        "source_table_path": str(source_table_path),
        "models": [],
    }

    for model_key in model_keys:
        model_path = MODEL_SPECS[family][model_key].resolve()
        print(f"[retune] family={family} model={model_key} path={model_path}", flush=True)
        model_report = {
            "model_key": model_key,
            "model_path": str(model_path),
            "steps": [],
        }

        if not skip_attention_prefill:
            records, step_report = tune_attention_prefill_for_model(
                model_path=model_path,
                device_id=device_id,
                hw_profile=hw_profile,
                base_table=base_table,
                source_table_path=source_table_path,
                records=records,
                warmup=attention_warmup,
                iters=attention_iters,
                seq_lens=prefill_list,
            )
            model_report["steps"].append(step_report)
            print(
                f"  [attention-prefill] best={step_report['best']['candidate_label']} "
                f"avg_median_ms={step_report['best']['avg_median_ms']:.6f}",
                flush=True,
            )

        if not skip_attention_decode:
            records, step_report = tune_attention_decode_for_model(
                model_path=model_path,
                device_id=device_id,
                hw_profile=hw_profile,
                base_table=base_table,
                source_table_path=source_table_path,
                records=records,
                warmup=attention_warmup,
                iters=attention_iters,
                kv_lens=kv_lens,
            )
            model_report["steps"].append(step_report)
            print(
                f"  [attention-decode] best_params={json.dumps(step_report['best']['impl_params'], sort_keys=True)} "
                f"avg_median_ms={step_report['best']['avg_median_ms']:.6f}",
                flush=True,
            )

        if not skip_linear:
            linear_plan = []
            if not skip_linear_decode:
                linear_plan.extend((kind, "decode", 1) for kind in LINEAR_DECODE_KINDS)
            if not skip_linear_prefill:
                for m in prefill_list:
                    for kind in LINEAR_PREFILL_KINDS:
                        linear_plan.append((kind, "prefill", m))
            for kind, stage, m in linear_plan:
                records, step_report = tune_linear_shape(
                    model_path=model_path,
                    device_id=device_id,
                    hw_profile=hw_profile,
                    base_table=base_table,
                    source_table_path=source_table_path,
                    records=records,
                    kind=kind,
                    stage=stage,
                    m=m,
                    warmup=linear_warmup,
                    iters=linear_iters,
                )
                model_report["steps"].append(step_report)
                candidate_name = (
                    "baseline" if step_report["best"]["algo_index"] is None
                    else f"algo_{step_report['best']['algo_index']}"
                )
                print(
                    f"  [linear] stage={stage} kind={kind} m={m} "
                    f"best={candidate_name} median_ms={step_report['best']['median_ms']:.6f}",
                    flush=True,
                )

        family_report["models"].append(model_report)

    extra_metadata = {
        "retune_session": {
            "generator": "scripts/tune/retune_qwen_operator_tables.py",
            "device_id": device_id,
            "hw_profile": hw_profile,
            "families": [family],
            "prefill_list": prefill_list,
            "kv_lens": kv_lens,
            "attention_warmup": attention_warmup,
            "attention_iters": attention_iters,
            "linear_warmup": linear_warmup,
            "linear_iters": linear_iters,
            "skip_linear_decode": skip_linear_decode,
            "skip_linear_prefill": skip_linear_prefill,
            "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    }
    write_table(table_path, base_table=base_table, records=records, extra_metadata=extra_metadata)
    family_report["updated_table_path"] = str(table_path)
    family_report["final_record_count"] = len(records)
    return family_report


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.device_id)
    hw_profile = resolve_target_hw_profile(args.hw_profile)

    families = parse_csv(args.families)
    prefill_list = [int(item) for item in parse_csv(args.prefill_list)]
    kv_lens = [int(item) for item in parse_csv(args.kv_lens)]
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    family_to_models = {
        "llm": parse_csv(args.llm_models),
        "vlm": parse_csv(args.vlm_models),
    }

    report = {
        "started_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device_id": args.device_id,
        "hw_profile": hw_profile,
        "families": {},
    }
    for family in families:
        if family not in MODEL_SPECS:
            raise ValueError(f"Unsupported family: {family}")
        selected_models = family_to_models[family]
        unsupported = [model_key for model_key in selected_models if model_key not in MODEL_SPECS[family]]
        if unsupported:
            raise ValueError(f"Unsupported {family} model keys: {unsupported}")
        report["families"][family] = run_family(
            family=family,
            model_keys=selected_models,
            device_id=args.device_id,
            hw_profile=hw_profile,
            prefill_list=prefill_list,
            kv_lens=kv_lens,
            attention_warmup=args.attention_warmup,
            attention_iters=args.attention_iters,
            linear_warmup=args.linear_warmup,
            linear_iters=args.linear_iters,
            output_dir=output_dir,
            skip_attention_prefill=args.skip_attention_prefill,
            skip_attention_decode=args.skip_attention_decode,
            skip_linear=args.skip_linear,
            skip_linear_decode=args.skip_linear_decode,
            skip_linear_prefill=args.skip_linear_prefill,
        )

    report["completed_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report_path = result_path(
        "efm_retune_qwen_tables_",
        f"retune_report_device{args.device_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json",
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report_path": str(report_path), "families": list(report["families"].keys())}, indent=2))


if __name__ == "__main__":
    main()
