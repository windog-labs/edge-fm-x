#!/usr/bin/env python3
"""Materialize a declared native host pipeline and hash-bound raw observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_profile(profile, inputs, samples):
    import numpy as np

    if profile.get("schema") != "vlaforge.native_host_profile/1" or type(samples) is not int or samples <= 0:
        raise ValueError("explicit native host profile and positive sample count required")
    dtypes = {"uint8": "u8", "float32": "f32", "float64": "f64",
              "int32": "i32", "int64": "i64", "bool": "bool"}

    def tensor(dtype, shape):
        if dtype not in dtypes.values() or not shape or any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError("unsupported dtype or host tensor shape")
        if math.prod(shape) > (1 << 31):
            raise ValueError("host tensor exceeds bounded profile")
        return '{VLAFORGE_DTYPE_' + dtype.upper() + ', {' + ','.join(map(str, shape)) + '}}'

    outputs = {port["name"]: (index, port["payload"]) for index, port in enumerate(inputs)}
    if len(outputs) != len(inputs) or any(port["id"] != index or port["payload"]["kind"] != "tensor" for index, port in enumerate(inputs)):
        raise ValueError("dense unique tensor input IDs required")
    raw = profile["raw_tensors"]
    names = [item["name"] for item in raw]
    texts = profile["texts"]
    if len(set(names)) != len(names) or len(set(texts)) != len(texts):
        raise ValueError("raw/text sources must be unique")
    raw_specs = [tensor(dtypes[item["dtype"]], item["shape"]) for item in raw]
    output_specs = [tensor(port["payload"]["dtype"], port["payload"]["shape"]) for port in inputs]
    covered, operations = set(), []

    def target(name, dtype):
        if name not in outputs or name in covered or outputs[name][1]["dtype"] != dtype:
            raise ValueError("each correctly typed target must be assigned exactly once")
        covered.add(name)
        index, spec = outputs[name]
        return index, math.prod(spec["shape"])

    def literal(value):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("finite numeric FLOAT parameter required")
        with np.errstate(over="ignore"):
            narrowed = np.float32(value)
        if not np.isfinite(narrowed):
            raise ValueError("FLOAT parameter overflows")
        text = format(float(narrowed), ".9g")
        return text + ("F" if any(c in text for c in ".eE") else ".0F")

    def vector(values):
        return "{" + ",".join(literal(value) for value in values) + "}"

    def rounding(op):
        if op.get("rounding") not in ("none", "bf16-rne-f32"):
            raise ValueError("explicit native carrier rounding policy required")
        return str(op["rounding"] == "bf16-rne-f32").lower()

    for op in profile["operations"]:
        kind = op["kind"]
        if kind == "tokenize":
            index, count = target(op["output"], "i32")
            mask_dtype = outputs.get(op["mask"], (None, {}))[1].get("dtype")
            if mask_dtype not in ("bool", "i32"):
                raise ValueError("token mask must be explicitly BOOL or INT32")
            mask, mask_count = target(op["mask"], mask_dtype)
            if (type(op["max_length"]) is not int or op["max_length"] != count or mask_count != count
                    or type(op["pad_id"]) is not int or not 0 <= op["pad_id"] <= 2147483647
                    or type(op["add_special_tokens"]) is not bool or type(op["truncate"]) is not bool
                    or op["padding_side"] not in ("left", "right") or op["truncation_side"] not in ("left", "right")
                    or not isinstance(op["suffix"], str)):
                raise ValueError("invalid explicit tokenization/padding policy")
            op["suffix"].encode("utf-8")
            operations.append(f'vlaforge::tools::TokenizeText{{{texts.index(op["source"])}, {index}, {mask}, {op["pad_id"]}, '
                + f'{str(op["add_special_tokens"]).lower()}, {str(op["truncate"]).lower()}, '
                + f'vlaforge::runtime::SequenceSide::k{op["padding_side"].title()}, '
                + f'vlaforge::runtime::SequenceSide::k{op["truncation_side"].title()}, {json.dumps(op["suffix"])}}}')
        elif kind == "scatter-divide-f32":
            source = names.index(op["source"])
            raw_spec = raw[source]
            index, count = target(op["output"], "f32")
            shape = outputs[op["output"]][1]["shape"]
            indices, divisors = op["indices"], op["divisors"]
            if (raw_spec["dtype"] != "float32" or not indices or len(indices) != len(divisors)
                    or raw_spec["shape"][-1] != len(indices) or len(set(indices)) != len(indices)
                    or any(type(value) is not int or not 0 <= value < shape[-1] for value in indices)
                    or count != math.prod(raw_spec["shape"][:-1]) * shape[-1]):
                raise ValueError("invalid explicit scatter shape or indices")
            values = vector(divisors)
            if any(float(np.float32(value)) == 0 for value in divisors):
                raise ValueError("scatter divisors must remain nonzero in FLOAT")
            mask = "std::nullopt"
            if op.get("mask") is not None:
                mask_index, mask_count = target(op["mask"], "f32")
                if mask_count != count or outputs[op["mask"]][1]["shape"][-1] != shape[-1]:
                    raise ValueError("scatter mask must match the declared output rows")
                mask = str(mask_index)
            operations.append(f'vlaforge::tools::ScatterDivide{{{source}, {index}, {shape[-1]}, {mask}, '
                + "{" + ",".join(map(str, indices)) + "}, " + values + ", " + rounding(op) + "}")
        elif kind == "images-u8-to-chw-f32":
            sources = op["sources"]
            index, count = target(op["output"], "f32")
            shape = outputs[op["output"]][1]["shape"]
            width, height = op["width"], op["height"]
            if (not sources or len(set(sources)) != len(sources) or len(shape) < 3 or shape[-3] != 3
                    or type(width) is not int or type(height) is not int or width <= 0 or height <= 0
                    or shape[-2:] != [height, width] or count != len(sources) * width * height * 3
                    or type(op["pad_to_square"]) is not bool or op["filter"] not in ("bilinear", "bicubic", "lanczos")
                    or len(op["background"]) != 3 or any(type(value) is not int or not 0 <= value <= 255 for value in op["background"])
                    or len(op["mean"]) != 3 or len(op["stddev"]) != 3
                    or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in op["stddev"])
                    or type(op["rescale"]) not in (int, float) or not math.isfinite(op["rescale"]) or op["rescale"] <= 0):
                raise ValueError("invalid explicit image layout, filter or normalization")
            source_ids, sizes = [], []
            for name in sources:
                position = names.index(name)
                declaration = raw[position]
                if declaration["dtype"] != "uint8" or len(declaration["shape"]) != 3 or declaration["shape"][-1] != 3:
                    raise ValueError("image source must be a contiguous three-byte pixel tensor")
                source_ids.append(str(position))
                sizes.append("{" + str(declaration["shape"][1]) + "," + str(declaration["shape"][0]) + "}")
            rescale = format(float(op["rescale"]), ".17g")
            if not any(character in rescale for character in ".eE"):
                rescale += ".0"
            operations.append('vlaforge::tools::PrepareImages{{' + ",".join(source_ids) + f'}}, {index}, '
                + "{" + ",".join(sizes) + "}, "
                + f'{{{width}, {height}, {str(op["pad_to_square"]).lower()}, '
                + "{" + ",".join(map(str, op["background"])) + "}, "
                + 'vlaforge::backends::ImageFilter::k' + op["filter"].title() + "}, " + rescale + ", "
                + vector(op["mean"]) + ", " + vector(op["stddev"]) + ", " + rounding(op) + "}")
        elif kind == "fill-bool":
            index, _ = target(op["output"], "bool")
            if type(op["value"]) is not bool:
                raise ValueError("constant BOOL must be explicit")
            operations.append(f'vlaforge::tools::FillBool{{{index}, {str(op["value"]).lower()}}}')
        else:
            source = names.index(op["source"])
            raw_spec = raw[source]
            dtype = dtypes[raw_spec["dtype"]]
            destination_dtype = "f32" if kind in ("scale-u8", "standardize-f32") else dtype
            index, count = target(op["output"], destination_dtype)
            if count != math.prod(raw_spec["shape"]):
                raise ValueError("native host operation changes element count")
            if kind == "scale-u8":
                if dtype != "u8" or type(op["divisor"]) not in (int, float) or op["divisor"] <= 0:
                    raise ValueError("U8 scaling requires a positive divisor")
                operations.append(f'vlaforge::tools::ScaleU8{{{source}, {index}, {literal(op["divisor"])}}}')
            elif kind == "standardize-f32":
                if (dtype != "f32" or not op["mean"] or len(op["mean"]) != len(op["stddev"])
                        or count % len(op["mean"]) or type(op["epsilon"]) not in (int, float) or op["epsilon"] < 0
                        or any(type(value) not in (int, float) or value < 0 for value in op["stddev"])):
                    raise ValueError("invalid explicit FLOAT statistics")
                operations.append(f'vlaforge::tools::StandardizeF32{{{source}, {index}, '
                                  + vector(op["mean"]) + ", " + vector(op["stddev"]) + ", " + literal(op["epsilon"]) + "}")
            elif kind == "copy":
                operations.append(f'vlaforge::tools::CopyTensor{{{source}, {index}}}')
            else:
                raise ValueError("unsupported native host operation")
    if covered != set(outputs):
        raise ValueError("native host plan leaves model inputs unassigned")
    has_images = any(op["kind"] == "images-u8-to-chw-f32" for op in profile["operations"])
    return ('#pragma once\n#define VLAFORGE_NATIVE_PROFILE_HAS_IMAGES ' + str(int(has_images))
            + '\n#include "declared_host_pipeline.h"\n'
            + 'const std::vector<vlaforge::tools::HostTensorSpec> raw_specs = {' + ",\n".join(raw_specs) + '};\n'
            + 'const std::vector<vlaforge::tools::HostTensorSpec> output_specs = {' + ",\n".join(output_specs) + '};\n'
            + 'const std::vector<vlaforge::tools::HostOperation> operations = {' + ",\n".join(operations) + '};\n'
            + f'constexpr std::size_t sample_count = {samples};\nconstexpr std::size_t text_count = {len(texts)};\n')


def main():
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--model-ir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    if digest(args.raw_manifest) != profile["raw_manifest_sha256"] or digest(args.model_ir) != profile["model_ir_sha256"]:
        raise ValueError("raw manifest/model interface identity differs")
    for asset in profile["assets"].values():
        if digest(Path(asset["path"])) != asset["sha256"]:
            raise ValueError("host pipeline asset identity differs")
    raw = json.loads(args.raw_manifest.read_text())
    module = json.loads(args.model_ir.read_text())
    header = render_profile(profile, module["inputs"], len(raw["records"]))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "host_profile.h").write_text(header)
    template = Path(__file__).with_name("native_observation_probe.cpp.in")
    (args.output / "probe.cpp").write_bytes(template.read_bytes())
    helper = Path(__file__).with_name("declared_host_pipeline.h")
    (args.output / helper.name).write_bytes(helper.read_bytes())
    records = []
    for index, item in enumerate(raw["records"]):
        source = (args.raw_manifest.parent / item["path"]).resolve(strict=True)
        if not source.is_relative_to(args.raw_manifest.parent.resolve()) or digest(source) != item["sha256"] or item["index"] != index:
            raise ValueError("raw observation identity/order differs")
        folder = args.output / "raw" / str(index)
        folder.mkdir(parents=True)
        with np.load(source, allow_pickle=False) as arrays:
            for position, declaration in enumerate(profile["raw_tensors"]):
                value = arrays[declaration["name"]]
                expected = item["arrays"][declaration["name"]]
                if (list(value.shape) != declaration["shape"] or str(value.dtype) != declaration["dtype"]
                        or hashlib.sha256(value.tobytes()).hexdigest() != expected["sha256"] or not np.isfinite(value).all()):
                    raise ValueError("raw tensor contents/type/shape differ")
                np.ascontiguousarray(value).tofile(folder / (str(position) + ".bin"))
        for position, name in enumerate(profile["texts"]):
            (folder / ("text-" + str(position) + ".txt")).write_bytes(item["text"][name].encode("utf-8"))
        records.append({"sample": index, "files": {path.name: digest(path) for path in folder.iterdir()}})
    report = {"schema": "edgefm.native_observation_preparation/1", "status": "prepared-not-executed",
              "scope": "native input preparation only; not model benchmark", "profile_sha256": digest(args.profile),
              "raw_manifest_sha256": digest(args.raw_manifest), "model_ir_sha256": digest(args.model_ir),
              "script_sha256": digest(__file__), "generated_files": {name: digest(args.output / name)
                  for name in ("host_profile.h", "probe.cpp", "declared_host_pipeline.h")}, "records": records}
    (args.output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "samples": len(records), "header_sha256": digest(args.output / "host_profile.h")}))


if __name__ == "__main__":
    main()
