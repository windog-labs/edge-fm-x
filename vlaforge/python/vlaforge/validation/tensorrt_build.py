"""Fail-closed parsing of recorded TensorRT child-build outcomes.

A collector process can exit successfully after recording a failed trtexec.
These checks validate that receipt only, not engine bytes, execution or fidelity.
"""


def require_successful_trt_build(record, *, source_sha256=None):
    def sha(value):
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

    if (not isinstance(record, dict) or record.get("schema") != "edgefm.orin_trt_build/1"
            or record.get("status") != "built-not-validated"
            or type(record.get("exit_code")) is not int or record["exit_code"] != 0):
        raise ValueError("TensorRT child build did not report successful engine creation")
    if not sha(record.get("source_sha256")) or (source_sha256 is not None and record["source_sha256"] != source_sha256):
        raise ValueError("TensorRT source identity is missing or differs")
    engine = record.get("engine")
    if (not isinstance(engine, dict) or not isinstance(engine.get("path"), str) or not engine["path"]
            or type(engine.get("bytes")) is not int or engine["bytes"] <= 0 or not sha(engine.get("sha256"))):
        raise ValueError("TensorRT build lacks a complete positive-size engine receipt")
