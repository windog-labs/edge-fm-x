# Qwen2 Generate Correctness Debug Log

This document is the active record for `tests/engine/test_qwen2_generate.py`
correctness work. Keep it current: every experiment below records the build,
module path, command, result, and the current conclusion. Obsolete hypotheses
are removed or moved to the excluded section.

## Current Status

- Acceptance scope: Qwen2.5 LLM correctness plus Qwen2.5-VL 0.5B correctness.
- Platform: RTX 3060 / `build-3060` / `horizon_quant`.
- Current commit: `6d31130041c321d48f5479343934450d49ffcb1d`.
- Trusted Python module:
  `/home/zhangzimo/Repos/private/edge-fm-x/build-3060/install/python/edge_fm.cpython-310-x86_64-linux-gnu.so`
- Latest trusted install timestamp:
  `2026-04-27 11:03:28 +0800`
- LLM no-graph correctness: passing.
- LLM CUDA graph token alignment: passing.
- VLM 0.5B no-graph token alignment: passing.
- VLM 0.5B CUDA graph token alignment: passing.
- Supporting fused SwiGLU/KVManager/attention prefill tests: passing where
  runnable.
- Reduced LLM benchmark smoke: passing.

## Root Cause

The LLM decode divergence was caused by the TensorRT-LLM SM80 fused-MoE SwiGLU
kernel being used by default on RTX 3060 SM86 during the decode-only
`gate_up + SwiGLU` fast path.

Evidence:

- The failing sequence diverged at generated token index 3:
  `edge_fm=14589`, `ref=3730`.
- Generate trace showed correct token handoff and KV length protocol through the
  divergence point: `30 -> 358 -> 2776`, `decode_cache_kv_len=9`, `d_kv_len=9`.
- Full-prefill EdgeFM on appended reference prefixes matched Transformers,
  which localized the issue to the decode-only execution path.
- Temporarily disabling the fused decode SwiGLU path restored exact LLM token
  alignment before the production guard fix.
- Layer-level diagnostics under `cuda_sm86` showed decode fused SwiGLU mismatches
  against the two-stage BF16 `gate/up + SiLU` path across 1.5B layers.

Fix:

- `src/operators/fused_gate_up_activation_op.cu` now enables the TensorRT-LLM
  SM80 fused decode SwiGLU path only on actual SM80.
- Temporary generate trace and environment-gated diagnostic overrides were
  removed after the root cause was confirmed.
- `tests/operators/test_fused_gate_up_activation.py` now covers the SM86 default
  guard so this path does not silently re-enable on RTX 3060.
- `tests/scripts/dump_qwen2_5_vl_decode.py` now supports the local 0.5B
  `llava`-wrapped checkpoint by rendering chat text from the tokenizer fallback,
  remapping checkpoint keys into Transformers `LlavaForConditionalGeneration`,
  and omitting `position_ids.npy` for the non-M-RoPE Llava path.

## Build Log

### Trusted current-HEAD build

Command:

```bash
source /home/zhangzimo/miniconda3/bin/activate horizon_quant
cmake --preset 3060 -DPython_EXECUTABLE=$CONDA_PREFIX/bin/python > /tmp/edgefm_config_3060_cleanup.log 2>&1
cmake --build --preset 3060 --target edge_fm_python --parallel 2 > /tmp/edgefm_build_3060_cleanup.log 2>&1
cmake --install build-3060 >> /tmp/edgefm_build_3060_cleanup.log 2>&1
```

Result:

- Build and install completed.
- Installed Python module timestamp:
  `2026-04-27 11:03:28 +0800`.
- Installed `libedge_fm.so` timestamp:
  `2026-04-27 11:03:27 +0800`.
- Installed library contains the new guard string:
  `decode fused SwiGLU path only enables TensorRT-LLM's SM80 kernel on SM80 by default`.
- Installed library no longer contains the removed diagnostic trace or SwiGLU
  override environment strings.

Conclusion:

- Subsequent results are valid for the current patched build.

## Experiment Log

### 2026-04-27: Initial stale binary check

Command:

```bash
git rev-parse HEAD
stat -c '%y %n' build-3060/install/python/edge_fm*.so build-3060/python/edge_fm*.so
```

Result:

- HEAD: `6d31130041c321d48f5479343934450d49ffcb1d`.
- Initial installed Python extension timestamp was `2026-04-22 14:22:06 +0800`.
- Recent refactor commits were from `2026-04-26`.

Conclusion:

- Earlier failures against the `2026-04-22` extension were not trusted evidence.

### 2026-04-27: LLM reference dump verification

Command:

```bash
source /home/zhangzimo/miniconda3/bin/activate horizon_quant
python - <<'PY'
from pathlib import Path
import numpy as np
from tests.engine.test_qwen2_generate import _generate_dump
root = Path('/home/zhangzimo/Repos/private/edge-fm-x')
model = root / 'examples/qwen2.5-1.5b-instruct/qwen2.5-1.5b-instruct'
out = root / '.tmp_codex/qwen2_generate/llm_dump_check'
_generate_dump(str(model), out)
old = np.load(root / 'tests/data/decode_dump/decode_tokens.npy')
new = np.load(out / 'decode_tokens.npy')
print(np.array_equal(old, new))
PY
```

Result:

- Existing and regenerated `decode_tokens.npy` matched exactly.
- Reference tokens:
  `[30, 358, 2776, 3730, 2244, 11, 9702, 498, 0, 2585, 646, 358, 7789, 498, 3351, 30, 358, 2776, 3330, 369, 1045]`

Conclusion:

- LLM dump under `tests/data/decode_dump` is trusted.

### 2026-04-27: Direct LLM generate after fix

Command:

```bash
source /home/zhangzimo/miniconda3/bin/activate horizon_quant
export EDGE_FM_BUILD_DIR=$PWD/build-3060
export PYTHONPATH=$PWD/build-3060/install/python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$PWD/build-3060/install/lib:$PWD/build-3060/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
python - <<'PY'
import json
import numpy as np
import edge_fm
from tests.engine.test_qwen2_generate import DUMP_DIR, _create_engine_config
manifest = json.loads((DUMP_DIR / 'manifest.json').read_text())
token_ids = np.load(DUMP_DIR / 'token_ids.npy').flatten().tolist()
ref = np.load(DUMP_DIR / 'decode_tokens.npy').tolist()
engine = edge_fm.EdgeFM(_create_engine_config(manifest['model_path'], len(token_ids), manifest['num_decode_steps']))
got = engine.generate(edge_fm.Request(0, token_ids)).token_ids()
print(got)
print([(i, got[i], ref[i]) for i in range(min(len(got), len(ref))) if got[i] != ref[i]])
PY
```

Result:

- Imported module:
  `/home/zhangzimo/Repos/private/edge-fm-x/build-3060/install/python/edge_fm.cpython-310-x86_64-linux-gnu.so`.
- Generated tokens matched the trusted reference exactly.
- Mismatch list: `[]`.

Conclusion:

- The code fix restores direct LLM generate correctness without any diagnostic
  override.

### 2026-04-27: LLM no-graph pytest correctness

Command:

```bash
source /home/zhangzimo/miniconda3/bin/activate horizon_quant
EDGE_FM_BUILD_DIR=$PWD/build-3060 \
PYTHONPATH=$PWD/build-3060/install/python:${PYTHONPATH:-} \
LD_LIBRARY_PATH=$PWD/build-3060/install/lib:$PWD/build-3060/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-} \
python -m pytest -s tests/engine/test_qwen2_generate.py \
  -k 'test_generate_token_alignment or test_generate_logits_cosine_similarity or test_generate_checkpoint_alignment'
```

Result:

- `7 passed, 5 deselected in 14.71s`.

Conclusion:

- LLM no-graph correctness is passing on the trusted patched build.

### 2026-04-27: LLM CUDA graph pytest correctness

Command:

```bash
source /home/zhangzimo/miniconda3/bin/activate horizon_quant
EDGE_FM_BUILD_DIR=$PWD/build-3060 \
PYTHONPATH=$PWD/build-3060/install/python:${PYTHONPATH:-} \
LD_LIBRARY_PATH=$PWD/build-3060/install/lib:$PWD/build-3060/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-} \
python -m pytest -s tests/engine/test_qwen2_generate.py \
  -k 'test_generate_token_alignment_cuda_graph'
```

Result:

- `1 passed, 11 deselected in 7.44s`.

Conclusion:

- LLM CUDA graph token alignment is passing on the trusted patched build.

### 2026-04-27: Fused SwiGLU guard regression test

Command:

```bash
source /home/zhangzimo/miniconda3/bin/activate horizon_quant
EDGE_FM_BUILD_DIR=$PWD/build-3060 \
PYTHONPATH=$PWD/build-3060/install/python:${PYTHONPATH:-} \
LD_LIBRARY_PATH=$PWD/build-3060/install/lib:$PWD/build-3060/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-} \
python -m pytest -s tests/operators/test_fused_gate_up_activation.py \
  -k 'decode_fused_gate_up_swiglu'
```

Result:

- `1 passed, 1 skipped, 12 deselected in 7.02s`.
- The new non-SM80 Ampere guard test passed on RTX 3060.
- The old output-comparison test skipped because the fast path is intentionally
  unavailable on SM86 by default.

Conclusion:

- The regression test now protects the root-cause fix.

### 2026-04-27: KVManager supporting test

Command:

```bash
source /home/zhangzimo/miniconda3/bin/activate horizon_quant
EDGE_FM_BUILD_DIR=$PWD/build-3060 \
PYTHONPATH=$PWD/build-3060/install/python:${PYTHONPATH:-} \
LD_LIBRARY_PATH=$PWD/build-3060/install/lib:$PWD/build-3060/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-} \
python -m pytest -s tests/engine/test_kvcache.py
```

Result:

- `6 passed in 0.08s`.

Conclusion:

- Standalone Python `KVManager` tests are aligned with the backend-neutral
  default host allocator. `StandardEngine` still injects the CUDA allocator.

### 2026-04-27: Attention supporting tests

Commands:

```bash
source /home/zhangzimo/miniconda3/bin/activate horizon_quant
EDGE_FM_BUILD_DIR=$PWD/build-3060 \
PYTHONPATH=$PWD/build-3060/install/python:${PYTHONPATH:-} \
LD_LIBRARY_PATH=$PWD/build-3060/install/lib:$PWD/build-3060/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-} \
python -m pytest -s tests/operators/test_attention_decode.py \
  -k 'matches_flashinfer_reference or graph_like_path_matches_non_graph'

EDGE_FM_BUILD_DIR=$PWD/build-3060 \
PYTHONPATH=$PWD/build-3060/install/python:${PYTHONPATH:-} \
LD_LIBRARY_PATH=$PWD/build-3060/install/lib:$PWD/build-3060/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-} \
python -m pytest -s tests/operators/test_attention_prefill.py -k correctness
```

Result:

- `test_attention_decode.py`: module skipped at import because `flashinfer` is
  unavailable in the current `horizon_quant` environment; pytest returned code 5
  due to zero runnable items.
- `test_attention_prefill.py`: `3 passed, 3 deselected in 1.14s`.

Conclusion:

- Prefill attention correctness is passing.
- Decode attention pytest was not runnable in this environment; the LLM/VLM
  end-to-end generate tests still exercise decode attention through EdgeFM.

### 2026-04-27: VLM 0.5B dump generation

Command:

```bash
source /home/zhangzimo/miniconda3/bin/activate horizon_quant
python tests/scripts/dump_qwen2_5_vl_decode.py \
  --model_path examples/qwen2.5-vl-0.5b/qwen2.5-vl-0.5b \
  --output_dir .tmp_codex/qwen2_generate/vl_0_5b_dump
```

Result:

- Dump written to `.tmp_codex/qwen2_generate/vl_0_5b_dump`.
- Existing `tests/data/decode_dump_vl` 3B dump was not modified.
- Manifest model type: `llava`.
- `embed_token_id=151665`.
- `image_embeddings_shape=(576, 896)`.
- Reference decode tokens:
  `[3862, 374, 264, 38703, 389, 279, 31556, 13, 151645, 198, 3872, 279, 38703, 389, 279, 31556, 476, 537, 30, 151645, 198]`.

Conclusion:

- VLM 0.5B reference dump is available in an isolated local directory.

### 2026-04-27: VLM 0.5B pytest correctness

Command:

```bash
source /home/zhangzimo/miniconda3/bin/activate horizon_quant
EDGE_FM_BUILD_DIR=$PWD/build-3060 \
PYTHONPATH=$PWD/build-3060/install/python:${PYTHONPATH:-} \
LD_LIBRARY_PATH=$PWD/build-3060/install/lib:$PWD/build-3060/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-} \
EDGE_FM_QWEN_VL_MODEL_PATH=$PWD/examples/qwen2.5-vl-0.5b/qwen2.5-vl-0.5b \
EDGE_FM_QWEN_VL_DUMP_DIR=$PWD/.tmp_codex/qwen2_generate/vl_0_5b_dump \
python -m pytest -s tests/engine/test_qwen2_generate.py -k 'test_generate_vl_token_alignment'
```

Result:

- `2 passed, 10 deselected in 7.55s`.
- No-graph VLM alignment: `20/20` steps aligned.
- CUDA graph VLM alignment: `20/20` steps aligned.

Conclusion:

- VLM 0.5B correctness is passing for both no-graph and CUDA graph decode.

### 2026-04-27: Reduced LLM benchmark smoke

Command:

```bash
source /home/zhangzimo/miniconda3/bin/activate horizon_quant
EDGE_FM_BUILD_DIR=$PWD/build-3060 \
PYTHONPATH=$PWD/build-3060/install/python:${PYTHONPATH:-} \
LD_LIBRARY_PATH=$PWD/build-3060/install/lib:$PWD/build-3060/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-} \
EDGE_FM_BENCH_LLM_MODELS=0.5b \
EDGE_FM_BENCH_PREFILL_LIST=512 \
EDGE_FM_BENCH_DECODE_LIST=32 \
python -m pytest -s tests/engine/test_qwen2_generate.py -k test_benchmark_llm
```

Result:

- `1 passed, 11 deselected in 9.77s`.
- Case: Qwen2.5-0.5B-Instruct, prefill `512`, decode `32`.
- Cleanup rerun: `1 passed, 11 deselected in 8.93s`.
- Transformers total latency: `304.9 ms`.
- EdgeFM CUDA graph total latency: `132.8 ms`.
- Speedup: `2.30x`.
- EdgeFM runs were stable: `[132.8, 132.8, 132.8, 132.8, 132.8] ms`.

Conclusion:

- Reduced performance smoke passed after correctness fixes.

## Excluded Hypotheses

- Stale Python extension caused the current failure: excluded after rebuilding
  current HEAD and reproducing the same LLM mismatch before the SwiGLU fix.
- LLM reference dump was stale: excluded because regenerated Transformers dump
  matched `tests/data/decode_dump/decode_tokens.npy` exactly.
- Token handoff, response buffer indexing, `d_kv_len`, or layer0 write pointer
  caused the divergence: mostly excluded by env-gated generate trace.
- FlashInfer attention dynamic-length decode path caused the divergence:
  excluded by end-to-end LLM/VLM generate passing after only the SwiGLU guard
  change; standalone decode attention pytest was skipped because `flashinfer`
  is unavailable in `horizon_quant`.

## Pending Work

1. Optional: install/repair `flashinfer` in `horizon_quant` if standalone
   `tests/operators/test_attention_decode.py` must be runnable as a separate
   supporting test.
2. Optional: keep `.tmp_codex/qwen2_generate/vl_0_5b_dump` as the local
   non-committed 0.5B VLM reference unless a tracked fixture is explicitly
   requested.
