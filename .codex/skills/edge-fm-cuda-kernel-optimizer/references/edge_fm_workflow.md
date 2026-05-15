# edge-fm-x Repo Workflow

当目标 kernel 在 `edge-fm-x` 仓库内部，而不是一个可直接 `solve(...)` benchmark 的独立文件时，先按这里做热点定位、隔离和回归。

## 常见目标文件

- `src/operators/*.cu`
  典型是 operator 级 CUDA 实现，如 linear / attention / norm / activation
- `src/layers/*.cu`
  典型是 layer 级组合逻辑或封装
- `src/utils/device/*.cu`
  典型是 runtime kernel、decode/prefill 辅助 kernel
- `src/tuning/*.cpp`
  可能涉及 launch config、operator tuner 或策略逻辑

## 常见测试映射

- `src/operators/attention_op.cu`
  优先看 `tests/operators/test_attention_decode.py` 和 `tests/operators/test_attention_prefill.py`
- `src/layers/attention.cu`
  优先看 `tests/layers/test_attn.py`
- `src/operators/linear_impl.cu`
  优先看 `tests/operators/test_decode_linear.py` 和 `tests/operators/test_prefill_linear.py`
- `src/layers/linear.cu`
  优先看 `tests/layers/test_linear.py`
- `src/operators/norm_op.cu`
  优先看 `tests/operators/test_norm_sampler.py`
- `src/layers/layernorm.cu`
  优先看 `tests/layers/test_layernorm.py`
- `src/operators/activation_op.cu`
  优先看 `tests/layers/test_activation.py`
- `src/operators/fused_gate_up_activation_op.cu`
  优先看 `tests/operators/test_fused_gate_up_activation.py`
- `src/utils/device/decode_runtime_kernels.cu`
  往往需要 `tests/engine/test_qwen2_generate.py` 或自定义 decode repro

如果一个实现同时被 layer 测试和 engine 测试覆盖，先跑 operator / layer 级入口，再决定是否需要 engine 级验证。

## 热点定位顺序

### 1. 先确认是不是目标 kernel 真慢

优先用仓库现有测试或脚本做 NCU：

```bash
ncu --set full --target-processes all -o ncu_reports/attn_profile \
  python -m pytest tests/layers/test_attn.py -v
```

或使用仓库脚本：

```bash
python tests/scripts/profile_edgefm.py
python scripts/profile/profile_operator_comparison.py
python scripts/profile/profile_edgefm_generate_case.py
```

需要 NSYS attribution 时，优先采 graph-off mapping trace；CUDA graph 最终行为再补 graph-on formal trace：

```bash
nsys profile -o .tmp_codex/nsys/edgefm_mapping \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  python3 scripts/profile/profile_edgefm_generate_case.py \
    --model-path /path/to/model --prefill-len 1024 --decode-len 32 --profile-range

nsys profile -o .tmp_codex/nsys/edgefm_formal \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  python3 scripts/profile/profile_edgefm_generate_case.py \
    --model-path /path/to/model --prefill-len 1024 --decode-len 32 \
    --use-cuda-graph --profile-range
```

VLM 用 `scripts/profile/profile_vlm_prepared_case.py`，让 ViT 留在 profiled region 外。

如果用户只说“整模型慢”，不要直接进入多轮 kernel 优化。先拿到热点 kernel 名、shape 和调用路径。

### 2. 尽量缩到 operator / layer 级

不要默认在 `tests/engine/test_qwen2_generate.py` 里直接调优。整条 generate 链路里同时存在：

- cuda graph
- host 侧调度
- KV cache 管理
- 多个 operator 串联

这会掩盖单 kernel 收益。优先缩小到单 operator 或单 layer 测试。

### 3. 必要时建立最小复现

如果生产 kernel 不能直接接进 `.codex/skills/edge-fm-cuda-kernel-optimizer/scripts/benchmark.py`，建议：

1. 从目标实现中抽出核心 CUDA/CUTLASS 逻辑；Triton 只在已有 baseline 或用户明确指定时作为外部对照
2. 在 `deliverables/kernel_opt/<kernel_name>/` 下放最小 repro
3. 提供：
   - baseline kernel
   - `ref.py`
   - 固定 dims / shape
4. 先在 repro 上做多轮优化
5. 确认收益后再回迁到 `src/`

这样可以避免把探索性试错直接堆进正式实现。

## 构建与安装

如果正式代码发生变更，需要重新编译并安装 Python 扩展。通用方式：

```bash
cmake -S . -B build -DPLATFORM=a100
cmake --build build -j
cmake --install build
```

如果仓库里已经有平台对应的 build 目录，例如 `build-3060` 或 `build-orin`，可以复用，但不要假设所有环境都有相同目录。

## 回归验证

改动回迁到 `src/` 后，至少做一类最贴近的验证：

- operator / layer 改动

```bash
pytest -s tests/operators/test_decode_linear.py
pytest -s tests/operators/test_attention_decode.py
pytest -s tests/layers/test_linear.py
pytest -s tests/layers/test_attn.py
```

- engine / runtime 改动

```bash
pytest -s tests/engine/test_qwen2_generate.py
```

优先跑和目标文件最贴近的测试，不要默认全量 `pytest tests/`。

## NCU 与 benchmark 建议

- 若还没建立 standalone repro，先在 pytest 上跑 NCU，确认热点和 shape
- 若已经有 standalone repro，用本 skill 自带 `benchmark.py` 跑 correctness + latency
- 若收益只在 standalone repro 上存在，回到 repo 里重新核对：
  - 输入 shape 是否一致
  - launch config 是否一致
  - 编译选项 / arch 是否一致
  - 正式链路是否因更上层调度掩盖了单 kernel 收益

## 建议的产物目录

为了不污染正式代码，推荐把探索性工件集中在：

```text
deliverables/
  kernel_opt/
    <kernel_name>/
      baseline.cu
      ref.py
      env.json
      run_YYYYMMDD_HHMMSS/
```

如果用户明确要求直接在原文件旁边产出工件，再按用户要求放置。
