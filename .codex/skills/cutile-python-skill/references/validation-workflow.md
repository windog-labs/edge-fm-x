# Validation Workflow

## Prerequisites

根据仓库自带 README / docs，cuTile Python 常见前置条件包括：

- Python 3.10+
- CUDA Toolkit 13.1+，或带 `tileiras` 的 Python 包
- 支持的 NVIDIA GPU
- 构建 C++ extension 所需的 CMake / C++17 编译器

如果验证失败，不要先猜代码问题；先确认环境：

1. `tileiras` 是否可用
2. CUDA / 驱动版本是否满足
3. Python 环境里依赖是否齐全

## Build From Source

按仓库 README，最直接的是 editable install：

```bash
cd third_party/cutile-python
pip install -e .
```

如果已经做过 editable install，只改了 C++ extension，优先尝试：

```bash
make -C third_party/cutile-python/build
```

如果要启用 experimental autotune 包：

```bash
pip install ./third_party/cutile-python/experimental
```

## Fast Validation Targets

### Quick smoke tests

```bash
python3 third_party/cutile-python/samples/quickstart/VectorAdd_quickstart.py
pytest third_party/cutile-python/test/test_frontpage_example.py -q
pytest third_party/cutile-python/test/test_readme_example.py -q
```

### Run a focused API / behavior test

```bash
pytest third_party/cutile-python/test/test_load_store.py -q
pytest third_party/cutile-python/test/test_copy.py -q
pytest third_party/cutile-python/test/test_mma.py -q
pytest third_party/cutile-python/test/test_export_compat.py -q
pytest third_party/cutile-python/test/test_cache.py -q
pytest third_party/cutile-python/test/test_autotuner.py -q
```

### Run samples

```bash
pytest third_party/cutile-python/samples -q
```

## How To Choose Tests

- 改 `ct.load` / `ct.store` / gather / scatter：
  跑 `test/test_load_store.py`、`test/test_gather_scatter.py`
- 改 `ct.mma` / matmul / scaled MMA：
  跑 `test/test_mma.py`、`test/test_mma_scaled.py`、相关 sample
- 改导出 / ABI / mangling：
  跑 `test/test_export_compat.py`、`test/test_name_mangling.py`
- 改 cache / JIT / compiler context：
  跑 `test/test_cache.py`、`test/test_compiler_options.py`
- 改 autotune：
  跑 `test/test_autotuner.py` 和相关 sample

## Debug-Oriented Environment Variables

验证或排障时常用：

```bash
CUDA_TILE_LOGS=CUTILEIR
CUDA_TILE_ENABLE_CRASH_DUMP=1
CUDA_TILE_COMPILER_TIMEOUT_SEC=30
CUDA_TILE_CACHE_DIR=/tmp/cutile-cache
```

如果问题与 profile、Nsight、底层 CUDA 运行时强相关，联动 `cuda-skill`。
