---
name: edge-fm-dev
description: Guides coding, building, and Python alignment testing for edge-fm (C++ core with pybind11 bindings). Use when editing C++/CUDA, adding Python bindings, compiling, or running pytest alignment tests against transformers/flashinfer.
---

# edge-fm 开发流程

项目为 C++/CUDA 实现，通过 pybind11 暴露接口，便于用 Python 做与 transformers/flashinfer 的结果对齐测试。

## 代码位置

| 用途 | 路径 |
|------|------|
| C++ 源码 | `src/`（含 `engine/`、`layers/`、`models/` 等） |
| pybind11 绑定（供 Python 调用） | `src/python/pybind_debug.cpp` |
| Python 对齐测试 | `tests/`（`engine/`、`layers/`、`utils/`） |

测试通过 `build/install/python` 下的 `edge_fm` 模块调用 C++，修改 C++ 或绑定后需重新编译并 `make install` 再跑测试。

## 编译

在项目根目录执行：

```bash
mkdir -p build && cd build
cmake .. -DPLATFORM=a100   # 首次或改 PLATFORM 时执行；可选 5090/4050/a100/thor/j6m
make -j && make install
```

- **horizon_quant (Python 3.10)**：需用 CUDA 12.8 构建以匹配 PyTorch，否则运行时报 `undefined symbol: __cudaLaunchKernel`。一键脚本：`bash scripts/build_horizon_quant.sh`
- **全量并行编译**：使用 `make -j`（不指定线程数），由 make 占满可用 CPU 核数；当前环境 CPU 与内存充足，无需限制编译线程。
- 默认安装前缀为 `build/install`，Python 扩展会安装到 `build/install/python/`（如 `edge_fm*.so`）。
- 关闭 Python 绑定：`cmake .. -DBUILD_PYTHON=OFF`。

## Python 对齐测试

- 依赖：在 `tests/` 下安装 `pip install -r tests/requirements.txt`（含 torch、transformers、safetensors、pytest-order 等）。
- 运行方式（在**项目根目录**执行）：
  - 用脚本跑：`bash tests/test_all.sh`（脚本内用 pytest 跑若干用例）。
  - **horizon_quant (Python 3.10)**：`HORIZON_PYTHON=/path/to/horizon_quant/bin/python bash tests/test_all.sh`，或一键：`bash scripts/run_tests_horizon_quant.sh`
  - 直接跑 pytest，例如：
    - 全量：`pytest -s tests/` 或按目录：`pytest -s tests/engine/`、`pytest -s tests/layers/`、`pytest -s tests/utils/`
    - 单文件：`pytest -s tests/engine/test_engine_generate.py`、`pytest -s tests/layers/test_attn.py`
  - **Benchmark (Python 3.10)**：`EDGE_FM_QWEN_MODEL_PATH=... EDGE_FM_DEVICE_ID=0 bash scripts/run_benchmark_horizon_quant.sh`（单 GPU 时需 `EDGE_FM_DEVICE_ID=0`）

测试文件会自行把 `project_root / "build" / "install" / "python"` 加入 `sys.path` 以导入 `edge_fm`；无需手动设置 `PYTHONPATH`，只要先完成 `make install`。

## 开发闭环

1. **改 C++ 或绑定**：编辑 `src/` 或 `src/python/pybind_debug.cpp`。
2. **编译安装**：`cd build && make -j && make install`（全量并行，不限制线程数）。
3. **跑对齐测试**：在项目根目录执行 `pytest -s tests/<子目录或文件>` 或 `bash tests/test_all.sh`。

若修改了 CMake 配置或新增源文件，需在 `build` 中重新执行 `cmake ..` 再 `make -j && make install`。
