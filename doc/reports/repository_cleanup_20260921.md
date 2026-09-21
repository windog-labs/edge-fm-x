# VLA 分支整理与验证

日期：2026-09-21。起点为 `592fc320d57398571dcb4ab686d5fb55fe9bac86` 的
detached worktree；同期的模型代码、测试和报告此前没有提交。整体归档提交为
`573df78251f0dc13a180f4484529041e2b8f4224`，后续提交补齐 J6P cache ABI 示例。

## 整理范围

- 保留并提交已有 runtime、编译工具、模型适配和测试；平铺 adapter 文件的
  删除对应其迁入模型子包，不代表删除这些模型支持。
- 修复迁移后仍指向旧 adapter 路径的文档，以及 SmolVLA capture 工具计算
  source hash 时使用的旧文件路径。
- 清理本工作区 `tmp/` 下七个生成文件（CogACT runner 源码副本与 PDF 页面图）
  和 Python 缓存字节码；增加 `tmp/` ignore，正式实验数据继续留在 `artifacts/`。
- 将重复的中间实验状态汇总改为最终报告入口；旧内容仍可从 Git 历史恢复。
  H20 报告增加时间说明，避免把 9 月 10 日的 J6M 暂缓状态误用为当前状态。
- 保留有效的回归测试、历史失败报告和审计记录。未发现可安全删除的重复测试，
  不通过移除失败测试减少测试数量。
- [J6P VLM 文档](../j6p_vlm_deployment_and_kv_cache.md) 和对应 Python 示例
  使用两个完整 HBM，显式描述 prefill 输出与 decode 输入、decode 输出回接的 ABI。
  无逐层调度和自定义算子；provider SDK 接入尚未实现，不声称已在 J6P 部署。

原始 HBM、checkpoint 和大型原始实验数据不进入 Git；其本地及远端归档位置
和 hash 由已有实验报告记录。此次同步的是代码、测试与报告，不是完整模型包。

## 验证结果

| 检查 | 结果 |
|---|---|
| 第一轮整理后的 Python 全套回归 | 2317 passed、74 skipped、6 subtests passed |
| 补齐 J6P 两模型 cache ABI 后的全套回归 | 2328 passed、74 skipped、6 subtests passed；78.82 s |
| host-cpu CMake 构建与 CTest | 11/11 passed |
| J6P 编排定向测试 | 16 passed；包含在全套回归中 |
| Git whitespace 检查 | passed |

完整 CPU 回归从 `vlaforge/` 执行：

```sh
PATH=/home/zhangzimo/miniconda3/bin:$PATH CUDA_VISIBLE_DEVICES= \
PYTHONPATH=python:tools \
/home/zhangzimo/.venvs/edgefm-vla-report-py313-20260906/bin/python \
  -m pytest -q --tb=short \
  --junitxml=../artifacts/repository-cleanup-20260921/full-cpu.xml
```

第一次回归的 13 个 generated-C++ 测试因子进程 PATH 缺少 Ninja 而在 CMake
配置阶段失败。系统 GCC/G++ 已存在；补全 PATH 后全套通过，无代码绕过或测试删除。
74 项 skip 覆盖显式 opt-in SDK/CUDA/真实 checkpoint 和缺失可选依赖，未计为通过。

从仓库根目录执行的本机构建命令：

```sh
cmake --preset host-cpu
cmake --build --preset host-cpu --parallel
ctest --preset host-cpu --output-on-failure
```

这次工作没有重跑 H20/J6M 正式测时，没有新增 J6P 模型实验，也未修改实验目标。
