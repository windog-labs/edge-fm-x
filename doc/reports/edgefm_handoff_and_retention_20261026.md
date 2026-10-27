# 自包含论文交接与冷模型产物清理

后续阶段见[本地大目录与NAS清理结果](local_bulk_cleanup_20261026.md)。用户随后明确授权删除表列的本机实验/case目录，NAS冷模型清理也已完成；下文“本机原始来源仍保留”“NAS未选中”等是本阶段当时的历史状态，不代表更晚清理后的现状。自包含论文包保持不变并通过清理后复查。

日期：2026-10-26，Asia/Shanghai。本次操作先验证交接包可独立使用，再按用户明确授权删除已定论实验的冷模型产物，没有重跑模型或改变实测结论。

## 交接包

位置：`/home/zhangzimo/Repos/private/edge-fm-x/paper_handoff/edgefm_three_platform_20261026`，同目录旁提供 `.zip`。

- 272 个内容文件，约 75 MB；含总汇总、EdgeFM/baseline 对照说明、分平台报告、PNG/PDF、原始时延/CDF/质量/消融/资源表和校验工具。
- 包内 282 个本地 Markdown 链接均为相对路径，所有引用内容随包携带。
- 已移除本机、NAS 和连接地址；来源仅保留无机器地址的记录标识、字节数和 SHA256。旧执行命令不作为交接内容。
- JSON/CSV 的数值不变；元数据经过明确标注的去路径处理，大 JSON 为明确标注的审计节选，不冒充完整原审计或新实验。
- ZIP 已独立解压到另一临时目录，使用包内标准库脚本自检通过；不需要原工作区、模型权重或台架连接。
- `01_edgefm_results.md` 明确 H20 `session-required` 与 Orin replay/Arena 是框架实测路径，同时声明并非五模型统一启用全部 Agent 算子优化的 Full EdgeFM 配置。

[严格自包含验证记录](../../artifacts/paper-summary-20261026/handoff-closeout-003.json)。原先001/002为此前打包快照，已由003替代。

## 本轮清理结果

| 位置 | 已删除路径 | 唯一 inode 已分配空间回收 GiB | 清理后可用空间快照 GiB |
|---|---:|---:|---:|
| 本机 Orin 实验副本 | 211 | 120.200481 | 与下一行共用文件系统 |
| 本机 H20 历史 Qwen 临时模型 | 3 | 9.199696 | 本机合计约214.45 |
| Orin EdgeFM 实验目录 | 48 | 42.626907 | 约234.22 |

本机合计回收约 **129.400177 GiB**；Orin 合计约 **42.626907 GiB**。以删除前唯一 inode 的已分配块计数，硬链接不重复相加；可用空间会受其他后台写入影响，不将磁盘快照差值包装为额外回收量。

范围仅为两个本机实验根和 Orin 的 EdgeFM 实验根下、>=64MiB 的已确认冷模型/engine/导出权重。无后缀外部权重通过 ONNX 元数据解析确认；`.ts` 仅在确认是 PyTorch/TorchScript 模型或权重归档时清理，不视为 TypeScript 源码。

Qwen0.8B 的已完成部署产物和已取消的 Orin2B准备副本可以退役；取消不改称2B的Orin测试通过。共享 NAS、其他仓库/工作树、环境/SDK、未完成的 J6M 完整模型路径、不确定用途的文件和小型元数据不在本轮删除范围。

## 保留与复查

- 每次 unlink 前保存完整 SHA256、大小/inode/权限与写前日志；拒绝非授权根、符号链接、其他所有者、跨设备目录及已观察到的打开/mmap 文件。
- 原始时延 CSV/JSON、动作/reference/观测/noise、质量及失败记录、CDF 数据/PNG/PDF、日志、源代码和协议保留；未以清理为名删除慢样本。
- 清理前后105个论文来源文件的 SHA256/大小完全一致，52行表格与109个原仓库证据链接复查通过；交接包独立校验仍通过。
- Orin 48个选定路径均已不存在，无运行中的本任务单元；原Supervisor PID4237不变，两项Java服务仍以PID643146/643080处于Running，SDK查询正常。没有停止服务、修改SDK、清缓存或改变功耗/频率。
- 可见性限于当前用户能读取的打开文件和进程映射，不宣称全平台独占或应用功能健康验收。

历史 engine/源模型闭合记录仍描述当时真实实验，**不表示清理后可直接再推理**。以后重新运行完整模型，需要重新准备被退役的权重/ONNX/engine；本次不承诺大文件备份。

## 审计入口

- [用户授权](../../artifacts/storage-maintenance-20261026/authorization.json)
- [本机 Orin 计划](../../artifacts/storage-maintenance-20261026/local-orin-plan-001.json)、[结果](../../artifacts/storage-maintenance-20261026/local-orin-cleanup-001.json)
- [本机 H20 计划](../../artifacts/storage-maintenance-20261026/local-h20-plan-002.json)、[结果](../../artifacts/storage-maintenance-20261026/local-h20-cleanup-001.json)
- [Orin 计划](../../artifacts/storage-maintenance-20261026/target-plan-001.json)、[结果](../../artifacts/storage-maintenance-20261026/target-cleanup-001.json)、[终态检查](../../artifacts/storage-maintenance-20261026/target-postcheck-001.json)
- [清理前数据复查](../../artifacts/storage-maintenance-20261026/evidence-before-cleanup-001.json)、[清理后数据复查](../../artifacts/storage-maintenance-20261026/evidence-after-cleanup-001.json)
- [最终保留验证](../../artifacts/storage-maintenance-20261026/post-cleanup-verification-001.json)

本报告是本机运维记录，不放入发给论文修改同学的自包含材料；交接包不依赖这些原机路径或连接。
