# 本地大目录与NAS清理结果

日期：2026-10-26，Asia/Shanghai。按用户“这些全部删掉吧”和“.codex中的能清理的也清理一下”的明确授权执行。未重跑模型、未修改实验数值或验收门槛。

## 结果

| 位置 | 本阶段结果 | 空间口径 |
|---|---|---|
| 本机 | 23个授权路径完成，无跳过/延期；约171万个文件/目录元数据节点 | 删除前唯一inode已分配块约369.914902GiB |
| 本机系统盘 | 清理后约582GiB可用，使用率33%，总容量约916GiB | 即时df快照，不把后台写入/释放算作额外清理量 |
| NAS本任务runs | 196个冷模型文件完成，计划/结果/逐项SHA256/删除日志核对通过 | 唯一inode已分配块约797.968254GiB；不承诺NFS物理配额实际回收量 |
| Orin | 本轮追加复查为0个合格冷模型文件，无新增删除 | 之前48路径/42.626907GiB属于前阶段，不重复计数 |

本机与NAS不是同一文件系统，不能将NAS清理量计入本机可用空间。此前本机129.40GiB的模型退役属于历史阶段，也不与本阶段重复相加计算当前余量。

## 本机范围

| 内容 | 本阶段处理 |
|---|---|
| OpenPI约36GiB | 删除四个pi0/pi0.5权重及转换副本目录；未连带删除dataset、processor和环境 |
| OpenVLA约15GiB | 删除模型缓存目录，其他vlaforge环境保留 |
| 旧EdgeFM实验约96GiB | 删除明确授权的20260906-044509整棵实验目录，包括旧中间数据 |
| 两份BEV构建约24GiB | 保存补丁、新增源码及独立Horizon Git历史后删除；仅移除这两份已删worktree的Git登记，不全局prune |
| Horizon下载包约5GiB | 删除三个tar/tgz，保留已安装环境和已解压runtime |
| 两个感知仓库索引约10GiB | 删除.codegraph；后续结构查询需重建索引，不自动初始化 |
| 感知case约104GiB | 按本次明确授权删除case目录，包含此前提示的近期修复case；未承诺原始MCAP/回放大文件备份 |
| f801约82GiB实验产物 | 删除Orin、J6M可行性/性能、H20恢复审计及旧goal产物；保留源码、文档、看板、当前索引和清理记录，工作树清理后约132MiB |
| .codex额外缓存 | 删除cache、.tmp和两个旧索引SQLite备份；不删除当前会话、归档会话、附件、记忆、插件、登录配置或当前状态库 |

各行是清理前目录近似值，不直接相加作为保证回收量；硬链接按跨目录唯一inode核算。删除前生成逐文件元数据清单，清单已gzip压缩保存；这次整目录清理不声称对每个case/模型文件新做完整内容哈希。NAS冷模型退役则保留每个选定文件的完整SHA256。

## 数据与复现边界

论文交接文件夹和ZIP保留，清理后使用包内标准库脚本独立复查通过：272个内容文件、282个相对链接，ZIP SHA256仍为`8d70859f0d11efbf221f61b352398939c7f4dfdc857d9c45122d37815c2a1740`。

[论文交接文件夹](/home/zhangzimo/Repos/private/edge-fm-x/paper_handoff/edgefm_three_platform_20261026/README.md)包含汇总主表、CDF、质量、消融和已声明边界，不依赖已删的本机实验根。包本身未改动，仍自包含。

**本阶段授权比先前“只删冷权重”更广，已删除本机原始实验/case目录。** 之前105来源文件/109原机链接的核验是删除前的历史事实，不再声称这些源路径现在仍可访问。原机报告与看板中的旧证据路径可能失效，论文修改以自包含包为入口。NAS原始测量格式未选入196文件删除清单，剩余CSV/JSON/数组等仅做存在性/格式统计，未声称全量内容重哈希。

J6M未完成完整链的本机产物也按本次授权退役，不把它改标为完整E2E通过；模型权重、engine/HBM和原始case大文件不承诺备份。未来完整重新推理/回放需重新准备数据和构建产物。感知case不属于论文交接内容，删除后不能靠论文包恢复。

## 源码保存与审计

清理记录与源码保存位于[维护目录](/home/zhangzimo/Repos/private/edge-fm-x/artifacts/storage-maintenance-20261026)。f801源码仍在原路径；另保存其补丁/新增源码。BEV副本中未提交的Horizon修复已保存staged/unstaged补丁、新文件归档和可独立核验的Git bundle；共享Git数据库与其他工作树未清理。

- [本机计划](/home/zhangzimo/Repos/private/edge-fm-x/artifacts/storage-maintenance-20261026/authorized-local-tree-plan-001.json)
- [本机结果](/home/zhangzimo/Repos/private/edge-fm-x/artifacts/storage-maintenance-20261026/authorized-local-tree-cleanup-001.json)
- [本机写前删除日志](/home/zhangzimo/Repos/private/edge-fm-x/artifacts/storage-maintenance-20261026/authorized-local-tree-cleanup-001.journal.jsonl)
- [压缩元数据清单](/home/zhangzimo/Repos/private/edge-fm-x/artifacts/storage-maintenance-20261026/authorized-local-tree-manifest-001.jsonl.gz)
- [源码保存索引](/home/zhangzimo/Repos/private/edge-fm-x/artifacts/storage-maintenance-20261026/source-preservation.json)
- [NAS终态复查](/home/zhangzimo/Repos/private/edge-fm-x/artifacts/storage-maintenance-20261026/nas-postcheck-001.json)
- [本地清理后源码/交接包/表格及模型计数复查](/home/zhangzimo/Repos/private/edge-fm-x/artifacts/storage-maintenance-20261026/bulk-cleanup-postcheck-002.json)

本机删除前检查当前用户可见的打开文件、mmap及cwd；目录内容变更会延期而不是强删。可见性不代表全系统独占。未停止台架/业务服务、未使用sudo、未改SDK或功耗参数；Codex进程保留，看板仅为启用退役说明而重载。

看板对明确列入本次退役根的原始证据返回HTTP410并提供清理记录入口，Markdown入口标记“原件已退役”；未知证据仍为404，未声明的文件缺失仍使检查失败。模型验收与采样门槛未因存储退役而放宽。

复查：24项合成清理安全测试、15项看板回归通过；原汇总177行表格文本逐行不变，模型和run记录不变。论文包272文件/282链接、压缩清单解压SHA、四组源码保存及独立Git bundle重核通过；仅重载本任务看板服务以启用已退役入口，未停止Codex或台架服务。
