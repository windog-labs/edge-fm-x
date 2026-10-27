# 本地磁盘占用盘点

后续状态：用户已明确授权清理，下面是**清理前**的只读快照，不是当前可删除候选。[清理结果](local_bulk_cleanup_20261026.md)：本阶段约369.91GiB已分配块退役，系统盘现约583GiB可用；f801源码、论文交接包和Codex会话保留。

日期：2026-10-26，Asia/Shanghai；详细快照时间10:55。本轮只读盘点，没有据此追加删除本地目录。此前已完成的129.40GiB模型产物清理已反映在当前空间中，不再重复计入可释放量。

## 文件系统

| 项目 | 当前快照 |
|---|---:|
| 根分区 | `/dev/nvme0n1p2`，ext4 |
| 总容量 | 916 GiB |
| 已用 | 约656 GiB |
| 普通用户可用 | 10:55快照约212.75 GiB；首轮df显示约214G，盘点期间仍有后台写入 |
| 使用率 | 76% |
| 用户主目录 | 约561 GiB |

目录数值来自`du -x`，取已分配空间近似值。父子目录不能相加；分别扫描的目录也可能有硬链接/共享依赖，不把目录体积直接当作保证回收量。`/var`、`/opt`等有权限不可见区域，其目录结果仅为可见部分。

## 优先查看的候选

| 优先级 | 目录 / 内容 | 占用 | 建议与风险 |
|---|---|---:|---|
| 高 | `~/.cache/edgefm-vla-goal/openpi`四个模型权重/转换目录 | 约36.04 GiB | pi0/pi0.5原权重和转换副本，可在确认不再推理后删除；以后需重新下载或转换。保留processor、数据集和必要配置 |
| 高 | `~/.cache/vlaforge/openvla-7b` | 约15 GiB | 大模型缓存候选；需确认旧OpenVLA任务不再依赖。不要因此整删`.cache/vlaforge`，其中另有环境 |
| 高、先分离 | 原仓库`edge-fm-x/artifacts/edgefm-vla-goal/20260906-044509` | 约95.67 GiB | 历史实验混合目录，权重/导出可清，唯一原始数据/失败记录先核验；不承诺整目录均可释放 |
| 中 | `edge-fm-x/artifacts/disk-cleanup-20260907T112748Z` | 约6 GiB | 历史清理留存目录，先确认是否是尚需恢复的备份/唯一内容 |
| 中 | `edge-fm-x/examples/smolvla` | 约8.22 GiB | 示例模型/相关产物，分离源码、输入和大文件后再清，不能只按examples名称整删 |
| 中、其他项目 | `Repos/private/model-delivery-builds` | 约24 GiB | 两套BEV交付构建副本；通常可重建，但先确认当前交付与回放任务是否还引用 |
| 中 | `~/Packages`中的Horizon压缩安装包 | 约5.1 GiB | 可仅评估tar/tgz下载包；不要误删3.9GiB的已解压HBDK runtime及其他正在使用工具 |
| 中、需重建 | 两个感知仓库的`.codegraph` | 约9.9 GiB | 索引可重建，但需确认索引任务不在运行，删后暂时失去结构查询能力 |
| 中、其他项目 | `xs-perception/output` | 约8.8 GiB | 编译/运行输出需分类型核实，不直接混删case证据和交付包 |
| 小 | Docker当前报告的可回收镜像 | 约4.46 GB，Docker口径 | 镜像总量15.97GB不等于全部可回收；build-cache显示5.787GB，但当前reclaimable仅12.29kB，不重复相加 |

以上是候选，不是删除许可或确定的回收承诺。尤其其他项目、环境和原始case，需要按用途确认。

## 大目录构成

| 目录 | 占用 | 说明 |
|---|---:|---|
| `~/Repos` | 约334 GiB | 主要为下列三个仓库和交付构建，不能与子项重复求和 |
| `Repos/private/perception_detection_solo` | 约117 GiB | case约104GiB为最大项；其他含repo元数据、依赖和索引 |
| `Repos/private/edge-fm-x` | 约114 GiB | artifacts约101.67GiB、examples约8.22GiB、git约1.32GiB |
| `Repos/private/xs-perception` | 约73 GiB | git约42GiB、thirdparty约9.8GiB、output约8.8GiB、索引约7.2GiB |
| `~/.codex` | 约100 GiB | worktrees约83GiB，其余是附件、会话、备份和工具；不可整删 |
| `~/.cache` | 约61 GiB | EdgeFM模型缓存约38GiB；vlaforge约23GiB，其中MindDrive环境约7.3GiB，不是全可删下载缓存 |
| `~/miniconda3` | 约23 GiB | envs约12GiB、base lib约9.6GiB、pkgs约1.3GiB；按环境和依赖清理，不整删 |
| `~/.venvs` | 约12 GiB | OpenPI CUDA环境约8.1GiB、CPU环境约1.9GiB；报告环境174MiB建议保留 |
| `~/Packages` | 约9.4 GiB | 下载归档约5.1GiB，HBDK host runtime约3.9GiB，另有工具 |
| `/opt` | 可见约20 GiB | NVIDIA约18GiB，含运行工具/SDK，不作为普通临时文件清理 |
| `/usr/local` | 约34 GiB | CUDA11.8约9.1GiB、CUDA12.8约11GiB、TensorRT10.15约14GiB；环境/SDK项，暂不动 |
| `/var/log` | 可见约4.1 GiB | 若要清需按日志服务保留策略处理，不能直接删除正在写的日志 |

## 三个需要特别注意的目录

### 旧EdgeFM实验：约96 GiB

根目录：`/home/zhangzimo/Repos/private/edge-fm-x/artifacts/edgefm-vla-goal/20260906-044509`。

| 子目录 | 占用 GiB | 分类 |
|---|---:|---|
| `openpi-inventory` | 37.55 | 权重/检查与中间结果混合，优先核实重复 |
| `fresh-smolvla` | 11.13 | 历史模型准备/结果副本 |
| `operator-profiles` | 9.34 | 算子产物与数值/性能数据混合 |
| `numerical-probes` | 5.18 | 数值探针，失败/唯一验证数据需保留或归档 |
| `int8-native-v1` | 3.47 | 历史低精度构建/验证产物 |
| `local-smolvla` | 3.29 | 历史模型准备/本机副本 |

这不是刚才清理的f801实验根，之前的129.4GiB回收没有涵盖该目录。

### 感知case：约104 GiB

根目录：`/home/zhangzimo/Repos/private/perception_detection_solo/case`。

| 子目录 | 占用 GiB | 备注 |
|---|---:|---|
| `uturn_3issues_20261024` | 88.92 | 最大项；需确认原始MCAP、复现产物及归档是否齐全 |
| `stage2_v013_rl_j6p_deployment_fix_20261026` | 7.06 | 当日部署修复case，不建议默认删除 |
| `rl_stage2_cases_20261023_155253` | 4.25 | 归因case，确认评测是否结束 |
| `stage2_v013_rl_j6p_alignment_20261025` | 2.30 | 对齐case，确认评测是否结束 |
| `rl_stage2_attribution_20261023` | 0.79 | 归因证据，确认后处理 |

### 当前Codex工作树：约83 GiB

`~/.codex/worktrees`目前只看到f801这一棵大工作树，并非还有很多废弃工作树可删。其主要artifact：

| 数据组 | 占用 GiB | 建议 |
|---|---:|---|
| `artifacts/orin` | 45.71 | 清理后的原始/诊断/参考与部分工具等混合，不能整删 |
| `artifacts/j6m-feasibility-20260911` | 32.38 | 含未完成完整链所需的模型和评估材料，不能默认删 |
| `artifacts/recovery-audit-20260909` | 3.68 | H20保存证据，保留唯一数据 |
| `artifacts/j6m-performance-20260914` | 0.06 | 最终阶段表/CDF证据，保留 |

`xs-perception/.git`也不能直接删：只读`git count-objects`显示pack约34.77GiB，garbage为0。正常GC不保证能腾出这34.77GiB；不能为了空间擅自改写历史或破坏worktree/submodule引用。

补充只读查看`.codex`其他占用：附件约5.5GiB、归档会话约2.5GiB、packages约2.6GiB、backups约1.5GiB，另有旧迁移/恢复副本。它们不等于废弃工作树，也不视为已获准删除的缓存；直接删除附件或会话可能丢失历史材料。浏览器配置约5GiB同样不是普通模型产物，不纳入本次候选删除。

## 建议顺序

1. 先确认是否释放OpenPI权重/转换副本约36GiB，以及不用的OpenVLA缓存约15GiB；不连带删除环境或数据集。
2. 再按文件类型整理旧EdgeFM约96GiB历史实验，先核验唯一证据，不承诺一次全删。
3. 最后处理其他感知项目的构建输出和已经归档的case；保留近期修复case、Git历史和必要SDK。

`/home/zhangzimo/data`已不存在，`~/Data`几乎为空；先前删除的数据不再占用本地磁盘。论文交接包及ZIP约0.08GiB，应保留。

NAS上的约1.1TiB实验目录是另一文件系统，相关清理单独记录，不计入上面的本地可释放量。

[只读原始盘点记录](../../artifacts/storage-maintenance-20261026/local-disk-inventory-001.json)保留逐目录精确字节数、Docker原输出和采集时间；没有执行本地删除。
