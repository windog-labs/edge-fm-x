# EdgeFM VLA Goal Status

Goal 启动：2026-09-06 04:45 UTC。当前Goal允许的H20证据交付已闭合；
全论文板端验收未完成。
Qwen3.5 0.8B/2B固定profile原生部署现已在H20完成，四组正式campaign均通过
独立审计、完整输出byte核对和CDF检查。CogACT autonomous C++ RNG路径也已完成
同计时边界的official/native正式对照。

## 2026-09-10 最终状态

本节为当前入口；以下实施记录中的运行中/待完成描述均为历史快照。
统一入口：[H20实验交付与剩余工作](reports/h20_vla_experiment_delivery_20260910.md)。
逐项Goal验收索引：
`artifacts/recovery-audit-20260909/goal-completion-audit-002.json`。
Qwen原生正式结果：[Qwen3.5 Native H20 Formal Deployment](reports/qwen35_native_formal_20260910.md)。
Qwen证据索引：
`artifacts/recovery-audit-20260909/qwen-native-evidence-index-002.json`，SHA256
`c13c977de929a0cfadc403093e2aae1c45b74d3830186ed51e2c6657c1b2c4ae`。
CogACT同边界正式结果：
[CogACT Boundary-Aligned Formal Comparison](reports/cogact_timing_boundary_formal_20260910.md)。
CogACT证据索引：
`artifacts/recovery-audit-20260909/cogact-timing-boundary-evidence-index-001.json`。

| 范围 | 当前验收状态 | 剩余工作及估时 |
|---|---|---|
| 五VLA原始输入 official/Session | passed：50/50进程，51200测量，149760完整张量byte exact，远端/本地审计和CDF全部完成 | 当前profile无需补跑 |
| 五VLA resident及host-model-tensor | passed：各15行正式表、76800测量、224640完整张量byte exact | 无 |
| 六类算子 | passed：每类5对独立进程，完整输出/ownership审计；保留Attention持平、Linear负结果 | 固定recipe局部比较，不授予自主kernel生成或整模集成收益 |
| Qwen3.5 0.8B/2B官方多模态 | passed：各5进程、5120测量、5760完整token输出exact，CDF完成 | 仅官方Python参考；原生适配已单列完成 |
| Qwen3.5原生部署 | passed（H20固定profile）：0.8B/2B各first/16-token共20进程、20480测量、46080完整输出张量exact；四组独立审计、CDF及80项Session生命周期检查完成 | 无当前profile补跑；旧cache-copy失败保留为历史路线 |
| CogACT autonomous native同边界对照 | passed：5+5独立进程、5120+5120测量；输出与RNG全byte exact，同输入/seed/精度/GPU和同一计时边界独立审计通过；official/native mean 129.258/88.751ms，mean ratio 1.456x | 无当前profile补跑；worker-4重尾保留，median paired-GPU ratio 1.393x也单列 |
| 最终回归/交付索引 | passed：CogACT收尾回归2276通过/62跳过/0失败；Qwen收尾回归2275通过/62跳过/0失败；此前集合反例4通过；2935证据身份重核 | 无 |
| Orin/J6M/BPU | deferred-hardware | 无硬件，不承诺完成日期 |
| 稿件回填 | 已提供表图和PDF主张检查，原PDF未修改 | 有可编辑稿源后约2-4小时，不含补实验与作者审阅 |

CogACT正式controller143713及最后worker166189均已成功结束，10/10进程与57600
完整五端口张量审计通过，official/Session均值120.286270/79.971651ms。此前3/10
状态已失效，禁止重复启动。五模型统一报告位于当前worktree
`artifacts/recovery-audit-20260909/raw-five-summary-001/report.json`，SHA256
`077362fc3ec81e74c8440374c677f525dd1ae6271b5d196e753849b7d2b9856c`。

当前范围的H20正式实验已无排队作业。Qwen原生固定profile已完成；Orin/J6M/BPU
和任意输入/流式逐token边界仍不继承该结果。
raw-input CogACT正式流程使用7.63B实际checkpoint、公共依赖配置和归档外部RNG
tape，整个流程含Python主机预后处理并从已解码CPU输入开始；独立autonomous
native路径已改用C++ `CudaRngProvider`并完成同边界对照。原Meta gated配置等价性
仍未验证。各项论文边界见
`reports/h20_paper_scope_review_20260910.md`。估时为条件判断，不是运行倒计时。

## 原始输入实施记录（历史快照）

五模型host-model-tensor正式表和全部远端/本地审计已闭合，见
`reports/h20_host_tensor_formal_table_20260909.md`。原始输入通用计时器及OpenPI
双实现已落地，pi0/pi0.5各10个正式进程已完成远端独立审计、本地完整输出复核
及CDF出图检查。两模型共20480正式测量/46080完整输出张量byte exact；每个
native进程1152 replay、0普通回退。official/Session required均值分别为pi0
221.305113/108.152217ms、pi0.5 256.840307/116.853524ms。controller
pi0.5 80976/GPU1、pi0 85392/GPU2均已成功结束，禁止重复启动。
计时从CPU RAM中的已解码RGB/state/text/noise开始，包含预处理和完整CPU动作
输出；不包含文件解码、初始化或机器人通信，此Python主机流程不授予no-Python证明。
源码302文件快照保持不变。SmolVLA新增适配在独立303文件快照中，16条原始观测
已提取校验；69项定向及完整CPU2235通过/74跳过，GPU验证仍待完成。
详见 `reports/raw_input_host_pipeline_20260910.md`。SmolVLA/RDT/CogACT原始输入
正式长测、最终五模型统一表和Qwen原生部署仍待完成。

继续推进：SmolVLA原始输入在两张H20上各完成official/native pilot及远端/本地
完整输出审计，每组96调用/192张量byte exact，native48 replay/0回退。H20-1
formal controller3202649因新出现的foreign owner在启动任何worker前被拒绝；
失败记录保留。H20-2 GPU1新campaign-002正式controller113987已成功结束，10/10
worker、远端独立审计、本地完整输出复核及CDF全部完成。SmolVLA official/Session
原始输入均值206.161639/50.726747ms，共10240测量/23040完整张量byte exact；
每个native进程1152次十步replay/0回退。不可重复启动。
RDT新304文件隔离适配在H20-2 GPU0验证全部16条原始输入及保存噪声exact，
official/native pilot均通过远端与本地完整输出审计，96调用/192完整张量byte
exact，native48次五步replay/0回退；正式controller119430与首worker119581实查
存活。最新完整CPU2249通过/74跳过，另H20依赖齐全的CPU定向53通过/1 CUDA跳过。
详见 `reports/raw_smolvla_rdt_pipeline_20260910.md`；RDT尚未正式完成。
CogACT原始输入adapter已进入新305文件隔离快照，90项定向及完整CPU2259通过/
74跳过；真实CPU预处理48张量exact。official/native pilot现已通过远端独立审计
及本地完整复核：96调用/480完整五端口张量byte exact，native48次十步replay/
0回退。正式controller143713在H20-2 GPU2实查存活，尚无正式完成结论。详见
`reports/raw_cogact_host_pipeline_20260910.md`。
Qwen2B新的同口径原生诊断在隔离目录`qwen-native-2b-20260910-001`执行；
controller142389/worker142395已实查存活，原始图像预处理及完整token与既有正式
参考exact，trace/export仍在验证。旧0.8B导出失败不能直接写成2B实测失败。

## 当前一览（2026-09-09 续接核对）

最新收尾：五模型host-model-tensor正式75/75进程、远端独立审计、本地完整输出
复核、15行统一表及五组CDF全部完成；76800测量/224640完整输出张量byte exact。
权威入口为 `reports/h20_host_tensor_formal_table_20260909.md`。下方22:32运行中
快照已过期，禁止重复启动。pi0.5原始RGB/state/text到完整双输出的native/official
路径各48次pilot自检通过，但尚未独立审计或完成正式长测；其他模型原始输入
路径及五模型同口径官方对照仍未闭合。Goal保持active。

本节和最新恢复记录优先于以下历史快照。按用户最新要求，新的 GPU 实验只在
H20 执行，H100/RTX 已有结果保留为历史证据，不再把缺少 H100 全模型矩阵作为
当前任务依赖。`huoshan-private` 与两台 H20 共用 NAS，同步优先走该通道。

| 编号 | 状态 | 一句话结论 |
|---|---|---|
| G0 | running | 论文映射和证据地图持续维护 |
| G1 | running | CUDA/原生生命周期与 graph-memory 有限场景已闭合；非零分配/全栈接管未声明 |
| G2 | historical snapshot; autonomous RNG later closed | 五个真实 VLA 各有 CUDA replay/CDF；H100 pi0 16帧normalized及dual-output native诊断已通过；CogACT 新数值 provider 002 已闭合；该时点原 Meta 配置/自主 RNG 未闭合，后续 autonomous RNG 已由2026-09-10同边界报告闭合 |
| G3 | scoped H20 formal complete; wider boundary pending | 五模型15策略行已重核76800时延/224640张量；新pi0.5 H20双输出formal及CDF完成，required均值95.344ms；完整输入边界和official/vendor对照仍缺 |
| G4 | scoped operator table complete; integration pending | 六类H20算子各5对进程、完整张量及ownership审计通过，CSV/图表已检查；保留Attention持平和Linear负结果，不外推整模收益 |
| G5 | running | SmolVLA/RTX 工作集：INT8 held-out 0/8，FP16/BF16 Python/no-Python/off/replay/error-budget/profiler 均完成；模型族泛化待做 |
| G6 | native formal complete on H20 fixed profile | first/16-token四组共20进程/20480测量/46080完整输出张量byte exact；四组独立审计、CDF及80项Session生命周期检查完成；旧cache-copy失败保留为历史 |
| D1/D2 | deferred-hardware | 数据包/契约已准备，板端 driver/SDK/实机未验收 |

## 2026-09-09 主机张量计时实现

公共runner和Python/C++测试已完成，新增显式host-model-tensor模式：每次H2D、
绑定、Session完成及完整D2H计入同一调用，分段CSV与主时延逐次核对并绑定SHA。
定向167通过，完整CPU回归2204通过/74跳过/0失败。源码按当前592fc320及298个
实现文件冻结，未合并已前进的远端主分支。详见
`reports/host_tensor_boundary_20260909.md`。

H20-2隔离目录 `runs/host-tensor-20260909-001/` 中，五模型三策略构建和pilot
远端/本地审计全部完成：720次调用/1872完整输出张量byte exact。正式controller
Smol33468、pi0 33483、pi0.5 31003、RDT35605、CogACT39618及各自supervisor均
已实查存活。22:32 CST快照为11/75正式worker自检通过；尚未完成全量正式审计。
原始图像预处理和统一official基线仍待接入，已有resident表不改名为端到端表。
本节优先于下方历史快照中的“此接口尚未实现”。

## 2026-09-09 21:49 CST 当前剩余工作

Qwen3.5 0.8B/2B正式Python基准、完整输出审计、本地复核和图表已完成。
TTFT均值分别152.937/156.302ms，固定16-token输出的输入到CPU token均值
558.794/545.668ms。仅为固定真实图像和prompt的官方Python路径，不是VLAForge
原生部署或优化前后质量对照。详见 `reports/qwen35_natural_profile_20260909.md`。

本轮用户要求内的剩余项：

1. 五VLA完整输入计时与同口径official/direct/Session对照。已完成表仍是CUDA
   resident model-tensor边界。主机模型张量模式需要通用runner显式测量每次
   H2D、绑定、Session完成、所有输出D2H，并由审计器验证分段求和与完整调用
   时间一致；原始图像预处理须另接入和计时，不能由该模式自动推定完成。
   此接口尚未实现，当前不接受host边界协议，也没有新的host正式GPU数据。
2. Qwen原生cache/state适配、导出和C++/数值门禁，随后才可测量原生部署性能。
   当前两帧TS候选诊断通过，ExportedProgram仍未通过cache-copy动态尺寸检查。
   Goal允许无法完成时交付可审计阻塞报告；不得把Python结果升级为原生结果。
3. 最终统一证据索引、论文表格口径和回归收尾。六类算子微基准已完成，但只支持
   固定编译recipe的局部比较，不支持完整自主Agent优化或整模收益主张。
   CogACT的公共依赖/外部随机数profile限制仍必须保留。

工期是条件估算而非已启动任务的倒计时：已有结果交付整理约1-3小时；五模型
主机张量模式实现及补测约0.5-1.5工作日，若包括原始输入预处理与统一official
基线，VLA部分粗估1-3工作日。Qwen原生路线另估2-5工作日，cache适配可能超出。
前提是H20资源持续可用且不出现新的后端问题；Orin/J6M继续暂缓。
G1/G5历史扩展项不自动扩大本轮作者补实验要求。

## 2026-09-09 08:25 UTC 证据状态校正

最新闭合：pi0.5 H20正式15进程已完成并独立审计，controller4125122已退出。
17,280调用/34,560完整双输出张量逐byte一致，off/batch/required均值为
180.262594/169.152800/95.344118ms；CDF已生成并查看。五模型H20汇总为76,800条
测量和224,640份完整张量，见 `doc/reports/h20_vla_formal_table_20260909.md`。
这完成resident-tensor正式比较，不能升级为完整输入E2E、vendor或板端结果。

算子续接：新增注册握手版重复工具与完整张量离线审计器，修复旧 missing-proc
误判自有进程的问题。完整CPU回归2180 passed/73 skipped/0 failed。x8-1 GPU4
solver重复及首次x8-2 GPU0恢复因外部owner停止，均未记成五对进程完成；GPU6
恢复controller4169563已完成solver五对进程和远端/本地独立审计，操作耗时点估计
降低48.6221%，不代表整模收益。其余五类由controller4175354在GPU6继续，不能
提前标记六类全部完成。详见
`doc/reports/operator_h20_confirmation_20260909.md`；旧RoPE时延不追认新ownership保证。

后续恢复以 `doc/reports/recovery_integrity_20260909.md` 和
`doc/reports/qwen35_h20_pilot_20260909.md` 为准。两套 Qwen Python pilot 已完成并
独立审计，源码没有新增模型专用 core 分支。最新完整 CPU 回归为2180 passed、
73 skipped、0 failed，随后独立CLI入口修复的49项定向测试通过；跳过硬件/依赖项目
不计通过。pi0.5 H20正式双输出现已闭合，完整输入边界仍待补。
后处理 eager/reload/AOTI 完整700个F64值及拒绝检查已通过；双输出整模direct、C++
三调用和16帧direct校验也已通过，参考推理未重跑。三策略构建及144调用/288张量
pilot审计通过，7项真实证据反例测试通过。正式controller4125122、supervisor4125254、
首个worker4125319已实查存活；后续需15进程完成、report/audit/CDF，不得重复启动。
尚不填H20正式表。见 `doc/reports/pi05_h20_output_recovery_20260909.md`。

- 更正本节先前的平台误判：`local-pi05-replay-20260908-001` 是 RTX 3060
  双输出正式实验，不是 H20。原协议和首个正式 worker telemetry 都绑定
  `GPU-ce878329-6b58-5666-292d-94185f5e5585`。其 off/batch-only/required
  `362.055789/360.010064/350.053228` ms 不得放入 H20 主表。H20 的旧 9/15
  中断和新物化 pilot 仍各自保留，完整 H20 双输出正式证据需要补齐。
- CogACT 的旧 legacy run 004 不是当前可信版本；新的
  `cogact-numerical-h20-20260908-002` 已完成 provider-bound 三策略正式实验，
  15 个独立进程、17280 次完整调用、86400 个完整输出张量逐字节一致。均值为
  `84.869524/86.191139/76.230755` ms，required 每进程记录 1152 replay、0
  ordinary fallback。原 Meta 配置、自主 RNG 和 nominal 3B 标签仍未通过。
- 当前已确认四个模型的 H20 scoped formal，不是五个。继续优先补 pi0.5
  H20 双输出，其次是算子消融和 Qwen3.5；不重复已经完成的 scoped 实验。
- Qwen3.5 运行时探测见 `doc/reports/qwen35_runtime_probe_20260909.md`：两套真实
  权重和离线 wheel 均可用，但 Transformers 4.57.3 不识别 `qwen3_5`，因此没有
  加载权重或启动 GPU 推理。后续新隔离 5.12.1 已通过两模型配置/processor
  探测及 0.8B 图像+文本 smoke；旧 4.57.3 错误不是当前不可解决的阻塞。
  smoke 只属于官方 Python baseline，不是 VLAForge/no-Python 性能。
- H20 RoPE 算子补测见 `doc/reports/operator_rope_h20_20260909.md`：真实 OpenPI
  rotary-frequency 子图在 H20 GPU0 上完成 ATen/Inductor 各 5 个独立进程，两个
  `[1,816,256]` BF16 输出全部 bitwise exact；Inductor 的 graph-batch 操作均值
  比 ATen 低 61.7583%。该数字只属于算子微基准，不代表整模 E2E 收益，也未选择
  该候选进入部署。

## 2026-09-09 06:10 UTC 恢复与空间优化

- 本地存储优化已执行，不再只是预案：22 份内容完全相同的大模型包合并硬链接，
  释放 19,962,724,352 bytes（18.59GiB），项目约136GiB降至117GiB。
  185个候选文件所有路径、大小和SHA前后相同，源码/文档/Git保持不变；未删除
  独有产物、环境或失败证据。见 `doc/reports/storage_dedup_20260909.md`。
- Qwen两模型各13文件匹配固定revision的镜像元数据，LFS SHA、Git blob、
  safetensors keys/ranges/index验证通过。asset report SHA
  `fed0a8c669974cf0df96903cccd167998eaa0519c99df3e08f5351e788bc53fa`；
  这不是模型推理。旧“权重未发现/等待授权”状态不再适用。
- pi0.5 H20已具备完整checkpoint（SHA `7ed2fb2f...`）、9个归一化asset、
  v2 capture、AOTI和单帧normalized native。旧formal的03-off因foreign owner
  被监控终止，9个完成worker不扩大为15进程通过，也不重新启动原目录。
- `build_session_variants.py --materialize-aoti` 新增显式选项，复用既有公共
  物化部署API，保留计算/数值lineage，真实payload在NAS持久目录加载，不依赖
  native package loader的硬编码 `/tmp`。相关196项CPU测试通过；GPU验证待收尾。
- H20-1 GPU0在启动前被他人占用，门禁拒绝，未启动任务；随后绑定空闲GPU4
  `GPU-d1d90db4-423d-4393-a8c1-8b8ced20338e`，控制器PID2865031，正在构建并
  准备物化pilot。隔离目录为 NAS `runs/recovery-20260909/pi05-h20x1-materialized-001`。
  H20-2原始9轮结果保留，跨GPU参考和物化结果单独标记，不混入旧formal统计。
- 更正了平台覆盖表中pi0/pi0.5 normalized数字串位；OpenPI低精度frames8-15
  是看过全16帧后的事后子集，不是预先锁定held-out。没有追认独立验证通过。

下一步：完成上述pilot的全部输出/所有权/NAS加载核验，再做H20 pi0.5双输出与
正式矩阵。其余 G1/G4/G5、官方/vendor、完整输入边界、Qwen推理及板端仍未完成。

后续收尾：上述控制器已exit0，三策略pilot各48次调用（16 warmup+32 measured），
共144份normalized输出对官方参考byte exact，源码未变，owner监控通过。
根盘空闲前后为3647557632/3647627264 bytes，物化路径未再向根盘解包。
这是另一H20 UUID上的normalized pilot，不补齐旧9/15 formal，也不是双输出/
完整输入E2E。controller本地副本：
`artifacts/h20-recovery-20260909/pi05-materialized-controller.json`。

投稿收尾估算（条件：稳定空闲H20、无新后端大改；不是正收益保证）：
H20五模型resident-tensor统一表/完整动作/CDF与证据整理约1-2工作日；
加上完整输入边界、official/vendor对照、G1/G4/G5与回归/论文主张核对，
整体约5-10工作日，不简单累加。Qwen完整多模态部署另2-4工作日，P1排后。
Orin后端可用后约2-4日；BPU先确认SKU/OE/provider，缺后端时周级且暂不承诺总期。

## 2026-09-09 02:35 CST / 2026-09-08 18:35 UTC 恢复推进

资源核对：H20-2 八卡、H100-4 四卡、H100-8 八卡均无 compute owner；
H20-1 八卡满载且属于其他任务，未启动新作业。H100 pi0.5 formal benchmark
的 prepare 已修复两类问题并重跑成功：

1. H100 official `actions.npz` 是 NpzFile，不能用 `value.shape` 直接编码。
   公共 `session_benchmark.load_reference_array` 按声明完整输出 shape/dtype
   选择唯一的参考数组，`benchmark_session.prepare` 接入；47 项
   `test_session_benchmark_multi_output.py` 定向测试通过。CPU030 隔离全量回归
   为 2135 passed / 83 skipped / 0 failed，源码前后 SHA 不变。
2. 根目录 `pi05-h100-benchmark-protocol.json` 的 bundles 仍全部指向同一个 off
   bundle，prepare 被严格门禁正确拒绝；改用已构建的
   `pi05-h100-benchmark-variants/protocol.json` 后 prepare 成功。错误保留在
   `pi05-h100-prepared-bad-bundle-protocol` 与 `pi05-h100-prepared-bad-bundle-protocol.log`。

pilot 三策略各 32 measured：off / batch-only / required mean 为
128.168668 / 129.374862 / 66.377779 ms，required 相对 off 约 -48.2%
（pilot，不是正式统计）。

后续已闭合：H100 pi0.5 normalized 三策略正式 replay 完成并独立审计。
15 个独立进程、各 128 warmup + 1024 measured，共 17280 calls /
17280 normalized tensors 对 H100 official direct/eager 参考逐 byte exact；
no-Python maps、required 每进程 10 步 replay / 0 ordinary 通过。
off / batch-only / required mean 为 133.087191 / 135.039983 /
65.299905 ms，p99 为 148.596515 / 152.641024 / 71.752993 ms，
chunks/s 为 7.513871 / 7.405214 / 15.313958。required 相对 off mean
降低 50.934493%。聚合 report SHA
`1f0365e670946e2e5258a0584bf826e9a459e8d8eff86afae6f608a123581761`，
独立审计 SHA
`f65a27b7911edd3bf3ec2d2d8a4b7c13e0c4ff9052918280cfd1d9f1a207df28`，
CDF PNG/PDF/CSV 与图表 report 已生成并检查。完整报告：
`doc/reports/h100_pi05_native_replay_formal_20260908.md`。
仍只覆盖 normalized 单输出，physical/native 双输出 H100 正式表、同平台
五模型矩阵、official/vendor 对照与板端验收未闭合；`full_paper_acceptance=false`。

## 2026-09-09 03:35 CST / 2026-09-08 19:35 UTC H100 pi0 16帧闭合

H100 pi0 的 16 帧 no-Python native 验证已闭合：复用
`h100-series-inputs/{0..15}` 观测/noise 包与 `pi0-native-session-001`
同一 ATen sm_90 bundle，逐帧生成 H100 official normalized/physical
reference，再逐帧 native exec。16 个 `run-0.bin` 全部与对应
normalized reference byte exact，maps 无 Python，native exit0；
frame0 还与既有 `pi0-recorded-reference-001` exact。remote audit SHA
`fdf6ebd80ff3d3eb82175e98ac3d220c7d955ef655727a8a310b6abbb73fb249`，
local evidence 在
`artifacts/edgefm-vla-goal/20260906-044509/h100-pi05-aten-recovery-20260909/pi0-h100-16-20260908/`。
详见 `doc/reports/h100_pi0_native16_20260908.md`。这不是三策略 replay、
physical/double-output 正式表或板端结果；`full_paper_acceptance=false`。

## 2026-09-09 04:25 CST / 2026-09-08 20:25 UTC H100 pi0 正式闭合

同一 evidence chain 继续推进：H100 pi0 normalized 三策略正式 replay 完成并
独立审计。15 个独立进程，各 128 warmup + 1024 measured，共 17280 calls /
17280 normalized tensors 对 H100 official direct/eager 参考逐 byte exact；
no-Python maps 与 required replay counter 通过。off / batch-only / required
mean 为 109.108080 / 108.722066 / 59.051777 ms，p99 为 119.294383 /
123.295946 / 63.362188 ms，chunks/s 为 9.165224 / 9.197765 /
16.934291。required 相对 off mean 降低 45.877724%。聚合 report SHA
`d605e307029847115ca3200ad5965d3fc885798cab705203c345fa16d9ba2b8d`，
独立审计 SHA
`e69b7c99420ecade23c87a2ac4ac8835534ffa697eff06efcece8d7684eaec03`，
CDF PNG/PDF/CSV 与图表 report 已生成并检查。完整报告：
`doc/reports/h100_pi0_native_replay_formal_20260908.md`。
仍只覆盖 normalized 单输出，physical/double-output H100 正式表、同平台
五模型矩阵、official/vendor 对照与板端验收未闭合；`full_paper_acceptance=false`。

## 2026-09-09 05:10 CST / 2026-09-08 21:10 UTC H100 physical 缺口定位

尝试从既有 H100 pi0 native Session 直接扩展 physical/double-output：
`pi0-native-session-001/report.json` 已满足
`openpi_output_bundle` 的 normalized native source 条件，但所绑定的 H100
capture 仍是 legacy/1 partial context，且缺 `processor_config`。公共
`openpi_output_capture` 新增 `processor_config_for_output` 与可选
`--processor-checkpoint-dir`，可从 checkpoint provenance 补 processor
路径并保留 source_capture 绑定；20 项定向测试通过。实际 H100 运行后仍被
`NumericalContextError: legacy /1 context ... recapture required` 拒绝，
这是正确的硬门禁，不是数值失败。失败目录/日志保留在
`pi0-output-processor-20260908-failed-legacy-config`。

结论：H100 physical/double-output 不能从旧 legacy capture 追加，必须先按
当前 adapter 重新生成 v2 capture（含 processor_config 与 enforceable
context），再重跑相应 artifact audit/native Session 和 output-stage。
CPU031 隔离全量回归 2138 passed / 83 skipped / 0 failed，源前后 SHA
不变；tests.xml SHA `7b70c6906aa6e0caae19e5b04d900345c461a747454072e7b638f82fb72c01ec`。

## 2026-09-09 05:10 CST / 2026-09-08 21:10 UTC H100 pi0 dual-output 闭合

同一 v2 重建链路继续完成：H100 pi0 从 legacy capture 升级为全新 v2
capture，independent reload、AOTI audit、normalized no-Python native
Session、output-stage processor 全部通过后，`openpi_output_bundle`
生成 dual-output bundle。带 owner 监控的 native exec 3 次完整调用：
normalized F32 与 native F64 均对 H100 official 和 direct full-IR 输出
byte exact，maps 无 Python。verify SHA
`e71897a41942ad77dde04ad2860e3dc615de74d79eeb0cbb6736a2b8b5540efa`，
v2 capture SHA `9f22db202ce34e668ec943f65db441fa84e63db8c651de92461a79ff48df3e5f`，
AOTI audit SHA
`9f8977694c186e6352d948774386dc5e2fdacb6a965610ca695704d524e52c06`，
output processor SHA
`4e72416f06bd7bbfc2e24e29421fa08d32a391abf747f6a019dba0453088153f`，
v2 native Session SHA
`edd5fb0159264e8886091abc5af1a0de72a6a8de98c251e6f271af5827ef6e97`。
详见 `doc/reports/h100_pi0_dual_output_20260908.md`。这是单帧三次调用的
诊断 native dual-output，不是 formal replay/CDF；pi0.5 H100 dual-output
与五模型 formal 矩阵仍未闭合。`full_paper_acceptance=false`。

## 2026-09-09 05:40 CST / 2026-09-08 21:40 UTC H100 pi0.5 dual-output 闭合

同一 v2 重建流程已应用到 pi0.5：v2 capture、independent reload、AOTI
audit、normalized no-Python native Session、output-stage processor 全部
通过，`openpi_output_bundle` 生成 dual-output bundle。带 owner 监控的
native exec 3 次完整调用：normalized F32 与 native F64 均对 H100 official
和 direct full-IR 输出 byte exact，maps 无 Python。verify SHA
`95fa269c386e92d123d8f3a39796e9d2dc1815c4d227bac4065abc63af117ee3`，
v2 capture SHA `c257bc38576119051f14ae71e29024be227250a5f489958866798005ae7fd5e0`，
AOTI audit SHA
`aabc48a35dbfcb98e4e811286d10e94732256de3c4f4ddeb32752850b626c0bd`，
output processor SHA
`e7671abde74ca56b1012b18a81c736ac9c379139c0be64254444e1a9e240558b`，
v2 native Session SHA
`85dc4192e6825d3a596cb661106e6a9a9c8a827e3ff9b3dba83b2fc09be8d30c`。
详见 `doc/reports/h100_pi05_dual_output_20260908.md`。这是单帧三次调用
诊断 native dual-output，不是 formal replay/CDF；五模型 formal 双输出
矩阵和板端仍未闭合。`full_paper_acceptance=false`。

## 2026-09-09 07:15-07:25 CST / 2026-09-08 23:15-23:25 UTC H100 双输出正式闭合

pi0/pi0.5 的 H100 dual-output 16 帧三策略正式 replay 已完成并通过独立
审计：

- pi0：off/batch/required mean `107.459741/108.077134/59.860582 ms`，
  required 相对 off 降 `44.294877%`；17280 calls / 34560 tensors 全部
  byte exact。aggregate SHA
  `ac822e8860351d598e28db09b4fb4f64278025b04a31aa572a01472b9b9a8b50`，
  audit SHA `3f155ce6f3394af3bb5751beb56a65739005aecdb0df2c4beb7b8823f7180b48`。
- pi0.5：off/batch/required mean `126.277332/123.479209/64.436106 ms`，
  required 相对 off 降 `48.972547%`；17280 calls / 34560 tensors 全部
  byte exact。aggregate SHA
  `b42681b44d6d3a54ddfa2a83c0e54a7accd4d62cbcf15a944fd9d3dfcf557939`，
  audit SHA `2ee27314eafa8bcf09cf93874a67f12dc3ccf7f7dafa98072f576f7e1d6614d2`。

报告：
`doc/reports/h100_pi0_dual_replay_formal_20260908.md`、
`doc/reports/h100_pi05_dual_replay_formal_20260908.md`；本地汇总与图在
`artifacts/edgefm-vla-goal/20260906-044509/h100-pi05-aten-recovery-20260909/dual-formal-summaries/`。
这是 H100 resident model-tensor 边界，不是 Orin/BPU；
`full_paper_acceptance=false`。

## 2026-09-09 07:40 CST / 2026-09-08 23:40 UTC 平台覆盖与传输负证据

新增 `doc/reports/cuda_platform_formal_coverage_20260908.md`：按 H100/H20
分别列出当前正式行，并明确缺失格。H100 已覆盖 Smol/pi0/pi0.5，H20 已覆盖
Smol/RDT/pi0/CogACT，均不是完整同平台五模型表。

H20 pi0.5 缺口：partial checkpoint 已有 17/27 ranges，剩 2.67GB。H100 到
H20 无直连 DNS；经本机转发的实测约 0.15-0.6MB/s，并行两路未见改善，未继续
浪费带宽。此传输负证据保留；下一步需要在 H20 上提供更高速的资产通道，或由
H20 本机已有镜像/授权账号直接放置 pi0.5 权重，而不是继续跨机搬 6.8GB。

## 2026-09-09 09:30 CST / 2026-09-09 01:30 UTC G5 OpenPI 全 Linear FP16

新增公共整图变换 `lower_exported_all_linear_halves`：不改图签名/state
存储，把 `aten.linear` 输入、weight、bias 显式 cast 到 half，Linear 结果
再 cast 回原 dtype；不依赖 scheduler/step 输入，不按模型名分支。
`test_linear_precision.py` 20 项全过。

H100 真实 pi0/pi0.5 step Region 已分别重写 131/167 个 Linear 为 FP16
compute，并保存/重载。16 帧 Python free-running 全部过原 paper gate：

- pi0：worst MSE `1.3312e-06`、worst max-abs `0.006966`、最低 cosine
  `0.99998424`。
- pi0.5：worst MSE `3.3693e-07`、worst max-abs `0.004152`、最低 cosine
  `0.99999710`。

证据与说明见 `doc/reports/openpi_h100_all_linear_half_20260908.md`。
这是无 calibration/held-out 划分的 Python 全模型质量结果，不是 native、
时延、双输出或板端验收；不称为无损。

最新回归（2026-09-09 03:20 UTC）：CPU027为2213 passed/83 skipped/0 failed，
报告SHA `aae953e5c3eb85786e7648b470b14e7e9b35007539a2e24053b05c3a574ea52b`。

最新推进（2026-09-09 05:10 UTC）：H100 pi0.5 ATen no-Python Session真实通过，
explicit numerical provider enforced，2次完整动作与same-artifact及official均exact。
remote report SHA `073c6926e339b05b07291db42358f5389727157b9da4fc07be929bde6bdaf68a`。
为此新增legacy compile binding与v2 projection公共修复；CPU028为2216 passed/83
skipped/0 failed，报告SHA
`2906e5c224240183f67765b8d1e0841190062f2bbe479613683e7493935b5c08`。
详见 `doc/reports/h100_pi05_native_20260909.md`。

最新推进（2026-09-09 05:20 UTC）：H100 pi0 ATen no-Python Session也真实通过，
native PID1858650，2次完整normalized action与same artifact/official均exact，
provider enforced。remote report SHA
`1bc27e11bef115ed5d2dc5f460e1d22b4df55be724990cfca1174814f7371c7e`。
详见 `doc/reports/h100_pi0_native_20260909.md`。

最新推进（2026-09-09 05:40 UTC）：pi0.5参考版本差异已确认并归档。H100 frame0
输入与本地frame0逐byte相同，config/revision相同，但normalized reference不同；
本地16帧包不能直接并入H100正式矩阵，需以H100生成的全帧reference为准。
详见 `doc/reports/pi05_reference_version_discrepancy_20260909.md`。

最新推进（2026-09-09 06:20 UTC）：H100 pi0.5已生成16帧自洽official references，
同一native bundle逐帧运行全部byte exact。normalized/physical 16帧hash均互异；
reference生成与native验证在H100闭环。formal replay仍待做。
详见 `doc/reports/h100_pi05_native16_20260909.md`。

最新推进（2026-09-08 23:35 UTC，优先于以下历史快照）：G5新增通用FP16/BF16完整轨迹
质量分支。公共`deployment/half_linear.py`与`linear_precision.py precision_kind`
支持float16/bfloat16替换；RTX3060真实原IR自由推进每种16条轨迹（前8校准/后8
held-out）全部通过原MSE<=1e-5/cos>=0.9999。FP16 worst all MSE2.550426e-6、held-out
2.434350e-6；BF16 worst all MSE5.291621e-6、held-out5.109796e-6。FP16 audit SHA
`08264a3bfd3671f41de8737e90eab836b4e393e52f738880d714b08da82105a0`，BF16 audit SHA
`f7e3f514b11124270f6334522ba67d04c0ee3e08196cb9b4bf01e782d3571cba`。supervisor
PIDs2067408/2074915 exit0/最终owner空。这仍是Python IR诊断，不是no-Python原生/
性能/无损；先前INT8四策略0/8失败保留。CPU026为2213 passed/83 skipped/0 failed，
报告SHA `488a594551611a6ed8f9a1afd5ae497f0b85b69b6be670874940ec3b2eb8580a`。
下一步原生FP16/BF16 artifact与同源性能/内存。详见
`doc/reports/half_free_running_20260908.md`。

继续推进（2026-09-08 23:40 UTC）：FP16 step Region已完成公共TorchScript导出，
`fp16-step-native-v1/step-half.pt`为907403934B，SHA
`c33e79922bb9f09925bdf7aa5174bfad716694530e9e5f2822e8ce347f768b7e`；11个验证
case（原始example+10个step index）save/reload全部bitwise。这是no-Python完整Session
前的Region传输闭环，不是完整原生验证。下一步组装三个共享原Region TS + 新FP16 step
TS并运行full-model Session。

最新推进（2026-09-08 23:45 UTC，优先于以下历史快照）：FP16 no-Python完整模型
Session已闭合。RTX3060上共享三Region+FP16 step四TS、一个quantized-lane C++ bundle、
单原生Session 16完整动作全部byte exact于FP16 Python轨迹；16/16过原质量门，
held-out8/8。maps无libpython/libtorch_python；native PID2077538 exit0。
protocol SHA `c5e34be3e5890d28322ce5a835e8d22fd9e5fdf3f4f2c755035ba1ee424d43c1`，
trace/build/native/execution/audit SHA
`4b91b95ffe077a653572d7ae2b50ef289854fd31051de8249ee438218afe52e5`、
`eeec953aa1d662bf10b33faf35f0af13d950d1eb32cf7def2bdafc779718b370`、
`873bc3fb92515f2edcf0a9f0905fdcc6eaa92d824e2eb6ae97d329e0d1c43e2e`、
`ed2342bdd1011aa742001434ad0de0447c72e5b94d2209a795f0e720e38e6f12`、
`35c660042d1785847ef2c98a3b529876446830118108d75b95e22022c827af85`。
这不是正式时延/内存/replay/BF16/无损。详见
`doc/reports/fp16_native_full_20260908.md`。

最新推进（2026-09-08 23:55 UTC，优先于以下历史快照）：BF16 no-Python完整模型
Session也闭合。RTX3060四TS+C++ bundle+单原生Session16完整动作byte exact于BF16
Python轨迹；16/16过质量门，held-out8/8；maps无Python。protocol/trace/build/
native/audit SHA `6a6e7aef96bd846fd6d854067290f75a0b69251f87b931c3015836a7a336d96d`、
`71f98150707c29108c6fb30815911bdf0f5a09597035f5cacf732fc5230d2e26`、
`2332246f59df50bb3b73c91107b8ad7c029fcf04e3a11ab04d9eb9e52d91e631`、
`9fcb7974b85f73e95cf4b0cb1443355828b131d3feea24b993bb7f95c55315ef`、
`0002d4d3ce0b3f99db79ad6cb63f7302ffc4a9c1033fc78cfff21aa097cfc7f9`。
详见 `doc/reports/bf16_native_full_20260908.md`。

最新推进（2026-09-09 00:00 UTC，优先于以下历史快照）：baseline/FP16/BF16同源
RTX原生pilot已跑通。单进程16warmup+32measured、每lane48调用byteexact、maps无
Python。mean baseline/FP16/BF16=93.137304/94.474481/94.115055ms，half不更快。
allocator steady requests baseline410912 vs half413152（+70/call），allocated peak
baseline1884473856 vs half1884520448（+46608B），reserved peak与destroy后active
相同；不是零分配或省内存。这是descriptive pilot不是formal CDF/主表。
详见 `doc/reports/half_native_benchmark_pilot_20260908.md`。

最新推进（2026-09-09 00:50 UTC，优先于以下历史快照）：三路正式5x1024原生benchmark
完成并独立重算。mean baseline/FP16/BF16=93.351683/94.182185/93.792310ms，FP16慢
0.8896%、BF16慢0.4719%，无原生加速；allocated peak半精度+46608B、reserved不变、
steady allocator请求+70/call。formal analysis SHA
`ef615c48383372763e4a1668b4eaf4096762d4956faefa3862f1d19853bfe074`；CDF PNG SHA
`1b171248a257c6953b09df026982a6a587f8ab2f6292eda5b216623af2efcc9e`。
off-only普通执行，无replay/profiler/板端。详见
`doc/reports/half_native_benchmark_formal_20260909.md`。

最新推进（2026-09-09 01:10 UTC，优先于以下历史快照）：FP16/BF16逐步误差预算完成。
step0 head输入byte exact，step9 worst held-out MSE FP16/BF16=1.149572e-4/1.261380e-4，
max-abs0.0625-0.09375；最终动作MSE仍2.434350e-6/5.109796e-6过门。中间状态发散不能被
表述为逐张量lossless；report SHA
`6386d259119f70279a4c26b7d6be454d1f4be3b0386d611bad8a7a1c23d4e279`。
详见 `doc/reports/half_stepwise_error_20260909.md`。

最新推进（2026-09-09 01:20 UTC，优先于以下历史快照）：RTX NSYS CUDA profiler三路
完成（1warmup+1measured，CPU采样不可用关闭）。GPU kernel采样总时
baseline/FP16/BF16=167.510/167.362/167.804ms，top为模型既有BF16 attention/GEMM；
half action head非主导kernel。profiler overhead不计formal。详见
`doc/reports/half_native_profiler_20260909.md`。

最新推进（2026-09-09 01:40 UTC，优先于以下历史快照）：baseline/FP16/BF16三策略
whole-loop replay pilot完成，48调用/策略byte exact，required counter
`REPLAY_FINAL,11,1,10,48,0`。required pilot mean baseline/FP16/BF16=
76.290/76.611/76.780ms，较同lane off约降18-19%；descriptive仅，正式replay CDF待补。
详见 `doc/reports/half_native_replay_pilot_20260909.md`。

最新推进（2026-09-09 02:20 UTC，优先于以下历史快照）：三路required正式5x1024 CDF
完成。mean baseline/FP16/BF16=77.170051/77.217964/77.271447ms，每进程counter
`REPLAY_FINAL,11,1,10,1152,0`；相对formal off降17.33%/18.01%/17.61%，half不改变
replay收益。详见 `doc/reports/half_native_replay_formal_20260909.md`。

最新推进（2026-09-08 15:08 UTC，优先于以下历史快照）：CogACT数值绑定002正式15进程、
report、修正独立审计、本地取回复算和CDF均完成。17280调用/86400完整五端口张量/
6099840值对official+direct逐byte exact；required每进程N10 capture/1152 replay/
0 ordinary，mean76.230755ms、p9978.860851ms、13.118065chunks/s，相对同源off
mean降低10.1788822%，采样显存+140MiB。remote formal audit SHA
`92df2490f9383ad6ba75bafb3eaf8bb4bd568168116f8abaeb0d438632eb7014`，原链失败保留，
audit recovery SHA `95d44cd91d869d2a50e4eec7339df6a3aa9352b17fc45d4d2ec175bbdad4bc77`，
本地871文件/127375828B取回复算SHA `504f3905251847167d9c81ef9f79460f8f82201697c6b17b9c93fdf98d2e50dc`。
跨campaign汇总005含6模型（Smol/RDT/pi0/pi0.5/CogACT）18策略行、92160测量调用、
259200完整输出张量复核，summary SHA `c146271c79d5072d8fe5d351ef6dac7f1b355646f4d0098280c2f195b522de36`；
不合并设备/不生成跨设备加速比。H100同源graph-memory retain/scoped对照也已完成：
30次Session生命周期、480调用/960完整F32张量/288000值exact；required下retain销毁后
reserved每周期+4MiB（净+16MiB），scoped-reclaim每次销毁2次scoped device free且
reserved恒定1937768448B。remote audit SHA
`bafebd980163b9f7fe0f4234aea648f29c4110229a994aa0f1b871d5623908c3`，retrieval SHA
`e1d2272246924a8919f8b0a0dae866d178971acd43849f44aca6f11f2f3a2b0d`。这是有限5次
Session/thread和仅Torch2.10路径的内存回收证据，不是零分配、所有权全覆盖或性能数据。
全量CPU022-025为2119-2209通过、83跳过；新summary/lifecycle测试含真实归档负测。
详见 `doc/reports/cogact_numerical_formal_20260908.md`、
`doc/reports/session_lifecycle_memory_pair_20260908.md` 和
`doc/reports/session_lifecycle_numerical_20260908.md`。

本轮继续（2026-09-08 12:50 UTC）：上一轮为真实进展；本轮已核对H20-2无遗留本任务
进程，未重复启动pilot。新CogACT数值绑定002正式控制器PID3803445、采样PID3803499
仍实查存活，三策略首组3/15已通过完整输出检查；仍用原冻结包，128+1024、五进程。
`formal-pipeline-001.json`记录新正式链。新汇总审计器的JSON列表/tuple比较错误已用
历史真实formal数据复现（1失败/9通过），独立002修正后10项通过；运行中的原脚本不改，
原链最终审计预计拒绝后将只补审计，不重复模型采样。全阶段完成仍待正式输出/独立审计。
公共新增模型无关`tools/summarize_session_campaigns.py`，逐完整输出和raw CDF复核后
按campaign与实际GPU分行，不计算跨设备加速比。当前6项接口测试、9项真实归档负测
通过；CPU022为2119通过/83跳过。新增索引变更拒绝后CPU023正在运行，不能继承022。
首次跨campaign汇总因pi0旧归档缺serialized reference子集而拒绝，尚未发布表格；
独立view补齐按原冻结hash检查，不改旧归档。既有模型证明未因此改写，板端继续延期。

最新结项范围（2026-09-08 12:22 UTC，仅本轮子任务，不是Goal结项）：SmolVLA H20同机
官方参考、完整双输出IR、三策略原生pilot/正式15进程、独立审计、本地648文件取回/
34560张量10368000值逐byte复算及CDF实际查看均完成。off/batch/required mean为
117.632460/118.379805/47.112908ms，required p9949.797612ms、21.225606chunks/s。
相对off平均降低59.9490584%，但采样显存增加778MiB；不是省内存或Agent算子收益。
远端audit SHA `66a2f82f7669e6369206bb2d92af8c144485152e4a273b7330b8ee539d3807bb`；
本地closeout SHA `c039235b976528a9d5ebb0e4087d1f467f18888386d1d32352433ba67f6eb77d`。
GPU独占登记通过，但主机CPU/NAS与本项目CogACT构建共享；此工况保留，不宣称无干扰极限。

CogACT新数值绑定002的四Region重新编译、16帧完整五输出IR和三策略C++ pilot现均
通过。原生独立audit为144调用/720张量/50832值exact，并实测provider-required约束。
原始pipeline最后因缺pytest失败，未改其failed状态；隔离支持库后12项实际证据测试
补跑通过，模型环境/模型执行/原生包均未更改或重跑。548文件本地取回、720完整张量
复算也通过，closeout SHA `c8430b442226c2a4ca73b320b1ac5fcfadbaa4d06f8473414e68e2b6d33e2868`。
这只闭合新candidate的provider缺口；旧formal/CDF不继承，新版正式长测尚未执行。
原Meta配置、7.63B规模差异、自主RNG、板端限制仍在。公共编译接口CPU021为2113通过/
83跳过，另外2项真实C++ CPU策略测试通过，后者为fixture接口证据，不充当模型结果。

资源实查 `resource-preflight-20260908-121459.json`：H20-x8-2八卡、H100-x4四卡及
H100-x8的1/3/4/5/6/7号六卡未见计算owner，共18张；H20-x8-1及H100-x8的0/2号卡
有其他任务，不占用。空闲不是预约；启动前再次检查。GPU条件充足，不等待新增GPU。
本轮运行与审计进程已退出；下一步先按新数值约束版本做CogACT正式复测与同平台矩阵，
补官方/vendor和更受控工况对照，再推进G4/G5缺口。细节/恢复命令见
`doc/reports/smolvla_h20_20260908.md` 和 `doc/reports/torchscript_numerical_compile_20260908.md`。
以下历史“运行中/缺provider”等记录按各自时间和产物理解，不代表最新002状态。

最新推进（2026-09-08 11:01 UTC，优先于以下历史快照）：四机资源已再次核对，GPU不是
当前阻塞。10:31记录中H20-1已出现其他owner、H100-x8八卡被占用，H20-2八卡与
H100-x4四卡仍空闲；瞬时空闲不是预约。随后在H20-2 GPU0启动新的SmolVLA同平台链，
已完成真实权重/339项上游源/16真实episode输入核验、新官方参考与完整IR exact、
五TS Region含官方反归一化、三策略原生pilot和独立144调用/288完整张量审计。九项
真实证据测试通过后正式15进程长测已启动，PID3775194，控制器3766345；尚未完成。
pilot audit SHA `9a9004795b3014783daff065466e50fb50d240ede0ce1d825384d1052482cd66`。
慢权重rsync的主动终止/主站连接失败均保留，固定revision镜像传输的12文件全量SHA/size
核验通过才启动模型。详情见 `doc/reports/smolvla_h20_20260908.md`。
公共TorchScript新增观测数值策略的编译接口/CLI，既不自动改全局flags，也不把compile
记录当runtime provider证据。CPU021为2113 passed/83 skipped/0 failed，冻结源前后不变。
CogACT新数值绑定001因错误比较捕获/重载图hash而停止，真实480节点probe复现仅命名
变化、canonical结构/状态/提供的Region输出一致；保留失败，用实际编译图identity建立
新契约并单列旧capture来源。002现于GPU1编译，完整五输出与C++ provider仍待新验收。
见 `doc/reports/torchscript_numerical_compile_20260908.md`。原Meta配置、自主RNG、低比特
held-out质量、稳定Agent整模收益、Qwen与板端任务均不因此标成完成。

最新推进（2026-09-08 09:58 UTC）：09:46四机再次实查，H20-x8-1现有其他任务约94GB/卡，
不占用；H20-x8-2八卡、H100-x4四卡、H100-x8八卡均无compute owner，共20张瞬时空闲。
原始记录 `resource-preflight-20260908-0946.json`，GPU资源不是当前阻塞。
SmolVLA H100-x4正式三策略与本地归档已完成。新H100-x8 solver实际同包整模三路正式
实验也已完成：17280调用/34560完整张量/10368000值exact，远端独立审计、本地复算和
CDF实际查看均通过。original/control/candidate mean44.875762/44.090836/43.748341ms；
候选相对control点估计降低0.776794%，但配对CI为[-0.740279,1.587530]ms、sign-flip
p=0.28125，不成立稳定收益；selection/2记录真实集成true、数值true、最终rejected。
正式audit SHA `a1a6189a28ebe776871890ff8429667894ca3a9cb540e924830ce0d411c87021`。
这修补了G4“同一候选接回整模”的实测证据缺口，但尚无被选中kernel的稳定E2E收益。
CogACT004所有15正式worker、汇总、远端独立审计、本地766文件取回/原始复算及CDF
实际查看均已完成：86400完整五端口张量/6099840值exact，required mean75.538978ms、
p9978.687230ms，较off点估计降低11.7295225%，采样显存增加140MiB。正式audit SHA
`63d15591ad3b9bce8cb11f38cedb0a23c1d3c10d7f9b340ab6faa25966b0eb9f`，本地SHA
`4337a597236190824f9c855edf116ef185b6ed334ff28192dc0bf99b8892975b`。原Meta配置、
实际7.63B规模、外部随机数带与legacy provider缺口仍保留，paper-table eligible仍false。
本轮所有模型、构建、正式实验、审计及传输进程均已退出，没有重复启动原作业。
另修复公共独立AOTI加载器依赖初始化及参考读取重复I/O：缓存仅限一次校验，所有实际
完整输出仍逐个验证；真实pi05的34560张量/39744000值新旧校验器绑定完全一致，参考读取
由69120降至960，原数据不变。audit SHA `b6b5dc4e4d8190161be7be6c84c659b7fd6f7663899af24fce25b60bf37cc728`。
缓存不改运行中的冻结源码，也不算模型时延收益。最新CPU017为2102 passed/83 skipped/
0 failed，冻结源前后不变。详情见Smol与operator报告，下文旧“未集成”仅为历史阶段。

剩余工时与条件（09-08工程估计，不是GPU排队时间）：当前CUDA P0约5-10个有效工作日，
取决于数值/后端兼容问题；不是保证完成日期。主要剩余为G1后端内存与状态覆盖、G2
CogACT原配置/自主RNG/provider补齐、G3同平台矩阵与官方/vendor对照、G4稳定正收益
候选和skill复用、G5 held-out低比特质量（当前仍0/8过门槛）。现有CUDA资源可继续推进，
无需等待新增GPU；CogACT原配置需要可验证原文件或相应获取权限。G6真实多模态Qwen
另估2-4个工作日起，遇新增算子/capture问题需调整。D1/D2需要实机访问、具体SKU、
JetPack或OE/运行时版本及测试工况，尚不能给可靠全论文完成日期；data-only包不是实机成绩。

最新推进（2026-09-08 09:08 UTC，优先于下文历史快照）：新增H100-x4 SmolVLA三策略
正式15进程、远端独立审计、本地644文件/220024511B取回与原始复算、CDF实际查看均完成。
17280调用/34560完整F32双输出张量/10368000值逐byteexact；off/batch/required mean
118.170567/113.808212/45.971896ms，required p9957.069162ms、21.752420chunks/s。
平均时延降低61.0969997%，采样显存增加566MiB；所有长尾保留，不声称省内存或Agent
kernel收益。远端audit SHA `b2171d87c7d364cd3b408daea3a86587d7ab37352f6742c52f0f1f661af89f4c`，
本地audit SHA `42824393d545385a7c88d326586b90edce15cd9d549341339d48fda898bd66e9`。
详见 `doc/reports/smolvla_h100x4_20260908.md`；不是同平台五模型矩阵或板端结果。
CogACT004现13/15正式worker通过，原pipeline继续运行。H100-x8 GPU1新独立solver
工作负载已重新生成16帧同机官方参考、160真实step边界，完整split IR exact；三候选
各320完整step输出exact。微基准中位数ATen/Inductor/preserving分别3.897455/2.255696/
3.766021us，仅独立微基准，未作kernel E2E收益。整模集成首次AOTI延迟导入失败，
第二次导入早于GPU登记被门禁拒绝，均保留；新003恢复使用原测量packageSHA不重编译，
完整direct/构建尚在执行。GPU不是阻塞，其他owner不终止。

最新恢复快照（2026-09-08 08:26 UTC，优先于下文历史状态）：四台主机在07:32 UTC
实查均无GPU compute owner，合计16张H20和12张H100；不是GPU短缺。原始快照为
`artifacts/edgefm-vla-goal/20260906-044509/resource-preflight-20260908-0732.json`。
新增授权的H100-x4已纳入Goal，作业分别隔离，不把瞬时空闲当预约。
CogACT `cogact-replay-20260908-004` 已从本地冻结源码完成三策略构建与新pilot，
独立审计及12项真实证据测试通过；pipeline PID3724808已进入15进程正式长测，未重复启动。
H100-x4 `smolvla-h100x4-20260908-001` 已核验本地源码/16帧输入/真实权重/339项上游源，
参考捕获worker PID1550835退出0，16帧新同机官方参考/捕获完整IR exact。
追加官方反归一化与五个新TS Region，完整双输出IR exact；CMake首次缺OpenSSL开发
路径失败，单变量指定既有 `/opt/conda` 后配置通过，失败目录保留。新 `complete-002`
三策略原生pilot、独立审计及9项真实证据测试通过，正式15进程实验已启动，尚未出齐。
本地另复算48完整输出/14400值exact，并用NumPy独立核对mean/std逆变换，audit SHA
`63f05ab34fe9db70e1dac7be0cf31710185c9f9eea10b23bb6631d2c1a1e4d65`。
所有旧失败、源码和原传输收据保留，不将此本地IR复算冒充C++或正式结果。
另重核G4旧配对记录：H20 Embedding AOTI候选被错误配到RTX静态预计算的
original-to-unchanged-control对照，不能作为同候选同平台集成证据；120项原始证据哈希
重核通过，不修改历史decision。独立审计SHA
`1a94c630c256432056a7e9f707768b69e56c898f7ee49bd62b5af934d0246911`；
当前无选中kernel的有效E2E集成结果。公共selection/2现已收紧候选artifact、GPU类型、
测量/输入/输出/数值契约与双lane均值绑定，显式要求置信门禁；29项定向测试和实际旧
误配记录的拒绝测试通过。它仅校验元数据一致性，不能代替真实bundle/执行原始证据审计。
全量CPU回归015为2097 passed/83 skipped/0 failed，源码前后不变；上述远端作业继续
使用启动时冻结源码，没有将新selection改动同步到运行中的实验目录。

恢复执行核对（2026-09-08）：实查并保留已有RDT正式控制器PID3607880、监督进程
PID3607882，未重复启动；隔离run `rdt-replay-20260908-003` 现已完成三策略正式实验、
汇总、远端独立审计、本地原始数据复算和CDF实际查看。15进程各128预热+1024测量，
15360正式样本、17280完整调用、34560完整BF16张量/157040640值全部对官方及同artifact
逐byteexact，MSE/max-abs为0。required每进程5步capture/1152replay/0ordinary；
off/batch/required mean176.710522/172.734449/164.397820ms，required p99168.415117ms，
平均时延降低6.967724%。采样显存由13675增到14283MiB，不声称省内存。
远端audit SHA `95a17d627aed2fb59c591bda725f1ea74137d7236ec85e7481b817845693b8fe`，
本地audit SHA `47d5a91a1c307f1a31a56e8e3edc99c0b04fcce225c745bbfb3c25b696977d9e`。
所有本轮GPU执行、report、audit、rsync进程均已退出；旧pilot文件逐一确认未改。
上次未完成任务引用的旧单输出001失败记录、恢复002主动中止记录均保留，不代替双输出。
详情和复核命令见 `doc/reports/rdt_execution_variants_20260908.md`。
本段最新状态优先于下文历史摘要。RDT运行使用已冻结292文件源码；本地新增的data-only
板端数据契约不rsync进该运行目录。五模型完整输出离线包现均已移目录复验，
Smol重新核对34560张量、CogACT重新核对5750张量；后者历史audit哈希断链单列，未作为
通过依据。详情 `doc/reports/vla_board_data_completion_20260908.md`。新全量CPU回归013为
2074 passed/83 skipped/0 failed，源前后不变；板端仍未执行。
pi0.5追加本地三策略多观测pilot：16帧144调用/288完整F32+F64张量逐byteexact，
10步capture/48replay/0ordinary；audit SHA `d33b2d488ff32d11798269327db8cc1fd0879e53c519f1cc31094faecd4cea5c`，
6项真实证据审计测试通过，所有本地pilot进程退出。首次prepare权限失败保留，新prepare-002
通过后才执行。正式三策略于2026-09-08 04:34 UTC启动，恢复后未重复启动；现15进程各
128预热+1024测量、独立审计、完整表和CDF实际查看均完成，所有worker退出。
17280调用/34560完整F32+F64张量/39744000值逐byteexact；off/batch/required mean
362.055789/360.010064/350.053228ms，required p99351.884728ms，平均时延降低3.315114%，
每进程10步capture/1152replay/0ordinary。采样显存+120MiB，不声称省内存。
audit SHA `b50b40f7c889e56a5186987815c91a0556dbb7e5d8fd8c7e096e08f516641f11`。
详见 `doc/reports/pi05_execution_variants_20260908.md`。
CogACT新16帧公共协议三策略已构建；run001 off/batch pilot通过，required在首调用
CUDA capture失败，pipeline已退出且未启动正式采样。真实step定位CPU常量向CUDA复制，
通用literal-only precompute仅新增512B常量后真实N10 graph三次replay exact，保留动态
timestep/CFG/noise。run002全IR16帧80完整输出exact，但捕获证据归档错误使base构建失败；
新run003已用公共保存接口重新捕获，独立完整IR16帧80输出exact，新base/required C++
构建与48调用/240完整张量pilot通过。H20-2 GPU0出现其他owner后拒绝启动，转到空闲
GPU1；native PID3719342退出0，实际10步capture/48replay/0ordinary，旧失败/占用记录
保留。五端口保留F64 native与U8/I64 exact输出；独立audit SHA
`981f98dce0442563c3972c7f3971f4cab5c6fe3e3e046ce14310478ec7926458`，12项真实证据测试
通过。正式同产物三策略长测仍待执行；原Meta配置/自主RNG和legacy数值provider缺口未验收。
本地另重核240张量/16944值exact，audit SHA
`e0ae324fa447fdced9264999042434e40a5b33913bb1e468ae4783bad77be345`。首个指标JSON精确比较
因63处cosine/norm差1 ULP失败，保留失败记录；显式跨主机模式仅容许这些归约元数据
至多4 ULP差异，不放宽原始字节/MSE/max-abs/整数门禁，14项真实数据测试通过。
本轮本地/远端作业均已退出；07:12 UTC H20-2 owner为空仅为瞬时快照。
公共修复回归014为2079 passed/83 skipped/0 failed，另11项C++ CPU测试通过。
详见 `doc/reports/cogact_execution_variants_20260908.md`。

最新核对：2026-09-08（北京时间）。Smol、RDT、pi0、pi0.5 的三策略正式replay与完整输出审计现均完成，但硬件不同，不能拼成同平台模型矩阵。pi0新H20 required mean84.651419ms/p9986.142677ms，相对off平均时延降低39.373919%，无ordinary回退；采样显存反增668MiB。CogACT 新完整C++ required replay pilot现已通过48调用/240张量，正式公共协议稳态CDF仍待补；旧115帧/10进程没有剔除预热。pi05正式及CogACT pilot GPU进程均已退出，下一任务启动前仍须重查资源。

维护核对：2026-09-07（本地）。磁盘维护计划 `disk-maintenance-20260907/` 已按固定计划哈希 `2200df3fb2871335b18877861dac87e4c2c45a30dbd5766c401c40264a7716c1` 执行并完成 88/88 个动作；释放约 58.6 GB（可用空间由约 3.2 GB 增至约 57 GB）。所有保留权重、`.pt2` 包和受保护报告哈希未变化，未终止任何外部进程。该维护只改变可重建传输分片、重复包和可从已保留包恢复的编译缓存，不改变任何验证结论。

Replay 入口核对：OpenPI 输出 bundle Adapter 现接受显式 `--loop-execution off|batch-only|required`，默认仍为 `off`；`test_openpi_output_ir.py` 与 `test_replay_codegen.py` 定向回归为 33 passed、2 个显式 CUDA replay skip。该改动只开放通用构建参数，尚未把历史 off bundle 追认为 replay bundle。

回归核对：`resume-cpu-regression-011/report.json` 记录 2049 passed、71 skipped、0 failed，pytest 70.48 s（外层72.53 s）；日志 SHA `2ad3199ed343756faa6827f8ff456e5eb943615fea3eba34d118290cd05907c7`。核心源码前后 SHA 一致；skipped 不计作真实模型或硬件验收。

执行定义：[Goal prompt](edgefm_vla_paper_goal_prompt.md)。本地源：`/home/zhangzimo/Repos/private/edge-fm-x`，启动 HEAD：`592fc320d57398571dcb4ab686d5fb55fe9bac86`；工作树已有通用 frontend/loop/cache 改动和用户论文，全部保留。开发态结果必须保留 dirty/source hashes，不能当作冻结版本正式成绩。

## 任务矩阵

| ID | 状态 | 已验证/正在执行 | 剩余与下一步 |
|---|---|---|---|
| G0 基线与论文映射 | running | `edgefm_vla_claim_evidence.md` 已逐项映射 III-A 至 III-D；source/checkpoint/dataset/hash 与回归记录已归档，H100 隔离环境通过 | 随后续切片更新冻结源及证据；不能用开发快照成绩作正式主表 |
| G1 通用 runtime/memory/state | running, bounded lifecycle slice passed | unused-state完整IR peak reserved由14.370降到6.780GB且exact；H100同源retain/scoped 30次真实Session/480调用/960张量exact；scoped required销毁后reserved恒定，retain每周期+4MiB | 仅Torch2.10显式scoped路径；默认retain不继承；有限5次生命周期，非零分配/完整arena接管；线程/全部backend/失败路径另验 |
| G2 真实 VLA | historical snapshot; autonomous RNG later closed | Smol/RDT/pi0/pi05三策略双输出与CogACT三策略五输出正式replay已分别完成；新增H100 Smol34560张量、H20 Cog86400张量、H100 pi0 16帧normalized native byte exact及H100 pi0 dual-output native诊断3 calls exact，远端/本地独立审计通过；Cog每进程N10 capture/1152replay/0ordinary | 该时点CogACT原Meta配置、规模标签、自主RNG和统一vendor对照仍未闭合，外部随机带沿用；autonomous RNG后续已由2026-09-10同边界报告闭合，原Meta配置和统一vendor对照仍未闭合 |
| G3 CUDA 主表/CDF/保真 | running, per-model scoped CUDA CDFs archived | 五模型现均有公共128预热+1024测量/五进程/三策略的完整输出正式CDF；H100 pi0/pi0.5 dual-output 16帧三策略正式replay通过（各34560 tensors exact）；Cog旧10x115和legacy unbound另列不混入；Cog新数值002 mean76.230755ms/p9978.860851ms，H20 PNG已生成并通过像素非空检查 | 同平台五模型矩阵、官方/vendor和板端主表未完成；不能混不同GPU做排名，也不把model-tensor当sensor-to-action |
| G4 Agent/算子/执行消融 | running, exact same-package E2E measured; no candidate selected | H100新solver三候选各160step exact；同微基准AOTI包接入完整模型，三路正式34560张量exact；candidate/control mean43.748341/44.090836ms，点收益0.776794%但CI跨0，selection/2正确rejected | 稳定正收益与选中kernel仍缺；其余Attention/GEMM/Norm/Embedding/RoPE匹配整模及同平台覆盖、Agent skill复用待补；旧H20/RTX误配无效不追认。见 `doc/reports/operator_agent_budget_20260907.md` |
| G5 逐步低精度 | running; formal off/replay CDF complete, model-family coverage started | INT8四策略held-out仍0/8；Smol FP16/BF16 Python/no-Python全链8/8 held-out；OpenPI pi0/pi0.5 H100全Linear FP16 16帧Python free-running均过原paper gate；RTX formal off/required CDF与error budget/profiler完成；CPU030/CPU031通过 | OpenPI native低精度、held-out split/校准纪律、INT8正结果或收敛负结果待补；INT8负结果保留，不因half通过而宣称INT8质量无损 |
| G6 Qwen3.5 补充 | passed on H20 fixed profile | 两套完整权重的原生vision+prefill+decode、first/16-token共20进程、20480测量和46080完整输出张量已完成；四组独立审计与CDF通过 | 任意输入、流式逐token、Orin/J6M/BPU外推不在当前结果内 |
| D1 Orin 验收 | deferred-hardware, five complete-output data packages prepared | Smol/RDT/pi0/pi05完整双输出与CogACT115帧五输出包均保留原输入/noise及类型并移目录复验；CogACT旧audit断链明确排除，5750原始张量另独立重核 | target driver/SDK未知pending；实机主表/CDF/功耗热稳态及target完整输出待补；均非板端可执行包 |
| D2 BPU 验收 | deferred-hardware, data and driver contract prepared | 五模型可移机输入/完整输出包、typed契约和四阶段dispatch脚本齐备，五包共40次Orin/BPU shell入口均exit2/pending且未启动driver | 确认 SKU/OE/provider并实现真实driver；不能用HBM/模拟或离线包验证替代实机 |

历史矩阵曾将 pi0.5 概括为无 replay；该状态已由 `local-pi05-replay-required-009` 的 10步/3调用计数证据更新。原日志保持不变；短运行不替代正式连续推理实验。

## 模型阶段

以下状态仅指本 Goal 当前运行，旧报告不计作本轮通过。

| 模型 | Source / 权重 | eager / capture | artifact / no-Python C++ | 正式 CUDA benchmark |
|---|---|---|---|---|
| SmolVLA | policy/VLM SHA256本地/H100一致；strict完整加载；恢复so100统计有完整历史权重逐bit证明 | 恢复配置16episode真实双相机state变换及完整serialized IR重载exact；新递归effect audit四真实EP通过 | 三策略完整normalized及官方反归一化输出exact，正式002补冻实际runtime DSO；旧AOTI失真和旧DSO缺口记录保留 | RTX双输出正式002三策略15进程15360测量/17280调用/34560张量exact；replay mean80.006704ms/p9980.902536ms、12.498953chunks/s；旧77.410ms仅旧normalized边界，不混用 |
| pi0 | 官方33对象验签、严格转换；H20 Torch2.10与H100 Torch2.7.1转换7GB权重SHA一致；不声称JAX FP32 parity | 新H20 v2 capture/reload完整exact；16不同帧/保存noise的官方reference及完整IR双输出全部exact；原Parquet独立核验 | 原系列001普通正式exact；新replay002三策略15进程/17280调用/34560张量独立exact，完整runtime DSO/模型payload/source/遥测与CDF通过 | H20新off/batch/required mean139.628717/139.362490/84.651419ms，required p9986.142677ms；每进程10步capture/1152replay/0ordinary；同episode16帧，无物理标定/板端或vendor对照 |
| pi0.5 | 官方29对象验签；完整转换权重已核验 | RTX v2 reference/capture、pruned完整IR和三输入checked输出processor通过 | 新16帧三策略正式17280调用/34560张量/39744000值逐byteexact，required每进程10步/1152replay/0ordinary；数值provider/DSO/源与输入冻结复核通过 | RTX三策略各5x1024正式样本；required mean350.053228ms/p99351.884728ms、2.856708chunks/s，较off降低3.315114%，显存+120MiB；同episode16帧，vendor/板端未验收 |
| RDT-1B | policy/T5/SigLIP共50,511,104,233bytes完整SHA验签；官方cu121与cu128独立环境；实际dataset分片验签 | H20在线T5+6图像+5步reference；原7 Region完整EP IR exact；新增官方机器人变换和完整双输出IR16真实history exact | 新8 TS双输出三策略17280调用/34560张量/157040640值对官方及同TS逐byteexact，完整v2/provider/DSO与replay计数通过；同一episode，非物理标定 | 新H20 off/batch/required mean176.710522/172.734449/164.397820ms，required p99168.415117ms、6.082806chunks/s；各5x1024正式测量/CDF独立通过；采样显存+608MiB，无板端或vendor成绩 |
| CogACT | 官方30.52GB checkpoint逐Tensor安装核验，实际7,630,224,071元素；Meta原config未取得，依赖candidate明确标识 | 真实Fractal episode 0 全115帧、N10 CFG/DDIM逐步/CFG/RNG全链exact；四strict EP完整链exact；新literal-only保存动态timestep/CFG/noise，16帧完整IR exact | run004三策略正式17280调用/86400完整五端口张量/6099840值对双参考byteexact，N10 capture/1152replay/0ordinary；远端及本地raw审计通过；非自主PRNG、legacy provider未绑定 | H20 off/batch/required mean85.576719/84.403288/75.538978ms，required p9978.687230ms，13.238199chunks/s；平均降低11.7295225%，显存+140MiB；配置/provider限制使paper-table eligible仍false，无vendor或板端成绩 |
| CogACT 002 numerical bound | 同一官方30.52GB public-dependency checkpoint与上述元素口径 | 四Region重新编译以实际加载图建立identity，16帧完整五输出IR exact | 新数值绑定三策略17280调用/86400完整五端口张量/6099840值对双参考byteexact；实际LibTorch provider约束与无Python进程映射通过 | H20 off/batch/required mean84.869524/86.191139/76.230755ms，required p9978.860851ms、13.118065chunks/s；相对off降10.1788822%，显存+140MiB；原Meta配置/外部RNG和板端限制仍使paper-table eligible false |

## 资源检查

当前授权为H20-x8-1、H20-x8-2、H100-x4、H100-x8。2026-09-08 07:32 UTC
重查四机所有28张GPU均0MiB/0%且无compute owner；两个同步根目录可写。
CogACT正式恢复绑定H20-2 GPU1（UUID `GPU-29fb9c6e-2a11-e303-5f26-ad079fa86fa5`），
Smol同机参考绑定H100-x4 GPU0（UUID `GPU-3251e3fd-849d-6a9d-0d06-7e4290aecec8`）。
下列09-06占用记录仅是历史，不应再用作当前GPU资源阻塞理由。

2026-09-06 04:46 UTC 左右通过 SSH/nvidia-smi 实查：

- `zzm-h100-x8`：8 x H100 80GB，driver 595.71.05；显存占用均为 0 MiB，compute-apps 为空。系统 Python 3.11.14 / torch 2.9.1+cu128 / nvcc 12.6。后续使用独立环境，不修改系统安装。
- `zzm-h20-x8-2`：8 x H20 96GB，driver 535.161.08；显存占用均为 0 MiB，compute-apps 为空。与 H20-1 共用 `/xs-train-nas/zzm/repos/`。
- `zzm-h20-x8-1`：已有 GPU 显存占用和计算负载，不安排本轮 GPU 作业，不中断既有任务。
- 本机：RTX 3060 12GB，driver 595.58.03，图形/已有负载约 415 MiB；优先承担 CPU 回归和本地代码工作。
- 空闲是瞬时快照，不是预约；每次实际启动 GPU 任务前必须重查。只绑定所需 GPU，不能默认占用全部 8 卡。

约 05:42 UTC 重查：H20-2 GPU0-6 已有 3.1-3.5 GiB 占用，GPU7 空闲，暂不安排 GPU 任务。H100 使用 GPU0，其他卡不占用；OpenPI 另用独立 CPU 环境，转换前检查资源。

约 06:40 UTC 重查：H100 全部8卡已有其他任务，不再启动新GPU任务；H20-2当时空闲，准备隔离环境。H20 SSH直传约100KB/s且高重传，Python工具链已传完；wheelhouse/输入两条自有慢传已停止并保留partial，不影响他人进程。改用本地固定manifest指定的同SHA wheel镜像下载；独有输入按分片传输重组后验原始SHA。所有env/tmp/cache放NAS，不占仅7.4GB可用的overlay。尚未启动H20模型实验。

SmolVLA 可用文件位于 `examples/smolvla/SmolVLA-Base/` 和 `examples/smolvla/SmolVLM2-500M-Video-Instruct/`。HF hub 中同名目录只有 refs，不是可用 snapshot。旧文档 `/home/zhangzimo/Archives/...` 路径在本机不存在，不继续按该路径假定资产齐备。

## 日志与产物

本轮本地归档根目录为 `artifacts/edgefm-vla-goal/20260906-044509/`。H100 已有 eager/IR、L2、L3 运行成功记录，但还不是完整部署或正式主表验收。

- `execution-context/report.json`：共享 context C++ smoke、CPU/CUDA CTest、命令/源码与 artifact hash/ldd/日志。不是整段 replay。
- `execution-context/generated-cuda-audit.json`：生成器自动绑定 shared context，session/invocation residency 两种模式及负例通过。
- bounded replay 基础测试：真实 AOTI 完整4步、两 Region，3次不同输入 replay 对普通执行 max-abs=0；valid capture abort 后可重捕获。invalidated CUDA capture 使 context poisoned，独立进程隔离测试通过，不允许静默 ordinary fallback。尚未接通 generated Session 整段 replay，也不是 VLA 实测。
- `local-smolvla/eager-ir.json`、`eager-ir-trace.json`、`eager-ir-strict-attempt1.log`：本机真实权重 eager/IR，退出 0；单图人工输入功能审计，非正式基准。
- `local-smolvla/frontend-l2.json`、`exports/*.capture.json`、`frontend-l2.log`：3 Region 捕获通过，退出 0；导出误差均为 0。
- `local-smolvla/artifacts/prepare_prefix.compile.json`：conservative profile、sm_86 原生 prefix 编译通过；编译不等于 artifact 数值或 C++ 已通过。
- `local-smolvla/smolvla-l3.json`：3 Region conservative artifact 数值审计，旧 max-abs=0.05/mean-abs=0.01 门槛下通过；不是论文无损验收。
- `local-smolvla/smolvla-l4.json`：完整 chunk 对同 artifact 最大误差 0、无 libpython、无有效 Python 环境，152 次事务序列和失败重试通过。仍为单图人工输入 queue 模式功能审计，不是 fresh-chunk CDF。
- `local-smolvla/smolvla-l3-eager-numerics.json`：独立候选保留融合间低精度舍入/除法舍入；cosine=0.9999307966，MSE=0.0001214326，max-abs=0.03285438。完整 raw reference/candidate 及分维度结果已保存；未消除编译误差，不事后改容差。
- `local-smolvla/real-observations-16/manifest.json`：固定官方数据集 `lerobot/svla_so100_pickplace` revision `728583b5eaf9e739a7f119e2def466fa1d552402`，连续帧 0-15、top/wrist 两实际相机。`dataset-source-verified.json` 将所有文件与 Hub Git/LFS 对象核验一致。注意：该包调用已发布 processor，但后来发现旧统计 namespace 可能未被新版 normalizer 消费；manifest 中原 normalization 描述不是实际数值变换已验证的证明。该包只用于兼容性诊断，不进入正式主表/物理单位保真，待明确 statistics profile 后重生成。
- `fresh-smolvla/`：第0个真实观测完成 fresh-chunk IR/Plan/4 Region export 精确对照（MSE/max-abs=0），27 项 CPU 测试通过；尚无 fresh artifact/C++ 或物理单位验收。发现 checkpoint postprocessor 可能输出恒等变换，正加统计消费硬门禁；不要把尚未修订的 `*_native` 文件名理解为物理单位证据。
- `fresh-smolvla/statistics-audit/report.json`：官方当前及迁移提交的5个小配置/统计文件与本地完全一致；published pre/post 统计消费门禁均正确拒绝。显式选择 `so100` 后300个 action 元素确实按 `x*std+mean` 变换并逐值匹配，仍只是动作尺度候选，不是机器人标定物理单位证据。迁移前固定 revision `3326b100...` 的 config 声明 STATE=MEAN_STD，safetensors header 存在 state mean/std；完整旧权重下载/摘要核验 running，尚不作为可用恢复统计。
- 上项后续已完成：完整迁移前 checkpoint 906720008bytes / SHA256 `8f8dc071d5b933e79edd2b73b8d6b5cca482ef0437c099ea3ec13ab978a38fc8` 验证通过；`fresh-smolvla/recovered-so100-policy/recovery.json` 记录500模型张量/450046176元素逐bit等于current，所有namespace两组actionstats均一致。新adapter `smolvla_migration.py` 只生成独立canonical stats处理器目录，不改发布包，不拟合dataset stats；signed-zero/weight/统计篡改/缺项/robot错配负例包含在9项测试。
- `local-smolvla/recovered-observations-spaced-16/manifest.json`：strict-statistics模式，16个episode的16帧真实双相机观测，state数值全部按恢复mean/std逐值验证通过。normalization_statistics_verified=true；physical units/机器人标定/完整预处理链仍未验收。恢复源及生成profile文件摘要均写入输入包；恢复和输入工具合计20项测试实际环境通过。
- `fresh-smolvla/published-v2-compile/`：真实双相机兼容性输入的fresh capture/AOTI/direct，MSE=0.0010596353、max-abs=0.09716463、mean-abs=0.02355985、cosine=0.999920029；技术和论文MSE门槛均failed，不改变门槛，保留fullraw并继续审计同artifact C++调度。恢复配置用不同目录，不能混用两套输入/输出成绩。
- `local-smolvla/strict-input-statistics-gate.log`：新工具默认 strict-statistics，在实际发布包上因缺 observation.state stats 正确退出1，未生成输入目录。`published-observations-spaced-16/manifest.json` 是显式兼容性模式 v2 包，16帧 stride=1200，统计/物理尺度/完整前处理验收标志均 false。工具11项测试在真实SmolVLA环境通过；CPU最小环境缺safetensors的1项依赖测试单列跳过。
- `local-smolvla/environment.lock`、`smolvla-weights.sha256`、`h100-smolvla-weights.sha256`：依赖与本地/远端权重身份。
- `source-base.bundle`、`source/`：起始源码快照；后续开发改动尚需重新冻结。CPU 中途回归 360 passed / 11 skipped（metrics 测试之后又增加，最终需重跑）。
- `cpu-regression.xml`/`.log`：本轮较新完整 CPU 回归 403 passed / 13 skipped；跳过的真实模型/CUDA项单独跟踪，不计通过。后续改动仍需重跑。
- H100 隔离根 `/mnt/data/LingXiTeam/workspace/zzm/repos/edgefm-vla-goal-20260906/`：`source-baseline/`、`assets/smolvla/`、`assets/lerobot-8fff0fde/` 已 rsync 完成。系统 Torch 不改动。
- H100 在线 Torch 2.10 安装因 DNS 失败，退出 1，日志在远端 `runs/h100-20260906-044509/pip-torch.log`。随后改用 cp311 wheelhouse；哈希校验、离线安装均通过，新独立 `envs/smolvla-cu128-py311` 的 Torch2.10+cu128/vision0.25/audio2.10/SmolVLA import 和 CUDA 可用检查通过。首次远端 checksum 在错误 cwd 下失败，切换 wheelhouse cwd 重验全部 OK；不改系统环境。
- `source-0534/`、`source-0534.sha256`、`source-0534-dirty.patch`：H100 Python 阶段新冻结源，后续新增 replay/fresh Adapter 不属于该快照。
- H100 L4 失败日志保留：首次 `_git` 依赖 cwd；attempt2 缺 OpenSSL 开发路径；attempt3 冻结源缺 `<cstdint>` 在 H100 编译器报错。当前本地已修复 cwd/实际GPU架构/错误日志输出，已有 `<cstdint>` 修复纳入 `source-0609/`；只在隔离源目录设 rsync `--chown=0:0` 处理远端 root 与本地 UID 不同，不改全局 Git 配置。指定 `OPENSSL_ROOT_DIR=/opt/conda` 的独立 CMake 配置已通过，L4 继续重跑，失败不计完成。
- H100 attempt4 实际退出0，`h100-reports/smolvla-l4.json` status=passed：完整 `[1,50,6]` 对同AOTI artifact精确一致、152次事务、cache/reset/failure-retry通过、无libpython且无有效Python环境。远端bundle `runs/h100-20260906-044509/l4-bundle-attempt4/`，runner SHA256 `9fdf5ab73019f65420d74dede0ed1303996bfdae1a3902689bdee720a4dece1a`；本地保留JSON、ldd、runner source、输入/输出binary。该queue功能审计不是152次fresh推理或CDF。H100 L3 MSE=7.160220140882103e-5、cosine=0.9999711904，论文MSE门槛仍失败。
- 本机 SmolVLA 首两次启动分别缺 imageio、pyserial；独立 venv 补齐后 strict 运行通过。失败不计通过。

## 07:00 UTC 后续证据

- `execution-context/copy-ordering-audit/README.md`：旧SmolVLA fresh C++第0次准确、第1次漂移。保持旧二进制/输入/artifact不变，只用诊断interposer补齐copy完成等待后10/10逐bit一致；最终源修复同步CopyBytes契约，非整个device同步。旧源重链接定向回归exit3，新源通过；单卡未覆盖peer-copy。旧L4成功记录只能说明当次成功，不能推断不存在该竞态。
- `fresh-smolvla/recovered-v2-compile/session-attempts/copy-ordering-fixed/session/report.json`：正式重建后10/10完整 `[1,50,6]` fresh动作与同AOTI逐bit一致，无libpython/无有效Python环境；不是10个不同真实观测。原失败 `session/report.json` 保留。
- `fresh-smolvla/recovered-v2-compile/direct/report.json`：recovered真实观测AOTI MSE=3.05882e-5、max-abs=0.01631856、cosine=0.9999948；论文1e-5仍失败。`action-scale-audit/report.json` 中官方postprocessor确实改变300/300元素且公式一致，但不代表已标定物理单位。
- `execution-context/generated-replay/gpu-20260906T070025393720Z.json`：两种范式真实generated Session普通/整段replay各次完整raw输出逐bit一致；eager MSE/cosine/max-abs、changed-input/拒绝事务/reset与fatal78独立进程证据齐备，2/2测试通过。只是通用机制小模型，不是5个VLA或性能实验。off/required当前同时改变host同步与graph提交，不能将收益单独归因graph。
- `openpi-inventory/pi0-download-verified.json`、`pi0-strict-conversion-report.json`、`h100-pi0-policy-load.json`：pi0真实权重验签、严格转换及完整官方Policy加载已闭合，3,501,372,176参数；转换model SHA256 `07e8a2ef8438e3cf839bc0992b2f2e07a95dac6049b156696bd47bbe79cb1518`。未运行action；详见独立model card。
- `numerical-probes/`：独立中间输出诊断工具正在实跑，额外输出会改变fusion，不能冒充原artifact内部值或性能。首次因export往返图文本摘要变化而拒绝，随后改用capture清单绑定两个源文件SHA；11项定向测试通过。实际诊断结果待运行结束。

## 07:35 UTC 后续证据

- `fresh-smolvla/recovered-v2-eager-numerics/`：使用与conservative完全相同的序列化捕获输入/图，在新目录独立编译，单样本MSE=9.864825893515602e-6、max-abs=0.008535862、cosine=0.999998074，通过论文数值门槛但不逐bit一致，也不构成全样本验收。
- `fresh-smolvla/recovered-spaced16/`：扩展到16个不同episode真实观测，4个region capture对例子误差0，重新编译eager-numerics并逐个完整direct。技术门槛16/16，论文门槛5/16；失败indices为1,2,4,5,6,7,8,9,10,11,13，最坏MSE=7.618073052335563e-5，均值MSE=2.1699754315922916e-5，最坏max-abs=0.0308970958，最低cosine=0.999978934。不放宽1e-5；完整raw与分样本报告保留。16样本重新捕获的文件摘要不同，不冒用旧artifact；Python源冻结在`source-smolvla-0717/`及SHA清单。
- `operator-profiles/rtx-fresh-conservative-mapping/`：NSYS 2024.6.2真实C++执行3个fresh chunk，`.nsys-rep`/SQLite/CSV与命令/runner/bundle摘要已归档。三份输出SHA逐bit等于修复后的原run0。GEMM和Attention为主要GPU热点；含cold load及profiler开销，不采作正式延时/CDF。12项通用算子输入提取测试覆盖stride/offset/alias/in-place/dropout门禁，真实节点提取待GPU交接。
- `cpu-regression-0706`记录1个新增OpenPI实际依赖测试缺transformers的失败；修正其依赖边界且在真实环境单列验证后，`cpu-regression-0718`为569 passed/37 skipped。真实CUDA/模型opt-in与最小CPU环境缺少的safetensors/CRC依赖跳过均未计通过；Smol真实环境另跑32项依赖/诊断测试通过。之后batch-only等改动有独立测试，需再次全量回归。
- OpenPI真实pi0单帧10步CPU reference与partition已完成：normalized `[1,50,32]` 1600值与官方native ALOHA尺度`[50,14]` 700值均exact、MSE/max-abs=0、cosine=1，`h100-pi0-recorded-partition-001-report.json`及actions.npz归档。不是JAX parity、GPU、编译或板端验收；机器人标定/物理单位标志false。pi0.5全29对象12,441,749,581bytes验签完成，6个composite使用CRC32C且md5_verified=false，原TLS失败partial与续传日志保留，严格转换尚待完成。
- H20-2的107个固定依赖已逐SHA匹配并离线安装；default driver小FP32/BF16 GEMM、Norm/SDPA、Inductor BF16程序真实执行通过。prefix EP经43分片重组、archive及原始EP SHA双重验证后落NAS；新`source-probe-0710`与本地文件摘要一致。此时还没有H20完整VLA成绩。

## 08:15 UTC 后续证据

- `execution-context/compiler-loop-policy/report.json`：同一原始语义IR可显式选择off/batch-only/required，原始IR摘要与编译IR/Plan/certificate分别绑定，不改变步数、body或carry。真实16样本SmolVLA三路各32次完整C++输出exact；required实际captured_steps=10/replay_count=32，batch-only不capture。不是仅改变命令行标签。
- `fresh-smolvla/recovered-spaced16/export-verification/report.json`：新进程重载序列化4regions，16/16完整输出与官方参考逐bit一致；AOTI误差不能归咎于这组输入上的IR保存重载。
- `execution-context/formal-latency-v1-final/`：三路pilot已通过144次完整输出，正式15进程正在运行；每进程128warmup+1024measured，16真实输入均衡循环并递增revision。计时边界为常驻CUDA张量到Session::run完成，排除init/H2D/D2H/核验；逐次保存所有输出、原始时延、MSE/cosine/max-abs及1Hz温度/频率/owner。当前quality_gate=failed固定，不能通过重复采样变成论文无损主表。
- `openpi-inventory/h100-pi0-replay-003-high-report.json`：同一pi0 EP/IR/真实输入，在saved-only进程恢复官方构造函数设置的float32 matmul high策略后1600值exact；默认highest时MSE=2.423884e-5/max-abs=0.03133583的失败保留。当前补通用完整数值上下文记录及新capture/replay，不能让模型名分支或默认值偷偷修复报告，也不声称C++已执行该契约。
- `openpi-inventory/pi05-strict-conversion-report.json`及`h100-pi05-recorded-input-partition-001/report.json`：pi0.5严格转换SHA256 `7ed2fb2f91b084efc387022383035f6908aa620f6bfa0da52204a4e52e7851b6`；真实10步CPU normalized/native动作逐值一致，actions SHA256 `d082da940c133b010bfffd7b4f2f7b95dc8c00715449a14c8cdbfdb9bf652f47`。仍不代表GPU、AOTI、C++、物理单位或板端通过。
- `numerical-probes/h20-firstvision-conservative/`：真实H20 prefix诊断已运行，第一Norm MSE=2.560459e-6/max-abs=0.0625，与RTX观察相近。额外输出改变fusion，仅作诊断，不能外推H20完整动作成绩。
- `operator-profiles/actual-prefix-operator-examples-attempt2/`及`h20-prefix-examples/report.json`：真实图执行时提取Attention/GEMM/Norm/Embedding实际输入与输出，保留dtype/shape/stride/storage_offset，四个独立operator EP在RTX/H20各自全部eager bitwise通过。非法alias/in-place/dropout与provenance门禁有21项CPU测试；首个immutable-list导出失败保留。
- H20 `runs/operator-microbench-h20-v1/`：12候选的初轮实际微基准启动。每个候选独立进程，完整输出bitwise门禁，实际Triton库CUDA graph batch均值原样保存，不伪称逐调用CDF；驱动逐秒查GPU0 owner，只终止本任务进程。尚无候选选中或完整模型接回收益。

## 09:05 UTC 后续证据

- `execution-context/formal-latency-v1-final/README.md`、`latency-table.csv`、`independent-final-audit.json`：15个独立进程已完成，三策略各5120次正式测量，连同warmup共17280个完整动作对同AOTI逐bit一致。off/batch-only/required均值分别62.707/62.312/59.338ms，p99为63.382/62.994/60.098ms；图replay对off约降低5.4%时延。原始逐次时延、所有输出、CDF、温度/频率/owner及源码/二进制摘要均归档。这是RTX常驻模型张量边界，不是sensor-to-action、H20或板端成绩；官方eager数值门槛仍失败，全部行禁止进入论文无损主表。
- `execution-context/raw-aoti-replay/report.json`：raw `.so` loader缺少single-threaded选项导致图捕获失败；修复后真实双Region/N4/重置/abort/fatal独立进程验证通过。同一个raw产物在旧冻结runtime上稳定复现失败；没有不安全fallback。package方式正式基准与raw修复证据分开记录。
- `rdt/gpu1-reference/report.json`：官方Torch2.1 CUDA完整参考通过，T5在线、实际6幅图、5步DPMSolver timesteps `[999,799,599,400,200]`，normalized `[1,64,128]`及官方robot `[1,64,14]`完整输出归档。在线参数分别RDT 1,228,319,872、T5 encoder 4,762,310,656、vision 428,225,600；不得以policy的1.2B代称全pipeline显存。输入保留上游OpenCV数组到PIL的实际颜色语义，未宣称物理标定。此处尚非IR/AOTI/C++或正式吞吐。
- `operator-profiles/h20-microbench-v2/`：16次候选尝试已结束。Norm的两种常规Inductor候选被bitwise数值门禁拒绝；linear的常规候选比ATen更慢；Attention基本持平。两个Embedding复用因config校验失败、linear保留ATen候选因proxy参数失败，保留原失败。每候选只有一个进程，保存的是实际graph-batch均值，不是逐调用CDF；没有选中算子，也没有完整模型接回收益。
- `fresh-smolvla/recovered-spaced16-aten-preserving*`：新的通用后端候选仍属诊断。v1/v2编译失败，v3四Region编译通过但加载因非有限常量JSON失败；v4用相同真实输入/IR重新编译，补齐ATen scalar overload、schema默认参数和非有限常量原值的native lowering。每项后端图改写均记账；不能称全部ATen或已经无损。断点续跑现在同时核验具体config及后端pass源SHA，不能仅用profile名称接受旧产物。
- OpenPI新capture记录20个实际Python数值上下文字段，同进程参考exact；完整上下文guard的saved-only冷启动仍有误差，单独high设置某些顺序下exact。旧单项high成功不能推广为完整上下文已验证，诊断仍在进行。`execution-context/numerical-transport-audit.md`明确observed/configured/runtime-checked/output-verified四层证据，通用schema实现中；C++暂不自动改全局Torch策略，也不把getter相等当作输出保真。
- `cpu-regression-0843.log/xml`：713 passed、45 skipped；其后dispatch/续跑校验专项66 passed，算子tool新增专项14 passed。全量跳过项仍为明确依赖或实际GPU/模型opt-in，未计作真实实验通过。
- 08:58 UTC H100八卡再次空闲，OpenPI仅分配GPU2，启动前重查apps；H20-2 GPU1为RDT、GPU0为根任务算子实验。H20与本地的权重传输限隔离目录，四路仅约0.45MB/s；不因网络慢占用其他任务卡或混用未验完整资产。

## 09:45 UTC 后续证据

- `openpi-inventory/h100-pi0-cuda-capture-005-image-layout/`、`h100-pi0-cuda-saved-only-006/`及pi05对应`h100-pi05-cuda-capture-002-image-layout/`、`h100-pi05-cuda-saved-only-003/`：两模型H100官方真实10步reference、partition、4 Region捕获保存重载和无OpenPI构造的独立IR重放均exact（normalized1600值与native700值；saved-only对1600值）。pi0 GPU初次partition失败由官方channels-last图像被强转连续布局导致；Adapter将布局恢复显式记录进导出图，旧失败保留。CPU数值getter的首次prefix差异另行未闭合，不用CUDA成功抹去。
- `rdt/gpu1-torch210-capture-fixed/capture/`：真实在线T5+6图像、5个denoiser输出和完整8192值对同版官方reference exact，7/7 Region实际捕获通过。RDT CPU0D系数迁到device0D的BF16乘法舍入差异已在真实tensor最小回归与完整模型验证修复；未更改官方DPMSolver方程。`rdt/aoti-solver/full-solver-reference-v3.json`：真正编译的solver五步sample及所有carry exact，尚仅1/7 compiled，不是完整无Python部署。其余6 Region串行编译继续。
- RDT官方Torch2.1→2.10配对参考MSE=9.2775e-7/cosine=0.99998244/max-abs=0.01953125，602个值不同；input/noise/timesteps一致。此项跨框架版本差异独立归档，不等于同版本exact或编译验收。
- `execution-context/numerical-schema-slice1/report.json`：通用数值Policy/CompileRecord/Requirement/RegionBinding及canonical SHA已实现，不import Torch。policy-bearing artifact/certificate/bundle分别v4/v3/v5防降级，legacy原版本保留；runtime enforcement固定unimplemented，build/codegen提前拒绝执行。322 CPU passed、6显式CUDA skip，C++ CTest10/10；正在实现optional provider及C++校验，不能把schema当真实runtime enforcement。
- `operator-profiles/proxy-enum-audit.md`：真实小AOTI包暴露Torch2.10 export enum与C++ c10 enum编码不一致；同机器码/权重仅翻译typed dtype/layout字段后bitwise通过，实际CUDA回归1 passed。审查另修正并发目标文件保护及package audit完整性校验，99 CPU passed。全Smol v5虽修复枚举仍因C shim把整数Scalar编码成double而破坏resize索引；独立小resize复现后整数算术选native lowering，v6整模重新编译。小resize不再运行失败，但非精确缩放仍有2.384e-7浮点偏差，不能描述为任意resize bitwise无损。
- `operator-profiles/h20-microbench-v3/`：6个补测全部运行且bitwise门禁通过；默认schema参数修复linear，Embedding候选复用重新数值/计时验证。`h20-embedding-repeats-v1/`：ATen与Inductor-ATen各5独立进程均bitwise通过，进程中位数均值0.0021265963/0.0018667301ms，当前实际工作负载device时延约降低12.2%，绝对约0.260us。原始graph-batch均值不是逐调用CDF；尚未注册部署skill或接回完整模型，没有E2E收益声明。
- `cpu-regression-0925.log/xml`：827 passed、48明确skipped，12 warnings；之后helper复用/整型Scalar修复有专项测试，最终需再次全量回归。失败/跳过从未合并计通过。

## 10:15 UTC 后续证据

- `fresh-smolvla/recovered-spaced16-aten-preserving-v6/direct/report.json`：修复proxy枚举/整数索引后四Region AOTI实际完整执行16样本，技术门槛16/16、论文门槛4/16、逐bit0/16；最坏MSE=5.471928163680649e-5/max-abs=0.0246362686，最低cosine=0.999986895。运行错误已消除不等于保真通过，不晋升该候选，也没有v6 C++或正式时延成绩。
- `fresh-smolvla/torchscript-native-002/`：探索现有TorchScript后端的真实序列化ATen候选，四Region单独trace/save/load均exact，但完整16组输入默认JIT仅2/16论文门槛；`torchscript-native-003-unoptimized/`关闭JIT优化也仅6/16，最坏MSE=5.040515217386413e-5。两条均0/16逐bit；局部样例通过不能替代完整链。仍是Python诊断，未扩展现有CPU-only后端、未宣称无Python/CUDA部署；trace守卫也需单列审计。
- RDT所有7个区域AOTI已实际编译完成；新进程加载全部7个EP并执行完整IR，对同Torch官方8192值exact。正在执行全部AOTI的完整动作链，不把编译成功算作输出通过。
- pi0 time/finite/action-step AOTI已完成；prefix仍用EP的明确hybrid诊断完整10步MSE=6.09156777e-7/max-abs=0.00361311436/cosine=0.99999769045，严格零容差失败。实际prefix AOTI继续编译；不是完整artifact或严格无损证据。

## 11:25 UTC 后续证据

- `execution-context/numerical-runtime-slice2/`：optional typed numerical provider ABI、runtime-owned lease、Session两阶段acquire/load及Run/drain/commit检查已实现。CPU provider正负例包括跨Region/Session共享、冲突、缺失provider、加载/绑定/执行失败和drain失败隔离；353 CPU passed、7 CUDA opt-in skipped，ASan/UBSan CTest11/11。这里的provider是明确测试实现，不是AOTI/TS/TRT数值策略已执行；这些后端继续拒绝未支持的requirement。
- `fresh-smolvla/torchscript-native-006-preserve-devices/`：同一四Region archive在不做device remap且关闭JIT优化时，真实16组完整动作全部bitwise；只保留device但默认JIT的`007`仍失败。CPU零维常量设备和JIT优化分别是必要控制，旧002/003失败不改写。
- `fresh-smolvla/torchscript-public-008/`及`torchscript-public-009-direct/`：公开通用`export_torchscript_region`实际编译这四个真实EP，effect audit、保存重载完整Region输出通过；独立完整IR的16/16动作对官方bitwise。接口拒绝动态shape、隐式输入/state写回与RNG，不包含SmolVLA名称分支。
- `fresh-smolvla/torchscript-session-002-public/`：冻结`source-torchscript-1135/vlaforge/`，通用IR/Plan/build生成无Python C++，16个真实episode观测各两次、所有revision递增，32/32完整`[1,50,6]`对同TS和官方reference逐bit一致，MSE/max-abs=0；exit0、ldd和实际process maps无libpython，所有full binary与SHA归档。这是RTX3060常驻模型张量链的真实保真通过，不是正式时延、传感器预处理、机器人标定、整段replay或零分配证明。该backend明确不支持共享context/replay/numerical provider。先前`session-001`同样32次exact，但使用审查修复前源码，不能替代002。
- `execution-context/torchscript-native/`：12/12 CTest和CUDA原生full-byte smoke通过，覆盖CPU标量保留、JIT caller恢复、Session模块隔离、重新加载清除绑定、错误设备拒绝、输出metadata失败不部分发布。公开导出/CLI专项19 passed；Python3.10哈希兼容处理另已补入，真实运行环境仍为3.13/Torch2.10。
- `rdt/full-aoti-0940/`：全部7 AOTI和3次完整C++执行已结束，C++8192值逐bit等于同AOTI。**主验收必须使用实际14活动维度**：MSE=4.668274110437259e-5、max-abs=0.025390625，论文门槛失败；128维中114个无效零值会稀释MSE。`active-output-acceptance.json`明确覆盖旧internal-space报告的主表解释，完整raw保持不变。尚无多观测/正式CDF或板端验收。
- `openpi-inventory/h100-pi0-aoti-eager-007*`及pi05全artifact audit002：四Region实际运行均完成，pi0 MSE=1.7464798115476964e-5/max-abs=0.02789402008，pi05 MSE=1.0663439368697745e-6/max-abs=0.005490541459，严格零容差均失败。pi0 ATen-preserving新prefix的36个KV输出bitwise，但time包空列表proxy错误仍在真实小Region诊断，不能写为完整候选通过。
- `openpi-inventory/h100-pi05-aoti-eager-004-native-openssl/`：实际无Python C++三次完整动作稳定，exit0，但对同AOTI Python亦非bitwise（MSE=2.4303444139178836e-7/max-abs=0.002809226513），对官方MSE=1.0333326646675647e-6/max-abs=0.004783749580。这是L4 execution而非L4验收；输入/layout/loader/数值上下文逐Region诊断继续。
- `cpu-regression-1130.log/xml`：944 passed、48明确skipped、13 warnings，51.16秒。缺依赖/真实CUDA与模型opt-in逐项保留，真实模型独立运行不以此skip冒充通过。测试之后仅有哈希兼容和局部导入格式修订，后续公共编译修改仍需重跑。

## 11:55 UTC 后续证据

- `execution-context/torchscript-native/formal-latency-v1/`：5个独立原生进程各128warmup+1024measured均完成，5760份完整动作对同TS与官方参考全部bitwise；独立raw重验通过。5120正式时延均值94.717645ms、p99 96.573343ms、std0.802624ms，模型张量边界吞吐10.557695 calls/s。实际CDF CSV/PNG/PDF、逐次raw/完整输出/1Hz遥测/源码与二进制摘要齐备。该保真通过路径比旧失真AOTI更慢，不声明提速，也不是板端或sensor-to-action主表；eligible_for_lossless_paper_table仍false。仅普通策略可用，不能伪造其它策略数据。
- 通用benchmark新增显式策略subset，原三策略协议保持默认；声明quality passed也必须逐输出实测通过，bitwise模式另对官方完整字节核验。29项专项回归通过，未支持的策略/缺失输出/伪造quality负例均拒绝。实际pilot48调用单列，不混入正式样本。
- CogACT `cogact/reference-candidate-001/remote/`真实权重候选参考已执行：完整checkpoint安装后7,630,224,071参数元素逐Tensor核对；Fractal真实观测、seed42/43、N10、CFG1.5，plain与记录hook的完整16x7输出及随机消耗一致。**仍是有标识public-dependency reference candidate**，不是已取得Meta原config或严格参考验收；max_positions差异继续做单字段完整对照。不是论文所写3B规模，需修订实际在线/常驻参数叙述。
- pi05匹配原生默认策略、single-threaded loader、stream与contiguous边界的Python多因素诊断两次全1600值等于native。尚不能归因某一个设置，也没有修复公共runtime；独立单因素对照继续。

## 12:55 UTC 后续证据

- `fresh-smolvla/torchscript-context-001-off/`、`002-batch/`、`003-required/`：同四个公开TS archive、16个真实输入各两次，新通用共享context三策略共96份完整动作均对同TS与官方bitwise。required实际记录captured_steps=10、replay_count=32、ordinary_count=0；batch-only只普通执行32次。不是正式计时。新的上下文variant与旧同步variant保持独立，LibTorch graph provider提取为AOTI/TS共享接口。
- `execution-context/torchscript-native/context-compat-pytest-v1.log/xml`：23 passed，包括真实AOTI package/raw共享context、连续/AR生成整段replay及TS原生full-byte/fatal隔离回归。`cpu-regression-context-v1.log/xml`：1034 passed、49明确skipped、16 warnings。后续pending-binding/异常边界收紧另在v2复验，不把v1证据移贴到新源码。
- RDT `rdt/torchscript-v1/`：7/7公开TS编译，3次独立完整IR及3次真实无Python C++输出全部bitwise，完整8192个BF16值及896个活动值分别核验；活动维度MSE/max-abs=0、cosine=1。目前是1个观测/seed的重复，16帧真实输入包及逐输入重新官方参考正在扩展，不声明多episode。
- pi0 `h100-pi0-aoti-aten-012-public-full-audit`：公开pre-AOT+post-grad双阶段空列表lowering修复后，13次真实Region调用及最终1600值全部bitwise等于官方和saved-EP；属于L3，不是新候选C++ L4。编译修复证据另见`doc/aoti_empty_proxy_lists_20260906.md`，保留最初失败，未将shape[]改为[1]。
- pi05 `008/009/native010`：相同loader/stream/拷贝/InferenceMode设置，仅数值policy切换的两组Python输出分别逐bit等于native004和原Python AOTI。实际native010与Python008的26次调用、940份全量边界张量全部bitwise，定位此处C++额外漂移为numeric policy；不是已经修复runtime provider，也不解决旧包对官方误差。完整trace压缩包及SHA已本地重验。
- `openpi-inventory/splitk-v1-context-001.json`：真实Torch2.10 CPU诊断证明旧20字段guard遗漏FP16/BF16 splitK状态，caller(false,false)被恢复为(false,true)，旧snapshot却错误显示相等。原进程完整pair已显式还原，v1保留为partial历史记录；新增版本化API域和完整pair恢复正在修复，不能声称旧guard覆盖全部策略。
- CogACT `cogact/maxpos-full-comparison.json`：candidate配置max_positions2048与单字段4096对照，在此实际位置0..286、两seed、完整输出/每步CFG及11次RNG消费全部bitwise，但没有证明Meta整个原config等价。`partition-candidate-001/`完整partition exact；后续003错误receipt/PAD/NaN事务实际拒绝，strict export首次失败已保存，尚未capture通过。显式RNG tape由外部生产，C++自主随机状态推进仍是未闭合项。

## 14:32 UTC 后续证据

- `execution-context/torchscript-native/formal-context-v2/independent-final-audit.json`：新冻结共享context v2三策略各5个独立进程、各128 warmup+1024 measured，15个PID和17280份完整F32动作逐字节复核通过，正式样本15360。off/batch-only/required的mean为92.831385/91.426849/77.409843ms，p99为94.255179/93.050652/78.132581ms；required对同一context普通执行mean降低16.6124%，对batch-only降低15.3314%。每个required进程记录10步capture、1152次replay、0 ordinary；无静默回退。完整raw/CDF CSV、PNG/PDF、温度频率owner、源与二进制hash齐备，图已检查。所有行仍是RTX3060常驻模型张量边界，不含初始化/H2D/D2H/传感器；`full_paper_acceptance=false`、`numerical_provider_enforcement=false`，不能继承后续数值provider的证明。
- `fresh-smolvla/torchscript-context-004-off-v2`、`005-batch-v2`、`006-required-v2`各32完整真实输出bitwise。v2对pending-work期间重绑/重载、失败加载清理、C ABI未知异常收紧；`context-compat-pytest-v2.log/xml`真实CUDA 23 passed，`context-v2-ctest.log` 12/12，旧CPU ABI与base CUDA smoke仍通过。旧v1记录保持原源码身份。
- `openpi-inventory/libtorch-numerical-cpu-evidence-verified.json`：真正LibTorch2.10共享provider读取22个字段加release/API，591项native CPU检查通过，包含所有三态FP16/BF16 pair、4边界实际setter漂移、跨Session冲突和线程TLS。实际生成的bound/legacy小TS Session各6独立进程通过，ldd/maps无Python；这是实际后端契约验证，不是VLA模型输出证明。旧Torch2.7.1 CPU构建也实测明确拒绝2.10策略并保持原状态。独立worker初始化API及Python生成助手随后实现，原生故障/恢复检查进行中；Session不调用setter。
- Python `numerical_context` v2修复已完成：新观察保留完整Torch版本/API域及22字段，完整pair恢复通过真实2.10 CPU 45项检查；`splitk-v2-context-003-final.json`保存实际反例修复。v1仍为partial历史记录，不能填补未观测splitK；不支持的release/API失败关闭。OpenPI两模型旧capture仍是v1，H100八卡被其他任务占用，v2真实GPU重捕获没有启动。H20仅4个对象6,594,393,489bytes已独立验签，不是完整权重可用。
- pi05新`h100-pi05-aoti-aten-012-public-full-audit`也已完成13次Region调用及完整1600输出，对官方/同EP逐bit一致，与pi0同属L3。新包与v2数值provider的完整无Python部署仍未闭合，不改写旧失真结果。
- `rdt/torchscript-series16-v2/`：同一个真实episode的16个分散帧、16个独立seed/noise、在线T5+六图像、全部5步新参考均完成；同一IR解释器和同一C++ Session更新revision执行16组，完整8192值与活动896值对同TS及同Torch官方全部bitwise。本地原始BF16独立重验通过。元数据非有限scheduler配置导致首次导出失败的记录保留；仅给非有限配置值显式标签，未改调度公式或输入数值门禁。
- RDT新通用typed benchmark：BF16原始字节与F32参考必须可无损roundtrip，活动/存储维度分开门禁；FP16/BF16全65536 bit patterns实际C++解码与NumPy对照通过。首次pilot因容器PID与NVML host PID不一致在Run前停止，无计时成绩。新显式CUDA注册→本进程context reset→同PID重注册握手实测通过，只在任何输入/Session/device工作之前执行，不能device-level reset或豁免未知PID。新pilot48次完整输出bitwise，正式5进程已启动；详见`doc/session_benchmark_typed_tensors_20260906.md`。
- CogACT `cogact/partition-slice-closeout-008/report.json`：已修复真实strict capture失败，四Region保存重载及两seed完整N10输出/CFG/RNG均exact。Adapter使用原生decoder层、等价timm索引tuple，明确batch1无padding位置/causal语义，不改上游全局源或导出图数值。public递归effect audit检查prefix 97个子图通过；37项CPU与真实无效事务反例通过。仍是public-dependency candidate、一个实际观测和外部11-draw tape，Meta原config/自主RNG/编译/C++/正式计时未验收。
- `numerical-probes/smolvla-recursive-effects-v1/report.json`：PID1353649独立重新加载原四真实EP，用新递归auditor逐图检查通过，包含24处dropout=0 SDPA及215处不别名外部存储的局部写操作。源和四EP摘要前后核验不变，仅是新版静态effect审计，不伪称重新编译/推理。`recursive-audit-cpu-v3.log/xml`148 CPU passed；worker renderer定向54 passed，最初异常类型测试错误保留v1 XML、修复后v2通过。

## 16:05 UTC 后续证据

- `rdt/benchmark-ordinary-v2/` 正式off-only五进程全部完成：5760个完整8192值与活动896值分别对同Torch官方和TS逐字节一致，5120个正式延迟样本mean174.162718ms/p99178.194143ms/std1.376267ms、5.741757 calls/s。CSV/CDF PNG已检查，完整raw本地独立重验通过；最慢194.190ms样本保留。仍是H20同一episode的16帧，非RTX/板端/传感器成绩，也没有RDT replay消融。v1 owner-namespace失败保留。
- `cogact/torchscript-slice-closeout-004/report.json`：四个真实Region全部公开TS保存重载，对应完整IR和同一无Python C++ Session执行42/43/42，全部五输出及每步carry对candidate参考逐字节一致，ldd/maps无Python。原始raw/normalized/native动作、RNG receipt与draw count均保存；是public-dependency candidate，一个真实观测、两seed、外部11-draw tape，非Meta原配置、自主C++随机状态或正式CDF。002/003的dtype/Plan失败及004显式81处IR dtype规范化账本保留。
- `openpi-inventory/libtorch-bootstrap-verified-summary.json`：显式worker初始化14个真实native进程用例及7种实际renderer/Session模式通过，包含完整22字段readback、medium策略、3x3 reduction pair、初始化前后lease约束、线程身份复用、部分失败完整恢复和恢复失败poison。旧thread-id复用误接受原生反例保留；新实现用存活thread token拒绝。是实际LibTorch后端契约，不冒充真实VLA绑定验收。
- `operator-profiles/rtx-rope-repeats-v1/`：原已捕获RoPE工作负载明确迁到RTX并建立本地实际参考；ATen与Inductor各五独立进程的完整两份BF16输出，对源输出/本地eager/候选均byte exact。进程中位数均值32.067544us与7.854182us，约降低75.5074%、绝对24.213361us。原始源manifest未观测GPU字段保持空，不回填；不是H100/H20成绩、单调用CDF、多shape或整模收益。
- `execution-context/allocator-pilot-v1/`：三策略各16warmup+32measured，144完整动作bitwise。稳态32调用off/batch各410912次LibTorch allocator请求，required52512；三者device alloc/free/retry/OOM均0、reserved无增长，但required多4MiB。销毁后均有2个块共9568256B active/allocated，所有权与长期增长未知。独立430文件hash及完整输出复核通过，不能称零分配、零泄漏或完整内存接管。
- `execution-context/allocator-v2-revalidation.json`：新增守恒/duplicate-key/raw-report-execution绑定门禁，107 CPU tests通过并对三策略实际原始快照重算通过。原v1 process report未改、没有新binding；此为纯离线新复核，非新GPU运行。v2/v3测试报告中的旧aggregate fixture缺冻结manifest失败保留，v4修复测试fixture后通过；生产门禁没有放宽。
- `operator-profiles/constant-precompute-vision-position-v1/`：正式e23源EP的两处固定位置Embedding由通用不可变快照预计算，保留全部原state，新增3145728B常量。original/unchanged-clone/folded三组各16完整direct和32完整C++动作全部对官方及同TS byte exact，全部33个prefix返回值亦独立核验。CPU独立raw/ZIP复核48 direct+96 native及501个原state通过。三组正式E2E已启动，尚无性能结论；这是静态预计算消融，不能冒称旧H20 Embedding kernel接回收益。
- OpenPI在H20完整33对象验签、CPU Torch2.10严格转换后，7002873776B模型SHA与H100 Torch2.7.1转换完全相同。新v2 reference/capture worker3139186与退出后的独立reload worker3140829均完整normalized1600/native700值exact；reference/capture peak reserved7.330GB，reload14.374GB，尚未解决全部EP的冗余驻留/加载峰值。H20 GPU2继续新数值绑定AOTI/C++，不借旧v1上下文证明新链。
- `vlaforge/spec/precision-calibration.md`与`precision-calibration-cpu-v1.xml`：26项离线校准契约测试通过，含真实CPU BF16存储哈希，不是完整VLA校准。所有plan的native低精度、held-out验证和无损标志均false。完整真实逐步采集、误差预算、kernel lowering及后端/性能仍未闭合。

## 17:00 UTC 后续证据

- 静态预计算正式三组各5进程完成，`operator-profiles/constant-precompute-vision-position-v1/formal-comparison.json`保留15360条正式时延与17280完整动作独立byte核验。folded比unchanged-control均值慢0.262075ms（约0.277%），进程级探索性bootstrap收益区间跨0，没有稳定收益，不选入部署。不与旧派生Embedding微基准或另一冻结runtime的replay数字混用。
- 通用多输出协议v2的CogACT H20 pilot已完成：同Session 48调用/240完整Tensor对candidate eager及TS全部exact，含F64 native、U8 receipt和I64 counter。mean90.953ms只作pilot，仍1观测/2seeds、非Meta原配置/自主PRNG/正式CDF。冻结127 CPU tests，后续aggregate CSV绑定门禁131 CPU tests；跨NumPy环境的F64 norm/cosine JSON精确重算有1ULP差异而严格拒绝，原环境重验通过，原始输出字节一致，失败记录不删除或放宽。
- `openpi-inventory/h20-unused-state-001/`：通用pass只删未使用的lifted/orphan state，不删任何计算或用户端口。四真实Region各自完整original/pruned/saved-reloaded输出byte一致，prefix37个返回值全部检查。独立本地audit PID1395292复核132冻结源文件、12个完整输出树及owner；报告SHA `38b569f56357bc0ea15a0c7ed728052e6e3e4aab64b5ac48589ce090c060488f`。源v1和当前新增signature/ledger封印的v2分开，后者21 CPU tests通过；不将旧真实结果追认为新guard执行。
- `openpi-inventory/h20-openpi-pruned-fullir-verified-summary.json`：original PID3159456与pruned PID3160530为H20 GPU2两个fresh worker，同真实输入/noise/IR/数值v2，完整13调用、1600normalized及700 CPU原生后处理值全部exact。peak allocated为14,186,335,744/6,636,933,632B，reserved为14,369,685,504/6,780,092,416B。是完整saved-only IR内存消融，不是C++/latency/12GB RTX实证；当前单真实观测，不是泛化验收。
- OpenPI native001因Adapter的数值contract schema仍为legacy而在部署前拒绝；002实际编译后runner exit127，原因为provider共享库没有随bundle复制，未进入main。公共builder现在从CMake实际shared targets清单复制并hash到lib，使用`$ORIGIN/../lib`及显式SDK路径；不会将Torch/CUDA整目录伪装随包交付。实际CPU policy/legacy两模式搬移+清env+完整输出+删坏库拒绝通过；v6新增无空RUNPATH门禁及8helper/1external build，共11passed。v5是basetemp父目录未建的11 setup errors，无native执行，原XML保留。native003仅冻结未运行，native004使用最新修复独立构建/运行中。
- 通用PrecisionCalibration/Plan经独立审查加上结构化group键、声明摘要锁、可表示FP32尺度和owned统计快照，39 CPU tests通过。新的原EP observer首片合计66 CPU tests通过，参考输出入口快照及真实state内容守恒继续补；完整IR工具准备固定16episode前8校准/后8 held-out。尚无低比特kernel或低精度无损结论。
- 当前CPU回归在`source-cpu-regression-1700/`独立源快照执行，XML为`cpu-regression-1700.xml`；启动不是通过，等待退出再更新。首次rsync因目标父目录不存在失败、pytest未启动，随后创建目录重同步；未修改原实验文件。

## 17:35 UTC 后续证据

- `execution-context/session-lifecycle-v1/`实际三策略各5次Session创建/16完整调用/销毁，共240完整动作逐字节一致；独立审计674源hash、44runtime文件、160输入/参考文件与75 allocator快照。**bounded retention验收失败**：每个周期销毁后allocated/active/requested为9568256、19136512、28704768、38273024、47841280B，最后10个活动分配。off/batch reserved每周期增加1925185536B，required增加1929379840B。尚未归因，不能称已证明全局cache或零泄漏；见`doc/session_lifecycle_20260907.md`。原失败保留，下一步针对stream-key BLAS workspace候选做地址证据与最小复现。
- OpenPI `native004`实际native PID3164543退出0，三次完整normalized `[1,50,32]`对官方及同AOTI byte exact，完整v2数值provider在worker显式初始化后保持；maps无libpython，provider来自bundle/lib，runner/provider RUNPATH无临时目录或空分量。native报告SHA `3f310fb9e477ad0d18e6ad8056fa101ae6b5442593f8722fe176c6175a58496c`，独立汇总SHA `2cc0a18d6a03099cd84289cdbf63197d288b7eae281ac61d9accc385023011d2`。不是native700值已在C++输出；后处理初次CUDA标量FP64除法严格零门槛失败20/700值、max-abs 1.11e-16，独立单因素定位后改为同FP64 Tensor分母，CUDA003的eager/capture/reload/AOTI完整700值exact。新双输出IR/bundle继续，不改原报告。
- pi05转换后7233650408B资产已从H100分片只读传至本地，SHA `7ed2fb2f91b084efc387022383035f6908aa620f6bfa0da52204a4e52e7851b6`，独立CPU核验812个BF16张量/3616757520元素和norm资产；实际resume与错误SHA拒绝测试保留。这只是新v2捕获的资产准备，不是模型执行。
- `cpu-regression-1700.xml`实际12failed/1469passed/55skipped，包含2旧monitor mock、5旧protocol fixture及5隔离快照缺根文档。只修测试输入与冻结目录，不放宽生产门禁。`cpu-regression-interface-fixes-1705.xml`实际工作树47passed（含Git-index架构检查）；随后完整冻结CPU切片1483passed/55skipped/1deselected。源封印driver仍报告失败，因为wheel测试改写egg-info/SOURCES.txt并生成build/lib；独立`cpu-regression-1705-generated-output-audit.json`验证无原实现变化且每个新增副本与原源一致，状态`passed_with_recorded_packaging_outputs`。保留原失败，不把生成元数据解释成未运行或实现遭改；后续新增源码仍需新回归。
- `precision-probe-v1/attempt-003/independent-audit.json`确认16个不同episode完整原轨迹、208 Region调用、1136实际观察、160份完整FP32 `[1,50,720]` head输入；前8的568观察为唯一拟合源，后8不fit。16完整动作与官方bytes exact，实际step_index/timestep、148冻结源码和153输入/源文件重验通过。attempt002的singleton stride字节视图失败保留；修复owned CPU规范拷贝后重跑，没有修改原图。当前Plan严格JSON恢复/INT8模块/规范拷贝专项117 CPU tests通过，与旧冻结observer版本分开。
- `int8-kernel-v1/attempt-001/`实际RTX PID1415668退出0：四策略各10个模块、每模块16真实head输入，共640份完整局部输出；INT8xINT8的int32累加对独立CPU int64逐值exact，严格export/save/reload全输出exact。40份实际profiler trace中`aten::_int_mm`归属`ampere_igemm_int8_128x128_ldg4_nn` CUDA kernel。权重原step EP/官方checkpoint逐byte同一性的独立报告在`precision-probe-v1/head-weight-identity/`。此时仅核验原轨迹保存输入上的局部head：held-out最坏MSE global/site=0.000103346719、step=0.000100484576、step-group=0.000101856516，最低cosine约0.99996265。不把高cosine当无损，不把head误差当完整动作误差；未选部署、无free-running/性能/板端结论。

## 18:10 UTC 后续证据

- `execution-context/retention-diagnosis-v1/diagnosis-report.json`将原残留每个活动地址绑定到LibTorch cuBLAS/cuBLASLt两个stream-key workspace：8519680+1048576B。无TS/模型的getter最小例也累增，no-BLAS为0、同context复用恒定。公共runtime只改`execution_context.cpp`，引入跨device区分、同时存活不共享的stream lease池，仅drain成功且未poison归还，进程存活pool holder避免全局析构顺序UB。v3源码SHA `95591c0edb4c1d82dca279d1be6693f882d7cbffb96bce646584a50a78da16ba`；三策略5生命周期共240完整动作及75快照独立通过，活动内存固定9568256B。off/batch reserved固定1925185536B，required仍每生命周期多4194304B；只读pool-id证明确为各自完全inactive/allocated=0的旧graph私有池，不是已解决全内存增长。179 CPU pass/1opt-in skip，独立真实CUDA并发lease小回归1pass；graph作用域回收仍单独验证，不clear全局allocator。
- 新`cpu-regression-1800-report.json`源快照含上述v3 runtime，967主要文件前后SHA一致。实际1644passed/56skipped/1deselected，64.30秒；真实工作树Git-index架构检查另1passed/2collection skip。生成build/cache/egg-info排除范围预先记录。准备时误附不存在的顶层reports导致rsync exit23，但实际CMakeLists/vlaforge/doc已复制，随后源冻结和完整测试通过；原命令失败与修正命令在`cpu-regression-1800-README.md`，不删除记录。
- `int8-kernel-independent-v1/attempt-002/report.json`独立审核1920完整raw、640整数累加器/反量化结果、40个EP和40个CUDA kernel归属，补上执行driver/controller前后SHA、PID/NSpid、argv、exit0与19个owner快照。报告SHA `78d9e85deed0ef7a554a4248b661fb2492cbb2008a3d290e901a40c038fab0c7`。全部FP32 reference的原始SHA直接等于原完整轨迹head输出，不只是一次新的F.linear计算。001缺预启动driver身份的局限保留。
- `deployment/linear_precision.py`只替换显式生产者到指定`aten.linear`，真实weight/bias由原signature绑定，设备int64[1] step选已拟合尺度；原浮点节点、scheduler与外部ABI不重写，不复用旧证书。133专项CPU测试通过，v1/v2 FakeTensor复制/domain失败保留；当前保存重载及非法step guard验证通过。`int8-free-running-v1/attempt-001/`实际四策略各16轨迹完成，原16观测/原noise/前8拟合，后8不fit。独立复核64完整[1,50,6]动作、4544原始Tensor、原全部state与未选择数学节点；每条step0 head输入对原轨迹exact、step1已分歧，上一量化sample输出与下一步输入SHA一致，证明不是teacher forcing。
- 上述量化完整动作的held-out纸面MSE门槛四组均0/8通过：最坏MSE global/site=4.287605236885381e-5、step=5.0809271346018674e-5、step-group=5.060397781240626e-5；最低cosine分别0.99999391376、0.99999323111、0.99999298871。执行/证据审核通过不代表质量通过，保持lossless=false、same_precision=false、no_python_deployment=false、performance=false。四新step EP的公开TS/完整C++作为独立precision-changing部署准备，不改官方门槛或原参考。
- OpenPI checked finish005在H20 GPU7完成完整700 F64+bool的eager/capture/reload/AOTI；独立CPU全字节及EP/AOTI绑定复核通过。其后新完整双输出session001及先前GPU2 finish004均遭外部owner插入，监控只终止自身，未完成新完整C++。CPU实际资产/IR/Plan/5数值契约准备通过，172 CPU pass/1explicit real skip。H100八卡80GB与H20八卡外部任务占用时不抢占。下一新run控制器`runs/openpi-pi0-cpu-inputs-001/run_openpi_output_bundle_002.py`已隔离就绪，当前未启动。

## 18:25 UTC 后续证据

- `board-handoff/smolvla-real16-data-v1/manifest.json` SHA `30192dd7524e7cff23da134654b59d003b12184103ec8e548601309889d14cf0`：237个文件、147090789B，16不同真实episode，每样本8输入及完整F32 `[1,50,6]` native-scale动作，总4800动作值。原NPZ/输入/参考SHA独立核验后，把包复制到不依赖原绝对路径的目录再次验证。原协议保持旧数值provider未强制的身份，不追认后续G5策略；恢复so100尺度不代表机器人物理标定。公共实现47新+65已有typed测试通过；Orin/BPU各四阶段实际exit2/pending、无driver子进程。此项只完成数据和契约准备，没有板端driver或实机结果；见`doc/reports/board_handoff_preparation_20260907.md`。
- OpenPI双输出session002曾在H20 GPU7空闲后启动，controller PID3203314；约90秒后出现foreign owner，监控只终止自身worker，exit-15，未进入完整模型compute。这是第二次完整链被资源变化中止，不记direct/native通过。pi05资产已在本地完整核验，H20尚无完整文件；隔离上传PID1437803使用四路rsync、总限速16MiB/s，保留partial与逐片/最终SHA验收，不能把传输启动当作安装完成。
- `doc/vla_model_metric_boundaries_20260907.md`及`model-count-scopes-v1/`从已核验实际模型报告生成范围表：RDT在线pipeline共6418856128元素（含T5/SigLIP），当前CogACT安装candidate为7630224071元素。Smol模型state计数和OpenPI完整实例计数另列，不能把名义action-policy规模、完整pipeline与在线活跃/驻留权重混为一个指标。
- 新默认关闭的graph-scoped回收策略和同步失败sticky poison正在独立编译/回归，不继承旧v3实际输出证据。INT8新七个TS archive与四个quantized-lane新contract/bundle使用稳定冻结源独立验证；目前尚无完整低精度C++通过记录。

## 18:55 UTC 后续证据

- `execution-context/session-lifecycle-scoped-reclaim-v4/report.json` SHA `432aea28ba4a64d1592f0a24b13cf6834de712903bba00feb97a6ecd9b0481ab`；独立audit SHA `a12861514032773a6ae4079cce0199e9feedf1316caeccec979a55122357f0a8`。off/batch/required各5次Session生命周期，共240完整动作byteexact、75 allocator快照和921冻结源文件核验通过。三者销毁后active恒定9568256B、reserved恒定1925185536B；required每次2次真实devicefree，其他两策略0。新上下文sticky同步失败poison与checked graph exec/rawgraph/pool销毁分别测试；故障路径明确quarantine而不是假装释放成功，void析构ABI不向caller返回错误。27完整小图输出及真实故障包装检查通过，不冒充实际驱动自然故障。
- 公共`build_artifact_compile_bundle(..., libtorch_graph_memory_policy="scoped-reclaim")`已接入，实际CPU-only CUDA SDK小包构建确认CMakeCache/target flags为ON，metadata/build_configuration.json纳入bundle哈希，20专项测试通过。版本未知或非CUDA LibTorch合同拒绝；默认`retain`保留兼容，不能把scoped结果追认所有默认部署。销毁时间另同源retain/scoped对照在准备，旧formal延迟不继承。
- `int8-native-v1/attempt-002/independent-audit.json` SHA `dfa1fdba9f2f91cc5d0c895bd56f0525f63f522dc4c90c510b1489ab06ddecd6`：761源文件、7TS/47完整Region case、16新contract/4certificate与4实际原生PID全部复核；64完整F32 `[1,50,6]`逐byte等于四组量化Python free-running候选，实际maps无libpython、provider绑定完整v2。官方held-out质量仍每组0/8，不改门槛。native-report SHA `43790da860af2fec12953087b741f32d602c227c42366d3ff1cccbda3a9ec89c`。原target错误/旧未用controller目录缺陷保留；尚无新native kernel profiler或正式性能。
- `operator-profiles/solver-slice-v4/`实际RTX worker1449510 exit0，通用`extract_tensor_slice`复制原Smol step三个ATen节点，不运行或改写原整模。16episode的160次原noise/velocity边界与重建carry逐步匹配原trace SHA，保存800完整Tensor，小EP保存重载一致，数值v2/RNG未变；独立审计继续。旧per-call stride未观测，实验显式恢复capture的singleton index stride1600；不伪称原carry raw一直存在。前三次status/metadata/raw-view失败保留。最新公共slice/operator/probe定向102 CPU tests通过；solver微基准与整模部署收益尚未完成。
- pi05本机CPU真实prepare-only PID1442664 exit0：严格完整权重和官方ALOHA输入、10完整Tensor、22字段v2；58.17秒、maxRSS15616484KiB，未运行GPU动作。外层controller schema名字错误导致postcheck失败保留，独立validator使用实际公共SCHEMA复核报告通过（SHA `95159d20e2938ea6ba8de08e24dace14860f38bf4b467bcd73dda5de3ec37d4e`）。新本机CUDA任务采用单owner reference/persist、每EP独立prune重载、最后全pruned IR，避免把H20原始14GB多EP驻留直接搬到12GB RTX；这是待验证方案，不借pi0内存值证明pi05已完成。
- `source-cpu-regression-1855/`已冻结当前公共源码，`cpu-regression-1855-started.json`记录实际测试子进程，结果等待退出后更新。新回归显式隐藏CUDA设备，不与π05争用；不是GPU验收。

## 19:35 UTC 后续证据

- `openpi-inventory/local-pi05-v2-pruned-001/`：RTX worker1451461控制的新v2真实reference/capture及四EP original/pruned/reload、fresh完整pruned IR全部通过，1600 normalized/700 canonical native值byteexact；独立audit SHA `0c1a10574529db5e569e367745263b616f29974f147fbb049bd0bec7421e81ea`。随后`local-pi05-v2-pruned-aoti-001/audit/report.json` SHA `300ddb2431c67ef01aafd8a6e2df647c0100d98ae843721aadc55fcca7e3499a`：原EP trace worker1462389先退出，fresh pure-AOTI worker1471921逐一核验13调用所有实际输入/输出exact。NVML峰值7022MiB；Torch allocator只报133848576B，因AOTI常量外部管理，不能拿该小值当整模GPU峰值。仍一观测，C++尚未验收。
- H20 pi0双输出session003在GPU7空闲时启动，controller3228463/worker3228465/NVML1024118；约143秒仍在hash证明核验时foreign1031837进入，监控只终止自有worker，未完成direct/C++。证据`openpi-inventory/h20-openpi-native-output-session-003/`，不连续抢占重试。
- `execution-context/session-destruction-comparison-v1.json` SHA `84bb7265241c73a994b009096882cd1011056c26dc07220dc06febf6b873f723`：同冻结runtime/16输入、唯一CMake retain/scoped变量，三策略各5周期，合计480完整动作与150 allocator快照独立exact。required销毁+drain中位数37.609881/38.139672ms，范围36.481693-44.836868/36.288846-46.074646ms，仅5点且范围重叠，不能将差值归因2次devicefree或声称显著。retain旧私有池增长复现，scoped恒定。
- solver独立CPU audit SHA `a77642d2c23cadac0b87c51eea98372ebb85b705e0d9fe57e0f11e991f635dd2`确认800raw、160步骤和原完整动作绑定。`operator-profiles/solver-candidates-v1/summary/report.json` SHA `d9b0e6bab7b603862ab1bfdd1a00098eed278915a2270ce753c43b6b9ff29b0f`：三候选各160步骤的两输出byteexact，单进程ATen/Inductor/preserving均值3.068011/1.726649/2.942838us。两份独立all-case validator最初缺显式codecache import，在加载AOTI前失败；`revalidate-002/`仅修validator、绑定原timing/artifact后重新通过，原失败保留。不是五进程/CDF/E2E收益。
- `int8-trajectory-errors-v2/report.json` SHA `ea870795336b34b9c5a2137ef6ed18368419a73004fefcd7f592b08e720fdac6`：纯CPU对3211源文件绑定，生成2560项逐步carry/velocity指标、64最终动作重算和active6/full32曲线；所有最终metric逐字段等于原free-running报告。step首步held-out平均MSE 1.113001e-6低于global/site 1.325157e-6，最终却为2.068598e-5高于1.948666e-5；四组最终仍0/8过门槛，没有从held-out再fit。原v1分析脚本错误比较两种sample schema，失败记录保留；v2按实际compact/full identity分别验证。
- **尺度勘误**：实际Smol finish EP仅`slice[:,:,0:6] + contiguous + finite`，正式TS、allocator、solver/INT8及原板包完整F32[1,50,6]均是归一化输出。18:25文字及板包v1的native-scale标签错误，不撤销raw字节正确性，也不追认为已反归一化。原文件/报告保持不变，单列勘误和新语义包；真正反归一化Region及完整C++输出仍是G2/G3待补项，不能用早期单样本postprocessor公式检查代替。
- `cpu-regression-1855-report.json`已结束：996主要文件前后SHA一致，1774passed/58skipped/1deselected；Git-index另1pass。之后公开benchmark显式worker bootstrap与OpenPI join新增，另冻`int8-native-pilot-v1/public-cpu-regression-v3/`得1826passed/58skipped/0failed。之前v2在Smol环境有13个RDT版本硬拒及5个快照外层依赖缺失，失败保留；v3使用既有validation环境和完整快照，不把跳过的RDT/CUDA/真实模型项算通过。

## 2026-09-07 03:55 UTC 恢复执行

- RDT `retrace-v2-002/` 的八个完整新TS archive已实际H20编译、验证所有solver/denoiser索引及state/RNG不变；12857393533B trace目录通过唯一一条原有rsync拉回，session95157退出0。没有重复同步或改变鉴权/SSH配置。报告SHA `8b23b8b4d6599016e90336b04553afcc0b14b002fedcf06e83faa5b93e7fccca`。
- RDT `dual-output-003/independent-native-audit-001.json` SHA `59d29384b6ef629fbe3f7755ab4d780c4101f42fe47ac5e8853f49de182a3c72`：H20-2 GPU6、实际C++ PID3368196/NVML3655299，16预热+16诊断共32调用；每次统一BF16[1,64,128]和官方机器人BF16[1,64,14]全部逐byte等于原同Torch官方及同TS。64完整张量/290816值、112原输入、8个12.86GB总trace archive本地逐SHA、IR/Plan/certificate、完整v2/provider随包/实际maps无Python均独立核验。该数字不是16独立episode或机器人物理标定。
- 新`adapters/output_stage.py`通过显式来源映射把source output、原acceptance及可选输入送入纯Tensor后处理，原predicate与后处理finite共同决定原子双输出commit。RDT、Smol和后续OpenPI扩展复用接口，公共runtime/IR不按模型名分支。31项回归通过；初次回归临时目录父层缺失3项setup errors原XML保留，修正运行目录后无改代码重跑通过。
- RDT前两次build分别拒绝漏写numerical schema及ordinary off路径误声明shared-context变体，没有native运行。第三次native本身exit0，旧收尾器却找`samples.csv`而实际monitor保存为`stdout.log`；独立audit直接核验原stdout及raw通过，旧failure/旧build-only报告不改，未来driver修正文件名。不是通过修改原报告制造成功。
- `operator-profiles/terminal-tail-native-v1/independent-native-audit-001.json` SHA `b6177b644732be34136c947abf3adddbd3b92744ef9759612ca20a278896a491`：三native PID3361365/3362647/3363532，48完整normalized动作与同H20原EP及对应TS全部逐byte相同，完整provider和无Python maps通过。H20对旧RTX official 0/16 byteexact、4/16门槛，不能当作H20官方baseline；仍无选中Inductor tail/E2E收益。独立auditor首次误把内嵌certificate当FileRecord的失败保留，按实际结构修后审计通过。
- `execution-context/aoti-extraction-v1/resume-cpu-001/report.json` SHA `a4b93a36edfdd3f4dedfb08e67cdd974d0b13a52e0210d62027d89a655a082db`：通用显式私有目录解包当前源码80 CPU通过/1显式skip，真实CPU proxy+嵌入/blob权重、C++ Region/sequence及Python loader均执行。root需canonical/owned/0700，流式解包和双SHA校验，不清foreign/tmp，不改已安装Torch/环境。此项不等于pi0 H20已解除/tmp失败；CUDA完整模型重新验收仍待执行。
- Smol `smol-native-output-cuda-001/stage/report.json` SHA `4d9eed5ea9acb562d89493b6d3c2afbdaf4cf7221cffd31b8bf8478a01e2ab82`：实际RTX PID1550683，16真实normalizedrefs经官方反归一化、显式旧acceptance、capture/reload/TS全部300值exact；还不是完整新Session。
- RDT `dual-output-formal-001/` 在本节起始时已完成pilot并开始正式采样；最终结果见下面04:20 UTC记录。旧174.163ms对应旧unified-only/旧runtime边界，不转贴新包。
- 当前全CPU快照首轮1950 passed/74 skipped/2 failed：测试temp在repo内，两个AutoVLA no-Git fixture向上发现主repo而拒绝；不是模型执行失败。`resume-cpu-regression-002/`把temp移到repo外、未改模型/身份门禁，最终1952 passed/74 skipped，1057快照文件前后SHA一致，其中514文件逐SHA匹配当前本地实现和测试；原失败保留。缺依赖/原生opt-in的74项明确未验收，不能作为新增模型通过。

## 2026-09-07 04:20 UTC 新RDT正式结果

- `rdt/dual-output-formal-001/independent-audit-001.json` SHA `58c104c196481fa450036161cf7bbb205cc34103ab68afd975b12d9e0295f543`：5进程各128warmup+1024measured，5120正式时延、5760完整调用、11520张量/52346880值对官方及同TS逐byteexact。445项冻结文件和571个实际文件重新SHA核验；前一阶段32次诊断不混入正式样本，正式每进程为1152调用。
- Mean175.006495743ms、p50175.205363ms、p95176.837480ms、p99179.051004ms、std1.307760971ms；模型张量边界吞吐5.714073616chunks/s，全稳态wall边界5.706979191chunks/s。所有outlier保留，最大188.307264ms，无硬deadline保证。
- 实际native PID3373222/3375002/3376734/3378566/3380186，NVML对应3762876/3787619/3812266/3837184/3861482；无外部GPU owner冲突、完整v2/provider/无Python maps。48调用pilot单列，不合并进正式统计。此阶段结束后GPU6已释放。
- `retrieval/files/prepared/latency-table.csv`、`off-cdf.csv`、`latency-cdf.png`及绘图源码/来源hash齐全，图已实际检查。独立NumPy重算5120样本mean/std/nearest-rank quantile和CDF秩次一致；本次只回传221002811B新证据，未重复传输12.86GB模型archive。
- 本边界包括在线T5、6图像SigLIP、N5 DPMSolver、官方关节重排/夹爪缩放、统一输出和机器人输出原子commit；不含相机/预处理、H2D/D2H、初始化、核验、日志和机器人传输。不是Orin/BPU成绩、物理标定，也不把与旧快照的差值归因给后处理。

优先下一步：Smol真正反归一化完整Session；H20用新私有AOTI加载器重新验收pi0双输出；后续OpenPI保留旧acceptance的新接口需新冻结输出证据。G3更多观测和匹配官方/vendor baseline、G4选中kernel真实E2E/完整Agent/skill消融、G5质量过门槛策略、D1/D2三模型输入包及真实driver仍缺。G6保持P1。当前硬件阶段与全论文验收均未完成。

## 后续更新：Smol整链与两个新缺口

- `smol-native-output-session-001/independent-audit-001.json` SHA `4026cd9186b27e2a218723be633f39da18974ee5bda257a5858e9bb7c0cd0ee7`：16个真实episode，完整IR以及off/batch-only/required三路C++96调用、192个完整归一化/官方动作尺度输出、57600值逐byteexact。新`codegen/session_runner.py`从模型接口生成typed C++数据读入/绑定/全部输出核验，OpenPI旧入口改为复用，默认生成C++文本与旧实现逐byte一致。定向55项通过。
- `smol-native-output-formal-001/`三个策略各5进程、每进程128预热+1024测量已实跑，公共工具报完整34560张量保真通过并生成CDF。mean为94.293003977/92.621583626/80.068477882ms，p99为95.383038/93.851600/80.747883ms。注意：下面的runtime DSO身份缺口尚未闭合，当前只保留实测数据，不能写成最终冻结验收完成。
- 新独立审计发现基准runner实际加载`prepared/build/<policy>/vlaforge_runtime/libvlaforge_libtorch_numerical_backend.so`，不是原bundle内的副本，两者SHA也不同；旧公共`prepare`只冻结了主binary和源码，没有冻结重建DSO。RDT新正式记录同样有这个缺口，原审计仅检查库名。已补公共benchmark的构建DSO/随包DSO启动前hash清单、实际maps到完整路径/内容校验及汇总复验，49项定向测试通过；新门禁下的CUDA正式重验尚未启动。旧日志/统计/报告不改，不追认为原来已防止库替换。
- `openpi-inventory/h20-pi0-private-loader-001/`用当前本地冻结源在H20 GPU7运行，实际PID3391903/NVML4124307。新的3输入输出Region包含原acceptance，700个F64输出的eager、保存重载、AOTI以及false/完整normalized非finite拒绝均通过；但ExitStack关闭私有解包时NFS出现ENOTEMPTY，capture阶段退出1，后续完整部署未启动。
- 独立cleanup诊断PID3395351使用同SHA `2e2039cfa22297fb215fd8ab6407c993ee834a074d58c7f1af16fb591ffcaa54`包重现：关闭runner后库仍在maps，NAS产生映射中的`.nfs`文件；删除输出、GC、CUDA同步及两次延时重试仍EBUSY。ELF存在GNU UNIQUE符号。不是NAS容量不足或可以靠重试解决的瞬时错误；不清外部/tmp，也不把关闭失败降级为成功。证据在该目录`retrieval/files/cleanup-diagnostic-001/`。下一步应实现有逐文件签名的构建期物化AOTI产物/加载契约，避免运行时依赖DSO卸载删除临时目录；公共原raw.so入口可复用，但proxy/cubin/blob必须一起绑定，不能只拿裸.so绕过校验。
- 全量当前CPU快照`resume-cpu-regression-003/`为1969 passed/74 skipped，源码前后不变；随后补runtime库门禁另49项定向通过，整套新门禁尚待实际CUDA重验。Orin/BPU实机仍延期，P1 Qwen3.5仍未启动。

当前优先顺序：完成runtime库冻结后的Smol/RDT重验；构建期物化AOTI并闭合pi0完整新Session；pi05扩大真实观测/拒绝接口验收；CogACT原配置/更多观测；官方/vendor baseline、Agent调优E2E收益、低精度质量及板端包。总体Goal保持active，绝非全部完成。

## 2026-09-07 08:30 UTC 网络恢复后继续

- 恢复时实际检查本地/远端进程：此前两条 exec handle 已失效，本地无相关 pytest/benchmark，H20-2 无相关实验进程、8 卡 compute owner 为空。没有修改代理、SSH 或凭据配置，也没有重启仍存活的实验。
- `aoti-materialized-cpu-001/tests.xml` 已在中断期间完成 61 passed；随后默认 C++ manifest 入口不再依赖私有解包 root，`aoti-materialized-cpu-003/tests.xml` 新 61 passed。002 因测试父目录未预建产生 31 passed/30 setup errors，原 XML 保留，未算部署失败或测试通过。
- `aoti-materialized-bundle-cpu-002/tests.xml` 4 passed：实际 CPU AOTI/TS、policy/legacy 四组合，构建目录清理及 bundle 搬移后完整输出一致，实际 maps 无 Python，原物化目录删除后仍运行，损坏 payload/provider 拒绝。001 为测试错误字段访问导致 1 failed/1 passed，后按真实 NumericalCompileRecord 字段更正，不放宽生产门禁。
- 通用实现新增 `ArtifactKind.AOTI_MATERIALIZED`、规范 manifest/逐文件 SHA 与 size、原 compile record 到物化身份的 lineage、公共 bundle 自动携带 payload、Python 与原生 raw runner 加载。文件归部署资产持有，Session close 不删 DSO，不伪称已解决旧 NFS 卸载。OpenPI adapter 只选择该通用后端路径，内存/IR/Session 无模型名分支。
- RDT `dual-output-formal-002` 远端前三次 00/01/02 报告均 passed，第四/第五目录恢复前不存在。新 `resume_formal_002.py` 逐项核验旧 raw/数值 worker/实际 DSO/冻结文件并记录旧目录所有字节，仅执行 03/04。实际 controller PID3433005，03 worker3433372 已完成，04 worker3435084 继续；不是重跑前三次。最终聚合/独立审计尚待结束。
- pi0 `h20-pi0-materialized-loader-001` 使用新隔离源码，H20 GPU7 controller3435263、capture3435279 实际运行。先重验带 incoming acceptance 的完整输出 processor，再接完整 IR/C++。本地审查发现旧冻结 build 的清单比较漏 MD5 字段，已在公开代码修复及补测试，不热改远端冻结源；后续只从可证明通过的阶段续接，不重算已有参考/权重。
- 本段属于实质进展；G0-G6/D1-D2 总 Goal 保持 active。当前可用硬件阶段与全论文均未完成，Orin/BPU 延期范围未变。

### 本轮完成项

- RDT 正式002已结束并拉回本地，`rdt/dual-output-formal-002/independent-audit-001.json` SHA `46ce66c4dd154e03ffe7afefb6b4fd4eca90043b0a092a9b9d7437b736c3a6e5`：449冻结条目/849实际文件复核，实际 maps 的 DSO 完整路径及字节等于启动前清单；五进程5120正式样本、5760完整调用、11520张量/52346880值对官方及同TS逐byteexact。独立逐文件证明前三重复未改且未重跑，后两重复新执行；mean175.169159ms、p99182.002621ms、max195.236780ms、std1.477469ms、5.708768chunks/s。完整CDF及尾部图已生成并实际查看；同episode16history、model-tensor边界、无replay/板端/物理标定，旧001身份缺口仍保留。
- `aoti-materialized-cpu-004/tests.xml` 84 passed：增加默认入口原生 manifest/path/device/内容/缺文件/symlink 错误拒绝，以及 MD5 记录回归。`resume-cpu-regression-004/report.json` SHA `9ba6fd80b53123ff2611407ae7e8bdb4b4bf8e0fec69cce9574ac45b979bf35f`：2003 passed/83 skipped，冻结源前后不变；跳过项不计新增模型或原生验收，实际 CPU AOTI/搬移执行见单独 opt-in 结果。
- pi0 物化001 capture PID3435279、direct PID3438522均exit0：新3输入processor各700值与官方exact并保留false/NaN拒绝，完整IR两输出已通过。processor报告SHA `768ce72d6258980a565ccc3410f02f5c866226393deeff0e98ff81bd55f35e5f`；direct SHA `0c730334251899d60a79cffcea82cf6939e5ac8b96ec4aa2e3701dc7c06508d1`。旧冻结build在已定位的记录字段比较处拒绝，未改原报告。
- pi0 物化002用当前本地新冻结源，逐hash绑定并复用001的capture/direct，没有重算模型或修改旧artifact。实际build/native/verify均已结束通过；新的C++执行完整normalized F32[1,50,32]和official native F64[1,50,14]三次调用，完整数值v2显式bootstrap后由provider验收。独立 ZIP成员/实际maps/原始输出字节审计及回传正在进行，正式CDF/多观测/同模型replay仍未做。

### 08:44 UTC 独立归档完成

- pi0远端独立audit PID3444163通过，已回传到 `openpi-inventory/h20-pi0-materialized-loader-002/retrieval/files/independent-audit-001.json`，SHA `5fb30c751c1be444ad9c166a1fb2484c8d0523f2f3ebf3d72cfe16c1deb53e47`。实际native PID3442053/NVML1041813，6完整张量/6900值对官方和同artifact逐byteexact；5物化库实际maps、39成员与原ZIP逐一SHA/size、策略及原compile记录lineage通过。该audit没有新增GPU执行，也不等于多观测、replay或正式CDF。
- 本地 `local-independent-audit-001.json` SHA `9d2ad2f944acf380ba71f03ecae421d0dcd61608c8690b2605374f27237ffa48`：两次retrieval每个文件及冻结源复核，另拉回官方actions/prepared_inputs/capture三文件并绑定原SHA；全部输入bin等于官方预处理数组，重新比较3次完整normalized/native输出，逐维MSE/max-abs为0并保留cosine。大权重及payload仍在NAS，不声称本地已镜像完整可运行bundle。
- 本轮所有controller、CPU回归、独立audit和rsync均终止完成；最后H20-2及本地compute-apps检查为空。空闲只是快照，下轮启动前仍需检查，不预占资源。
- 下一步入口已准备：本地 `run_smol_output_formal.py` 改为每次冻结当前本地runtime，保存新source-files并在各阶段/结束重验；旧001的controller-source不改。仅 `--help` 已检查，正式002尚未启动。命令：

```sh
env CUDA_VISIBLE_DEVICES= /home/zhangzimo/.venvs/edgefm-vla-smolvla-py313-20260906/bin/python /home/zhangzimo/Repos/private/edge-fm-x/artifacts/edgefm-vla-goal/20260906-044509/run_smol_output_formal.py --label 002
```

后续仍为Smol正式DSO重验、OpenPI多观测/新拒绝语义及正式性能、CogACT原配置、匹配官方/vendor baseline、Agent完整调优与E2E收益、低精度质量、五模型板端包及G6。没有缩小Goal，也未将当前硬件阶段或全论文标为完成。

## 2026-09-07 09:33 UTC 状态核对

- Smol `smol-native-output-formal-002/independent-audit-001.json` SHA `6ff7e36c77ed416a629f85bf0cf3bfc9505319f8aa09c4f43afc42ea0e136e9d` 已通过：15进程、15360正式测量、17280完整调用、34560张量/10368000值；16真实episode，normalized和官方反归一化输出对官方及同TS逐byteexact。511冻结文件/1243实际文件、实际DSO、完整CDF和replay计数均独立重验。off/batch-only/required mean94.420255/92.565081/80.006704ms，p9995.708937/94.121297/80.902536ms；required对同源off mean降低15.2653%，不是Agent kernel收益。图已检查，旧001身份缺口不改写。
- pi0 `openpi-inventory/openpi-series-001/` 使用同一真实ALOHA episode的16不同帧，不是16episode。原Parquet解码、各相机/state/task与已保存noise逐数组独立核验，audit SHA `1e50296c737c66df944336357e42a28c950b8b60b8b6b99d7f8a5459bfefcf34`。H20 reference PID3451394及direct PID3452162均通过，完整双输出exact；控制器PID3451369继续原有正式任务，当前stdout列出重复0至3完成。没有重复启动或修改远端冻结源；最终统计和独立审计待补。
- pi0.5 `openpi-inventory/local-pi05-native-output-cuda-002/processor/report.json` SHA `a869c7909a0405ea4d6a1f795c2119c7f96efc1eabb0d0a40dbe55ee950666d7`：新三输入processor保留incoming acceptance，eager/reload/AOTI拒绝false及完整source非finite。`local-pi05-native-output-session-003/` controller1794466、native1795347、verify1795468均结束；三次完整双输出6900值逐byteexact，数值provider通过，冻结源前后不变。verify SHA `54a766af9a506d435db2936a51ab326cf653538dc655122f9f896b4ba90ab067`，controller SHA `45abbe133069487206ee2112631b04bdb81a47653b6e48051d7c7ffe367b2286`。仍一观测、非正式计时，独立证据审计尚待补。
- `resume-cpu-regression-005/report.json` SHA `cd1cfd0e6fa0450a32c701bcd577201d51d7a171c688bba30c9c6d5707081118`：2014 passed/83 skipped，源前后不变。此快照早于后续OpenPI显式bootstrap及series archive支持，不能替代这些最新改动的全量回归；对应定向回归与真实003结果单列。

## 2026-09-07 14:02 UTC pi0.5 required 运行核验

- `openpi-inventory/local-pi05-replay-required-002/` 已由当前三输入 processor-002、typed native join、`required` policy 完成 prepare/build。bundle 在本地 RTX3060 上通过三阶段 owner handshake 和显式 numerical worker bootstrap；无 Python C++ runner 以 1 warmup + 2 measured 调用 exit 0，3 次 normalized F32/native F64 完整输出均对 direct/eager 参考逐字节一致，MSE/max-abs=0，独立审计 `native4/independent-replay-audit.json` SHA `c0d6262e77fd9a0bc1931d0c5ac64ecaf38b75785c4097a0bc3e2fc7a07f3d4f`。
- 该旧 runner 的实现契约是 typed output/finite/exact 验证，不打印 `vlaforge_model_session_get_replay_info`；因此旧 `replay_analysis.json` 只能证明 compile-time candidate。该旧项已被 14:48 UTC 的 `local-pi05-replay-required-009` telemetry 复验 supersede，不把旧 exact 输出单独计作 replay。
- 通用 runner 已补 `replay_policy` telemetry hook：从 IR/Plan 自动枚举 bounded-loop task，在 C++ 中校验并打印 `REPLAY` state/capture/replay/ordinary 计数；OpenPI builder 已接入，47 个定向测试通过。上述旧 bundle 生成早于该 hook，需重新 build 才能闭合 runtime counter，模型适配器不增加模型名分支。
- telemetry warmup 边界修正后，`local-pi05-replay-required-009` 新 bundle 在本地真实运行通过：task 13 记录 `captured_steps=10`、`replay_count=1/2/3`、`ordinary_count=0`，exit 0；双输出 warmup + measured 均 exact。独立审计 SHA `3e4b34e11fdf5eecb8705a22e47c8d470f0ca14c72e93bdad4970bb49f9f901b`。这是单观测短运行，不等于所有模型 replay 或正式 replay CDF 已完成。

### 剩余工作与估时

以下为2026-09-07的工程估算，不是实验保证；按连续可执行、GPU可用、传输无长时间中断计算，不简单累加并行项，也不承诺未通过的科学结论必然成立。

| 范围 | 尚未完成 | 估计有效工作时间 / 依赖 |
|---|---|---|
| G2/G3 CUDA模型与基准 | pi0正式归档；pi05多观测/正式CDF、runtime replay counter；CogACT原配置、更多观测与完整语义；匹配官方/vendor对照及统一测量边界 | 约2-4天；CogACT原配置授权/资产等待另计 |
| G1/G0 通用实现收尾 | 最新完整回归、跨生成范式真实覆盖、后端内存边界和长稳/异常矩阵、论文主张修订 | 随上项推进；仍不能宣称全栈零分配或全arena接管 |
| G4 Agent与算子 | 六类热点完整同平台候选/选择/skill流程、预算和复用消融；选中kernel接回整模并复测 | 约2-4天；孤立算子加速不能代替整模收益 |
| G5 低精度 | 现有四种INT8策略均未通过held-out门槛；进一步策略、完整动作质量及收益验收 | 约2-5天探索；可能仍得到负结果，需要据实收敛论文主张 |
| G6 可选Qwen | 0.8B/2B真实vision+prefill+decode、数值与性能 | VLA优先项之后约1-3天，取决于资产与后端支持 |
| D1/D2 交付准备 | 五模型完整输出数据包、目标provider/driver与真实运行入口 | 部分已备，SDK未知部分未完成；不能把数据包叫可执行部署包 |
| D1 Orin实机 | SDK/驱动确认、目标构建、五模型实测/CDF/功耗热稳态 | 硬件可用且后端可构建后约2-4天；显存或不支持的格子需如实记录 |
| D2 BPU实机 | SKU/OE/provider与fallback实现，再做五模型实测 | 目前不能可靠给出总工期；若需新provider，应按周级估算，不是简单启动脚本 |

CUDA P0整体建议预留约1-2周；低精度或CogACT依赖未解决会更久。全论文验收目前不能报确定日期，必须另满足Orin/BPU条件。优先继续当前pi0正式归档、pi05多观测和匹配参考；不得为估时删减模型、降低门槛或把pending标passed。

09:38 UTC 增量核对：pi0系列001 stdout已有重复0至4全部完成，原SSH执行返回exit0，远端controller3451369已退出，controller.json为passed。没有重启任务。完整结果仍待回传及独立审计，因此当前最终独立验收的正式模型仍计Smol/RDT两项，不把自动汇总通过当成已完成额外审计。

## 2026-09-08 当前进度与估时

本节是当前核对，优先于上文各历史时点的进行态。当前硬件阶段与全论文均未完成。

- 已有真实权重无 Python C++ 完整动作输出记录的模型为五个；其中 CogACT 保留
  public-dependency candidate 身份，原 Meta 配置和自主随机状态推进未闭合。
- 已有公共正式协议普通执行数据和完整双输出审计的是 Smol、RDT、pi0、pi0.5。
  Smol 另完成三策略正式 replay；pi0.5 replay 只有单观测短运行，不能继承 off 正式样本。
- pi0 新实验 `openpi-inventory/h20-pi0-replay-002` 完成三策略 pilot：16真实帧，
  每策略16预热+32测量，共144调用/288张量对官方及同artifact逐byteexact。
  独立audit SHA `ce544e13e722d1997cd50cbc8f1309944aa81798a40cf92b18734e9d2e22db1c`。
  新正式控制器H20-2 PID3511813正在运行；每策略5独立进程、每进程128预热+1024测量。
  日志：`/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/pi0-replay-20260908-002/formal.log`。
- 最新完整CPU回归011为2049 passed、71 skipped、0 failed，核心源码前后未变。
  真实模型/硬件opt-in跳过项不计作通过。

以下为粗略有效工作时间，假设算力持续可用、依赖获取正常、没有重大后端重写；
不是通过保证，也不按已投入时间线性推算完成率。采样时间与研究性改动分开估计：

| 范围 | 当前剩余 | 粗估 |
| --- | --- | --- |
| 当前pi0实验 | 三策略正式进程、全量输出审计、CDF/报告归档 | 00:32北京时间核对时约40-60分钟，不保证共享机器始终空闲 |
| 同精度CUDA收尾 | pi05正式replay；RDT/CogACT replay；CogACT统一稳态CDF；同源官方/vendor对照与边界统一 | 约2-4个工作日，配置/授权等待另计 |
| Agent G4 | 完整选择/skill复用与预算消融，选中kernel整模重测 | 约2-4个工作日；没有正收益则如实保留负结果 |
| 低精度 G5 | 修正held-out质量失败，最终双输出与性能/内存验收 | 约2-5个工作日探索，质量或收益仍可能失败 |
| CUDA P0整体 | 含G0/G1回归、生命周期与论文主张对齐及上述G2-G5 | 建议预留1-2周量级，非各项简单相加 |
| Qwen G6 | H20固定profile已完成；任意输入、流式逐token和板端外推未验收 | 当前profile无剩余；扩展边界需另立范围 |
| Orin D1 | SDK确认、目标构建、5模型实测/CDF/热稳态 | 硬件和后端可用后约2-4天；仍可能有OOM/unsupported单元格 |
| BPU D2 | SKU/OE/provider/fallback实现和实机矩阵 | 当前无法可靠给出总工期；需要新provider时按周级而非小时估计 |

板端目前只有部分可移机数据/reference包与通用接入/调度契约，并非五模型完整可执行
部署包。当前CUDA成绩多为resident model-tensor边界，不能填入Orin/BPU端到端主表；
CogACT历史测量另含H2D/D2H和首轮开销，不能不注明差异就合并。

00:46北京时间数据交付增量：`board-handoff/pi0-dual-data-002/pack`已完成，322文件/
64,678,337B，manifest SHA `5f455275e12602ac36685b6749b51cf3eb2c4d55eb9a270edb3df7f3bd2911c4`。
16真实帧的160个输入逐byte等于官方prepared NPZ，保存noise另对原dataset审计核验；
32个完整normalized/native输出对均exact。移目录复验通过，8次未知target dispatch全部
exit2/pending，无driver启动。001缺本地input index的失败保留，补原件验SHA后另建002；
不是新推理或板端可执行包。详见 `doc/reports/pi0_board_data_handoff_20260908.md`。

pi0.5数据包另完成 `board-handoff/pi05-dual-data-001/pack`：328文件/64,715,108B，
manifest SHA `5257c4c07c0b758eb19c5cbeef4c2aeeb813a211c47ae0e9fbb9813061af6d63`。
来源为pi05正式003的独立参考，不复用pi0输出；160输入逐byte等于该模型官方prepared
NPZ，保存noise对原审计一致，32完整双输出对exact，移目录复验通过。仅追加已有转换
checkpoint身份到data-only protocol，原协议字段不变；不是pi05正式replay或板端执行。

## 2026-09-08 pi0 三策略正式闭合

恢复核对时旧exec句柄已失效；远端原controller及runner已退出，15份`execution.json`
均exit0，`formal.log`含全部PROCESS_COMPLETE，因此没有重跑。实际GPU实验结束于
2026-09-07 17:14:30 UTC，共47.83分钟；06:43北京时间启动离线report，随后完成归档。

- `openpi-inventory/h20-pi0-replay-002/independent-formal-audit.json` SHA
  `7405e50ad5348d0fa4e088f388c2a8300b25ed29eba4f4d9ccd5d1221cbb90e4`：三策略15进程、
  15360正式样本、17280完整调用、34560张量/39744000值全部对官方及同artifact逐byte
  exact；MSE/max-abs0，最低cosine0.9999999999999998。每required进程10步capture、
  1152replay、0ordinary，预热亦核验；应用预热不计入时延统计，未删慢样本。
- 同源off/batch-only/required mean为139.628717/139.362490/84.651419ms；
  p99为151.099301/158.724181/86.142677ms，required吞吐11.813151fresh chunks/s。
  相对off平均时延降低39.373919%，属于执行层replay收益，不是Agent选中kernel收益。
  普通进程均值135.644317-148.464923ms有差异，全部保留；共享流单独无稳定明显收益。
- `post-execution-assets-audit.json` SHA
  `5d25c9e62c322fc696a7fad402b05edd0ccb5069e4bc03282635648c0f03e2ad`：573冻结条目、
  290实现源文件、179bundle文件和每进程五个实际模型库maps复核；本地完整输出审计
  另验真实runtime DSO字节、bootstrap与raw遥测。无foreign compute owner；采样SM
  1980MHz，NVML peak used off6893/required7561MiB，不能称省显存或allocator峰值。
- `prepared/latency-table.csv`、三份CDF CSV及`figures/latency-cdf-tail.png/.pdf`
  已生成、独立重算并查看。大型模型payload留在NAS，不声称本地已镜像完整部署包。
- 独立审计负例9项、公开handoff/variant/runner定向72项通过；核心代码与全量
  CPU regression011一致，2049 passed/71 skipped不扩大解释。
- G4选择链已补入公共 `vlaforge/deployment/operator_selection.py`，要求候选微基准、
  完整输出正确性、实际完整模型E2E和显式置信门禁同时通过。真实H20 Embedding尝试
  的微基准点估计为+13.920929%，完整静态预计算E2E仅+0.214856%，五进程bootstrap
  `[-0.3720797,0.6559971]ms`跨零；`operator-selection-v1/decision.json`因此明确
  rejected/selected_for_deployment=false。选择API及operator测试47 passed，未把
  孤立算子加速或不稳定点估计写成收益。

下一步仍按完整Goal推进：pi05同源三策略正式实验；RDT/CogACT replay及CogACT统一
稳态协议；官方/vendor等价对照；G4候选装回完整模型的收益与skill复用；G5修正held-out
失败；剩余完整板端数据包/实际provider与G6。CUDA P0与整篇论文均未完成。

已完成项离线重验命令（需要Python NumPy和当前源码，输出使用新路径）：

```sh
REPO=/home/zhangzimo/Repos/private/edge-fm-x
RUN=$REPO/artifacts/edgefm-vla-goal/20260906-044509/openpi-inventory/h20-pi0-replay-002
PYTHONPATH="$REPO/vlaforge/python" python "$RUN/audit_formal.py" \
  --output /absolute/new/pi0-formal-revalidation.json
```

上述audit会要求当前核心源码仍等于本实验快照；后续源码变化时这个默认命令会拒绝，
需基于保留的冻结源另设可追溯的重验入口，不能关闭身份门禁或改写旧审计。不得在原
prepared目录重新启动`--stage run`覆盖证据。
