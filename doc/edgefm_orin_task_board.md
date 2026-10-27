# EdgeFM Orin 实验任务看板

最后核验：2026/10/27 10:35:27，Asia/Shanghai。此为已核验快照，不是实时台架遥测。

**总状态：已完成必需验收。当前位置：已选实验验收与边界不变。运行时a61917d与交付6214dbd已推送origin/codex/vlaforge-paper-artifact；独立Git检出3060 Python/16 Node/11浏览器及272文件包核验通过，CPU CTest14通过。旧主目录完整stash备份后已快进；旧实验根退役，45小型记录与13原图提供版本化入口。。**

网页版：<http://127.0.0.1:4317>。若端口已占用，以启动器实际返回的地址为准。

本文件与网页共用 [结构化状态](edgefm_orin_task_board_state.json)，由 `node tools/orin-task-board/sync.mjs` 生成。
后续每次有进展，更新同一状态文件；运行中的本地服务会同步生成本文件并刷新网页。

## 一眼看进度

| 交付项 | 已验证进度 | 说明 |
|---|---|---|
| 完整链路真机 pilot | 5/5 模型 | 不等于五模型正式交付 |
| 正式 E2E / CDF | 5/5 模型 | 不是按工作量计算的整体完成百分比 |
| 算子 / 集成消融 | 范围内已完成 | 保留负收益、数值合同和计时边界；不等于全模型加速 |
| 资源 / 功耗 | 部分完成 | 已有热稳态及匹配功耗证据；物理峰值归因和其他功耗模式仍有边界 |

## 当前所在位置

已正式验收：SmolVLA、RDT-1B、pi0、pi0.5、CogACT -> **已选实验验收与边界不变。运行时a61917d与交付6214dbd已推送origin/codex/vlaforge-paper-artifact；独立Git检出3060 Python/16 Node/11浏览器及272文件包核验通过，CPU CTest14通过。旧主目录完整stash备份后已快进；旧实验根退役，45小型记录与13原图提供版本化入口。** -> 公共消融及最终交付 -> 可选 Qwen3.5。

已选范围完成：五个必需VLA+Qwen3.5-0.8B；2B用户取消，不计测试或通过。数据交付完成不代表全部论文主张通过，私有GPU峰值/第二功耗模式/机器人任务质量仍有明确边界。

同一 Orin、同精度 / 输入 / noise：标准化完整 action 主门槛 MSE < 1e-5、cosine > 0.9999；native 误差和 H20 差异单列。Runtime 对照固定 engine，图消融显式记录 engine 差异。

## 模型主线

| 模型 | 完整部署 | 动作核验 | 正式采样 / CDF | 下一步 |
|---|---|---|---|---|
| SmolVLA | 已验证，7 engines / 10 步 | 正式数据已审计 | 已完成 | 正式、Norm负收益回集成、热稳态及匹配功耗已审计；转最终资源范围/交付验收 |
| RDT-1B | 已验证，11 engines / 5 步 | 正式数据已审计 | 已完成 | 正式/CDF/15进程资源与恢复已审计；保留shared负收益和私有归因边界，转最终交付 |
| pi0 | 已验证，8 engines / 10 步 | 正式数据已审计 | 已完成 | D1-D3及实际scalar/cast绑定已核对；仅arena无replay、约0.4%且两轮负收益，抖动未达目标。H20 normalized0/16失败保留；转最终资源范围/交付验收。 |
| pi0.5 | 已验证，8 engines / 10 步 | 正式数据已审计 | 已完成 | D1-D3及实际scalar/cast绑定已核对，原两Java已恢复；Arena无replay、约0.146%且两轮负收益、std未达目标。H20 normalized0/16失败保留；转最终验收。 |
| CogACT | 已验证，12 engines / 10 步 | 正式数据已审计 | 已完成 | D1-D3及五模型CSV完整验收，两项业务已恢复；Arena无replay、约0.087%且一轮负收益，std未达目标。转最终资源范围/交付验收，不扩大为源模型/机器人质量无损。 |
| Qwen3.5 0.8B（可选） | 已验证，5 engines / 15 步 | 正式数据已审计 | 已完成 | 已验收；Arena均值仅改善约0.41%，两项最大时延更高，无whole-loop replay或官方源等价主张。 |
| Qwen3.5 2B（可选） | 用户取消，不测试 | 用户取消，不测试 | 用户取消 | 按2026-10-25用户最新指令取消2B构建、模型运行和采样；仅保留历史主机准备，不计实测或通过，不作为Goal完成条件 |

CogACT 保留实际约 7.63B 公开依赖配置身份，不改称已测名义 3B。未开始不等于不支持。

## 当前采样快照

| 项目 | 核验事实 |
|---|---|
| 采集状态 | 正式数据、CDF、资源及服务恢复已审计 |
| 台架 | Jetson AGX Orin，序列号 `1421524312594` |
| 软件 / 功耗 | TensorRT 8.5.2.2 · CUDA 11.4 · L4T 35.4.1；既有 MAXN，未标为 60W |
| 协议 | 2profiles各5独立进程x128warm+1024measured，AB/BA顺序GPU；在线native文本与RGB，17神经+16控制，16新tokens，16GiB准入/4GiB保护，已授权Java按需临停，原Supervisor恢复另行核验/无whole-loop replay |
| 完成进程 | 10/10 |
| 当前进程 | 第 5 轮 verified；1,152 次调用，128 次预热，1,024/1,024 条测量 |
| 总 measured | 10,240/10,240 |
| 已记录质量 | 全部184320tokens精确；320完整logit/ID见证byteexact，MSE0/cos1。跨设备9/16tokens不同单列。 |
| 估时 | 已完成，无模型计算任务运行 |
| 控制器 | PID `230214` / start ticks `43409003` |
| worker | PID `601943` / start ticks `45105672` |
| Boot ID | `3c801f5d-6897-4d97-b021-3c52db38a750` |
| 数据目录 | `/data/datalog/edgefm-orin/20261025-21/native-resident-formal-20261025-21-006/native-output` |

板端 civil clock 不正确，使用 monotonic time。SSH 观察超时不代表任务失败；恢复时核对实际 PID/start ticks/argv 与日志，不自动重启。

## 任务队列

| ID | 任务 | 状态 | 已有证据 / 剩余动作 | 依赖 |
|---|---|---|---|---|
| prefix-split-deployment | 通用四段前缀部署与完整计算保留 | 已完成 | pi0完整source-preserving repeated-call正式十进程/10240 measured及23040完整动作tensor、资源/服务恢复/冻结文件联合通过，覆盖3/5。旧NaN/准入失败及H20 normalized0/16均保留；pi0.5独立绑定与模型验收另列。<br>待完成：pi0.5源组合不外推为模型通过，转由pi05任务独立部署/核验 | native-numeric-region, openpi-native-input |
| openpi-session-binding | 真实 artifact 的通用 Session 绑定 | 已完成 | 通用绑定组件已覆盖pi0真实11region及pi0.5独立10region；两者完整目标链及D1-D3均已独立正式验收。pi05全23040tensor/230文件/资源及原管理器恢复联合通过，正式模型4/5；组件通过仍不外推CogACT。 | native-numeric-region |
| native-e2e-driver | 原生输入 API 与 E2E 驱动集成 | 已完成 | 五个必需VLA均已完成实际Orin原生输入/独立Session资源绑定、完整动作及D1-D3联合验收；CogACT实际ARM64十二TRT/四native完整十步和57600输出通过。该组件完成不代表公共D4或可选Qwen完成。 | openpi-native-input, openpi-session-binding |
| openpi-native-input | OpenPI 原生输入准备 | 已完成 | 通用输入API已在Orin原生编译；320/320张量对新鲜native CLI与冻结host命名配置均字节一致，Prepare零文件IO。CPU组件完成；完整神经链与E2E集成归模型/驱动任务。 | 无 |
| native-numeric-region | 原生 F64 后处理 Region | 已完成 | 5个实际schema provider原生编译/GPU核验通过，128冻结后处理块89600 F64值全字节一致；这是非神经组件验收，完整模型集成仍归pi0/pi0.5主任务。 | 无 |
| scalar-sincos-target | F64 SinCos 插件真机验证 | 已完成 | 旧smoke不改为模型通过。新独立追溯两正式OpenPI：step engine SHA/实际build源图SHA/成功导入日志/闭合资源一致；每图512个F64频率字节SHA等于原十步CPU源shadow。后续完整模型质量/资源/服务联合验收已通过，非仅host准备。保留无完整OpenPI replay/任意新shape/硬件故障认证限制，不扩大单组件两流smoke。 | rdt-audit |
| checked-cast-region | 通用受检 dtype 转换 Region | 已完成 | 两正式OpenPI的3个受检cast DSO与实际原生CUDA11.4build注册/完整模型资源闭合SHA吻合，workspace384/448/1056及F64->F32/I64->I32原接口核验通过。要求同步、无external capture/context支持；原F64状态保留，原3345 CPU断言和128chunk/97772 GPU功能断言证据边界不改。正式全模型后续通过不等于任意shape/CUDA提交硬件原子性或故障分支全认证。 | 无 |
| rdt-partial-evidence | RDT 中断证据归档与输出核验 | 已完成 | 范围内归档与输出检查完成：25894 个保留动作张量逐字节一致；正式协议仍未完成，不产出正式主表或 CDF。 | 无 |
| rdt-formal | RDT 三模式正式采样 | 已完成 | 15/15完整128+1024、三模式各5120 measured，34560张量字节一致；全数据/尾部/CDF、采样floor及原服务真实新进程恢复合同审计通过，旧中断数据未合并。 | rdt-raw-pilot |
| operator-ablation | 同平台算子调优消融 | 已完成 | 六族成对算子实验和源signature/数值/计时边界、单候选预算、12选定engine实际build成本已闭合，123200测量完整保留；选定build wall合116.830536192s，非全部历史搜索或Agent思考时长。新两族actual ABI/159360值exact、40进程/40960测量和521hash/71出生终态通过，CDF已实看；保留Embedding负轮、RoPE resident更差max、state配置限定及旧104/110精度边界。Norm完整回集成慢6.43%按负结果完成，GEMM/Attention/newNorm失败不采用，不追加正收益门槛，不宣称自主kernel合成或整Goal完成。 | 无 |
| rdt-audit | RDT 全量审计、CDF 与归档 | 已完成 | 15/15完整128+1024、三模式各5120 measured，34560张量字节一致；全数据/尾部/CDF、采样floor及原服务真实新进程恢复合同审计通过，旧中断数据未合并。 | rdt-formal |
| openpi-host-preparation | pi0/pi0.5 主机导出与后处理准备 | 已完成 | 两OpenPI主机源导出/后处理/原生输入和受检类型组件已进入真实完整Orin链，pi0/pi0.5独立D1-D3均联合验收；源BF16/H20误差保留，不由组件状态外推其他模型或源精度等价。 | 无 |
| cogact-host-preparation | CogACT 主机并行导出准备 | 已完成 | 主机准备交付：四个真实阶段及额外portable CPU前缀，主任务独立复验822文件/46.26GB。270 action组件对照通过；完整计算前缀join的最终动作通过，但cognition MSE超阈值，整体failed保留。此任务完成仅指主机准备，不是CogACT目标通过。 | 无 |
| external-sequence-preparation | 静态Sequence改写的外部权重支持 | 已完成 | 外部权重结构lowering完成：51项及真实5.636GB前缀/37 split/110 access检查通过，源图/控制快照竞态已修复。主任务185项组合回归通过；运行时索引consumer拒绝保留，须先lower完整源图再分区，不是TRT或整模型通过。 | 无 |
| pi0 | pi0 完整 TRT 部署与正式实验 | 已完成 | formal-002十进程/10240 measured及23040完整tensor、11026遥测/52419守护采样、两Java原管理器新进程恢复/216冻结文件联合通过。均值1129.870→1125.379ms，仅1.00399x，无replay；两轮负收益和H20补充失败保留。必需D1-D3覆盖3/5，整个Goal未完成。 | rdt-audit |
| pi05 | pi0.5 独立配置与完整实验 | 已完成 | 独立8TRT+2native正式D1-D3联合验收：十进程/10240测量/23040完整tensor全部byte-exact，主Gate0/1、native补充0；12057遥测/58325守护、原管理器恢复及230文件哈希通过。1258.793->1256.956ms仅0.146%差异，两轮负收益/p50变差/std未达1.2ms保留，无replay。正式覆盖4/5，非整体完成。 | pi0 |
| cogact | CogACT 容量、完整链与正式实验 | 已完成 | 本批10进程/11520调用/10240measured、57600完整输出独立exact，normalized MSE0/cos1、native补充0、U8 RNG/I64计数exact；7330遥测/33156guard、58187目标文件hash/36出生终态及原管理器恢复联合通过。两项业务临停8465.231494208秒后恢复且其他受管程序不变。694.393388→693.790506ms，仅约0.087%、一轮负收益，std2.970001ms未达1.2ms；不宣称显著加速，旧03/109失败不合并。 | pi05 |
| operator-integration | 有效算子候选回集成 | 已完成 | 代表模型回集成实验已完成，性能负收益而非未完成：Norm全部10240 measured/23040输出重审原audit与CSV hash完全一致，标准化主门槛通过、native补充误差保留，E2E125.900306->133.994095ms且五对均回退。正收益不是Goal额外完成条件；不采用候选、不改主表、不接受GEMM/Attention失败或新Norm不足计数pilot。九检查通过，仅闭合此子项而非全部D4。 | operator-ablation |
| memory | 内存观测与限制归档 | 已完成 | 五模型60正式进程/78667遥测、35分钟热稳态、覆盖CUDART 217/217及1024稳态零受检事件已按原Goal完成有界观测，并发布进程VmHWM/共享内存/私有归因unknown的最终表、逐项验收和claim-gap。48检查、资源CSV新旧字节一致通过。完成的是观测与明确限制，不是私有SDK/driver物理峰值或全局零分配证明；unknown仍为null，不以获取root无限延长。 | rdt-audit |
| power | 匹配功耗基线与另一合法模式 | 已完成 | 代表模型匹配功耗15进程/34560动作exact、2423原始/1854合格功耗读数已验收：普通/共享/replay38.438594/38.552769/42.267198W，replay更高。五模型主模式完成后实际只读nvpmodel确认MAXN/0及15W/30W/50W配置名；UID1000无status/CPU/GPU频率写权限，CLI非setuid/无filecap，EMC读取EACCES13保留。未sudo、未尝试切换、无第二模式性能，按Goal D4.5具体限制交付，不宣称平台无其他模式或MAXN=60W。 | cogact |
| qwen08 | Qwen3.5 0.8B 原生 VLM（可选） | 已完成 | 五VLA与Qwen0.8B完整实测及31项验收完成；2B用户取消。原始样本/CDF/质量/资源/服务恢复已核验；H20差异、RSS缺口、负收益及论文强主张边界保留。 | cogact |
| qwen2 | Qwen3.5 2B 原生 VLM（可选） | 用户取消 | 用户明确取消2B；不再执行构建、模型运行或正式采样，不作为Goal完成条件。保留既有主机准备与源logits差异，未做Orin模型实测，不标unsupported或通过。剩余仅0.8B按原质量门槛和正式协议收尾，之后结束整个Goal。 | 无 |
| delivery | 五模型最终论文证据包 | 已完成 | 五VLA与Qwen0.8B完整实测及31项验收完成；2B用户取消。原始样本/CDF/质量/资源/服务恢复已核验；H20差异、RSS缺口、负收益及论文强主张边界保留。 | cogact, operator-integration, memory, power |
| environment | Orin 原生 TRT 基础链路 | 已完成 | 真实台架的原生运行路径已验证，可复用。 | 无 |
| smol-formal | SmolVLA 正式 E2E 与连续 CDF | 已完成 | 三模式正式协议完成，完整动作逐字节一致。 | environment |
| rdt-raw-pilot | RDT 原生输入与完整动作 pilot | 已完成 | 已达到正式采样的同 Orin 门槛。 | environment |
| smol-runtime | 代表模型 runtime 与热稳态 | 已完成 | 完成限定范围内的功能与稳态验证，未宣称全部资源归因。 | smol-formal |

## 验收与风险

- **原机实验目录已按新授权退役，论文使用自包含包**：前阶段105来源/52数值行/109链接的检查是删除前事实。本阶段用户明确授权删除本机旧实验、Orin/J6M/H20产物及感知case，共23路径；不再声称原机来源全部保留。NAS196冷模型逐项SHA删除，原始测量格式不选入；Orin追加计划0。论文包272文件/282相对链接清理后复查通过，数字和未测边界不变；f801源码仍在，BEV未提交改动/新增源码及独立Git历史已保存。历史验收不代表当前可复跑，J6M完整E2E不改标通过。
- **本轮已释放磁盘，后续大文件任务仍需准入检查**：本阶段按授权退役本机369.914902GiB唯一inode已分配块，清理后复查可用582.90GiB；23路径无跳过。NAS另退役797.968254GiB，不承诺物理配额精确回收。f801约132MiB源码/文档/活动索引和看板保留，Codex会话/附件/登录、主Git和SDK环境不删；两感知索引已删需重建。旧阶段129.40/42.63GiB不重复计入本次或与后台磁盘变化混算。
- **历史单体前缀内存保护失败保留**：旧003单体完整解析后优化阶段触发2GiB采样保护，非算子不支持/内核OOM；该次失败不回改。后续source-preserving repeated-call链完成独立完整动作/资源验收，五模型D1-D3现5/5。实际配置与旧单体不同，不把旧构建失败改为通过或降低资源门槛。
- **历史Session workspace初始化失败与修复保留**：首次workspace contract mismatch/exit86及后续pilot002 acceptance失败均保留。修复绑定已有Plan workspace_buffer而非额外StaticArena，size/alignment/device严格校验不变。后续独立完整模型数据另行验收；不以组件修复追溯接受旧失败pilot，也不以历史失败描述当前正式进度。
- **旧whole-part01数值失败保留，不混同后续通过配置**：原engine d9ed520...同18输入独立执行后12/51输出NaN，与原Session51输出含NaN bits逐字节一致，不能归责arena/路由。后续source-preserving repeated-call配置有独立完整动作及正式验收，不证明旧whole-part01问题根因已定位；原失败/负证据不改，不作全FP32/量化回退或跳过acceptance。
- **旧 RDT 压力窗口保留，新服务窗口不得混合**：旧低内存/swap耗尽中断数据保留且不合并。后续新服务窗口十五进程完整数据及原管理器恢复已独立审计，16GiB准入/2GiB保护不松。CogACT另批24/4GiB和两原生业务临停/恢复独立验收，不承诺全平台隔离或未来资源不变。
- **GPU 进程检查并非全覆盖**：历史owner快照和后续普通用户device-FD检查均存在权限缺口，不能证明全设备独占。不同采样窗口仅临停各自获批程序；最新CogACT两原生业务已原管理器恢复且其他受管程序不变，不据此宣称已清空全平台。
- **抖动目标未达成**：正式pooled分布未达草稿<1.2ms目标；SmolVLA replay std也高于常规模式，CogACT Arena最大时延更高。更低mean不等于所有尾部改善，35分钟单窗口1.060070ms不替换正式pooled结果。
- **保留负收益与微小差异，不包装为显著加速**：RDT共享流负pilot保留，后续正式结果单列；pi0/pi0.5有两轮负收益，CogACT有一轮负收益且Arena最大时延更高。Norm完整回集成通过主质量门槛但慢约6.43%，已按负结果完成，不要求凑正收益。
- **跨设备误差单列**：H20 差异不删除、不改阈值；主要门槛是同一 Orin、同精度常规 TRT 与优化路径。
- **已选范围完成，不扩大为全部论文主张**：五VLA与Qwen0.8B完整实测及31项验收完成；2B用户取消。原始样本/CDF/质量/资源/服务恢复已核验；H20差异、RSS缺口、负收益及论文强主张边界保留。Qwen三段RSS错误及缺口保留，私有GPU峰值和其他功耗模式未测；same-Orin一致不等于官方源/机器人质量等价。
- Orin 不安装或运行 Torch 模型
- 不以全模型 FP32 回退凑表，不开展量化搜索
- 板端顺序运行，不争用其他任务
- 不删慢样本，不缩减模型、进程或迭代次数

剩余模型工期取决于导出、算子和容量实测；RDT 纯采样时间不能外推为整项目工期。全五模型及必需交付未闭合前，Goal 不标完成。

## 证据入口

- [代码文档提交范围、验证结果与未复测边界](reports/repository_closeout_20261027.md)
- [45小型记录和13张原图的路径/字节/SHA对应索引](evidence/retained-index-20261027.json)
- [用户授权的本机/NAS清理与源码保存及复现边界](reports/local_bulk_cleanup_20261026.md)
- [23路径/压缩清单/源码归档/论文包清理后独立核验](evidence/storage-maintenance-20261026/bulk-cleanup-postcheck-002.json)
- [NAS196路径计划/结果/SHA写前日志/不存在核对](evidence/storage-maintenance-20261026/nas-postcheck-001.json)
- [本地磁盘只读盘点与候选清理顺序，不含新增删除](reports/local_disk_inventory_20261026.md)
- [10:55本地目录、文件系统及Docker只读快照](evidence/storage-maintenance-20261026/local-disk-inventory-001.json)
- [NAS本任务196个冷模型文件计划，原始测量不选中](evidence/storage-maintenance-20261026/nas-plan-001.json)
- [Orin追加只读复查：剩余合格冷模型文件为0](evidence/storage-maintenance-20261026/orin-remaining-results-plan-002.json)
- [自包含交接与最终冷模型清理说明](reports/edgefm_handoff_and_retention_20261026.md)
- [214本机/48Orin路径删除日志及105来源不变复查](evidence/storage-maintenance-20261026/post-cleanup-verification-001.json)
- [论文交接文件夹/ZIP、EdgeFM方法标签与移位解压核验](evidence/paper-summary-20261026/handoff-closeout-003.json)
- [Actual006 seven complete native processes/7168measured; eighth started and final scope still pending（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Actual006 six complete native processes/6144measured and3full paired quality checks; observer005 active（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [372 CPU tests plus138subtests: same006modelcohort, preserved003/004observerprefixes and required005final; no data imputation（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Actual006 halfcohort five complete processes/5120measured; numerical cohort stillrunning（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [RSS004failure after5083valid prefix rows/3births; original source and gap preserved（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Actual005reader current-cgroup-only samples; no numerical model/controller restart（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Actual006 four complete processes/4096measured and2full paired quality checks; not whole-cohort acceptance（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Actual006 three complete native processes/3072measured; corrected RSS observer active after exit（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [368 CPU tests plus138subtests; explicit observer prefix/gap/final join, not model restart or platform acceptance（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [RSS observer003 failure and3598valid prefix rows; numerical model cohort not restarted（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Actual RSS observer004 active on same006parent and third native PID; observation gap not imputed（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Actual006 first complete ordinary/verified pair passes32full logit witnesses;2of10notwholecohortacceptance（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Actual006 first1152-row process passes tokens/logits/guard/PID checks; whole-cohort and restoration still pending（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Current managed006 sampling scope, true preflight, planning headroom and preserved failures](reports/orin_qwen_managed_sampling_20261026.md)
- [Actual006 outer/compute/RSS active, two authorized Java stops and full native preflight（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [353 CPU tests plus117 subtests, current006 lineage and post-terminal commands; not formal acceptance（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Two complete native processes retained from failed004 controller; not merged into006（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [005capacity32GiB refused before model start; both Java restored through original manager（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Fresh host-only retained five-VLA recheck:60processes/61440measured; Qwen still pending（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [First complete1152-row process passes native PID/guard/output checks; 1of10 is not whole-cohort acceptance（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [222 CPU tests plus117 subtests: full-data/resource/CDF/finalizer tooling ready; running cohort not accepted（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Complete first native process retained; cohort003 failed metadata and is not merged（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Actual cgroup discovery sees guard/timeout/native and complete outputs; not formal timing（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Read-only native RSS/HWM sidecar002 bound to actual cohort004（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Actual cohort004 process births and native RSS/HWM sample, not full-cohort acceptance（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [204 CPU checks plus57 subtests for full-data/CDF/protocol/memory preparation; no platform acceptance（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [quality005原RGB+在线native文本16观测64请求主GateMSE0/cos1；跨设备7同9异及logits补充单列（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [quality005终态空cgroup/MainPID0/manager和所有程序不变（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [quality005两模式各两周期32真实native请求，非正式长采样（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [005同semantic/IO完整33调用，marker4/output_49数组index48（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [004单观测prepared-text四完整call字节/finite/count/guard/maps/来源独立审计（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [004全部完整tokens/logits/CSV及首call慢样本保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [004归档终态成功/空cgroup/服务不变（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [004 compiled顺序observer绑定/同步失败抑制7项通过归档（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [002 observer source/compiled region顺序错误，保留失败非模型数值失败（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [官方五文件后host16×327 IDs byteexact，native另见quality005（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [0.8B完整常驻IR准备/无参数控制真机/首pilot边界及剩余验收](reports/orin_qwen_resident_preparation_20261025.md)
- [无参数Pad/greedy/计数/有限性控制真机19完整输出2101285元素byteexact，非模型通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整5Region33调用off/verified同语义摘要，三神经engine不变；尚未目标编译/执行（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [实际CPP→两prefix→15decoder单观测17调用/835记录，16tokens冻结参考exact，源差异/非resident边界保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [新prefix后15实际decoder735输出完整/state控制边审计，portable差异不隐藏（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [两真实prefix engine及全部pins/SDK/空cgroup/两Java原管理器恢复独立审计（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [用户明确取消2B，仅完整收尾0.8B后关闭Goal，质量与协议不变（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Orin纯C++预处理对16完整冻结pixel tensor byteexact，CPU unit空，非在线Session或正式（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [修复后prefix producer实际生成a9aff933/2109749460B，尚未推理核验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [通用C++原RGB预处理对16完整冻结pixel tensor byteexact/maxabs0，仅host未集成target（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [修复完整prefix双阶段真机构建与后续fresh接续验收边界](reports/orin_qwen_fresh_prefix_20261025.md)
- [双prefix图/profile/3.42GB共享原权重及SDK工具真机重哈希，参数IO与CPUparity绑定（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [持续授权下双prefix顺序守护构建/24GiB准入4GiB底线/原服务恢复合同（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整19decoder15实际调用/735输出112657920元素finite/所有state与控制边/16token接续一致，非fresh或正式（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [19连续诊断unit终态空cgroup/管理器与原服务快照不变（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整19decoder同旧native prefix全部49输出finite，logits cos0.99997454/MSE0.00090755（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [同19engine有效CPU state对照全部49finite，logits cos0.99997705，source补充非主Gate（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整decoder19已生成/全部pins重核/两cgroup空/唯一data343.932秒后原管理器恢复（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整原130节点mixer修复后3输出实测通过，投影cos0.84155到0.99999608（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整schema修复decoder19守护构建/持续授权/原服务恢复合同（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [原opset18 norm engine实际ReduceMean形状1x1而非16x1，推理前拒绝（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [独立19输出norm核验通过/最终exact；同suite的core构建失败单列（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整decoder79归约轴schema转换/49输出x15CPU步byteexact/参数精度不变（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [精确CPU中间输入/冻结权重的独立投影及原残差2输出maxabs5.96e-8（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [声明式gated-norm/core原图CPU全部父输出exact，原参数/精度保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [有效CPU cache对照仍差异/原3输出mixer定位/下一独立投影与norm核验边界](reports/orin_qwen_decoder_remaining_boundary_20261025.md)
- [同engine有效CPU前缀state49输出finite，但完整logits源差异仍在（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [原3输入3输出mixer实际成功，投影边界cos0.84155/状态近参考，非根因定论（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [17全部15实际调用735输出112657920元素finite/state边核验，但源token仍不同（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实query缩放原图/折叠图均byteexact MSE0/cos1，假设排除（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [3观察图selector全匹配但原FP16/FP32边界tactic失败，未放宽精度/未判原模型unsupported（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [修复后完整decoder引擎12b873c6生成/unmatched0/pins和空cgroup原程序独立审计（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [首recurrent恢复cos0.99999869但完整logits仍偏差，49输出finite/非主Gate（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [修复后15实际状态调用finite但源token不同/空unit/管理器业务未变（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [视觉前缀两段8+10静态Exp修复后，各50完整输出CPU前后byteexact，非真机（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实首层query缩放Pow/Reciprocal/Mul，3→1节点CPU源输出byteexact（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [静态Exp实测定位/stock同引擎复核/4真机修复与完整decoder回集成边界](reports/orin_qwen_readonly_exp_repair_20261025.md)
- [输入和参数exact/静态Exp首错/官方runner同错/只读折叠4真机恢复（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整15步每步49输出只读修复前后CPU byteexact/原始参数与ABI不变（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [16旧静态层选择器严格失败/无engine，全部pins和cgroup原程序独立闭合（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [仅将18已折叠计算与源层storage约束显式承接为FLOAT字面量，动态义务不变（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [两9节点真实gating最小化复现/设备序号工具修正/受控续跑](reports/orin_qwen_gate_operator_diagnosis_20261025.md)
- [原节点权重/实际源激活，两9节点对子图及49parent CPU byteexact（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [首微图复现与CUDA ordinal工具错误终态/空unit/管理器程序未变（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [明确GPU0/复用两引擎，续跑真实live3/4cases，非模型验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [同输入54约束受控试验：局部改善但32输出NaN明确拒绝，不过滤样本（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [14引擎真实构建/唯一data恢复344.027秒与全部pins/cgroup闭合审计（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [14首推理写出全部原始输出但非有限检查拒绝/管理器业务未变/空unit（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [同图仅54FLOAT约束补全/真实解析5695层unresolved0、building未验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [首recurrent CPU半非线性/半Conv代理未复现目标大误差，根因未证（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [735实际输出finite/状态边核验，但同输入源/同prefix portable差异保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [15实际调用终态/空unit/管理器业务状态未变，非质量（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [单步49实际输出7510528元素重核finite/argmax11952，非完整链（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [单步raw诊断真实完成/空unit/管理器和业务状态未变（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [新单步4031019296B引擎生成/无guard临停，pins/cgroup/原程序独立审计（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [单步engine实际exit0/314.982秒构建，非推理或质量（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [4215节点新单步实际解析5695层/unresolved0、building未完成（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [60个wide整数仅正向Slice end且INT32裁剪等价，未改源图（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [实际前缀50原输出重核/48语义映射/51decoder输入与KV零槽（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [单步真机shape边界与恢复、通用整数修正](reports/orin_qwen_reusable_decode_target_20261025.md)
- [单步解析失败/无guard，data恢复/pins/cgroup独立审计（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [移除12常量shape scatter/保留12动态KV原节点（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [修正前后完整15步49输出byteexact，非目标（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [保存重载单步全部49输出x15步eager byteexact，非目标（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [6633节点单步ONNX结构通过，rawBF16 runtime parity明确skipped（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [4251节点/51入49出通用部署准备，非真机（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [两portable完整15步49输出byteexact，16token相同/源logit误差保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [动态源16token/logits byteexact；固定缓存token相同但logits误差保留，非Orin（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CPU完整缓存接续worker007 exit0/无guard，不是目标性能（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [28GiB重试仍4GiB保护/无engine，三lease原管理器恢复及pins/cgroup独立审计（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [整段资源重试闭合，单步复用源/ABI/状态准备与原始输入回收边界](reports/orin_qwen_cached_decode_preparation_20261025.md)
- [16原PNG/CHW/原始manifest和NPZarchive重核，HWC typed transport/非推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [官方图像配置在既有主机环境复现16完整像素tensor byteexact/MSE0，非目标（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [28GiB配置真实解析通过15234network layers/精度未匹配0，构建仍进行（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [共享整片仍内存保护退出，无engine/所有pins重核与data恢复及cgroup空（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [当前独占台架指令的四项资源范围解释，非新用户回复/非历史窗口复用（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [四程序PID/出生/原管理器/可执行文件及JAR/SDK/图资源只读核验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [同图/精度28GiB准入/4GiB底线，三项按需临停后真实解析，恢复待终态（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [28范围/顺序/固定底线/原deadline/先清compute后逐lease恢复保护测试（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [首共享常量片真实解析通过15234层/精度未匹配0、构建进行（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [实际归档Orin前缀对应新首片134完整CPU参考，非目标推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [通用规则节点逐字节等于实测SDK子图的库CSE案例，非整片修复（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [受控CPU构建资源证据/通用精确规则/完整CPU核验/新整片尝试边界](reports/orin_qwen_constant_resource_sharing_20261025.md)
- [64原插件初始化资源仅数MiB，非完整构建根因证明，无GEMM/模型推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [两真实3-MatMul build-only CPU峰3880→1940MiB，GPU相同/非完整模型（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [18通用共享规则/完整输出/精度义务/只读边界测试，非目标验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [67共享/分段/BF16/precision profile邻近回归，非模型指标（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [五片仅去重复常量转置，全部2805BF16/15步/参数/状态/精度保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整五片16token/完整logits再次byteexact，304中间finite/非目标（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [首11865节点实际解析/仅data临停/原24与4GiB保护（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [首片4GiB内存保护/无engine、资产重核及cgroup空/Java原出生不变（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [实际归档Orin前缀输入的首片134输出CPU参考finite，非目标推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [57codec/layout/完整输出与无服务诊断标签软件测试，非模型结果（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [首片真实解析完成15629network layers，4366精度/输出名全部匹配，构建进行未engine（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [首小tail片真实构建/完整host分段parity及未闭合目标门槛](reports/orin_qwen_bounded_tail_build_20261025.md)
- [五图/精度源覆盖独立重核，目标所有传输与复用权重/工具重哈希（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [首12260节点尾片真实解析/owned进程及资源/Java未停快照（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [新持续授权派生的首片3600秒task/7200秒window、原管理器恢复保护（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [304中间输出finite/完整16token及所有logits分段前后byte-exact，非目标（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实host CPU worker exit0、24/3GiB保护未触发，非性能实验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [35未改动保护/原生收据检查再次通过，非模型验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [整图主动中止/无guard触发，两Java原Supervisor恢复、cgroup空及全部pins重核（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [仅停止pinned owned解析窗口以改小图，非用户暂停/超时/unsupported/质量失败（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [五个<=16000节点原计算/精度保留候选及所有typed routes，未上板（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [七段完整源IR/Plan，两最终输出及所有状态保留，非实际engine绑定（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [首递归别名候选发布前正确拒绝，原失败/源码保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [118窗口/receipt/Slice参考/typed chain/直通输出及既有路由软件检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [26770+31471节点备用分段，NodeProto/完整输出/精度投影保留，未上板或数值核验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [115软件检查含48个极端Slice端点ONNX参考，不是TRT模型核验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [三段完整源链IR/Plan与50类型精确状态路由，目标绑定及I64/I32适配仍待（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [67窗口/receipt/告警/类型路由及既有组合回归通过，非模型验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整15步tail真实原生构建/长时授权派生窗口及未闭合门槛](reports/orin_qwen_complete_tail_20261025.md)
- [完整tail parsing及实际unit/进程/服务/内存只读检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [897溢出常量仅静态有界Slice端点，非整模型等价（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [36窗口/receipt/整数告警分类软件测试，非模型验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实完整prefix输出pilot/参考差异/原管理器恢复，非16token或正式指标](reports/orin_qwen_native_prefix_pilot_20261025.md)
- [全部50真实输出7170050值，整数匹配/浮点参考误差完整保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [两段实际输出重哈希/仅data临停62.630秒后恢复/owned cgroup空（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实两段I/O与enqueue文件链，非resident或性能实验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [73工具/codec/布局/恢复测试通过，独立于目标实测（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [用户明确允许长时自主使用台架，安全/恢复/正式质量协议门槛仍保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [现有SDK导出102绑定名覆盖52入50出、顺序不同，不能替代typed API（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [无服务干预的启动检查低于16GiB正确拒绝，未启动GPU（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实单观测327IDs及图像张量7375416B，未模型执行（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [14项实际engine/数据/工具板端重哈希，未GPU或服务动作（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [58通用raw连接/codec/短窗口保护测试，非模型验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实全前缀诊断准备/严格边界/新30分钟许可待回复](reports/orin_qwen_native_prefix_diagnostic_20261025.md)
- [两段实际engine及源资产重哈希，唯一data临停/原管理器恢复/owned cgroup空（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [producer exit0后slot断言及空layer导出；SDKbuildOnly源码/日志取证，非推理通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整参数前缀第二段engine生成，exit0/893.945秒，未输出核验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [独立ExecStopPost清理owned compute后原Supervisor恢复，未超原两小时（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [复用已审计engine元数据检查器原源码，仅CPU编译未加载engine（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整参数前缀第一段engine生成/exit0，终态发现post-build内部断言，未推理验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [7项非法请求在插件/CUDA初始化前拒绝，未模型推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [第一段原生构建进行/11.55GiB可用，仅data临停，非模型通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [新完整前缀分段构建窗口/独立恢复/原生输出工具边界报告](reports/orin_qwen_partition_window_20261025.md)
- [通用原生全输出工具CPU编译通过，未执行/未加载engine（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [第一段完整参数前缀真实解析结束/构建进行，仅data临停（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [新窗口复验16项准入/权限/时限/恢复保护测试（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [新bbcd9f1f两小时独立构建/pilot授权，不含正式采样（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [新窗口模型/权重/SDK/Java出生及原Supervisor只读检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [新独立系统守护与完整参数前缀两段原生构建job（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [三连续Goal回合Orin资源/许可阻塞审计，非完成或不支持](reports/orin_qwen_resource_block_20261025.md)
- [22:57UTC15499516KiB低于16GiB，两Java原出生运行/无trtexec（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [单观测全部50状态7170050值裁剪改写前后同表示byte-exact（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实prefix输出接完整15步tail，单观测16tokens exact/来源logits误差保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [前缀全部50输出实际有限，来源state误差仍非无损（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [原始真实state接完整tail，16tokens exact但logits非byte-exact（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [前缀187BF16/117源低FP16/655源FLOAT规划，未目标精度认证（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Tail2805BF16/270源低FP16/195源FLOAT规划，大参数需JSON构建器（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [22通用类型化阶段/精度证据测试，拒绝隐式修复和不完整证据（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [新版2B前缀同步safe-softmax，190400节点/50输出，数值未验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [新版前缀190400→32776节点，浮点与公开IO不变（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整tail86886→47261整数折叠，不求值浮点（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Tail原生BF16 provider/舍入边界保留，存储载体不代表精度通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [2B完整15步tail86886节点/50入2出ONNX checker通过，runtime parity skipped（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [2B完整tail CPU导出602.031秒/exit0/无guard，非目标性能（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [2B tail256文件NAS逐一fsync/校验，本地全manifest再匹配（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [导出归档核验器6项正负例，损坏数据不发成功receipt（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [2B前缀190040→32566节点，不求值浮点/权重与IO不变（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [2B前缀整数折叠156.049秒/exit0/无guard，非数值核验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [21:54UTC Orin16183924KiB低于16GiB，两Java原出生运行/无trtexec（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [更新scratch保护后的独立大内存主机配置5项准入检查通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [21:34UTC Orin16310080KiB低于16GiB，两Java原出生运行/无trtexec（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [2B tail隔离002仍自身RSS越24GiB，保留主机边界非Orin失败（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [独立ONNX reference复现4个全遮罩浮点mask NaN红例（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [通用SDPA安全softmax及外部数据/BF16等88相邻测试通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [远端旧Torch环境39通过5失败负例保留，非完整模型执行（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [远端51通过1个NAS临时目录清理失败，未改成成功（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [远端任务私有依赖/非NAS scratch下52项通用导出回归通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [大内存CPU主机独立资源配置5项准入正负例，拒绝本机或GPU（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [2B前缀190040节点/71266张量完整载荷指纹不变，仅存储整理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [2B前缀外置整理28.011秒/exit0/无guard（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [2B tail001主机可用内存保护退出-15，未Orin执行或不支持结论（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [2B完整prefix190040节点/50输出ONNX导出checker通过，runtime parity skipped（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [2B完整prefix195.064秒/exit0/无guard，非目标性能（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [44项通用GQA与composite opmath回归通过，含预期负例（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [2B完整prefix30处SDPA，视觉16头/语言GQA，输出不变（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [2B完整15步tail90处真实SDPA头数8/2/2，输出不变（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [2B独立prefix形状/完整输出观测104.036秒/exit0/无guard（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [2B独立15步tail形状/完整输出观测36.013秒/exit0/无guard（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [2B完整参数保留prefix3→50/tail50→2保存重载全输出exact，非ONNX/目标支持（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [2B两阶段序列化198.062秒/exit0/无guard，全部worker终态（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [16真实观测128输入/参考文件匹配原hash，未重跑模型（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [资产核验器7正负例通过，缺失或改变证据不发成功receipt（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [20:55UTC16158152KiB低于16GiB，无trtexec，两Java原出生运行（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [2B完整50状态接续16tokens/logits对同CPU源byte-exact，H20误差单列（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [2B完整CPU接续有界worker126.031秒/exit0/无guard，非目标性能（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [2B完整源包SHA及首样本8输入/参考文件核验，非模型推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [2B主机输入准备有界4.002秒成功，CUDA隐藏/未改环境（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [实际2B saved JIT首argmax后50运行时Tensor，静态图边界非执行证明（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [2B独立主机源与状态评估，Orin结果仍未验收](reports/orin_qwen35_2b_preparation_20261025.md)
- [20:43UTC内存不足16GiB，两Java原出生身份运行，无新服务或模型操作（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [20:37UTC低于16GiB，无trtexec/服务未动，新许可待明确回复（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整15步2805BF16/270FP16/195源FLOAT策略，CLI单参数超限不删规则（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [原生C++ JSON构建器编译及大profile正/非法配置负例，不加载模型（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [小型非模型SDK构建正例/缺失精度名拒绝，未Qwen部署或推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [20:22UTC低于16GiB、无trtexec/服务未动，新许可待回复（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [单变量保留参数表示首token恢复源32，state误差不隐藏（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [同一完整15步tail生成16token全等源，logits非byte-exact单列（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [优先参数表示9735+11314节点/50输出byte-exact/计算不重复（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [参数版两图/precision/载荷已staged核SHA，不含执行或服务许可（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [裁剪/形状索引/整数/分段等111针对回归通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [1800秒预算退出124，四actor消失/owned组空/业务未变，非OOM或unsupported（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [从原DAG独立推导新增裁剪整数控制，不使用内部标注作为证明（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [独立裁剪控制固化后完整50输出CPU byte-exact（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [9633+11202完整前缀分段，50输出byte-exact/每个原计算一次（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [源公开常量保留/不可变数据/完整分段等167回归通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整后15步58241节点/50输入2输出部署准备，未目标验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整host16token运行但与冻结源不全相同，logits MSE2.03/cos0.895失败诊断（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [已安装ONNX IR host-only CSE同输入/初始化器/输出等价，不改部署图（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整50状态前缀部署准备，NN与全部公开IO/权重保持（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整50输出7170050值索引改写CPU byte-exact，不是目标验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [18个负Pad裁剪后完整50输出7170050值CPU byte-exact（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [首轮完整前缀暴露卷积状态负Pad，不删除缓存输出（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [第二轮完整前缀16GiB/4GiB/1800秒唯一构建，不停业务（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [187原生BF16/69源低精度dense/631源FLOAT，裁剪未改浮点节点和规则（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [纯裁剪Pad→Slice通用边界/dtype/NaN等价测试34通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [源类型/裁剪/索引/整数/Norm/Trilu/BF16相邻183回归通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整剩余15步tail85806→46631节点，浮点/公开IO保持（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [12节点恢复原依赖，真机5.768秒复现同ForeignNode/Slice断言（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [只增workspace至2048MiB仍同Slice断言，非扩容修复（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Min显式上界候选真机仍失败，负例不掩盖（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Range/Gather候选真机引擎生成，层信息slot错误单列（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [局部真实输入334848完整原生输出byte-exact/MSE0/cosine1，非整模型（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [显式上界候选36源语义边界测试通过，不代表target可用（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [有界Range/Gather候选50空输入/dtype/NaN等价测试通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实首Scatter全部边界捕获，1/7节点CPU byte-exact最小图（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [7节点真机2.530秒Myelin断言，与整图报错不同/组空/业务不动（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [8节点含原Embedding真机3.268秒同Myelin断言，不宣称整图根因（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [12节点恢复token IDs-mask原依赖，CPU边界byte-exact，待真机诊断（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [源Embedding权重块压紧复制/逐范围校验，非硬链接或参数改动（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [两完整阶段结构导出/50状态匹配与新鲜服务恢复联合checkpoint，非目标验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整剩余15步tail ONNX85806节点/50入2出结构通过，runtime skipped（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [保留共享参数tail导出183.066秒/峰值12.567GiB，无守护触发（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [保留共享参数的两完整阶段保存重加载状态及16token输出byte-exact（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [冻结tail主机导出触发既定24GiB RSS守护，非Orin或系统OOM（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [失败首Scatter精确NonZero/Slice动态边界、505 Scatter清单摘要，不预判根因（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [独立终态审计：两Java恢复/其他程序及全部图权重hash未变/own units空（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整prefix含50状态输出188677节点ONNX结构通过，runtime parity skipped（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [window003失败后原管理器恢复成功、starter不动/计算cgroup空（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [v4首段744.456秒融合Embedding/ScatterND workspace失败，非OOM或模型推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整prefix和剩余15步保存/重加载全部状态与生成输出byte-exact，非目标验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [50张量状态接续全部16token/最终logits与同CPU源byte-exact，非目标核验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [33窗口/profile/range/通用拆分软件检查通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [新获批最多2小时构建/pilot按需临停许可，不含正式长采样（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [window003已启动，仅data-collection临停、starter保留、第一段构建中（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [28窗口保护/源profile/权重范围检查通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [最小真实GEMM支持FP16内部结果，build-only非模型推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [最小真实Conv支持FP16内部结果，不需新增FP32格式岛（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [v4显式内部FP16结果及重验要求，源FLOAT与BF16保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [120权重数组无FP16溢出但203值下溢，激活/完整质量未验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [16范围/准入/固定截止/独立清理恢复软件检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [第二次整段build在QKV Gemm无实现终止，未触发资源守护/权重业务未变/组空（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [第二次真实trtexec完整构建记录及已核对SHA的失败日志索引（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [首次整段buildOnly格式失败、无资源守护触发、业务未变/进程组空（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [从真实失败构建提取同权重/shape的单Conv最小复现，预算2组合（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [单Conv FP16计算/FLOAT输出复现同一格式失败，非模型推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [仅单Conv改FP32计算/FLOAT输出可构建，未执行模型（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [v3只加实测格式所需的单Conv FP32岛，保留其他混合精度（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [140导出/变换/分区/原builder与单格式例外检查通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [20723节点单图007仍预算退出，未模型推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整首token精确两段9732+10991节点，5静态前沿，无计算复制（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整两段CPU拼接token及所有logits byte-exact，不是目标plugin核验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [第一段真实Orin解析exit0/零错误，未构建engine（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [第二段真实Orin解析exit0/零错误，未构建engine（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [由原始dense dtype派生混合精度约束，不做全FP32替代（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [5源hash/原builder/noTF32/obey/参数上限检查通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [通用整数/shape折叠首token图至20723节点，保留浮点计算（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [折叠前后完整CPU portable token及全部logits逐字节一致，非目标推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整16token图261070至79298节点，保留全部生成计算（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [134导出/BF16/shape/分区及完整核验器检查通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [006真实目标解析900秒预算退出，业务进程身份未变，未推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [封存源实际root/prefill接口及状态边界结构检查，未执行模型（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整CPU source/portable对比：token exact但logits严格诊断失败，非目标质量结论（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实TRT原生裁剪图解析900秒预算退出，未构建engine或执行推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整16token主机导出277382节点checker通过，runtime parity跳过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整源108注意力位点实际shape与完整CPU输出不变证明（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [004真实图文导出186935节点/18原生16bit FLOAT opmath，不是目标验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [57显式scale/float/half/BF16/mask/GQA及既有导出回归通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [保留003的portable ORT复数scalar类型加载失败，不冒充输出数值对比（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [004全部源张量外置前后完整payload指纹一致（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [实际CPU source完整token/logits及输入不变参考，非其他平台性能实验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Qwen0.8B真实图文首token导出、GQA边界及完整任务剩余范围](reports/orin_qwen35_support_20261024.md)
- [封存0.8B源包及8真实图文输入参考文件完整SHA/类型验证（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [18实际SDPA形状与完整CPU输出observer/direct byte-exact（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [实际首轮导出GQA不支持失败/CPU资源/完整日志SHA（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [实际二轮12视觉后未知语言头维保守拒绝，非模型unsupported（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [003实际CPU导出115.037773s/exit0，非其他平台性能实验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实图文first-token ONNX checker/186956nodes/479initializers，不含运行parity（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [51通用GQA/调用点与源字节绑定/原opmath及外部权重回归通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [按用户数据优先保留策略清理模型/engine及重新推理限制](reports/orin_storage_retention_20261024.md)
- [清理后316路径缺失/完整删除SHA日志、原69120请求/18CDF及48检查复核](evidence/storage-maintenance-20261024/post-cleanup-verification-001.json)
- [108本机冷模型路径清理结果，逐项SHA且未删除原始数据](evidence/storage-maintenance-20261024/local-payload-cleanup-001.json)
- [208真实Orin冷模型路径清理、身份/idle/完整SHA日志](evidence/storage-maintenance-20261024/target-payload-cleanup-002.json)
- [保留策略变更后的当前31项数据/文档复核，不代表二进制仍存在（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [清理后48数据/文档/边界回归通过](evidence/storage-maintenance-20261024/post-cleanup-core-tests-001.xml)
- [17文件类型/变更/打开句柄/硬链接/保留数据安全检查通过](evidence/storage-maintenance-20261024/cleanup-safety-tests-002.xml)
- [31项Goal逐项验收、同请求边界/内存限制与复跑命令](reports/orin_core_acceptance_20261024.md)
- [更新交付/claim-gap后的29核心/2可选逐项审查，Goal false（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [9份CSV/Markdown复跑字节一致、组件复查、文档SHA/48检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [最终文档状态下48项真实数据/反例/数字/边界/链接复查通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [活动Goal 31项逐项审查：29核心有界测量、2可选待评估，最终交付未完成（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [69120同请求时间分段/36边界分布核验，非新独立实验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [61440唯一测量请求的三种关联计时边界统计（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [六项真实计时分段及非法边界回归（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [板卡只读身份/SDK/业务状态复查，自有活动unit为0（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [五正式模型69120请求/61440测量及18CDF文件重hash导出（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整五模型12配置Orin主表CSV（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [五模型质量CSV，numeric与exact-only辅助输出分别计数（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [五模型60个独立正式进程原始均值表（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [五正式模型完整p50/p95/p99/max/std与质量表（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [两完整正式模型的Scalar512真实频率/三cast DSO集成追溯（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [全部五模型60进程/78667遥测envelopes与私有峰值unknown（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [60个正式进程RSS/VmHWM/共享余量及范围CSV（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [12配置采样进程峰值与非孤立GPU物理峰值限定（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [六族真实源signature/预算/合同及十二actual build收据汇总（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [十二选定engine实际构建成本，未观测Agent思考时间为null（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [五模型CSV/组件/资源/六族成本新交付与未完成边界](reports/orin_delivery_reconciliation_20261024.md)
- [34实际数据/类型端口/组件合同/资源unknown/成本回归检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Embedding/RoPE 40独立进程/40960测量及完整末输出核验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [原顺序/argv/资源保护和负收益轮次独立审计（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [全部四任务空、521目标hash/71出生终态、业务不变（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [33实际源/严格Gate/短计数/错误顺序与资源反例检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [新两族完整计时表与原合同/边界/负收益限定（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [四实际engine构建预算/成本和非物理峰值RSS（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [两族四边界全部rank CDF PNG，已实际查看](../paper_handoff/edgefm_three_platform_20261026/figures/Orin/orin__20261024-06__publication-001__operator-cdf.png)
- [原始rank/CSV/PNG/PDF/构建成本发布pins（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [五模型后实际MAXN/配置及UID1000无验证切换权限限制（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Embedding/RoPE新FLOAT合同实际构建/完整输出与边界](reports/orin_embedding_rope_float_islands_20261024.md)
- [四实际FLOAT算子引擎构建与原业务身份不变（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [新引擎SDK ABI及159360完整输出值独立exact审计（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [120目标路径/20出生终态及原业务无操作（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [27源/guard/完整输出严格阈值CPU检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT实际约7.63B完整Orin D1-D3正式报告与限制](reports/orin_cogact_formal_e2e_20261024.md)
- [CogACT完整typed cohort D1-D3联合验收，非整个Goal完成（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT 57600完整原始输出独立逐字节核验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT十进程7330遥测/33156guard资源审计（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [两项获批业务原Supervisor恢复及四小时窗口审计（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT恢复后58187目标路径hash/36记录出生终态（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT完整10240测量原始时延分布与逐轮负收益（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT完整两模式正式时延表（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT完整pooled CDF图，已实际查看](../paper_handoff/edgefm_three_platform_20261026/figures/Orin/orin__20261024-05__publication-curves-001__cdf-pooled.png)
- [CogACT完整pooled CDF矢量图](../paper_handoff/edgefm_three_platform_20261026/figures/Orin/orin__20261024-05__publication-curves-001__cdf-pooled.pdf)
- [CogACT十独立进程CDF图，已实际查看](../paper_handoff/edgefm_three_platform_20261026/figures/Orin/orin__20261024-05__publication-curves-001__cdf-processes.png)
- [CogACT十独立进程CDF矢量图](../paper_handoff/edgefm_three_platform_20261026/figures/Orin/orin__20261024-05__publication-curves-001__cdf-processes.pdf)
- [CogACT十二条未改rank曲线与验收hash绑定回执（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT服务恢复后完整终态归档传输回执（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-080（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-079（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-078（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-077（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-076（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-075（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-074（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-073（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-072（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-071（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-070（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-069（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-068（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-067（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-066（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-065（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-064（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-063（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-062（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-061（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-060（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-059（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-058（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-057（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-056（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-055（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-054（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-053（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-052（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-051（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-050（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-049（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-048（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-047（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-046（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-045（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-044（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-043（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-042（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-041（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-040（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-039（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-038（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-037（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-036（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-035（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [D4逐项剩余清单：已完成负收益/有界热稳态功耗与真实缺项分开，不添加隐藏验收条件](reports/orin_d4_remaining_execution_20261024.md)
- [独立FLOAT算子构建准备：原四图/精度声明不改，禁FP16/INT8且noTF32；尚未上板（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [四个真实算子图/源证明/构建器的冻结主机计划，须五模型正式验收与恢复归档后才激活（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [11项CPU源图/构建参数/原始文件未改检查通过，不是新engine或性能验证（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [04:14UTC仅两获批原生业务临停，其他受管程序/原配置未变，恢复待采样结束（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-034（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [终态只读归档入口已准备：活跃窗口先拒绝，实际依赖/noise/输出/记录进程逐一核验，尚未实际执行（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [77项离线NONMODEL检查通过：70整批typed合同及7目标归档mock，不是实际模型或服务恢复验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [03:52UTC仅两获批原生业务仍临停，原管理器/其他受管程序/配置不变，实际恢复待采样结束（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-027（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [主机端typed整批验收入口：保留十进程/样本/慢尾及固定质量，实际结果未生成（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [68项离线合同通过：完整六类型NONMODEL样例、精确整数输出、恢复失败不得接纳（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [03:39UTC仅两获批原生业务临停，其他受管程序及原配置不变，恢复仍待结束（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-021（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [仅主机：通用显式程序窗口终态恢复核验，不运行模型/操作服务，不接受模型性能（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [53项离线检查通过：22项新恢复合同含实时未恢复拒绝，31项既有边界回归（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [03:32UTC服务检查：仅获批两项临停，其他受管程序不变，恢复仍待当前采样结束（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-017（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [03:25UTC只读检查：仅两项获批原生服务临停，其他受管程序不变，恢复待采样结束（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-011（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [历史/当前实验合并31项检查，按源文件隔离同名模块加载（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [保留合并测试首次29通过/1导入串用失败，不是板端故障（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [两个获批业务仍停止、其他原管理器程序未变，实际恢复仍待采样结束（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-010（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [实际只读服务观察：两项获批业务保持停止，其他原管理器程序未变，尚未恢复不标完成（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [19项原生/解释型显式身份、回滚与恢复就绪条件检查，模拟不作硬件验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-006（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-002（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT原生业务服务新窗口的完整正式激活规格（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT新窗口只读PID/出生/cgroup/资源探针及正式计数 checkpoint-001（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [仅两原生业务临停/原管理器恢复/新四小时，不禁用自启（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [13项按声明管理原生/解释型程序、回滚/固定计算参数检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [独立新4h外层与计算cgroup，900s恢复预留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [旧两Java批次1/10后准入失败，5760输出exact/5804hash/恢复已审计（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [26项实际部分数据拒绝整批/恢复和门槛检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT只读PID/出生/cgroup/计数与服务状态核验 checkpoint-007（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT只读PID/出生/cgroup/计数与服务状态核验 checkpoint-004（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT只读PID/出生/cgroup/计数与服务状态核验 checkpoint-003（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [代表VLA完整算子回集成实验闭合，负收益与全Goal完成严格分开（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [九项真实重审/门槛及计数不可放宽检查，保留来源工具版本差异（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [实验完成不等于正收益；原Norm数据hash完全重现，其他D4仍未完](reports/orin_integration_requirement_reconciliation_20261024.md)
- [CogACT只读PID/出生/cgroup/计数与服务状态核验 checkpoint-002（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [新授权CogACT正式激活spec：两模式5x128+1024、24/4GiB不变（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT只读PID/出生/cgroup/计数与服务状态核验 checkpoint-001（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [独立新许可：两Java最多4小时，原管理器恢复；非旧pilot权限（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [新固定窗口与无条件ExecStopPost，预留900秒恢复（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [15项独立权限/固定预算/先清理自有计算后恢复的边界检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [本机ENOSPC监听配额下恢复只读看板，改为单文件stat轮询，实测HTTP/PID（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [12项看板回归含原子替换、错误后恢复及关闭，不改系统配额（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [通用FP32 Norm不采用：计数不足、配对负收益，未进入完整模型（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [20进程20479原始记录：1023/1024拒绝正式计时，不删慢样本（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [全部20479条TRT原始微基准记录，pilot-only（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [18项完整数值/篡改拒绝/缺样本不通过/保留负收益检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [两unit success/inactive、原Java未变，可用22264460KiB低于24GiB（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [通用FP32 Norm完整微核验通过但不采用，未关闭D4](reports/orin_generic_norm_candidate_20261024.md)
- [实机通用Norm完整786432值严格门槛通过、来源/精度及终态核验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [当前NV插件对新候选、5配对两TRT边界，非VLA E2E（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [单候选通用FP32 Norm：真实签名、严格完整输出门槛、冻结微基准预算（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [公开SDK层诊断：66保存输出对/1254运行期检查，217hash/11终态身份（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [冻结引擎/真实输入/SSA分区路由，无新精度或完整E2E声明（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [15项实际SDK/完整prefix输出/篡改拒绝/图表和内存口径检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [SDK报告层均值和上下文需求图，非E2E/物理峰值（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [第二版分离图例/注释轨道PNG/PDF和源hash凭据（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [692个SDK报告层的16次原始计时统计CSV（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [引擎文件/上下文需求/探针IO独立内存口径，不求物理总量（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [公开SDK无CUPTI prefix层诊断、格式转换开销和上下文需求报告](reports/orin_sdk_layer_reservations_20261024.md)
- [01:02两自有unit均已退出，最新容量仍不足CogACT24GiB（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CUPTI两API权限拒绝，52目标文件/9终态身份，非内存测量（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Activity注册返回35，未进入分配控制main，无有效trace（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [独立Callback订阅返回35，domains未启用且未申请分配（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [9项实际权限边界/空trace不当零/源与清理检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CUPTI观测权限边界、无新模型结果及后续非CUPTI接口口径](reports/orin_cupti_memory_boundary_20261024.md)
- [23:08再次核实unit inactive/MainPID0、15完整进程与当前容量（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整匹配功耗补充联合闭包：34560输出、192文件、33终态身份（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [原完整15worker审计器核验34560动作tensor和全部新CDF（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [全部15进程功耗窗：2423原始/1854合格样本，保留排除和负结果（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [匹配功耗PNG/PDF和全部原始轨迹生产端hash/像素检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [三模式匹配功耗窗口与全部2423原始读数图（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [18项实际完整数据/窗口/错误cohort拒绝/图和终态检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [15/15采集终态及MainPID0/success真实快照（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [结束后5次约22.4GiB内存不足CogACT24GiB、应用保持运行（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [22:50匹配功耗14/15完整进程、14336测量、2279功耗行快照（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [22:44同批次11/15完整进程、12103测量、1910功耗行快照（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [四接受模型57600原始行/51200测量独立统计与14CDF资产核验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [四模型十配置完整mean/p50/p95/p99/max/std/吞吐/ratio机器表（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [原完整动作审计继承质量表，CogACT正式数值不填充（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [四接受模型主表和质量Markdown，五模型缺项显式N/A（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [14项真实cohort与部分接受/错误边界/H20主Gate/篡改CSV或图拒绝（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [22:33真实同批次7/15完整进程、8071测量、1262功耗行快照（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [15项源/实际窗口和真实live-archive拒绝检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [终态归档器实际拒绝活跃unit，无复制/重启实验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [22:20三模式首轮完成3/15进程与真实活跃控制器只读快照（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [首完整进程2304动作tensor独立exact，非整批接受（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [14项完整首动作/真实功耗窗变异拒绝与原整批门槛检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [同完整模型匹配功耗运行口径、首动作核验及全部未决边界](reports/orin_matched_power_supplement_20261024.md)
- [同7engine三模式匹配功耗冻结计划，模型/采集器及完整计数不变（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [4项冻结协议/真实身份与观测窗/无应用管理源检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [新自有user compute unit启动回执，原应用无停启授权/操作（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [22:14真实PID/出生/argv及首进程744测量和129功耗行只读快照（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [4实际refit构建仍精度不合格，55路径/10身份终态核验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [9项固定命令差分和真实归档/进程/不提升formal检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [固定refit编译可观测性负结果及资源/证据报告](reports/orin_refit_observability_diagnostic_20261024.md)
- [固定refit编译诊断计划，原图/profile不变且无推理计时授权（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [6项固定refit唯一命令差分与原资源线/无应用管理测试（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [终态不完整批次5788路径/5进程闭包，1/10不是formal接受（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [8项真实部分输出/资源/完整门槛拒绝及无应用管理检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT首完整进程与下一入场容量失败、无应用控制及终态报告](reports/orin_cogact_no_service_partial_attempt_20261024.md)
- [首进程结束后下一入场内存失败，unit终态/Java未变快照（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [首完整1152调用/5760输出全部byte-exact，单进程非整批正式验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [首已完成进程24GiB/4GiB采样资源审计通过，非完整cohort（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [失败后自有cgroup已空的post记录，无原应用控制（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [保持原应用运行的CogACT正式采集范围、环境限制与运行状态](reports/orin_cogact_formal_no_application_control_20261024.md)
- [无应用停启的完整正式协议，24GiB/4GiB/十进程不缩减（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [原Goal内保持应用运行的推理路径，不捏造新停服授权（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [自有四小时compute unit启动回执，services_stopped=false（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [实际unit/collector/worker身份及481measured实时快照，非终审（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [无应用管理/不伪造新授权/原完整协议真实归档三项检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Orin真实CPU NONMODEL cgroup机制资格，不是生产服务窗/模型通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [正常/超时/自身主进程SIGKILL三CPU case原始命令与post记录（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [3自有单元已消失、原Java进程仍运行及旧6h已过期的只读复查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [20项实际证据/反例/纯4h预算检查，不授权任务（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [旧窗口边界审查、用户级cgroup真机诊断与生产集成剩余项](reports/orin_cogact_window_mechanism_20261024.md)
- [通用显式资源合同的collector/auditor整文件inverse，未部署目标（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [160既有真实typed pilot输出复核与通用collector质量schema兼容，非新推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实归档trace及93字节alias兼容核验，不是服务租约或现场预检（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [24GiB/4GiB与typed数据源码资格封存，formal activation false（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [83项新源资源/typed/真实归档兼容检查，不计正式数据（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [47项旧源完整行为及归档身份检查，原文件不改（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [不可运行的正式template，独立人类授权和租约仍空（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [正式资源/五输出合同离线准备、检查范围及未满足启动条件](reports/orin_cogact_formal_source_readiness_20261024.md)
- [真实独立ordinary复验16请求80tensor对原参考全部exact，非formal（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实普通复验24GiB/4GiB资源与原Manager/Jar/cwd/argv/config恢复审计（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [127路径含16共享资源16.45GB完整SHA与3进程终态核验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [26本机归档/坏数据/原policy检查及19子测试，非新正式数据（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [仅关闭普通复验前置的未激活formal草稿，独立授权/窗口/预检仍空（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT真实常规复验/原服务恢复及正式剩余前置报告](reports/orin_cogact_ordinary_repeat_20261024.md)
- [Attention真实全动作主gate失败及88目标文件/10进程终态闭包（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整34动作对审计：normalized最坏MSE1.2796944e-5失败，baseline原参考exact（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [标准化action主门槛明确拒绝，native补充且阈值不改（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [复用同源重建控制完整动作比较仍失败，原审计不被替换（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [45项真实失败复现/源与精度/原GEMM控制回归（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Attention完整模型回集成失败及同源重建补充/资源报告](reports/orin_operator_attention_integration_20261024.md)
- [Attention整prefix目标构建进行中，非engine/全动作/性能通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [28项既有elision/source检查现时复跑通过，非目标动作结果（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实两算子159360值exact/160文件SHA/26进程终态，内部精度未认证未计时（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实Embedding完整46080值exact，SDK ABI通过，融合存储精度未认证（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实RoPE两观测113280值exact，原生SDK ABI通过但内部融合未认证（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [66项真实SDK/全输出及旧source/typed回归，未认证精度不被晋级（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Embedding/RoPE真实目标数值、SDK ABI及内部精度边界报告](reports/orin_embedding_rope_target_20261024.md)
- [五输出正式核验器离线47测试及原源inverse通过，非真实formal数据（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [27旧回归+20typed测试，模拟57600tensor覆盖不计Orin采样（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [17:58UTC两Java原Supervisor Running的只读进程状态，非应用健康（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实同Orin完整16对80tensor全byte-exact，normalized MSE0/cos1，非formal（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [D1-D2完整质量pilot及原管理器恢复/资源门槛独立通过，formal false（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT完整双路径质量pilot报告及H20负例、资源/范围限制](reports/orin_cogact_complete_quality_pilot_20261024.md)
- [112输入严格匹配的H20/Orin补充48完整action比较，非主gate/加速比（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [固定10进程/10240measured/57600完整tensor的未授权未激活formal草稿（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [额外ordinary复验24GiB拒绝，未启动worker且原服务恢复，非质量失败（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实元数据绑定完整两个Session源码，唯一Plan差异Arena，非目标质量（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整ordinary/Arena Session两个库实际ARM64编译，未据此标质量（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实常规完整16帧80tensor finite/计数11/RNG末态核验，未比较Arena（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Arena未启动：22849736KiB低于24GiB准入，未降线（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [两授权Java临停后仍不足24GiB，未启动worker且原服务恢复（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [12engine真实I/O+4provider实际Load/workspace/cleanup及原服务恢复独立核验，非模型（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [13/13查询终态169.697s且原服务恢复、进程退出，非完整模型（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实4provider CUDA Load/query/cleanup的80事件，无Run/Synchronize模型计算（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [固定查询policy与worker/preflight/helper及原始身份44本机回归（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [只查询的固定scope控制器派生：原授权/前窗/13worker不可替换，非pilot（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [全部12真实引擎构建终态、原服务恢复观察；非完整模型（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [构建窗后新鲜原Manager/Jar/argv/cwd服务恢复检查，非应用健康（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实共享loader发现zero-cap sidecar冲突，17HOST探针回归，未目标query（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [通用control省去无能力optional扩展：原三库红/新三库绿及完整源inverse（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [checked provider编译variant命名空间修复，核心与380B/align256/ID4不变（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [ABI相容源修复后4库真实ARM64/CUDA11.4重编，未query（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [ABI/资源清理探针真实ARM64编译，程序未执行（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [原生操作窗及拒绝活构建查询请求28本机测试，非实际GPU查询（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [7/12真实引擎已编译、第八段进行，临停保护仍有效（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [后续query/pilot窗口23本机测试通过，未在目标启用（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [复用审查生命周期的原生操作窗口派生，绝对截止不续期，非实际查询（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [4/12真实引擎已编译、第五段进行，控制器及服务状态仍有效（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [3/12真实引擎已编译、第四段进行，非完整模型/恢复验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [2/12真实引擎已编译、第三段进行，部分窗口观察非模型验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [16Region实际静态ABI与编译ID审计，原序号≠编译ID，真实query未做（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [实际canonicalize排序/typed ports/workspace声明10父NONMODEL回归（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [首完整prefix阶段真实TRT引擎构建363.542s/1.528GB，非模型质量（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [实际4765层：MatMul139 FP16与FP32回转可追溯，非动作核验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [实际1/12引擎已编译，第二阶段进行中的部分窗口观察（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [3control+1checked-integer真实ARM64/CUDA11.4编译，未加载/功能核验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Orin ARM64原生输入16帧112完整tensor独立exact、564源文件/二进制核验，非模型（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [523 half-dense显式FP16结果profile派生；96FP32/Erf等保留，源inverse精确（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [FP16结果profile派生及既有通用API32父回归通过，非模型质量（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [重试24小元数据文件目标fullSHA/size一致，非模型参数改变（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实首段重试进行中：服务临停及controller/builder/trtexec身份复核，原截止不延长（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实首构建失败边界及原Supervisor/JAR/cwd/argv/config恢复独立核验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [3个实际源形状TRT微图：FP16/FP32输出失败，FP16输出构建通过，非模型验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT全部59文件/29.826GB实际目标SHA/size一致，无构建（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [构建窗口和原服务helper47本机回归含内外spawn信号/慢停恢复（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [独立审查：两中断阻塞修复，2项HOST回归通过，非真机恢复证据（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [窗口已激活：两授权服务临停，首段TRT实际构建及进程birth复核（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [目标内存guard中断临界区派生，源及12请求inverse精确，模型pin不变（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [构建专用最多6h窗口/旧服务helper42本机测试，无台架操作（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [首段实际源/data/profile及9项请求输入目标fullSHA/size一致，非构建（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [3个原生源码包已传目标且SHA/size一致，尚未执行（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5全十进程D1-D3/完整动作/资源/恢复/230文件联合正式通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整23040张量含warmup独立byte-exact，主Gate MSE0/cos1（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整12057遥测/58325守护采样合同通过，9.622s间隙保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi05两Java原管理器恢复及新进程身份合同独立通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [230项真实目标文件完整流式SHA核验通过，含两实际模式库（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [10240测量全量统计/CDF：普通1258.793/Arena1256.956ms，仅0.146%差异（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [正式已验收全部CDF rank的PNG/PDF，数值未改（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5正式E2E/质量/资源/负收益与来源差异报告](reports/orin_pi05_formal_e2e_20261023.md)
- [pi05十进程均终态、恢复守护结束、两Java原管理器Running（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [新用户授权：仅CogACT首轮构建/pilot最多6h临停并原管理器恢复，尚未激活（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [受检I64/I32 CUDA组件主机编译及59注入测试通过，真实CUDA待验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [12真实源/精度绑定构建请求33父元数据测试通过，未上板（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5正式9/10，最后Arena真实活跃/72预热（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [九完整进程20736tensor独立全字节一致，非整组正式（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [九进程10853遥测/52537守护，采样间隙与权限保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5正式8/10，第九ordinary真实活跃/840测量（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [八完整进程18432tensor独立全字节一致，非整组正式（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [八进程9610遥测/46641守护，采样间隙与权限保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [实际16region源IR/25命名跨段路由父核验及整个源inverse通过，unbound（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [三控制provider105父注入测试/840控制输出字节一致，非CUDA目标（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整GPU分支raw runner主机链接/CPU原始112输入字节核验，非目标（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整16region Module公开header/Cpp/ModuleIO对已编译runner全部字节一致（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [显式记录source/runtime schema标签差异及whole Module inverse，无tensor转换（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [原生输入/完整typed HOST runner及18.35MB源包冻结，ARM64待验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5正式7/10，第八ordinary真实活跃/181测量（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [七完整进程16128tensor独立全字节一致，非整组正式（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [七进程8417遥测/40837守护，9.622s间隙与权限保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [实际神经step25LN/77dense完整源图与精度重放一致，非目标（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [实际保存IR全部源operand独立重建及whole-Module inverse通过，unbound（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [通用多阶段组合/实际IR/65新旧测试冻结，非目标（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [11段实际源29.467GB/970range父全字节复核通过，非目标（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整六类型输出handler父76 HOST NONMODEL测试通过，非整runner目标（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5正式6/10，第七Arena真实活跃/145测量（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [六完整进程13824tensor独立全字节一致，非整组正式（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [六进程7230遥测/35053守护，采样/权限限制保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整step实际源节点/权重/typed路由/四输出join独立核验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [29节点控制核20调用200完整输出逐字节一致，非神经step（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整控制分区43测试及源/主机证据冻结，目标未验证（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [11段prefix父独立重放：源/权重路由inverse一致，非目标（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [通用typed transport父66注入测试，实际CUDA调用0（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5正式5/10，第六Arena真实活跃/818测量（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [五完整进程11520tensor独立全字节一致，非整组正式（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [五进程6028遥测/29204守护，采样/权限限制保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5正式4/10，第五ordinary真实活跃/296测量（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [四完整进程9216tensor独立全字节一致，非整组正式（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [四进程4821遥测/23335守护，采样/权限限制保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [12节点私有系数C++核100完整输出exact，非神经/目标（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [通用Sub/静态Expand/溢出不发布/旧源码一致13测试（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [可执行私有系数核、源保持与目标/全step限制（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5正式3/10，第四ordinary真实活跃/500测量（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [三完整进程6912tensor独立字节一致，非整组正式（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [三进程3624遥测/17532守护，采样限制保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT原生主机准备：形状/控制核/输入完整证据和目标缺口](reports/orin_cogact_native_preparation_20261023.md)
- [CogACT缺失静态形状源证明及精确F64系数接口（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [initialize/finish实际主机C++核父重放20完整输出字节一致（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [13形状+57控制核父回归70通过，无神经/目标执行（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [独立原始NPZ对照112输入exact及16Prepare受检零文件IO（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [原生输入/域检查/fresh数据和trace审计43通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5正式2/10完整，第三worker实际活跃/101测量（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [两完整进程4608完整tensor独立全字节一致，非整组正式（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [两进程2415遥测/11673守护，保留5.799s间隙和权限限制（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [新closure/服务审计78项父NONMODEL回归通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [双真实库/原授权/正式协议metadata配置父复验，实际终审未执行（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实授权hash/问题ID/原话绑定通过，活动窗口仍pending，非恢复证明（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT源精度与类型静态准备，保留未证形状/源MSE失败，非目标通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT源准备26项父NONMODEL测试通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT原生输入主机16帧112tensor全字节一致，非Orin模型结果（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [原生输入扩展后旧OpenPI32帧320tensor回归全字节一致（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [原生输入扩展后原23项API坏参数/生命周期检查通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5正式1/10完整进程，第二Arena真实活跃身份（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [首完整进程2304完整tensor独立全字节一致，非整组正式（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [首1203遥测/5811守护资源审计，保留间隙/权限/传感器限制（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5正式首进程128预热+518测量只读核验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [主机终审工具201项NONMODEL父回归通过，不计模型实验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5实际独立ABI/源绑定169pins验证（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5十个真实独立目标资源copy及全部SHA匹配（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5正式首进程128预热+120测量身份/在线标记核验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5 H20原raw/noise对齐补充父重跑，0/16达标，非主Gate（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5 H20补充完整误差报告，负结果/混杂因素保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5通用离线补充审计28项父重跑NONMODEL测试（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5同Orin两模式128完整tensor主Gate通过，非正式全量（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [本次pi0.5临停/原管理器恢复独立用户授权（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5已激活正式固定协议/新lease SHA冻结（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5本次服务窗口激活快照，非终态恢复证明（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5恢复窗口真实启动身份（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5正式首进程活跃身份/13预热/服务临停只读核验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [恢复守护父进程重跑5项NONMODEL测试通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5完整raw普通/arena paired pilot，正式采样另计](reports/orin_pi05_raw_pilot_20261023.md)
- [32请求64完整tensor独立核验/重复字节一致，仅record-only（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [16观测same-Orin完整参考，77f2cf...，非H20/官方源等价证明（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [time/finite真机120完整控制case字节一致，非完整动作质量（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [八引擎真实构建/ABI及finite修正边界](reports/orin_pi05_target_engines_20261023.md)
- [stage003实际built-not-validated，585秒/1458999614B（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [stage003真实132端口ABI，未创建context/未推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [step神经引擎guard23307实际启动，非构建完成（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [五个真实前缀engine共5642021145B本机副本独立SHA/字节数一致（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [旧H20第三方安装包17份清理3.825GiB，保留版本/SHA及原始日志](evidence/storage-maintenance-20261023/public-wheel-cache-cleanup-001.json)
- [用户指定data目录已清理约70GiB，模型/环境/实验证据保留](evidence/storage-maintenance-20261023/cleanup-summary-001.json)
- [stage002实际built-not-validated，577秒/1389687337B（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [stage002真实95端口ABI，未创建context/未推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [最后前缀段stage003 guard8681实际启动，未宣称完成（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [consumer实际built-not-validated，513秒/901004425B，非模型验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [consumer真实72端口ABI，context需求非峰值/未推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [stage002目标构建guard4185853实际启动（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [共享视觉段实际built-not-validated，139秒/484582813B，非模型通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [共享段真实3端口ABI/17380864B context需求，未推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [consumer目标构建guard4171789启动，非构建完成（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [共享视觉段目标构建启动，guard4165845，非构建完成（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [共享视觉段源/权重/profile、设备/资源只读准入，未模型推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [首段实际构建/18端口ABI/934守护样本审计，非模型验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [共享段/consumer共2762950656B权重传板，9文件回读一致，仅文件传输（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0正式E2E报告：完整核验/细微均值差/负收益/资源缺口](reports/orin_pi0_formal_e2e_20261023.md)
- [pi0.5首段源/权重/profile pins和设备/资源只读核验，未推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [首段实际目标构建启动guard4151193，非构建完成/模型通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0最终D1-D3联合验收通过，非全Goal完成（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [23040完整normalized/native动作tensor全字节一致（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [11026遥测和52419守护采样完整核验，缺失/权限保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [原管理器恢复两个新进程，非应用健康/内存恢复证明（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [最终目标216冻结文件完整pins核对（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [全部10240正式时延/逐进程CDF，均值差仅约0.4%（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [已验收原始rank数据的中性标题PNG/PDF，无数值修改（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0前9完整worker20736动作tensor独立全字节一致，最后进程仍在运行（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0已9/10完整进程，最后worker4103897 live匹配，累计9221measured；恢复仍待结束（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [前9worker资源9909遥测/47181守护采样，缺失和权限保留，非全批资源通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0前8完整worker18432动作tensor独立全字节一致，全批尚未验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0已8/10完整进程，第九worker4078052 live匹配；累计8990measured，服务窗仍运行（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [前8worker资源8818遥测/41961守护采样，缺失/权限保留，非全批验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [父复跑58项NONMODEL联合审计测试，不能把partial/混批/失败输入升格为正式通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [实际台架冻结文件核验器在采样未完时正确pending、不读取大文件、不执行模型/服务（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0前7完整worker：16128完整动作tensor独立全字节一致，非全批正式验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0已7/10完整进程、第八worker4052414 live匹配；累计7715measured，恢复仍待结束（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [前7worker资源7729遥测/36745守护采样，权限与缺失保留；不是全批资源通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5共享callable和consumer实际源发布：2.76GB/290范围字节一致，非目标模型通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5实际源10-region组合：3fresh共享调用/逆向全Module相等，全部uncompiled/unbound（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [父只读重新组合并核对5份完整JSON/32pins一致，无目标/数值/能力伪造（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [父复跑19项NONMODEL source IR组合测试，非目标模型/数值验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0仍5/10完整进程，第六worker live匹配且403measured，累计5523/10240，服务窗仍运行（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0前5完整worker独立核验11520动作tensor全字节一致；半批完成非正式验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0已5/10完整进程，第六worker4001829 live匹配；累计5225measured、服务窗仍运行（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [前5worker资源5561遥测/26394守护采样、缺失与权限错误保留，非全批资源验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [native服务窗终审/只读恢复观察器已准备；当前真实恢复仍pending，未调用服务操作（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [父复跑36项NONMODEL服务窗测试：新birth、原管理器与真实恢复顺序投影；非真实恢复通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实pi0.5 source三份1400-node callable精确等价与652 consumer边界，未物化/目标通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [父只读复验3份完整JSON-canonical对象及31 pins一致，无忽略字段/模型执行（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [父复跑15项NONMODEL重复子图证明测试，非目标数值/完整模型验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0前4完整worker：9216完整动作tensor独立逐字节一致，局部完成非全批正式通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0已4/10完整进程，第五worker3976292实际live匹配；累计4291measured，服务恢复仍待完成（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [前4worker资源：4460遥测/21155守护采样；退出计数缺失/无效传感器和权限错误均保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0前三完整worker：6912完整动作tensor独立全字节一致，仍非完整正式批次（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0已3/10完整进程，第四worker3950841 live匹配；累计3329measured，服务恢复仍待批次结束（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [前三worker资源：3365遥测/15947守护采样、RSS缺失/无效温度/功耗权限显式保留，非全批资源验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [20项NONMODEL资源审计回归：身份/hash/采样时序/内存准入与安全线、缺失不能补零（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [用户授权本机清理：pip可重下载缓存6075.8MB/1133文件；权重/环境/实验证据保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0前两完整worker独立核验4608动作tensor全字节一致，仍为selected-cases非全批正式通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0已2/10完整进程、累计2350measured；第三worker3924745实际live匹配，服务守护继续运行（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [native全量审计后才允许统计/CDF：保留完整尾部/负收益，source/服务/资源最终验收独立（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [父进程独立复跑42项后处理NONMODEL测试全通过，非神经模型/正式时延结果（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0第二进程328measured、累计1352/10240；守护/collector/worker live匹配，恢复未完成（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0首个完整正式worker：2304完整动作tensor独立逐字节一致，仅selected-case不宣称全批通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0已1/10完整进程；第二worker3898841真实live匹配、24measured，原服务仍待守护恢复（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [27项NONMODEL审计回归通过：完整尾部/计数/类型/finite/相似度与partial不得正式通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0新批次128warmup+297measured，完整在线flags正常；守护/collector/worker live匹配，非正式通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [本轮人类明确授权：仅两Java临停且结束/失败/中断原管理器恢复（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0新formal-002冻结协议：双模式各5进程，授权/lease快照与脚本哈希绑定（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [本次live只读核验：守护/collector/worker身份相符，其他受管进程不变，非正式通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [本次临停恢复窗口与采样守护状态，恢复尚待批次结束（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5四段11.26GB source只做板端文件缓存字节核对，无engine或模型执行（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5独立source结构/路由及11.26GB范围字节审计，非目标数值证明（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5通用6到9region source IR及inverse等价，全部unbound，未目标构建（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5通用元数据4段计划：9589节点恰一次、37输出保留、无目标容量或峰值内存承诺（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5只发布首个自包含source stage0/2.80GB，非完整模型或目标通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [独立核验首段节点/18路由/193引用权重范围，保留原37remainder输出，非目标数值证明（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5 prefix v5：165处LayerNorm独立图重放、I/O/initializer不变，仅主机准备（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0.5源dtype精度声明：806 BF16 MatMul提议FP16、4个原FLOAT dense保留FP32，非目标证明（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [既有通用LayerNorm改写50项软件回归，非模型通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0最终resume：D1/D2 pilot通过、formal准入终态cases0/待JAR新授权、H20supp失败边界（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0合并主报告：完整raw/D2同Orin通过、负Arena pilot、H20supp大差异、formal内存准入终态失败](reports/orin_pi0_raw_pilot_20261023.md)
- [H20补充边界：强身份/noise对齐但normalized0/16，不证官方BF16/机器人质量，不改formal primary（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [H20完整tensor离线supp审计cos0.668535/MSE0.0682653、0/16，非主Gate或formal阻塞（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [H20 supp独立复跑JSON与首审字节一致，原差异/失败保留，不放宽阈值（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [首次formal001内存准入failed-preserved/cases=[]、0正式数据，非live/采样或quality失败（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0 D2同Orin独立paired Gate passed：128完整tensor全字节exact/MSE0/cos1，非D3正式（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [优化same11合同/Module/control/resources、仅arena、memo禁用/loop off preflight，非quality通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [ordinary-only现有native库/11资源/SDK/exports/strictFP只读审计，非优化库或模型验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [16观测same-Orin ordinary参考已冻结SHA94b9b4fa...，非paired质量通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [优化source verified/loop off、Plan仅arena、memo禁用、capability不变，未target-qualified（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整ordinary rawpilot32请求完成，recorded-not-compared，非paired quality/formal通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整rawpilot64完整F32/F64 tensor bytes/SHA/shape/finite复验，record-only非D2/D3验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整11-region/10-step原生库CPU preflight通过，34依赖/header pins不变，非模型验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [11资源独立copy共7336379858B/source-dest SHA通过，无原badengine，非运行/质量通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [repeated canonical DAG/params/shape/逐node precision exact源证明，非运行时binding/完整模型验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [原652节点consumer实际构建901156854B/649815...，构建不等于模型通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [原18输入+3路GPU features诊断consumer51输出finite，1样本中段而非整模quality（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [同final-only engine三路实际activation各524288值finite；callable equivalence/IR proof pending，非模型通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [同1400节点单视角final-only实际构建484615455B/3c5259...，非完整模型验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [单视角仅linear_162实际524288 F32全finite(-20.75..20.75)，非多视图/完整模型通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [仅GraphOutputs逆操作恢复旧697输出整ModelProto字节，1400源节点不变（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [root20视觉producer实际build001成功488599688B/d6912acc...，未数值通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [原18输入独立执行视觉producer三输出全NaN，不含Gemma，非模型通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [视觉producer诊断guard exit0仅采集完成，三输出全NaN失败保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [root20 source-preserving二分4852=4200+652节点、权重/precision不变，已发布未target-qualified（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [最终resume checkpoint：原part01独立复现12/51 NaN、全51输出字节同Session，正式2/5，74pass/2skip，本轮GPU终态（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [单视角tail真机697输出/1624113332B统计finite、同步成功；rawbins仅板端，非原whole part01通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实Orin首个MLP诊断probe17观测输出全finite，非完整prefix/模型通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [MLP局部probe真实trtexec输入dump执行guard exit0，非正式采样（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整单视角尾段697观测输出probe准备及逐节点precision谱系，非执行通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CPU FP16 IO模拟695探针+51输出finite，不替代Orin精度/模型结果（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [744同步dump实际audit：part01 transpose_330图像tokens NaN，已记录SSA边界0不匹配；非模型通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pilot002完整guard：exit86/source acceptance拒绝两输出，无完整action（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [untimed同步dump诊断guard：仍由真实acceptance拒绝，非性能或验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [28节点/17输出最小vision MLP实际Orin build002 exit0，仅built-not-validated（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [part01 CPU shadow初轮51输出finite，非目标精度/性能或完整模型验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [root17真实编译CPU shared-plugin软件ABI回归15通过，明确NONMODEL（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Session004严格绑定已有Plan workspace源码，全10实际合同，无目标验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [修复Session004实际原生链接234344B/e7765d45...，非D1通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Session003/004 Module/Plan/10 artifact contracts字节不变，无model/precision变化（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [已有Plan workspace绑定的codegen/Plan回归28通过，非模型执行（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [workspace replay回归19通过/2 opt-in真实CUDA跳过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [首次actual raw pilot初始化workspace检查exit86完整guard，无神经enqueue（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [前缀第三段actual Orin built-not-validated，未推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [前缀第四段actual Orin built-not-validated，未推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [前缀000实际反序列化8输入/10输出ABI，无context或推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [前缀001实际反序列化18输入/51输出ABI，无context或推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [前缀002实际反序列化69输入/26输出ABI，无context或推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [前缀003实际反序列化95输入/37输出ABI，无context或推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [四段实际engine/inspection接入十区域绑定请求，非模型通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Session003全10真实区域源码已生成missing=[]，非目标验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Session003原生共享库234344B链接成功receipt，非D1通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [嵌套缩进严格逐task guarded wrapper 12项软件测试通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [全Session尝试前目标快照：同serial及两Java PID/start，非隔离证明（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [四段build/ABI、全十区域源码/原生链接及首次workspace失败证据](reports/orin_prefix_split_deployment_20261023.md)
- [前缀第二段真实Orin引擎构建成功1.39GB，无输出核验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [历史前缀第三段活动构建快照，已由终态替代，不是模型通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [实际四段源NodeProto/路由/类型独立核验，非数值或目标证明（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [四段实际ONNX/完整权重范围发布，810个dense各一次，尚非数值通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [前缀首段真实Orin引擎构建成功2.46GB，未神经输出核验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [历史前缀第二段活动构建快照，已由终态替代，非整模型验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [审核插件切分/外部流式发布/原切分77项主机回归（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [6/7实际Region合同已核验/物理化，唯一前缀未绑定，无Session（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实Orin控制图61完整GPU用例通过，非VLA模型输出（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [通用E2E驱动及固定normalized主门槛的软件交付（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Orin驱动目标分支已编译，未加载真实Session/未推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实step 42个I/O及来源明确的Scalar布局检查，无推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [前缀I64最终范围证明：197个Slice等价、1个死锥（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0前缀资源停止/控制GPU核验/原生驱动与整链缺口](reports/orin_openpi_integration_checkpoint_20261023.md)
- [前缀003优化阶段触发采样内存保护，仅本组SIGTERM，非模型通过或内核OOM（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Orin输入CPU API原生编译/320完整张量/零Prepare IO，非神经推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [原生输入API/图像分词库实际链接清单与两配置公共消费者（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0前缀001解析失败：TensorRT8.5未导入原生LayerNormalization（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [通用LayerNorm改写/外部元数据不可变：50项主机回归（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [前缀003构建活动快照，非构建或完整模型通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [通用计时内输入API：32用例/320张量及零文件IO，尚未Orin构建（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [四图源dtype驱动通用提案与27项CPU测试，无目标自动激活（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [profile007改写链独立校验/32项测试，目标artifact仍明确unbound（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0 step真机引擎构建成功640MB，未执行神经推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [168个dense节点可见IO/权重格式符合声明，非累加器/数值证明（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [5个实际Region GPU通过128冻结块/89600值全字节一致，非模型E2E（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0真实目标阶段检查点与完整链剩余边界](reports/orin_pi0_target_stage_checkpoint_20261023.md)
- [源dtype推导：6个FLOAT区域与162个BF16核心的显式部署策略（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0 step 38个越界I64常量：1死锥/37切片界限饱和等价（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实schema/Region ID的原生CUDA provider准备与14项CPU测试（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Orin CUDA11.4原生17条编译命令/5个provider通过，GPU未运行（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0 003整图导入通过、RoPE MatMul格式构建失败（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0 004保留源RoPE FP32后，attention格式约束失败（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0 005声明dense FP16输出后，Shuffle格式边界仍失败（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [常量表示/受限静态控制/SinCos共51项主机回归（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [原始RoPE FP32外积节点/完整shape与源autocast关闭边界（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0 step 002：越过原崩溃后静态Slice轴导入限制，exit 1（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0 step 首次真实TRT导入失败：exit -11，原图保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [同图GDB回溯：导入器误读Constant形状后Unsqueeze崩溃（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [通用ONNX常量表示等价改写：11项主机回归（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [RDT完整数据/资源采样/真实恢复联合审计通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [RDT34560张量/15360 measured及replay完整审计（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [RDT正式Orin E2E：3065.625->2975.587ms，1.0303x](reports/orin_rdt_formal_e2e_20261023.md)
- [RDT15360 measured正式pooled CDF](../paper_handoff/edgefm_three_platform_20261026/figures/Orin/orin__20261022-01__formal-cdf-001__e2e-cdf.png)
- [RDT正式pooled CDF PDF](../paper_handoff/edgefm_three_platform_20261026/figures/Orin/orin__20261022-01__formal-cdf-001__e2e-cdf.pdf)
- [RDT十五独立进程CDF](../paper_handoff/edgefm_three_platform_20261026/figures/Orin/orin__20261022-01__formal-cdf-001__e2e-cdf-processes.png)
- [原服务恢复后PID/start/argv/父进程/JAR/config/boot新观察（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实Orin原生ScalarSinCos/BF16数值/两流replay smoke exit0，非模型通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [两模型八份opset17图与精度请求；受检转换/目标执行仍未验证（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [245项主机输入/输出/时序/Sequence/SinCos/opset/精度/谓词回归通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [第五轮replay2304完整张量/全hash一致，1152次/5步/回退0，累计14/15（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [四份FLOAT载体图与carrier-first SinCos候选；不证明硬件计算精度或源等价（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [第五轮共享流2304完整张量/全hash一致，累计13/15（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [当前并行：原生输入、数值Region、受检转换；目标仍RDT独占（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [主任务重编/复验128完整后处理块/89600 F64值字节一致（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [OpenPI F64后处理公式/源码/完整CPU验证与独立来源族（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [51项结构改写及真实大图/源与标量快照一致性验证（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [外部权重结构lowering说明，最新快照修复见revision003](reports/orin_external_sequence_lowering_20261022.md)
- [OpenPI TRT源准备：不降F64、后处理/源shadow与剩余目标门槛](reports/orin_openpi_trt_preparation_20261022.md)
- [ScalarSinCosF64主机编译/ABI/生命周期及未执行的目标smoke](reports/orin_scalar_sincos_plugin_preparation_20261022.md)
- [两个十步源时间序列20480特征CPU字节一致，非CUDA或整模型（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [185项原生类型/完整输出/时序/代码生成/Sequence/SinCos主机回归（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [第四轮replay另2304完整张量/全hash通过，1152次/5步/普通回退0，累计12/15非整组验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0/pi0.5原始F64 producer分区，不降精度；原生执行/目标支持仍未验收](reports/orin_openpi_source_partition_20261022.md)
- [两份真实source分区节点/权重/类型独立静态核验，非数值或目标通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [主机并行后续分工：OpenPI后处理、通用Sequence改写及主任务dtype分区（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [CogACT完整四阶段/CPU参考交付，保留cognition MSE与BF16 Conv失败，非目标通过](reports/orin_cogact_parallel_preparation_20261022.md)
- [CogACT组件通过与完整前缀join失败的分层收据及三方误差（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [主任务独立复验CogACT全部822文件/46.26GB与manifest hash，非数值或目标验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [pi0/pi0.5八张真实CPU ONNX导出、源身份与完整类型合同；非Orin通过](reports/orin_openpi_parallel_preparation_20261022.md)
- [八图/外部权重13.09GB及来源、工具、收据的最终交付manifest（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [主任务独立重哈希全部114个交付文件/报告27.21GB，不是数值或目标验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [通用host F64/I64字节保真与非有限值检查，不是TensorRT/完整模型支持](reports/orin_native_host_wide_types_20261022.md)
- [121项主机类型保真、旧buffer接口兼容、完整输出/时序和代码生成回归通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [用户要求并行后的主机任务启动、独立写入范围和GPU资源边界（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [第四轮共享流另2304完整张量/全hash通过，ordinary符合设计，累计11/15非整组验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [第四轮常规另2304完整张量/全hash通过，无非预期REPLAY，累计10/15非整组验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [第三轮共享流另2304完整张量/全hash通过，ordinary符合设计，累计9/15非整组验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [第三轮常规另2304完整张量/全hash通过，无非预期REPLAY，累计8/15非整组验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [第三轮replay另2304完整张量/全hash通过，1152次/5步/普通回退0，累计7/15非整组验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [第二轮常规另2304完整张量/全hash通过，无非预期REPLAY，累计6/15非整组验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [第二轮replay另2304完整张量/全部hash通过，1152次/5步/普通回退0，累计5/15非整组验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [第二轮共享流2304张量/全部hash通过，按设计ordinary1152次，累计4/15非整组验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Typed审计/等待状态同步后八项看板状态和服务测试通过](evidence/orin-task-board/typed-audit-node-tests-014.xml)
- [Typed审计状态同步后九项桌面/平板/手机浏览器回归通过](evidence/orin-task-board/typed-audit-browser-tests-015.xml)
- [INT32/非dense审计模块、完整数据与精度分开、归档复查及未来门槛](reports/orin_typed_operator_audit_20261022.md)
- [旧Attention真机归档完整786432值/既有精度复查，未新运行模型或计时（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Embedding实际无目标capture，typed审计exit3 pending且不宣称通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [RoPE实际无目标capture，typed审计exit3 pending且不宣称通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [58项typed/原生/旧请求组合回归：真实归档及临时反例，非新板端实验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [RDT首轮完整段/缺失字段状态同步后八项看板回归通过](evidence/orin-task-board/first-round-node-tests-013.xml)
- [RDT首轮状态同步后九项桌面/平板/手机浏览器回归通过](evidence/orin-task-board/first-round-browser-tests-014.xml)
- [新replay首进程2304张量和全部hash通过，1152次/5步/普通fallback0，非整组验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [原生用例进展同步后八项看板状态/服务/动态标题回归通过](evidence/orin-task-board/native-cases-node-tests-012.xml)
- [原生用例进展同步后九项桌面/平板/手机浏览器回归通过](evidence/orin-task-board/native-cases-browser-tests-013.xml)
- [Embedding/RoPE原生用例/请求和非dense审计缺口，非目标结果](reports/orin_embedding_rope_native_preparation_20261022.md)
- [Embedding有界INT32 ABI与完整CPU对照，未目标构建/质量/计时（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Embedding标准板端请求：一个不同指令，原精度映射不变（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [RoPE原payload按hash内联及两个观测完整CPU对照，非目标结果（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [RoPE标准板端请求：三个完整输入、两个真实观测（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [29项原生ABI/磁盘ONNX完整CPU输出/损坏/旧请求回归通过，非dense不得空通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Attention源候选进展同步后八项看板状态/服务/动态标题回归通过](evidence/orin-task-board/attention-source-node-tests-011.xml)
- [Attention源候选进展同步后九项桌面/平板/手机浏览器回归通过](evidence/orin-task-board/attention-source-browser-tests-012.xml)
- [Attention完整prefix源组合：只删除一个零mask Add，未目标构建/质量/E2E验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [已核验Attention配方、原完整source/build/formal/quality的固定来源绑定（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [43项通用组合/来源/不覆盖与旧GEMM/admission回归通过，非板端模型测试（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Attention整图源候选准备、通用设计边界、完整证据和后续目标门槛](reports/orin_attention_integration_preparation_20261022.md)
- [完成段资源证据同步后八项看板状态/服务/动态标题回归通过](evidence/orin-task-board/completed-resource-node-tests-010.xml)
- [完成段资源证据同步后九项桌面/平板/手机浏览器回归通过](evidence/orin-task-board/completed-resource-browser-tests-011.xml)
- [十四个完成段43070资源行/系统保护线核验，含两行旧字段缺失，非全窗口验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [明确十四个已完成目录、固定源/helper/设备/服务窗口的资源审计请求（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [RDT完成段资源包络、replay缺失字段/原失败保留、采样/权限限制及复跑命令](reports/orin_rdt_completed_resources_20261022.md)
- [13项资源回归：首轮9421真实行、缺失不补零、坏单位/保护线/owner仍拒绝（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [新共享流完整进程2304张量/全部哈希字节一致，仅完成段范围，非整组验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [源资产进展同步后八项看板状态/服务/动态标题回归通过](evidence/orin-task-board/source-contract-node-tests-009.xml)
- [源资产进展同步后九项浏览器回归通过，覆盖桌面/平板/手机](evidence/orin-task-board/source-contract-browser-tests-010.xml)
- [五模型D0源资产评估：仅主机元数据/数据验证，非目标部署验收](reports/orin_source_asset_assessment_20261022.md)
- [五个原始数据包路径、固定manifest/checkpoint pin及真实IR步数（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [4581文件/179存储样本/703完整参考对离线验证；原源未改且未新推理（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [分离数据包/H20 raw profile、dtype/shape/RNG/参数来源，不引入性能数据（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [七项源合同回归通过：拒绝假板端通过、dtype/文本长度篡改，读取真实IR步数（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [八项看板状态/服务/动态标题回归：隔离fixture允许2/5进展但不完成项目](evidence/orin-task-board/managed-node-tests-008.xml)
- [九项当前看板浏览器回归：pilot/formal使用隔离fixture，未修改真实验收数据](evidence/orin-task-board/managed-audit-board-green-009.xml)
- [新常规首进程2304张量及全部哈希独立核验通过，仅1/15范围，非整组验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [91项完成段/窗口/恢复/原完整动作与CDF后处理组合回归通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [动态已核验计数标题及状态切换后的九项浏览器回归通过](evidence/orin-task-board/managed-audit-board-green-008.xml)
- [部分交付索引：正式模型仍为1/5，包含已验证主表、负结果与缺口](reports/edgefm_orin_experiment_delivery.md)
- [新窗口完整数据 / 恢复最终审计步骤，非已完成结果](reports/orin_managed_campaign_audit_20261022.md)
- [真实当前采样冻结绑定已核验：明确不证明数据完成或进程活性（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [只读恢复观测真实返回pending：当前采样中，未启动模型或改服务（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [86 项新窗口 / 恢复 / 冻结 / 完整动作与CDF后处理回归通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [RDT 采集和两服务恢复守护运行快照；须同时核验实际PID，不当作完成证据（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [新服务窗口三模式完整 raw pilot：120 张量字节一致，实际 replay 无回退（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [全新 RDT 正式十五进程协议、服务窗与源 hash；不合并旧数据（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [新采样开始前的旧中断 run 快照：11 个完整进程和 147 条部分测量，非新进度](evidence/orin-task-board/pre-rdt-fresh-campaign-state-20261022.json)
- [36 项服务范围 / 退出恢复 / 哈希 / 完整模型和正式协议冻结回归通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [RDT 明确授权的两服务临时窗口、首次保护恢复和新 pilot 门槛](reports/orin_rdt_resource_requalification_20261022.md)
- [用户直接确认的两个精确程序临时停机范围，非全平台清理授权（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [两服务 PID 0 / Exited，配置和其他受管程序不变；需在窗口结束恢复（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [RDT 停两服务后只读资源快照：可用约 43.49 GiB，非精确模型内存（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [22 项严格 PID / 程序范围 / 终态 / 恢复纯测试通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [最终 Norm 负结果看板九项浏览器回归通过：桌面 / 平板 / 390 / 320px](evidence/orin-task-board/norm-raw-final-board-green-006.xml)
- [Norm 正式 10 进程 / 10240 measured / 23040 张量完整审计：负收益不抹除（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Norm 完整模型 E2E 对照：主质量通过，但候选慢 6.43%](reports/orin_operator_layernorm_raw_e2e_20261022.md)
- [Norm 完整原始输入 E2E 全尾部 CDF：含所有慢样本](../paper_handoff/edgefm_three_platform_20261026/figures/Orin/orin__20261022-00__norm-raw-e2e-pair-003__formal-cdf-001__raw-model-pair-cdf.png)
- [Norm 正式窗口资源：采样内存 / 温度及未确认 CV -256 读数均保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [141 项算子 / 完整回集成 / 原始正式数据 / 未确认温度回归通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [139 项全部算子 / 回集成组合回归通过，非全源码测试替代（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [四类算子当前实测汇总，保留负收益与融合精度限制；非 E2E 主表](reports/orin_operator_ablation_results_20261022.md)
- [32 项原始输入 / 输出口径 / 缺行拒绝 / owned-child / 多输入回归通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Norm 原始输入 pilot：80 完整张量与各自参考字节一致；负收益保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Norm 两端 ordinary / 五进程对 / 128+1024 冻结正式协议与安全工具 hash（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Norm 正式采集终态：全部进程 exit 0，采集器及 workers 已退出（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Attention 算子配对时延、完整 CDF、精度和首轮短记录限制](reports/orin_operator_attention_pair_20261022.md)
- [Attention 原始全部尾部 CDF：不是 VLA E2E](../paper_handoff/edgefm_three_platform_20261026/figures/Orin/orin__20261022-00__operator-attention-pair-001__cdf-002__attention-operator-cdf.png)
- [用户确认标准化 action 主门槛，native 误差单列；历史收据不改（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Attention 20 进程 / 20800 行完整审计，非 VLA E2E（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [首轮 1023 行拒绝证据与保留的完整 20479 行，未放宽门槛（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [多输入计时、正零证明、实际混合精度及单输入兼容回归：38 passed（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Attention 三输入冻结五进程对协议，所有输入显式加载（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Attention 四次 pilot / 512 条记录及完整最后输出审计（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Attention 配对计时进行中快照：11/20 完成，非最终审计（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Attention 真实观测的三输入完整请求，缩放及混合精度配置保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Attention 真机 786432 值字节一致与 FP16 QK/PV、FP32 Softmax 证据（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [多输入请求、正零 elision、保留精度约束及旧格式兼容回归：28 passed（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [看板长字段换行、桌面 / 平板 / 390 / 320px 浏览器回归：9 passed](evidence/orin-task-board/state-board-green-004.xml)
- [保留的 320px 长字段溢出失败回归，修复前证据](evidence/orin-task-board/state-long-key-red-003.xml)
- [状态配置对照 20 进程 / 20480 行审计：无均值加速，融合实际精度限制保留（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [状态配置对照全部 20480 条原始记录，不移除慢样本（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [状态配置口径算子完整 CDF，非 VLA E2E / 非实际精度认证](../paper_handoff/edgefm_three_platform_20261026/figures/Orin/orin__20261022-00__operator-state-pair-001__paired-timing-audit-001__cdf__state-operator-cdf.png)
- [VLA 状态投影配置对照报告：负收益、代价及融合证据限制](reports/orin_operator_state_pair_20261022.md)
- [状态输入 / 数值 / 融合边界 / 计时范围组合回归：50 passed（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [状态投影计时进行中快照：18/20 完成，尚非最终审计（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [状态投影冻结配对协议：仅同 OBEY 配置口径，不宣称融合实际精度相同（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [状态投影四次 pilot / 512 条导出记录及最后输出审计，不是主结果（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [状态投影 16 输入数值通过；仅配置精度口径，融合内部仍不可视（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [状态投影两构建 / 32 个 native 完整输出捕获，非计时（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [输入布局、融合显式边界、数值及旧 GEMM 兼容回归：32 passed（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [VLA 状态投影真机请求：16 份不同状态、原 FP16 投影配置（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [通用记录布局、路径边界、旧 GEMM 兼容及 admission 回归：25 passed（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [相同源图重建的 34 个完整动作对：标准化通过，native 同阈值失败（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [GEMM 对新重建控制的完整动作比较：标准化 MSE 1.3759e-5 仍失败（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整回集成 / 同源控制 / 匹配数据与失败门槛回归：43 passed（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [原 ONNX / precision profile 字节不变的重建控制来源审计（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [相同源图重建的真机编译和资源保护证据，非完整动作验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [GEMM 整模型 MSE 失败与未改图重建控制，未发布 E2E 收益](reports/orin_operator_gemm_integration_20261022.md)
- [图嵌入、独立组成、admission 与真实完整动作门槛回归：38 passed（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [相同 ONNX / precision profile 的重建控制运行快照，非质量通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [GEMM 全模型 34 个动作对：标准化 / native MSE 均失败，未进入 E2E（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [GEMM 完整 prefix 真机编译与资源保护记录，非整模型通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [重新读取完整图 / 精度约束 / 43 个绑定的独立配对审计（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整图独立审计、splice 和低内存拒绝回归：15 passed（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [GEMM 完整 prefix 构建快照：admission 15.1 GiB，非编译通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [GEMM 单算子候选嵌回完整 prefix 的结构 / 原权重 / 精度映射证明，非真机通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整图 splice 和原 MatMul 候选回归：27 passed（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [GEMM 20 进程 / 20480 条记录计时审计：resident 1.448x，IO compute 略退化（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [GEMM 全部 20480 条原始计时记录，不剔除尾部（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [GEMM 算子两种边界的完整 CDF，非 VLA E2E](../paper_handoff/edgefm_three_platform_20261026/figures/Orin/orin__20261022-00__operator-matmul-pair-001__paired-timing-audit-002__cdf__gemm-operator-cdf.png)
- [GEMM 真机配对、质量、成本及边界相关负收益报告](reports/orin_operator_gemm_pair_20261022.md)
- [GEMM / 通用捕获 / 精度 ABI / 计时参考与旧数据兼容回归：47 passed（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [GEMM 微基准采集中快照：12/20 完成，非最终结果（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [GEMM 冻结微基准协议：两种边界各五组独立进程对，不删慢样本（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [GEMM 四次计时 pilot / 512 条记录及完整输出审计，不是主结果（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [GEMM 两份真实输入 / 113280 值真机质量与 FP16 数据合同审计（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [GEMM 真机双构建及四个完整输出捕获，非计时（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [GEMM / 通用请求 / 精度 ABI / 计时回归：39 passed（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [GEMM 通用真机请求：两份真实输入、原 FP16 请求，无输出精度覆盖（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [通用算子请求 / admission / owned-child 回归：15 passed（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [全模型 34 个动作对：标准化通过 / native 同阈值失败，非正式 E2E（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [LayerNorm 完整七 engine 回集成 pilot 与保留的 native 误差](reports/orin_operator_layernorm_integration_20261022.md)
- [完整配对 / 来源 / admission / owned-cleanup 回归：22 passed（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [完整 prefix 的 Norm-only 来源和精度审计，非整模型通过（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [Norm 回集成来源审计回归：8 passed（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [LayerNorm 20 个独立进程 / 20480 条记录计时审计：非 VLA E2E（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [LayerNorm 两种边界的全部未裁剪计时 CSV（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [LayerNorm 原语 CDF：完整尾部，不是 VLA 端到端曲线](../paper_handoff/edgefm_three_platform_20261026/figures/Orin/orin__20261022-00__operator-layernorm-pair-001__paired-timing-audit-001__cdf__layernorm-operator-cdf.png)
- [20 项原语捕获 / 计时范围 / 全记录与不利尾部检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [LayerNorm Orin 新配对完整输出门槛：已通过、未计时（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [LayerNorm Orin 小内存配对构建 / 输出核验与边界](reports/orin_operator_layernorm_pair_20261022.md)
- [中断后 11 个完整进程的文件 / 全部动作核验：非正式验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [第 12 个部分进程已写入输出核验：不计完整进程（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [25 项安全 / 中断范围 / 文件哈希回归（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [RDT 仅终止本任务 worker 的资源安全中断回执（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [RDT 资源安全中断：保留范围与未完成验收](reports/orin_rdt_safety_interruption_20261022.md)
- [RDT 第 11 进程完整退出 / 第 12 replay 进程身份快照](evidence/orin-task-board/checkpoint-20261021214032352.json)
- [05:27 RDT 只读资源快照：余量约 1.19 GiB / swap 满（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [9 项三快照 / 原始哈希 / 资源解析检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [RDT 第 10 进程完整退出 / 第 11 共享流进程身份快照](evidence/orin-task-board/checkpoint-20261021204044022.json)
- [04:25 RDT 只读资源复查：第二个冻结快照（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [8 项双快照 / 原始哈希 / 资源解析检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [RDT 前三轮 9 个进程完整退出 / 第 10 进程身份快照](evidence/orin-task-board/checkpoint-20261021194223159.json)
- [已保留的网页搜索计数假设失败：非模型实验失败](evidence/orin-task-board/browser-tests-resource-warning-failed-001.xml)
- [RDT 只读资源快照：原始日志 / 进程状态与哈希（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [RDT 系统内存压力与 owner 覆盖限制](reports/orin_rdt_resource_observation_20261022.md)
- [6 项资源解析 / 有界只读 / 原始哈希检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [RDT 第 8 进程完整退出 / 第 9 共享流进程启动身份快照](evidence/orin-task-board/checkpoint-20261021184241955.json)
- [RDT 第 7 进程完整退出 / 第 8 进程启动身份快照](evidence/orin-task-board/checkpoint-20261021174346579.json)
- [RDT 前两轮 6 个进程退出 / 第 7 进程身份核验快照](evidence/orin-task-board/checkpoint-20261021164607427.json)
- [RDT 第五进程退出 / 第六进程身份核验的冻结快照](evidence/orin-task-board/checkpoint-20261021154625053.json)
- [真实 VLA 状态填充 / 投影配对：仅主机准备（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [39 项真实状态用例与 12 份配置 / 构建器检查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [143 项六类算子与构建器组合回归（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实语言 Embedding 配对：单个真实指令、仅主机准备（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [37 项 Embedding / INT32 索引边界与完整权重重算（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [104 项五类真实算子 / 静态索引回归（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [135 项真实算子与独立构建器组合回归（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [31 项 v3 构建器主机检查；不是板端编译结果（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实 Attention Core 配对：仅主机准备（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [67 项 Attention / GEMM / RoPE / Norm / 静态索引组合回归（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实 FP16 请求的 GEMM 配对：仅主机准备（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [48 项 LayerNorm / RoPE / GEMM / 静态索引组合回归（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [测试夹具外部权重独立副本修复记录（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [真实 RoPE/cache 前后配对：仅主机准备（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [21 项真实 RoPE / 静态索引回归（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [15 项静态索引依赖 / 标签 / 形状边界（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [最新全量主机回归：2639 passed / 74 skipped（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [算子消融准备及真实输入边界](reports/orin_operator_ablation_preparation_20261021.md)
- [真实 LayerNorm 前后配对：未做板端性能测量（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [68 项真实用例 / 重写 / 签名回归（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [网页看板：6 项状态与只读服务测试](evidence/orin-task-board/node-tests.xml)
- [网页看板：桌面 / 手机及采集终态浏览器测试](evidence/orin-task-board/browser-tests.xml)
- [详细状态与历史证据](edgefm_orin_goal_status.md)
- [Orin 环境检查](reports/orin_environment_20261020.md)
- [SmolVLA 正式 E2E / CDF 报告](reports/orin_smolvla_formal_e2e_20261021.md)
- [SmolVLA soak 与资源边界](reports/orin_smolvla_soak_memory_20261021.md)
- [SmolVLA 正式 pooled CDF（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [SmolVLA 独立进程 CDF（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [RDT raw E2E pilot 与采样入口](reports/orin_rdt_native_raw_e2e_20261021.md)
- [RDT pilot 完整动作独立审计（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [RDT 恢复、审计与绘图命令（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [37 项采集 / 审计 / 绘图回归（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [七个已验证源图的算子逻辑清单（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [24 项算子签名 / 架构回归（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [主机全量回归：2630 passed / 74 skipped（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [论文主张与证据缺口](reports/orin_claim_gap_20261021.md)
- [当前采样进程 / CSV 只读核验](evidence/orin-task-board/checkpoint-20261022220103106.json)
- [0.8B原生正式固定协议与全部输入/engine/质量pins（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [0.8B正式10进程有界队列启动，不临停服务（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [正式采样只读最新进度，非整批验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [0.8B原生正式固定协议与全部输入/engine/质量pins（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [0.8B正式10进程有界队列启动，不临停服务（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [正式采样只读最新进度，非整批验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [0.8B原生正式固定协议与全部输入/engine/质量pins（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [0.8B正式10进程有界队列启动，不临停服务（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [正式采样只读最新进度，非整批验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [0.8B原生正式固定协议与全部输入/engine/质量pins（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [0.8B正式10进程有界队列启动，不临停服务（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [正式采样只读最新进度，非整批验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [0.8B原生正式固定协议与全部输入/engine/质量pins（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [0.8B正式10进程有界队列启动，不临停服务（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [正式采样只读最新进度，非整批验收（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [最终31项验收及发布后复查（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [0.8B完整实测、资源和目标终态归档（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [最终Orin已选范围报告与主表](reports/orin_selected_goal_acceptance_20261026.md)
- [TTFT及完整16token实际CDF视觉核验（原件已退役）](reports/local_bulk_cleanup_20261026.md)
- [H20/J6M/Orin最终论文数据统一汇总](reports/edgefm_h20_j6m_orin_paper_summary_20261026.md)
- [三平台汇总数值、链接与CDF哈希复查](evidence/paper-summary-20261026/verification-002.json)

## 同步规则

1. 每次阶段进展、状态变化、新证据、失败或阻塞，更新同一 JSON 状态，并同步本看板。
2. 每次 Goal 恢复先读看板和详细状态，核验实际进程后再填进度，不根据旧 PID 或意图推断。
3. 同步核验时间、当前位置、相应任务、下一道门槛、计数与证据入口。
4. pilot、编译成功、静态清单和运行中数据不得写成正式通过；负收益和失败保留。
5. 用户无需再次提醒。未运行网页服务时，更新状态后运行 `node tools/orin-task-board/sync.mjs`。

## 最近变更

| 时间 | 进展 | 说明 |
|---|---|---|
| 2026/10/27 10:35:27 | 运行时和交付已正常推送，独立Git检出复核通过 | a61917d/6214dbd已推送既有origin/codex/vlaforge-paper-artifact，远端与主目录0/0且无force push。385旧文件及唯一旧golden差异完整保存于stash 6b6e03fc后再快进，不复活已重构adapters；原ZIP从备份恢复且SHA8d70859f不变。最初普通archive在旧父仓库下的3个架构检查因读父Git索引失败，负记录保留；真正独立Git clone的同套3060 Python/57skip/17deselected、16 Node/11浏览器/272文件282链接全通过，原检查不改。Cargo未安装，硬件/Rust编译不复测，平台数字与验收门槛不变。 |
| 2026/10/27 10:22:35 | 运行时代码已提交，交付文件通过检查并整理版本化证据 | a61917d包含通用TRT/native输入/region-chain/ONNX适配；wheel缺少CUDA插件和Rust源码的回归先失败后4项通过。Python3060/57skip/17deselected/6subtests，CPU CTest14，Node16与浏览器11通过；没有重跑任何论文硬件实验。45小记录逐字节保留，13张原图按原路径身份/原始与打包SHA匹配，不替换缺失SmolVLA原件；看板展示由metadata声明的实际RDT曲线并正确显示已完成。主目录旧副本需先保存到本机Git stash，再快进合并已有远端重构，防止复活旧adapters。实验model/run记录和177行汇总表格不改。 |
| 2026/10/26 11:43:28 | 本机23路径/NAS196冷模型退役，交接包与源码保存复查通过 | 用户明确确认表中八类删除并清理.codex可清项。先保存四组源码状态，Horizon独立Git位于待删副本内的保护门拒绝发生在删除前，改为完整bundle后核验；4项合成安全测试通过。清理本机1711690元数据节点/369.914902GiB已分配块，无跳过；原case/实验根不再保留，不再宣称旧105来源原机链接有效。NAS196路径/797.968254GiB逐SHA删除，终态核对通过，原始格式未选入。f801活动源码/看板保留约132MiB；包272文件/282链接及ZIP SHA不变。两份BEVworktree仅定点移除登记，其他主Git/环境/Codex会话/附件/登录不动，实验计数及结论未改。 |
| 2026/10/26 11:14:21 | 本地磁盘盘点完成，NAS冷模型清理进行中 | 本地10:55快照总916GiB、已用约656GiB、可用约213GiB；本轮只读，没有追加删除候选。OpenPI四个权重/转换目录约36GiB、OpenVLA缓存约15GiB；旧EdgeFM约96GiB需分离证据，感知case/SDK/Git/当前工作树不整删。Orin追加计划0文件；NAS仅本任务runs下196个已确认冷模型文件逐项hash/unlink中，原始测量与共享模型库不在计划，不提前记为完成或本地空间回收。实验验收和采样计数不变。 |
| 2026/10/26 10:28:21 | 自包含交接包与模型产物清理完成 | 包内本机/NAS/连接引用已去除，独立解压及282本地链接校验通过。随后删除214本机/48Orin选定冷模型产物，回收129.40/42.63GiB；105论文来源哈希不变，原始实测/质量/CDF保留。环境、共享NAS、其他仓库和未完成J6M未动；Orin原管理器/两Java仍Running。 |
| 2026/10/26 09:52:42 | 论文交接包完成，明确baseline和EdgeFM实测路径 | H20 session-required为EdgeFM native Session；Orin replay/Arena为框架优化路径，baseline是同一Session普通TRT执行而非独立vendor baseline。新交接文件夹及约12MB ZIP已做独立解压移位自检，272个内容文件、282个本地链接通过；不含权重/engine/HBM，巨大审计明确节选，原实验数值和验收状态不变。 |
| 2026/10/26 09:36:40 | 三平台论文实验数据已统一汇总 | 新增一份Markdown统一H20/J6M/Orin主表、CDF、质量、算子/回集成、资源及未测边界；没有重跑实验，不更改已验收计数或结论。 |
| 2026/10/26 07:39:06 | 五VLA与0.8B已选范围最终验收完成 | 五VLA与Qwen0.8B完整实测及31项验收完成；2B用户取消。原始样本/CDF/质量/资源/服务恢复已核验；H20差异、RSS缺口、负收益及论文强主张边界保留。Qwen TTFT333.350ms、full161586.510ms、10.085tokens/s；原管理器和两Java服务恢复核验通过。 |
| 2026/10/25 23:08:15 | 0.8B quality005在线native文本及16观测主Gate通过，正式未启动 | quality005主Gate通过：16真RGB+在线native Rust tokenizer，off/verified各两周期32请求，共64请求/192完整tokens-logits-IDs文件，全部跨模式/重复byteexact、MSE0/cos1、原327IDs一致；varying reset及两个独立进程通过，未测并发Session，终态空/服务不变。CPU/H20补充仅7/16 tokens同、9/16异，logits worstMSE8.65417583235/mincos0.6973061539单列，不影响同Orin主Gate，不触发重建/额外gate。004单观测及002 observer负例保留。正式5×(128+1024)×2预计约5.2h但尚未启动；main冻结plan、agent2准备worker/tests，后续main接管文档。2B取消、五VLA不变、Goal未完成。 第二周期median off TTFT332.094672/full16 1588.910768ms，verified333.960528/1587.708448ms仅pilot。 |
| 2026/10/25 23:05:24 | 004 prepared-text单观测resident pilot独立核验通过，002 observer负例保留 | 004离线独立核验off/verified各2call单观测，8完整输出3973632B全byteexact/finite/33counts有效，guard0/终态空/服务不变/maps无Python或Torch；79archive源与attempt004原源/build/binary/DSO pins绑定，marker4/output_49数组index48。002因source与canonical region顺序不同observer失败，保留负例非模型数值失败；004归档7回归通过 第二call full16 off1581.044512/verified1579.464192ms，TTFT330.455776/329.939616ms仅pilot。单观测MSE0/cos1不作16观测主Gate或正式CDF。 |
| 2026/10/25 22:22:37 | 补齐常驻编排控制真机验证，生成完整双配置IR/Plan | 两无参数控制只执行KV padding/greedy历史/计数与finite判定，真机19输出2101285元素byteexact，神经参数/精度不变。首失败为8.5CLI不接受bool IOFormats，已保留；新实际元数据严格核验源dtype/layout。完整程序5常驻Region/33调用包含17原神经调用与16控制，off/verified语义和IO摘要相同且prefix缓存关闭。可选完成观察在同步成功后借用输出，真实CPU九次回调和sync失败抑制测试通过，默认codegen不改。尚未完整Orin Session编译/执行，首pilot prepared IDs非最终raw-text性能；主Gate/正式仍待，2B取消不再扩展，五VLA数据不变。 |
| 2026/10/25 21:44:14 | 0.8B新前缀完整单观测计算闭合，转最后常驻与正式收尾 | 实际Orin CPP RGB像素→两修复prefix→15decoder均真机执行，17模型调用/835输出记录完整、735decoder输出112657920元素finite/控制与state carries验证；16token文件SHAf74f95a8与原CPU/H20冻结参考exact。source logits MSE0.00471986/cos0.99975058、prefix误差及同实际prefix portable tokens不同均保留为补充，不宣称源无损/主Gate。两engine a9aff933/eed2a800通过独立pins/空cgroup审计，两Java临停1816.855/1804.498秒后原Supervisor恢复4062194/4062004，其他服务不变。2B用户取消，仅0.8Bresident/reset隔离/16观测同平台质量/正式指标与交付待完成。14C++、59链路、40旧host pipeline、14看板单测及11浏览器检查通过，Goal仍active。 |
| 2026/10/25 21:27:56 | 用户取消2B，仅完整收尾0.8B后结束Goal | 根据用户最新明确指令，2B改为用户取消，不再构建/运行/采样、不是tested/passed/unsupported，历史主机准备保留但不再作为完成条件。0.8B提升为最后必须闭合P0任务：完整native链、16观测同平台质量、TTFT/16token/CDF/内存及最终交付均须完成，原质量门槛/5进程128+1024协议不变。当前尚不标整个Goal完成，五VLA验收证据不变。 |
| 2026/10/25 21:18:45 | 通用C++16图预处理Orin实测闭合，首prefix engine生成 | 同host证明源码在Orin纯CPU编译执行，16完整FLOAT[1200,1536]与原冻结输入byteexact/maxabs0，300s/1GiB/单CPU cgroup记录compiler PID/start/argv与输出hash，现已terminal/empty，无模型/GPU/服务控制。prefix producer已生成a9aff933/2109749460B，原生912.397秒/12126层/unmatched0；consumer4037206解析中，两Java恢复待构建终态。预处理组件完成不冒充resident在线集成/完整模型主Gate/正式指标，五VLA数据不变。 |
| 2026/10/25 21:08:35 | 并行补齐通用C++patchify，16完整像素张量host逐字节验证 | common host preprocessing新增声明式PatchifySpec/F32变换，保留普通scale/standardize，显式frame/channel/矩形patch/merge/temporal padding，无模型名分支；尺寸溢出/shape/count/alias/nonfinite在写前拒绝。按现有官方processor融合顺序，原16RGB到FLOAT[1200,1536]全部byteexact/maxabs0，14CPU CTests及59链路/receipt Python tests通过。只代表host准备，目标CPP/online Session/正式未验证。prefix producer12126层全部精度名匹配且building，consumer待，两原Java lease恢复待终态，五VLA数据不变。 |
| 2026/10/25 20:58:59 | 转入修复后完整prefix双阶段真机构建 | 旧native prefix不再用于后续fresh主张；两原opset17图只读Exp8+10修复/all50CPU输出前后exact，原参数/ABI/79+108BF16MatMul不变。图/profile/3.42GB原共享权重/工具均重核后新20窗口实际启动，producer4013285在parsing，consumer顺序待。24GiB准入按需只停两Java至27647052KiB，router/compotask原出生及其他服务不变；4GiB底线/原管理器恢复仍执行，待终态。此为部署推进，非fresh/resident/主Gate/正式通过，五VLA数据不变。 |
| 2026/10/25 20:51:05 | 完整15步decoder连续核验闭合，16tokens同输入接续一致 | 735输出112657920元素独立finite，48state carries及每步token/counter边实际验证，完整16tokens与同旧native prefix的portable接续一致。逐步teacher-forced logits MSE0.00049079..0.00144257/min cos0.99996069保留为源补充差异，非源无损或同Orin主Gate。所有unit/lease闭合、data343.932秒后原管理器恢复/其他3项原出生不变。119关联host/13看板tests通过；转已CPU验证修复prefix真机、fresh/resident/16观测primary/正式和2B，不用文件链耗时作E2E，五VLA验收不变。 |
| 2026/10/25 20:45:32 | 完整decoder两类状态输入49输出实测，启动15步连续核验 | 全部49输出7510528元素finite，旧native prefix同输入logits MSE2.463→0.00090755/cos0.92747→0.99997454；有效CPU state同engine logits MSE0.00098758/cos0.99997705/首token11952，首recurrent cos0.999999099。仅schema修复、精度不变，源误差单列补充、不重定义主Gate。两单步unit已闭合/程序未变，15实际状态调用启动，fresh/resident/正式和2B仍待。 |
| 2026/10/25 20:43:02 | 完整schema修复decoder已生成并完成原服务恢复 | 99317ab3/4031068039B，原生311.73秒，5695层全部约束匹配，min8811088KiB/无guard。独立原始assets/DSO/授权/SDK hashes重核，两cgroup空，唯一data临停343.932秒后恢复3985484/start41405109，其他3项原出生不变。当前已启动单步51真实输入/49全输出诊断；不是完整fresh模型/主Gate/性能，后续15步与resident/正式仍待。119关联host tests和13看板tests通过，五VLA验收数据不变。 |
| 2026/10/25 20:36:06 | 完整首mixer实测修复，启动完整decoder19目标构建 | 原130节点/3真实输入/3原输出mixer仅改schema后实际完成，投影cos0.841550016→0.999996082/MSE1.01e-8，conv exact/recurrent cos0.999999564，无精度变化。完整decoder19启动，24GiB准入只按需停data至28061568KiB，其他原服务未停；4GiB运行底线/原管理器恢复合同不变，待终态审计。78关联host tests通过；非完整VLM/同平台主Gate/正式性能验收，五VLA数据不变。 |
| 2026/10/25 20:35:00 | 实测ReduceMean错轴边界，既有schema转换恢复norm | 原opset18norm engine将[-1]归约的16x1变为1x1，ABI在推理前拒绝。既有通用18to17转换后19实际输出shape/finite/诊断均通过，末端norm byteexact；core观察另有tactic失败，suite保持partial。完整decoder转换79轴/18Split/12ScatterElements，参数/IO/精度映射不变、49输出x15CPU步exact；新增none/add/mul ScatterElements回归先3红后35全绿，min/max仍拒绝。原mixer首staging路径错误已保留并纠正，不计模型失败；target复核推进，五VLA不变，Qwen仍无主Gate/正式。 |
| 2026/10/25 20:26:03 | 独立投影未复现大偏差，转上游norm/core原图定位 | 原投影Transpose/BF16MatMul/Round/真实残差Add共4节点，2实际输出maxabs5.96e-8/MSE3.47e-18，无服务控制且unit闭合。并非修改生产builder空约束验证：保留真实原残差Add及其F32约束。两个新声明式cut保留原NodeProto/全部精度，父49CPU输出与参考exact，norm19节点19输出/core66节点68输出实际启动。非完整模型/主Gate/性能验收，不改五VLA结论。 |
| 2026/10/25 20:10:00 | 排除旧cache单因，原始三输出mixer实测缩小残余误差范围 | 同17engine用已验证CPU state仍logits MSE2.41403/cos0.92880，49finite/首recurrent cos0.9999991。130原节点/42172416冻结参数字节的原3输入3输出mixer真机成功，状态近参考而投影输出MSE0.00217461/cos0.84155，定位残差/MLP之前，不直接判MatMul错误。接下来用精确CPU中间输入与冻结权重核验单独out-projection，再norm/core；不重建同模型碰运气/不放宽精度或主Gate。五VLA不变，full/fresh/resident/正式/2B仍未完成。 |
| 2026/10/25 19:58:09 | 15步独立核验/排除query缩放假设，观察边界失败归档并释放台架 | 17全部15调用735输出112657920元素finite，实际carry/counter/token边重核，源整体logit/token仍不同，不作完整模型/主Gate/正式通过。query原3节点与折叠1节点实际均byteexact MSE0/cos1，假设排除。3个侵入mixer观察图虽selector全部匹配，但原FP16Shuffle/publicFP32边界无tactic，保留负证据不放宽精度、不判原模型unsupported。所有unit空/无lease/管理器程序未变；下一步有效CPU前缀state及原3个输出/最小兼容节点对照，prefixCPU修复ready。五VLA与完整Goal范围不变。 |
| 2026/10/25 19:24:12 | 完整decoder恢复finite和首recurrent，残余输出偏差转query缩放微图定位 | 17engine12b873c6/4030963431B真实构建/unmatched0，独立原程序/pins/cgroup闭合，无临停guard。49输出finite，首recurrent MSE2.65e-9/cos0.99999869但logits MSE2.46307/cos0.92747；15实际状态调用均finite、source token仍不同，不宣称模型/主Gate/正式。prefix8+10Exp修复各50输出CPU前后exact准备，尚无新目标prefix；真实query缩放Pow/Reciprocal三节点只读对照launched，原精度/真实数据不变。所有负证据及五VLA验收保留，fresh/resident/正式与2B未完成。 |
| 2026/10/25 18:16:25 | 静态Exp路径定位并修复4真机微图，完整decoder CPU验证完成 | 输入echo/参数Cast byteexact，首错静态Exp、Softplus近参考；官方stock同engine同错，非自写raw runner独有。现有库只读折叠4case全部finite/maxabs5.96e-8，通用工具111tests，原参数/动态数据/18FP16dense13FLOATdense187BF16不改；完整15步49输出CPU前后byteexact。16仅18不再运行的源层selector未匹配而严格拒绝，独立pins/cgroup/原服务闭合、无临停；17不改graph/weights，只显式投影到FLOAT字面量，全部动态约束保留。无完整目标/主Gate/正式通过，五VLA不变。 |
| 2026/10/25 16:59:53 | 9节点真实微图复现大误差，修正设备序号后复用引擎续跑 | 两处各9原节点/真实权重/实际源激活，完整49parent/子图CPU byteexact。首policy13真机输出多为0/1，MSE0.20218/maxabs0.99964；已不需整个模型复现局部误差。case index错用作CUDA ordinal已实证修正，10测试，旧unit空/业务不变，非硬件故障；新scope无服务控制、明确GPU0/复用两引擎，verified live PID3700169/3of4cases，600s/16GiB4GiB守护。无完整模型/主Gate/正式数据，五VLA及14负结果不变。 |
| 2026/10/25 16:31:04 | 受控约束候选真机非有限被拒绝，全部资源/服务已闭合 | 14真实推理写49输出，32输出3219968NaN、首layer8 recurrent/全部logits；源同输入首layer0 cos0.42178→0.998716/MSE2.5845e-6仅局部改善。不替换或过滤NaN、不跑该候选15步/CDF、不宣称修复；严格核验失败保留，worker002仅完整失败审计。所有unit空，data唯一临停344.026676224秒原Supervisor恢复3647862、其他程序/config不变。13有限15调用735输出和源差异保留，转最小算子边界定位；同Orin主Gate不变且仍未测，fresh/resident/正式与2B未完，五VLA不变。 |
| 2026/10/25 16:19:02 | 54约束受控版本实际构建，CPU代理未确认误差根因 | 相同4215节点/权重/187BF16/18FP16dense/13FLOATdense，原义务全保留仅补54FLOAT nonlinear，真实解析5695层/unresolved0并building，恢复受独立守护。98tests通过；CPU非线性/conv半精度代理对首native recurrent cos仍约0.42178，未复现误差因果，不全FP32/不放宽阈值。13重复普通首步49输出byteexact仅重复性、735finite与源负证据保留，无主Gate/正式通过，五VLA不变。 |
| 2026/10/25 16:10:15 | 15步实际状态链完成但源差异明确，定位缺失非线性精度合同 | 735完整输出112657920元素finite，state carry/token/counter独立核对、unit空/管理器业务状态不变。源与同actual prefix portable tokens均不同；同输入首logits MSE1.84037/cos0.94816、首recurrent cos0.42178，保留补充负结果不改主Gate。36Exp+18Softplus缺FLOAT约束，通用规则red→98tests；14同图/权重/BF16与dense策略只补54约束，根因/数值修复待实测；非wholeFP32或正式pair，五VLA不变。 |
| 2026/10/25 15:57:07 | 新decoder真机49输出完整finite，15步复用诊断已启动 | 独立7510528元素/49输出重核finite，logits[1,1,248320]/argmax11952，实际I/O和checked integer codec匹配；unit空/管理器程序未变。15通用carry/argmax/control测试通过，15实际调用每次49输出诊断运行中，仍用归档真实prefix，不是fresh全链/resident/E2E/主Gate或正式指标，五VLA不变。 |
| 2026/10/25 15:50:10 | 单步TRT engine真实生成并独立闭合，开始全输出诊断 | 4031019296B/SHA cc4594e9/exit0，wrapper314.9819秒/compute318.4226秒、min12833856KiB、无guard无临停；所有source/profile/weights/tools/权限pins及原程序births/config不变、cgroup空。SDK权重警告和CPU误差保留，51实际prefix输入准备，49完整输出诊断下一步；非完整16token模型/主Gate/正式指标，五VLA不变。 |
| 2026/10/25 15:46:48 | 修正单步真实解析通过并构建，wide哨兵/状态桥已核验 | 4215节点→5695SDK层/unresolved0，native building，24/4GiB不变、无需临停；187BF16/18FP16dense/13源FLOAT累加保持。60wide整数仅Slice end且INT32边界裁剪等价；实际prefix50原输出重核、48语义映射与51输入准备，KV仅追加15零槽。无新engine/完整16token目标/主质量或正式指标，五VLA不变。 |
| 2026/10/25 15:42:10 | 单步解析边界/恢复审计闭合，通用整数修正全链通过 | 12解析constant shape scatter失败，无engine/无guard；data27.4718616秒原管理器恢复3582918，其他births/config/pins不变、cgroup空。通用规则94测试，4251→4215，12常量shape删除/12动态KV byteexact保留，全15步49输出portable exact/16token一致，精度与源误差保留。13新图staging，尚无完整目标/主Gate/正式通过，五VLA不变。 |
| 2026/10/25 15:30:58 | 可复用单步序列化/导出/完整portable链通过，准备新图目标构建 | 49输出x15步重载对eager exact；6633节点ONNX结构通过，通用部署4251节点/51入49出。before/after portable15步所有输出exact，16token匹配固定CPU但logits MSE0.00097958/maxabs0.15625/cos0.99995202保留，非源无损/目标或同Orin主Gate。新图staging/24GiB准入4GiB底线准备，暂无native engine/完整目标链/正式数据，五VLA不变。 |
| 2026/10/25 15:23:45 | 可复用缓存CPU接续完成，源误差保留；下一步单步序列化核验 | 477原source张量exact，prefill trace重排按原返回表达式恢复；动态16token/完整logits对冻结源byteexact。固定342cache全15步/48状态合同通过，16token相同，但logits MSE0.000893488747/maxabs0.15625/cos0.999953631，非源无损或同Orin主Gate。007exit0/无guard；008单步trace/reload49输出x15步验证中。无新目标engine/完整链/正式指标，五VLA及09恢复审计不变。 |
| 2026/10/25 15:07:25 | 整片资源重试审计/恢复闭合，转单步源验证；原始输入已恢复并复现 | 09min4191656KiB/1709.879秒被4GiB守护停止、无engine；独立审计源/工具/权限pins未变、空cgroup、三服务原Supervisor恢复3498108/3498074/3498022，compotask及其他程序/config未变，无live Orin任务。不重复边缘增加余量，源参数/状态/单步复用CPU检查保留15步/16token及精度，尚未源输出验收/导出/目标。16原RGB/CHW/archive/manifest与128参考文件恢复/rehashed，官方processor16完整像素tensor byteexact/MSE0；仅准备，非模型质量/正式，五VLA不变。 |
| 2026/10/25 14:05:21 | 提高余量的相同源图已真实解析通过，继续构建 | native receipt building/15234network layers/unresolved0，同graph/precision/builder/plugin不变，28GiB准入/4GiB底线。仅两Java/router条件临停，compotask保持运行，独立原Supervisor恢复待终态；没有engine/完整模型或正式通过。原资源失败、所有CPU源参考差异边界和五VLA验收保持不变。 |
| 2026/10/25 13:59:25 | 共享首片内存保护闭合；相同图/精度提高可用余量重测 | 08min3916184KiB/1727.263秒被4GiB守护停止、无engine，独立审计所有pins不变、data原Supervisor恢复/其他出生与配置不变、cgroup空。CSE受控CPU贡献真实但不等于整片修复。09相同图/精度/builder/DSO，仅准入28GiB，4GiB保护不降；当前独占台架指令派生四项已知可恢复程序范围，历史native许可只是范围/恢复先例，不当新回复或旧窗口复用。28保护测试通过，实际仅两Java/router需停达到29757028KiB，compotask未停，原管理器恢复待终态。真实parsing，无目标完整模型/主Gate/正式数据，五VLA不变。 |
| 2026/10/25 13:18:05 | 新首片真实解析通过，完整134输出CPU参考再次一致 | 首11865节点变体真实native解析完成15234network layers/unresolved0，receipt building，尚无engine/推理。仍原24/4GiB保护及data-only临停，原Supervisor恢复待终态。实际归档Orin前缀输入重算全部134CPU输出，对旧oracle dtype/shape/bytes逐一相同；通用规则节点与实测子图库CSE逐字节相同。不是目标精度或完整阶段修复/正式指标，五VLA验收不变。 |
| 2026/10/25 13:12:45 | 构建CPU常量冗余有实证，通用规则与完整CPU核验后重测整片 | 64插件初始化仅数MiB，未支持句柄初始化根因、不改句柄。两真实3-MatMul SDK子图仅共享相同常量权重转置，CPU allocator峰3880→1940MiB，GPU峰1455MiB/scratch512755712B及约1GBengine不变；不是推理加速/整片修复。通用规则18/67测试，五片仅去395/514/510/187/264常量转置，所有2805BF16/15步/算术/参数/状态/精度保留；全五片CPU16token/full logits再次byteexact、304中间finite，旧源误差保留。新首片11865在真实parsing，同builder/DSO/24与4GiB保护，持续授权窗口仅data临停、starter未停、原管理器恢复待终态；完整target engine/模型/正式仍缺。 |
| 2026/10/25 11:58:13 | 首尾片构建触发内存保护，独立终态审计完成 | min available4132752KiB<4GiB，compute1843.451秒，outerexit-15/build-interrupted，native最后building，无engine；不是3600秒超时/unsupported/数值失败。所有资产/authority pins重核，两owned cgroup空；两Java未停且原出生、Supervisor4237/config/JAR/其他服务未变。下一步原生资源快探针区分初始化/权重workspace/后台占用，不原样重试、不改精度或安全线；根因尚未证实。全五片CPU最终16token/logits byteexact仍独立，真实前缀首片134 CPU参考finite/57诊断工具测试，未新目标推理或正式数据。 |
| 2026/10/25 11:25:41 | 首较小尾片真实解析通过，精度规则无未匹配项 | native receipt building、network_layers15629、unresolved_precision_names=[]；4366条precision及output请求均匹配。原FP16/noTF32/obey/1024MiB约束未省略，尚无engine或完整输出验证。24/4GiB窗口继续，约25.92GiB可用，两Java未停；CPU完整分段parity与目标模型门槛分开。 |
| 2026/10/25 11:23:15 | 首较小尾片真实Orin构建；完整五片CPU结果逐字节一致 | 12,260节点片/50入134出已实际native解析，4366精度和4366输出请求完整保留，派生长期授权有界窗口24GiB准入/4GiB线，3600秒task/7200秒window，未停Java。源计算和全部精度覆盖独立重核、五片资产及复用权重/工具板端重哈希，35旧保护测试复验通过。完整五片CPU顺序执行304中间输出finite，最终16tokens及全部logits与未分段同表示参考byte-exact/MSE0；仅host partition parity，不改变官方源误差或代替Orin quality/performance。目标engine/完整native链/resident/主Gate/正式仍待，五VLA不变。 |
| 2026/10/25 11:10:22 | 整图解析主动结束并恢复；五小片/七段完整源IR准备完成 | 约1967秒仍parsing/no engine时仅停止owned窗口以控图规模，非5400秒超时/资源/unsupported/精度失败。独立审计两Java原Supervisor恢复、所有资产pins未变、owned cgroup空，Goal仍active。五片12260/14510/14413/7742/9316保留每原NodeProto和精度恰一次、原initializer及两终端输出；84早期结果直接原生产者路由不加Identity/cast，首别名拒绝负记录保留。七纯uncached Region IR/Plan、118软件和13看板测试通过，不是target绑定或模型质量。下一步小图构建/完整16token/resident/主Gate/正式指标，Qwen2B仍等待，五VLA验收未改。 |
| 2026/10/25 10:56:04 | 备用分段保留完整原节点/输出/精度，未替换当前构建 | 26770+31471节点，每个原NodeProto逐字节不变且恰一次，两最终输出Proto不变、初始化量/外部范围不变，所有precision/output映射恰一次；140边界张量。未做候选数值parity/目标构建/推理，不计部署通过，当前whole-tail未中断且仍parsing/约28.41GiB可用。ONNX极端端点/正负步长补充48例后共115软件回归通过，非目标完整质量或正式指标。 |
| 2026/10/25 10:47:22 | 三段完整源链IR/Plan及全部50状态类型连接完成，非目标绑定 | 复用现有InvocationBuilder/compile_module，三pure uncached Regions、50明确源顺序状态路由、完整两最终输出。所有部署carrier shape/dtype精确，不合成状态/隐式cast；目标I64/I32存储适配、真实engine绑定及resident Session仍待，外部commit许可不作模型质量证据。67组合软件检查通过。Orin完整tail仍parsing/约28.65GiB可用，无新目标质量或正式指标。 |
| 2026/10/25 10:39:07 | 完整15步tail真实native解析；长时授权派生有界窗口 | 58241节点/50入2出及全部20689精度/输出映射通过JSON文件接口构建，无全FP32回退或步数缩减。新长时授权独立派生5400秒task/7200秒window，24GiB准入高于平台16GiB，4GiB运行线未变；两获批Java临停，原Supervisor恢复待终态。897超范围整数仅证明静态Slice端点语义，36工具保护测试通过，不是tail engine/16token/同Orin主质量或正式指标。prefix实测及参考差异保留，五VLA验收不变。 |
| 2026/10/25 10:02:18 | 真实两段prefix/全50状态核验与恢复收束；继续完整tail/16token | 真实Orin inspector及C++context/enqueue全部成功，实际接口2→50/52→50。最终50张量7170050值有限，首token32/RoPE-280经精确整数storage codec与CPU portable一致；浮点MSE最坏0.0711422827223/maxabs5.765625、mincos0.981940520329，原误差保留，不宣称参考无损/同Orin优化主门槛通过。两engine/图/工具/输入及metadata/pilot/raw输出再次独立重哈希；仅data临停62.629870304秒后原Supervisor恢复3132518/start37483668，starter原出生不变、其他程序/config/JAR不变，owned cgroup空，原30分钟内闭合。旧slot详细导出错误保留，此次读接口/上下文/输出可用。73工具测试分开记。用户明确放宽台架执行时段，后续长阶段自主安排，既有短窗口不重写/续期；完整15步tail/resident Session/16token/正式指标仍缺。 |
| 2026/10/25 09:26:57 | 真实输入与全部状态连接已staged，58工具检查通过，待新pilot许可 | 一个真实源观测仅载入CPU输入、不执行源模型：IDs INT64[1,327]范围13..248069、图像F32[1200,1536]范围-1..1，共7375416B。通用stdlib文件链先真实inspector再原生Cpp输出，consumer仅取实际producer输出，dtype转换只允许声明且完整范围通过的I64→I32；没有预计算cache/补零/隐式浮点cast。14板端文件重哈希，source98352917.../plan95d8982d...。58合成工具/恢复测试通过，不是目标I/O/输出/性能结论。13587732KiB低于16GiB，两Java运行；新最多30分钟pilot已请求但未获明确答复，尚无auth文件/活动窗口/新GPU或服务动作。旧窗口不续期，producer内部错误和完整tail/正式指标仍缺。 |
| 2026/10/25 08:55:34 | producer构建后slot内部错误已保留，SDKbuildOnly/导出源码与时间顺序取证 | 日志先报告Engine built及deserialized0.9242秒，随后slot.h::decode断言/层JSON0B，再跳过推理并reportPass；consumer反序列化0.505905秒/层JSON1272951B有效，无错误行。设备安装源码buildOnly调用printLayerInfo(engine,nullptr)后reportPass，导出函数无条件返回true；源码SHA bf899d4e...已保存，不假定二进制由此份源码编译或错误仅限导出。故保留exit0和负证据，未提升为可用engine对/实际输出/质量或正式指标通过；没有新GPU/服务动作。 |
| 2026/10/25 08:51:45 | 两段engine已生成，终态发现第一段SDK内部错误；服务已恢复，未模型通过 | producer/consumer原始exit0/954.541654144与893.944740416秒，actual SHA089f62e1.../297a44a7...再次核验；producer反序列化后slot.h::decode内部断言且layer文件0B，consumer层JSON1272951B有效。原日志/错误不重写，不把buildOnly reportPass当上下文/输出质量或正式数据。仅data临停1879.792253888秒，原Supervisor恢复PID3061628/start37129667；starter原出生未动，其他managed snapshot/config/JAR不变，owned cgroup空/原两小时内恢复。原窗口关闭不续用，当前14490560KiB低于16GiB，无新GPU/服务动作；下一步真实ABI、全部50状态、完整tail和同平台质量/正式指标。 |
| 2026/10/25 08:38:59 | 第二段真实解析完成，实际绑定检查器已就绪但未运行 | consumer日志Finish parsing network model、Unused Input input_0/input_2，仍为building无新错误/4GiB保护；不可把SDK warning当实际I/O读取或填充被裁输入。producer layers.json为空，不充当布局证明。复用已审计元数据inspector源码原生编译exit0、binary d7488706...，未执行或加载engine、不与当前GPU构建并行运行。原服务恢复策略和主验收门槛保持。 |
| 2026/10/25 08:34:14 | 完整参数前缀第一段原生engine构建成功，第二段开始 | 真实同一Orin TRT第一段exit0/built-not-validated，954.541654144秒，2109829697B，engine SHA089f62e1e5682ed31fa8f27ea001c1753f224b3c2861ae84388bb215e8b5c9f5。第二段重新16GiB准入后已进入解析/构建，4GiB及1800秒线不改。仅data临停，starter原出生不变。未执行新模型推理，未作完整前缀/16token/同平台质量或正式指标验收；窗口仍独立受保护。 |
| 2026/10/25 08:30:34 | 全输出工具7项pre-CUDA拒绝通过，模型构建仍进行且日志零错误 | 相同原生binary拒绝缺参/重复key/尾随JSON/错误schema/错误serial/错误boot/空bindings，均在创建输出目录、插件或CUDA初始化前退出，无engine加载、推理或额外服务操作。真实第一段模型构建日志5.78MB中[E]计数0，仍running而非通过；MemAvailable约11.59GiB，恢复守护不变。 |
| 2026/10/25 08:27:01 | 完整参数前缀继续原生构建，通用raw输出工具CPU编译通过 | 同一Orin第一段已解析，trtexec仍运行且4GiB保护未触发；当前约11.55GiB，starter原出生运行。通用C++17 raw-engine-pilot以CUDA隐藏CPU编译/链接2.982403616秒exit0，binary c432abeb...，未执行/未加载engine/未推理。工具要求全部实际名字/dtype/shape及engine/DSO/input SHA，二进制完整输出，拒绝静默cast；不是原生模型质量或性能通过。13看板回归再次通过，保留固定主门槛及五VLA证据。 |
| 2026/10/25 08:21:58 | 完整参数前缀第一段解析完成，engine构建继续；仅data临停 | 同一Orin真实日志Finish parsing network model，构建仍running且未产生接受engine/推理。data临停后准入22333260KiB，starter原出生1465114/28963358保持运行；当前18831552KiB约17.96GiB、compute/window均active。原Supervisor/服务恢复保护继续，不增停其他服务、不开始正式采样。 |
| 2026/10/25 08:17:33 | 新独立2小时许可收到，0.8B完整参数前缀分段构建窗口启动 | 原台架/boot/Supervisor4237/两Java出生和JAR核验，SDK8.5.2及MAXN不改；只读preflight15795544KiB低于16GiB、无并行trtexec，磁盘余242655256576B。新bbcd9f1f许可绑定独立job及systemd恢复，旧许可不续用。图/精度/3.423GB权重逐一hash确认。最多7200秒，预留900秒恢复，仅按需data后starter，16窗口保护测试通过；未执行模型推理或正式采样，不变更五VLA结论。 |
| 2026/10/25 07:05:04 | 主机完整串接与50状态改写核验收束，三回合资源/许可阻塞，Goal已blocked | 本轮源侧真实计算非空等：2B单个真实观测prefix→15步tail全部16tokens等于来源/输出有限，但logits MSE0.012132943794/cos0.998849929274/maxabs0.5，prefix最坏state MSE0.262105/cos0.941399，均保留且不称来源无损/主Orin gate。裁剪前后全部50tensor7170050值同表示byte-exact；源dtype声明187/2805 BF16、117/270 FP16dense、655/195源FLOAT，22通用测试通过，无全FP32回退。所有host worker终态。连续三个Goal回合20:55/21:54/22:57真读均<16GiB，新服务grant未回复，旧window003恢复后不续期。工具已返回Goal blocked，下一步实际engine/I/O/常驻链与采样必须等准入或新有界授权；未新服务控制或Orin模型执行，不改范围/不标完成或unsupported。 |
| 2026/10/25 06:17:07 | 2B两阶段规则已同步，整数控制及BF16部署表示继续推进 | 新prefix196.064秒导出/checker通过，190400节点/624initializer/50输出/30safe-softmax位点；runtime parity skipped，50状态source合同仍与tail逐项一致。外部429文件按内容哈希多重集不变，但图编号导致名称不同，不冒充算子绑定或数值证明。整数折叠32776/47261不求值浮点；tail原生BF16 MatMul2805、舍入边界11562/输入48，数据7.527GB，不做全FP32回退或精度认证。Prefix BF16转换和tail通用部署改写新worker运行中；Orin内存仍低于16GiB/新许可待回复，无新目标或服务操作，Goal active。 |
| 2026/10/25 05:54:54 | 2B完整tail导出/归档闭合，原始50状态合同匹配，仍非Orin通过 | 602.031秒CPU导出exit0/无guard，86886节点/321initializer/50入2出，90 SDPA及safe-softmax全部记录；checker通过而runtime parity显式skipped，不把不同Torch2.7/2.10主机推成数值等价。metadata SHAe458b93f...，254外部文件3.764GB，含模型/report共256文件NAS fsync/原始哈希复核，本地全manifest亦匹配；6归档正负例通过。PID626517/626518消失，原prefix整数折叠190040→32566/无浮点求值，50输出合同逐项等tail输入，但prefix旧策略仍须同步通用修复及完整数值验证。Orin21:54UTC16183924KiB<16GiB，两Java原出生运行/无trtexec，新许可待回复，未新服务控制或模型执行；两Qwen正式指标与Goal未完成。 |
| 2026/10/25 05:35:34 | 2B完整15步tail已在大内存主机CPU导出，非GPU实验或目标通过 | Source archive/input/profile SHA0d2b92a5.../5b06df99.../3abbc78d...本地与远端一致，CUDA隐藏/2CPU/900秒，独立资源保护物理≥512GiB/可用≥128GiB/RSS≤64GiB和scratch32准入/16运行GiB；新增保护配置5测试再次通过。新通用exporter已52远端/88本机回归，任务实际进入ONNX导出并确认句柄live，不先报完整模型或数值通过。Orin21:34UTC16310080KiB仍不足16GiB、无trtexec、两Java原出生运行，未新临停/目标模型执行，不改本机24或Orin16GiB线。 |
| 2026/10/25 05:30:29 | 本机tail固定RSS边界确认，大内存主机CPU导出环境独立回归通过 | 隔离002 RSS26047975424B越24GiB，内存可用6.073GB>3GiB，124.711秒保护退出；原负例保留，不再抬本机门槛重试。大内存主机只读复用现有依赖时39/5失败揭示全mask NaN和测试注册路径，新增ONNX reference4红例后通用safe-softmax仍委托原SDPA；本机88相邻通过，远端经非NAS scratch52全过，原NAS51/1保留。全局环境/SDK未改，CUDA隐藏，尚未跑大模型。独立512GiB物理/128GiB可用/64GiB RSS/900秒/2CPU配置5正负例通过，不改本机24或Orin16GiB门槛。2B源包已上传正在核SHA，Orin未新临停/模型执行，指标仍缺。 |
| 2026/10/25 05:07:48 | 2B tail主机全局内存保护触发，保留负例并做原限额隔离重试 | 001完整tail115.858秒exit-15，末RSS24789962752B<24GiB而主机可用2615320576B<3GiB；与前缀权重整理后段重叠，未发布完整tail ONNX，非Orin OOM或算子unsupported。前缀外置28.011秒通过，190040节点及71266张量4430639608B metadata/payload指纹不变。原两worker终态，新002只移除大型并发，保持原archive/input/profile/exporter/15步/2threads/24/3GiB/900秒，不先报问题已修复。无新服务或目标模型执行，Goal active。 |
| 2026/10/25 05:04:14 | 2B完整前缀ONNX导出/checker通过，完整15步tail导出启动 | 前缀195.064秒exit0/guard null，190040节点/624initializer/50输出，429外部文件4429672704B；原未用mask消除后2实际输入，不伪造mask/状态。保留全部公开状态，metadata SHAdbdb1321...。runtime parity显式skipped，不报ONNX数值/Orin支持通过。完整15步tail按自身90位点profile开始有界导出，未套0.8Bprofile或减步；44导出器GQA/opmath回归通过。Orin新临停许可仍待回复，无新服务或模型执行，正式Qwen指标仍缺。 |
| 2026/10/25 05:00:44 | 2B真实阶段30/90注意力位点观测通过，完整前缀ONNX开始导出 | 两阶段独立profile绑定实际archive/input SHA，完整CPU输出不变。Prefix30处：视觉24处16/16/16、语言6处8/2/2；tail90处8/2/2，不能用0.8B视觉profile代替。104.036/36.013秒worker退出0/无guard。复用与冻结003 SHA一致的通用公共导出器，有界完整prefix导出已启动，runtime parity显式关闭，不先报checker/数值/目标通过；不删生成步或精度规则。Orin内存仍不足16GiB/无新临停，整体Goal active。 |
| 2026/10/25 04:55:51 | 2B两阶段保存/重载全输出一致，16观测128文件齐备，未计目标指标 | 参数保留的prefix3→50/tail50→2均保存重载并核对全部源输出byte-exact，198.062秒/exit0/guard null，峰值RSS10.273GiB；源接续完整16tokens/logits证明及H20补充误差均保留。16个真实观测128文件全部匹配历史SHA/bytes，未重跑历史推理；7核验器正负例通过。所有主机worker终态。Orin20:55UTC16158152KiB仍低于16GiB、trtexec无、两Java原出生，未新临停或目标模型执行。下一步完整ONNX/实际精度/Orin部署与正式指标，不把CPU准备当模型验收，Goal active。 |
| 2026/10/25 04:49:39 | 2B完整源接续逐字节通过，实际状态与H20差异独立记录 | 同CPU完整原调用vs前缀+后15步，全部16tokens及248320 BF16 logits byte-exact、公共输入不变；126.031秒退出0/无guard。真实50状态48BF16+2I64共14340112B，状态shape来自执行非模型名外推；H20 tokens exact，logits MSE0.02411/maxabs0.90625/cos0.997724仍补充保留。通用保留注册参数的两阶段保存/重载新worker实际运行中，未预报通过。无Orin新执行/服务操作，两模型TTFT/CDF仍缺，Goal active。 |
| 2026/10/25 04:47:01 | 台架等待准入期间恢复2B真实完整源包，独立状态评估开始 | NAS实际4.431GB完整16token归档SHA818881968f...与原bundle及validation一致，首样本8文件核SHA通过，输入准备4.002秒成功。静态112580节点/16argmax，首token后50非恒定Tensor，不按0.8B外推；完整CPU接续证明新worker运行中，未先报等价通过。Orin20:44UTC15992320KiB仍不足16GiB，两Java原出生运行，未新临停服务或模型构建。两Qwen正式指标仍缺，Goal active。 |
| 2026/10/25 04:43:22 | 核对同一回复已用于结束窗口，保持两服务运行，不自动续期 | 当前ea9de85c问题回复与window003授权文件question_item_id完全相同，14:46UTC终态审计已证明服务临停766.161秒后按原Supervisor恢复、owned compute为空。不能据同一条旧回复重开2小时窗口。20:43UTC只读15965052KiB<16777216KiB、trtexec空、两Java PID/start未变，未临停或新启动完整模型。此前另发的新窗口询问仍待明确回复；13看板回归通过，Qwen指标和整个Goal仍未完成。 |
| 2026/10/25 04:37:45 | 完整tail长精度参数障碍以原生JSON接口验证，模型分段仍等资源许可 | 没有重复问许可或动服务：最新MemAvailable15869212KiB仍不足16GiB，trtexec无，新2小时许可待回复。完整15步tail有2805BF16/270源低FP16/195源FLOAT策略，precision和output各20689规则，单CLIflag324597B超Linux上限，台架SDK源码同名参数不合并。采用现有JsonCpp/TRT8.5原生C++文件构建器，编译/大profile纯校验通过，3非法JSON拒绝；独立小型非模型SDK构建通过，missing_layer精度名明确拒绝。无Python TRT安装、无规则删减/全FP32、无模型推理，managed snapshots不变。工具smoke不能替代Qwen完整native生成/TTFT/CDF或2B验收，Goal active。 |
| 2026/10/25 04:22:00 | 参数表示完整16token恢复源一致，logits误差保留；分段已备齐，等待资源许可 | 在同一源模型/同Torch50状态byte-exact已证基础上，只切换共享参数序列化表示并复用全部通用变换，源首token32（冻结表示357）恢复；同一完整15步tail16token全等同CPU冻结源。logits MSE0.005419/cos0.999728及state最坏cos0.9522仍披露、不宣称源无损/不改同Orin主门槛。参数版完整9735+11314分段/source计算不重复/50输出byte-exact/常量状态精确保留，187BF16+69FP16+631源FLOAT保持，111回归通过，两模型/precision/载荷已上传核SHA。20:22UTC16738824KiB低于16GiB，trtexec无，两Java原PID/出生未变；已申请新的2小时条件临停，尚未收到明确授权，因此未停服务或启动构建。完整Orin生成/TTFT/CDF/2B未过，Goal active。 |
| 2026/10/25 03:55:51 | 整图预算退出已清理；完整分段50状态验证闭合，source生成漂移单列失败 | 整图第二次构建1800.296秒guard124，无engine，最低11982520KiB，4GiB线未触发，非OOM或unsupported；独立审计4个原actor及owned组消失，业务快照不变，不重启旧job。裁剪新增控制从原DAG独立形状推导且50输出byte-exact，不放宽reviewer。分段工具精确保留源直接initializer输出（含RoPE偏移），两红例/167回归；完整9633+11202节点/source每个计算一次/50输出byte-exact已准备，尚未新增真机。完整后15步58241节点准备，host参考CSE不改部署图/RSS降7.19GiB；完整16token虽有限可运行但与同CPU冻结源token不同，logits MSE2.0327/cos0.8951为失败source诊断，单列不替换同Orin常规TRT/优化固定主门槛。完整目标生成/TTFT/CDF和2B未过，Goal active。 |
| 2026/10/25 03:17:12 | 完整50状态回集成与CPU门槛闭合；保留负Pad失败并进入第二轮完整构建 | 不再只处理首token子集：完整状态前缀188677→31537原始节点，部署20837节点，全部50输出7170050值的索引改写CPU byte-exact。外部BF16 dtype保守排除修复有真实红例，不算浮点、不放宽plugin静态形状证明。首轮完整构建暴露卷积缓存Pad负裁剪[-323]；通用Pad→Slice覆盖18节点、34单测、183相邻回归，再次完整50输出byte-exact，精度选中节点与规则未改。第二轮完整构建仅本任务actor、16GiB准入/4GiB运行/1800秒，真实guard2669599/child2669601运行，无服务操作；原失败及最小图slot错误保留。剩余15步tail85806→46631节点并行本机准备。尚无完整16token Orin/TTFT/CDF或2B验收，Goal仍active。 |
| 2026/10/25 01:46:41 | 动态Slice局部兼容路径真机完整输出逐字节通过；转完整状态链回集成 | 12节点保留源IDs-mask依赖后复现同Slice维度断言，1024/2048MiB workspace均失败；Min显式上界36主机测试虽通过，真机仍失败。零起点单位步长一维Slice改为按真实extent截断的Range/Gather后，50主机语义测试及真实CPU边界byte-exact，真机57.581秒构建1017433644B引擎。保留trtexec层信息slot错误/空JSON，不声称placement通过。实际INT32/FLOAT绑定按dtype文件装载、整型范围检查，2次enqueue完成CUDA同步，334848值全部finite并与源边界byte-exact/MSE0/cosine1。无服务操作、managed snapshots不变。局部一个样本不替代完整16token模型/TTFT/CDF；下一步直接回集成完整50状态前缀，Goal仍active。 |
| 2026/10/24 23:00:22 | 完整状态链两阶段ONNX结构导出闭合；Orin完整构建/推理仍未过 | CPU SSA前缀50状态接续全部16token/最终logits byte-exact，冻结与保留共享参数两表示保存重加载也完全一致。完整prefix188677节点/50输出、tail85806节点/50入2出结构checker均通过，50状态shape/dtype/顺序逐项一致；runtime parity明确skipped。冻结tail旧导出24GiB守护失败保留，新共享参数tail183.066秒、峰值12.567GiB、无guard。33软件及13看板回归通过。新鲜target检查两Java运行/未停starter出生不变/own units空；本轮只曾临停data766.161秒并按原管理器恢复。首段v4动态NonZero/Slice/Scatter融合失败仍未解决，无Qwen目标完整链/TTFT/CDF或2B验收，Goal继续active。 |
| 2026/10/24 22:49:29 | v4完整首段workspace/动态Scatter边界失败已恢复；完整CPU状态链与prefix导出推进 | 新7200秒构建/pilot许可只实际临停data-collection 766.161秒，starter未动，独立systemd后处理与再次审计确认原Supervisor恢复/其他程序及图profile权重SHA不变、own cgroup空。首段744.456秒失败，SDK称Embedding到ScatterND融合workspace不足，同时Slice维度断言，不能认定容量唯一根因；guard未触发、consumer未启动，无Qwen推理或正式指标。完整CPU图按SSA分50张量状态，3入50出prefix接50入2出剩余15步，序列化重加载后全部16token/最终logits仍byte-exact；H20差异补充。prefix ONNX188677节点/50输出checker通过，runtime skipped，tail在本机导出。33软件检查通过，旧失败保留，Goal仍active。 |
| 2026/10/24 21:41:16 | 第二次整段构建终止于QKV GEMM实现选择，未遗留本任务测评进程 | 单Conv格式修正后的完整第一段构建623.568420秒，原Conv格式错误消失，但vision QKV Gemm出现Could not find any implementation。无资源守护触发（最低6575860KiB）、外置权重hash未变、四业务身份不变、进程组空。原始build.json/build.log已回传并核对日志SHA；当前只读确认没有本任务原生解析/构建进程，不标engine/推理/性能通过。下一步精确单GEMM诊断与通用适配。工作目录已核对：原仓库与f801独立，继续按Goal在f801保存证据，未向原仓库复制/覆盖文件；Goal仍active。 |
| 2026/10/24 20:33:59 | 真实整段格式失败已最小化；只加一个有实测证据的FP32岛再构建 | 第一段build357.640572秒失败在/prefill/vision//proj/Conv的No supported formats，内存守护未触发、最低14460120KiB，四业务身份不变/进程组空。原始单Conv同shape/权重/workspace的FP16计算/FLOAT输出3.763116秒同错；FP32计算/FLOAT输出5.795300秒buildOnly通过，非整模型FP32替代/推理。v3只改这一个源低精度Conv为显式FP32岛，68FP16 dense、613源FLOAT累加及187BF16 provider保持。140相关检查通过；第二次完整第一段构建实际进行，仍无目标输出或正式性能。 |
| 2026/10/24 20:04:48 | 两段完整首token真实Orin解析通过，启动受保护engine构建 | 单图007仍900.415秒预算退出，未标unsupported。通用分区保留全部20723节点为9732+10991、5静态前沿，无复制/公式变化，完整CPU拼接token/logits byte-exact。008两段各229.870/297.001秒真实解析exit0/零错误，四业务身份不变。原始dtype约束69FP16 dense/613源FLOAT累加dense/187BF16 provider，原builder同SHA且5兼容检查通过；第一段仅buildOnly，16GiB准入/4GiB运行/1800秒上限，无服务操作。还没有目标推理、完整16token/state或正式性能。 |
| 2026/10/24 19:37:20 | 首token整数化简完整输出exact，完整16token图同步减小；目标解析仍未验收 | 模型中立的有界整数/BOOL与独立shape折叠首token可达图163476→20723；存活计算/浮点载荷检查、完整CPU portable token及248320维logits前后byte-exact，MSE0，不宣称源BF16或目标plugin等价。完整16token保留2992BF16 MatMul、12517舍入及全部迭代，261070→79298节点。134相关回归通过，所有旧CPU源差异/脚本失败保留。006真实目标解析900.917s预算退出且四业务身份未变，007固定点图继续实际解析；无Orin推理、engine、正式指标，2B待独立评估。 |
| 2026/10/24 19:01:04 | 完整16token源图导出完成；CPU源诊断失败与TRT解析预算边界保留 | 本次状态核对确认五VLA主表/CDF/同Orin质量与限定范围D4均有正式数据。Qwen0.8B首token portable完整CPU比较token相同，但logits MSE0.018800586/cos0.998918418未过严格源诊断；实际原生解析Norm/Trilu修正后，163476节点裁剪图仍900秒预算退出，非OOM/unsupported/engine验收。完整16token源及8参考SHA复用，108shape位点观测不改完整CPU输出；新导出exit0/294.105199s、277382节点/480initializers、结构checker通过但runtime parity跳过。仍无Qwen Orin TTFT/生成/CDF指标，2B独立待评估；未新增目标GPU任务或变更服务。 |
| 2026/10/24 17:07:49 | Qwen结构导出后发现真实类型错误，修正SDPA opmath并继续完整输出核验 | 003权重外置70071载荷前后完整指纹一致，portable395舍入边界生成，但ORT加载因视觉SDPA Cast(to=15)成为complex128而失败，未执行/接受输出对比。源Torch SDPA文档确认half/BF16 math中间值FLOAT，通用可选改写现在显式ONNX dtype及FLOAT opmath保留原输出舍入，不做全局dtype替换。57相关检查通过；004真实导出exit0/122.037639s、186935节点/18opmath调用，外置载荷完整指纹一致，portable2已生成。真实CPU完整token/logits参考保存，当前完整类型/ORT对比运行；无目标GPU推理或性能、新16token支持或2B验收。 |
| 2026/10/24 16:37:43 | Qwen0.8B真实图文首token ONNX通过，完整目标链仍待验证 | 源包e173a9f1.../8真实输入参考完整SHA与旧H20封存一致。首轮GQA不支持、二轮未知语言头维均失败且保留；可选通用GQA扩展与archive/input/source-site-bound实际形状profile后，003 exit0/115.037773s，186956节点/479权重/18GQA18specializations/checker通过，runtime parity skipped。18源SDPA形状observer/direct完整CPU输出byte-exact，51回归通过；初始失败导出器源SHA83f019...已精确保留，不覆盖旧数据或模型精度。没有Qwen目标engine/16token/TTFT/CDF结果，接下来外置权重/BF16/TRT及完整state链，2B独立待评估；目标Torch和新跨平台性能实验均无。 |
| 2026/10/24 16:03:47 | 按用户明确许可清理冷模型，本机/Orin分别恢复到约248/238GiB余量 | 用户明确优先实测论文数据/文档，不要求保留所有复现模型。先14副本/6组精确合并释放29.19GiB，继而108本机冷payload释放194.35GiB，Orin208路径释放190.12GiB；全删除前SHA/字节/元数据/写前日志保留且实际不存在。目标首轮Python3.8 API预检查失败在unlink前，独立兼容版重试完成，失败记录不抹除。五模型69120原始/61440测量、18CDF重hash、主/质量/过程表与原字节相同，48数据及17清理检查通过，SDK/业务/环境/PDF不动。文档明确历史closure不代表engine当前可运行，无大文件备份，需重备模型后才能新推理。Qwen未清理且无新目标推理，整个Goal仍active。 |
| 2026/10/24 15:27:06 | 核心五模型有界交付闭合，转Qwen完整链支持评估 | 按活动Goal原31项发布逐项验收，修正旧claim-gap中CogACT6/10、未计时算子和功耗模式待评估等过时行。新文档发布同请求三边界、60资源envelope、共享内存不双算与私有SDK/GPU未知；9份CSV/Markdown与原输出字节一致，组件重查及48项回归/最终文档复查通过，PDF不变。memory/delivery仅在有界观察和核心出版范围done，未解决私有物理峰值/全局零分配/第二模式实测或机器人主张。开始后置可选Qwen0.8B完整图文与状态资产评估，未启动Qwen目标构建/采样，Goal仍active。 |
| 2026/10/24 15:14:24 | 核心测量已逐项审查，转最终文档与证据索引收尾 | 按当前活动Goal冻结文本逐项审查31项，29核心项有明确范围内测量证据，2项后置可选Qwen待评估。69120原请求计时分段和36组分布核验，61440唯一测量不重复计数，不把关联子区间冒充新独立实验；6项回归通过。板卡只读复查自有活动unit为0，未运行新模型或改变服务。内存进程峰值/覆盖CUDART观测有效，私有SDK/driver物理峰值仍未知；另一功耗模式未测限制保留。最终报告/旧claim-gap证据表/可复跑索引尚待统一，Goal仍active。 |
| 2026/10/24 14:38:47 | 完整五模型CSV及60进程资源交付，组件旧TODO已据实闭合 | 重算69120请求/61440测量/12配置、60进程及18CDF文件pins，旧四模型数值完全一致；CogACT只加完整已验收批次。两OpenPI实际源图各512F64频率与原源shadow同字节SHA，step build/导入/engine及三cast DSO注册/workspace与两完整joint相连；旧smoke不冒充模型。六族合同/预算/12选定build成本116.830536192s已汇总，Agent思考未知，旧失败不删除。五模型60资源envelope/78667遥测不双算，私有GPU/SDK峰值仍null未知。34检查通过，无新GPU/服务/SDK操作。Goal仍active，最终内存范围/逐项交付及Qwen评估待完成。 |
| 2026/10/24 14:05:15 | Embedding/RoPE完整正式补测通过，另一功耗模式权限边界已实测 | 40进程每进程128warm+1024measured，全40960测量/完整末输出/原轮次/8GiB准入2GiB保护通过；RoPE resident1.361492x/IO1.144595x，Embedding差异小且两边界各3负轮。521目标hash/71出生终态、四units空、业务不变；33检查、成本CSV/CDF PNG/PDF已归档并实看。nvpmodel当前MAXN/0，配置有15W/30W/50W但关键路径对UID1000不可写且CLI无提权属性，EMC读取EACCES保留，未切模式。转最终五模型CSV和跨子任务逐项验收，Goal仍未完成，Qwen后置。 |
| 2026/10/24 13:51:00 | Embedding/RoPE四新引擎及159360完整值exact核验通过，尚未计时 | 原四图/profile保持字节不变，新FLOAT-only SDK数值合同禁止FP16/INT8/TF32许可并保留OBEY。实际四构建/四SDK ABI/六次原始点执行通过，三输出对byte-exact；120目标路径hash/20出生终态与两个own units空，业务身份配置不变。27CPU检查通过，旧内部精度未观测结果不回改，不作全模型FP32/隐藏物理精度/新VLA E2E通过。下一步冻结五轮128+1024算子计时。 |
| 2026/10/24 13:37:29 | CogACT完整D1-D3联合验收，五个必需VLA正式主线5/5 | 十进程/10240measured/57600完整typed输出exact、资源/CDF/原Supervisor恢复/58187目标路径hash/36出生终态全部通过；两项获批业务已恢复且原配置/其他受管程序不变。仅Arena无replay，均值694.393388→693.790506ms差约0.087%、一轮负收益、std未达1.2ms，不宣称显著加速。03/109失败分开保留；当前位置转D4剩余算子、资源/功耗边界与汇总，整个Goal仍未完成，可选Qwen未启动。 |
| 2026/10/24 12:14:57 | CogACT 6/10完整进程、6144测量；D4缺项对照及FLOAT算子新合同仅主机准备 | 第七Arena进程9warmup，实时flags通过，原24/4GiB门槛不变；其他受管程序未变，正式仍4/5。D4矩阵明确旧Norm回集成负收益实验已完成，热稳态/匹配功耗已有证据，物理峰值/私有归因不伪造。06四份未修改FLOAT算子图通过主机完整类型检查，通用新构建仅去掉FP16权限、保留noTF32/OBEY，11检查通过；不重标旧104/110结果，不上板、不并发GPU，等待五模型完整验收/恢复/归档。 |
| 2026/10/24 11:54:28 | CogACT 4/10完整进程，累计4615正式测量；终态归档入口已准备 | 第五进程128warm+519measured，实时flags通过、16765612KiB高于4GiB运行线；仅两项获批业务临停。主机新增只读归档入口，先验真机恢复/空cgroup再读大模型和全量文件，核对实际manifest依赖及noise和记录PID出生，绑定同一服务回执；77项NONMODEL检查通过。当前未实际归档/批量传输活跃输出，不改变部署代码、模型/精度/采样门槛，正式接受仍4/5。 |
| 2026/10/24 11:39:41 | CogACT第四进程继续正式采样，累计3518；typed整批验收入口准备完成 | 3/10进程完整，第四128warm+446measured，实时质量flags通过，运行内存17188476KiB高于4GiB。仅两项获批业务临停，其余受管程序不变。主机端整批joining保留固定计数、轮换、慢样本与负收益，整数/opaque按原始字节不伪造MSE；68项NONMODEL检查通过，但没有真实终态恢复/完整原始输出/CDF联合验收，正式仍4/5。部署代码及模型/精度/样本协议未变。 |
| 2026/10/24 11:33:16 | CogACT 3/10进程完整结束，第四进程预热；终态恢复审计入口已准备 | 累计3072正式样本，第四普通路径20次预热，实时flags通过；退出后再次通过原24GiB下一进程准入，未降低门槛。新增主机端通用恢复合同审计，22新/53合并检查通过，包括拒绝当前仍临停状态；这不是实际恢复或模型接受。完整typed输出、资源、CDF、目标文件/进程与实际恢复审计仍待全批结束，正式4/5。 |
| 2026/10/24 11:25:12 | CogACT第三独立进程进入正式采样，当前2443条 | 2/10进程完整结束，第三进程128warm+395measured，实时质量flags通过。仅router_manager和compotask_service临停，其他受管程序不变；未修改自启、未新增或重启采样。全批输出、资源、CDF和原管理器恢复待结束审计，正式接受仍4/5。 |
| 2026/10/24 11:20:03 | 首轮两配置均完整结束，2/10进程，第三进程已实际启动 | ordinary-source与verified-arena首轮各128warm+1024measured已完整终态，累计2048measured；第三进程为下一轮Arena，79warmup/0measured。实时输出flags通过，运行可用17907400KiB>4GiB；跨两个进程退出后的新入场仍遵守原24GiB，无降线或部分数据拼接。只读服务观察确认两项授权业务仍停止、其他程序未变，恢复仍pending。主机合并测试发现同名模块缓存误载旧Jar策略，只隔离测试加载后31通过；原29/1失败保留，部署三脚本hash均未变。Goal/第五模型未接受。 |
| 2026/10/24 11:09:38 | 首正式进程完整结束，第二配置实际通过24GiB下一进程准入 | 新05窗口首ordinary完整128warm+1024measured已终态；verified-arena已实际运行128warm+252measured，累计1276measured，1/10完整。实时native flags均通过，尚未做整批离线全字节审计，不发布成对加速比。运行可用18921448KiB>4GiB，24GiB仍为下一进程准入；不是降线通过。原Supervisor只读确认两项获批原生业务停止，其他管理程序未变，当前恢复就绪为false；19检查不冒充实际恢复，Goal/第五模型未完成。 |
| 2026/10/24 10:52:01 | 无需root路径已实际启动：两原生业务临停，跨进程资源探针通过 | 新批准仅临停router_manager/compotask_service，不改自启、不停Java或其他服务。两短完整进程退出后32482752/32441996KiB均通过原24GiB，正式新批实际运行。0/10终态，当前ordinary-source 56warm+0measured；实时flags通过，尚待完整独立输出/资源/CDF/恢复审计，正式接受仍4/5。新通用服务描述按program/executable/argv/cwd/父进程验证，不假设Java；不重新执行server.run、不改自启/CAN/网络/时钟。32探针调用不计入正式，旧批次不合并。 |
| 2026/10/24 10:01:29 | 更正回集成子项：完整实测已完成，正收益不是额外完成前提 | 重读11520完整调用/23040输出与10240测量，原audit SHA0e154ce0及CSV完全重现，资源也相同；源工具版本hash不同单列，但冻结候选图/精度控制仍完全重现。标准化主门槛通过、native补充失败保留，五对E2E均回退约6.43%。九检查通过，仅闭合代表模型集成实验，不宣称正收益/默认采用、自主kernel合成、全部D4或五模型完成；当前CogACT仍在新4小时许可下正式采样。 |
| 2026/10/24 09:54:57 | CogACT新正式双模式批次已实际启动，独立4小时许可与恢复机制 | 新4小时正式许可已绑定，只停两Java并预留900秒恢复；临停后29334088KiB通过24GiB。0/10终态、当前ordinary-source 67 warm+0 measured，native PID1447975/出生已核实，完整输出独立审计未完成。外层/计算分别独立unit，RuntimeMaxSec扣除900秒恢复，post先清理自有计算再原Supervisor恢复；未变更原权重、精度、步数、noise或正式计数。整体仍4/5，不把实时flags当完整动作独立接受。 |
| 2026/10/24 09:41:37 | 本机监听配额耗尽下恢复同一个看板，不改系统或实验 | 旧看板进程缺失；新启动被inotify配额ENOSPC拒绝。公共服务改为单状态文件stat轮询，12检查含原子替换、错误恢复和关闭通过；现PID4674及出生/argv、HTTP/证据路由实测通过。未修改sysctl/Orin应用，不启动实验；accepted run仍原Smol匹配功耗补充、formal4/5，目标容量仍为01:36快照22264460KiB<24GiB。 |
| 2026/10/24 09:36:22 | 通用Norm完整数值通过，但冻结计时少1条且收益不一致，不采用 | 单候选CUB/float4与当前NV插件比较，786432值MSE1.7843e-15/maxabs1.907e-6通过严格门槛。20TRT进程实际20479/20480，resident-1-after仅1023，拒绝正式计时；resident44.866573/43.539719us有两对负收益、IO296.315950/299.373519us有四对负收益，均仅pilot。不构建完整模型、不修改原主表或旧Norm慢6.43%。18检查、质量43文件/7身份、计时112文件/22身份终态通过，两unit空cgroup、Java未变；最新22264460KiB仍<24GiB，CogACT不重跑、正式4/5且Goal/D4未完成。 |
| 2026/10/24 09:02:16 | 普通权限SDK层计时跑通，保留旧Norm负收益及严格范围 | 无需请求被拒CUPTI；两冻结prefix引擎共1254运行期完整输出比较、66保存输出对独立exact，SDK层46/646、Reformat37/205且均值合计0.363006/10.075274ms，上下文需求438248960/155889152bytes。层数非kernel数、层和非E2E、需求非物理峰值；不新增模型或精度资格。217目标hash/11身份终态、Java未变、15检查/PNG-PDF通过；5条兼容warning保留，粗设备字段相同不证明原因。最新可用22885500KiB<24GiB，正式仍4/5、Goal未完成。 |
| 2026/10/24 07:53:12 | CUPTI两条观测路径权限均拒绝，不将空日志当零内存 | Activity注册/独立Callback订阅均35，三编译命令成功但原生程序分别112/122退出；未执行声明分配控制或模型，API头/库版本14一致。52目标hash/9终态出生身份通过，两ownunit空cgroup且Java未变；9边界检查通过不等于采集成功。未提权或改驱动/SDK，sudo只读列表需密码；余下私有分配/物理峰值仍未知。可另评估普通TensorRT接口，不提高CUPTI为推理门槛；CogACT固定24GiB不足、正式仍4/5。 |
| 2026/10/24 07:01:25 | 完整匹配功耗补充已审计，原应用未停启，整体仍4/5 | 15进程/15360测量/34560完整动作exact，2423原始读数/1854功耗窗合格。常规/共享/replay功耗38.438594/38.552769/42.267198W，replay更高，时延132.803564/132.111825/112.148095ms；不推算精确每次能耗、不替换旧主表。18检查、192目标hash/33身份终态通过，自己cgroup空、Java942765/942805未变。新5次内存约22.4GiB<固定24GiB，CogACT不重跑、不停后台；另功耗模式待五核心模型后。 |
| 2026/10/24 06:50:17 | 匹配功耗14/15进程完成，等最后常规路径收尾 | 14336measured/2279功耗行，既有unit及collector保持真实出生/argv，未启动其他GPU工作。最后进程/整批输出/功耗/终态闭包仍待核验，不用已完成进程代替完整cohort。主表四模型已独立重算、补pi0.5且旧数据未改；CogACT容量和其余D4缺口不变，正式4/5。 |
| 2026/10/24 06:44:08 | 补齐pi0.5主表并独立重算四接受模型全部原始时延 | 57600原始请求/51200measured逐组复核5×128+1024，mean/p50/p95/p99/max/总体std/吞吐与原封存统计全部匹配；14CDF PNG/PDF按生产receipt重hash。主/质量CSV和Markdown导出，14实际/变异测试通过。没有新模型推理/完整tensor-byte核验，CogACT正式数字留空，功耗补充不替代旧主表；负收益、两OpenPI的无replay/Arena口径及全部尾部保留。功耗同unit仍11/15进程采样，正式模型4/5、Goal未完成。 |
| 2026/10/24 06:33:19 | 匹配功耗7/15进程完成，当前baseline继续测量 | 8071measured/1262原始功耗行，实际unit1171108/collector1171318/当前native身份复查，未重启。首完整2304动作tensor独立exact，15源/真实窗口/实际live归档拒绝检查通过；原完整15worker审计和新功耗/源/进程闭包已准备，仍待整批结束。原Java不变，不改电源SDK，不宣称精确每次能耗或整批通过；正式模型仍4/5，CogACT固定24GiB前置与待批范围不变。 |
| 2026/10/24 06:20:28 | 匹配功耗首轮3/15完成，首完整2304动作张量独立exact | 三模式第一轮完成3072measured/492功耗行，原controller真实出生/argv仍活跃，第三native已终态属轮次间隙不重启。首ordinary1152calls/2304tensor全exact，norm0/1/native0，首功耗窗130样本，14源/实际变异拒绝检查通过。原整批审计拒绝实际running cohort，不因部分质量通过发布全批功耗/能耗；原始错误本机reference目录和过迟guard测试预期的拒绝均保留，未换baseline/未改原审计器。仍需完整15进程/34560tensor/功耗/终态证据，formal4/5。 |
| 2026/10/24 06:14:25 | D4同完整模型匹配功耗批次已真实运行，非五模型新验收 | CogACT固定24GiB不足且未获新临停范围，转做可独立推进的D4。Smol原7engine/模型输入noise/worker/collector不改，三模式各5×128+1024，读既有jtop非重叠输入rail，5秒warmup观测保护，不推算精确每次能耗。真实unit/monitor1171108/collector1171318/native1171530出生/argv复查，首744measured/129功耗行，原两Java保持运行。预计采集35–45分钟后独立输出/功耗/终态审计；正式仍4/5，不替代旧时间或降低CogACT门槛。 |
| 2026/10/24 06:01:45 | 4实际refit构建完成，仍不满足精度可观测性，不计时 | 原4graph/profile与FP16/noTF32/OBEY不变，仅加refit并单列诊断元数据。所有构建exit0但0组完成原typed精度覆盖，未推理/新ABI/计时、不复用104输出证明新engine。9测试、55目标hash/10身份终态通过，自己的unit cgroup为空，原Java未变。该负结果不补齐D4，也不改变CogACT因24GiB资源待新完整cohort，formal保持4/5。 |
| 2026/10/24 05:58:22 | D4固定编译诊断源准备通过，不计为新性能数据 | 仅4原算子图/profile追加TRT refit构建标志，6源测试通过；8GiB入场/2GiB运行线、256MiBworkspace与360s阶段边界不变。无模型推理/计时、原应用停启或量化搜索。CogACT完整批次仍因24GiB资源待恢复，不重跑109部分cohort。 |
| 2026/10/24 05:48:40 | CogACT不完整批次输出/资源/清理独立核验完成 | 首完整1152调用/5760tensor全exact，normalized0/1/native0、运行floor4GiB通过；下一启动前24GiB准入失败，不启动第二进程。5788路径SHA和5任务身份终态、8反例/实际检查通过，原JavaPID未变，post cgroup空。保留1/10/1024measured为部分证据，不拼接/补写/降阈值/自动重试，无整批CDF或加速比。当前需要稳定空闲资源或先评估明确新临停范围，申请未答复前不动后台，formal接受4/5。 |
| 2026/10/24 05:28:18 | 首完整进程通过输出核验，整批被下一内存准入拒绝 | 首ordinary1152调用含128warm/1024measured及5760真实输出独立全exact，normalized0/1/native0、资源floor4GiB通过。下一case启动前collector正式内存入场失败；结束环境约19523884KiB及后续19.4/19.23百万KiB低于24GiB固定线，未记录失败瞬间精确值不伪造。unitfailed/MainPID0/post空cgroup，原JavaPID不变。仅1/10完整进程不可正式验收，不重启/降线/擅停后台；已问稳定空闲资源或明确额外临停评估范围，formal4/5。 |
| 2026/10/24 05:21:01 | CogACT完整正式batch已启动，原Java不停启 | 现场可用26.97/26.79百万KiB满足固定24GiB，无需行使未批准临停；原Goal已要求正式推理，纠正把停服专属授权泛化为所有计算前置的过宽门槛。services_stopped=false/control=none，不伪造新批准或复用过期6h。实际目标validate-only通过后启动自有4h compute unit；0/10终态，首128warm+481measured/609calls，三进程PID/argv匹配、live gates通过，原JavaPID不变。全57600输出/资源/无停启/终态与CDF审计仍待，formal4/5。 |
| 2026/10/24 04:42:17 | Orin自有CPU单元清理机制三case通过，非正式模型实验 | UID1000现有systemd245user manager运行新命名瞬态CPU任务，无root/原应用停启/模型/GPU操作。正常、6s超时及自身main SIGKILL后，独立session子进程清理、post cgroup为空，旁路存活并自行回收；最终3unit消失，原JavaPID不变。20证据和最多4h纯预算检查通过，不代表真实collector/lease/restore或GPU故障恢复资格。原6hbuild/pilot权限已到期，正式独立授权、生产服务窗集成和fresh预检仍缺，formal4/5。 |
| 2026/10/24 04:03:29 | 正式24GiB/4GiB与五输出源合同离线验证完成 | 107新源按声明连接准入/guard参数/安全floor/元数据/独立审计，保留旧16/2默认及全部原浮点/opaque/计数/恢复逻辑，整源inverse通过。83新源和47旧源检查通过，旧归档身份对新源的首次拒绝保留且原身份检查未删改。160既有实际pilot输出/上游trace与93真实字节alias验证，不增加推理或样本。不可运行template保持human授权/lease/window空，新源未部署目标；正式独立授权、最终原Manager窗和fresh预检仍缺，formal4/5。 |
| 2026/10/24 03:17:53 | CogACT常规独立复验通过，关闭正式前一项门槛 | 原6h绝对截止内一次106短pilot，仅原两Java临停75.825s后原Manager恢复。24GiB准入27716920KiB，221样本最低10374132KiB>4GiB；16请求80完整tensor全部exact，主Gate0/1、native/RNG/I64 exact。127目标路径含16共享资源16.45GB重哈希和3进程终态通过，新JavaPID942765/942805及原Jar/cwd/argv/config独立确认。26测试/19子测通过；正式源草稿仅补复验/typed审计，保留24/4容量，formal授权/窗口/fresh预检仍空，正式4/5不变。 |
| 2026/10/24 02:44:26 | Attention完整动作核验失败，禁止进入正式计时 | 105真实整prefix构建/双完整七阶段十步执行完成，34输出对独立审计：normalized最坏MSE1.2796944e-5>1e-5，仅sample5失败，cos通过不抵消；native误差单列。现有同源重建控制匹配输入/IR/Plan/其他六engine，对控制仍1.2985516e-5失败，不复用其baseline擦除原结果。45回归、88目标路径SHA与10进程终态通过，原两JavaPID/Manager不变。无正式E2E/集成加速比，VLA正式仍4/5，CogACT独立授权仍待。 |
| 2026/10/24 02:28:27 | Attention整模型prefix候选真实构建中 | 105只构建原源证明候选，不改原模型其他stage/噪声/精度/步数。原guard/builder SHA和现场PID/start匹配，准入20532756KiB；原12GiB启动/2GiB运行线及900秒超时不变，两Java保持运行，无CogACT正式采样。28source回归通过，尚无新engine或全动作/计时通过。 |
| 2026/10/24 02:17:36 | Embedding/RoPE真实输出exact，内部精度未认证且未计时 | 复用原固定请求/精度与8GiB微基准门槛，不临停服务。两组真实build/infer的3输出对159360值全byte-exact；4原生SDK查询排除RoPE内部slice别名造成的外部ABI误判。数值/精度逻辑未放宽，opaque fusion仍让两组precision-unqualified且未计时。66测试、160目标文件和26进程退出核验通过，原两JavaPID/Manager不变。五模型正式仍4/5，CogACT正式授权待答复。 |
| 2026/10/24 02:00:00 | CogACT五输出核验器47测试通过，formal仍4/5 | 原32浮点审计器无法检查U8/I64；103按公共dtype通用扩展，整数/opaque不转float且全字节exact，原MSE/cos、native规则、协议和原源SHA保持，整源inverse通过。27旧+20typed测试及57600tensor模拟覆盖已封存；模拟/16帧pilot不算formal。只读确认两Java原Supervisor Running，无新服务临停或模型执行。独立formal权限及ordinary复验/资源准入仍待，原6h权限不复用续期。 |
| 2026/10/24 01:41:07 | CogACT完整双路径质量pilot通过，pilot5/5、formal4/5 | 原两Java范围内实时24GiB准入恢复到26080124KiB，16对80完整tensor父独立全字节一致，normalizedMSE0/cos1且native/U8/I64 exact。75.809s窗口原Manager恢复审计通过，未停额外daemon，旧拒绝/负例保留。H20同112输入补充normalized13/16诊断通过且阈值不改。正式草稿未激活，独立正式权限待答复，额外ordinary复验资源拒绝已恢复服务；不能用不同CPU服务状态的pilot时间算加速比。 |
| 2026/10/24 01:12:37 | CogACT常规完整模型16帧已执行，Arena因资源准入待额外授权 | 两完整Session实际ARM64编译，同16资源fullSHA且regular硬链接共享；常规16真实帧80tensor父独立finite/hash/dtype/shape/11draws/RNG末态通过，sameOrin参考已冻结，但没有跨模式质量通过。Arena单独及两Java临停后均未达到固定24GiB，worker未启动，原Manager服务已恢复；其他daemon约6GiB私有PSS不擅停，额外许可已问。MSE/cos及样本数/原截止未改，正式覆盖4/5。 |
| 2026/10/24 00:35:50 | 真实12engine I/O及4provider query/cleanup核验通过，原服务已恢复 | 13串行query全部exit0，169.697s终态；源与实际I/O名称/类型/shape/LINEAR匹配，control workspace0/align1和checked380/align256由实际SDK得到。Optional扩展缺失明确不支持context/capture，未伪造Sync/ports/host峰值。原Manager/Jar/argv/cwd/config及进程退出独立审计通过。仅元数据/资源生命周期证据，尚未运行完整Session或动作，正式4/5不变；原6h截止不续期。 |
| 2026/10/23 23:50:44 | 12/12真实引擎已编译并恢复原服务，provider ABI兼容修复重编就绪 | 原窗口3803.442s终态，controller消失/两Java原Supervisor恢复并新鲜检查。真实loader HOST检查发现静态审计未覆盖的zero-cap可选sidecar和checked命名空间冲突；按原规范修复provider，公共validator/算法不变，新4库与原生probe实际ARM64编译。28本机保护/请求测试通过，但真实engine/provider query及完整Session/动作/pilot/正式仍待。原6h绝对截止不续期，正式4/5不变。 |
| 2026/10/23 23:21:27 | CogACT引擎构建7/12，后续查询保护窗口本机测试通过 | prefix000..006已真实编译，prefix007与原controller指纹核验；仍无完整模型/动作通过。后续query/pilot控制器仅替换任务语义，所有权/信号mask/嵌套清理/原管理器恢复保留且源inverse；23本机测试拒绝续期、活前窗、错误布尔和fake模型通过。未启用新目标服务窗或GPU查询，正式采样另授权。 |
| 2026/10/23 23:03:17 | CogACT引擎构建完成4/12，第五段进行 | prefix000..003实际构建完成；prefix004 builder649987/start24960156、controller621796/start24821585及argv复核通过，两授权Java仍原Manager临停。原6h截止不延长；完整query/ABI/Session/动作/pilot及正式尚待，4/5正式覆盖不变。 |
| 2026/10/23 22:59:09 | CogACT真实引擎构建完成3/12，第四段进行 | 第三段prefix002生成1574072870B/SHA d512d484...；第四段builder643463/start24926523、controller621796/start24821585出生指纹复核，两授权服务仍原Manager临停。保持原六小时截止与恢复保护，完整模型/动作/pilot及真实provider ABI尚未验收，正式覆盖仍4/5。 |
| 2026/10/23 22:54:41 | 2/12引擎已编译；16Region绑定需按编译后名称ID映射 | prefix000/001实际引擎1.528/1.751GB，prefix002出生指纹复核；仍是build-not-validated。静态ABI/typedports/排序审计及10父回归通过，control00/01/02映射compiledID1/3/0而非source0/2/1，checked4/prefix5..15不变；真实workspace/caps/engine external I/O/Session与完整动作仍待。原6h绝对截止和原Supervisor保护继续有效，正式4/5不变。 |
| 2026/10/23 22:47:24 | 首完整阶段引擎构建通过，1/12；4原生组件库完成ARM64编译 | prefix000实际363.542s/1.528GB引擎生成，4765层检查确认旧失败MatMul139 FP16输入/结果和FP32回转，不是微图替代完整stage。prefix001正在构建，原6h截止不延长。3control+checked转换4库真实ARM64/CUDA11.4编译8命令exit0，但未加载或CUDA功能核验；输入16/112真机exact已独立审计。完整provider/Session/动作/pilot/正式仍待，4/5不变。 |
| 2026/10/23 22:38:12 | ARM64输入组件通过；显式半精结果配置重建启动 | 实际native输入16帧112tensor父逐字节exact，564源码/数据和二进制fullSHA通过；full runner仅ARM64编译未运行。通用profile只改523 half-dense结果输出，96源FP32 dense及Erf/Norm等不动；32父测试与24目标小文件核验通过。原6h绝对截止不延长，重试服务窗relative20004s含300s恢复预留，首段TRT真实构建中，0/12闭合；完整Session/动作/pilot/正式均待。 |
| 2026/10/23 22:22:00 | 首构建格式失败已定位，原服务恢复核验通过 | 75.206s编译报MatMul_139无可用格式；98.270s保护窗口终态，原Manager/JAR/cwd/argv/config与其他服务行独立核验通过，所有owned构建进程消失。3个[1,16,256,72]×[1,16,72,256]真机微图验证：FP16计算+FP32输出有/无BF16Round都失败，FP16输出含plugin可构建；下一步显式profile结果舍入后重建，数值门槛不变。仅机制定位，不计模型通过或全FP32回退。 |
| 2026/10/23 22:13:07 | CogACT首段真实TRT构建已启动，最多6h临停窗激活 | 59源/请求文件实际目标SHA/size一致，47回归及2独立中断边界测试通过；两Java由原Supervisor临停，窗口controller及builder/trtexec出生指纹已复核。24GiB准入及0可见GPU持有者通过，268进程权限缺口不称独占证明。当前0/12构建闭合，模型/质量/正式仍未通过，整体必需模型4/5不变；完成/失败/中断恢复原服务。 |
| 2026/10/23 22:02:30 | 构建窗口42本机测试通过，其余阶段源传输中 | 新窗口绑定实际授权和源请求，最多6h含300s恢复预留；失败/中断/超时清理及原管理器恢复路径已用NONMODEL manager与CPU子进程验证。现场两Java仍在原Supervisor运行；独立审查及完整文件就绪后才激活窗口，不宣称目标构建或模型通过。 |
| 2026/10/23 21:37:20 | CogACT首段源/3.037GB数据及原生源包已上板并核验，尚未构建 | prefix-stage000的graph/data/profile/compaction已传目标，连同既有5工具/plugin全部9项request输入目标fullSHA/size/stability一致；3原生源码archive目标SHA/size匹配。目标最初102.27GB空余满足全部47GB搬运准入。本阶段没有编译/模型/GPU运行或新服务临停，6h授权已记录，恢复守护及fresh管理/资源检查后才激活窗口。正式仍4/5，CogACT未验收。 |
| 2026/10/23 21:14:46 | pi0.5正式联合验收通过，必需模型4/5；转CogACT | 十进程/10240 measured/23040完整tensor、12057遥测/58325守护、原Supervisor恢复和230目标文件SHA联合通过。首闭包工具错误保留，SDK角色专门修正31新旧测试/实际目标哈希通过，不改模型输出或门槛。普通1258.792712/Arena1256.955605ms仅0.146%差异，无replay，前两轮负收益/p50变差/std不达1.2ms保留。CogACT源部署包和12请求已主机验证；新授权最多6h构建/pilot临停尚未激活，正式采样另授权。整体Goal未完成。 |
| 2026/10/23 20:37:33 | pi0.5已9/10进程，20736完整tensor独立一致，最后Arena运行中 | 九完整进程含全部warmup输出独立逐字节一致，主Gate MSE0/cos1及native0；10853遥测/52537守护采样线通过，最小18671116KiB、GPU最大57.062C，9.622s间隙与权限缺失保留。最后Arena worker470796/start24090919真实活跃，72预热/0测量；collector和原管理器恢复守护身份匹配，两个授权Java仍临停。所有10进程整批输出/资源/服务恢复/闭包/CDF联合终审未完成，正式仍3/5。 |
| 2026/10/23 20:31:46 | pi0.5八进程18432tensor一致；CogACT完整原生源部署包主机闭合 | 八完整进程全输出含warmup独立字节一致，主Gate0/1及native0；9610遥测/46641守护采样线通过，9.622s间隙/权限保留。第九ordinary441966/start23942137实际活跃840测量，正式仍3/5，恢复/终审待。CogACT实际16region源IR/25命名跨段路由/42父测试、三控制105父注入测试/840控制输出一致、完整raw runner主机链接及112原输入独立字节核验通过；38别名/11package测试。标签规范化source/runtime schema差异显式记录，尚无目标模型验收。 |
| 2026/10/23 19:53:17 | pi0.5七进程16128tensor一致；CogACT实际神经step和typed IR源核验推进 | 七完整进程全输出含warmup独立字节一致，主Gate0/1及native0；8417遥测/40837守护样本采样线通过，9.622s间隙/权限保留。第八ordinary413431/start23795680真实活跃，181测量；原管理器恢复仍待。CogACT实际神经step25LN/77dense源重放、typed IR独立operand/inverse及65组合测试、完整输出76父软件测试通过，全部目标未验收。19:53条件排期：必需VLA/D4预留2-4天，含可选Qwen暂4-7天；CogACT首个完整Orin pilot后修正，不是完成承诺。 |
| 2026/10/23 19:28:11 | pi0.5六进程13824tensor一致；CogACT完整step控制源/CPU核验通过 | 六完整进程含全部预热输出独立逐字节一致，主Gate0/1，native0；7230遥测/35053守护样本通过固定采样线，间隙/权限保留。第七Arena385139/start23649669真实活跃，145测量；原管理器恢复仍待。CogACT29控制节点/200输出exact，原始77dense及全部权重保留；43专项、prefix17父测试/11段inverse及transport66父注入测试通过，尚无目标完整模型结果，正式仍3/5。 |
| 2026/10/23 19:17:48 | pi0.5五进程11520完整tensor独立一致，第六进程818测量 | 五个闭合进程含全部预热输出独立逐字节一致，normalized MSE0/cos1、native补充0；6028遥测/29204守护样本通过固定采样保护线，最小20087416KiB、GPU最大56.781C，5.992032864s间隙与权限缺失保留。第六Arena worker356490/start23502083真实活跃，128预热+818测量；collector与原Supervisor恢复守护身份一致，两授权Java仍临停、其他受管进程不变。正式覆盖仍3/5，完整cohort/CDF/闭包/恢复未验收。 |
| 2026/10/23 18:42:09 | pi0.5四进程9216tensor通过；CogACT系数核100私有输出exact | 四完整进程全部输出/原始warmup独立字节一致、主Gate0/1、native0；4821遥测/23335guard样本保护线通过，最小20668820KiB，间隙/权限保留。第五worker327698/start23353929真实活跃，128预热+296测量。源保留12节点F64系数核两输入×十合法索引共100输出exact，13编译/溢出/广播/旧源码回归通过，非神经step或Orin模型通过；正式仍3/5，完整cohort/CDF/恢复未验收。 |
| 2026/10/23 18:21:54 | pi0.5三进程6912tensor通过；CogACT主机控制核/输入补齐 | 第三闭合进程正常退出，前三6912完整tensor独立exact、主Gate MSE0/cos1/native0；3624遥测/17532守护按采样合同通过，间隙/权限/无效值保留。第四ordinary299173/start23207478真实活跃，500测量在线正常。CogACT缺失形状源证明、两CPU控制核20输出exact及70父测试，原生112输入exact+16受检零文件IO窗口+43测试冻结。仅主机，不冒充Orin/完整模型；继续prefix/sourcepartition/CUDAtransport，正式总验收3/5。 |
| 2026/10/23 17:48:59 | pi0.5两进程4608tensor独立通过；CogACT原生输入主机112tensor exact | 正式2/10，第三worker270557/start23059580实际活跃，128预热+101测量在线正常。两完整进程主Gate/全字节及2415遥测/11673守护审计通过，保留5.798869152s间隙/权限限制，仍非整组。新版终审78父测试+真实授权pending正确，CogACT26源测试及112完整输入主机exact、旧320+23API回归通过，未在Orin运行或改权重。正式验收仍3/5，服务恢复守护继续。 |
| 2026/10/23 17:20:49 | pi0.5首完整进程2304tensor独立通过，继续1/10采样 | 首进程exit0且PID已退出，128warmup+1024measured、2304完整tensor独立逐字节一致，MSE0/cos1/native0。1203遥测/5811guard样本，保护线通过；5.408544256s最大间隙、权限/无效CV读数保留。第二worker241684/start22911629真实活跃、63预热，服务恢复守护正常。selected-case不是整组正式，仍3/5验收，不提前主表/CDF。 |
| 2026/10/23 17:08:34 | 首518正式测量正常；201软件终审回归通过，组件状态纠正 | 实际collector/worker及临停服务守护继续正常，在线主Gate通过不冒充全量独立审计。201项NONMODEL回归通过，旧closure对新native构建收据尚需按schema扩展；不在采样期间读大资源。OpenPI两模型真实绑定/host准备组件已完成，pi0.5正式仍运行、总验收3/5。CogACT仅主机并行，无第二个Orin GPU任务。 |
| 2026/10/23 17:00:14 | pi0.5正式首120测量正常，H20补充失败分列 | 实际collector/worker身份继续一致，128预热+120测量在线Gate正常，MemAvailable23271812KiB。H20完整原raw/noise强对齐补充normalized MSE0.01640955/cos0.89430129、0/16，父重放结果逐字节一致，28软件测试通过；不影响same-Orin主Gate，不宣称源BF16等价。正式3/5，服务窗口继续临停守护，完整输出/CDF/资源/恢复终审仍待。 |
| 2026/10/23 16:55:19 | pi0.5 paired主Gate通过，新授权正式采样已实启 | 128完整tensor含warmup独立全字节一致，normalized MSE0/cos1、native补充0。两模式各5进程×128+1024固定，首worker13预热，身份PID/start/argv已核验。两Java按本次授权原Supervisor临停，其他程序/配置未变，守护结束/失败/中断恢复。约4.1小时纯采样加终审，正式验收仍3/5；无replay/显著收益或H20等价声明。 |
| 2026/10/23 16:24:36 | pi0.5完整部署pilot独立通过，参考已冻结，正式仍3/5 | 32请求/64完整动作tensor全finite且SHA/shape/dtype复验、16组warmup/measured重复字节一致。普通路径16测量均值1270.853ms只作pilot；不是paired质量。参考77f2cf... same-Orin/device/schema/processor/noise/policy冻结；后续优化完整动作Gate后才正式。runtime replay当前provider声明不足，不翻能力标记或冒充capture。 |
| 2026/10/23 14:18:43 | 五前缀构建/ABI齐备，step串行启动 | stage003 engine1458999614B/SHA2118428d...、585.227秒、132端口，未推理；step guard23307/start21828203启动，准入29480104KiB。五前缀实际engine后续本机5.642GB缓存逐文件SHA/字节匹配，不是占位，完整Session/动作仍待验收。 |
| 2026/10/23 14:30:12 | 清理旧H20第三方安装包缓存3.825GiB，实验证据未删 | 17个torch/triton/nvidia wheel逐文件包名/版本/SHA/字节/stat核验并保留清单后删除，日志和已安装环境不动；无模型或测量数据删除。首轮纳秒整数经JS序列化损失精度被严格前置检查拒绝、删除0份；plan002改十进制字符串并重新hash后通过，原失败保留。实际wheel清理4107051008分配字节，后续离线重装需重新获取同版本/哈希。 |
| 2026/10/23 14:06:57 | stage002构建/ABI完成，stage003串行启动 | stage002 engine1389687337B/SHA3c25d5dd...、577.228秒、95个端口、context需求268633600B仅元数据，未推理；stage003 guard8681/start21757589已启动。完整模型/动作未通过，正式3/5。 |
| 2026/10/23 14:10:11 | 用户指定data目录已清理，系统盘可用约78GiB | 仅删除明确授权/home/zhangzimo/data，约74998190080已分配字节，确认无软链接/子挂载及删除后不存在。唯一Codex工作树仍在使用且dirty，未删除；历史100GiB实验仍有pi0.5数据合同与论文原始证据引用，未整目录删或声称归档完成。 |
| 2026/10/23 13:54:34 | consumer构建/72端口ABI完成，串行进入stage002 | consumer901004425B/SHA089a2ab8...，513.255秒；context需求476546048B是元数据，未创建context/未推理。旧三个实际PID均缺失、guard终态0后执行检查，再启动stage002 guard4185853/start21683303。两Java运行、固定阈值不变；三个阶段构建/ABI不等于完整模型通过，正式3/5。 |
| 2026/10/23 13:43:24 | 共享段构建/ABI完成，串行进入consumer构建 | 共享段engine484582813B/SHA0e388152...，139.065秒，3个LINEAR端口/context需求17380864B仅元数据；无context/推理。原guard终态0后才检查并启动consumer guard4171789/start21616295；两Java运行、固定内存阈值保持。完整模型/动作未验收。 |
| 2026/10/23 13:39:04 | pi0.5共享视觉段目标构建启动 | guard4165845/start21590280，源镜像驱动请求，16GiB准入/2GiB安全线保持，MemAvailable30756784KiB。两Java运行，目标GPU串行；源200-token/cache968/十步/native0/2不改。其他三请求仅准备，完整Session/动作未验收，覆盖3/5。 |
| 2026/10/23 13:33:55 | pi0.5首段构建/18端口ABI检查完成，非完整模型通过 | engine1407746956B/SHA abaa024c...、228.998秒；原构建/检查guard均child-exited/0，934内存采样最低18681448KiB。8输入/10输出LINEAR、200-token/cache968，context需求43588096B仅元数据非峰值；未创建context/未推理。共享段/consumer的2.76GB源9文件已传板回读一致。两Java保持运行，不复用旧服务窗；下一步其余引擎/实际Session及完整动作核验，正式3/5。 |
| 2026/10/23 13:27:50 | pi0.5首段目标构建启动，尚无模型验收 | 原source/profile及2802293696B权重pin复核，serial/boot/SDK/MAXN/温度/进程记录保留；MemAvailable28235048KiB，无可见GPU FD持有者但权限不全。guard4151193/start21522950按16GiB准入/2GiB安全线执行，两个恢复Java保持运行。Python3.8不支持Path.readlink的首轮只读快照错误已保留v1源码，v2用os.readlink通过；非模型失败。 |
| 2026/10/23 13:19:48 | pi0正式D1-D3联合验收，必需模型覆盖3/5，两个服务已恢复 | 十进程11520请求/10240 measured、23040完整tensor全字节一致，normalized MSE0/cos1/native补充MSE0；资源/原管理器新进程恢复/216冻结文件联合通过。均值1129.870→1125.379ms仅约0.4%差异且两轮负收益，不证显著加速/官方BF16/机器人质量。std目标失败、H20补充失败保留。collector和worker均终态；下一模型pi0.5尚无目标engine/推理，公共D4与CogACT仍未完成。 |
| 2026/10/23 12:51:26 | pi0已9/10完整进程，20736动作tensor独立逐字节一致，最后进程运行中 | 10368已完成请求含warmup/measured完整normalized/native输出独立核对通过，MSE0/cos1；最后worker4103897/start21288018实际live，128warmup+5measured，累计9221/10240。9909遥测/47181守护采样最低MemAvailable22274308KiB；权限与缺失不补零。两Java仍仅授权临停、collector/guard live。最后进程/全量审计/CDF/恢复/目标文件终审尚未完成，正式模型仍2/5。 |
| 2026/10/23 12:44:19 | pi0已8/10完整进程，18432动作tensor独立全字节一致，第九进程继续运行 | 前8worker9216请求含全部warmup/measured的normalized/native输出独立核对，MSE0/cos1；第九worker4078052/start21156280真实live，128warmup+798measured，累计8990/10240。8818遥测/41961守护采样最低MemAvailable22696620KiB，功耗不可读不补零。两Java仅授权临停、守护与collector live；全批/CDF/恢复未验收，正式模型仍2/5。 |
| 2026/10/23 12:42:42 | 联合验收58软件测试/冻结文件核验18测试通过，真实最终证据尚未齐备 | 最终联合器只允许同一终态campaign/spec/protocol、十进程完整动作/资源/服务恢复/时延输出/冻结产物闭合；whole_goal_complete恒false。冻结读器在真实Orin运行中返回pending、无大文件读取或模型/服务操作；source002已适配Python3.8并补三项cohort hash绑定，只有结束后才验全部engine/provider/runner/SDK/raw-noise/reference。未伪造final pass。 |
| 2026/10/23 12:17:39 | pi0已7/10完整进程，16128动作tensor独立全字节一致，第八进程运行中 | 全部已完成的8064请求/16128 normalized/native完整输出含warmup均独立核对，MSE0/cos1。第八worker4052414/start21024635实际live，128warmup+547measured、累计7715/10240；7729遥测/36745守护采样最低MemAvailable23072872KiB，权限/缺失保持。两Java仍仅授权临停、守护/collector live。全批/CDF/恢复尚未验收，正式模型覆盖2/5。 |
| 2026/10/23 12:15:29 | pi0.5两源组件2.76GB已实际发布，10-region全未绑定IR及父逆向重放通过 | 复用原提取/分区/streaming compactor/common repeat_invocation，shared1400节点/963512320B、consumer652节点/1799438336B，248+42范围前后及输出SHA一致；旧source/profile不改。9到10region组合的3fresh调用无memoize，native0/2、200token/cache968/十步及whole-Module控制保持，父19软件测试/5 JSON/32pins复验通过。源payload在tmpfs；无engine/target/数值/NaN消除/性能通过声明。 |
| 2026/10/23 11:25:46 | pi0采样过半：5/10完整进程、11520完整动作tensor独立全字节一致 | 全部已完成worker含warmup/measured的normalized/native动作独立核对通过，MSE0/cos1；第六worker4001829/start20763150实际live，128warmup+105measured、累计5225/10240。5561遥测/26394守护采样最低MemAvailable24243484KiB；权限/缺失/无效传感器保留。全批/CDF/服务恢复仍未验收，模型正式覆盖2/5。 |
| 2026/10/23 11:21:29 | 服务窗最终审计和只读观察器已准备，36项父软件回归通过，真实恢复尚未发生 | 明确匹配native --job控制器/真实collector/不可变lease与可变恢复记录；新birth而非新PID判恢复，按真实inspection顺序验证raw actions后仅内存投影给旧通用合同。未改封存helper或raw证据，未执行SSH/服务操作，当前真实snapshot正确pending。 |
| 2026/10/23 11:21:29 | pi0.5真实重复子图source proof通过：3×1400节点、652 consumer，父只读复验一致 | 数据依赖发现真实frontier，类型/attrs/opset/每节点precision/参数range SHA均精确一致，closure不重叠且内部不外泄；15软件测试和父3份全字段JSON-canonical重放/31pins通过，未忽略字段。仍仅metadata candidate，无新权重/图/IR/绑定/engine或GPU/Orin执行，未证明目标NaN消除或模型质量。 |
| 2026/10/23 11:05:28 | pi0已4/10完整进程，9216动作tensor及采样资源独立核对，第五进程运行中 | 两mode各2个完整worker，含预热/测量的完整normalized/native动作均逐字节一致，MSE0/cos1；第五worker3976292/start20630963实际live，128warmup+195measured、累计4291/10240。前4worker4460遥测/21155守护采样最低MemAvailable24662724KiB、GPU最高56.593C；功耗持续不可读，不产能耗结论。仅selected-case/resource，全批Cdf和恢复未验收，模型覆盖仍2/5。 |
| 2026/10/23 10:44:41 | pi0已3/10完整进程，6912动作tensor及前三进程采样资源已独立核对 | 完整normalized/native输出全字节一致，MSE0/cos1；第四worker3950841/start20499558实际live，128warmup+257measured、累计3329/10240。前三进程3365遥测/15947守护采样最低MemAvailable25159392KiB、GPU最高56.25C；记录退出RSS缺失和CV无效温度、不补零。INA3221持续PermissionError，不产功耗/能耗结论；20资源软件测试通过。仍为selected-case/resource，不是全批正式或服务恢复通过，正式模型覆盖2/5。 |
| 2026/10/23 10:22:27 | 本机授权清理完成：约5.7GiB下载缓存，磁盘恢复约8.3GiB可用 | 只清pip HTTP/wheel可重下载缓存，1133文件/254目录/6075.8MB；已安装依赖、权重、环境、代码和成功/失败实验数据未动。conda无可清压缩包；journal因sudo需密码未改。未中断Orin、未调整SDK或服务窗。 |
| 2026/10/23 10:23:25 | pi0前2/10正式进程完成，4608完整动作tensor独立全字节一致 | 两mode各完成首个1152请求worker，含全部warmup/measured的normalized/native完整输出SHA/shape/dtype/finite/主Gate独立核对通过，MSE0/cos1；仍为selected-cases、formal_sampling=false。第三worker3924745/start20366275实际live、128warmup+302measured，累计2350/10240；collector与服务守护live、仅授权两Java停止，其他受管程序不变，正式模型覆盖仍2/5。 |
| 2026/10/23 10:06:49 | native时延/CDF后处理已交付并独立复跑42项软件测试，不提前生成正式图 | 按冻结spec读取两mode，各5进程/128+1024，保留每个慢样本及逐进程rank CDF，负加速不删；明确拒绝pilot/运行中/局部审计。PNG与raster PDF仅NONMODEL绘图测试，正式数值/曲线等待完整model-data审计；本工具不验收服务恢复/资源或整Goal。 |
| 2026/10/23 09:55:57 | pi0首个正式进程完成并独立核验2304完整动作tensor，第二模式运行中 | 首worker exit0/1152请求，含128warmup+1024measured的全部normalized/native输出逐文件SHA、shape/dtype/finite与参考对比均通过，完整字节一致/MSE0/cos1。仅selected-case，formal_sampling=false；第二worker3898841/start20232961真实live、128warmup+24measured，累计1048/10240。独立审计27项软件回归通过；原两服务临停、其他服务不变，守护与collector live，完整正式验收仍2/5。 |
| 2026/10/23 09:38:37 | pi0正式批次第1进程已进入measured：128预热+297测量 | 425次完整raw请求在线finite/normalized/native/source acceptance flags正常，累计297/10240 measured；0/10完整进程、全量独立审计未完成。守护/collector/worker实际身份继续相符，两Java停止、其他受管程序不变，MemAvailable27483388KiB。看板双模式协议校验修正且本地服务重启，API HTTP200，正式覆盖仍2/5。 |
| 2026/10/23 09:31:14 | pi0新正式批次已启动：两Java原管理器临停/守护恢复，其他受管服务不变 | 当前明确人类授权已绑定新spec/lease快照；守护PID3871611/start20094526、collector3872054/start20096270、worker3872125/start20096646实际argv匹配。首42次warmup flags通过，MemAvailable27533908KiB；10进程/10240measured计划，预计纯采样约3.5–4小时。旧memory-refused批次0样本不合并、全量审计/恢复/CDF未完成，正式仍2/5；H20supp失败和负Arena pilot均保留。 |
| 2026/10/23 09:30:00 | pi0.5四段源文件和9-region unbound IR就绪，非目标模型通过 | 9589 used nodes恰一次，37输出、165LN、3176Round和806+4 dense声明保留；独立审计11,257,119,680B，仅文件缓存已同步Orin。6到9region inverse相等，native0/2、200-token/cache968/十步/F64保持，22+8项软件回归通过，无enginebuild/模型加载/forward。 |
| 2026/10/23 06:54:43 | 原whole-part01独立复现12/51 NaN，51完整输出含NaN bits与Session字节一致 | 已读取比较文件all_output_bytes_equal_session实际true，并独立逐字节复核本地51完整direct输出。原engine同18真实输入在Session外同步成功后仍NaN，非Session arena/路由特有；Task14存活arena声明范围无重叠，source pins不变。exit0仅诊断采集成功，pilot002失败保留，正式仍2/5，pi0无baseline。剩余为原engine内部数值/融合路径定位后完整raw核验，不增dense FP32例外/全FP32/quant；聚合74通过/2 CUDA opt-in跳过，父核验本轮已知GPU child均终态/源服务未停。本轮只收敛同一看板，不扩写历史报告或追加任务。 |
| 2026/10/23 06:45:49 | 单视角tail真机697输出统计finite，下一步原part01同18输入独立诊断 | results001逐项697输出/1624113332B finite==elements、NaN/Inf=0、cudaStreamSynchronize-success。rawbins因空间仅板端保留，本地完整统计而非输出字节复验。额外观测输出/单视角不证明原whole part01通过或等价；父准备原engine同18真实输入区分Session与engine融合/数值问题，SDK/精度不放宽。pilot002失败保留，正式仍2/5，无完整action/baseline。 |
| 2026/10/23 06:40:02 | 首个MLP真机17输出finite，继续尾段697输出probe定位更晚层NaN | MLP infer001 guard3663165/start19047940 exit0，17额外观测outputs全finite；CPU shadow/FP16 IO模拟各695 probes+51 outputs finite，不替代Orin结果或证明原engine数值正确。尾段697 outputs控制器3665131/start19057246已exit0，本地build receipt/执行输出待回；保留原part01 NaN/acceptance拒绝，正式仍2/5，无完整action/baseline。 |
| 2026/10/23 06:34:48 | 744同步dump定位part01图像tokens NaN，下一步为具体数值原因而非workspace/绑定 | 逐文件核验744份/767531682B size/SHA；part00十输出finite，part01第0输出transpose_330图像768tokens/196608值全NaN，48文本tokens/12288值finite。audit已记录SSA同buffer边界0不匹配，不外推为所有路由正确。最小MLP 28nodes/17outputs实际build002 exit0仅built-not-validated；CPU shadow初轮51输出finite，均不计完整模型通过。父继续数值探针诊断，真实acceptance/旧失败保留，无完整action/baseline，正式仍2/5。 |
| 2026/10/23 06:19:26 | pilot002初始化通过后首请求被source acceptance拒绝exit86，无完整action | 控制器记录3641035/start18940517退出86。父报告所有native/TRT初始化已过，native finishing返回source acceptance rejected both output ports，未发布完整action；不称32请求完成或冻结baseline。详细guard待回传，父进行untimed真实Task同步边界I/O诊断，不改/跳过acceptance；旧失败保留，D1/D2未通过，正式仍2/5。 |
| 2026/10/23 06:17:37 | Session004原生链接完成，修复后pilot002父任务运行中，仍非模型通过 | 库234344B/e7765d45...；既有Plan workspace按task正确绑定，非新增StaticArena，LoadRegion严校size/alignment/device。Module/Plan/10合同与003字节一致，28回归通过/replay19通过2跳过。pilot002仅核验控制器启动3641035/start18940517，未收终态/质量；首轮完整exit86失败及lifetime限制保留，正式仍2/5。 |
| 2026/10/23 06:12:51 | 全10区域Session已原生链接，首次raw pilot在workspace初始化退出86 | Session003库234344B/8bde9318...，链接不算D1通过。控制器/完整guard记录child3633695/start18905371退出86；workspace contract mismatch、retained_to_process_exit=true，无神经enqueue。根因是codegen未绑定既有Plan的384B/448B、align256 CUDA workspace，非缺额外StaticArena，不改零/松检查/改精度，正式仍2/5。 |
| 2026/10/23 06:12:51 | 四段actual Orin构建和ABI检查完成，全10绑定源码生成成功 | 四段均built-not-validated，inspection未创建context/未推理；all10 missing=[]。旧生成002失败为4/6空格嵌套缩进导致334项old_count=0，不是重复代码；新严格逐task guard全部exactly1且12软件测试通过，旧sealed源码/失败保留，无完整模型动作或D1/D2通过。 |
| 2026/10/23 04:38:22 | π0实际合同已绑定6/7，前缀是唯一缺失构件 | 三个实际引擎/三个原生provider合同经源图、GPU控制输出和布局证明合并，time仅移除私有未用参数。69项CPU测试通过；无前缀占位或Session。拆分后备停在custom-domain API边界，服务窗口尚待确认，模型正式验收仍2/5。 |
| 2026/10/23 04:26:44 | 控制图真实GPU核验和原生驱动编译完成，仍缺前缀与完整Session | time零输入1例/finite完整1600输入60例全部输出字节正确；驱动Orin编译，固定完整normalized主MSE/cosine门槛，native误差补充，未模型运行。实际三引擎接口已检查，前缀仍无engine，正式覆盖2/5；等待服务窗口授权并准备拆分后备。 |
| 2026/10/23 03:41:00 | 前缀完整解析后触发内存保护，停止本组而不停止业务服务 | guard记录4537条采样，优化阶段可用内存最低1438432KiB，低于2097152KiB保护线，终止本组3420513/3420527，未生成engine。非算子不支持/内核OOM；旧内部running记录未收尾，以guard终态和实际PID缺失为准。已询问扩展临停Java服务窗口，期间推进小控制图/绑定/驱动，不降低门槛。 |
| 2026/10/23 03:29:14 | 原生输入API板端CPU组件验收，接入完整神经链是下一步 | Orin原生Pillow/SentencePiece/API构建成功，32用例320/320完整张量对新鲜原生CLI与host命名配置全字节一致；23不变量与6审计检查通过，Prepare零文件IO和GPU打开。历史失败未覆盖，未运行神经模型或计时实验；正式覆盖仍2/5。 |
| 2026/10/23 03:18:49 | 前缀LayerNorm兼容构建与四路完整链集成同时推进 | 前缀001原生LN导入失败保留；通用改写165节点并保持权重/epsilon，50项回归通过。003 guard子进程3420513/start17845413实测存活；24GiB准入/2GiB采样保护只终止本组，不改业务服务。输入API板端CPU构建、Session绑定、驱动和控制图并行，未进入正式动作或CDF采样。 |
| 2026/10/23 02:56:00 | 并行输入API、IR改写链和通用精度提案均交付 | 输入内存API32用例/320张量对新鲜CLI字节一致、23原生+7审计检查通过；IR32项测试/165旧记录不变，实际artifact仍unbound。四图源dtype提案27项测试，pi0-step与已构建007全字节相同，其他图未激活。目标完整前缀/十步动作/正式采样仍是下一关，正式覆盖2/5。 |
| 2026/10/23 02:51:10 | pi0 step引擎构建成功，原生组件GPU核验完成 | 006引擎640218807 bytes/b130aa7e...，168个dense节点可见格式符合请求；编译耗时不是推理时延。5个实际Region GPU通过128冻结后处理块/89600 F64值全字节一致。完整前缀/十步动作/正式采样未验，模型正式覆盖仍2/5。 |
| 2026/10/23 02:45:00 | 源精度边界定位后进入引擎tactic阶段；原生Region实机编译完成 | 源6个FP32区域/162个BF16核心分别声明FP32/FP16部署，007 profile用于006构建，非全FP32回退或源数值等价证明。I64告警的38个切片哨兵已逐用途核实；5个真实schema provider已CUDA11.4编译，GPU和完整动作仍待验。 |
| 2026/10/23 02:32:15 | pi0完整图导入通过，转入混合精度格式最小复现 | 两处导入阻塞已越过；003-005均真实完整导入，构建阶段分别定位RoPE、attention、Shuffle的格式约束。保持FP16 dense/源数值岛，不做全模型FP32回退；输入API、Region集成、整数边界与IR/报告并行推进。完整动作与正式采样未开始。 |
| 2026/10/23 02:24:00 | pi0越过原导入崩溃，继续处理真实Slice控制量限制 | v3图937处常量只改表示不改值，真实TRT构建002不再SIGSEGV；导入器拒绝经Cast/Reshape表示的静态Slice轴。002失败完整保留，独立通用静态控制量改写正在准备，不降低动作门槛或声称模型通过。 |
| 2026/10/23 02:17:00 | 四路主机工作并行；pi0导入阻塞已定位并重试 | 并行推进导入诊断、计时内原生输入API、真实schema的CUDA Region测试、报告证据同步。Orin GPU仍串行，step001失败/GDB证据保留；常量表示改写通过11项回归后启动002，未改变FP16 dense策略或完整动作门槛。 |
| 2026/10/23 01:58:47 | RDT正式验收通过，必需模型覆盖2/5；转入pi0 | 15进程/15360 measured/34560张量字节一致；普通3065.625218ms、replay2975.587343ms，1.030258858x。全CDF、共享流负收益、46125资源采样和真实恢复证据保留，非全Goal完成。 |
| 2026/10/23 01:58:47 | Orin原生插件数值/两流replay通过，pi0 step引擎构建中 | 未改全局SDK/时间/旧runtime；新源包校验一致，CUDA11.4原生编译，直接smoke exit0并保存PID/log hash。ctest命令不在PATH，用等价二进制，不升级为完整模型通过。 |
| 2026/10/23 01:32:22 | RDT 正式采样完成 15/15 个进程 | 当前批次 rdt-raw-e2e-services-20261022-001 累计 15,360 条 measured；完整输出与正式结果仍待独立审计。 |
| 2026/10/23 01:25:19 | 两模型八份opset17阶段图准备完成，245项主机回归通过 | 外置原有Cast、仅对Gather索引要求受检I32接口、删除已证明未用的F64参数。图/权重/输出合同保留；165个原生LayerNorm/前缀及mask/time显式FP32，dense请求FP16，时间插件内部FP64。尚无TensorRT导入/真实精度/完整动作通过，所有运行时转换仍必须实际执行。 |
| 2026/10/23 00:35:41 | RDT14/15完成，32256完整张量独立一致；只剩末常规进程 | 第五轮replay完整128+1024/exit0，另2304张量/全部hash通过，1152次/5步/普通回退0。十四段43070资源行保护线通过，旧缺失两行保留、无新增。末进程3222446/start16870506匹配；共享流/replay5/5、常规4/5，正式表/CDF/恢复不提前验收。 |
| 2026/10/23 00:32:46 | RDT 正式采样完成 14/15 个进程 | 当前批次 rdt-raw-e2e-services-20261022-001 累计 14,336 条 measured；完整输出与正式结果仍待独立审计。 |
| 2026/10/23 00:10:37 | 并行准备两模型四份载体图，按正确顺序插入F64时间插件 | 3176/598与3176/670个显式BF16舍入边界保留；FLOAT仅为存储/部署表示，实际计算精度未认证。两份carrier-first step候选的512项F64频率及shadow hash与源级候选一致。受检I64/I32边界、F64输入投影/原Cast producer和目标完整动作仍待验证，不作全模型FP32回退。 |
| 2026/10/22 23:45:34 | RDT13/15完成，累计29952完整张量一致；最后两个进程继续 | 第五轮共享流完整128+1024/exit0，另2304张量/全部hash通过。十三段40114资源行保护线通过，旧缺失两行保留、无新增，最大间隙4.615秒。共享流5/5、其他4/5；下一replay3155729/start16524491匹配，正式表/CDF/恢复仍待整组审计。 |
| 2026/10/22 23:45:34 | 并行闭合F64后处理/时间特征准备，185项相关主机回归通过 | 主任务重编验证128完整后处理块/89600值字节一致；结构lowering快照修复经51项/真实大图验证；SinCos源匹配13项/20480真实时间特征shadow通过，插件仅主机编译。原生输入/Region/受检转换继续，未新增目标模型通过。 |
| 2026/10/22 23:35:09 | RDT 正式采样完成 13/15 个进程 | 当前批次 rdt-raw-e2e-services-20261022-001 累计 13,312 条 measured；完整输出与正式结果仍待独立审计。 |
| 2026/10/22 22:38:59 | RDT新批次12/15完成，累计27648完整张量独立一致，三模式各4/5 | 第四轮replay完整128+1024且exit0，另2304张量/所有hash通过，1152次/5步/普通回退0。十二段37008资源行系统保护线通过，两行旧字段缺失保留，新段无新增，最大间隙4.478秒。下一共享流3086136/start16164441身份匹配；正式表/CDF/恢复不提前验收，模型正式完成仍1/5。 |
| 2026/10/22 22:35:45 | RDT 正式采样完成 12/15 个进程 | 当前批次 rdt-raw-e2e-services-20261022-001 累计 12,288 条 measured；完整输出与正式结果仍待独立审计。 |
| 2026/10/22 22:27:59 | OpenPI两份F64源分区逐节点/权重/类型检查通过，不降精度 | pi0 3492节点完整分成55+3437，pi05 3843分成54+3789。所有源节点恰一次保留，initializer/sidecar和端口类型不变；consumer无内部F64算子输出，但未用F64 state口与producer原生执行仍待解决，不宣称TRT或模型通过。 |
| 2026/10/22 22:23:37 | CogACT四阶段及CPU参考交付，主任务复验822文件/46.26GB | 保留实际7.63B公开依赖身份、十步与完整F32/F64及精确RNG端口。270组件比较/九项guard测试通过；完整prefix join的两个cognition MSE比较失败、最终动作通过，整体failed不改。无目标/GPU性能实验；尚不能增加Orin部署/正式通过数。后续OpenPI原生后处理与Sequence通用代码继续并行。 |
| 2026/10/22 22:11:19 | 并行交付pi0/pi0.5八张ONNX，主任务独立复验114文件/27.21GB | 图及外部权重13.09GB，八图checker和输入/输出类型合同通过；完整checkpoint已重新核对，原F64/I64/BF16、48/200-token、十步来源保留。CPU导出无GPU/Orin执行；后续通用分区、native F64后处理和目标完整动作仍待验收，正式模型数保持1/5。 |
| 2026/10/22 21:57:09 | F64原生runner兼容旧冻结输入包接口，121项主机回归通过 | 额外编译/执行真实模板中的finite函数，发现并修复中间版本对新buffer方法的依赖；失败XML及首版源快照保留。最终模板只使用原字段/data接口，F64非有限值仍拒绝；不要求重做旧输入包，不替换活动RDT程序。 |
| 2026/10/22 21:51:31 | 并行补齐通用host F64/I64保真缓冲，120项主机回归通过 | 保留原始类型/所有字节，拒绝隐式降精度，增加F64 NaN/Inf及精确元素宽度溢出检查；F32/图像/文本逻辑不变。仅主机代码和软件测试，不替换活动RDT程序，不宣称F64 TRT计算、完整预处理或目标模型通过。 |
| 2026/10/22 21:39:03 | 按用户要求并行主机准备，Orin正式采样仍独占GPU | 已启动独立OpenPI(pi0/pi0.5)与CogACT主机任务，分别处理实际源资产和神经图导出准备；写入范围不交叠。主任务继续RDT采集/独立审计；主机GPU导出须协调，不新增其他平台性能实验，不缩减正式协议或质量门槛。目标完整链仍逐个验收，未新增部署通过。 |
| 2026/10/22 21:39:03 | RDT新批次11/15完成，累计25344完整张量独立一致 | 第四轮共享流完整128+1024且exit0，另2304张量/所有hash通过，ordinary1152符合设计。十一段34073资源行系统保护线通过，两行旧字段缺失保留，新段无新增。下一replay3019073/start15817980身份匹配；常规/共享流4/5、replay3/5，正式表/CDF/恢复不提前验收，模型正式完成仍1/5。 |
| 2026/10/22 21:37:26 | RDT 正式采样完成 11/15 个进程 | 当前批次 rdt-raw-e2e-services-20261022-001 累计 11,264 条 measured；完整输出与正式结果仍待独立审计。 |
| 2026/10/22 20:40:00 | RDT新批次10/15完成，累计23040完整张量独立一致 | 第四轮常规完整128+1024且exit0，另2304张量/所有hash通过，stderr无非预期REPLAY。十段31051资源行系统保护线通过，两行旧进程字段缺失保留，新段无新增，最大采样间隙4.338秒。下一共享流进程身份匹配；常规4/5、其他3/5，正式表/CDF/恢复不提前验收，模型正式完成仍1/5。 |
| 2026/10/22 20:38:08 | RDT 正式采样完成 10/15 个进程 | 当前批次 rdt-raw-e2e-services-20261022-001 累计 10,240 条 measured；完整输出与正式结果仍待独立审计。 |
| 2026/10/22 19:39:56 | RDT新批次9/15完成，累计20736完整张量独立一致，三模式各3/5 | 第三轮共享流完整128+1024且exit0，另2304张量/所有hash通过，ordinary1152为设计路径。九段28008资源行系统保护线通过，两行旧进程字段缺失保留，新段无新增。下一常规进程身份匹配；各模式仍非5进程/5120测量，正式表/CDF/恢复不提前验收，模型正式完成仍1/5。 |
| 2026/10/22 19:39:03 | RDT 正式采样完成 9/15 个进程 | 当前批次 rdt-raw-e2e-services-20261022-001 累计 9,216 条 measured；完整输出与正式结果仍待独立审计。 |
| 2026/10/22 18:42:10 | RDT新批次8/15完成，累计18432完整张量独立一致，下一共享流运行中 | 第三轮常规完整128+1024且exit0，另2304张量/所有文件hash通过，stderr无非预期REPLAY。八段24965资源行系统保护线通过，两行旧进程字段缺失保留，新段无新增。常规/replay仍仅3/5、共享流2/5；正式表/CDF/恢复不提前验收，模型正式完成仍1/5。 |
| 2026/10/22 18:40:47 | RDT 正式采样完成 8/15 个进程 | 当前批次 rdt-raw-e2e-services-20261022-001 累计 8,192 条 measured；完整输出与正式结果仍待独立审计。 |
| 2026/10/22 17:41:45 | RDT新批次7/15完成，累计16128完整张量独立一致，下一常规运行中 | 第三轮replay完整128+1024且exit0，另2304张量/所有文件hash通过，实际1152次/5步/普通回退0。七段21825资源行系统保护线通过，两行旧进程字段缺失保留，新段无新增。replay仍仅3/5、其他2/5；正式表/CDF/恢复不提前验收，模型正式完成仍1/5。 |
| 2026/10/22 17:40:12 | RDT 正式采样完成 7/15 个进程 | 当前批次 rdt-raw-e2e-services-20261022-001 累计 7,168 条 measured；完整输出与正式结果仍待独立审计。 |
| 2026/10/22 16:46:31 | RDT新批次6/15完成，累计13824完整张量独立一致，三模式各完成2/5 | 第二轮常规完整128+1024且exit0，另2304张量/所有文件hash通过，stderr无非预期REPLAY；仅复制已结束目录。六段18824资源行系统保护线通过，两行旧进程字段缺失保留，新段无新增。第七replay身份匹配；各模式仍缺3个进程，正式表/CDF/恢复不提前验收，模型正式完成仍1/5。 |
| 2026/10/22 16:42:57 | RDT 正式采样完成 6/15 个进程 | 当前批次 rdt-raw-e2e-services-20261022-001 累计 6,144 条 measured；完整输出与正式结果仍待独立审计。 |
| 2026/10/22 15:47:48 | RDT新批次5/15完成，累计11520完整动作张量独立核验一致 | 第二轮replay完整128+1024且exit0，另2304张量/全部文件hash通过；实际1152次/5步/普通回退0。五段15685资源行系统保护线通过，原两行进程字段缺失保留，新段无新缺失。第六常规进程身份已匹配，活动文件未归档。仍缺每模式完整5进程、CDF/全量审计及恢复，模型正式完成仍1/5。 |
| 2026/10/22 15:43:24 | RDT 正式采样完成 5/15 个进程 | 当前批次 rdt-raw-e2e-services-20261022-001 累计 5,120 条 measured；完整输出与正式结果仍待独立审计。 |
| 2026/10/22 14:47:37 | RDT新批次4/15完成，累计9216个完整动作张量独立核验一致 | 新增第二轮共享流完整128+1024且exit0，另2304张量/全文件hash通过；ordinary1152为该模式设计路径，不是replay失败。四段12594资源行系统保护线通过，两行进程内存字段缺失保留；下一replay已启动。全15进程/CDF/恢复未验收，模型正式完成仍1/5。 |
| 2026/10/22 14:44:08 | RDT 正式采样完成 4/15 个进程 | 当前批次 rdt-raw-e2e-services-20261022-001 累计 4,096 条 measured；完整输出与正式结果仍待独立审计。 |
| 2026/10/22 14:35:55 | INT32/非dense审计代码及58项组合回归就绪，目标缺口保持未完成 | 可分开报告全输出数值与可观测精度；真实INT32/Constant记录及旧Attention归档复查通过，未来新采集器添加生产端inspector hash但未运行。Embedding/RoPE实际exit3 pending；空dense/未知融合/元数据自一致但少样本不能放行。无新GPU任务，RDT继续采样，模型正式完成仍1/5。 |
| 2026/10/22 13:49:40 | RDT首轮三模式6912完整张量核验通过，当前3/15，下一轮已启动 | 每进程完整128+1024且exit0，replay实际1152次/5步/普通fallback0；只归档结束目录。9421资源行的系统保护线通过，replay末行缺失RSS等计数显式保留、原因未定，不补零。原额外审计失败/v1来源保留，13项回归通过，不代表全15进程或恢复验收。 |
| 2026/10/22 13:44:09 | RDT 正式采样完成 3/15 个进程 | 当前批次 rdt-raw-e2e-services-20261022-001 累计 3,072 条 measured；完整输出与正式结果仍待独立审计。 |
| 2026/10/22 13:37:32 | 按用户授权初始化本机CodeGraph索引，MCP可检索 | 初始化/索引命令exit0，MCP状态与结构查询通过；索引仅在本机，沿用现有忽略规则，AGENTS/项目配置未变。未操作Orin模型或新实验，RDT仍持续采样，正式模型完成数不变。 |
| 2026/10/22 13:27:44 | 保存ONNX完整CPU输出及原始来源pin复查通过，仍非板端结果 | 重新加载两对图，核对Embedding一个指令和RoPE两个观测的所有完整输出，原source/profile/sidecar未改。29项主机、8项看板和9项浏览器回归通过；RDT采集器/worker/恢复守护实际身份匹配，两个服务仍停止，无新GPU任务。 |
| 2026/10/22 13:21:38 | Embedding/RoPE原生用例与请求完成主机准备，目标实验仍待执行 | 有界Gather INT64->INT32 ABI显式声明，RoPE外置payload原样内联；完整CPU输出与原源/参考字节一致，27项回归通过。现dense/FLOAT审计不支持这两类，缺口保留，不空证据放行。无新Orin构建/推理，RDT持续采样，正式模型仍1/5。 |
| 2026/10/22 13:06:26 | Attention完整prefix源候选已准备，尚未目标构建或验证整模型收益 | 重施已微基准核验的同一配方，只删除/Add_138，2088->2087节点；840原initializer、片段外节点、完整IO/类型及其他精度声明不变。43项回归通过，两个源准备版ONNX字节相同，仍是一份候选；无新增Orin任务，RDT继续采样，正式模型完成数仍1/5。 |
| 2026/10/22 12:52:42 | RDT新首轮replay完成128预热，进入正式测量 | worker2399781/start12629557与采集器/守护实际身份仍匹配，已写正式测量行；两个授权服务仍PID0/Exited。完整进程保持2/15，活动replay数据不提前归档或认定通过。资源看板更新后的八项Node及九项浏览器回归通过，正式模型数仍1/5。 |
| 2026/10/22 12:51:51 | RDT两个完成段6330资源样本独立核验，仍非整组正式结果 | 复用原资源/ownership与保护线检查，两个约59分钟窗口最小可用23.894/23.025GiB；GPU频率采样均1300.5MHz、最高有效温度60.187C。保留最大3.669秒间隙、受限owner、18990个无效温度读数及无功率数据，9项回归通过。只处理已归档主机数据，不改活动采集或启动GPU任务。 |
| 2026/10/22 12:45:49 | 新共享流完成段全部2304张量独立核验通过，进度2/15 | 完整128预热+1024测量、exit0、所有文件hash及完整native/unified动作均通过；只归档结束的round-00-batch-only，与已检查常规段合计4608张量。下一replay worker实际身份匹配，正式整组质量/主表/CDF及服务恢复仍待最终审计。 |
| 2026/10/22 12:45:31 | RDT 正式采样完成 2/15 个进程 | 当前批次 rdt-raw-e2e-services-20261022-001 累计 2,048 条 measured；完整输出与正式结果仍待独立审计。 |
| 2026/10/22 12:42:20 | 五模型既有数据/源合同完成离线核对，目标正式完成仍为1/5 | 4581文件/179存储样本/703完整参考对字节一致，十项私有副本拒绝检查及七项源合同回归通过；不重哈希大权重、不新推理/导出、不占用Orin GPU。保留pi0/pi0.5文本与F64差异、CogACT 7.63B/外部noise及H20 LibTorch RNG边界。RDT仍为当前唯一GPU采样任务。 |
| 2026/10/22 11:46:12 | 新常规首进程完成并独立检查2304张量，进入共享流预热 | 首段完整128+1024且exit0，全部文件哈希和完整native/unified动作字节一致；原整组十五进程门槛不变，当前仅1/15，不输出提前主表/CDF。只归档结束的round-00-off，未复制活动部分输出；第二段worker2329235身份实时匹配。 |
| 2026/10/22 11:45:27 | RDT 正式采样完成 1/15 个进程 | 当前批次 rdt-raw-e2e-services-20261022-001 累计 1,024 条 measured；完整输出与正式结果仍待独立审计。 |
| 2026/10/22 11:25:48 | RDT 新窗口最终审计链就绪，真实未完成任务被拒绝为pending | 当前冻结绑定验证通过，但不认证模型数据或进程活性；终态审计复用原十五进程完整动作 / 时延 / replay检查，并新增全窗口资源和真实恢复身份门槛。86项组合回归通过，当前RDT继续顺序采样。部分交付索引已整理，五模型范围和未完成状态不变。 |
| 2026/10/22 10:53:23 | 新 RDT 首进程完成128预热，进入正式测量 | 采集器和native worker实际PID/start/argv再次匹配，首轮常规已写7条measured，原生有限值和参考一致标志正常；仍为0/15完整进程，全部动作及CDF须结束后独立核验。36项窗口/冻结/恢复回归、六项看板和九项浏览器检查通过；当前不是正式结果。 |
| 2026/10/22 10:49:37 | 新 RDT 完整十五进程采样已启动，旧中断记录不合并 | 新三模式 pilot 全部120张量字节一致、实际五步replay无回退；保持原engine/DSO/输入/noise/精度/求解合同。采集器2258939和worker2260072已实时核验，完整协议未缩减。预计约14.52小时纯采样，结束或失败后由独立守护恢复两服务；运行中不是正式通过。 |
| 2026/10/22 10:35:10 | RDT 获得两服务临时停机授权，资源窗口建立，正在重新核验 pilot | 仅两个精确 Supervisor 程序被临时停止，其他受管 PID / 状态及配置不变；首次严格终态检查触发自动恢复，失败未隐藏，第二次确认实际 Exited/PID0。可用内存约43.49GiB，恢复守护已启动。新 pilot 和完整十五进程将独立于旧中断数据，不拼接。 |
| 2026/10/22 10:13:05 | Norm 完整 E2E 审计结束，候选慢 6.43%，不计入整模型加速收益 | 五进程对 / 每进程 128+1024 全部保留，10240 measured 与 23040 完整张量独立核验通过。标准化主门槛通过、native MSE 0.0121001 单列；125.900306 -> 133.994095 ms，五组方向一致。完整 CDF、资源、141 项回归及失败历史均归档，所有本任务推理进程已退出；五模型 Goal 仍未完成。 |
| 2026/10/22 10:01:56 | 算子与回集成 139 项组合回归通过 | 保留旧 failed 收据和 GEMM 标准化失败，新增标准化 primary 口径、完整 raw 输出、缺行 / producer / 内存篡改拒绝及 owned-child 清理覆盖。Norm 正式采样仍在运行，不以测试通过代替真实最终审计。 |
| 2026/10/22 09:42:42 | Norm 完整原始输入对照开始正式采样 | 先完成两端各 4 warmup+16 measured 原始输入 pilot，80 张量字节一致到各自冻结参考。短 pilot 候选较慢，未宣称收益；两端普通调度、相同 IR/Plan/输入/noise/其余六 engine，正式五进程对 128+1024 不删样本。模板版本不匹配的首次编译失败保留，改用已验证通用冻结模板。 |
| 2026/10/22 09:28:00 | 用户确认标准化 action 主门槛，native 误差单列 | 阈值和原始误差不变；Norm 原 failed all-output receipt 保留，新收据按确认口径放行原始输入验证。GEMM 标准化 MSE 自身失败，仍不进入正式 E2E。 |
| 2026/10/22 09:34:00 | Attention 20 进程 / 20800 行通过完整独立审计 | resident compute 1.458263x，IO event 1.239668x；全部 Q/K/V 显式加载，完整最后输出逐值核验。首轮 1023 行进程仍拒绝并归档；重跑只增加 CLI 请求余量，1024 最低导出门槛不变，全部 1040 行保留。非整模型 E2E。 |
| 2026/10/22 09:22:51 | Attention 三输入五进程对采样已完成 11/20 | 四次 pilot / 512 条记录及完整输出已核验后冻结协议；resident 十个进程和 IO 首个已完成。漏 K、重复输入、字节数错误及单 / 多输入混合协议会在启动前拒绝；38 项回归通过，主结果仍待整体独立审计。 |
| 2026/10/22 09:13:17 | Attention 三输入真机上下文输出逐字节一致 | 观测 6 的全部 786432 值 MSE=0、cosine=1。两端 QK/PV 数据 FP16、Softmax FP32 可观测；只删除实际正零 mask 加法，缩放不变。28 项请求 / 策略回归通过；准备多输入计时，尚无时延或整模型收益结论。 |
| 2026/10/22 09:00:03 | 看板长实验字段的 320px 溢出已修复 | 实验记录和阈值未改；进展段落可完整换行，不隐藏溢出、不缩小字体。保留失败 trace / XML，固定长字段夹具先红后绿；桌面、平板、390px、320px 等九项浏览器回归通过。 |
| 2026/10/22 08:52:37 | 状态配置对照已审计：无均值加速，保留全部负结果及精度可视性限制 | 20 个 native 进程 / 20480 行全部保留：resident 29.811 -> 30.403 us（0.981x），IO event 39.361 -> 40.169 us（0.980x）；resident max 也变差。50 项回归及完整 CDF 就绪，same_effective_precision_main_table_qualified=false 不变，不选该候选回集成。RDT 不重启、五模型 Goal 未完成。 |
| 2026/10/22 08:49:03 | 状态投影配置对照完成 18/20 个进程 | resident 十个进程和 IO 八个进程已结束，第十九个已启动。所有原始导出仍待整体审计；融合实际精度缺口继续保留，不因采样结束变为同实际精度主表通过。 |
| 2026/10/22 08:46:56 | 状态投影配置口径配对计时协议已冻结并启动 | 四次 pilot / 512 条记录及最后输出已核验；两种边界各五组独立进程对、至少 1024 measured，所有记录保留。协议显式 same_effective_precision_main_table_qualified=false，不以计时完成抹掉 baseline 融合内部不可视的限制。 |
| 2026/10/22 08:43:32 | 状态投影完整输出通过，融合实际精度缺口显式保留 | 全部 16 输入 / 15360 值通过 MSE/cosine 门槛。baseline Myelin 内部不可视，candidate FP16/Half 可见；同 FP16 请求和 OBEY 配置经 SDK 合同及实际 argv 检查。仅做配置口径计时，实际精度完全相同主张未通过；32 项回归通过。 |
| 2026/10/22 08:35:10 | VLA 状态投影 16 份输入的真机配对已启动 | 两图 / 精度 profile / 全部 16 个不同输入与期望输出均 hash 绑定。通用请求补充显式单端口与两种明确路径约定，25 项回归通过；未改原 case 或旧脚本。尚无质量、计时或模型收益通过结论，RDT 不重启。 |
| 2026/10/22 08:29:12 | GEMM 回集成及重建控制已审计，候选不进入整模型计时 | 原图重建标准化 MSE 4.519e-7 通过，但 GEMM 相对该新控制仍为 1.376e-5、超过 1e-5。两次各 34 个完整动作对保留，原模型对候选失败不被新 baseline 取代。43 项回归通过。微基准仍只支持其独立边界，四类算子 / 有效 E2E 及五模型主线仍未闭合。 |
| 2026/10/22 08:25:31 | 未改图 / 同 profile 的完整 prefix 控制重建通过 | 原 ONNX 和 precision profile hash 均与旧模型完全相同，数值 CLI 设置一致，生成新 engine b0b77654…。已组装同一七 engine / 十步求解进行完整动作控制核验，结果仍待审计；不是第二个优化候选，原 GEMM MSE 失败不被抹掉。 |
| 2026/10/22 08:22:40 | GEMM 完整动作标准化 MSE 失败，正式 E2E 不启动 | 34 个完整动作对：标准化最坏 MSE 1.324344964e-5 超阈值，native MSE 0.0124503239；baseline 与原采用 pilot 字节一致。38 项回归通过。未改图 / 未改 profile 的重建控制 PID 2096349 / start 11040202 已确认运行，用于区分编译漂移与改写，不抹掉失败或新增量化搜索。 |
| 2026/10/22 08:16:30 | 完整 GEMM prefix 编译与来源配对审计通过 | 构建约 162 秒，328 个内存观测最低 8.6 GiB，未触发安全线。独立读完整图确认仅指定 GEMM 片段改变，精度映射与已报告绑定不变；三处缺失描述留给 native guard。正在组装完整七 engine / 十步求解，尚无动作或 E2E 通过结论。 |
| 2026/10/22 08:14:32 | 完整 GEMM prefix 的 Orin 构建已确认运行 | 构建控制子进程 PID 2086450 / start 10990726；启动前可用内存 15859892 KiB，12 GiB admission 和 2 GiB safety floor 生效。尚无 engine / 模型通过结论，不重启 RDT，不更改其他服务。 |
| 2026/10/22 08:11:51 | GEMM 候选已嵌回完整 prefix，结构与边界检查通过 | 只替换已测单个 MatMul；其他源节点、840 个原 initializer、全部 compute / output precision 映射不变。27 项组合回归通过，候选文件已传至 Orin；拟按 12 GiB admission / 2 GiB 安全线构建，编译不等于整模型通过。RDT 仍停止。 |
| 2026/10/22 08:01:02 | GEMM 两种五进程对计时审计完成，负结果保留 | 全部 20480 行保留：resident GPU compute 38.142 -> 26.339 us（1.448x），IO event 59.467 -> 58.993 us（1.008x）；IO compute 略退化、两组 IO event 负收益及较高 CPU 诊断误差均保留。完整 CDF 和 47 项回归就绪，不是整模型收益。RDT 未重启。 |
| 2026/10/22 07:57:47 | GEMM 微基准完成 12/20 个进程，整体审计待执行 | resident 边界十个进程均完成；IO 边界已完成两个并启动第三个。新版计时审计对旧 LayerNorm 全部 20480 条记录回归通过，含错误 engine / boot / 输入 / producer 的拒绝测试，共六项通过。此快照不认证整体完成或 VLA E2E。 |
| 2026/10/22 07:55:58 | GEMM 算子计时 pilot 通过，五进程对协议已冻结 | 四次 pilot / 512 条完整导出记录及最后输出检查通过；两种边界各五组独立进程对、交替顺序、每进程最低 1024 measured，保留全部记录。原生 CLI 计时边界不改，不以 pilot 或算子收益代替 VLA E2E。 |
| 2026/10/22 07:53:37 | GEMM 两份真实输入的真机精度门槛通过 | 113280 个完整输出值：最坏 MSE 3.836792e-9、最低 cosine 0.9999980099，不是逐字节一致。相同 FP16 dense 数据 / Half 权重，vendor tactic 不同；候选对 CPU FLOAT 的误差更高已保留。39 项回归通过，尚未测收益。 |
| 2026/10/22 07:50:08 | GEMM 原 FP16 合同的真机配对已启动 | 数据清单绑定两份真实输入、两图 / 精度配置 hash；通用前置回归 15 项通过。实际 baseline 构建已进入 native 正确性 probe，未开始配对计时，也未宣称收益。RDT 控制器和 worker 均不匹配，保持停止。 |
| 2026/10/22 07:45:04 | LayerNorm 完整动作配对已核验，native MSE 失败保留 | 两端均为七 engine / 十步求解，34 个完整动作对：标准化最坏 MSE 5.30e-6、cosine 0.9999985；native MSE 0.0121 未过同一阈值。候选与原同 engine 输出字节一致。22 项回归通过；未启动正式 E2E，未重启 RDT 或终止既有服务。 |
| 2026/10/22 07:36:01 | LayerNorm 全模型回集成的来源配对通过 | 精确重跑通用改写得到同一完整 prefix ONNX hash；50 处 Norm 替换，非 Norm 权重 / 节点 / 精度不变。尚未验证完整动作或 E2E；三处缺失 inspector 描述留给 native ABI 验证。RDT 未重启。 |
| 2026/10/22 07:20:09 | LayerNorm 五进程对计时完成独立审计 | resident compute 4.329x / IO event latency 1.644x，仅单个真实算子签名。20 个进程、20480 条记录全部保留；候选 std 和部分最大值不更好。仍未做全模型回集成，RDT 未重启。 |
| 2026/10/22 06:54:15 | LayerNorm 小内存 Orin 配对通过全输出门槛 | 两端新 engine 同为 256 MiB workspace / 原 FP32 归一化合同；786432 个值 MSE 2.29e-15、cosine 近 1，但非 byte-exact。尚未计时，不算整模型 E2E 或 RDT 正式结果。 |
| 2026/10/22 06:23:11 | RDT 保留数据完成限定范围核验 | 11 个完整进程及第 12 个已写入的部分数据共 25894 个动作张量逐字节一致；原始文件未裁剪，正式门槛仍拒绝未完成 campaign。当前无采样 worker，不自动重启。 |
| 2026/10/22 05:54:33 | RDT 资源安全中断，完整 / 部分证据保留 | 持续低内存和满 swap 下，仅结束本任务 worker；控制器记录失败并退出。11 个完整进程及末进程 147 条 measured 保留，外部服务不变；不是正式完成或数值失败。 |
| 2026/10/22 05:40:32 | RDT 正式采样完成 11/15 个进程 | 累计 11,264 条 measured；完整输出与正式结果仍待独立审计。 |
| 2026/10/22 05:29:53 | 系统内存耗尽风险已更新 | 05:27 快照：可用约 1.19 GiB、swap 满，当前 RDT worker VmSwap 0；已有服务 RSS / 线程数继续增加。9 项证据检查通过，未作泄漏或延迟因果结论，采样及服务配置不变。 |
| 2026/10/22 04:40:44 | RDT 正式采样完成 10/15 个进程 | 累计 10,240 条 measured；完整输出与正式结果仍待独立审计。 |
| 2026/10/22 04:26:49 | RDT 资源压力复查已归档 | 系统可用内存约 2.39 GiB，较前次降约 0.71 GiB；新 worker VmSwap 0，已有 Java RSS / 线程数增加。8 项检查通过，两个快照保留；不作泄漏或无性能影响结论，采样继续。 |
| 2026/10/22 03:42:23 | RDT 正式采样完成 9/15 个进程 | 累计 9,216 条 measured；完整输出与正式结果仍待独立审计。 |
| 2026/10/22 03:38:57 | 资源警告页及搜索回归通过 | 7 项浏览器检查覆盖桌面 / 手机风险页，无文字溢出；搜索用例不再写死实时任务匹配数量。原失败记录保留，未改真实任务或 Orin 实验。 |
| 2026/10/22 03:29:07 | RDT 系统资源压力与可见性限制已归档 | 当前 worker VmSwap 0；系统可用内存约 3.10 GiB、swap 近满，已有 Java 服务占用高。6 项证据 / 解析检查通过。未改环境、未下泄漏或无性能影响结论，正式采样继续。 |
| 2026/10/22 02:42:41 | RDT 正式采样完成 8/15 个进程 | 累计 8,192 条 measured；完整输出与正式结果仍待独立审计。 |
| 2026/10/22 01:43:46 | RDT 正式采样完成 7/15 个进程 | 累计 7,168 条 measured；完整输出与正式结果仍待独立审计。 |
| 2026/10/22 00:46:07 | RDT 正式采样完成 6/15 个进程 | 累计 6,144 条 measured；完整输出与正式结果仍待独立审计。 |
| 2026/10/21 23:46:25 | RDT 正式采样完成 5/15 个进程 | 累计 5,120 条 measured；完整输出与正式结果仍待独立审计。 |
| 2026/10/21 23:39:50 | 第六类 VLA 状态用例完成主机准备 | 16 个不同真实输入、15360 个状态 token 值逐字节一致；原投影及 FP16 请求不变，输入哈希链与独立重算已核验。143 项组合回归通过；仍无新 Orin 构建、测时或热点结论。 |
| 2026/10/21 23:19:48 | 真实语言 Embedding 与独立构建器完成主机检查 | 46080 个实际输出及完整缩放表重算一致；16 观测共享一个指令，不夸大提示覆盖。新 v3 不补造计算 / 输出精度约束，135 项组合回归通过；尚未传到 Orin 或产生新性能结果。 |
| 2026/10/21 22:58:34 | 真实 Attention Core 配对完成主机准备 | 原始观测重算 Q/K/V 与 786432 个上下文值逐字节一致；只移除静态零掩码，保留缩放及原精度请求。67 项组合回归通过；没有新 Orin 构建、测时或收益结论。 |
| 2026/10/21 22:48:46 | 真实 GEMM 配对与组合回归完成主机验证 | 保留原 FP16 请求、不增加输出精度约束；两观测 113280 个原始 CPU 输出值逐字节一致。48 项组合回归通过，原外部权重未变。板端配对仍未构建、核验或测时，不计优化收益。 |
| 2026/10/21 22:46:38 | RDT 正式采样完成 4/15 个进程 | 累计 4,096 条 measured；完整输出与正式结果仍待独立审计。 |
| 2026/10/21 21:50:23 | RDT 正式采样完成 3/15 个进程 | 累计 3,072 条 measured；完整输出与正式结果仍待独立审计。 |
| 2026/10/21 21:31:38 | 真实 RoPE/cache 输入配对已准备 | 两观测 113280 个原始输出值逐字节一致，21 项回归通过；保留相同数学表达式和精度合同。尚无新 Orin 构建、正确性或性能结果。 |
| 2026/10/21 21:31:38 | 通用静态索引改写补齐依赖保护 | 拒绝三探针假静态、可覆盖 initializer 和中间形状注解；使用 SSA 输出区分匿名/重名节点。15 项边界测试、全量 2639 项主机测试通过；冻结 RDT engine 未改动。 |
| 2026/10/21 20:53:22 | RDT 正式采样完成 2/15 个进程 | 累计 2,048 条 measured；完整输出与正式结果仍待独立审计。 |
| 2026/10/21 20:22:38 | 真实输入的首个算子配对已准备 | LayerNorm 同接口 / 常量 / epsilon；原表达式和语义影子通过 CPU 门槛，68 项回归通过。新配对尚未在 Orin 构建、核验或测时，不计为算子收益。 |
| 2026/10/21 20:06:10 | 网页版任务看板已就绪 | 任务总览、模型矩阵、证据与 CDF；网页和 Markdown 共用状态源。6 项服务测试、7 项浏览器测试通过，覆盖桌面、平板及两种手机宽度。 |
| 2026/10/21 19:52:33 | RDT 正式采样完成 1/15 个进程 | 累计 1,024 条 measured；完整输出与正式结果仍待独立审计。 |
| 2026/10/21 19:25:00 | 主机全量回归通过 | 2630 passed，74 skipped，6 个子测试通过；不新增 Orin 模型计算。 |
| 2026/10/21 19:17:00 | 算子消融输入准备推进 | 七个 SmolVLA 实际源图、236 个逻辑签名组；尚无算子前后时延。 |
| 2026/10/21 19:17:00 | RDT 活跃进程已核验 | 首进程 367 条正式测量，已写入行 finite / direct_exact 均通过。 |
| 2026/10/21 19:04:00 | 正式后处理流程已就绪 | 独立审计与 pooled / 逐进程 CDF 回归合计 37 项通过。 |
| 2026/10/21 18:50:00 | RDT raw pilot 通过，进入正式采样 | 112 个输入与 120 个完整动作张量逐字节一致；正式结果仍待完结审计。 |
