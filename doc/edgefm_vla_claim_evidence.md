# EdgeFM Paper Claim / Evidence Map

2026-09-10最终核对：依据 `ICRA_2027_EdgeFM.pdf` 与作者新增要求，不修改原PDF。
状态入口：`edgefm_vla_goal_status.md`；完整主张审查：
[PDF与H20证据边界](reports/h20_paper_scope_review_20260910.md)；逐项验收：
`artifacts/recovery-audit-20260909/goal-completion-audit-002.json`。

五个VLA原始输入正式流程已全部完成，50进程、51200正式测量、149760完整输出
张量byte exact；远端/本地独立审计、逐模型及统一CDF均已完成。CogACT最后补齐
的official/Session均值为120.286270/79.971651ms，不再处于pilot或运行中。
resident及host-model-tensor两套正式表、六类算子和两规模Qwen官方多模态
formal均已完成。Qwen3.5 0.8B/2B固定profile原生部署也已完成：first/16-token
共20进程、20480测量、46080完整输出张量byte exact，四组独立审计与CDF通过。
CogACT autonomous C++ RNG路径完成同输入、seed、精度、GPU序列和计时边界的
official/native正式对照：5120+5120测量，完整输出与RNG byte exact，mean ratio
1.456x。最新CogACT收尾CPU回归2276通过/62跳过，2935个证据文件身份重核保持一致。

| 主张 | 当前边界 |
|---|---|
| H20 VLA吞吐/时延/CDF及动作一致性 | 支持已记录profile、计时边界和数值策略；完整CPU动作输出，保留有限输入覆盖 |
| 端到端无Python、相机到执行器zero-copy | 原始输入表含Python主机及显式H2D/D2H，不能据此声称 |
| 算子调优 | 支持六类固定recipe对比和负结果，不证明自主生成kernel或这些候选的整模收益 |
| Qwen原生适配 | H20固定profile完成：0.8B/2B first/16-token原生C++ Session、完整输出byte exact、CDF和负例门禁均通过；不支持任意输入、流式逐token或板端外推 |
| CogACT autonomous RNG及同边界native对照 | 支持已记录public-dependency profile：C++ provider自主生成/推进状态，official/native同边界独立审计通过，输出与RNG全byte exact |
| CogACT完整3B模型、原Meta依赖 | 不支持；实际checkpoint7.63B，公共依赖配置及原Meta gated配置未验证必须保留 |
| Orin/J6M/BPU、硬实时、物理任务成功 | 暂缓或未验证，H20结果不替代 |

统一证据与工期入口为
[H20实验交付](reports/h20_vla_experiment_delivery_20260910.md)。
Qwen原生正式结果见
[Qwen3.5 Native H20 Formal Deployment](reports/qwen35_native_formal_20260910.md)。
CogACT同边界正式结果见
[CogACT Boundary-Aligned Formal Comparison](reports/cogact_timing_boundary_formal_20260910.md)。
当前H20证据交付闭合，不等于全论文或Orin/J6M/BPU验收。

## 历史实施记录

以下实施时点为历史快照，冲突时以本段、最终交付和状态入口为准。

2026-09-10：通用原始输入计时器与OpenPI双实现已落地；pi0/pi0.5各10个正式进程
全部通过远端/本地完整输出审计与CDF检查，共20480测量/46080张量byte exact。
每个native进程1152 replay且0回退；controller 80976/85392已成功结束。两模型
official/Session均值分别221.305113/108.152217、256.840307/116.853524ms。
计时包含原始已解码RGB/state/text预处理和完整CPU双输出，排除文件解码与初始化。
SmolVLA下一隔离适配的完整CPU回归2235通过/74跳过，真实GPU流程未验收。详见
`reports/raw_input_host_pipeline_20260910.md`。此Python主机流程不授予no-Python
部署证明；其余三模型原始输入后续闭合，Qwen原生部署也已在
`reports/qwen35_native_formal_20260910.md`完成固定profile验收。

同日续接：SmolVLA和RDT原始输入适配均已通过official/native pilot的远端独立
审计及本地完整输出复核，每组96调用/192张量byte exact。SmolVLA先在H20-1
通过pilot，正式启动因foreign owner拒绝后转入H20-2新GPU绑定协议及独立pilot。
SmolVLA H20-2正式controller113987现已成功完成10/10进程、远端/本地全部审计
及CDF检查，official/Session均值206.161639/50.726747ms，10240测量/23040完整
张量byte exact；RDT119430正式长测仍未验收。CogACT新原始输入adapter的最新
完整CPU2259通过/74跳过。CogACT原始输入pilot已完成远端/本地独立复核，96调用/
480完整张量byte exact，native48十步replay/0回退；正式controller143713实查存活，
不能写成formal完成。详见 `reports/raw_smolvla_rdt_pipeline_20260910.md` 和
`reports/raw_cogact_host_pipeline_20260910.md`。Qwen2B独立原生诊断已匹配既有正式
输入和完整token，后续trace/export仍待验收；不由0.8B失败推断2B已经实测失败。

最新收尾：H20五模型host-model-tensor正式75进程、远端/本地完整输出审计、统一
15行表和五组CDF已完成，见 `reports/h20_host_tensor_formal_table_20260909.md`。
新增76800正式测量/224640完整张量，计入每次H2D/绑定/Session完成/全部D2H；
仍不包含原始图像预处理。下方运行中快照为历史。pi0.5原始输入native与official
路径各48次pilot通过自检，仅支持流程可行性，不能写入正式时延表。

2026-09-09 22:32 CST：公共主机张量计时模式及Python/C++测试完成，完整CPU回归
2204通过/74跳过。五模型的新三策略pilot远端/本地审计全部通过，720调用/1872完整
输出张量byte exact；五个formal controller实查存活，11/75 worker自检通过。
正式汇总/独立审计未完成，也不包含原始图像预处理。见
`reports/host_tensor_boundary_20260909.md`。旧resident正式表保持独立。

2026-09-09 21:49 CST最新：H20五VLA resident-tensor正式表、完整输出/CDF已闭合，
共76800正式测量/224640完整张量；六类算子各5对进程的局部比较及完整输出审计
已闭合；Qwen3.5 0.8B/2B真实图像+文本官方Python基准各5x1024完成并经远端/
本地审计，CDF已检查。权威报告分别为 `reports/h20_vla_formal_table_20260909.md`、
`reports/h20_operator_table_20260909.md`、`reports/qwen35_natural_profile_20260909.md`。
VLA完整输入计时及同口径官方性能对照、Qwen原生部署仍未完成。算子结果不外推
自主Agent或整模收益；CogACT profile限制和Orin/J6M暂缓保持。下面内容为历史记录。

2026-09-08 23:35 UTC：G5新增通用FP16/BF16完整轨迹质量分支，真实RTX 3060全原IR
执行。公共`deployment/half_linear.py`与`linear_precision.py precision_kind`提供
float16/bfloat16静态替换；两种精度各16条真实轨迹（前8校准/后8 held-out）全部通过
原MSE<=1e-5/cos>=0.9999。FP16 worst held-out2.434350e-6，BF16 worst held-out
5.109796e-6。这是首批通过held-out的低精度完整轨迹，但不等于原生部署、无损、正式
性能或物理任务成功。independent audits SHA
`08264a3bfd3671f41de8737e90eab836b4e393e52f738880d714b08da82105a0` 和
`f7e3f514b11124270f6334522ba67d04c0ee3e08196cb9b4bf01e782d3571cba`。
先前的INT8四策略0/8失败仍保留；下一步原生FP16/BF16和同源计时/内存对照。

2026-09-08 23:45 UTC：FP16 no-Python完整模型已闭合。四TS Region（三共享+FP16
step）、C++ bundle、单原生Session 16完整动作byte exact且held-out8/8过质量门，
maps无Python。native PID2077538 exit0；independent audit SHA
`35c660042d1785847ef2c98a3b529876446830118108d75b95e22022c827af85`。这不是
formal latency/memory/replay/BF16/无损/板端。INT8原生仍0/8过质量门保留。

2026-09-08 23:55 UTC：BF16 no-Python完整模型也闭合。四TS Region（三共享+BF16
step）、C++ bundle、单原生Session 16完整动作byte exact且held-out8/8过质量门，
maps无Python。independent audit SHA
`0002d4d3ce0b3f99db79ad6cb63f7302ffc4a9c1033fc78cfff21aa097cfc7f9`。
仍不是formal latency/memory/replay/无损/板端。

2026-09-09 00:00 UTC：baseline/FP16/BF16 RTX同源原生pilot完成，单进程16warmup+
32measured，每lane48调用byteexact且maps无Python。mean
93.137304/94.474481/94.115055ms，half无加速；allocator steady请求
410912/413152/413152、allocated peak+46608B、reserved不变。descriptive仅，正式
CDF未完成。

2026-09-09 00:50 UTC：三路正式5x1024同源原生benchmark完成并独立重算。mean
baseline/FP16/BF16=93.351683/94.182185/93.792310ms，half无加速；allocated peak
+46608B、reserved不变、steady allocator+70/call。formal analysis SHA
`ef615c48383372763e4a1668b4eaf4096762d4956faefa3862f1d19853bfe074`。off-only普通
执行，无replay/profiler/板端。

2026-09-09 01:10 UTC：FP16/BF16逐步误差预算完成。step0 head输入byte exact；
held-out step9 worst head-input MSE FP16/BF16=1.149572e-4/1.261380e-4，
max-abs0.0625-0.09375；final action MSE仍过1e-5门。中间状态不可写成逐张量lossless。
report SHA `6386d259119f70279a4c26b7d6be454d1f4be3b0386d611bad8a7a1c23d4e279`。

2026-09-09 01:20 UTC：RTX NSYS CUDA profiler三路完成，top为既有BF16 attention/GEMM，
half head非主导kernel；CPU采样不可用关闭，profiler不计formal。

2026-09-09 01:40 UTC：baseline/FP16/BF16三策略replay pilot完成，48调用byte exact，
required counter `REPLAY_FINAL,11,1,10,48,0`；required pilot mean约76.3-76.8ms，
较off约降18-19%，descriptive仅。

2026-09-09 02:20 UTC：三路required正式5x1024 CDF完成，mean
77.170051/77.217964/77.271447ms，counter `REPLAY_FINAL,11,1,10,1152,0`；相对formal
off降17.33%/18.01%/17.61%，half不改变replay收益。

2026-09-08 15:08 UTC：CogACT新002数值provider正式表闭合。15独立进程/15360测量、
17280完整调用/86400五端口张量/6099840值byteexact；off/batch-only/required mean
84.869524/86.191139/76.230755ms，required p9978.860851ms、13.118065chunks/s，相对
同源off mean降10.1788822%，采样显存+140MiB。原链audit的JSON顺序bug保留，修正002
仅重审不重跑；remote audit SHA
`92df2490f9383ad6ba75bafb3eaf8bb4bd568168116f8abaeb0d438632eb7014`。仍不能填充
Orin/BPU主表，原Meta配置/7.63B规模/外部RNG/低比特缺口不变。
H100 SmolVLA同源retain/scoped graph memory对照通过远端独立
审计和本地完整取回。两策略各30次真实Session create/run/destroy、480调用/960完整
F32张量/288000值byteexact；off/batch-only销毁后reserved恒定。required在retain模式
每次新增Session销毁后reserved增加4MiB、5次净+16MiB；scoped-reclaim每次销毁发生2次
scoped device free且reserved保持1937768448B。这闭合“同candidate同platform下整段
replay重复Session内存”的有限场景，不是零分配/零释放/全部workspace接管；仅限Torch
2.10 scoped路径。remote audit SHA
`bafebd980163b9f7fe0f4234aea648f29c4110229a994aa0f1b871d5623908c3`，retrieval SHA
`e1d2272246924a8919f8b0a0dae866d178971acd43849f44aca6f11f2f3a2b0d`。CogACT新002
formal 15 workers与report已完成，独立修正审计进行中，旧unbound formal仍不继承。

2026-09-08 12:22 UTC：SmolVLA新H20正式15进程、独立远端/本地完整输出审计及CDF
实际查看完成。34560完整双输出张量/10368000值exact；required mean47.112908ms、
p9949.797612ms、21.225606chunks/s，相对off点估计降低59.9490584%，采样显存反增778MiB。
不是Agent单算子收益；GPU独占但CPU/NAS与本项目构建共享，不声称无干扰平台极限。
CogACT新002的四Region编译、16帧完整五输出和三策略原生pilot/独立审计/本地复算
通过，720张量50832值exact，实际LibTorch数值provider约束通过。缺pytest的原pipeline
失败保留，隔离依赖后的12项证据测试恢复通过。新的formal/CDF尚未运行，旧unbound
性能记录不能自动附加provider保证。Meta配置、实际规模、自主RNG及低比特/板端缺口
不变。公共CPU021为2113通过/83跳过，另2项原生CPU测试仅属于fixture接口验证。
两项本轮closeout SHA分别为
`c039235b976528a9d5ebb0e4087d1f467f18888386d1d32352433ba67f6eb77d` 和
`c8430b442226c2a4ca73b320b1ac5fcfadbaa4d06f8473414e68e2b6d33e2868`。

2026-09-08 11:01 UTC：新增H20 SmolVLA同机真实16episode参考、完整双输出IR与
三策略C++ pilot通过；独立144调用/288张量审计SHA
`9a9004795b3014783daff065466e50fb50d240ede0ce1d825384d1052482cd66`，九项真实证据测试
通过，正式长测仍运行，未形成新的正式表/CDF结论。公共编译器现在实测绑定源文件、
实际加载图、产物和编译前后数值策略，CPU021为2113通过/83跳过；不是runtime完成证据。
CogACT数值绑定001的图hash断言失败保留，真实save/load证明变量重命名而非已发现的
输出变化；002重新编译并以实际编译图建立新identity，C++ provider仍待验收。
详见 `reports/smolvla_h20_20260908.md` 与 `reports/torchscript_numerical_compile_20260908.md`。
本段优先于下文“本轮所有作业退出”的历史描述。Goal与板端验收仍未完成。

2026-09-08 09:58 UTC最终补充：CogACT004三策略正式实验、远端独立审计、本地原始复算
及CDF实际查看现均通过。86400完整五端口张量/6099840值逐byteexact；required每进程
N10 capture/1152replay/0ordinary，mean75.538978ms、p9978.687230ms、13.238199chunks/s；
相对off平均降低11.7295225%，采样显存增加140MiB，不支持省内存或Agent收益结论。
远端audit SHA `63d15591ad3b9bce8cb11f38cedb0a23c1d3c10d7f9b340ab6faa25966b0eb9f`，
本地audit SHA `4337a597236190824f9c855edf116ef185b6ed334ff28192dc0bf99b8892975b`。
Meta原配置、实际7.63B与论文3B标签差异、自主RNG及legacy数值provider仍未验收，
公共表的lossless-paper eligible仍false。不能因测量和输出通过而删除这些限制。
五模型的分平台三策略CUDA切片已各自产生正式CDF，但不是同平台矩阵或Orin/BPU主表。
本轮所有执行、审计和传输作业已退出。详细限制与产物见CogACT和SmolVLA报告。

2026-09-08 09:47 UTC：G4新增真正的同候选同平台正式整模证据。H100-x8上原模型、
拆分TS control、接入原微基准AOTI包三路各5x1024，34560完整双输出张量全部byteexact。
远端与本地独立审计通过，包SHA `73c3d5f22c7cfdb50e7b7bc4b9d973c482afc7bcc1480466c6842a9cfe5799f9`
未重编译或替换。点收益0.776794%，配对CI[-0.740279,1.587530]ms跨零、p=0.28125，
因此性能门禁不通过，不能支持III-D稳定kernel加速声明。selection/2如实记
`end_to_end_integrated=true`、`correctness_passed=true`、`status=rejected`。
正式audit SHA `a1a6189a28ebe776871890ff8429667894ca3a9cb540e924830ce0d411c87021`。
公共加载依赖和参考I/O修复后CPU017为2102通过/83跳过；真实pi05新旧校验34560张量
绑定相同，校验优化不计模型收益。CogACT所有正式worker与汇总通过，独立审计仍在运行。

2026-09-08 09:08 UTC追加：SmolVLA在新增H100-x4已完成三策略正式实验、远端独立审计、
本地34560完整张量/10368000值逐byte复算和CDF实际查看。required mean45.971896ms、
p9957.069162ms，较off平均降低61.0969997%，采样显存增加566MiB；不是省内存或Agent
kernel收益。远端audit SHA `b2171d87c7d364cd3b408daea3a86587d7ab37352f6742c52f0f1f661af89f4c`。
详见 `doc/reports/smolvla_h100x4_20260908.md`，当前硬件Goal和全论文验收仍未完成。
CogACT正式13/15worker通过仍在运行。新H100-x8 solver三候选各160真实step输出exact，
原候选包接回完整模型验证中，不能从2.255696us微基准推导E2E收益。

2026-09-08 08:26 UTC更新：用户追加H100-x4资源，四机实查共28张GPU空闲快照已归档，
不是当前GPU缺口。CogACT新run004在三策略pilot/独立审计/12项测试通过后已进入正式
长测；SmolVLA新H100同机参考16帧、五TS Region和完整双输出IR/三策略native pilot、
独立审计/9项测试通过，正式长测运行中，两者尚不作为完成的正式结果。
III-D旧H20 Embedding与RTX静态预计算original-to-control误配已独立判无效，不是已
接入kernel的负收益实验。公共selection/2已增加候选产物和同平台/同口径显式绑定，
真实误配拒绝测试通过，仍不能用元数据门禁代替真实集成执行。全量CPU015为
2097通过、83跳过；远端继续原冻结源码，不覆盖运行中作业。

2026-09-08 D1/D2准备更新：五模型完整输出数据包已完成离线/移目录校验；通用
data-only schema不虚构CogACT115帧的正式benchmark次数。Smol/RDT新增完整双输出包，
CogACT5750个原始输出张量另逐byte重核，保留公共依赖候选和外部随机数带限制。
CogACT历史audit引用的上一文件SHA与现文件不符，该断链排除出验收证据，旧文件和失败
记录保留。详见 `doc/reports/vla_board_data_completion_20260908.md`。这只完成数据准备，
不成立板端执行/性能/质量或论文全验收结论。当前CPU回归013为2074通过、83跳过。

2026-09-08 恢复补充：RDT现完成8 Region完整双输出的三策略真实H20正式实验，
15进程/15360测量/17280调用/34560张量/157040640值对官方及同artifact逐byteexact，
MSE/max-abs为0。required每进程实际5步capture、1152 replay、0 ordinary；
保留DPMSolver carry和原数值策略，
没有将多步scheduler替换为Euler。独立audit SHA
`95a17d627aed2fb59c591bda725f1ea74137d7236ec85e7481b817845693b8fe`，
本地完整raw/时延/遥测复验、生成表及CDF实际查看通过。required mean164.397820ms、
p99168.415117ms，相比同源off平均时延降低6.967724%；采样显存反增608MiB。
该证据推进III-A通用跨模型整段replay，不支持省内存或Agent kernel收益声明。
G5低比特、vendor及板端不继承。详见 `doc/reports/rdt_execution_variants_20260908.md`。

pi0.5新16帧三策略正式实验现已完成15进程15360测量、17280调用/34560完整F32+F64
张量/39744000值逐byteexact；MSE/max-abs=0，cosine最低0.9999999999999998。
每个required进程10步capture/1152replay/0ordinary；mean350.053228ms/p99351.884728ms，
较同源off平均时延降低3.315114%，采样显存增加120MiB。pilot排除出正式CDF；原始输出、
数值bootstrap/DSO、每个CDF rank和汇总独立审计通过，PNG实际查看，worker均已退出。
audit SHA `b50b40f7c889e56a5186987815c91a0556dbb7e5d8fd8c7e096e08f516641f11`。
此为resident-tensor执行策略收益，不是Agent kernel收益、物理任务成功或省内存证明。
详见 `doc/reports/pi05_execution_variants_20260908.md`。

CogACT新增通用literal-only precompute仅预计算512B固定频率向量，真实step的N10
CUDA graph capture/replay及完整carry逐byteexact，动态timestep/CFG/noise未改。原
required C++ pilot因CPU常量复制失败；新run003完整C++ required pilot已通过48调用/
240完整张量逐byteexact、10步capture/48replay/0ordinary。独立五端口audit SHA
`981f98dce0442563c3972c7f3971f4cab5c6fe3e3e046ce14310478ec7926458`，12项真实证据测试
通过。短pilot不作正式性能对照，也不是Agent kernel收益。run002捕获状态报告误作
详细证书使base构建失败；run003公共保存接口重新捕获及完整IR参考通过，旧失败全部保留。
公共源码CPU回归014为2079 passed/83 skipped及11项C++通过，不替代真实部署门禁。
CogACT本地原始证据也逐byte复核通过，audit SHA
`e0ae324fa447fdced9264999042434e40a5b33913bb1e468ae4783bad77be345`；两台CPU的部分
cosine/norm末位1 ULP差异独立保存，未改变原始输出、模型门禁或远端指标。完整动作
保真依据仍为逐字节一致及零MSE/max-abs，不是相近的cosine。

2026-09-08 当前核对：pi0 同源 off/batch-only/required 三策略真实 H20 正式实验
已完成15进程/15360测量/17280调用/34560完整双输出张量独立逐byte核验；每个required
进程实测10步capture、1152 replay、0 ordinary。off/batch/required mean分别为
139.628717/139.362490/84.651419ms，required p9986.142677ms，相对off平均时延降低
39.373919%。独立审计SHA `7405e50ad5348d0fa4e088f388c2a8300b25ed29eba4f4d9ccd5d1221cbb90e4`，
完整raw、CDF图、runtime DSO/模型payload及290源文件均复核。pilot没有填入正式CDF；
采样显存由off6893MiB增至required7561MiB，不声称省内存或Agent kernel收益。详见
`doc/reports/pi0_execution_variants_20260908.md`。通用typed runner复用公共benchmark的
严格replay/poison检查，应用预热计入调用数，不豁免预热检查；全量CPU回归2049通过、
71跳过，不以skip代替硬件或模型通过。该路径只有同episode16帧、resident-tensor边界、
保存noise；官方/vendor性能对照、原始传感器边界及板端验收均不继承本结果。

CogACT统计勘误：下文历史目录/标题中的“正式”保持原称呼，但10x115调用没有按公共
协议剔除128次预热，且每进程测量不足1024次；不能将其计作统一稳态CDF验收完成。
完整输出保真结论仍有效，详见 `doc/reports/cogact_formal_cuda_20260907.md`。

2026-09-07 恢复勘误：下表中的历史 Smol/RDT benchmark 确实保存了完整输出与时延，
但旧 prepare 未冻结重建的 runtime DSO，不能据此声称完整运行库身份闭合。
Smol 新双输出诊断 96 调用已有独立核验；正式 001 数据保留、DSO 门禁重验待补。
RDT 正式 002 已使用启动前 DSO 清单与实际 maps 校验，网络中断后保留已完成的前三次，
只续跑后两次；最终独立审计结果以状态文档的新记录为准，不追认旧运行执行过新门禁。
通用 AOTI 构建期物化现已实现且 CPU 真实包/搬移测试通过，H20 pi0 完整新 Session
仍在验证中，不能从该 CPU 结果推出 CUDA 全模型或板端通过。

12:08 UTC CogACT series 更新：从锁定 Fractal episode 0 的 16 个唯一真实帧逐帧加载官方
reference，并在同一模型进程中验证 10 个 DDIM step 的 sample、每次 CFG 的
`x/timestep/z/output`、11 次 CUDA RNG draw 和完整动作输出；series-004 report SHA
`d7a15c16bc9c12ee2b8d3b02537880431e002a09a4d720cfc056c84953f28898`。这些输入已物化为
现有 TorchScript bundle 的二进制契约；5 个独立 H20 进程共 80 次跨帧 C++ 调用全部逐字节
匹配 native action，pilot 的 mean/p50/p95/p99 为 171.812444/136.563095/666.485731/
735.979237 ms。pilot 仍低于预声明的 1000 fresh-chunk 正式门槛，不构成最终 CDF 或 vendor
对照结果。

13:18 UTC CogACT 正式 CUDA series 更新：扩展到同一锁定 Fractal episode 的全部 115 个唯一
真实帧；series-005 官方 partition 115/115 逐步、CFG、RNG 和完整动作 exact。现有无 Python
TorchScript bundle 在 H20-2 上由 10 个独立进程各执行 115 帧，共 1150 次调用，全部 native
action 逐字节匹配；随后独立 audit-002 对 raw/normalized/native 三组输出逐字节复核，SHA
`9c6946c96a5e3574a047fa8026dbb9ede33f4cdaf028cdf388ff210e278c22e9`。
三组的 MSE/max-abs 均为 0，cosine 最低分别为 0.999999826/0.999999757/0.9999999999999998。
resident-tensor 边界的 mean/p50/p95/p99/max 为 139.889050/134.535095/140.828603/
156.612784/745.119087 ms，原始样本、CSV 和 CDF PNG 已归档于
`artifacts/edgefm-vla-goal/20260906-044509/cogact/series-005-native-formal-010/remote/`。
该结果包含显式 H2D/D2H；
Session/bundle 创建在计时循环前完成，首个 timed run 可能包含首次执行开销；仍不是
sensor-to-action、vendor baseline、replay 或板端结果。

14:02 UTC pi0.5 `required` 运行核验：使用三输入 processor-002 和 typed native
join 生成 bundle，IR/build 均通过；本地 RTX3060 由无 Python C++ runner 完成 owner
registered-1/reset/registered-2 三阶段握手、numerical worker bootstrap、1 次 warmup
及 2 次 measured 调用，exit 0。3 次 normalized F32 与 native F64 输出均对 direct/eager
参考逐字节一致，MSE/max-abs=0，finite/exact 全通过；独立审计见
`artifacts/edgefm-vla-goal/20260906-044509/openpi-inventory/local-pi05-replay-required-002/native4/independent-replay-audit.json`
（SHA `c0d6262e77fd9a0bc1931d0c5ac64ecaf38b75785c4097a0bc3e2fc7a07f3d4f`）。该 runner
只验证 required bundle 的真实输出，不导出 `vlaforge_model_session_get_replay_info`
计数；因此该旧证据仍记为 compile-time candidate/runtime telemetry unobserved，已被
14:48 UTC 的新 runner telemetry 复验 supersede，不能单独把这 3 次 exact 执行写成 replay。

实现补充：`render_resident_tensor_runner(..., replay_policy=...)` 现可从 IR/Plan 自动
枚举 bounded-loop task，在生成的通用 C++ runner 中校验并打印 `REPLAY` 的 state、
captured_steps、replay_count 和 ordinary_count；OpenPI bundle builder 将显式 policy
传入该接口。47 个 Python 定向测试通过；下一次重新 build 的 required bundle 才能使用
该 telemetry 形成运行期 replay 证据，当前旧 bundle 不回填。

Replay 构建入口更新：`openpi_output_bundle` 现将 bounded-loop policy 作为显式
`off|batch-only|required` 参数传入 compiler 和 bundle manifest，默认保持 `off`；33 个
IR/replay 定向测试通过。pi0.5 新 `required` bundle 已完成真实 C++ exact 输出核验，
但通用 typed runner 未导出 runtime replay counter；只能作为 required 构建/输出证据，
不能把它计作 replay 结果。

14:48 UTC replay telemetry 修正后复验：`local-pi05-replay-required-009` 新 bundle
（bundle SHA `b1094517ffcdfb92180630cb8065c9218f72709fb4f8e5b8546d744a1ac4a87f`，
runner SHA `0419ea9b101c88e22dffc54565b8749528bfa2fa52e90af3184247207fc5bf17`）在本地
无 Python C++ runner exit 0。task 13 的三条运行记录为
`captured_steps=10`、`replay_count=1/2/3`、`ordinary_count=0`；warmup 和两次 measured
的 normalized/native 完整输出均逐字节 exact。独立审计见
`.../local-pi05-replay-required-009/native/independent-replay-audit.json`，SHA
`3e4b34e11fdf5eecb8705a22e47c8d470f0ca14c72e93bdad4970bb49f9f901b`。这是短运行单观测
证据，不替代多观测 replay CDF。

同一源码随后完成全量 CPU 回归：2038 passed、71 skipped、82 warnings，75.57 s；结构化
报告 `artifacts/edgefm-vla-goal/20260906-044509/resume-cpu-regression-010.json`，日志
SHA `dc6125da89fb07ccaf0278fa42e2912ec84748229419bf4818577f477bcb9a0c`。跳过项仍是
明确的 CUDA/真实权重/额外依赖门禁，不计入板端或模型实验完成。

08:44 UTC 新结果覆盖上述进行态：H20 pi0 物化002完整无Python C++三次双输出
6900值逐byteexact，5实际映射库/39原ZIP成员与numerical lineage独立通过；本地
又按锁定官方数组复核输入及完整输出。仅1观测，无正式CDF或板端结论。RDT正式002
五进程5120样本已补冻实际DSO并独立通过，mean175.169159ms/p99182.002621ms；
历史表格数字只代表各自旧运行，不把新门禁追认到旧实验。

09:33 UTC 更新：Smol双输出正式002现已独立通过实际DSO及完整输出门禁，
15进程15360测量、34560完整张量对官方/同TS逐byteexact；三策略mean
94.420255/92.565081/80.006704ms，完整N10 replay对同源off降低15.2653%。
包含官方反归一化，但不含输入预处理/H2D/D2H/机器人传输。audit SHA
`6ff7e36c77ed416a629f85bf0cf3bfc9505319f8aa09c4f43afc42ea0e136e9d`。
pi0新增16不同观测reference/完整IR通过；正式五重复已由独立审计通过，5120
测量/11520完整张量，mean138.846511ms/p99 150.009736ms。
pi0.5新三输入processor保留旧acceptance且拒绝false/NaN；正式003使用16个
真实观测、五进程5120样本，完整C++双输出11520张量逐byteexact，mean
362.610401ms/p99 364.221098ms，CDF与运行库映射独立通过。上述新结果不回填到
下表或后文历史运行的冻结源码中；G4整模收益、G5质量门槛及板端仍未闭合。

| 论文主张 | 对应代码 | 当前证据 | 未闭合项 / 文稿约束 |
|---|---|---|---|
| III-A 两阶段、context 一次、迭代生成 | `frontend/invocation.py`；variadic `vla.for`；模型 Adapter | 通用连续/AR fixture 与 C++；SmolVLA 真实权重 eager/IR 10 步及完整 chunk 一致 | 两个逻辑阶段可以包含多个物理 artifact；不能把上游本来已有的 prefix-once 当成新增 N 倍节省 |
| III-A 单流 C++ 调度 | `runtime/execution_context.cpp`；AOTI/TS sidecar；`codegen/cpp.py` | generated Session自动绑定非默认stream；SmolVLA真实TS共享context三策略各32次完整输出对官方bitwise；AOTI兼容回归通过 | 同一 stream 不意味着硬实时或没有 host synchronization；TRT/BPU 不自动继承 LibTorch 实测结果 |
| III-A 整个 N 步图 replay | `runtime/bounded_replay.cpp` / 共享LibTorch graph provider / Python-IR-Plan-codegen | 新TS共享context三策略各5进程5120正式时延，17280完整动作对官方bitwise；10步replay无ordinary回退；mean92.831/91.427/77.410ms | 仅RTX3060常驻模型张量边界，replay对同context off降16.6%；不是sensor-to-action或板端证明；旧AOTI保真失败独立保留 |
| III-A predictive arena 与零分配 | `plan/memory.py`；runtime arena；allocator/lifecycle observer；unused-state pass | OpenPI完整IR fresh peak reserved由14.370降到6.780GB且全输出exact；独占stream lease+显式scoped graph回收后三策略240完整动作exact，销毁后active9568256B/reserved1925185536B恒定 | 仅Torch2.10显式scoped策略；每次graph销毁2次devicefree，不是零分配/零释放。默认retain不继承，失败quarantine仍保留资源；Plan未接管全部后端/权重 |
| III-A 跨 cycle 固定文本 prefix | builder 自动依赖；三域 revision；runtime cache | 显式/自动/default 版本碰撞、图像/state/mask 失效；SmolVLA real L4 cache/reset | 多模态混合 prefix 必须随视觉/state 变化失效；只有模型依赖允许时才能声称静态文本子图单独复用 |
| III-A sensor-to-action zero copy / deadline | 输入/输出/borrowed-view ABI | 无板端相机 DMA 或控制器 buffer 实测 | CUDA tensor ABI 不能替代 Orin/BPU 零拷贝链；设备零拷贝也不构成 deadline 概率保证 |
| III-B 等价低精度 / 每步尺度 | `analysis/precision_calibration.py`、`precision_probe.py`、`deployment/int8_linear.py`、`linear_precision.py` | Smol16episode的8/8隔离校准；真实INT8 kernel/64完整free-running轨迹；四新C++候选64完整动作byteexact；同源五路pilot240输出exact，native NSYS确认每次10个INT8 GEMM | 四组held-out仍0/8过MSE，cosine>0.99999不能称无损；pilot未证明稳定E2E收益或省内存，NSYS兼容警告保留；normalized空间，不事后改门槛 |
| III-B 非线性/掩码/RoPE 改写严格等价 | 对应后端 lowering / operator contract | H20真实OpenPI RoPE子图在ATen/Inductor各5独立进程中两路`[1,816,256]` BF16输出均bitwise exact；MSE/max-abs为0 | 该证据只覆盖已捕获的RoPE频率子图；tanh/polynomial GELU 与 erf GELU不能无条件当作同一浮点函数，有限mask仍需覆盖边界及全mask情况 |
| III-C heterogeneous backend | RegionExecutable v1/v2 + optional execution sidecar + plugin loader | 旧插件无 sidecar 仍可用；ABI/空 provider/失败清理测试 | Orin/BPU provider、fallback 分段、权重/arena/workspace 实测仍待补；不得写五模型双平台均已成功 |
| III-D Agent skill loop / E2E 收益 | `analysis/operator_capture.py`、`tensor_slice.py`、`terminal_tail.py`、`deployment/operator_selection.py`、公开benchmark与部署contract | H20真实 Attention/Linear/Norm/Embedding/RoPE 及 VLA solver 输入均有完整输出 correctness；H20 RoPE ATen/Inductor 各5进程、两输出bitwise exact；selection/2候选/平台/口径绑定及误配拒绝测试通过 | RoPE graph-batch 微基准 Inductor 比 ATen 低61.7583%，但尚无同候选整模E2E收益；其他候选有持平/失败/不稳定结果，未选择部署。不能把微基准收益外推到论文系统收益；Agent预算复用消融仍未完成 |
| IV-E MSE < 1e-5、cosine > 0.9999、严格无损 | `validation/deployment_metrics.py`；SmolVLA L3/L4 | 见下方真实权重结果 | cosine 条件通过仍可能 MSE 条件失败；误差约束不推出机器人行为数学上等价 |
| IV-E 数值策略可追溯与部署校验 | `numerical_context.py`；数值contract；LibTorch provider；显式worker bootstrap | LibTorch2.10完整22字段；pi05新C++3次双输出exact；RDT新C++32次统一8192+机器人896值exact，完整v2/provider随包且实际maps无Python | pi0完整双输出/tmp修复后待重新验；RDT新policy来自重载EP的全字段观察，不补填旧capture；Session只require-current、不拦外部setter；旧成绩不追认 |
| IV VLA完整部署与连续CDF | 公共TS导出、IR/Plan/C++、typed benchmark | H20上的SmolVLA、RDT-1B、pi0、CogACT有三策略正式CDF及完整输出审计；pi0.5的350.053ms正式结果由原始telemetry证实属于RTX 3060，不计入H20 | H20 pi0.5完整双输出正式实验仍缺；resident model-tensor不是sensor-to-action；各模型的episode、public-dependency、外部RNG和官方/vendor对照限制保留 |
| VLM 为单次无 action 子图 | 通用 bounded generation / state 接口 | 现有 AR fixture 验证通用语义 | 文本生成仍有 prefill + 多步 decode；Qwen3.5 vision/prefill/decode 不是单次前向的退化情形 |

## 已实测的数值边界

原始报告位于 `artifacts/edgefm-vla-goal/20260906-044509/local-smolvla/`。
profile 为真实 SmolVLA-Base 权重、人工单相机输入、N=10、完整 `[1,50,6]` normalized chunk，硬件 RTX 3060；不是正式真实观测实验，也不是板端结果。

| 比较 | cosine | MSE | max-abs | 结论 |
|---|---:|---:|---:|---|
| 官方 eager vs Invocation IR | 未在该旧报告统计 | 未在该旧报告统计 | 0 | 当前输入完整 chunk 精确一致 |
| 官方 eager vs exported execution | 未在该旧报告统计 | 未在该旧报告统计 | 0 | 当前输入精确一致 |
| 官方 eager vs AOTI conservative | 0.9999583668 | 0.0001106217 | 0.0222523808 | 旧宽容差通过，论文 MSE 条件未通过 |
| 官方 eager vs AOTI eager-numerics | 0.9999307966 | 0.0001214326 | 0.0328543782 | 仍有偏差，不选作消除误差的证据 |
| 直接 AOTI vs generated C++ Session | 未在该旧报告统计 | 未在该旧报告统计 | 0 | 部署调度没有新增数值偏差，不代表 artifact 对 eager 无损 |

表中数字分别来自 `eager-ir.json`、`smolvla-l3.json`、
`smolvla-l3-conservative-fidelity.json`、`smolvla-l3-eager-numerics.json`、`smolvla-l4.json`。
两个 fidelity 报告保留 raw reference/candidate、分维度统计和最坏元素。
同一 checkpoint 的 H100 eager/IR 与 L2 也已通过；H100 L3 在旧宽容差下通过，max-abs=0.0235134、MSE=7.16022e-5、cosine=0.9999711904；H100 L4 已实际通过完整chunk对同artifact exact和事务/cache/reset。H100同样未通过论文MSE门槛，不能由L4 exact抹去编译阶段偏差。

真实双相机published兼容性输入的fresh AOTI单样本误差更大（MSE=0.0010596353、max-abs=0.09716463），技术容差也未通过。旧发布processor丢失state统计的问题已通过完整历史checkpoint与500模型张量/所有action统计逐bit核对后，在独立恢复profile中修复；16episode真实state变换精确验证通过。

恢复配置单个真实观测已独立重跑：AOTI MSE=3.05882e-5、max-abs=0.01631856、cosine=0.9999948，技术门槛通过但论文MSE仍失败。generated C++连续fresh推理曾暴露CUDA copy与非阻塞stream竞态；修复后10次完整chunk逐bit等于同AOTI。这个结果证明此输入上的调度修复，不证明编译无损、多输入验收或延时CDF。

独立视觉前段诊断发现，暴露add/BF16边界或保留eager舍入后第一层偏差显著缩小；额外输出/clone会改变fusion，因此这些图仅用于定位，不冒充原artifact中间值。完整recovered模型的eager-numerics已实测16episode：论文门槛5/16、最坏MSE=7.618073e-5/max-abs=0.0308970958，最低cosine=0.999978934；11个失败不放宽门槛。相同16样本serialized IR完整重载全部exact。正式延时15进程已完成，17280完整输出对同artifact exact，但quality_gate保持failed。

SmolVLA ATen-preserving v6实际完整16样本技术门槛全过，但论文门槛4/16，最坏MSE=5.471928e-5。TorchScript早期默认/关闭JIT优化分别仅2/16与6/16，因为两者仍做archive device remap。后续同时保留CPU零维常量设备并关闭JIT优化后，公开通用导出接口四Region保存重载和16个真实观测完整IR全部bitwise；`torchscript-session-002-public`真实无Python C++的32次完整动作对官方亦全部bitwise。旧同步普通路径正式5进程/5120测量、5760完整动作对官方bitwise，mean94.718ms/p99 96.573ms。新`torchscript-aten-context/1` v2三策略正式15进程已完成：off/batch/required均值92.831/91.427/77.410ms，p99为94.255/93.051/78.133ms，合计17280完整动作对官方及同TS全字节一致、15360正式样本；required实际每进程10步capture、1152次replay、0 ordinary。完整raw/CDF/遥测/源码摘要独立复核，replay较同context off降低16.6%平均时延。三套campaign不混用，后续数值provider不追认到旧冻结源；没有零分配或板端/sensor-to-action证明。

pi0另暴露实际数值上下文缺口：官方模型初始化设matmul precision为high，独立EP默认highest产生偏差；旧20字段guard恢复的CPU saved-only冷启动仍有误差，单列未闭合。H100另发现官方channels-last图像被连续ABI准备流程改变布局，Adapter现将原布局恢复记录进导出图。修复后pi0/pi0.5真实CUDA reference、partition、4 Region捕获及独立进程完整IR均exact（normalized1600值；partition另核验native700值）。此项成功不抹去CPU失败，也不代表与JAX FP32对照通过。两模型旧四AOTI严格零容差均失败；pi05旧C++对同AOTI也有非零偏差。后续真正LibTorch2.10 provider已实现22字段实际getter/typed策略租约，591原生CPU检查与生成小TS Session正负例通过；尚未闭合新真实VLA capture和policy-bound C++，TRT仍未支持。observed/configured/runtime-checked/output-verified分开，不能把后端契约测试表述为部署保持所有数学语义。

RDT已完成H20真实官方完整参考、7 Region capture/reload及完整serialized IR8192值exact，包含在线T5、6个实际图像和全部5步动作。7/7 AOTI及3次无Python C++完整执行已结束，C++对同artifact bitwise；主活动14维MSE=4.668274110437259e-5，论文数值失败，不能用114个无效零维稀释成通过。Torch2.1与2.10的独立官方参考有602个值不同（MSE=9.2775e-7），单列跨框架版本差异。通用迭代接口携带模型输出历史、有效历史数及设备step index；Adapter保持官方DPMSolver方程及CPU scalar opmath舍入，不以Euler更新替代。

后续独立候选：pi0及pi05公开双阶段ATen-preserving四artifact的完整1600值均对官方bitwise（各audit012），尚无这两新包C++验收。RDT七TS已扩展同一真实episode的16帧及独立seed/noise，每帧在线T5、六图像编码和N5官方参考、IR与同一实际C++ Session全部8192值bitwise，活动14维MSE/max-abs=0；不是16episode泛化结果。H20的off-only正式benchmark已完成：5个独立进程各128 warmup + 1024 measured，5120正式时延及5760完整BF16动作对同TS/同Torch官方全部bitwise；mean174.162718ms、p99 178.194143ms、std1.376267ms，模型张量边界吞吐5.741757 fresh chunks/s。48调用pilot单列，不并入正式样本。原始输出、CDF CSV/PNG、遥测及源码/二进制身份已归档并在本地独立复核，见[RDT独立审计](/home/zhangzimo/Repos/private/edge-fm-x/artifacts/edgefm-vla-goal/20260906-044509/rdt/benchmark-ordinary-v2/independent-local-audit.json)和[时延表](/home/zhangzimo/Repos/private/edge-fm-x/artifacts/edgefm-vla-goal/20260906-044509/rdt/benchmark-ordinary-v2/latency-table.csv)。该边界不含输入预处理/H2D、输出D2H和机器人传输；活动动作语义不能重标为已物理标定。尚无RDT replay/Agent收益或板端验收，`full_paper_acceptance=false`，旧AOTI失真结果保留。

数值上下文另有明确失败边界：真实Torch2.10的reduction getter包含三个状态，旧20字段Python guard遗漏splitK，恢复后snapshot相等但完整状态不同。v1只能作为partial历史记录，不能补填未知值。新v2完整22字段、具体release/API域及三态pair恢复已在真实2.10 CPU环境45项验证通过；旧2.7 bool API明确标记splitK不可观察。pi05单因素policy对照及940份全量native/Python边界证据定位额外C++漂移，但旧模型capture并不包含v2数据。

显式worker bootstrap的原生验收已完成，见[独立汇总](/home/zhangzimo/Repos/private/edge-fm-x/artifacts/edgefm-vla-goal/20260906-044509/openpi-inventory/libtorch-bootstrap-verified-summary.json)：14个独立native CPU用例覆盖完整字段、9组三态pair组合、历史lease、一次故障回滚、不可恢复poison及线程身份复用；7种生成CPU fixture Session模式覆盖normal/multisession/initdrift/entrydrift/threaddrift/exitdrift/missing-bootstrap，全部通过，进程maps无Python。旧实现确实接受被复用thread id的新线程，修复后同一实际复用场景拒绝，原失败和修复源码身份分别保留。初始化仅由worker显式请求，Session仍require-current，不调用setter、不拦截外部setter；不保证第三方任意并发全局策略修改隔离。该报告明确`gpu_execution=false`、`pretrained_model_output_validation=false`，是实际LibTorch契约和故障恢复证据，不是OpenPI两模型v2捕获/部署已完成；旧Smol/RDT/CogACT冻结成绩也不能继承此证明。

CogACT两seed的完整reference candidate、partition、Invocation及四strict EP保存重载全N10输出和11次CUDA随机draw消费均bitwise；DDIM eta=0仍保留官方每步draw。递归effect audit检查prefix 97个嵌套子图通过。Adapter保持实际batch1无padding的原生causal decoder语义，未用图改写掩盖偏差；旧strict失败链完整保留。max_positions2048/4096单字段对照在本次实际position范围内exact，不等于已取得Meta原config。

CogACT公开device-preserving TS的ordinary部署切片已闭合，见[torchscript-slice-closeout-004](/home/zhangzimo/Repos/private/edge-fm-x/artifacts/edgefm-vla-goal/20260906-044509/cogact/torchscript-slice-closeout-004/report.json)：四真实EP经公共导出接口保存/重载及完整N10执行通过；H20同一个无Python C++ Session按seed42/43/42连续三次fresh，全部raw/normalized/native `[16,7]`动作、16字节receipt和int64 draw-count输出对同TS及已有捕获逐字节一致，三种动作亦对独立官方candidate逐字节一致。旧IR中受支持dtype别名经显式conversion ledger转换，保留原/新IR摘要，公开compile重建证书；未修改历史TensorType解析语义，也未继承旧证书。仍只有一个真实Fractal观测、两个seed和`official-code-public-dependency-candidate`身份；Meta原config、更多观测、正式时延/CDF、replay、运行时数值provider绑定及板端均未验收，`paper_fidelity_gate=not_verified`。外部producer负责真实11次CUDA RNG消费，C++仅消费tape/receipt，不是自主PRNG。完整checkpoint安装有7,630,224,071元素，其中未消费lm_head不在在线forward中计算，不能沿用论文3B规模标签。

## 内存主张的新增实测边界

[allocator-pilot-v1独立审计](/home/zhangzimo/Repos/private/edge-fm-x/artifacts/edgefm-vla-goal/20260906-044509/execution-context/allocator-pilot-v1/independent-review.json)重新核验430个冻结文件哈希及三策略共144个完整`[1,50,6]`输出，全部对direct/eager字节一致。各策略16次warmup后32次调用，off/batch-only各410912次LibTorch分配器请求，required为52512；区间均值分别12841/12841/1641次每调用，不能作为逐调用分配分布。各策略稳态device alloc/free、retry、OOM及allocator事件回收同步计数增量为0，reserved current/累计peak均无增长；required在warmup后比off/batch多保留4194304B。allocator同步计数不是全部CUDA同步次数。

这些计数只覆盖进程/设备的LibTorch native allocator，不包括runtime arena/caller cudaMalloc、CPU heap及其他CUDA库分配；device alloc为0不等于零分配。快照位于计时区间外，单次时延仍为Session run加CUDA完成，计数差分则跨完整测量循环及图外校验。累计peak不等于独立稳态阶段峰值，cudaMemGetInfo不是进程所有权数据。销毁Session后仍有2个active/allocated块共9568256B，尚未做跨生命周期增长和存活分配归因，不能宣称无泄漏或已证实全局cache。v1审查另发现计数守恒与聚合证据绑定门禁缺口，实际raw记录通过独立守恒/哈希检查。后续[v2离线重验](/home/zhangzimo/Repos/private/edge-fm-x/artifacts/edgefm-vla-goal/20260906-044509/execution-context/allocator-v2-revalidation.json)使用修复后的严格JSON解析、相邻计数守恒及raw/report/execution绑定校验，三策略均通过；报告SHA为`59aee16b450333d76f2c9152f7e7cf56e1cdd56aedd1abdacc9efd3e7d9b1a61`。这是新验证器对原始文件的离线审计，无新GPU执行；旧v1报告未改，仍不含新版process binding，不能回写为v1已防篡改。该pilot既不是整链零分配证明，也不替代长稳/高水位/内存优化消融。

## 优先收敛的叙述

### 后续证据不追认历史

20:08 UTC更新：RDT当前BF16[1,64,128]是统一模型动作，不是官方机器人输出；sorted active14还需左右关节重排和BF16夹爪缩放，不能把有效维度数相同当作同一动作空间。新Adapter接受显式Tensor `output_transform`，原路径不变，新路径保留统一动作aux；完整新Session尚未验收。Smol normalized v2与RDT unified板端数据包已移机核验，均没有target driver或实机结果。

G4新terminal-tail三路direct/IR在16真实观测上96动作、960完整carry exact，独立核验2462文件/507权重常量项；新分区没有继承旧TS或C++证书。原EP经未改计算control重存时有72项内部FakeTensor storage_offset重建，边界ABI和所有算子数学不变且实测exact；不能写成归档metadata完全不变。

G5新五路原生pilot已独立核验240输出；四量化策略仍全部不通过held-out质量门禁，单进程32点没有稳定时延收益，allocated peak/归档大小及allocator请求未减少。独立NSYS证实每次完整量化调用10个INT8 GEMM，但保留工具兼容警告，不把profile时间混入pilot/CDF。逐步误差分析的velocity差异使用各自free-running carry，不是teacher-forced的单算子量化误差。

pi05新RTX原生三次normalized1600值已exact且完整v2策略/无Python通过；独立native700后处理CUDA capture/AOTI亦exact，双输出新build首次因Region id和新IR顺序不符被公共门禁拒绝，修正Adapter映射后需新运行。pi0双输出004由PyTorch2.10 legacy AOTI loader硬编码/tmp且overlay已满导致失败；TMPDIR指向NAS不能修复该上游行为，005未启动，不触碰其他任务临时文件。

19:35 UTC更新：Smol正式TS、allocator和INT8的F32[1,50,6]是**归一化**输出。源finish只有slice/copy/finite，未执行反归一化；旧板包v1及部分进度文字的native-scale标签有误，原manifest和raw不改，单列勘误及新normalized包。公式校验过的独立postprocessor不是正式Session已包含后处理的证明。还必须补真实反归一化Region、保留normalized aux并重新核验完整C++；旧时延只覆盖原张量边界。

新增`int8-trajectory-errors-v2/`从原始free-running轨迹与逐步hash验证的参考carry生成2560指标和active6/full32曲线；最终64动作metric逐字段匹配原报告，四组held-out仍0/8。逐步尺度首步均值较低，最终均值未优于全局尺度，没有取得论文逐步低精度优越性证据。新INT8无Python C++已对对应候选全部exact，但执行保真不等于官方质量通过，性能/反归一化验收仍分别待补。

新增graph scoped回收路径已在完整240动作/75快照验证健康生命周期恒定内存，默认retain不继承。另同源retain/scoped共480动作/150快照核验，required销毁+drain中位数37.610/38.140ms；仅各5点、范围重叠，不宣称差异显著或归因于2次devicefree。详细失败/quarantine与版本边界仍见独立生命周期报告。

上述旧数值/provider/allocator段落保留各自冻结时点。2026-09-06 18:25 UTC的新状态以首表和`edgefm_vla_goal_status.md`为准：pi0 native004已完成新v2绑定的normalized输出部署；native后处理700值的CUDA捕获/AOTI已exact，但两次完整双输出部署均因外部GPU owner出现而安全中止，未记通过。Smol跨生命周期活动内存增长已有地址级BLAS归因和stream lease修复，仍留独立graph私有pool增长问题；新scoped回收不得追认为旧v3运行过。

真实INT8完整轨迹的审核仅证明执行和记录可靠，不能写成数值质量通过：四策略held-out均未通过预声明MSE门槛，且此时无低精度C++或性能证明。D1/D2已有16真实episode的可移机数据包、通用target descriptor及四阶段脚本，八次离线pending路径均无driver执行；板端driver/SDK/实机仍未提供。参数口径与当前可引用时延见`vla_model_metric_boundaries_20260907.md`，当前CogACT实际安装模型不是3B。

统一生成公式应表述为 `(state_next, outputs) = F(state, context, schedule, rng)` 的有界状态转移。
`sample + dt * velocity` 是 SmolVLA/OpenPI 对应配置的模型公式，不覆盖 RDT 多步 scheduler、CogACT CFG 或 AR token/KV。
观测 tuple 需包含实际 proprioceptive state、masks 与显式 noise/随机状态，不能仅列 image 和 text。

“determinism-first”可表述设计目标和实测分布，不能由 batch=1/single stream 推出严格 deadline 保证。
模型参数量需分别记录完整 pipeline、在线活跃与常驻部分，尤其 RDT 编码器与 CogACT 主干。
“首个”“weeks to minutes”“zero accuracy degradation”和当前 PDF 中尚无本轮证据的 Orin/BPU 数字应在补证前保持待验证，不能作为完成事实。
