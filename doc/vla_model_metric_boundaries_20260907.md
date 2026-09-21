# VLA Model Metric Boundaries

本表离线汇总本 Goal 已有 strict-load/转换/权重恢复记录，源报告 SHA 再次记录在
`artifacts/edgefm-vla-goal/20260906-044509/model-count-scopes-v1/report.json`。
不是一次新模型推理，也不是根据文件大小推测参数或显存。

## 规模口径

| 模型/实际版本 | 记录中的元素数 | 必须保留的范围说明 |
|---|---:|---|
| SmolVLA recovered-statistics | 450,046,176 | 500 个模型状态张量逐字节核验；此项原始记录为模型张量元素数，不额外推断全部属于可训练参数 |
| RDT-1B 在线 pipeline | 6,418,856,128 | policy 1,228,319,872 + T5 encoder 4,762,310,656 + SigLIP encoder 428,225,600；不能只用 1.2B 解释完整端到端成本 |
| pi0 转换后的实例化模型 | 3,501,372,176 | 包含 263,323,648 个显式未使用、补零的 expert vocabulary-head 元素；不是这些补零值也是已训练执行权重 |
| pi0.5 转换后的实例化模型 | 3,616,757,520 | 同样包含 263,323,648 个上述未使用 head 元素；JAX/PyTorch 数值等价未由转换证明 |
| CogACT-Base public-dependency candidate | 7,630,224,071 | 完整安装栈含 LLM/两视觉主干/projector/DiT；本次不是论文所写的 3B 实例，原 Meta 配置仍未取得 |

这些计数的语义并不完全相同，不能不带范围说明直接做参数效率排名。除 RDT
已有单独在线子网络计数外，其余行不在本表推断在线活跃元素总数。移除未计算
模块、移除未使用的导出状态、序列化文件缩小和实际 CUDA 常驻/峰值下降是不同指标。

CogACT 不应继续标成 3B。论文应标注实际测试的 CogACT-Base 完整栈，或另行确定
目标 3B checkpoint 并完成新的参考与部署实验，不能重标已有 7.63B 结果。

## 性能口径

当前可用正式 CUDA 数字来自不同设备，不构成跨模型公平速度排名：

历史 Smol/RDT 正式记录存在重建 runtime DSO 启动前身份未冻结的缺口，原数值与
时延保留，但不算完整运行库身份验收。下面另列 Smol/RDT 正式 002，已补该门禁并
独立复核；不能把新门禁追认给历史运行。

| 模型/执行方式 | 实际设备 | 正式测量数 | Mean / p99 (ms) | 已核验范围 |
|---|---|---:|---|---|
| SmolVLA shared-context off | RTX 3060 | 5 x 1024 | 92.831385 / 94.255179 | 与同一冻结 runtime 的另外两策略对照；完整动作对官方及同 TS byte exact |
| SmolVLA shared-context batch-only | RTX 3060 | 5 x 1024 | 91.426849 / 93.050652 | 同上 |
| SmolVLA full N=10 replay | RTX 3060 | 5 x 1024 | 77.409843 / 78.132581 | 对同 context off mean 降低 16.6124%；不是加入后续数值 provider/内存修复后的新成绩 |
| RDT 完整在线编码器 + N=5 普通执行 | H20 | 5 x 1024 | 174.162718 / 178.194143 | 同 episode 16 分散帧/16 seeds，全 8192 BF16 值及 896 活动值分别 byte exact；无 replay 对照 |
| RDT 新v2完整统一输出 + 官方机器人输出 | H20 | 5 x 1024 | 175.006496 / 179.051004 | 每次8192 unified+896 robot值逐byteexact，含预热5760调用/11520完整张量；5.714074 chunks/s，std1.307761ms；同episode16history |
| RDT 正式002，实际 runtime DSO 冻结 | H20 | 5 x 1024 | 175.169159 / 182.002621 | 5760调用/11520完整张量对官方及同TS逐byteexact；5.708768 chunks/s，std1.477469ms；原前三重复不重跑，仅补两次 |
| pi0 系列001 off，实际DSO冻结 | H20 | 5 x 1024 | 138.846511 / 150.009736 | 16真实同episode frames，normalized/native双输出11520张量逐byteexact；7.202198 chunks/s |
| pi0.5 正式003 off，实际DSO冻结 | RTX 3060 | 5 x 1024 | 362.610401 / 364.221098 | 16真实同episode frames，normalized/native双输出11520张量逐byteexact；2.757781 chunks/s |
| Smol 双输出正式002 off，实际DSO冻结 | RTX 3060 | 5 x 1024 | 94.420255 / 95.708937 | 完整normalized及官方反归一化输出，16真实episode；10.590948 chunks/s |
| Smol 双输出正式002 batch-only，实际DSO冻结 | RTX 3060 | 5 x 1024 | 92.565081 / 94.121297 | 同一冻结runtime及完整双输出；10.803210 chunks/s |
| Smol 双输出正式002 N=10 replay，实际DSO冻结 | RTX 3060 | 5 x 1024 | 80.006704 / 80.902536 | 同源off mean降低15.2653%，三策略共34560张量逐byteexact；12.498953 chunks/s；不是Agent收益 |

这些边界是输入 CUDA Tensor 已就绪到完整模型动作完成，不含相机/解码/传输，
也不含初始化与 H2D/D2H。正式时延、全部输出、CDF、owner/温度/频率记录分别在
`execution-context/torchscript-native/formal-context-v2/` 和
`rdt/benchmark-ordinary-v2/`。后续源码修复不能继承这些已冻结时延。

上表旧Smol三行是归一化输出，旧RDT 174.162718ms行是统一动作存储空间；RDT的896个活动值不等于官方
896个机器人动作值，后者还包含左右关节重排和BF16夹爪缩放。新增
`rdt/dual-output-003/`已对32次完整C++调用的两个输出端口逐字节核验通过，
但旧174.162718ms不包含这次新增输出Region。新`rdt/dual-output-formal-001/`
已完成五进程正式采样和独立全字节/CDF复核，对应上表新增行，不据两次不同
runtime快照之间的差值单独归因后处理成本。其完整robot输出仍不等于实机
物理标定，所有时延排除D2H、核验和机器人传输。

`rdt/dual-output-formal-002/independent-audit-001.json` SHA
`46ce66c4dd154e03ffe7afefb6b4fd4eca90043b0a092a9b9d7437b736c3a6e5`
核验 449 个冻结条目、849 个实际文件和全部输出；新 CDF 保留 max195.236780ms。
这次是网络中断后同一冻结工作负载的分时段进程重复，不是同一连续进程运行，
也不能把与001的小差异归为优化收益。Smol 双输出正式001保留其历史实测数据；
正式002已补冻DSO并独立验收，证据在
`smol-native-output-formal-002/independent-audit-001.json`，SHA
`6ff7e36c77ed416a629f85bf0cf3bfc9505319f8aa09c4f43afc42ea0e136e9d`。
其三策略15进程共15360测量、17280调用/34560完整张量和CDF已复核；不能用
旧归一化成绩代替新双输出成绩，也不将两次不同runtime的差值归因为后处理成本。

CogACT 目前的 90.953 ms 是单观测/两 seed/48 次调用的 pilot，不进入正式主表。
pi0/pi0.5 已完成当前 resident-tensor CUDA 正式性能矩阵，但没有 replay 或
vendor 对照。Orin/BPU 没有本轮实机成绩，不能用上述 CUDA
数字、已编译 artifact 或待运行输入包填充相应格子。

## 质量口径

同精度完整参考、对同产物 C++ 一致性、跨版本/设备一致性和量化误差分别报告。
完整动作包括实际输出维度，RDT 另报告活动维度，整数辅助输出不转成浮点比较。
SmolVLA 四种 INT8 head 尺度策略的真实 free-running 留出集均为 0/8 通过
MSE < 1e-5，尽管余弦都超过 0.99999。因此不把高余弦、编译成功或候选 C++
一致性表述为无损，也不掩盖已归档的失败方案。

通用terminal-tail的新H20三路C++共48个完整normalized输出与同卡原EP/TS
全部逐字节一致，但H20原EP相对既有RTX官方参考本身0/16逐字节一致、4/16
达到原数值门槛。因此仅证明本次同H20分区/部署未增加观测到的差异；尚不能
把RTX官方参考当作同H20 eager baseline，也没有选中优化kernel的E2E收益证明。
