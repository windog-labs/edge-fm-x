# VLAForge Invocation IR v0.2：被动式有状态模型部署编译框架

## 1. 决策状态

本文档是 VLAForge 当前的权威架构决策。它覆盖 v0.1 中把
`ClockDomain`、`Policy.clock`、`RunTick`、物理时间 freshness 和 action
publish 放进核心语义的设计。

VLAForge/EdgeFM 的调用者是底软。VLAForge 只负责把一次 VLA 模型调用及其
跨调用状态编译成高性能部署代码，不负责：

- 传感器时间同步、帧率维持、周期调度、timer、sleep 或 rate control；
- 丢帧策略、ROS/Cyber topic、消息队列；
- 底盘动作发布或任何外部控制 I/O；
- 通用实时系统调度。

核心抽象是：

```text
底软 BindInput(TensorView, optional InputStamp)
     -> Session::Run()
     -> ReadOutput()
     -> 底软自行发布/执行输出
```

每次 `Run()` 是一次被动、同步、有界的 invocation。VLAForge 内部不得启动
维持频率的线程或定时器。

## 2. v0.2 Semantic IR

v0.2 schema：

```text
vlaforge.semantic_ir/0.2
```

核心只保留六类业务语义：

1. Stamped Input；
2. Versioned Authoritative State；
3. Pure TensorRegion Invoke；
4. 结构化 `if` 与有界 `for`；
5. Exact Cache/Reuse Contract；
6. Transactional Output。

建议的数据模型：

```text
Module {
  name
  inputs: InputPort[]
  states: StateSlot[]
  regions: TensorRegion[]
  invocations: Invocation[]
}

InputPort {
  name
  payload_type
}

InputStamp {
  revision: uint64?       // exact identity；缺省时每次 Bind 都生成新 revision
  timestamp_ns: uint64?   // 可选 metadata，不做同步、不驱动调度
}

StateSlot {
  name
  payload_type
  retention
  reset_on_episode
  ownership
}

Invocation {
  name
  body
}
```

核心 operation：

```text
vla.input.read
vla.txn.begin
vla.state.read_latest
vla.snapshot.value
vla.invoke
vla.if
vla.for
vla.validate
vla.state.stage_write
vla.output.create
vla.txn.commit
vla.return
```

`async/await`、多物理 clock、deadline/jitter、内部事件调度、复杂
`StateScope` 不属于 v0.2 核心。以后如果确有模型内部并行需求，应首先作为
backend/CUDA Graph 的执行细节，而不是传感器或底软调度语义。

## 3. InputRevision 与 exact cache

### 3.1 输入身份

`InputRevision` 只回答“这次绑定的数据与上次是否是同一份逻辑输入”，不回答
传感器是否同步。

- 调用者显式提供相同 revision：允许 exact cache 命中；
- revision 变化：所有依赖该输入的 exact cache 必须失效；
- 调用者不提供 revision：runtime 为每次 `BindInput` 分配新的单调 revision，
  禁止跨 `Run()` 复用；
- `timestamp_ns` 仅随 trace/metadata 保留。核心不等待、不对齐、不丢帧。

### 3.2 exact key

exact cache key 必须由编译器证明并显式生成：

```text
ExactCacheKey = (
  episode,
  model_identity,
  artifact_identity,
  region_identity,
  InputRevision...,
  StateSnapshot.version...
)
```

禁止用 Tensor 地址、内容 hash 猜测、时间间隔或“看起来没变”作为 exact key。

### 3.3 三种常见 VLA cache

- VLM prefix/condition embedding：适合跨 `Run()` exact revision cache；
- autoregressive decode KV：属于一次 `Run()` 内的 loop-carried SSA，不是
  authoritative state；
- DiT condition feature：可用 exact revision cache + LICM；
- 近似 DiT feature/residual reuse：未来单独使用 explicit guarded reuse
  contract。它必须报告误差/guard，不得伪装成 exact memoize。

## 4. State 与事务

### 4.1 authoritative state

`StateSlot` 只表示不能静默丢弃的跨 `Run()` 模型状态，例如：

- action queue 与 cursor；
- previous action；
- recurrent hidden state；
- 显式 RNG state。

读取总是 `read_latest`，结果包含不可变 payload 和内部单调 `version`。IR 不再
计算 `EpochExpr.current/next`。

### 4.2 commit

一次 invocation 的状态流：

```text
read_latest -> compute -> stage_write -> validate output -> txn.commit
```

成功 commit：

- StateStore 为每个 staged state 内部分配 `version + 1`；
- committed output 与这些新 state version 原子可见；
- `ReadOutput()` 返回本次 committed output。

abort、backend 失败或 validation failure：

- staged state 全部丢弃；
- state version 不增加；
- 不覆盖上一次 committed output；
- `Run()` 返回错误。

`ResetEpisode()` 清除声明为 episode-reset 的 authoritative state、清空 committed
output，并使所有 derived cache 失效。

## 5. Transactional Output

`vla.action.publish` 从核心删除。新语义是：

```text
candidate tensor
  -> validate
  -> vla.output.create
  -> vla.txn.commit
  -> vla.return committed_output
```

生成的 C++ Session API：

```cpp
Status BindInput(
    uint32_t input_id,
    const TensorView& value,
    const InputStamp* stamp = nullptr) noexcept;

Status InitializeState(
    uint32_t state_id,
    const TensorView& value) noexcept;

Status Run() noexcept;

Status ReadOutput(CommittedOutput* output) const noexcept;

Status ResetEpisode(uint64_t new_episode) noexcept;
```

`ReadOutput` 只读内存，不执行 topic publish、RPC 或底盘 I/O。

## 6. 四类内存

| 类别 | 生命周期 | 能否静默失效 | 例子 |
|---|---|---:|---|
| External input/output | 调用方管理 | 不适用 | image/state/token input，committed output |
| Per-Run temporary/static arena | 一次 `Run()` | 可以 | activations，loop carry，workspace |
| Authoritative persistent state | 跨 `Run()`/episode | 否 | action queue/cursor，previous action，RNG |
| Derived cache | 跨 `Run()`，可重算 | 是 | VLM prefix/KV，condition embedding，DiT feature |

编译器和论文必须明确区分 authoritative state 与 derived cache。cache miss 或
reset 可以重算 derived cache；authoritative state 丢失会改变模型语义。

## 7. 旧 → 新 API/IR migration map

| v0.1 | v0.2 | 迁移规则 |
|---|---|---|
| `ClockDomain(period/deadline/jitter)` | 删除 | 底软负责调用周期与 deadline |
| `Policy(clock, tick)` | `Invocation(name, body)` | invocation 无物理时钟 |
| `Session::RunTick(Epoch)` | `Session::Run()` | 外部连续调用 |
| `Epoch(clock, sequence, timestamp, episode)` | `InputStamp(revision?, timestamp?)` + `episode` | revision 只标识输入；episode 由 Session 管理 |
| `vla.sample_input -> value, EpochType` | `vla.input.read -> value, InputRevisionType` | 无 revision 时 runtime 自动生成唯一值 |
| `FreshnessConstraint(max_age)` | 核心删除/metadata | 底软或 adapter 做 freshness policy |
| `EpochExpr.current/next` | 删除 | state version 由 commit 内部分配 |
| `vla.state.read(epoch)` | `vla.state.read_latest` | 返回 `Snapshot<T, version>` |
| `vla.state.stage_write(epoch)` | `vla.state.stage_write` | 不携带 next epoch |
| `ActionType` | `PendingOutputType` | 输出不等于外部发布 |
| `CommittedActionType` | `CommittedOutputType` | 事务提交后的模型输出 |
| `vla.action.publish` | 删除 | `vla.return` + `ReadOutput` |
| `ActionQueue` runtime | `OutputStore` | 只保存 latest committed output |
| epoch cache dependency | input revision/state version dependency | exact key 加 episode/model/artifact identity |
| `vla.async/await` | v0.2 非核心 | backend overlap/CUDA Graph 后置 |
| 多种 `StateScope` | episode-reset authoritative state | 复杂 scope 后置 |

旧 textual/JSON schema 不保证兼容；转换脚本只有在实现成本很低时提供。代码和
测试以 v0.2 为唯一 release gate。

## 8. 编译流水线

```text
Restricted Python/model adapter
        |
        v
Invocation Semantic IR v0.2
        |
        +-- verifier:
        |     input revision provenance
        |     state snapshot/version
        |     bounded control flow
        |     validation dominates commit
        |     one successful commit/output
        |
        +-- exact memoization / invocation LICM
        |
        v
Scheduled Execution Plan
        |
        +-- per-Run temporary arena
        +-- authoritative state arena
        +-- derived cache arena
        |
        v
C++ Session::Run / ReadOutput
```

compiler profile 的准确表述是
“legality-checked stateful invocation whole-program transformations”，不是
通用实时调度或任意 plan synthesis。

## 9. 验收矩阵

必须自动化验证：

1. Session 无 timer/sleep/rate control；外部可连续 `Run()`；
2. 同 revision exact cache hit，新 revision invalidation；
3. 未提供 revision 时连续 Bind 默认 miss；
4. commit 后每个 staged state version 恰好 `+1`；
5. abort/validation failure state version 不变；
6. `ResetEpisode()` 清 state/output/cache；
7. SmolVLA/π0 action queue 跨 `Run()` 正确消费与 refill；
8. OpenVLA decode KV 是 loop-carried SSA；
9. Semantic interpreter、Plan executor、generated C++ 的 output/state trace 等价；
10. generated runner 不链接 Python；
11. Orin arm64 build 通过；真机 latency/power/closed-loop 后置。

## 10. 论文主张

论文不再主张“VLA 时序调度器”。推荐主线：

> VLAForge is a stateful invocation whole-program compiler that makes
> persistent model state, exact input identity, derived caches, and
> transactional outputs explicit, then lowers the verified program to a
> no-Python C++ deployment session.

三项贡献调整为：

1. VLA-specific Invocation IR：区分 stamped input、authoritative state、
   derived cache 和 transactional output；
2. revision/version legality-checked whole-program transformations：exact
   memoization、LICM、state physicalization、static arena；
3. verified C++ AOT lowering：`Run/ReadOutput`、state/output atomicity和
   Semantic/Plan/C++ trace validation。

底软调度、传感器同步和动作发布明确在系统边界之外。
