# EdgeFM 论文实验交接包

日期：2026-10-26，Asia/Shanghai。同步整个文件夹即可，不需要原工作区。

## 阅读顺序

1. [EdgeFM指标与baseline对应关系](01_edgefm_results.md)：先确认哪些列是我们框架的实测结果。
2. [三平台总汇总](02_all_platform_results.md)：主表、Qwen、CDF、质量核验、算子/集成消融、资源和未测边界。
3. [框架对照CSV](tables/edgefm_framework_comparison.csv)、[H20带方法名主表](tables/H20_vla_framework_labeled.csv)、[Orin带方法名主表](tables/Orin_vla_framework_labeled.csv)。

## 目录

- `figures/`：按H20/J6M/Orin分组的PNG/PDF，可用于论文制图参考。
- `data/`：原始测量/精确CDF/质量/算子/资源CSV、保存的审计或明确标注的审计节选。
- `reports/`：各平台详细报告可移植副本，旧状态按各自日期解释；最新范围以总汇总为准。
- `tables/`：便于论文表格编辑的CSV；paper_method是新增说明列，不改变数值。
- `paper/`：原论文PDF，仅供对照，未修改。
- `MANIFEST.json`、`SHA256SUMS`：包内文件与源文件哈希/转换记录；`evidence/`保留未打包历史引用说明。

## 不能省略的边界

H20/Orin有五VLA完整框架部署与所列runtime模式实测，但不是统一启用全部Agent算子优化的Full EdgeFM配置；算子回集成的负收益/失败单列。J6M只有SmolVLA阶段结果，没有合格完整resident E2E。H20已测Qwen2B，Orin2B用户取消。Orin baseline是同一EdgeFM Session普通TRT执行，不是vendor独立baseline；小收益、负结果、跨设备误差与RSS缺口必须保留。

本包是论文修改材料，不是模型推理复现包。不含模型权重、engine/HBM、巨大逐tensor输出或构建环境。大于2MiB的审计JSON提供带源SHA256的`.extract.json`，明确不是完整原审计，也不冒充重新审计。JSON中只保留来源哈希及不含机器地址的记录标识，本包不依赖原工作区。

## 自检

```bash
python3 tools/verify_package.py
```

需要Python3.9或更新版本，仅使用标准库，不联网、不连接台架、不运行模型。所有Markdown本地链接和图片均已改为包内相对路径。
