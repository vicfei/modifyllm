# modifyllm 设计文档

> **口号：Modify with receipts —— 带回执的模型改造。**
> 一个接口，三个算子（编辑 / 微调 / 蒸馏，外加合并），每个修改默认携带：匹配对照筛查、三段式评估、可回滚快照、机器可读回执。

版本：v0.2 草案 · 2026-11-07 · 状态：待评审
变更：v0.1 → v0.2 依据外部评审修订——修正编辑符号约定、三段数据协议、对照统计学、事务状态机、Receipt 结构化、验收口径拆分、v0 范围收缩。

---

## 1. 定位

**是什么**：一个统一"模型改造算子"的 Python 库 + CLI（skill 打包后置）。三个算子共享目标描述与回执格式，使"同一目标、不同算子"的经验对照可比。

**不是什么**：不是训练框架。微调委托 LLaMA-Factory / Unsloth，蒸馏委托 easydistill，合并委托 mergekit；只有**权重编辑**（weight-space programming）自研——当前的工具真空地带，也是技术核心。

**理论对齐**：[awesome-agentic-ml](https://github.com/vicfei/awesome-agentic-ml) "三修改算子"框架的实物化——能力获取（微调）、能力转移（蒸馏）、行为手术（权重编辑）。

## 2. 设计原则

| # | 原则 | 含义 |
|---|---|---|
| P1 | **回执一等公民** | 任何算子的任何修改（含失败与放弃）都产出 receipt；无回执的修改视为未发生 |
| P2 | **验证先于应用** | 编辑在暂存副本上做；通过筛查才原子提交；选择与最终评估严格分离 |
| P3 | **算子统一、实现委托** | 共同 GoalSpec + EvalSpec，算子差异放 OperatorSpec 扩展；不重造轮子 |
| P4 | **安全有边界** | 不提供移除安全/拒绝行为的预设；自定义配对开放但默认留痕、完整性可验证 |
| P5 | **agent 原生** | CLI 可无交互驱动；输出机器可读；skill 打包后置到 v0.2 |

## 3. 概念模型

```
ModelHandle   模型引用：路径或 HF id + revision + 文件清单哈希
GoalSpec      共同目标：行为描述 + GraderSpec（可序列化、带版本；CLI 下禁止内联代码）
EvalSpec      共同评估：三段数据 + 指标定义（数值+单位+阈值）+ 随机种子
OperatorSpec  算子扩展：
              · EditSpec:    pairs、目标层、recipe、α 网格、搜索预算
              · FinetuneSpec / DistillSpec / MergeSpec: 各自后端参数
ScreenProtocol 匹配对照 + 门禁（困惑度/回归）+ 置换检验；阈值全部进协议
Receipt       事务回执（见 §6）
Session       ~/.modifyllm/runs/<run_id>/  独立运行目录 + 锁
```

**事务不变式（两阶段提交）**：

```
started → running → (passed → applied) | (failed → discarded) | interrupted
                                                    └→ rolled_back（回滚追加状态并生成事件回执）
源快照 → 暂存候选(临时目录) → 筛查 → 原子提交(rename) 或 丢弃 → 终态回执
```

- **原子提交**：多分片权重先写入临时目录 + manifest，再 rename 到正式快照目录；崩溃时运行目录标记 `interrupted`，下次扫描可清理或续报。
- **并发**：每个 run_id 独占目录 + 文件锁；快照目录只增不改。
- **回滚**：恢复**源快照**（修改前状态）；原回执追加 `rolled_back` 状态，并生成一条 rollback 事件回执（引用原 receipt_id）。`rollback` 字段永远指向源快照，与 `model_out`（产物快照）严格区分。

## 4. 架构

```
modifyllm/
├── src/modifyllm/
│   ├── core/
│   │   ├── probe.py        # 隐状态采集与方向提取（§5.1）
│   │   ├── edit.py         # rank-1 投影编辑（§5.2）
│   │   ├── screen.py       # 筛查编排：对照、门禁、置换检验、预算控制
│   │   ├── receipt.py      # 回执 schema、状态机、写入、校验、迁移
│   │   └── txn.py          # 两阶段提交、快照、回滚、运行锁
│   ├── adapters/           # finetune / distill / merge（v0.3）
│   ├── evals/              # exact / mc / GraderSpec 运行器
│   ├── cli.py              # edit / verify / rollback / receipts / runs
│   └── schema/             # pydantic 模型（GoalSpec/EvalSpec/Receipt）
├── tests/
│   ├── unit/               # 无 GPU：数学确定性测试（含 ±α 符号测试）
│   ├── stat/               # 合成信号检出 + 零假设假阳性率实验
│   ├── txn/                # 提交/崩溃/回滚路径
│   ├── integration/        # tiny 模型端到端（CPU 分钟级）
│   └── golden/             # 回执 schema 与不变量比对（数值用容差）
├── examples/
└── docs/
```

依赖 v0 极简：`torch / transformers / safetensors / tydantic? → pydantic / typer / numpy`；量化编辑（GGUF/AWQ）明确出 v0 范围。

## 5. 核心算法规格（v0：单 recipe rank-1 编辑）

### 5.1 方向提取（probe）

**采集约定**（逐项固定，不留实现自由度）：

1. 对每个配对 `(prompt, wanted, unwanted)`：以 **teacher forcing** 分别前向完整 response；
2. 取 **response 区间全部 token**（不含 prompt）在**目标层残差流**（该层输出、即 down_proj 写入之后的位置）的隐状态，按 token 取**均值**得到单个向量 `h_wanted ∈ R^d`、`h_unwanted ∈ R^d`；
3. 配对差分 `δ_i = h_unwanted_i − h_wanted_i`（**注意方向：待消融行为在前**），跨配对取均值后 **L2 归一化为单位向量 u**；
4. v0 固定单层单方向（不做 PCA、不做多层拼接）；目标层默认 `layers.{14,17}.mlp.down_proj`（可配置，层数默认 2）。

### 5.2 编辑公式

对每个目标层，`W' = W − α·u(uᵀW)`，其中：

- `u` 为 **单位向量**（§5.1），维度 = `d_model` = down_proj 输出维，与 `uᵀW ∈ R^{d_ff}` 相容；
- **整矩阵编辑**（不做列子集选择——v0 收缩项），即投影扣除作用于该层全部 FFN 列；
- `α = 1` 恰为精确投影移除；`α ∈ {0.5, 1, 2, 4, 8}` 网格，`α > 1` 语义为过度投影（外插消融），其有效性不预设、由数据判定；
- **符号自检测试（强制）**：合成张量上，编辑后 `W'u` 的范数应随 α 单调下降且 `±α` 不对称——该测试进 unit suite，防止 §5.1 方向约定被实现反转。

### 5.3 数据三段协议（EvalSpec 必填）

| 段 | 用途 | 约束 |
|---|---|---|
| **方向提取集** | 生成 u（配对样本） | 不参与任何评估 |
| **选择集** | 选层对与 α、决定 pass/fail 升级 | 参与调参；结果只报 `selection` 性质 |
| **冻结测试集** | 配置冻结后**只评估一次** | 最终 CI 只来自此段；样本不足时回执标 `exploratory`，不判 pass |

### 5.4 筛查协议（ScreenProtocol）

按顺序执行，全部指标带数值+单位+阈值+样本量写入回执：

1. **困惑度门**：held-out 文本 PPL 相对基线的**相对漂移** `Δppl = ppl'/ppl − 1`，阈值 `≤ +0.05`（相对值，无量纲）；
2. **匹配对照**：n=24 个与 u **同范数（=1）的随机单位方向**，同层、同 α、同评估流程；候选编辑在选择集上的效应 Δgoal 须 **超过全部对照的最大值**（秩检验名义水平 1/(n+1) ≈ 4%；实测校准见 stat 套件——曾用 n=8，实测零假设假阳性率 ≈ 9.5% ≈ 1/9，据此提标），且对配对差做**符号置换检验 p < 0.05**；（更多对照类型——配对打乱、空目标、参数预算匹配扰动——backlog，按成本启用）
3. **回归门**：MMLU-Pro 固定子集，绝对掉点 `≤ 1.0 个百分点`（绝对 pp，非相对）；
4. **冻结终评**：通过 1–3 后在冻结测试集上单次评估，配对差 **bootstrap 95% CI**（按题目重采样，B=2000）不含 0 且方向正确 → 判 `passed`。

**搜索预算（硬上限）**：候选 = 层对方案 × α 网格 ≤ 12 个；每候选只在选择集上评估；对照 8×仅对进入终评的 1 个候选执行。成本模型（前向次数、显存、暂存空间）在运行前预算并写入回执 `budget` 字段。

**升级阶梯**：α 扩展 → 增层对 → 诚实报告 `failed`（含失败阶段与已完成检查）。

## 6. Receipt 规格（v0.2）

```json
{
  "schema_version": "0.2",
  "receipt_id": "rcp_01H…",
  "run_id": "run_01H…",
  "parent_receipt_id": null,
  "status": "applied",
  "ts_started": "…", "ts_finished": "…",
  "env": {"code_version": "…", "command": "…", "config_digest": "…",
          "torch": "…", "gpu": "…", "seeds": {"data": 1234, "bootstrap": 42}},
  "model_in":  {"id": "Qwen/Qwen3.5-0.8B", "revision": "…", "files_sha256": "…"},
  "model_out": {"snapshot": "snap_out_01H…", "files_sha256": "…"},
  "rollback_to": "snap_src_01H…",
  "operator": {"kind": "edit", "spec": {"recipe": "rank1",
               "layers": ["…#14.mlp.down_proj", "…#17.mlp.down_proj"], "alpha": 4.0,
               "u_norm": 1.0, "params_touched_pct": 0.016}},
  "goal": {"kind": "preset:math-loops", "pairs_ref": "sha256:…", "grader": {"type": "exact@math500-subset", "version": "…"}},
  "data_protocol": {"direction_set": {"n": 32}, "selection_set": {"n": 120},
                    "frozen_test": {"n": 200, "evaluated_once": true}},
  "screen": {
    "ppl_gate":   {"value": 0.018, "unit": "relative_drift", "threshold": 0.05, "n_tokens": 50000, "verdict": "pass"},
    "controls":   {"type": "norm_matched_random_unit", "n": 8, "max_control_effect": 0.011,
                   "candidate_effect": 0.402, "permutation_p": 0.002, "verdict": "pass"},
    "regression": {"metric": "mmlu_pro_sub", "value": -0.4, "unit": "percentage_points", "threshold": -1.0, "n": 200, "verdict": "pass"},
    "frozen_eval":{"value": 40.2, "unit": "paired_delta_pp", "ci95": [33.8, 46.5],
                   "bootstrap": {"unit": "item", "B": 2000}, "verdict": "pass"}
  },
  "budget": {"forward_passes": 48632, "gpu_mem_peak_gb": 9.1, "staging_gb": 1.7},
  "verdict": "passed",
  "warnings": [], "failure": null, "human_overrides": []
}
```

规则：状态机 `started/running/passed/failed/applied/discarded/interrupted/rolled_back`；`--no-screen` 产物判 `screen_bypassed`（永不 pass）；数值一律合法 JSON（无 `+` 前缀、无百分号字符串）；schema 带 `schema_version` 与迁移说明；失败回执必填 `failure.stage/error/passed_checks`。回执分私有（含 pairs 引用原文路径）与公开（仅哈希+脱敏摘要）两种发布形态。

## 7. CLI 草案

```bash
modifyllm edit --model Qwen/Qwen3.5-0.8B --goal goals/math-loops.yaml \
               --data-split splits/math300.yaml --out runs/exp01
modifyllm receipts list / show <id> / compare <idA> <idB>
modifyllm rollback --receipt <id>
# v0.3+：finetune / distill / merge（同一 GoalSpec+EvalSpec，OperatorSpec 扩展）
```

## 8. 里程碑（工程与研究拆分）

| 版本 | 交付 | 验收（全部可有限次执行） |
|---|---|---|
| **v0.1 工程闭环**（2 周） | §5 全规格实现 + 事务 + 回执 + CLI | ① 数学确定性测试（投影/形状/**±α 符号**/列-整矩阵语义）全绿；② 合成植入信号可检出 + 零假设重复实验**报告**实测假阳性率（不承诺为零）；③ 通过/失败/绕过/崩溃注入/回滚五路径均产出合法回执，回滚后哈希 == 源快照；④ tiny 模型 CPU 端到端 ≤ 10 分钟 |
| **v0.2 研究验证**（+2 周） | 真实模型夜间实验管线 + 预设族 | 固定 revision/数据/配置，报告效应、CI、耗时、显存、存储；**允许结论为"无效"**；skill 打包 |
| **v0.3 算子扩展**（+4 周） | 三适配器 + 统一回执 | 同 GoalSpec 多算子可 `receipts compare` |
| **v1.0 对照报告** | **同目标算子对照报告**（原"等价报告"更名——只做经验对照，不主张等价） | 对照表含预算/数据/调参披露 |

## 9. 安全设计

1. 无去对齐预设（白名单制）；
2. 自定义配对开放，回执默认留痕 + 完整性可验证（哈希）；**不主张身份证明**——操作者归属由外部系统（如 CI/审批流）提供；
3. `--no-screen` → 判定 `screen_bypassed`，醒目写入 verdict；
4. 私有/公开回执分层 + 数据保留策略配置；
5. 代码 Apache-2.0，文档 CC-BY-4.0。

## 10. 风险

| 风险 | 应对 |
|---|---|
| 量化权重编辑（GGUF/AWQ）深坑 | 出 v0；v0.3 评估 dequant→edit→requant 路线 |
| 模型版本迭代 | 适配层隔离 + 兼容矩阵；实验固定 revision 并入回执 |
| 评估噪声/多重比较 | 三段协议 + 置换检验 + 单次冻结终评 + 预算硬上限 |
| 与 di37 skill 等先发同质 | 差异化 = 回执事务体系 + 统一算子接口 + 统计严谨性 |
| 真实模型复现失败 | v0.1/v0.2 拆分后，研究失败不阻塞工程交付 |

## 11. 与五图矩阵联动

`awesome-agentic-ml` 三算子章节 → 本工具（理论→实践闭环）；发布与矩阵首帖合并为一次推广战役；`awesome-ai-native` 的 skill CI/CD 议题引用回执作为"改造审计链"案例。

## 12. 命名

推荐 `modifyllm`（GitHub 全域零重名）；备选 `modiforge`。
