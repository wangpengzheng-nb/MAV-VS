# ToolUsePlannerAgent — 工具使用规划智能体

## 什么是 Planner？

在 MAV-VS 中：

- **Strategy**（策略）：说明科学上想做什么（如"用 smina 对接 10 万分子"）
- **Planner**（规划器）：决定用哪些工具、以什么顺序执行
- **DAG Executor**（执行器）：真正运行工具

Planner 是中间的"翻译官"——它把科学意图翻译成可执行的工具调用计划。

## 输入

Planner 接收：

| 输入 | 来源 | 说明 |
|------|------|------|
| Strategy | 策略生成+进化+投票 | 科学目标和推荐步骤 |
| InputManifest | 用户输入 | 分子库、靶结构、口袋参数 |
| Capabilities | 系统检测 | 哪些工具可用、哪些降级、哪些不可用 |
| Action Contracts | `autovs/planning/contracts.py` | 每个工具的输入/输出合约 |
| Artifact Registry | `autovs/planning/contracts.py` | 每种数据格式的定义 |
| Constraints | 用户或系统配置 | CPU only、GPU限制、时间预算等 |

## 输出

```
PlannerResult
├── plan: WorkflowPlan        ← 交给 DAG Executor 执行
├── decisions: [...]          ← 为什么选择/跳过每个步骤
├── warnings: [...]           ← 风险和警告
├── capability_gaps: [...]    ← 不可用的能力
└── alternatives_considered   ← 考虑过但未选的方案
```

## Artifact 如何连接步骤？

每个工具步骤就像一个"工厂"：
- **输入（required_inputs）**：它需要什么原材料
- **输出（outputs）**：它产出什么产品

例如：
```
INPUT_VALIDATION: screening_library → normalized_library
PROTEIN_PREPARATION: target_structure → receptor_pdb + receptor_pdbqt
MOLECULAR_DOCKING: receptor_pdbqt + prepared_library + pocket_center → docked_poses + scores_csv
```

Planner 自动根据这些合约：
1. 检查每个步骤的输入是否已被前面的步骤产出
2. 如果没有，自动插入能产出的步骤
3. 优先选择 source producer（从无到有创建数据的步骤）

## capability 的三种状态

| 状态 | 含义 | Planner 行为 |
|------|------|-------------|
| **available** | 工具正常可用 | 正常参与选择 |
| **degraded** | 工具可用但有风险 | 可以选择，但提高风险评分 |
| **unavailable** | 工具不可用 | 如果是 required → 任务失败；如果是 optional → 跳过 |

## 成本和风险如何影响选择？

Planner 使用简单的评分公式（所有权重定义在 `autovs/planning/scoring.py`）：

```
候选评分 = 科学重要性惩罚 + 降级惩罚 + 归一化成本 + 失败风险 + GPU约束惩罚
```

分数越低越好。评分最高的候选被选中。

## CPU-only 如何工作？

设置 `PlannerConstraints(cpu_only=True)` 时：
- 需要 GPU 的步骤（MD、结构预测等）会自动跳过
- 如果是 required 但没有 CPU 替代方案 → 产生 capability gap

## Planner 失败时怎么办？

失败时会在任务目录生成 `tool_planning_error.json`，包含：
- 错误类型
- 具体原因
- 缺失的能力或 artifact

系统会自动回退到 legacy `compile_strategy()`（原有的线性编译器）。

可以通过环境变量切换模式：
```bash
export AUTOVS_PLANNER_MODE=legacy    # 使用旧编译器
export AUTOVS_PLANNER_MODE=tool_use  # 使用新 Planner（默认）
```

## 标准虚拟筛选 DAG 示例

一个有上传 PDB 的标准对接任务会生成如下 DAG：

```
input-validation
├── target-structure-acquisition（仅当无上传PDB时）
├── molecule-standardization（配体准备分支）
│   └── (产生 prepared_library)
├── pocket-definition（口袋分支）
│   └── (产生 pocket_center, pocket_size)
└── protein-preparation（蛋白准备分支）
    └── (产生 receptor_pdb, receptor_pdbqt)

molecular-docking ← 汇聚三个分支
    ↓
pose-extraction
    ↓
interaction-analysis（可并行）
    ↓
final-ranking
    ↓
report-generation
```

注意：蛋白准备和配体准备是**并行**的（互不依赖），这与旧编译器的线性链不同。

## 如何运行测试

```bash
cd /users_home/wangpengzheng/药物筛选智能体

# 运行所有测试
python3 -m pytest tests/ -q

# 只运行 Planner 相关测试
python3 -m pytest tests/test_tool_use_planner.py tests/test_artifact_contracts.py -v
```

## legacy compile_strategy 回退

旧的 `compile_strategy()` 仍然保留在 `autovs/compiler.py` 中。

如果 ToolUsePlanner 失败（如 capability 不足、递归过深），系统会：
1. 记录错误到 `tool_planning_error.json`
2. 自动使用 `compile_strategy()` 生成线性计划
3. 在任务日志中记录回退原因
