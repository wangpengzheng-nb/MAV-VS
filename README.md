# AutoVS-Agent

AutoVS-Agent 是面向非共价小分子结构虚拟筛选的无人值守系统。新任务唯一必填输入是自然语言任务；蛋白–配体 PDB 和严格 SMI 分子库均为建议输入。缺少分子库时锁定内置 PocketXMol 87K，缺少 PDB 时在策略进化后从调研验证的 RCSB 共晶候选中受控获取结构。

当前版本已真实跑通 CPU 基线：

```text
输入校验 → 口袋解析 → RDKit标准化/PAINS/3D构象
→ OpenBabel蛋白准备 → smina对接 → 姿态分数解析
→ 骨架多样性Top 20 → Markdown/HTML报告
```

PLIP、ADMET-AI、GNINA 和分级 GROMACS MD 通过能力目录明确报告可用性；未成功执行的证据不会用 mock 数据填充。

## 口袋科学预检

用户上传 PDB 后，该坐标文件会被锁定，调研只补充文献和靶点证据，不再下载替代结构。未上传 PDB 时，系统只允许从调研得到的同 UniProt、同物种、含配体实验结构中选择最多5个候选。口袋坐标始终由确定性工具计算，优先级为：

1. 用户坐标：必须通过蛋白空间范围与 box 相交校验。
2. 上传 PDB 中的合理非共价共晶配体：排除水、离子和常见结晶添加剂，并记录几何接触与 PLIP 证据。
3. 受控下载的 RCSB 共晶结构中的合理非共价配体。
4. 调研或用户给出的关键残基：必须在最终锁定 PDB 中实际匹配后才计算口袋。

无可信口袋时任务会在预检阶段失败，不会盲目全蛋白对接。策略智能体共享 `InputManifest v1` 约束，但无权替换 `screening_library`、用户 PDB 或口袋坐标。

## 分子库格式

新任务只接受 UTF-8、无表头的 `.smi`/`.smiles` 文件，每行恰好两列并使用 Tab 分隔：

```text
molecule_id<TAB>SMILES
aspirin<TAB>CC(=O)Oc1ccccc1C(=O)O
```

空格/逗号分隔、CSV、SDF、额外列、注释和空行会整库报错。无效 SMILES、重复 ID 和重复规范化结构会进入隔离表，其余分子继续执行。用户分子 ID 会作为 `source_id` 保留到最终报告。

## 安装

```bash
python -m pip install -e '.[test]'
autovs doctor
```

生产环境定义见 `environments/autovs-core.yml` 和 `environments/autovs-admet.yml`。工具路径、Conda 环境、Apptainer 镜像和 Slurm 资源统一位于 `config/tools.toml`，也可使用 `AUTOVS_*` 环境变量覆盖。

## CLI

```bash
# 环境与能力检查
autovs doctor

# 正式多智能体流水线
autovs run \
  --query '寻找靶向BCL2结合口袋的非共价小分子抑制剂'

# 可选：锁定用户分子库和预处理 PDB
autovs run \
  --query '寻找靶向BCL2结合口袋的非共价小分子抑制剂' \
  --protein preprocessed_receptor.pdb \
  --library compounds.smi \
  --center -15.36 2.24 -9.56

# 当前无GPU时的真实CPU基础链路诊断
autovs run \
  --query 'BCL2 CPU baseline validation' \
  --protein receptor.pdb \
  --library compounds.smi \
  --center -15.36 2.24 -9.56 \
  --cpu-only --baseline --wait

autovs status TASK_ID
autovs resume TASK_ID
autovs report TASK_ID
```

`--baseline` 只跳过 LLM 调研/投票，因而必须提供 `--protein`；计算步骤仍调用真实 RDKit、OpenBabel 和 smina，不会生成模拟评分。

## Web 与 MCP

```bash
# Web，默认仅本机访问
python -m web_app.server

# MCP Streamable HTTP: http://127.0.0.1:8765/mcp
autovs-tools-mcp
```

MCP 只暴露固定工具：能力发现、健康检查、工作流校验、步骤提交、作业查询、产物查询、日志和显式确认取消。它不接受任意 shell 命令。

Web 会先调用 `POST /api/targets/resolve`，将自然语言解析为结构化筛选要求并通过 UniProt 验证靶点身份。高置信度结果自动继续；低置信度结果要求用户从候选 UniProt 条目中确认。任务内的 `research.json` 使用 v2 schema，保存身份指纹、各 API 状态、结构准备度和证据缺口。没有实验共晶结构时调研仍成功，但策略会要求 `target_structure_prediction`；在 AlphaFold/Boltz 适配器接入前，该路线会返回明确的 capability gap。

真实 API 冒烟检查（不属于默认测试集）：

```bash
python scripts/smoke_target_research.py
python scripts/smoke_target_research.py --full
```

## WorkflowPlan v1

策略生成器和进化器只能输出已注册 action。执行前统一转换为严格的 `WorkflowPlan 1.0`：

```json
{
  "plan_version": "1.0",
  "strategy_id": "example",
  "steps": [{
    "step_id": "dock",
    "action_type": "molecular_docking",
    "requires": ["protein-preparation"],
    "inputs": [],
    "outputs": [],
    "parameters": {"exhaustiveness": 4, "num_modes": 3},
    "quality_gates": [],
    "resource_profile": {
      "executor": "slurm", "environment": "smina_stage2",
      "cpus": 10, "memory_gb": 4, "gpu_required": false,
      "timeout_seconds": 3600
    }
  }]
}
```

未知 action、向前依赖、额外字段、缺失口袋和越界文件路径会在计算前被拒绝。每个任务、作业和带 SHA256 的产物均写入 SQLite，Web/CLI 重启后仍可查询。

## 测试

```bash
pytest -q
python -m compileall -q autovs src web_app
```

GPU恢复后的科学验收顺序为：先用 BCL2/60OK 已知活性与 decoy 做富集验证，再用新靶点完成无人值守泛化验证。
