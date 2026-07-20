# AutoVS-Agent

AutoVS-Agent 是面向非共价小分子结构虚拟筛选的无人值守系统。用户提供自然语言任务、蛋白 PDB 和 SMI/CSV/SDF 分子库，系统持久化执行调研、策略生成、全排列投票、策略进化、可执行性校验、计算工具 DAG、候选排序和可复现报告。

当前版本已真实跑通 CPU 基线：

```text
输入校验 → 口袋解析 → RDKit标准化/PAINS/3D构象
→ OpenBabel蛋白准备 → smina对接 → 姿态分数解析
→ 骨架多样性Top 20 → Markdown/HTML报告
```

PLIP、ADMET-AI、GNINA 和分级 GROMACS MD 通过能力目录明确报告可用性；未成功执行的证据不会用 mock 数据填充。

## 口袋科学预检

用户必须上传预处理后的蛋白 PDB。口袋中心可选，但推荐在已知时显式填写。口袋在策略生成之前确定，优先级为：

1. 用户坐标：必须通过蛋白空间范围与 box 相交校验。
2. 上传 PDB 中的合理非共价共晶配体：排除水、离子和常见结晶添加剂，并记录几何接触与 PLIP 证据。
3. TargetScout 通过API验证的配体中心：只有上传 PDB HEADER 与调研 PDB ID 一致时才能使用。
4. 调研或用户给出的关键残基：必须在上传 PDB 中实际匹配后才计算口袋。

无可信口袋时任务会在预检阶段失败，不会盲目全蛋白对接。策略智能体可读取已验证的 `PocketResolution v1`，但无权替换坐标。

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
  --query '寻找靶向BCL2结合口袋的非共价小分子抑制剂' \
  --protein receptor.pdb \
  --library compounds.smi \
  --center -15.36 2.24 -9.56

# 口袋坐标留空：先调研，再从上传PDB中的配体或可映射关键残基确定
autovs run \
  --query '寻找靶向BCL2结合口袋的非共价小分子抑制剂' \
  --protein preprocessed_receptor.pdb \
  --library compounds.smi

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

`--baseline` 只跳过 LLM 调研/投票，计算步骤仍调用真实 RDKit、OpenBabel 和 smina；它不会生成模拟评分。

## Web 与 MCP

```bash
# Web，默认仅本机访问
python -m web_app.server

# MCP Streamable HTTP: http://127.0.0.1:8765/mcp
autovs-tools-mcp
```

MCP 只暴露固定工具：能力发现、健康检查、工作流校验、步骤提交、作业查询、产物查询、日志和显式确认取消。它不接受任意 shell 命令。

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
