# 🧬 AutoVS-Agent — AI 驱动的自动化虚拟筛选智能体

AutoVS-Agent 是一个基于大语言模型（LLM）的多智能体协作系统，能够自动化完成从靶点调研、策略生成、策略审评到策略进化的完整虚拟筛选管线。

## 🏗️ 架构总览

```
用户输入任务
     │
     ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Step 0       │    │ Step 1           │    │ Step 2           │
│ 靶点调研      │───▶│ 策略生成          │───▶│ 策略审评          │
│              │    │                  │    │                  │
│ UniProt      │    │ 基于调研报告      │    │ 三人设独立评审     │
│ PDB          │    │ 生成 8-10 个     │    │ (漏斗/需求/产出)   │
│ ChEMBL       │    │ 虚拟筛选策略      │    │ 瑞士制锦标赛      │
│ PubMed       │    │                  │    │ Elo 排名         │
│ ClinicalTrials│   │                  │    │                  │
└──────────────┘    └──────────────────┘    └────────┬─────────┘
                                                      │
                                                      ▼
┌──────────────┐    ┌──────────────────┐
│ 输出结果      │◀───│ Step 3           │
│              │    │ 策略进化          │
│ 调研报告      │    │                  │
│ 进化策略      │    │ 弱点靶向修复      │
│ 排名榜单      │    │ 迷你锦标赛验证    │
└──────────────┘    └──────────────────┘
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- DeepSeek API Key（或其他兼容 OpenAI 接口的 LLM）

### 安装

```bash
git clone <repo-url>
cd 药物筛选智能体
pip install openai pydantic langgraph python-dotenv
pip install fastapi uvicorn  # Web 界面需要
```

### 配置

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY="sk-your-api-key"
DEEPSEEK_API_BASE="https://api.deepseek.com"
```

### 命令行运行

```bash
# 完整管线（调研 → 生成 → 审评 → 进化）
python test_tournament.py
```

### Web 界面运行

```bash
cd web_app && python server.py
# 访问 http://localhost:8080
```

## 📋 四步管线详解

### Step 0: 靶点调研（TargetScoutAgent）

从真实 API 获取数据，防止 LLM 幻觉：

| 数据源 | 用途 |
|---|---|
| **UniProt API** | 蛋白功能、基因、物种验证 |
| **RCSB PDB API** | 结构搜索、分辨率验证、物种校验 |
| **ChEMBL API** | 真实 IC50/Ki/Kd 活性数据 |
| **PubMed API** | 文献检索 |
| **ClinicalTrials.gov API** | 临床试验检索 |

产出：`research_report.md` — 包含靶点生物学、结构分析、已知配体、可药性评估 4 个章节。

### Step 1: 策略生成（StrategyGeneratorAgent）

基于调研报告，LLM 一次性生成 8-10 个差异化虚拟筛选策略，每个策略包含：
- 完整步骤管线（工具名称、参数、阈值）
- 存活率估算
- 应急预案
- 优劣势分析

### Step 2: 策略审评（TournamentReviewer + StrategyJudge）

三位评审官独立评分 + 瑞士制锦标赛:

| 评审官 | 关注维度 | 权重 |
|---|---|---|
| **漏斗工程评审官** | 存活链完整性、前后端效率、工具资源 | 30% |
| **需求匹配评审官** | 靶点适配性、用户约束、数据利用 | 35% |
| **产出质量评审官** | 化学空间覆盖、骨架多样性、命中率 | 35% |

评审后采用**瑞士制锦标赛**配对辩论，通过裁判裁决和 Elo 积分系统对策略排名。

### Step 3: 策略进化（StrategyEvolver）

对 Top 3 策略做弱点靶向修复：
- 提取三官评审 + 裁判判例中的所有弱点
- LLM 定向修复每个弱点
- 迷你锦标赛验证进化效果

产出：`evolved_strategies/` — 进化版策略，包含 `evolution_changelog`。

## 📁 项目结构

```
药物筛选智能体/
├── src/
│   ├── agents/
│   │   ├── target_scout.py        # Step 0: 靶点调研
│   │   ├── strategy_generator.py  # Step 1: 策略生成
│   │   ├── expert_committee.py    # Step 2: 三人设评审
│   │   ├── judge_agent.py         # Step 2: 裁判 + Elo
│   │   └── strategy_evolver.py    # Step 3: 策略进化
│   ├── graph/                     # LangGraph 工作流（v1 兼容）
│   └── tools/                     # 分子工具
├── web_app/
│   ├── server.py                  # FastAPI 服务 + SSE
│   ├── pipeline_runner.py         # 管线包装 + 进度回调
│   └── static/                    # 前端页面
├── test_tournament.py             # CLI 入口
├── main.py                        # v1 CLI 入口（旧版）
├── config/                        # 配置文件
├── .env                           # API 密钥（不入库）
└── README.md
```

## ⚙️ test_tournament.py 配置项

```python
YOUR_QUERY = "你的虚拟筛选任务描述"
SKIP_RESEARCH = True      # 跳过调研，从已有目录加载
SKIP_STRATEGY = True      # 跳过策略生成
SKIP_EVALUATION = True    # 跳过审评（从已有文件加载评审和锦标赛结果）
LOAD_FROM_DIR = "分析文件/任务_xxx"  # 已有任务目录路径
EVOLVE_TOP_N = 3          # 进化 Top N 策略；设为 0 跳过进化
SWISS_ROUNDS = 4          # 瑞士制轮数
```

## 🔑 技术特性

- **防幻觉机制**: PDB ID/IC50 坐标由 API 数据覆盖 LLM 输出
- **多段并行生成**: 调研报告 4 章节各一次独立 LLM 调用，防止 token 预算耗尽
- **瑞士制配对**: 10 策略 × 4 轮 = 20 场，相比全量配对 45 场节省 56%
- **动态 Elo**: K 因子 24/32/48 + 冷门惩罚 1.5×
- **实时进度推送**: Web 界面通过 SSE 推送 5 节点进度

## 📄 License

MIT
