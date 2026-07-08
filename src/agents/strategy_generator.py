"""
AutoVS-Agent v2.0: Strategy Generator (详细策略生成器)
========================================================
基于深度调研报告, 生成 5-10 个极其详细的虚拟筛选策略。
每个策略包含: 具体步骤、每步工具、评价指标、具体数值阈值。
"""

from __future__ import annotations

import json, os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from openai import OpenAI
from pydantic import BaseModel, Field


# =============================================================================
# 1. Pydantic Schema
# =============================================================================

class StrategyStep(BaseModel):
    step_number: int = Field(..., description="步骤编号 1-N")
    step_name: str = Field(..., description="步骤名称")
    tool: str = Field(..., description="使用的工具: GNINA/smina/RDKit/PLIP/ADMET-AI/GROMACS/DeepSeek")
    action: str = Field(..., description="具体操作描述 (100-200字)。包括运行参数、输入输出")
    metric: str = Field(..., description="评价指标: 对接分数/CNN_VS/PLIP_score/MW/LogP/PAINS等")
    threshold: str = Field(..., description="具体阈值或判断标准。如'CNN_VS > 5.0'或'对接分数前10%'")
    rationale: str = Field(..., description="为什么这样设计 (50-100字)")


class DetailedStrategy(BaseModel):
    strategy_name: str = Field(..., description="策略名称")
    strategy_tagline: str = Field(..., description="一句话总结")
    approach_type: str = Field(..., description="structure_based/ligand_based/hybrid/ml_driven/fragment_based")
    rationale: str = Field(..., description="为什么这个策略适合该靶点 (引用调研报告中的发现)")
    pipeline_steps: List[StrategyStep] = Field(..., min_length=3, max_length=10)
    survival_estimate: str = Field(..., description="每步后的估算存活率")
    contingency: str = Field(..., description="应急预案: 如果存活<10, 放宽哪些步骤?")
    strengths: List[str] = Field(...)
    weaknesses: List[str] = Field(...)
    estimated_runtime: str = Field(default="", description="估算计算时间")
    suitable_when: str = Field(default="", description="什么情况下这个策略最优")


class DetailedStrategyOutput(BaseModel):
    generation_rationale: str = Field(..., description="为什么选择这些策略方向")
    strategies: List[DetailedStrategy] = Field(..., min_length=5, max_length=10)


DetailedStrategyOutput.model_rebuild()
DetailedStrategy.model_rebuild()
StrategyStep.model_rebuild()


# =============================================================================
# 2. Prompt
# =============================================================================

STRATEGY_DETAILED_PROMPT = """\
# 人设: 虚拟筛选策略架构师

你基于深度调研报告, 设计 5-10 个极其详细的虚拟筛选策略。
每个策略必须具体到: 每一步用什么工具、跑什么参数、看什么指标、什么数值才算好。

# 策略设计要求
- 每个策略包含 3-10 个具体步骤
- 每步说明: 工具、操作、评价指标、具体阈值
- 阈值必须有科学依据 (引用调研报告中的配体活性值/口袋特征)
- **严禁**使用泛泛的"good score"/"reasonable value"
- 必须给出具体数值: "CNN_VS > 5.5", "MW 在 300-700", "PLIP score ≥ 12"
- 必须覆盖至少 5 个差异化方向:
  1) 严格SBDD 2) 宽松探索 3) 配体相似性 4) ML/AI驱动 5) 药效团优先 6) 片段筛选 7) 共价筛选 8) 多轮迭代
"""


# =============================================================================
# 3. Agent
# =============================================================================

class StrategyGeneratorAgent:
    def __init__(self, model="deepseek-chat", api_key=None, api_base=None, temperature=0.4, max_tokens=8192):
        self.model = model
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_base = api_base or os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=f"{self.api_base}/v1")
        return self._client

    def generate_strategies(self, research_report: dict, target_info: Optional[dict] = None) -> Dict[str, Any]:
        report_text = research_report.get("full_report_text", "")
        if not report_text:
            report_text = json.dumps(research_report, ensure_ascii=False, indent=2)

        prompt = f"""\
## 靶点深度调研报告

{report_text[:6000]}

### 任务
基于以上调研报告, 设计 5-10 个极其详细的虚拟筛选策略。
每个策略必须包含 pipeline_steps (每步的具体工具/参数/指标/阈值)。
**阈值必须引用调研报告中的具体数值!**
"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": STRATEGY_DETAILED_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "system", "content": f"JSON Schema:\n{json.dumps(DetailedStrategyOutput.model_json_schema(), indent=2, ensure_ascii=False)}"},
                ],
                response_format={"type": "json_object"},
            )
            validated = DetailedStrategyOutput.model_validate(json.loads(resp.choices[0].message.content.strip()))
            return {"strategies": [s.model_dump() for s in validated.strategies],
                    "generation_rationale": validated.generation_rationale}
        except Exception as e:
            return self._fallback(report_text, str(e))

    def _fallback(self, report: str, err: str) -> Dict[str, Any]:
        strategies = []
        for i, (name, tag, atype, steps) in enumerate([
            ("严格SBDD策略", "基于共晶结构精准对接+PLIP严格筛选",
             "structure_based",
             [{"step_number":1,"step_name":"SMILES→3D SDF","tool":"RDKit ETKDGv3","action":"生成3D构象, MMFF94优化","metric":"conformer_count","threshold":"有效构象>90%","rationale":"对接需3D结构"},
              {"step_number":2,"step_name":"GNINA精对接","tool":"GNINA refinement","action":"exhaustiveness=64, num_modes=9","metric":"CNN_VS","threshold":"CNN_VS > 5.0","rationale":"基于调研报告的共晶结构"},
              {"step_number":3,"step_name":"PLIP相互作用","tool":"PLIP","action":"分析氢键/疏水/盐桥","metric":"PLIP_score","threshold":">=15","rationale":"保留关键残基匹配的分子"},
              {"step_number":4,"step_name":"ADMET过滤","tool":"ADMET-AI","action":"预测40+属性","metric":"PAINS_alert","threshold":"PAINS=0 AND hERG=low","rationale":"排除假阳性"},
              {"step_number":5,"step_name":"理化性质过滤","tool":"RDKit","action":"MW/LogP/TPSA","metric":"MW","threshold":"150<MW<600","rationale":"类药性"}]),
            ("探索性宽松策略", "扩大小分子化学空间, 优先召回率",
             "hybrid",
             [{"step_number":1,"step_name":"GNINA粗对接","tool":"GNINA rough","action":"exhaustiveness=8","metric":"CNN_VS","threshold":"CNN_VS > 3.0","rationale":"降低门槛避免漏筛"},
              {"step_number":2,"step_name":"PAINS过滤","tool":"RDKit SMARTS","action":"排除已知PAINS子结构","metric":"PAINS_count","threshold":"PAINS=0","rationale":"唯一红线"},
              {"step_number":3,"step_name":"药效团筛选","tool":"RDKit","action":"保留含关键药效团的分子","metric":"pharmacophore_match","threshold":"至少1个","rationale":"基于SAR"}]),
            ("配体相似性策略", "以已知活性配体为模板进行相似性搜索",
             "ligand_based",
             [{"step_number":1,"step_name":"Tanimoto指纹筛选","tool":"RDKit Morgan2","action":"与已知配体计算Tanimoto","metric":"Tanimoto","threshold":">0.35","rationale":"化学空间聚焦"},
              {"step_number":2,"step_name":"对接","tool":"GNINA","action":"exhaustiveness=16","metric":"CNN_VS","threshold":">4.0","rationale":"相似分子优先对接"}]),
            ("ML驱动策略", "用ADMET-AI+对接分数训练分类器排序",
             "ml_driven",
             [{"step_number":1,"step_name":"特征工程","tool":"RDKit","action":"ECFP4+理化性质+对接分数","metric":"feature_dim","threshold":"2048+","rationale":"ML输入"},
              {"step_number":2,"step_name":"模型预测","tool":"ADMET-AI","action":"预测ADMET+活性概率","metric":"probability","threshold":"active_prob>0.7","rationale":"综合排序"}]),
            ("多轮迭代策略", "第一轮宽筛→分析→第二轮精筛",
             "hybrid",
             [{"step_number":1,"step_name":"第一轮粗筛","tool":"GNINA","action":"exhaustiveness=8, Top20%","metric":"CNN_VS","threshold":"保留前20%","rationale":"宽筛"},
              {"step_number":2,"step_name":"分析第一轮","tool":"PLIP+RDKit","action":"富集模式分析","metric":"enrichment","threshold":"富集率>2倍","rationale":"指导第二轮"},
              {"step_number":3,"step_name":"第二轮精筛","tool":"GNINA refinement","action":"exhaustiveness=64","metric":"CNN_VS","threshold":">5.5","rationale":"精确筛选"}]),
        ]):
            strategies.append({
                "strategy_name": name, "strategy_tagline": tag, "approach_type": atype,
                "rationale": f"基于调研报告的自动生成策略。{err[:100]}",
                "pipeline_steps": steps, "survival_estimate": f"~{100-(i+1)*15}%",
                "contingency": "放宽阈值20%", "strengths": ["自动化"], "weaknesses": ["需人工审查"],
                "estimated_runtime": "~2-5天", "suitable_when": "通用",
            })
        return {"strategies": strategies, "generation_rationale": f"[FALLBACK] {err[:200]}"}

    def generate_strategies_from_text(self, query: str, research_report: dict) -> Dict[str, Any]:
        return self.generate_strategies(research_report)


def strategy_generation_node(state: dict) -> dict:
    agent = StrategyGeneratorAgent()
    profile = state.get("target_profile", {})
    result = agent.generate_strategies(profile, state.get("target_info"))
    strategies = result["strategies"]
    elo = {s["strategy_name"]: state["tournament_state"]["elo_initial_rating"] for s in strategies}
    pairings = [[strategies[i]["strategy_name"], strategies[j]["strategy_name"]]
                for i in range(len(strategies)) for j in range(i+1, len(strategies))]
    now = datetime.now(timezone.utc).isoformat()
    return {
        "pipeline_stage": "strategy_generation", "candidate_strategies": strategies,
        "tournament_state": {**state["tournament_state"], "elo_ratings": elo, "pairings_queue": pairings,
                             "completed_debates": 0, "current_leader": strategies[0]["strategy_name"] if strategies else ""},
        "updated_at": now,
        "event_log": [f"[{now}] [StrategyGen] {len(strategies)} strategies, {len(pairings)} debates."],
    }
