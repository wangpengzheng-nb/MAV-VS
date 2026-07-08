"""
AutoVS-Agent v2.0: Target Scout Agent (靶点侦察兵)
====================================================
职责: 深度调研靶点蛋白，生成标准化的靶点画像 (TargetProfile)。
     不直接生成筛选方案！只为下游 StrategyGeneration 提供情报基础。

输入: TargetInfo
输出: TargetProfile (Pydantic 结构化画像)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field


# =============================================================================
# 1. 结构化输出: TargetProfileSchema
# =============================================================================

class StructuralAssessmentSchema(BaseModel):
    has_experimental_structure: bool = Field(
        ..., description="该靶点是否有实验解析的三维结构 (X-ray / Cryo-EM / NMR)"
    )
    pdb_ids: List[str] = Field(
        default_factory=list, description="已知的 PDB 结构 ID 列表"
    )
    resolution_range: str = Field(
        default="unknown", description="结构分辨率范围"
    )
    has_cocrystal_with_ligand: bool = Field(
        ..., description="是否有与配体/抑制剂的共晶结构"
    )
    pocket_type: str = Field(
        ...,
        description=(
            "结合口袋类型: deep_cleft (深裂隙，如激酶ATP口袋) / "
            "shallow_groove (浅沟槽) / flat_ppi (平坦PPI界面) / "
            "allosteric (变构口袋) / cryptic (隐蔽口袋)"
        ),
    )
    pocket_volume_estimate: str = Field(
        ..., description="口袋体积估算: small (<300Å³) / medium (300-800) / large (>800)"
    )
    pocket_polarity: str = Field(
        ..., description="口袋极性: hydrophobic / mixed / polar"
    )
    flexibility_concern: str = Field(
        ..., description="蛋白柔性: rigid / moderate_flexibility / highly_flexible"
    )


class KnownLigandInfoSchema(BaseModel):
    has_known_active_ligands: bool = Field(
        ..., description="是否有已知的活性配体/抑制剂?"
    )
    representative_ligands: List[str] = Field(
        default_factory=list,
        description="代表性配体名称或 SMILES (最多 5 个)"
    )
    binding_affinity_range: str = Field(
        default="unknown", description="已知配体的亲和力范围 (nM / uM / mM)"
    )
    key_pharmacophore_features: List[str] = Field(
        default_factory=list,
        description="已知配体的关键药效团特征"
    )
    relevant_patents_or_papers: List[str] = Field(
        default_factory=list, description="相关文献 PMID/DOI"
    )


class PriorityMetricsSchema(BaseModel):
    primary_metrics: List[str] = Field(
        ...,
        min_length=1,
        description="最重要的评价指标。例如: ['氢键与ASP103/TRP144的匹配度', '疏水互补面积']"
    )
    secondary_metrics: List[str] = Field(
        default_factory=list, description="次要指标"
    )
    red_flags: List[str] = Field(
        default_factory=list,
        description="绝对红线。例如: ['PAINS子结构', 'hERG抑制风险']"
    )
    suggested_thresholds: Dict[str, str] = Field(
        default_factory=dict,
        description="建议的软阈值。例如: {'MW': '<600 (PPI可放宽至1000)', 'LogP': '1-5'}"
    )


class TargetProfileSchema(BaseModel):
    """Target Scout 产出的标准化靶点画像。"""

    target_name: str = Field(..., description="靶点名称")
    structural_assessment: StructuralAssessmentSchema = Field(
        ..., description="结构可用性评估"
    )
    known_ligand_info: KnownLigandInfoSchema = Field(
        ..., description="已知配体信息"
    )
    priority_metrics: PriorityMetricsSchema = Field(
        ..., description="靶点特定的优先级评价指标"
    )
    drug_design_challenges: List[str] = Field(
        default_factory=list,
        description="该靶点的药物设计主要挑战。例如: ['口袋极度平坦，缺乏氢键锚点', 'KRAS对GDP/GTP的极高亲和力']"
    )
    recommended_approaches: List[str] = Field(
        ...,
        min_length=1,
        description="推荐的技术路线: SBDD / LBDD / FBDD / covalent / PROTAC / molecular_glue / ml_screening"
    )
    key_references: List[str] = Field(
        default_factory=list, description="支持本画像的关键文献"
    )
    profile_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


TargetProfileSchema.model_rebuild()
StructuralAssessmentSchema.model_rebuild()
KnownLigandInfoSchema.model_rebuild()
PriorityMetricsSchema.model_rebuild()


# =============================================================================
# 2. 核心系统提示词
# =============================================================================

TARGET_SCOUT_SYSTEM_PROMPT = """\
# 人设: 资深靶点情报分析师 (Target Intelligence Analyst)

你是一位在药物发现领域有 20 年经验的**靶点情报分析师**。
你的唯一职责是对给定的药物靶点进行**深度侦察和画像**，
而不是直接给出筛选方案。

你的工作成果将交给下游的"策略生成器(Strategy Generator)"，
它将基于你的画像生成多套差异化的虚拟筛选策略。
因此，你的画像必须**客观、详尽、数据驱动**。

---

# 侦察清单 (必须逐项回答)

## 1. 结构可用性评估
- 是否有实验确定的三维结构 (X-ray, Cryo-EM, NMR)?
- 列出已知 PDB ID，标注分辨率。
- 是否有与配体/抑制剂的共晶结构? (这对 SBDD 至关重要)
- 结合口袋的几何特征:
  * 深裂隙 (如激酶 ATP 口袋) — 适合传统小分子
  * 浅沟槽 — 需要更大的分子量
  * 平坦 PPI 界面 — bRo5 空间，适合大环/肽类
  * 变构口袋 — 可能缺乏内源性配体竞争
  * 隐蔽口袋 (cryptic) — 需要 MD 揭示，对接可能不准确
- 口袋体积估算: 小 (<300 Å³) / 中 (300-800) / 大 (>800)
- 口袋极性: 疏水为主? 极性为主? 混合?
- 蛋白柔性: 刚性 (适合刚性对接) / 中度柔性 / 高度柔性 (需诱导契合)

## 2. 已知配体情报
- 是否有已知的活性配体或已上市药物?
- 代表化合物的名称/SMILES 及其亲和力
- 已知配体的关键药效团特征
- 相关的关键文献/专利

## 3. 优先级评价指标
根据靶点特征，什么指标应该是**最高优先级**?
- 例如: Bcl-2 PPI → 疏水互补面积 > 氢键网络
- 例如: 激酶 ATP 口袋 → 铰链区氢键匹配度 > 疏水接触
- 例如: KRAS G12D → 共价弹头反应性 + switch-II 口袋占据
- 必须列出明确的**红线 (Red Flags)**

## 4. 药物设计挑战
- 该靶点的主要难点是什么?
- 选择性问题? 口袋适应性? 缺乏化学起点?

## 5. 推荐技术路线
基于以上分析，推荐哪些方法学?
- SBDD (基于结构的药物设计): 有高质量共晶结构
- LBDD (基于配体的药物设计): 有已知活性配体
- FBDD (基于片段的药物设计): 口袋适合片段筛选
- Covalent (共价抑制剂): 存在可靶向的半胱氨酸/赖氨酸
- PROTAC / 分子胶: 传统抑制剂难以阻断的 PPI
- ML 虚拟筛选: 有足够的训练数据

---

# 输出约束
1. 必须输出完整的 TargetProfileSchema JSON。
2. 对于不确定的信息，请注明"unknown"或"inferred"，不得编造。
3. 对于关键判断 (如 pocket_type)，请提供依据。
"""


# =============================================================================
# 3. StrategyAgent 重写为 TargetScoutAgent
# =============================================================================

class TargetScoutAgent:
    """Target Scout Agent — 靶点深度侦察兵。"""

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ):
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

    def generate_profile(self, target_info: dict) -> Dict[str, Any]:
        """生成靶点深度画像。

        Args:
            target_info: TargetInfo TypedDict。

        Returns:
            TargetProfile dict (Pydantic-validated)。
        """
        user_prompt = f"""\
## 靶点侦察任务

### 靶点基本信息
- 名称: {target_info.get('target_name', 'Unknown')}
- UniProt ID: {target_info.get('uniprot_id', 'N/A')}
- PDB ID: {target_info.get('pdb_id', 'N/A')}
- 靶点类别: {target_info.get('target_class', 'Unknown')}
- 物种: {target_info.get('organism', 'Homo sapiens')}
- 已知关键残基: {', '.join(target_info.get('key_residues', [])) or '未提供'}
- 结合位点中心: {target_info.get('binding_site_center', '未提供')}
- 结合位点尺寸: {target_info.get('binding_site_size', '未提供')}

### 功能描述
{target_info.get('description', '无额外信息。')}

### 任务
请根据上述信息，对该靶点进行完整的深度侦察，
输出 TargetProfileSchema JSON。
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": TARGET_SCOUT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "system", "content": f"请严格输出 JSON:\n{json.dumps(TargetProfileSchema.model_json_schema(), indent=2, ensure_ascii=False)}"},
                ],
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            validated = TargetProfileSchema.model_validate(parsed)
            return validated.model_dump()
        except Exception as e:
            # 降级: 返回最小画像
            return {
                "target_name": target_info.get("target_name", "Unknown"),
                "structural_assessment": {
                    "has_experimental_structure": bool(target_info.get("pdb_id")),
                    "pdb_ids": [target_info.get("pdb_id")] if target_info.get("pdb_id") else [],
                    "resolution_range": "unknown",
                    "has_cocrystal_with_ligand": False,
                    "pocket_type": "unknown",
                    "pocket_volume_estimate": "unknown",
                    "pocket_polarity": "unknown",
                    "flexibility_concern": "unknown",
                },
                "known_ligand_info": {
                    "has_known_active_ligands": False,
                    "representative_ligands": [],
                    "binding_affinity_range": "unknown",
                    "key_pharmacophore_features": [],
                    "relevant_patents_or_papers": [],
                },
                "priority_metrics": {
                    "primary_metrics": [f"基于{target_info.get('target_class', 'Unknown')}类靶点的通用指标"],
                    "secondary_metrics": [],
                    "red_flags": ["PAINS", "BRENK alerts"],
                    "suggested_thresholds": {},
                },
                "drug_design_challenges": ["Insufficient data for detailed analysis"],
                "recommended_approaches": ["SBDD"],
                "key_references": [],
                "profile_timestamp": datetime.now(timezone.utc).isoformat(),
                "_fallback": True,
                "_error": str(e)[:200],
            }


# =============================================================================
# 4. LangGraph 节点
# =============================================================================

def target_scout_node(state: dict) -> dict:
    """Step 1: Target Scout — 靶点深度侦察。"""
    from datetime import datetime as dt

    agent = TargetScoutAgent()
    profile = agent.generate_profile(state.get("target_info", {}))

    now = dt.now(timezone.utc).isoformat()
    return {
        "pipeline_stage": "target_scout",
        "target_profile": profile,
        "updated_at": now,
        "event_log": [
            f"[{now}] [TargetScout] Profile generated. "
            f"Pocket: {profile.get('structural_assessment', {}).get('pocket_type', '?')}. "
            f"Approaches: {profile.get('recommended_approaches', [])}"
        ],
    }
