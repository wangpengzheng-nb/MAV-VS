"""
AutoVS-Agent v2.0: Strategy Generator
======================================
基于调研报告一次性生成5-10个策略。一次LLM调用+高质量fallback, 无多步脆弱链。
"""

from __future__ import annotations

import json, os, re
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from openai import OpenAI
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════
# v4: 结构化Action输出 — 策略只定义"做什么", 工具选择留给下游
# ═══════════════════════════════════════════

class ActionInput(BaseModel):
    type: str = Field(default="compound_library")
    size: Any = Field(default="")  # str or int
    format: str = Field(default="SMILES")

class ActionOutput(BaseModel):
    type: str = Field(default="filtered_compounds")
    size: Any = Field(default="")  # str or int
    format: str = Field(default="SDF")

class PipelineAction(BaseModel):
    step_id: str = Field(default="")  # UUID, Python端自动生成
    step_number: int = Field(default=1)
    action_type: str = Field(default="molecular_docking")
    action_name: str = Field(default="")
    description: str = Field(default="")
    input: ActionInput = Field(default_factory=ActionInput)
    output: ActionOutput = Field(default_factory=ActionOutput)
    parameters: dict = Field(default_factory=dict)
    quality_criteria: str = Field(default="")
    cardinality_estimate: str = Field(default="")
    computational_cost: str = Field(default="medium")
    requires: List[str] = Field(default_factory=list)

class TargetProfile(BaseModel):
    target_class: str = Field(default="")
    pocket_type: str = Field(default="")
    pocket_volume_approx: str = Field(default="")
    pocket_polarity: str = Field(default="")
    recommended_mw_range: List[float] = Field(default_factory=list)
    recommended_logp_range: List[float] = Field(default_factory=list)
    has_experimental_structure: bool = Field(default=True)
    has_known_active_ligands: bool = Field(default=True)
    rule_category: str = Field(default="Ro5")

class ContingencyPlan(BaseModel):
    trigger: str = Field(default="survivors < 10")
    actions: List[str] = Field(default_factory=list)

class ApplicabilityConditions(BaseModel):
    requires_structure: bool = Field(default=True)
    requires_ligands: bool = Field(default=False)
    min_library_size: str = Field(default="100K")
    suitable_target_types: List[str] = Field(default_factory=list)

class DetailedStrategy(BaseModel):
    strategy_id: str = Field(default="")
    strategy_name: str = Field(default="")
    strategy_tagline: str = Field(default="")
    approach_category: str = Field(default="")
    rationale: str = Field(default="")
    target_profile: TargetProfile = Field(default_factory=TargetProfile)
    pipeline: List[PipelineAction] = Field(default_factory=list)
    survival_estimate: str = Field(default="")
    contingency_plan: ContingencyPlan = Field(default_factory=ContingencyPlan)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    estimated_runtime_category: str = Field(default="days")
    knowledge_dependencies: List[str] = Field(default_factory=list)
    applicability_conditions: ApplicabilityConditions = Field(default_factory=ApplicabilityConditions)
    problem_focus: str = Field(default="")
    target_evidence_refs: List[str] = Field(default_factory=list)
    user_requirement_coverage: List[str] = Field(default_factory=list)
    diversity_axis: str = Field(default="")
    risk_level: Literal["low", "medium", "high"] = "medium"
    why_this_strategy_fits_target: str = Field(default="")
    execution_status: Literal[
        "currently_executable", "partially_executable", "future_capability_required",
    ] = "currently_executable"
    required_capabilities: List[str] = Field(default_factory=list)
    missing_capabilities: List[str] = Field(default_factory=list)

    # 向后兼容别名
    @property
    def pipeline_steps(self): return self.pipeline
    @property
    def approach_type(self): return self.approach_category
    @property
    def contingency(self): return self.contingency_plan.trigger
    @property
    def estimated_runtime(self): return self.estimated_runtime_category
    @property
    def suitable_when(self):
        return f"requires_structure={self.applicability_conditions.requires_structure}, requires_ligands={self.applicability_conditions.requires_ligands}"


# 注册 model_rebuild
class StrategyContext(BaseModel):
    target_name: str = "Unknown target"
    gene_symbol: str = ""
    uniprot_id: str = ""
    user_query: str = ""
    target_class: str = "other"
    pocket_type: str = "unknown"
    pocket_polarity: str = "unknown"
    rule_category: str = "Ro5"
    mw_range: list[float] = Field(default_factory=lambda: [250, 600])
    logp_range: list[float] = Field(default_factory=lambda: [0.0, 5.0])
    has_experimental_structure: bool = False
    has_holo_structure: bool = False
    predicted_structure_required: bool = False
    has_known_active_ligands: bool = False
    best_pdb_id: str = ""
    best_pdb_resolution: float | None = None
    activity_range_nm: list[float] = Field(default_factory=list)
    selectivity_constraints: list[str] = Field(default_factory=list)
    excluded_targets: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    evidence_levels: dict[str, str] = Field(default_factory=dict)


class StrategyBlueprint(BaseModel):
    blueprint_id: str
    strategy_name: str
    approach_category: str
    problem_focus: str
    diversity_axis: str
    risk_level: Literal["low", "medium", "high"] = "medium"
    preferred_actions: list[str]
    future_actions: list[str] = Field(default_factory=list)
    requires_ligands: bool = False
    rationale_hint: str = ""


# 注册 model_rebuild
for _m in [ActionInput, ActionOutput, PipelineAction, TargetProfile, StrategyContext,
            StrategyBlueprint, ContingencyPlan, ApplicabilityConditions, DetailedStrategy]:
    _m.model_rebuild()


EXECUTABLE_STRATEGY_ACTIONS = {
    "library_preparation", "protein_preparation", "binding_site_detection",
    "physicochemical_filtering", "molecular_docking", "interaction_analysis",
    "final_ranking", "report_generation", "pose_validation", "admet_filtering",
    "pocket_prediction", "diffdock_docking", "geometric_pocket_detection",
}
FUTURE_ACTION_CAPABILITIES = {
    "molecular_dynamics": "GROMACS/MD production workflow",
    "short_md": "GROMACS short MD quality gate",
    "similarity_screening": "2D/3D ligand similarity screening engine",
    "pharmacophore_screening": "pharmacophore model generation and screening",
    "shape_matching": "3D shape/electrostatic similarity engine",
    "fragment_screening": "fragment docking/growing workflow",
    "consensus_scoring": "multi-engine consensus scoring workflow",
    "machine_learning_scoring": "ML rescoring model",
}


class StrategyGeneratorAgent:
    def __init__(self, model="deepseek-chat", api_key=None, api_base=None, temperature=0.5, max_tokens=8192):
        self.model = model
        self.api_key = os.getenv("DEEPSEEK_API_KEY") if api_key is None else api_key
        self.api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com") if api_base is None else api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=f"{self.api_base}/v1")
        return self._client

    @staticmethod
    def _report_text(research_report: dict) -> str:
        text = research_report.get("executive_summary") or research_report.get("full_report_text") or ""
        return text or json.dumps(research_report, ensure_ascii=False, indent=2)

    @classmethod
    def build_strategy_context(cls, research_report: dict) -> StrategyContext:
        text = cls._report_text(research_report)
        km = research_report.get("key_metrics") if isinstance(research_report.get("key_metrics"), dict) else {}
        readiness = research_report.get("structure_readiness") or {}
        structures = research_report.get("verified_pdb_structures") or []
        activities = research_report.get("chembl_activities") or []
        intent = research_report.get("intent") or {}
        execution_context = research_report.get("_execution_context") if isinstance(research_report.get("_execution_context"), dict) else {}
        target_binding = execution_context.get("target") if isinstance(execution_context.get("target"), dict) else {}
        uploaded_target_locked = bool(target_binding.get("locked") and target_binding.get("source") == "user")
        query = research_report.get("_user_query") or intent.get("raw_query", "")
        gene = research_report.get("gene_symbol") or research_report.get("target_gene") or ""
        target_name = research_report.get("target_name") or research_report.get("protein_name") or gene or "Unknown target"
        uniprot_id = research_report.get("uniprot_id") or research_report.get("target_uniprot_id") or ""

        target_class, class_level = cls._infer_target_class(gene, target_name, text)
        pocket_type, pocket_level = cls._infer_pocket_type(km, text, target_class)
        mw_range = cls._numeric_pair(km.get("known_ligand_mw_range")) or cls._mw_range_for(target_class, pocket_type)
        logp_range = cls._numeric_pair(km.get("known_ligand_logp_range")) or ([1.0, 6.5] if target_class == "PPI" else [0.0, 5.0])
        rule_category = str(km.get("recommended_rule_category") or ("bRo5" if target_class == "PPI" or "ppi" in pocket_type.lower() else "Ro5"))
        activity_range = cls._numeric_pair(km.get("known_ligand_ic50_range_nm")) or cls._activity_range_from_records(activities)

        holo = [item for item in structures if isinstance(item, dict) and item.get("has_ligand")]
        best = next(iter(holo or structures), {}) if structures else {}
        best_pdb = str(readiness.get("recommended_pdb_id") or research_report.get("recommended_pdb_for_docking") or best.get("pdb_id") or "")
        best_res = cls._as_float(km.get("best_pdb_resolution") or best.get("resolution"))
        has_structure = bool(uploaded_target_locked or structures or readiness.get("experimental_holo_available") or readiness.get("experimental_apo_available"))
        predicted_required = False if uploaded_target_locked else bool(readiness.get("predicted_structure_required"))
        selectivity, excluded = cls._extract_user_constraints(query, intent)
        selectivity.extend(str(item) for item in km.get("selectivity_residues", []) if item)
        refs = cls._evidence_refs(best_pdb, best_res, activity_range, structures, activities, text)
        if uploaded_target_locked:
            refs.insert(0, "uploaded_target_structure:locked")

        return StrategyContext(
            target_name=target_name, gene_symbol=gene, uniprot_id=uniprot_id, user_query=query,
            target_class=target_class, pocket_type=pocket_type,
            pocket_polarity=str(km.get("pocket_polarity") or "mixed" if pocket_type != "unknown" else "unknown"),
            rule_category=rule_category, mw_range=mw_range, logp_range=logp_range,
            has_experimental_structure=has_structure, has_holo_structure=bool(holo or readiness.get("experimental_holo_available")),
            predicted_structure_required=predicted_required, has_known_active_ligands=bool(activities or activity_range),
            best_pdb_id=best_pdb, best_pdb_resolution=best_res, activity_range_nm=activity_range,
            selectivity_constraints=list(dict.fromkeys(selectivity)),
            excluded_targets=list(dict.fromkeys(excluded)),
            evidence_refs=refs,
            evidence_levels={
                "target_class": class_level, "pocket_type": pocket_level,
                "structure": "direct_api" if has_structure else ("direct_api" if predicted_required else "unknown"),
                "ligands": "direct_api" if activities else ("direct_api" if activity_range else "unknown"),
                "user_constraints": "direct_api" if intent else ("text_inferred" if query else "unknown"),
            },
        )

    @staticmethod
    def _infer_target_class(gene: str, target_name: str, text: str) -> tuple[str, str]:
        hay = f"{gene} {target_name} {text[:5000]}".lower()
        if any(x in hay for x in ("bcl-2", "bcl2", "bcl-xl", "bclxl", "apoptosis regulator", "ppi", "protein-protein")):
            return "PPI", "text_inferred"
        if any(x in hay for x in ("kinase", "egfr", "abl1", "jak", "mapk", "atp-binding")):
            return "Kinase", "text_inferred"
        if any(x in hay for x in ("g protein-coupled", "gpcr", "receptor")):
            return "GPCR", "text_inferred"
        if any(x in hay for x in ("protease", "proteinase", "caspase")):
            return "Protease", "text_inferred"
        return "other", "unknown"

    @staticmethod
    def _infer_pocket_type(km: dict, text: str, target_class: str) -> tuple[str, str]:
        raw = str(km.get("pocket_type") or "").strip()
        if raw and raw.lower() != "unknown":
            return raw, "direct_api"
        hay = text[:6000].lower()
        if target_class == "PPI" or any(x in hay for x in ("bh3", "hydrophobic groove", "shallow groove")):
            return "shallow_groove", "text_inferred"
        if target_class == "Kinase" or "hinge" in hay or "atp" in hay:
            return "deep_cleft", "text_inferred"
        if "allosteric" in hay or "别构" in hay:
            return "allosteric", "text_inferred"
        return "unknown", "unknown"

    @staticmethod
    def _numeric_pair(value: Any) -> list[float]:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return [float(value[0]), float(value[1])]
            except (TypeError, ValueError):
                return []
        return []

    @staticmethod
    def _as_float(value: Any) -> float | None:
        try:
            return float(value) if value not in (None, "", "N/A") else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _mw_range_for(target_class: str, pocket_type: str) -> list[float]:
        if target_class == "PPI" or "ppi" in pocket_type.lower() or "groove" in pocket_type.lower():
            return [350, 900]
        if target_class == "Kinase":
            return [250, 550]
        return [250, 650]

    @staticmethod
    def _activity_range_from_records(records: list) -> list[float]:
        values = []
        for item in records:
            if isinstance(item, dict):
                try:
                    values.append(float(item.get("standard_value")))
                except (TypeError, ValueError):
                    pass
        return [min(values), max(values)] if values else []

    @staticmethod
    def _extract_user_constraints(query: str, intent: dict) -> tuple[list[str], list[str]]:
        selectivity, excluded = [], []
        for req in intent.get("requirements", []) if isinstance(intent, dict) else []:
            if isinstance(req, dict) and req.get("category") in {"selectivity", "off_target"}:
                selectivity.append(str(req.get("original_text") or req.get("normalized_key") or "selectivity requirement"))
        excluded.extend(str(item) for item in (intent.get("excluded_targets", []) if isinstance(intent, dict) else []) if item)
        if re.search(r"不要|禁止|排除|避免|avoid|exclude|selectiv|选择性|同源", query, re.I):
            selectivity.append(query[:240])
            excluded.extend(re.findall(r"\b[A-Z][A-Z0-9-]{1,15}\b", query))
        return selectivity, [item for item in excluded if item.upper() not in {"PDB", "SMI", "IC50", "ADMET"}]

    @staticmethod
    def _evidence_refs(best_pdb: str, best_res: float | None, activity_range: list[float],
                       structures: list, activities: list, text: str) -> list[str]:
        refs = []
        if best_pdb:
            refs.append(f"verified_structure:{best_pdb}" + (f":{best_res:g}A" if best_res else ""))
        if structures:
            refs.append(f"rcsb_structures:{len(structures)}")
        if activities:
            refs.append(f"chembl_activities:{len(activities)}")
        if activity_range:
            refs.append(f"activity_range_nm:{activity_range[0]:g}-{activity_range[1]:g}")
        if "BH3" in text or "bh3" in text:
            refs.append("text_evidence:BH3 binding groove")
        return refs or ["evidence:limited_target_research"]

    def plan_strategy_blueprints(self, context: StrategyContext, *, count: int = 8) -> list[StrategyBlueprint]:
        blueprints = [
            StrategyBlueprint(
                blueprint_id="structure_precision", strategy_name="结构精筛对接漏斗",
                approach_category="structure_precision_docking", problem_focus="利用可信口袋进行高精度非共价结构筛选",
                diversity_axis="structure_precision", risk_level="low",
                preferred_actions=["library_preparation", "protein_preparation", "binding_site_detection", "physicochemical_filtering", "molecular_docking", "interaction_analysis", "final_ranking", "report_generation"],
                rationale_hint="优先把结构证据转化为可执行对接和相互作用质量门禁。",
            ),
            StrategyBlueprint(
                blueprint_id="wide_exploration", strategy_name="宽松探索型筛选漏斗",
                approach_category="recall_first_exploration", problem_focus="在靶点信息不完备时提高召回率并保留新骨架",
                diversity_axis="recall_and_scaffold_coverage", risk_level="medium",
                preferred_actions=["library_preparation", "physicochemical_filtering", "molecular_docking", "diversity_selection", "final_ranking", "report_generation"],
                rationale_hint="用较宽阈值减少漏筛，再靠多样性和排序压缩候选。",
            ),
            StrategyBlueprint(
                blueprint_id="selectivity_guard", strategy_name="选择性避靶筛选策略",
                approach_category="selectivity_aware_screening", problem_focus="回应用户选择性和同源靶点排除需求",
                diversity_axis="selectivity_constraints", risk_level="medium",
                preferred_actions=["physicochemical_filtering", "molecular_docking", "interaction_analysis", "final_ranking", "report_generation"],
                future_actions=["counter_docking", "selectivity_panel_scoring"],
                rationale_hint="围绕差异残基、避靶和相互作用指纹筛掉同源靶点风险。",
            ),
            StrategyBlueprint(
                blueprint_id="ligand_sar", strategy_name="已知配体/SAR 聚焦策略",
                approach_category="ligand_sar_guided_screening", problem_focus="利用已知活性配体和SAR收缩化学空间",
                diversity_axis="known_ligand_sar", risk_level="medium", requires_ligands=True,
                preferred_actions=["library_preparation", "physicochemical_filtering", "molecular_docking", "final_ranking", "report_generation"],
                future_actions=["similarity_screening", "pharmacophore_screening"],
                rationale_hint="用已知活性和药效团约束提升命中率，同时标注配体依赖风险。",
            ),
            StrategyBlueprint(
                blueprint_id="pocket_family_specific", strategy_name="靶点口袋特异性策略",
                approach_category="target_class_specific_screening", problem_focus="针对靶点类别和口袋形态设置专属理化与相互作用规则",
                diversity_axis="target_class_rules", risk_level="medium",
                preferred_actions=["physicochemical_filtering", "molecular_docking", "interaction_analysis", "final_ranking", "report_generation"],
                rationale_hint="PPI走bRo5/疏水沟槽；激酶走hinge/ATP口袋；蛋白酶走催化口袋约束。",
            ),
            StrategyBlueprint(
                blueprint_id="admet_first", strategy_name="成药性与毒性风险优先策略",
                approach_category="developability_first_screening", problem_focus="降低湿实验前明显ADMET和反应性风险",
                diversity_axis="developability_risk", risk_level="low",
                preferred_actions=["library_preparation", "physicochemical_filtering", "molecular_docking", "final_ranking", "report_generation"],
                future_actions=["admet_filtering"],
                rationale_hint="把早期不适合推进的小分子风险前置处理。",
            ),
            StrategyBlueprint(
                blueprint_id="fragment_shape_future", strategy_name="片段/形状探索未来路线",
                approach_category="fragment_shape_exploration", problem_focus="探索非模板骨架和局部口袋占位方式",
                diversity_axis="novel_chemotypes", risk_level="high",
                preferred_actions=["binding_site_detection", "molecular_docking", "final_ranking", "report_generation"],
                future_actions=["fragment_screening", "shape_matching"],
                rationale_hint="为后续片段库、形状匹配或药效团工具接入预留科学路线。",
            ),
            StrategyBlueprint(
                blueprint_id="stability_validation", strategy_name="姿态稳定性验证增强策略",
                approach_category="pose_stability_validation", problem_focus="减少对接假阳性，优先输出姿态稳定候选",
                diversity_axis="dynamic_stability", risk_level="high",
                preferred_actions=["molecular_docking", "interaction_analysis", "final_ranking", "report_generation"],
                future_actions=["molecular_dynamics", "consensus_scoring"],
                rationale_hint="把少量Top候选送入动态稳定性和多评分验证。",
            ),
        ]
        if context.predicted_structure_required:
            for bp in blueprints:
                if "target_structure_prediction" not in bp.future_actions:
                    bp.future_actions.insert(0, "target_structure_prediction")
        if not context.has_known_active_ligands:
            for bp in blueprints:
                if bp.blueprint_id == "ligand_sar":
                    bp.problem_focus = "已知配体证据不足时，仅作为未来 ChEMBL/SAR 增强路线"
                    bp.risk_level = "high"
        if context.selectivity_constraints:
            blueprints.insert(1, blueprints.pop(next(i for i, bp in enumerate(blueprints) if bp.blueprint_id == "selectivity_guard")))
        return blueprints[:count]

    @staticmethod
    def _build_context_block(context: StrategyContext) -> str:
        return json.dumps({
            "target": context.model_dump(),
            "instructions": [
                "策略必须引用 target.evidence_refs 中的证据或说明 evidence:limited_target_research",
                "策略阈值必须从 mw_range/logp_range/rule_category 出发，不能套用无关默认值",
                "用户选择性/避靶需求必须进入 user_requirement_coverage",
                "暂未接入工具必须进入 missing_capabilities，不能伪装可执行",
            ],
        }, ensure_ascii=False, indent=2)

    def generate_strategies(self, research_report: dict, target_info=None, prior_knowledge: str = ""):
        context = self.build_strategy_context(research_report)
        report_text = self._report_text(research_report)
        target_name = context.target_name
        target_gene = context.gene_symbol
        target_uniprot = context.uniprot_id
        vector_db_path = research_report.get("_vector_db_path", "")
        prediction_required = context.predicted_structure_required
        blueprints = self.plan_strategy_blueprints(context, count=8)

        # 构建一次共享上下文（metrics + constraints），复用给每次单策略调用
        metrics_block = self._build_context_block(context)
        pdb_note = f"\n⚠️ 已验证结构: {context.best_pdb_id or '无明确推荐PDB'}。\n" if context.has_experimental_structure else ""
        if prediction_required:
            pdb_note = (
                "\n⚠️ 调研已确认没有可用实验共晶结构。每个策略必须以 "
                "target_structure_prediction 动作开始，由能力层在未来选择 AlphaFold/Boltz；"
                "不得虚构PDB或口袋坐标。\n"
            )

        constraint_block = ""
        if context.user_query:
            constraint_block = f"""## 🚨 用户约束
{context.user_query}
❌ 禁止忽略用户库/口袋/数值约束
---
"""

        prior_block = ""
        if prior_knowledge and prior_knowledge.strip():
            prior_block = f"""## 🧠 先验知识
{prior_knowledge.strip()}
---
"""

        if not self.api_key:
            fallback = self._fallback_from_context(context, blueprints)
            return {
                "strategies": fallback,
                "strategy_context": context.model_dump(),
                "generation_rationale": "离线规则化策略蓝图兜底：无 LLM API key",
            }

        from time import sleep as _sleep
        MIN_SUCCESS = 5
        strategies: list[dict] = []
        existing_summaries: list[str] = []

        for idx, blueprint in enumerate(blueprints):
            single_prompt = self._build_single_strategy_prompt(
                target_name, target_gene, target_uniprot,
                report_text, metrics_block, pdb_note,
                constraint_block, prior_block,
                existing_summaries=existing_summaries,
                strategy_index=idx + 1,
                blueprint=blueprint,
                context=context,
            )
            generated = False
            for attempt in range(2):
                if attempt > 0:
                    _sleep(2)
                result, raw = self._call_llm(single_prompt, target_name, target_gene, vector_db_path)
                if result and len(result) >= 1:
                    strategies.append(result[0])
                    existing_summaries.append(self._summarize_strategy(result[0]))
                    generated = True
                    print(f"  ✅ 策略{idx+1}: {result[0].get('strategy_name','?')} [{result[0].get('approach_category','?')}]", flush=True)
                    break
                if raw:
                    print(f"  ⚠️ 策略{idx+1}第{attempt+1}次失败 (raw={len(raw)}chars)", flush=True)
            if not generated:
                print(f"  ❌ 策略{idx+1} 2次都失败，使用规则化蓝图兜底", flush=True)
                fallback = self._strategy_from_blueprint(context, blueprint, idx + 1)
                strategies.append(fallback)
                existing_summaries.append(self._summarize_strategy(fallback))

        # 不足时回退批量模式兜底（prompt 要求 4-6 个）
        if len(strategies) < MIN_SUCCESS:
            print(f"\n⚠️ 单策略仅产出{len(strategies)}个, 回退批量模式...", flush=True)
            _sleep(2)
            batch_prompt = self._build_strategy_prompt(
                target_name, target_gene, target_uniprot,
                report_text, context.user_query, prior_knowledge,
            )
            for attempt in range(2):
                if attempt > 0:
                    _sleep(4)
                batch_strategies, _ = self._call_llm(batch_prompt, target_name, target_gene, vector_db_path)
                if batch_strategies and len(batch_strategies) >= MIN_SUCCESS:
                    existing_names = {s.get("strategy_name", "") for s in strategies}
                    for s in batch_strategies:
                        if s.get("strategy_name", "") not in existing_names:
                            strategies.append(s)
                    break

        if len(strategies) >= MIN_SUCCESS:
            strategies = self._quality_gate_strategies(
                self._ensure_structure_prediction(strategies, prediction_required),
                context, blueprints,
            )
            return {"strategies": strategies,
                    "strategy_context": context.model_dump(),
                    "generation_rationale": f"成功生成{len(strategies)}个策略 (逐策略独立生成, 各占独立token配额)"}

        # 简化 prompt 最后尝试
        _sleep(4)
        simple_prompt = f"""为{target_name}({target_gene})设计3个虚拟筛选策略。输出JSON。
不可变上下文: {prior_knowledge[:2000]}"""
        strategies, _ = self._call_llm(simple_prompt, target_name, target_gene, vector_db_path)
        if strategies and len(strategies) >= MIN_SUCCESS:
            strategies = self._quality_gate_strategies(
                self._ensure_structure_prediction(strategies, prediction_required),
                context, blueprints,
            )
            return {"strategies": strategies, "strategy_context": context.model_dump(),
                    "generation_rationale": "简化prompt重试成功"}

        # 兜底：确定性模板
        print(f"\n⚠️ 所有LLM调用失败, 使用确定性模板。", flush=True)
        fallback_strategies = self._fallback_from_context(context, blueprints)
        return {"strategies": fallback_strategies, "strategy_context": context.model_dump(),
                "generation_rationale": "规则化策略蓝图兜底"}

    @staticmethod
    def _ensure_structure_prediction(strategies: list[dict], required: bool) -> list[dict]:
        if not required:
            return strategies
        for strategy in strategies:
            pipeline = strategy.get("pipeline") or strategy.get("pipeline_steps") or []
            if not isinstance(pipeline, list):
                pipeline = []
            if not any(step.get("action_type") == "target_structure_prediction" for step in pipeline):
                pipeline.insert(0, {
                    "step_number": 1,
                    "action_type": "target_structure_prediction",
                    "action_name": "预测并验证靶点结构",
                    "description": "调用能力层提供的结构预测工具，并在进入口袋识别与对接前完成结构和置信度质量门禁。",
                    "input": {"type": "verified_target_identity", "format": "JSON"},
                    "output": {"type": "predicted_target_structure", "format": "PDB/CIF"},
                    "parameters": {}, "quality_criteria": "结构置信度与口袋质量门禁通过",
                    "cardinality_estimate": "1 target -> 1 validated structure",
                    "computational_cost": "high", "requires": ["verified_target_identity"],
                })
            for index, step in enumerate(pipeline, 1):
                step["step_number"] = index
            strategy["pipeline"] = pipeline
            strategy.setdefault("target_profile", {})["has_experimental_structure"] = False
        return strategies

    def _fallback_from_context(self, context: StrategyContext,
                               blueprints: list[StrategyBlueprint] | None = None) -> list[dict]:
        blueprints = blueprints or self.plan_strategy_blueprints(context, count=8)
        return self._quality_gate_strategies(
            [self._strategy_from_blueprint(context, blueprint, index + 1)
             for index, blueprint in enumerate(blueprints)],
            context, blueprints,
        )

    def _strategy_from_blueprint(self, context: StrategyContext,
                                 blueprint: StrategyBlueprint, index: int) -> dict:
        actions = list(dict.fromkeys([
            *("target_structure_prediction" for _ in [0] if context.predicted_structure_required),
            *blueprint.preferred_actions,
            *blueprint.future_actions,
        ]))
        pipeline = [
            self._action_from_blueprint(action, context, blueprint, step_number)
            for step_number, action in enumerate(actions, 1)
        ]
        target_label = f"{context.target_name}({context.gene_symbol or context.uniprot_id})"
        evidence = context.evidence_refs[:4]
        status, missing = self._execution_status_for(pipeline)
        coverage = context.selectivity_constraints[:2] or [context.user_query[:160] if context.user_query else "按靶点调研证据设计筛选约束"]
        return {
            "strategy_id": f"{(context.gene_symbol or 'TARGET').replace('-', '').upper()}_{blueprint.blueprint_id}_{index:02d}",
            "strategy_name": f"{target_label} {blueprint.strategy_name}",
            "strategy_tagline": blueprint.problem_focus,
            "approach_category": blueprint.approach_category,
            "rationale": (
                f"{blueprint.rationale_hint} 靶点上下文显示 target_class={context.target_class}, "
                f"pocket_type={context.pocket_type}, rule_category={context.rule_category}, "
                f"结构证据={context.best_pdb_id or '无明确共晶结构'}。"
            ),
            "target_profile": {
                "target_class": context.target_class,
                "pocket_type": context.pocket_type,
                "pocket_volume_approx": "large" if context.rule_category == "bRo5" else "medium",
                "pocket_polarity": context.pocket_polarity,
                "recommended_mw_range": context.mw_range,
                "recommended_logp_range": context.logp_range,
                "has_experimental_structure": context.has_experimental_structure,
                "has_known_active_ligands": context.has_known_active_ligands,
                "rule_category": context.rule_category,
            },
            "pipeline": pipeline,
            "survival_estimate": self._survival_for(blueprint, status),
            "contingency_plan": {
                "trigger": "survivors < 10 或关键证据缺失",
                "actions": ["回退到宽松阈值", "保留更多骨架进入人工复核", "等待缺失工具接入后重跑对应策略"],
            },
            "strengths": [blueprint.problem_focus, "策略证据和能力缺口可追溯"],
            "weaknesses": ["依赖当前调研证据质量"] + (["含暂未接入工具，不能直接执行完整路线"] if missing else []),
            "estimated_runtime_category": "days" if status == "currently_executable" else "future",
            "knowledge_dependencies": evidence,
            "applicability_conditions": {
                "requires_structure": "molecular_docking" in actions or "target_structure_prediction" in actions,
                "requires_ligands": blueprint.requires_ligands,
                "min_library_size": "87K",
                "suitable_target_types": [context.target_class, context.pocket_type],
            },
            "problem_focus": blueprint.problem_focus,
            "target_evidence_refs": evidence,
            "user_requirement_coverage": coverage,
            "diversity_axis": blueprint.diversity_axis,
            "risk_level": blueprint.risk_level,
            "why_this_strategy_fits_target": (
                f"该策略针对 {context.target_class}/{context.pocket_type} 场景设计，"
                f"采用 {context.rule_category} 与用户约束共同限定筛选路线。"
            ),
            "execution_status": status,
            "required_capabilities": actions,
            "missing_capabilities": missing,
        }

    @staticmethod
    def _action_from_blueprint(action: str, context: StrategyContext,
                               blueprint: StrategyBlueprint, step_number: int) -> dict:
        names = {
            "target_structure_prediction": "预测并验证靶点结构",
            "library_preparation": "分子库标准化与构象准备",
            "protein_preparation": "蛋白结构准备",
            "binding_site_detection": "证据驱动口袋定义",
            "pocket_prediction": "ML口袋预测（P2Rank apo结构）",
            "diffdock_docking": "扩散模型分子对接（DiffDock）",
            "geometric_pocket_detection": "几何口袋检测（fpocket）",
            "physicochemical_filtering": "靶点适配理化过滤",
            "molecular_docking": "口袋导向分子对接",
            "interaction_analysis": "关键相互作用指纹分析",
            "diversity_selection": "骨架多样性保护",
            "admet_filtering": "ADMET风险过滤",
            "molecular_dynamics": "候选姿态稳定性MD验证",
            "similarity_screening": "已知配体相似性筛选",
            "pharmacophore_screening": "药效团约束筛选",
            "shape_matching": "三维形状匹配",
            "fragment_screening": "片段探索筛选",
            "consensus_scoring": "多评分共识排序",
            "final_ranking": "综合证据排序",
            "report_generation": "可追溯报告生成",
        }
        params: dict[str, Any] = {}
        if action == "physicochemical_filtering":
            params = {"mw_range": context.mw_range, "logp_range": context.logp_range,
                      "rule_category": context.rule_category, "pains_filter": True}
        elif action == "molecular_docking":
            params = {"exhaustiveness": 12 if blueprint.risk_level == "low" else 6,
                      "num_modes": 5, "pocket_type": context.pocket_type}
        elif action == "interaction_analysis":
            params = {"prioritize_selectivity": bool(context.selectivity_constraints),
                      "key_constraints": context.selectivity_constraints[:5]}
        elif action in FUTURE_ACTION_CAPABILITIES:
            params = {"capability_gap": FUTURE_ACTION_CAPABILITIES[action]}
        return {
            "step_number": step_number,
            "action_type": action,
            "action_name": names.get(action, action),
            "description": (
                f"{names.get(action, action)}：服务于“{blueprint.problem_focus}”。"
                f"参数依据 {context.target_class}/{context.pocket_type} 和证据 {', '.join(context.evidence_refs[:2])}。"
            ),
            "input": {"type": "screening_context", "size": "task_bound", "format": "JSON/SMI/PDB"},
            "output": {"type": "screening_artifact", "size": "stage_dependent", "format": "CSV/SDF/JSON"},
            "parameters": params,
            "quality_criteria": "输出必须保留 source_id、证据来源和失败原因",
            "cardinality_estimate": "stage-dependent",
            "computational_cost": "high" if action in {"molecular_docking", "molecular_dynamics", "target_structure_prediction"} else "medium",
            "requires": ["screening_library"] if action in {"library_preparation", "physicochemical_filtering", "similarity_screening"} else ["target_structure", "screening_library"],
        }

    @staticmethod
    def _execution_status_for(pipeline: list[dict]) -> tuple[str, list[str]]:
        missing = []
        for step in pipeline:
            action = str(step.get("action_type", ""))
            if action in FUTURE_ACTION_CAPABILITIES:
                missing.append(f"{action}: {FUTURE_ACTION_CAPABILITIES[action]}")
            elif action not in EXECUTABLE_STRATEGY_ACTIONS:
                missing.append(f"{action}: no registered production adapter")
        missing = list(dict.fromkeys(missing))
        if not missing:
            return "currently_executable", []
        executable_count = sum(str(step.get("action_type", "")) in EXECUTABLE_STRATEGY_ACTIONS for step in pipeline)
        if executable_count >= 3:
            return "partially_executable", missing
        return "future_capability_required", missing

    @staticmethod
    def _survival_for(blueprint: StrategyBlueprint, status: str) -> str:
        if blueprint.risk_level == "low":
            return "87K→40K→5K→500→20"
        if status == "future_capability_required":
            return "待缺失能力接入后定义可执行压缩漏斗"
        return "87K→60K→10K→1K→20"

    def _quality_gate_strategies(self, strategies: list[dict], context: StrategyContext,
                                 blueprints: list[StrategyBlueprint]) -> list[dict]:
        cleaned: list[dict] = []
        seen_axes: set[str] = set()
        by_blueprint = {bp.diversity_axis: bp for bp in blueprints}
        for raw in strategies:
            item = self._normalize_strategy_metadata(raw, context)
            axis = item.get("diversity_axis") or item.get("approach_category") or item.get("strategy_name", "")
            if axis in seen_axes:
                continue
            seen_axes.add(axis)
            cleaned.append(item)
        for bp in blueprints:
            if len(cleaned) >= 8:
                break
            if bp.diversity_axis not in seen_axes:
                cleaned.append(self._strategy_from_blueprint(context, bp, len(cleaned) + 1))
                seen_axes.add(bp.diversity_axis)
        return cleaned[:8]

    def _normalize_strategy_metadata(self, strategy: dict, context: StrategyContext) -> dict:
        item = dict(strategy)
        pipeline = item.get("pipeline") if isinstance(item.get("pipeline"), list) else item.get("pipeline_steps", [])
        item["pipeline"] = pipeline if isinstance(pipeline, list) else []
        status, missing = self._execution_status_for(item["pipeline"])
        item.setdefault("problem_focus", item.get("strategy_tagline") or item.get("rationale", "")[:120] or "解决靶点筛选任务")
        item.setdefault("target_evidence_refs", context.evidence_refs[:4])
        item.setdefault("user_requirement_coverage", context.selectivity_constraints[:2] or [context.user_query[:160] if context.user_query else "无显式用户约束，按靶点证据设计"])
        item.setdefault("diversity_axis", item.get("approach_category") or "strategy_axis")
        item.setdefault("risk_level", "medium")
        item.setdefault("why_this_strategy_fits_target", f"匹配 {context.target_class}/{context.pocket_type} 与 {context.rule_category} 约束")
        item["execution_status"] = item.get("execution_status") if item.get("missing_capabilities") else status
        item["missing_capabilities"] = list(dict.fromkeys([*item.get("missing_capabilities", []), *missing]))
        if item["missing_capabilities"] and item["execution_status"] == "currently_executable":
            item["execution_status"] = status
        item["required_capabilities"] = list(dict.fromkeys(
            item.get("required_capabilities", []) or [str(step.get("action_type", "")) for step in item["pipeline"] if step.get("action_type")]
        ))
        item.setdefault("target_profile", {})
        item["target_profile"] = {
            **{
                "target_class": context.target_class, "pocket_type": context.pocket_type,
                "recommended_mw_range": context.mw_range, "recommended_logp_range": context.logp_range,
                "has_experimental_structure": context.has_experimental_structure,
                "has_known_active_ligands": context.has_known_active_ligands,
                "rule_category": context.rule_category,
            },
            **item.get("target_profile", {}),
        }
        return item

    def _build_strategy_prompt(self, target_name, target_gene, target_uniprot, report_text, user_query="", prior_knowledge=""):
        metrics_block = self._build_metrics_block(report_text)

        # 🆕 从报告提取已验证PDB列表(防止LLM说"无实验结构")
        pdbs_in_report = re.findall(r'\b[0-9][A-Z0-9]{3}\b', report_text[:2000])
        pdb_note = ""
        if pdbs_in_report:
            unique_pdbs = list(dict.fromkeys(pdbs_in_report))[:8]
            pdb_note = f"\n⚠️ 调研报告已验证PDB结构包括: {', '.join(unique_pdbs)}。如果此列表非空, 禁止在策略中说'无实验结构'或建议'同源建模'! 必须推荐使用已验证的PDB进行对接!\n"

        # 🆕 用户约束前置 (强化版)
        constraint_block = ""
        if user_query:
            constraint_block = f"""## 🚨 用户任务约束 — 必须逐条满足! (违反单项扣10分)

{user_query}

请从用户任务中提取所有操作细节, 逐条对照填入策略:
1. **库来源**: 用户指定了什么化合物库? 库大小? → 必须使用用户指定的库! 禁止改用ZINC/Enamine!
2. **口袋/靶点类型**: PPI? 激酶? 别构? → 必须设计匹配的科学步骤；具体工具只由下游能力目录绑定!
3. **排除条件**: 用户说了"不要X/禁止X/避开X"? → 策略中必须有对应的排除/过滤步骤!
4. **特殊要求**: 分子量范围? 选择性要求? ADMET? → 策略中必须明确体现!
5. **数值约束**: 用户给了具体数值(库大小/MW/IC50)? → 策略阈值必须基于此, 禁止用泛化默认值!

❌ 禁止: 忽略用户库来源改用ZINC/Enamine
❌ 禁止: 忽略用户口袋类型用通用方法
❌ 禁止: 用泛化默认值替代用户指定的数值

---

"""

        # 🆕 先验知识块
        prior_block = ""
        if prior_knowledge and prior_knowledge.strip():
            prior_block = f"""## 🧠 领域先验知识 (必须遵守的专家规则!)

{prior_knowledge.strip()}

以上是领域专家的先验知识。策略中的工具选择、方法设计必须遵守这些规则!
- 如果先验知识指定了某种场景下的工具 → 策略中该场景必须使用该工具
- 如果先验知识与用户约束冲突 → 优先遵守用户约束

---

"""

        return f"""为靶点 {target_name} ({target_gene}, UniProt:{target_uniprot}) 设计4-6个虚拟筛选策略。

{constraint_block}{prior_block}{metrics_block}{pdb_note}
## 调研报告全文
{report_text[:6000]}

## 🎨 策略设计要求（最重要！）

⚠️ 禁止生成"模板化"策略! 每个策略必须基于调研报告中的**靶点特异性信息**定制:
  - 基于口袋特征：口袋是deep cleft还是flat PPI? 体积多大? 极性如何?
  - 基于已知配体：IC50/MW/LogP范围? 有什么药效团特征?
  - 基于用户约束：选择性要求? 排除条件? 特殊需求?
  - 基于结构可用性：有共晶结构吗? 分辨率多少?

🚫 禁止套路:
  - 禁止给所有靶点都生成相同的"SBDD/LBDD/ML/片段/共价"五件套
  - 禁止在无Cys的靶点上强行设计共价策略
  - 禁止在有共晶结构时还建议同源建模
  - 禁止忽略调研报告中的真实IC50/MW/LogP数据而使用泛化默认值

✅ 鼓励创新:
  - 组合多种方法 (如"药效团预筛→对接精筛→MD验证")
  - 利用靶点特异性特征 (如别构口袋、蛋白-蛋白界面、选择性残基)
  - 考虑实际约束 (计算资源、时间、库大小)
  - 为同一靶点设计风险不同的策略 (激进vs保守, 探索vs聚焦)
  - 针对用户特定需求设计专属策略 (如抗衰老、PROTAC、双靶点等)

## JSON格式要求 (v4 Action-based — 不指定具体工具!)

⚠️ 关键变化: 策略只描述"做什么"(Action), 不绑定具体工具! 工具选择由下游智能体完成。

第一版系统只支持有明确结合口袋的非共价小分子SBDD。action_type **只能**从以下已注册能力中选择:
library_preparation, protein_preparation, binding_site_detection,
target_structure_prediction,
physicochemical_filtering, diversity_selection, molecular_docking,
interaction_analysis, admet_filtering, molecular_dynamics, final_ranking,
report_generation

严禁输出药效团、形状、共价、FEP、片段生长、生成式设计、人工目视检查等未注册步骤；
策略必须能在上述能力集合内完整执行。molecular_dynamics只能用于最终少量候选，不能用于大库粗筛。
🚨 禁止使用 input_validation 和 target_structure_acquisition！后者仅指RCSB实验结构下载；无共晶结构时允许 target_structure_prediction。

{{
  "strategies": [{{
    "strategy_id": "TARGET_METHOD_001",
    "strategy_name": "策略名称",
    "strategy_tagline": "一句话描述核心创新点",
    "approach_category": "方法本质(自由描述, 如 pharmacophore_guided_consensus_docking)",
    "rationale": "设计原理(200-400字, 引用调研报告中的IC50/PDB/口袋数据)",
    "target_profile": {{
      "target_class": "PPI / Kinase / GPCR / Protease / other",
      "pocket_type": "deep_cleft / shallow_groove / flat_ppi / allosteric",
      "pocket_volume_approx": "small / medium / large",
      "pocket_polarity": "hydrophobic / mixed / polar",
      "recommended_mw_range": [250, 600],
      "recommended_logp_range": [1.0, 5.0],
      "has_experimental_structure": true,
      "has_known_active_ligands": true,
      "rule_category": "Ro5 / bRo5 / custom"
    }},
    "pipeline": [{{
      "step_number": 1,
      "action_type": "physicochemical_filtering",
      "action_name": "基于理化性质的预过滤",
      "description": "详细描述(100-300字): 阐述本步骤要完成什么操作、为什么需要这一步、基本原理",
      "input": {{"type": "compound_library", "size": "10M", "format": "SMILES"}},
      "output": {{"type": "filtered_library", "size": "~5M", "format": "SDF"}},
      "parameters": {{"mw_range": [400, 800], "logp_range": [3, 8], "pains_filter": true}},
      "quality_criteria": ">95%分子通过过滤",
      "cardinality_estimate": "10M → 5M",
      "computational_cost": "low",
      "requires": ["compound_library_smiles"]
    }}],
    "survival_estimate": "10M→5M→100K→1K→100→20",
    "contingency_plan": {{"trigger": "survivors < 10", "actions": ["放宽阈值至...", "扩大库至..."]}},
    "strengths": ["优势1", "优势2"],
    "weaknesses": ["劣势1", "劣势2"],
    "estimated_runtime_category": "days",
    "knowledge_dependencies": ["known_ligand_SAR"],
    "applicability_conditions": {{
      "requires_structure": true,
      "requires_ligands": true,
      "min_library_size": "1M",
      "suitable_target_types": ["PPI"]
    }}
  }}]
}}

输出纯JSON, 所有字符串用双引号, 不要markdown代码块。必须输出4-6个差异化策略!

🚨 requires 字段是字符串数组，填前置资源描述如 "compound_library"、"protein_structure"，绝对禁止填步骤编号数字！"""

    def _build_single_strategy_prompt(
        self, target_name, target_gene, target_uniprot,
        report_text, metrics_block, pdb_note,
        constraint_block, prior_block,
        existing_summaries=None, strategy_index=1,
        blueprint: StrategyBlueprint | None = None,
        context: StrategyContext | None = None,
    ):
        """构建单策略 prompt — LLM 只需输出 1 个完整策略，独占 8K token 配额。"""
        existing_note = ""
        if existing_summaries:
            items = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(existing_summaries))
            existing_note = f"""## ⚠️ 已生成策略（必须差异化！）
{items}

你的新策略必须与以上 {len(existing_summaries)} 个已有策略**明显不同**：不同 approach_category、不同 pipeline 流程、不同风险偏好!
"""
        blueprint_note = ""
        if blueprint is not None:
            blueprint_note = f"""## 本策略蓝图（必须遵守）
{blueprint.model_dump_json(indent=2)}

你只能围绕这个蓝图生成一个策略。可以补充科学细节，但不能把它改成另一个策略族。
"""
        context_note = ""
        if context is not None:
            context_note = f"""## 结构化靶点上下文
{context.model_dump_json(indent=2)}

字段 evidence_levels 标记证据等级。若为 assumed/unknown，必须在 rationale 中说明不确定性。
"""
        return f"""为靶点 {target_name} ({target_gene}, UniProt:{target_uniprot}) 设计第{strategy_index}个虚拟筛选策略。只需输出 1 个完整策略对象，不要 strategies 数组。

{constraint_block}{prior_block}{existing_note}{blueprint_note}{context_note}{metrics_block}{pdb_note}
## 调研报告参考
{report_text[:4000]}

## 设计要求
- 基于靶点口袋和已知配体数据，description 简洁 (50-150字), rationale (100-200字)
- 与已有策略明显差异化：不同方法组合、不同过滤顺序、不同风险偏好
- 必须填写 problem_focus、target_evidence_refs、user_requirement_coverage、diversity_axis、risk_level、why_this_strategy_fits_target
- 暂未接入工具允许写入 pipeline，但 execution_status 不能是 currently_executable，missing_capabilities 必须写清楚

action_type 可从以下选择:
当前可执行/接近可执行: library_preparation, protein_preparation, binding_site_detection, physicochemical_filtering, molecular_docking, interaction_analysis, final_ranking, report_generation
未来能力路线: target_structure_prediction, admet_filtering, molecular_dynamics, similarity_screening, pharmacophore_screening, shape_matching, fragment_screening, consensus_scoring, machine_learning_scoring

🚨 禁止使用 input_validation 和 target_structure_acquisition！无共晶结构时允许 target_structure_prediction。

## 输出格式（单策略对象，不含 strategies 数组！）
{{
  "strategy_name": "名称",
  "strategy_tagline": "一句话创新点",
  "approach_category": "方法本质",
  "rationale": "设计原理(100-200字)",
  "target_profile": {{
    "target_class": "PPI / Kinase / GPCR / Protease / other",
    "pocket_type": "deep_cleft / shallow_groove / flat_ppi / allosteric",
    "pocket_volume_approx": "small / medium / large",
    "pocket_polarity": "hydrophobic / mixed / polar",
    "recommended_mw_range": [250, 600],
    "recommended_logp_range": [1.0, 5.0],
    "has_experimental_structure": true,
    "has_known_active_ligands": true,
    "rule_category": "Ro5 / bRo5 / custom"
  }},
  "pipeline": [{{
    "step_number": 1,
    "action_type": "physicochemical_filtering",
    "action_name": "步骤名",
    "description": "简洁描述(50-150字)",
    "input": {{"type": "compound_library", "size": "1M", "format": "SMILES"}},
    "output": {{"type": "filtered_library", "size": "~500K", "format": "SDF"}},
    "parameters": {{"mw_range": [250, 600]}},
    "quality_criteria": "",
    "cardinality_estimate": "",
    "computational_cost": "low",
    "requires": []
  }}],
  "survival_estimate": "",
  "contingency_plan": {{"trigger": "survivors < 10", "actions": []}},
  "strengths": [],
  "weaknesses": [],
  "estimated_runtime_category": "days",
  "knowledge_dependencies": [],
  "applicability_conditions": {{
    "requires_structure": true,
    "requires_ligands": false,
    "min_library_size": "100K",
    "suitable_target_types": []
  }},
  "problem_focus": "要解决的核心问题",
  "target_evidence_refs": ["verified_structure:xxxx", "chembl_activities:20"],
  "user_requirement_coverage": ["如何覆盖用户需求或为什么无显式需求"],
  "diversity_axis": "必须等于或贴近蓝图 diversity_axis",
  "risk_level": "low / medium / high",
  "why_this_strategy_fits_target": "解释它为什么适配该靶点和口袋特征",
  "execution_status": "currently_executable / partially_executable / future_capability_required",
  "required_capabilities": ["molecular_docking"],
  "missing_capabilities": []
}}

输出纯JSON，不要 markdown 代码块，不要外层 strategies 数组！只输出一个策略对象。

🚨 requires 字段必须填描述性字符串，绝对禁止填数字！
✅ "requires": ["compound_library_smiles"]
✅ "requires": ["protein_structure", "binding_site_definition"]
✅ "requires": []
❌ "requires": [1, 2, 3]
❌ "requires": [{{"step": 2}}]
requires 是前置资源描述，不是步骤编号！"""

    @staticmethod
    def _summarize_strategy(strategy: dict) -> str:
        """提取策略摘要，用于后续策略差异化提示。"""
        name = strategy.get("strategy_name", "?")
        tagline = strategy.get("strategy_tagline", "")
        category = strategy.get("approach_category", "")
        steps = [s.get("action_type", "?") for s in strategy.get("pipeline", [])]
        flow = " → ".join(steps[:6])
        strengths = ", ".join(strategy.get("strengths", [])[:2])
        return f"[{category}] {name}: {tagline} | {flow} | {strengths}"

    @staticmethod
    def _build_metrics_block(report_text: str) -> str:
        """从调研报告中提取key_metrics, 智能构建数据块。
        有数据时突出展示; 无数据时明确告知LLM使用通用准则。
        """
        km = {}
        km_match = re.search(r'"key_metrics"\s*:\s*\{[^}]+\}', report_text, re.DOTALL)
        if km_match:
            try: km = json.loads(km_match.group())
            except Exception: pass

        # 判断哪些数据可用
        has_ligands = bool(km.get("known_ligand_mw_range") and len(km.get("known_ligand_mw_range",[])) == 2 and km["known_ligand_mw_range"][1] > 0)
        has_structure = bool(km.get("pocket_type") and km["pocket_type"] != "unknown")
        has_selectivity = bool(km.get("selectivity_residues"))

        lines = ["## ⚠️ 关键约束数据 (策略阈值必须基于以下数据, 不可使用泛化默认值!)"]

        if has_ligands:
            mw = km["known_ligand_mw_range"]
            logp = km.get("known_ligand_logp_range", [])
            ic50 = km.get("known_ligand_ic50_range_nm", [])
            lines.append(f"\n### 已知配体数据 (直接引用!)")
            lines.append(f"- MW范围: [{mw[0]}, {mw[1]}] — 策略中的MW阈值必须覆盖此范围!")
            if logp and len(logp) == 2:
                lines.append(f"- LogP范围: [{logp[0]}, {logp[1]}] — LogP阈值不能比这更窄!")
            if ic50 and len(ic50) == 2:
                lines.append(f"- IC50范围: {ic50[0]:.2f}-{ic50[1]:.0f} nM")
            if km.get("representative_ligand_mw_max", 0) > 0:
                lines.append(f"- ⚠️ 最大已知配体MW={km['representative_ligand_mw_max']} — 如果这是PPI靶点, 必须用bRo5规则! MW上限不低于此值加50!")
            rule = km.get("recommended_rule_category", "")
            if rule and rule != "Ro5":
                lines.append(f"- 🚨 推荐规则: **{rule}** (不是Ro5! 不要默认套用MW<500!)")
        else:
            lines.append(f"\n### 已知配体: ⚠️ 无数据")
            lines.append(f"- 此靶点尚无已知配体或活性数据不足")
            lines.append(f"- 请基于口袋类型和结构特征推断合理的理化性质范围")
            lines.append(f"- 在rationale中明确标注'基于口袋特征推断, 待实验验证'")

        if has_structure:
            lines.append(f"\n### 结构特征")
            lines.append(f"- 口袋类型: {km.get('pocket_type','?')} | 体积: {km.get('binding_pocket_volume_ang3','?')}")
            if km.get("key_hbond_residues"):
                lines.append(f"- 关键氢键残基: {km['key_hbond_residues']}")
            if km.get("key_hydrophobic_residues"):
                lines.append(f"- 关键疏水残基: {km['key_hydrophobic_residues']}")
            if km.get("best_pdb_resolution", 99) < 99:
                lines.append(f"- 最佳PDB分辨率: {km['best_pdb_resolution']}Å")
            lines.append(f"- 共晶结构: {'有' if km.get('has_cocrystal') else '无'}")
            # 根据口袋类型给阈值建议
            pt = km.get("pocket_type", "")
            if "ppi" in pt.lower() or "flat" in pt.lower():
                lines.append(f"- 🚨 口袋类型={pt} → 传统Ro5不适用! 考虑bRo5 (MW<1000, LogP<8)")
            elif "deep" in pt.lower() or "cleft" in pt.lower():
                lines.append(f"- 口袋类型={pt} → 传统Ro5适用 (MW<500, LogP<5)")
        else:
            lines.append(f"\n### 结构特征: ⚠️ 无实验结构")
            lines.append(f"- 无可用PDB结构, 对接策略需注明基于同源建模或AlphaFold预测")
            lines.append(f"- 如使用同源建模, 必须在rationale中说明模板选择依据")

        if has_selectivity:
            lines.append(f"\n### 选择性约束")
            lines.append(f"- 差异残基: {km['selectivity_residues']}")
            lines.append(f"- 策略中必须包含基于这些残基的选择性过滤步骤!")

        return "\n".join(lines) + "\n"

    def _call_llm(self, prompt, target_name="", target_gene="", vector_db_path=""):
        raw = ""
        try:
            is_reasoner = "reasoner" in self.model.lower()

            messages = [
                {"role":"system","content":f"你是虚拟筛选策略专家。为{target_name}({target_gene})设计策略。核心规则: 1)用户query中'不要/禁止/避开'→策略有排除步骤 2)PDB结构→推荐使用 3)禁止固定模板 4)策略只描述Action(做什么),不绑定工具名(用什么做)→工具选择由下游智能体完成 5)action_type从给定标签中选择 6)输出纯JSON,参数阈值基于报告数据。如有不确定的数据,使用search_research_db工具查询。"},
                {"role":"user","content":prompt},
            ]

            # 🆕 Tool calling loop
            tools = None
            if vector_db_path:
                from src.tools.vector_store import ResearchVectorStore, SEARCH_TOOL_SCHEMA, set_research_vs
                try:
                    vs = ResearchVectorStore(vector_db_path)
                    set_research_vs(vs)
                    tools = [SEARCH_TOOL_SCHEMA]
                except Exception:
                    pass  # 向量库不可用时优雅降级

            for _ in range(5):  # 最多5轮tool call
                kwargs = dict(model=self.model, max_tokens=self.max_tokens, messages=messages)
                if not is_reasoner:
                    kwargs["temperature"] = self.temperature
                    kwargs["response_format"] = {"type":"json_object"}
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"

                resp = self.client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message

                # 检查是否有 tool calls
                if tools and msg.tool_calls:
                    messages.append(msg)  # 添加 assistant message with tool calls
                    for tc in msg.tool_calls:
                        if tc.function.name == "search_research_db":
                            args = json.loads(tc.function.arguments)
                            query = args.get("query", "")
                            print(f"    🔍 策略生成器查询向量库: {query[:60]}...", flush=True)
                            from src.tools.vector_store import _get_vs
                            vs2 = _get_vs()
                            if vs2:
                                result_text = vs2.search_formatted(query, top_k=3)
                            else:
                                result_text = "向量数据库不可用"
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result_text,
                            })
                    continue  # 继续下一轮

                # 正常的文本响应
                raw = msg.content or ""
                if not raw.strip():
                    raw = getattr(msg, "reasoning_content", "") or ""
                break  # 拿到最终响应, 退出循环

            if "{" in raw: raw = raw[raw.find("{"):]
            parsed = self._robust_parse(raw.strip())
            items = parsed.get("strategies", [])
            if not items and "strategy_name" in parsed:
                items = [parsed]
            if not items and len(raw) > 500:
                print(f"  ⚠️ JSON解析完全失败(raw={len(raw)}chars), 检查响应格式", flush=True)
                print(f"  📋 raw前200字符: {raw[:200]}", flush=True)
            result = []
            fail_count = 0
            for i, item in enumerate(items):
                # 去除 LLM 不应生成的步骤（input_validation, target_structure_acquisition — 系统自动注入）
                if "pipeline" in item:
                    item["pipeline"] = [s for s in item["pipeline"]
                                        if str(s.get("action_type", "")) not in {"input_validation", "target_structure_acquisition"}]
                # 归一化 requires 字段：LLM 常错误输出整数或对象，统一转为字符串
                for step in item.get("pipeline", []):
                    raw_req = step.get("requires", [])
                    if raw_req:
                        step["requires"] = [
                            str(r.get("step", r)) if isinstance(r, dict) else
                            f"step_{r}" if isinstance(r, (int, float)) else str(r)
                            for r in raw_req
                        ]
                    # 归一化 input/output 的 type/format 字段：LLM 有时输出列表而非字符串
                    for field_name in ("input", "output"):
                        field = step.get(field_name, {})
                        for key in ("type", "format"):
                            val = field.get(key)
                            if isinstance(val, list):
                                field[key] = ", ".join(str(v) for v in val)
                            elif isinstance(val, (int, float)):
                                field[key] = str(val)
                try:
                    s = DetailedStrategy(**item)
                    result.append(s.model_dump())
                except Exception as e:
                    fail_count += 1
                    if fail_count <= 3:
                        print(f"  ⚠️ 策略{i+1}校验失败: {e}", flush=True)
            if fail_count > 0:
                print(f"  📊 {len(items)}个原始策略, {fail_count}个校验失败, {len(result)}个成功", flush=True)
            # 🆕 Python端注入UUID (LLM不可靠)
            import uuid as _uuid
            for s in result:
                if not s.get("strategy_id"):
                    s["strategy_id"] = f"s-{_uuid.uuid4().hex[:8]}"
                for st in s.get("pipeline", []):
                    if not st.get("step_id"):
                        st["step_id"] = f"a-{_uuid.uuid4().hex[:8]}"
            return result, raw
        except Exception as e:
            print(f"  ❌ LLM调用异常: {e}", flush=True)
            return [], raw

    @staticmethod
    def _robust_parse(raw, verbose=True):
        """多层JSON修复: 处理LLM输出的各种格式问题, 尤其是大JSON的嵌套转义。"""
        errors = []
        # 0) 空输入
        if not raw or not raw.strip():
            return {}

        # 1) 直接解析
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            errors.append(f"direct:{e}")

        # 2) 去除markdown代码块
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```\w*\n', '', cleaned)
            cleaned = re.sub(r'\n```$', '', cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            errors.append(f"clean:{e}")

        # 3) raw_decode (找到第一个完整JSON对象)
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s >= 0 and e > s:
            try:
                decoder = json.JSONDecoder()
                result, _ = decoder.raw_decode(cleaned[s:e+1])
                return result
            except json.JSONDecodeError as ex:
                errors.append(f"raw_decode:{ex}")
                # 🆕 尝试修复常见问题后重试raw_decode
                try:
                    core = cleaned[s:e+1]
                    # 修复裸换行在字符串值中
                    core_fixed = re.sub(
                        r'(?<="\s*:\s*")(.*?)(?="\s*[,}\]])',
                        lambda m: m.group(1).replace('\n', '\\n').replace('\r', ''),
                        core, flags=re.DOTALL
                    )
                    result, _ = json.JSONDecoder().raw_decode(core_fixed)
                    if verbose:
                        print(f"  🔧 JSON换行修复后解析成功", flush=True)
                    return result
                except Exception:
                    pass

        # 4) 最后的兜底: 逐个策略正则提取
        if verbose:
            print(f"  ⚠️ JSON解析失败: {'; '.join(errors[-2:])}", flush=True)
            print(f"  🔧 尝试正则逐策略提取...", flush=True)

        strategies = []
        # 匹配每个策略块: {"strategy_name": "..." ...}
        strategy_blocks = re.findall(
            r'\{\s*"strategy_name"\s*:\s*"[^"]*".*?"suitable_when"\s*:\s*"[^"]*"\s*\}',
            cleaned, re.DOTALL
        )
        if not strategy_blocks:
            # 放宽: 匹配到下一个 "strategy_name" 或 ]
            strategy_blocks = re.findall(
                r'\{\s*"strategy_name"[^}]+(?:}[,;\s]*\}?)',
                cleaned, re.DOTALL
            )
        for block in strategy_blocks:
            try:
                strategies.append(json.loads(block))
            except json.JSONDecodeError:
                try:
                    strategies.append(json.JSONDecoder().raw_decode(block)[0])
                except Exception:
                    pass
        if strategies:
            if verbose:
                print(f"  ✅ 正则提取到 {len(strategies)} 个策略", flush=True)
            return {"strategies": strategies}

        return {}

    def _fallback(self, target_name, target_gene):
        """紧急 fallback — 当所有 LLM 调用失败时的基础策略模板 (v4 Action格式)。"""
        context = StrategyContext(target_name=target_name or "Unknown target",
                                  gene_symbol=target_gene or "",
                                  user_query=f"为{target_name}设计虚拟筛选策略")
        return {"strategies": self._fallback_from_context(context),
                "generation_rationale": f"Fallback: {target_name}({target_gene}) 8个自适应策略蓝图"}
        T, G = target_name, target_gene
        default_pipeline = [
            {"step_number":1,"action_type":"physicochemical_filtering","action_name":"类药性预过滤",
             "description":f"基于{T}的口袋特征筛选类药分子: MW 250-600, LogP 1-5, HBD≤5, HBA≤10, 去除PAINS和反应性基团。",
             "input":{"type":"compound_library","size":"1M","format":"SMILES"},
             "output":{"type":"filtered_library","size":"~500K","format":"SMILES"},
             "parameters":{"mw_range":[250,600],"logp_range":[1,5],"pains_filter":True},
             "quality_criteria":"PAINS=0, >95%通过理化过滤",
             "cardinality_estimate":"1M → 500K","computational_cost":"low",
             "requires":["compound_library_smiles"]},
            {"step_number":2,"action_type":"molecular_docking","action_name":"分子对接筛选",
             "description":f"使用{G}的PDB结构进行分子对接。定义结合位点盒子, 对接后按结合亲和力排序, 保留Top 10%化合物。",
             "input":{"type":"prepared_library","size":"500K","format":"SDF"},
             "output":{"type":"ranked_compounds","size":"~50K","format":"SDF"},
             "parameters":{"exhaustiveness":32,"num_modes":9},
             "quality_criteria":"对接成功>95%, 无原子冲突",
             "cardinality_estimate":"500K → 50K","computational_cost":"high",
             "requires":["protein_structure","binding_site_definition"]},
            {"step_number":3,"action_type":"admet_filtering","action_name":"ADMET 毒性过滤",
             "description":"对对接命中化合物进行ADMET预测: 肝毒性、hERG抑制、CYP450抑制、Ames致突变、Caco-2渗透性。排除有明显毒性风险的化合物。",
             "input":{"type":"ranked_compounds","size":"50K","format":"SDF"},
             "output":{"type":"safe_compounds","size":"~10K","format":"SDF"},
             "parameters":{"hERG":"low","CYP3A4":"negative","Ames":"negative"},
             "quality_criteria":"无高风险ADMET标志",
             "cardinality_estimate":"50K → 10K","computational_cost":"medium",
             "requires":["compound_structures"]},
            {"step_number":4,"action_type":"diversity_selection","action_name":"多样性筛选",
             "description":"使用Murcko骨架聚类, 每类骨架选取结合能最低的1-2个代表化合物, 确保最终输出具有化学多样性。",
             "input":{"type":"safe_compounds","size":"10K","format":"SDF"},
             "output":{"type":"diverse_hits","size":"~100","format":"SDF"},
             "parameters":{"clustering_method":"Murcko_scaffold","max_per_cluster":2},
             "quality_criteria":"≥10个不同骨架类型",
             "cardinality_estimate":"10K → 100","computational_cost":"low",
             "requires":["compound_structures"]},
        ]
        return {"strategies": [
            {"strategy_id":f"{G}_SBDD_001","strategy_name":f"基于结构的{T}虚拟筛选","strategy_tagline":"蛋白结构导向的多级筛选漏斗",
             "approach_category":"structure_based_screening","rationale":f"利用{T}({G})的结构信息进行基于结构的药物设计。",
             "target_profile":{"target_class":"other","pocket_type":"unknown","has_experimental_structure":True},
             "pipeline":default_pipeline,
             "survival_estimate":"1M→500K→50K→10K→100",
             "contingency_plan":{"trigger":"survivors<10","actions":["降低对接阈值","扩大初始库"]},
             "strengths":["基于实验结构","多级过滤","多样性保护"],
             "weaknesses":["依赖结构质量","通用模板,未针对靶点优化"],
             "estimated_runtime_category":"days",
             "knowledge_dependencies":["protein_structure"],
             "applicability_conditions":{"requires_structure":True,"requires_ligands":False,"min_library_size":"100K","suitable_target_types":["*"]}},
            {"strategy_id":f"{G}_LIG_001","strategy_name":f"基于配体的{T}相似性搜索","strategy_tagline":"已知配体导向的化学空间聚焦",
             "approach_category":"ligand_based_screening","rationale":f"利用{T}已知活性配体信息, 通过化学相似性聚焦搜索空间。",
             "target_profile":{"target_class":"other","pocket_type":"unknown","has_known_active_ligands":True},
             "pipeline":[
                 {"step_number":1,"action_type":"similarity_screening","action_name":"2D Tanimoto 相似性搜索",
                  "description":"计算库分子与已知活性配体的Morgan2指纹Tanimoto相似度, 保留>0.35的分子。",
                  "input":{"type":"compound_library","size":"1M","format":"SMILES"},
                  "output":{"type":"similar_compounds","size":"~200K","format":"SMILES"},
                  "parameters":{"fingerprint":"Morgan2_2048","tanimoto_cutoff":0.35},
                  "quality_criteria":">80%已知配体在相似集中",
                  "cardinality_estimate":"1M→200K","computational_cost":"medium","requires":["known_active_ligands"]},
             ],
             "survival_estimate":"1M→200K→50K→5K→200",
             "contingency_plan":{"trigger":"survivors<10","actions":["降低Tanimoto阈值至0.25"]},
             "strengths":["不依赖蛋白结构","快速富集"],
             "weaknesses":["依赖已知配体质量","可能局限化学空间"],
             "estimated_runtime_category":"days",
             "knowledge_dependencies":["known_active_ligands"],
             "applicability_conditions":{"requires_structure":False,"requires_ligands":True,"min_library_size":"100K","suitable_target_types":["*"]}},
            {"strategy_id":f"{G}_HYBRID_001","strategy_name":f"{T}宽筛漏斗","strategy_tagline":"低门槛粗筛→逐步收紧, 优先召回率",
             "approach_category":"hybrid_wide_funnel","rationale":"缺乏充分数据时, 用宽松阈值最大化覆盖, 逐步收紧。",
             "target_profile":{"target_class":"other","pocket_type":"unknown"},
             "pipeline":[
                 {"step_number":1,"action_type":"physicochemical_filtering","action_name":"粗过滤",
                  "description":"宽松的类药性过滤: MW 200-800, LogP 0-7, 排除PAINS。",
                  "input":{"type":"compound_library","size":"1M","format":"SMILES"},
                  "output":{"type":"filtered_library","size":"~800K","format":"SMILES"},
                  "parameters":{"mw_range":[200,800],"pains_filter":True},
                  "quality_criteria":"PAINS=0",
                  "cardinality_estimate":"1M→800K","computational_cost":"low","requires":[]},
                 {"step_number":2,"action_type":"molecular_docking","action_name":"低精度对接",
                  "description":"使用低exhaustiveness对接, 保留Top 30%。",
                  "input":{"type":"filtered_library","size":"800K","format":"SDF"},
                  "output":{"type":"ranked_compounds","size":"~240K","format":"SDF"},
                  "parameters":{"exhaustiveness":8},
                  "quality_criteria":"对接成功率>90%",
                  "cardinality_estimate":"800K→240K","computational_cost":"high","requires":["protein_structure"]},
             ],
             "survival_estimate":"1M→800K→240K→50K→500",
             "contingency_plan":{"trigger":"survivors<10","actions":["取消ADMET过滤","扩大初始库"]},
             "strengths":["高召回率","数据需求低"],
             "weaknesses":["假阳性率高","下游验证负担大"],
             "estimated_runtime_category":"days",
             "knowledge_dependencies":[],
             "applicability_conditions":{"requires_structure":False,"requires_ligands":False,"min_library_size":"100K","suitable_target_types":["*"]}},
        ], "generation_rationale": f"Fallback: {T}({G}) 3个基础模板"}


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
        "event_log": [f"[{now}] [StrategyGen] {len(strategies)} strategies."],
    }
