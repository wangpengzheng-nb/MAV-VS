"""
AutoVS-Agent v2.0: Strategy Generator
======================================
基于调研报告一次性生成5-10个策略。一次LLM调用+高质量fallback, 无多步脆弱链。
"""

from __future__ import annotations

import json, os, re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
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
for _m in [ActionInput, ActionOutput, PipelineAction, TargetProfile,
            ContingencyPlan, ApplicabilityConditions, DetailedStrategy]:
    _m.model_rebuild()


class StrategyGeneratorAgent:
    def __init__(self, model="deepseek-chat", api_key=None, api_base=None, temperature=0.5, max_tokens=16384):
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

    def generate_strategies(self, research_report: dict, target_info=None, prior_knowledge: str = ""):
        # 🆕 优先使用执行摘要, 否则 fallback 到完整报告
        summary = research_report.get("executive_summary", "")
        if summary:
            report_text = summary
        else:
            report_text = research_report.get("full_report_text", "")
        if not report_text:
            report_text = json.dumps(research_report, ensure_ascii=False, indent=2)

        target_name = research_report.get("target_name", "Unknown")
        target_gene = research_report.get("gene_symbol", "")
        target_uniprot = research_report.get("uniprot_id", "")

        # 🆕 向量数据库路径
        vector_db_path = research_report.get("_vector_db_path", "")

        user_query = research_report.get("_user_query", "")
        prompt = self._build_strategy_prompt(target_name, target_gene, target_uniprot,
                                              report_text, user_query, prior_knowledge)

        # 最多重试3次, 每次打印失败原因
        for attempt in range(3):
            strategies, raw_json = self._call_llm(prompt, target_name, target_gene, vector_db_path)
            if strategies and len(strategies) >= 3:
                return {"strategies": strategies, "generation_rationale": f"成功生成{len(strategies)}个策略 (第{attempt+1}次调用)"}
            print(f"\n⚠️ 策略生成第{attempt+1}次: {len(strategies) if strategies else 0}个策略。原始响应{len(raw_json)}字符", flush=True)
            if raw_json:
                print(f"  响应结尾: {raw_json[-300:]}", flush=True)

        # 3次都失败: 简化prompt再试
        simple_prompt = f"""为靶点{target_name}({target_gene})设计5个虚拟筛选策略。输出JSON。"""
        strategies, _ = self._call_llm(simple_prompt, target_name, target_gene, vector_db_path)
        if strategies and len(strategies) >= 3:
            return {"strategies": strategies, "generation_rationale": "简化prompt重试成功"}

        raise RuntimeError(
            f"策略生成失败: 4次LLM调用均返回<3个有效策略。"
            f"请检查调研报告是否完整, 或手动运行 test_tournament.py 查看详细日志。"
        )

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
2. **口袋/靶点类型**: PPI? 激酶? 别构? → 必须匹配对应工具! PPI用Diffdock, 其他用gnina!
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

        return f"""为靶点 {target_name} ({target_gene}, UniProt:{target_uniprot}) 设计8-10个虚拟筛选策略。

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

action_type 必须从以下标签中选择: library_preparation, protein_preparation, pharmacophore_screening, similarity_screening, shape_matching, admet_filtering, selectivity_filtering, diversity_selection, physicochemical_filtering, molecular_docking, covalent_docking, ensemble_docking, consensus_scoring, machine_learning_scoring, molecular_dynamics, free_energy_calculation, water_analysis, interaction_analysis, binding_mode_analysis, binding_site_detection, pocket_comparison, structure_alignment, de_novo_design, scaffold_hopping, fragment_growing, linker_design, r_group_enumeration, visual_inspection, positive_control_validation, decoy_validation, statistical_analysis

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

输出纯JSON, 所有字符串用双引号, 不要markdown代码块。必须输出8-10个差异化策略!"""


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
