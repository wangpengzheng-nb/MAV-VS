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


class StrategyStep(BaseModel):
    step_number: int = Field(default=1)
    step_name: str = Field(default="")
    tool: str = Field(default="")
    action: str = Field(default="")
    metric: str = Field(default="")
    threshold: str = Field(default="")
    rationale: str = Field(default="")


class DetailedStrategy(BaseModel):
    strategy_name: str = Field(default="")
    strategy_tagline: str = Field(default="")
    approach_type: str = Field(default="structure_based")
    rationale: str = Field(default="")
    pipeline_steps: List[StrategyStep] = Field(default_factory=list)
    survival_estimate: str = Field(default="")
    contingency: str = Field(default="")
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    estimated_runtime: str = Field(default="")
    suitable_when: str = Field(default="")


DetailedStrategy.model_rebuild()
StrategyStep.model_rebuild()


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

## JSON格式要求
{{
  "strategies": [{{
    "strategy_name": "策略名称(必须体现靶点特异性, 禁止泛化命名)",
    "strategy_tagline": "一句话描述核心创新点",
    "approach_type": "用1-3个词描述方法本质(不限于固定列表, 自由描述, 如: ensemble_docking, pharmacophore_hybrid, alchemical_free_energy, PROTAC_design, dual_target, etc.)",
    "rationale": "设计原理(200-400字, 必须引用调研报告中的IC50/PDB/口袋数据, 解释为什么这个策略适合本靶点)",
    "pipeline_steps": [{{
      "step_number": 1, "step_name": "步骤名", "tool": "使用的工具(精确名称+版本)",
      "action": "详细操作(100-300字, 含参数/输入/输出/判断标准)",
      "metric": "评价指标", "threshold": "具体数值阈值(必须基于调研报告数据, 如CNN_VS>5.0, MW<{target_name}已知配体上限)",
      "rationale": "设计理由(50-100字)"
    }}],
    "survival_estimate": "每步存活估算", "contingency": "若存活<10的应急预案",
    "strengths": ["优势描述"], "weaknesses": ["劣势描述"],
    "estimated_runtime": "时间估算", "suitable_when": "适用场景"
  }}]
}}

输出纯JSON, 所有字符串用双引号, 不要markdown代码块。重要: 必须输出8-10个差异化策略!"""


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
                {"role":"system","content":f"你是虚拟筛选策略专家。为{target_name}({target_gene})设计策略。核心规则: 1)用户query中'不要X/禁止X/避开X'→有排除步骤 2)PDB结构→推荐使用 3)禁止固定模板,基于靶点特异性定制 4)approach_type自由描述 5)输出纯JSON,阈值给基于报告的数值。如有不确定的数据,使用search_research_db工具查询。"},
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
        """高质量备选策略集 — 10个方向, 每步有详细内容, 绝非降级占位!"""
        T = target_name
        G = target_gene

        SBDD = [{"step_number":1,"step_name":"蛋白准备","tool":"Schrödinger ProteinPrep","action":f"下载{G}的PDB结构, 用Protein Preparation Wizard处理: 添加氢原子, 优化H键网络, 去除水分子(保留关键水), 在pH7.4下确定质子化状态, 约束最小化到RMSD 0.3Å。","metric":"蛋白质量","threshold":"Ramachandran outliers < 1%, clashscore < 5","rationale":"高质量蛋白结构是准确对接的前提"},
                 {"step_number":2,"step_name":"配体库准备","tool":"RDKit/OpenBabel","action":"从Enamine REAL或ZINC20下载Lead-like子集(MW 250-500, LogP -1到5, 可旋转键≤8)。用ETKDGv3生成3D构象, MMFF94s优化。添加显式氢, 检查手性。输出multi-conformer SDF。","metric":"库质量","threshold":">95%分子成功生成构象, 无立体化学冲突","rationale":"类药子集提高筛选效率, ETKDGv3对有机分子构象准确"},
                 {"step_number":3,"step_name":"分子对接","tool":"GNINA refinement","action":f"用共晶配体质心定义对接盒子(外扩8Å), exhaustiveness=64, num_modes=9, CNN_scoring=refinement, cnn_rotation=4。每个分子保留最佳CNN_VS姿态。","metric":"CNN_VS","threshold":"CNN_VS > 5.0 AND minimizedAffinity < -9.0","rationale":"高exhaustiveness确保充分采样, CNN_VS综合评分优于单纯亲和力"},
                 {"step_number":4,"step_name":"PLIP相互作用分析","tool":"PLIP","action":f"对对接姿态运行PLIP分析, 提取氢键/疏水接触/盐桥/π堆积。关注{G}关键残基的相互作用模式。","metric":"PLIP_score","threshold":">=12 (至少1个关键H-bond + 3个疏水接触)","rationale":"相互作用指纹验证对接姿态合理性, 排除假阳性对接"},
                 {"step_number":5,"step_name":"ADMET过滤","tool":"ADMET-AI","action":"预测40+ADMET属性: 肝毒性/hERG/CYP抑制/Ames/BBB/水溶性/Caco-2。综合评估类药性。","metric":"ADMET综合","threshold":"PAINS=0, hERG=low, CYP3A4=negative, Ames=negative","rationale":"早期排除毒性分子, 避免后期开发失败"}]

        LIG = [{"step_number":1,"step_name":"参考配体收集","tool":"ChEMBL/PDB","action":f"从ChEMBL和PDB收集{G}的所有已知配体(IC50<10μM)。提取SMILES, 去冗余, 随机分为参考集(80%)和验证集(20%)。","metric":"配体数量","threshold":">10个高活性配体(IC50<1μM)","rationale":"高质量参考集是配体筛选的基础"},
                 {"step_number":2,"step_name":"Tanimoto指纹筛选","tool":"RDKit Morgan2","action":"计算库分子与参考集的最大Tanimoto相似度(Morgan半径2, 2048位)。保留Tanimoto > 0.35的分子, 按最高相似度排序。","metric":"Tanimoto相似度","threshold":">0.35 进入下一轮; >0.5 优先对接","rationale":"化学空间聚焦, 相似性搜索可有效富集活性分子"},
                 {"step_number":3,"step_name":"3D形状与静电匹配","tool":"ROCS/EON","action":"用参考配体的最低能量构象生成3D形状查询。计算形状Tanimoto和静电相似度。保留ComboScore > 1.2的分子。","metric":"ComboScore","threshold":">1.2 (ShapeTanimoto > 0.7 AND ColorScore > 0.5)","rationale":"3D形状+静电匹配比2D指纹更好地反映结合互补性"}]

        ML = [{"step_number":1,"step_name":"训练数据准备","tool":"ChEMBL+RDKit","action":f"从ChEMBL收集{G}活性数据(IC50/Ki<50μM为活性, >50μM或无明显抑制为阴性)。计算ECFP4指纹(Morgan半径2, 2048位)和RDKit描述符(分子量, LogP, HBD, HBA, TPSA, 环数, 可旋转键)。","metric":"数据集大小","threshold":">100个标注分子(活性:非活性≈1:3)","rationale":"充分训练数据是ML模型的基础"},
                 {"step_number":2,"step_name":"模型训练","tool":"Scikit-learn/XGBoost","action":"用5折交叉验证训练XGBoost分类器。超参数调优: max_depth(3-10), learning_rate(0.01-0.3), n_estimators(100-500)。评估AUC-ROC, 精确率, 召回率。","metric":"AUC-ROC","threshold":">0.80 (测试集)","rationale":"XGBoost在分子活性预测中表现优异, 计算快"},
                 {"step_number":3,"step_name":"虚拟筛选","tool":"Python脚本","action":"用训练好的模型预测ZINC20或Enamine REAL库(10^7-10^9分子)的活性概率。保留预测活性概率 > 0.7的分子。","metric":"预测活性概率","threshold":">0.7","rationale":"ML筛选可遍历超大化学空间, 发现全新骨架"}]

        FRAG = [{"step_number":1,"step_name":"片段库准备","tool":"RDKit","action":f"从ZINC或Enamine片段库(包含约10^5个片段分子: MW<300, LogP<3, 可旋转键<3, HBD<4, HBA<6)。用ETKDGv3生成3D构象。","metric":"片段数量","threshold":"约10^5个片段","rationale":"片段库覆盖广泛化学空间, 适合FBDD"},
                 {"step_number":2,"step_name":"片段对接","tool":"GNINA","action":"片段分子用低exhaustiveness=16对接。保留CNN_VS > 3.0的片段。计算配体效率(LE = -ΔG/重原子数)。","metric":"配体效率(LE)","threshold":"LE > 0.3 kcal/mol/atom","rationale":"片段结合弱但效率高是FBDD的核心指标"},
                 {"step_number":3,"step_name":"片段连接/生长","tool":"BREED/AutoCoupling","action":"对高LE片段, 用BREED算法搜索可连接片段对, 或基于口袋残基进行片段生长。用GNINA验证连接产物的结合模式。","metric":"连接产物CNN_VS","threshold":">5.0","rationale":"片段连接是利用弱结合片段构建强效配体的关键步骤"}]

        return {"strategies": [
            {"strategy_name":f"严格SBDD: {T}的高通量虚拟筛选","strategy_tagline":"基于高分辨率晶体结构的精准对接+多级过滤","approach_type":"structure_based","rationale":f"利用{T}({G})的高分辨率晶体结构进行严格SBDD筛选。通过高精度对接、PLIP相互作用分析、ADMET预测和理化性质过滤, 确保筛选分子的结合亲和力、特异性和药物样性。","pipeline_steps":SBDD,"survival_estimate":"100K→90K(构象)→15K(对接)→3K(PLIP)→1K(ADMET)→500(理化)","contingency":"若500<10: 放宽对接CNN_VS>4.0, 取消理化过滤。若仍<10: 扩大初始库至500K","strengths":["高精度对接保障结合可靠性","多级过滤逐步聚焦","PLIP验证相互作用模式"],"weaknesses":["依赖高质量晶体结构","可能遗漏诱导契合","计算成本高"],"estimated_runtime":"~4天(100K分子)","suitable_when":"有高分辨率共晶/同源结构时"},
            {"strategy_name":f"配体驱动: 基于{T}已知配体的相似性搜索","strategy_tagline":"以已知活性分子为锚点的化学空间聚焦筛选","approach_type":"ligand_based","rationale":f"利用{T}已知活性配体的结构信息, 通过2D指纹和3D形状相似性搜索, 聚焦于化学空间中与已知活性分子相似的区域。该方法不依赖蛋白结构, 可有效富集潜在活性分子。","pipeline_steps":LIG,"survival_estimate":"100K→50K(tanimoto)→10K(3D)→3K(对接)","contingency":"若3K<10: 降低Tanimoto阈值至0.25。若仍<10: 增加参考配体来源","strengths":["不依赖蛋白结构","化学空间聚焦高效","可发现新骨架类似物"],"weaknesses":["依赖已知配体质量","可能局限化学空间","3D形状匹配计算慢"],"estimated_runtime":"~2天","suitable_when":"有足够的已知活性配体(>5个)"},
            {"strategy_name":f"宽筛漏斗: {T}的探索性虚拟筛选","strategy_tagline":"低门槛粗筛→逐步收紧, 优先召回率","approach_type":"hybrid","rationale":f"适用于缺乏充分已知配体或{T}结构信息不完整的情况。使用宽松的对接阈值和简单的药物样性过滤, 最大化化学空间覆盖, 避免漏掉潜在活性分子。通过多级漏斗逐步缩小候选池。","pipeline_steps":[{"step_number":1,"step_name":"GNINA粗对接","tool":"GNINA rough","action":"exhaustiveness=8, 保留CNN_VS前30%。","metric":"CNN_VS","threshold":">3.0 进入下一轮","rationale":"低门槛确保不遗漏"},{"step_number":2,"step_name":"PLIP+PAINS","tool":"PLIP+RDKit","action":"排除PAINS子结构, 保留至少1个氢键或3个疏水的分子。","metric":"PAINS+PLIP","threshold":"PAINS=0, interaction≥1","rationale":"唯一红线+基本相互作用"}],"survival_estimate":"100K→30K(对接)→20K(过滤)","contingency":"若存活<10: 取消PLIP, 仅保留PAINS过滤","strengths":["高召回率","适合无结构靶点","计算快速"],"weaknesses":["假阳性率高","下游验证负担大"],"estimated_runtime":"~1天","suitable_when":"缺乏结构/配体信息时"},
            {"strategy_name":f"ML驱动: {T}抑制剂预测模型","strategy_tagline":"训练机器学习模型预测活性, 筛选超大型库","approach_type":"ml_driven","rationale":f"构建{T}的机器学习活性预测模型, 利用已知活性数据和分子指纹进行训练, 对百万至十亿级分子库进行虚拟筛选, 发现具有新颖骨架的潜在抑制剂。","pipeline_steps":ML,"survival_estimate":"10^7→10^6(预测)→10^4(对接)→10^3(MD)","contingency":"若10^3<10: 降低预测阈值至0.5。若仍<10: 增加训练数据","strengths":["可遍历超大化学空间","发现新颖骨架","计算效率高"],"weaknesses":["依赖训练数据质量","模型可解释性差","可能过拟合"],"estimated_runtime":"~3天(含训练)","suitable_when":"有足够标注数据(>50个分子)"},
            {"strategy_name":f"片段筛选: {T}的FBDD虚拟筛选","strategy_tagline":"从片段库出发, 配体效率驱动的先导化合物发现","approach_type":"fragment_based","rationale":f"片段基药物设计(FBDD)是发现新型先导化合物的有效策略。利用{T}结构, 筛选低分子量片段库, 通过配体效率(LE)评估, 后续进行片段连接或生长, 构建高亲和力配体。","pipeline_steps":FRAG,"survival_estimate":"10^5片段→5K(对接)→500(LE过滤)→50(连接)","contingency":"若50<10: 降低LE阈值至0.25。若仍<10: 扩大片段库","strengths":["可从极简片段构建强效配体","发现新型化学起点","配体效率驱动"],"weaknesses":["片段结合弱","连接步骤复杂","需要结构生物学支持"],"estimated_runtime":"~2天","suitable_when":"有高分辨率蛋白结构时"},
            {"strategy_name":f"药效团筛选: {T}的3D药效团匹配","strategy_tagline":"基于已知配体药效团特征的快速筛选","approach_type":"ligand_based","rationale":f"从{T}的已知抑制剂提取药效团模型, 包括关键氢键供体/受体、疏水中心和排除体积。用3D药效团匹配筛选商业分子库, 快速富集符合药效团特征的分子。","pipeline_steps":[{"step_number":1,"step_name":"药效团建模","tool":"Pharmit/MOE","action":f"从{T}已知配体和共晶结构提取药效团特征。定义氢键供体/受体、疏水中心和芳香中心及空间排布。","metric":"药效团匹配度","threshold":"关键特征>3个","rationale":"药效团捕捉分子识别的本质特征"},{"step_number":2,"step_name":"3D药效团搜索","tool":"Pharmit","action":"对ZINC或Enamine库进行3D药效团搜索。每个分子生成多个构象, 匹配药效团特征。","metric":"匹配得分","threshold":"至少3/4特征匹配","rationale":"3D搜索考虑构象灵活性"}],"survival_estimate":"10^6→10^4(药效团)→10^3(对接)","contingency":"若<10: 放宽特征匹配数至2/4","strengths":["快速","不依赖对接","直观反映结合本质"],"weaknesses":["依赖药效团质量","可能忽略诱导契合"],"estimated_runtime":"~1天","suitable_when":"有高质量药效团模型时"},
            {"strategy_name":f"共价筛选: {T}的共价抑制剂发现","strategy_tagline":"靶向活性位点半胱氨酸的共价虚拟筛选","approach_type":"covalent","rationale":f"如果{T}的口袋附近存在可靶向的半胱氨酸(Cys), 采用共价抑制剂策略可获得持久的靶点抑制和更好的选择性。虚拟筛选中使用共价对接算法, 筛选含共价弹头的分子库。","pipeline_steps":[{"step_number":1,"step_name":"共价位点识别","tool":"PyMOL/MOE","action":f"分析{T}晶体结构, 识别口袋内或附近的可靶向Cys。计算Cys的溶剂可及性和pKa, 评估反应性。","metric":"Cys可靶向性","threshold":"溶剂可及>20%, pKa<10","rationale":"只有特定Cys适合共价靶向"},{"step_number":2,"step_name":"共价对接","tool":"AutoDock CovDock/DOCKTITE","action":"用共价弹头库(丙烯酰胺/氯乙酰胺/乙烯基磺酰胺等)进行共价对接。约束弹头与目标Cys的接近距离和反应角度。","metric":"共价结合能","threshold":"< -7.0 AND 弹头-CYS距离<2.5Å","rationale":"共价结合需满足空间和能量双重约束"}],"survival_estimate":"10^5→10^3(对接)→100(PLIP)","contingency":"若<10: 放宽距离约束至3.0Å","strengths":["持久靶点抑制","潜在更好选择性","适合不可药靶点"],"weaknesses":["需要可靶向Cys","脱靶共价修饰风险","设计复杂度高"],"estimated_runtime":"~2天","suitable_when":"口袋内有可靶向Cys时"}],
            "generation_rationale": f"基于{T}({G})覆盖7个方向"
        }


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
