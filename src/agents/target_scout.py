"""
AutoVS-Agent v2.0: Target Scout Agent (深度调研智能体)
========================================================
职责: 深度联网调研靶点蛋白, 生成 ~2000字的结构化调研报告。
     不做策略生成! 只为下游 StrategyGenerator 和 RedTeamReviewer 提供情报基础。

调研维度:
  1. 靶点生物学 (功能/通路/疾病关联)
  2. 结构分析 (PDB/口袋/关键残基/柔性)
  3. 已知配体与SAR (药物/活性化合物/药效团)
  4. 成药性评估 (口袋特征/挑战/推荐方法)
  5. 结合位点详细信息 (坐标/体积/极性)
  6. 参考文献

输入: 自由文本 query 或 TargetInfo dict
输出: ResearchReport (结构化数据 + ~2000字文本报告)
"""

from __future__ import annotations

import json
import os
import re
import ssl
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field


# =============================================================================
# 1. 结构化输出: 深度调研报告
# =============================================================================

class BindingSiteDetail(BaseModel):
    """结合位点详细信息。"""
    pocket_description: str = Field(..., description="口袋的几何和物理化学特征描述")
    volume_angstrom3: str = Field(default="unknown", description="估算口袋体积")
    polarity: str = Field(default="unknown", description="hydrophobic / mixed / polar")
    flexibility: str = Field(default="unknown", description="rigid / moderate / highly_flexible / cryptic")
    key_residues: List[Dict[str, str]] = Field(
        default_factory=list,
        description="关键残基列表, 每项: {name, role, chain, resnum}"
    )
    center_coordinates: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="对接盒子建议中心 [x, y, z] (从PDB共晶结构或文献获取)"
    )
    suggested_box_size: List[float] = Field(
        default_factory=lambda: [20.0, 20.0, 20.0],
        description="建议对接盒子尺寸 [sx, sy, sz]"
    )


class KnownLigandDetail(BaseModel):
    """已知活性配体详细信息。"""
    name: str = Field(..., description="化合物名称/代号")
    smiles: str = Field(default="", description="SMILES (如已知)")
    activity_type: str = Field(default="", description="IC50 / Ki / Kd / EC50")
    activity_value: str = Field(default="", description="活性数值 + 单位")
    mechanism: str = Field(default="", description="作用机制 (竞争性/共价/变构)")
    pdb_complex: str = Field(default="", description="共晶结构PDB ID (如有)")
    key_features: List[str] = Field(default_factory=list, description="关键结构特征/药效团要素")


class ResearchReport(BaseModel):
    """深度调研报告 — TargetScout 的最终产出。"""

    # ---- 基本信息 ----
    target_name: str = Field(..., description="靶点标准名称")
    gene_name: str = Field(default="", description="基因符号")
    uniprot_id: str = Field(default="", description="UniProt ID")
    organism: str = Field(default="Homo sapiens")

    # ---- 结构化关键数据 ----
    target_class: str = Field(default="", description="PPI/Kinase/GPCR/Protease/E3_ligase/...")
    binding_site: BindingSiteDetail = Field(default_factory=BindingSiteDetail)
    known_ligands: List[KnownLigandDetail] = Field(default_factory=list)
    pdb_structures: List[Dict[str, Any]] = Field(default_factory=list)

    # ---- 核心: 长篇文本报告 ----
    biology_overview: str = Field(
        ..., min_length=200,
        description="靶点生物学概述: 基因/蛋白功能、信号通路、疾病关联。300-500字。"
    )
    structural_analysis: str = Field(
        ..., min_length=200,
        description="结构分析: PDB结构质量、口袋几何、关键残基作用、蛋白柔性。300-500字。"
    )
    known_ligands_sar: str = Field(
        ..., min_length=150,
        description="已知配体与构效关系: 代表性配体、结合模式、SAR规律。200-400字。"
    )
    druggability_assessment: str = Field(
        ..., min_length=150,
        description="成药性评估: 口袋适应性、预测难点、推荐技术路线。200-400字。"
    )
    screening_recommendations: str = Field(
        ..., min_length=200,
        description="虚拟筛选建议: 适合的方法、关键评价指标、需注意的陷阱、对接参数建议。300-500字。"
    )
    references: List[str] = Field(
        default_factory=list,
        description="关键文献列表 (PMID/DOI/PDB ID)"
    )

    # ---- 元信息 ----
    full_report_text: str = Field(
        default="",
        description="合并后的完整调研报告文本 (~2000字)"
    )
    research_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    research_sources: List[str] = Field(default_factory=list)


ResearchReport.model_rebuild()
BindingSiteDetail.model_rebuild()
KnownLigandDetail.model_rebuild()


# =============================================================================
# 2. 深度调研系统提示词
# =============================================================================

DEEP_RESEARCH_SYSTEM_PROMPT = """\
# 人设: 资深药物靶点情报分析师 (Senior Target Intelligence Analyst)

你是一位在顶级药企工作了 20 年的**靶点情报分析专家**。
你的唯一职责是对给定的药物靶点进行**极其详尽的深度调研**,
产出一份约 2000 字的专业调研报告。

你下游的"策略生成器"和"专家评审团"都完全依赖你的报告来做决策。
如果报告粗糙, 整个虚拟筛选项目都会失败!

---

# 调研要求 (必须逐项深入)

## 1. 靶点生物学 (300-500字)
- 基因全称、蛋白功能、所属家族
- 参与的信号通路和在细胞中的角色
- 与疾病的明确关联 (哪些疾病? 什么机制? 有临床验证吗?)
- 是否有已知的动物模型或遗传学证据?
- 该靶点是否已被验证为药物靶点? 是否有已上市或临床阶段药物?

## 2. 结构分析 (300-500字)
- 是否有实验结构 (X-ray/Cryo-EM/NMR)? 列出全部PDB ID和分辨率
- 最关键: **结合位点的详细分析**:
  * 口袋的几何形状 (深裂隙/浅沟槽/平坦界面/隐蔽口袋)
  * 估算口袋体积
  * 极性分布 (疏水位点 vs 极性位点)
  * 关键残基及其作用 (哪些残基可供氢键? 疏水锚点在哪里?)
  * 蛋白柔性评估 (刚性对接是否可行? 是否需要诱导契合?)
- 是否有共晶结构? 配体的结合模式是什么?
- **给出具体的对接盒子建议中心和尺寸** (从共晶配体坐标推断)

## 3. 已知配体与SAR (200-400字)
- 列出所有已知活性配体 (至少3个, 越多越好)
- 每个配体: 名称、活性值(具体数值!)、作用机制、PDB共晶ID
- 总结这类靶点的构效关系 (SAR) 规律
- 药效团模型: 哪些特征是必需的? 哪些是加分的?
- 选择性问题: 如何避免脱靶?

## 4. 成药性评估 (200-400字)
- 该靶点的主要药物设计挑战
- 是否有可靶向的共价位点 (Cys/Lys)?
- 适合哪种分子类型? (传统小分子/大环肽/PROTAC/分子胶)
- 预测的口服生物利用度如何?
- 血脑屏障穿透是否需要?

## 5. 虚拟筛选具体建议 (300-500字)
- 推荐哪些对接软件? (GNINA/smina/Vina/Glide) 为什么?
- **对接盒子中心和尺寸的具体数值**
- 推荐的 exhaustiveness 参数
- 哪些评分函数更适合该靶点?
- MD模拟是否必须? 多长时间?
- 后处理的PLIP相互作用分析应关注哪些残基?
- 关键的过滤指标和阈值 (给出具体数值!)
- 容易出现的假阳性模式
- 建议的阳性对照化合物 (具体名称)

---

# 输出格式
严格输出 ResearchReport JSON Schema。
full_report_text 字段合并以上所有文本为一个完整报告。
**禁止使用占位符! 每个字段都必须是靶点特定的、有具体数值的真实信息。**
"""


# =============================================================================
# 意图解析 (保持)
# =============================================================================

class IntentParseResult(BaseModel):
    target_name: str = Field(..., description="靶点标准名称/基因符号")
    uniprot_id: str = Field(default="")
    pdb_id: str = Field(default="")
    target_class: str = Field(default="")
    domain: str = Field(default="")
    modification: str = Field(default="")
    known_drugs: List[str] = Field(default_factory=list)
    organism: str = Field(default="Homo sapiens")
    description: str = Field(default="")

IntentParseResult.model_rebuild()

INTENT_PARSE_SYSTEM_PROMPT = """\
你是一个精准的药物靶点识别系统。从用户自然语言中提取靶点信息。

关键规则:
1. 靶点名称必须保留完整的基因符号,包括:
   - 连字符: "BCL-2" 不是 "BCL", "BCL-xl" 不是 "BCL"
   - 数字后缀: "PGK2" 不是 "PGK", "CBLB" 不是 "CBL"
   - 突变标注: "EGFR T790M" 提取 name="EGFR", modification="T790M"
2. 如果用户提到了家族成员(如 BCL-2、BCL-xl),请精确识别是哪一种
3. 如果知道 UniProt ID 请填写,不确定则留空(系统会自动搜索)
4. target_class 推断: BCL-2/BCL-xl → PPI; EGFR → Kinase; KRAS → PPI
5. 靶点名称: 请使用标准的HUGO基因符号, 如BCL2, BCL2L1, EGFR, KRAS, PGK2, CBLB
"""


# =============================================================================
# 3. TargetScoutAgent
# =============================================================================

class TargetScoutAgent:
    """深度调研智能体 — 产出 ResearchReport。"""

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.3,
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

    # -------------------------------------------------------------------------
    # 🆕 入口: 深度调研
    # -------------------------------------------------------------------------

    def deep_research(self, query: str) -> Dict[str, Any]:
        """从自由文本出发, 执行完整的深度调研。

        流程:
          1. LLM 意图解析 → 提取靶点名称/ID
          2. 用 ID 查询 UniProt/PDB/ChEMBL API (真实数据)
          3. 用靶点名称搜索 UniProt (ID缺失时)
          4. 组装增强 Prompt (含用户查询 + API数据)
          5. LLM 深度推理 → ResearchReport (~2000字)
        """
        search_log = [f"Query: {query}"]

        # Step 1: 意图解析
        intent = self._parse_intent(query)
        search_log.append(f"Parsed: target_name={intent.get('target_name','?')}, "
                         f"uniprot={intent.get('uniprot_id','?')}, "
                         f"pdb={intent.get('pdb_id','?')}, "
                         f"class={intent.get('target_class','?')}")

        # Step 2: 用已识别的 ID 拉取公开数据库真实数据
        target_info = {
            "target_name": intent.get("target_name", ""),
            "uniprot_id": intent.get("uniprot_id", ""),
            "pdb_id": intent.get("pdb_id", ""),
            "target_class": intent.get("target_class", ""),
            "description": intent.get("description", query),
            "organism": intent.get("organism", "Homo sapiens"),
            "key_residues": intent.get("key_residues", []),
        }

        # Step 3: 如 ID 缺失, 用靶点名搜索 UniProt
        if not target_info["uniprot_id"] and target_info["target_name"]:
            found = self._search_uniprot_by_name(target_info["target_name"])
            if found:
                target_info["uniprot_id"] = found
                search_log.append(f"UniProt search → {found}")

        # Step 4: 拉取全部 API 数据
        api_data = self._fetch_all_research_data(target_info)
        search_log.append(f"Sources: {api_data.get('_sources', [])}")

        # Step 5: 构建深度研究 Prompt + LLM 生成报告
        user_prompt = self._build_deep_research_prompt(query, target_info, api_data)

        # 构建精简 JSON Schema (去掉嵌套 Field description, 避免 prompt 过大)
        slim_schema = {
            "type": "object",
            "required": ["target_name", "biology_overview", "structural_analysis",
                         "known_ligands_sar", "druggability_assessment", "screening_recommendations"],
            "properties": {
                "target_name": {"type": "string"},
                "gene_name": {"type": "string"},
                "uniprot_id": {"type": "string"},
                "organism": {"type": "string"},
                "target_class": {"type": "string"},
                "binding_site": {"type": "object", "properties": {
                    "pocket_description": {"type": "string"},
                    "volume_angstrom3": {"type": "string"},
                    "polarity": {"type": "string"},
                    "flexibility": {"type": "string"},
                    "key_residues": {"type": "array", "items": {"type": "object", "properties": {
                        "name": {"type": "string"}, "role": {"type": "string"},
                        "chain": {"type": "string"}, "resnum": {"type": "string"}}}},
                    "center_coordinates": {"type": "array", "items": {"type": "number"}},
                    "suggested_box_size": {"type": "array", "items": {"type": "number"}},
                }},
                "known_ligands": {"type": "array", "items": {"type": "object", "properties": {
                    "name": {"type": "string"}, "smiles": {"type": "string"},
                    "activity_type": {"type": "string"}, "activity_value": {"type": "string"},
                    "mechanism": {"type": "string"}, "pdb_complex": {"type": "string"},
                    "key_features": {"type": "array", "items": {"type": "string"}} }}},
                "pdb_structures": {"type": "array", "items": {"type": "object"}},
                "biology_overview": {"type": "string"},
                "structural_analysis": {"type": "string"},
                "known_ligands_sar": {"type": "string"},
                "druggability_assessment": {"type": "string"},
                "screening_recommendations": {"type": "string"},
                "references": {"type": "array", "items": {"type": "string"}},
            }
        }
        try:
            response = self.client.chat.completions.create(
                model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": DEEP_RESEARCH_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "system", "content": f"请输出以下JSON格式(所有字符串字段必须详细填写,禁止占位符):\n{json.dumps(slim_schema, indent=2, ensure_ascii=False)}"},
                ],
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            report = ResearchReport.model_validate(parsed).model_dump()

            # 合并完整报告文本
            sections = [
                report.get("biology_overview", ""),
                report.get("structural_analysis", ""),
                report.get("known_ligands_sar", ""),
                report.get("druggability_assessment", ""),
                report.get("screening_recommendations", ""),
            ]
            report["full_report_text"] = "\n\n".join(s for s in sections if s)
            report["research_sources"] = api_data.get("_sources", [])
            if "_search_log" not in report:
                report["_search_log"] = search_log
            return report

        except Exception as e:
            report = self._fallback_report(target_info, str(e))
            report["_search_log"] = search_log
            return report

    def _build_deep_research_prompt(
        self, query: str, target_info: dict, api_data: dict,
    ) -> str:
        """构建包含用户查询+API真实数据的深度调研 Prompt。"""
        parts = [f"""\
## 深度靶点调研任务

### 用户查询
{query}

### 基本信息
- 靶点名称: {target_info.get('target_name', '?')}
- UniProt ID: {target_info.get('uniprot_id', '?')}
- PDB ID: {target_info.get('pdb_id', '?')}
- 靶点类别: {target_info.get('target_class', '?')}
"""]
        # UniProt
        uni = api_data.get("uniprot")
        if uni:
            parts.append(f"""\
### UniProt 真实数据
- 蛋白全称: {uni.get('protein_name','?')}
- 基因名: {uni.get('gene_name','?')}
- 物种: {uni.get('organism','?')}
- 序列长度: {uni.get('length','?')} aa
- 功能: {uni.get('function','?')[:800]}
""")
        # PDB
        pdb = api_data.get("pdb")
        if pdb:
            parts.append(f"""\
### PDB 结构数据
- 分辨率: {pdb.get('resolution','?')} Å
- 实验方法: {pdb.get('method','?')}
- 结合配体: {pdb.get('bound_ligands',[])}
- PMID: {pdb.get('primary_pmid','?')}
""")
        # ChEMBL
        chembl = api_data.get("chembl")
        if chembl and chembl.get("top_compounds"):
            parts.append("### ChEMBL 已知活性化合物")
            for c in chembl["top_compounds"][:8]:
                parts.append(f"  - {c.get('molecule_chembl_id','?')}: "
                           f"{c.get('standard_type','?')}={c.get('standard_value','?')} {c.get('standard_units','?')}")
        parts.append("""\
### 任务
请生成一份极其详尽的靶点调研报告, 必须包含:
1. **生物学功能与疾病关联** (具体通路和临床证据)
2. **结构分析** (PDB质量、口袋几何、关键残基及坐标)
3. **已知配体与SAR** (具体化合物名称和活性数值)
4. **成药性评估** (口袋适应性、主要挑战)
5. **虚拟筛选建议** (具体对接盒子坐标/尺寸、关键过滤指标及数值、阳性对照)
6. **参考文献** (PMID/DOI/PDB ID)

每个部分都要有**具体数值**, 禁止泛泛而谈!""")
        return "\n\n".join(parts)

    # -------------------------------------------------------------------------
    # API 数据拉取
    # -------------------------------------------------------------------------

    def _fetch_all_research_data(self, target_info: dict) -> Dict[str, Any]:
        """拉取全部公开数据库数据。"""
        data: Dict[str, Any] = {}
        sources = []
        uid = target_info.get("uniprot_id", "").strip()
        pid = target_info.get("pdb_id", "").strip()
        tname = target_info.get("target_name", "").strip()

        if uid:
            try:
                uni = self._fetch_uniprot(uid)
                if uni: data["uniprot"] = uni; sources.append(f"UniProt:{uid}")
            except Exception: pass
        if pid:
            try:
                pdb = self._fetch_pdb(pid)
                if pdb: data["pdb"] = pdb; sources.append(f"PDB:{pid}")
            except Exception: pass
        if tname:
            try:
                chembl = self._fetch_chembl(tname)
                if chembl: data["chembl"] = chembl; sources.append(f"ChEMBL:{tname}")
            except Exception: pass
        data["_sources"] = sources
        return data

    def _fetch_uniprot(self, uid: str) -> Optional[Dict]:
        resp = self._http_get(f"https://rest.uniprot.org/uniprotkb/{uid}.json")
        if not resp: return None
        r = {}
        r["protein_name"] = resp.get("proteinDescription",{}).get("recommendedName",{}).get("fullName",{}).get("value","")
        for gn in resp.get("genes",[]):
            r["gene_name"] = gn.get("geneName",{}).get("value",""); break
        r["organism"] = resp.get("organism",{}).get("scientificName","")
        for c in resp.get("comments",[]):
            if c.get("commentType")=="FUNCTION":
                r["function"] = c.get("texts",[{}])[0].get("value","")[:800]; break
        r["length"] = resp.get("sequence",{}).get("length",0)
        return r

    def _fetch_pdb(self, pid: str) -> Optional[Dict]:
        resp = self._http_get(f"https://data.rcsb.org/rest/v1/core/entry/{pid}")
        if not resp: return None
        r = {}
        r["resolution"] = resp.get("rcsb_entry_info",{}).get("resolution_combined",[None])[0]
        r["method"] = resp.get("rcsb_entry_info",{}).get("experimental_method","")
        ligands = []
        nl = resp.get("rcsb_entry_info",{}).get("nonpolymer_bound_components",{})
        for k in nl:
            if isinstance(nl[k], list):
                for item in nl[k]:
                    if isinstance(item, dict):
                        ligands.append(item.get("comp_id",""))
        r["bound_ligands"] = ligands
        r["primary_pmid"] = str(resp.get("rcsb_primary_citation",{}).get("pdbx_database_id_PubMed","")) if isinstance(resp.get("rcsb_primary_citation"), dict) else ""
        return r

    def _fetch_chembl(self, name: str) -> Optional[Dict]:
        import urllib.parse
        url = f"https://www.ebi.ac.uk/chembl/api/data/target/search.json?q={urllib.parse.quote(name)}&limit=3"
        resp = self._http_get(url)
        if not resp: return None
        targets = resp.get("targets",[])
        if not targets: return None
        r = {"top_compounds": []}
        for t in targets[:2]:
            cid = t.get("target_chembl_id","")
            if cid:
                cr = self._http_get(f"https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id={cid}&limit=5&standard_type__in=IC50,Ki,Kd")
                if cr:
                    for a in cr.get("activities",[])[:5]:
                        r["top_compounds"].append({
                            "molecule_chembl_id": a.get("molecule_chembl_id",""),
                            "standard_type": a.get("standard_type",""),
                            "standard_value": a.get("standard_value",""),
                            "standard_units": a.get("standard_units",""),
                        })
                    break
        return r if r["top_compounds"] else None

    def _search_uniprot_by_name(self, name: str) -> str:
        import urllib.parse
        data = self._http_get(f"https://rest.uniprot.org/uniprotkb/search?query={urllib.parse.quote(name)}+AND+organism_id:9606&size=3")
        if data:
            results = data.get("results",[])
            if results: return results[0].get("primaryAccession","")
        return ""

    @staticmethod
    def _http_get(url: str, timeout: int = 15) -> Optional[Dict]:
        try:
            ctx = ssl.create_default_context()
            req = urllib.request.Request(url, headers={"User-Agent": "AutoVS-Agent/2.0"})
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                return json.loads(resp.read().decode())
        except Exception: return None

    # -------------------------------------------------------------------------
    # 意图解析
    # -------------------------------------------------------------------------

    def _parse_intent(self, query: str) -> Dict[str, Any]:
        try:
            resp = self.client.chat.completions.create(
                model=self.model, temperature=0.0, max_tokens=1024,
                messages=[
                    {"role": "system", "content": INTENT_PARSE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"用户查询: {query}"},
                    {"role": "system", "content": f"输出JSON:\n{json.dumps(IntentParseResult.model_json_schema(), indent=2, ensure_ascii=False)}"},
                ],
                response_format={"type": "json_object"},
            )
            return IntentParseResult.model_validate(json.loads(resp.choices[0].message.content.strip())).model_dump()
        except Exception: return self._heuristic_parse(query)

    def _heuristic_parse(self, query: str) -> Dict[str, Any]:
        # 匹配完整基因符号: BCL-2, BCL-xl, EGFR, KRAS, CBLB, PGK2 等
        # 支持: 大写字母(2+), 可选连字符+后缀(数字/小写字母)
        gene = re.search(r'(?:^|[一-鿿\s])'
                         r'([A-Z]{2,}'              # 至少2个大写字母开头
                         r'(?:-[0-9A-Za-z]+)?'       # 可选: -2, -xl, -L1 等
                         r'(?:\s*[A-Z]\d+[A-Z])?'    # 可选: T790M 等突变标注
                         r')',
                         query)
        raw_name = gene.group(1) if gene else ""

        # 分离突变标注
        mut_match = re.search(r'([A-Z]\d+[A-Z])', raw_name) if raw_name else None
        if mut_match:
            modification = mut_match.group(1)
            name = raw_name.replace(modification, '').strip()
        else:
            modification = ""
            name = raw_name

        # 如果上面没匹配到，尝试中文靶点名
        if not name:
            cn = re.search(r'(?:寻找|筛选|设计)\s*([一-鿿A-Za-z0-9\-]+?)\s*(?:的)?(?:抑制剂|配体|激动剂)', query)
            if cn: name = cn.group(1).strip()

        domain = ""
        dm = re.search(r'([一-鿿A-Za-z]*(?:结构域|口袋|位点|domain|pocket))', query)
        if dm: domain = dm.group(1)

        cls = ""
        for kws, c in [(['激酶','kinase'],'Kinase'),(['GPCR','受体'],'GPCR'),
                       (['PPI','BH3','凋亡','apoptosis','BCL-2','BCL-xl','Bcl-2'],'PPI'),
                       (['连接酶','ligase','E3','泛素','CBL','cbl'],'E3_ligase')]:
            if any(k.lower() in query.lower() for k in kws): cls = c; break

        return {"target_name": name or query[:60], "uniprot_id": "", "pdb_id": "", "target_class": cls,
                "domain": domain, "modification": modification,
                "description": query, "organism": "Homo sapiens", "key_residues": []}

    # -------------------------------------------------------------------------
    # 降级报告
    # -------------------------------------------------------------------------

    def _fallback_report(self, target_info: dict, error: str) -> Dict[str, Any]:
        name = target_info.get("target_name","Unknown")
        return {
            "target_name": name, "gene_name": "", "uniprot_id": target_info.get("uniprot_id",""),
            "organism": "Homo sapiens", "target_class": target_info.get("target_class",""),
            "binding_site": {"pocket_description":"","volume_angstrom3":"unknown","polarity":"unknown",
                             "flexibility":"unknown","key_residues":[],"center_coordinates":[0,0,0],
                             "suggested_box_size":[20,20,20]},
            "known_ligands": [], "pdb_structures": [],
            "biology_overview": f"靶点: {name}. 用户查询: {target_info.get('description','')}",
            "structural_analysis": "无可用结构数据。请设置 DEEPSEEK_API_KEY 以启用 LLM 深度调研。",
            "known_ligands_sar": "无已知配体数据。",
            "druggability_assessment": "无法评估。",
            "screening_recommendations": f"通用建议: 对接后用PLIP分析, 关注PAINS过滤。",
            "references": [],
            "full_report_text": f"[降级报告] {name}: {target_info.get('description','')}. LLM不可用: {error[:200]}",
            "research_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # 兼容旧接口
    def generate_profile(self, target_info: dict) -> Dict[str, Any]:
        return self.deep_research(target_info.get("description", target_info.get("target_name","")))
    def generate_profile_from_text(self, query: str) -> Dict[str, Any]:
        return self.deep_research(query)


# =============================================================================
# LangGraph节点
# =============================================================================

def target_scout_node(state: dict) -> dict:
    agent = TargetScoutAgent()
    profile = agent.deep_research(
        state.get("target_info", {}).get("description",
            state.get("target_info", {}).get("target_name", ""))
    )
    now = datetime.now(timezone.utc).isoformat()
    return {
        "pipeline_stage": "target_scout",
        "target_profile": profile,
        "updated_at": now,
        "event_log": [f"[{now}] [TargetScout] Research report: {len(profile.get('full_report_text',''))} chars"],
    }
