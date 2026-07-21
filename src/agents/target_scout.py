"""
AutoVS-Agent v2.0: Target Scout Agent (防幻觉 + 真实数据填充版)
=================================================================
核心原则: LLM只做归纳推理, 具体数值由Python工具函数从真实API/文件解析填入。

防幻觉 + 真实数据:
  1. Pydantic Schema: 删除诱导捏造的字段, 新增API数据字段
  2. System Prompt: "以检索为准(Grounding)" + 防幻觉红线
  3. API验证: PDB ID/UniProt ID 输出前逐一验证
  4. 🆕 工具函数: fetch_chembl_activity() + fetch_pdb_ligand_center()
"""

from __future__ import annotations

import json, os, re, ssl, urllib.request, urllib.error, traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field, field_validator


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Pydantic Schema — 真实数据字段                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class VerifiedPDBInfo(BaseModel):
    pdb_id: str
    resolution: Optional[float] = None
    method: str = ""
    has_ligand: bool = False
    title: str = ""
    deposition_year: int = 0
    ligand_ids: List[str] = Field(default_factory=list, description="结合的配体ID列表(HET code)")
    uniprot_mapped: bool = False


class ChemblActivity(BaseModel):
    """从ChEMBL API查询到的真实活性数据。"""
    molecule_chembl_id: str
    standard_type: str           # IC50, Ki, Kd, EC50
    standard_value: float        # 数值
    standard_units: str          # nM, uM, etc.
    target_chembl_id: str
    smiles: str = ""


class BindingSiteAnalysis(BaseModel):
    pocket_type: str = Field(default="unknown")
    pocket_description: str = Field(default="")
    key_residues_text: str = Field(default="")
    docking_box_strategy: str = Field(
        default="待分析",
        description="对接策略描述(文字)。具体坐标来自fetch_pdb_ligand_center()工具函数。"
    )
    structural_flexibility: str = Field(default="unknown")


class KeyMetrics(BaseModel):
    """🆕 结构化关键数据 — 下游StrategyGenerator直接读取, 避免从长文本中"理解"数值。"""
    known_ligand_mw_range: List[float] = Field(default_factory=list, description="已知配体分子量范围 [min, max], 如[150,974]")
    known_ligand_logp_range: List[float] = Field(default_factory=list, description="已知配体LogP范围 [min, max]")
    known_ligand_ic50_range_nm: List[float] = Field(default_factory=list, description="已知配体IC50范围nM [min, max]")
    representative_ligand_mw_max: float = Field(default=0.0, description="代表性最大配体的MW")
    binding_pocket_volume_ang3: str = Field(default="unknown", description="结合口袋体积范围")
    pocket_type: str = Field(default="unknown", description="deep_cleft/shallow_groove/flat_ppi/cryptic")
    recommended_rule_category: str = Field(default="Ro5", description="Ro5/bRo5/custom")
    key_hbond_residues: List[str] = Field(default_factory=list, description="关键氢键残基列表")
    key_hydrophobic_residues: List[str] = Field(default_factory=list, description="关键疏水残基列表")
    selectivity_residues: List[str] = Field(default_factory=list, description="与同源蛋白的差异残基(用于选择性设计)")
    best_pdb_resolution: float = Field(default=99.0, description="最佳PDB分辨率Å")
    has_cocrystal: bool = Field(default=False, description="是否有共晶结构")


class TargetResearchReport(BaseModel):
    """防幻觉版靶点调研报告 — 真实数据由Python工具函数填入。"""

    target_name: str = Field(default="Unknown")
    target_macromolecule_type: str = Field(default="Protein")
    gene_symbol: str = Field(default="")
    uniprot_id: str = Field(default="")
    target_uniprot_id: str = Field(default="")
    target_organism: str = Field(default="Homo sapiens")

    # 🆕 结构化关键数据块
    key_metrics: KeyMetrics = Field(default_factory=KeyMetrics)

    # ── 结构(来自PDB API验证) ──
    verified_pdb_structures: List[VerifiedPDBInfo] = Field(default_factory=list)
    recommended_pdb_for_docking: str = Field(default="")
    docking_center_from_pdb: List[float] = Field(default_factory=list)
    binding_site: BindingSiteAnalysis = Field(default_factory=BindingSiteAnalysis)

    # ── 配体(来自ChEMBL API) ──
    chembl_activities: List[ChemblActivity] = Field(default_factory=list)
    known_ligands_text: str = Field(default="")

    # ── 长篇文本 ──
    biology_overview: str = Field(default="")
    structural_analysis: str = Field(default="")
    druggability_assessment: str = Field(default="")
    screening_strategy: str = Field(default="", description="[已弃用] 策略生成由StrategyGenerator负责, 此处留空")
    references: List[str] = Field(default_factory=list)

    # ── 元信息 ──
    full_report_text: str = Field(default="")
    api_sources: List[str] = Field(default_factory=list)
    verification_log: List[str] = Field(default_factory=list)
    research_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator('target_organism', 'gene_symbol', 'target_macromolecule_type', 'target_name', mode='before')
    @classmethod
    def sanitize_and_default(cls, v, info):
        """内建Validator: 物理拦截 ? / Unknown / N/A / 空, 替换为默认值。"""
        if not isinstance(v, str):
            return v
        cleaned = v.strip()
        if cleaned.lower() in ("?", "unknown", "n/a", "none", "") or cleaned.startswith("?") or cleaned == "???":
            defaults = {"target_organism": "Homo sapiens", "gene_symbol": "N/A",
                        "target_macromolecule_type": "Protein", "target_name": "Unknown Target"}
            return defaults.get(info.field_name, cleaned)
        return cleaned


KeyMetrics.model_rebuild()
TargetResearchReport.model_rebuild()
VerifiedPDBInfo.model_rebuild()
ChemblActivity.model_rebuild()
BindingSiteAnalysis.model_rebuild()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  System Prompt — Grounding + 防幻觉                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

SCOUT_SYSTEM_PROMPT = """\
# 人设: 顶级药企的资深靶点评估专家

你的每一份报告决定数千万美元的筛选方向。请基于用户查询+真实API数据,
产出一份**极其详尽**的靶点深度调研报告。

---

# JSON字段 ↔ 报告章节映射 (严格遵守!)
#  biology_overview       → 第1节: 靶点生物学
#  structural_analysis    → 第2节: 结构生物学深度分析
#  known_ligands_text     → 第3节: 已知配体与构效关系
#  druggability_assessment → 第4节: 可药性评估与研究现状
#
# ⚠️ 你的职责是纯调研! 不要生成筛选策略或漏斗设计! 那是下游StrategyGenerator的工作!
#
# 📝 Markdown格式要求:
#   每个字段文本必须以 "## N. 标题" 作为一级标题开头
#   子节用 "### 子标题" 标记
#   禁止将多个章节合并, 禁止将子标题写成内联文本

---

## 第1节 → 字段 biology_overview (800-1500字)
以 "## 1. 靶点生物学" 开头, 包含:
- 基因/蛋白全称、家族、进化保守性、组织表达分布
- 三维结构: 结构域组成、活性位点、别构口袋、翻译后修饰
- 信号通路位置和上下游关系、生理功能
- 疾病关联: 哪些疾病、患者群体规模、遗传学证据(如GWAS/敲除表型)
- 动物模型和临床前验证数据
- **必须引用UniProt API数据中的真实功能注释**

## 第2节 → 字段 structural_analysis (800-1500字)
以 "## 2. 结构生物学深度分析" 开头, 包含:
- 所有已知PDB结构的详细比较(分辨率/方法/构象)
- 结合口袋精确描述(体积/极性/柔性/可药性评分)
- 关键残基相互作用(氢键/疏水/盐桥/π-π)
- **如果API返回了PDB结构: 必须详细分析每个共晶配体的结合模式**
- **如果API没返回结构: 诚实说明并讨论替代方案(同源建模需说明模板选择依据)**

## 第3节 → 字段 known_ligands_text (800-1500字)
以 "## 3. 已知配体与构效关系" 开头, 包含:
- 已上市药物的完整信息(适应症/疗效/局限性)
- 临床阶段候选化合物(阶段/开发机构/靶点选择性)
- 工具化合物(用途/活性/选择性)
- 基于ChEMBL API真实数据的SAR总结
- 药效团模型和选择性问题
- **必须引用ChEMBL API返回的真实IC50/Ki数据**

## 第4节 → 字段 druggability_assessment (500-1000字)
以 "## 4. 可药性评估与研究现状" 开头, 包含:
- 靶点可药性综合评分(基于口袋特征/已知配体/结构信息)
- 学术研究热度(里程碑发现/历年论文趋势)
- 工业界布局(药企/交易动态/专利分析)
- 临床试验现状(基于ClinicalTrials.gov API数据)
- 新技术趋势(PROTAC/分子胶/共价抑制剂/DEL)
- 未满足的临床需求和药物开发机会

---

# 🚨 防幻觉红线(不变)
1. PDB ID/IC50/XYZ坐标只能来自API数据, 禁止捏造
2. API无结果→诚实写"未检索到", 禁用训练记忆
3. 禁止跨物种/跨蛋白借用结构
4. 所有字段禁止"?"
"""


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  🆕 多段独立生成 — 每个章节一次独立LLM调用, 防止token预算耗尽              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

SECTION_CONFIGS = [
    {
        "field": "biology_overview",
        "title": "靶点生物学",
        "section_num": 1,
        "min_chars": 800,
        "max_chars": 1500,
        "system_prompt": """\
你是顶级药企的靶点生物学专家。基于用户查询和API真实数据,
撰写靶点生物学章节。输出纯Markdown文本(不要JSON包装)。

必须以 "## 1. 靶点生物学" 开头, 子节用 "### 子标题"。
必须覆盖: 基因/蛋白全称与家族、进化保守性、组织表达分布、
三维结构域组成、活性位点、翻译后修饰、信号通路位置和上下游关系、
生理功能、疾病关联与遗传学证据、动物模型与临床前验证。
引用UniProt API数据中的真实功能注释。

字数: 800-1500字。防幻觉: 不捏造数值, 不确定处注明"待实验验证"。
""",
    },
    {
        "field": "structural_analysis",
        "title": "结构生物学深度分析",
        "section_num": 2,
        "min_chars": 800,
        "max_chars": 1500,
        "system_prompt": """\
你是结构生物学专家。基于用户查询和PDB API真实数据,
撰写结构生物学深度分析章节。输出纯Markdown文本(不要JSON包装)。

必须以 "## 2. 结构生物学深度分析" 开头, 子节用 "### 子标题"。
必须覆盖: 所有已知PDB结构详细比较(分辨率/方法/构象)、
结合口袋精确描述(体积/极性/柔性/可药性评分)、
关键残基相互作用(氢键/疏水/盐桥/π-π)。
如果API有PDB结构: 必须详细分析每个共晶配体的结合模式。
如果API无结构: 诚实说明"未检索到实验结构", 讨论同源建模方案。

字数: 800-1500字。防幻觉: PDB ID和坐标只能来自API数据, 禁止捏造。
""",
    },
    {
        "field": "known_ligands_text",
        "title": "已知配体与构效关系",
        "section_num": 3,
        "min_chars": 800,
        "max_chars": 1500,
        "system_prompt": """\
你是药物化学专家。基于用户查询和ChEMBL API真实数据,
撰写已知配体与构效关系章节。输出纯Markdown文本(不要JSON包装)。

必须以 "## 3. 已知配体与构效关系" 开头, 子节用 "### 子标题"。
必须覆盖: 已上市药物完整信息(适应症/疗效/局限性)、
临床阶段候选化合物(阶段/开发机构/靶点选择性)、
工具化合物(用途/活性/选择性)、
基于ChEMBL API真实IC50/Ki数据的SAR总结、
药效团模型和选择性问题。

字数: 800-1500字。防幻觉: IC50/Ki只能来自API数据, 禁止捏造。
""",
    },
    {
        "field": "druggability_assessment",
        "title": "可药性评估与研究现状",
        "section_num": 4,
        "min_chars": 500,
        "max_chars": 1000,
        "system_prompt": """\
你是药物研发战略专家。基于用户查询、API数据和学术/工业界知识,
撰写可药性评估与研究现状章节。输出纯Markdown文本(不要JSON包装)。

必须以 "## 4. 可药性评估与研究现状" 开头, 子节用 "### 子标题"。
必须覆盖: 靶点可药性综合评分(基于口袋特征/已知配体/结构信息)、
学术研究热度(里程碑发现/历年论文趋势)、
工业界布局(药企/交易动态/专利分析)、
临床试验现状(基于ClinicalTrials.gov API数据)、
新技术趋势(PROTAC/分子胶/共价抑制剂/DEL)、
未满足的临床需求和药物开发机会。

字数: 500-1000字。防幻觉: 引用API真实数据, 不确定处标注信息来源。
""",
    },
]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  🆕 工具函数: 从真实API/文件获取数据 (非LLM)                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _http_get(url: str, timeout: int = 15) -> Optional[Dict]:
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(url, headers={"User-Agent": "AutoVS-Agent/2.0"})
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def _http_get_text(url: str, timeout: int = 15) -> Optional[str]:
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(url, headers={"User-Agent": "AutoVS-Agent/2.0"})
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return r.read().decode("utf-8", errors="ignore")
    except Exception:
        return None


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  🆕 实时文献检索: PubMed + Europe PMC (免费, 无需API Key)                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def search_pubmed(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """PubMed实时检索: 搜索文献标题+摘要。使用Entrez API, 免费。"""
    import urllib.parse, xml.etree.ElementTree as ET

    # Step 1: 搜索ID列表
    search_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        f"db=pubmed&retmax={max_results}&sort=relevance&term={urllib.parse.quote(query)}"
        "&retmode=json"
    )
    data = _http_get(search_url)
    if not data:
        return []
    ids = data.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    # Step 2: 获取摘要
    fetch_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
        f"db=pubmed&id={','.join(ids)}&retmode=xml&rettype=abstract"
    )
    xml_text = _http_get_text(fetch_url)
    if not xml_text:
        return []

    # Step 3: 解析XML
    results = []
    try:
        root = ET.fromstring(xml_text)
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID", "")
            title = article.findtext(".//ArticleTitle", "")
            abstract = article.findtext(".//AbstractText", "") or ""
            journal = article.findtext(".//Journal/Title", "")
            year = article.findtext(".//Journal/JournalIssue/PubDate/Year", "")
            doi = ""
            for eid in article.findall(".//ELocationID"):
                if eid.get("EIdType") == "doi":
                    doi = eid.text or ""
                    break
            if title or abstract:
                results.append({
                    "pmid": pmid, "title": title, "abstract": abstract[:600],
                    "journal": journal, "year": year, "doi": doi,
                })
    except Exception:
        pass

    return results[:max_results]


def search_clinical_trials(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """ClinicalTrials.gov API: 搜索临床试验。"""
    import urllib.parse
    url = (
        "https://clinicaltrials.gov/api/v2/studies?"
        f"query.term={urllib.parse.quote(query)}&pageSize={max_results}&format=json"
    )
    data = _http_get(url)
    if not data:
        return []
    results = []
    for study in data.get("studies", [])[:max_results]:
        prot = study.get("protocolSection", {})
        ident = prot.get("identificationModule", {})
        status = prot.get("statusModule", {})
        desc = prot.get("descriptionModule", {})
        results.append({
            "nct_id": ident.get("nctId", ""),
            "title": ident.get("briefTitle", ""),
            "status": status.get("overallStatus", ""),
            "phase": ";".join(status.get("phaseList", []) or []),
            "conditions": ";".join(ident.get("conditionList", []) or []),
            "description": (desc.get("briefSummary", "") or "")[:300],
        })
    return results


def fetch_chembl_activity(chembl_target_id: str) -> List[ChemblActivity]:
    """🆕 通过ChEMBL API查询靶点的真实IC50/Ki/Kd值。

    纯Python实现, 不依赖LLM。
    """
    activities = []
    # 查活性数据: IC50, Ki, Kd (取前20条最高效的)
    url = (f"https://www.ebi.ac.uk/chembl/api/data/activity.json?"
           f"target_chembl_id={chembl_target_id}&limit=20"
           f"&standard_type__in=IC50,Ki,Kd,EC50"
           f"&standard_units__in=nM,uM&standard_relation__in==,<,<=&order_by=standard_value")
    data = _http_get(url)
    if not data:
        return activities

    for a in data.get("activities", []):
        sv = a.get("standard_value")
        if sv is None:
            continue
        try:
            sv = float(sv)
        except (ValueError, TypeError):
            continue
        # 如果单位是uM, 转换为nM
        if a.get("standard_units") == "uM":
            sv = sv * 1000.0
        activities.append(ChemblActivity(
            molecule_chembl_id=a.get("molecule_chembl_id", ""),
            standard_type=a.get("standard_type", ""),
            standard_value=round(sv, 2),
            standard_units="nM",
            target_chembl_id=a.get("target_chembl_id", chembl_target_id),
            smiles=a.get("canonical_smiles", ""),
        ))
    # 去重(同一分子取最有效的值)
    seen = {}
    unique = []
    for a in sorted(activities, key=lambda x: x.standard_value):
        if a.molecule_chembl_id not in seen:
            seen[a.molecule_chembl_id] = True
            unique.append(a.model_dump())
    return unique[:10]  # top 10


def fetch_pdb_ligand_center(pdb_id: str) -> Optional[Dict[str, Any]]:
    """🆕 从真实PDB文件解析配体质心坐标。

    下载PDB文件, 找到HETATM配体, 计算质心XYZ。
    纯Python实现, 不依赖LLM!
    """
    try:
        ctx = ssl.create_default_context()
        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        req = urllib.request.Request(pdb_url, headers={"User-Agent": "AutoVS-Agent/2.0"})
        with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
            pdb_text = r.read().decode("utf-8", errors="ignore")
    except Exception:
        return None

    # 解析HETATM行, 按配体分组
    ligand_atoms: Dict[str, List[Dict]] = {}
    protein_residues = {
        "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
        "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
        "DA","DC","DG","DT","A","C","G","U",
    }
    skip_residues = {"HOH", "H2O", "DOD", "WAT", "NA", "CL", "K", "MG", "CA", "ZN", "SO4", "PO4", "GOL", "EDO", "ACT", "MPD", "PEG"}

    for line in pdb_text.split("\n"):
        if not line.startswith("HETATM"):
            continue
        resn = line[17:20].strip()
        if resn in skip_residues or resn in protein_residues:
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        chain = line[21:22].strip()
        resi = line[22:26].strip()
        key = f"{resn}_{chain}_{resi}"
        if key not in ligand_atoms:
            ligand_atoms[key] = []
        ligand_atoms[key].append({"x": x, "y": y, "z": z, "resn": resn, "chain": chain})

    if not ligand_atoms:
        return None

    # 对每个配体计算质心, 按原子数排序
    ligands = []
    for key, atoms in ligand_atoms.items():
        cx = sum(a["x"] for a in atoms) / len(atoms)
        cy = sum(a["y"] for a in atoms) / len(atoms)
        cz = sum(a["z"] for a in atoms) / len(atoms)
        ligands.append({
            "ligand_key": key,
            "residue_name": atoms[0]["resn"],
            "chain": atoms[0]["chain"],
            "atom_count": len(atoms),
            "center": [round(cx, 2), round(cy, 2), round(cz, 2)],
        })

    # 按原子数降序(通常是最大的配体)
    ligands.sort(key=lambda x: x["atom_count"], reverse=True)
    return {"pdb_id": pdb_id, "ligands_found": len(ligands), "top_ligands": ligands[:5]}


def verify_pdb_id(pdb_id: str) -> Optional[Dict[str, Any]]:
    data = _http_get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}")
    if not data:
        return None
    ei = data.get("rcsb_entry_info", {}) or {}
    res_list = ei.get("resolution_combined") or []
    nbc_raw = ei.get("nonpolymer_bound_components", {}) or {}
    # 提取配体HET codes (nbc可能是dict或list, 都要兼容)
    ligand_ids = []
    if isinstance(nbc_raw, dict):
        for het_group, items in nbc_raw.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        ligand_ids.append(item.get("comp_id", het_group))
    elif isinstance(nbc_raw, list):
        for item in nbc_raw:
            if isinstance(item, dict):
                ligand_ids.append(item.get("comp_id", ""))
            elif isinstance(item, str):
                ligand_ids.append(item)  # e.g. [\"XNV\", \"ADP\"]
    dep_year = (data.get("rcsb_accession_info") or {}).get("deposit_date", "0000")[:4]
    title = (data.get("struct") or {}).get("title", "")
    # 🆕 API经常漏报ligand: 从标题推断是否有配体
    has_lig = bool(ligand_ids)
    if not has_lig and title:
        has_lig = any(kw in title.lower() for kw in
                      ["complex", "inhibitor", "ligand", "bound", "compound", "drug", "agonist", "antagonist", "with"])
    return {
        "resolution": res_list[0] if res_list else None,
        "method": ei.get("experimental_method", ""),
        "has_ligand": has_lig,
        "title": title,
        "deposition_year": int(dep_year) if dep_year.isdigit() else 0,
        "ligand_ids": ligand_ids,
    }


def verify_uniprot_id(uniprot_id: str) -> Optional[Dict[str, Any]]:
    data = _http_get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json")
    if not data:
        return None
    gene = ""
    for gn in data.get("genes", []):
        gene = gn.get("geneName", {}).get("value", ""); break
    func_text = ""
    for c in data.get("comments", []):
        if c.get("commentType") == "FUNCTION":
            func_text = c.get("texts", [{}])[0].get("value", "")[:1000]; break
    return {"protein_name": data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
            "gene_symbol": gene, "organism": data.get("organism", {}).get("scientificName", ""), "function": func_text}


def search_pdb_by_uniprot(uniprot_id: str) -> List[str]:
    """🆕 通过UniProt ID精确检索PDB结构 (首选锚点, 比文字搜索准确100倍)。

    使用RCSB PDB Search API: rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession
    """
    if not uniprot_id:
        return []
    try:
        import urllib.parse
        query = json.dumps({
            "query": {
                "type": "terminal", "service": "text",
                "parameters": {
                    "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                    "operator": "exact_match",
                    "value": uniprot_id,
                },
            },
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": 50}},
        })
        url = "https://search.rcsb.org/rcsbsearch/v2/query?json=" + urllib.parse.quote(query)
        data = _http_get(url)
        return [r.get("identifier", "") for r in data.get("result_set", [])] if data else []
    except Exception:
        return []


def verify_pdb_organism(pdb_id: str) -> Optional[List[str]]:
    """获取PDB结构的源物种名称。通过Data API的polymer_entities解析。"""
    data = _http_get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}")
    if not data: return None
    orgs = set()
    # 从polymer_entities获取
    for eid, ent in (data.get("polymer_entities") or {}).items():
        for src in ent.get("entity_src_gen", []):
            sci = src.get("pdbx_gene_src_scientific_name", "") or src.get("organism_scientific", "")
            if sci: orgs.add(sci)
    # 如果polymer_entities为空, 从struct.title提取
    if not orgs:
        title = (data.get("struct") or {}).get("title", "")
        # 常见病毒/物种名匹配
        for keyword in ["SARS-CoV-2", "SARS-CoV", "MERS-CoV", "HIV-1", "HIV", "HCV",
                        "Hepatitis", "Ebola", "Zika", "Dengue", "Influenza",
                        "Homo sapiens", "Human", "Mus musculus", "Mouse",
                        "Escherichia coli", "Pseudomonas",
                        "Plasmodium", "falciparum", "Mycobacterium", "tuberculosis",
                        "Trypanosoma", "Leishmania", "Candida", "Aspergillus", "Cryptococcus",
                        "Saccharomyces", "Rattus", "Danio", "Drosophila"]:
            if keyword.lower() in title.lower():
                orgs.add(keyword)
        # 如果匹配到了寄生虫/细菌等, 从title中提取完整物种名
        if orgs and "Plasmodium" in title:
            for match in ["Plasmodium falciparum", "Plasmodium vivax", "Plasmodium knowlesi"]:
                if match.lower() in title.lower(): orgs.add(match)
    return sorted(orgs) if orgs else None


def search_rcsb_pdb_rna(target_name: str, expected_organism: str = "", max_results: int = 15) -> List[Dict[str, Any]]:
    """🆕 RNA/DNA靶点的PDB搜索: 使用nucleic acid文本服务。"""
    import urllib.parse
    search_value = f"{target_name} {expected_organism}".strip()
    query_json = json.dumps({
        "query": {
            "type": "group", "logical_operator": "and", "nodes": [
                {"type": "terminal", "service": "full_text", "parameters": {"value": search_value}},
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_entry_info.structure_determination_methodology", "operator": "exists"}},
            ],
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": max_results}},
    })
    data = _http_get("https://search.rcsb.org/rcsbsearch/v2/query?json=" + urllib.parse.quote(query_json))
    if not data: return []
    return [{"pdb_id": r.get("identifier", "")} for r in data.get("result_set", []) if r.get("identifier")]


def search_rcsb_pdb(target_name: str, expected_organism: str = "", max_results: int = 15) -> List[Dict[str, Any]]:
    """🆕 物种感知PDB搜索。物种名作为搜索词加入, 确保结果属于该物种。"""
    import urllib.parse
    # 将物种名和目标名一起作为搜索词
    search_value = f"{target_name} {expected_organism}".strip()
    query_json = json.dumps({
        "query": {"type": "terminal", "service": "full_text", "parameters": {"value": search_value}},
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": max_results}},
    })
    data = _http_get("https://search.rcsb.org/rcsbsearch/v2/query?json=" + urllib.parse.quote(query_json))
    if not data: return []
    results = []
    for r in data.get("result_set", []):
        pid = r.get("identifier", "")
        if pid: results.append({"pdb_id": pid})
    return results


def search_uniprot_by_name(name: str, organism: str = "") -> List[Dict]:
    import urllib.parse
    query_parts = [urllib.parse.quote(name)]
    if organism:
        query_parts.append(f"AND+organism_name:{urllib.parse.quote(organism)}")
    query_str = "+".join(query_parts)
    url = f"https://rest.uniprot.org/uniprotkb/search?query={query_str}&size=10"
    data = _http_get(url)
    if not data:
        return []
    return [{"uniprot_id": r.get("primaryAccession", ""),
             "gene": (r.get("genes", [{}])[0].get("geneName", {}).get("value", "") if r.get("genes") else ""),
             "protein": r.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "")}
            for r in data.get("results", [])]


def search_chembl_target(name: str) -> List[Dict]:
    import urllib.parse
    data = _http_get(f"https://www.ebi.ac.uk/chembl/api/data/target/search.json?q={urllib.parse.quote(name)}&limit=5")
    if not data:
        return []
    return [{"chembl_id": t.get("target_chembl_id", ""), "pref_name": t.get("pref_name", ""),
             "organism": t.get("organism", ""), "target_type": t.get("target_type", "")}
            for t in data.get("targets", [])[:3]]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TargetScoutAgent                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  🆕 后处理清洗器: 物理删除所有LLM走私的坐标数值                               ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝

_COORD_REPLACEMENT = "[COORDINATES REDACTED — must be computed by downstream tools from real PDB files]"

COORD_PATTERNS = [
    # [x.xx, y.yy, z.zz] 或 (x.xx, y.yy, z.zz)
    (re.compile(r'\[?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]?'), _COORD_REPLACEMENT),
    # x=X.XX, y=Y.YY, z=Z.ZZ 或 center_x=X.XX ...
    (re.compile(r'(?:center_?)?[xyz]\s*=\s*-?\d+\.?\d*\s*[,\s]*', re.IGNORECASE), ''),
    # centered at [...] / center at [...]
    (re.compile(r'(?:cent(?:er|red)\s*at)\s*\[?\s*-?\d+\.?\d*\s*,?\s*-?\d+\.?\d*\s*,?\s*-?\d+\.?\d*\s*\]?', re.IGNORECASE), f'centered at {_COORD_REPLACEMENT}'),
    # box size of XX[.]X Å / grid box of XX Å³
    (re.compile(r'(?:box|grid)\s*(?:size|dimension)s?\s*(?:of|is|=|:)?\s*\(?\s*-?\d+\.?\d*\s*[,×x]\s*-?\d+\.?\d*\s*[,×x]\s*-?\d+\.?\d*\s*(?:Å|A|angstrom)?\s*\)?', re.IGNORECASE), f'box/grid dimensions: {_COORD_REPLACEMENT}'),
    # Nx × Ny × Nz Å³ box
    (re.compile(r'-?\d+\.?\d*\s*[×x]\s*-?\d+\.?\d*\s*[×x]\s*-?\d+\.?\d*\s*(?:Å|A|angstrom)?\s*(?:³|3|box|cube)', re.IGNORECASE), _COORD_REPLACEMENT),
    # exhaustiveness = N (specific docking param smuggling)
    # Don't scrub exhaustiveness — it's acceptable as a recommendation
    # num_modes = N — also acceptable
    # --center_x X --center_y Y --center_z Z
    (re.compile(r'--center_[xyz]\s+-?\d+\.?\d*', re.IGNORECASE), '--center_[XYZ] REDACTED'),
    # --size_x X --size_y Y --size_z Z
    (re.compile(r'--size_[xyz]\s+\d+\.?\d*', re.IGNORECASE), '--size_[XYZ] REDACTED'),
]


def scrub_hallucinated_coordinates(text: str) -> str:
    """🆕 后处理清洗器: 物理删除所有LLM可能走私的坐标/盒子数值。

    不依赖Prompt约束, 直接在输出层用正则一刀切。
    对所有长文本字段(structure_analysis, screening_strategy等)执行。
    """
    if not text or not isinstance(text, str):
        return text

    cleaned = text
    for pattern, replacement in COORD_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)

    # 清理连续多个替换产生的冗余空格
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    cleaned = re.sub(r',\s*,', ',', cleaned)

    return cleaned


def scrub_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """🆕 清洗: 坐标 + PDB ID (API空时) + 应用默认值。"""
    text_fields = ["biology_overview", "structural_analysis", "druggability_assessment",
                   "screening_strategy", "known_ligands_text", "full_report_text"]

    # 🆕 如果API没查到任何结构, 物理删除所有走私的PDB ID
    has_pdb_results = bool(report.get("verified_pdb_structures", []))
    if not has_pdb_results:
        # 匹配 PDB XXXX (如 PDB 1YCR, pdb 4D1P)
        pdb_id_pattern = re.compile(
            r'\b(?:PDB|pdb)\s*:?\s*[0-9][A-Za-z0-9]{3}\b|'   # PDB: 1YCR
            r'\b[0-9][A-Za-z0-9]{3}\s*\((?:PDB|pdb)\)|'        # 1YCR(PDB)
            r'\bPDB\s+(?:entry|code|ID)\s+[0-9][A-Za-z0-9]{3}\b'  # PDB entry 1YCR
        )
        for field in text_fields:
            if field in report and isinstance(report[field], str):
                report[field] = pdb_id_pattern.sub(
                    "[PDB ID REDACTED — API returned no verified structures]", report[field]
                )

    for field in text_fields:
        if field in report and isinstance(report[field], str):
            report[field] = scrub_hallucinated_coordinates(report[field])
    bs = report.get("binding_site", {})
    if isinstance(bs, dict):
        for k in ("pocket_description", "key_residues_text", "docking_box_strategy"):
            if k in bs and isinstance(bs[k], str):
                bs[k] = scrub_hallucinated_coordinates(bs[k])
    return report


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TargetScoutAgent                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TargetScoutAgent:
    def __init__(self, model="deepseek-chat", api_key=None, api_base=None, temperature=0.2, max_tokens=16384):
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

    # =========================================================================
    # 主入口
    # =========================================================================

    def deep_research(self, query: str, *, fetch_structure_coordinates: bool = True) -> Dict[str, Any]:
        log = [f"Query: {query}"]
        api_sources = []

        # ── Step 1: LLM意图解析 ──
        intent = self._parse_intent(query)
        gene = intent.get("target_name", "")
        is_viral = intent.get("is_viral", False)
        is_pathogen = intent.get("is_pathogen", False)
        intent_organism = intent.get("organism", "")
        log.append(f"Intent: gene={gene}, organism={intent_organism or '?'}, viral={is_viral}, pathogen={is_pathogen}")

        # 🆕 非人类靶点全面检测: 防止细菌/病毒/真菌/寄生虫/模式生物错误默认Human
        # 核心逻辑: query中出现了任何非人类生物指示词 → 不强制Human, 让API自由搜索
        _query_lower = query.lower()
        _non_human_indicators = [
            # ── 细菌 ──
            "细菌", "bacterial", "dna旋转酶", "dna gyrase", "gyrase",
            "微生物", "microorganism", "microbial",
            "分枝杆菌", "mycobacterium", "结核", "tuberculosis", "mtb",
            "大肠杆菌", "ecoli", "e. coli", "枯草", "subtilis", "bacillus",
            "金黄色葡萄球菌", "aureus", "staph", "铜绿假单胞菌", "pseudomonas",
            "淋病", "gonorrhoeae", "沙门", "salmonella", "霍乱", "cholera",
            "肺炎链球菌", "pneumoniae", "链霉菌", "streptomyces",
            "抗菌", "antibacterial", "antibiotic", "耐药菌", "革兰",
            "幽门", "pylori", "弯曲", "campylobacter", "志贺", "shigella",
            "鲍曼", "baumannii", "克雷伯", "klebsiella", "艰难", "difficile",
            # ── 病毒 ──
            "病毒", "viral", "hiv", "hcv", "hbv", "sars", "cov-2", "covid",
            "流感", "influenza", "埃博拉", "ebola", "ebov", "寨卡", "zika",
            "登革", "dengue", "疱疹", "herpes", "腺病毒", "adenovirus",
            "乙肝", "丙肝", "cmv", "ebv", "rsv", "抗病毒", "antiviral",
            # ── 真菌 ──
            "真菌", "fungal", "fungus", "candida", "念珠菌",
            "aspergillus", "曲霉", "cryptococcus", "隐球菌",
            "抗真菌", "antifungal", "酵母", "yeast", "saccharomyces",
            "白色念珠", "pneumocystis", "组织胞浆", "histoplasma",
            # ── 寄生虫 ──
            "寄生虫", "parasitic", "parasite", "plasmodium", "疟原虫",
            "malaria", "疟疾", "trypanosoma", "锥虫", "leishmania", "利什曼",
            "弓形虫", "toxoplasma", "隐孢子", "cryptosporidium",
            "血吸虫", "schistosoma", "丝虫", "filarial",
            # ── 模式生物 (非人类) ──
            "小鼠", "mouse", "大鼠", "rat", "斑马鱼", "zebrafish",
            "果蝇", "drosophila", "线虫", "elegans", "拟南芥", "arabidopsis",
            # ── 通用非人类指示词 ──
            "病原", "pathogen", "致病", "感染", "infectious",
            "抗生素", "抗菌", "杀菌", "bactericid",
        ]
        _is_non_human = any(kw in _query_lower for kw in _non_human_indicators)
        if _is_non_human and not is_viral and not is_pathogen and not intent_organism:
            # 标记为非人类靶点: 防止 force_human 错误地将物种设为 Homo sapiens
            # 但保留 intent_organism 为空, 让API搜索时不做物种限制
            is_pathogen = True
            log.append(f"Non-human target detected from query → preventing Human default")

        # 🆕 PROTAC/药物开发上下文: 用户未指定物种时默认人类
        # ⚠️ 仅当查询不含非人类指示词时才生效, 避免细菌/病毒靶点被错误设为Human
        _query_lower = query.lower()
        _drug_dev_keywords = ["protac", "分子胶", "降解剂", "molecular glue", "degron",
                              "protac", "药物筛选", "先导化合物", "lead compound",
                              "药物发现", "drug discovery", "临床候选"]
        _is_drug_dev = any(kw in _query_lower for kw in _drug_dev_keywords)
        if _is_drug_dev and not intent_organism and not is_viral and not is_pathogen:
            intent_organism = "Homo sapiens"
            log.append(f"PROTAC/drug dev context detected → defaulting organism to Homo sapiens")

        # 🆕 多靶点查询检测: 检测用户是否同时对多个靶点提出要求
        _multi_target_keywords = [
            r'和.*受体', r'与.*酶', r'同时.*和', r'双靶', r'多靶',
            r'和\s*[A-Z][A-Z0-9-]+', r'与\s*[A-Z][A-Z0-9-]+',
            r'\+.*[A-Z][A-Z0-9-]+', r'and\s+[A-Z][A-Z0-9-]+',
            r'同时激活.*抑制', r'同时抑制.*激活',
        ]
        _is_multi_target = any(re.search(p, query) for p in _multi_target_keywords)
        # 额外检查: 用简单规则数目标名 (大写字母+数字组合出现 >1次)
        _gene_like = re.findall(r'\b[A-Z][A-Z0-9]{2,}(?:\s*[和,、+&/]\s*[A-Z][A-Z0-9]{2,})+\b', query)
        if _is_multi_target or _gene_like:
            log.append(f"⚠️ 多靶点查询检测: 当前系统仅深度调研主靶点({gene}), 其他靶点未被覆盖。"
                       f"建议对每个靶点分别运行调研。")

        # ── Step 2: UniProt搜索+验证 (🆕 仅蛋白质靶点) ──
        uniprot_data, uniprot_id, gene_symbol = None, "", ""
        mol_type = intent.get("macromolecule_type", "Protein")
        is_nucleic = mol_type in ("RNA", "DNA")
        if gene and not is_nucleic:
            # 🆕 短基因符号(≤2字符)不可靠, 用query中的英文全名替代
            search_gene = gene
            if len(gene.strip()) <= 2:
                # 从query中提取可能的英文全名或中文关键词翻译
                cn_to_en = {"雄激素受体": "androgen receptor", "雌激素受体": "estrogen receptor",
                            "糖皮质激素受体": "glucocorticoid receptor", "孕激素受体": "progesterone receptor",
                            "雄激素": "androgen receptor", "AR受体": "androgen receptor"}
                expanded = query
                for cn, en in cn_to_en.items():
                    if cn in query: expanded = en; break
                if expanded == query and "受体" in query:
                    expanded = query  # 无法翻译, 尝试用原query
                search_gene = expanded
                log.append(f"Short gene '{gene}', expanding search to '{search_gene[:60]}'")
            # 短基因符号(≤2)或没有明确病原体信息: 强制人类organism_id:9606过滤
            force_human = not is_viral and not is_pathogen and not intent_organism
            if is_viral or is_pathogen:
                candidates = self._search_uniprot_global(search_gene)
            elif force_human or len(gene.strip()) <= 2:
                candidates = search_uniprot_by_name(search_gene, organism=intent_organism)
            else:
                candidates = search_uniprot_by_name(search_gene, organism=intent_organism)
            if candidates:
                best = candidates[0]
                # 🆕 物种感知: query含病原体关键词→优先选非Human候选
                pathogen_genera = {"结核":"mycobacterium","分枝杆菌":"mycobacterium",
                    "plasmodium":"plasmodium","malaria":"plasmodium","ecoli":"escherichia"}
                for kw, genus in pathogen_genera.items():
                    if kw in query.lower():
                        for cand in candidates:
                            if genus in (cand.get("protein","")+cand.get("organism","")).lower():
                                best = cand; break
                        break
                # 短基因名验证
                found_gene = (best.get("gene") or "").upper()
                if found_gene and gene.upper() not in ("", "AR", found_gene) and len(gene) <= 2:
                    # 短基因符号可能匹配错误, 用全名重搜
                    fallback = search_uniprot_by_name(query if len(query) > 10 else f"{query} receptor")
                    if fallback and (fallback[0].get("gene") or "").upper() == gene.upper():
                        best = fallback[0]
                        log.append(f"Fallback search corrected: {best.get('gene')}")
                uniprot_id = best.get("uniprot_id", "")
                gene_symbol = best.get("gene") or gene
                uniprot_data = verify_uniprot_id(uniprot_id) if uniprot_id else None
                if uniprot_data:
                    api_sources.append(f"UniProt:{uniprot_id}(verified)")
                    log.append(f"UniProt: {uniprot_id} → {gene_symbol}")
                    # 🆕 基因符号一致性校验: UniProt返回的基因≠意图解析的→用原始query重搜
                    found_gene = (uniprot_data.get("gene_symbol") or "").upper()
                    if found_gene and gene.upper() != found_gene:
                        log.append(f"⚠️ Gene mismatch: intent={gene}, UniProt={found_gene}. Retrying with query.")
                        retry_candidates = self._search_uniprot_global(query)
                        if retry_candidates and retry_candidates[0].get("gene","").upper() == gene.upper():
                            retry_id = retry_candidates[0].get("uniprot_id","")
                            retry_data = verify_uniprot_id(retry_id)
                            if retry_data:
                                uniprot_id, gene_symbol, uniprot_data = retry_id, gene, retry_data
                                log.append(f"  ✅ Corrected: {uniprot_id} → {gene_symbol}")
        elif is_nucleic:
            log.append(f"RNA/DNA target: skipping UniProt, using nucleic acid PDB search")

        # ── Step 3: PDB搜索+验证 (🆕 物种感知 + 核酸分支) ──
        verified_pdbs, recommended_pdb, docking_center = [], "", []
        rejected_pdbs = []  # 🆕 被物种校验拒绝的PDB
        expected_organism = ""  # 初始化
        search_term = gene_symbol or gene
        if search_term:
            # 🆕 从UniProt数据或query中确定真实物种 (绝不默认Homo sapiens!)
            expected_organism = ""
            if uniprot_data and uniprot_data.get("organism"):
                expected_organism = uniprot_data["organism"]  # UniProt API是权威来源
            elif intent_organism:
                expected_organism = intent_organism  # LLM意图解析的物种
            elif is_viral or is_pathogen:
                for vname in ["SARS-CoV-2", "SARS-CoV", "MERS-CoV", "HIV-1", "HIV", "HCV", "HBV", "EBOV", "ZIKV", "DENV",
                              "HIV-1", "HIV", "Human immunodeficiency virus"]:
                    if vname.lower() in query.lower():
                        expected_organism = vname; break
            if not expected_organism:
                # 从query中搜索已知非人类物种关键词
                species_keywords = {
                    "malaria": "Plasmodium falciparum",
                    "plasmodium falciparum": "Plasmodium falciparum",
                    "plasmodium": "Plasmodium falciparum",
                    "pfalciparum": "Plasmodium falciparum",
                    "P. falciparum": "Plasmodium falciparum",
                    "tuberculosis": "Mycobacterium tuberculosis",
                    "mycobacterium": "Mycobacterium tuberculosis",
                    "ecoli": "Escherichia coli",
                    "pseudomonas": "Pseudomonas aeruginosa",
                    "influenza": "Influenza A virus",
                    "trypanosoma": "Trypanosoma brucei",
                    "leishmania": "Leishmania major",
                    "candida albicans": "Candida albicans",
                    "candida": "Candida albicans",
                    "aspergillus": "Aspergillus fumigatus",
                    "cryptococcus": "Cryptococcus neoformans",
                    "mouse": "Mus musculus",
                    "rat": "Rattus norvegicus",
                    "yeast": "Saccharomyces cerevisiae",
                    "zebrafish": "Danio rerio",
                    "drosophila": "Drosophila melanogaster",
                }
                for kw, species in species_keywords.items():
                    if kw.lower() in query.lower() or kw.lower() in search_term.lower():
                        expected_organism = species; break
            if not expected_organism:
                expected_organism = "Homo sapiens"  # 最后的fallback

            log.append(f"PDB search: term={search_term}, organism={expected_organism}, UniProt={uniprot_id}, type={mol_type}")

            # 🆕 双轨制检索 (蛋白质) / 核酸检索 (RNA/DNA)
            all_pdb_ids = set()
            uniprot_pdb_ids = set()

            if is_nucleic:
                # ── 核酸靶点: 跳过UniProt, 直接用核酸文本搜索 ──
                na_results = search_rcsb_pdb_rna(search_term, expected_organism)
                uniprot_pdb_ids = {r["pdb_id"] for r in na_results if r.get("pdb_id")}
                all_pdb_ids.update(uniprot_pdb_ids)
                log.append(f"  Track-NA(nucleic): '{search_term}' → {len(uniprot_pdb_ids)} structures")

            # ── 轨道1: UniProt ID精确检索 (仅蛋白质) ──
            if uniprot_id and not is_nucleic:
                uniprot_pdbs = search_pdb_by_uniprot(uniprot_id)
                uniprot_pdb_ids = set(uniprot_pdbs)
                all_pdb_ids.update(uniprot_pdb_ids)
                log.append(f"  Track1(UniProt:{uniprot_id}): {len(uniprot_pdb_ids)} structures")

            # ── 轨道2: 文字搜索 (非核酸的fallback/补充) ──
            base_name = search_term  # 初始化, 核酸靶点时也需此变量
            if not is_nucleic:
                import re as _re
                mutation = intent.get("modification", "")
                base_name = search_term
                if mutation:
                    base_name = _re.sub(r'\s*' + _re.escape(mutation) + r'\s*', ' ', search_term).strip()
                for extra in ["C481S", "T790M", "G12D", "C797S", "T315I", "V600E", "L858R"]:
                    base_name = _re.sub(r'\s*' + extra + r'\s*', ' ', base_name, flags=_re.IGNORECASE).strip()
                synonym_map = {"DHFR":"DHFR-TS dihydrofolate reductase","BTK":"Bruton's tyrosine kinase BTK",
                               "EGFR":"epidermal growth factor receptor EGFR","BCL2":"B-cell lymphoma 2 Bcl-2"}
                tier_searches = [(base_name, "Track2-R2"), (synonym_map.get(base_name.upper(), base_name), "Track2-R3")]
                for st, label in tier_searches:
                    results = search_rcsb_pdb(st, expected_organism)
                    new_ids = {r["pdb_id"] for r in results if r.get("pdb_id")}
                    all_pdb_ids.update(new_ids)
                    if new_ids: log.append(f"  {label}: '{st}' → +{len(new_ids)}")
            log.append(f"  Total PDB candidates: {len(all_pdb_ids)} (Track1+Track2{'/NA' if is_nucleic else ''})")

            # 🆕 动态标题关键词: 多层生成, 杜绝 gyrB→GYRB 匹配不到 Gyrase 的问题
            protein_keywords = set()
            for src in [gene_symbol, gene, base_name]:
                if not src:
                    continue
                upper = src.upper().replace("-"," ").replace("_"," ")
                # 1. 单词级
                for w in upper.split():
                    if len(w) >= 3:
                        protein_keywords.add(w)
                        # N-1前缀 (GYRB→GYR, CDK9→CDK)
                        for i in range(3, len(w)):
                            protein_keywords.add(w[:i])
                # 2. 连字符/下划线变体
                protein_keywords.add(src.upper().replace("-","_"))
                protein_keywords.add(src.upper().replace("_","-"))
            # 3. 从UniProt蛋白全名提取关键词 (最重要!)
            if uniprot_data and uniprot_data.get("protein_name"):
                pname = uniprot_data["protein_name"].upper()
                for w in pname.replace(","," ").replace(";"," ").replace("-"," ").split():
                    if len(w) >= 4:
                        protein_keywords.add(w)
            # 4. 通用生物学术语扩展 — 常见基因→蛋白名映射 (覆盖主流靶点家族)
            _common_expansions = {
                "GYRB": ["GYRASE"], "GYRA": ["GYRASE"], "PARC": ["TOPOISOMERASE"],
                "PARE": ["TOPOISOMERASE"], "TOP2": ["TOPOISOMERASE"],
                "INHA": ["ENOYL", "REDUCTASE", "FASII"], "KATG": ["CATALASE"],
                "FTSZ": ["TUBULIN"], "DHFR": ["DIHYDROFOLATE"],
                "DHPS": ["DIHYDROPTEROATE"], "FABI": ["ENOYL"],
                "DPRE1": ["DECAPRENYL"], "MMPL3": ["MYCOBACTERIAL"],
                "RNAP": ["POLYMERASE"], "RPOA": ["POLYMERASE"],
                "RPOB": ["POLYMERASE"], "RPOC": ["POLYMERASE"],
                "ABL": ["TYROSINE", "KINASE"], "EGFR": ["EPIDERMAL", "GROWTH", "FACTOR"],
                "ALK": ["LYMPHOMA", "KINASE"], "BTK": ["BRUTON", "TYROSINE"],
                "JAK1": ["JANUS", "KINASE"], "JAK2": ["JANUS", "KINASE"],
                "JAK3": ["JANUS", "KINASE"], "TYK2": ["TYROSINE", "KINASE"],
                "BRAF": ["SERINE", "THREONINE"], "KRAS": ["GTPASE"],
                "NRAS": ["GTPASE"], "HRAS": ["GTPASE"],
                "PIK3CA": ["PHOSPHATIDYLINOSITOL", "KINASE"], "MTOR": ["RAPAMYCIN"],
                "AKT1": ["KINASE"], "AKT2": ["KINASE"], "AKT3": ["KINASE"],
                "MEK": ["MAP", "KINASE"], "ERK": ["MAP", "KINASE"],
                "CDK1": ["CYCLIN", "DEPENDENT", "KINASE"], "CDK2": ["CYCLIN", "DEPENDENT", "KINASE"],
                "CDK4": ["CYCLIN", "DEPENDENT", "KINASE"], "CDK6": ["CYCLIN", "DEPENDENT", "KINASE"],
                "CDK7": ["CYCLIN", "DEPENDENT", "KINASE"], "CDK9": ["CYCLIN", "DEPENDENT", "KINASE"],
                "AURKA": ["AURORA", "KINASE"], "AURKB": ["AURORA", "KINASE"],
                "PLK1": ["POLO", "LIKE", "KINASE"], "PLK4": ["POLO", "LIKE", "KINASE"],
                "WEE1": ["KINASE"], "CHK1": ["CHECKPOINT", "KINASE"],
                "CHEK1": ["CHECKPOINT", "KINASE"], "CHEK2": ["CHECKPOINT", "KINASE"],
                "BCL2": ["LYMPHOMA", "APOPTOSIS"], "BCLXL": ["LYMPHOMA", "APOPTOSIS"],
                "MCL1": ["MYELOID", "LEUKEMIA"], "BAX": ["APOPTOSIS"],
                "BAK": ["APOPTOSIS"], "BIM": ["APOPTOSIS"],
                "MDM2": ["P53", "UBIQUITIN"], "TP53": ["P53", "TUMOR"],
                "BRD2": ["BROMODOMAIN"], "BRD3": ["BROMODOMAIN"], "BRD4": ["BROMODOMAIN"],
                "BRDT": ["BROMODOMAIN"], "BRD9": ["BROMODOMAIN"],
                "EP300": ["ACETYLTRANSFERASE"], "CREBBP": ["ACETYLTRANSFERASE"],
                "HDAC1": ["DEACETYLASE"], "HDAC2": ["DEACETYLASE"],
                "HDAC3": ["DEACETYLASE"], "HDAC6": ["DEACETYLASE"],
                "HDAC8": ["DEACETYLASE"], "SIRT1": ["SIRTUIN"],
                "SIRT2": ["SIRTUIN"], "SIRT3": ["SIRTUIN"],
                "DNMT1": ["METHYLTRANSFERASE"], "DNMT3A": ["METHYLTRANSFERASE"],
                "DNMT3B": ["METHYLTRANSFERASE"], "EZH2": ["METHYLTRANSFERASE"],
                "PRMT1": ["METHYLTRANSFERASE"], "PRMT5": ["METHYLTRANSFERASE"],
                "IDH1": ["ISOCITRATE", "DEHYDROGENASE"], "IDH2": ["ISOCITRATE", "DEHYDROGENASE"],
                "PARP1": ["POLY", "ADP", "RIBOSE"], "PARP2": ["POLY", "ADP", "RIBOSE"],
                "CNR2": ["CANNABINOID"], "CNR1": ["CANNABINOID"],
                "NR4A2": ["NUCLEAR", "RECEPTOR", "NURR1"], "PPARG": ["PEROXISOME", "PROLIFERATOR"],
                "ESR1": ["ESTROGEN", "RECEPTOR"], "ESR2": ["ESTROGEN", "RECEPTOR"],
                "AR": ["ANDROGEN", "RECEPTOR"], "PGR": ["PROGESTERONE", "RECEPTOR"],
                "NR3C1": ["GLUCOCORTICOID", "RECEPTOR"],
                "VDR": ["VITAMIN", "RECEPTOR"], "RARA": ["RETINOIC", "ACID"],
                "RARB": ["RETINOIC", "ACID"], "RARG": ["RETINOIC", "ACID"],
                "FXR": ["FARNESOID"], "LXRA": ["LIVER"],
                "PPARA": ["PEROXISOME"], "PPARD": ["PEROXISOME"],
                "PROTEASE": ["PROTEASE", "RETROVIRAL"], "REVERSE": ["TRANSCRIPTASE", "POLYMERASE"],
                "INTEGRASE": ["INTEGRASE", "RETROVIRAL"], "NS3": ["PROTEASE", "HELICASE"],
                "NS5A": ["HEPATITIS"], "NS5B": ["POLYMERASE", "HEPATITIS"],
                "M PRO": ["PROTEASE", "CORONAVIRUS"], "PL PRO": ["PROTEASE", "CORONAVIRUS"],
                "RDRP": ["POLYMERASE", "RNA"], "SPIKE": ["GLYCOPROTEIN", "RECEPTOR"],
                "CASPASE": ["CASPASE", "APOPTOSIS"], "CASP3": ["CASPASE"],
                "CASP6": ["CASPASE"], "CASP7": ["CASPASE"], "CASP8": ["CASPASE"],
                "CASP9": ["CASPASE"], "BAX": ["APOPTOSIS"],
                "BAK": ["APOPTOSIS"], "BID": ["APOPTOSIS"],
                "XIAP": ["APOPTOSIS", "INHIBITOR"], "CIAP": ["APOPTOSIS", "INHIBITOR"],
                "SMAC": ["DIABLO"], "DIABLO": ["APOPTOSIS"],
                "GLUT1": ["GLUCOSE", "TRANSPORTER"], "SGLT2": ["GLUCOSE", "TRANSPORTER"],
                "SLC6A3": ["DOPAMINE", "TRANSPORTER"], "SLC6A4": ["SEROTONIN", "TRANSPORTER"],
                "DRD2": ["DOPAMINE", "RECEPTOR"], "HTR2A": ["SEROTONIN", "RECEPTOR"],
                "ACHE": ["ACETYLCHOLINESTERASE", "CHOLINESTERASE"],
                "BCHE": ["BUTYRYLCHOLINESTERASE", "CHOLINESTERASE"],
                "COX1": ["CYCLOOXYGENASE", "PROSTAGLANDIN"], "COX2": ["CYCLOOXYGENASE", "PROSTAGLANDIN"],
                "PTGS1": ["CYCLOOXYGENASE"], "PTGS2": ["CYCLOOXYGENASE"],
                "MAOB": ["MONOAMINE", "OXIDASE"], "MAOA": ["MONOAMINE", "OXIDASE"],
                "COMT": ["METHYLTRANSFERASE", "CATECHOL"],
                "TH": ["TYROSINE", "HYDROXYLASE"],
                "DDC": ["DECARBOXYLASE", "DOPA"],
                "ADA": ["DEAMINASE", "ADENOSINE"],
                "HMGCR": ["REDUCTASE", "MEVALONATE"],
                "ACE": ["CONVERTING", "ANGIOTENSIN"], "ACE2": ["CONVERTING", "ANGIOTENSIN"],
                "RENIN": ["RENIN", "ANGIOTENSIN"],
                "PDE3": ["PHOSPHODIESTERASE"], "PDE4": ["PHOSPHODIESTERASE"],
                "PDE5": ["PHOSPHODIESTERASE"], "PDE10": ["PHOSPHODIESTERASE"],
                "CFTR": ["FIBROSIS", "TRANSMEMBRANE", "CONDUCTANCE"],
                "VHL": ["HIPPEL", "LINDAU", "UBIQUITIN", "LIGASE"],
                "CRBN": ["CEREBLON", "UBIQUITIN", "LIGASE"],
                "KEAP1": ["UBIQUITIN", "NUCLEAR", "FACTOR"],
                "NRF2": ["NUCLEAR", "FACTOR", "ERYTHROID"],
                "HIF1A": ["HYPOXIA", "INDUCIBLE", "FACTOR"],
                "STAT3": ["SIGNAL", "TRANSDUCER", "TRANSCRIPTION"],
                "NFKB": ["NUCLEAR", "FACTOR", "KAPPA"],
                "MYC": ["MYELOCYTOMATOSIS", "ONCOGENE"],
            }
            for kw in list(protein_keywords):
                for variant in _common_expansions.get(kw, []):
                    protein_keywords.add(variant)
            protein_keywords = [k for k in protein_keywords if k not in {"AND","THE","FOR","WITH","WT","WILD","TYPE","COMPLEX","STRUCTURE","CRYSTAL"}]

            # Track1优先排序, 总上限50 (P10415有161个PDB, 不需要全部)
            sorted_pdbs = sorted(all_pdb_ids,
                key=lambda p: (0 if p in uniprot_pdb_ids else 1, p))[:50]
            for pid in sorted_pdbs:
                meta = verify_pdb_id(pid)
                if not meta: continue
                title = (meta.get("title","") or "").upper().replace("_","-").replace("−","-")

                # 🆕 Track1(UniProt ID)结果100%可靠, 不受任何过滤
                from_track1 = pid in uniprot_pdb_ids if uniprot_id else False
                if not from_track1:
                    # Track2结果: 标题必须包含至少一个靶点关键词
                    title_match = any(kw in title for kw in protein_keywords)
                    if not title_match:
                        log.append(f"  ⚠️ SKIPPED {pid}: title lacks target keywords — {meta.get('title','')[:80]}")
                        continue

                # 物种校验: 容错匹配 (处理同物异名, 如 Mycobacterium vs Mycolicibacterium)
                orgs = verify_pdb_organism(pid)
                if orgs and expected_organism:
                    match = False
                    exp_lower = expected_organism.lower()
                    # 拆分物种名 (属名+种名) 用于部分匹配
                    exp_words = set(exp_lower.replace("."," ").split())
                    # 已知属名同义词 (Mycobacterium→Mycolicibacterium, 等)
                    _genus_synonyms = {
                        "mycobacterium": "mycolicibacterium",
                        "mycolicibacterium": "mycobacterium",
                    }
                    exp_words_expanded = set(exp_words)
                    for w in list(exp_words):
                        if w in _genus_synonyms:
                            exp_words_expanded.add(_genus_synonyms[w])
                    for o in orgs:
                        o_lower = o.lower().replace("."," ")
                        o_words = set(o_lower.split())
                        # 规则1: 完整包含
                        if exp_lower in o_lower or o_lower in exp_lower:
                            match = True; break
                        # 规则2: Homo sapiens ↔ human
                        if "homo sapiens" in exp_lower and "human" in o_lower:
                            match = True; break
                        # 规则3: 共享至少一个非通用词 (种名级别)
                        o_words_expanded = set(o_words)
                        for w in list(o_words):
                            if w in _genus_synonyms:
                                o_words_expanded.add(_genus_synonyms[w])
                        common = exp_words_expanded & o_words_expanded
                        common -= {"homo", "sapiens", "human", "mus", "musculus",
                                   "mouse", "sp", "spp", "strain", "isolate"}
                        if common:
                            match = True; break
                    if not match:
                        rejected_pdbs.append(f"{pid}(orgs={orgs})")
                        continue
                    verified_pdbs.append(VerifiedPDBInfo(
                        pdb_id=pid, resolution=meta.get("resolution"),
                        method=meta.get("method", ""), has_ligand=meta.get("has_ligand", False),
                        title=meta.get("title", ""), deposition_year=meta.get("deposition_year", 0),
                        ligand_ids=meta.get("ligand_ids", []), uniprot_mapped=from_track1,
                    ).model_dump())

            if rejected_pdbs:
                log.append(f"REJECTED(wrong organism): {rejected_pdbs}")

            # 🆕 选推荐PDB: 优先级 (最新+有配体+高分辨率)
            if verified_pdbs:
                with_ligand = [p for p in verified_pdbs if p.get("has_ligand")]
                candidates_pdb = with_ligand if with_ligand else verified_pdbs
                # 按年份降序排序
                candidates_pdb.sort(key=lambda x: (x.get("deposition_year", 0), -(x.get("resolution") or 99)), reverse=True)
                best_pdb = candidates_pdb[0]
                recommended_pdb = best_pdb["pdb_id"]
                api_sources.append(f"PDB:{len(verified_pdbs)}_verified")
                log.append(f"PDB: {len(verified_pdbs)} verified, recommended={recommended_pdb}")

                # 🆕 从真实PDB文件解析配体质心坐标
                if fetch_structure_coordinates:
                    center_data = fetch_pdb_ligand_center(recommended_pdb)
                    if center_data and center_data.get("top_ligands"):
                        top_lig = center_data["top_ligands"][0]
                        docking_center = top_lig["center"]
                        log.append(f"PDB ligand center({recommended_pdb}): {docking_center}")
                        api_sources.append(f"PDB_ligand_center:{recommended_pdb}")
                else:
                    log.append("PDB coordinate download deferred until strategy evolution and selection")

        # ── Step 4: 🆕 实时文献检索 ──
        search_query = f"{gene_symbol or gene} inhibitor drug discovery"
        pubmed_papers = search_pubmed(search_query)
        clinical_trials = search_clinical_trials(gene_symbol or gene)
        if pubmed_papers:
            api_sources.append(f"PubMed:{len(pubmed_papers)}_papers")
            log.append(f"PubMed: {len(pubmed_papers)} papers for '{search_query}'")
        if clinical_trials:
            api_sources.append(f"ClinicalTrials:{len(clinical_trials)}_trials")
            log.append(f"ClinicalTrials: {len(clinical_trials)} trials")

        # ── Step 5: ChEMBL搜索 + 真实活性数据 ──
        chembl_activities = []
        chembl_results = search_chembl_target(gene_symbol or gene)
        for cr in chembl_results[:2]:
            cid = cr.get("chembl_id", "")
            if cid:
                acts = fetch_chembl_activity(cid)  # 🆕 真实IC50/Ki
                chembl_activities.extend(acts)
                if acts:
                    log.append(f"ChEMBL {cid}: {len(acts)} activities fetched")
                    api_sources.append(f"ChEMBL_activity:{cid}")

        # ── Step 5: 🆕 将所有真实数据注入Prompt ──
        tool_data_text = self._format_tool_data(recommended_pdb, docking_center, verified_pdbs,
                                                 chembl_activities, pubmed_papers, clinical_trials)
        user_prompt = self._build_prompt(query, gene_symbol or gene, uniprot_id, uniprot_data,
                                          verified_pdbs, recommended_pdb, docking_center,
                                          chembl_activities, tool_data_text,
                                          expected_organism, mol_type)

        # ── Step 6: LLM归纳生成报告 ──
        report = self._call_llm_for_report(user_prompt)
        # 🆕 用真实工具数据覆盖LLM输出
        report["api_sources"] = api_sources
        report["verification_log"] = log
        report["uniprot_id"] = uniprot_id
        report["gene_symbol"] = gene_symbol
        report["verified_pdb_structures"] = verified_pdbs
        report["recommended_pdb_for_docking"] = recommended_pdb
        report["docking_center_from_pdb"] = docking_center
        report["chembl_activities"] = chembl_activities  # 🆕 真实活性数据
        report["target_organism"] = expected_organism if expected_organism else (uniprot_data.get("organism", "") if uniprot_data else "")
        report["target_uniprot_id"] = uniprot_id

        # 🆕 从API数据填充key_metrics
        km = report.get("key_metrics", {})
        if chembl_activities:
            values = [a.get("standard_value", 0) for a in chembl_activities if a.get("standard_value")]
            if values:
                km["known_ligand_ic50_range_nm"] = [min(values), max(values)]
        if verified_pdbs:
            resolutions = [p.get("resolution") for p in verified_pdbs if p.get("resolution")]
            if resolutions: km["best_pdb_resolution"] = min(resolutions)
            km["has_cocrystal"] = any(p.get("has_ligand", False) for p in verified_pdbs)
        report["key_metrics"] = km
        report["target_macromolecule_type"] = mol_type  # 🆕 核酸兼容
        if is_nucleic:
            report["gene_symbol"] = "N/A"
            report["uniprot_id"] = "N/A"
        report = scrub_report(report)  # 🆕 最终清洗
        # 🆕 靶点误识别检测: LLM自己在报告中发现了UniProt不匹配
        full_text = report.get("full_report_text", "")
        mismatch_keywords = ["无同源性", "不匹配", "不相关", "对应的是", "非预期", "wrong protein",
                             "mismatch", "not correspond", "unrelated", "different protein",
                             "不是目标", "错误", "incorrect target"]
        if full_text and any(kw in full_text for kw in mismatch_keywords):
            log.append(f"⚠️ Target mismatch detected in report. Re-searching with original query terms.")
            # 用原始query中的物种关键词重新搜索
            correct_org = expected_organism
            for kw, org in [("结核", "Mycobacterium tuberculosis"), ("分枝杆菌", "Mycobacterium tuberculosis"),
                            ("malaria", "Plasmodium falciparum"), ("plasmodium", "Plasmodium falciparum"),
                            ("ecoli", "Escherichia coli"), ("大肠杆菌", "Escherichia coli")]:
                if kw in query: correct_org = org; break
            if correct_org != expected_organism or not gene_symbol:
                log.append(f"  Retrying with organism={correct_org}")
                # 用物种关键词 + 原始query重新解析
                alt_gene = self._parse_organism_specific_gene(query, correct_org)
                if alt_gene and alt_gene != gene:
                    log.append(f"  Corrected gene: {gene} → {alt_gene}")
                    # 重新搜索UniProt
                    alt_candidates = search_uniprot_by_name(f"{alt_gene} {correct_org}")
                    if not alt_candidates:
                        alt_candidates = self._search_uniprot_global(f"{alt_gene} {correct_org}")
                    if alt_candidates:
                        alt_uniprot = alt_candidates[0].get("uniprot_id", "")
                        alt_data = verify_uniprot_id(alt_uniprot)
                        if alt_data:
                            gene_symbol = alt_data.get("gene_symbol", alt_gene)
                            uniprot_id = alt_uniprot
                            uniprot_data = alt_data
                            expected_organism = alt_data.get("organism", correct_org)
                            log.append(f"  ✅ Corrected: UniProt={uniprot_id}, gene={gene_symbol}")
                            # 重新搜索PDB
                            verified_pdbs = []
                            alt_pdbs = search_pdb_by_uniprot(uniprot_id)
                            for pid in alt_pdbs[:30]:
                                meta = verify_pdb_id(pid)
                                if meta:
                                    verified_pdbs.append(VerifiedPDBInfo(
                                        pdb_id=pid, resolution=meta.get("resolution"),
                                        method=meta.get("method",""), has_ligand=meta.get("has_ligand",False),
                                        title=meta.get("title",""), deposition_year=meta.get("deposition_year",0),
                                        ligand_ids=meta.get("ligand_ids",[]), uniprot_mapped=True,
                                    ).model_dump())
                            log.append(f"  Corrected PDBs: {len(verified_pdbs)} found")
        # 🆕 P0: 空报告兜底 — 检查完整性和单个章节, 绝不返回空白
        full_text = report.get("full_report_text", "")
        required_sections = ["biology_overview", "structural_analysis",
                             "druggability_assessment", "known_ligands_text"]
        empty_sections = [k for k in required_sections
                          if len(report.get(k, "").strip()) < 50]
        if len(full_text) < 500 or len(empty_sections) >= 3:
            log.append(f"LLM report incomplete (full_text={len(full_text)}chars, "
                       f"empty_sections={empty_sections}) — building API fallback report")
            report["full_report_text"] = self._build_api_fallback(
                target_name=gene_symbol or gene or "Unknown",
                uniprot_data=uniprot_data, verified_pdbs=verified_pdbs,
                chembl_activities=chembl_activities, pubmed_papers=pubmed_papers,
                clinical_trials=clinical_trials,
            )
            # 将拼装文本填入各字段(供test_tournament显示)
            ft = report["full_report_text"]
            report["biology_overview"] = ft
            report["structural_analysis"] = ""
            report["druggability_assessment"] = ""
            report["known_ligands_text"] = ""
        report["_search_log"] = log

        # ── 🆕 Step 7: 写入向量数据库 + 生成执行摘要 ──
        try:
            from src.tools.vector_store import ResearchVectorStore
            import hashlib as _hl
            # 向量库路径: 分析文件/vectordb/{task_hash}/
            vdb_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "分析文件", "vectordb",
                _hl.md5(query.encode()).hexdigest()[:12])
            os.makedirs(vdb_dir, exist_ok=True)
            vs = ResearchVectorStore(vdb_dir)

            # 写入各 API 数据
            if uniprot_data:
                vs.store_uniprot(uniprot_data)
            if verified_pdbs:
                vs.store_pdb(verified_pdbs)
            if chembl_activities:
                vs.store_chembl(chembl_activities)
            if pubmed_papers:
                vs.store_pubmed(pubmed_papers)
            if clinical_trials:
                vs.store_clinical(clinical_trials)
            # 完整报告分块存储
            full_text = report.get("full_report_text", "")
            if full_text:
                vs.store_full_report(full_text)

            # 生成执行摘要 (~6000 tokens)
            summary = self._generate_summary(
                target_name=report.get("target_name", gene_symbol or "Unknown"),
                gene=gene_symbol or gene,
                full_text=full_text[:15000],  # 传入前15000字符
            )
            report["executive_summary"] = summary
            report["_vector_db_path"] = vdb_dir
            log.append(f"VectorDB: stored in {vdb_dir}")
            log.append(f"Summary: {len(summary)} chars")
        except Exception as e:
            log.append(f"VectorDB/Summary error: {e}")
            report["executive_summary"] = report.get("full_report_text", "")[:6000]

        return report

    def _generate_summary(self, target_name: str, gene: str,
                           full_text: str) -> str:
        """生成结构化执行摘要 (JSON, ~2000 tokens), 只保留虚拟筛选决策关键数据。"""
        if not full_text:
            return ""
        is_reasoner = "reasoner" in self.model.lower()
        prompt = f"""从虚拟筛选角度提取靶点 {target_name} ({gene}) 的关键数据, 输出JSON:

{{
  "target_profile": {{
    "target_class": "PPI / Kinase / GPCR / Protease / Epigenetic / other",
    "pocket_type": "deep_cleft / shallow_groove / flat_ppi / allosteric / cryptic",
    "pocket_volume_approx": "small(<300) / medium(300-800) / large(>800) Å³",
    "pocket_polarity": "hydrophobic / mixed / polar",
    "rule_category": "Ro5 / bRo5 / custom",
    "key_selectivity_residues": "与同源蛋白的差异残基(如果有)"
  }},
  "structures": [
    {{"pdb_id":"XXXX","resolution":"X.XÅ","method":"X-ray/Cryo-EM","has_ligand":true/false,"ligand_ids":["XXX"],"key_residues":["Asp103","Trp144"]}}
  ],
  "known_ligands": {{
    "best_activity": {{"chembl_id":"XXX","type":"IC50/Ki/Kd","value":0.01,"unit":"nM"}},
    "mw_range":[250,600],
    "logp_range":[1.0,5.0],
    "key_pharmacophore":["H-bond donor","hydrophobic center"],
    "key_SAR":"简述构效关系(50字)"
  }},
  "druggability": {{
    "score":"0-10",
    "recommended_approaches":["SBDD","FBDD"],
    "red_flags":["selectivity_concern","PPI_flat_pocket"],
    "drug_design_notes":"关键注意事项(50字)"
  }}
}}

要求: 每个字段必须填, 不确定的写"待验证"。只输出纯JSON, 不要markdown。

## 原始报告
{full_text}"""

        try:
            kwargs = dict(model=self.model, max_tokens=2048,
                          messages=[{"role":"system","content":"你是药物研发数据提取专家。只提取虚拟筛选决策需要的结构化数据, 输出纯JSON。不编造数据, 不确定的填'待验证'。"},
                                    {"role":"user","content":prompt}])
            if not is_reasoner:
                kwargs["temperature"] = 0.1
                kwargs["response_format"] = {"type": "json_object"}
            resp = self.client.chat.completions.create(**kwargs)
            raw = (resp.choices[0].message.content or "").strip()
            if "{" in raw: raw = raw[raw.find("{"):]
            parsed = json.loads(raw)
            summary_text = json.dumps(parsed, ensure_ascii=False, indent=2)
            finish = getattr(resp.choices[0], "finish_reason", "stop")
            if finish == "length":
                print(f"  ⚠️ 摘要被截断(但JSON仍然有效)", flush=True)
            return summary_text
        except Exception as e:
            print(f"  ⚠️ 摘要生成失败: {e}, 使用全文前2000字符", flush=True)
            return full_text[:2000]

    @staticmethod
    def _build_api_fallback(target_name, uniprot_data, verified_pdbs,
                            chembl_activities, pubmed_papers, clinical_trials):
        """🆕 P0: 用已验证的API数据拼装报告。零LLM, 零幻觉。"""
        parts = [
            f"# 靶点深度调研报告: {target_name}\n",
            f"⚠️ **LLM文本生成失败, 本报告由API真实数据自动拼装。数据准确但缺乏分析深度。**\n",
            "---\n",
        ]
        # 1. UniProt
        if uniprot_data:
            parts.append("## 靶点生物学 (来源: UniProt API)\n")
            parts.append(f"- **蛋白全称**: {uniprot_data.get('protein_name','?')}\n")
            parts.append(f"- **基因**: {uniprot_data.get('gene_symbol','?')}\n")
            parts.append(f"- **物种**: {uniprot_data.get('organism','?')}\n")
            func = uniprot_data.get('function','')
            if func: parts.append(f"- **功能**: {func[:1500]}\n")
            parts.append("\n")
        # 2. PDB
        if verified_pdbs:
            parts.append(f"## 已知实验结构 ({len(verified_pdbs)}个, 来源: RCSB PDB API)\n\n")
            parts.append("| PDB ID | 分辨率 | 方法 | 含配体 | 年份 | 标题 |\n")
            parts.append("|--------|--------|------|--------|------|------|\n")
            for p in verified_pdbs[:12]:
                lig = "✓" if p.get("has_ligand") else ""
                parts.append(f"| {p.get('pdb_id','?')} | {p.get('resolution','?')}Å | {p.get('method','?')} | {lig} | {p.get('deposition_year','?')} | {p.get('title','?')[:60]} |\n")
            parts.append("\n")
        # 3. ChEMBL
        if chembl_activities:
            parts.append(f"## 已知配体活性数据 ({len(chembl_activities)}条, 来源: ChEMBL API)\n\n")
            parts.append("| ChEMBL ID | 类型 | 值 (nM) |\n|-----------|------|----------|\n")
            for a in chembl_activities[:15]:
                parts.append(f"| {a.get('molecule_chembl_id','?')} | {a.get('standard_type','?')} | {a.get('standard_value','?')} |\n")
            parts.append("\n")
        # 4. PubMed
        if pubmed_papers:
            parts.append(f"## 相关文献 ({len(pubmed_papers)}篇, 来源: PubMed)\n\n")
            for p in pubmed_papers[:10]:
                parts.append(f"- [{p.get('pmid','?')}] {p.get('title','?')} ({p.get('year','?')})\n")
            parts.append("\n")
        # 5. ClinicalTrials
        if clinical_trials:
            parts.append(f"## 临床试验 ({len(clinical_trials)}项, 来源: ClinicalTrials.gov)\n\n")
            for t in clinical_trials[:5]:
                parts.append(f"- {t.get('nct_id','?')}: {t.get('title','?')} [{t.get('phase','?')} / {t.get('status','?')}]\n")
            parts.append("\n")
        return "\n".join(parts)

    # =========================================================================
    # 工具数据格式化 (注入Prompt)
    # =========================================================================

    def _format_tool_data(self, recommended_pdb, docking_center, verified_pdbs,
                           chembl_activities, pubmed_papers, clinical_trials):
        lines = ["## 🛠️ 真实工具检索数据 (优先级高于历史知识!)"]
        if verified_pdbs:
            lines.append(f"\n### 已验证PDB结构 ({len(verified_pdbs)}个, 最新优先)")
            sorted_pdbs = sorted(verified_pdbs, key=lambda x: x.get("deposition_year", 0), reverse=True)
            for p in sorted_pdbs:
                lig = f" (配体: {p.get('ligand_ids',[])})" if p.get("has_ligand") else ""
                lines.append(f"- {p['pdb_id']}: {p.get('resolution','?')}Å {p.get('method','?')} "
                           f"年份={p.get('deposition_year',0)}{lig} — {p.get('title','?')[:80]}")
            if recommended_pdb:
                lines.append(f"\n**推荐对接结构**: {recommended_pdb}")
                lines.append(f"**对接盒子中心(来自PDB文件解析)**: {docking_center if docking_center else '无配体可解析'}")
        else:
            lines.append("\n### PDB: 未找到已验证结构")

        if chembl_activities:
            lines.append(f"\n### ChEMBL真实活性数据 ({len(chembl_activities)}条)")
            for a in chembl_activities[:10]:
                lines.append(f"- {a.get('molecule_chembl_id','?')}: "
                           f"{a.get('standard_type','?')}={a.get('standard_value','?')} {a.get('standard_units','nM')}")
            best = chembl_activities[0]
            lines.append(f"\n**最有效配体**: {best.get('molecule_chembl_id','?')} "
                        f"({best.get('standard_type','?')}={best.get('standard_value','?')} {best.get('standard_units','nM')})")
        else:
            lines.append("\n### ChEMBL: 未找到活性数据")

        if pubmed_papers:
            lines.append(f"\n### 📚 PubMed实时文献 ({len(pubmed_papers)}篇)")
            for p in pubmed_papers[:12]:
                lines.append(f"- [{p.get('pmid','?')}] {p.get('title','?')[:120]} ({p.get('year','?')})")
                if p.get('abstract'):
                    lines.append(f"  摘要: {p.get('abstract','?')[:300]}...")

        if clinical_trials:
            lines.append(f"\n### 🏥 ClinicalTrials.gov ({len(clinical_trials)}项)")
            for t in clinical_trials[:8]:
                lines.append(f"- {t.get('nct_id','?')}: {t.get('title','?')[:100]} [{t.get('phase','?')}/{t.get('status','?')}]")

        return "\n".join(lines)

    def _build_prompt(self, query, gene, uniprot_id, uniprot_data, verified_pdbs,
                      recommended_pdb, docking_center, chembl_activities, tool_data_text,
                      expected_organism="", mol_type="Protein"):
        parts = [f"## 靶点调研任务\n查询: {query}\n基因/靶点: {gene}\nUniProt: {uniprot_id}\n物种: {expected_organism}\n类型: {mol_type}"]
        if uniprot_data:
            parts.append(f"### UniProt数据\n蛋白: {uniprot_data.get('protein_name','?')}\n功能: {uniprot_data.get('function','?')[:600]}")
        parts.append(tool_data_text)
        parts.append("""### 任务
生成纯调研报告JSON。⚠️ 不要生成筛选策略或漏斗设计(下游StrategyGenerator负责)!
- 每个文本字段至少800字
- 如果工具数据中有ChEMBL活性数据, 必须在known_ligands_text中详细引用SAR
- 如果工具数据中有PDB结构, 必须在structural_analysis中逐一分析共晶配体结合模式
- 必须包含: 已上市药物/临床候选化合物/研究现状/竞争格局
- 禁止捏造任何数值! 不确定的信息标注"待实验验证" """.format(
            rec_pdb=recommended_pdb or "无",
            center=docking_center if docking_center else "无配体可解析",
        ))
        return "\n\n".join(parts)

    @staticmethod
    def _sanitize_text(text: str, max_chars: int = 3000) -> str:
        """🆕 幻觉防护: 截断过长文本, 检测并清除重复模式。

        处理 LLM 可能产生的两种输出异常:
          1. 单字段超长 (如 NR4A2 口袋残基生成了5000个Leu)
          2. 短模式大量重复 (如 "Leu1234, Leu1235, ..." 连续出现)
        """
        if not text or not isinstance(text, str):
            return text or ""

        # 1. 硬截断
        if len(text) > max_chars:
            # 尝试在最后一个完整句子处截断
            trunc_point = max_chars
            for sep in ["\n\n", "。", ". ", "\n"]:
                last_sep = text.rfind(sep, 0, max_chars)
                if last_sep > max_chars * 0.7:
                    trunc_point = last_sep + len(sep)
                    break
            text = text[:trunc_point].rstrip()
            if not text.endswith(("。", ".", "!", "?", "\n")):
                text += "\n\n[内容过长已截断]"

        # 2. 重复模式检测: 同一短词连续出现 >20次
        import re
        # 检测 "LeuNNNN, LeuNNNN, ..." 这种重复残基列表
        residue_repeat = re.findall(r'((?:Leu|Val|Ile|Phe|Ala|Gly|Ser|Thr|Cys|Met|Pro|Trp|Tyr|Asn|Gln|Asp|Glu|Lys|Arg|His)\d{2,4},\s*){20,}', text)
        if residue_repeat:
            # 找到第一个大量重复的位置并截断
            for pattern in residue_repeat:
                idx = text.find(pattern)
                if idx > 0:
                    text = text[:idx].rstrip()
                    text += "\n\n[重复残基列表已清除 — 疑似LLM幻觉]"
                    break

        return text

    def _call_llm_for_report(self, user_prompt: str) -> Dict[str, Any]:
        """🆕 多段独立生成: 4个章节各一次独立LLM调用, 并行执行。

        每段只需输出纯Markdown文本(无JSON开销), token预算充足。
        单段失败不影响其他段, 最终汇总验证。
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        is_reasoner = "reasoner" in self.model.lower()
        results: Dict[str, str] = {}

        def _call_section(cfg: dict) -> tuple:
            """单章节LLM调用。返回 (field_name, text_content, error_msg)。"""
            field = cfg["field"]
            try:
                msgs = [
                    {"role": "system", "content": cfg["system_prompt"]},
                    {"role": "user", "content": user_prompt},
                    {"role": "system", "content": (
                        f"请只输出 '{cfg['title']}' 章节的纯Markdown文本。"
                        f"以 '## {cfg['section_num']}. {cfg['title']}' 开头。"
                        f"约{cfg['min_chars']}-{cfg['max_chars']}字。不要输出JSON,不要输出其他章节。"
                        f"防幻觉: 不捏造数值, API无数据则诚实说明。"
                    )},
                ]
                api_kwargs = dict(
                    model=self.model, max_tokens=self.max_tokens,
                    messages=msgs,
                )
                if not is_reasoner:
                    api_kwargs["temperature"] = self.temperature
                resp = self.client.chat.completions.create(**api_kwargs)
                raw = resp.choices[0].message.content or ""
                finish = getattr(resp.choices[0], "finish_reason", "stop")
                if finish == "length":
                    print(f"  ⚠️ [{field}] 输出被截断(finish_reason=length)", flush=True)
                # 清洗输出: 去掉可能的markdown代码块标记
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = re.sub(r'^```\w*\n?', '', raw)
                    raw = re.sub(r'\n?```$', '', raw)
                if len(raw) < 100:
                    return (field, "", f"输出过短({len(raw)}字符)")
                print(f"  ✅ [{field}] {len(raw)} chars (finish={finish})", flush=True)
                return (field, raw, "")
            except Exception as e:
                return (field, "", str(e))

        # 并行调用4个章节
        print(f"\n📝 多段并行生成报告 (4个独立LLM调用)...", flush=True)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_call_section, cfg): cfg for cfg in SECTION_CONFIGS}
            for future in as_completed(futures):
                field, text, err = future.result()
                if err:
                    print(f"  ❌ [{field}] 失败: {err}", flush=True)
                results[field] = text

        # 组装报告 — 从user_prompt提取靶点名称, 避免Unknown Target
        m = re.search(r'基因/靶点:\s*(\S+)', user_prompt)
        target_name = m.group(1) if m else "Unknown Target"
        report = TargetResearchReport(target_name=target_name).model_dump()
        for cfg in SECTION_CONFIGS:
            field = cfg["field"]
            text = results.get(field, "")
            report[field] = self._sanitize_text(text)

        sections = [report.get(k, "") for k in
                    ["biology_overview", "structural_analysis",
                     "known_ligands_text", "druggability_assessment"]]
        report["full_report_text"] = "\n\n".join(s for s in sections if s)
        report = scrub_report(report)

        # 完整性验证
        empty_sections = [cfg["field"] for cfg in SECTION_CONFIGS
                          if len(report.get(cfg["field"], "").strip()) < 100]

        if empty_sections:
            print(f"⚠️ 缺失章节: {empty_sections}", flush=True)
            if len(report.get("full_report_text", "")) < 300:
                print(f"  → 报告过短, 回退到单次调用模式重试...", flush=True)
                return self._call_llm_single_fallback(user_prompt)

        return report

    def _call_llm_single_fallback(self, user_prompt: str) -> Dict[str, Any]:
        """单次调用兜底: 当多段并行全部失败时, 回退到原来的单次JSON模式。"""
        is_reasoner = "reasoner" in self.model.lower()
        kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                      messages=[{"role":"system","content":SCOUT_SYSTEM_PROMPT},
                                {"role":"user","content":user_prompt},
                                {"role":"system","content":"输出纯Markdown文本, 按 ## 1. ## 2. ## 3. ## 4. 格式组织4个章节。不要JSON包装。"}])
        if not is_reasoner:
            kwargs["temperature"] = self.temperature
        try:
            resp = self.client.chat.completions.create(**kwargs)
            raw = (resp.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = re.sub(r'^```\w*\n?', '', raw)
                raw = re.sub(r'\n?```$', '', raw)
            # 按 ## 数字. 分割章节
            parts = re.split(r'\n(?=## \d\.)', raw)
            m2 = re.search(r'基因/靶点:\s*(\S+)', user_prompt)
            fb_target = m2.group(1) if m2 else "Unknown Target"
            report = TargetResearchReport(target_name=fb_target).model_dump()
            field_map = {"1": "biology_overview", "2": "structural_analysis",
                         "3": "known_ligands_text", "4": "druggability_assessment"}
            for part in parts:
                m = re.match(r'## (\d)\.', part.strip())
                if m and m.group(1) in field_map:
                    report[field_map[m.group(1)]] = self._sanitize_text(part.strip())
            sections = [report.get(k, "") for k in
                        ["biology_overview", "structural_analysis",
                         "known_ligands_text", "druggability_assessment"]]
            report["full_report_text"] = "\n\n".join(s for s in sections if s)
            return scrub_report(report)
        except Exception as e:
            print(f"  ❌ 兜底调用也失败: {e}", flush=True)
            return TargetResearchReport(target_name="Unknown").model_dump()

    @staticmethod
    def _robust_json_parse(raw: str) -> Dict[str, Any]:
        """多层JSON修复: 处理LLM输出的各种格式问题。"""
        # 1) 直接解析
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # 2) 去除markdown代码块标记
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```\w*\n', '', cleaned)
            cleaned = re.sub(r'\n```$', '', cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # 3) 修复未转义换行: 找到字符串值内嵌的裸换行并转义
        # 简单策略: 提取第一个{到最后一个}之间的内容, 替换裸换行为\n
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            core = cleaned[start:end+1]
            # 在JSON字符串值内部, 将裸换行替换为转义换行
            # 更安全的方式: 用json.JSONDecoder.raw_decode逐步解析
            try:
                decoder = json.JSONDecoder()
                result, _ = decoder.raw_decode(core)
                return result
            except json.JSONDecodeError:
                pass
        # 4) 最后尝试: 逐字段正则提取
        result = {}
        for field in ["target_name","biology_overview","structural_analysis",
                       "druggability_assessment","screening_strategy","known_ligands_text"]:
            m = re.search(rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned, re.DOTALL)
            if m:
                result[field] = m.group(1).replace('\\n', '\n').replace('\\"', '"')
        return result if result else {}

    # =========================================================================
    # 意图解析 + PDB搜索
    # =========================================================================

    def _parse_intent(self, query: str) -> Dict[str, Any]:
        is_reasoner = "reasoner" in self.model.lower()
        kwargs = dict(model=self.model, max_tokens=512,
                      messages=[{"role":"system","content":(
                          "提取靶点基因符号和物种。中文靶点翻译为英文HUGO符号: "
                          "BCL-2→BCL2, BCL-xl→BCL2L1, 雄激素受体→AR, EGFR→EGFR。"
                          "病毒/细菌靶点用原名(is_pathogen=true)。"
                          "🚨 PROTAC/分子胶/降解剂/药物筛选关键词 → organism默认'Homo sapiens'(这些技术几乎只用人类蛋白)。"
                          "输出JSON: {\"target_name\":\"基因符号\",\"organism\":\"物种\",\"is_pathogen\":false,\"macromolecule_type\":\"Protein\"}"
                      )},
                      {"role":"user","content":f"查询: {query}"}])
        if not is_reasoner:
            kwargs["temperature"] = 0.0
            kwargs["response_format"] = {"type":"json_object"}
        try:
            resp = self.client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content
            if not raw or not raw.strip():
                raw = getattr(resp.choices[0].message, "reasoning_content", "") or ""
                # 从推理末尾提取JSON
                if "{" in raw:
                    raw = raw[raw.rfind("{"):]
            parsed = json.loads(raw.strip())
            if "properties" in parsed and "target_name" not in parsed:
                kwargs2 = dict(model=self.model, max_tokens=256,
                               messages=[{"role":"user","content":f"提取靶点(json): {query}"},
                                         {"role":"system","content":'输出JSON: {"target_name":"靶点名"}'}])
                if not is_reasoner: kwargs2.update(temperature=0.0, response_format={"type":"json_object"})
                resp2 = self.client.chat.completions.create(**kwargs2)
                parsed = json.loads(resp2.choices[0].message.content.strip())
            return parsed
        except Exception:
            # 启发式: 提取英文基因符号 (≥2大写字母)
            m = re.search(r'\b([A-Z]{2,}[0-9]*[A-Za-z]*)\b', query)
            gene = m.group(1) if m else ""
            cn_map = {"雄激素受体":"AR","雌激素受体":"ESR1","BCL-2":"BCL2","EGFR":"EGFR","KRAS":"KRAS"}
            for cn, en in cn_map.items():
                if cn in query: gene = en; break
            return {"target_name":gene,"uniprot_id":"","target_class":"","is_viral":False,"is_pathogen":False,"organism":""}

    @staticmethod
    def _parse_organism_specific_gene(query: str, organism: str) -> str:
        """从用户查询中识别病原体特异性基因名。"""
        import re as _re
        # Mycobacterium tuberculosis targets
        mtb_genes = {"InhA": "inhA", "inhA": "inhA", "KatG": "katG", "katG": "katG",
                     "AhpC": "ahpC", "FabG1": "fabG1", "MabA": "mabA", "KasA": "kasA",
                     "DprE1": "dprE1", "MmpL3": "mmpL3", "PknB": "pknB", "EthR": "ethR"}
        if "mycobacterium" in organism.lower() or "结核" in query:
            for gene_name, corrected in mtb_genes.items():
                if gene_name.lower() in query.lower():
                    return corrected
        # Plasmodium targets
        pf_genes = {"DHFR": "dhfr-ts", "DHPS": "dhps", "CRT": "crt"}
        if "plasmodium" in organism.lower() or "疟原虫" in query:
            for gene_name, corrected in pf_genes.items():
                if gene_name.lower() in query.lower():
                    return corrected
        # Extract any gene-like pattern
        m = _re.search(r'(?:^|[一-鿿\s])([A-Za-z][A-Za-z0-9]{2,})', query)
        return m.group(1) if m else ""

    def _search_uniprot_global(self, name: str) -> List[Dict]:
        """全局搜索UniProt (不限物种: 病毒/细菌/寄生虫/人类)。"""
        import urllib.parse
        data = _http_get(f"https://rest.uniprot.org/uniprotkb/search?"
                         f"query={urllib.parse.quote(name)}&size=10")
        if not data:
            return []
        return [{"uniprot_id": r.get("primaryAccession", ""),
                 "gene": (r.get("genes", [{}])[0].get("geneName", {}).get("value", "") if r.get("genes") else ""),
                 "protein": r.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "")}
                for r in data.get("results", [])]

    def _search_pdb_by_gene(self, gene: str) -> List[str]:
        if not gene: return []
        try:
            import urllib.parse
            # full_text 可搜索所有文本字段(基因名/蛋白名/摘要等)
            query_json = json.dumps({
                "query": {"type": "terminal", "service": "full_text",
                          "parameters": {"value": gene}},
                "return_type": "entry",
                "request_options": {"paginate": {"start": 0, "rows": 15}},
            })
            url = "https://search.rcsb.org/rcsbsearch/v2/query?json=" + urllib.parse.quote(query_json)
            data = _http_get(url)
            return [r.get("identifier", "") for r in data.get("result_set", [])] if data else []
        except Exception:
            return []

    def generate_profile(self, target_info: dict) -> Dict[str, Any]:
        return self.deep_research(target_info.get("description", target_info.get("target_name", "")))


def target_scout_node(state: dict) -> dict:
    agent = TargetScoutAgent()
    query = state.get("target_info", {}).get("description", state.get("target_info", {}).get("target_name", ""))
    report = agent.deep_research(query or "unknown target")
    return {"pipeline_stage":"target_scout","target_profile":report,
            "updated_at":datetime.now(timezone.utc).isoformat(),
            "event_log":[f"[TargetScout] PDBs:{len(report.get('verified_pdb_structures',[]))} "
                         f"ChEMBL:{len(report.get('chembl_activities',[]))}"]}
