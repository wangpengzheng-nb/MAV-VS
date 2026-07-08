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


class TargetResearchReport(BaseModel):
    """防幻觉版靶点调研报告 — 真实数据由Python工具函数填入。"""

    target_name: str = Field(default="Unknown")
    target_macromolecule_type: str = Field(
        default="Protein",
        description="靶点类型: Protein / RNA / DNA / Complex。决定检索策略。无法确定时默认Protein。"
    )
    gene_symbol: str = Field(default="", description="蛋白质靶点填HUGO基因符号; RNA/DNA靶点填N/A")
    uniprot_id: str = Field(default="", description="蛋白质靶点填UniProt Accession; RNA/DNA靶点填N/A")
    target_uniprot_id: str = Field(default="", description="蛋白质靶点的首选PDB检索锚点")
    target_organism: str = Field(
        default="Homo sapiens",
        description="目标物种。从用户查询提取。无法确定时默认Homo sapiens。禁止填'?'或空!"
    )

    # ── 结构(来自PDB API验证 + PDB文件解析) ──
    verified_pdb_structures: List[VerifiedPDBInfo] = Field(default_factory=list)
    recommended_pdb_for_docking: str = Field(
        default="",
        description="推荐用于对接的PDB ID(已通过API验证, 优先选择最新+最高分辨率+含配体的)"
    )
    docking_center_from_pdb: List[float] = Field(
        default_factory=list,
        description="对接盒子中心[x,y,z] — 由fetch_pdb_ligand_center()从真实PDB文件解析, 非LLM生成!"
    )
    binding_site: BindingSiteAnalysis = Field(default_factory=BindingSiteAnalysis)

    # ── 配体(来自ChEMBL API) ──
    chembl_activities: List[ChemblActivity] = Field(
        default_factory=list,
        description="从ChEMBL API查询到的真实活性数据(IC50/Ki/Kd)"
    )
    known_ligands_text: str = Field(
        default="",
        description="已知配体文字描述(基于chembl_activities归纳)"
    )

    # ── 长篇文本 ──
    biology_overview: str = Field(default="")
    structural_analysis: str = Field(default="")
    druggability_assessment: str = Field(default="")
    screening_strategy: str = Field(default="")
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


TargetResearchReport.model_rebuild()
VerifiedPDBInfo.model_rebuild()
ChemblActivity.model_rebuild()
BindingSiteAnalysis.model_rebuild()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  System Prompt — Grounding + 防幻觉                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

SCOUT_SYSTEM_PROMPT = """\
# 人设: 严谨的靶点情报分析师

你基于用户查询 + 真实API检索数据, 归纳生成靶点调研报告。

---

# 🚨 核心原则: 以检索为准 (Grounding)!

## 绝对禁止使用历史训练记忆覆盖最新检索结果!

- 如果API检索到了2024年后发表的PDB结构, 必须在正文中明确指出该最新结构,
  并推荐使用它进行对接。禁止说"没有实验结构"或建议"同源建模"!
- 如果API检索到了冷冻电镜(Cryo-EM)或X-ray晶体结构,
  必须优先推荐它们, 而不是建议AlphaFold预测!
- 报告的结构分析和筛选策略必须基于真实API数据, 不得凭空推理!

## 防幻觉红线(不变)

1. 禁止捏造PDB ID! 只能填写verified_pdb_structures中已验证的!
2. 禁止捏造配体IC50/Ki值! 只能引用chembl_activities中从ChEMBL API查询到的真实数据!
3. 禁止捏造对接盒子XYZ坐标! 具体坐标由fetch_pdb_ligand_center()从真实PDB文件解析,
   在docking_center_from_pdb字段中填入, 你只需在docking_box_strategy中描述策略!
4. 禁止张冠李戴! 不将其他靶点的配体归于本靶点!
5. 🆕 禁止跨物种借用PDB结构! 所有PDB结构的源物种必须与用户查询中的物种严格匹配!
6. 🆕 严禁使用训练记忆中的PDB ID! (记忆隔离墙)
   如果search_rcsb_pdb工具返回空, 你必须写"未检索到可用实验结构", 绝对禁止写"我记得有4D1P"!
   你只能引用verified_pdb_structures中经API验证的PDB ID。如果觉得应该有但没查到,
   写"[系统检索可能存在遗漏, 需人工复核 UniProt: XXXX]", 不能自己列出PDB编号!
7. 🆕 禁止输出"?"或空字符串占位!
   target_organism无法确定→填"Homo sapiens"; gene_symbol未知→从UniProt数据提取;
   所有字段必须填具体内容, 用"?"代表未知是不可接受的!

8. 禁止跨蛋白做同源建模! (同源建模处理原则)
   如果目标是突变体(如BTK C481S)且没有直接结构, 必须优先使用同蛋白野生型高分辨率结构!
   在筛选策略中建议"基于WT结构做in silico氨基酸突变", 不允许用其他蛋白(如C-Src)替代!
   例如: BTK没有C481S结构→用BTK WT结构(如5P9J)做点突变, 绝不用C-Src(如3OOM)做模板!
   所有PDB标题已通过protein_keywords过滤, 不属于目标蛋白的结构已被自动排除。
"""


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
    return {
        "resolution": res_list[0] if res_list else None,
        "method": ei.get("experimental_method", ""),
        "has_ligand": bool(ligand_ids),
        "title": (data.get("struct") or {}).get("title", ""),
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


def search_uniprot_by_name(name: str) -> List[Dict]:
    import urllib.parse
    data = _http_get(f"https://rest.uniprot.org/uniprotkb/search?query={urllib.parse.quote(name)}+AND+organism_id:9606&size=5")
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
    def __init__(self, model="deepseek-chat", api_key=None, api_base=None, temperature=0.2, max_tokens=4096):
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

    def deep_research(self, query: str) -> Dict[str, Any]:
        log = [f"Query: {query}"]
        api_sources = []

        # ── Step 1: LLM意图解析 ──
        intent = self._parse_intent(query)
        gene = intent.get("target_name", "")
        is_viral = intent.get("is_viral", False)
        is_pathogen = intent.get("is_pathogen", False)
        intent_organism = intent.get("organism", "")
        log.append(f"Intent: gene={gene}, organism={intent_organism or '?'}, viral={is_viral}, pathogen={is_pathogen}")

        # ── Step 2: UniProt搜索+验证 (🆕 仅蛋白质靶点) ──
        uniprot_data, uniprot_id, gene_symbol = None, "", ""
        mol_type = intent.get("macromolecule_type", "Protein")
        is_nucleic = mol_type in ("RNA", "DNA")
        if gene and not is_nucleic:
            if is_viral:
                candidates = self._search_uniprot_global(gene)
            elif is_pathogen or intent_organism:
                candidates = self._search_uniprot_global(gene)  # 病原体不用人类filter
            else:
                candidates = search_uniprot_by_name(gene)
            if candidates:
                best = candidates[0]
                uniprot_id = best.get("uniprot_id", "")
                gene_symbol = best.get("gene") or gene
                uniprot_data = verify_uniprot_id(uniprot_id) if uniprot_id else None
                if uniprot_data:
                    api_sources.append(f"UniProt:{uniprot_id}(verified)")
                    log.append(f"UniProt: {uniprot_id} → {gene_symbol}")
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

            # 🆕 标题过滤: 去掉明确属于其他蛋白的结构
            # 从base_name提取蛋白关键词用于标题匹配
            protein_keywords = list(set(
                w for w in base_name.upper().replace("-", " ").split()
                if len(w) >= 3 and w not in {"AND", "THE", "FOR", "WITH", "WT", "WILD", "TYPE"}
            ))
            # 扩展: 如果base_name是基因符号, 也加全名关键词
            name_keywords = {
                "BTK": ["BTK", "BRUTON", "TYROSINE KINASE BTK"],
                "EGFR": ["EGFR", "EPIDERMAL GROWTH", "ERBB1"],
                "BCL2": ["BCL-2", "BCL2", "B-CELL LYMPHOMA"],
                "SHP2": ["PTPN11", "SHP2", "SHP-2", "TYROSINE-PROTEIN PHOSPHATASE", "TYROSINE PHOSPHATASE"],
                "PTPN11": ["PTPN11", "SHP2", "SHP-2", "TYROSINE-PROTEIN PHOSPHATASE"],
                "CYP51": ["CYP51", "STEROL 14-ALPHA", "14-ALPHA DEMETHYLASE", "ERG11"],
                "DHFR": ["DHFR", "DIHYDROFOLATE REDUCTASE"],
            }
            if base_name.upper() in name_keywords:
                protein_keywords = name_keywords[base_name.upper()]

            filtered_pdbs = []
            for pid in sorted(all_pdb_ids)[:20]:
                meta = verify_pdb_id(pid)
                if not meta:
                    continue
                title = (meta.get("title", "") or "").upper()
                # 🆕 如果标题中出现了明确的"其他蛋白"关键词, 排除!
                # 例如: "C-Src" 在BTK搜索中出现 → 排除
                cross_contamination = {
                    "BTK": ["C-SRC", "SRC KINASE", "LCK", "LYN", "ITK", "TEC KINASE", "BMX"],
                    "EGFR": ["ERBB2", "HER2", "ERBB3", "ERBB4", "C-MET"],
                    "BCL2": ["BCL-XL", "BCL2L1", "MCL-1", "BAX", "BAK"],
                }
                is_contaminated = False
                if base_name.upper() in cross_contamination:
                    for contaminant in cross_contamination[base_name.upper()]:
                        if contaminant in title and not any(kw in title for kw in protein_keywords):
                            is_contaminated = True; break
                if is_contaminated:
                    log.append(f"  🚫 FILTERED {pid}: cross-contamination (wrong protein) — {meta.get('title','')[:80]}")
                    continue

                # 🆕 标题关键词匹配: Track1/NA结果不因标题缺失而误杀
                title_match = any(kw in title for kw in protein_keywords) if protein_keywords else True
                from_track1 = pid in uniprot_pdb_ids if (uniprot_id or is_nucleic) else False
                if not title_match and not from_track1:
                    if all_pdb_ids and len(all_pdb_ids) > 20:
                        log.append(f"  ⚠️ SKIPPED {pid}: title lacks protein keywords — {meta.get('title','')[:80]}")
                        continue

                # 物种校验
                orgs = verify_pdb_organism(pid)
                if orgs and expected_organism:
                    match = any(expected_organism.lower() in o.lower() for o in orgs)
                    if not match:
                        rejected_pdbs.append(f"{pid}(orgs={orgs})")
                        continue
                    orgs = verify_pdb_organism(pid)
                    if orgs and expected_organism:
                        # orgs有值 → 检查物种匹配
                        match = any(expected_organism.lower() in o.lower() for o in orgs)
                        if not match:
                            rejected_pdbs.append(f"{pid}(orgs={orgs})")
                            continue  # 丢弃跨物种结构!
                    # orgs为None → 无法判断, 不拒绝(来自search_rcsb_pdb的物种搜索词已足够)
                    verified_pdbs.append(VerifiedPDBInfo(
                        pdb_id=pid, resolution=meta.get("resolution"),
                        method=meta.get("method", ""), has_ligand=meta.get("has_ligand", False),
                        title=meta.get("title", ""), deposition_year=meta.get("deposition_year", 0),
                        ligand_ids=meta.get("ligand_ids", []),
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
                center_data = fetch_pdb_ligand_center(recommended_pdb)
                if center_data and center_data.get("top_ligands"):
                    top_lig = center_data["top_ligands"][0]
                    docking_center = top_lig["center"]
                    log.append(f"PDB ligand center({recommended_pdb}): {docking_center}")
                    api_sources.append(f"PDB_ligand_center:{recommended_pdb}")

        # ── Step 4: ChEMBL搜索 + 🆕 真实活性数据 ──
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

        # ── Step 5: 🆕 将真实工具数据注入Prompt ──
        tool_data_text = self._format_tool_data(recommended_pdb, docking_center, verified_pdbs, chembl_activities)
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
        report["target_uniprot_id"] = uniprot_id  # 🆕 首选检索锚点
        report["target_macromolecule_type"] = mol_type  # 🆕 核酸兼容
        if is_nucleic:
            report["gene_symbol"] = "N/A"
            report["uniprot_id"] = "N/A"
        report = scrub_report(report)  # 🆕 最终清洗
        report["_search_log"] = log
        return report

    # =========================================================================
    # 工具数据格式化 (注入Prompt)
    # =========================================================================

    def _format_tool_data(self, recommended_pdb, docking_center, verified_pdbs, chembl_activities):
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
            # 取最有效的作为参考
            best = chembl_activities[0]
            lines.append(f"\n**最有效配体**: {best.get('molecule_chembl_id','?')} "
                        f"({best.get('standard_type','?')}={best.get('standard_value','?')} {best.get('standard_units','nM')})")
        else:
            lines.append("\n### ChEMBL: 未找到活性数据")

        return "\n".join(lines)

    def _build_prompt(self, query, gene, uniprot_id, uniprot_data, verified_pdbs,
                      recommended_pdb, docking_center, chembl_activities, tool_data_text,
                      expected_organism="", mol_type="Protein"):
        parts = [f"## 靶点调研任务\n查询: {query}\n基因/靶点: {gene}\nUniProt: {uniprot_id}\n物种: {expected_organism}\n类型: {mol_type}"]
        if uniprot_data:
            parts.append(f"### UniProt数据\n蛋白: {uniprot_data.get('protein_name','?')}\n功能: {uniprot_data.get('function','?')[:600]}")
        parts.append(tool_data_text)
        parts.append("""### 任务
生成TargetResearchReport JSON。记住:
- 如果工具数据中有2024年后的PDB结构, structural_analysis必须推荐它, 禁止说"没有结构"!
- 如果工具数据中有ChEMBL活性数据, 在known_ligands_text中引用具体数值
- 禁止捏造PDB ID/IC50值/XYZ坐标!
- 推荐对接PDB: {rec_pdb}, 对接盒子中心(来自PDB文件): {center}""".format(
            rec_pdb=recommended_pdb or "无",
            center=docking_center if docking_center else "无配体可解析",
        ))
        return "\n\n".join(parts)

    def _call_llm_for_report(self, user_prompt: str) -> Dict[str, Any]:
        schema = {"type":"object","required":["target_name"],"properties":{
            "target_name":{"type":"string"},"biology_overview":{"type":"string"},
            "structural_analysis":{"type":"string"},"druggability_assessment":{"type":"string"},
            "screening_strategy":{"type":"string"},"known_ligands_text":{"type":"string"},
            "references":{"type":"array","items":{"type":"string"}},
            "binding_site":{"type":"object","properties":{
                "pocket_type":{"type":"string"},"pocket_description":{"type":"string"},
                "key_residues_text":{"type":"string"},"docking_box_strategy":{"type":"string"},
                "structural_flexibility":{"type":"string"}}}}}
        raw = ""
        try:
            resp = self.client.chat.completions.create(
                model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                messages=[{"role":"system","content":SCOUT_SYSTEM_PROMPT},
                          {"role":"user","content":user_prompt},
                          {"role":"system","content":f"严格输出合法JSON(所有字符串中的换行用\\n转义, 引号用\\\"转义):\n{json.dumps(schema,indent=2,ensure_ascii=False)}"}],
                response_format={"type":"json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            # 多层JSON修复
            parsed = self._robust_json_parse(raw)
            report = TargetResearchReport.model_validate(parsed).model_dump()
            sections = [report.get(k,"") for k in ["biology_overview","structural_analysis","druggability_assessment","screening_strategy"]]
            report["full_report_text"] = "\n\n".join(s for s in sections if s)
            report = scrub_report(report)  # 🆕 清洗走私坐标
            return report
        except Exception as e:
            import sys as _sys
            print(f"\n⚠️ LLM报告生成失败: {e}", file=_sys.stderr)
            # 降级: 尝试从raw中提取部分文本
            fallback = TargetResearchReport(target_name="Unknown").model_dump()
            fallback = scrub_report(fallback)  # 🆕 降级也清洗
            try:
                # 至少填入已有的API数据
                partial = self._robust_json_parse(raw) if raw else {}
                fallback["biology_overview"] = str(partial.get("biology_overview", ""))[:500]
                fallback["structural_analysis"] = str(partial.get("structural_analysis", ""))[:500]
            except Exception:
                pass
            return fallback

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
        try:
            resp = self.client.chat.completions.create(
                model=self.model, temperature=0.0, max_tokens=512,
                messages=[{"role":"system","content":(
                    "提取靶点标识和物种。规则: "
                    "- 人类靶点→HUGO基因符号, organism=Homo sapiens "
                    "- 病毒靶点→病毒名+蛋白名, organism=病毒学名 "
                    "- 寄生虫/细菌→学名+蛋白名, organism=拉丁学名 "
                    "- RNA/DNA靶点→macromolecule_type=RNA或DNA, organism=物种 "
                    "- 示例: HIV-1 TAR RNA→target_name=HIV-1 TAR RNA, macromolecule_type=RNA "
                    "- 输出JSON: {\"target_name\":\"ID\",\"organism\":\"物种\",\"is_pathogen\":false,\"macromolecule_type\":\"Protein\"}"
                )},
                          {"role":"user","content":f"查询: {query}"},
                          {"role":"system","content":'输出JSON: {"target_name":"靶点ID","organism":"物种学名","is_pathogen":false}'}],
                response_format={"type":"json_object"},
            )
            parsed = json.loads(resp.choices[0].message.content.strip())
            if "properties" in parsed and "target_name" not in parsed:
                resp2 = self.client.chat.completions.create(
                    model=self.model, temperature=0.0, max_tokens=256,
                    messages=[{"role":"user","content":f"提取靶点(json): {query}"},
                              {"role":"system","content":'输出JSON: {"target_name":"靶点名"}'}],
                    response_format={"type":"json_object"},
                )
                parsed = json.loads(resp2.choices[0].message.content.strip())
            return parsed
        except Exception:
            return {"target_name":"","uniprot_id":"","target_class":"","is_viral":False,"is_pathogen":False,"organism":""}

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
