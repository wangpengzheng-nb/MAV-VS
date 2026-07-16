"""
Research Vector Store — ChromaDB 封装
======================================
用于存储靶点调研的 API 原始数据并支持语义搜索。
每个 collection 自动打标签, 供下游 Agent 按需查询。
"""
from __future__ import annotations
import os, json
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings


class ResearchVectorStore:
    """靶点调研向量数据库。

    Collections:
      - uniprot:    蛋白功能、基因、物种
      - pdb:        结构信息 (pdb_id, resolution, has_ligand, method)
      - chembl:     活性数据 (IC50/Ki/Kd)
      - pubmed:     文献摘要
      - clinical:   临床试验
      - full_report: 完整报告 (分块存储)
    """

    def __init__(self, persist_dir: str):
        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

    # ════════════════════════════════════════════════════════════
    # 存储方法
    # ════════════════════════════════════════════════════════════

    def store_uniprot(self, data: dict) -> None:
        """存储 UniProt 蛋白数据。"""
        if not data:
            return
        coll = self._get_or_create("uniprot")
        doc_id = data.get("uniprot_id", "unknown")
        text = (
            f"UniProt ID: {data.get('uniprot_id','?')}\n"
            f"Gene: {data.get('gene_symbol','?')}\n"
            f"Protein: {data.get('protein_name','?')}\n"
            f"Organism: {data.get('organism','?')}\n"
            f"Function: {data.get('function','?')}"
        )
        coll.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[{
                "source": "uniprot",
                "gene": data.get("gene_symbol", ""),
                "organism": data.get("organism", ""),
            }],
        )

    def store_pdb(self, structures: list) -> None:
        """存储 PDB 结构列表。"""
        if not structures:
            return
        coll = self._get_or_create("pdb")
        ids, docs, metas = [], [], []
        for s in structures:
            pid = s.get("pdb_id", "unknown")
            ids.append(pid)
            docs.append(
                f"PDB: {pid} | 分辨率: {s.get('resolution','?')}Å | "
                f"方法: {s.get('method','?')} | "
                f"含配体: {s.get('has_ligand',False)} | "
                f"配体ID: {s.get('ligand_ids',[])} | "
                f"标题: {s.get('title','?')}"
            )
            metas.append({
                "source": "pdb",
                "pdb_id": pid,
                "resolution": str(s.get("resolution", "")),
                "has_ligand": s.get("has_ligand", False),
                "method": s.get("method", ""),
            })
        if ids:
            coll.upsert(ids=ids, documents=docs, metadatas=metas)

    def store_chembl(self, activities: list) -> None:
        """存储 ChEMBL 活性数据。"""
        if not activities:
            return
        coll = self._get_or_create("chembl")
        ids, docs, metas = [], [], []
        for i, a in enumerate(activities):
            aid = a.get("chembl_id", f"chembl_{i}")
            ids.append(aid)
            docs.append(
                f"ChEMBL: {aid} | {a.get('assay_type','?')}={a.get('value','?')} "
                f"{a.get('unit','nM')} | 靶点: {a.get('target','?')}"
            )
            metas.append({
                "source": "chembl",
                "chembl_id": aid,
                "assay_type": a.get("assay_type", ""),
            })
        if ids:
            coll.upsert(ids=ids, documents=docs, metadatas=metas)

    def store_pubmed(self, papers: list) -> None:
        """存储 PubMed 文献。"""
        if not papers:
            return
        coll = self._get_or_create("pubmed")
        ids, docs, metas = [], [], []
        for i, p in enumerate(papers):
            pid = p.get("pmid", f"pubmed_{i}")
            ids.append(pid)
            docs.append(
                f"PMID: {pid} | 标题: {p.get('title','?')} | "
                f"年份: {p.get('year','?')} | 摘要: {p.get('abstract','?')[:500]}"
            )
            metas.append({
                "source": "pubmed",
                "pmid": pid,
                "year": str(p.get("year", "")),
            })
        if ids:
            coll.upsert(ids=ids, documents=docs, metadatas=metas)

    def store_clinical(self, trials: list) -> None:
        """存储 ClinicalTrials 数据。"""
        if not trials:
            return
        coll = self._get_or_create("clinical")
        ids, docs, metas = [], [], []
        for i, t in enumerate(trials):
            nct = t.get("nct_id", f"trial_{i}")
            ids.append(nct)
            docs.append(
                f"NCT: {nct} | {t.get('title','?')} | "
                f"阶段: {t.get('phase','?')} | 状态: {t.get('status','?')}"
            )
            metas.append({
                "source": "clinical",
                "nct_id": nct,
                "phase": t.get("phase", ""),
                "status": t.get("status", ""),
            })
        if ids:
            coll.upsert(ids=ids, documents=docs, metadatas=metas)

    def store_full_report(self, text: str, chunk_size: int = 800) -> None:
        """将完整报告分块存储。"""
        if not text:
            return
        coll = self._get_or_create("full_report")
        # 简单按段落分块
        paragraphs = text.split("\n\n")
        chunks, current = [], ""
        for p in paragraphs:
            if len(current) + len(p) < chunk_size:
                current += p + "\n\n"
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = p + "\n\n"
        if current.strip():
            chunks.append(current.strip())

        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}"
            ids.append(chunk_id)
            docs.append(chunk)
            # 判断章节
            section = "unknown"
            if "靶点生物学" in chunk[:100]:
                section = "biology"
            elif "结构生物" in chunk[:100]:
                section = "structure"
            elif "已知配体" in chunk[:100]:
                section = "ligands"
            elif "可药性" in chunk[:100]:
                section = "druggability"
            metas.append({"source": "full_report", "section": section, "chunk": i})
        if ids:
            coll.upsert(ids=ids, documents=docs, metadatas=metas)

    # ════════════════════════════════════════════════════════════
    # 搜索
    # ════════════════════════════════════════════════════════════

    def search(self, query: str, top_k: int = 5,
               sources: Optional[List[str]] = None) -> List[Dict]:
        """跨 collection 语义搜索。

        Args:
            query: 自然语言查询
            top_k: 返回条数
            sources: 限定 collection (如 ["pdb","chembl"]), None=全部

        Returns:
            [{"content": "...", "source": "pdb", "metadata": {...}}, ...]
        """
        collections = sources or ["uniprot", "pdb", "chembl", "pubmed", "clinical", "full_report"]
        results = []
        for name in collections:
            try:
                coll = self._client.get_collection(name)
                res = coll.query(query_texts=[query], n_results=top_k)
                if res and res.get("documents") and res["documents"][0]:
                    for i, doc in enumerate(res["documents"][0]):
                        meta = (res.get("metadatas", [[]])[0][i]
                                if res.get("metadatas") and res["metadatas"][0] else {})
                        results.append({
                            "content": doc,
                            "source": meta.get("source", name),
                            "metadata": meta,
                            "distance": (res.get("distances", [[]])[0][i]
                                         if res.get("distances") and res["distances"][0] else 0),
                        })
            except Exception:
                pass  # collection 不存在则跳过

        # 按距离排序, 取 top_k
        results.sort(key=lambda x: x.get("distance", 999))
        return results[:top_k]

    def search_formatted(self, query: str, top_k: int = 5,
                         sources: Optional[List[str]] = None) -> str:
        """搜索并格式化为纯文本, 供 LLM tool call 返回。"""
        results = self.search(query, top_k, sources)
        if not results:
            return "未找到相关信息。请尝试换个关键词查询。"

        lines = [f"🔍 查询: {query}\n找到 {len(results)} 条结果:\n"]
        for i, r in enumerate(results, 1):
            src = r.get("source", "?")
            lines.append(f"[{i}] 来源: {src}\n{r['content'][:600]}")
        return "\n---\n".join(lines)

    def clear(self) -> None:
        """清空所有 collection (用于测试)。"""
        for name in ["uniprot", "pdb", "chembl", "pubmed", "clinical", "full_report"]:
            try:
                self._client.delete_collection(name)
            except Exception:
                pass

    # ════════════════════════════════════════════════════════════
    # 辅助
    # ════════════════════════════════════════════════════════════

    def _get_or_create(self, name: str):
        try:
            return self._client.get_collection(name)
        except Exception:
            return self._client.create_collection(name)


# ════════════════════════════════════════════════════════════
# 工具函数: 供 StrategyGenerator 的 LLM tool call
# ════════════════════════════════════════════════════════════

def search_research_db(query: str, top_k: int = 5) -> str:
    """搜索靶点调研向量数据库。

    全局变量 _research_vs 需在调用前设置。
    """
    vs = _get_vs()
    if vs is None:
        return "错误: 向量数据库未初始化"
    return vs.search_formatted(query, top_k)


_research_vs: Optional[ResearchVectorStore] = None


def _get_vs() -> Optional[ResearchVectorStore]:
    return _research_vs


def set_research_vs(vs: ResearchVectorStore) -> None:
    global _research_vs
    _research_vs = vs


# OpenAI function calling schema
SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_research_db",
        "description": (
            "搜索靶点调研向量数据库，获取PDB结构、IC50/Ki/Kd活性数据、"
            "已知配体SAR、临床试验等详细信息。当摘要信息不足以确定策略参数时调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "自然语言查询，如 'BCL-2的已知配体IC50范围和结合模式' "
                                   "或 'SETDB1 Tudor结构域的PDB ID和关键残基'"
                },
            },
            "required": ["query"],
        },
    },
}
