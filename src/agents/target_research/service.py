from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from .client import ResearchHttpClient
from .models import (
    ChemblActivity,
    ResearchSourceResult,
    StructureReadiness,
    TargetIdentity,
    TargetIntent,
    TargetResearchReport,
    VerifiedStructure,
    stable_fingerprint,
)
from .resolver import TargetResolver


class TargetResearchError(RuntimeError):
    pass


class TargetResolutionRequired(TargetResearchError):
    def __init__(self, payload: dict[str, Any]):
        super().__init__(payload.get("reason", "target confirmation is required"))
        self.payload = payload


class UnsupportedTargetError(TargetResearchError):
    pass


class TargetResearchService:
    def __init__(self, *, snapshot_dir: Path | None = None, model: str = "deepseek-chat"):
        self.http = ResearchHttpClient(snapshot_dir)
        self.resolver = TargetResolver(self.http, model=model)
        self.model = model

    def resolve_target(self, query: str, *, target_hint: str = "", selected_accession: str = "") -> dict[str, Any]:
        intent = self.resolver.parse_intent(query, target_hint)
        result = self.resolver.resolve(intent, selected_accession=selected_accession)
        return self._dump(result)

    def research(self, query: str, *, target_hint: str = "", selected_accession: str = "") -> dict[str, Any]:
        resolution = self.resolver.resolve(
            self.resolver.parse_intent(query, target_hint), selected_accession=selected_accession,
        )
        if resolution["status"] == "unsupported":
            raise UnsupportedTargetError(resolution["reason"])
        if resolution["status"] != "resolved":
            raise TargetResolutionRequired(self._dump(resolution))
        intent: TargetIntent = resolution["intent"]
        identity: TargetIdentity = resolution["identity"]
        source_results: list[ResearchSourceResult] = list(resolution.get("source_results", []))

        structures, structure_source = self._fetch_structures(identity)
        source_results.extend(structure_source)
        activities, chembl_sources = self._fetch_chembl(identity)
        source_results.extend(chembl_sources)
        literature, pubmed_sources = self._fetch_pubmed(identity, intent)
        source_results.extend(pubmed_sources)
        trials, trial_source = self._fetch_trials(identity)
        source_results.append(trial_source)

        holo = [item for item in structures if item.has_ligand and item.uniprot_mapped]
        apo = [item for item in structures if not item.has_ligand and item.uniprot_mapped]
        recommended = sorted(holo, key=lambda item: (item.resolution or 99.0, -item.deposition_year))[0] if holo else None
        readiness = StructureReadiness(
            experimental_holo_available=bool(holo), experimental_apo_available=bool(apo),
            predicted_structure_required=not bool(holo),
            recommended_pdb_id=recommended.pdb_id if recommended else "",
            acquisition_recommendations=(
                ["Use the verified holo RCSB structure and derive the pocket from its bound ligand"] if recommended else
                ["Run target_structure_prediction through the capability layer (AlphaFold/Boltz when configured)",
                 "Validate the predicted pocket before molecular docking"]
            ),
            pocket_risks=[] if recommended else ["No verified experimental holo pocket is available"],
        )
        gaps = []
        if not activities:
            gaps.append("ChEMBL returned no accession-matched quantitative activities")
        if not literature:
            gaps.append("PubMed evidence is unavailable or empty")
        if readiness.predicted_structure_required:
            gaps.append("No verified experimental holo structure; structure prediction is required")
        unavailable = [item.source for item in source_results if item.status in {"unavailable", "invalid"}]
        gaps.extend(f"{source} source unavailable" for source in unavailable)

        fingerprint = stable_fingerprint({
            "query": query, "intent": intent.model_dump(mode="json"),
            "identity_fingerprint": identity.identity_fingerprint,
        })
        deterministic = self._deterministic_report(identity, structures, activities, literature, trials, readiness, gaps)
        summary = self._summarize(identity, intent, deterministic) or deterministic[:3000]
        report = TargetResearchReport(
            status="degraded" if gaps else "succeeded", intent=intent, identity=identity,
            source_results=source_results, verified_pdb_structures=structures,
            chembl_activities=activities, literature=literature, clinical_trials=trials,
            structure_readiness=readiness, evidence_gaps=gaps, executive_summary=summary,
            full_report_text=deterministic, input_fingerprint=fingerprint,
            biology_overview=self._biology_text(identity, literature),
            structural_analysis=self._structure_text(structures, readiness),
            known_ligands_text=self._ligand_text(activities),
            druggability_assessment=self._readiness_text(readiness, gaps),
            verification_log=[
                f"Verified identity: {identity.canonical_gene_symbol} / {identity.uniprot_accession}",
                f"Organism: {identity.organism_name} (assumed={identity.organism_assumed})",
                f"Experimental structures: {len(structures)}; holo: {len(holo)}",
            ],
            key_metrics={
                "has_cocrystal": bool(holo),
                "best_pdb_resolution": recommended.resolution if recommended else None,
                "known_ligand_ic50_range_nm": self._activity_range(activities),
            },
        )
        return report.model_dump(mode="json")

    def _fetch_structures(self, identity: TargetIdentity) -> tuple[list[VerifiedStructure], list[ResearchSourceResult]]:
        query = {
            "query": {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match", "value": identity.uniprot_accession,
            }},
            "return_type": "entry", "request_options": {"paginate": {"start": 0, "rows": 30}},
        }
        data, search_source = self.http.get_json(
            "rcsb_search", "https://search.rcsb.org/rcsbsearch/v2/query",
            params={"json": json.dumps(query, separators=(",", ":"))},
        )
        ids = [item.get("identifier", "").upper() for item in (data or {}).get("result_set", [])]
        if not ids and search_source.status == "success":
            search_source.status = "empty"
        structures: list[VerifiedStructure] = []
        detail_sources: list[ResearchSourceResult] = []
        skip = {"HOH", "DOD", "WAT", "NA", "CL", "K", "MG", "CA", "ZN", "SO4", "PO4", "GOL", "EDO"}
        for pdb_id in ids[:20]:
            if not re.fullmatch(r"[0-9][A-Z0-9]{3}", pdb_id):
                continue
            entry, source = self.http.get_json("rcsb_entry", f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}")
            detail_sources.append(source)
            if not entry:
                continue
            info = entry.get("rcsb_entry_info") or {}
            ligand_raw = info.get("nonpolymer_bound_components") or []
            ligand_ids: list[str] = []
            if isinstance(ligand_raw, list):
                ligand_ids = [str(item.get("comp_id", "") if isinstance(item, dict) else item).upper()
                              for item in ligand_raw]
            elif isinstance(ligand_raw, dict):
                ligand_ids = [str(key).upper() for key in ligand_raw]
            ligand_ids = sorted(set(item for item in ligand_ids if item and item not in skip))
            resolutions = info.get("resolution_combined") or []
            date = (entry.get("rcsb_accession_info") or {}).get("deposit_date", "")
            structures.append(VerifiedStructure(
                pdb_id=pdb_id, resolution=resolutions[0] if resolutions else None,
                method=info.get("experimental_method", ""), title=(entry.get("struct") or {}).get("title", ""),
                deposition_year=int(date[:4]) if date[:4].isdigit() else 0,
                has_ligand=bool(ligand_ids), ligand_ids=ligand_ids, uniprot_mapped=True,
            ))
        combined = ResearchSourceResult(
            source="rcsb_entry", status="success" if structures else ("unavailable" if any(s.status == "unavailable" for s in detail_sources) else "empty"),
            attempts=sum(item.attempts for item in detail_sources), latency_ms=sum(item.latency_ms for item in detail_sources),
            error_type=next((item.error_type for item in detail_sources if item.error_type), ""),
            snapshot_paths=[path for item in detail_sources for path in item.snapshot_paths],
        )
        structures.sort(key=lambda item: (not item.has_ligand, item.resolution or 99.0, -item.deposition_year))
        return structures, [search_source, combined]

    def _fetch_chembl(self, identity: TargetIdentity) -> tuple[list[ChemblActivity], list[ResearchSourceResult]]:
        data, target_source = self.http.get_json(
            "chembl_target", "https://www.ebi.ac.uk/chembl/api/data/target.json",
            params={"target_components__accession": identity.uniprot_accession, "limit": 10, "format": "json"},
        )
        targets = [item for item in (data or {}).get("targets", [])
                   if any(comp.get("accession") == identity.uniprot_accession for comp in item.get("target_components", []))]
        if not targets and target_source.status == "success":
            target_source.status = "empty"
        activities: list[ChemblActivity] = []
        sources = [target_source]
        for target in targets[:3]:
            target_id = target.get("target_chembl_id", "")
            if not target_id:
                continue
            payload, source = self.http.get_json(
                "chembl_activity", "https://www.ebi.ac.uk/chembl/api/data/activity.json",
                params={"target_chembl_id": target_id, "standard_type__in": "IC50,Ki,Kd,EC50",
                        "standard_units__in": "nM,uM", "limit": 100, "format": "json"},
            )
            sources.append(source)
            for raw in (payload or {}).get("activities", []):
                try:
                    value = float(raw.get("standard_value"))
                except (TypeError, ValueError):
                    continue
                units = raw.get("standard_units", "")
                if units == "uM":
                    value *= 1000.0
                if units not in {"nM", "uM"}:
                    continue
                activities.append(ChemblActivity(
                    molecule_chembl_id=raw.get("molecule_chembl_id", ""),
                    standard_type=raw.get("standard_type", ""), standard_value=value,
                    target_chembl_id=target_id, smiles=raw.get("canonical_smiles", "") or "",
                ))
        unique: dict[str, ChemblActivity] = {}
        for item in sorted(activities, key=lambda row: row.standard_value):
            unique.setdefault(item.molecule_chembl_id, item)
        return list(unique.values())[:20], sources

    def _fetch_pubmed(self, identity: TargetIdentity, intent: TargetIntent) -> tuple[list[dict[str, Any]], list[ResearchSourceResult]]:
        term = f'("{identity.canonical_gene_symbol}" OR "{identity.protein_name}") AND ({intent.desired_effect} OR drug discovery)'
        params: dict[str, Any] = {"db": "pubmed", "retmax": 20, "sort": "relevance", "term": term,
                                  "retmode": "json", "tool": os.getenv("NCBI_TOOL", "AutoVSAgent")}
        if os.getenv("NCBI_EMAIL"):
            params["email"] = os.getenv("NCBI_EMAIL")
        if os.getenv("NCBI_API_KEY"):
            params["api_key"] = os.getenv("NCBI_API_KEY")
        interval = 0.1 if os.getenv("NCBI_API_KEY") else 0.34
        search, search_source = self.http.get_json(
            "pubmed_search", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params=params, min_interval=interval,
        )
        ids = (search or {}).get("esearchresult", {}).get("idlist", [])
        if not ids:
            if search_source.status == "success":
                search_source.status = "empty"
            return [], [search_source]
        summary_params = {"db": "pubmed", "id": ",".join(ids), "retmode": "json", "tool": params["tool"]}
        if "email" in params:
            summary_params["email"] = params["email"]
        if "api_key" in params:
            summary_params["api_key"] = params["api_key"]
        summary, summary_source = self.http.get_json(
            "pubmed_summary", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params=summary_params, min_interval=interval,
        )
        results = []
        body = (summary or {}).get("result", {})
        for pmid in ids:
            item = body.get(pmid, {})
            if item.get("title"):
                results.append({"pmid": pmid, "title": item.get("title", ""),
                                "journal": item.get("fulljournalname", ""), "pubdate": item.get("pubdate", "")})
        return results, [search_source, summary_source]

    def _fetch_trials(self, identity: TargetIdentity) -> tuple[list[dict[str, Any]], ResearchSourceResult]:
        payload, source = self.http.get_json(
            "clinical_trials", "https://clinicaltrials.gov/api/v2/studies",
            params={"query.term": identity.canonical_gene_symbol or identity.protein_name, "pageSize": 10, "format": "json"},
        )
        trials = []
        for study in (payload or {}).get("studies", []):
            protocol = study.get("protocolSection", {})
            ident = protocol.get("identificationModule", {})
            status = protocol.get("statusModule", {})
            design = protocol.get("designModule", {})
            trials.append({"nct_id": ident.get("nctId", ""), "title": ident.get("briefTitle", ""),
                           "status": status.get("overallStatus", ""), "phases": design.get("phases", [])})
        return trials, source

    def _summarize(self, identity: TargetIdentity, intent: TargetIntent, report: str) -> str:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return ""
        try:
            client = OpenAI(api_key=api_key, base_url=f"{os.getenv('DEEPSEEK_API_BASE', 'https://api.deepseek.com').rstrip('/')}/v1")
            response = client.chat.completions.create(
                model=self.model, temperature=0.1, max_tokens=1800,
                messages=[
                    {"role": "system", "content": "基于给定证据生成简洁中文靶点调研摘要。不得补充证据中没有的ID、数值或结构。"},
                    {"role": "user", "content": json.dumps({"identity": identity.model_dump(mode="json"),
                                                               "intent": intent.model_dump(mode="json"),
                                                               "evidence_report": report[:14000]}, ensure_ascii=False)},
                ],
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            return ""

    @staticmethod
    def _biology_text(identity: TargetIdentity, literature: list[dict[str, Any]]) -> str:
        return (f"## 靶点身份与生物学\n\n{identity.protein_name}（{identity.canonical_gene_symbol}，"
                f"UniProt {identity.uniprot_accession}）来源于 {identity.organism_name}。"
                f"已检索到 {len(literature)} 条相关 PubMed 记录。")

    @staticmethod
    def _structure_text(structures: list[VerifiedStructure], readiness: StructureReadiness) -> str:
        lines = ["## 结构证据", ""]
        for item in structures[:20]:
            lines.append(f"- {item.pdb_id}: resolution={item.resolution or 'N/A'} A; holo={item.has_ligand}; ligands={','.join(item.ligand_ids) or 'none'}")
        if not structures:
            lines.append("- 未检索到与已验证 UniProt accession 精确映射的实验结构。")
        lines.extend(["", f"预测结构是否必需: {readiness.predicted_structure_required}"])
        return "\n".join(lines)

    @staticmethod
    def _ligand_text(activities: list[ChemblActivity]) -> str:
        lines = ["## 已知配体活性", ""]
        for item in activities[:20]:
            lines.append(f"- {item.molecule_chembl_id}: {item.standard_type}={item.standard_value:g} nM ({item.target_chembl_id})")
        if not activities:
            lines.append("- 未找到 accession 匹配且单位可归一化的 ChEMBL 定量活性。")
        return "\n".join(lines)

    @staticmethod
    def _readiness_text(readiness: StructureReadiness, gaps: list[str]) -> str:
        lines = ["## 虚拟筛选准备度", "", f"- 实验共晶结构: {readiness.experimental_holo_available}",
                 f"- 需要预测结构: {readiness.predicted_structure_required}"]
        lines.extend(f"- 证据缺口: {item}" for item in gaps)
        return "\n".join(lines)

    def _deterministic_report(self, identity: TargetIdentity, structures: list[VerifiedStructure],
                              activities: list[ChemblActivity], literature: list[dict[str, Any]],
                              trials: list[dict[str, Any]], readiness: StructureReadiness,
                              gaps: list[str]) -> str:
        return "\n\n".join([
            self._biology_text(identity, literature), self._structure_text(structures, readiness),
            self._ligand_text(activities),
            f"## 临床证据\n\n已检索到 {len(trials)} 项 ClinicalTrials.gov 记录。",
            self._readiness_text(readiness, gaps),
        ])

    @staticmethod
    def _activity_range(activities: list[ChemblActivity]) -> list[float]:
        values = [item.standard_value for item in activities]
        return [min(values), max(values)] if values else []

    @staticmethod
    def _dump(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return {key: TargetResearchService._dump(item) for key, item in value.items()}
        if isinstance(value, list):
            return [TargetResearchService._dump(item) for item in value]
        return value
