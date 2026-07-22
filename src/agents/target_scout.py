"""Compatibility facade for the v2 target-research subsystem.

Callers should keep importing ``TargetScoutAgent`` from this module. Identity
resolution, evidence acquisition and report validation live in
``src.agents.target_research``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agents.target_research import (
    ResearchSourceResult,
    ScreeningRequirement,
    StructureReadiness,
    TargetIdentity,
    TargetIdentityCandidate,
    TargetIntent,
    TargetResearchReport,
    TargetResearchService,
    TargetResolutionRequired,
    UnsupportedTargetError,
)


class TargetScoutAgent:
    def __init__(self, model: str = "deepseek-chat", api_key: str | None = None,
                 api_base: str | None = None, temperature: float = 0.1,
                 max_tokens: int = 4096, snapshot_dir: str | Path | None = None):
        # api_key/api_base/temperature/max_tokens remain accepted for compatibility;
        # credentials are read by the isolated LLM summarizer from the environment.
        del api_key, api_base, temperature, max_tokens
        self.model = model
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None

    def resolve_target(self, query: str, *, target_hint: str = "",
                       selected_accession: str = "") -> dict[str, Any]:
        return TargetResearchService(snapshot_dir=self.snapshot_dir, model=self.model).resolve_target(
            query, target_hint=target_hint, selected_accession=selected_accession,
        )

    def deep_research(self, query: str, *, fetch_structure_coordinates: bool = True,
                      target_hint: str = "", selected_accession: str = "",
                      snapshot_dir: str | Path | None = None) -> dict[str, Any]:
        # Coordinate download remains service-owned and occurs only after strategy
        # selection. The parameter is retained so existing callers do not break.
        del fetch_structure_coordinates
        output_dir = Path(snapshot_dir) if snapshot_dir else self.snapshot_dir
        return TargetResearchService(snapshot_dir=output_dir, model=self.model).research(
            query, target_hint=target_hint, selected_accession=selected_accession,
        )

    def generate_profile(self, target_info: dict[str, Any]) -> dict[str, Any]:
        query = target_info.get("description") or target_info.get("target_name") or ""
        return self.deep_research(
            query,
            target_hint=str(target_info.get("target_name") or ""),
            selected_accession=str(target_info.get("uniprot_accession") or ""),
        )


def target_scout_node(state: dict[str, Any]) -> dict[str, Any]:
    target_info = state.get("target_info", {})
    query = target_info.get("description") or target_info.get("target_name") or ""
    report = TargetScoutAgent().generate_profile({**target_info, "description": query})
    return {
        "pipeline_stage": "target_scout",
        "target_profile": report,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "event_log": [
            f"[TargetScout v2] target={report.get('target_uniprot_id', '?')} "
            f"PDBs={len(report.get('verified_pdb_structures', []))} "
            f"ChEMBL={len(report.get('chembl_activities', []))}"
        ],
    }


__all__ = [
    "ResearchSourceResult", "ScreeningRequirement", "StructureReadiness",
    "TargetIdentity", "TargetIdentityCandidate", "TargetIntent",
    "TargetResearchReport", "TargetResearchService", "TargetResolutionRequired",
    "TargetScoutAgent", "UnsupportedTargetError", "target_scout_node",
]
