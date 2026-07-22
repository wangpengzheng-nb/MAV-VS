from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ResearchModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RequirementPriority(str, Enum):
    MUST = "must"
    PREFER = "prefer"
    AVOID = "avoid"


class ScreeningRequirement(ResearchModel):
    category: Literal[
        "mechanism", "physicochemical", "selectivity", "off_target",
        "binding_site", "assay_context", "workflow", "other",
    ] = "other"
    priority: RequirementPriority
    original_text: str = Field(min_length=1, max_length=1000)
    normalized_key: str | None = None
    operator: Literal["eq", "ne", "lt", "le", "gt", "ge", "between", "contains"] | None = None
    value: str | float | list[str] | list[float] | None = None
    unit: str | None = None


class TargetIntent(ResearchModel):
    raw_query: str = Field(min_length=1, max_length=5000)
    target_text: str = Field(default="", max_length=200)
    organism_hint: str = Field(default="", max_length=200)
    organism_assumed: bool = False
    macromolecule_type: Literal["protein", "rna", "dna", "unknown"] = "protein"
    desired_effect: Literal["inhibit", "activate", "degrade", "bind", "modulate", "unknown"] = "unknown"
    requirements: list[ScreeningRequirement] = Field(default_factory=list)
    excluded_targets: list[str] = Field(default_factory=list)
    multiple_targets_detected: bool = False


class TargetIdentityCandidate(ResearchModel):
    canonical_gene_symbol: str = ""
    protein_name: str = ""
    uniprot_accession: str
    organism_name: str
    taxonomy_id: int | None = None
    aliases: list[str] = Field(default_factory=list)
    reviewed: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    match_evidence: list[str] = Field(default_factory=list)

    def compute_identity_fingerprint(self) -> str:
        return stable_fingerprint({
            "uniprot_accession": self.uniprot_accession,
            "taxonomy_id": self.taxonomy_id,
            "canonical_gene_symbol": self.canonical_gene_symbol,
        })


class TargetIdentity(TargetIdentityCandidate):
    status: Literal["verified"] = "verified"
    organism_assumed: bool = False
    verified_at: str = Field(default_factory=utcnow)
    identity_fingerprint: str = ""

    @model_validator(mode="after")
    def populate_fingerprint(self) -> "TargetIdentity":
        expected = TargetIdentityCandidate(**self.model_dump(
            exclude={"status", "organism_assumed", "verified_at", "identity_fingerprint"}
        )).compute_identity_fingerprint()
        if self.identity_fingerprint and self.identity_fingerprint != expected:
            raise ValueError("identity_fingerprint does not match the verified identity")
        self.identity_fingerprint = expected
        return self


class ResearchSourceResult(ResearchModel):
    source: str
    status: Literal["success", "empty", "unavailable", "invalid"]
    attempts: int = Field(default=1, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    error_type: str = ""
    message: str = ""
    retrieved_at: str = Field(default_factory=utcnow)
    snapshot_paths: list[str] = Field(default_factory=list)


class VerifiedStructure(ResearchModel):
    pdb_id: str = Field(pattern=r"^[0-9][A-Za-z0-9]{3}$")
    resolution: float | None = None
    method: str = ""
    title: str = ""
    deposition_year: int = 0
    has_ligand: bool = False
    ligand_ids: list[str] = Field(default_factory=list)
    uniprot_mapped: bool = True


class ChemblActivity(ResearchModel):
    molecule_chembl_id: str
    standard_type: str
    standard_value: float
    standard_units: str = "nM"
    target_chembl_id: str
    smiles: str = ""


class StructureReadiness(ResearchModel):
    experimental_holo_available: bool = False
    experimental_apo_available: bool = False
    predicted_structure_required: bool = False
    recommended_pdb_id: str = ""
    acquisition_recommendations: list[str] = Field(default_factory=list)
    pocket_risks: list[str] = Field(default_factory=list)


class TargetResearchReport(ResearchModel):
    research_version: Literal["2.0"] = "2.0"
    status: Literal["succeeded", "degraded"] = "succeeded"
    intent: TargetIntent
    identity: TargetIdentity
    source_results: list[ResearchSourceResult] = Field(default_factory=list)
    verified_pdb_structures: list[VerifiedStructure] = Field(default_factory=list)
    chembl_activities: list[ChemblActivity] = Field(default_factory=list)
    literature: list[dict[str, Any]] = Field(default_factory=list)
    clinical_trials: list[dict[str, Any]] = Field(default_factory=list)
    structure_readiness: StructureReadiness = Field(default_factory=StructureReadiness)
    evidence_gaps: list[str] = Field(default_factory=list)
    executive_summary: str = ""
    full_report_text: str = ""
    input_fingerprint: str
    research_timestamp: str = Field(default_factory=utcnow)

    # Compatibility fields consumed by the existing strategy and pocket stages.
    target_name: str = ""
    target_macromolecule_type: str = "Protein"
    gene_symbol: str = ""
    uniprot_id: str = ""
    target_uniprot_id: str = ""
    target_organism: str = ""
    recommended_pdb_for_docking: str = ""
    docking_center_from_pdb: list[float] = Field(default_factory=list)
    biology_overview: str = ""
    structural_analysis: str = ""
    druggability_assessment: str = ""
    known_ligands_text: str = ""
    api_sources: list[str] = Field(default_factory=list)
    verification_log: list[str] = Field(default_factory=list)
    key_metrics: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def derive_compatibility_fields(self) -> "TargetResearchReport":
        identity = self.identity
        self.target_name = identity.protein_name or identity.canonical_gene_symbol
        self.gene_symbol = identity.canonical_gene_symbol
        self.uniprot_id = identity.uniprot_accession
        self.target_uniprot_id = identity.uniprot_accession
        self.target_organism = identity.organism_name
        self.recommended_pdb_for_docking = self.structure_readiness.recommended_pdb_id
        self.api_sources = [f"{item.source}:{item.status}" for item in self.source_results]
        if not self.target_name:
            raise ValueError("a verified report cannot have an empty target_name")
        return self
